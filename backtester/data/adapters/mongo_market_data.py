"""MongoDB-backed market-data cache.

Drop-in replacement for :class:`CachedMarketDataAdapter` (parquet
disk cache) for **backtest / sweep pages**. Signal pages keep using
the parquet cache because they want a clean ``bypass_today`` opt-in
and don't need cross-machine cache sharing.

Schema
------
Single collection per interval-source pair (e.g. ``ohlcv_eodhd``,
``ohlcv_massive``, ``ohlcv_yfinance``). Each document::

    {
        symbol, interval, start, end,         # query key (compound index)
        status: "success" | "fail",
        fetched_at: datetime (UTC),
        is_today_partial: bool,               # only on success
        data: [{ts, open, high, low, close, volume}, ...],   # success
        error_type, error_msg: str,           # fail
    }

Today-partial policy (user spec)
--------------------------------
- "해당일이 지나고 저장한 데이터는 계속 저장" — Past-day data is final.
- "해당일 중에 저장한 데이터는 나중에 다시 저장" — Same-day data is
  re-fetched on subsequent days.

Implementation:
- When fetched with ``end >= today``, mark ``is_today_partial=True``.
- On read, if ``is_today_partial`` and ``today > fetched_at.date()``,
  treat as miss → refetch + overwrite.
- When fetched with ``end < today``, mark ``is_today_partial=False`` →
  always served from cache (final).

Negative cache markers (``status="fail"``) follow the same TTL logic
as the parquet version (default 24h).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from data.domain.ports import MarketDataPort


class _CachedFetchError(RuntimeError):
    """Raised when a cached negative-marker is hit.

    Inherits RuntimeError so :class:`FallbackMarketDataAdapter`'s
    blanket ``except Exception`` catches it and routes to the
    fallback adapter, identical to a fresh upstream throw.
    """


class MongoMarketDataAdapter(MarketDataPort):
    """OHLCV cache backed by MongoDB.

    Constructor
        upstream:           wrapped data source
        collection_name:    Mongo collection (one per upstream usually)
        mongo_url:          mongodb URL (defaults to ``MONGO_URL`` env)
        mongo_db:           database name (defaults to ``MONGO_DB`` env
                            or "pattern_finder")
        bypass_today:       signal-page opt-in. When True, requests
                            with ``end >= today`` skip the cache and
                            re-hit upstream. Defaults False (backtest).
        fail_ttl_hours:     negative-cache marker TTL (default 24h).
    """

    def __init__(
        self,
        upstream: MarketDataPort,
        collection_name: str,
        mongo_url: str | None = None,
        mongo_db: str | None = None,
        *,
        bypass_today: bool = False,
        fail_ttl_hours: int = 24,
        client: MongoClient | None = None,
    ) -> None:
        self._upstream = upstream
        url = mongo_url or os.environ.get("MONGO_URL", "mongodb://mongo:27017")
        db_name = mongo_db or os.environ.get("MONGO_DB", "pattern_finder")
        self._client = client or MongoClient(url, serverSelectionTimeoutMS=5000)
        self._coll: Collection = self._client[db_name][collection_name]
        # Compound unique index — one doc per (symbol, interval, start, end).
        self._coll.create_index(
            [("symbol", ASCENDING), ("interval", ASCENDING),
             ("start", ASCENDING), ("end", ASCENDING)],
            unique=True,
            name="cache_key",
        )
        self._bypass_today = bypass_today
        self._fail_ttl = timedelta(hours=fail_ttl_hours)

    # ---- public API ------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        # Signal-page opt-out: bypass cache when end includes today.
        if self._bypass_today and end >= date.today():
            return self._upstream.fetch_ohlcv(
                symbol, start, end, interval=interval
            )

        key = self._key(symbol, start, end, interval)
        entry = self._coll.find_one(key)

        # ---- existing entry handling ----
        if entry is not None:
            status = entry.get("status")
            if status == "success" and self._is_fresh(entry):
                return self._records_to_df(entry.get("data", []))
            if status == "fail" and self._is_fail_fresh(entry):
                raise _CachedFetchError(
                    f"cached upstream failure for {symbol} {interval}: "
                    f"{entry.get('error_type', 'Error')}: "
                    f"{entry.get('error_msg', '')}"
                )
            # else: stale (partial-today expired OR fail TTL expired)
            #       → fall through to refetch + overwrite.

        # ---- live fetch ----
        try:
            df = self._upstream.fetch_ohlcv(
                symbol, start, end, interval=interval
            )
        except Exception as exc:
            self._write_fail(key, exc)
            raise

        self._write_success(key, end, df)
        return df

    def peek_cache(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Cross-source cache check used by :class:`FallbackMarketDataAdapter`.

        Returns a DataFrame only on a fresh success entry. Stale
        today-partial entries, fail markers, and missing entries
        all return None — the fallback chain then tries the next
        source's peek or proceeds to live fetch.
        """
        entry = self._coll.find_one(self._key(symbol, start, end, interval))
        if entry is None:
            return None
        if entry.get("status") != "success":
            return None
        if not self._is_fresh(entry):
            return None
        return self._records_to_df(entry.get("data", []))

    # ---- internals -------------------------------------------------

    @staticmethod
    def _key(symbol: str, start: date, end: date, interval: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "interval": interval,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    @staticmethod
    def _is_fresh(entry: dict[str, Any]) -> bool:
        """Apply the today-partial refresh policy.

        Final past-day entries (``is_today_partial=False``) are always
        fresh — they never need re-fetching. Same-day partial entries
        (``is_today_partial=True``) are only fresh on the day they
        were fetched; once the system date advances past
        ``fetched_at.date()`` they become stale and must be refetched.
        """
        if not entry.get("is_today_partial", False):
            return True  # past-day data: permanent
        fetched_at = entry.get("fetched_at")
        if not isinstance(fetched_at, datetime):
            return False
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        # Compare in UTC-day terms (consistent with insertion time).
        return fetched_at.astimezone(timezone.utc).date() >= datetime.now(
            timezone.utc
        ).date()

    def _is_fail_fresh(self, entry: dict[str, Any]) -> bool:
        fetched_at = entry.get("fetched_at")
        if not isinstance(fetched_at, datetime):
            return False
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - fetched_at < self._fail_ttl

    def _write_success(
        self,
        key: dict[str, Any],
        end: date,
        df: pd.DataFrame,
    ) -> None:
        is_partial = end >= date.today()
        doc = {
            **key,
            "status": "success",
            "is_today_partial": is_partial,
            "fetched_at": datetime.now(timezone.utc),
            "data": self._df_to_records(df),
        }
        try:
            self._coll.replace_one(key, doc, upsert=True)
        except Exception:
            # Write failures are non-fatal — we still return data to
            # caller. Next call will re-fetch + try cache again.
            pass

    def _write_fail(self, key: dict[str, Any], exc: Exception) -> None:
        doc = {
            **key,
            "status": "fail",
            "fetched_at": datetime.now(timezone.utc),
            "error_type": type(exc).__name__,
            "error_msg": str(exc)[:500],
        }
        try:
            self._coll.replace_one(key, doc, upsert=True)
        except Exception:
            pass

    # ---- DataFrame ↔ records -------------------------------------

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        out: list[dict[str, Any]] = []
        # Iterate via numpy view for speed; ``itertuples`` is the
        # fastest pandas iter that preserves the typed-index value.
        cols = {c: df.columns.get_loc(c) for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns}
        tz = df.index.tz
        for row in df.itertuples(index=True):
            ts = row[0]
            rec: dict[str, Any] = {"ts": ts.isoformat()}
            for col, i in cols.items():
                rec[col.lower()] = float(row[i + 1])
            out.append(rec)
        return out

    @staticmethod
    def _records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            )
        df = pd.DataFrame.from_records(records)
        df.index = pd.to_datetime(df["ts"])
        df = df.drop(columns=["ts"])
        # Restore canonical capitalization expected by detectors.
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        return df
