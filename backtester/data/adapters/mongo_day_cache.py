"""MongoDB-backed **day-chunked** OHLCV cache.

Replaces :class:`MongoMarketDataAdapter` (window-keyed) for the
production backtest stack. Each business day's bars live in a single
document so windows that overlap reuse the same per-day docs — a
sweep on ``2026-04-11~05-11`` and a page query on
``2026-04-20~05-15`` share every day between 4/20–5/11.

Schema
------
**``bars_{source}`` collection** (one per upstream source)::

    {
        symbol: "ATRA", interval: "1m", date: "2026-04-11",  # compound key
        status: "success",
        bars: [{ts, o, h, l, c, v}, ...],   # [] = non-trading day (holiday)
        is_partial: bool,                   # True iff this day was today at fetch
        fetched_at: datetime,
    }

    # On upstream exception, status="fail" doc replaces the per-day entry::
    { ..., status: "fail", error_type, error_msg, fetched_at }

Today-partial policy
--------------------
- A day fetched while ``date == today`` is stored ``is_partial=True``.
- On read, a stale partial (``is_partial=True`` AND
  ``fetched_at.date() < today``) is treated as a miss → refetched.
- Past-day docs (``is_partial=False``) are permanent.

Cross-window reuse
------------------
Queries are sliced to (start, end) on read — the same per-day doc
serves any window that contains its date. Sweep iterations on a
fixed window cost zero network after the first iter; page queries
on overlapping windows hit a mix of cached + new days.

Holiday handling
----------------
Upstream returns no bars for market holidays / non-trading weekdays.
We store ``status="success"`` with ``bars=[]`` for those days — the
next request sees the doc as a hit (not missing) and skips the
network call. Weekends are filtered out before lookup via
``pd.bdate_range``.

Negative cache (per-day)
------------------------
On upstream throw, a ``status="fail"`` doc with TTL
(``fail_ttl_hours``, default 24h) replaces the per-day entry for
each missing day in the fetched range. Subsequent requests for
those days raise :class:`_CachedFetchError` without hitting the
network — and :class:`FallbackMarketDataAdapter` catches the raise
to route to the secondary source.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

from data.domain.ports import MarketDataPort

NY_TZ = "America/New_York"


class _CachedFetchError(RuntimeError):
    """Raised when a per-day negative marker is hit.

    Inherits RuntimeError so :class:`FallbackMarketDataAdapter`'s
    blanket ``except Exception`` catches it like a fresh throw.
    """


class MongoDayCacheAdapter(MarketDataPort):
    """Day-chunked OHLCV cache backed by MongoDB.

    Constructor
        upstream:           wrapped data source
        source_name:        suffix for collection names (e.g. ``"eodhd"``)
        mongo_url:          mongodb URL (``MONGO_URL`` env default)
        mongo_db:           database name (``MONGO_DB`` env default)
        bypass_today:       signal-page opt-in — when True, requests
                            with ``end >= today`` skip the cache and
                            re-hit upstream. Default False.
        fail_ttl_hours:     per-day negative-marker TTL (default 24h).
    """

    def __init__(
        self,
        upstream: MarketDataPort,
        source_name: str,
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
        self._bars_coll: Collection = self._client[db_name][f"bars_{source_name}"]
        # Unique compound index — one doc per (symbol, interval, date).
        self._bars_coll.create_index(
            [("symbol", ASCENDING), ("interval", ASCENDING),
             ("date", ASCENDING)],
            unique=True,
            name="day_key",
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

        expected_days = self._business_days(start, end)
        if not expected_days:
            # All-weekend range — nothing to fetch, nothing cached.
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            )

        cached = self._load_cached(symbol, interval, expected_days)
        missing, fresh_fail = self._classify(cached, expected_days)

        if fresh_fail and not missing:
            # All un-cached days have fresh fail markers — raise so
            # FallbackMarketDataAdapter can route to the secondary.
            sample = next(iter(fresh_fail.values()))
            raise _CachedFetchError(
                f"cached upstream failure for {symbol} {interval}: "
                f"{sample.get('error_type', 'Error')}: "
                f"{sample.get('error_msg', '')}"
            )

        if missing:
            # Fetch from upstream — contiguous range covering all
            # missing days. Returned data is split per-day and
            # upserted; days that didn't get returned are stored as
            # empty-bars success docs (= holidays).
            fetch_start = min(missing)
            fetch_end = max(missing)
            try:
                df = self._upstream.fetch_ohlcv(
                    symbol, fetch_start, fetch_end, interval=interval
                )
            except Exception as exc:
                # Persist per-day fail markers for every missing day
                # so subsequent calls skip upstream entirely.
                self._write_fail_days(symbol, interval, missing, exc)
                raise
            self._write_success_days(
                symbol, interval, df, fetch_start, fetch_end
            )
            # Reload after upsert for the final concat below.
            cached = self._load_cached(symbol, interval, expected_days)

        return self._concat_slice(cached, start, end)

    def peek_cache(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Cross-source cache check used by :class:`FallbackMarketDataAdapter`.

        Returns a DataFrame iff *every* business day in (start, end)
        has a fresh ``status="success"`` doc. Stale partials, fail
        markers, or any missing day all return None so the fallback
        chain tries the next source's peek or proceeds to live fetch.
        """
        expected_days = self._business_days(start, end)
        if not expected_days:
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            )
        cached = self._load_cached(symbol, interval, expected_days)
        for d in expected_days:
            doc = cached.get(d.isoformat())
            if doc is None:
                return None
            if doc.get("status") != "success":
                return None
            if doc.get("is_partial") and self._is_stale_partial(doc):
                return None
        return self._concat_slice(cached, start, end)

    # ---- internals -------------------------------------------------

    @staticmethod
    def _business_days(start: date, end: date) -> list[date]:
        """Weekday Mon-Fri inclusive. Holidays are handled lazily —
        upstream returns no bars and we store an empty-bars doc."""
        idx = pd.bdate_range(start, end)
        return [ts.date() for ts in idx]

    def _load_cached(
        self,
        symbol: str,
        interval: str,
        expected_days: list[date],
    ) -> dict[str, dict[str, Any]]:
        """Fetch all per-day docs in [min(days), max(days)] in one query."""
        if not expected_days:
            return {}
        cursor = self._bars_coll.find({
            "symbol": symbol,
            "interval": interval,
            "date": {
                "$gte": expected_days[0].isoformat(),
                "$lte": expected_days[-1].isoformat(),
            },
        })
        return {doc["date"]: doc for doc in cursor}

    def _classify(
        self,
        cached: dict[str, dict[str, Any]],
        expected_days: list[date],
    ) -> tuple[list[date], dict[str, dict[str, Any]]]:
        """Split expected days into (missing, fresh_fail).

        Returns the list of days that need fetching and a map of
        fresh-fail days (for raising _CachedFetchError when there's
        nothing else to fetch). Stale fails and stale partials are
        treated as missing — the new fetch will overwrite.
        """
        missing: list[date] = []
        fresh_fail: dict[str, dict[str, Any]] = {}
        for d in expected_days:
            doc = cached.get(d.isoformat())
            if doc is None:
                missing.append(d)
                continue
            status = doc.get("status")
            if status == "success":
                if doc.get("is_partial") and self._is_stale_partial(doc):
                    missing.append(d)
                # else: fresh success — covered.
            elif status == "fail":
                if self._is_fresh_fail(doc):
                    fresh_fail[d.isoformat()] = doc
                else:
                    missing.append(d)  # TTL expired — retry
            else:
                missing.append(d)
        return missing, fresh_fail

    @staticmethod
    def _is_stale_partial(doc: dict[str, Any]) -> bool:
        fetched_at = doc.get("fetched_at")
        if not isinstance(fetched_at, datetime):
            return True
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return (
            fetched_at.astimezone(timezone.utc).date()
            < datetime.now(timezone.utc).date()
        )

    def _is_fresh_fail(self, doc: dict[str, Any]) -> bool:
        fetched_at = doc.get("fetched_at")
        if not isinstance(fetched_at, datetime):
            return False
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - fetched_at < self._fail_ttl

    def _write_success_days(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
        fetch_start: date,
        fetch_end: date,
    ) -> None:
        """Bulk-upsert one doc per business day in (fetch_start, fetch_end).

        Days returned by upstream get their bars; days NOT returned
        (holidays) get empty-bars success docs so the next request
        treats them as covered.

        **서브데일리 히스토리 한계 보호**: yfinance 15m처럼 히스토리
        깊이가 제한된 소스는 오래된 날짜에 봉을 안 주는데, 이를
        휴장일로 오인해 영구 빈 문서를 쓰면 다른 소스(Alpha Vantage
        등)가 채울 기회까지 크로스소스 peek이 막아버린다. 그래서
        서브데일리 interval에서는 **첫 반환 봉 이전 날짜는 캐시하지
        않고**(커버리지 불명), 전부 미반환이면 아무것도 쓰지 않는다.
        일봉 이상은 전 소스가 풀 히스토리라 기존 동작 유지.
        """
        today_iso = date.today().isoformat()
        now = datetime.now(timezone.utc)
        per_day_bars: dict[str, list[dict[str, Any]]] = {}
        if df is not None and not df.empty:
            local_df = df
            if local_df.index.tz is None:
                local_df = local_df.copy()
                local_df.index = local_df.index.tz_localize(NY_TZ)
            elif str(local_df.index.tz) != NY_TZ:
                local_df = local_df.copy()
                local_df.index = local_df.index.tz_convert(NY_TZ)
            # Group by date in NY tz so a 23:30 ET bar lands on the
            # right calendar day (vs UTC, which would split it).
            for ts, group_df in local_df.groupby(local_df.index.date):
                per_day_bars[ts.isoformat()] = self._df_to_records(group_df)

        sub_daily = interval not in ("1d", "1wk", "1mo")
        first_bar_date = min(per_day_bars) if per_day_bars else None

        ops = []
        for d in self._business_days(fetch_start, fetch_end):
            date_iso = d.isoformat()
            bars = per_day_bars.get(date_iso, [])
            if sub_daily and not bars:
                # 봉이 없는 날: 소스 히스토리 시작 이전이면 휴장이
                # 아니라 '못 주는 날' — 캐시하지 않고 다음 소스에 맡긴다.
                if first_bar_date is None or date_iso < first_bar_date:
                    continue
            ops.append(UpdateOne(
                {"symbol": symbol, "interval": interval, "date": date_iso},
                {"$set": {
                    "symbol": symbol,
                    "interval": interval,
                    "date": date_iso,
                    "status": "success",
                    "bars": bars,
                    "is_partial": date_iso == today_iso,
                    "fetched_at": now,
                }, "$unset": {
                    # Drop any prior fail-marker fields from this doc
                    # — overwrite cleanly.
                    "error_type": "", "error_msg": "",
                }},
                upsert=True,
            ))
        if ops:
            try:
                self._bars_coll.bulk_write(ops, ordered=False)
            except Exception:
                pass  # write failures are non-fatal — caller has data

    def _write_fail_days(
        self,
        symbol: str,
        interval: str,
        days: list[date],
        exc: Exception,
    ) -> None:
        now = datetime.now(timezone.utc)
        err_type = type(exc).__name__
        err_msg = str(exc)[:500]
        ops = []
        for d in days:
            date_iso = d.isoformat()
            ops.append(UpdateOne(
                {"symbol": symbol, "interval": interval, "date": date_iso},
                {"$set": {
                    "symbol": symbol,
                    "interval": interval,
                    "date": date_iso,
                    "status": "fail",
                    "error_type": err_type,
                    "error_msg": err_msg,
                    "fetched_at": now,
                }, "$unset": {
                    "bars": "", "is_partial": "",
                }},
                upsert=True,
            ))
        if ops:
            try:
                self._bars_coll.bulk_write(ops, ordered=False)
            except Exception:
                pass

    @staticmethod
    def _concat_slice(
        cached: dict[str, dict[str, Any]],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Concat all success-status bars in [start, end] into one DataFrame."""
        all_records: list[dict[str, Any]] = []
        for date_iso in sorted(cached.keys()):
            if not (start.isoformat() <= date_iso <= end.isoformat()):
                continue
            doc = cached[date_iso]
            if doc.get("status") != "success":
                continue
            all_records.extend(doc.get("bars", []))
        return MongoDayCacheAdapter._records_to_df(all_records)

    # ---- DataFrame ↔ records -------------------------------------

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        out: list[dict[str, Any]] = []
        # Precompute column index — itertuples gives us a row tuple
        # with positional access by ordinal.
        col_idx = {
            c: df.columns.get_loc(c)
            for c in ("Open", "High", "Low", "Close", "Volume")
            if c in df.columns
        }
        for row in df.itertuples(index=True):
            ts = row[0]
            rec: dict[str, Any] = {
                "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            }
            for col, i in col_idx.items():
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
        # ``ts`` is stored as ISO with the NY offset at write time.
        # Across DST boundaries one cache slice can contain both
        # ``-04:00`` and ``-05:00`` offsets, which ``pd.to_datetime``
        # rejects with "Mixed timezones detected". Parse as UTC
        # first (normalizes the offsets) and then convert back to
        # NY so the index matches what upstream adapters return.
        df.index = pd.DatetimeIndex(
            pd.to_datetime(df["ts"], utc=True)
        ).tz_convert(NY_TZ)
        df = df.drop(columns=["ts"])
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        # Ensure all expected columns exist (some sources return
        # subset; downstream code expects the full set).
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c not in df.columns:
                df[c] = 0.0
        return df[["Open", "High", "Low", "Close", "Volume"]]
