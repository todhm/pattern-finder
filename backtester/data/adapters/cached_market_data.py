import hashlib
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.domain.ports import MarketDataPort


class _CachedFetchError(RuntimeError):
    """Raised when a cached negative-marker is hit.

    Inherits from RuntimeError so :class:`FallbackMarketDataAdapter`'s
    blanket ``except Exception`` catches it and routes to the fallback,
    same as if the upstream had thrown the original error.
    """


class CachedMarketDataAdapter(MarketDataPort):
    """Disk-cache decorator for any ``MarketDataPort``.

    First call for a given ``(symbol, start, end)`` hits the upstream
    adapter and persists the result as a parquet file. Subsequent calls
    with the same key load from disk — useful when iterating over
    strategy knobs without changing the fetch window, since yfinance
    calls are the scan's dominant cost (~2 min for 500 tickers).

    Cache key includes start/end so a new date range misses correctly.
    Files live under ``cache_dir`` (defaults to
    ``/tmp/pattern-finder-cache`` so it survives Streamlit reloads and
    rebuilds without bloating the repo).

    Negative caching
        When the upstream throws (HTTP 4xx/5xx, parse error, etc.) a
        small ``.fail.json`` marker is written next to where the
        parquet would live. Subsequent calls within ``fail_ttl_hours``
        re-raise the stored error *without* hitting the network — so
        when a ``FallbackMarketDataAdapter`` chain has primary=Cached
        (EODHD) + fallback=Cached(Massive), a ticker EODHD doesn't
        cover stays routed to Massive after the first failure instead
        of burning an EODHD HTTP per iteration.
    """

    def __init__(
        self,
        upstream: MarketDataPort,
        cache_dir: str | os.PathLike = "/tmp/pattern-finder-cache",
        *,
        bypass_today: bool = False,
        fail_ttl_hours: int = 24,
    ):
        """
        ``bypass_today``:
            When True, requests with ``end >= today`` skip the cache
            entirely and re-hit the upstream adapter. Required for
            **signal pages** that scan live data — caching a mid-
            session bar would serve stale OHLC/Volume on subsequent
            calls within the same trading day.

            Backtest / parameter-sweep pages should leave this False
            (default) — they only consume historical bars and re-
            running the same window across iterations is the dominant
            cost driver. With the flag off, sweeps reuse parquet
            caches even when ``end`` is today.
        """
        self._upstream = upstream
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._bypass_today = bypass_today
        self._fail_ttl = timedelta(hours=fail_ttl_hours)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        # Optional bypass: signal pages set ``bypass_today``
        # so the live intraday bar isn't served from a stale cached
        # snapshot. Backtest/sweep pages leave it off — re-using the
        # cache across iterations is the whole point of caching.
        if self._bypass_today and end >= date.today():
            return self._upstream.fetch_ohlcv(symbol, start, end, interval=interval)

        path = self._cache_path(symbol, start, end, interval)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                # Corrupted cache file — fall through to re-fetch.
                path.unlink(missing_ok=True)

        # Negative-cache check: did a recent upstream call throw for
        # this exact key? Re-raise the stored error so a Fallback
        # wrapper routes to its secondary without hitting the network.
        fail_path = self._fail_path(symbol, start, end, interval)
        if fail_path.exists():
            try:
                stat = fail_path.stat()
                age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
                if age < self._fail_ttl:
                    payload = json.loads(fail_path.read_text())
                    raise _CachedFetchError(
                        f"cached upstream failure for {symbol} {interval}: "
                        f"{payload.get('error_type', 'Error')}: "
                        f"{payload.get('error_msg', '')}"
                    )
                # Stale marker — drop and try fresh.
                fail_path.unlink(missing_ok=True)
            except _CachedFetchError:
                raise
            except Exception:
                fail_path.unlink(missing_ok=True)

        try:
            df = self._upstream.fetch_ohlcv(
                symbol, start, end, interval=interval
            )
        except Exception as exc:
            # Persist negative marker so subsequent calls skip the
            # upstream and re-raise from disk. If a previous success
            # parquet exists somehow, leave it alone — only the
            # fail marker is touched.
            try:
                fail_path.write_text(json.dumps({
                    "symbol": symbol,
                    "interval": interval,
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc)[:500],
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                }))
            except Exception:
                pass
            raise
        try:
            df.to_parquet(path)
            # Successful fetch invalidates any stale negative marker.
            fail_path.unlink(missing_ok=True)
        except Exception:
            pass
        return df

    def peek_cache(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Read-only cache lookup — no upstream call, no fail-marker check.

        Returns the cached DataFrame if a parquet exists for this exact
        ``(symbol, start, end, interval)`` key, else ``None``. Used by
        :class:`FallbackMarketDataAdapter` to do a cross-source cache
        sweep before any live fetch (so a Massive parquet is preferred
        over an EODHD live call that would otherwise burn quota).

        Negative cache markers (``.fail.json``) are *not* treated as
        hits — a marker means "primary couldn't fetch" not "no data
        anywhere", and the parquet from the secondary source might
        cover the request fine.

        ``bypass_today`` is ignored here: the caller is asking
        explicitly "do you have it?" and is willing to accept stale
        bars in exchange for skipping a network call. Signal pages
        that need fresh data shouldn't be using cross-source peek.
        """
        path = self._cache_path(symbol, start, end, interval)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            # Corrupted parquet — drop and report miss.
            path.unlink(missing_ok=True)
            return None

    def _cache_path(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> Path:
        # Keep the legacy key for "1d" so existing parquet caches stay
        # valid; tag every other interval explicitly to avoid collisions.
        if interval == "1d":
            key = f"{symbol}_{start.isoformat()}_{end.isoformat()}"
        else:
            key = f"{symbol}_{interval}_{start.isoformat()}_{end.isoformat()}"
        safe = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"{symbol}_{safe}.parquet"

    def _fail_path(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> Path:
        # Same hash as the parquet cache so a successful refetch can
        # locate and clear its own negative marker.
        return self._cache_path(symbol, start, end, interval).with_suffix(
            ".fail.json"
        )
