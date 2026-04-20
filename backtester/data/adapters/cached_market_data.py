import hashlib
import os
from datetime import date
from pathlib import Path

import pandas as pd

from data.domain.ports import MarketDataPort


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
    """

    def __init__(
        self,
        upstream: MarketDataPort,
        cache_dir: str | os.PathLike = "/tmp/pattern-finder-cache",
    ):
        self._upstream = upstream
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        path = self._cache_path(symbol, start, end)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                # Corrupted cache file — fall through to re-fetch.
                path.unlink(missing_ok=True)
        df = self._upstream.fetch_ohlcv(symbol, start, end)
        try:
            df.to_parquet(path)
        except Exception:
            pass
        return df

    def _cache_path(self, symbol: str, start: date, end: date) -> Path:
        key = f"{symbol}_{start.isoformat()}_{end.isoformat()}"
        safe = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"{symbol}_{safe}.parquet"
