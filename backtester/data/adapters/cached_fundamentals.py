"""Disk-cache decorator for FundamentalsPort.

Float / splits don't change daily — float updates with insider
trades / lock-up expirations (weekly at most), splits are point-in-
time historical events that never change once recorded. A multi-day
TTL on the cache cuts the per-scan API calls on subsequent runs to
zero for tickers we've already seen.

Cache layout::

    {cache_dir}/{symbol}.json
    {
        "fetched_at": "2026-05-10T...",
        "float_shares": 1234567.0,   // null if missing
        "splits": [
            {"date": "2025-04-08", "ratio": 0.1},
            ...
        ]
    }

A cache file older than ``ttl_days`` is treated as stale and refetched.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data.domain.ports import FundamentalsPort, TickerFundamentals

log = logging.getLogger(__name__)


class CachedFundamentalsAdapter(FundamentalsPort):
    def __init__(
        self,
        delegate: FundamentalsPort,
        cache_dir: str = "/tmp/pattern-finder-cache/fundamentals",
        ttl_days: int = 7,
    ) -> None:
        self._delegate = delegate
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = timedelta(days=ttl_days)

    def fetch(self, symbol: str) -> TickerFundamentals:
        cache_path = self._cache_dir / f"{symbol}.json"
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached
        result = self._delegate.fetch(symbol)
        # Always write cache, even when both fields are None — saves
        # repeating the failed fetch on next run within TTL. Tickers
        # that legitimately have no float (ETFs, OTC-only) will benefit.
        self._write_cache(cache_path, result)
        return result

    # ---- read ----

    def _read_cache(self, path: Path) -> TickerFundamentals | None:
        if not path.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > self._ttl:
                return None
            payload = json.loads(path.read_text())
        except Exception as exc:
            log.debug("Cache read failed for %s: %s", path, exc)
            return None
        symbol = payload.get("symbol", path.stem)
        float_shares = payload.get("float_shares")
        splits_rows = payload.get("splits") or []
        splits = None
        if splits_rows:
            try:
                idx = pd.DatetimeIndex(
                    [pd.Timestamp(r["date"]) for r in splits_rows]
                )
                vals = [float(r["ratio"]) for r in splits_rows]
                splits = pd.Series(vals, index=idx).sort_index()
            except Exception:
                splits = None
        return TickerFundamentals(
            symbol=symbol,
            float_shares=float(float_shares) if float_shares is not None else None,
            splits=splits,
        )

    # ---- write ----

    def _write_cache(self, path: Path, fund: TickerFundamentals) -> None:
        splits_rows = []
        if fund.splits is not None and len(fund.splits) > 0:
            for ts, ratio in fund.splits.items():
                splits_rows.append({
                    "date": ts.strftime("%Y-%m-%d"),
                    "ratio": float(ratio),
                })
        payload = {
            "symbol": fund.symbol,
            "fetched_at": datetime.now().isoformat(),
            "float_shares": fund.float_shares,
            "splits": splits_rows,
        }
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(tmp, path)
        except Exception as exc:
            log.debug("Cache write failed for %s: %s", path, exc)
