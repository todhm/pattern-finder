"""H18 — 직전 N년 실현 L*=μ/σ² 로 TQQQ/QLD/QQQ 선택 (순수 pandas/numpy). 목표 비중 DataFrame(열 TQQQ·QLD·QQQ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.regime_slots import COLS
from strategy.adapters.walk_forward_select import quarter_starts


@dataclass(frozen=True)
class LstarParams:
    years: int = 5
    hi: float = 3.0
    lo: float = 2.0
    switch: bool = True

    @property
    def name(self) -> str:
        return f"lstar N{self.years} hi{self.hi:g} lo{self.lo:g}{' +sw18' if self.switch else ''}"


def trailing_lstar(close: pd.Series, asof: pd.Timestamp, years: int) -> float:
    r = close.pct_change().dropna()
    r = r[(r.index < asof) & (r.index >= asof - pd.DateOffset(years=years))]
    if len(r) < 250:
        return np.nan
    mu = float(r.mean() * 252); var = float(r.var() * 252)
    return mu / var if var > 0 else np.nan


def lstar_weights(qqq: pd.Series, p: LstarParams, first: pd.Timestamp, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, pd.Series]:
    idx = qqq.dropna().index
    qs = quarter_starts(idx, first)
    out = pd.DataFrame(0.0, index=idx, columns=COLS); picks = {}
    out.loc[idx < qs[0], "TQQQ"] = 1.0
    for i, q in enumerate(qs):
        if rng is not None:
            pick = ["TQQQ", "QLD", "QQQ"][int(rng.integers(3))]; L = np.nan
        else:
            L = trailing_lstar(qqq, q, p.years)
            pick = "TQQQ" if (np.isnan(L) or L > p.hi) else ("QLD" if L >= p.lo else "QQQ")
        picks[q] = (pick, L)
        nxt = qs[i + 1] if i + 1 < len(qs) else idx[-1] + pd.Timedelta(days=1)
        out.loc[(idx >= q) & (idx < nxt), pick] = 1.0
    if p.switch:
        d = deep_correction_filter(qqq, 200, 0.18, 20).reindex(idx).ffill().fillna(1.0)
        out = out.mul(d, axis=0)
    return out, pd.Series({k: v[0] for k, v in picks.items()})
