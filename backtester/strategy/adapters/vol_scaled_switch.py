"""H28 — 변동성 비례 이탈 문턱 스위치 (순수 pandas/numpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import realized_vol


@dataclass(frozen=True)
class VolScaledParams:
    k: float = 0.75
    vol_lookback: int = 60
    lo: float = 0.10
    hi: float = 0.30
    n: int = 200
    reenter_n: int = 20

    @property
    def name(self) -> str:
        return f"volscaled k{self.k:g} L{self.vol_lookback}"


def vol_scaled_switch(signal_close: pd.Series, p: VolScaledParams) -> pd.Series:
    c = signal_close.dropna(); idx = c.index
    sma = c.rolling(p.n).mean(); ddv = (c / c.rolling(252, min_periods=20).max() - 1.0).to_numpy()
    thr = (p.k * realized_vol(c, p.vol_lookback)).clip(lower=p.lo, upper=p.hi).ffill().fillna(p.hi).to_numpy()
    off = (c < sma).to_numpy() & (ddv < -thr); on = (c > c.rolling(p.reenter_n).mean()).to_numpy()
    nan = sma.isna().to_numpy(); state = np.ones(len(c)); cur = 1.0
    for i in range(len(c)):
        if nan[i]:
            state[i] = 0.0; continue
        if cur > 0 and off[i]:
            cur = 0.0
        elif cur == 0 and on[i] and not off[i]:
            cur = 1.0
        state[i] = cur
    return pd.Series(state, index=idx, name=p.name)
