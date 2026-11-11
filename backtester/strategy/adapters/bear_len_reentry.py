"""H14 — 약세 길이에 따른 재진입 속도 (순수 pandas/numpy). 이탈 = 스위치 18."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BearLenParams:
    long_days: int = 120
    slow_mode: str = "sma50"     # sma50 | monthly20
    n: int = 200
    dd: float = 0.18
    fast_n: int = 20

    @property
    def name(self) -> str:
        return f"bearlen L{self.long_days} slow={self.slow_mode}"


def bear_len_switch(signal_close: pd.Series, p: BearLenParams) -> pd.Series:
    c = signal_close.dropna(); idx = c.index
    sma = c.rolling(p.n).mean(); ddv = c / c.rolling(252, min_periods=20).max() - 1.0
    off = ((c < sma) & (ddv < -p.dd)).to_numpy()
    fast_on = (c > c.rolling(p.fast_n).mean()).to_numpy()
    slow_on = (c > c.rolling(50).mean()).to_numpy() if p.slow_mode == "sma50" else fast_on
    per = idx.to_period("M"); last = np.zeros(len(c), dtype=bool)
    last[pd.Series(np.arange(len(c)), index=idx).groupby(per).last().to_numpy()] = True
    nan = sma.isna().to_numpy()
    state = np.ones(len(c)); cur = 1.0; days_out = 0
    for i in range(len(c)):
        if nan[i]:
            state[i] = 0.0; continue
        if cur > 0:
            if off[i]:
                cur = 0.0; days_out = 0
        else:
            days_out += 1
            if days_out < p.long_days:
                ok = fast_on[i]
            else:
                ok = slow_on[i] if p.slow_mode == "sma50" else (fast_on[i] and last[i])
            if ok and not off[i]:
                cur = 1.0
        state[i] = cur
    return pd.Series(state, index=idx, name=p.name)
