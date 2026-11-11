"""H24 — 스위치 18 이탈을 조용한 날로 미루기 (순수 pandas/numpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalmExitParams:
    r: float = 0.02
    max_delay: int = 5
    up_only: bool = False
    n: int = 200
    dd: float = 0.18
    reenter_n: int = 20

    @property
    def name(self) -> str:
        return f"calmexit r{self.r:.0%} d{self.max_delay}{' up' if self.up_only else ''}"


def calm_exit_switch(ohlcv: pd.DataFrame, p: CalmExitParams) -> pd.Series:
    c = ohlcv["Close"].dropna(); o = ohlcv["Open"].reindex(c.index)
    sma = c.rolling(p.n).mean(); ddv = c / c.rolling(252, min_periods=20).max() - 1.0
    off = ((c < sma) & (ddv < -p.dd)).to_numpy(); on = (c > c.rolling(p.reenter_n).mean()).to_numpy()
    ret = (c / c.shift(1) - 1.0).to_numpy(); gap = (o / c.shift(1) - 1.0).to_numpy()
    calm = (np.abs(ret) < p.r) & (np.abs(gap) < p.r)
    if p.up_only:
        calm &= ret >= 0
    nan = sma.isna().to_numpy()
    state = np.ones(len(c)); cur = 1.0; pending = -1
    for i in range(len(c)):
        if nan[i]:
            state[i] = 0.0; continue
        if cur > 0:
            if off[i]:
                if pending < 0:
                    pending = 0
                else:
                    pending += 1
                if calm[i] or pending >= p.max_delay:
                    cur = 0.0; pending = -1
            else:
                pending = -1                                   # 이탈 조건이 풀리면 보류 취소
        else:
            if on[i] and not off[i]:
                cur = 1.0
        state[i] = cur
    return pd.Series(state, index=c.index, name=p.name)
