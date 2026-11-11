"""H7 — 스위치 18 의 재진입 변형 (순수 pandas/numpy). 이탈 = 종가 < N일선 & 52주 고점 대비 낙폭 > dd (고정).

재진입 = 종가 > reenter_n 일선 [그리고 breadth > breadth_min (폭 유효일 때만)]. 재진입 뒤 lock_days 거래일 동안 재이탈 금지.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReentryParams:
    reenter_n: int = 20
    breadth_min: float | None = None
    lock_days: int = 0
    n: int = 200
    dd: float = 0.18

    @property
    def name(self) -> str:
        b = "none" if self.breadth_min is None else f">{self.breadth_min:.0%}"
        return f"reentry sma{self.reenter_n} breadth{b} lock{self.lock_days}"


def reentry_switch(signal_close: pd.Series, breadth: pd.Series | None, p: ReentryParams) -> pd.Series:
    c = signal_close.dropna()
    sma = c.rolling(p.n).mean()
    dd = c / c.rolling(252, min_periods=20).max() - 1.0
    off = ((c < sma) & (dd < -p.dd)).to_numpy()
    on = (c > c.rolling(p.reenter_n).mean()).to_numpy()
    if p.breadth_min is not None and breadth is not None:
        b = breadth.reindex(c.index)
        on = on & (b.isna() | (b > p.breadth_min)).to_numpy()
    sma_nan = sma.isna().to_numpy()
    state = np.ones(len(c)); cur = 1.0; locked_until = -1
    for i in range(len(c)):
        if sma_nan[i]:
            state[i] = 0.0; continue
        if cur == 1.0:
            if off[i] and i > locked_until:
                cur = 0.0
        else:
            if on[i] and not off[i]:                 # 이탈 조건이 아직 참이면 재진입 안 함(`deep_correction_filter` 와 동일한 우선순위)
                cur = 1.0; locked_until = i + p.lock_days
        state[i] = cur
    return pd.Series(state, index=c.index, name=p.name)
