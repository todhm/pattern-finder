"""H12 — 스위치 18 의 톱니 비용 변형 (순수 pandas/numpy): 밴드·월말 판정·2단계 재진입. 목표 비중 {0, 0.5, 1}."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WhipsawParams:
    band: float = 0.0
    monthly: bool = False
    two_stage: bool = False
    stage_days: int = 10
    n: int = 200
    dd: float = 0.18
    reenter_n: int = 20

    @property
    def name(self) -> str:
        return f"whipsaw band{self.band:.2f} {'monthly' if self.monthly else 'daily'} {'2stage' if self.two_stage else '1stage'}"


def whipsaw_switch(signal_close: pd.Series, p: WhipsawParams) -> pd.Series:
    c = signal_close.dropna()
    sma = c.rolling(p.n).mean(); dd = c / c.rolling(252, min_periods=20).max() - 1.0
    off = ((c < sma * (1.0 - p.band)) & (dd < -p.dd)).to_numpy()
    on = (c > c.rolling(p.reenter_n).mean() * (1.0 + p.band)).to_numpy()
    sma_nan = sma.isna().to_numpy()
    decide = np.ones(len(c), dtype=bool)
    if p.monthly:
        per = c.index.to_period("M")
        last = pd.Series(np.arange(len(c)), index=c.index).groupby(per).last().to_numpy()
        decide[:] = False; decide[last] = True
    state = np.ones(len(c)); cur = 1.0; stage_at = -1
    for i in range(len(c)):
        if sma_nan[i]:
            state[i] = 0.0; continue
        if decide[i]:
            if cur > 0.0 and off[i]:
                cur = 0.0; stage_at = -1
            elif cur == 0.0 and on[i] and not off[i]:
                cur = 0.5 if p.two_stage else 1.0; stage_at = i
        if p.two_stage and cur == 0.5 and stage_at >= 0 and i >= stage_at + p.stage_days and on[i]:
            cur = 1.0                                         # stage_days 뒤에도 20일선(×밴드) 위면 100%
        state[i] = cur
    return pd.Series(state, index=c.index, name=p.name)


def round_trips(target: pd.Series, start: str, end: str) -> int:
    t = target[(target.index >= start) & (target.index <= end)]
    d = (t > 0).astype(int).diff().fillna(0)
    return int((d == -1).sum())
