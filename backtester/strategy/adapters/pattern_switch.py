"""H19 — 패턴 신호(날짜 목록) 로 만드는 스위치 (순수 pandas/numpy). 신호 계산은 호출 쪽(감지기 어댑터)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternSwitchParams:
    mode: str = "pattern"        # pattern | switch18_early
    sell_set: str = "all"
    reenter: str = "sma20"       # pattern: wedge_pop | sma20 | either
    dd_alarm: float = 0.05       # switch18_early: 매도 패턴 날 낙폭 ≥ dd_alarm 이면 이탈
    n: int = 200
    dd: float = 0.18
    reenter_n: int = 20

    @property
    def name(self) -> str:
        return f"pattern sell={self.sell_set} re={self.reenter}" if self.mode == "pattern" else f"sw18+pattern sell={self.sell_set} dd{self.dd_alarm:.2f}"


def _mask(index: pd.DatetimeIndex, dates) -> np.ndarray:
    s = pd.Series(False, index=index)
    d = pd.DatetimeIndex(pd.to_datetime(list(dates))).tz_localize(None) if len(dates) else pd.DatetimeIndex([])
    s.loc[s.index.isin(d)] = True
    return s.to_numpy()


def pattern_switch(close: pd.Series, sell_dates, buy_dates, p: PatternSwitchParams) -> pd.Series:
    c = close.dropna(); idx = c.index
    sell = _mask(idx, sell_dates); buy = _mask(idx, buy_dates)
    sma_re = (c > c.rolling(p.reenter_n).mean()).to_numpy()
    if p.mode == "pattern":
        if p.reenter == "wedge_pop":
            on = buy
        elif p.reenter == "sma20":
            on = sma_re
        elif p.reenter == "either":
            on = buy | sma_re
        else:
            raise ValueError(p.reenter)
        off = sell
    elif p.mode == "switch18_early":
        sma = c.rolling(p.n).mean(); ddv = (c / c.rolling(252, min_periods=20).max() - 1.0).to_numpy()
        base_off = ((c < sma) & (ddv < -p.dd)).to_numpy()
        off = base_off | (sell & (ddv <= -p.dd_alarm))
        on = sma_re
    else:
        raise ValueError(p.mode)
    state = np.ones(len(c)); cur = 1.0
    warm = c.rolling(p.n).mean().isna().to_numpy()
    for i in range(len(c)):
        if warm[i]:
            state[i] = 0.0; continue
        if cur > 0 and off[i]:
            cur = 0.0
        elif cur == 0 and on[i] and not off[i]:
            cur = 1.0
        state[i] = cur
    return pd.Series(state, index=idx, name=p.name)
