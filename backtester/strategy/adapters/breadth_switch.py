"""H2 — 시장 폭(구성 종목 중 200일선 위 비율) 스위치 (순수 pandas/numpy).

이탈: breadth < exit_pct 그리고 지수 종가 < price_sma 일선. 재진입: breadth > reenter_pct. 이탈 중 노출 off_fraction.
폭이 무효(유효 종목 < min_names)인 날은 이탈하지 않는다(보유 유지). 시작 상태 보유.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreadthParams:
    exit_pct: float = 0.40
    reenter_pct: float = 0.60
    off_fraction: float = 0.0
    breadth_sma: int = 200
    price_sma: int = 50
    min_names: int = 30

    @property
    def name(self) -> str:
        return f"breadth x{self.exit_pct:.0%}/y{self.reenter_pct:.0%} off{self.off_fraction:g}"


def breadth_series(closes: pd.DataFrame, sma: int = 200, min_names: int = 30) -> pd.Series:
    """종목별 종가 > SMA 비율. 종가·SMA 가 모두 있는 종목만 분모. 분모 < min_names 면 NaN."""
    ma = closes.rolling(sma, min_periods=sma).mean()
    valid = closes.notna() & ma.notna()
    above = (closes > ma) & valid
    n = valid.sum(axis=1)
    b = above.sum(axis=1) / n.where(n > 0)
    return b.where(n >= min_names).rename("breadth")


def breadth_switch(signal_close: pd.Series, breadth: pd.Series, p: BreadthParams) -> pd.Series:
    c = signal_close.dropna()
    b = breadth.reindex(c.index)
    sma = c.rolling(p.price_sma).mean()
    off = ((b < p.exit_pct) & (c < sma)).to_numpy()          # NaN 비교는 False → 이탈 안 함
    on = (b > p.reenter_pct).to_numpy()
    state = np.ones(len(c)); cur = 1.0
    for i in range(len(c)):
        if off[i]:
            cur = 0.0
        elif on[i]:
            cur = 1.0
        state[i] = cur
    s = pd.Series(state, index=c.index, name=p.name)
    return s.clip(lower=p.off_fraction) if p.off_fraction > 0 else s
