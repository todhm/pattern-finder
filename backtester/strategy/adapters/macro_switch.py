"""H1 — 거시 레짐으로 이탈 문턱을 바꾸는 폭락 스위치 (순수 pandas/numpy).

스위치 18 (`leveraged_etf_lab.deep_correction_filter`) 과 같은 뼈대 — 이탈 = 종가 < N일선 **그리고** 52주 고점 대비 낙폭 > dd,
재진입 = 종가 > reenter_n 일선 — 에서 dd 만 그날의 거시 레짐(위험/중립/건강)에 따라 셋 중 하나를 쓴다.
거시 점수는 `macro_regime.composite_score` (0=전부 위험 … 1=전부 건강), 전날까지의 값.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroSwitchParams:
    dd_risk: float = 0.12
    dd_mid: float = 0.18
    dd_healthy: float = 0.24
    risk_thr: float = 0.5       # m < risk_thr → 위험
    healthy_thr: float = 0.75   # m ≥ healthy_thr → 건강
    n: int = 200
    reenter_n: int = 20

    @property
    def name(self) -> str:
        return f"macro_sw r{self.dd_risk:.2f}/m{self.dd_mid:.2f}/h{self.dd_healthy:.2f} thr{self.risk_thr:g}-{self.healthy_thr:g}"


def regime_series(macro_score: pd.Series, index: pd.DatetimeIndex, risk_thr: float, healthy_thr: float) -> pd.Series:
    """점수 → -1(위험)/0(중립)/+1(건강). 점수 없는 날(지표 이전)은 중립."""
    m = macro_score.reindex(index, method="ffill")
    reg = pd.Series(0, index=index, dtype=int)
    reg[m < risk_thr] = -1
    reg[m >= healthy_thr] = 1
    return reg


def dd_threshold_series(macro_score: pd.Series, index: pd.DatetimeIndex, p: MacroSwitchParams) -> pd.Series:
    reg = regime_series(macro_score, index, p.risk_thr, p.healthy_thr)
    return reg.map({-1: p.dd_risk, 0: p.dd_mid, 1: p.dd_healthy}).astype(float)


def macro_switch(signal_close: pd.Series, macro_score: pd.Series, p: MacroSwitchParams) -> pd.Series:
    """목표 비중 0/1. 시작 상태 보유, N일선 미계산 구간 0."""
    c = signal_close.dropna()
    sma = c.rolling(p.n).mean()
    drawdown = c / c.rolling(252, min_periods=20).max() - 1.0
    dd = dd_threshold_series(macro_score, c.index, p)
    off = ((c < sma) & (drawdown < -dd)).to_numpy()
    on = (c > c.rolling(p.reenter_n).mean()).to_numpy()
    sma_nan = sma.isna().to_numpy()
    state = np.ones(len(c)); cur = 1.0
    for i in range(len(c)):
        if sma_nan[i]:
            state[i] = 0.0; continue
        if off[i]:
            cur = 0.0
        elif on[i]:
            cur = 1.0
        state[i] = cur
    return pd.Series(state, index=c.index, name=p.name)


def regime_share(macro_score: pd.Series, index: pd.DatetimeIndex, p: MacroSwitchParams) -> dict:
    reg = regime_series(macro_score, index, p.risk_thr, p.healthy_thr)
    return {"risk": float((reg == -1).mean()), "mid": float((reg == 0).mean()), "healthy": float((reg == 1).mean())}
