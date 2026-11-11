"""H3 — TQQQ ↔ 금 ↔ 현금(T-bill) 듀얼 모멘텀 (순수 pandas). 월말 판정, 목표 비중 DataFrame(열 TQQQ·GOLD)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DualMomParams:
    variant: str = "rel"        # rel: max(TQQQ, GOLD) > T-bill 이면 그쪽, 아니면 현금 / abs: TQQQ > T-bill 이면 TQQQ, 아니면 GOLD > T-bill 이면 GOLD, 아니면 현금
    lookback_m: int = 12
    skip_m: int = 1
    signal: str = "tqqq"        # TQQQ 모멘텀을 tqqq 자체로 / qqq(1배) 로

    @property
    def name(self) -> str:
        return f"dualmom {self.variant} L{self.lookback_m} skip{self.skip_m} sig={self.signal}"


def month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)


def momentum(close: pd.Series, me: pd.DatetimeIndex, lookback_m: int, skip_m: int) -> pd.Series:
    """월말 시계열에서 (t−skip)/(t−lookback) − 1."""
    m = close.reindex(me).ffill()
    return m.shift(skip_m) / m.shift(lookback_m) - 1.0


def tbill_momentum(cash_rate: pd.Series | None, index: pd.DatetimeIndex, me: pd.DatetimeIndex, lookback_m: int, skip_m: int) -> pd.Series:
    if cash_rate is None:
        return pd.Series(0.0, index=me)
    daily = cash_rate.reindex(index).ffill().fillna(0.0) / 252.0
    cum = (1.0 + daily).cumprod()
    return momentum(cum, me, lookback_m, skip_m)


def dual_momentum_weights(tqqq: pd.Series, qqq: pd.Series, gold: pd.Series, cash_rate: pd.Series | None, p: DualMomParams) -> pd.DataFrame:
    idx = tqqq.dropna().index
    me = month_end_index(idx)
    sig_t = momentum(qqq if p.signal == "qqq" else tqqq, me, p.lookback_m, p.skip_m)
    sig_g = momentum(gold, me, p.lookback_m, p.skip_m)
    tb = tbill_momentum(cash_rate, idx, me, p.lookback_m, p.skip_m)
    w = pd.DataFrame(0.0, index=me, columns=["TQQQ", "GOLD"])
    t_ok = sig_t > tb
    g_ok = sig_g > tb                     # 금 자료 없으면 NaN > x = False
    if p.variant == "rel":
        pick_t = t_ok & (sig_t >= sig_g.fillna(-np.inf))
        pick_g = g_ok & (sig_g > sig_t.fillna(-np.inf))
    elif p.variant == "abs":
        pick_t = t_ok
        pick_g = (~t_ok) & g_ok
    else:
        raise ValueError(p.variant)
    w.loc[pick_t, "TQQQ"] = 1.0
    w.loc[pick_g, "GOLD"] = 1.0
    # 모멘텀 계산 전(초반 lookback 개월) 은 TQQQ 보유(스위치와 같은 '시작 상태 보유')
    warm = sig_t.isna()
    w.loc[warm, "TQQQ"] = 1.0; w.loc[warm, "GOLD"] = 0.0
    return w.reindex(idx).ffill().fillna(0.0)
