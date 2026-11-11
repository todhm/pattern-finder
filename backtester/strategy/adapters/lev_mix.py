"""H11 — QLD(2×) 보유 ± 스위치, TQQQ·QQQ 분기 혼합 ± 스위치(TQQQ 몫만). 목표 비중 DataFrame(열 TQQQ·QLD·QQQ)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.list_tqqq_mix import drift_weights, rebalance_days


@dataclass(frozen=True)
class LevMixParams:
    kind: str = "qld"            # qld | mix
    w_tqqq: float = 0.5          # mix 전용
    switch_dd: float | None = None   # None = 스위치 없음
    n: int = 200
    reenter_n: int = 20

    @property
    def name(self) -> str:
        sw = "off" if self.switch_dd is None else f"dd{self.switch_dd:.2f}"
        return f"qld sw={sw}" if self.kind == "qld" else f"mix tqqq{self.w_tqqq:.2f}+qqq sw={sw}"


def lev_mix_weights(prices: pd.DataFrame, qqq_signal: pd.Series, p: LevMixParams) -> pd.DataFrame:
    idx = prices.index
    w = pd.DataFrame(0.0, index=idx, columns=["TQQQ", "QLD", "QQQ"])
    if p.kind == "qld":
        w["QLD"] = 1.0
    elif p.kind == "mix":
        px2 = prices[["TQQQ", "QQQ"]]
        rd = rebalance_days(idx, "Q")
        d = drift_weights(px2, pd.Series({"TQQQ": p.w_tqqq, "QQQ": 1.0 - p.w_tqqq}), idx, rd)
        w["TQQQ"] = d["TQQQ"]; w["QQQ"] = d["QQQ"]
    else:
        raise ValueError(p.kind)
    if p.switch_dd is not None:
        s = deep_correction_filter(qqq_signal, p.n, p.switch_dd, p.reenter_n).reindex(idx).ffill().fillna(1.0)
        if p.kind == "qld":
            w["QLD"] = w["QLD"] * s
        else:
            w["TQQQ"] = w["TQQQ"] * s
    return w.clip(0.0, 1.0)
