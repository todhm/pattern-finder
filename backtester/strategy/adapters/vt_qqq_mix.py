"""H4 — 변동성 타기팅 TQQQ 비중 + 나머지 QQQ (+ 스위치 18) 목표 비중 DataFrame (순수 pandas)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.vol_target_regime_lab import predicted_vol


@dataclass(frozen=True)
class VtMixParams:
    target: float = 0.45
    predictor: str = "rv20"
    switch: str = "none"        # none | cash(이탈 시 전부 현금) | qqq(이탈 시 QQQ 유지)
    dd: float = 0.18
    n: int = 200
    reenter_n: int = 20
    max_exposure: float = 1.0

    @property
    def name(self) -> str:
        return f"vtmix t{self.target:.2f} {self.predictor} sw={self.switch}"


def vt_mix_weights(tqqq: pd.Series, qqq: pd.Series, p: VtMixParams) -> pd.DataFrame:
    sigma = predicted_vol(tqqq, p.predictor)
    w = (p.target / sigma).clip(lower=0.0, upper=p.max_exposure)
    w = w.where(sigma.notna(), 1.0)                       # 변동성 미계산 초반은 TQQQ 보유(시작 상태 보유)
    out = pd.DataFrame({"TQQQ": w, "QQQ": 1.0 - w}, index=tqqq.index)
    if p.switch != "none":
        d = deep_correction_filter(qqq, p.n, p.dd, p.reenter_n).reindex(tqqq.index).ffill().fillna(1.0)
        out["TQQQ"] = out["TQQQ"] * d
        if p.switch == "cash":
            out["QQQ"] = out["QQQ"] * d
        elif p.switch == "qqq":
            out["QQQ"] = 1.0 - out["TQQQ"]
        else:
            raise ValueError(p.switch)
    return out.clip(0.0, 1.0)
