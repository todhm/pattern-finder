"""H8 — 반도체 1배 ↔ TQQQ 상대 모멘텀 (순수 pandas). 월말 판정, 목표 비중 DataFrame(열 TQQQ·SOXX)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy.adapters.dual_momentum import momentum, month_end_index


@dataclass(frozen=True)
class SectorRotParams:
    lookback_m: int = 6
    threshold: float = 0.0
    base: str = "qqq"           # 상대 비교 기준: qqq(1배) | tqqq(3배)

    @property
    def name(self) -> str:
        return f"sectorrot L{self.lookback_m} thr{self.threshold:+.0%} vs={self.base}"


def sector_rotation_weights(tqqq: pd.Series, qqq: pd.Series, sector: pd.Series, p: SectorRotParams) -> pd.DataFrame:
    idx = tqqq.dropna().index
    me = month_end_index(idx)
    ref = qqq if p.base == "qqq" else tqqq
    rel = momentum(sector, me, p.lookback_m, 0) - momentum(ref, me, p.lookback_m, 0)
    pick_sector = (rel > p.threshold).fillna(False)          # 섹터 자료 없으면 TQQQ
    w = pd.DataFrame({"TQQQ": (~pick_sector).astype(float), "SOXX": pick_sector.astype(float)}, index=me)
    return w.reindex(idx).ffill().fillna({"TQQQ": 1.0, "SOXX": 0.0})
