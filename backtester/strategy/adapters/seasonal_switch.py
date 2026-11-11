"""H6 — 달력 규칙 TQQQ 노출 (순수 pandas). 목표 비중 DataFrame(열 TQQQ·QQQ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonParams:
    kind: str = "halloween"     # halloween | earnings
    months: tuple = (11, 12, 1, 2, 3, 4)       # halloween: TQQQ 월
    weeks: tuple = (2, 3, 4)                   # earnings: 1·4·7·10월의 몇째 주(월 안 주차, 1 = 1~7일)
    off_asset: str = "qqq"                     # qqq | cash

    @property
    def name(self) -> str:
        if self.kind == "halloween":
            return f"season halloween m{self.months[0]}-{self.months[-1]} off={self.off_asset}"
        return f"season earnings w{self.weeks[0]}-{self.weeks[-1]} off={self.off_asset}"


def in_season(index: pd.DatetimeIndex, p: SeasonParams) -> pd.Series:
    if p.kind == "halloween":
        on = index.month.isin(p.months)
    elif p.kind == "earnings":
        wk = (index.day - 1) // 7 + 1
        on = index.month.isin((1, 4, 7, 10)) & np.isin(wk, p.weeks)
    else:
        raise ValueError(p.kind)
    return pd.Series(on, index=index)


def seasonal_weights(index: pd.DatetimeIndex, p: SeasonParams) -> pd.DataFrame:
    on = in_season(index, p).astype(float)
    w = pd.DataFrame({"TQQQ": on, "QQQ": 0.0}, index=index)
    if p.off_asset == "qqq":
        w["QQQ"] = 1.0 - on
    elif p.off_asset != "cash":
        raise ValueError(p.off_asset)
    return w
