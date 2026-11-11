"""H5 — 1× 종목 리스트 곡선 + TQQQ 혼합, 분기(또는 연) 리밸런스, 스위치 18 은 TQQQ 쪽만 (순수 pandas).

목표 비중 DataFrame(열 TQQQ·LIST). 리밸런스일에 (w, 1−w) 로 맞추고 그 사이에는 가격으로 굴린 드리프트 비중을 목표로 넣어
다자산 원장이 사이에 거래하지 않게 한다(밴드 5% 안). 스위치 off 구간은 TQQQ 몫을 0(현금) 으로, LIST 몫은 드리프트 그대로.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter


@dataclass(frozen=True)
class ListMixParams:
    w_tqqq: float = 0.5
    rebalance: str = "Q"        # Q | A
    switch: bool = False
    dd: float = 0.18
    n: int = 200
    reenter_n: int = 20

    @property
    def name(self) -> str:
        return f"listmix w{self.w_tqqq:.1f} {self.rebalance} sw={'18' if self.switch else 'off'}"


def rebalance_days(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    per = index.to_period("Q" if freq == "Q" else "Y")
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(per).first().values)


def drift_weights(prices: pd.DataFrame, w0: pd.Series, index: pd.DatetimeIndex, rdays: pd.DatetimeIndex) -> pd.DataFrame:
    """리밸런스일에 w0, 그 사이는 가격으로 굴린 비중(합 1 유지)."""
    px = prices.reindex(index).ffill()
    out = pd.DataFrame(np.nan, index=index, columns=px.columns)
    rset = set(rdays)
    cur = None; base_px = None
    for t in index:
        if t in rset or cur is None:
            cur = w0.copy(); base_px = px.loc[t]
        rel = px.loc[t] / base_px
        v = cur * rel
        out.loc[t] = v / v.sum() if v.sum() > 0 else cur
    return out


def list_mix_weights(prices: pd.DataFrame, qqq: pd.Series, p: ListMixParams) -> pd.DataFrame:
    """prices: 열 TQQQ·LIST (LIST 는 리스트 자산 곡선). LIST 가 없는 날은 TQQQ 몫만(나머지 현금)."""
    idx = prices.index
    have_list = prices["LIST"].notna()
    rd = rebalance_days(idx[have_list], p.rebalance)
    w0 = pd.Series({"TQQQ": p.w_tqqq, "LIST": 1.0 - p.w_tqqq})
    w = drift_weights(prices[have_list], w0, idx[have_list], rd).reindex(idx)
    w.loc[~have_list, "TQQQ"] = p.w_tqqq; w.loc[~have_list, "LIST"] = 0.0
    if p.switch:
        d = deep_correction_filter(qqq, p.n, p.dd, p.reenter_n).reindex(idx).ffill().fillna(1.0)
        w["TQQQ"] = w["TQQQ"] * d
    return w.clip(0.0, 1.0)
