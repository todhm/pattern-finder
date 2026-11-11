"""H23 — 워크포워드 규칙 로테이션 (순수 pandas/numpy). 분기마다 직전 N년 세후 성적 최고 부품을 다음 분기에."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams
from strategy.adapters.multi_asset_sim import simulate_weights
from strategy.adapters.regime_slots import COLS


@dataclass(frozen=True)
class WalkForwardParams:
    years: int = 3
    metric: str = "mult"        # mult | calmar(mult ÷ |MDD|)
    sticky: bool = False        # 현재 부품보다 10% 이상 좋을 때만 교체
    years2: int | None = None   # 두 번째 룩백(성적 평균, H25)
    top_k: int = 1              # 상위 k 개 동일비중 혼합(H25)

    @property
    def name(self) -> str:
        yrs = f"N{self.years}" + (f"+{self.years2}" if self.years2 else "")
        return f"wf {yrs} {self.metric}{' sticky' if self.sticky else ''}{' top' + str(self.top_k) if self.top_k > 1 else ''}"


def quarter_starts(index: pd.DatetimeIndex, first: pd.Timestamp) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    qs = pd.DatetimeIndex(s.groupby(index.to_period("Q")).first().values)
    return qs[qs >= first]


def trailing_score(prices: pd.DataFrame, target: pd.DataFrame, start, end, costs: CostParams, cash_rate, metric: str) -> float:
    px = prices[(prices.index >= start) & (prices.index <= end)]
    t = target.reindex(px.index).ffill()
    if t.iloc[0].isna().any():
        prev = target[target.index < px.index[0]]; t = t.fillna(prev.iloc[-1] if len(prev) else 0.0)
    r = simulate_weights(px, t, costs, cash_rate, "wf")
    m = r.final_after_liquidation / costs.initial_capital
    if metric == "calmar":
        return m / max(abs(float(r.metrics.get("mdd", -1.0))), 0.05)
    return m


def walk_forward(prices: pd.DataFrame, library: dict[str, pd.DataFrame], p: WalkForwardParams, costs: CostParams, cash_rate,
                 first: pd.Timestamp, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """반환: (스티치 목표 비중, 분기별 선택 이름). rng 가 있으면 무작위 선택(플라세보)."""
    idx = prices.index
    qs = quarter_starts(idx, first)
    names = list(library)
    out = pd.DataFrame(np.nan, index=idx, columns=COLS); picks = {}
    cur: list[str] | None = None
    for i, q in enumerate(qs):
        if rng is not None:
            cur = list(rng.choice(names, size=p.top_k, replace=False))
        else:
            end = idx[idx < q][-1]
            scores = {}
            for n, f in library.items():
                sc = trailing_score(prices, f, q - pd.DateOffset(years=p.years), end, costs, cash_rate, p.metric)
                if p.years2:
                    sc = 0.5 * (sc + trailing_score(prices, f, q - pd.DateOffset(years=p.years2), end, costs, cash_rate, p.metric))
                scores[n] = sc
            ranked = sorted(scores, key=scores.get, reverse=True); best = ranked[: p.top_k]
            if cur is None or not p.sticky or scores[best[0]] > scores[cur[0]] * 1.10:
                cur = best
        picks[q] = "+".join(cur)
        nxt = qs[i + 1] if i + 1 < len(qs) else idx[-1] + pd.Timedelta(days=1)
        m = (idx >= q) & (idx < nxt)
        blend = sum(library[n].reindex(idx[m]).ffill().fillna(0.0) for n in cur) / len(cur)
        out.loc[m] = blend.to_numpy()
    out = out.ffill().fillna({"TQQQ": 1.0, "QLD": 0.0, "QQQ": 0.0})
    return out, pd.Series(picks)
