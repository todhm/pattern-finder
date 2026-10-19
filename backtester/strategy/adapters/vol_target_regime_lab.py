"""Phase 1 — 변동성 타기팅 × 낙폭 레짐 (로드맵 `TQQQ_초과수익_로드맵_2026-10.md` §6).

질문: "직전 변동성으로 노출을 w = min(1, 목표σ / 예측σ) 로 정하고(실효 레버리지 3w), ⑩ 낙폭 레짐을 겹치면
참사 5년(2000·01·02·08·22)을 피하면서 폭발 15년을 얼마나 지키는가."

- 예측기: 실현 20일·60일, EWMA(λ=0.94, RiskMetrics), max(20일, 60일). GARCH 는 과거 전체로 적합하면 선행 참조라 보류.
- σ 는 **레버리지 ETF(합성 3배) 자체의 연환산 변동성**. 목표 30/45/60% 는 기초지수 10/15/20% 에 해당.
- 레짐: ⑩ ``deep_correction_filter`` (200일선 아래 & 52주 고점 대비 낙폭 > 18% → 0, 20일선 회복 → 1).
- 체결: 신호 종가 확정 → 다음날 종가(exec_lag 1). 리밸런싱 밴드 5%. 세금 22%(연 250만 공제) + 토스 수수료.
엔진은 `leveraged_etf_lab.simulate_exposure` 그대로.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import (
    TRADING_DAYS,
    CostParams,
    LabResult,
    combine,
    deep_correction_filter,
    realized_vol,
    simulate_exposure,
)

PREDICTORS = ("rv20", "rv60", "ewma94", "max20_60")


def ewma_vol(close: pd.Series, lam: float = 0.94) -> pd.Series:
    r = np.log(close / close.shift(1)).dropna()
    var = np.empty(len(r)); v = float(r.iloc[:20].var()) if len(r) > 20 else float(r.var())
    for i, x in enumerate(r.values):
        v = lam * v + (1 - lam) * x * x
        var[i] = v
    return pd.Series(np.sqrt(var) * math.sqrt(TRADING_DAYS), index=r.index).reindex(close.index)


def predicted_vol(lev_close: pd.Series, predictor: str) -> pd.Series:
    if predictor == "rv20":
        return realized_vol(lev_close, 20)
    if predictor == "rv60":
        return realized_vol(lev_close, 60)
    if predictor == "ewma94":
        return ewma_vol(lev_close, 0.94)
    if predictor == "max20_60":
        return pd.concat([realized_vol(lev_close, 20), realized_vol(lev_close, 60)], axis=1).max(axis=1)
    raise ValueError(predictor)


@dataclass(frozen=True)
class VtParams:
    target: float = 0.45          # 레버리지 ETF 기준 목표 변동성
    predictor: str = "rv20"
    regime: bool = True           # ⑩ 겹치기
    dd_depth: float = 0.18
    dd_n: int = 200
    dd_reenter: int = 20
    max_exposure: float = 1.0

    @property
    def name(self) -> str:
        return f"vt{int(self.target * 100)}_{self.predictor}" + ("_dd" if self.regime else "")


def target_weights(lev_close: pd.Series, signal_close: pd.Series, p: VtParams) -> pd.Series:
    """목표 비중 시계열 (선행 참조 없음: t 일 종가까지의 정보로 t 일 목표 → exec_lag 로 t+1 체결)."""
    sigma = predicted_vol(lev_close, p.predictor)
    w = (p.target / sigma).clip(lower=0.0, upper=p.max_exposure).fillna(0.0)
    if p.regime:
        d = deep_correction_filter(signal_close, p.dd_n, p.dd_depth, p.dd_reenter).reindex(lev_close.index).ffill().fillna(0.0)
        w = combine(w, d)
    return w.rename(p.name)


def run_one(lev_close: pd.Series, signal_close: pd.Series, p: VtParams, costs: CostParams,
            cash_rate, start: date, end: date | None = None) -> LabResult:
    w = target_weights(lev_close, signal_close, p)
    lev = lev_close[lev_close.index >= pd.Timestamp(start)]
    if end is not None:
        lev = lev[lev.index <= pd.Timestamp(end)]
    return simulate_exposure(lev, w.reindex(lev.index).ffill().fillna(0.0), costs, cash_rate, p.name)


def run_hold(lev_close: pd.Series, costs: CostParams, cash_rate, start: date, end: date | None = None) -> LabResult:
    lev = lev_close[lev_close.index >= pd.Timestamp(start)]
    if end is not None:
        lev = lev[lev.index <= pd.Timestamp(end)]
    return simulate_exposure(lev, pd.Series(1.0, index=lev.index), costs, cash_rate, "hold")


def summarize(res: LabResult, hold: LabResult) -> dict:
    yrs = len(res.equity) / TRADING_DAYS
    mult = res.final_after_liquidation / res.equity.iloc[0]
    hmult = hold.final_after_liquidation / hold.equity.iloc[0]
    return {"name": res.name, "years": round(yrs, 1), "mult": mult, "cagr": mult ** (1 / yrs) - 1,
            "hold_mult": hmult, "hold_cagr": hmult ** (1 / yrs) - 1, "ratio_to_hold": mult / hmult,
            "mdd": res.metrics.get("mdd"), "hold_mdd": hold.metrics.get("mdd"),
            "trades": res.trades, "tax": res.tax, "fees": res.fees, "time_in_market": res.time_in_market,
            "avg_exposure": float(res.exposure.mean())}


def yearly(res: LabResult) -> pd.Series:
    e = res.equity
    return e.groupby(e.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)


def rolling(lev_close: pd.Series, signal_close: pd.Series, p: VtParams, costs: CostParams, cash_rate,
            years: int, first_start: date, step_months: int = 3) -> pd.DataFrame:
    idx = lev_close.index
    starts = pd.date_range(first_start, idx[-1] - pd.DateOffset(years=years), freq=f"{step_months}MS")
    w_full = target_weights(lev_close, signal_close, p)
    rows = []
    for s in starts:
        e = s + pd.DateOffset(years=years)
        if e > idx[-1]:
            break
        lev = lev_close[(lev_close.index >= s) & (lev_close.index <= e)]
        if len(lev) < 50:
            continue
        hold = simulate_exposure(lev, pd.Series(1.0, index=lev.index), costs, cash_rate, "hold")
        st = simulate_exposure(lev, w_full.reindex(lev.index).ffill().fillna(0.0), costs, cash_rate, p.name)
        hm = hold.final_after_liquidation / hold.equity.iloc[0]; sm = st.final_after_liquidation / st.equity.iloc[0]
        rows.append({"start": s.date(), "hold_mult": hm, "strat_mult": sm, "ratio": sm / hm,
                     "hold_cagr": hm ** (1 / years) - 1, "strat_cagr": sm ** (1 / years) - 1,
                     "hold_mdd": hold.metrics.get("mdd"), "strat_mdd": st.metrics.get("mdd")})
    return pd.DataFrame(rows)


def rolling_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {"windows": int(len(df)), "win_rate": float((df["ratio"] > 1).mean()), "ratio_median": float(df["ratio"].median()),
            "ratio_min": float(df["ratio"].min()), "strat_cagr_median": float(df["strat_cagr"].median()),
            "strat_cagr_min": float(df["strat_cagr"].min()), "hold_cagr_median": float(df["hold_cagr"].median()),
            "hold_cagr_min": float(df["hold_cagr"].min()), "strat_mdd_median": float(df["strat_mdd"].median()),
            "hold_mdd_median": float(df["hold_mdd"].median()), "strat_loss_rate": float((df["strat_mult"] < 1).mean()),
            "hold_loss_rate": float((df["hold_mult"] < 1).mean())}
