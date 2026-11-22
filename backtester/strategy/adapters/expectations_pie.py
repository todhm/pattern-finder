"""모부신·래퍼포트 『예측투자(Expectations Investing)』 — 주가에 담긴 기대(PIE) 역산과 기저율 대조 (순수 pandas/numpy).

체크리스트 `docs/strategy_notes/모부신_예측투자_고평가제외_체크리스트_2026-11.md`.
기업가치(EV)를 맞추는 10년 매출 성장률을 역산(`implied_growth`)하고, 그 성장률 이상을 실제로 해낸 회사의 비율을
Base Rate Book(Credit Suisse 2016, 표 4: 출발 매출 크기별 10년 실질 매출 CAGR 분포)에서 읽는다(`base_rate_tail`).
책은 예측 기간(MIFP)을 풀지만 과거 시점 컨센서스가 없어 기간을 고정하고 성장률을 푼다 — 책 그대로가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Base Rate Book 표 4 — 10년 실질 매출 CAGR 분포(%). 칸: <-25, -25~-20, …, 40~45, >45 (16칸). 구간 상한은 2015년 달러 $M.
_EDGES = np.array([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=float) / 100.0
BASE_RATE_10Y: list[tuple[float, str, list[float]]] = [
    (325, "$0~325M", [0.0, 0.0, 0.2, 0.5, 0.7, 3.5, 16.7, 29.3, 20.4, 10.5, 6.2, 4.2, 2.7, 1.8, 1.1, 2.1]),
    (700, "$325~700M", [0.0, 0.0, 0.1, 0.5, 1.6, 5.8, 24.3, 32.1, 14.9, 5.2, 2.5, 1.1, 0.7, 0.2, 0.1, 0.1]),
    (1250, "$700~1,250M", [0.2, 0.1, 0.2, 0.8, 2.2, 9.2, 31.9, 32.1, 14.8, 4.9, 2.0, 1.1, 0.3, 0.1, 0.0, 0.0]),
    (2000, "$1,250~2,000M", [0.0, 0.1, 0.4, 0.7, 1.8, 9.8, 36.8, 31.2, 12.3, 4.0, 1.6, 0.7, 0.3, 0.1, 0.2, 0.1]),
    (3000, "$2,000~3,000M", [0.0, 0.1, 0.1, 0.4, 2.6, 12.1, 40.0, 28.8, 9.8, 4.1, 1.1, 0.8, 0.1, 0.1, 0.1, 0.0]),
    (4500, "$3,000~4,500M", [0.0, 0.0, 0.0, 0.8, 2.6, 15.0, 40.1, 28.1, 8.7, 3.2, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0]),
    (7000, "$4,500~7,000M", [0.0, 0.1, 0.1, 1.0, 4.1, 15.5, 40.8, 26.5, 7.7, 2.7, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0]),
    (12000, "$7,000~12,000M", [0.0, 0.1, 0.6, 1.0, 4.4, 18.5, 40.8, 25.1, 6.3, 2.1, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0]),
    (25000, "$12,000~25,000M", [0.0, 0.1, 0.5, 1.4, 5.7, 20.2, 41.7, 20.8, 6.3, 2.5, 0.6, 0.1, 0.1, 0.0, 0.0, 0.1]),
    (float("inf"), "$25,000M 초과", [0.1, 0.2, 0.8, 2.5, 9.0, 22.9, 37.1, 20.2, 5.8, 1.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
]
CPI_FROM_2015 = {2021: 1.17, 2022: 1.26, 2023: 1.31, 2024: 1.34, 2025: 1.36, 2026: 1.40}     # [가정] 미국 소비자물가 대략값(2015 = 1)


@dataclass(frozen=True)
class PieParams:
    wacc: float = 0.08                  # 자본비용 (호출 쪽이 기준일 10년 국채 + 5%p 로 넣는다)
    tax: float = 0.21                   # 현금 세율 [가정]
    inflation: float = 0.02             # 기대 물가 [가정]
    years: int = 10                     # 예측 기간 [가정] — 기저율 표의 가장 긴 구간
    min_margin: float = 0.05            # 이익률이 이보다 낮으면 섹터 중앙값으로 [가정]
    max_invest: float = 0.50            # 증분 투자율 상한 [가정]
    exclude_below: float = 0.05         # 기저율 P 가 이보다 낮으면 '지나친 고평가' [가정]


def dcf_value(s0: float, g, margin: float, invest: float, p: PieParams) -> np.ndarray:
    """매출 성장률 g(스칼라·배열)일 때의 기업가치. 잔존가치 = 마지막 해 세후 영업이익의 물가 반영 영구가치(그 뒤 투자는 자본비용만 번다)."""
    g = np.atleast_1d(np.asarray(g, dtype=float))[:, None]
    t = np.arange(1, p.years + 1)[None, :]
    sales = s0 * (1 + g) ** t
    prev = s0 * (1 + g) ** (t - 1)
    nopat = sales * margin * (1 - p.tax)
    fcf = nopat - (sales - prev) * invest
    disc = (1 + p.wacc) ** t
    cv = nopat[:, -1] * (1 + p.inflation) / (p.wacc - p.inflation)
    return (fcf / disc).sum(axis=1) + cv / (1 + p.wacc) ** p.years


_GRID = np.arange(-0.20, 1.0001, 0.0025)


def implied_growth(ev: float, s0: float, margin: float, invest: float, p: PieParams = PieParams()) -> float:
    """EV 를 맞추는 10년 명목 매출 성장률. 격자 안에서 처음 넘는 지점(직선 보간). 범위 밖이면 ±inf, 계산 불가면 nan."""
    if not all(np.isfinite([ev, s0, margin, invest])) or s0 <= 0 or margin <= 0 or ev <= 0:
        return float("nan")
    v = dcf_value(s0, _GRID, margin, invest, p)
    if v[0] >= ev:
        return float("-inf")
    hit = np.nonzero(v >= ev)[0]
    if not len(hit):
        return float("inf")
    i = int(hit[0])
    return float(_GRID[i - 1] + (ev - v[i - 1]) / (v[i] - v[i - 1]) * (_GRID[i] - _GRID[i - 1]))


def size_bucket(sales_musd_2015: float) -> int:
    return next(i for i, (hi, _, _) in enumerate(BASE_RATE_10Y) if sales_musd_2015 < hi)


def base_rate_tail(real_growth: float, sales_musd_2015: float) -> float:
    """그 크기 구간에서 10년 실질 매출 CAGR 이 real_growth 이상이었던 회사의 비율(0~1). 5%p 칸 안은 직선 보간."""
    if not np.isfinite(real_growth):
        return 0.0 if real_growth == float("inf") else (1.0 if real_growth == float("-inf") else float("nan"))
    dist = np.array(BASE_RATE_10Y[size_bucket(sales_musd_2015)][2], dtype=float) / 100.0     # 정규화하지 않는다: 원문 $325~700M 열은 합이 89.2%로 인쇄돼 있고(가운데 칸 오식으로 보임) 꼬리 칸은 그대로 쓴다
    if real_growth < _EDGES[0]:
        return 1.0
    if real_growth >= _EDGES[-1]:
        return float(dist[-1]) * 0.5                                  # 열린 꼬리 칸(>45%): 절반으로 본다
    k = int(np.searchsorted(_EDGES, real_growth, side="right"))       # real_growth 가 든 칸 = dist[k], 구간 [_EDGES[k-1], _EDGES[k])
    frac = (_EDGES[k] - real_growth) / (_EDGES[k] - _EDGES[k - 1])
    return float(dist[k] * frac + dist[k + 1:].sum())


def _ttm(g: pd.DataFrame, col: str, k: int = 0) -> float:
    n = len(g)
    return float(g[col].iloc[n - 4 - k: n - k].sum(min_count=4)) if n - k >= 4 else float("nan")


def fundamentals(stmts_asof: pd.DataFrame) -> pd.DataFrame:
    """기준일까지 발표된 분기 재무제표 → 종목별 역산 입력(S₀·GAAP 영업이익률·증분 투자율·순차입금·주식 수)."""
    rows = {}
    for sym, g in stmts_asof.groupby("symbol"):
        g = g.sort_values("fiscal_date")
        rev0, rev3 = _ttm(g, "revenue"), _ttm(g, "revenue", 12)
        oi = _ttm(g, "operating_income")
        nwc = (g["total_current_assets"] - g["cash_and_st_investments"].fillna(0)) - (g["total_current_liabilities"] - g["short_term_debt"].fillna(0))
        d_nwc = float(nwc.iloc[-1] - nwc.iloc[-13]) if len(g) >= 13 and np.isfinite(nwc.iloc[-1]) and np.isfinite(nwc.iloc[-13]) else 0.0
        net_capex = float((g["capex"].abs() - g["da"].fillna(0)).iloc[-12:].sum(min_count=8)) if len(g) >= 12 else float("nan")
        d_rev = rev0 - rev3 if np.isfinite(rev3) else float("nan")
        last = lambda c: float(g[c].dropna().iloc[-1]) if g[c].notna().any() else 0.0  # noqa: E731
        rows[sym] = {"rev_ttm": rev0, "op_margin": oi / rev0 if np.isfinite(oi) and rev0 > 0 else np.nan,
                     "invest_rate": (net_capex + d_nwc) / d_rev if np.isfinite(net_capex) and np.isfinite(d_rev) and d_rev > 0 else np.nan,
                     "net_debt": last("total_debt") - last("cash_and_st_investments"), "shares": last("shares_outstanding"),
                     "sbc_to_rev": _ttm(g, "sbc") / rev0 if rev0 > 0 else np.nan, "sector": g["sector"].dropna().iloc[-1] if g["sector"].notna().any() else None}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("symbol")


def pie_table(f: pd.DataFrame, mcap: pd.Series, year: int, p: PieParams = PieParams()) -> pd.DataFrame:
    """`fundamentals` 결과 + 시가총액 → 내재 성장률·기저율 P·제외 여부. 섹터 중앙값은 넘겨받은 f 안에서 구한다."""
    d = f.copy()
    d["mcap"] = mcap.reindex(d.index)
    d["ev"] = d["mcap"] + d["net_debt"]
    sec_m = d[d["op_margin"] > 0].groupby("sector")["op_margin"].median()
    inv = d["invest_rate"].clip(0, p.max_invest)
    sec_k = inv.groupby(d["sector"]).median()
    low = ~(d["op_margin"] >= p.min_margin)
    d["margin_used"] = np.where(low, d["sector"].map(sec_m), d["op_margin"])
    d["margin_from_sector"] = low
    d["invest_used"] = inv.fillna(d["sector"].map(sec_k)).fillna(float(inv.median()) if inv.notna().any() else 0.0)
    d["g_implied"] = [implied_growth(ev, s, m, k, p) for ev, s, m, k in zip(d["ev"], d["rev_ttm"], d["margin_used"], d["invest_used"])]
    d["g_real"] = (1 + d["g_implied"]) / (1 + p.inflation) - 1
    cpi = CPI_FROM_2015.get(year, 1.0)
    d["sales_2015_musd"] = d["rev_ttm"] / 1e6 / cpi
    ok = d["sales_2015_musd"].notna() & d["g_implied"].notna()
    d["bucket"] = [BASE_RATE_10Y[size_bucket(s)][1] if o else None for s, o in zip(d["sales_2015_musd"], ok)]
    d["bucket_median"] = [_bucket_median(s) if o else np.nan for s, o in zip(d["sales_2015_musd"], ok)]
    d["base_rate_p"] = [base_rate_tail(g, s) if o else np.nan for g, s, o in zip(d["g_real"], d["sales_2015_musd"], ok)]
    d["excluded"] = d["base_rate_p"] < p.exclude_below
    d["ev_to_sales"] = d["ev"] / d["rev_ttm"]
    return d


_MEDIANS = [9.8, 6.7, 5.7, 5.1, 4.5, 4.1, 3.7, 3.2, 2.7, 1.8]           # 표 4 의 10년 중앙값(%)


def _bucket_median(sales_musd_2015: float) -> float:
    return _MEDIANS[size_bucket(sales_musd_2015)] / 100.0
