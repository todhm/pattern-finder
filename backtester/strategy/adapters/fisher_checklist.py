"""필립 피셔 『위대한 기업에 투자하라』 15 포인트 — 수치로 계산되는 부분과 집계 (순수 pandas).

체크리스트 `docs/strategy_notes/피셔_위대한기업_체크리스트_2026-11.md` 의 [자동] 항목:
포인트 1(매출 성장)·3(R&D 회수)·5(이익률)·6(이익률 추세)·12(재투자)·13(희석·자금)을 **기준일까지 발표된 분기 재무제표만으로** 계산한다.
문서 판독(15 포인트 전부)은 `fisher_judge.py`, 파이프라인은 `fisher_cli.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXCLUDED_SECTORS = ("FINANCIAL SERVICES", "REAL ESTATE")     # [가정] 15 포인트(R&D·영업 조직·영업이익률)가 맞지 않는 업종
POINT_NAMES = {1: "시장 잠재력(매출 성장)", 2: "신제품 개발 의지", 3: "R&D 효과", 4: "영업 조직", 5: "이익률", 6: "이익률 유지·개선 노력",
               7: "노사·인사 관계", 8: "임원 관계", 9: "경영진의 깊이", 10: "원가·회계 관리", 11: "업종 특유의 우위", 12: "장기 시각",
               13: "증자·희석", 14: "솔직함", 15: "정직성"}
QUANT_POINTS = (1, 3, 5, 6, 12, 13)


def asof(stmts: pd.DataFrame, as_of) -> pd.DataFrame:
    """기준일까지 **발표된** 분기만. 발표일이 없으면 분기 말 + 60일로 본다(보수적)."""
    d = stmts.copy()
    d["fiscal_date"] = pd.to_datetime(d["fiscal_date"])
    rep = pd.to_datetime(d["report_date"]).fillna(d["fiscal_date"] + pd.Timedelta(days=60))
    return d[rep <= pd.Timestamp(as_of)].sort_values(["symbol", "fiscal_date"]).drop_duplicates(["symbol", "fiscal_date"], keep="last")


def _split_adjusted_shares(sh: pd.Series) -> pd.Series:
    """분기 사이 주식 수가 40% 넘게 뛰면 분할로 보고 앞쪽을 비율만큼 맞춘다(Alpha Vantage 주식 수는 분할 미조정)."""
    s = sh.astype(float).copy()
    v = s.to_numpy(copy=True)
    for i in range(len(v) - 1, 0, -1):
        if np.isfinite(v[i]) and np.isfinite(v[i - 1]) and v[i - 1] > 0:
            r = v[i] / v[i - 1]
            if r > 1.4 or r < 1 / 1.4:
                v[:i] = v[:i] * r
    return pd.Series(v, index=s.index)


def _one(g: pd.DataFrame) -> dict:
    g = g.sort_values("fiscal_date")
    n = len(g)
    ttm = lambda c, k=0: float(g[c].iloc[n - 4 - k: n - k].sum(min_count=4)) if n - k >= 4 else np.nan  # noqa: E731
    last = lambda c: float(g[c].dropna().iloc[-1]) if g[c].notna().any() else np.nan  # noqa: E731
    out: dict = {"n_quarters": n, "last_fiscal": g["fiscal_date"].iloc[-1], "sector": g["sector"].dropna().iloc[-1] if g["sector"].notna().any() else None}
    rev = {k: ttm("revenue", k) for k in (0, 4, 8, 12, 16, 20)}
    out["rev_ttm"] = rev[0]
    cagr = lambda yrs: (rev[0] / rev[4 * yrs]) ** (1 / yrs) - 1 if rev[0] > 0 and np.isfinite(rev[4 * yrs]) and rev[4 * yrs] > 0 else np.nan  # noqa: E731
    out["rev_cagr3"], out["rev_cagr5"] = cagr(3), cagr(5)
    out["rev_yoy"] = rev[0] / rev[4] - 1 if np.isfinite(rev[4]) and rev[4] > 0 else np.nan
    yoy = [rev[k] / rev[k + 4] - 1 for k in (0, 4, 8, 12, 16) if np.isfinite(rev[k]) and np.isfinite(rev[k + 4]) and rev[k + 4] > 0]
    out["growth_years"] = int(sum(y > 0 for y in yoy)); out["growth_years_of"] = len(yoy)
    ta = last("total_assets")
    out["goodwill_share"] = last("goodwill") / ta if np.isfinite(ta) and ta > 0 and np.isfinite(last("goodwill")) else np.nan
    # 이익률
    oi = {k: ttm("operating_income", k) for k in (0, 4, 8, 12)}; gp = {k: ttm("gross_profit", k) for k in (0, 12)}
    om = {k: oi[k] / rev[k] if np.isfinite(oi[k]) and np.isfinite(rev[k]) and rev[k] > 0 else np.nan for k in (0, 4, 8, 12)}
    out["om_ttm"] = om[0]; out["om_avg3"] = float(np.nanmean([om[0], om[4], om[8]])) if any(np.isfinite([om[0], om[4], om[8]])) else np.nan
    out["om_chg3"] = om[0] - om[12] if np.isfinite(om[0]) and np.isfinite(om[12]) else np.nan
    gm0 = gp[0] / rev[0] if np.isfinite(gp[0]) and rev[0] > 0 else np.nan; gm3 = gp[12] / rev[12] if np.isfinite(gp[12]) and np.isfinite(rev[12]) and rev[12] > 0 else np.nan
    out["gm_ttm"] = gm0; out["gm_chg3"] = gm0 - gm3 if np.isfinite(gm0) and np.isfinite(gm3) else np.nan
    # R&D
    rnd0 = ttm("rnd"); rnd3 = float(g["rnd"].iloc[-12:].sum(min_count=8)) if n >= 12 else np.nan
    out["rnd_to_rev"] = rnd0 / rev[0] if np.isfinite(rnd0) and rev[0] > 0 else np.nan
    out["rnd_payoff"] = (gp[0] - gp[12]) / rnd3 if np.isfinite(rnd3) and rnd3 > 0 and np.isfinite(gp[0]) and np.isfinite(gp[12]) else np.nan   # 3년간 늘어난 매출총이익 ÷ 3년간 쓴 R&D [가정]
    rnd1 = ttm("rnd", 4)
    out["rnd_yoy"] = rnd0 / rnd1 - 1 if np.isfinite(rnd0) and np.isfinite(rnd1) and rnd1 > 0 else np.nan
    # 판관비 효율
    sga0, sga3 = ttm("sga"), ttm("sga", 12)
    out["rev_per_sga_chg"] = (rev[0] - rev[12]) / (sga0 - sga3) if np.isfinite(sga0) and np.isfinite(sga3) and sga0 - sga3 > 0 and np.isfinite(rev[12]) else np.nan
    # 재투자·현금
    ocf, capex = ttm("operating_cf"), abs(ttm("capex")) if np.isfinite(ttm("capex")) else np.nan
    out["fcf_ttm"] = ocf - capex if np.isfinite(ocf) and np.isfinite(capex) else np.nan
    out["fcf_margin"] = out["fcf_ttm"] / rev[0] if np.isfinite(out["fcf_ttm"]) and rev[0] > 0 else np.nan
    reinv = np.nansum([capex, rnd0]) if (np.isfinite(capex) or np.isfinite(rnd0)) else np.nan
    out["reinvest_to_rev"] = reinv / rev[0] if np.isfinite(reinv) and rev[0] > 0 else np.nan
    # 희석·부채
    sh = _split_adjusted_shares(g["shares_outstanding"])
    out["sh_chg3_ann"] = (sh.iloc[-1] / sh.iloc[-13]) ** (1 / 3) - 1 if n >= 13 and np.isfinite(sh.iloc[-1]) and np.isfinite(sh.iloc[-13]) and sh.iloc[-13] > 0 else np.nan
    sbc = ttm("sbc"); out["sbc_to_rev"] = sbc / rev[0] if np.isfinite(sbc) and rev[0] > 0 else np.nan
    cash, debt, ebitda = last("cash_and_st_investments"), last("total_debt"), ttm("ebitda")
    out["net_cash"] = (cash if np.isfinite(cash) else 0.0) - (debt if np.isfinite(debt) else 0.0)
    out["netdebt_to_ebitda"] = -out["net_cash"] / ebitda if np.isfinite(ebitda) and ebitda > 0 else np.nan
    return out


def quant_features(stmts_asof: pd.DataFrame) -> pd.DataFrame:
    rows = {sym: _one(g) for sym, g in stmts_asof.groupby("symbol")}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("symbol")


def _pct_in_sector(f: pd.DataFrame, col: str) -> pd.Series:
    return f.groupby("sector")[col].rank(pct=True)


def quant_scores(f: pd.DataFrame) -> pd.DataFrame:
    """수치 포인트 6개의 0~3 점수와 평균 `quant_avg`. 문턱은 체크리스트 §3 의 [가정]."""
    s = pd.DataFrame(index=f.index)
    g3, g5 = f["rev_cagr3"], f["rev_cagr5"].fillna(f["rev_cagr3"])
    lo = np.minimum(g3, g5)
    consistent = f["growth_years"] >= (f["growth_years_of"] - 1).clip(lower=1)
    p1 = np.select([lo >= 0.15, lo >= 0.08, lo > 0.0], [3, 2, 1], default=0).astype(float)
    p1 = np.where((p1 == 3) & ~consistent, 2, p1)            # 한두 해 급등이 전부면 3점을 주지 않는다
    s["p1"] = np.where(g3.isna(), np.nan, p1)
    pay = _pct_in_sector(f, "rnd_payoff")
    p3 = np.select([(pay >= 0.75) & (f["rnd_yoy"] > -0.10), pay >= 0.5, pay >= 0.2], [3, 2, 1], default=0).astype(float)
    s["p3"] = np.where(f["rnd_payoff"].isna() | (f["rnd_to_rev"].fillna(0) < 0.01), np.nan, p3)     # R&D 가 거의 없는 업종은 해당 없음
    omr = _pct_in_sector(f, "om_avg3")
    p5 = np.select([(omr >= 0.75) & (f["om_ttm"] > 0), (omr >= 0.5) & (f["om_ttm"] > 0), f["om_ttm"] > 0], [3, 2, 1], default=0).astype(float)
    s["p5"] = np.where(f["om_ttm"].isna(), np.nan, p5)
    dom, dgm = f["om_chg3"], f["gm_chg3"].fillna(0)
    p6 = np.select([(dom >= 0.03) & (dgm >= 0), dom >= -0.01, dom >= -0.04], [3, 2, 1], default=0).astype(float)
    s["p6"] = np.where(dom.isna(), np.nan, p6)
    rr = _pct_in_sector(f, "reinvest_to_rev")
    p12 = np.select([(rr >= 0.6) & (f["rnd_yoy"].fillna(0) > -0.05), rr >= 0.3], [2, 1], default=0).astype(float)     # 숫자만으로는 2 가 상한
    s["p12"] = np.where(f["reinvest_to_rev"].isna(), np.nan, p12)
    dil = f["sh_chg3_ann"]; fcf_ok = f["fcf_ttm"] > 0; lev = f["netdebt_to_ebitda"]
    p13 = np.select([(dil <= 0.0) & fcf_ok & (f["net_cash"] >= 0), (dil <= 0.02) & fcf_ok & ~(lev > 3), (dil <= 0.05) & ~(lev > 4)], [3, 2, 1], default=0).astype(float)
    s["p13"] = np.where(dil.isna(), np.nan, p13)
    s["quant_n"] = s.notna().sum(axis=1)
    s["quant_avg"] = s[[f"p{k}" for k in QUANT_POINTS]].mean(axis=1)
    return s


def screen(f: pd.DataFrame, min_rev: float = 1e8, min_quarters: int = 16) -> pd.Series:
    """'선전 단계 회사' 제외(매출 이력·규모) + 업종 제외. 참 = 검토 대상."""
    return (f["rev_ttm"] >= min_rev) & (f["n_quarters"] >= min_quarters) & ~f["sector"].isin(EXCLUDED_SECTORS) & f["sector"].notna()


# ---------------------------------------------------------------------------
# 판독 결과 집계
# ---------------------------------------------------------------------------
def fisher_verdict(scores: dict[int, float | None], promotional: bool = False, min_avg: float = 2.0, max_zero: int = 2, max_unknown: int = 5) -> dict:
    """15 포인트 점수(모름 = None) → 평균·통과 여부. 포인트 15 가 0 이면 탈락(거부권)."""
    known = {k: v for k, v in scores.items() if v is not None}
    avg = float(np.mean(list(known.values()))) if known else float("nan")
    zeros = [k for k, v in known.items() if v == 0]
    unknown = [k for k in range(1, 16) if scores.get(k) is None]
    why = []
    if scores.get(15) == 0:
        why.append("포인트 15(정직성) 0점 — 거부권")
    if promotional:
        why.append("선전 단계 회사")
    if not known or avg < min_avg:
        why.append(f"평균 {avg:.2f} < {min_avg}")
    if len(zeros) > max_zero:
        why.append(f"0점 {len(zeros)}개")
    if len(unknown) > max_unknown:
        why.append(f"모름 {len(unknown)}개")
    return {"avg": avg, "n_known": len(known), "zeros": zeros, "unknown": unknown, "passed": not why, "fail_reasons": why}


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 5:
        return float("nan"), len(d)
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())), len(d)
