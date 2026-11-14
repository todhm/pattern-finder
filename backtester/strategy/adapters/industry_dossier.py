"""산업 도시에(구조 카드) — 그 시점까지 SEC 에 제출된 숫자만으로 산업의 '성장의 질과 구조' 를 한 장으로 (순수 pandas).

filing_summary : 제출물 한 건 → 흐름 항목(매출·이익·현금흐름…)의 이번 기간/전년 같은 기간, 잔액 항목(재고·이연 매출…)의 최신 값
snapshot       : 기준일 T 에 회사별 최신 제출물(190 일 안)
card           : 산업 한 곳의 카드(dict) — 성장·마진·증분 마진·집중도·진입·희석·선수금·재고·투자, 전부 비율
render         : 카드를 사람이 읽는 글로(이름·연도·금액 없음 → 가린 판정용)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLOW = ("rev", "gp", "cogs", "op", "ni", "rnd", "capex", "cfo", "sbc", "issue", "buyback", "dsh")
STOCK = ("inv", "defrev", "ar", "cash", "debt", "assets", "eq", "gw")
ID = ["cik", "name", "sic", "form", "filed"]


def filing_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """raw(cik, name, sic, form, filed, item, ddate, qtrs, value) → 제출물별 한 줄. 흐름 항목은 같은 길이(qtrs)의 1 년 전 값이 같이 실린 가장 긴 기간을 쓴다."""
    f = raw[raw["item"].isin(FLOW) & (raw["qtrs"] > 0)]; k = ["cik", "filed", "item"]
    cur = f[f["ddate"] == f.groupby(k)["ddate"].transform("max")]
    m = cur.merge(f[k + ["qtrs", "ddate", "value"]], on=k + ["qtrs"], suffixes=("", "_p")); gap = (m["ddate"] - m["ddate_p"]).dt.days; m = m[(gap >= 350) & (gap <= 380)]
    m = m.sort_values("qtrs").groupby(k, as_index=False).tail(1)
    a = m.pivot_table(index=ID, columns="item", values=["value", "value_p"], aggfunc="first"); a.columns = [f"{i}_{'cur' if v == 'value' else 'prev'}" for v, i in a.columns]
    s = raw[raw["item"].isin(STOCK) & (raw["qtrs"] == 0)]; s = s[s["ddate"] == s.groupby(k)["ddate"].transform("max")]
    b = s.pivot_table(index=ID, columns="item", values="value", aggfunc="first"); b.columns = [f"{i}_bal" for i in b.columns]
    return a.join(b, how="outer").reset_index()


def snapshot(fs: pd.DataFrame, t, fresh_days: int = 190) -> pd.DataFrame:
    x = fs[(fs["filed"] <= t) & (fs["filed"] > pd.Timestamp(t) - pd.Timedelta(days=fresh_days))].sort_values("filed").groupby("cik").tail(1)
    return x.set_index("cik")


def _g(a, b):
    """합계 비율 − 1. 둘 다 있고 분모가 양수인 회사만."""
    ok = a.notna() & b.notna() & (b > 0)
    return float(a[ok].sum() / b[ok].sum() - 1.0) if ok.sum() >= 3 else np.nan


def _col(x: pd.DataFrame, c: str) -> pd.Series:
    return x[c] if c in x else pd.Series(np.nan, index=x.index)


def card(now: pd.DataFrame, y1: pd.DataFrame, y2: pd.DataFrame) -> dict:
    """now·y1·y2 = 같은 산업의 T·T−1 년·T−2 년 snapshot. 값이 모자라면 NaN."""
    c = lambda n: _col(now, n)  # noqa: E731
    rev, prev = c("rev_cur"), c("rev_prev"); ok = rev.notna() & (prev > 0); o: dict = {"n_firms": int(len(now)), "n_firms_1y": int(len(y1)), "n_firms_2y": int(len(y2))}
    if ok.sum() < 5:
        return o
    g = (rev / prev - 1.0)[ok]; tot = rev[ok].sum(); top = rev[ok].nlargest(3).index; rest = g.drop(top)
    gp, gpp = c("gp_cur").fillna(rev - c("cogs_cur")), c("gp_prev").fillna(prev - c("cogs_prev")); okg = ok & gp.notna() & gpp.notna(); op, opp = c("op_cur"), c("op_prev"); oko = ok & op.notna() & opp.notna()
    o.update({"rev_g": float(tot / prev[ok].sum() - 1), "rev_g_median": float(g.median()), "pct_growing": float((g > 0).mean()), "pct_hot": float((g > 0.25).mean()), "top3_share": float(rev[top].sum() / tot),
              "leaders_g": float(rev[top].sum() / prev[top].sum() - 1), "rest_g_median": float(rest.median()) if len(rest) >= 3 else np.nan, "new_filer_share": float((~now.index.isin(y2.index)).mean()) if len(y2) else np.nan})
    if okg.sum() >= 3:
        o["gm"] = float(gp[okg].sum() / rev[okg].sum()); o["gm_chg"] = o["gm"] - float(gpp[okg].sum() / prev[okg].sum())
    if oko.sum() >= 3:
        dr = rev[oko].sum() - prev[oko].sum(); o["opm"] = float(op[oko].sum() / rev[oko].sum()); o["opm_chg"] = o["opm"] - float(opp[oko].sum() / prev[oko].sum()); o["pct_profitable"] = float((op[oko] > 0).mean())
        o["incr_margin"] = float((op[oko].sum() - opp[oko].sum()) / dr) if dr > 0 and dr / prev[oko].sum() > 0.02 else np.nan
    for it in ("capex", "rnd", "sbc", "issue", "buyback", "cfo"):
        v = c(f"{it}_cur"); k2 = ok & v.notna()
        if k2.sum() >= 3:
            o[f"{it}_to_rev"] = float(v[k2].sum() / rev[k2].sum())
    o["capex_g"] = _g(c("capex_cur"), c("capex_prev")); o["rnd_g"] = _g(c("rnd_cur"), c("rnd_prev")); sh = (c("dsh_cur") / c("dsh_prev") - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    o["share_g_median"] = float(sh.median()) if len(sh) >= 3 else np.nan; o["pct_diluting"] = float((sh > 0.05).mean()) if len(sh) >= 3 else np.nan
    cap, cfo = c("capex_cur"), c("cfo_cur"); k3 = cap.notna() & cfo.notna(); o["cfo_covers_capex"] = float(cfo[k3].sum() / cap[k3].sum()) if k3.sum() >= 3 and cap[k3].sum() > 0 else np.nan
    both = now.index.intersection(y1.index)                                                         # 잔액 항목은 1 년 전 제출물과 비교(같은 회사만)
    for it in ("inv", "defrev", "ar"):
        a, b = _col(now, f"{it}_bal").reindex(both), _col(y1, f"{it}_bal").reindex(both); o[f"{it}_g"] = _g(a, b)
    if len(y1):
        r1, p1 = _col(y1, "rev_cur"), _col(y1, "rev_prev"); k4 = r1.notna() & (p1 > 0); o["rev_g_1y_ago"] = float(r1[k4].sum() / p1[k4].sum() - 1) if k4.sum() >= 5 else np.nan
    return o


def _p(v, d=0, sign=True):
    return "자료 없음" if v is None or not np.isfinite(v) else (f"{v:+.{d}%}" if sign else f"{v:.{d}%}")


def render(c: dict, label: str) -> str:
    """가린 카드: 이름·연도·금액 없음. 가격 줄은 price_12m/36m 가 있으면 붙는다."""
    g = lambda k: c.get(k, np.nan)  # noqa: E731
    L = [f"### 산업 {label}", f"- 회사 수: {g('n_firms'):.0f} 곳 (1 년 전 {g('n_firms_1y'):.0f}, 2 년 전 {g('n_firms_2y'):.0f}) · 2 년 사이 새로 들어온 회사 비율 {_p(g('new_filer_share'), sign=False)} · 상위 3 곳 매출 비중 {_p(g('top3_share'), sign=False)}",
         f"- 매출: 산업 합계 전년비 {_p(g('rev_g'))} (1 년 전에는 {_p(g('rev_g_1y_ago'))}) · 회사 중앙값 {_p(g('rev_g_median'))} · 매출이 는 회사 {_p(g('pct_growing'), sign=False)} · 25% 넘게 는 회사 {_p(g('pct_hot'), sign=False)}",
         f"- 확산: 상위 3 곳 합계 {_p(g('leaders_g'))} vs 나머지 중앙값 {_p(g('rest_g_median'))}",
         f"- 수익성: 매출총이익률 {_p(g('gm'), sign=False)} (전년 대비 {_p(g('gm_chg'), 1)}p) · 영업이익률 {_p(g('opm'), 1, False)} (전년 대비 {_p(g('opm_chg'), 1)}p) · 늘어난 매출 1 원당 영업이익 {_p(g('incr_margin'), sign=False)} · 영업흑자 회사 {_p(g('pct_profitable'), sign=False)}",
         f"- 투자: 설비투자 전년비 {_p(g('capex_g'))} (매출 대비 {_p(g('capex_to_rev'), 1, False)}) · 연구개발 전년비 {_p(g('rnd_g'))} (매출 대비 {_p(g('rnd_to_rev'), 1, False)}) · 영업현금 ÷ 설비투자 {g('cfo_covers_capex'):.1f} 배" if np.isfinite(g("cfo_covers_capex")) else f"- 투자: 설비투자 전년비 {_p(g('capex_g'))} (매출 대비 {_p(g('capex_to_rev'), 1, False)}) · 연구개발 전년비 {_p(g('rnd_g'))} (매출 대비 {_p(g('rnd_to_rev'), 1, False)})",
         f"- 자금: 주식 수 증가(중앙값) {_p(g('share_g_median'), 1)} · 주식 수 5% 넘게 늘린 회사 {_p(g('pct_diluting'), sign=False)} · 증자 유입 매출 대비 {_p(g('issue_to_rev'), 1, False)} · 자사주 매입 매출 대비 {_p(g('buyback_to_rev'), 1, False)} · 주식 보상 매출 대비 {_p(g('sbc_to_rev'), 1, False)}",
         f"- 수요 흔적: 선수금(이연 매출) 전년비 {_p(g('defrev_g'))} · 재고 전년비 {_p(g('inv_g'))} · 매출채권 전년비 {_p(g('ar_g'))}"]
    if "price_12m" in c:
        L.append(f"- 주가(산업 내 종목 중앙값): 지난 12 달 {_p(g('price_12m'))} · 지난 36 달 {_p(g('price_36m'))} · 유동 종목 {g('n_stocks'):.0f} 개")
    return "\n".join(L)


def firm_rows(now: pd.DataFrame, y1: pd.DataFrame, top: int = 5) -> list[dict]:
    """산업 안 매출 상위 top 곳의 회사별 줄: 매출 비중·성장(지금/1 년 전)·마진 변화·증분 마진·R&D·설비투자·선수금·주식 수·영업권(인수 흔적)."""
    c = lambda n: _col(now, n)  # noqa: E731
    rev, prev = c("rev_cur"), c("rev_prev"); ok = rev.notna() & (prev > 0); tot = rev[ok].sum(); out = []
    gp, gpp = c("gp_cur").fillna(rev - c("cogs_cur")), c("gp_prev").fillna(prev - c("cogs_prev")); r1 = _col(y1, "rev_cur") / _col(y1, "rev_prev").where(_col(y1, "rev_prev") > 0) - 1.0
    for cik in rev[ok].nlargest(top).index:
        v = lambda s_: float(s_.get(cik, np.nan)) if cik in s_.index else np.nan  # noqa: E731
        R, P = rev[cik], prev[cik]; d = R - P; bal = lambda it: (v(_col(now, f"{it}_bal")) / v(_col(y1, f"{it}_bal")) - 1.0) if cik in y1.index and v(_col(y1, f"{it}_bal")) > 0 else np.nan  # noqa: E731
        out.append({"name": str(now.at[cik, "name"]).title(), "share": float(R / tot), "rev_g": float(R / P - 1), "rev_g_1y": v(r1), "gm": v(gp) / R, "gm_chg": v(gp) / R - v(gpp) / P, "opm": v(c("op_cur")) / R, "opm_chg": v(c("op_cur")) / R - v(c("op_prev")) / P,
                    "incr": (v(c("op_cur")) - v(c("op_prev"))) / d if d / P > 0.02 else np.nan, "rnd": v(c("rnd_cur")) / R, "capex_g": v(c("capex_cur")) / v(c("capex_prev")) - 1.0 if v(c("capex_prev")) > 0 else np.nan,
                    "defrev_g": bal("defrev"), "inv_g": bal("inv"), "gw_g": bal("gw"), "sh_g": v(c("dsh_cur")) / v(c("dsh_prev")) - 1.0 if v(c("dsh_prev")) > 0 else np.nan})
    return out


def render_firms(rows: list[dict], names: bool = False) -> str:
    """회사별 줄(가린 판은 '회사 1' …). 영업권이 20% 넘게 늘면 인수 흔적으로 표시."""
    L = []
    for i, r in enumerate(rows):
        acq = " · 영업권 " + _p(r["gw_g"]) + "(인수 흔적)" if np.isfinite(r.get("gw_g", np.nan)) and r["gw_g"] > 0.2 else ""
        L.append(f"  - {r['name'] if names else f'회사 {i + 1}'} (산업 매출의 {_p(r['share'], sign=False)}): 매출 {_p(r['rev_g'])} (1 년 전 {_p(r['rev_g_1y'])}) · 총이익률 {_p(r['gm'], sign=False)} ({_p(r['gm_chg'], 1)}p) · 영업이익률 {_p(r['opm'], sign=False)} ({_p(r['opm_chg'], 1)}p)"
                 f" · 증분 이익 {_p(r['incr'], sign=False)} · R&D {_p(r['rnd'], sign=False)} · 설비투자 {_p(r['capex_g'])} · 선수금 {_p(r['defrev_g'])} · 재고 {_p(r['inv_g'])} · 주식 수 {_p(r['sh_g'], 1)}{acq}")
    return "\n".join(L)
