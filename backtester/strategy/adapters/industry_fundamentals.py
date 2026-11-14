"""산업별 '제출된 그대로' 성장 지표 — SEC 재무제표 자료의 한 제출물 안에 든 올해 값과 1 년 전 비교 값으로 전년비를 만든다(순수 pandas).

한 제출물(10-Q·10-K)에는 이번 기간과 전년 같은 기간이 같이 실린다 → 제출일에 공개된 숫자만으로 성장률이 나오고, 뒤의 재작성이 섞이지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLOW = ("rev", "op", "ni", "rnd", "capex")


def filing_growth(raw: pd.DataFrame) -> pd.DataFrame:
    """raw(cik, sic, form, filed, ddate, qtrs, item, value) → 제출물 한 줄: cik, sic, filed, {item}_cur, {item}_prev (기간 = 10-K 는 4 분기, 그 외 1 분기)."""
    d = raw[raw["item"].isin(FLOW)].copy(); want = np.where(d["form"].str.startswith(("10-K", "20-F", "40-F")), 4, 1); d = d[d["qtrs"] == want]
    key = ["cik", "filed"]; d = d.groupby(key + ["sic", "item", "ddate"], as_index=False)["value"].max()          # 같은 항목 태그 여럿이면 가장 큰 값(총매출)
    d["last"] = d.groupby(key + ["item"])["ddate"].transform("max"); cur = d[d["ddate"] == d["last"]]
    m = cur.merge(d[key + ["item", "ddate", "value"]], on=key + ["item"], suffixes=("", "_p")); gap = (m["ddate"] - m["ddate_p"]).dt.days; m = m[(gap >= 350) & (gap <= 380)]
    out = m.pivot_table(index=key + ["sic"], columns="item", values=["value", "value_p"], aggfunc="first")
    out.columns = [f"{i}_{'cur' if v == 'value' else 'prev'}" for v, i in out.columns]; return out.reset_index()


def sic_to_ind(sic: pd.Series, table: pd.DataFrame) -> pd.Series:
    """French 49 산업 SIC 구간표(lo, hi, ind) → 산업 이름. 어디에도 없으면 Other."""
    s = sic.to_numpy(dtype=float); out = np.full(len(s), "Other", dtype=object)
    for lo, hi, ind in table.itertuples(index=False):
        out[(s >= lo) & (s <= hi)] = ind
    return pd.Series(out, index=sic.index)


def _ratio(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna() & (b > 0)
    return float(a[ok].sum() / b[ok].sum() - 1.0) if ok.sum() >= 5 else np.nan


def industry_panel(fg: pd.DataFrame, month_ends: pd.DatetimeIndex, fresh_days: int = 190) -> dict[str, pd.DataFrame]:
    """월말 t 마다: filed ≤ t 이고 fresh_days 안인 회사별 최신 제출물만 모아 산업 지표. 이름 → (월말 × 산업) 표."""
    fg = fg.sort_values("filed"); rows = {}
    for t in month_ends:
        x = fg[(fg["filed"] <= t) & (fg["filed"] > t - pd.Timedelta(days=fresh_days))].groupby("cik").tail(1); r = {}
        for ind, g in x.groupby("ind"):
            ok = g["rev_cur"].notna() & (g["rev_prev"] > 0); fg_ = (g["rev_cur"] / g["rev_prev"] - 1.0)[ok]; has = lambda c: f"{c}_cur" in g  # noqa: E731
            r[ind] = {"g_agg": _ratio(g["rev_cur"], g["rev_prev"]), "g_med": float(fg_.median()) if len(fg_) >= 5 else np.nan, "hot_frac": float((fg_ > 0.25).mean()) if len(fg_) >= 5 else np.nan,
                      "shrink_frac": float((fg_ < 0).mean()) if len(fg_) >= 5 else np.nan, "filers": float(len(g)),
                      "rnd_g": _ratio(g["rnd_cur"], g["rnd_prev"]) if has("rnd") else np.nan, "capex_g": _ratio(g["capex_cur"], g["capex_prev"]) if has("capex") else np.nan,
                      "margin_chg": (float(g.loc[ok, "op_cur"].sum() / g.loc[ok, "rev_cur"].sum() - g.loc[ok, "op_prev"].sum() / g.loc[ok, "rev_prev"].sum()) if ok.sum() >= 5 and has("op") else np.nan)}
        rows[t] = pd.DataFrame(r).T
    names = ["g_agg", "g_med", "hot_frac", "shrink_frac", "filers", "rnd_g", "capex_g", "margin_chg"]
    return {n: pd.DataFrame({t: v[n] for t, v in rows.items() if len(v)}).T.reindex(month_ends) for n in names}
