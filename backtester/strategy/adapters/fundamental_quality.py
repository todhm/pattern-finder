"""재무제표 읽기 — 성장주 품질 체크리스트 (순수 pandas/numpy, 데이터 포트 없음).

설계서: docs/strategy_notes/성장주_보유_전략_설계_2026-10.md §11.
입력은 분기 재무제표 long 표(``QuarterlyStatement.model_dump()`` 행: symbol, fiscal_date, report_date, revenue … sbc).
한 분기 행마다 **그 분기까지의 TTM(최근 4분기 합)** 으로 12개 항목을 판정한다. 각 항목은 True/False/NaN(자료 없음).

    C1  매출 성장      매출 TTM YoY ≥ 15%
    C2  성장 지속      3년 매출 CAGR(TTM vs 12분기 전 TTM) ≥ 12%
    C3  이익 성장      순이익 TTM YoY ≥ 15% (적자→흑자 전환은 통과, 지금 적자는 실패)
    C4  마진의 질      매출총이익률 TTM ≥ 35% 그리고 1년 전 대비 −2%p 이상 안 줄음
    C5  영업 레버리지  영업이익률 TTM ≥ 8% 그리고 1년 전 대비 −1%p 이상 안 줄음
    C6  현금 뒷받침    영업현금흐름 TTM ÷ 순이익 TTM ≥ 0.8 (순이익 ≤ 0 이면 영업현금흐름 > 0)
    C7  잉여현금흐름   (영업현금흐름 − |capex|) TTM ÷ 매출 TTM ≥ 5%
    C8  운전자본 건전  매출채권 YoY ≤ 매출 YoY + 10%p 그리고 재고 YoY ≤ 매출 YoY + 10%p (분기 vs 전년 동기, 미너비니 2-9)
    C9  재무 건전      순차입(차입 − 현금) ≤ 0(순현금) 이면 통과, 아니면 순차입 ÷ EBITDA TTM ≤ 2.5 그리고 이자보상배율(EBIT÷이자) ≥ 5
    C10 희석 없음      유통주식 수 YoY ≤ +3%
    C11 재투자         (R&D + capex) TTM ÷ 매출 ≥ 10% 또는 R&D TTM YoY ≥ 15%
    C12 자본 효율      ROE TTM(순이익 ÷ 평균 자본) ≥ 15% (자본 ≤ 0 이면서 흑자면 자사주 소각형 → 통과)

q_avail = 판정 가능한 항목 수, q_score = 통과 ÷ 판정 가능 × 12 (0~12). 판정 가능 < min_available 이면 전략에서 '자료 부족' 으로 뺀다.
유효일 effective = 발표일 + lag_days (발표일 없으면 분기 말 + 90일). 소비자(growth_hold_cli)는 effective ≤ 그날 인 마지막 행을 쓴다.

[가정] 문턱값은 "성장주 재무제표를 사람이 읽을 때 보는 보통 기준" 으로 미리 정했고 데이터로 맞추지 않았다.
[함정] Alpha Vantage 재무제표는 **지금 시점에 재작성된 값**(재분류·정정 반영)이라 엄밀한 point-in-time 이 아니다. 발표일은 EARNINGS 의 것.
       유통주식 수는 소스 품질이 들쭉날쭉해 YoY 가 −50%~+200% 밖이면 자료 오류로 보고 NaN 처리.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

CHECKS: list[tuple[str, str, str]] = [            # (열 이름, 짧은 이름, 설명)
    ("c1_rev", "매출 성장", "매출 TTM YoY ≥ 15%"),
    ("c2_cagr", "성장 지속", "3년 매출 CAGR ≥ 12%"),
    ("c3_ni", "이익 성장", "순이익 TTM YoY ≥ 15% (적자→흑자 통과)"),
    ("c4_gm", "마진의 질", "매출총이익률 ≥ 35% & 1년 전 대비 ≥ −2%p"),
    ("c5_om", "영업 레버리지", "영업이익률 ≥ 8% & 1년 전 대비 ≥ −1%p"),
    ("c6_ocf", "현금 뒷받침", "영업CF ÷ 순이익 ≥ 0.8"),
    ("c7_fcf", "잉여현금흐름", "FCF 마진 ≥ 5%"),
    ("c8_wc", "운전자본 건전", "매출채권·재고 YoY ≤ 매출 YoY + 10%p"),
    ("c9_bal", "재무 건전", "순현금, 또는 순차입/EBITDA ≤ 2.5 & 이자보상 ≥ 5"),
    ("c10_dil", "희석 없음", "유통주식 YoY ≤ +3%"),
    ("c11_inv", "재투자", "(R&D+capex)/매출 ≥ 10% 또는 R&D YoY ≥ 15%"),
    ("c12_roe", "자본 효율", "ROE ≥ 15%"),
]
CHECK_COLS = [c for c, _, _ in CHECKS]
AUX_COLS = ["q_score", "q_avail", "q_rev", "q_cagr3", "q_ni", "q_gm", "q_dgm", "q_om", "q_dom", "q_ocfni", "q_fcfm", "q_roe",
            "q_netcash", "q_nodil", "q_rnd", "q_lowsbc",                        # growth_hold_cli aux 로 올라가는 열(높을수록 좋음)
            "accruals", "sbc_fcfm", "dso_chg", "gw_share", "roic", "inc_roic", "om_peak_frac", "cur_ratio", "q_rev_accel",
            "roa", "cfo_assets", "roa_var", "sg_var", "rnd_assets", "capex_assets", "sbc_rev", "sh_chg", "nd_ebitda", "netdebt", "ocf_ttm", "ni_ttm", "f_score"]   # 고급 판(§12.2)·G/F 용
CYCLICAL = {"ENERGY", "BASIC MATERIALS", "INDUSTRIALS", "CONSUMER CYCLICAL"}
EXCLUDED_SECTORS = {"FINANCIAL SERVICES", "REAL ESTATE", "UTILITIES"}
ADV_CHECKS = [("a1_relgrowth", "업종 상대 성장", "매출 TTM YoY 섹터 백분위 ≥ 0.67"), ("a2_accel", "성장 가속", "분기 매출 YoY 변화 ≥ −5%p"),
              ("a3_organic", "유기적 성장", "영업권 증가 ÷ 매출 증가 ≤ 0.5"), ("a4_relmargin", "업종 상대 마진", "OM 섹터 백분위 ≥ 0.5 & ΔOM ≥ −1%p"),
              ("a5_accruals", "이익의 질", "(순이익−영업CF)/자산 ≤ 5%"), ("a6_sbcfcf", "SBC 차감 FCF", "(영업CF−capex−SBC)/매출 ≥ 3%"),
              ("a7_dso", "매출채권 회전", "DSO 1년 변화 ≤ +10%"), ("a8_incroic", "증분 자본수익률", "ΔNOPAT/Δ투하자본 ≥ 15%"),
              ("a9_cycle", "사이클 위치", "경기섹터 피크 마진 & 감속 아님"), ("a10_flex", "재무 유연성", "순차입/EBITDA ≤ 1.5 & 유동비율 ≥ 1.2"),
              ("a11_dilsbc", "희석·주식보상", "주식 YoY ≤ 2% & SBC/매출 ≤ 10%"), ("a12_relroic", "업종 상대 ROIC", "ROIC 섹터 백분위 ≥ 0.5")]


@dataclass(frozen=True)
class QualityParams:
    lag_days: int = 1               # 발표일 뒤 며칠부터 유효(달력일)
    min_available: int = 8          # 판정 가능 항목 최소 수
    rev_yoy_min: float = 0.15
    cagr3_min: float = 0.12
    ni_yoy_min: float = 0.15
    gm_min: float = 0.35
    gm_drop_max: float = 0.02
    om_min: float = 0.08
    om_drop_max: float = 0.01
    ocf_ni_min: float = 0.8
    fcf_margin_min: float = 0.05
    wc_slack: float = 0.10
    nd_ebitda_max: float = 2.5
    int_cov_min: float = 5.0
    dilution_max: float = 0.03
    reinvest_min: float = 0.10
    rnd_yoy_min: float = 0.15
    roe_min: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)


FLOW = ["revenue", "gross_profit", "operating_income", "net_income", "ebit", "ebitda", "interest_expense",
        "operating_cf", "capex", "rnd", "sbc", "da"]


def _ttm(s: pd.Series, dates: pd.Series) -> pd.Series:
    """4분기 합. 4분기가 400일 안에 들어야 한다(결산 변경·누락 분기 보호). 값 하나라도 없으면 NaN."""
    v = s.astype(float).rolling(4).sum()
    span = (dates - dates.shift(3)).dt.days
    ok = (span <= 400) & s.astype(float).notna() & s.shift(1).notna() & s.shift(2).notna() & s.shift(3).notna()
    return v.where(ok)


def _yoy(s: pd.Series) -> pd.Series:
    prev = s.shift(4)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / prev.abs() - 1).where(prev.abs() > 0)


def _one(df: pd.DataFrame, p: QualityParams) -> pd.DataFrame:
    """한 종목(분기 오름차순)의 체크리스트 표."""
    d = df.sort_values("fiscal_date").reset_index(drop=True)
    fd = pd.to_datetime(d["fiscal_date"])
    T = {k: _ttm(d[k], fd) if k in d else pd.Series(np.nan, index=d.index) for k in FLOW}
    rev, gp, oi, ni, ocf = T["revenue"], T["gross_profit"], T["operating_income"], T["net_income"], T["operating_cf"]
    capex = T["capex"].abs(); rnd = T["rnd"]; ebitda = T["ebitda"]; ebit = T["ebit"]; intexp = T["interest_expense"].abs()
    with np.errstate(invalid="ignore", divide="ignore"):
        rev_yoy = _yoy(rev)
        cagr3 = (rev / rev.shift(12)) ** (1 / 3) - 1
        cagr3 = cagr3.where((rev.shift(12) > 0) & ((fd - fd.shift(12)).dt.days <= 1200))
        ni_prev = ni.shift(4)
        ni_yoy = _yoy(ni)
        gm = gp / rev; dgm = gm - gm.shift(4)
        om = oi / rev; dom = om - om.shift(4)
        ocf_ni = ocf / ni
        fcfm = (ocf - capex.fillna(0)) / rev
        fcfm = fcfm.where(ocf.notna())
        # 운전자본: 분기 vs 전년 동기
        q_rev_yoy = _yoy(d["revenue"].astype(float))
        recv_yoy = _yoy(d["receivables"].astype(float)) if "receivables" in d else pd.Series(np.nan, index=d.index)
        inv_yoy = _yoy(d["inventory"].astype(float)) if "inventory" in d else pd.Series(np.nan, index=d.index)
        cash = d.get("cash_and_st_investments", pd.Series(np.nan, index=d.index)).astype(float)
        debt = d.get("total_debt", pd.Series(np.nan, index=d.index)).astype(float)
        netdebt = debt.fillna(0) - cash.fillna(0)
        netdebt = netdebt.where(debt.notna() | cash.notna())
        nd_ebitda = netdebt / ebitda
        int_cov = ebit / intexp
        shares = d.get("shares_outstanding", pd.Series(np.nan, index=d.index)).astype(float)
        sh_chg = _yoy(shares)
        sh_chg = sh_chg.where((sh_chg > -0.5) & (sh_chg < 2.0))
        reinvest = (rnd.fillna(0) + capex.fillna(0)) / rev
        reinvest = reinvest.where(rnd.notna() | capex.notna())
        rnd_yoy = _yoy(rnd)
        eq = d.get("equity", pd.Series(np.nan, index=d.index)).astype(float)
        eq_avg = eq.rolling(5, min_periods=2).mean()
        roe = ni / eq_avg
        sbc_rev = T["sbc"] / rev
        rnd_int = rnd / rev
        # ---- 고급 판(설계서 §12.2)·G/F-score 용 연속 지표 ----
        assets = d.get("total_assets", pd.Series(np.nan, index=d.index)).astype(float)
        assets_avg = assets.rolling(5, min_periods=2).mean()
        accruals = (ni - ocf) / assets_avg                                   # Sloan 발생액
        sbc_fcfm = (ocf - capex.fillna(0) - T["sbc"].fillna(0)) / rev          # 주식보상 차감 FCF 마진
        sbc_fcfm = sbc_fcfm.where(ocf.notna())
        q_rev = d["revenue"].astype(float)
        recv = d.get("receivables", pd.Series(np.nan, index=d.index)).astype(float)
        dso = recv / q_rev
        dso_chg = dso / dso.shift(4) - 1
        gw = d.get("goodwill", pd.Series(np.nan, index=d.index)).astype(float)
        d_rev4 = rev - rev.shift(4)
        gw_share = ((gw - gw.shift(4)) / d_rev4).where(d_rev4 > 0)
        gw_share = gw_share.where(gw.notna(), 0.0).where(rev.notna())        # 영업권 자료 없음 = 0 (인수 없음으로 본다)
        tax_rate = (T["income_tax"] / T["pretax_income"]).clip(0, 0.35) if "income_tax" in T else pd.Series(np.nan, index=d.index)
        nopat = ebit * (1 - tax_rate.fillna(0.21))
        ic = eq.fillna(0) + debt.fillna(0) - cash.fillna(0)
        ic = ic.where(eq.notna())
        ic_avg = ic.rolling(5, min_periods=2).mean()
        roic = nopat / ic_avg
        roic = roic.where(ic_avg > 0)
        d_ic = ic - ic.shift(4); d_nopat = nopat - nopat.shift(4)
        inc_roic = (d_nopat / d_ic).where(d_ic > 0)
        inc_roic = inc_roic.where(~((d_ic <= 0) & (d_nopat > 0)), 9.99)      # 자본 줄고 이익 늘면 '매우 좋음'
        om_peak_frac = om / om.rolling(20, min_periods=8).max()
        om_peak_frac = om_peak_frac.where(om > 0)
        cur_ratio = d.get("total_current_assets", pd.Series(np.nan, index=d.index)).astype(float) / d.get("total_current_liabilities", pd.Series(np.nan, index=d.index)).astype(float)
        q_rev_accel = q_rev_yoy - q_rev_yoy.shift(1)
        roa = ni / assets_avg; cfo_assets = ocf / assets_avg
        roa_q = d["net_income"].astype(float) / assets
        roa_var = roa_q.rolling(8, min_periods=6).std()
        sg_var = q_rev_yoy.rolling(8, min_periods=6).std()
        rnd_assets = rnd / assets_avg; capex_assets = capex / assets_avg
        # Piotroski F (연간 → TTM, 1년 전 대비)
        ltd = d.get("long_term_debt", pd.Series(np.nan, index=d.index)).astype(float)
        lev = ltd / assets
        turn = rev / assets_avg
        f_items = pd.DataFrame({"f1": (roa > 0), "f2": (ocf > 0), "f3": (roa - roa.shift(4) > 0), "f4": (ocf > ni), "f5": (lev - lev.shift(4) < 0),
                                "f6": (cur_ratio - cur_ratio.shift(4) > 0), "f7": (sh_chg <= 0), "f8": (dgm > 0), "f9": (turn - turn.shift(4) > 0)}).astype(float)
        f_known = pd.DataFrame({"f1": roa.notna(), "f2": ocf.notna(), "f3": (roa - roa.shift(4)).notna(), "f4": (ocf.notna() & ni.notna()), "f5": (lev - lev.shift(4)).notna(),
                                "f6": (cur_ratio - cur_ratio.shift(4)).notna(), "f7": sh_chg.notna(), "f8": dgm.notna(), "f9": (turn - turn.shift(4)).notna()})
        f_score = (f_items.where(f_known).sum(axis=1) / f_known.sum(axis=1) * 9).where(f_known.sum(axis=1) >= 7)
    nan = pd.Series(np.nan, index=d.index)

    def chk(cond: pd.Series, known: pd.Series) -> pd.Series:
        out = cond.astype(float)
        return out.where(known.astype(bool), np.nan)

    c1 = chk(rev_yoy >= p.rev_yoy_min, rev_yoy.notna())
    c2 = chk(cagr3 >= p.cagr3_min, cagr3.notna())
    turn = (ni_prev <= 0) & (ni > 0)
    c3 = chk(((ni > 0) & (ni_prev > 0) & (ni_yoy >= p.ni_yoy_min)) | turn, ni.notna() & ni_prev.notna())
    c4 = chk((gm >= p.gm_min) & (dgm >= -p.gm_drop_max), gm.notna() & dgm.notna())
    c5 = chk((om >= p.om_min) & (dom >= -p.om_drop_max), om.notna() & dom.notna())
    c6 = chk(((ni > 0) & (ocf_ni >= p.ocf_ni_min)) | ((ni <= 0) & (ocf > 0)), ocf.notna() & ni.notna())
    c7 = chk(fcfm >= p.fcf_margin_min, fcfm.notna())
    recv_ok = (recv_yoy <= q_rev_yoy + p.wc_slack) | recv_yoy.isna()
    inv_ok = (inv_yoy <= q_rev_yoy + p.wc_slack) | inv_yoy.isna()
    c8 = chk(recv_ok & inv_ok, q_rev_yoy.notna() & (recv_yoy.notna() | inv_yoy.notna()))
    lev_ok = (netdebt <= 0) | ((ebitda > 0) & (nd_ebitda <= p.nd_ebitda_max))
    cov_ok = (netdebt <= 0) | int_cov.isna() | (intexp <= 0) | (int_cov >= p.int_cov_min)     # 이자보상은 순차입이 있을 때만 본다
    c9 = chk(lev_ok & cov_ok, netdebt.notna())
    c10 = chk(sh_chg <= p.dilution_max, sh_chg.notna())
    c11 = chk((reinvest >= p.reinvest_min) | (rnd_yoy >= p.rnd_yoy_min), reinvest.notna())
    c12 = chk((roe >= p.roe_min) | ((eq_avg <= 0) & (ni > 0)), roe.notna() | ((eq_avg <= 0) & ni.notna()))
    checks = pd.DataFrame({"c1_rev": c1, "c2_cagr": c2, "c3_ni": c3, "c4_gm": c4, "c5_om": c5, "c6_ocf": c6, "c7_fcf": c7, "c8_wc": c8,
                           "c9_bal": c9, "c10_dil": c10, "c11_inv": c11, "c12_roe": c12})
    avail = checks.notna().sum(axis=1)
    passed = checks.fillna(0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (passed / avail * 12).where(avail > 0)
    rd = pd.to_datetime(d["report_date"]) if "report_date" in d else pd.Series(pd.NaT, index=d.index)
    effective = rd.fillna(fd + pd.Timedelta(days=90)) + pd.Timedelta(days=p.lag_days)
    out = pd.DataFrame({"symbol": d["symbol"], "fiscal_date": fd, "report_date": rd, "effective": effective,
                        "q_score": score, "q_avail": avail, "q_pass": passed,
                        "q_rev": rev_yoy, "q_cagr3": cagr3, "q_ni": ni_yoy, "q_gm": gm, "q_dgm": dgm, "q_om": om, "q_dom": dom,
                        "q_ocfni": ocf_ni, "q_fcfm": fcfm, "q_roe": roe, "q_netcash": -nd_ebitda.where(netdebt > 0, 0.0).where(netdebt.notna()),
                        "q_nodil": -sh_chg, "q_rnd": rnd_int, "q_lowsbc": -sbc_rev,
                        "rev_ttm": rev, "ni_ttm": ni, "ocf_ttm": ocf, "fcf_ttm": ocf - capex.fillna(0), "capex_ttm": capex, "rnd_ttm": rnd, "sbc_ttm": T["sbc"],
                        "ebitda_ttm": ebitda, "netdebt": netdebt, "nd_ebitda": nd_ebitda, "int_cov": int_cov, "cash": cash, "debt": debt, "equity": eq,
                        "shares": shares, "sh_chg": sh_chg, "recv_yoy": recv_yoy, "inv_yoy": inv_yoy, "q_rev_yoy": q_rev_yoy, "reinvest": reinvest, "rnd_yoy": rnd_yoy,
                        "accruals": accruals, "sbc_fcfm": sbc_fcfm, "dso_chg": dso_chg, "gw_share": gw_share, "roic": roic, "inc_roic": inc_roic, "om_peak_frac": om_peak_frac,
                        "cur_ratio": cur_ratio, "q_rev_accel": q_rev_accel, "roa": roa, "cfo_assets": cfo_assets, "roa_var": roa_var, "sg_var": sg_var,
                        "rnd_assets": rnd_assets, "capex_assets": capex_assets, "sbc_rev": sbc_rev, "f_score": f_score,
                        "sector": d["sector"].iloc[0] if "sector" in d else None})
    return pd.concat([out, checks], axis=1)


def quality_features(statements: pd.DataFrame, p: QualityParams = QualityParams()) -> pd.DataFrame:
    """long 재무제표 표 → 분기별 체크리스트 특징(전 종목). effective 오름차순."""
    parts = [_one(g, p) for _, g in statements.groupby("symbol", sort=True)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out.sort_values(["symbol", "effective"]).reset_index(drop=True)


def daily_aux(feats: pd.DataFrame, days: pd.DatetimeIndex, tickers: list[str], cols: list[str] = AUX_COLS,
              min_available: int = 8) -> dict[str, np.ndarray]:
    """(T,N) aux 열. 그날 d 에는 effective ≤ d 인 마지막 분기 행. 판정 가능 항목 < min_available 이면 q_score NaN."""
    T, N = len(days), len(tickers)
    out = {c: np.full((T, N), np.nan) for c in cols}
    day_ns = days.values.astype("datetime64[ns]").astype(np.int64)
    by = {s: g for s, g in feats.groupby("symbol")}
    for j, t in enumerate(tickers):
        g = by.get(t)
        if g is None or g.empty:
            continue
        g = g.sort_values("effective")
        eff = g["effective"].values.astype("datetime64[ns]").astype(np.int64)
        k = np.searchsorted(eff, day_ns, side="right") - 1
        has = k >= 0
        kk = np.maximum(k, 0)
        for c in cols:
            v = g[c].to_numpy(dtype=float)[kk]
            out[c][:, j] = np.where(has, v, np.nan)
        if "q_score" in out:
            av = g["q_avail"].to_numpy(dtype=float)[kk]
            out["q_score"][:, j] = np.where(has & (av >= min_available), out["q_score"][:, j], np.nan)
    return out


def row_as_of(feats: pd.DataFrame, ticker: str, day: pd.Timestamp) -> pd.Series | None:
    g = feats[(feats["symbol"] == ticker) & (feats["effective"] <= day)]
    return None if g.empty else g.sort_values("effective").iloc[-1]


def _m(v: float | None, unit: float = 1e6, fmt: str = "{:,.0f}") -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v / unit)


def _p(v: float | None, digits: int = 0, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 10:
        return f"{v:+.0%}" if signed else f"{v:.0%}"
    return f"{v:+.{digits}%}" if signed else f"{v:.{digits}%}"


def _x(v: float | None) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}×"


def explain_card(ticker: str, r: pd.Series | None, day: pd.Timestamp) -> str:
    """사람이 읽는 재무제표 카드(markdown)."""
    if r is None:
        return f"### {ticker} — 기준일 {day.date()} 에 쓸 수 있는 재무제표 없음"
    mark = lambda v: "➖" if pd.isna(v) else ("✅" if v >= 1 else "❌")
    L = [f"### {ticker} — 기준일 {day.date()} · 마지막 분기 {r['fiscal_date'].date()} (발표 {r['report_date'].date() if pd.notna(r['report_date']) else '?'}) · **점수 {r['q_score']:.1f}/12** (판정 {int(r['q_avail'])}개, 통과 {int(r['q_pass'])}개)",
         "",
         f"- 규모(TTM, $M): 매출 {_m(r['rev_ttm'])} · 순이익 {_m(r['ni_ttm'])} · 영업CF {_m(r['ocf_ttm'])} · FCF {_m(r['fcf_ttm'])} · capex {_m(r['capex_ttm'])} · R&D {_m(r['rnd_ttm'])} · 주식보상 {_m(r['sbc_ttm'])}",
         f"- 재무상태($M): 현금성 {_m(r['cash'])} · 차입 {_m(r['debt'])} · 순차입 {_m(r['netdebt'])} · 자본 {_m(r['equity'])} · 유통주식 {_m(r['shares'])}M",
         "", "| # | 항목 | 판정 | 숫자 |", "|---|---|---|---|"]
    nums = {
        "c1_rev": f"매출 TTM YoY {_p(r['q_rev'])}",
        "c2_cagr": f"3년 매출 CAGR {_p(r['q_cagr3'])}",
        "c3_ni": f"순이익 TTM YoY {_p(r['q_ni'])}",
        "c4_gm": f"매출총이익률 {_p(r['q_gm'], signed=False)} (1년 전 대비 {_p(r['q_dgm'], 1)}p)",
        "c5_om": f"영업이익률 {_p(r['q_om'], signed=False)} (1년 전 대비 {_p(r['q_dom'], 1)}p)",
        "c6_ocf": f"영업CF ÷ 순이익 {_x(r['q_ocfni'])}",
        "c7_fcf": f"FCF 마진 {_p(r['q_fcfm'], signed=False)}",
        "c8_wc": f"매출채권 YoY {_p(r['recv_yoy'])} · 재고 YoY {_p(r['inv_yoy'])} vs 분기 매출 YoY {_p(r['q_rev_yoy'])}",
        "c9_bal": f"순차입/EBITDA {_x(r['nd_ebitda']) if (pd.notna(r['netdebt']) and r['netdebt'] > 0) else '순현금'} · 이자보상 {_x(r['int_cov'])}",
        "c10_dil": f"유통주식 YoY {_p(r['sh_chg'], 1)}",
        "c11_inv": f"(R&D+capex)/매출 {_p(r['reinvest'], signed=False)} · R&D YoY {_p(r['rnd_yoy'])}",
        "c12_roe": f"ROE {_p(r['q_roe'], signed=False)}",
    }
    for col, name, desc in CHECKS:
        L.append(f"| {col.split('_')[0].upper()} | {name} ({desc}) | {mark(r[col])} | {nums[col]} |")
    return "\n".join(L)


def study_forward(feats: pd.DataFrame, prices: dict[str, pd.DataFrame], qqq: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                  horizon_days: int = 252, lists: dict[pd.Timestamp, set[str]] | None = None, min_available: int = 8) -> dict:
    """항목별 예측력: 매월 첫 거래일에 (유니버스 전체 / 성장 리스트) 종목의 체크 통과·실패 별 앞 horizon 거래일 수익(QQQ 대비 포함)."""
    cal = pd.to_datetime(qqq.index); cal = cal[(cal >= start) & (cal <= end)]
    month_starts = [d for i, d in enumerate(cal) if i == 0 or cal[i - 1].month != d.month]
    qc = qqq["Close"].astype(float)
    close = {t: d["Close"].astype(float) for t, d in prices.items() if t in set(feats["symbol"])}
    full_cal = pd.to_datetime(qqq.index)
    pos = {d: i for i, d in enumerate(full_cal)}
    recs = []
    for d in month_starts:
        i = pos[d]; j = i + horizon_days
        if j >= len(full_cal):
            continue
        d2 = full_cal[j]
        q_ret = qc.get(d2, np.nan) / qc.get(d, np.nan) - 1
        lst = None
        if lists is not None:
            keys = [k for k in lists if k < d]
            lst = lists[max(keys)] if keys else set()
        for t, c in close.items():
            r = row_as_of(feats, t, d - pd.Timedelta(days=1))
            if r is None or r["q_avail"] < min_available:
                continue
            p0, p1 = c.get(d, np.nan), c.get(d2, np.nan)
            if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
                continue
            rec = {"day": d, "ticker": t, "fwd": p1 / p0 - 1, "fwd_rel": (p1 / p0 - 1) - q_ret, "in_list": (t in lst) if lst is not None else False,
                   "q_score": r["q_score"], **{c_: r[c_] for c_ in CHECK_COLS}}
            recs.append(rec)
    df = pd.DataFrame(recs)
    tables: dict = {}; L = [f"# 재무제표 항목별 예측력 — {start.date()} ~ {end.date()}, 앞 {horizon_days} 거래일 수익 (매월 첫 거래일 표본, 체결 그날 종가 → {horizon_days} 거래일 뒤 종가)", ""]
    for pop, sub in (("유니버스 전체", df), ("성장 리스트 종목만", df[df["in_list"]])):
        if sub.empty:
            continue
        L += [f"## {pop} — 표본 {len(sub):,} (종목·월), 전체 중앙 {sub['fwd'].median():+.1%} · QQQ 대비 중앙 {sub['fwd_rel'].median():+.1%} · QQQ 이긴 비율 {(sub['fwd_rel'] > 0).mean():.0%}", "",
              "| 항목 | 통과 n | 통과 중앙 | 통과 QQQ 대비 | 통과 QQQ 이김 | 실패 n | 실패 중앙 | 실패 QQQ 대비 | 실패 QQQ 이김 | 차이(중앙 QQQ 대비) |", "|---|---|---|---|---|---|---|---|---|---|"]
        rows = []
        for col, name, desc in CHECKS:
            a, b = sub[sub[col] >= 1], sub[sub[col] == 0]
            if len(a) < 30 or len(b) < 30:
                continue
            diff = a["fwd_rel"].median() - b["fwd_rel"].median()
            rows.append({"check": col, "name": name, "pass_n": len(a), "pass_med": a["fwd"].median(), "pass_rel": a["fwd_rel"].median(), "pass_win": (a["fwd_rel"] > 0).mean(),
                         "fail_n": len(b), "fail_med": b["fwd"].median(), "fail_rel": b["fwd_rel"].median(), "fail_win": (b["fwd_rel"] > 0).mean(), "diff": diff})
            L.append(f"| {name} ({desc}) | {len(a):,} | {a['fwd'].median():+.1%} | {a['fwd_rel'].median():+.1%} | {(a['fwd_rel'] > 0).mean():.0%} | {len(b):,} | {b['fwd'].median():+.1%} | {b['fwd_rel'].median():+.1%} | {(b['fwd_rel'] > 0).mean():.0%} | {diff:+.1%} |")
        L += ["", "| 점수 구간 | n | 중앙 | QQQ 대비 중앙 | QQQ 이김 | 평균 | 하위 10% |", "|---|---|---|---|---|---|---|"]
        bins = [(-1, 5.999, "≤ 5"), (6, 7.999, "6~7"), (8, 9.999, "8~9"), (10, 12.001, "10~12")]
        brows = []
        for lo, hi, lab in bins:
            s = sub[(sub["q_score"] >= lo) & (sub["q_score"] <= hi)]
            if len(s) < 30:
                continue
            brows.append({"bin": lab, "n": len(s), "med": s["fwd"].median(), "rel": s["fwd_rel"].median(), "win": (s["fwd_rel"] > 0).mean(), "mean": s["fwd"].mean(), "p10": s["fwd"].quantile(0.1)})
            L.append(f"| {lab} | {len(s):,} | {s['fwd'].median():+.1%} | {s['fwd_rel'].median():+.1%} | {(s['fwd_rel'] > 0).mean():.0%} | {s['fwd'].mean():+.1%} | {s['fwd'].quantile(0.1):+.1%} |")
        # 연도별 점수 상위(10~12) vs 하위(≤7) 차이
        yr = []
        for y, s in sub.groupby(sub["day"].dt.year):
            hi_, lo_ = s[s["q_score"] >= 10], s[s["q_score"] <= 7]
            if len(hi_) >= 20 and len(lo_) >= 20:
                yr.append({"year": int(y), "hi_n": len(hi_), "hi_rel": hi_["fwd_rel"].median(), "lo_n": len(lo_), "lo_rel": lo_["fwd_rel"].median()})
        if yr:
            L += ["", "| 연도(표본 시작) | 점수 10~12 n | QQQ 대비 중앙 | 점수 ≤7 n | QQQ 대비 중앙 | 차이 |", "|---|---|---|---|---|---|"]
            for y in yr:
                L.append(f"| {y['year']} | {y['hi_n']} | {y['hi_rel']:+.1%} | {y['lo_n']} | {y['lo_rel']:+.1%} | {y['hi_rel'] - y['lo_rel']:+.1%} |")
        tables[pop] = {"checks": rows, "bins": brows, "years": yr, "n": len(sub)}
        L.append("")
    return {"md": "\n".join(L), "tables": tables, "samples": df}


def _xrank_rows(a: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    for i in range(a.shape[0]):
        row = a[i]; m = ~np.isnan(row); k = int(m.sum())
        if k == 1:
            out[i, m] = 0.5
        elif k > 1:
            out[i, m] = np.argsort(np.argsort(row[m], kind="stable"), kind="stable") / (k - 1)
    return out


def sector_rank(a: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    """(T,N) 값을 날마다 **같은 섹터 안에서** 백분위(0~1)로. 섹터 None 은 NaN."""
    out = np.full(a.shape, np.nan)
    for sec in set(sectors.tolist()) - {None, "None", ""}:
        cols = np.flatnonzero(sectors == sec)
        if len(cols):
            out[:, cols] = _xrank_rows(a[:, cols])
    return out


def sector_median(a: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    for sec in set(sectors.tolist()) - {None, "None", ""}:
        cols = np.flatnonzero(sectors == sec)
        if len(cols):
            with np.errstate(all="ignore"):
                med = np.nanmedian(a[:, cols], axis=1)
            out[:, cols] = med[:, None]
    return out


def advanced_scores(aux: dict[str, np.ndarray], sectors: np.ndarray, min_available: int = 8) -> dict[str, np.ndarray]:
    """고급 판 A(12항목, 설계서 §12.2) + Mohanram G(7) 를 (T,N) 으로. 금융·부동산·유틸리티는 NaN.
    aux 는 daily_aux(AUX_COLS) 출력(+ q_rev·q_om·q_dom 등). 반환: a_score, a_avail, g_score, 항목별 a1..a12."""
    sec = np.asarray([None if s_ in (None, "None", "") else str(s_).upper() for s_ in sectors], dtype=object)
    excluded = np.isin(sec, list(EXCLUDED_SECTORS)) | (sec == None)  # noqa: E711
    cyc = np.isin(sec, list(CYCLICAL))
    T = next(iter(aux.values())).shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_g = sector_rank(aux["q_rev"], sec); rel_om = sector_rank(aux["q_om"], sec); rel_roic = sector_rank(aux["roic"], sec)
        nd = aux["nd_ebitda"]; netdebt = aux["netdebt"]
        a = {
            "a1_relgrowth": (rel_g >= 0.67, ~np.isnan(rel_g)),
            "a2_accel": (aux["q_rev_accel"] >= -0.05, ~np.isnan(aux["q_rev_accel"])),
            "a3_organic": (aux["gw_share"] <= 0.5, ~np.isnan(aux["gw_share"])),
            "a4_relmargin": ((rel_om >= 0.5) & (aux["q_dom"] >= -0.01), ~np.isnan(rel_om) & ~np.isnan(aux["q_dom"])),
            "a5_accruals": (aux["accruals"] <= 0.05, ~np.isnan(aux["accruals"])),
            "a6_sbcfcf": (aux["sbc_fcfm"] >= 0.03, ~np.isnan(aux["sbc_fcfm"])),
            "a7_dso": (aux["dso_chg"] <= 0.10, ~np.isnan(aux["dso_chg"])),
            "a8_incroic": (aux["inc_roic"] >= 0.15, ~np.isnan(aux["inc_roic"])),
            "a9_cycle": (~(cyc[None, :] & (aux["om_peak_frac"] >= 0.95) & (aux["q_rev_accel"] < 0)), ~np.isnan(aux["om_peak_frac"]) | ~cyc[None, :].repeat(T, 0)),
            "a10_flex": (((netdebt <= 0) | (nd <= 1.5)) & (aux["cur_ratio"] >= 1.2), ~np.isnan(netdebt) & ~np.isnan(aux["cur_ratio"])),
            "a11_dilsbc": ((aux["sh_chg"] <= 0.02) & (np.nan_to_num(aux["sbc_rev"], nan=0.0) <= 0.10), ~np.isnan(aux["sh_chg"])),
            "a12_relroic": (rel_roic >= 0.5, ~np.isnan(rel_roic)),
        }
        passed = np.zeros(aux["q_rev"].shape); avail = np.zeros(aux["q_rev"].shape)
        out: dict[str, np.ndarray] = {}
        for k, (cond, known) in a.items():
            v = np.where(known, cond.astype(float), np.nan)
            out[k] = v; passed += np.nan_to_num(v); avail += known.astype(float)
        a_score = np.where(avail >= min_available, passed / np.maximum(avail, 1) * 12, np.nan)
        a_score[:, excluded] = np.nan
        out["a_score"] = a_score; out["a_avail"] = avail
        # Mohanram G (업종 중앙값 대비, 광고비 제외 7점)
        g_items = [aux["roa"] > sector_median(aux["roa"], sec), aux["cfo_assets"] > sector_median(aux["cfo_assets"], sec), aux["ocf_ttm"] > aux["ni_ttm"],
                   aux["roa_var"] < sector_median(aux["roa_var"], sec), aux["sg_var"] < sector_median(aux["sg_var"], sec),
                   aux["rnd_assets"] > sector_median(aux["rnd_assets"], sec), aux["capex_assets"] > sector_median(aux["capex_assets"], sec)]
        g_known = [~np.isnan(aux["roa"]), ~np.isnan(aux["cfo_assets"]), ~np.isnan(aux["ocf_ttm"]) & ~np.isnan(aux["ni_ttm"]), ~np.isnan(aux["roa_var"]), ~np.isnan(aux["sg_var"]),
                   ~np.isnan(np.nan_to_num(aux["rnd_assets"], nan=0.0) + aux["roa"]), ~np.isnan(aux["capex_assets"])]
        gp = sum(np.where(k_, c_.astype(float), 0.0) for c_, k_ in zip(g_items, g_known)); ga = sum(k_.astype(float) for k_ in g_known)
        g_score = np.where(ga >= 5, gp / np.maximum(ga, 1) * 7, np.nan); g_score[:, excluded] = np.nan
        out["g_score"] = g_score
    return out
