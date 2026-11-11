"""재무제표 읽기 품질 체크리스트 — 합성 분기 표로 판정·TTM·유효일·일별 정렬을 고정한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.fundamental_quality import (
    AUX_COLS, CHECK_COLS, QualityParams, advanced_scores, daily_aux, explain_card, quality_features, row_as_of, sector_rank,
)


def _stmts(symbol: str, n: int = 20, rev0: float = 1000.0, g: float = 0.08, gm: float = 0.6, om: float = 0.2,
           ocf_mult: float = 1.2, capex_frac: float = 0.05, debt: float = 0.0, cash: float = 500.0,
           shares: float = 100.0, sh_g: float = 0.0, rnd_frac: float = 0.12, equity: float = 2000.0,
           recv_mult: float = 1.0, inv_mult: float = 1.0, interest: float = 0.0) -> pd.DataFrame:
    """분기마다 매출이 g 씩(복리) 느는 회사. 분기 말은 3·6·9·12월 말, 발표일은 분기 말 + 30일."""
    rows = []
    rev = rev0; sh = shares
    for i in range(n):
        fd = pd.Timestamp("2016-03-31") + pd.offsets.QuarterEnd(i)
        ni = rev * om * 0.8
        rows.append(dict(symbol=symbol, fiscal_date=fd, report_date=fd + pd.Timedelta(days=30),
                         revenue=rev, gross_profit=rev * gm, operating_income=rev * om, net_income=ni, ebit=rev * om, ebitda=rev * om * 1.1,
                         interest_expense=interest, operating_cf=ni * ocf_mult, capex=rev * capex_frac, rnd=rev * rnd_frac, sbc=rev * 0.02,
                         total_debt=debt, cash_and_st_investments=cash, equity=equity, shares_outstanding=sh,
                         receivables=rev * 0.2 * recv_mult ** i, inventory=rev * 0.1 * inv_mult ** i, da=rev * 0.02))
        rev *= 1 + g; sh *= 1 + sh_g
    return pd.DataFrame(rows)


def test_good_company_passes_all_twelve():
    f = quality_features(_stmts("GOOD"))
    last = f.iloc[-1]
    assert last["q_avail"] == 12 and last["q_pass"] == 12 and abs(last["q_score"] - 12) < 1e-9
    # 8% 분기 복리 → TTM YoY ≈ 36%, 3년 CAGR ≈ 36%
    assert 0.34 < last["q_rev"] < 0.38 and 0.34 < last["q_cagr3"] < 0.38
    assert abs(last["q_gm"] - 0.6) < 1e-9 and abs(last["q_ocfni"] - 1.2) < 1e-9
    # 유효일 = 발표일 + 1일
    assert last["effective"] == last["report_date"] + pd.Timedelta(days=1)


def test_ttm_needs_four_quarters_and_cagr_needs_twelve_more():
    f = quality_features(_stmts("X", n=8))
    assert f["q_rev"].notna().sum() == 1           # TTM(4분기) 의 YoY 는 8번째 분기에서 처음
    assert f["q_cagr3"].isna().all()                # 16분기 필요
    assert f.iloc[3]["rev_ttm"] > 0 and np.isnan(f.iloc[2]["rev_ttm"])


def test_bad_signals_fail_specific_checks():
    s = _stmts("BAD", g=0.0, gm=0.2, om=0.03, ocf_mult=0.3, capex_frac=0.3, debt=5000.0, cash=100.0, sh_g=0.05, rnd_frac=0.0, equity=10000.0,
               recv_mult=1.15, interest=200.0)
    f = quality_features(s); last = f.iloc[-1]
    failed = {c for c in CHECK_COLS if last[c] == 0}
    assert {"c1_rev", "c2_cagr", "c3_ni", "c4_gm", "c6_ocf", "c7_fcf", "c8_wc", "c9_bal", "c10_dil", "c12_roe"} <= failed
    # 영업이익률 3% 지만 안 줄었으니 C5 는 문턱(8%) 미달로 실패
    assert last["c5_om"] == 0
    # 재투자: capex 30% 라 C11 통과
    assert last["c11_inv"] == 1
    assert last["q_score"] < 2


def test_turnaround_counts_as_profit_growth_and_negative_equity_buyback_passes_roe():
    s = _stmts("TURN")
    s.loc[s.index < 8, "net_income"] = -10.0            # 처음 8분기 적자 → 9~12분기 TTM 전환
    s["equity"] = -500.0                                  # 자사주 소각형 음(−)의 자본
    f = quality_features(s)
    row = f[f["fiscal_date"] == pd.Timestamp("2019-03-31")].iloc[0]   # 13번째 분기: TTM 흑자, 1년 전 TTM 적자
    assert row["c3_ni"] == 1 and row["c12_roe"] == 1


def test_missing_balance_sheet_marks_checks_unknown_not_failed():
    s = _stmts("NOBS")
    for c in ("total_debt", "cash_and_st_investments", "equity", "shares_outstanding", "receivables", "inventory"):
        s[c] = np.nan
    f = quality_features(s); last = f.iloc[-1]
    assert np.isnan(last["c9_bal"]) and np.isnan(last["c10_dil"]) and np.isnan(last["c12_roe"]) and np.isnan(last["c8_wc"])
    assert last["q_avail"] == 8 and abs(last["q_score"] - 12) < 1e-9      # 통과 8/8 → 12점 환산


def test_daily_aux_steps_on_effective_date_and_respects_min_available():
    f = quality_features(pd.concat([_stmts("A"), _stmts("B", n=6)]))
    days = pd.bdate_range("2020-10-01", "2021-02-28")
    aux = daily_aux(f, days, ["A", "B", "Z"], cols=["q_score", "q_avail", "rev_ttm"])
    rowA = f[(f.symbol == "A") & (f.fiscal_date == "2020-12-31")].iloc[0]
    eff = rowA["effective"]                                 # 2021-01-30 + 1 = 01-31 (일요일) → 02-01 부터 보임
    before = aux["rev_ttm"][days < eff, 0]; after = aux["rev_ttm"][days >= eff, 0]
    assert np.allclose(after, rowA["rev_ttm"]) and (before < rowA["rev_ttm"]).all()
    assert np.isnan(aux["q_score"][:, 2]).all()           # 자료 없는 종목
    assert np.isnan(aux["q_score"][:, 1]).all()           # B: 6분기뿐 → 판정 가능 항목 < 8 → 점수 없음(다른 열은 있을 수 있음)


def test_row_as_of_and_card_text():
    f = quality_features(_stmts("CARD"))
    r = row_as_of(f, "CARD", pd.Timestamp("2021-02-15"))
    assert r is not None and r["fiscal_date"] == pd.Timestamp("2020-12-31")
    md = explain_card("CARD", r, pd.Timestamp("2021-02-15"))
    assert "점수 12.0/12" in md and md.count("✅") == 12 and "매출 TTM YoY" in md
    assert "쓸 수 있는 재무제표 없음" in explain_card("CARD", None, pd.Timestamp("2016-01-01"))


def test_share_count_glitch_is_unknown():
    s = _stmts("GL")
    s.loc[s.index == 19, "shares_outstanding"] = 100.0 * 1000     # 단위 오류(천 배)
    f = quality_features(s)
    assert np.isnan(f.iloc[-1]["c10_dil"])


def test_sector_rank_is_within_sector_only():
    a = np.array([[1.0, 2.0, 3.0, 10.0, 20.0]])
    r = sector_rank(a, np.array(["T", "T", "T", "H", "H"], dtype=object))
    assert np.allclose(r[0], [0.0, 0.5, 1.0, 0.0, 1.0])


def test_advanced_scores_good_company_and_excluded_sector():
    good = _stmts("GOOD"); good["sector"] = "TECHNOLOGY"; good["goodwill"] = 0.0; good["total_assets"] = 5000.0
    good["total_current_assets"] = 1500.0; good["total_current_liabilities"] = 500.0; good["income_tax"] = good["net_income"] * 0.2; good["pretax_income"] = good["net_income"] * 1.2
    bad = _stmts("BAD", g=0.01, om=0.03, ocf_mult=0.2, sh_g=0.05); bad["sector"] = "TECHNOLOGY"; bad["goodwill"] = np.linspace(0, 3000, len(bad)); bad["total_assets"] = 500.0
    bad["total_current_assets"] = 400.0; bad["total_current_liabilities"] = 500.0; bad["income_tax"] = 0.0; bad["pretax_income"] = bad["net_income"]
    bank = _stmts("BANK"); bank["sector"] = "FINANCIAL SERVICES"; bank["total_assets"] = 5000.0
    f = quality_features(pd.concat([good, bad, bank]))
    days = pd.bdate_range("2020-12-01", "2021-02-28")
    aux = daily_aux(f, days, ["GOOD", "BAD", "BANK"], cols=AUX_COLS)
    out = advanced_scores(aux, np.array(["TECHNOLOGY", "TECHNOLOGY", "FINANCIAL SERVICES"], dtype=object), min_available=6)
    last = -1
    assert np.isnan(out["a_score"][last, 2]) and np.isnan(out["g_score"][last, 2])          # 금융 제외
    assert out["a_score"][last, 0] > out["a_score"][last, 1]                                 # 좋은 회사 > 나쁜 회사
    assert out["a5_accruals"][last, 0] == 1 and out["a5_accruals"][last, 1] == 0            # 발생액: 영업CF 0.2× 순이익 → 실패
    assert out["a11_dilsbc"][last, 1] == 0                                                   # 5% 희석 실패
    assert out["a3_organic"][last, 1] == 0 and out["a3_organic"][last, 0] == 1              # 영업권으로 산 성장
