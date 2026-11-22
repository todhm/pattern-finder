"""피셔 15 포인트: 기준일 자르기, 수치 포인트, 통과 규칙, 10-K 항목 자르기·이름 가림."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.adapters.sec_edgar_filings import mask_company, split_items, window
from strategy.adapters import fisher_checklist as fc


def _stmts(sym: str, sector: str, quarters: int = 28, rev0: float = 100e6, g: float = 0.04, om: float = 0.2, rnd: float = 0.1, sh_g: float = 0.0,
           split_at: int | None = None) -> pd.DataFrame:
    fd = pd.date_range("2019-03-31", periods=quarters, freq="QE")
    rev = rev0 * (1 + g) ** np.arange(quarters)
    sh = 100e6 * (1 + sh_g) ** np.arange(quarters)
    if split_at is not None:
        sh[split_at:] *= 10
    return pd.DataFrame({"symbol": sym, "fiscal_date": fd, "report_date": fd + pd.Timedelta(days=30), "revenue": rev, "gross_profit": rev * 0.6, "operating_income": rev * om,
                         "rnd": rev * rnd, "sga": rev * 0.2, "ebitda": rev * (om + 0.05), "total_assets": rev * 8, "goodwill": rev * 0.5, "cash_and_st_investments": rev * 2,
                         "total_debt": rev * 0.5, "shares_outstanding": sh, "operating_cf": rev * (om + 0.03), "capex": -rev * 0.05, "sbc": rev * 0.02, "sector": sector})


def test_asof_uses_report_date_only():
    s = _stmts("A", "TECHNOLOGY")
    cut = fc.asof(s, "2024-01-15")                           # 2023-12-31 분기는 2024-01-30 에 발표 → 아직 모른다
    assert cut["fiscal_date"].max() == pd.Timestamp("2023-09-30")
    f_now = fc.quant_features(cut)
    later = s.copy(); later.loc[later["fiscal_date"] > "2023-09-30", "revenue"] *= 100
    assert fc.quant_features(fc.asof(later, "2024-01-15")).loc["A", "rev_ttm"] == pytest.approx(f_now.loc["A", "rev_ttm"])


def test_quant_features_and_scores():
    frames = [_stmts("FAST", "TECHNOLOGY", g=0.05, om=0.30, rnd=0.10), _stmts("SLOW", "TECHNOLOGY", g=0.0, om=0.05, rnd=0.10),
              _stmts("DILUTE", "TECHNOLOGY", g=0.03, om=0.15, sh_g=0.03), _stmts("SPLIT", "TECHNOLOGY", g=0.03, om=0.15, split_at=20),
              _stmts("SHOP", "CONSUMER CYCLICAL", g=0.03, om=0.10, rnd=0.0), _stmts("BANK", "FINANCIAL SERVICES")]
    f = fc.quant_features(fc.asof(pd.concat(frames), "2026-01-01"))
    assert f.loc["FAST", "rev_cagr3"] == pytest.approx(1.05 ** 4 - 1, rel=1e-6) and f.loc["FAST", "om_ttm"] == pytest.approx(0.30)
    assert f.loc["SPLIT", "sh_chg3_ann"] == pytest.approx(0.0, abs=1e-9)            # 10:1 분할은 희석이 아니다
    assert f.loc["DILUTE", "sh_chg3_ann"] == pytest.approx(1.03 ** 4 - 1, rel=1e-6)
    ok = fc.screen(f)
    assert ok["FAST"] and not ok["BANK"]
    s = fc.quant_scores(f[ok])
    assert s.loc["FAST", "p1"] == 3 and s.loc["SLOW", "p1"] == 0 and s.loc["FAST", "p5"] == 3 and s.loc["SLOW", "p5"] <= 1
    assert np.isnan(s.loc["SHOP", "p3"])                                             # R&D 없는 업종은 해당 없음
    assert s.loc["DILUTE", "p13"] == 0 and s.loc["FAST", "p13"] == 3
    assert s.loc["FAST", "quant_avg"] > s.loc["SLOW", "quant_avg"]


def test_verdict_rules():
    good = {k: 2 for k in range(1, 16)}
    assert fc.fisher_verdict(good)["passed"]
    assert not fc.fisher_verdict({**good, 15: 0})["passed"]                          # 정직성 거부권
    assert not fc.fisher_verdict(good, promotional=True)["passed"]
    assert not fc.fisher_verdict({**good, 1: 0, 2: 0, 3: 0})["passed"]               # 0점 3개
    unk = {**good, **{k: None for k in (4, 7, 8, 9, 14, 10)}}
    v = fc.fisher_verdict(unk)
    assert not v["passed"] and len(v["unknown"]) == 6 and v["avg"] == 2.0
    assert fc.fisher_verdict({**good, **{k: 3 for k in (1, 5)}, 4: None})["avg"] == pytest.approx((12 * 2 + 6) / 14)


def test_split_items_prefers_body_over_toc_and_masks_names():
    toc = "Item 1. Business 3\nItem 1A. Risk Factors 9\nItem 3. Legal Proceedings 20\nItem 7. Management's Discussion and Analysis 30\nItem 9A. Controls and Procedures 60\n"
    body = ("PART I\nItem 1.\nBusiness\n" + "Acme Widgets Inc makes widgets. " * 200 + "\nHuman Capital\nWe had 5,000 employees.\n"
            + "Item 1A. | Risk Factors\n" + "risk " * 500 + "\nItem 3. Legal Proceedings\nNone.\nItem 7. Management’s Discussion and Analysis of Financial Condition\n" + "md&a " * 400
            + "\nItem 9A. Controls and Procedures\nEffective.\nItem 9B. Other Information\nNone.")
    it = split_items(toc + body)
    assert len(it["1"]) > 5000 and "Human Capital" in it["1"] and it["7"].count("md&a") == 400 and "Effective." in it["9a"] and "risk" in it["1a"]
    assert window(it["1"], r"human\s+capital", 60).startswith("Human Capital")
    m = mask_company("Acme Widgets Inc and ACME WIDGETS sell under ticker ACME; ACMEX is unrelated.", ["ACME WIDGETS INC"], ["ACME"])
    assert "Acme" not in m and "ACME WIDGETS" not in m and "ACMEX" in m and "[티커]" in m
