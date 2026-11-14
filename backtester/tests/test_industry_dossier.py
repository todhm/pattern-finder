import numpy as np
import pandas as pd

from strategy.adapters.industry_dossier import card, filing_summary, render, snapshot


def _raw(filed="2016-08-20", dd="2016-06-30", scale=1.0):
    rows = []; prev = str(pd.Timestamp(dd) - pd.DateOffset(years=1))[:10]
    for cik in range(1, 7):
        for item, cur, old, q in (("rev", 130.0 * scale, 100.0, 1), ("rev", 250.0 * scale, 200.0, 2), ("op", 30.0 * scale, 10.0, 2), ("capex", 20.0, 10.0, 2), ("dsh", 110.0, 100.0, 2), ("gp", 150.0 * scale, 100.0, 2)):
            rows += [(cik, f"C{cik}", 3674, "10-Q", filed, item, dd, q, cur), (cik, f"C{cik}", 3674, "10-Q", filed, item, prev, q, old)]
        rows.append((cik, f"C{cik}", 3674, "10-Q", filed, "inv", dd, 0, 50.0 * scale))
    d = pd.DataFrame(rows, columns=["cik", "name", "sic", "form", "filed", "item", "ddate", "qtrs", "value"]); d["filed"] = pd.to_datetime(d["filed"]); d["ddate"] = pd.to_datetime(d["ddate"]); return d


def test_summary_takes_longest_period_with_year_ago_match():
    fs = filing_summary(_raw()); a = fs.iloc[0]
    assert a["rev_cur"] == 250.0 and a["rev_prev"] == 200.0 and a["capex_cur"] == 20.0 and a["inv_bal"] == 50.0 and len(fs) == 6


def test_card_ratios_and_no_future_filings():
    fs = pd.concat([filing_summary(_raw()), filing_summary(_raw("2015-08-20", "2015-06-30", 0.5))])
    assert len(snapshot(fs, "2016-08-01")) == 0                                    # 아직 제출 전 + 1 년 전 것은 낡음
    now, y1 = snapshot(fs, "2016-08-31"), snapshot(fs, "2015-08-31"); c = card(now, y1, y1.iloc[:0])
    assert abs(c["rev_g"] - 0.25) < 1e-9 and abs(c["incr_margin"] - 0.4) < 1e-9 and abs(c["gm"] - 0.6) < 1e-9 and abs(c["gm_chg"] - 0.1) < 1e-9
    assert abs(c["inv_g"] - 1.0) < 1e-9 and abs(c["share_g_median"] - 0.1) < 1e-9 and c["pct_diluting"] == 1.0 and abs(c["top3_share"] - 0.5) < 1e-9
    txt = render({**c, "price_12m": 0.2, "price_36m": np.nan, "n_stocks": 6}, "K"); assert "C1" not in txt and "2016" not in txt and "자료 없음" in txt
