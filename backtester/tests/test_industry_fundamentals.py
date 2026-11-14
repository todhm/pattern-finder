import pandas as pd

from strategy.adapters.industry_fundamentals import filing_growth, industry_panel, sic_to_ind


def _raw():
    rows = []
    for cik in range(1, 7):
        for filed, dd, cur in (("2015-05-10", "2015-03-31", 120.0 + cik), ("2015-08-10", "2015-06-30", 150.0)):
            prev = str(pd.Timestamp(dd) - pd.DateOffset(years=1))[:10]
            rows += [(cik, 3674, "10-Q", filed, dd, 1, "rev", cur), (cik, 3674, "10-Q", filed, prev, 1, "rev", 100.0), (cik, 3674, "10-Q", filed, dd, 2, "rev", 999.0), (cik, 3674, "10-Q", filed, dd, 1, "rev", cur - 50)]
    d = pd.DataFrame(rows, columns=["cik", "sic", "form", "filed", "ddate", "qtrs", "item", "value"]); d["filed"] = pd.to_datetime(d["filed"]); d["ddate"] = pd.to_datetime(d["ddate"]); return d


def test_growth_comes_from_one_filing_and_ignores_ytd_rows():
    fg = filing_growth(_raw()); a = fg[(fg["cik"] == 1) & (fg["filed"] == "2015-05-10")].iloc[0]
    assert a["rev_cur"] == 121.0 and a["rev_prev"] == 100.0 and len(fg) == 12          # 누적(qtrs 2) 줄과 작은 태그 값은 버린다


def test_panel_uses_only_filings_already_filed():
    fg = filing_growth(_raw()); fg["ind"] = sic_to_ind(fg["sic"], pd.DataFrame([(3674, 3674, "Chips")], columns=["lo", "hi", "ind"]))
    me = pd.to_datetime(["2015-04-30", "2015-05-31", "2015-08-31", "2016-06-30"]); p = industry_panel(fg, me)
    assert pd.isna(p["g_agg"].loc["2015-04-30"]).all()                                # 아직 아무도 제출 안 함
    assert abs(p["g_agg"].loc["2015-05-31", "Chips"] - (sum(120 + c for c in range(1, 7)) / 600 - 1)) < 1e-9
    assert abs(p["g_agg"].loc["2015-08-31", "Chips"] - 0.5) < 1e-9 and p["hot_frac"].loc["2015-08-31", "Chips"] == 1.0
    assert pd.isna(p["g_agg"].loc["2016-06-30"]).all()                                # 190 일 넘은 제출물은 낡아서 뺀다
    assert sic_to_ind(pd.Series([9999.0]), pd.DataFrame([(3674, 3674, "Chips")], columns=["lo", "hi", "ind"])).iloc[0] == "Other"
