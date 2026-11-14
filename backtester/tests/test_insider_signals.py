import numpy as np
import pandas as pd

from strategy.adapters.insider_signals import attach, buy_windows, first_cross, forward_returns, normalize_symbol


def _tx():
    rows = [("AAA", "2020-01-02", 1, "P", 5e4, True, False), ("AAA", "2020-01-10", 2, "P", 6e4, True, True), ("AAA", "2020-01-20", 1, "P", 7e4, True, False),
            ("AAA", "2020-03-15", 3, "P", 5e4, True, False), ("AAA", "2020-01-11", 9, "S", 9e6, True, False), ("AAA", "2020-01-12", 8, "P", 5e3, True, False),
            ("BBB", "2020-01-05", 4, "P", 2e6, False, False)]
    t = pd.DataFrame(rows, columns=["symbol", "filing_date", "owner", "code", "value", "od", "ceo"]); t["filing_date"] = pd.to_datetime(t["filing_date"])
    return t


def test_normalize_symbol():
    assert normalize_symbol(" nyse: abc ") == "ABC" and normalize_symbol("BRK.B") == "BRK-B"
    assert normalize_symbol("NONE") is None and normalize_symbol(np.nan) is None and normalize_symbol("NOT A TICKER AT ALL") is None


def test_buy_windows_counts_distinct_buyers_in_window_only():
    bw = buy_windows(_tx()).set_index(["symbol", "date"])
    a = bw.loc["AAA"]
    assert list(a["n_buyers"]) == [1, 2, 2, 1]                  # 같은 사람 두 번은 한 명, 매도·소액·비임원 제외, 3월은 창 밖
    assert a["value"].iloc[2] == 5e4 + 6e4 + 7e4 and bool(a["ceo"].iloc[1])
    assert "BBB" not in bw.index.get_level_values(0)
    assert "BBB" in buy_windows(_tx(), od_only=False)["symbol"].values


def test_first_cross_cooldown():
    bw = buy_windows(_tx())
    e = first_cross(bw, min_buyers=2)
    assert list(e["date"].dt.strftime("%m-%d")) == ["01-10"]    # 01-20 은 쿨다운 안
    assert len(first_cross(bw, min_buyers=1, cooldown_days=30)) == 2


def test_forward_returns_and_attach_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=10)
    p = pd.DataFrame({"date": idx, "symbol": "AAA", "O": np.arange(10.0, 20.0), "C": np.arange(10.5, 20.5), "delisted": True})
    f = forward_returns(p, horizons=(2, 20))
    r0 = f.iloc[0]
    assert np.isclose(r0["e2"], 12.5 / 11.0 - 1)                # 다음날 시가 11 → 2 거래일째 종가 12.5
    assert np.isclose(r0["e20"], 19.5 / 11.0 - 1)               # 폐지 종목은 마지막 종가
    ev = pd.DataFrame({"symbol": ["AAA", "AAA", "ZZZ"], "date": pd.to_datetime(["2020-01-04", "2020-03-01", "2020-01-02"])})   # 토요일 공시 / 너무 뒤 / 패널에 없음
    a = attach(ev, f)
    assert len(a) == 1 and a["date"].iloc[0] == pd.Timestamp("2020-01-03") and a["filing"].iloc[0] == pd.Timestamp("2020-01-04")
