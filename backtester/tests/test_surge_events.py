"""급등주 패널 — 분할 조정·폐지일 자르기·특징의 미래 비참조·유동성 필터."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import add_features, build_panel, is_common, liquid, load_symbol


def _raw(n=400, seed=3, split_at=None, ratio=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    c = 20 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    split = np.ones(n)
    if split_at is not None:
        c[split_at:] = c[split_at:] / ratio; split[split_at] = ratio     # 원가격은 분할일에 1/ratio 로 뛴다
    o = c * (1 + rng.normal(0, 0.005, n))
    v = rng.integers(200_000, 400_000, n).astype(float)
    return pd.DataFrame({"o": o, "h": np.maximum(o, c) * 1.01, "l": np.minimum(o, c) * 0.99, "c": c, "adj": c, "v": v, "div": 0.0, "split": split}, index=idx)


def test_is_common():
    assert is_common("AAPL") and is_common("F")
    for s in ("BC/PA", "ABCDW", "ABCDU", "XYZ-WS", "XYZ-U", "BRK.B", "XYZ-P-A"):
        assert not is_common(s), s


def test_split_adjust_removes_fake_jump(tmp_path):
    for ratio in (10.0, 0.1):                                             # 10:1 분할, 1:10 역분할
        raw = _raw(split_at=200, ratio=ratio); p = tmp_path / f"S{ratio}.parquet"; raw.to_parquet(p)
        d = load_symbol(p)
        r = d["C"].pct_change().abs()
        assert r.iloc[200] < 0.2                                          # 분할일 가짜 급등락 없음
        assert np.isclose(d["C"].iloc[-1], raw["c"].iloc[-1])             # 최근 가격은 원가격 그대로
        assert np.allclose(d["dv"], raw["c"] * raw["v"])                  # 거래대금은 원가격 기준
        f = add_features(d)
        assert f["p_split_near"].iloc[201] == 1 and f["p_split_near"].iloc[150] == 0


def test_delisting_cut_and_short_history(tmp_path):
    raw = _raw(); p = tmp_path / "X.parquet"; raw.to_parquet(p)
    cut = raw.index[250]
    d = load_symbol(p, delisting_date=cut)
    assert d.index.max() <= cut + pd.Timedelta(days=5) and len(d) < len(raw)
    assert load_symbol(p, delisting_date=raw.index[10]) is None           # 30일 미만이면 버림


def test_features_use_only_past():
    raw = _raw(); raw.index.name = None
    d = pd.DataFrame({"O": raw["o"], "H": raw["h"], "L": raw["l"], "C": raw["c"], "V": raw["v"], "raw_c": raw["c"], "dv": raw["c"] * raw["v"], "split": 1.0})
    f0 = add_features(d)
    k = 300
    d2 = d.copy(); d2.iloc[k:, :5] *= 3.0; d2.iloc[k:, d2.columns.get_loc("raw_c")] *= 3.0; d2["dv"] = d2["raw_c"] * d2["V"]
    f1 = add_features(d2)
    pcols = [c for c in f0.columns if c.startswith("p_")]
    # k 일(포함) 이후를 바꿔도 k 일의 사건 전 특징은 그대로 — 당일·미래를 보지 않는다
    assert np.allclose(f0.loc[f0.index[: k + 1], pcols].to_numpy(dtype=float), f1.loc[f1.index[: k + 1], pcols].to_numpy(dtype=float), equal_nan=True)
    # 결과: 다음날 시가 매수 → n 일 뒤 종가
    i = 100
    assert np.isclose(f0["fwd5"].iloc[i], d["C"].iloc[i + 5] / d["O"].iloc[i + 1] - 1)
    assert np.isclose(f0["fwd_max20"].iloc[i], d["H"].iloc[i + 1 : i + 21].max() / d["O"].iloc[i + 1] - 1)
    assert f0["days_left"].iloc[-1] == 0


def test_build_panel_and_liquid(tmp_path):
    _raw(seed=1).to_parquet(tmp_path / "AAA.parquet"); _raw(seed=2).to_parquet(tmp_path / "BBB.parquet"); _raw(seed=4).to_parquet(tmp_path / "CCCCW.parquet")
    listing = pd.DataFrame({"symbol": ["AAA", "BBB", "CCCCW", "NOFILE"], "ipo_date": ["2019-01-02", "2019-01-02", "", ""],
                            "delisting_date": [None, "2020-01-15", None, None], "exchange": ["NASDAQ"] * 4})
    pan = build_panel(tmp_path, listing, since="2019-01-01")
    assert set(pan["symbol"]) == {"AAA", "BBB"}
    assert pan.loc[pan["symbol"] == "BBB", "date"].max() <= pd.Timestamp("2020-01-20") and pan.loc[pan["symbol"] == "BBB", "delisted"].all()
    ok = liquid(pan)
    assert ok.sum() > 0 and (pan.loc[ok, "p_dv20"] >= 1e6).all() and (pan.loc[ok, "p_raw_c"] >= 1.0).all()
    assert not liquid(pan, min_price=1e9).any()


def test_test_tickers_and_real_surge():
    from strategy.adapters.surge_events import real_surge
    for s in ("ZVZZT", "ZXZZT", "ZWZZT", "ZJZZT", "NTEST", "PTEST", "NTEST-G"):
        assert not is_common(s), s
    f = pd.DataFrame({"ret": [0.6, 0.6, 0.2], "dv": [5e6, 0.0, 5e6]})
    assert real_surge(f, 0.5).tolist() == [True, False, False]
