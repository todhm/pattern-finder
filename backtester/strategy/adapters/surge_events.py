"""급등주 탐색 — 패널 조립·급등 사건·사건 전 특징 (순수 pandas/numpy).

원천: `surge_delisted_cli.py` 가 받은 종목별 AV 원본(o,h,l,c,adj,v,div,split). 여기서
- 분할만 소급 조정한 가격(O,H,L,C)·거래량(V) 과 원가격 종가(raw_c)·거래대금(dv = 원가격 × 원거래량) 을 만든다.
  수익률은 분할 조정가로, 가격대·유동성 필터는 원가격으로 본다(역분할 소급 조정이 만드는 가짜 가격대 방지).
- 폐지 종목은 폐지일 뒤 행을 버린다(티커 재사용 = 다른 회사).
계획 docs/strategy_notes/surge_search_runs/급등주_탐색_계획.md
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# 워런트·유닛·권리·우선주 등 보통주가 아닌 표기 (AV: -WS, -U, -R, -P-, Nasdaq 5글자 끝 W/U/R)
_NONCOMMON = re.compile(r"(-WS|-WT|-U|-UN|-R|-RT|-P-|-P$|/|\.)|^[A-Z]{4}[WUR]$")
# 거래소 시험용 티커(ZVZZT·ZXZZT·NTEST·PTEST …) — 가격이 임의로 찍혀 가짜 급등이 수백 건 나온다
_TEST = re.compile(r"^Z[A-Z]ZZT$|TEST|^ZZZ|^ZAZZT|^ZBZX|^ZEXIT|^ZIEXT|^ZXIET|^ZTST")


def is_common(symbol: str) -> bool:
    return _NONCOMMON.search(symbol) is None and _TEST.search(symbol) is None


def load_symbol(path: Path, delisting_date: pd.Timestamp | None = None, ipo_date: pd.Timestamp | None = None) -> pd.DataFrame | None:
    raw = pd.read_parquet(path)
    if raw.empty:
        return None
    raw = raw.sort_index()
    if delisting_date is not None:
        raw = raw[raw.index <= delisting_date + pd.Timedelta(days=5)]
    if ipo_date is not None:
        raw = raw[raw.index >= ipo_date - pd.Timedelta(days=5)]
    raw = raw[(raw["c"] > 0)]
    if len(raw) < 30:
        return None
    split = raw["split"].replace(0.0, 1.0)
    later = split[::-1].cumprod()[::-1].shift(-1).fillna(1.0)     # 그날 이후 발효되는 분할 계수의 곱
    out = pd.DataFrame({"O": raw["o"] / later, "H": raw["h"] / later, "L": raw["l"] / later, "C": raw["c"] / later,
                        "V": raw["v"] * later, "raw_c": raw["c"], "dv": raw["c"] * raw["v"], "split": split}, index=raw.index)
    return out


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """한 종목 프레임에 사건 전 특징(전부 전날까지 = shift(1)) 과 사건 후 결과를 붙인다."""
    c, o, h, v, dv = d["C"], d["O"], d["H"], d["V"], d["dv"]
    f = d.copy()
    f["ret"] = c / c.shift(1) - 1.0
    f["gap"] = o / c.shift(1) - 1.0
    f["hi_ret"] = h / c.shift(1) - 1.0
    p = lambda s: s.shift(1)  # noqa: E731 — 전날까지
    f["p_raw_c"] = p(d["raw_c"])
    f["p_dv20"] = p(dv.rolling(20, min_periods=10).mean())
    f["p_dv5"] = p(dv.rolling(5, min_periods=3).mean())
    f["p_vol_ratio5"] = p(v.rolling(5, min_periods=3).mean() / v.rolling(60, min_periods=20).mean())
    f["p_ret5"] = p(c / c.shift(5) - 1.0); f["p_ret20"] = p(c / c.shift(20) - 1.0); f["p_ret60"] = p(c / c.shift(60) - 1.0)
    f["p_dd252"] = p(c / c.rolling(252, min_periods=60).max() - 1.0)
    f["p_from_low252"] = p(c / c.rolling(252, min_periods=60).min() - 1.0)
    f["p_vs_sma50"] = p(c / c.rolling(50, min_periods=30).mean() - 1.0); f["p_vs_sma200"] = p(c / c.rolling(200, min_periods=100).mean() - 1.0)
    r = c.pct_change()
    f["p_vol20"] = p(r.rolling(20, min_periods=10).std()); f["p_vol60"] = p(r.rolling(60, min_periods=30).std())
    f["p_range10"] = p((h.rolling(10).max() - d["L"].rolling(10).min()) / c)
    f["p_big_days_252"] = p((r >= 0.3).rolling(252, min_periods=60).sum())          # 지난 1년 +30% 이상 날 수(급등 이력)
    f["p_days_listed"] = np.arange(len(d))
    f["p_split_near"] = p((d["split"] != 1.0).rolling(10, min_periods=1).max())      # 최근 10일 분할(가짜 급등 의심)
    # 사건 후: 다음날 시가에 사서 n 일 뒤 종가
    nxt_o = o.shift(-1)
    f["n_open_gap"] = nxt_o / c - 1.0
    for n in (1, 5, 20, 60):
        f[f"fwd{n}"] = c.shift(-n) / nxt_o - 1.0
    f["fwd_max20"] = h.shift(-1)[::-1].rolling(20, min_periods=1).max()[::-1] / nxt_o - 1.0
    f["days_left"] = np.arange(len(d))[::-1]
    return f


def build_panel(cache: Path, listing: pd.DataFrame, since: str = "2017-01-01") -> pd.DataFrame:
    """listing: symbol, ipo_date, delisting_date(폐지면 값), exchange. 보통주만."""
    frames = []
    for row in listing.itertuples(index=False):
        s = row.symbol
        if not is_common(s):
            continue
        path = cache / (s.replace("/", "_") + ".parquet")
        if not path.exists():
            continue
        dl = pd.Timestamp(row.delisting_date) if isinstance(row.delisting_date, str) and row.delisting_date else None
        ipo = pd.Timestamp(row.ipo_date) if isinstance(row.ipo_date, str) and row.ipo_date else None
        d = load_symbol(path, dl, ipo if dl is None else None)
        if d is None:
            continue
        f = add_features(d); f = f[f.index >= since]
        if f.empty:
            continue
        f.insert(0, "symbol", s); f["delisted"] = dl is not None
        frames.append(f)
    out = pd.concat(frames); out.index.name = "date"
    return out.reset_index()


def real_surge(f: pd.DataFrame, thr: float, min_day_dv: float = 1e6) -> pd.Series:
    """급등 사건: 종가 수익 ≥ thr 이고 그날 실제로 거래가 붙었다(거래대금 ≥ min_day_dv).
    ADR 비율 변경·티커 재사용·호가만 바뀐 날(거래량 0)은 가격만 뛰고 거래가 없다."""
    return (f["ret"] >= thr) & (f["dv"] >= min_day_dv)


def liquid(f: pd.DataFrame, min_price: float = 1.0, min_dv20: float = 1e6) -> pd.Series:
    """전날 기준 체결 가능: 원가격 ≥ min_price, 20일 평균 거래대금 ≥ min_dv20, 최근 분할 없음."""
    return (f["p_raw_c"] >= min_price) & (f["p_dv20"] >= min_dv20) & (f["p_split_near"].fillna(0) == 0)


def hold_returns(f: pd.DataFrame, horizons=(1, 5, 20), cap: float = 5.0) -> pd.DataFrame:
    """패널(date, symbol 행)에 "당일 시가 매수 → n 일째 종가" 수익 h{n} 과 전날 수익 p_ret1 을 붙인다.
    폐지로 n 일째가 없으면 마지막 종가로 청산. 살아 있는 종목의 패널 끝은 NaN. 한 건 수익은 cap 에서 자른다."""
    f = f.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = f.groupby("symbol", sort=False)
    last_c = g["C"].transform("last")
    f["p_ret1"] = g["ret"].shift(1)
    for n in horizons:
        c_n = g["C"].shift(-(n - 1))
        c_n = c_n.where(c_n.notna() | ~f["delisted"], last_c)
        f[f"h{n}"] = (c_n / f["O"] - 1.0).clip(upper=cap)
    return f


def event_frame(f: pd.DataFrame, min_ret: float = 0.15, horizons=(20, 60, 120), cap: float = 5.0, min_day_dv: float = 1e6) -> pd.DataFrame:
    """급등 사건 행만 뽑아 사건일 특징(dd_now·close_pos·vol_mult)과 "다음날 시가 → n 일째 종가" 수익 e{n} 을 붙인다.
    폐지로 n 일째가 없으면 마지막 종가 청산(다음날 시가조차 없으면 사건 제외), 살아 있는 종목의 패널 끝은 NaN."""
    f = f.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = f.groupby("symbol", sort=False)
    last_c = g["C"].transform("last")
    nxt_o = g["O"].shift(-1)
    cols = {}
    for n in horizons:
        c_n = g["C"].shift(-n)
        c_n = c_n.where(c_n.notna() | ~f["delisted"], last_c)
        cols[f"e{n}"] = (c_n / nxt_o - 1.0).clip(upper=cap)
    cols["dd_now"] = g["p_dd252"].shift(-1)                                  # 사건일 종가 기준 52주 고점 대비
    rng = (f["H"] - f["L"]).replace(0, np.nan)
    cols["close_pos"] = ((f["C"] - f["L"]) / rng).fillna(0.5)
    cols["vol_mult"] = f["dv"] / f["p_dv20"]
    f = f.assign(**cols)
    ok = liquid(f) & real_surge(f, min_ret, min_day_dv) & nxt_o.notna()
    return f[ok].reset_index(drop=True)
