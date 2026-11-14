"""내부자 공개시장 매수(Form 4, 거래 코드 P) 군집 신호 (순수 pandas/numpy).

입력 tx: symbol, filing_date, owner, code(P|S), value, od(임원·이사 여부), ceo(대표·재무책임자 여부).
시점은 전부 **공시일(filing_date)** 기준 — 거래일이 아니라 시장이 알게 된 날.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_BAD = {"", "NONE", "NA", "N/A", "NULL", "NAN"}


def normalize_symbol(s) -> str | None:
    """공시에 적힌 티커를 패널 표기로. 'NYSE: abc' → ABC, 'BRK.B' → BRK-B. 쓸 수 없으면 None."""
    if not isinstance(s, str):
        return None
    t = s.strip().upper()
    if ":" in t:
        t = t.split(":")[-1].strip()
    t = re.sub(r"[./ ]", "-", t.strip("()[]\"' ")).strip("-")
    return None if t in _BAD or not re.fullmatch(r"[A-Z0-9-]{1,8}", t) else t


def buy_windows(tx: pd.DataFrame, window_days: int = 30, min_value: float = 1e4, od_only: bool = True) -> pd.DataFrame:
    """매수 공시일마다 직전 window_days(달력일) 안의 서로 다른 매수자 수·매수 금액 합.
    반환: symbol, date(공시일), n_buyers, value, ceo(창 안에 대표·재무책임자 매수가 있나). (symbol, date) 당 한 줄."""
    b = tx[(tx["code"] == "P") & (tx["value"] >= min_value)]
    if od_only:
        b = b[b["od"]]
    b = b.sort_values(["symbol", "filing_date"])
    win = np.timedelta64(window_days, "D"); rows = []
    for sym, g in b.groupby("symbol", sort=False):
        d = g["filing_date"].to_numpy(); o = g["owner"].to_numpy(); v = g["value"].to_numpy(); c = g["ceo"].to_numpy()
        for day in np.unique(d):
            m = (d > day - win) & (d <= day)
            rows.append((sym, day, len(set(o[m])), float(v[m].sum()), bool(c[m].any())))
    return pd.DataFrame(rows, columns=["symbol", "date", "n_buyers", "value", "ceo"])


def first_cross(bw: pd.DataFrame, min_buyers: int = 1, min_total: float = 0.0, cooldown_days: int = 90) -> pd.DataFrame:
    """조건(매수자 ≥ min_buyers, 금액 합 ≥ min_total)을 채운 첫 공시일만. 같은 종목은 cooldown_days 안에 다시 세지 않는다."""
    x = bw[(bw["n_buyers"] >= min_buyers) & (bw["value"] >= min_total)].sort_values(["symbol", "date"])
    keep = []; last_sym, last_day = None, None; cd = np.timedelta64(cooldown_days, "D")
    for sym, day in zip(x["symbol"].to_numpy(), x["date"].to_numpy()):
        ok = sym != last_sym or day - last_day >= cd
        keep.append(ok)
        if ok:
            last_sym, last_day = sym, day
    return x[np.array(keep, dtype=bool)].reset_index(drop=True)


def forward_returns(panel: pd.DataFrame, horizons=(63, 126, 252), cap: float = 5.0) -> pd.DataFrame:
    """패널(date, symbol, O, C, delisted)에 "다음날 시가 매수 → n 거래일째 종가" 수익 e{n}.
    폐지로 n 일째가 없으면 마지막 종가, 살아 있는 종목의 패널 끝은 NaN."""
    f = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = f.groupby("symbol", sort=False); last_c = g["C"].transform("last"); nxt_o = g["O"].shift(-1)
    for n in horizons:
        c_n = g["C"].shift(-n)
        c_n = c_n.where(c_n.notna() | ~f["delisted"], last_c)
        f[f"e{n}"] = (c_n / nxt_o - 1.0).clip(upper=cap)
    return f


def attach(events: pd.DataFrame, panel: pd.DataFrame, tolerance_days: int = 5) -> pd.DataFrame:
    """공시일 이전 가장 가까운 거래일의 패널 행(특징·앞 수익)을 붙인다. 그날 종가까지 알려진 것으로 보고 다음날 시가 매수."""
    ev = events.sort_values("date").rename(columns={"date": "filing"})
    ev["date"] = ev["filing"]
    out = pd.merge_asof(ev, panel.sort_values("date").assign(sig_date=lambda x: x["date"]), on="date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=tolerance_days))
    out = out.dropna(subset=["C"]).reset_index(drop=True)
    out["date"] = out.pop("sig_date")                      # date = 신호를 본 거래일(공시일 또는 그 직전 거래일)
    return out
