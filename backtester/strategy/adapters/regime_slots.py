"""프로토콜 §5 — 레짐 라벨(point-in-time)·에피소드·슬롯 평가·스티치 (순수 pandas/numpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams
from strategy.adapters.multi_asset_sim import simulate_weights

CODES = ("A", "B", "C", "D")
NAMES = {"A": "강세", "B": "조정·짧은 약세", "C": "긴 약세장", "D": "회복 초기"}
COLS = ["TQQQ", "QLD", "QQQ"]


@dataclass(frozen=True)
class RegimeParams:
    n: int = 200
    dd_thr: float = 0.10
    long_days: int = 120
    recover_days: int = 60
    confirm_below: int = 5      # 200일선 아래 '약세 상태' 진입에 필요한 연속 일수(히스테리시스)
    confirm_above: int = 10     # 약세 상태에서 벗어나는 데 필요한 연속 일수
    min_bear_for_recovery: int = 20   # 이만큼 이상 약세였던 뒤의 복귀만 R-D
    c_depth: float | None = None      # H26: 약세 상태에서 낙폭 ≥ c_depth 이면 체류 일수와 무관하게 R-C


def regime_labels(qqq_close: pd.Series, p: RegimeParams = RegimeParams()) -> pd.Series:
    """R-A/B/C/D 라벨(point-in-time, 히스테리시스). 우선순위 D > C > B > A. SMA 미계산 구간은 'A'.

    약세 상태(bear) 는 종가 < 200일선이 confirm_below 일 연속이면 켜지고, > 200일선이 confirm_above 일 연속이면 꺼진다.
    bear 가 켜진 동안 체류 일수 ≥ long_days 면 C, 아니면 B. bear 가 꺼지면(그 bear 가 ≥ min_bear_for_recovery 일이었을 때) recover_days 동안 D.
    bear 가 아니어도 52주 고점 대비 낙폭 ≥ dd_thr 이면 B.
    """
    c = qqq_close.dropna()
    sma = c.rolling(p.n).mean()
    dd = c / c.rolling(252, min_periods=20).max() - 1.0
    below = (c < sma).to_numpy(); ddv = dd.to_numpy(); nan = sma.isna().to_numpy()
    out = np.empty(len(c), dtype=object)
    bear = False; run_below = 0; run_above = 0; bear_days = 0; since_recover = 10**9; last_bear_len = 0
    for i in range(len(c)):
        if nan[i]:
            out[i] = "A"; continue
        if below[i]:
            run_below += 1; run_above = 0
        else:
            run_above += 1; run_below = 0
        if not bear and run_below >= p.confirm_below:
            bear = True; bear_days = run_below; since_recover = 10**9
        elif bear and run_above >= p.confirm_above:
            bear = False; last_bear_len = bear_days; bear_days = 0; since_recover = run_above - 1
        if bear:
            bear_days += 1 if run_below >= p.confirm_below and bear_days >= run_below else 1
        else:
            since_recover += 1
        if not bear and since_recover < p.recover_days and last_bear_len >= p.min_bear_for_recovery:
            out[i] = "D"
        elif bear and (bear_days >= p.long_days or (p.c_depth is not None and ddv[i] <= -p.c_depth)):
            out[i] = "C"
        elif bear or ddv[i] <= -p.dd_thr:
            out[i] = "B"
        else:
            out[i] = "A"
    return pd.Series(out, index=c.index, name="regime")


def episodes(labels: pd.Series, min_days: int = 20) -> pd.DataFrame:
    """연속 구간 (code, start, end, days). min_days 미만은 평가에서 제외(flag)."""
    v = labels.to_numpy(); idx = labels.index
    change = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.concatenate([[0], change]); ends = np.concatenate([change - 1, [len(v) - 1]])
    rows = [{"code": v[s], "start": idx[s], "end": idx[e], "days": e - s + 1, "eval": (e - s + 1) >= min_days} for s, e in zip(starts, ends)]
    return pd.DataFrame(rows)


def as_frame(target: pd.Series | pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """부품 목표 비중을 열 TQQQ·QLD·QQQ DataFrame 으로."""
    if isinstance(target, pd.Series):
        f = pd.DataFrame({"TQQQ": target}, index=target.index)
    else:
        f = target.copy()
    for c in COLS:
        if c not in f.columns:
            f[c] = 0.0
    return f[COLS].reindex(index).ffill().fillna(0.0)


def run_slice(prices: pd.DataFrame, target: pd.DataFrame, start, end, costs: CostParams, cash_rate) -> float:
    """구간 [start, end] 을 자본 1 로 돌린 세후 청산 배수."""
    px = prices[(prices.index >= start) & (prices.index <= end)]
    if len(px) < 5:
        return np.nan
    t = target.reindex(px.index).ffill()
    if t.iloc[0].isna().any():
        prev = target[target.index < px.index[0]]
        t = t.fillna(prev.iloc[-1] if len(prev) else 0.0)
    r = simulate_weights(px, t, costs, cash_rate, "slice")
    return r.final_after_liquidation / costs.initial_capital


def episode_table(prices: pd.DataFrame, comp: pd.DataFrame, hold: pd.DataFrame, eps: pd.DataFrame, costs: CostParams, cash_rate) -> pd.DataFrame:
    rows = []
    for e in eps[eps["eval"]].itertuples(index=False):
        m = run_slice(prices, comp, e.start, e.end, costs, cash_rate); h = run_slice(prices, hold, e.start, e.end, costs, cash_rate)
        rows.append({"code": e.code, "start": e.start, "end": e.end, "days": e.days, "comp": m, "hold": h, "ratio": m / h if h > 0 else np.inf})
    return pd.DataFrame(rows)


def slot_stats(tbl: pd.DataFrame, code: str) -> dict:
    t = tbl[tbl["code"] == code]
    if t.empty:
        return {"n": 0}
    return {"n": int(len(t)), "win_rate": float((t["ratio"] > 1).mean()), "ratio_median": float(t["ratio"].median()),
            "ratio_min": float(t["ratio"].min()), "ratio_max": float(t["ratio"].max()), "days": int(t["days"].sum())}


def slot_pass(st: dict, placebo_pct: float | None) -> bool:
    return st.get("n", 0) >= 3 and st["win_rate"] >= 0.70 and st["ratio_median"] >= 1.0 and st["ratio_min"] >= 0.5 and (placebo_pct is None or placebo_pct >= 95.0)


def placebo_slot_medians(tbl: pd.DataFrame, code: str, n_seeds: int = 30, seed0: int = 20261110) -> list[float]:
    """에피소드 라벨을 섞어 같은 개수의 가짜 code 에피소드로 부품 비율 중앙값 분포."""
    k = int((tbl["code"] == code).sum()); rng = np.random.default_rng(seed0); out = []
    ratios = tbl["ratio"].to_numpy()
    for _ in range(n_seeds):
        pick = rng.choice(len(ratios), size=k, replace=False)
        out.append(float(np.median(ratios[pick])))
    return out


def stitch(slots: dict[str, pd.DataFrame], labels: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    lab = labels.reindex(index).ffill().fillna("A")
    out = pd.DataFrame(0.0, index=index, columns=COLS)
    for code, f in slots.items():
        m = (lab == code).to_numpy()
        out.loc[m] = f.reindex(index).ffill().fillna(0.0).loc[m].to_numpy()
    return out


def shuffle_labels(labels: pd.Series, seed: int) -> pd.Series:
    """에피소드 단위로 라벨을 섞는다(에피소드 길이·개수 보존)."""
    eps = episodes(labels, 1); rng = np.random.default_rng(seed)
    codes = eps["code"].to_numpy().copy(); rng.shuffle(codes)
    out = labels.copy()
    for (s, e), c in zip(eps[["start", "end"]].itertuples(index=False), codes):
        out.loc[s:e] = c
    return out
