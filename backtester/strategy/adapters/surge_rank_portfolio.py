"""생존 편향 없는 패널 위의 "상위 K 종목 월 교체" 포트폴리오 (순수 pandas/numpy).

신호는 리밸런스일 종가까지, 체결은 다음 거래일 시가. 다음 리밸런스 체결 시가까지 동일비중 보유.
폐지로 가격이 끊긴 종목은 마지막 종가에 현금으로 바뀐 것으로 본다. 레버리지·공매도 없음.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankRules:
    rank: str = "mom12_1"          # mom12_1 | mom6 | mom3 | near_high | random
    k: int = 10
    min_price: float = 5.0
    min_dv20: float = 2e7
    every: int = 21                # 리밸런스 간격(거래일)
    cost: float = 0.005            # 교체 한 번(매도+매수)에 드는 비율 — 바뀐 비중에만
    max_ret20: float | None = None # 직전 20일 수익이 이보다 크면 제외(급등 직후 회피)
    above_sma200: bool = False


def wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """long 패널 → 날짜×종목 행렬. C_ff/O_ff 는 폐지 뒤 마지막 종가로 채움."""
    out = {c: panel.pivot(index="date", columns="symbol", values=c) for c in ("C", "O", "raw_c", "dv")}
    alive = out["C"].notna()
    started = alive.cummax()
    c_ff = out["C"].ffill().where(started)
    out["alive"] = alive
    out["C_ff"] = c_ff
    out["O_ff"] = out["O"].where(alive, c_ff)
    return out


def scores(w: dict[str, pd.DataFrame], rank: str) -> pd.DataFrame:
    c = w["C"]
    if rank == "mom12_1":
        return c.shift(21) / c.shift(252) - 1.0
    if rank == "mom6":
        return c / c.shift(126) - 1.0
    if rank == "mom3":
        return c / c.shift(63) - 1.0
    if rank == "near_high":
        return c / c.rolling(252, min_periods=200).max() - 1.0
    raise ValueError(rank)


def run_rank(w: dict[str, pd.DataFrame], rules: RankRules, start: str, end: str, rng: np.random.Generator | None = None,
             score: pd.DataFrame | None = None, extra_ok: pd.DataFrame | None = None, market_ok: pd.Series | None = None) -> pd.DataFrame:
    """리밸런스 구간별 표: date(체결일), ret(비용 차감), n, turnover, picks."""
    idx = w["C"].index
    pos = np.arange(len(idx))[(idx >= start) & (idx <= end)]
    sig = pos[::rules.every]
    sig = sig[sig + 1 < len(idx)]
    c = w["C"]
    dv20 = w["dv"].rolling(20, min_periods=10).mean()
    if rules.rank != "random" and score is None:
        score = scores(w, rules.rank)
    ret20 = c / c.shift(20) - 1.0 if rules.max_ret20 is not None else None
    sma200 = c.rolling(200, min_periods=150).mean() if rules.above_sma200 else None
    o_ff = w["O_ff"].to_numpy(); cols = w["C"].columns.to_numpy()
    rows, prev = [], set()
    for j, s in enumerate(sig):
        ok = w["alive"].iloc[s] & (w["raw_c"].iloc[s] >= rules.min_price) & (dv20.iloc[s] >= rules.min_dv20)
        if ret20 is not None:
            ok &= ret20.iloc[s] <= rules.max_ret20
        if sma200 is not None:
            ok &= c.iloc[s] > sma200.iloc[s]
        if extra_ok is not None:                      # 호출 쪽이 만든 추가 소거 조건(날짜×종목 bool, 신호일 종가까지의 정보)
            ok &= extra_ok.iloc[s].fillna(False)
        if market_ok is not None and not bool(market_ok.iloc[s]):   # 시장 필터 꺼짐 → 이번 구간 현금
            ok &= False
        if rules.rank == "random":
            cand = np.flatnonzero(ok.to_numpy())
            pick = rng.choice(cand, size=min(rules.k, len(cand)), replace=False) if len(cand) else np.array([], dtype=int)
        else:
            sc = score.iloc[s].where(ok).dropna()
            pick = c.columns.get_indexer(sc.nlargest(rules.k).index)
        e = s + 1
        x = (sig[j + 1] + 1) if j + 1 < len(sig) else min(len(idx) - 1, e + rules.every)
        if x <= e:
            break
        if len(pick):
            r = o_ff[x, pick] / o_ff[e, pick] - 1.0
            r = np.where(np.isfinite(r), r, 0.0)
            gross = float(r.mean())
        else:
            gross = 0.0
        now = set(cols[pick])
        turn = 0.0 if not now else (1.0 if not prev else len(now - prev) / len(now))   # 현금 구간은 비용 없음
        rows.append({"date": idx[e], "exit": idx[x], "ret": (1 + gross) * (1 - rules.cost * turn) - 1.0, "n": len(pick), "turnover": turn, "picks": ",".join(sorted(now))})
        prev = now
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame) -> dict:
    eq = (1 + t["ret"]).cumprod()
    yrs = max((t["exit"].iloc[-1] - t["date"].iloc[0]).days / 365.25, 0.1)
    return {"mult": float(eq.iloc[-1]), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "mdd": float((eq / eq.cummax().clip(lower=1.0) - 1).min()),
            "win": float((t["ret"] > 0).mean()), "turnover": float(t["turnover"].mean())}


def bench_table(o: pd.Series, t: pd.DataFrame) -> pd.DataFrame:
    """같은 체결일·청산일에 시가로 사고판 벤치마크."""
    b = t[["date", "exit"]].copy()
    b["ret"] = o.reindex(b["exit"]).to_numpy() / o.reindex(b["date"]).to_numpy() - 1.0
    b["turnover"] = 0.0
    return b


def daily_curve(w: dict[str, pd.DataFrame], t: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """run_rank 표 → 일별 자산 곡선(종가 평가). 체결 전·표가 끝난 뒤는 직전 값 유지(현금)."""
    c_ff, o_ff = w["C_ff"], w["O_ff"]
    eq = pd.Series(np.nan, index=index); v = 1.0
    eq.iloc[0] = 1.0
    for row in t.itertuples():
        picks = [s for s in row.picks.split(",") if s]
        seg = index[(index >= row.date) & (index < row.exit)]
        if picks and len(seg):
            rel = (c_ff.loc[seg, picks] / o_ff.loc[row.date, picks]).fillna(1.0).mean(axis=1)
            eq.loc[seg] = v * rel
        elif len(seg):
            eq.loc[seg] = v
        v *= 1.0 + row.ret
        if row.exit in eq.index:
            eq.loc[row.exit] = v                      # 청산일 시가 기준 확정값(같은 날 새 구간이 덮어쓴다)
    return eq.ffill()


def tranche_curve(w: dict[str, pd.DataFrame], tables: list[pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    """시차를 둔 분할 계좌들의 평균 곡선(처음에 1/N 씩 나눠 각자 굴림, 계좌 간 재조정 없음)."""
    return pd.concat([daily_curve(w, t, index) for t in tables], axis=1).mean(axis=1)


def curve_stats(eq: pd.Series) -> dict:
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    return {"mult": float(eq.iloc[-1] / eq.iloc[0]), "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1), "mdd": float((eq / eq.cummax() - 1).min())}
