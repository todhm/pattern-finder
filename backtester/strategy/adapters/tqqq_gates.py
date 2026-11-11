"""TQQQ 넘기 공용 평가기 — 자산 곡선 → 창 묶음 표 → 관문 G1~G6 판정 (프로토콜 §1, 2026-11-10 고정).

순수 pandas/numpy. 기준선(창별 TQQQ·QQQ 세후)은 ``tqqq_baseline_windows_2026-11-10.csv`` 를 **읽기만** 한다 — 재계산하지 않는다.
전략은 목표 비중 시계열(0~1, 전 구간) 하나로 넘기고, 창마다 자본 1 로 새로 ``simulate_exposure`` 를 돌린다(세금은 창마다 새로).
관문 수치(70%·1.0·0.5·−80%·0.7·60%·95+min(4,N/50)) 는 루프 중 수정 금지.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams, simulate_exposure

BUNDLES = ("A-10", "A-∞", "B-10", "S-5")
EDITIONS = {"dev": "개발(끝 2024-06-30)", "sealed": "봉인 해제(끝 2026-10-24)"}
GATE_COSTS = CostParams(buy_fee=0.001 + 0.0005, sell_fee=0.001 + 0.0000229 + 0.0005, tax_rate=0.22,
                        tax_deduction=2_500_000.0, initial_capital=100_000_000.0, rebalance_band=0.05, exec_lag=1)
# H13 — 한국 세제 정밀: 연 통산·250만 공제·이월 없음(원장 기본) + 배당세 15.4%(가격이 배당 재투자 조정이라 드래그로). 연 배당수익률 가정.
DIVIDEND_YIELDS = {"TQQQ": 0.007, "QQQ": 0.006, "QLD": 0.003, "SOXX": 0.008, "GOLD": 0.0, "LIST": 0.012}
GATE_COSTS_KR = replace(GATE_COSTS, dividend_yield=DIVIDEND_YIELDS["TQQQ"], dividend_tax_rate=0.154)
DOTCOM_TOP = pd.Timestamp("2000-03-01")

TargetLike = pd.Series | Callable[[pd.Timestamp, pd.Timestamp], pd.Series]


# ----------------------------------------------------------------------------- 기준선

def load_baseline(path: str) -> pd.DataFrame:
    """CSV(판, 창, 시작, 끝, TQQQ 세전/세후/MDD/CAGR, QQQ 세후/MDD) → 영문 열."""
    raw = pd.read_csv(path)
    ed = {v: k for k, v in EDITIONS.items()}
    df = pd.DataFrame({
        "edition": raw["판"].map(ed), "bundle": raw["창"],
        "start": pd.to_datetime(raw["시작"]), "end": pd.to_datetime(raw["끝"]),
        "tqqq_pre": raw["TQQQ 세전"].astype(float), "tqqq_ret": raw["TQQQ 세후"].astype(float),
        "tqqq_mdd": raw["TQQQ MDD"].astype(float), "tqqq_cagr": raw["TQQQ 세후 CAGR"].astype(float),
        "qqq_ret": raw["QQQ 세후"].astype(float), "qqq_mdd": raw["QQQ MDD"].astype(float),
    })
    if df["edition"].isna().any():
        raise ValueError(f"알 수 없는 판 이름: {sorted(raw['판'][df['edition'].isna()].unique())}")
    return df


# ----------------------------------------------------------------------------- 창 표

def run_window(lev_close: pd.Series, target: pd.Series | pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
               costs: CostParams = GATE_COSTS, cash_rate: pd.Series | float | None = None,
               prices: pd.DataFrame | None = None, dividend_yields: dict[str, float] | None = None) -> dict:
    """창 하나를 자본 1 로 새로 돌린다. 반환: 세후 청산 수익률·MDD·CAGR·매매·체류.

    ``target`` 이 DataFrame(열 = 자산) 이면 ``prices``(같은 열, TQQQ 포함) 로 다자산 원장(`multi_asset_sim`) 을 쓴다.
    """
    lev = lev_close[(lev_close.index >= start) & (lev_close.index <= end)].dropna()
    if len(lev) < 50:
        raise ValueError(f"창 {start.date()}~{end.date()} 가격 {len(lev)} 일 — 너무 짧다")
    tgt = target.reindex(lev.index).ffill()
    if isinstance(target, pd.DataFrame):
        if tgt.iloc[0].isna().any():
            prev = target[target.index < lev.index[0]]
            tgt = tgt.fillna(prev.iloc[-1] if len(prev) else 0.0)
        from strategy.adapters.multi_asset_sim import simulate_weights
        if prices is None:
            raise ValueError("DataFrame 목표 비중에는 prices 가 필요하다")
        res = simulate_weights(prices.reindex(lev.index), tgt, costs, cash_rate, "strategy", dividend_yields)
    else:
        if tgt.isna().iloc[0]:                       # 창 시작 전에 신호가 없으면 그 앞 마지막 값
            prev = target[target.index < lev.index[0]]
            tgt = tgt.fillna(float(prev.iloc[-1]) if len(prev) else 0.0)
        res = simulate_exposure(lev, tgt, costs, cash_rate, "strategy")
    mult = res.final_after_liquidation / costs.initial_capital
    years = (lev.index[-1] - lev.index[0]).days / 365.25
    return {"strat_ret": mult - 1.0, "strat_mdd": float(res.metrics["mdd"]),
            "strat_cagr": mult ** (1.0 / years) - 1.0 if mult > 0 else -1.0,
            "trades": int(res.trades), "time_in_market": float(res.time_in_market),
            "tax": float(res.tax), "fees": float(res.fees)}


def window_table(lev_close: pd.Series, target: TargetLike, baseline: pd.DataFrame, edition: str = "dev",
                 bundles: tuple[str, ...] = BUNDLES, costs: CostParams = GATE_COSTS,
                 cash_rate: pd.Series | float | None = None, prices: pd.DataFrame | None = None,
                 min_start: pd.Timestamp | None = None, dividend_yields: dict[str, float] | None = None) -> pd.DataFrame:
    """판·묶음의 모든 창을 돌려 기준선과 나란히 붙인 표.

    ``target`` 은 전 구간 Series 또는 ``f(start, end) -> Series``(창마다 다시 맞춰야 하는 전략).
    """
    base = baseline[(baseline["edition"] == edition) & (baseline["bundle"].isin(bundles))]
    if min_start is not None:                        # 자료가 늦게 시작하는 전략: 그 뒤 시작 창만(G4 의 2006 규칙과 같은 처리)
        base = base[base["start"] >= pd.Timestamp(min_start)]
    rows = []
    for b in base.itertuples(index=False):
        tgt = target(b.start, b.end) if callable(target) else target
        r = run_window(lev_close, tgt, b.start, b.end, costs, cash_rate, prices, dividend_yields)
        r.update({"bundle": b.bundle, "start": b.start, "end": b.end, "tqqq_ret": b.tqqq_ret, "tqqq_mdd": b.tqqq_mdd,
                  "tqqq_cagr": b.tqqq_cagr, "qqq_ret": b.qqq_ret, "qqq_mdd": b.qqq_mdd})
        r["ratio"] = (1.0 + r["strat_ret"]) / (1.0 + b.tqqq_ret) if (1.0 + b.tqqq_ret) > 1e-9 else np.inf
        r["beats_tqqq"] = r["strat_ret"] > b.tqqq_ret
        r["beats_qqq"] = r["strat_ret"] > b.qqq_ret
        r["mdd_shallower"] = r["strat_mdd"] > b.tqqq_mdd
        r["has_2000_03"] = bool(b.start <= DOTCOM_TOP <= b.end)
        rows.append(r)
    cols = ["bundle", "start", "end", "strat_ret", "strat_mdd", "strat_cagr", "trades", "time_in_market", "tax", "fees",
            "tqqq_ret", "tqqq_mdd", "tqqq_cagr", "qqq_ret", "qqq_mdd", "ratio", "beats_tqqq", "beats_qqq",
            "mdd_shallower", "has_2000_03"]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------- 관문

@dataclass
class Gate:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)


def _bundle(tbl: pd.DataFrame, name: str) -> pd.DataFrame:
    return tbl[tbl["bundle"] == name]


def gate_g1(tbl: pd.DataFrame) -> Gate:
    d, ok = {}, True
    for b in ("A-10", "A-∞"):
        t = _bundle(tbl, b)
        if t.empty:
            return Gate("G1", False, {"error": f"{b} 창 없음"})
        d[b] = {"n": int(len(t)), "win_rate": float((t["ratio"] > 1).mean()), "ratio_median": float(t["ratio"].median()),
                "ratio_min": float(t["ratio"].min()), "ratio_max": float(t["ratio"].max())}
        ok &= d[b]["win_rate"] >= 0.70 and d[b]["ratio_median"] >= 1.0 and d[b]["ratio_min"] >= 0.5
    return Gate("G1", bool(ok), d)


def gate_g2(tbl: pd.DataFrame) -> Gate:
    t = tbl[tbl["bundle"].isin(["A-10", "A-∞"])]
    if t.empty:
        return Gate("G2", False, {"error": "A 창 없음"})
    d = {"n": int(len(t)), "shallower_rate": float(t["mdd_shallower"].mean()), "worst_mdd": float(t["strat_mdd"].min()),
         "per_bundle": {b: float(_bundle(t, b)["mdd_shallower"].mean()) for b in ("A-10", "A-∞")},
         "mdd_median": float(t["strat_mdd"].median()), "tqqq_mdd_median": float(t["tqqq_mdd"].median())}
    return Gate("G2", d["shallower_rate"] >= 0.70 and d["worst_mdd"] >= -0.80, d)


def gate_g3(tbl: pd.DataFrame) -> Gate:
    t = _bundle(tbl, "S-5")
    if t.empty:
        return Gate("G3", False, {"error": "S-5 창 없음"})
    rows = [{"start": str(r.start.date()), "end": str(r.end.date()), "strat_ret": float(r.strat_ret),
             "floor": float(0.7 * r.tqqq_ret), "tqqq_ret": float(r.tqqq_ret), "ok": bool(r.strat_ret >= 0.7 * r.tqqq_ret)}
            for r in t.itertuples(index=False)]
    return Gate("G3", all(r["ok"] for r in rows), {"windows": rows, "n_fail": sum(not r["ok"] for r in rows)})


def gate_g4(tbl: pd.DataFrame, b_start_from: pd.Timestamp | None = None) -> Gate:
    t = _bundle(tbl, "B-10")
    if b_start_from is not None:
        t = t[t["start"] >= pd.Timestamp(b_start_from)]
    if t.empty:
        return Gate("G4", False, {"error": "B-10 창 없음"})
    dot = t[t["has_2000_03"]]
    d = {"n": int(len(t)), "beat_tqqq_rate": float(t["beats_tqqq"].mean()), "beat_qqq_rate": float(t["beats_qqq"].mean()),
         "worst_cagr": float(t["strat_cagr"].min()), "ret_median": float(t["strat_ret"].median()),
         "tqqq_ret_median": float(t["tqqq_ret"].median()), "qqq_ret_median": float(t["qqq_ret"].median()),
         "dotcom_n": int(len(dot)), "dotcom_strat_median": float(dot["strat_ret"].median()) if len(dot) else np.nan,
         "dotcom_qqq_median": float(dot["qqq_ret"].median()) if len(dot) else np.nan,
         "loss_windows": int((t["strat_ret"] < 0).sum()), "worst_mdd": float(t["strat_mdd"].min())}
    ok = (d["beat_tqqq_rate"] >= 0.70 and d["beat_qqq_rate"] >= 0.60 and d["worst_cagr"] >= 0.0
          and (len(dot) == 0 or d["dotcom_strat_median"] >= d["dotcom_qqq_median"]))
    return Gate("G4", bool(ok), d)


def g5_threshold(n_trials: int) -> float:
    return 95.0 + min(4.0, n_trials / 50.0)


def gate_g5(tbl: pd.DataFrame, placebo_medians: list[float] | None, n_trials: int) -> Gate:
    """전략의 A-10 ∪ A-∞ 창별 비율 중앙값이 플라세보(같은 통계) 분포에서 몇 백분위인가."""
    if not placebo_medians:
        return Gate("G5", False, {"error": "플라세보 없음"})
    t = tbl[tbl["bundle"].isin(["A-10", "A-∞"])]
    med = float(t["ratio"].median())
    arr = np.asarray(placebo_medians, dtype=float)
    pct = float(100.0 * (arr < med).mean())
    thr = g5_threshold(n_trials)
    d = {"strat_median": med, "percentile": pct, "threshold": thr, "n_placebo": int(len(arr)),
         "placebo_median": float(np.median(arr)), "placebo_max": float(arr.max()), "placebo_min": float(arr.min())}
    return Gate("G5", pct >= thr, d)


def gate_g6(neighbor_verdicts: list[dict] | None) -> Gate:
    """이웃 조합(핵심 손잡이 ±20%)의 verdict dict 목록 — 과반이 G1·G2 모두 통과."""
    if not neighbor_verdicts:
        return Gate("G6", False, {"error": "이웃 없음"})
    ok = [bool(v["G1"]["passed"] and v["G2"]["passed"]) for v in neighbor_verdicts]
    return Gate("G6", sum(ok) > len(ok) / 2, {"n": len(ok), "n_pass": sum(ok)})


def gate_verdicts(tbl: pd.DataFrame, n_trials: int = 0, placebo_medians: list[float] | None = None,
                  neighbor_verdicts: list[dict] | None = None, b_start_from: pd.Timestamp | None = None,
                  stop_at_first_fail: bool = False) -> dict:
    """G1~G6 순서대로. 반환 {"G1": {...}, ..., "first_fail": "G3"|None, "passed_through": 2}.

    ``stop_at_first_fail=False`` 면 모든 관문의 숫자를 다 계산해 둔다(기록용) — 판정은 첫 탈락 관문으로.
    """
    gates = [gate_g1(tbl), gate_g2(tbl), gate_g3(tbl), gate_g4(tbl, b_start_from),
             gate_g5(tbl, placebo_medians, n_trials), gate_g6(neighbor_verdicts)]
    out: dict = {}
    first_fail, n_pass = None, 0
    for g in gates:
        out[g.name] = {"passed": bool(g.passed), **g.detail}
        if first_fail is None:
            if g.passed:
                n_pass += 1
            else:
                first_fail = g.name
                if stop_at_first_fail:
                    break
    out["first_fail"] = first_fail
    out["passed_through"] = n_pass
    return out


def sealed_verdicts(tbl_sealed: pd.DataFrame) -> dict:
    """G7 — 봉인 해제 판 표로 G1~G3 만 다시. 호출 자체가 '열람 1회' 이므로 호출 쪽이 장부에 적는다."""
    gates = [gate_g1(tbl_sealed), gate_g2(tbl_sealed), gate_g3(tbl_sealed)]
    out = {g.name: {"passed": bool(g.passed), **g.detail} for g in gates}
    out["passed"] = all(g.passed for g in gates)
    return out


# ----------------------------------------------------------------------------- 플라세보

def _runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """같은 값이 이어지는 구간 → (값 배열, 길이 배열)."""
    if len(values) == 0:
        return np.array([]), np.array([], dtype=int)
    change = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [len(values)]]))
    return values[starts], lengths


def placebo_target(target: pd.Series, seed: int, mode: str = "runs") -> pd.Series:
    """노출 통계를 보존한 무작위 목표 비중.

    - ``runs``: 같은 값 구간의 **길이를 값별로 섞어** 재배열 — 전환 횟수·시장 체류 비율·구간 길이 분포 보존, 시점만 무작위.
      (0/1 스위치, 몇 단계 비중 전략용. 연속 비중이면 구간이 전부 길이 1 이라 거의 그대로다 → ``shift`` 를 쓴다)
    - ``shift``: 무작위 순환 이동 — 시계열 모양 전부 보존, 정렬만 무작위.
    """
    rng = np.random.default_rng(seed)
    if isinstance(target, pd.DataFrame):            # 행 상태(자산 조합) 를 코드로 바꿔 런 셔플 → 자산별 체류·전환 수 보존
        codes, uniq = pd.factorize(pd.Series([tuple(r) for r in target.round(6).to_numpy()], index=target.index))
        shuffled = placebo_target(pd.Series(codes.astype(float), index=target.index), seed, mode).to_numpy().astype(int)
        return pd.DataFrame([uniq[c] for c in shuffled], index=target.index, columns=target.columns)
    v = target.to_numpy(dtype=float)
    if mode == "shift":
        k = int(rng.integers(1, len(v)))
        return pd.Series(np.roll(v, k), index=target.index, name=f"placebo{seed}")
    if mode != "runs":
        raise ValueError(mode)
    vals, lens = _runs(v)
    new_lens = lens.copy()
    for val in np.unique(vals):
        idx = np.flatnonzero(vals == val)
        new_lens[idx] = rng.permutation(lens[idx])
    out = np.repeat(vals, new_lens)
    return pd.Series(out, index=target.index, name=f"placebo{seed}")


def placebo_medians(lev_close: pd.Series, target: pd.Series, baseline: pd.DataFrame, n_seeds: int = 30,
                    seed0: int = 20261110, mode: str = "runs", edition: str = "dev",
                    costs: CostParams = GATE_COSTS, cash_rate: pd.Series | float | None = None,
                    progress: Callable[[int, float], None] | None = None, prices: pd.DataFrame | None = None,
                    dividend_yields: dict[str, float] | None = None) -> list[float]:
    """seed 마다 플라세보 목표 비중으로 A-10 ∪ A-∞ 창을 돌려 창별 비율의 중앙값 목록."""
    meds = []
    for i in range(n_seeds):
        pt = placebo_target(target, seed0 + i, mode)
        tbl = window_table(lev_close, pt, baseline, edition, ("A-10", "A-∞"), costs, cash_rate, prices, dividend_yields=dividend_yields)
        meds.append(float(tbl["ratio"].median()))
        if progress:
            progress(i, meds[-1])
    return meds


# ----------------------------------------------------------------------------- 보고

def bundle_summary(tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BUNDLES:
        t = _bundle(tbl, b)
        if t.empty:
            continue
        rows.append({"bundle": b, "n": len(t), "strat_ret_median": t["strat_ret"].median(), "strat_ret_min": t["strat_ret"].min(),
                     "tqqq_ret_median": t["tqqq_ret"].median(), "ratio_median": t["ratio"].median(), "ratio_min": t["ratio"].min(),
                     "win_rate": (t["ratio"] > 1).mean(), "beat_qqq_rate": t["beats_qqq"].mean(),
                     "strat_mdd_median": t["strat_mdd"].median(), "strat_mdd_worst": t["strat_mdd"].min(),
                     "tqqq_mdd_median": t["tqqq_mdd"].median(), "mdd_shallower_rate": t["mdd_shallower"].mean(),
                     "strat_cagr_median": t["strat_cagr"].median(), "strat_cagr_min": t["strat_cagr"].min(),
                     "trades_median": t["trades"].median(), "time_in_market_median": t["time_in_market"].median()})
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return "∞" if not np.isfinite(x) else f"{x:+.0%}"


def verdict_markdown(name: str, tbl: pd.DataFrame, verdict: dict, edition: str = "dev", note: str = "") -> str:
    lines = [f"### {name} — {EDITIONS[edition]}", ""]
    if note:
        lines += [note, ""]
    ff = verdict.get("first_fail")
    lines.append(f"**판정: {'G1~G6 통과' if ff is None else f'{ff} 에서 탈락'} (통과 관문 {verdict.get('passed_through', 0)})**")
    lines.append("")
    g1, g2, g3, g4, g5, g6 = (verdict.get(k, {}) for k in ("G1", "G2", "G3", "G4", "G5", "G6"))
    if "A-10" in g1:
        for b in ("A-10", "A-∞"):
            x = g1[b]
            lines.append(f"- G1 {b}: 승률 {x['win_rate']:.0%} · 비율 중앙 {x['ratio_median']:.2f} · 최악 {x['ratio_min']:.2f} · 최고 {x['ratio_max']:.2f} → {'✅' if g1['passed'] else '❌'}")
    if "shallower_rate" in g2:
        lines.append(f"- G2: MDD 얕은 창 {g2['shallower_rate']:.0%} (A-10 {g2['per_bundle']['A-10']:.0%} · A-∞ {g2['per_bundle']['A-∞']:.0%}) · 최악 MDD {g2['worst_mdd']:.0%} · 중앙 {g2['mdd_median']:.0%} (TQQQ {g2['tqqq_mdd_median']:.0%}) → {'✅' if g2['passed'] else '❌'}")
    if "windows" in g3:
        parts = [f"{w['start'][:7]}~{w['end'][:7]} {_pct(w['strat_ret'])}(기준 {_pct(w['floor'])}){'' if w['ok'] else ' ✗'}" for w in g3["windows"]]
        lines.append(f"- G3 S-5: {' · '.join(parts)} → {'✅' if g3['passed'] else '❌'}")
    if "beat_tqqq_rate" in g4:
        lines.append(f"- G4 B-10({g4['n']}창): TQQQ 이김 {g4['beat_tqqq_rate']:.0%} · QQQ 이김 {g4['beat_qqq_rate']:.0%} · 최악 CAGR {g4['worst_cagr']:+.1%} · 손실 창 {g4['loss_windows']} · 세후 중앙 {_pct(g4['ret_median'])}(TQQQ {_pct(g4['tqqq_ret_median'])}·QQQ {_pct(g4['qqq_ret_median'])}) · 닷컴 {g4['dotcom_n']}창 중앙 {_pct(g4['dotcom_strat_median'])} vs QQQ {_pct(g4['dotcom_qqq_median'])} · 최악 MDD {g4['worst_mdd']:.0%} → {'✅' if g4['passed'] else '❌'}")
    if "percentile" in g5:
        lines.append(f"- G5 플라세보 {g5['n_placebo']} seed: 전략 중앙 비율 {g5['strat_median']:.2f} · 플라세보 중앙 {g5['placebo_median']:.2f}(최대 {g5['placebo_max']:.2f}) · 백분위 {g5['percentile']:.0f}% (문턱 {g5['threshold']:.0f}%) → {'✅' if g5['passed'] else '❌'}")
    elif g5:
        lines.append(f"- G5: {g5.get('error', '미계산')}")
    if "n" in g6:
        lines.append(f"- G6 이웃 {g6['n_pass']}/{g6['n']} 통과 → {'✅' if g6['passed'] else '❌'}")
    elif g6:
        lines.append(f"- G6: {g6.get('error', '미계산')}")
    lines.append("")
    s = bundle_summary(tbl)
    lines.append("| 묶음 | 창 | 전략 세후 중앙(최소) | TQQQ 세후 중앙 | 비율 중앙(최악) | 승률 | QQQ 이김 | 전략 MDD 중앙(최악) | TQQQ MDD 중앙 | MDD 얕음 | 매매 중앙 | 체류 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s.itertuples(index=False):
        lines.append(f"| {r.bundle} | {r.n} | {_pct(r.strat_ret_median)} ({_pct(r.strat_ret_min)}) | {_pct(r.tqqq_ret_median)} | "
                     f"{r.ratio_median:.2f} ({r.ratio_min:.2f}) | {r.win_rate:.0%} | {r.beat_qqq_rate:.0%} | {r.strat_mdd_median:.0%} ({r.strat_mdd_worst:.0%}) | "
                     f"{r.tqqq_mdd_median:.0%} | {r.mdd_shallower_rate:.0%} | {r.trades_median:.0f} | {r.time_in_market_median:.0%} |")
    lines.append("")
    return "\n".join(lines)


def window_markdown(tbl: pd.DataFrame, bundles: tuple[str, ...] = ("A-10", "A-∞", "S-5")) -> str:
    lines = ["| 묶음 | 시작 | 끝 | 전략 세후 | 전략 MDD | 매매 | 체류 | TQQQ 세후 | TQQQ MDD | 비율 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tbl[tbl["bundle"].isin(bundles)].itertuples(index=False):
        lines.append(f"| {r.bundle} | {r.start.date()} | {r.end.date()} | {_pct(r.strat_ret)} | {r.strat_mdd:.0%} | {r.trades} | {r.time_in_market:.0%} | "
                     f"{_pct(r.tqqq_ret)} | {r.tqqq_mdd:.0%} | {r.ratio:.2f} |")
    return "\n".join(lines)

# ----------------------------------------------------------------------------- 일간 승수 곡선 모드 (H20 장중 규칙 등)

def window_table_curve(mult: pd.Series, baseline: pd.DataFrame, edition: str = "dev", bundles: tuple[str, ...] = ("A-10", "A-∞", "S-5"),
                       tax_rate: float = 0.22, deduction_frac: float = 0.025) -> pd.DataFrame:
    """전략이 '일간 승수' 로만 주어질 때(장중 체결) 의 창 표. 세금은 연간 순이익 과세(이월 없음), 창마다 자본 1. 자료 없는 창은 건너뛴다."""
    from strategy.adapters.intraday_rules import after_tax_curve
    base = baseline[(baseline["edition"] == edition) & (baseline["bundle"].isin(bundles))]
    rows = []
    for b in base.itertuples(index=False):
        m = mult[(mult.index > b.start) & (mult.index <= b.end)]
        if len(m) < 50 or mult.index[0] > b.start + pd.Timedelta(days=7):
            continue
        eq, final = after_tax_curve(m, tax_rate, deduction_frac)
        dd = float((eq / eq.cummax() - 1.0).min()); years = (m.index[-1] - m.index[0]).days / 365.25
        r = {"bundle": b.bundle, "start": b.start, "end": b.end, "strat_ret": final - 1.0, "strat_mdd": dd,
             "strat_cagr": final ** (1.0 / years) - 1.0 if final > 0 else -1.0, "trades": int((m != 1.0).sum()), "time_in_market": float((m != 1.0).mean()),
             "tax": np.nan, "fees": np.nan, "tqqq_ret": b.tqqq_ret, "tqqq_mdd": b.tqqq_mdd, "tqqq_cagr": b.tqqq_cagr, "qqq_ret": b.qqq_ret, "qqq_mdd": b.qqq_mdd}
        r["ratio"] = final / (1.0 + b.tqqq_ret) if (1.0 + b.tqqq_ret) > 1e-9 else np.inf
        r["beats_tqqq"] = r["strat_ret"] > b.tqqq_ret; r["beats_qqq"] = r["strat_ret"] > b.qqq_ret
        r["mdd_shallower"] = dd > b.tqqq_mdd; r["has_2000_03"] = bool(b.start <= DOTCOM_TOP <= b.end)
        rows.append(r)
    return pd.DataFrame(rows)
