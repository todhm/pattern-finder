"""성장주 보유 전략 — "리스트를 들고, 싸진 것에 더 싣는다" (순수 numpy/pandas, 데이터 포트 없음).

설계서: docs/strategy_notes/성장주_보유_전략_설계_2026-10.md
  [A] 리스트  미너비니 성장 리스트(weekly_lists, 일별 lmask). 편입: 리스트에 entry_weeks 주 연속 / 편출: 빠진 지 exit_weeks 주.
  [B] 비중    동일비중(1) + "싸진 것"(cheap) × tilt. 상한 max_names 는 rank 로 자른다. 종목당 name_cap.
  [C] 리밸런스 월/분기 첫 거래일 시가 체결(판정은 전날 종가). band 안이면 그 종목은 안 건드린다. 현금 0.
  [D] 폭락 스위치 off/full/half — QQQ 종가 < 200일선 & 52주 고점 −18% → 현금(또는 절반), 20일선 회복 → 복귀 (leveraged_etf_lab.deep_correction_filter).
  비용 편도 fee_side, 세금은 growth_dip_scalp._tax_equity(연 22%, 공제 0.025, 이월 없음).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from strategy.adapters.growth_dip_scalp import Arrays, SimResult, _tax_equity, summarize


@dataclass(frozen=True)
class HoldRules:
    max_names: int = 40                 # 0 = 전부
    rank: str = "per_discount"          # per_discount(지금 PER/원래 PER 낮은 순) | drawdown(52주 고점 대비 낙폭 큰 순) | eps_yoy(큰 순) | random(플라세보)
                                        # | quality(재무제표 체크리스트 점수, CLI quality_features) | fcf | roe | revgrowth | combo:q_score+mom6 …
                                        # | momentum(12-1개월 수익 큰 순, aux["mom12"]) | mom6(6개월) | per_low(TTM PER 절대값 낮은 순) | near_high(52주 고점에 가까운 순)
                                        # | surprise(최근 분기 EPS 서프라이즈 큰 순, aux["surp"]) | ereact(최근 발표 2일 반응 큰 순, aux["ereact"])
                                        # | news_sent(30일 관련도 가중 감성, aux["nsent30"]) | news_vol(30일 기사 수 ÷ 180일 평균, aux["nvolrel"])
                                        # | combo:a+b+c — 열(mom6|mom12|surp|ereact|nsent30|nvolrel|nbull30|lyoy|dd) 의 횡단면 백분위 평균, 높을수록 우선
    require: str | None = None          # "surp>0,nneg30==0" — aux 열 조건을 전부 만족하는 날만 리스트로 인정(편입 streak·편출에 그대로 반영)
    entry_require: str | None = None    # "dd>=0.1,per<=30" — **편입할 때만** 만족해야 하는 조건(보유 중엔 안 본다). 저평가 진입 변형(설계서 §11.5)
    rerank: bool = False                # True: 리밸런스마다 점수 상위 max_names 로 다시 고른다(보유 중이어도 밀리면 매도). False: 보유 중인 것 우선(빈 자리만 채움)
    tilt: float = 2.0                   # cheap 종목 가중 배수 (1 = 동일비중)
    cheap_def: str = "all"              # per | q1 | q2 | all — per: PER ≤ 원래×(1−x), q1: 종가 ≤ 250일 중앙×0.9, q2: 종가 ≤ 200일선×0.9, all: 셋 다
    cheap_x_pct: float = 30.0
    contam_max: float = 1.5             # PER 창 최대 ÷ 원래 ≥ 이면 cheap 제외
    name_cap: float = 0.10              # 0 = 없음
    rebalance: str = "M"                # M | Q
    band: float = 0.30                  # 목표 대비 상대 편차가 이 안이면 안 건드림. 0 = 매번 맞춤
    entry_weeks: int = 2
    exit_weeks: int = 8
    eps_exit: bool = True               # TTM EPS ≤ 0 → 다음 리밸런스에 매도
    tilt_sticky: bool = False           # True: 한 번 cheap 으로 산 종목은 보유 중 tilt 유지 (PER 회복해도)
    crash_switch: str = "off"           # off | full | half
    trend_filter: str | None = None     # 리스트 단계(CLI weekly_lists) — 캐시 키용
    fee_side: float = 0.0015
    seed: int = 0


def crash_exposure(qqq_close: pd.Series, days: pd.DatetimeIndex, mode: str) -> np.ndarray:
    """(T,) 노출 배수 1/0(full) 또는 1/0.5(half). 신호는 종가, 적용은 다음날(호출 쪽에서 하루 민다)."""
    if mode == "off":
        return np.ones(len(days))
    from strategy.adapters.leveraged_etf_lab import deep_correction_filter
    st = deep_correction_filter(qqq_close.astype(float), 200, 0.18, 20, 0.5 if mode == "half" else 0.0)
    return st.reindex(days).ffill().fillna(1.0).to_numpy(dtype=float)


def streaks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,N) bool → (연속 True 일수, 연속 False 일수) — 그날 포함."""
    T, N = mask.shape
    on = np.zeros((T, N), dtype=int); off = np.zeros((T, N), dtype=int)
    for i in range(T):
        if i == 0:
            on[i] = mask[i].astype(int); off[i] = (~mask[i]).astype(int)
        else:
            on[i] = np.where(mask[i], on[i - 1] + 1, 0); off[i] = np.where(mask[i], 0, off[i - 1] + 1)
    return on, off


def cheap_mask(arr: Arrays, rules: HoldRules) -> np.ndarray:
    """(T,N) '싸진 것' — 그날 종가 기준 (리밸런스는 전날 값을 쓴다)."""
    C = arr.C
    with np.errstate(invalid="ignore", divide="ignore"):
        per_ok = (arr.EPS is not None and arr.PREF is not None) and (C / arr.EPS <= arr.PREF * (1 - rules.cheap_x_pct / 100))
        q1 = (arr.PXMED is not None) and (C <= arr.PXMED * 0.9)
        q2 = (arr.S200 is not None) and (C <= arr.S200 * 0.9)
        clean = np.ones(C.shape, dtype=bool) if (arr.PWMAX is None or arr.PREF is None) else ~(arr.PWMAX / arr.PREF >= rules.contam_max)
    m = {"per": per_ok, "q1": q1, "q2": q2, "all": per_ok & q1 & q2}[rules.cheap_def]
    return np.asarray(m, dtype=bool) & clean & ~np.isnan(C)


_HIGH_IS_GOOD = {"momentum": "mom12", "mom6": "mom6", "surprise": "surp", "ereact": "ereact", "news_sent": "nsent30", "news_vol": "nvolrel",
                 "quality": "q_qual", "fcf": "q_fcfm", "roe": "q_roe", "revgrowth": "q_rev", "peg": "peg_inv",
                 "advanced": "a_score", "gscore": "g_score", "fscore": "f_score",
                 "qplus": "q_plus", "nbeat": "nbeat4", "surpavg": "surp_avg4"}   # 서프라이즈 보탠 점수(설계서 §15)   # peg: PER ÷ 매출 TTM 성장률(%) 낮은 순 (CLI aux)     # 재무제표 품질(설계서 §11): q_qual = 점수 + 백분위 보조


def xrank(a: np.ndarray) -> np.ndarray:
    """행(날짜)마다 횡단면 백분위 0~1 (NaN 은 NaN 그대로, 값 1개면 0.5)."""
    out = np.full(a.shape, np.nan)
    for i in range(a.shape[0]):
        row = a[i]; m = ~np.isnan(row)
        k = int(m.sum())
        if k == 1:
            out[i, m] = 0.5
        elif k > 1:
            out[i, m] = np.argsort(np.argsort(row[m], kind="stable"), kind="stable") / (k - 1)
    return out


def _column(name: str, arr: Arrays, hi252: np.ndarray | None, aux: dict[str, np.ndarray]) -> np.ndarray:
    """조건·콤보에 쓰는 열. 높을수록 좋은 방향으로 맞춘다(dd 는 낙폭이 클수록 큰 값)."""
    nan = np.full(arr.C.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        if name == "lyoy":
            return arr.LYOY if arr.LYOY is not None else nan
        if name == "dd":
            return (1 - arr.C / hi252) if hi252 is not None else nan
        if name == "per_disc":
            return -(arr.C / arr.EPS) / arr.PREF if (arr.EPS is not None and arr.PREF is not None) else nan
    return aux[name] if name in aux else nan



def beat_history(surp_by_report: np.ndarray, k: np.ndarray) -> dict[str, np.ndarray]:
    """발표별 EPS 서프라이즈(시간순, NaN 허용) → 날짜별 서프라이즈 이력 열 (설계서 §15.2).

    k = 그 날짜까지 나온(유효해진) 발표 수(searchsorted side=right). 발표 없으면 NaN.
    nbeat4: 최근 4번 발표 중 beat(> 0) 수 — 유효 서프라이즈가 3개 미만이면 NaN. surp_avg4: 최근 4번 평균. beat_streak: 연속 beat 수(miss·NaN 에서 0).
    """
    s = np.asarray(surp_by_report, dtype=float); n = len(s)
    nb4 = np.full(n, np.nan); avg4 = np.full(n, np.nan); streak = np.zeros(n)
    for q in range(n):
        w = s[max(0, q - 3): q + 1]; w = w[np.isfinite(w)]
        if len(w) >= 3:                                   # 최근 4발표 중 유효 서프라이즈 3개 이상이면 횟수를 센다
            nb4[q] = float((w > 0).sum())
        if len(w) >= 1:
            avg4[q] = float(w.mean())
        streak[q] = (streak[q - 1] if q > 0 else 0.0) + 1.0 if (np.isfinite(s[q]) and s[q] > 0) else 0.0
    k = np.asarray(k); idx = np.maximum(k - 1, 0); has = k >= 1
    return {"nbeat4": np.where(has, nb4[idx], np.nan), "surp_avg4": np.where(has, avg4[idx], np.nan),
            "beat_streak": np.where(has, streak[idx], np.nan)}

_COND = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*(-?[0-9.]+)\s*$")


def require_mask(expr: str, arr: Arrays, hi252: np.ndarray | None, aux: dict[str, np.ndarray] | None) -> np.ndarray:
    """"surp>0,nneg30==0" → (T,N) bool. 값이 NaN 이면 False."""
    aux = aux or {}
    m = np.ones(arr.C.shape, dtype=bool)
    ops = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}
    for part in expr.split(","):
        g = _COND.match(part)
        if not g:
            raise ValueError(f"require 조건 모름: {part!r}")
        col = _column(g.group(1), arr, hi252, aux)
        with np.errstate(invalid="ignore"):
            m &= ops[g.group(2)](col, float(g.group(3))) & ~np.isnan(col)
    return m


def rank_score(arr: Arrays, rules: HoldRules, hi252: np.ndarray | None, rng: np.random.Generator,
               aux: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """(T,N) 낮을수록 우선. aux 는 CLI 가 만든 (T,N) 보조 열(mom12·mom6·surp·ereact·nsent30 등)."""
    C = arr.C
    aux = aux or {}
    nan = np.full(C.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        if rules.rank in _HIGH_IS_GOOD:
            col = _HIGH_IS_GOOD[rules.rank]
            s = -aux[col] if col in aux else nan
        elif rules.rank.startswith("combo:"):
            cols = [xrank(_column(n, arr, hi252, aux)) for n in rules.rank[6:].split("+")]
            s = -np.nanmean(np.stack(cols), axis=0)
        elif rules.rank == "per_discount":
            s = (C / arr.EPS) / arr.PREF if (arr.EPS is not None and arr.PREF is not None) else nan
        elif rules.rank == "drawdown":
            s = C / hi252 - 1 if hi252 is not None else nan
        elif rules.rank == "near_high":
            s = 1 - C / hi252 if hi252 is not None else nan
        elif rules.rank == "eps_yoy":
            s = -arr.LYOY if arr.LYOY is not None else nan
        elif rules.rank == "per_low":
            s = np.where(arr.EPS > 0, C / arr.EPS, np.nan) if arr.EPS is not None else nan
        elif rules.rank == "random":
            s = rng.random(C.shape)
        else:
            raise ValueError(f"rank {rules.rank!r} 모름")
    return np.where(np.isnan(s), np.inf, s)       # 값 없으면 맨 뒤


def rebalance_days(days: pd.DatetimeIndex, mode: str) -> np.ndarray:
    key = days.year * 100 + (days.month if mode == "M" else (days.month - 1) // 3)
    first = np.r_[True, key[1:] != key[:-1]]
    return first


def run_hold(arr: Arrays, lmask: np.ndarray, rules: HoldRules, qqq_close: pd.Series | None = None,
             hi252: np.ndarray | None = None, record: bool = False, aux: dict[str, np.ndarray] | None = None) -> SimResult:
    """record=True 면 stats["보유 이력"] 에 리밸런스마다 (날짜, {티커: 목표 비중}) 과 stats["현금 비율"] 일별 시리즈를 넣는다(진단용)."""
    T, N = arr.C.shape
    O, C = arr.O, arr.C
    rng = np.random.default_rng(rules.seed)
    if rules.require:
        lmask = lmask & require_mask(rules.require, arr, hi252, aux)
    emask = require_mask(rules.entry_require, arr, hi252, aux) if rules.entry_require else None
    on, off = streaks(lmask)
    cheap = cheap_mask(arr, rules)
    score = rank_score(arr, rules, hi252, rng, aux)
    expo = crash_exposure(qqq_close, arr.days, rules.crash_switch) if qqq_close is not None else np.ones(T)
    rb = rebalance_days(arr.days, rules.rebalance)
    fee = rules.fee_side
    shares = np.zeros(N); cost = np.zeros(N); sticky = np.zeros(N, dtype=bool); ever = np.zeros(N, dtype=bool)
    cash = 1.0
    equity = np.empty(T); realized = np.zeros(T); trades: list[dict] = []
    prev_expo = 1.0
    hist: list = []; cash_frac = np.ones(T)
    for i in range(T):
        px_open = O[i]; px_close = C[i]
        ok_open = ~np.isnan(px_open)
        e = float(expo[i - 1]) if i > 0 else 1.0            # 전날 종가 신호 → 오늘 시가
        do_rb = rb[i] and i > 0
        expo_changed = e != prev_expo
        if (do_rb or expo_changed) and i > 0:
            j = i - 1                                        # 판정은 전날 종가
            held = shares > 0
            valid = ~np.isnan(C[j]) & ok_open
            entering = (on[j] >= max(1, rules.entry_weeks * 5)) & valid      # entry_weeks 0 이어도 그날 리스트에는 있어야 한다
            if emask is not None:
                entering &= emask[j]                                             # 진입 조건(저평가 등)은 새로 살 때만
            leaving = (off[j] >= rules.exit_weeks * 5)
            if rules.eps_exit and arr.EPS is not None:
                leaving |= held & (np.isnan(arr.EPS[j]) | (arr.EPS[j] <= 0))
            keep = held & ~leaving & valid
            cand = (keep | entering)
            if rules.max_names > 0:
                order = np.argsort(score[j], kind="stable") if rules.rerank else np.lexsort((score[j], ~keep))   # rerank: 점수만 / 아니면 보유 중인 것 먼저
                chosen = np.zeros(N, dtype=bool)
                for jj in order:
                    if not cand[jj]:
                        continue
                    chosen[jj] = True
                    if chosen.sum() >= rules.max_names:
                        break
            else:
                chosen = cand
            ch = cheap[j] & chosen
            if rules.tilt_sticky:
                sticky = (sticky & chosen) | ch
                ch = sticky & chosen
            w = np.where(chosen, np.where(ch, rules.tilt, 1.0), 0.0)
            if w.sum() > 0:
                w = w / w.sum()
                if rules.name_cap > 0:
                    for _ in range(5):
                        over = w > rules.name_cap
                        if not over.any():
                            break
                        excess = (w[over] - rules.name_cap).sum(); w[over] = rules.name_cap
                        under = chosen & ~over
                        if under.any():
                            w[under] += excess * w[under] / w[under].sum()
            w = w * e                                       # 폭락 스위치 노출
            if record:
                hist.append((str(arr.days[i].date()), {arr.tickers[jj]: float(w[jj]) for jj in np.flatnonzero(w > 0)}))
            eq_open = cash + float(np.nansum(np.where(ok_open, shares * px_open, shares * np.nan_to_num(C[j]))))
            cur_w = np.where(ok_open, shares * px_open, 0.0) / eq_open
            target_val = w * eq_open
            for jj in np.flatnonzero((np.abs(cur_w - w) > 1e-12) & ok_open):
                if w[jj] > 0 and cur_w[jj] > 0 and rules.band > 0 and abs(cur_w[jj] / w[jj] - 1) <= rules.band and not expo_changed:
                    continue                                 # 밴드 안 → 그대로
                diff_val = target_val[jj] - shares[jj] * px_open[jj]
                if abs(diff_val) < 1e-9 * eq_open:
                    continue
                if diff_val < 0:                             # 매도
                    sell_sh = min(shares[jj], -diff_val / px_open[jj])
                    proceeds = sell_sh * px_open[jj] * (1 - fee)
                    basis = cost[jj] * (sell_sh / shares[jj]) if shares[jj] > 0 else 0.0
                    realized[i] += proceeds - basis
                    cash += proceeds; cost[jj] -= basis; shares[jj] -= sell_sh
                    if shares[jj] <= 1e-12:
                        trades.append({"day": arr.days[i], "ticker": arr.tickers[jj], "ret": proceeds / basis - 1 if basis > 0 else 0.0,
                                       "why": "편출" if leaving[jj] else ("스위치" if expo_changed and e < prev_expo else ("교체" if (rules.max_names > 0 and not chosen[jj]) else "리밸런스")), "hold": 0, "exit": arr.days[i]})
                        shares[jj] = 0.0; cost[jj] = 0.0
                else:                                        # 매수
                    spend = min(diff_val, cash)
                    if spend <= 0:
                        continue
                    buy_sh = spend * (1 - fee) / px_open[jj]
                    shares[jj] += buy_sh; cost[jj] += spend; cash -= spend; ever[jj] = True
            prev_expo = e
        equity[i] = cash + float(np.nansum(np.where(np.isnan(px_close), 0.0, shares * px_close)))
        cash_frac[i] = cash / equity[i] if equity[i] > 0 else 1.0
    eq_s = pd.Series(equity, index=arr.days); pnl_s = pd.Series(realized, index=arr.days)
    st = summarize(eq_s, _tax_equity(eq_s, pnl_s), trades, arr.days)
    turn = sum(abs(t.get("ret", 0)) >= 0 for t in trades)
    st["매도 건수"] = int(turn)
    st["연 매도 건수"] = float(turn / max(1e-9, (arr.days[-1] - arr.days[0]).days / 365.25))
    st["세금 총액"] = float((eq_s.iloc[-1]) - _tax_equity(eq_s, pnl_s).iloc[-1])
    st["고유 종목 수"] = int(ever.sum())
    if record:
        st["보유 이력"] = hist
        st["현금 비율"] = {str(d.date()): float(v) for d, v in zip(arr.days, cash_frac)}
        st["세후 곡선"] = {str(d.date()): float(v) for d, v in _tax_equity(eq_s, pnl_s).items()}
    return SimResult(eq_s, trades, st)


def expand_grid(base: HoldRules, grid: dict[str, list]) -> list[HoldRules]:
    import itertools
    keys = list(grid)
    out = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        out.append(HoldRules(**{**asdict(base), **dict(zip(keys, vals))}))
    return out
