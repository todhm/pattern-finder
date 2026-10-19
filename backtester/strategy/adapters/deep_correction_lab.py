"""⑩ 급락 회피 전용 분석 — 회피 구간 일지, 시작 시점별(롤링) 세후 성과, 파라미터 격자.

규칙은 :func:`leveraged_etf_lab.deep_correction_filter`. 여기서는 "언제 시작했어도 장기보유보다 나았나" 를 묻는다:
모든 시작월에서 N년 보유한 결과를 장기보유와 짝지어 비율·승률·최악을 본다 (47 페이지의 단일 구간 비교를 분포로 확장).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams, LabResult, deep_correction_filter, simulate_exposure
from strategy.adapters.risk_metrics import compute_risk_metrics


@dataclass(frozen=True)
class DeepParams:
    n: int = 200
    dd: float = 0.15
    reenter_n: int = 20
    off_fraction: float = 0.0


# ── 확정 전략 "TQQQ 폭락 스위치 18" (2026-10-19) ───────────────────────────────
# 10월 검증 138개 변형(변동성 타기팅·조기경보·재진입 게이트·동승자 혼합·지수 로테이션) 중
# 세후 2010~ 보유의 90% 를 지킨 유일한 계열. 근거: docs/strategy_notes/study_cards/README.md (F2·F5·F7).
CRASH_SWITCH_NAME = "TQQQ 폭락 스위치 18"
CRASH_SWITCH_PARAMS = DeepParams(n=200, dd=0.18, reenter_n=20, off_fraction=0.0)
CRASH_SWITCH_RULE = ("평소 TQQQ 100%. QQQ 종가가 200일선 아래이면서 52주 고점 대비 −18% 아래로 마감 → 다음 거래일 전량 매도(현금). "
                     "현금 상태에서 QQQ 종가가 20일선 위로 마감 → 다음 거래일 전량 재매수. 그 외엔 아무것도 하지 않는다.")


def signal(signal_close: pd.Series, p: DeepParams) -> pd.Series:
    return deep_correction_filter(signal_close, p.n, p.dd, p.reenter_n, p.off_fraction)


def run_pair(lev_close: pd.Series, sig: pd.Series, start: date | pd.Timestamp, end: date | pd.Timestamp | None,
             costs: CostParams, cash_rate=None) -> tuple[LabResult, LabResult]:
    """(장기보유, 급락 회피) 한 쌍을 같은 구간·비용으로."""
    lev = lev_close[lev_close.index >= pd.Timestamp(start)]
    if end is not None:
        lev = lev[lev.index <= pd.Timestamp(end)]
    tgt = sig.reindex(lev.index).ffill().fillna(0.0)
    hold = simulate_exposure(lev, pd.Series(1.0, index=lev.index), costs, cash_rate, "① 장기보유")
    strat = simulate_exposure(lev, tgt, costs, cash_rate, "⑩ 급락 회피")
    return hold, strat


def avoided_segments(res: LabResult, lev_close: pd.Series, signal_close: pd.Series) -> pd.DataFrame:
    """이탈 → 재진입 구간마다 그 사이 레버리지 ETF·신호 지수 수익률 (회피 효과)."""
    ex = (res.exposure > 0.5).astype(int)
    d = ex.diff().fillna(0)
    exits, entries = list(d[d == -1].index), list(d[d == 1].index)
    rows = []
    for x in exits:
        nxt = next((e for e in entries if e > x), res.exposure.index[-1])
        rows.append({"이탈": x.date(), "재진입": nxt.date(), "일수": (nxt - x).days,
                     "회피 구간 ETF": float(lev_close.loc[nxt] / lev_close.loc[x] - 1.0),
                     "회피 구간 지수": float(signal_close.loc[nxt] / signal_close.loc[x] - 1.0),
                     "진행 중": nxt == res.exposure.index[-1] and ex.iloc[-1] == 0})
    return pd.DataFrame(rows)


def rolling_study(lev_close: pd.Series, sig: pd.Series, years: int, costs: CostParams, cash_rate=None,
                  step_months: int = 3, first_start: date | None = None) -> pd.DataFrame:
    """시작월을 step_months 간격으로 옮기며 years 년 보유 — 시작월별 (장기보유 배수, 급락 회피 배수, 비율, MDD)."""
    idx = lev_close.index
    starts = pd.date_range(first_start or idx[0], idx[-1] - pd.DateOffset(years=years), freq=f"{step_months}MS")
    rows = []
    for s in starts:
        e = s + pd.DateOffset(years=years)
        if e > idx[-1]:
            break
        hold, strat = run_pair(lev_close, sig, s, e, costs, cash_rate)
        if len(hold.equity) < 50:
            continue
        hm, sm = hold.equity.iloc[-1] / hold.equity.iloc[0], strat.equity.iloc[-1] / strat.equity.iloc[0]
        rows.append({"start": s.date(), "end": hold.equity.index[-1].date(), "hold_mult": hm, "strat_mult": sm, "ratio": sm / hm,
                     "hold_cagr": hm ** (1 / years) - 1, "strat_cagr": sm ** (1 / years) - 1,
                     "hold_mdd": hold.metrics["mdd"], "strat_mdd": strat.metrics["mdd"], "trades": strat.trades})
    return pd.DataFrame(rows)


def rolling_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "windows": int(len(df)), "win_rate": float((df["ratio"] > 1).mean()),
        "ratio_median": float(df["ratio"].median()), "ratio_p10": float(df["ratio"].quantile(0.1)), "ratio_p90": float(df["ratio"].quantile(0.9)),
        "ratio_min": float(df["ratio"].min()), "ratio_max": float(df["ratio"].max()),
        "strat_cagr_median": float(df["strat_cagr"].median()), "hold_cagr_median": float(df["hold_cagr"].median()),
        "strat_cagr_min": float(df["strat_cagr"].min()), "hold_cagr_min": float(df["hold_cagr"].min()),
        "strat_mdd_median": float(df["strat_mdd"].median()), "hold_mdd_median": float(df["hold_mdd"].median()),
        "strat_mdd_worst": float(df["strat_mdd"].min()), "hold_mdd_worst": float(df["hold_mdd"].min()),
        "strat_loss_rate": float((df["strat_mult"] < 1).mean()), "hold_loss_rate": float((df["hold_mult"] < 1).mean()),
    }


def param_grid(lev_close: pd.Series, signal_close: pd.Series, start: date, costs: CostParams, cash_rate=None,
               dds=(0.10, 0.12, 0.15, 0.18, 0.20, 0.25), reenters=(10, 20, 50), n: int = 200) -> pd.DataFrame:
    rows = []
    hold = None
    for dd in dds:
        for re in reenters:
            sig = deep_correction_filter(signal_close, n, dd, re)
            h, s = run_pair(lev_close, sig, start, None, costs, cash_rate)
            hold = hold or h
            rows.append({"dd": dd, "reenter": re, "mult": s.equity.iloc[-1] / s.equity.iloc[0], "cagr": s.metrics["cagr"],
                         "mdd": s.metrics["mdd"], "trades": s.trades, "vs_hold": (s.equity.iloc[-1] / s.equity.iloc[0]) / (hold.equity.iloc[-1] / hold.equity.iloc[0])})
    return pd.DataFrame(rows)


def current_state(signal_close: pd.Series, p: DeepParams) -> dict:
    c = signal_close.dropna()
    sma = c.rolling(p.n).mean()
    hi = c.rolling(252, min_periods=20).max()
    sig = signal(c, p)
    return {"as_of": c.index[-1].date(), "close": float(c.iloc[-1]), "sma": float(sma.iloc[-1]), "dist_sma": float(c.iloc[-1] / sma.iloc[-1] - 1),
            "drawdown": float(c.iloc[-1] / hi.iloc[-1] - 1), "exit_level": float(hi.iloc[-1] * (1 - p.dd)),
            "reenter_sma": float(c.rolling(p.reenter_n).mean().iloc[-1]), "holding": bool(sig.iloc[-1] > 0.5)}


# ----------------------------------------------------------------------------- 오늘의 판단

def decision(signal_close: pd.Series, p: DeepParams, hypothetical_close: float | None = None) -> dict:
    """오늘(마지막 봉 또는 가정 종가) 기준 **보유 / 매도 / 현금 유지 / 재진입** 판단과 그 근거·트리거 레벨.

    상태는 전날까지의 신호로 정하고, 오늘 종가(실제 또는 가정)에 규칙을 한 번 더 적용해 행동을 낸다.
    - 보유 중: 종가 < SMA(n) **이고** 종가 < 52주 고점×(1−dd) 면 **매도**, 아니면 **보유**.
    - 현금 중: 종가 > SMA(reenter) 이고 이탈 조건이 풀려 있으면 **재진입 매수**, 아니면 **현금 유지**
      (필터와 동일하게 이탈 조건이 우선 — 아직 200일선 아래·낙폭 안이면 20일선을 넘어도 안 산다).
    """
    c = signal_close.dropna().astype(float)
    if hypothetical_close is not None:
        nxt = c.index[-1] + pd.offsets.BDay(1)
        c = pd.concat([c, pd.Series([float(hypothetical_close)], index=[nxt])])
    sig = signal(c, p)
    prev_state = bool(sig.iloc[-2] > 0.5) if len(sig) > 1 else True
    sma_n = float(c.rolling(p.n).mean().iloc[-1])
    sma_re = float(c.rolling(p.reenter_n).mean().iloc[-1])
    hi = float(c.rolling(252, min_periods=20).max().iloc[-1])
    close = float(c.iloc[-1])
    exit_level = hi * (1 - p.dd)
    below_sma, deep = close < sma_n, close < exit_level
    above_re = close > sma_re
    after = bool(sig.iloc[-1] > 0.5)
    action = {(True, True): "보유", (True, False): "매도", (False, True): "재진입 매수", (False, False): "현금 유지"}[(prev_state, after)]
    changes = sig[sig.diff().fillna(0) != 0]
    last_change = changes.index[-1].date() if len(changes) else None
    state_days = int((c.index[-1] - changes.index[-1]).days) if len(changes) else None
    return {
        "as_of": c.index[-1].date(), "hypothetical": hypothetical_close is not None, "close": close, "prev_holding": prev_state,
        "action": action, "holding_after": after,
        "sma_n": sma_n, "sma_reenter": sma_re, "hi252": hi, "exit_level": exit_level,
        "dist_sma": close / sma_n - 1, "drawdown": close / hi - 1, "dist_exit": close / exit_level - 1, "dist_reenter": close / sma_re - 1,
        "cond_below_sma": below_sma, "cond_deep_dd": deep, "cond_above_reenter": above_re,
        "sell_trigger": min(sma_n, exit_level),      # 보유 중이면 이 가격 아래 종가에서 매도
        "buy_trigger": max(sma_re, min(sma_n, exit_level)),   # 현금 중이면 이 가격 위 종가에서 매수 (이탈 조건 해제 포함)
        "last_change": last_change, "state_days": state_days,
    }


# ----------------------------------------------------------------------------- 월말 SMA 규칙 (고변동 자산용 — SOXL)

@dataclass(frozen=True)
class MonthlySmaParams:
    """각 달 마지막 거래일 종가가 N일 SMA 위면 다음 달 보유, 아래면 ``off_fraction`` 만 남긴다.

    SOXL 탐색(soxl_search.py, 285개 규칙)에서 세후로 장기보유를 넘긴 계열: SOXX 100~125일 SMA 월말 판정 + 이탈 시 1/3 잔여.
    변동성이 큰 자산은 매일 판정하면 휩소가 수익을 다 먹는다 — 월 1회가 낫다.
    """

    n: int = 100
    off_fraction: float = 0.33


def signal_monthly_sma(signal_close: pd.Series, p: MonthlySmaParams) -> pd.Series:
    from strategy.adapters.leveraged_etf_lab import month_end_only, sma_filter
    s = month_end_only(sma_filter(signal_close.dropna(), p.n))
    return s.clip(lower=p.off_fraction) if p.off_fraction > 0 else s


def decision_monthly(signal_close: pd.Series, p: MonthlySmaParams, hypothetical_close: float | None = None) -> dict:
    """월말 규칙의 오늘 판단 — 상태는 지난 월말에 정해졌고, 다음 판정은 이달 마지막 거래일. '오늘이 월말이라면' 결과도 같이 낸다."""
    c = signal_close.dropna().astype(float)
    if hypothetical_close is not None:
        c = pd.concat([c, pd.Series([float(hypothetical_close)], index=[c.index[-1] + pd.offsets.BDay(1)])])
    sma = c.rolling(p.n).mean()
    s = signal_monthly_sma(c, p)
    held = bool(s.iloc[-1] > 0.5)
    close, sma_v = float(c.iloc[-1]), float(sma.iloc[-1])
    today = c.index[-1]
    month_end = (today + pd.offsets.BMonthEnd(0))
    if_month_end_now = close > sma_v
    last_decision = (today - pd.offsets.BMonthEnd(1)) if today != month_end else today
    changes = s[s.diff().fillna(0) != 0]
    return {
        "as_of": today.date(), "hypothetical": hypothetical_close is not None, "close": close, "sma": sma_v, "dist_sma": close / sma_v - 1,
        "holding": held, "exposure_now": float(s.iloc[-1]), "last_decision": last_decision.date(), "next_decision": month_end.date(),
        "days_to_decision": int((month_end - today).days), "if_month_end_now": "보유" if if_month_end_now else ("잔여 " + f"{p.off_fraction:.0%}" if p.off_fraction > 0 else "현금"),
        "action": "보유" if held else (f"잔여 {p.off_fraction:.0%} 유지" if p.off_fraction > 0 else "현금 유지"),
        "would_change": if_month_end_now != held,
        "last_change": changes.index[-1].date() if len(changes) else None,
    }


def param_grid_monthly(lev_close: pd.Series, signal_close: pd.Series, start: date, costs: CostParams, cash_rate=None,
                       ns=(75, 100, 125, 150, 200), offs=(0.0, 0.33)) -> pd.DataFrame:
    rows = []; hold = None
    for n in ns:
        for off in offs:
            s = signal_monthly_sma(signal_close, MonthlySmaParams(n, off))
            h, r = run_pair(lev_close, s, start, None, costs, cash_rate)
            hold = hold or h
            rows.append({"n": n, "off": off, "mult": r.equity.iloc[-1] / r.equity.iloc[0], "cagr": r.metrics["cagr"], "mdd": r.metrics["mdd"],
                         "trades": r.trades, "vs_hold": (r.equity.iloc[-1] / r.equity.iloc[0]) / (hold.equity.iloc[-1] / hold.equity.iloc[0])})
    return pd.DataFrame(rows)
