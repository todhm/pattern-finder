"""TQQQ 스위치·사다리·긴 약세장 패키지 (H43, R60 — 2026-11-11 G1~G7 합격) 의 운용용 판정.

규칙 수치는 바꾸지 않는다(프로토콜 §4). 부품 엔진은 그대로 쓰고(`ladder_reentry.ladder_switch`, `regime_slots.regime_labels`,
`whipsaw_switch.whipsaw_switch`), 여기서는 ① 날마다 어떤 부품이 왜 비중을 바꿨는지(사다리 칸·이탈 단계·레짐) 추적,
② 오늘의 행동과 다음 트리거 가격, ③ 비중 변경 이력 표를 만든다. 순수 pandas.

설명 문서 docs/strategy_notes/TQQQ_탐색_최고결과_쉬운설명_2026-11.md, 기술 원본 TQQQ_탐색_검증_2026-11.md H43.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.ladder_reentry import LadderParams, ladder_switch
from strategy.adapters.leveraged_etf_lab import CostParams, LabResult, simulate_exposure
from strategy.adapters.regime_slots import RegimeParams, regime_labels
from strategy.adapters.whipsaw_switch import WhipsawParams, whipsaw_switch

PACKAGE_NAME = "TQQQ 스위치·사다리·긴 약세장 패키지 (H43)"
PACKAGE_RULE = ("평소 TQQQ 100%. QQQ 종가가 200일선 아래이면서 52주 고점 대비 −18% → 다음 거래일 절반 매도, −24% → 전량 매도. "
                "현금·부분 상태에서 QQQ 종가 > 20일선 → 전량 재매수(단 200일선 아래 & −18% 면 보류). "
                "현금·부분 상태에서 TQQQ 가 자기 52주 고점 대비 −50/−65/−80% 에 닿을 때마다 ⅓ 씩 재매수(낙폭 에피소드당 칸마다 한 번, 새 고점에서 재장전). "
                "긴 약세장(QQQ 200일선 아래 약세 상태 & 고점 −35% 이상 또는 체류 120거래일 이상)이면 그 달 마지막 거래일에만 스위치 18(−18% 전량 매도 / 20일선 전량 매수) 판정, 사다리 없음.")
PACKAGE_COSTS = CostParams(buy_fee=0.001 + 0.0005, sell_fee=0.001 + 0.0000229 + 0.0005, tax_rate=0.22,
                           tax_deduction=2_500_000.0, initial_capital=100_000_000.0, rebalance_band=0.05, exec_lag=1)


@dataclass(frozen=True)
class PackageParams:
    n: int = 200
    dd1: float = 0.18          # 절반 매도
    dd2: float = 0.24          # 전량 매도
    reenter_n: int = 20
    rungs: tuple = (0.50, 0.65, 0.80)
    sizes: tuple = (1 / 3, 1 / 3, 1 / 3)
    c_depth: float = 0.35      # 긴 약세장 낙폭
    long_days: int = 120       # 긴 약세장 체류

    @property
    def ladder(self) -> LadderParams:
        return LadderParams(self.rungs, self.sizes, n=self.n, reenter_n=self.reenter_n, exit_steps=((self.dd1, 0.5), (self.dd2, 0.0)))

    @property
    def regime(self) -> RegimeParams:
        return RegimeParams(n=self.n, long_days=self.long_days, c_depth=self.c_depth)

    @property
    def monthly(self) -> WhipsawParams:
        return WhipsawParams(0.0, True, False, n=self.n, dd=self.dd1, reenter_n=self.reenter_n)


PACKAGE_PARAMS = PackageParams()


# ----------------------------------------------------------------------------- 사다리 부품 추적 (ladder_switch 와 같은 상태 기계 + 이벤트)

def ladder_trace(qqq: pd.Series, tqqq: pd.Series, p: PackageParams) -> pd.DataFrame:
    """`ladder_switch(qqq, tqqq, p.ladder)` 와 똑같은 상태(state)에 칸 소진 수(rung_i)·이탈 단계(step_i)·그날 이벤트를 붙인다.
    테스트가 state 열이 ladder_switch 와 일치함을 고정한다."""
    lp = p.ladder
    c = qqq.dropna(); idx = c.index; t = tqqq.reindex(idx).ffill()
    sma = c.rolling(lp.n).mean(); ddq = (c / c.rolling(252, min_periods=20).max() - 1.0).to_numpy()
    steps = tuple(lp.exit_steps)
    below = (c < sma).to_numpy(); off = below & (ddq < -steps[-1][0])
    on = (c > c.rolling(lp.reenter_n).mean()).to_numpy()
    hi = t.rolling(252, min_periods=20).max(); ddt = (t / hi - 1.0).to_numpy(); hiv = hi.to_numpy(); tv = t.to_numpy()
    nan = sma.isna().to_numpy()
    n = len(c); state = np.ones(n); rung = np.zeros(n, dtype=int); step = np.zeros(n, dtype=int); ev = np.array([""] * n, dtype=object)
    cur = 1.0; rung_i = 0; step_i = 0
    for i in range(n):
        if nan[i]:
            state[i] = 0.0; continue
        if tv[i] >= hiv[i]:
            rung_i = 0
        e = []
        if cur >= 1.0:
            step_i = 0
            while step_i < len(steps) and below[i] and ddq[i] < -steps[step_i][0]:
                cur = steps[step_i][1]; step_i += 1; e.append(f"exit{step_i}")
        else:
            if on[i] and not off[i] and not (below[i] and ddq[i] < -steps[0][0]):
                cur = 1.0; step_i = 0; e.append("reenter")
            elif step_i < len(steps) and below[i] and ddq[i] < -steps[step_i][0]:
                cur = min(cur, steps[step_i][1]); step_i += 1; e.append(f"exit{step_i}")
            else:
                while rung_i < len(lp.rungs) and ddt[i] <= -lp.rungs[rung_i]:
                    cur = min(1.0, cur + lp.sizes[rung_i]); rung_i += 1; e.append(f"ladder{rung_i}")
        state[i] = cur; rung[i] = rung_i; step[i] = step_i; ev[i] = "+".join(e)
    return pd.DataFrame({"state": state, "rung_i": rung, "step_i": step, "event": ev}, index=idx)


# ----------------------------------------------------------------------------- 패키지 프레임

EVENT_KO = {"exit1": "−18% 절반 매도", "exit2": "−24% 전량 매도", "reenter": "20일선 재진입 전량 매수",
            "ladder1": "사다리 ① TQQQ −50% ⅓ 매수", "ladder2": "사다리 ② TQQQ −65% ⅓ 매수", "ladder3": "사다리 ③ TQQQ −80% ⅓ 매수"}


def package_frame(qqq: pd.Series, tqqq: pd.Series, p: PackageParams = PACKAGE_PARAMS) -> pd.DataFrame:
    """날마다: 가격·선·낙폭·레짐·부품 비중·최종 목표 비중(target)·이유(reason)."""
    c = qqq.dropna(); idx = c.index; t = tqqq.reindex(idx).ffill()
    tr = ladder_trace(c, t, p)
    lab = regime_labels(c, p.regime).reindex(idx).ffill().fillna("A")
    msw = whipsaw_switch(c, p.monthly).reindex(idx).ffill().fillna(1.0)
    is_c = (lab == "C").to_numpy()
    target = np.where(is_c, msw.to_numpy(), tr["state"].to_numpy())
    f = pd.DataFrame({"QQQ": c, "TQQQ": t, "sma200": c.rolling(p.n).mean(), "sma20": c.rolling(p.reenter_n).mean(),
                      "hi252": c.rolling(252, min_periods=20).max(), "tqqq_hi": t.rolling(252, min_periods=20).max(),
                      "regime": lab, "ladder_w": tr["state"], "monthly_w": msw, "target": target,
                      "rung_i": tr["rung_i"], "step_i": tr["step_i"], "ladder_event": tr["event"]}, index=idx)
    f["qqq_dd"] = f["QQQ"] / f["hi252"] - 1.0; f["tqqq_dd"] = f["TQQQ"] / f["tqqq_hi"] - 1.0
    prev_t = f["target"].shift(1); prev_c = pd.Series(is_c, index=idx).shift(1).fillna(False).astype(bool)
    reasons = []
    for i, d in enumerate(idx):
        chg = i > 0 and abs(f["target"].iat[i] - prev_t.iat[i]) > 1e-9
        if not chg:
            reasons.append(""); continue
        if is_c[i] and not prev_c.iat[i]:
            reasons.append("긴 약세장 진입 → 월말 스위치 상태로 맞춤")
        elif is_c[i]:
            reasons.append("긴 약세장 월말 판정: " + ("전량 매수" if f["target"].iat[i] > prev_t.iat[i] else "전량 매도"))
        elif prev_c.iat[i]:
            reasons.append("긴 약세장 해제 → 사다리 상태로 맞춤")
        else:
            reasons.append(" + ".join(EVENT_KO.get(e, e) for e in f["ladder_event"].iat[i].split("+") if e) or "비중 변경")
    f["reason"] = reasons
    return f


def change_table(f: pd.DataFrame) -> pd.DataFrame:
    """목표 비중이 바뀐 날 표(최신 위). 다음 변경까지 TQQQ 수익률 = 그 비중으로 지낸 구간의 TQQQ 움직임."""
    chg = f[f["target"].diff().fillna(0.0).abs() > 1e-9]
    rows = []
    dates = list(chg.index)
    for k, d in enumerate(dates):
        nxt = dates[k + 1] if k + 1 < len(dates) else f.index[-1]
        rows.append({"날짜": d.date(), "이전 비중": float(f["target"].shift(1).loc[d]), "이후 비중": float(f["target"].loc[d]), "이유": f["reason"].loc[d],
                     "레짐": f["regime"].loc[d], "QQQ 고점 대비": float(f["qqq_dd"].loc[d]), "QQQ vs 200일선": float(f["QQQ"].loc[d] / f["sma200"].loc[d] - 1.0),
                     "TQQQ 고점 대비": float(f["tqqq_dd"].loc[d]), "다음 변경까지 TQQQ": float(f["TQQQ"].loc[nxt] / f["TQQQ"].loc[d] - 1.0),
                     "다음 변경": nxt.date() if k + 1 < len(dates) else None})
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


# ----------------------------------------------------------------------------- 오늘의 판단

def _month_last_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    e = (d + pd.offsets.MonthEnd(0)).normalize()
    while e.weekday() >= 5:
        e -= pd.Timedelta(days=1)
    return e


def decision(qqq: pd.Series, tqqq: pd.Series, p: PackageParams = PACKAGE_PARAMS,
             hypothetical_close: float | None = None, hypothetical_tqqq: float | None = None) -> dict:
    """마지막 종가(또는 가정 종가)로 오늘의 행동. 가정 QQQ 만 주면 TQQQ 는 전날 종가 × (1 + 3 × QQQ 변화) 로 근사한다."""
    q = qqq.dropna().copy(); t = tqqq.reindex(q.index).ffill().copy()
    hyp = hypothetical_close is not None and hypothetical_close > 0
    if hyp:
        q.iloc[-1] = float(hypothetical_close)
        t.iloc[-1] = float(hypothetical_tqqq) if hypothetical_tqqq else float(t.iloc[-2] * (1.0 + 3.0 * (hypothetical_close / float(qqq.dropna().iloc[-2]) - 1.0)))
    f = package_frame(q, t, p); last = f.iloc[-1]; prev = f.iloc[-2]
    d = f.index[-1]; tgt = float(last["target"]); ptg = float(prev["target"]); is_c = last["regime"] == "C"
    month_end = _month_last_trading_day(d); is_month_end = d >= month_end or (d.month != (d + pd.tseries.offsets.BDay(1)).month)
    if abs(tgt - ptg) < 1e-9:
        if is_c:
            action = "긴 약세장 — " + ("보유 유지" if tgt > 0.5 else "현금 유지") + ("" if is_month_end else " (월말까지 판정 없음)")
        else:
            action = {1.0: "보유", 0.0: "현금 유지"}.get(round(tgt, 3), f"부분 보유 유지 ({tgt:.0%})")
    elif tgt > ptg:
        action = f"매수 — 비중 {ptg:.0%} → {tgt:.0%}"
    else:
        action = f"매도 — 비중 {ptg:.0%} → {tgt:.0%}"
    rung_i = int(last["rung_i"]); next_rung = p.rungs[rung_i] if rung_i < len(p.rungs) else None
    below = bool(last["QQQ"] < last["sma200"])
    # 다음 트리거
    trig = []
    if is_c:
        trig.append(f"긴 약세장 모드: 판정은 {month_end.date()}(이 달 마지막 거래일) 종가로만. 그날 QQQ 가 200일선 아래 & {last['hi252'] * (1 - p.dd1):.2f}(고점 −{p.dd1:.0%}) 아래면 전량 매도, "
                    f"20일선({last['sma20']:.2f}) 위면 전량 매수. 사다리 없음.")
    else:
        if tgt >= 1.0:
            trig.append(f"절반 매도: QQQ 종가가 200일선({last['sma200']:.2f}) 아래이면서 {last['hi252'] * (1 - p.dd1):.2f}(고점 −{p.dd1:.0%}, 현재가 대비 {last['hi252'] * (1 - p.dd1) / last['QQQ'] - 1:+.1%}) 아래로 마감")
            trig.append(f"전량 매도: 같은 조건에서 {last['hi252'] * (1 - p.dd2):.2f}(고점 −{p.dd2:.0%}, {last['hi252'] * (1 - p.dd2) / last['QQQ'] - 1:+.1%}) 아래로 마감")
        else:
            if int(last["step_i"]) == 1:
                trig.append(f"나머지 전량 매도: QQQ 종가가 200일선 아래이면서 {last['hi252'] * (1 - p.dd2):.2f}(고점 −{p.dd2:.0%}, {last['hi252'] * (1 - p.dd2) / last['QQQ'] - 1:+.1%}) 아래")
            trig.append(f"전량 재매수: QQQ 종가가 20일선({last['sma20']:.2f}, 현재가 대비 {last['sma20'] / last['QQQ'] - 1:+.1%}) 위로 마감 (단 200일선 아래 & −{p.dd1:.0%} 이면 보류)")
            if next_rung is not None:
                trig.append(f"사다리 {rung_i + 1}칸: TQQQ 가 {last['tqqq_hi'] * (1 - next_rung):.2f}(52주 고점 {last['tqqq_hi']:.2f} 대비 −{next_rung:.0%}, 현재가 대비 {last['tqqq_hi'] * (1 - next_rung) / last['TQQQ'] - 1:+.1%}) 이하로 마감하면 ⅓ 매수")
            else:
                trig.append("사다리: 세 칸 모두 소진(TQQQ 가 새 52주 고점을 찍어야 재장전)")
        trig.append(f"긴 약세장 모드 진입: QQQ 가 200일선 아래 약세 상태로 {last['hi252'] * (1 - p.c_depth):.2f}(고점 −{p.c_depth:.0%}) 아래이거나 아래 체류 {p.long_days}거래일 이상")
    return {"as_of": d.date(), "hypothetical": hyp, "close": float(last["QQQ"]), "tqqq": float(last["TQQQ"]), "action": action, "target": tgt, "prev_target": ptg,
            "reason": last["reason"], "regime": last["regime"], "is_month_end": bool(is_month_end), "month_end": month_end.date(),
            "sma200": float(last["sma200"]), "sma20": float(last["sma20"]), "hi252": float(last["hi252"]), "qqq_dd": float(last["qqq_dd"]),
            "tqqq_hi": float(last["tqqq_hi"]), "tqqq_dd": float(last["tqqq_dd"]), "rung_used": rung_i, "next_rung": next_rung, "step_i": int(last["step_i"]),
            "cond_below_sma": below, "cond_dd1": bool(last["qqq_dd"] <= -p.dd1), "cond_dd2": bool(last["qqq_dd"] <= -p.dd2),
            "cond_above_sma20": bool(last["QQQ"] > last["sma20"]), "cond_depth": bool(last["qqq_dd"] <= -p.c_depth),
            "half_level": float(last["hi252"] * (1 - p.dd1)), "full_level": float(last["hi252"] * (1 - p.dd2)), "depth_level": float(last["hi252"] * (1 - p.c_depth)),
            "ladder_levels": [float(last["tqqq_hi"] * (1 - r)) for r in p.rungs], "triggers": trig,
            "state_days": int((f["target"].iloc[::-1] == tgt).cummin().sum()), "last_change": f.index[f["target"].diff().fillna(0).abs() > 1e-9][-1].date() if (f["target"].diff().fillna(0).abs() > 1e-9).any() else None}


# ----------------------------------------------------------------------------- 성적

def run_pair(lev_close: pd.Series, target: pd.Series, start: date | pd.Timestamp, end: date | pd.Timestamp | None,
             costs: CostParams = PACKAGE_COSTS, cash_rate=None) -> tuple[LabResult, LabResult]:
    lev = lev_close[lev_close.index >= pd.Timestamp(start)]
    if end is not None:
        lev = lev[lev.index <= pd.Timestamp(end)]
    tgt = target.reindex(lev.index).ffill()
    if tgt.isna().iloc[0]:
        prev = target[target.index < lev.index[0]]
        tgt = tgt.fillna(float(prev.iloc[-1]) if len(prev) else 0.0)
    hold = simulate_exposure(lev, pd.Series(1.0, index=lev.index), costs, cash_rate, "TQQQ 보유")
    strat = simulate_exposure(lev, tgt, costs, cash_rate, PACKAGE_NAME)
    return hold, strat
