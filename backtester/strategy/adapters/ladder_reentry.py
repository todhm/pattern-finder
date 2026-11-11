"""H27 — 현금 상태의 바닥 사다리 재매수 (순수 pandas/numpy). 목표 비중 {0, 1/3, 2/3, 1} 또는 {0, 1/2, 1}."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LadderParams:
    rungs: tuple = (0.50, 0.65, 0.80)
    sizes: tuple = (1 / 3, 1 / 3, 1 / 3)
    n: int = 200
    dd: float = 0.18
    reenter_n: int = 20
    reenter_mode: str = "sma"     # sma(reenter_n 일선) | time(이탈 뒤 time_days 경과 & 종가 > time_sma 일선) — H37
    time_days: int = 60
    time_sma: int = 200
    exit_steps: tuple = ()        # H42: ((dd1, 남는 비중), (dd2, 0.0)) — 비어 있으면 dd 한 단계

    @property
    def name(self) -> str:
        r = "/".join(f"{x:.0%}" for x in self.rungs); s = "/".join(f"{x:.2f}" for x in self.sizes)
        tail = f" time{self.time_days}/sma{self.time_sma}" if self.reenter_mode == "time" else ""
        if self.exit_steps:
            tail += " exit" + "/".join(f"{d:.0%}→{f:g}" for d, f in self.exit_steps)
        return f"ladder {r} w{s}{tail}"


def ladder_switch(qqq: pd.Series, tqqq: pd.Series, p: LadderParams, gate: pd.Series | None = None, gate_size: float = 0.0,
                  monthly_gate: pd.Series | None = None, monthly_reentry: bool = True, monthly_exit: bool = True) -> pd.Series:
    """``gate`` 가 True 인 날은 사다리 매수를 막거나(gate_size 0) 비중을 gate_size 배로 줄인다(H29). 칸은 그래도 소비된다.

    ``monthly_gate`` 가 True 인 날(H32: 긴 약세장) 은 스위치의 이탈(과 ``monthly_reentry`` 면 재진입) 판정을 그 달 마지막 거래일에만 한다.
    """
    c = qqq.dropna(); idx = c.index; t = tqqq.reindex(idx).ffill()
    g = gate.reindex(idx).fillna(False).to_numpy(dtype=bool) if gate is not None else np.zeros(len(idx), dtype=bool)
    mg = monthly_gate.reindex(idx).fillna(False).to_numpy(dtype=bool) if monthly_gate is not None else np.zeros(len(idx), dtype=bool)
    last = np.zeros(len(idx), dtype=bool); last[pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).last().to_numpy()] = True
    sma = c.rolling(p.n).mean(); ddq = c / c.rolling(252, min_periods=20).max() - 1.0
    steps = tuple(p.exit_steps) if p.exit_steps else ((p.dd, 0.0),)
    below = (c < sma).to_numpy(); ddq_v = ddq.to_numpy()
    off = below & (ddq_v < -steps[-1][0])          # 완전 이탈 조건(마지막 단계)
    if p.reenter_mode == "time":
        above = (c > c.rolling(p.time_sma).mean()).to_numpy()
    else:
        on = (c > c.rolling(p.reenter_n).mean()).to_numpy()
    hi = t.rolling(252, min_periods=20).max(); ddt = (t / hi - 1.0).to_numpy(); hiv = hi.to_numpy(); tv = t.to_numpy()
    nan = sma.isna().to_numpy()
    state = np.ones(len(c)); cur = 1.0; rung_i = 0; days_out = 0; step_i = 0
    for i in range(len(c)):
        if nan[i]:
            state[i] = 0.0; continue
        if tv[i] >= hiv[i]:                          # 가격이 새 52주 고점을 찍으면 사다리 재장전(낙폭 에피소드당 한 번; 고점이 창 밖으로 빠져도 재장전 안 함)
            rung_i = 0
        can_exit = (not mg[i]) or (not monthly_exit) or last[i]
        can_enter = (not mg[i]) or (not monthly_reentry) or last[i]
        if p.reenter_mode == "time":
            on_i = days_out >= p.time_days and above[i]
        else:
            on_i = on[i]
        if cur >= 1.0:
            step_i = 0
            if can_exit:
                while step_i < len(steps) and below[i] and ddq_v[i] < -steps[step_i][0]:
                    cur = steps[step_i][1]; step_i += 1; days_out = 0
        else:
            days_out += 1
            if on_i and not off[i] and can_enter and not (below[i] and ddq_v[i] < -steps[0][0]):
                cur = 1.0; step_i = 0
            elif step_i < len(steps) and can_exit and below[i] and ddq_v[i] < -steps[step_i][0]:
                cur = min(cur, steps[step_i][1]); step_i += 1
            else:
                while rung_i < len(p.rungs) and ddt[i] <= -p.rungs[rung_i]:
                    cur = min(1.0, cur + p.sizes[rung_i] * (gate_size if g[i] else 1.0)); rung_i += 1
        state[i] = cur
    return pd.Series(state, index=idx, name=p.name)
