"""주도주 "끝까지 들고 가기" 칸(slot) 포트폴리오 (순수 numpy).

칸이 비면 그날 종가 기준 적격 종목 중 점수 상위를 다음 거래일 시가에 사고,
보유 종목은 추세가 꺾일 때(종가가 이탈선 아래 / 보유 중 고점 대비 trail / 폐지)만 다음 거래일 시가에 판다.
레버리지·공매도 없음. 새로 살 때는 남은 현금을 빈 칸 수로 나눈다(칸 사이 재조정 없음 = 이긴 종목이 커지게 둔다).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HoldRules:
    slots: int = 5
    trail: float | None = None      # 보유 중 종가 고점 대비 이만큼 빠지면 매도
    max_hold: int | None = None     # 거래일 기한(없으면 무기한)
    cost: float = 0.0025            # 한쪽 체결 비용


def run_leader_hold(c: np.ndarray, o: np.ndarray, alive: np.ndarray, elig: np.ndarray, score: np.ndarray, rules: HoldRules,
                    start: int, end: int, exit_line: np.ndarray | None = None, rng: np.random.Generator | None = None):
    """c/o: 폐지 뒤 마지막 종가로 채운 종가·시가 (T,N). elig: 그날 종가 기준 매수 적격. score: 클수록 먼저(rng 가 있으면 무작위).
    exit_line: 종가가 이 값 아래면 매도(예: 200일선). 반환: (자산 곡선 [start..end], 거래 목록)."""
    cash = 1.0; pos: dict[int, list] = {}; sells: list[int] = []; buys: list[int] = []
    eq = np.empty(end - start + 1); trades = []
    for t in range(start, end + 1):
        for j in sells:                                   # 아침: 어제 정한 매도·매수를 시가에
            sh, ent_t, ent_p, _ = pos.pop(j); px = o[t, j] if np.isfinite(o[t, j]) else c[t - 1, j]
            cash += sh * px * (1 - rules.cost); trades.append((j, ent_t, t, px / ent_p - 1.0))
        for j in buys:
            free = rules.slots - len(pos)
            if free <= 0 or j in pos or not np.isfinite(o[t, j]) or o[t, j] <= 0:
                continue
            amt = cash / free; pos[j] = [amt * (1 - rules.cost) / o[t, j], t, o[t, j], o[t, j]]; cash -= amt
        eq[t - start] = cash + sum(p[0] * c[t, j] for j, p in pos.items())
        sells = []
        for j, p in pos.items():                          # 종가: 내일 팔 것
            p[3] = max(p[3], c[t, j])
            if (not alive[t, j]) or (exit_line is not None and c[t, j] < exit_line[t, j]) or (rules.trail is not None and c[t, j] < p[3] * (1 - rules.trail)) \
                    or (rules.max_hold is not None and t - p[1] >= rules.max_hold):
                sells.append(j)
        free = rules.slots - (len(pos) - len(sells)); buys = []
        if free > 0:                                      # 종가: 내일 살 것
            cand = np.flatnonzero(elig[t])
            cand = cand[[j not in pos for j in cand]]     # 방금 팔기로 한 종목도 바로 다시 사지 않는다
            if len(cand):
                if rng is not None:
                    buys = list(rng.choice(cand, size=min(free, len(cand)), replace=False))
                else:
                    s = score[t, cand]; ok = np.isfinite(s); cand, s = cand[ok], s[ok]
                    buys = list(cand[np.argsort(-s, kind="stable")[:free]])
    for j, p in pos.items():
        trades.append((j, p[1], -1, c[end, j] / p[2] - 1.0))   # 끝까지 보유 중(exit −1)
    return eq, trades
