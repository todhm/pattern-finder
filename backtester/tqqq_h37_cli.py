"""H37 — 20일선 없이 사다리 + 시간 정지 재진입 (설계 tqqq_search_runs/H37_설계.md). H34 CLI 의 LADDERS 를 파라미터 객체로 바꿔 실행."""

from __future__ import annotations

import json
import sys

import tqqq_h34_cli as h34
from strategy.adapters.ladder_reentry import LadderParams, ladder_switch

T = (1 / 3,) * 3
PARAMS = {"50/65/80 · 60일/200선": LadderParams((0.50, 0.65, 0.80), T, reenter_mode="time", time_days=60, time_sma=200),
          "45/60/75 · 60일/200선": LadderParams((0.45, 0.60, 0.75), T, reenter_mode="time", time_days=60, time_sma=200),
          "50/65/80 · 120일/200선": LadderParams((0.50, 0.65, 0.80), T, reenter_mode="time", time_days=120, time_sma=200),
          "45/60/75 · 120일/200선": LadderParams((0.45, 0.60, 0.75), T, reenter_mode="time", time_days=120, time_sma=200),
          "50/65/80 · 60일/50선": LadderParams((0.50, 0.65, 0.80), T, reenter_mode="time", time_days=60, time_sma=50),
          "45/60/75 · 60일/50선": LadderParams((0.45, 0.60, 0.75), T, reenter_mode="time", time_days=60, time_sma=50)}


def _ladder(qqq, tqqq, p, **kw):
    return ladder_switch(qqq, tqqq, PARAMS[_ladder.current], **kw)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    h34.LADDERS = {k: v.rungs for k, v in PARAMS.items()}
    _orig_LP = h34.LadderParams

    class _LP:                       # h34 가 LadderParams(rungs, sizes) 로 만들면 이름으로 실제 파라미터를 찾는다
        def __new__(cls, rungs, sizes):
            return next(v for v in PARAMS.values() if v.rungs == rungs and v.time_days == _LP.td and v.time_sma == _LP.ts)
    h34.LadderParams = _orig_LP
    # 단순화: 이름별 순차 실행을 위해 h34.main 내부 루프가 LADDERS 를 돌며 LadderParams(rungs, sizes) 를 만들므로 rungs 만으로는 모호 → 이름 기반 매핑
    def _ls(qqq, tqqq, p, **kw):
        key = [k for k, v in PARAMS.items() if v.rungs == p.rungs and k not in _ls.done]
        name = key[0]; _ls.done.add(name)
        return ladder_switch(qqq, tqqq, PARAMS[name], **kw)
    _ls.done = set()
    h34.ladder_switch = _ls
    h34.main(cfg)
