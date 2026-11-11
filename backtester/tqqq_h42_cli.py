"""H42 — 두 단계 이탈 + 사다리 (설계 tqqq_search_runs/H42_설계.md). H34 CLI 를 파라미터 매핑으로 실행."""

from __future__ import annotations

import json
import sys

import tqqq_h34_cli as h34
from strategy.adapters.ladder_reentry import LadderParams, ladder_switch

T = (1 / 3,) * 3; R = (0.50, 0.65, 0.80)
PARAMS = {"15/21 ½": LadderParams(R, T, exit_steps=((0.15, 0.5), (0.21, 0.0))), "12/18 ½": LadderParams(R, T, exit_steps=((0.12, 0.5), (0.18, 0.0))),
          "18/24 ½": LadderParams(R, T, exit_steps=((0.18, 0.5), (0.24, 0.0))), "15/21 ⅔남김": LadderParams(R, T, exit_steps=((0.15, 2 / 3), (0.21, 0.0))),
          "15/21 ⅓남김": LadderParams(R, T, exit_steps=((0.15, 1 / 3), (0.21, 0.0)))}

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    h34.LADDERS = {k: v.rungs for k, v in PARAMS.items()}; order = list(PARAMS)

    def _ls(qqq, tqqq, p, **kw):
        name = order[_ls.i]; _ls.i += 1
        return ladder_switch(qqq, tqqq, PARAMS[name], **kw)
    _ls.i = 0
    h34.ladder_switch = _ls
    h34.main(cfg)
