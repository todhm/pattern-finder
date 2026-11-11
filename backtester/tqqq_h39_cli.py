"""H39 — 스위치 문턱 축 + 사다리 (설계 tqqq_search_runs/H39_설계.md). H34 CLI 를 dd 축으로 실행."""

from __future__ import annotations

import json
import sys

import tqqq_h34_cli as h34
from strategy.adapters.ladder_reentry import LadderParams, ladder_switch

T = (1 / 3,) * 3
PARAMS = {f"dd {dd:.0%} + 사다리 50/65/80": LadderParams((0.50, 0.65, 0.80), T, dd=dd) for dd in (0.15, 0.18, 0.21, 0.24)}

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    h34.LADDERS = {k: v.rungs for k, v in PARAMS.items()}
    order = list(PARAMS)

    def _ls(qqq, tqqq, p, **kw):
        name = order[_ls.i]; _ls.i += 1
        return ladder_switch(qqq, tqqq, PARAMS[name], **kw)
    _ls.i = 0
    h34.ladder_switch = _ls
    h34.main(cfg)
