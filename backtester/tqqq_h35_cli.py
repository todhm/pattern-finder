"""H35 — QQQ 낙폭 기준 사다리 → 관문 (설계 tqqq_search_runs/H35_설계.md). H34 CLI 의 LADDERS·낙폭 시계열만 바꿔 실행."""

from __future__ import annotations

import json
import sys

import pandas as pd

import tqqq_h34_cli as h34
from strategy.adapters.ladder_reentry import LadderParams, ladder_switch

LADDERS = {"QQQ 3칸 20/28/36": (0.20, 0.28, 0.36), "QQQ 3칸 18/26/34": (0.18, 0.26, 0.34), "QQQ 3칸 22/30/38": (0.22, 0.30, 0.38),
           "QQQ 2칸 20/30": (0.20, 0.30), "QQQ 4칸 18/25/32/39": (0.18, 0.25, 0.32, 0.39)}

_orig = ladder_switch


def _ladder_on_qqq(qqq, tqqq, p, **kw):
    return _orig(qqq, qqq, p, **kw)          # 낙폭 시계열을 QQQ 로


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    h34.LADDERS = LADDERS; h34.ladder_switch = _ladder_on_qqq
    h34.main(cfg)
