"""H30 — 워크포워드 선택기 + 사다리·스티치 부품 (설계 tqqq_search_runs/H30_설계.md). H25 CLI 의 라이브러리를 확장해 실행."""

from __future__ import annotations

import json
import sys

import pandas as pd

from strategy.adapters.ladder_reentry import LadderParams, ladder_switch
from strategy.adapters.regime_slots import as_frame, regime_labels, RegimeParams, stitch
from strategy.adapters.walk_forward_select import WalkForwardParams
from strategy.adapters.whipsaw_switch import WhipsawParams, whipsaw_switch
import tqqq_h25_cli as h25

_base_lib = h25.library_v2


def library_v3(px, prices):
    lib = _base_lib(px, prices); idx = px.index; q = px["QQQ"]
    lad = as_frame(ladder_switch(q, px["TQQQ"], LadderParams((0.50, 0.65, 0.80), (1 / 3,) * 3)), idx); lib["사다리 50/65/80"] = lad
    lab = regime_labels(q, RegimeParams(c_depth=0.30)); msw = as_frame(whipsaw_switch(q, WhipsawParams(0.0, True, False)).reindex(idx).ffill().fillna(1.0), idx)
    lib["스티치 R48"] = stitch({"A": lad, "B": lad, "D": lad, "C": msw}, lab, idx)
    return lib


def combos():
    return [WalkForwardParams(3, "mult", True, None, 1), WalkForwardParams(3, "mult", True, None, 2), WalkForwardParams(5, "mult", True, None, 1), WalkForwardParams(5, "mult", True, None, 2),
            WalkForwardParams(5, "mult", False, None, 1), WalkForwardParams(5, "mult", False, None, 2)]


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    h25.library_v2 = library_v3; h25.combos = combos
    h25.main(cfg)
