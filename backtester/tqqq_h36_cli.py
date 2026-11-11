"""H36 — 스티치 R-C 부품 교체 (설계 tqqq_search_runs/H36_설계.md). H31 CLI 의 build/combos 를 바꿔 실행."""

from __future__ import annotations

import json
import sys

import pandas as pd

import tqqq_h31_cli as h31
from strategy.adapters.ladder_reentry import LadderParams, ladder_switch
from strategy.adapters.regime_slots import as_frame, regime_labels, RegimeParams, stitch
from strategy.adapters.whipsaw_switch import WhipsawParams, whipsaw_switch


def build(px, prices, depth, c_comp):
    idx = px.index; q = px["QQQ"]
    lab = regime_labels(q, RegimeParams(c_depth=depth))
    lad = as_frame(ladder_switch(q, px["TQQQ"], h31.LADDER), idx)
    msw = whipsaw_switch(q, WhipsawParams(0.0, True, False)).reindex(idx).ffill().fillna(1.0)
    comps = {"QLD 보유": pd.DataFrame({"TQQQ": 0.0, "QLD": 1.0, "QQQ": 0.0}, index=idx),
             "QLD+월말 스위치": pd.DataFrame({"TQQQ": 0.0, "QLD": msw.values, "QQQ": 0.0}, index=idx),
             "기본 규칙 ½": lad * 0.5}
    slots = {"A": lad, "B": lad, "D": lad, "C": comps[c_comp]}
    return lab, slots, stitch(slots, lab, idx)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    h31.build = build
    src = open(h31.__file__).read()
    # combos 는 main 안에 하드코딩 → 작은 래퍼: main 을 복사해 combos 만 교체
    import types
    code = src.replace('combos = [(dep, c) for dep in (0.30, 0.25) for c in ("월말 스위치 18", "현금", "QLD+월말 스위치")]',
                       'combos = [(0.30, "QLD 보유"), (0.30, "QLD+월말 스위치"), (0.30, "기본 규칙 ½"), (0.25, "QLD 보유")]').replace("[{i + 1}/6]", "[{i + 1}/4]").replace("# H31 — 레짐 스티치", "# H36 — 스티치 R-C 부품 교체")
    mod = types.ModuleType("h36_run"); mod.__dict__["__file__"] = h31.__file__
    exec(compile(code, "h36_run", "exec"), mod.__dict__)
    mod.build = build
    mod.main(cfg)
