"""H38 — 스티치 긴 약세장 깊이 35/40% (설계 tqqq_search_runs/H38_설계.md). H31 CLI 를 build/combos 교체로 실행."""

from __future__ import annotations

import json
import sys
import types

import pandas as pd

import tqqq_h31_cli as h31
from strategy.adapters.ladder_reentry import ladder_switch
from strategy.adapters.regime_slots import as_frame, regime_labels, RegimeParams, stitch
from strategy.adapters.whipsaw_switch import WhipsawParams, whipsaw_switch

COMBOS = [(0.30, 120), (0.35, 120), (0.35, 10**6), (0.40, 120), (0.40, 10**6)]


def build(px, prices, depth, c_comp):
    long_days = int(c_comp)                      # c_comp 자리에 체류 일수를 넣어 전달
    idx = px.index; q = px["QQQ"]
    lab = regime_labels(q, RegimeParams(long_days=long_days, c_depth=depth))
    lad = as_frame(ladder_switch(q, px["TQQQ"], h31.LADDER), idx)
    msw = as_frame(whipsaw_switch(q, WhipsawParams(0.0, True, False)).reindex(idx).ffill().fillna(1.0), idx)
    slots = {"A": lad, "B": lad, "D": lad, "C": msw}
    return lab, slots, stitch(slots, lab, idx)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    src = open(h31.__file__).read()
    code = (src.replace('combos = [(dep, c) for dep in (0.30, 0.25) for c in ("월말 스위치 18", "현금", "QLD+월말 스위치")]',
                        'combos = [(d, str(l)) for d, l in [(0.30, 120), (0.35, 120), (0.35, 10**6), (0.40, 120), (0.40, 10**6)]]')
               .replace('name = f"stitch C{dep:.2f}={c} | else=sw18+ladder"', 'name = f"stitch depth{dep:.2f} 체류{\'120일\' if c == \'120\' else \'끔\'} C=월말 | else=sw18+ladder"')
               .replace("[{i + 1}/6]", "[{i + 1}/5]").replace("# H31 — 레짐 스티치", "# H38 — 스티치 긴 약세장 깊이 35/40%")
               .replace('(q["depth"] == r["depth"] or q["c_comp"] == r["c_comp"])', '(abs(q["depth"] - r["depth"]) <= 0.051 or q["c_comp"] == r["c_comp"])'))
    mod = types.ModuleType("h38_run"); mod.__dict__["__file__"] = h31.__file__
    exec(compile(code, "h38_run", "exec"), mod.__dict__)
    mod.build = build
    mod.main(cfg)
