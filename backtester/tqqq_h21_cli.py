"""H21 — 15분봉 극단일 회피(갭·첫 1시간) → 관문 (설계 tqqq_search_runs/H21_설계.md). H20 CLI 의 평가 경로 재사용."""

from __future__ import annotations

import json
import sys

from strategy.adapters.intraday_rules import IntradayParams
import tqqq_h20_cli as h20


def combos():
    out = [IntradayParams("gap", x, 0, rc) for x in (0.04, 0.06, 0.08) for rc in (False, True)]
    out += [IntradayParams("firsthour", y) for y in (0.04, 0.06)]
    return out


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    h20.combos_all = combos
    h20.render = (lambda f: (lambda df, v, t: f(df, v, t).replace("# H20 — 15분봉 장중 규칙", "# H21 — 15분봉 극단일 회피")))(h20.render)
    h20.main(cfg)
