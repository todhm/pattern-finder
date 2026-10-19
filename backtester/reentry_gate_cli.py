"""약세장 재진입 게이트 — ⑩ 의 이탈은 그대로, 재진입만 바꿨을 때의 세후 성과.

    docker compose exec backtester python reentry_gate_cli.py '{"out": "/app/sweep_results/rg.json", "md": "/app/sweep_results/rg.md"}'
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date

import pandas as pd

from crash_early_warning_cli import SAMPLES, WHIPSAW_YEARS, _metrics, _rolling, build_signals
from strategy.adapters.crash_early_warning import reentry_gate_filter
from strategy.adapters.leveraged_etf_lab import CostParams, deep_correction_filter, simulate_exposure
from vol_target_regime_cli import BOOM_YEARS, CRASH_YEARS, load

V_YEARS = [2009, 2020, 2023]  # V자 반등 — 재진입 지연의 비용이 드러나는 해


def main(cfg: dict) -> dict:
    end = date.fromisoformat(cfg.get("as_of", "2026-10-16"))
    t0 = time.perf_counter()
    lev, ndx, cash = load(end)
    sigs = build_signals(ndx, end)
    alarms = {
        "신용": sigs["신용스프레드 급등"], "VIX>30": sigs["VIX > 30"], "청구": sigs["실업수당 청구 급증"],
        "신용|VIX|청구": sigs["신용 OR VIX기간 OR 청구"].where(sigs["VIX 기간구조 역전"].notna(), sigs["신용스프레드 급등"]),
        "신용|VIX30": pd.concat([sigs["신용스프레드 급등"], sigs["VIX > 30"]], axis=1).max(axis=1),
        "Sahm": sigs["실업률 Sahm"], "12m추세": sigs["12개월 추세 하회"],
    }
    variants = {"⑩ dd18 (기준)": deep_correction_filter(ndx, 200, 0.18, 20)}
    for name, a in alarms.items():
        variants[f"a 금지 [{name}]"] = reentry_gate_filter(ndx, a, gate_mode="block")
        variants[f"b 절반 [{name}]"] = reentry_gate_filter(ndx, a, gate_mode="partial", partial=0.5)
    for n2 in (50, 100, 200):
        variants[f"c 가격 20일&{n2}일선"] = reentry_gate_filter(ndx, None, gate_mode="price", reenter_n2=n2)
    variants["c 가격 50일선만"] = reentry_gate_filter(ndx, None, gate_mode="price", reenter_n=50)
    costs = CostParams(tax_rate=float(cfg.get("tax", 0.22)), exec_lag=1, rebalance_band=0.05)
    out = {"as_of": end.isoformat(), "strategies": {}, "yearly": {}, "rolling": {}}
    for vname, w in variants.items():
        w = w.reindex(lev.index).ffill().fillna(0.0); rec = {}
        for sname, start in SAMPLES.items():
            l = lev[lev.index >= pd.Timestamp(start)]
            hold = simulate_exposure(l, pd.Series(1.0, index=l.index), costs, cash, "hold")
            r = simulate_exposure(l, w.reindex(l.index), costs, cash, vname)
            rec[sname] = _metrics(r, hold)
            if sname == "C_1986":
                ys = r.equity.groupby(r.equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
                hy = hold.equity.groupby(hold.equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
                rec["crash_mean"] = float(ys[CRASH_YEARS].mean()); rec["whipsaw_mean"] = float(ys[WHIPSAW_YEARS].mean())
                rec["v_mean"] = float(ys[V_YEARS].mean()); rec["boom_capture"] = float(ys[BOOM_YEARS].mean() / hy[BOOM_YEARS].mean())
                out["yearly"][vname] = {int(k): float(v) for k, v in ys.items()}
        out["strategies"][vname] = rec
    print(f"strategies done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    base = out["strategies"]["⑩ dd18 (기준)"]
    picks = ["⑩ dd18 (기준)"] + sorted([v for v, r in out["strategies"].items() if v != "⑩ dd18 (기준)" and r["A_2010"]["ratio"] >= 0.88 and r["B_1999"]["mdd"] > base["B_1999"]["mdd"] + 0.03],
                                   key=lambda v: out["strategies"][v]["B_1999"]["mdd"], reverse=True)[:7]
    for v in picks:
        w = variants[v].reindex(lev.index).ffill().fillna(0.0)
        out["rolling"][v] = {"@1986": _rolling(lev, w, costs, cash, date(1986, 1, 1)), "@1999": _rolling(lev, w, costs, cash, date(1999, 12, 1))}
    print(f"rolling done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    if cfg.get("out"):
        json.dump(out, open(cfg["out"], "w"), ensure_ascii=False, indent=1, default=str)
    if cfg.get("md"):
        open(cfg["md"], "w").write(render(out))
    return out


def render(out: dict) -> str:
    L = [f"# 약세장 재진입 게이트 (세후 22%, lag 1, ~{out['as_of']})", "",
         "| 변형 | A 2010~ 비율 / MDD / 매매 | B 1999~ 배수 / MDD | C 1986~ 배수 / MDD | 참사5년 | 휩소3년 | V자3년(09·20·23) | 폭발 포착 |", "|---|---|---|---|---|---|---|---|"]
    for v, rec in sorted(out["strategies"].items(), key=lambda kv: kv[1]["B_1999"]["mdd"], reverse=True):
        a, b, c = rec["A_2010"], rec["B_1999"], rec["C_1986"]
        L.append(f"| {v} | **{a['ratio']:.2f}** / {a['mdd']:.0%} / {a['trades']} | {b['mult']:.0f}x / **{b['mdd']:.0%}** | {c['mult']:.0f}x / {c['mdd']:.0%} | {rec['crash_mean']:+.0%} | {rec['whipsaw_mean']:+.0%} | {rec['v_mean']:+.0%} | {rec['boom_capture']:.0%} |")
    L += ["", "## 10년 롤링 (A ≥ 0.88 이고 B MDD 3%p 이상 개선된 변형)", "", "| 변형 | @1986 승률 / 최악 CAGR / 원금손실 | @1999 승률 / 최악 CAGR / 원금손실 |", "|---|---|---|"]
    for v, r in out["rolling"].items():
        a, b = r["@1986"], r["@1999"]
        L.append(f"| {v} | {a['win_rate']:.0%} / **{a['worst_cagr']:+.0%}** / {a['loss_rate']:.0%} | {b['win_rate']:.0%} / **{b['worst_cagr']:+.0%}** / {b['loss_rate']:.0%} |")
    ys = out["yearly"]; names = [n for n in ("⑩ dd18 (기준)", "a 금지 [신용|VIX30]", "b 절반 [신용|VIX30]", "c 가격 20일&50일선", "c 가격 20일&200일선") if n in ys]
    L += ["", "## 연도별 (C 표본)", "", "| 연도 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for y in [2000, 2001, 2002, 2008, 2022, 2009, 2020, 2023, 2011, 2018, 2025, 2003, 2013, 2017, 2019, 2021]:
        L.append(f"| {y} | " + " | ".join(f"{ys[n][y]:+.0%}" for n in names) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
