"""H22 — 레짐 슬롯 장부: 라벨 → 에피소드 → 부품 라이브러리 슬롯 평가 → 킵 → 스티치 관문 (프로토콜 §5).

    docker compose exec backtester python tqqq_h22_cli.py '{"out": "/app/sweep_results/tqqq_h22.json", "md": "/app/sweep_results/tqqq_h22.md", "n_trials": 12}'
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.lev_mix import LevMixParams, lev_mix_weights
from strategy.adapters.regime_slots import (
    CODES, COLS, NAMES, RegimeParams, as_frame, episode_table, episodes, placebo_slot_medians, regime_labels, shuffle_labels,
    slot_pass, slot_stats, stitch,
)
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, verdict_markdown, window_table
from strategy.adapters.vt_qqq_mix import VtMixParams, vt_mix_weights
from strategy.adapters.whipsaw_switch import WhipsawParams, whipsaw_switch
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate
from tqqq_h11_cli import load_qld


def library(px: pd.DataFrame, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx = px.index; q = px["QQQ"]
    lib = {
        "TQQQ 보유": pd.Series(1.0, index=idx),
        "현금": pd.Series(0.0, index=idx),
        "스위치 18": deep_correction_filter(q, 200, 0.18, 20),
        "스위치 24": deep_correction_filter(q, 200, 0.24, 20),
        "월말 스위치 18": whipsaw_switch(q, WhipsawParams(0.0, True, False)),
        "QLD+스위치 18": lev_mix_weights(prices, q, LevMixParams("qld", switch_dd=0.18)),
        "vt60+스위치 18": vt_mix_weights(px["TQQQ"], q, VtMixParams(0.60, "rv20", "cash")),
    }
    return {k: as_frame(v, idx) for k, v in lib.items()}


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    qld = load_qld(px.index, cash)
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": qld, "QQQ": px["QQQ"]})
    rp = RegimeParams(**cfg.get("regime", {}))
    dev_end = pd.Timestamp("2024-06-30")
    labels = regime_labels(px["QQQ"], rp)
    eps = episodes(labels[labels.index <= dev_end])
    lib = library(px, prices)
    if cfg.get("library") == "v2":                       # H26: H25 의 넓은 라이브러리 + 현금
        from tqqq_h25_cli import library_v2
        lib = {**library_v2(px, prices), "현금": lib["현금"]}
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    share = labels[(labels.index >= "1986-01-01") & (labels.index <= dev_end)].value_counts(normalize=True).to_dict()
    print(f"라벨 비율 1986~: " + ", ".join(f"{c} {share.get(c, 0):.0%}" for c in CODES) + f" · 에피소드 {len(eps)}(평가 {int(eps['eval'].sum())}) ({time.time() - t0:.0f}s)", file=sys.stderr)
    hold = lib["TQQQ 보유"]
    tables = {name: episode_table(prices, f, hold, eps, GATE_COSTS, cash) for name, f in lib.items()}
    print(f"에피소드 평가 완료 ({time.time() - t0:.0f}s)", file=sys.stderr)
    slot_rows, slots, kept = [], {}, {}
    for code in CODES:
        best = None
        for name, tbl in tables.items():
            st = slot_stats(tbl, code)
            if st.get("n", 0) == 0:
                continue
            meds = placebo_slot_medians(tbl, code, seeds)
            pct = float(100.0 * (np.asarray(meds) < st["ratio_median"]).mean())
            ok = slot_pass(st, pct)
            slot_rows.append({"code": code, "component": name, **st, "placebo_pct": pct, "pass": ok})
            if ok and (best is None or st["ratio_median"] > best[1]["ratio_median"]):
                best = (name, st)
        if best:
            slots[code] = lib[best[0]]; kept[code] = best[0]
        else:
            slots[code] = hold; kept[code] = "TQQQ 보유(임시)"
        print(f"슬롯 R-{code} {NAMES[code]}: 킵 = {kept[code]}", file=sys.stderr)
    st_target = stitch(slots, labels, px.index)
    tbl = window_table(px["TQQQ"], st_target, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices)
    # G5: 에피소드 라벨 셔플 스티치
    meds = []
    for s in range(seeds):
        sh = shuffle_labels(labels, 20261110 + s)
        t2 = stitch(slots, sh, px.index)
        tb = window_table(px["TQQQ"], t2, base, "dev", ("A-10", "A-∞"), GATE_COSTS, cash, prices)
        meds.append(float(tb["ratio"].median()))
    # G6: 레짐 손잡이 ±20% 이웃
    nb = []
    for q in (RegimeParams(rp.n, rp.dd_thr * 0.8, rp.long_days, rp.recover_days), RegimeParams(rp.n, rp.dd_thr * 1.2, rp.long_days, rp.recover_days),
              RegimeParams(rp.n, rp.dd_thr, int(rp.long_days * 0.8), rp.recover_days), RegimeParams(rp.n, rp.dd_thr, int(rp.long_days * 1.2), rp.recover_days)):
        lb = regime_labels(px["QQQ"], q); t3 = stitch(slots, lb, px.index)
        nb.append(gate_verdicts(window_table(px["TQQQ"], t3, base, "dev", ("A-10", "A-∞"), GATE_COSTS, cash, prices), n_trials))
    v = gate_verdicts(tbl, n_trials, meds, nb)
    # 정보용: 슬롯마다 '통과 안 해도 중앙 최고' 부품으로 스티치
    greedy = {}
    for code in CODES:
        cand = [r for r in slot_rows if r["code"] == code]
        greedy[code] = max(cand, key=lambda r: r["ratio_median"])["component"] if cand else "TQQQ 보유"
    g_target = stitch({c: lib[greedy[c]] for c in CODES}, labels, px.index)
    g_tbl = window_table(px["TQQQ"], g_target, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices)
    g_v = gate_verdicts(g_tbl, n_trials)
    out = {"cfg": cfg, "regime": rp.__dict__, "label_share": share, "episodes": json.loads(eps.to_json(orient="records", date_format="iso")),
           "slot_rows": slot_rows, "kept": kept, "stitch_verdict": v, "greedy": greedy, "greedy_verdict": g_v,
           "stitch_windows": json.loads(tbl.to_json(orient="records", date_format="iso")), "elapsed_s": time.time() - t0}
    md = render(out, tbl, g_tbl, eps)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:3000])
    return out


def render(out, tbl, g_tbl, eps) -> str:
    L = [f"# H22/H26 — 레짐 슬롯 (개발 판, 끝 2024-06-30) regime={out['regime']}", "",
         "라벨 비율 1986~: " + ", ".join(f"R-{c} {NAMES[c]} {out['label_share'].get(c, 0):.0%}" for c in CODES) + f" · 에피소드 {len(eps)}", ""]
    L += ["## 에피소드(평가 대상, ≥20일)", "", "| 레짐 | 시작 | 끝 | 일수 |", "|---|---|---|---|"]
    for e in eps[eps["eval"]].itertuples(index=False):
        L.append(f"| R-{e.code} | {pd.Timestamp(e.start).date()} | {pd.Timestamp(e.end).date()} | {e.days} |")
    L += ["", "## 슬롯 평가 (부품 ÷ TQQQ 보유, 에피소드별 세후 배수 비율)", "", "| 레짐 | 부품 | 에피소드 | 승률 | 비율 중앙 | 최악 | 최고 | 플라세보 백분위 | §5.2 통과 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(out["slot_rows"], key=lambda r: (r["code"], -r["ratio_median"])):
        L.append(f"| R-{r['code']} | {r['component']} | {r['n']} | {r['win_rate']:.0%} | {r['ratio_median']:.2f} | {r['ratio_min']:.2f} | {r['ratio_max']:.2f} | {r['placebo_pct']:.0f}% | {'✅' if r['pass'] else '—'} |")
    L += ["", "## 킵한 슬롯", "", "| 슬롯 | 부품 |", "|---|---|"] + [f"| R-{c} {NAMES[c]} | {out['kept'][c]} |" for c in CODES]
    L += ["", "## 스티치(킵 슬롯) 관문", "", verdict_markdown("스티치(킵)", tbl, out["stitch_verdict"], "dev")]
    L += ["", f"## 정보: 통과 여부 무시하고 레짐별 중앙 최고 부품 스티치 — " + ", ".join(f"R-{c} {out['greedy'][c]}" for c in CODES), "", verdict_markdown("스티치(탐욕)", g_tbl, out["greedy_verdict"], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
