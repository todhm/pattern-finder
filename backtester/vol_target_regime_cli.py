"""Phase 1 배치 — 변동성 타기팅 × ⑩ 낙폭 레짐, 세 표본 · 세후 · lag 1.

    docker compose exec backtester python vol_target_regime_cli.py '{"out": "/app/sweep_results/vt_regime.json", "md": "/app/sweep_results/vt_regime.md"}'

표본: A 실제 TQQQ 2010-02-11~ · B 1999-12-01~ (합성 3배 → 2010 실데이터 스플라이스) · C 1986-01-01~ (^NDX 합성).
기초·신호 지수는 ^NDX (1985~). 격자: 목표 30/45/60 × 예측기 4 × 레짐 on/off = 24 + ⑩ 단독 + 보유.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.band_rebalance_strategy import CALIBRATED_DRAG, build_synthetic_leveraged, splice_series
from strategy.adapters.leveraged_etf_data import _close
from strategy.adapters.leveraged_etf_lab import CostParams, deep_correction_filter, simulate_exposure
from strategy.adapters.vol_target_regime_lab import (
    PREDICTORS, VtParams, rolling, rolling_summary, run_hold, run_one, summarize, yearly,
)

SAMPLES = {"A_2010": date(2010, 2, 11), "B_1999": date(1999, 12, 1), "C_1986": date(1986, 1, 2)}
CRASH_YEARS = [2000, 2001, 2002, 2008, 2022]
BOOM_YEARS = [1989, 1991, 1995, 1996, 1998, 1999, 2003, 2009, 2012, 2013, 2014, 2017, 2019, 2021, 2023]


def load(end: date):
    md = build_default_market_data()
    ndx = _close(md.fetch_ohlcv("^NDX", date(1985, 1, 1), end))
    cash = _close(md.fetch_ohlcv("^IRX", date(1985, 1, 1), end)) / 100.0
    syn = build_synthetic_leveraged(ndx, 3.0, annual_expense=CALIBRATED_DRAG[3.0], financing_rate=cash)
    real = _close(md.fetch_ohlcv("TQQQ", date(2010, 2, 11), end))
    lev = splice_series(syn, real)
    return lev, ndx, cash


def main(cfg: dict) -> dict:
    end = date.fromisoformat(cfg.get("as_of", "2026-10-16"))
    t0 = time.perf_counter()
    lev, ndx, cash = load(end)
    print(f"data {len(lev)} days {lev.index[0].date()}~{lev.index[-1].date()} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    targets = cfg.get("targets", [0.30, 0.45, 0.60])
    tax = cfg.get("tax", 0.22)
    costs = CostParams(tax_rate=tax, exec_lag=1, rebalance_band=float(cfg.get("band", 0.05)))
    grid = [VtParams(target=t, predictor=pr, regime=rg) for t in targets for pr in PREDICTORS for rg in (False, True)]
    out = {"as_of": end.isoformat(), "tax": tax, "band": costs.rebalance_band, "samples": {}, "yearly": {}, "rolling": {}}
    for sname, start in SAMPLES.items():
        hold = run_hold(lev, costs, cash, start)
        rows = [{**summarize(hold, hold)}]
        # ⑩ 단독
        d = deep_correction_filter(ndx, 200, 0.18, 20).reindex(lev.index).ffill().fillna(0.0)
        levs = lev[lev.index >= pd.Timestamp(start)]
        dd_only = simulate_exposure(levs, d.reindex(levs.index), costs, cash, "dd18_only")
        rows.append(summarize(dd_only, hold))
        for p in grid:
            r = run_one(lev, ndx, p, costs, cash, start)
            rows.append(summarize(r, hold))
            if sname == "C_1986":
                out["yearly"][p.name] = {int(k): float(v) for k, v in yearly(r).items()}
        if sname == "C_1986":
            out["yearly"]["hold"] = {int(k): float(v) for k, v in yearly(hold).items()}
            out["yearly"]["dd18_only"] = {int(k): float(v) for k, v in yearly(dd_only).items()}
        out["samples"][sname] = rows
        print(f"{sname} done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    # 롤링 10년: A 표본에서 세후 보유의 90% 이상이면서 B 에서 비율 최고인 상위 3 + ⑩
    a = {r["name"]: r for r in out["samples"]["A_2010"]}
    b = sorted((r for r in out["samples"]["B_1999"] if r["name"] not in ("hold",)), key=lambda r: -r["ratio_to_hold"])
    picks = [r["name"] for r in b if a[r["name"]]["ratio_to_hold"] >= 0.9][:3]
    picks = list(dict.fromkeys(picks + ["dd18_only"] + [b[0]["name"]]))
    out["picks"] = picks
    for name in picks:
        if name == "dd18_only":
            p = None
        else:
            t, pr = name.split("_")[0], "_".join(name.split("_")[1:]).replace("_dd", "")
            p = VtParams(target=int(t[2:]) / 100, predictor=pr, regime=name.endswith("_dd"))
        for first in (date(1986, 1, 1), date(1999, 12, 1)):
            if p is None:
                dfp = _rolling_dd(lev, ndx, costs, cash, first)
            else:
                dfp = rolling(lev, ndx, p, costs, cash, 10, first)
            out["rolling"][f"{name}@{first.year}"] = rolling_summary(dfp)
        print(f"rolling {name} done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    # 참사/폭발 연도 요약 (C 표본 연도별)
    out["crash_boom"] = {}
    for name, ys in out["yearly"].items():
        cr = [ys[y] for y in CRASH_YEARS if y in ys]; bo = [ys[y] for y in BOOM_YEARS if y in ys]
        out["crash_boom"][name] = {"crash_mean": sum(cr) / len(cr) if cr else None, "crash_worst": min(cr) if cr else None,
                                  "boom_mean": sum(bo) / len(bo) if bo else None,
                                  "boom_capture": (sum(bo) / len(bo)) / (sum(out["yearly"]["hold"][y] for y in BOOM_YEARS if y in ys) / len(bo)) if bo else None}
    out["elapsed_sec"] = round(time.perf_counter() - t0)
    if cfg.get("out"):
        with open(cfg["out"], "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    if cfg.get("md"):
        with open(cfg["md"], "w") as f:
            f.write(render_md(out))
    return out


def _rolling_dd(lev, ndx, costs, cash, first):
    d = deep_correction_filter(ndx, 200, 0.18, 20).reindex(lev.index).ffill().fillna(0.0)
    idx = lev.index
    starts = pd.date_range(first, idx[-1] - pd.DateOffset(years=10), freq="3MS")
    rows = []
    for s in starts:
        e = s + pd.DateOffset(years=10)
        if e > idx[-1]:
            break
        l = lev[(lev.index >= s) & (lev.index <= e)]
        hold = simulate_exposure(l, pd.Series(1.0, index=l.index), costs, cash, "hold")
        st = simulate_exposure(l, d.reindex(l.index), costs, cash, "dd")
        hm = hold.final_after_liquidation / hold.equity.iloc[0]; sm = st.final_after_liquidation / st.equity.iloc[0]
        rows.append({"start": s.date(), "hold_mult": hm, "strat_mult": sm, "ratio": sm / hm, "hold_cagr": hm ** 0.1 - 1,
                     "strat_cagr": sm ** 0.1 - 1, "hold_mdd": hold.metrics.get("mdd"), "strat_mdd": st.metrics.get("mdd")})
    return pd.DataFrame(rows)


def render_md(out: dict) -> str:
    L = [f"# Phase 1 — 변동성 타기팅 × ⑩ 레짐 (세후 {out['tax']:.0%}, lag 1, 밴드 {out['band']:.0%}, ~{out['as_of']})", ""]
    for sname, rows in out["samples"].items():
        L += [f"## 표본 {sname}", "", "| 설정 | 배수 | CAGR | 보유 대비 | MDD | 보유 MDD | 매매 | 평균 노출 | 세금(억) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -r["ratio_to_hold"]):
            L.append(f"| {r['name']} | {r['mult']:.1f}x | {r['cagr']:+.1%} | **{r['ratio_to_hold']:.2f}** | {r['mdd']:.0%} | {r['hold_mdd']:.0%} | {r['trades']} | {r['avg_exposure']:.0%} | {r['tax'] / 1e8:.1f} |")
        L.append("")
    L += ["## 10년 롤링 창", "", "| 설정@시작 | 창 | 승률 | 비율 중앙 | 최악 | 전략 CAGR 중앙/최악 | 보유 CAGR 중앙/최악 | MDD 중앙 전략/보유 | 원금손실 전략/보유 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in out["rolling"].items():
        if s:
            L.append(f"| {k} | {s['windows']} | **{s['win_rate']:.0%}** | {s['ratio_median']:.2f} | {s['ratio_min']:.2f} | {s['strat_cagr_median']:+.0%} / {s['strat_cagr_min']:+.0%} | {s['hold_cagr_median']:+.0%} / {s['hold_cagr_min']:+.0%} | {s['strat_mdd_median']:.0%} / {s['hold_mdd_median']:.0%} | {s['strat_loss_rate']:.0%} / {s['hold_loss_rate']:.0%} |")
    L += ["", "## 참사 5년 / 폭발 15년 (C 표본 연도별 수익)", "", "| 설정 | 참사 평균 | 참사 최악 | 폭발 평균 | 폭발 포착률(보유 대비) |", "|---|---|---|---|---|"]
    for name, cb in sorted(out["crash_boom"].items(), key=lambda kv: -(kv[1]["boom_capture"] or 0)):
        L.append(f"| {name} | {cb['crash_mean']:+.0%} | {cb['crash_worst']:+.0%} | {cb['boom_mean']:+.0%} | {cb['boom_capture']:.0%} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
