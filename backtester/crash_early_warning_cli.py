"""참사 조기경보 배치 — 신호별 혼동행렬 + "⑩ 에 조기 이탈 조건 추가" 의 세후 성과.

    docker compose exec backtester python crash_early_warning_cli.py '{"out": "/app/sweep_results/cew.json", "md": "/app/sweep_results/cew.md"}'
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.fred_adapter import FredCsvAdapter
from strategy.adapters.crash_early_warning import (
    confusion, early_exit_filter, episodes, sig_claims, sig_credit_spread, sig_sahm, sig_term_spread,
    sig_trend_12m, sig_vix_level, sig_vix_term,
)
from strategy.adapters.leveraged_etf_data import _close
from strategy.adapters.leveraged_etf_lab import CostParams, deep_correction_filter, simulate_exposure
from vol_target_regime_cli import BOOM_YEARS, CRASH_YEARS, load

SAMPLES = {"A_2010": date(2010, 2, 11), "B_1999": date(1999, 12, 1), "C_1986": date(1986, 1, 2)}
WHIPSAW_YEARS = [2011, 2018, 2025]


def build_signals(ndx: pd.Series, end: date) -> dict[str, pd.Series]:
    f = FredCsvAdapter(); md = build_default_market_data()
    s0 = date(1985, 1, 1)
    idx = ndx.index
    vix = _close(md.fetch_ohlcv("^VIX", s0, end)); vix3 = _close(md.fetch_ohlcv("^VIX3M", s0, end))
    sigs = {
        "기간스프레드 역전": sig_term_spread(f.fetch_series("T10Y3M", s0, end), idx),
        "신용스프레드 급등": sig_credit_spread(f.fetch_series("BAA10Y", s0, end), idx),
        "실업률 Sahm": sig_sahm(f.fetch_series("UNRATE", s0, end), idx),
        "실업수당 청구 급증": sig_claims(f.fetch_series("ICSA", s0, end), idx),
        "VIX > 30": sig_vix_level(vix, idx),
        "VIX 기간구조 역전": sig_vix_term(vix, vix3, idx),
        "12개월 추세 하회": sig_trend_12m(ndx),
    }
    sigs["신용 OR VIX기간 OR 청구"] = pd.concat([sigs["신용스프레드 급등"], sigs["VIX 기간구조 역전"].fillna(0.0), sigs["실업수당 청구 급증"]], axis=1).max(axis=1)
    return sigs


def _metrics(res, hold):
    yrs = len(res.equity) / 252
    m = res.final_after_liquidation / res.equity.iloc[0]; hm = hold.final_after_liquidation / hold.equity.iloc[0]
    return {"mult": m, "cagr": m ** (1 / yrs) - 1, "ratio": m / hm, "mdd": res.metrics.get("mdd"), "trades": res.trades, "tax": res.tax}


def _rolling(lev, w, costs, cash, first, years=10):
    idx = lev.index; rows = []
    for s in pd.date_range(first, idx[-1] - pd.DateOffset(years=years), freq="3MS"):
        e = s + pd.DateOffset(years=years)
        if e > idx[-1]:
            break
        l = lev[(lev.index >= s) & (lev.index <= e)]
        h = simulate_exposure(l, pd.Series(1.0, index=l.index), costs, cash, "h")
        r = simulate_exposure(l, w.reindex(l.index).ffill().fillna(0.0), costs, cash, "s")
        hm = h.final_after_liquidation / h.equity.iloc[0]; sm = r.final_after_liquidation / r.equity.iloc[0]
        rows.append({"ratio": sm / hm, "strat_cagr": sm ** (1 / years) - 1, "strat_mult": sm})
    df = pd.DataFrame(rows)
    return {"windows": len(df), "win_rate": float((df["ratio"] > 1).mean()), "worst_cagr": float(df["strat_cagr"].min()),
            "loss_rate": float((df["strat_mult"] < 1).mean())}


def main(cfg: dict) -> dict:
    end = date.fromisoformat(cfg.get("as_of", "2026-10-16"))
    t0 = time.perf_counter()
    lev, ndx, cash = load(end)
    sigs = build_signals(ndx, end)
    base = deep_correction_filter(ndx, 200, 0.18, 20)
    eps = episodes(base, lev)
    out = {"as_of": end.isoformat(), "episodes": [{"exit": str(e.exit_date.date()), "reentry": str(e.reentry_date.date()) if e.reentry_date is not None else None,
                                                   "etf_ret": e.hold_ret, "kind": e.kind} for e in eps],
           "confusion": {}, "strategies": {}, "yearly": {}}
    for name, s in sigs.items():
        out["confusion"][name] = confusion(s, eps)
    print(f"signals+confusion done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    costs = CostParams(tax_rate=float(cfg.get("tax", 0.22)), exec_lag=1, rebalance_band=0.05)
    dd_alarms = cfg.get("dd_alarm", [0.05, 0.08, 0.12])
    variants = {"⑩ dd18 (기준)": base}
    for name, s in sigs.items():
        for da in dd_alarms:
            variants[f"⑩ + {name} @dd{int(da * 100)}"] = early_exit_filter(ndx, s, 200, 0.18, da, 20)
    for vname, w in variants.items():
        w = w.reindex(lev.index).ffill().fillna(0.0)
        rec = {}
        for sname, start in SAMPLES.items():
            l = lev[lev.index >= pd.Timestamp(start)]
            hold = simulate_exposure(l, pd.Series(1.0, index=l.index), costs, cash, "hold")
            r = simulate_exposure(l, w.reindex(l.index), costs, cash, vname)
            rec[sname] = _metrics(r, hold)
            if sname == "C_1986":
                ys = r.equity.groupby(r.equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
                hy = hold.equity.groupby(hold.equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
                rec["crash_mean"] = float(ys[CRASH_YEARS].mean()); rec["whipsaw_mean"] = float(ys[WHIPSAW_YEARS].mean())
                rec["boom_capture"] = float(ys[BOOM_YEARS].mean() / hy[BOOM_YEARS].mean())
                out["yearly"][vname] = {int(k): float(v) for k, v in ys.items()}
        out["strategies"][vname] = rec
    print(f"strategies done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    # 롤링: 기준 + A 표본 비율이 기준 이상인 변형 전부
    base_a = out["strategies"]["⑩ dd18 (기준)"]["A_2010"]["ratio"]
    picks = ["⑩ dd18 (기준)"] + [v for v, rec in out["strategies"].items() if v != "⑩ dd18 (기준)" and rec["A_2010"]["ratio"] >= base_a - 0.02 and rec["crash_mean"] > out["strategies"]["⑩ dd18 (기준)"]["crash_mean"]]
    out["rolling"] = {}
    for v in picks[:8]:
        w = variants[v].reindex(lev.index).ffill().fillna(0.0)
        out["rolling"][v] = {"@1986": _rolling(lev, w, costs, cash, date(1986, 1, 1)), "@1999": _rolling(lev, w, costs, cash, date(1999, 12, 1))}
    out["elapsed"] = round(time.perf_counter() - t0)
    if cfg.get("out"):
        json.dump(out, open(cfg["out"], "w"), ensure_ascii=False, indent=1, default=str)
    if cfg.get("md"):
        open(cfg["md"], "w").write(render(out))
    return out


def render(out: dict) -> str:
    L = [f"# 참사 조기경보 — 신호별 혼동행렬과 ⑩ 조기 이탈 변형 (세후, ~{out['as_of']})", "", "## ⑩ dd18 이탈 에피소드", "",
         "| 이탈 | 재진입 | 구간 ETF 수익 | 라벨 |", "|---|---|---|---|"]
    for e in out["episodes"]:
        L.append(f"| {e['exit']} | {e['reentry'] or '진행중'} | {e['etf_ret']:+.0%} | {e['kind']} |")
    L += ["", "## 신호별 혼동행렬 (⑩ 이탈 대비 선행 영업일; 양수 = 먼저 울림)", "", "| 신호 | 데이터 시작 | 참사 포착 | 참사 평균 선행(일) | 거짓 경보 | 에피소드별 선행 |", "|---|---|---|---|---|---|"]
    for name, c in out["confusion"].items():
        per = ", ".join(f"{r['exit']}({r['kind'][0]}):{r['lead_bdays'] if r['lead_bdays'] is not None else ('—' if r['measurable'] else 'n/a')}" for r in c["rows"])
        lead = f"{c['crash_lead_mean']:+.0f}" if c["crash_lead_mean"] is not None else "—"
        L.append(f"| {name} | {c['coverage_start']} | {c['crash_caught']}/{c['crash_measurable']} | {lead} | {c['n_false']} | {per} |")
    L += ["", "## 세후 성과 — ⑩ 에 조기 이탈 조건 추가 (경보 ON & 낙폭 > dd_alarm & 200일선 아래)", "",
          "| 변형 | A 2010~ 비율 / MDD / 매매 | B 1999~ 배수 / MDD | C 1986~ 배수 / MDD | 참사5년 평균 | 휩소3년 평균 | 폭발 포착 |", "|---|---|---|---|---|---|---|"]
    for v, rec in sorted(out["strategies"].items(), key=lambda kv: -kv[1]["A_2010"]["ratio"]):
        a, b, c = rec["A_2010"], rec["B_1999"], rec["C_1986"]
        L.append(f"| {v} | **{a['ratio']:.2f}** / {a['mdd']:.0%} / {a['trades']} | {b['mult']:.0f}x / {b['mdd']:.0%} | {c['mult']:.0f}x / {c['mdd']:.0%} | {rec['crash_mean']:+.0%} | {rec['whipsaw_mean']:+.0%} | {rec['boom_capture']:.0%} |")
    L += ["", "## 10년 롤링 (A 비율 ≥ 기준 −0.02 이고 참사 평균이 개선된 변형만)", "", "| 변형 | @1986 승률 / 최악 CAGR / 원금손실 | @1999 승률 / 최악 CAGR / 원금손실 |", "|---|---|---|"]
    for v, r in out["rolling"].items():
        a, b = r["@1986"], r["@1999"]
        L.append(f"| {v} | {a['win_rate']:.0%} / {a['worst_cagr']:+.0%} / {a['loss_rate']:.0%} | {b['win_rate']:.0%} / {b['worst_cagr']:+.0%} / {b['loss_rate']:.0%} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
