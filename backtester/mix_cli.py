"""프런티어 옮기기 — TQQQ(⑩) + 동승자(장기국채 1x/3x·금) 혼합의 세후 성과, 여섯 기준 판정.

    docker compose exec backtester python mix_cli.py '{"out": "/app/sweep_results/mix.json", "md": "/app/sweep_results/mix.md"}'
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.fred_adapter import FredCsvAdapter
from strategy.adapters.band_rebalance_strategy import splice_series
from strategy.adapters.leveraged_etf_data import _close
from strategy.adapters.leveraged_etf_lab import CostParams, deep_correction_filter
from strategy.adapters.mix_lab import calendar_days, simulate_mix, synthetic_long_bond
from vol_target_regime_cli import BOOM_YEARS, CRASH_YEARS, load

SAMPLES = {"A_2010": date(2010, 2, 11), "B_1999": date(1999, 12, 1), "C_1986": date(1986, 1, 2)}


def load_assets(end: date):
    lev, ndx, cash = load(end)
    md = build_default_market_data(); f = FredCsvAdapter()
    # DGS20 은 1987~1993 미발행, DGS30 은 2002~2006 미발행 → 소스별 듀레이션으로 일간 수익을 만들고 우선순위로 이어붙인다
    rets = []
    for sid, dur, conv in (("DGS20", 12.0, 180.0), ("DGS30", 15.0, 280.0), ("DGS10", 8.0, 80.0)):
        y = f.fetch_series(sid, date(1985, 1, 1), end); y.index = pd.DatetimeIndex(y.index)
        rets.append(synthetic_long_bond(y, dur, conv).pct_change())
    r = rets[0].combine_first(rets[1]).combine_first(rets[2]).reindex(lev.index).fillna(0.0)
    syn = (1 + r).cumprod() * 100.0
    vustx = _close(md.fetch_ohlcv("VUSTX", date(1999, 11, 1), end))
    ltb1 = splice_series(syn.dropna(), vustx).reindex(lev.index).ffill()
    r1 = ltb1.pct_change().fillna(0.0); c = cash.reindex(lev.index).ffill().fillna(0.0) / 252
    ltb3 = (1 + 3 * r1 - 2 * c - 0.0105 / 252).cumprod() * 100.0
    gold = _close(md.fetch_ohlcv("GC=F", date(2000, 8, 30), end)).reindex(lev.index).ffill()
    prices = pd.DataFrame({"TQQQ": lev, "LTB1": ltb1, "LTB3": ltb3, "GOLD": gold})
    return prices, ndx, cash


def build_portfolios(prices: pd.DataFrame, ndx: pd.Series) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    idx = prices.index
    sw = deep_correction_filter(ndx, 200, 0.18, 20).reindex(idx).ffill().fillna(0.0)
    sw_change = sw.diff().fillna(0.0) != 0
    ports = {}
    def add(name, w_t, div=None, w_div=0.0, off_to="cash", switch=True, freq="quarterly", gold=0.0):
        s = sw if switch else pd.Series(1.0, index=idx)
        t = pd.DataFrame(0.0, index=idx, columns=prices.columns)
        t["TQQQ"] = w_t * s
        if div: t[div] = w_div
        if off_to != "cash":
            t[off_to] = t[off_to] + w_t * (1 - s)
        if gold: t["GOLD"] = gold
        td = calendar_days(idx, freq) | (sw_change if switch else False)
        ports[name] = (t, td)
    add("⑩ TQQQ 100% (기준)", 1.0)
    for off in ("LTB1", "LTB3"):
        add(f"⑩ 100%, 이탈 시 {off}", 1.0, off_to=off)
    for w in (0.9, 0.8, 0.7):
        for div in ("LTB1", "LTB3"):
            for freq in ("quarterly", "annual"):
                add(f"⑩ {int(w*100)}% + {div} {int((1-w)*100)}% ({freq[0].upper()})", w, div, 1 - w, "cash", True, freq)
                add(f"⑩ {int(w*100)}% + {div} {int((1-w)*100)}% ({freq[0].upper()}), 이탈 시 {div}", w, div, 1 - w, div, True, freq)
    for w in (0.55, 0.7):
        add(f"보유 {int(w*100)}% + LTB3 {int((1-w)*100)}% (Q, 스위치 없음)", w, "LTB3", 1 - w, "cash", False, "quarterly")
    add("⑩ 80% + LTB1 10% + GOLD 10% (Q)", 0.8, "LTB1", 0.1, "cash", True, "quarterly", gold=0.1)
    add("⑩ 70% + LTB1 15% + GOLD 15% (Q)", 0.7, "LTB1", 0.15, "cash", True, "quarterly", gold=0.15)
    add("⑩ 70% + LTB3 15% + GOLD 15% (Q), 이탈 시 LTB1", 0.7, "LTB3", 0.15, "LTB1", True, "quarterly", gold=0.15)
    return ports


def run_sample(prices, targets, td, costs, cash, start, end=None, name=""):
    p = prices[prices.index >= pd.Timestamp(start)]
    if end is not None: p = p[p.index <= pd.Timestamp(end)]
    cols = [c for c in p.columns if (targets[c].abs() > 0).any()]
    p = p[cols].dropna()
    return simulate_mix(p, targets[cols], td, costs, cash, name)


def main(cfg: dict) -> dict:
    end = date.fromisoformat(cfg.get("as_of", "2026-10-16")); t0 = time.perf_counter()
    prices, ndx, cash = load_assets(end)
    ports = build_portfolios(prices, ndx)
    costs = CostParams(tax_rate=float(cfg.get("tax", 0.22)), exec_lag=1, rebalance_band=0.05)
    hold_t = pd.DataFrame(0.0, index=prices.index, columns=prices.columns); hold_t["TQQQ"] = 1.0
    hold_td = pd.Series(True, index=prices.index)  # 단일 자산 100% 는 밴드 안이라 실제 거래는 첫날뿐
    out = {"as_of": end.isoformat(), "portfolios": {}, "rolling": {}, "yearly": {}}
    holds = {s: run_sample(prices, hold_t, hold_td, costs, cash, st, name="hold") for s, st in SAMPLES.items()}
    for name, (t, td) in ports.items():
        rec = {}
        uses_gold = (t["GOLD"].abs() > 0).any()
        for s, st in SAMPLES.items():
            if uses_gold and s == "C_1986": rec[s] = None; continue
            r = run_sample(prices, t, td, costs, cash, st, name=name); h = holds[s]
            yrs = len(r.equity) / 252; m = r.final_after_liquidation / r.equity.iloc[0]; hm = h.final_after_liquidation / h.equity.iloc[0]
            rec[s] = {"mult": m, "cagr": m ** (1 / yrs) - 1, "ratio": m / hm, "mdd": r.metrics["mdd"], "trades": r.trades, "tax": r.tax}
            if s == ("B_1999" if uses_gold else "C_1986"):
                ys = r.equity.groupby(r.equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
                out["yearly"][name] = {int(k): float(v) for k, v in ys.items()}
        out["portfolios"][name] = rec
    print(f"portfolios done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    # 롤링: A ≥ 0.85 인 것 전부 (+기준)
    picks = [n for n, r in out["portfolios"].items() if r["A_2010"]["ratio"] >= 0.85]
    for n in picks:
        t, td = ports[n]; res = {}
        for first in (date(1986, 1, 1), date(1999, 12, 1)):
            if (t["GOLD"].abs() > 0).any() and first.year == 1986: continue
            rows = []
            for s in pd.date_range(first, prices.index[-1] - pd.DateOffset(years=10), freq="3MS"):
                e = s + pd.DateOffset(years=10)
                if e > prices.index[-1]: break
                h = run_sample(prices, hold_t, hold_td, costs, cash, s, e); r = run_sample(prices, t, td, costs, cash, s, e)
                hm = h.final_after_liquidation / h.equity.iloc[0]; sm = r.final_after_liquidation / r.equity.iloc[0]
                rows.append({"ratio": sm / hm, "cagr": sm ** 0.1 - 1, "mult": sm})
            df = pd.DataFrame(rows)
            res[f"@{first.year}"] = {"windows": len(df), "win_rate": float((df.ratio > 1).mean()), "worst_cagr": float(df.cagr.min()), "loss_rate": float((df.mult < 1).mean())}
        out["rolling"][n] = res
        print(f"rolling {n} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    # 여섯 기준 판정
    out["verdict"] = {}
    for n, r in out["portfolios"].items():
        a, b = r["A_2010"], r["B_1999"]; ro = out["rolling"].get(n, {})
        crit = {"A_ret": a["ratio"] >= 0.90, "A_mdd": a["mdd"] >= -0.60, "B_ret": b["mult"] >= 10 * (holds["B_1999"].final_after_liquidation / holds["B_1999"].equity.iloc[0]), "B_mdd": b["mdd"] >= -0.65}
        for k, v in ro.items():
            crit[f"C_win{k}"] = v["win_rate"] >= 0.70; crit[f"C_worst{k}"] = v["worst_cagr"] >= 0
        out["verdict"][n] = {"pass": sum(crit.values()), "of": len(crit), "crit": crit}
    if cfg.get("out"): json.dump(out, open(cfg["out"], "w"), ensure_ascii=False, indent=1, default=str)
    if cfg.get("md"): open(cfg["md"], "w").write(render(out))
    return out


def render(out):
    L = [f"# 프런티어 옮기기 — TQQQ(⑩) + 동승자 혼합 (세후 22%, lag 1, ~{out['as_of']})", "",
         "| 포트폴리오 | A 2010~ 비율 / MDD / 매매 | B 1999~ 배수 / MDD | C 1986~ 배수 / MDD | 통과 | 롤링 @1986 승률/최악 | @1999 승률/최악 |", "|---|---|---|---|---|---|---|"]
    for n, r in sorted(out["portfolios"].items(), key=lambda kv: -(out["verdict"][kv[0]]["pass"] * 10 + kv[1]["A_2010"]["ratio"])):
        a, b, c = r["A_2010"], r["B_1999"], r["C_1986"]; v = out["verdict"][n]; ro = out["rolling"].get(n, {})
        cs = f"{c['mult']:.0f}x / {c['mdd']:.0%}" if c else "—"
        r86 = f"{ro['@1986']['win_rate']:.0%} / {ro['@1986']['worst_cagr']:+.0%}" if "@1986" in ro else "—"
        r99 = f"{ro['@1999']['win_rate']:.0%} / {ro['@1999']['worst_cagr']:+.0%}" if "@1999" in ro else "—"
        L.append(f"| {n} | **{a['ratio']:.2f}** / {a['mdd']:.0%} / {a['trades']} | {b['mult']:.0f}x / **{b['mdd']:.0%}** | {cs} | {v['pass']}/{v['of']} | {r86} | {r99} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
