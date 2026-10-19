"""레버리지 지수 상대 모멘텀 로테이션 × ⑩ — "매달 가장 강한 지수의 3배" (로드맵 기둥 1, 카드 C4).

    docker compose exec backtester python rotation_cli.py '{"out": "/app/sweep_results/rot.json", "md": "/app/sweep_results/rot.md"}'

유니버스(1x 지수 → ^IRX 조달 + 드래그 2.51% 합성 3x): NDX(실제 TQQQ 스플라이스) · SPX · MID(S&P400) · RUT(1987~) · XAU(금광).
월말에 룩백 수익률(1x)로 순위 → 상위 k 를 다음 달 보유. 순위가 안 바뀌면 거래 없음(세금). 절대 모멘텀/⑩ 로 현금 전환.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.band_rebalance_strategy import CALIBRATED_DRAG, build_synthetic_leveraged
from strategy.adapters.leveraged_etf_data import _close
from strategy.adapters.leveraged_etf_lab import CostParams, deep_correction_filter
from strategy.adapters.mix_lab import simulate_mix
from vol_target_regime_cli import BOOM_YEARS, CRASH_YEARS, load

SAMPLES = {"A_2010": date(2010, 2, 11), "B_1999": date(1999, 12, 1), "C_1986": date(1986, 1, 2)}
IDX = {"SPX": "^GSPC", "MID": "^MID", "RUT": "^RUT", "XAU": "^XAU"}


def load_universe(end: date):
    lev, ndx, cash = load(end); md = build_default_market_data()
    one = {"NDX": ndx}; three = {"NDX": lev}
    for k, sym in IDX.items():
        c = _close(md.fetch_ohlcv(sym, date(1985, 1, 1), end)).reindex(lev.index).ffill()
        one[k] = c
        three[k] = build_synthetic_leveraged(c.dropna(), 3.0, annual_expense=CALIBRATED_DRAG[3.0], financing_rate=cash).reindex(lev.index)
    return pd.DataFrame(one), pd.DataFrame(three), ndx, cash


def momentum(one: pd.DataFrame, lookback_m: int, skip_m: int = 1) -> pd.DataFrame:
    """월말 기준 (t−skip) / (t−lookback) − 1. 일별 인덱스로 ffill."""
    me = one.groupby(one.index.to_period("M")).last()
    mom = me.shift(skip_m) / me.shift(lookback_m) - 1
    mom.index = one.groupby(one.index.to_period("M")).apply(lambda x: x.index[-1]).values
    return mom.reindex(one.index).ffill()


def build_targets(one, three, ndx, cash, universe, lookback, top, switch, abs_mom):
    idx = three.index
    mom = momentum(one[universe], lookback)
    me_days = pd.Series(False, index=idx); me_days.loc[one.groupby(one.index.to_period("M")).apply(lambda x: x.index[-1]).values] = True
    tbill = (cash.reindex(idx).ffill().fillna(0.0))  # 연환산
    sw = deep_correction_filter(ndx, 200, 0.18, 20).reindex(idx).ffill().fillna(0.0) if switch == "ndx" else None
    t = pd.DataFrame(0.0, index=idx, columns=three.columns); trade = pd.Series(False, index=idx)
    cur = []
    sw_prev = 1.0
    for d in idx:
        if me_days[d]:
            m = mom.loc[d].dropna()
            if abs_mom:
                m = m[m > tbill[d] * lookback / 12]
            new = list(m.sort_values(ascending=False).index[:top])
            if new != cur:
                cur = new; trade[d] = True
        w = (1.0 / top) if cur else 0.0
        for a in cur: t.at[d, a] = w
        if sw is not None:
            s = float(sw[d])
            if s != sw_prev: trade[d] = True; sw_prev = s
            t.loc[d] *= s
    return t, trade


def main(cfg):
    end = date.fromisoformat(cfg.get("as_of", "2026-10-16")); t0 = time.perf_counter()
    one, three, ndx, cash = load_universe(end)
    costs = CostParams(tax_rate=float(cfg.get("tax", 0.22)), exec_lag=1, rebalance_band=0.05)
    grid = []
    for uni_name, uni in (("NDX+SPX", ["NDX", "SPX"]), ("NDX+SPX+MID", ["NDX", "SPX", "MID"]), ("NDX+SPX+MID+RUT", ["NDX", "SPX", "MID", "RUT"]), ("4지수+XAU", ["NDX", "SPX", "MID", "RUT", "XAU"])):
        for lb in (3, 6, 12):
            for top in (1, 2):
                if top == 2 and len(uni) < 3: continue
                for switch in ("none", "ndx"):
                    for abs_mom in (False, True):
                        if switch == "none" and not abs_mom: continue  # 현금 전환이 전혀 없는 건 제외
                        grid.append((f"{uni_name} | {lb}m top{top} | " + ("⑩" if switch == "ndx" else "") + ("+절대" if abs_mom else ""), uni, lb, top, switch, abs_mom))
    hold_t = pd.DataFrame(0.0, index=three.index, columns=three.columns); hold_t["NDX"] = 1.0
    hold_td = pd.Series(True, index=three.index)
    holds = {s: simulate_mix(three[["NDX"]][three.index >= pd.Timestamp(st)], hold_t[["NDX"]], hold_td, costs, cash, "hold") for s, st in SAMPLES.items()}
    out = {"as_of": end.isoformat(), "portfolios": {}, "rolling": {}, "yearly": {}}
    base_t = pd.DataFrame(0.0, index=three.index, columns=three.columns); base_t["NDX"] = deep_correction_filter(ndx, 200, 0.18, 20).reindex(three.index).ffill().fillna(0.0)
    base_td = (base_t["NDX"].diff().fillna(0) != 0)
    grid.insert(0, ("⑩ TQQQ 100% (기준)", None, None, None, None, None))
    targets = {}
    for name, uni, lb, top, switch, abs_mom in grid:
        if uni is None: t, td = base_t, base_td
        else: t, td = build_targets(one, three, ndx, cash, uni, lb, top, switch, abs_mom)
        targets[name] = (t, td); rec = {}
        for s, st in SAMPLES.items():
            cols = [c for c in t.columns if (t[c].abs() > 0).any()]
            p = three[cols][three.index >= pd.Timestamp(st)].dropna()
            r = simulate_mix(p, t[cols], td, costs, cash, name); h = holds[s]
            yrs = len(r.equity) / 252; m = r.final_after_liquidation / r.equity.iloc[0]; hm = h.final_after_liquidation / h.equity.iloc[0]
            rec[s] = {"mult": m, "cagr": m ** (1 / yrs) - 1, "ratio": m / hm, "mdd": r.metrics["mdd"], "trades": r.trades, "tax": r.tax}
            if s == "C_1986":
                ys = r.equity.groupby(r.equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
                out["yearly"][name] = {int(k): float(v) for k, v in ys.items()}
                rec["crash_mean"] = float(ys[CRASH_YEARS].mean()); rec["boom_mean"] = float(ys[BOOM_YEARS].mean())
        out["portfolios"][name] = rec
    print(f"{len(grid)} portfolios done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    picks = [n for n, r in out["portfolios"].items() if r["A_2010"]["ratio"] >= 0.85 or n.startswith("⑩ TQQQ")]
    for n in picks:
        t, td = targets[n]; cols = [c for c in t.columns if (t[c].abs() > 0).any()]; res = {}
        for first in (date(1986, 1, 1), date(1999, 12, 1)):
            rows = []
            for s in pd.date_range(first, three.index[-1] - pd.DateOffset(years=10), freq="3MS"):
                e = s + pd.DateOffset(years=10)
                if e > three.index[-1]: break
                p = three[cols][(three.index >= s) & (three.index <= e)].dropna()
                h = simulate_mix(p[["NDX"]], hold_t[["NDX"]], hold_td, costs, cash, "h"); r = simulate_mix(p, t[cols], td, costs, cash, "s")
                hm = h.final_after_liquidation / h.equity.iloc[0]; sm = r.final_after_liquidation / r.equity.iloc[0]
                rows.append({"ratio": sm / hm, "cagr": sm ** 0.1 - 1, "mult": sm})
            df = pd.DataFrame(rows)
            res[f"@{first.year}"] = {"windows": len(df), "win_rate": float((df.ratio > 1).mean()), "worst_cagr": float(df.cagr.min()), "loss_rate": float((df.mult < 1).mean())}
        out["rolling"][n] = res
    print(f"rolling done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    hb = holds["B_1999"].final_after_liquidation / holds["B_1999"].equity.iloc[0]
    out["verdict"] = {}
    for n, r in out["portfolios"].items():
        a, b = r["A_2010"], r["B_1999"]; ro = out["rolling"].get(n, {})
        crit = {"A_ret": a["ratio"] >= 0.90, "A_mdd": a["mdd"] >= -0.60, "B_ret": b["mult"] >= 10 * hb, "B_mdd": b["mdd"] >= -0.65}
        for k, v in ro.items(): crit[f"C_win{k}"] = v["win_rate"] >= 0.70; crit[f"C_worst{k}"] = v["worst_cagr"] >= 0
        out["verdict"][n] = {"pass": sum(crit.values()), "of": len(crit)}
    if cfg.get("out"): json.dump(out, open(cfg["out"], "w"), ensure_ascii=False, indent=1, default=str)
    if cfg.get("md"): open(cfg["md"], "w").write(render(out))
    return out


def render(out):
    L = [f"# 레버리지 지수 상대 모멘텀 로테이션 × ⑩ (세후 22%, lag 1, ~{out['as_of']})", "",
         "| 포트폴리오 | A 2010~ 비율 / MDD / 매매 | B 1999~ 배수 / MDD | C 1986~ 배수 / MDD | 참사5년 | 폭발15년 | 통과 | 롤링 @1986 승률/최악 | @1999 승률/최악 |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(out["portfolios"].items(), key=lambda kv: -(out["verdict"][kv[0]]["pass"] * 10 + kv[1]["A_2010"]["ratio"])):
        a, b, c = r["A_2010"], r["B_1999"], r["C_1986"]; v = out["verdict"][n]; ro = out["rolling"].get(n, {})
        r86 = f"{ro['@1986']['win_rate']:.0%} / {ro['@1986']['worst_cagr']:+.0%}" if "@1986" in ro else "—"
        r99 = f"{ro['@1999']['win_rate']:.0%} / {ro['@1999']['worst_cagr']:+.0%}" if "@1999" in ro else "—"
        L.append(f"| {n} | **{a['ratio']:.2f}** / {a['mdd']:.0%} / {a['trades']} | {b['mult']:.0f}x / **{b['mdd']:.0%}** | {c['mult']:.0f}x / {c['mdd']:.0%} | {r['crash_mean']:+.0%} | {r['boom_mean']:+.0%} | {v['pass']}/{v['of']} | {r86} | {r99} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
