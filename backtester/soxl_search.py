"""고변동 레버리지 ETF(SOXL)용 레짐 규칙 탐색 — 신호 지수(SOXX·QQQ·SPY) × 월말/주봉 SMA·Faber·깊은 낙폭·크로스·잔여비중 격자.

    docker compose exec backtester python soxl_search.py '{"etf": "SOXL", "out": "/tmp/soxl_search.json"}'

평가: 세후(토스 0.1% + 양도세 22%) 2010~ 실데이터, 2002~ 합성 포함, 하위 구간(2010~17 / 2018~26), 상위 규칙은 5·10년 롤링 승률.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.deep_correction_lab import rolling_study, rolling_summary
from strategy.adapters.leveraged_etf_data import load_lab_prices
from strategy.adapters.leveraged_etf_lab import (
    CostParams,
    deep_correction_filter,
    month_end_only,
    simulate_exposure,
    sma_filter,
    vol_target,
)
from regime_search import cross, faber, sma_asym


def grid_for(c: pd.Series, tag: str) -> dict[str, pd.Series]:
    g: dict[str, pd.Series] = {}
    for n in (50, 75, 100, 125, 150, 200):
        g[f"{tag}:sma{n}_m"] = month_end_only(sma_filter(c, n))
        g[f"{tag}:sma{n}_w"] = sma_filter(c, n, weekly=True)
    for m in (3, 4, 5, 6, 8, 10):
        g[f"{tag}:faber{m}"] = faber(c, m)
    for n in (100, 150, 200):
        for dd in (0.20, 0.25, 0.30, 0.35, 0.40):
            for re in (20, 50, 100):
                g[f"{tag}:dd{n}_{int(dd*100)}_re{re}"] = deep_correction_filter(c, n, dd, re)
            g[f"{tag}:dd{n}_{int(dd*100)}_re50_m"] = month_end_only(deep_correction_filter(c, n, dd, 50))
    for f, s in ((50, 200), (50, 150), (20, 100), (30, 150)):
        g[f"{tag}:cross{f}/{s}"] = cross(c, f, s)
    for a, b in ((200, 50), (150, 50), (100, 20)):
        g[f"{tag}:asym{a}/{b}"] = sma_asym(c, a, b)
    return g


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    etf = cfg.get("etf", "SOXL")
    md = build_default_market_data()
    early = date.fromisoformat(cfg.get("early_start", "2002-06-01"))
    late = date.fromisoformat(cfg.get("late_start", "2010-04-01"))
    end = date.today()
    pr = load_lab_prices(md, etf, "SOXX", early, end, synthetic=True)
    lev, cash = pr.lev_close, pr.cash_rate
    closes = {"SOXX": pr.signal_close}
    for sym in cfg.get("extra_signals", ["QQQ", "SPY"]):
        df = md.fetch_ohlcv(sym, early - pd.Timedelta(days=400), end)
        c = df["Close"].astype(float)
        if c.index.tz is not None:
            c.index = c.index.tz_localize(None)
        closes[sym] = c[~c.index.normalize().duplicated()].set_axis(c.index.normalize()[~c.index.normalize().duplicated()])
    sigs: dict[str, pd.Series] = {}
    for tag, c in closes.items():
        sigs.update(grid_for(c, tag))
    # 잔여 비중·변동성 타기팅 결합 변형 (상위 후보군에만)
    vt = vol_target(lev, 0.60, 20, 1.0)
    for k in list(sigs):
        if k.endswith(("_m", "faber6", "faber4")) and ("sma100" in k or "sma75" in k or "sma125" in k or "faber" in k):
            sigs[k + "_off33"] = sigs[k].clip(lower=0.33)
            sigs[k + "*vt60"] = sigs[k].reindex(lev.index).ffill().fillna(0.0) * vt
    print(f"[soxl] rules={len(sigs)}", file=sys.stderr)
    costs = CostParams(rebalance_band=0.02)
    notax = replace(costs, tax_rate=0.0)
    periods = {"2010~": (late, None), "2002~": (early, None), "2010~17": (late, date(2017, 12, 31)), "2018~": (date(2018, 1, 1), None)}

    def mult(s, a, b, cst):
        l = lev[lev.index >= pd.Timestamp(a)]
        if b is not None:
            l = l[l.index <= pd.Timestamp(b)]
        r = simulate_exposure(l, s.reindex(l.index).ffill().fillna(0.0), cst, cash)
        return r.equity.iloc[-1] / r.equity.iloc[0], r.metrics["mdd"], r.trades

    hold = {k: mult(pd.Series(1.0, index=lev.index), a, b, costs) for k, (a, b) in periods.items()}
    rows = []
    for name, s in sigs.items():
        row = {"rule": name}
        for k, (a, b) in periods.items():
            m, mdd, tr = mult(s, a, b, costs)
            row[f"r_{k}"] = m / hold[k][0]; row[f"mdd_{k}"] = mdd; row[f"tr_{k}"] = tr
        row["r_2010~_notax"] = mult(s, late, None, notax)[0] / hold["2010~"][0]
        rs = [row["r_2010~"], row["r_2002~"], row["r_2010~17"], row["r_2018~"]]
        row["worst"] = min(rs)
        row["score"] = (rs[0] * rs[1] * rs[2] * rs[3]) ** 0.25          # 네 구간 기하평균
        row["active"] = row["tr_2010~"] > 1                               # 2010~ 에서 한 번도 안 움직인 규칙(=장기보유) 제외
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df["active"]].sort_values("score", ascending=False)
    print(f"\n{etf} 장기보유: " + " ".join(f"{k} {v[0]:.0f}x(MDD {v[1]:+.0%})" for k, v in hold.items()))
    print(f"{'rule':34} {'기하평균':>7} {'최악':>5} | {'2010~':>6} {'MDD':>5} {'매매':>4} | {'2002~':>6} | {'10~17':>6} | {'18~':>6} | {'세전2010':>8}")
    for _, r in df.head(30).iterrows():
        print(f"{r['rule']:34} {r['score']:7.2f} {r['worst']:5.2f} | {r['r_2010~']:6.2f} {r['mdd_2010~']:+5.0%} {int(r['tr_2010~']):4d} | {r['r_2002~']:6.2f} | "
              f"{r['r_2010~17']:6.2f} | {r['r_2018~']:6.2f} | {r['r_2010~_notax']:8.2f}")
    print("\n2010~ 세후 비율 상위 10 (활성 규칙)")
    for _, r in df.sort_values("r_2010~", ascending=False).head(10).iterrows():
        print(f"{r['rule']:34} 2010~ {r['r_2010~']:5.2f} MDD {r['mdd_2010~']:+5.0%} 매매 {int(r['tr_2010~']):3d} | 2002~ {r['r_2002~']:5.2f} | 10~17 {r['r_2010~17']:5.2f} | 18~ {r['r_2018~']:5.2f} | 세전 {r['r_2010~_notax']:5.2f}")
    # 상위 8 규칙 롤링 5·10년 승률
    print("\n상위 규칙 롤링 (세후, 시작월 3개월 간격)")
    roll = {}
    for _, r in df.head(8).iterrows():
        s = sigs[r["rule"]]
        res = {}
        for y in (5, 10):
            d = rolling_study(lev, s, y, costs, cash, step_months=3)
            sm = rolling_summary(d); res[y] = sm
        roll[r["rule"]] = res
        print(f"{r['rule']:34} 5년 승률 {res[5]['win_rate']:.0%} 중앙 {res[5]['ratio_median']:.2f} 최악 {res[5]['ratio_min']:.2f} | "
              f"10년 승률 {res[10]['win_rate']:.0%} 중앙 {res[10]['ratio_median']:.2f} 최악 {res[10]['ratio_min']:.2f} | ⑩ 최악 CAGR {res[10]['strat_cagr_min']:+.0%} vs 보유 {res[10]['hold_cagr_min']:+.0%}")
    json.dump({"etf": etf, "hold": {k: v[0] for k, v in hold.items()}, "table": df.to_dict(orient="records"), "rolling": roll},
              open(cfg.get("out", "/tmp/soxl_search.json"), "w"), ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
