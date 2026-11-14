"""옵션 탐색 O1 — TQQQ(보유 | H43 패키지) 위에 TQQQ 옵션을 얹으면 TQQQ·H43 을 넘는가. 실제 일별 호가(surge_opt_cli.py fetch).

제약의 옵션 판: 매수 옵션은 프리미엄 전액 지불, 매도 콜은 보유 주식 이하, 매도 풋은 행사가 전액 현금 담보. 노출은 TQQQ 전액 보유를 넘지 않는다.
체결 두 가지: 호가 전부 건넘(fill 1.0, 보수적) · 중간값과 호가 사이 절반(fill 0.5). 세전·배당 미포함, 현금 이자 ^IRX.
실행: docker compose exec backtester python surge_o1_cli.py '{"out": "/app/sweep_results/surge_o1"}'
봉인: '{"end": "2026-11-10", "start": "2024-01-02", "only": [...]}' — 장부에 열람 기록 후(체인도 그때 받는다).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from strategy.adapters.option_overlay import Chain, Leg, Spec, run_overlay
from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame
from surge_l2_cli import roll_beat

SR = Path("/app/sweep_results")


def load_chain(sym: str, start: str, end: str) -> Chain:
    fs = sorted(f for f in (SR / "surge_opt_cache" / sym).glob("*.parquet") if start <= f.stem <= end); parts = []
    for f in fs:
        d = pd.read_parquet(f, columns=["expiration", "type", "strike", "bid", "ask", "delta"])
        if len(d):
            d["date"] = pd.Timestamp(f.stem); parts.append(d)
    return Chain.from_frame(pd.concat(parts, ignore_index=True))


def strategies(h43: pd.Series, fear: pd.Series) -> dict[str, dict]:
    """이름 → Spec 인자. h43 = 전날 기준 H43 비중, fear = 전날 기준 QQQ < 200일선."""
    one = h43 * 0 + 1.0; S: dict[str, dict] = {"TQQQ 보유": {"w": one}, "H43 패키지": {"w": h43}}
    C = lambda d, dte=30, cover=1.0, **k: Leg(call=True, side=-1, delta=d, dte=dte, cover=cover, **k)        # noqa: E731  커버드 콜
    P = lambda d, dte, roll, **k: Leg(call=False, side=1, delta=d, dte=dte, roll_dte=roll, **k)              # noqa: E731  보호 풋
    W = lambda d, dte=30, **k: Leg(call=False, side=-1, delta=d, dte=dte, basis="cash", **k)                 # noqa: E731  현금 담보 풋 매도
    for d in (0.1, 0.2, 0.3):
        S[f"A 보유 + 커버드콜 Δ{d} 30일"] = {"w": one, "legs": [C(d)]}
    S["A 보유 + 커버드콜 Δ0.2 30일 절반만"] = {"w": one, "legs": [C(0.2, cover=0.5)]}
    S["A 보유 + 커버드콜 Δ0.2 7일"] = {"w": one, "legs": [C(0.2, dte=7)]}
    S["A 보유 + 커버드콜 Δ0.2, QQQ 200일선 아래일 때만"] = {"w": one, "legs": [C(0.2, active=fear)]}
    S["A 보유 + 커버드콜 Δ0.2, QQQ 200일선 위일 때만"] = {"w": one, "legs": [C(0.2, active=~fear)]}
    for d in (0.1, 0.2, 0.3):
        S[f"B 보유 + 보호 풋 Δ{d} 90일(30일 남으면 교체)"] = {"w": one, "legs": [P(d, 90, 30)]}
    S["B 보유 + 보호 풋 Δ0.2 180일(60일 남으면 교체)"] = {"w": one, "legs": [P(0.2, 180, 60)]}
    S["B 보유 + 보호 풋 Δ0.2 30일(만기까지)"] = {"w": one, "legs": [P(0.2, 30, 0)]}
    S["C 보유 + 칼라(풋 Δ0.2 90일 + 콜 Δ0.2 30일)"] = {"w": one, "legs": [P(0.2, 90, 30), C(0.2)]}
    for d in (0.2, 0.3, 0.5):
        S[f"D 현금 + 풋 매도 Δ{d} 30일"] = {"w": one * 0, "legs": [W(d)]}
    S["D 현금 + 풋 매도 Δ0.3 7일"] = {"w": one * 0, "legs": [W(0.3, dte=7)]}
    S["D 주식 반 + 나머지 현금으로 풋 매도 Δ0.3 30일"] = {"w": one, "stock_frac": 0.5, "legs": [W(0.3)]}
    for d, dte, roll in ((0.8, 365, 90), (0.7, 180, 60), (0.9, 365, 90), (0.6, 90, 30)):
        S[f"E 현금 + 콜 매수 Δ{d} {dte}일(같은 주식 수)"] = {"w": one * 0, "legs": [Leg(call=True, side=1, delta=d, dte=dte, roll_dte=roll, basis="notional")]}
    S["E 현금 + 콜 매수 Δ0.8 365일(주식 수 1.25 배 = 델타 1)"] = {"w": one * 0, "legs": [Leg(call=True, side=1, delta=0.8, dte=365, roll_dte=90, basis="notional", cover=1.25)]}
    out = h43 < 1.0; full = h43 >= 1.0
    for d in (0.1, 0.2):
        S[f"G H43 + 커버드콜 Δ{d} 30일"] = {"w": h43, "legs": [C(d)]}
    for d in (0.3, 0.5):
        S[f"H H43 + 쉬는 현금으로 풋 매도 Δ{d} 30일"] = {"w": h43, "legs": [W(d, active=out)]}
    S["H H43 + 쉬는 현금으로 풋 매도 Δ0.3 7일"] = {"w": h43, "legs": [W(0.3, dte=7, active=out)]}
    for d in (0.3, 0.5):
        S[f"I 보유 + H43 이 쉬라는 동안만 보호 풋 Δ{d} 60일"] = {"w": one, "legs": [P(d, 60, 20, active=out)]}
    S["J H43 + 쉬는 동안 콜 매수 Δ0.3 90일(자산의 3%)"] = {"w": h43, "legs": [Leg(call=True, side=1, delta=0.3, dte=90, roll_dte=30, basis="budget", cover=0.03, active=out)]}
    S["K H43 + 들고 있을 때 보호 풋 Δ0.1 90일"] = {"w": h43, "legs": [P(0.1, 90, 30, active=full)]}
    S["L H43 쉬는 동안만: 풋 매도 Δ0.3 + 콜 매수 3%"] = {"w": h43, "legs": [W(0.3, active=out), Leg(call=True, side=1, delta=0.3, dte=90, roll_dte=30, basis="budget", cover=0.03, active=out)]}
    return S


def strategies_qqq(h43: pd.Series, fear: pd.Series) -> dict[str, dict]:
    """O2 — 옵션은 QQQ. 콜 수량 n = 자산 ÷ QQQ 주가의 n 배(델타 합 ≤ 3 = TQQQ 전액 보유의 노출 이하)."""
    one = h43 * 0 + 1.0; zero = one * 0; full = h43 >= 1.0; out = h43 < 1.0
    S: dict[str, dict] = {"TQQQ 보유": {"w": one}, "H43 패키지": {"w": h43}}
    N = lambda d, dte, roll, n, **k: Leg(call=True, side=1, delta=d, dte=dte, roll_dte=roll, basis="notional", cover=n, **k)   # noqa: E731
    B = lambda d, dte, roll, b, **k: Leg(call=True, side=1, delta=d, dte=dte, roll_dte=roll, basis="budget", cover=b, **k)     # noqa: E731
    for d, n in ((0.8, 3.0), (0.8, 3.75), (0.7, 3.0), (0.7, 4.0), (0.9, 3.0), (0.6, 4.0), (0.5, 4.0), (0.5, 6.0)):
        S[f"M 현금 + QQQ 콜 Δ{d} 365일 × {n}배(델타 {d * n:.1f})"] = {"w": zero, "legs": [N(d, 365, 90, n)]}
    S["M 현금 + QQQ 콜 Δ0.8 730일 × 3.0배"] = {"w": zero, "legs": [N(0.8, 730, 180, 3.0)]}
    S["M 현금 + QQQ 콜 Δ0.8 180일 × 3.0배"] = {"w": zero, "legs": [N(0.8, 180, 60, 3.0)]}
    S["M 현금 + QQQ 콜 Δ0.8 90일 × 3.0배"] = {"w": zero, "legs": [N(0.8, 90, 30, 3.0)]}
    for b in (0.15, 0.25, 0.35):
        S[f"N 현금 + QQQ 콜 Δ0.5 365일, 자산의 {b:.0%}"] = {"w": zero, "legs": [B(0.5, 365, 90, b)]}
    S["N 현금 + QQQ 콜 Δ0.3 365일, 자산의 15%"] = {"w": zero, "legs": [B(0.3, 365, 90, 0.15)]}
    for d, n in ((0.8, 3.0), (0.8, 3.75), (0.7, 4.0), (0.5, 6.0)):
        S[f"P H43 신호(전량 보유일 때만) QQQ 콜 Δ{d} 365일 × {n}배"] = {"w": zero, "legs": [N(d, 365, 90, n, active=full)]}
    S["P QQQ 200일선 위일 때만 QQQ 콜 Δ0.8 365일 × 3.0배"] = {"w": zero, "legs": [N(0.8, 365, 90, 3.0, active=~fear)]}
    S["Q TQQQ 반 + QQQ 콜 Δ0.8 365일 × 1.5배"] = {"w": one, "stock_frac": 0.5, "legs": [N(0.8, 365, 90, 1.5)]}
    S["Q H43 반 + (전량 보유일 때) QQQ 콜 Δ0.8 365일 × 1.5배"] = {"w": h43, "stock_frac": 0.5, "legs": [N(0.8, 365, 90, 1.5, active=full)]}
    for d in (0.1, 0.2, 0.3):
        S[f"R TQQQ 보유 + QQQ 보호 풋 Δ{d} 90일 × 3배"] = {"w": one, "legs": [Leg(call=False, side=1, delta=d, dte=90, roll_dte=30, cover=3.0)]}
    S["R TQQQ 보유 + H43 이 쉬라는 동안만 QQQ 풋 Δ0.5 60일 × 3배"] = {"w": one, "legs": [Leg(call=False, side=1, delta=0.5, dte=60, roll_dte=20, cover=3.0, active=out)]}
    for b in (0.03, 0.06):
        S[f"T H43 + 쉬는 동안 QQQ 콜 Δ0.3 90일(자산의 {b:.0%})"] = {"w": h43, "legs": [B(0.3, 90, 30, b, active=out)]}
    S["T H43 + 쉬는 동안 QQQ 콜 Δ0.5 180일(자산의 6%)"] = {"w": h43, "legs": [B(0.5, 180, 60, 0.06, active=out)]}
    return S


def strategies_boost(h43: pd.Series, fear: pd.Series) -> dict[str, dict]:
    """O3 — H43 을 그대로 들고, 전량 보유 중일 때 자산의 일부로 콜을 더 산다. ⚠️ 노출이 TQQQ 전액 보유를 넘는다(계획서의 상한 밖, 사람 결정용)."""
    one = h43 * 0 + 1.0; full = h43 >= 1.0; S: dict[str, dict] = {"TQQQ 보유": {"w": one}, "H43 패키지": {"w": h43}}
    B = lambda d, dte, roll, b, **k: Leg(call=True, side=1, delta=d, dte=dte, roll_dte=roll, basis="budget", cover=b, **k)     # noqa: E731
    for d, dte, roll in ((0.3, 90, 30), (0.5, 90, 30), (0.3, 365, 180), (0.5, 365, 180), (0.7, 365, 180)):
        for b in (0.02, 0.05, 0.10):
            S[f"U H43 + 보유 중 콜 Δ{d} {dte}일, 교체 때마다 자산의 {b:.0%}"] = {"w": h43, "legs": [B(d, dte, roll, b, active=full)]}
    S["U TQQQ 보유 + 항상 콜 Δ0.5 365일, 자산의 5%"] = {"w": one, "legs": [B(0.5, 365, 180, 0.05)]}
    S["U H43 + 항상(쉬는 동안 포함) 콜 Δ0.5 365일, 자산의 5%"] = {"w": h43, "legs": [B(0.5, 365, 180, 0.05)]}
    return S


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_o1")); start, end = cfg.get("start", "2010-02-11"), cfg.get("end", "2023-12-29"); mid = pd.Timestamp(cfg.get("mid", "2017-12-29")); only = cfg.get("only")
    sname = cfg.get("set", "tqqq"); qset = sname in ("qqq", "boost_qqq"); ch = load_chain("QQQ" if qset else "TQQQ", start, end); px = pd.read_parquet(SR / "surge_av_cache_long" / "TQQQ.parquet"); px = px[(px.index >= ch.days[0]) & (px.index <= end) & (px["c"] > 0)]
    raw_c, split = px["c"], px["split"].replace(0.0, 1.0); split.iloc[0] = 1.0; days = raw_c.index
    bl = pd.read_parquet(SR / "surge_bench_long.parquet"); bl = bl[bl.index <= end]; qq = bl["QQQ_C"]
    h43 = package_frame(qq, bl["TQQQ_C"])["target"].astype(float).shift(1).reindex(days).fillna(1.0)
    fear = (qq < qq.rolling(200).mean()).shift(1).reindex(days).fillna(False).astype(bool)
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"]
    oq = pd.read_parquet(SR / "surge_av_cache_long" / "QQQ.parquet") if qset else None; okw = {"opt_raw_c": oq["c"], "opt_split": oq["split"].replace(0.0, 1.0)} if qset else {}
    specs = {"tqqq": strategies, "qqq": strategies_qqq, "boost_tqqq": strategies_boost, "boost_qqq": strategies_boost}[sname](h43, fear); fills = cfg.get("fills", [1.0, 0.5]); curves, res = {}, []
    half = lambda e: (float(e[e.index <= mid].iloc[-1] / e.iloc[0]), float(e.iloc[-1] / e[e.index <= mid].iloc[-1]))  # noqa: E731
    for name, kw in specs.items():
        if only and name not in only and name not in ("TQQQ 보유", "H43 패키지"):
            continue
        for f in fills if kw.get("legs") else fills[:1]:
            eq, log = run_overlay(ch, raw_c, split, irx, Spec(fill=f, stock_cost=0.002 if kw["w"] is h43 else 0.001, **kw), **okw); eq = eq / eq.iloc[0]
            curves[f"{name} | {f}"] = eq; s = curve_stats(eq); h = half(eq)
            res.append({"name": name, "fill": f, **s, "h1": h[0], "h2": h[1], "trades": int((log["what"] == "열기").sum())})
            print(f"{name} | fill {f}: ×{s['mult']:.2f} MDD {s['mdd']:+.0%} 옵션 진입 {res[-1]['trades']}", file=sys.stderr, flush=True)
    df = pd.DataFrame(res); bt, bh = curves[f"TQQQ 보유 | {fills[0]}"], curves[f"H43 패키지 | {fills[0]}"]; t, h = df.iloc[0], df.iloc[1]
    df["vs"] = df["mult"] / t["mult"]; df["vs1"] = df["h1"] / t["h1"]; df["vs2"] = df["h2"] / t["h2"]; df["vs_h43"] = df["mult"] / h["mult"]; df["vs_h43_1"] = df["h1"] / h["h1"]; df["vs_h43_2"] = df["h2"] / h["h2"]
    df["roll"] = [roll_beat(curves[f"{r['name']} | {r['fill']}"], bt)[0] for _, r in df.iterrows()]; df["roll_h43"] = [roll_beat(curves[f"{r['name']} | {r['fill']}"], bh)[0] for _, r in df.iterrows()]
    pm = lambda x: f"{x:+.0%}"  # noqa: E731
    L = [f"# {dict(tqqq='O1 — TQQQ 옵션 얹기', qqq='O2 — QQQ 옵션 섞기', boost_tqqq='O3 — H43 + TQQQ 콜 더 사기', boost_qqq='O3 — H43 + QQQ 콜 더 사기')[sname]} ({days[0].date()}~{days[-1].date()})", "",
         "실제 일별 매수·매도 호가(Alpha Vantage HISTORICAL_OPTIONS). 체결 1.0 = 호가 전부 건넘, 0.5 = 중간값과 호가 사이 절반. 세전·배당 미포함, 현금 이자 ^IRX.",
         f"전반 = ~{mid.date()}, 후반 = 그 뒤. 대비 열은 같은 구간 배수의 비율(1 보다 크면 이김).", "",
         "| 방법 | 체결 | 배수 | CAGR | MDD | TQQQ 대비 | 전/후반 | H43 대비 | 전/후반 | 1년 창 TQQQ 이긴% | 1년 창 H43 이긴% | 옵션 진입 수 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['name']} | {r['fill']} | ×{r['mult']:.1f} | {pm(r['cagr'])} | {pm(r['mdd'])} | {r['vs']:.2f} | {r['vs1']:.2f} / {r['vs2']:.2f} | {r['vs_h43']:.2f} | {r['vs_h43_1']:.2f} / {r['vs_h43_2']:.2f} | {r['roll']:.0%} | {r['roll_h43']:.0%} | {r['trades']} |")
    new = df.iloc[2:]; c1 = new[new["fill"] == fills[0]]
    L += ["", f"보수적 체결({fills[0]}) 기준 TQQQ 를 넘은 방법 {int((c1['vs'] > 1).sum())} / {len(c1)} · H43 도 넘은 것 {int((c1['vs_h43'] > 1).sum())} / {len(c1)} · 전·후반 모두 H43 이상 {int(((c1['vs_h43_1'] >= 1) & (c1['vs_h43_2'] >= 1)).sum())} / {len(c1)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); pd.DataFrame(curves).to_parquet(f"{out}_curves.parquet"); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
