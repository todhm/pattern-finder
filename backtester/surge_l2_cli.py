"""2차 탐색 L2·L3 — H43 패키지(TQQQ 스위치·사다리)와 주도주 계단식 묶음(S5)의 결합 vs TQQQ.

 L2  a × (TQQQ 보유 | H43) + (1−a) × 묶음, 63 거래일마다 비중 되돌림
 L3  H43 이 현금인 몫을 묶음에 넣는다(H43 비중 w → TQQQ w, 묶음 1−w)
세전·배당 미포함(1차 탐색과 같은 자). H43 비중 변경 비용 0.2%, 혼합 재조정 비용 0.25%(바뀐 몫).
실행: docker compose exec backtester python surge_l2_cli.py '{"out": "/app/sweep_results/surge_l2"}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")


def mix_curve(ra: pd.Series, rb: pd.Series, a: float, every: int = 63, cost: float = 0.0025) -> pd.Series:
    """두 일별 수익률을 a : 1−a 로 섞고 every 거래일마다 비중을 되돌린다."""
    va, vb = a, 1.0 - a; out = np.empty(len(ra))
    for i, (x, y) in enumerate(zip(ra.to_numpy(), rb.to_numpy())):
        va *= 1 + x; vb *= 1 + y
        if i and i % every == 0:
            tot = va + vb; tot -= cost * abs(va - a * tot) * 2; va, vb = a * tot, (1 - a) * tot
        out[i] = va + vb
    return pd.Series(out, index=ra.index)


def roll_beat(eq: pd.Series, bench: pd.Series, win: int = 252, step: int = 21) -> tuple[float, float]:
    """win 거래일 창을 step 씩 밀며 벤치를 이긴 비율, 창 수익 비율의 중앙."""
    i = np.arange(0, len(eq) - win, step); a = eq.to_numpy()[i + win] / eq.to_numpy()[i]; b = bench.to_numpy()[i + win] / bench.to_numpy()[i]
    return float((a > b).mean()), float(np.median(a / b))


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_l2")); end = cfg.get("end", "2023-12-31"); mid = pd.Timestamp(cfg.get("mid", "2020-12-31"))
    curves = pd.read_parquet(cfg.get("curves", SR / "surge_s5_curves.parquet")); pools = cfg.get("pools", ["0.51_20", "0.53_10", "0.51_10"]); days = curves.index
    tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet")["C"]; qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")["C"]
    tq, qq = tq[tq.index <= end], qq[qq.index <= end]
    w = package_frame(qq, tq)["target"].astype(float)
    r_t = tq.pct_change().reindex(days).fillna(0.0); w_prev = w.shift(1).reindex(days).fillna(1.0)
    r_h = w_prev * r_t - 0.002 * w.diff().abs().shift(1).reindex(days).fillna(0.0)
    r_q = qq.pct_change().reindex(days).fillna(0.0)
    r_t.iloc[0] = r_h.iloc[0] = r_q.iloc[0] = 0.0                      # 첫날 = 1 로 맞춘다
    r = {"TQQQ": r_t, "H43": r_h}; r_pool = {k: curves[k].pct_change().fillna(0.0) for k in pools}
    eq = {"TQQQ 보유": (1 + r_t).cumprod(), "H43 패키지": (1 + r_h).cumprod()}
    for base in ("TQQQ", "H43"):                                      # 대조: 묶음 자리에 QQQ·현금
        for a in (0.25, 0.5, 0.75):
            eq[f"대조 {base} {a:.0%} + QQQ {1 - a:.0%}"] = mix_curve(r[base], r_q, a)
        eq[f"대조 {base} 50% + 현금 50%"] = mix_curve(r[base], r_q * 0.0, 0.5)
    for k in pools:
        eq[f"묶음 {k}"] = curves[k] / curves[k].iloc[0]
        for base in ("TQQQ", "H43"):
            for a in (0.25, 0.5, 0.75):
                eq[f"L2 {base} {a:.0%} + 묶음 {k} {1 - a:.0%}"] = mix_curve(r[base], r_pool[k], a)
        eq[f"L3 H43 현금 몫 → 묶음 {k}"] = (1 + r_h + (1 - w_prev) * r_pool[k]).cumprod()
        eq[f"L3 H43 현금 몫 절반 → 묶음 {k}"] = (1 + r_h + 0.5 * (1 - w_prev) * r_pool[k]).cumprod()
    bt, bh = eq["TQQQ 보유"], eq["H43 패키지"]; st = curve_stats(bt)
    half = lambda e: (float(e[e.index <= mid].iloc[-1] / e.iloc[0]), float(e.iloc[-1] / e[e.index <= mid].iloc[-1]))  # noqa: E731
    ht = half(bt); pm = lambda x: f"{x:+.0%}"  # noqa: E731
    L = [f"# L2·L3 — H43 패키지 × 주도주 묶음 ({days[0].date()}~{days[-1].date()})", "",
         "묶음 = S5 계단식(소거 4 개·12-1 모멘텀 상위 K·63일 교체 9 분할), 이름은 오른 날 문턱_K. 세전·배당 미포함. H43 현금 비중 평균 " + f"{(1 - w_prev).mean():.0%}.", "",
         "| 방법 | 배수 | CAGR | MDD | 전반 | 후반 | TQQQ 대비 | 전/후반 대비 | H43 대비 | 1년 창 TQQQ 이긴% | 창 비율 중앙 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    res = []
    for name, e in eq.items():
        s = curve_stats(e); h = half(e); rb = roll_beat(e, bt)
        row = {"name": name, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st["mult"], "vs1": h[0] / ht[0], "vs2": h[1] / ht[1], "vs_h43": s["mult"] / float(bh.iloc[-1]), "roll_beat": rb[0], "roll_med": rb[1]}
        res.append(row)
        L.append(f"| {name} | ×{row['mult']:.2f} | {pm(row['cagr'])} | {pm(row['mdd'])} | ×{h[0]:.2f} | ×{h[1]:.2f} | {row['vs']:.2f} | {row['vs1']:.2f} / {row['vs2']:.2f} | {row['vs_h43']:.2f} | {rb[0]:.0%} | {rb[1]:.2f} |")
    df = pd.DataFrame(res); new = df[df["name"].str.startswith("L")]
    L += ["", f"TQQQ 를 넘은 새 방법 {int((new['vs'] > 1).sum())} / {len(new)} · H43 도 넘은 것 {int((new['vs_h43'] > 1).sum())} / {len(new)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); pd.DataFrame(eq).to_parquet(f"{out}_curves.parquet"); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
