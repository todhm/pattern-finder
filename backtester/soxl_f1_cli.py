"""SOXL 사이클 탐색 F1 — 업계 지표 하나로 SOXL 을 들고 쉬면 타이밍이 되는가.

지표 15 종 × 규칙 6 종. 지표 값은 관측월 말 + lag 개월 뒤부터(기본 1 → 9월 값은 11월 1일부터), 신호는 전날 기준, 당일 종가 체결.
 단독(sig)  · 가격 결합: and200 = 신호 AND SOX 200일선 위 · or200 = 신호 OR SOX 200일선 위
보는 것: 세 창(1995~2009·2010~2017·2018~2023) 배수와 MDD, 기준선(SOXL 보유·SOX 200일선·TQQQ H43) 대비,
        신호의 매수·매도가 F0 사이클 저점·고점에서 몇 달 떨어졌나(중앙값), 신호 켜짐/꺼짐 때의 SOXL 앞 6 개월 수익.
봉인(2024-01~)은 열지 않는다. 실행: docker compose exec backtester python soxl_f1_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from soxl_f0_cli import curve
from strategy.adapters.cycle_signals import RULES, features, flips, rule_state, to_daily
from strategy.adapters.surge_rank_portfolio import curve_stats

SR = Path("/app/sweep_results")
WINDOWS = {"1995~2009": ("1995-01-03", "2009-12-31"), "2010~2017": ("2010-01-04", "2017-12-29"), "2018~2023": ("2018-01-02", "2023-12-29")}
NAMES = {"kr_semi_px": "한국 반도체 수출물가($)", "kr_semi_exp": "한국 반도체 수출금액", "kr_it_exp": "한국 IT 수출금액", "kr_elec_ship": "한국 전자부품 출하", "kr_elec_inv": "한국 전자부품 재고(뒤집음)",
         "kr_inv_ship": "한국 전자부품 재고/출하(뒤집음)", "kr_elec_prod": "한국 전자부품 생산", "us_semi_ip": "미국 반도체 생산", "us_semi_util": "미국 반도체 가동률", "us_semi_ppi": "미국 반도체 생산자물가",
         "us_ce_orders": "미국 컴퓨터전자 신규주문", "us_ce_ship": "미국 컴퓨터전자 출하", "us_book_bill": "미국 컴퓨터전자 주문/출하", "kr_exports": "한국 총수출", "t10y3m": "미 장단기 금리차(수준)", "baa10y": "미 신용 스프레드(뒤집음)"}
RULE_KR = {"yoy_pos": "전년비 +", "accel3": "전년비 3달 개선", "accel6": "전년비 6달 개선", "yoy_neg": "전년비 −(역발상)", "trough": "바닥권 반전 매수·꼭지권 꺾임 매도", "accel3_or_pos": "+ 이거나 개선 중"}


def load_ind() -> pd.DataFrame:
    d = pd.read_parquet(SR / "soxl_indicators.parquet")
    d["kr_inv_ship"] = d["kr_elec_ship"] / d["kr_elec_inv"]; d["kr_elec_inv"] = 1.0 / d["kr_elec_inv"]          # 재고는 낮을수록 좋다 → 뒤집어 "높을수록 좋음" 으로
    d["us_book_bill"] = d["us_ce_orders"] / d["us_ce_ship"]; d["baa10y"] = 1.0 / d["baa10y"]
    return d


def monthly_state(ind: pd.DataFrame, key: str, rule: str) -> pd.Series:
    x = ind[key].dropna()
    if key in ("t10y3m", "us_book_bill"):                      # 수준 지표: 전년비 대신 12 개월 차이
        f = pd.DataFrame({"yoy": x - (x.rolling(12).mean() if key == "us_book_bill" else 0.0)}); f["accel3"] = f["yoy"] - f["yoy"].shift(3); f["accel6"] = f["yoy"] - f["yoy"].shift(6); f["pct"] = f["yoy"].rolling(60, min_periods=36).rank(pct=True)
    else:
        f = features(x)
    return rule_state(f, rule)


def main(cfg):
    out = Path(cfg.get("out", SR / "soxl_f1")); lag = int(cfg.get("lag", 1)); end = cfg.get("end", "2023-12-29")
    b = pd.read_parquet(SR / "soxl_bench_1994.parquet"); b = b[b.index <= end]; sox, soxl = b["SOX"], b["SOXL"]; days = b.index
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"].reindex(days).ffill().fillna(0.0) / 252.0; ret = soxl.pct_change().fillna(0.0)
    above = (sox > sox.rolling(200).mean()).astype(float).shift(1).fillna(0.0); ind = load_ind()
    turns = [(pd.Timestamp(d), k) for d, k, _ in json.load(open(SR / "soxl_f0_turns.json")) if d <= end]
    lows = np.array([d.value for d, k in turns if k == "저점"]); highs = np.array([d.value for d, k in turns if k == "고점"])
    fwd = soxl.shift(-126) / soxl - 1.0

    def stats(e):
        o = {}
        for wn, (a, z) in WINDOWS.items():
            s = e[(e.index >= a) & (e.index <= z)]; st = curve_stats(s / s.iloc[0]); o[wn] = (st["mult"], st["mdd"])
        return o

    base = {"SOXL 보유": stats(curve(ret, above * 0 + 1, irx)), "SOX 200일선": stats(curve(ret, above, irx))}
    L = [f"# F1 — 업계 지표 하나로 SOXL 타이밍 (개발 ~{end}, 발표 지연 {lag} 개월)", "",
         "값 = 세전 배수(MDD). 굵게 = 그 창에서 'SOX 200일선' 기준선을 넘음. 매수·매도 거리 = 신호가 바뀐 날에서 가장 가까운 F0 사이클 저점(매수)·고점(매도)까지 개월 수 중앙값(+ 는 전환점보다 늦음).", "",
         "| 기준선 | " + " | ".join(WINDOWS) + " |", "|---|---|---|---|"] + [f"| {k} | " + " | ".join(f"×{m:,.2f} ({d:+.0%})" for m, d in v.values()) + " |" for k, v in base.items()]
    L += ["", "| 지표 | 규칙 | 결합 | " + " | ".join(WINDOWS) + " | 세 창 모두 200일선 초과 | 보유 비중 | 매수 거리 | 매도 거리 | 켜짐 뒤 6달 | 꺼짐 뒤 6달 | 왕복 수 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = []; ref = base["SOX 200일선"]
    for key, rule in itertools.product(NAMES, RULES):
        if key not in ind or ind[key].dropna().index[0] > pd.Timestamp("2001-01-01") and key != "kr_semi_exp":
            continue
        sig = to_daily(monthly_state(ind, key, rule), days, lag).fillna(0.0).shift(1).fillna(0.0)
        for comb, w in (("단독", sig), ("그리고 200일선", sig * above), ("또는 200일선", ((sig + above) > 0).astype(float))):
            e = curve(ret, w, irx); st = stats(e); fl = flips(w); beat = [st[wn][0] > ref[wn][0] for wn in WINDOWS]
            dist = lambda side, arr: float(np.median([(d.value - arr[np.argmin(np.abs(arr - d.value))]) / 86400e9 / 30.4 for d in fl[fl["side"] == side]["date"]])) if (fl["side"] == side).any() else np.nan  # noqa: E731
            row = {"key": key, "rule": rule, "comb": comb, **{f"{wn}_mult": st[wn][0] for wn in WINDOWS}, **{f"{wn}_mdd": st[wn][1] for wn in WINDOWS}, "beat_all": all(beat), "n_beat": sum(beat), "hold": float(w.mean()),
                   "buy_dist": dist("매수", lows), "sell_dist": dist("매도", highs), "fwd_on": float(fwd[w > 0].mean()), "fwd_off": float(fwd[w == 0].mean()), "trips": int((fl["side"] == "매수").sum())}
            res.append(row)
            L.append(f"| {NAMES[key]} | {RULE_KR[rule]} | {comb} | " + " | ".join((f"**×{st[wn][0]:,.2f}**" if bb else f"×{st[wn][0]:,.2f}") + f" ({st[wn][1]:+.0%})" for wn, bb in zip(WINDOWS, beat))
                     + f" | {'예' if all(beat) else '-'} | {row['hold']:.0%} | {row['buy_dist']:+.0f} | {row['sell_dist']:+.0f} | {row['fwd_on']:+.0%} | {row['fwd_off']:+.0%} | {row['trips']} |")
    df = pd.DataFrame(res)
    L += ["", f"세 창 모두 SOX 200일선 기준선을 넘은 조합 {int(df['beat_all'].sum())} / {len(df)} · 두 창 이상 {int((df['n_beat'] >= 2).sum())} / {len(df)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L[:9])); print(L[-1])
    top = df.sort_values(["n_beat", "2018~2023_mult"], ascending=False).head(25); print(top[["key", "rule", "comb", "1995~2009_mult", "2010~2017_mult", "2018~2023_mult", "hold", "buy_dist", "sell_dist", "fwd_on", "fwd_off", "trips"]].round(2).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
