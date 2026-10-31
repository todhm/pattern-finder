"""근거중심매매 스크리너 내보내기(54 페이지 CSV) 재검증 자료 — 기준일 종가로 자동 판독 + 책 규칙 레벨 + 차트.

    docker compose exec backtester python evidence_recheck_cli.py '{"csv": "/app/sweep_results/evidence_screen_2026-10-28.csv",
        "as_of": "2026-10-28", "min_weekly_yes": 4, "out_dir": "/app/sweep_results/evidence_recheck_2026-10-28"}'

출력: rows.json(종목별 자동값·책 레벨·플래그), auto.md(표), QQQ/SPY 지수 판독, ev_NN.png(4종목씩 주봉+일봉).
사람은 이 자료로 §5 수동 체크리스트(M-A~M-H, v3)를 적용한다 — 결과 문서 예: docs/strategy_notes/근거중심매매_2026-10-28_스크리너_*_수동검증.md
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters.yfinance_bulk import download_many
from evidence_manual_bt import JUNK, draw_panel, setup_font
from strategy.adapters.evidence_backtest import SimRules, entry_levels, precompute, reports_at
from strategy.adapters.evidence_checklist import EvidenceParams, fvg_text


def main(cfg: dict) -> None:
    as_of = pd.Timestamp(cfg["as_of"])
    out = Path(cfg.get("out_dir", f"/app/sweep_results/evidence_recheck_{as_of.date()}")); out.mkdir(parents=True, exist_ok=True)
    rows_in = list(csv.DictReader(open(cfg["csv"], encoding="utf-8-sig")))
    min_yes = int(cfg.get("min_weekly_yes", 4))
    tickers = [r["티커"] for r in rows_in if int(r.get("주봉 예", 0)) >= min_yes]
    junk = [t for t in tickers if len(t) == 5 and t[-1] in JUNK]
    tickers = [t for t in tickers if t not in junk]
    print(f"내보내기 {len(rows_in)} · 주봉 예 ≥ {min_yes}: {len(tickers) + len(junk)} · 정크 제외 {len(junk)} → {len(tickers)}", file=sys.stderr)
    prices = download_many(tickers + ["QQQ", "SPY"], date(2015, 1, 1), as_of.date())
    p = EvidenceParams(); rules = SimRules()
    data = precompute({t: d[d.index <= as_of] for t, d in prices.items()}, p)
    rows = []
    for t in ["QQQ", "SPY"] + tickers:
        if t not in data:
            rows.append({"티커": t, "오류": "데이터 없음"}); continue
        d, w = data[t]
        rep = reports_at(d, w, as_of, p, rules)
        if rep is None:
            rows.append({"티커": t, "오류": "봉 부족"}); continue
        wr, dr = rep
        c = float(d.at[as_of, "Close"]); e20 = float(d.at[as_of, "EMA20"])
        dv = float((d["Close"] * d["Volume"]).tail(20).mean()) / 1e6
        fb = wr.fib
        pb = float(wr.df["Low"].iloc[int(fb.hi_pos):].min()) if fb is not None else None
        lv = entry_levels(wr, c, rules, pb)
        rows.append({
            "티커": t, "종가": round(c, 2), "거래대금($M)": round(dv, 1), "20MA 대비": round(c / e20 - 1, 3),
            "주봉 예": wr.yes_count, "일봉 예": dr.yes_count,
            "5대(주)": {k: (v.ok, v.why[:80]) for k, v in wr.five.items()},
            "4분면": wr.channel.quadrant, "배열(주)": wr.ma.alignment, "배열(일)": dr.ma.alignment, "주봉 20MA 위": wr.ma.above_ema20,
            "주봉 MA 접촉": wr.ma.touching, "채널(주)": wr.channel.touch, "채널(일)": dr.channel.touch,
            "피보(주)": fb.text if fb else None, "레벨(주)": ({k: round(v, 2) for k, v in fb.levels.items() if k in (0.382, 0.5, 0.618, 0.786, 1.0)} if fb else None),
            "캔들(주)": wr.candle.text[:90], "캔들(일)": dr.candle.text[:90], "일봉 20MA 위": dr.candle.above_ema20, "거래량(일)": dr.candle.vol_ratio,
            "RSI(주)": round(wr.div.rsi), "RSI(일)": round(dr.div.rsi), "다이버(주)": wr.div.text, "하락다이버 2회": wr.div.bearish_count >= 2,
            "매물대": wr.supply.why[:80], "FVG(일)": fvg_text(dr.fvgs)[:80],
            "추격(20MA+5%)": c > e20 * 1.05,
            "책 레벨(종가 진입 가정)": (None if lv is None else {"2차": round(lv[1], 2) if lv[1] else None, "손절": round(lv[0], 2), "1:1": round(lv[2], 2), "최종": round(lv[3], 2),
                                           "손절%": round(lv[0] / c - 1, 3), "1:1%": round(lv[2] / c - 1, 3)}),
        })
    json.dump(rows, open(out / "rows.json", "w"), ensure_ascii=False, indent=1, default=str)
    # md 표
    L = [f"# 재검증 자동값 {as_of.date()} (종가 기준)", "", "| 티커 | 종가 | 거래대금 | 주봉 예 | 일봉 예 | 4분면 | 배열 주/일 | 주봉 MA | 채널 주/일 | 피보(주) | 캔들(주) | 캔들(일) | 일봉20MA | 하락다이버2회 | 추격 | 2차 | 손절 | 1:1 | 최종 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "오류" in r:
            L.append(f"| {r['티커']} | {r['오류']} |" + " |" * 17); continue
        b = r["책 레벨(종가 진입 가정)"] or {}
        L.append(f"| {r['티커']} | {r['종가']} | {r['거래대금($M)']} | {r['주봉 예']} | {r['일봉 예']} | {r['4분면']} | {r['배열(주)']}/{r['배열(일)']} | {', '.join(r['주봉 MA 접촉']) or '-'} | "
                 f"{r['채널(주)'] or '-'}/{r['채널(일)'] or '-'} | {(r['피보(주)'] or '-')[:60]} | {r['캔들(주)'][:40]} | {r['캔들(일)'][:40]} | {'위' if r['일봉 20MA 위'] else '아래'} | "
                 f"{'⚠️' if r['하락다이버 2회'] else '-'} | {'⚠️' if r['추격(20MA+5%)'] else '-'} | {b.get('2차', '-')} | {b.get('손절', '-')} ({b.get('손절%', 0):+.1%}) | {b.get('1:1', '-')} ({b.get('1:1%', 0):+.1%}) | {b.get('최종', '-')} |")
    (out / "auto.md").write_text("\n".join(L))
    # 차트
    setup_font()
    import matplotlib.pyplot as plt
    tks = [r["티커"] for r in rows if "오류" not in r]
    for i in range(0, len(tks), 4):
        grp = tks[i:i + 4]
        fig, axes = plt.subplots(len(grp), 2, figsize=(16, 4.2 * len(grp))); axes = np.atleast_2d(axes)
        for j, t in enumerate(grp):
            wr, dr = reports_at(data[t][0], data[t][1], as_of, p, rules)
            draw_panel(axes[j, 0], wr, 160, f"{t} 주봉 {as_of.date()} — 예 {wr.yes_count}/5 · {wr.ma.alignment} · 채널 {wr.channel.touch or '-'} · {wr.channel.quadrant} · {wr.fib.zone if wr.fib else '-'}")
            draw_panel(axes[j, 1], dr, 200, f"{t} 일봉 — 예 {dr.yes_count}/5 · {dr.ma.alignment} · 캔들 {','.join(dr.candle.patterns) or '-'} · RSI {dr.div.rsi:.0f}")
        fig.tight_layout(); fig.savefig(out / f"ev_{i // 4 + 1:02d}.png", dpi=80); plt.close(fig)
    print("out", out, "charts", (len(tks) + 3) // 4, file=sys.stderr)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
