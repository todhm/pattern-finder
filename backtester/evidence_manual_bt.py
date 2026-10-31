"""근거중심매매 **수동** 백테스트 보조 — 사람이 블라인드로 판정할 수 있게 자료만 만든다.

    python evidence_manual_bt.py data                       # 유니버스 일봉 캐시(parquet) + 레짐별 무작위 날짜
    python evidence_manual_bt.py candidates 2019-05-31 [min_yes] [out_dir]   # 그 금요일까지의 데이터로 주봉 후보 + 차트
    python evidence_manual_bt.py chart 2019-05-31 AAPL,MSFT [out_png]        # 지정 종목 차트(블라인드)
    python evidence_manual_bt.py holdings 2019-06-03 2019-06-07 '{"AAPL": {"entry_px": 1, "stop": 1, "second": 1, "target1": 1, "final": 1}}'
    python evidence_manual_bt.py levels AAPL 2019-06-03 152.3  # 책 규칙 레벨: 2차(0.618 이하 지정가)·손절(근거선)·1:1(3점 확장)·최종(1.618)
    python evidence_manual_bt.py settle decisions.json       # 사람이 적은 결정(진입·청산 날짜)을 다음날 시가로 체결해 손익 계산

결정 파일 형식: {"start": "...", "end": "...", "trades": [{"ticker": "AAPL", "legs": [{"date": "2019-06-03", "side": "buy", "frac": 0.1, "why": "..."},
  {"date": "2019-06-20", "side": "sell", "frac_of_pos": 0.75, "why": "1:1"}]}]}  frac = 자본 대비 비중, 체결은 '그 날짜 다음 거래일 시가'.
  2차 지정가는 {"date": "터치일", "side": "buy", "frac": 0.1, "limit": 138.2} — 그 날 저가가 limit 이하면 min(시가, limit) 로 **그 날** 체결.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.evidence_backtest import classify_regimes, entry_levels, pick_dates, precompute, reports_at, SimRules
from strategy.adapters.evidence_checklist import EvidenceParams, fvg_text

CACHE = Path("/app/sweep_results/evidence_manual_prices.parquet")
JUNK = ("U", "P", "O", "L", "N", "I", "M", "G", "W", "R")


def _load() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(CACHE)
    return {t: g.drop(columns="ticker").sort_index() for t, g in df.groupby("ticker")}


def cmd_data() -> None:
    from data.adapters.wikipedia_universe import default_universe_provider
    from data.adapters.yfinance_bulk import download_many
    prov = default_universe_provider()
    tk = list(dict.fromkeys(prov.get_tickers("sp500") + prov.get_tickers("nasdaq100"))) + ["QQQ", "SPY"]
    pr = download_many(tk, date(2003, 1, 1), date.today())
    df = pd.concat([g.assign(ticker=t) for t, g in pr.items()])
    df.to_parquet(CACHE)
    reg = classify_regimes(pr["QQQ"])
    print("cached", len(pr), "tickers")
    for n, s in ((4, 7), (3, 11)):
        print(f"seed {s} n_each {n}:", [(r, d.date().isoformat()) for r, d in pick_dates(reg, n, s, "2006-01-01", "2026-03-31")])


def _blind(d: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return d.loc[:as_of]


def cmd_candidates(as_of: str, min_yes: int = 3, out_dir: str = "/app/sweep_results/evidence_manual_imgs") -> None:
    as_of_ts = pd.Timestamp(as_of)
    prices = _load()
    p = EvidenceParams(); rules = SimRules()
    data = precompute({t: _blind(d, as_of_ts) for t, d in prices.items() if t not in ("QQQ", "SPY")}, p)
    rows = []
    for t, (d, w) in data.items():
        if len(t) == 5 and t[-1] in JUNK:
            continue
        rep = reports_at(d, w, as_of_ts, p, rules)
        if rep is None:
            continue
        wr, dr = rep
        if wr.yes_count < min_yes or "최악" in wr.channel.quadrant:
            continue
        dv = float((d["Close"] * d["Volume"]).tail(20).mean()) / 1e6
        if dv < 5:
            continue
        rows.append({"티커": t, "주봉 예": wr.yes_count, "일봉 예": dr.yes_count, "종가": round(float(d["Close"].iloc[-1]), 2),
                     "52주고 대비": round(float(d["Close"].iloc[-1] / d["High"].tail(252).max() - 1), 3), "거래대금M": round(dv),
                     "배열": wr.ma.alignment, "4분면": wr.channel.quadrant, "채널": wr.channel.touch or "-",
                     "MA접촉": ",".join(wr.ma.touching) or "-", "피보": wr.fib.zone if wr.fib else "-",
                     "피보 상세": (wr.fib.text if wr.fib else ""), "캔들(주)": ",".join(wr.candle.patterns) or "-", "캔들(일)": ",".join(dr.candle.patterns) or "-",
                     "일봉20MA위": dr.candle.above_ema20, "RSI주": round(wr.div.rsi), "다이버": ("상승" if wr.div.bullish else "") + (f"하락x{wr.div.bearish_count}" if wr.div.bearish_count else ""),
                     "매물대": wr.supply.why[:40], "FVG": fvg_text(dr.fvgs)[:90],
                     "레벨": {k: round(v, 2) for k, v in (wr.fib.levels.items() if wr.fib else {}) if k in (0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.382, 1.618, "ext1.272")}})
    rows.sort(key=lambda r: (-r["주봉 예"], abs(r["52주고 대비"])))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print(f"# {as_of} 후보 {len(rows)}개 (주봉 예 ≥ {min_yes})", file=sys.stderr)
    # 시장
    for t in ("QQQ", "SPY"):
        d = _blind(prices[t], as_of_ts); dd = precompute({t: d}, p)[t]
        rep = reports_at(dd[0], dd[1], as_of_ts, p, rules)
        if rep:
            wr, dr = rep
            print(f"# {t} 주봉 예 {wr.yes_count} ({wr.ma.text} | 채널 {wr.channel.touch} {wr.channel.quadrant} | {wr.fib.zone if wr.fib else '-'}) · 일봉 예 {dr.yes_count} ({dr.ma.text} | 캔들 {','.join(dr.candle.patterns) or '-'} RSI {dr.div.rsi:.0f})", file=sys.stderr)
    # 자동 1차 거름: 주봉 예 4+, 하락 채널 아님, (일봉 20MA 위 또는 일봉 상승 캔들), 일봉 하락 반전 캔들 없음 → 최대 5개 + QQQ 차트
    bear = ("하락장악형", "베어리쉬", "이브닝스타", "흑삼병", "하락다람쥐", "윗꼬리 강")
    fin = [r for r in rows if r["주봉 예"] >= 4 and not r["4분면"].startswith("하락") and (r["일봉20MA위"] or any(k in r["캔들(일)"] for k in ("아래꼬리 강", "상승장악형", "모닝스타", "백삼병", "불리쉬")))
           and not any(k in r["캔들(일)"] for k in bear) and not any(k in r["캔들(주)"] for k in bear)]
    print("# 1차 거름 통과:", [r["티커"] for r in fin], file=sys.stderr)
    if fin:
        cmd_chart(as_of, ",".join(["QQQ"] + [r["티커"] for r in fin[:5]]), f"{out_dir}/fin_{as_of}.png")


def setup_font() -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = "/app/pattern-finder-cache/fonts/AppleGothic.ttf"
    try:
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def draw_panel(ax, rep, bars: int, title: str) -> None:
    """주봉/일봉 패널: 캔들·EMA·SMA365·채널·피보나치·매물대·FVG·스윙·RSI (기준일까지만)."""
    d = rep.df.tail(bars); off = len(rep.df) - len(d); x = np.arange(len(d))
    up = d["Close"] >= d["Open"]
    ax.bar(x, (d["Close"] - d["Open"]).abs().where(lambda s: s > 0, 1e-9), bottom=d[["Open", "Close"]].min(axis=1), width=0.6, color=np.where(up, "#d62728", "#1f77b4"))
    ax.vlines(x, d["Low"], d["High"], color=np.where(up, "#d62728", "#1f77b4"), linewidth=0.6)
    for n, col in ((20, "red"), (50, "orange"), (100, "skyblue"), (200, "purple")):
        ax.plot(x, d[f"EMA{n}"].to_numpy(), color=col, linewidth=0.9)
    if "SMA365" in d:
        ax.plot(x, d["SMA365"].to_numpy(), color="black", linewidth=0.9)
    for ch, col in zip(rep.channel.channels, ("#b8860b", "#2e8b57")):
        ser = ch.series(len(rep.df)).iloc[off:]
        for nm, ls in (("upper", "-"), ("mid", ":"), ("lower", "-")):
            ax.plot(x, ser[nm].to_numpy(), color=col, linewidth=0.8, linestyle=ls)
    if rep.fib:
        for r, v in rep.fib.levels.items():
            if isinstance(r, float) and r in (0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618):
                ax.axhline(v, color="gray", linewidth=0.5, linestyle="--"); ax.text(len(d) - 1, v, f"{r}", fontsize=6, color="gray", va="bottom")
    for b in rep.boxes:
        if b.end >= off:
            ax.axhspan(b.bottom, b.top, xmin=max(0, (b.start - off)) / len(d), xmax=1, color="green", alpha=0.08)
    for f in rep.fvgs:
        if f.fill < 1 and f.pos >= off:
            ax.axhspan(f.bottom, f.top, xmin=(f.pos - off) / len(d), xmax=1, color="orange" if f.bullish else "gray", alpha=0.15)
    sw = rep.swings[rep.swings.pos >= off]
    ax.scatter(sw[sw.kind == "H"].pos - off, sw[sw.kind == "H"].price, marker="v", color="#ff7f0e", s=14)
    ax.scatter(sw[sw.kind == "L"].pos - off, sw[sw.kind == "L"].price, marker="^", color="#2ca02c", s=14)
    ax.set_yscale("log"); ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
    step = max(1, len(d) // 8); ax.set_xticks(x[::step]); ax.set_xticklabels([i.strftime("%y-%m-%d") for i in d.index[::step]], fontsize=6)
    ax2 = ax.twinx(); ax2.plot(x, d["RSI"].to_numpy(), color="#9467bd", linewidth=0.7, alpha=0.7); ax2.set_ylim(0, 300); ax2.set_yticks([])


def cmd_chart(as_of: str, tickers: str, out_png: str) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = "/app/pattern-finder-cache/fonts/AppleGothic.ttf"
    try:
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    as_of_ts = pd.Timestamp(as_of)
    prices = _load(); p = EvidenceParams(); rules = SimRules()
    tks = [t.strip() for t in tickers.split(",") if t.strip()]
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    panel = draw_panel
    fig, axes = plt.subplots(len(tks), 2, figsize=(16, 4.2 * len(tks)))
    axes = np.atleast_2d(axes)
    for i, t in enumerate(tks):
        dd = precompute({t: _blind(prices[t], as_of_ts)}, p)[t]
        rep = reports_at(dd[0], dd[1], as_of_ts, p, rules)
        if rep is None:
            continue
        wr, dr = rep
        panel(axes[i, 0], wr, 160, f"{t} 주봉 {as_of} — 예 {wr.yes_count}/5 · {wr.ma.alignment} · 채널 {wr.channel.touch or '-'} · {wr.channel.quadrant} · {wr.fib.zone if wr.fib else '-'}")
        panel(axes[i, 1], dr, 200, f"{t} 일봉 — 예 {dr.yes_count}/5 · {dr.ma.alignment} · 캔들 {','.join(dr.candle.patterns) or '-'} · RSI {dr.div.rsi:.0f}")
    fig.tight_layout(); fig.savefig(out_png, dpi=80); plt.close(fig)
    print("chart", out_png, tks, file=sys.stderr)


def cmd_holdings(d1: str, d2: str, pos_json: str) -> None:
    """보유 종목의 날짜별 재판정 자료: OHLC·레벨 대비·캔들·MA·RSI. 체결은 하지 않는다."""
    pos = json.loads(pos_json)
    prices = _load(); p = EvidenceParams(); rules = SimRules()
    days = pd.to_datetime(prices["QQQ"].loc[d1:d2].index)
    for t, lv in pos.items():
        full = prices[t]
        print(f"\n## {t} 진입가 {lv.get('entry_px')} 2차 {lv.get('second')} 손절 {lv.get('stop')} 1:1 {lv.get('target1')} 최종 {lv.get('final')}")
        for day in days:
            if day not in full.index:
                continue
            dd = precompute({t: _blind(full, day)}, p)[t]
            rep = reports_at(dd[0], dd[1], day, p, rules)
            r = full.loc[day]
            flags = []
            if lv.get("stop") and r.Close <= lv["stop"]: flags.append("⛔ 종가<손절")
            if lv.get("second") and r.Low <= lv["second"]: flags.append(f"② 2차 지정가 체결 {min(r.Open, lv['second']):.2f}")
            if lv.get("target1") and r.Close >= lv["target1"]: flags.append("🎯 1:1 도달")
            if lv.get("final") and r.Close >= lv["final"]: flags.append("🏁 최종 목표")
            extra = ""
            if rep:
                wr, dr = rep
                extra = f"캔들 {','.join(dr.candle.patterns) or '-'} | 20MA {'위' if dr.candle.above_ema20 else '아래'} | 거래량 {dr.candle.vol_ratio or 0:.1f}× | RSI {dr.div.rsi:.0f} | {dr.ma.alignment} | 채널 {dr.channel.touch or '-'}"
                if day.weekday() == 4:
                    extra += f" || 주봉 예 {wr.yes_count} {wr.candle.text[:60]} | 주봉 20MA {'위' if wr.ma.above_ema20 else '아래'}"
            print(f"{day.date()} 시 {r.Open:.2f} 고 {r.High:.2f} 저 {r.Low:.2f} 종 {r.Close:.2f} ({r.Close / full['Close'].shift(1).loc[day] - 1:+.1%}) {' '.join(flags)} | {extra}")


def cmd_levels(ticker: str, as_of: str, entry_px: str) -> None:
    """책 규칙(2026-10-29 재대조판) 레벨: 금요일 주봉 기준 2차·손절·1:1·최종. 진입가는 사람이 넣는다(보통 다음날 시가)."""
    prices = _load(); p = EvidenceParams(); rules = SimRules()
    day = pd.Timestamp(as_of)
    dd = precompute({ticker: _blind(prices[ticker], day)}, p)[ticker]
    rep = reports_at(dd[0], dd[1], day, p, rules)
    if rep is None:
        print("데이터 부족"); return
    wr, _ = rep
    fb = wr.fib
    if fb is None:
        print("피보나치 없음"); return
    pb = float(wr.df["Low"].iloc[int(fb.hi_pos):].min())
    lv = entry_levels(wr, float(entry_px), rules, pb)
    print(f"{ticker} {as_of} {fb.kind} 고 {fb.hi:.2f} 저 {fb.lo:.2f} 되돌림 저점 {pb:.2f} · 레벨 " +
          " ".join(f"{k}={v:.2f}" for k, v in fb.levels.items() if k in (0.382, 0.5, 0.618, 0.786, 1.0)))
    if lv is None:
        print("진입 불가(근거선 −20% 초과 또는 목표 없음)"); return
    stop, second, t1, fin = lv
    print(json.dumps({"entry_px": float(entry_px), "second": second, "stop": stop, "target1": round(t1, 2), "final": round(fin, 2)}))


def cmd_settle(path: str) -> None:
    dec = json.load(open(path))
    prices = _load()
    cal = pd.to_datetime(prices["QQQ"].index)
    fee = 0.0015
    cash = 1.0; eq_rows = []; book = {}
    trades_out = []
    events = []
    for tr in dec["trades"]:
        for leg in tr["legs"]:
            d = pd.Timestamp(leg["date"])
            nxt = d if leg.get("limit") else cal[cal > d][0]     # 지정가는 터치일 당일 체결
            events.append((nxt, tr["ticker"], leg))
    events.sort(key=lambda e: e[0])
    start, end = pd.Timestamp(dec["start"]), pd.Timestamp(dec["end"])
    days = cal[(cal >= start) & (cal <= end)]
    for day in days:
        for nxt, t, leg in [e for e in events if e[0] == day]:
            o = float(prices[t].loc[day, "Open"])
            if leg.get("limit"):
                if float(prices[t].loc[day, "Low"]) > float(leg["limit"]):
                    print(f"# {t} {day.date()} 지정가 {leg['limit']} 미체결(저가 {float(prices[t].loc[day, 'Low']):.2f})", file=sys.stderr)
                    continue
                o = min(o, float(leg["limit"]))
            b = book.setdefault(t, {"qty": 0.0, "cost": 0.0, "proceeds": 0.0, "legs": []})
            if leg["side"] == "buy":
                eq_now = cash + sum(v["qty"] * float(prices[k].loc[day, "Close"]) for k, v in book.items())
                amt = eq_now * leg["frac"]
                q = amt * (1 - fee) / o; cash -= amt; b["qty"] += q; b["cost"] += amt
                b["legs"].append((day.date().isoformat(), "buy", round(o, 2), round(leg["frac"], 3), leg.get("why", "")))
            else:
                q = b["qty"] * leg.get("frac_of_pos", 1.0)
                cash += q * o * (1 - fee); b["qty"] -= q; b["proceeds"] += q * o * (1 - fee)
                b["legs"].append((day.date().isoformat(), "sell", round(o, 2), round(leg.get("frac_of_pos", 1.0), 3), leg.get("why", "")))
        eq = cash + sum(v["qty"] * float(prices[k].loc[day, "Close"]) for k, v in book.items() if day in prices[k].index)
        eq_rows.append((day.date().isoformat(), round(eq, 5)))
    last = days[-1]
    for t, b in book.items():
        if b["qty"] > 1e-12:
            c = float(prices[t].loc[last, "Close"]); b["proceeds"] += b["qty"] * c * (1 - fee); b["legs"].append((last.date().isoformat(), "mark", round(c, 2), 1.0, "구간 종료 평가")); b["qty"] = 0
        trades_out.append({"티커": t, "손익(자본 대비)": round(b["proceeds"] - b["cost"], 4), "legs": b["legs"]})
    eq = pd.Series(dict(eq_rows))
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1); mdd = float((eq / eq.cummax() - 1).min())
    realized = sum(x["손익(자본 대비)"] for x in trades_out); tax = max(0.0, realized - 0.025) * 0.22
    bh = {t: float(prices[t].loc[start:end, "Close"].iloc[-1] / prices[t].loc[start:end, "Close"].iloc[0] - 1) for t in ("QQQ", "SPY")}
    out = {"start": dec["start"], "end": dec["end"], "수익률": ret, "세후": ret - tax, "MDD": mdd, "QQQ": bh["QQQ"], "SPY": bh["SPY"],
           "거래": trades_out, "equity": dict(eq_rows)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "data":
        cmd_data()
    elif cmd == "candidates":
        cmd_candidates(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3, sys.argv[4] if len(sys.argv) > 4 else "/app/sweep_results/evidence_manual_imgs")
    elif cmd == "chart":
        cmd_chart(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else f"/app/sweep_results/evidence_manual_imgs/chart_{sys.argv[2]}.png")
    elif cmd == "holdings":
        cmd_holdings(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "levels":
        cmd_levels(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "settle":
        cmd_settle(sys.argv[2])
