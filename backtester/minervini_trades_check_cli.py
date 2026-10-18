"""거래 로그(CSV: Ticker, Entry Date, …) 의 각 진입을 **진입 전날 기준** 미너비니 체크리스트로 재판정.

    docker compose exec backtester python minervini_trades_check_cli.py \\
        '{"csv": "/app/sweep_results/wedgepop_trades_2025-05_2026-04.csv", "out": "/app/sweep_results/minervini_trades_check.json",
          "report_md": "/app/sweep_results/minervini_trades_check.md", "charts": "/app/sweep_results/minervini_trades_imgs"}'

판정 소스: STAGE 0 = SPY·QQQ(`evaluate_market`) + 0-3 신고가/신저가(S&P500+Nasdaq100 유니버스, 현재 구성 종목 — 소폭 look-ahead),
STAGE 1 = `evaluate_trend_template` + RS 백분위(같은 유니버스), STAGE 2 = Alpha Vantage 분기·연간·서프라이즈·재무상태표 + 뉴스 가이던스,
STAGE 3 = `manual_auto`(0-4·3-1·3-2·3-3·4-1·4-10) — 3-4·3-5 는 사람 몫, STAGE 4 = `pivot_levels` + `stage4`(핵심 5/7 완화·엄격 둘 다).
4-12 는 실제 진입일 거래량 / 50일 평균. 차트 PNG(진입 전 130일 + 후 20일)를 4장씩 묶어 저장 — 눈판독용.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.adapters.alphavantage_adapter import AlphaVantageAdapter
from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from pages._shared.growth_sources import build_growth_adapters, fetch_growth_snapshot
from strategy.adapters.minervini_full_checklist import stage2, stage3, stage4
from strategy.adapters.minervini_manual_auto import manual_auto
from strategy.adapters.minervini_scan import fetch_with_retry
from strategy.adapters.minervini_screener import (
    distribution_days,
    evaluate_annual,
    evaluate_balance,
    evaluate_growth,
    evaluate_market,
    evaluate_surprise,
    evaluate_trend_template,
    momentum_score,
    pivot_levels,
    rs_percentiles,
)


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _clip(df: pd.DataFrame, as_of: date | None = None) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated()].sort_index()
    return df.loc[: pd.Timestamp(as_of)] if as_of else df


def _money(s: str) -> float:
    return float(str(s).replace("$", "").replace(",", "").replace("%", ""))


def _table(d: pd.DataFrame) -> str:
    cols = [str(c) for c in d.columns]
    rows = [["" if v is None else str(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def _mark(v) -> str:
    return "✅" if v is True else "❌" if v is False else "—"


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    trades = pd.read_csv(cfg["csv"], encoding="utf-8-sig")
    trades["entry_date"] = pd.to_datetime(trades["Entry Date"]).dt.date
    md = MongoDayCacheAdapter(AlphaVantageAdapter(), source_name="alphavantage")
    growth = build_growth_adapters()
    bal = next((a for a in growth if hasattr(a, "fetch_balance_sheet")), None)
    try:
        from data.adapters.alphavantage_news import AlphaVantageNewsAdapter
        news = AlphaVantageNewsAdapter()
    except Exception:  # noqa: BLE001
        news = None
    uni = default_universe_provider()
    universe = sorted(set(uni.get_tickers("sp500")) | set(uni.get_tickers("nasdaq100")) | set(trades["Ticker"]) | {"SPY", "QQQ"})
    first, last = min(trades["entry_date"]), max(trades["entry_date"])
    start, end = first - timedelta(days=720), min(last + timedelta(days=120), date.today())
    _log(f"유니버스 {len(universe)} 일봉 {start}~{end}")
    frames: dict[str, pd.DataFrame] = {}
    from concurrent.futures import ThreadPoolExecutor

    def _f(t):
        try:
            return t, _clip(fetch_with_retry(md, t, start, end))
        except Exception as e:  # noqa: BLE001
            return t, None
    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 6))) as ex:
        for i, (t, df) in enumerate(ex.map(_f, universe)):
            if df is not None and len(df):
                frames[t] = df
            if (i + 1) % 100 == 0:
                _log(f"  일봉 {i + 1}/{len(universe)}")
    _log(f"일봉 {len(frames)}/{len(universe)}")
    snaps: dict[str, object] = {}
    for t in sorted(set(trades["Ticker"])):
        try:
            snaps[t] = fetch_growth_snapshot(t, growth)
        except Exception as e:  # noqa: BLE001
            _log(f"  {t} 펀더멘털 실패 {type(e).__name__}")
            snaps[t] = None
    bals: dict[str, list] = {}
    if bal is not None:
        for t in snaps:
            try:
                bals[t] = bal.fetch_balance_sheet(t)
            except Exception:  # noqa: BLE001
                bals[t] = []

    rows = []
    for _, tr in trades.iterrows():
        t, ed = tr["Ticker"], tr["entry_date"]
        full = frames.get(t)
        if full is None or pd.Timestamp(ed) not in full.index:
            rows.append({"ticker": t, "entry_date": str(ed), "error": "일봉 없음"})
            continue
        pos = full.index.get_loc(pd.Timestamp(ed))
        as_of = full.index[pos - 1].date()            # 셋업일 = 진입 전 거래일
        df = full.loc[: pd.Timestamp(as_of)]
        # STAGE 0
        verdicts = {s: evaluate_market(_clip(frames[s], as_of), symbol=s) for s in ("SPY", "QQQ") if s in frames}
        dist = {s: distribution_days(_clip(frames[s], as_of)) for s in verdicts}
        nh = nl = 0
        scores = {}
        for u, udf in frames.items():
            if u in ("SPY", "QQQ"):
                continue
            ud = udf.loc[: pd.Timestamp(as_of)]
            if len(ud) < 253:
                continue
            c = float(ud["Close"].iloc[-1])
            hi, lo = float(ud["High"].tail(252).max()), float(ud["Low"].tail(252).min())
            is_nh = c >= 0.95 * hi
            nh += is_nh
            nl += (not is_nh) and c <= 1.05 * lo
            s = momentum_score(ud)
            if s is not None:
                scores[u] = s
        rs = rs_percentiles(scores).get(t)
        # STAGE 1
        tt = evaluate_trend_template(df)
        # STAGE 2
        snap = snaps.get(t)
        q = snap.quarters if snap else []
        gr = evaluate_growth(q, as_of) if q else None
        an = evaluate_annual(snap.annuals, as_of) if snap and snap.annuals else None
        sr = evaluate_surprise(q, as_of) if q else None
        br = evaluate_balance(bals.get(t, []), q, as_of) if (bals.get(t) and q) else None
        titles = None
        if news is not None:
            try:
                titles = [n.title for n in news.fetch_news(t, as_of - timedelta(days=120), as_of)]
            except Exception:  # noqa: BLE001
                titles = None
        s2 = stage2(gr, an, sr, br, titles)
        # STAGE 3 / 4
        pl = pivot_levels(df)
        ma = manual_auto(df, _clip(frames["QQQ"], as_of), pl) if pl is not None and "QQQ" in frames else None
        m = dict(ma.checks) if ma else {}
        dvol = float((df["Close"] * df["Volume"]).tail(20).mean())
        s3 = stage3(m, snap.industry if snap and snap.industry else "미분류", set(), liquidity_ok=dvol >= 5e6)
        s4s = stage4(df, pl) if pl is not None else None
        s4 = stage4(df, pl, min_core=5) if pl is not None else None
        vol50 = float(df["Volume"].tail(50).mean())
        day = full.iloc[pos]
        fwd = full.iloc[pos:]
        c0 = float(day["Close"])
        row = {
            "ticker": t, "entry_date": str(ed), "setup_date": str(as_of), "exit_reason": tr["Exit Reason"],
            "wp_pnl_pct": _money(tr["Net P&L (%)"]) / 100, "wp_days": (pd.to_datetime(tr["Exit Date"]).date() - ed).days,
            "industry": snap.industry if snap else None, "sector": snap.sector if snap else None,
            "market": {s: {"passed": v.ok if v else None, "trend_ok": v.trend_ok if v else None, "short_ok": v.short_ok if v else None,
                           "note": v.note if v else "", "dist": v.dist_days if v else dist.get(s)} for s, v in verdicts.items()},
            "nh": nh, "nl": nl, "breadth_ok": nh > nl,
            "tt": {**asdict(tt), "n_passed": sum(tt.checks.values())} if tt else None, "rs": rs,
            "growth": asdict(gr) if gr else None, "annual": asdict(an) if an else None,
            "surprise": asdict(sr) if sr else None, "balance": asdict(br) if br else None,
            "news_titles": (titles or [])[:15],
            "s2": asdict(s2), "s3": asdict(s3), "s4_strict": asdict(s4s) if s4s else None, "s4_core5": asdict(s4) if s4 else None,
            "manual_auto": {"checks": m, "notes": ma.notes if ma else {}},
            "pivot": asdict(pl) if pl else None,
            "entry_vol_ratio": float(day["Volume"] / vol50) if vol50 else None,
            "entry_close_vs_pivot": (c0 / pl.pivot - 1) if pl else None,
            "fwd_1m": float(fwd["Close"].iloc[min(21, len(fwd) - 1)] / c0 - 1), "fwd_3m": float(fwd["Close"].iloc[min(63, len(fwd) - 1)] / c0 - 1),
            "dollar_vol_m": dvol / 1e6,
        }
        rows.append(row)
        tn = sum(tt.checks.values()) if tt else None
        _log(f"{t:5s} {ed} S0 {'✅' if verdicts and all(v.ok for v in verdicts.values() if v) else '❌'} 0-3 {nh}/{nl} | S1 {tn}/7 RS {rs and round(rs)} "
             f"| S2 {s2.ok} | S3 {s3.ok} | S4 {s4.ok if s4 else None} | WP {row['wp_pnl_pct']:+.1%}")
    with open(cfg.get("out", "/tmp/minervini_trades_check.json"), "w") as f:
        json.dump({"config": cfg, "rows": rows}, f, ensure_ascii=False, indent=1, default=str)
    report = _report(rows)
    print(report)
    if cfg.get("report_md"):
        open(cfg["report_md"], "w").write(report)
    if cfg.get("charts"):
        _charts(rows, frames, cfg["charts"])


def _report(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if r.get("error"):
            out.append({"종목": r["ticker"], "진입": r["entry_date"], "오류": r["error"]})
            continue
        mk = r["market"]
        s0 = bool(mk) and all(v["passed"] for v in mk.values()) and r["breadth_ok"]
        warn = any((v["dist"] or 0) >= 5 for v in mk.values())
        tt = r["tt"]
        s1_n = tt["n_passed"] if tt else None
        s1 = bool(tt and tt["passed"] and (r["rs"] or 0) >= 70)
        s2i, s3i, s4i = r["s2"]["items"], r["s3"]["items"], (r["s4_core5"] or {}).get("items", {})
        s2_core = sum(s2i.get(k) is True for k in ("s2_1", "s2_2", "s2_3", "s2_4", "s2_7"))
        s2_alert = [k[3:] for k in ("s2_8", "s2_9", "s2_10") if s2i.get(k) is False]
        s4_core = sum(v is True for k, v in s4i.items() if k not in ("s4_3", "s4_10", "s4_11"))
        s4_guard = [k[3:] for k in ("s4_3", "s4_10", "s4_11") if s4i.get(k) is False]
        verdict = ("탈락 S0" if not s0 else "탈락 S1" if not s1 else "탈락 S2" if not r["s2"]["ok"] else "탈락 S3" if not r["s3"]["ok"]
                   else "워치(S4 미완)" if not (r["s4_core5"] or {}).get("ok") else "매수 후보")
        out.append({
            "종목": r["ticker"], "진입": r["entry_date"], "WP 손익": f"{r['wp_pnl_pct']:+.1%}", "보유일": r["wp_days"],
            "S0 시장": ("✅" if s0 else "❌") + (" ⚠분산" if warn else "") + f" {r['nh']}/{r['nl']}",
            "S1 템플릿": f"{s1_n}/7 RS{round(r['rs']) if r['rs'] is not None else '—'}" + (" ✅" if s1 else " ❌"),
            "S2 펀더": (f"{s2_core}/5" if s2i.get("s2_1") is not None else "미확인") + (f" ⛔{','.join(s2_alert)}" if s2_alert else "") + (" ✅" if r["s2"]["ok"] else " ❌"),
            "S3 주도": f"{_mark(s3i.get('s3_1'))}{_mark(s3i.get('s3_2'))}{_mark(s3i.get('s3_3'))}" + (" ✅" if r["s3"]["ok"] else " ❌"),
            "S4 VCP": f"{s4_core}/7" + (f" ⛔{','.join(s4_guard)}" if s4_guard else "") + (" ✅" if (r["s4_core5"] or {}).get("ok") else " ❌"),
            "피봇 대비": f"{r['entry_close_vs_pivot']:+.1%}" if r["entry_close_vs_pivot"] is not None else "—",
            "4-12 거래량": f"{r['entry_vol_ratio']:.1f}×" if r["entry_vol_ratio"] else "—",
            "자동 판정": verdict, "1M/3M": f"{r['fwd_1m']:+.0%}/{r['fwd_3m']:+.0%}",
        })
    df = pd.DataFrame(out)
    lines = ["# 미너비니 체크리스트 — 거래 로그 재판정 (진입 전날 기준)", "", _table(df), ""]
    ok = [r for r in rows if not r.get("error")]
    if ok:
        d = pd.DataFrame([{"S1": bool(r["tt"] and r["tt"]["passed"] and (r["rs"] or 0) >= 70), "S2": r["s2"]["ok"], "S3": r["s3"]["ok"],
                           "S4": bool((r["s4_core5"] or {}).get("ok")), "pnl": r["wp_pnl_pct"]} for r in ok])
        agg = []
        for k in ("S1", "S2", "S3", "S4"):
            for v in (True, False):
                s = d[d[k] == v]["pnl"]
                if len(s):
                    agg.append({"단계": k, "통과": v, "n": len(s), "승률": f"{(s > 0).mean():.0%}", "평균": f"{s.mean():+.1%}", "합계": f"{s.sum():+.1%}"})
        lines += ["## 단계 통과 여부 × WedgePop 손익", "", _table(pd.DataFrame(agg)), ""]
    return "\n".join(lines)


def _charts(rows: list[dict], frames: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import Rectangle
    from pathlib import Path
    fp = Path("/app/pattern-finder-cache/fonts/AppleGothic.ttf")
    if fp.exists():
        fm.fontManager.addfont(str(fp))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(fp)).get_name()
    plt.rcParams["text.parse_math"] = False
    plt.rcParams["axes.unicode_minus"] = False
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if not r.get("error")]
    for g in range(0, len(ok), 4):
        batch = ok[g:g + 4]
        fig, axes = plt.subplots(len(batch), 1, figsize=(14, 4.2 * len(batch)), squeeze=False)
        for ax, r in zip(axes[:, 0], batch):
            full = frames[r["ticker"]]
            pos = full.index.get_loc(pd.Timestamp(r["entry_date"]))
            w = full.iloc[max(0, pos - 130): pos + 21]
            x = np.arange(len(w))
            o, h, l, c, v = (w[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
            for i in range(len(w)):
                col = "#d33" if c[i] >= o[i] else "#1976d2"
                ax.vlines(x[i], l[i], h[i], color=col, lw=0.8)
                ax.add_patch(Rectangle((x[i] - 0.3, min(o[i], c[i])), 0.6, abs(c[i] - o[i]) + 1e-9, color=col))
            for n, col in ((50, "orange"), (150, "green"), (200, "purple")):
                sma = full["Close"].rolling(n).mean().iloc[max(0, pos - 130): pos + 21].values
                ax.plot(x, sma, color=col, lw=1, label=f"SMA{n}")
            ei = pos - max(0, pos - 130)
            ax.axvline(ei, color="k", ls="--", lw=0.8)
            ax.axvspan(ei, len(w), color="gray", alpha=0.08)
            if r["pivot"]:
                ax.axhline(r["pivot"]["pivot"], color="red", ls=":", lw=1, label=f"피봇 {r['pivot']['pivot']:.2f}")
            ax2 = ax.twinx()
            ax2.bar(x, v, color=["#d33" if c[i] >= o[i] else "#1976d2" for i in range(len(w))], alpha=0.25, width=0.6)
            ax2.set_ylim(0, v.max() * 4)
            ax2.set_yticks([])
            s4 = {k: str(v).replace("\u2212", "-") for k, v in (r["s4_core5"]["notes"] if r["s4_core5"] else {}).items()}
            if s4.get("s4_4", "").count("→") > 5:
                s4["s4_4"] = "… → " + " → ".join(s4["s4_4"].split(" → ")[-5:])
            ax.set_title(f"{r['ticker']} 진입 {r['entry_date']} (WP {r['wp_pnl_pct']:+.1%}, {r['exit_reason']}) · "
                         f"S1 {r['tt']['n_passed'] if r['tt'] else '-'}/7 RS{round(r['rs']) if r['rs'] else '-'} · 축소 {s4.get('s4_4', '')} · {s4.get('s4_7', '')} · {s4.get('s4_9', '')}", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)
            ax.set_xlim(-1, len(w))
            ticks = list(range(0, len(w), 20))
            ax.set_xticks(ticks)
            ax.set_xticklabels([w.index[i].strftime("%m-%d") for i in ticks], fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/trades_{g // 4 + 1:02d}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
