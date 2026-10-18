"""44 스크리너 내보내기 CSV(티커, 산업군, RS …) 의 후보 전부를 **기준일 종가 기준** 미너비니 STAGE 0~4 로 재판정
+ 눈판독용 차트 PNG + 종목별 도시에(주봉 26주·일봉 15일·헤드라인) 생성.

    docker compose exec backtester python minervini_watchlist_check_cli.py \\
        '{"csv": "/app/sweep_results/minervini_screener_2026-10-13.csv", "as_of": "2026-10-12",
          "out": "/app/sweep_results/minervini_watchlist_check.json", "report_md": "/app/sweep_results/minervini_watchlist_check.md",
          "charts": "/app/sweep_results/minervini_watchlist_imgs", "dossiers": "/app/sweep_results/minervini_watchlist_dossiers.md",
          "live_json": "/app/sweep_results/live_2026-10-13.json"}'

판정 소스: STAGE 0 = SPY·QQQ(`evaluate_market`) + 0-3 은 `live_json`(전체 유니버스 live 결과)이 있으면 그 값,
STAGE 1 = `evaluate_trend_template` + RS 는 CSV 값(스크리너 전체 유니버스 백분위), STAGE 2 = Alpha Vantage 분기·연간·서프라이즈·재무상태표 + 뉴스 가이던스,
STAGE 3 = `manual_auto`(0-4·3-1·3-2·3-3) + 3-4 CSV 후보 내 산업군 RS 순위 + 3-5 CSV 산업군 집계(≥2), STAGE 4 = `pivot_levels` + `stage4`(엄격·핵심 5/7).
`minervini_trades_check_cli.py`(거래 로그용)와 짝 — 이쪽은 '진입 전'이 아니라 '지금' 을 본다.
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
from pages._shared.growth_sources import build_growth_adapters, fetch_growth_snapshot
from strategy.adapters.minervini_full_checklist import stage2, stage3, stage4
from strategy.adapters.minervini_manual_auto import industry_rank, manual_auto
from strategy.adapters.minervini_scan import fetch_with_retry
from strategy.adapters.minervini_screener import (
    distribution_days,
    evaluate_annual,
    evaluate_balance,
    evaluate_growth,
    evaluate_market,
    evaluate_surprise,
    evaluate_trend_template,
    pivot_levels,
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


def _table(d: pd.DataFrame) -> str:
    cols = [str(c) for c in d.columns]
    rows = [["" if v is None else str(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def _mark(v) -> str:
    return "✅" if v is True else "❌" if v is False else "—"


def _col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    wl = pd.read_csv(cfg["csv"], encoding="utf-8-sig")
    wl["ticker"] = _col(wl, "티커", "Ticker").astype(str).str.strip()
    wl["csv_industry"] = _col(wl, "산업군", "industry") if _col(wl, "산업군", "industry") is not None else "미분류"
    wl["csv_rs"] = pd.to_numeric(_col(wl, "RS", "rs"), errors="coerce")
    as_of = date.fromisoformat(cfg["as_of"]) if cfg.get("as_of") else date.today()
    md = MongoDayCacheAdapter(AlphaVantageAdapter(), source_name="alphavantage")
    growth = build_growth_adapters()
    bal = next((a for a in growth if hasattr(a, "fetch_balance_sheet")), None)
    try:
        from data.adapters.alphavantage_news import AlphaVantageNewsAdapter
        news = AlphaVantageNewsAdapter()
    except Exception:  # noqa: BLE001
        news = None
    tickers = list(dict.fromkeys(wl["ticker"]))
    start = as_of - timedelta(days=720)
    _log(f"{len(tickers)} 종목 일봉 {start}~{as_of}")
    frames: dict[str, pd.DataFrame] = {}
    from concurrent.futures import ThreadPoolExecutor

    def _f(t):
        for s0 in (start, as_of - timedelta(days=400)):   # 상장 2년 미만 종목은 짧은 창으로 재시도
            try:
                return t, _clip(fetch_with_retry(md, t, s0, as_of), as_of)
            except Exception:  # noqa: BLE001
                continue
        return t, None
    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 6))) as ex:
        for t, df in ex.map(_f, tickers + ["SPY", "QQQ"]):
            if df is not None and len(df):
                frames[t] = df
    _log(f"일봉 {len(frames)}/{len(tickers) + 2}")

    # STAGE 0
    verdicts = {s: evaluate_market(frames[s], symbol=s) for s in ("SPY", "QQQ") if s in frames}
    dist = {s: distribution_days(frames[s]) for s in verdicts}
    live = json.load(open(cfg["live_json"])) if cfg.get("live_json") else None
    nh = nl = None
    if live:
        nh, nl = live["market"].get("nh"), live["market"].get("nl")
    market = {s: {"ok": v.ok if v else None, "trend_ok": v.trend_ok if v else None, "short_ok": v.short_ok if v else None,
                  "note": v.note if v else "", "dist": v.dist_days if v else dist.get(s), "close": float(frames[s]["Close"].iloc[-1])}
              for s, v in verdicts.items()}
    s0_ok = all(v["ok"] for v in market.values()) and (nh is None or nl is None or nh > nl)
    _log(f"S0 {'✅' if s0_ok else '❌'} " + " · ".join(f"{s} {v['note']} 분산 {v['dist']}" for s, v in market.items()) + f" · 0-3 {nh}/{nl}")

    # 3-4 / 3-5 (CSV 후보군 기준)
    ranks = industry_rank([(r.ticker, r.csv_industry, r.csv_rs) for r in wl.itertuples()])
    ind_counts = wl[wl["csv_industry"] != "미분류"]["csv_industry"].value_counts()
    leading = set(ind_counts[ind_counts >= 2].head(10).index)

    snaps, bals = {}, {}
    for t in tickers:
        try:
            snaps[t] = fetch_growth_snapshot(t, growth)
        except Exception as e:  # noqa: BLE001
            _log(f"  {t} 펀더멘털 실패 {type(e).__name__}")
            snaps[t] = None
        if bal is not None:
            try:
                bals[t] = bal.fetch_balance_sheet(t)
            except Exception:  # noqa: BLE001
                bals[t] = []

    rows = []
    for r in wl.itertuples():
        t = r.ticker
        df = frames.get(t)
        if df is None or len(df) < 60:
            rows.append({"ticker": t, "error": "일봉 없음"})
            continue
        tt = evaluate_trend_template(df)
        snap = snaps.get(t)
        q = snap.quarters if snap else []
        gr = evaluate_growth(q, as_of) if q else None
        an = evaluate_annual(snap.annuals, as_of) if snap and snap.annuals else None
        sr = evaluate_surprise(q, as_of) if q else None
        br = evaluate_balance(bals.get(t, []), q, as_of) if (bals.get(t) and q) else None
        items = None
        if news is not None:
            try:
                items = news.fetch_news(t, as_of - timedelta(days=120), as_of)
            except Exception:  # noqa: BLE001
                items = None
        titles = [n.title for n in items] if items is not None else None
        recent_news = [f"{n.published_at.date()} {n.title}" for n in (items or []) if n.published_at.date() >= as_of - timedelta(days=45)]
        s2 = stage2(gr, an, sr, br, titles)
        pl = pivot_levels(df)
        ma = manual_auto(df, frames["QQQ"], pl) if pl is not None and "QQQ" in frames else None
        m = dict(ma.checks) if ma else {}
        m["m3_4"] = ranks.get(t, (None, ""))[0]
        dvol = float((df["Close"] * df["Volume"]).tail(20).mean())
        industry = r.csv_industry if r.csv_industry and r.csv_industry != "미분류" else (snap.industry if snap and snap.industry else "미분류")
        s3 = stage3(m, industry, leading, liquidity_ok=dvol >= 5e6)
        s4s = stage4(df, pl) if pl is not None else None
        s4 = stage4(df, pl, min_core=5) if pl is not None else None
        c = float(df["Close"].iloc[-1])
        last_q = max((x for x in q if x.report_date and x.report_date <= as_of), key=lambda x: x.report_date, default=None)
        latest = [x for x in sorted(q, key=lambda x: x.fiscal_date) if x.effective_report_date <= as_of][-6:]
        rows.append({
            "ticker": t, "as_of": str(as_of), "industry": industry, "sector": snap.sector if snap else None,
            "csv": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r._asdict().items() if k not in ("Index", "ticker", "csv_industry", "csv_rs")},
            "rs": float(r.csv_rs) if pd.notna(r.csv_rs) else None, "close": c,
            "tt": {**asdict(tt), "n_passed": sum(tt.checks.values())} if tt else None,
            "growth": asdict(gr) if gr else None, "annual": asdict(an) if an else None,
            "surprise": asdict(sr) if sr else None, "balance": asdict(br) if br else None,
            "quarters": [{"fiscal": str(x.fiscal_date), "report": str(x.report_date), "eps": x.eps_actual, "est": x.eps_estimate, "revenue": x.revenue,
                          "net_income": x.net_income} for x in latest],
            "last_report": str(last_q.report_date) if last_q else None,
            "news_recent": recent_news[-25:], "news_n120": len(titles) if titles is not None else None,
            "s2": asdict(s2), "s3": asdict(s3), "s4_strict": asdict(s4s) if s4s else None, "s4_core5": asdict(s4) if s4 else None,
            "manual_auto": {"checks": m, "notes": {**(ma.notes if ma else {}), "m3_4": ranks.get(t, (None, ""))[1]}},
            "pivot": asdict(pl) if pl else None,
            "ret_10d": float(c / df["Close"].iloc[-11] - 1) if len(df) > 11 else None,
            "ret_1d": float(c / df["Close"].iloc[-2] - 1), "vol_ratio_1d": float(df["Volume"].iloc[-1] / df["Volume"].tail(50).mean()),
            "vs_sma50": float(c / df["Close"].rolling(50).mean().iloc[-1] - 1) if len(df) >= 50 else None,
            "vs_52w_high": float(c / df["High"].tail(252).max() - 1),
            "dollar_vol_m": dvol / 1e6,
        })
        _log(f"{t:5s} S1 {rows[-1]['tt']['n_passed'] if tt else '-'}/7 RS{r.csv_rs:.0f} | S2 {s2.ok} | S3 {s3.ok} | S4 {s4.ok if s4 else None} "
             f"| 피봇 {pl.dist_to_pivot:+.1%} 깊이 {pl.base_depth if pl.base_depth is None else round(pl.base_depth * 100)}% {pl.base_weeks and round(pl.base_weeks, 1)}주")
    out = {"config": cfg, "as_of": str(as_of), "market": market, "nh": nh, "nl": nl, "s0_ok": s0_ok,
           "leading_industries": sorted(leading), "rows": rows}
    with open(cfg.get("out", "/tmp/minervini_watchlist_check.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    report = _report(out)
    print(report)
    if cfg.get("report_md"):
        open(cfg["report_md"], "w").write(report)
    if cfg.get("dossiers"):
        open(cfg["dossiers"], "w").write(_dossiers(rows, frames))
    if cfg.get("charts"):
        _charts(rows, frames, cfg["charts"])


def _verdict(r: dict) -> str:
    s1 = bool(r["tt"] and r["tt"]["passed"] and (r["rs"] or 0) >= 70)
    if not s1:
        return "탈락 S1"
    if not r["s2"]["ok"]:
        return "탈락 S2"
    if not r["s3"]["ok"]:
        return "탈락 S3"
    if not (r["s4_core5"] or {}).get("ok"):
        return "워치(S4 미완)"
    return "매수 후보(S4 5/7)" if not (r["s4_strict"] or {}).get("ok") else "매수 후보(S4 전부)"


def _report(out: dict) -> str:
    rows = out["rows"]
    mk = out["market"]
    lines = [f"# 미너비니 체크리스트 — 스크리너 후보 재판정 ({out['as_of']} 종가 기준)", "",
             "STAGE 0: " + " · ".join(f"{s} {v['note']} (분산 {v['dist']})" for s, v in mk.items())
             + f" · 0-3 신고가/신저가 {out['nh']}/{out['nl']} → {'✅' if out['s0_ok'] else '❌'}",
             f"3-5 주도 산업군(CSV 2종목↑): {', '.join(out['leading_industries'])}", ""]
    tab = []
    for r in rows:
        if r.get("error"):
            tab.append({"종목": r["ticker"], "오류": r["error"]})
            continue
        tt = r["tt"]
        s1 = bool(tt and tt["passed"] and (r["rs"] or 0) >= 70)
        s2i, s3i, s4i = r["s2"]["items"], r["s3"]["items"], (r["s4_core5"] or {}).get("items", {})
        s2_core = sum(s2i.get(k) is True for k in ("s2_1", "s2_2", "s2_3", "s2_4", "s2_7"))
        s2_alert = [k[3:] for k in ("s2_8", "s2_9", "s2_10") if s2i.get(k) is False]
        s4_core = sum(v is True for k, v in s4i.items() if k not in ("s4_3", "s4_10", "s4_11"))
        s4_guard = [k[3:] for k in ("s4_3", "s4_10", "s4_11") if s4i.get(k) is False]
        pl = r["pivot"] or {}
        tab.append({
            "종목": r["ticker"], "산업군": r["industry"][:22], "RS": f"{r['rs']:.0f}" if r["rs"] is not None else "—",
            "종가": f"{r['close']:.2f}", "피봇(거리)": f"{pl.get('pivot', 0):.2f} ({pl.get('dist_to_pivot', 0):+.1%})",
            "S1": f"{tt['n_passed'] if tt else '-'}/7" + (" ✅" if s1 else " ❌"),
            "S2": (f"{s2_core}/5" if s2i.get("s2_1") is not None else "미확인") + (f" ⛔{','.join(s2_alert)}" if s2_alert else "") + (" ✅" if r["s2"]["ok"] else " ❌"),
            "S3": f"{_mark(s3i.get('s3_1'))}{_mark(s3i.get('s3_2'))}{_mark(s3i.get('s3_3'))}{_mark(s3i.get('s3_4'))}{_mark(s3i.get('s3_5'))}" + (" ✅" if r["s3"]["ok"] else " ❌"),
            "S4": f"{s4_core}/7" + (f" ⛔{','.join(s4_guard)}" if s4_guard else "") + (" ✅" if (r["s4_core5"] or {}).get("ok") else " ❌"),
            "베이스": f"{pl.get('base_weeks') and round(pl['base_weeks'], 1)}주/{pl.get('base_depth') and round(pl['base_depth'] * 100)}%" if pl else "—",
            "축소": (r["s4_core5"] or {}).get("notes", {}).get("s4_4", "")[-40:].replace("−", "−"),
            "드라이업": f"{pl.get('volume_dryup', 0):.2f}", "10일": f"{r['ret_10d']:+.0%}" if r["ret_10d"] is not None else "—",
            "거래대금": f"${r['dollar_vol_m']:.0f}M", "실적": r["last_report"] or "—", "자동": _verdict(r),
        })
    lines += [_table(pd.DataFrame(tab)), ""]
    ok = [r for r in rows if not r.get("error")]
    cnt = {"S1": sum(bool(r["tt"] and r["tt"]["passed"] and (r["rs"] or 0) >= 70) for r in ok), "S2": sum(r["s2"]["ok"] for r in ok),
           "S3": sum(r["s3"]["ok"] for r in ok), "S2+3": sum(r["s2"]["ok"] and r["s3"]["ok"] for r in ok),
           "S2+3+S4(5/7)": sum(r["s2"]["ok"] and r["s3"]["ok"] and bool((r["s4_core5"] or {}).get("ok")) for r in ok),
           "S4(5/7)만": sum(bool((r["s4_core5"] or {}).get("ok")) for r in ok)}
    lines += ["## 깔때기", "", _table(pd.DataFrame([{"후보": len(ok), **cnt}])), "",
              "S2+3 통과: " + ", ".join(r["ticker"] for r in ok if r["s2"]["ok"] and r["s3"]["ok"]),
              "S4(5/7) 통과: " + ", ".join(r["ticker"] for r in ok if (r["s4_core5"] or {}).get("ok")), ""]
    return "\n".join(lines)


def _dossiers(rows: list[dict], frames: dict) -> str:
    lines = ["# 종목별 판독 자료 (기준일 종가까지)", ""]
    for r in rows:
        if r.get("error"):
            continue
        t = r["ticker"]
        hist = frames[t]
        w = hist.tail(135).resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        v50 = float(hist["Volume"].tail(50).mean())
        lines += [f"## {t} — {r['industry']} · RS {r['rs'] and round(r['rs'])} · 종가 {r['close']:.2f} · 피봇 {r['pivot']['pivot']:.2f} ({r['pivot']['dist_to_pivot']:+.1%})", "",
                  "주봉 26주 (거래량 = 50일 평균 주간 대비)", "", "| 주말 | 고 | 저 | 종 | 거래량 |", "|---|---|---|---|---|"]
        for d, x in w.tail(26).iterrows():
            lines.append(f"| {d:%m-%d} | {x.High:.2f} | {x.Low:.2f} | {x.Close:.2f} | {x.Volume / (v50 * 5):.2f} |")
        lines += ["", "최근 15일", "", "| 일 | 시 | 고 | 저 | 종 | 등락 | 거래량 |", "|---|---|---|---|---|---|---|"]
        tail = hist.tail(16)
        for i in range(1, len(tail)):
            x, p = tail.iloc[i], tail.iloc[i - 1]
            lines.append(f"| {tail.index[i]:%m-%d} | {x.Open:.2f} | {x.High:.2f} | {x.Low:.2f} | {x.Close:.2f} | {x.Close / p.Close - 1:+.1%} | {x.Volume / v50:.2f} |")
        sma = {n: float(hist["Close"].rolling(n).mean().iloc[-1]) for n in (50, 150, 200)}
        g = r["growth"] or {}
        lines += ["", f"- SMA50/150/200: {sma[50]:.2f} / {sma[150]:.2f} / {sma[200]:.2f} · 52주 고가 대비 {r['vs_52w_high']:+.1%} · 50일선 대비 {r['vs_sma50']:+.1%} · 10일 {r['ret_10d']:+.1%} · 거래대금 ${r['dollar_vol_m']:.0f}M",
                  f"- 성장(2-1~2-4): {g.get('checks')} · 서프라이즈 {(r['surprise'] or {}).get('streak')}연속 최근 {(r['surprise'] or {}).get('latest_beat')} · 연간 {(r['annual'] or {}).get('checks')} 감속 {(r['annual'] or {}).get('decel_warning')} · 재무 {(r['balance'] or {}).get('passed')}",
                  f"- 분기: " + " | ".join(f"{x['fiscal'][:7]} EPS {x['eps']} 매출 {x['revenue'] and round(x['revenue'] / 1e6)}M" for x in r["quarters"]),
                  f"- S2 {r['s2']['items']} {r['s2']['notes']}", f"- S3 {r['s3']['items']} · {r['manual_auto']['notes']}",
                  f"- S4(5/7) {(r['s4_core5'] or {}).get('items')} · " + " · ".join(f"{k[1:]} {v}" for k, v in (r["s4_core5"] or {}).get("notes", {}).items() if v),
                  f"- 피봇 방식 {r['pivot']['method']} · {r['pivot']['status']} · 축소 {r['pivot']['contractions']}",
                  "- 최근 45일 헤드라인:"] + [f"  - {n}" for n in r["news_recent"]] + [""]
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
            w = full.tail(160)
            x = np.arange(len(w))
            o, h, l, c, v = (w[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
            for i in range(len(w)):
                col = "#d33" if c[i] >= o[i] else "#1976d2"
                ax.vlines(x[i], l[i], h[i], color=col, lw=0.8)
                ax.add_patch(Rectangle((x[i] - 0.3, min(o[i], c[i])), 0.6, abs(c[i] - o[i]) + 1e-9, color=col))
            for n, col in ((50, "orange"), (150, "green"), (200, "purple")):
                sma = full["Close"].rolling(n).mean().tail(160).values
                ax.plot(x, sma, color=col, lw=1, label=f"SMA{n}")
            if r["pivot"]:
                ax.axhline(r["pivot"]["pivot"], color="red", ls=":", lw=1, label=f"피봇 {r['pivot']['pivot']:.2f}")
            ax2 = ax.twinx()
            ax2.bar(x, v, color=["#d33" if c[i] >= o[i] else "#1976d2" for i in range(len(w))], alpha=0.25, width=0.6)
            ax2.set_ylim(0, v.max() * 4)
            ax2.set_yticks([])
            s4 = {k: str(v).replace("−", "-") for k, v in (r["s4_core5"]["notes"] if r["s4_core5"] else {}).items()}
            if s4.get("s4_4", "").count("→") > 5:
                s4["s4_4"] = "… → " + " → ".join(s4["s4_4"].split(" → ")[-5:])
            ax.set_title(f"{r['ticker']} {r['industry'][:20]} RS{round(r['rs']) if r['rs'] else '-'} · S1 {r['tt']['n_passed'] if r['tt'] else '-'}/7 "
                         f"S2 {'O' if r['s2']['ok'] else 'X'} S3 {'O' if r['s3']['ok'] else 'X'} S4 {'O' if (r['s4_core5'] or {}).get('ok') else 'X'} · "
                         f"축소 {s4.get('s4_4', '')} · {s4.get('s4_3', '')} · {s4.get('s4_7', '')} · {s4.get('s4_9', '')}", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)
            ax.set_xlim(-1, len(w))
            ticks = list(range(0, len(w), 20))
            ax.set_xticks(ticks)
            ax.set_xticklabels([w.index[i].strftime("%m-%d") for i in ticks], fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/watch_{g // 4 + 1:02d}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
