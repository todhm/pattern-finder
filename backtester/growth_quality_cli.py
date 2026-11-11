"""성장주 "재무제표 읽기" 품질 전략 — 자료 수집·특징·분석 카드 배치 (설계서 성장주_보유_전략_설계_2026-10.md §11).

    # 1) 재무제표 3종 수집 (Alpha Vantage, 종목당 손익·발표일(성장 캐시 재사용) + 재무상태표 1콜 + 현금흐름표 1콜)
    docker compose exec backtester python growth_quality_cli.py fetch '{"universe": "sp500+nasdaq100", "workers": 2, "pace": 1.0,
        "out": "sweep_results/growth_statements_2026-11-09.parquet"}'
    # 2) 분기별 체크리스트 특징 (point-in-time: 발표일 + lag 거래일부터 유효)
    docker compose exec backtester python growth_quality_cli.py features '{"statements": "...parquet", "out": "sweep_results/growth_quality_features_2026-11-09.parquet"}'
    # 3) 항목별 예측력 (리스트 종목·유니버스 전체, 체크 통과/실패 뒤 12개월 수익)
    docker compose exec backtester python growth_quality_cli.py study '{"features": "...", "universe": "sp500+nasdaq100", "start": "2016-01-04", "end": "2026-10-24", "out_md": "..."}'
    # 4) 종목·기준일의 재무제표 분석 카드 (사람이 읽는 형태)
    docker compose exec backtester python growth_quality_cli.py cards '{"features": "...", "picks": {"2024-01-02": ["NVDA", "..."]}, "out_md": "..."}'

특징 parquet 은 growth_hold_cli.py cfg "quality_features" 로 aux 열(q_score·q_rev·q_fcfm…)이 되어
rank quality|combo:q_score+mom6, require "q_score>=8" 에 쓰인다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from data.adapters.wikipedia_universe import default_universe_provider
from growth_dip_scalp_cli import log


def _tickers(uni) -> list[str]:
    prov = default_universe_provider()
    if isinstance(uni, list):
        return list(dict.fromkeys(uni))
    if isinstance(uni, dict):          # {"include": ["nasdaq_full"], "exclude": ["sp500", "nasdaq100"]} — 중소형(설계서 §12)
        inc = [t for u in uni.get("include", []) for t in prov.get_tickers(u)]
        exc = {t for u in uni.get("exclude", []) for t in prov.get_tickers(u)}
        return [t for t in dict.fromkeys(inc) if t not in exc]
    if "+" in uni:
        return list(dict.fromkeys(t for u in uni.split("+") for t in prov.get_tickers(u.strip())))
    return prov.get_tickers(uni)


def cmd_fetch(cfg: dict) -> None:
    from data.adapters.alphavantage_growth_fundamentals import AlphaVantageGrowthFundamentalsAdapter
    from data.adapters.alphavantage_statements import AlphaVantageStatementsAdapter
    tickers = _tickers(cfg.get("universe", "sp500+nasdaq100"))
    if cfg.get("max_tickers"):
        tickers = tickers[: int(cfg["max_tickers"])]
    if cfg.get("only_cached"):                                             # 성장 캐시(v2)가 있는 종목만 (중소형: 손익 3콜 절약)
        import glob, os
        have = {os.path.basename(f).split(".v2.json")[0] for f in glob.glob("/tmp/pattern-finder-cache/growth-fundamentals-av/*.v2.json")}
        tickers = [t for t in tickers if t.replace("/", "_").replace(".", "_") in have]
    pace = float(cfg.get("pace", 1.0))
    growth = AlphaVantageGrowthFundamentalsAdapter(cache_ttl_days=float(cfg.get("growth_ttl_days", 7)))
    stm = AlphaVantageStatementsAdapter(growth=growth, pace_s=pace, cache_ttl_days=float(cfg.get("ttl_days", 30)))
    rows: list[dict] = []; t0 = time.time(); done = [0]; empty: list[str] = []

    def one(t: str) -> list[dict]:
        try:
            snap = growth.fetch(t)                                        # v2 캐시 갱신(EARNINGS·INCOME·OVERVIEW, TTL 7일)
            if snap.quarters:
                time.sleep(pace)
            qs = stm.fetch_statements(t)
            sector = snap.sector
        except Exception as e:  # noqa: BLE001
            log(f"  {t}: 실패 {e!r}"); qs = []
        done[0] += 1
        if done[0] % 25 == 0:
            log(f"  재무제표 {done[0]}/{len(tickers)} ({time.time() - t0:.0f}s)")
        if not qs:
            empty.append(t)
        return [{**q.model_dump(), "sector": sector} for q in qs]

    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 2))) as ex:
        for part in ex.map(one, tickers):
            rows += part
    df = pd.DataFrame(rows)
    for c in ("fiscal_date", "report_date"):
        df[c] = pd.to_datetime(df[c])
    df.to_parquet(cfg["out"], index=False)
    has_bs = df["total_assets"].notna().groupby(df["symbol"]).any().sum(); has_cf = df["operating_cf"].notna().groupby(df["symbol"]).any().sum()
    log(f"저장 {cfg['out']}: 종목 {df.symbol.nunique()}/{len(tickers)} · 분기 행 {len(df)} · 재무상태표 있는 종목 {has_bs} · 현금흐름표 {has_cf} · 비어 있는 종목 {len(empty)} {empty[:30]} ({time.time() - t0:.0f}s)")


def cmd_features(cfg: dict) -> None:
    from strategy.adapters.fundamental_quality import QualityParams, quality_features
    st = pd.read_parquet(cfg["statements"])
    p = QualityParams(**cfg.get("params", {}))
    feats = quality_features(st, p)
    feats.to_parquet(cfg["out"], index=False)
    ok = feats["q_avail"] >= p.min_available
    log(f"저장 {cfg['out']}: 종목 {feats.symbol.nunique()} · 분기 행 {len(feats)} · 판정 가능 행 {int(ok.sum())} · q_score 분포(판정 가능) "
        + " ".join(f"{k}:{v}" for k, v in feats.loc[ok, 'q_score'].round().astype(int).value_counts().sort_index().items()))


def cmd_study(cfg: dict) -> None:
    from strategy.adapters.fundamental_quality import CHECKS, study_forward
    from data.adapters.yfinance_bulk import download_many
    feats = pd.read_parquet(cfg["features"])
    tickers = _tickers(cfg.get("universe", "sp500+nasdaq100"))
    start, end = pd.Timestamp(cfg.get("start", "2016-01-04")), pd.Timestamp(cfg.get("end", "2026-10-24"))
    prices = download_many(tickers + ["QQQ"], (start - pd.Timedelta(days=420)).date(), end.date())
    qqq = prices.pop("QQQ")
    lists = None
    if cfg.get("growth_list", True):
        from growth_dip_scalp_cli import load_snapshots
        from strategy.adapters.growth_dip_scalp import ScalpRules, precompute, weekly_lists
        from strategy.adapters.minervini_screener import evaluate_growth
        cal = pd.to_datetime(qqq.index)
        data = precompute({t: d for t, d in prices.items() if t in tickers}, cal)
        snaps, _ = load_snapshots(list(data))
        gc: dict = {}

        def growth_fn(t: str, as_of: date) -> bool:
            k = (t, as_of)
            if k not in gc:
                g = evaluate_growth(snaps.get(t, []), as_of, eps_min_yoy=0.25, rev_min_yoy=0.25, n_required=3)
                gc[k] = bool(g.data_available and g.passed)
            return gc[k]
        lists = weekly_lists(list(data), cal, growth_fn, data, ScalpRules(trend_filter=None, max_vol_ratio=None, min_days_to_earnings=0), start, end)
    out = study_forward(feats, prices, qqq, start, end, horizon_days=int(cfg.get("horizon", 252)), lists=lists)
    with open(cfg["out_md"], "w") as fh:
        fh.write(out["md"])
    with open(cfg["out_md"].replace(".md", ".json"), "w") as fh:
        json.dump(out["tables"], fh, ensure_ascii=False, indent=1, default=str)
    log(f"저장 {cfg['out_md']}")


def cmd_cards(cfg: dict) -> None:
    from strategy.adapters.fundamental_quality import explain_card, row_as_of
    feats = pd.read_parquet(cfg["features"])
    L = [f"# 재무제표 읽기 카드 — {cfg.get('title', '')}", "", "각 카드는 **기준일에 알 수 있었던 마지막 분기 재무제표**(발표일 + 지연 뒤)로 판정한 것. 금액 단위 $M, 비율은 TTM(최근 4분기 합). ✅ 통과 · ❌ 실패 · ➖ 자료 없음.", ""]
    for day, tks in cfg["picks"].items():
        L.append(f"## 기준일 {day}")
        for t in tks:
            r = row_as_of(feats, t, pd.Timestamp(day))
            L.append(explain_card(t, r, pd.Timestamp(day)))
            L.append("")
    with open(cfg["out_md"], "w") as fh:
        fh.write("\n".join(L))
    log(f"저장 {cfg['out_md']}")



# ---------------------------------------------------------------------------
# 블라인드 수동 분석 (설계서 §13.1) · 승자 시점별 특징 (§13.2)
# ---------------------------------------------------------------------------

def _fmt_m(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v / 1e6:,.0f}"


def _fmt_p(v, d=0) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}%}"


def _fmt_x(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}×"


def _price_on(close: pd.Series, day: pd.Timestamp):
    s = close[close.index <= day]
    return float(s.iloc[-1]) if len(s) else np.nan


def _blind_card(cid: str, st: pd.DataFrame, feats: pd.DataFrame, as_of: pd.Timestamp, close: pd.Series, sector: str | None, small: bool) -> str:
    """기준일까지(발표일 ≤ 기준일) 재무제표·파생·밸류에이션을 익명으로."""
    q = st[(pd.to_datetime(st["report_date"]).fillna(pd.to_datetime(st["fiscal_date"]) + pd.Timedelta(days=90)) <= as_of)].sort_values("fiscal_date")
    f = feats[feats["effective"] <= as_of].sort_values("effective")
    if q.empty or f.empty:
        return ""
    L = [f"## {cid} — 섹터 {sector or '?'} · {'중소형(나스닥 비지수)' if small else '대형(S&P500/나스닥100)'} · 기준일 {as_of.date()} · 마지막 분기 {q.iloc[-1]['fiscal_date'].date()}", ""]
    inc = q.tail(12)
    L += ["**손익계산서 (분기, $M)**", "", "| 분기 | 매출 | 매출총이익 | 영업이익 | 순이익 | R&D | SG&A | D&A |", "|---|---|---|---|---|---|---|---|"]
    for _, r in inc.iterrows():
        L.append(f"| {r['fiscal_date'].date()} | {_fmt_m(r['revenue'])} | {_fmt_m(r['gross_profit'])} | {_fmt_m(r['operating_income'])} | {_fmt_m(r['net_income'])} | {_fmt_m(r['rnd'])} | {_fmt_m(r['sga'])} | {_fmt_m(r['da'])} |")
    bs = q.tail(5)
    L += ["", "**재무상태표 (분기 말, $M; 주식 수 M)**", "", "| 분기 | 현금성 | 차입 | 자본 | 매출채권 | 재고 | 영업권 | 이연매출 | 총자산 | 유동비율 | 주식 수 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in bs.iterrows():
        cr = r["total_current_assets"] / r["total_current_liabilities"] if (pd.notna(r["total_current_assets"]) and pd.notna(r["total_current_liabilities"]) and r["total_current_liabilities"]) else np.nan
        L.append(f"| {r['fiscal_date'].date()} | {_fmt_m(r['cash_and_st_investments'])} | {_fmt_m(r['total_debt'])} | {_fmt_m(r['equity'])} | {_fmt_m(r['receivables'])} | {_fmt_m(r['inventory'])} | {_fmt_m(r['goodwill'])} | {_fmt_m(r['deferred_revenue'])} | {_fmt_m(r['total_assets'])} | {_fmt_x(cr)} | {_fmt_m(r['shares_outstanding'])} |")
    cf = q.tail(8)
    L += ["", "**현금흐름표 (분기, $M)**", "", "| 분기 | 영업CF | capex | 주식보상 | 자사주 | 배당 | 증자 |", "|---|---|---|---|---|---|---|"]
    for _, r in cf.iterrows():
        L.append(f"| {r['fiscal_date'].date()} | {_fmt_m(r['operating_cf'])} | {_fmt_m(r['capex'])} | {_fmt_m(r['sbc'])} | {_fmt_m(r['buybacks'])} | {_fmt_m(r['dividends'])} | {_fmt_m(r['stock_issuance'])} |")
    ft = f.tail(8)
    L += ["", "**파생 지표 (TTM, 발표일 기준 유효)**", "", "| 분기 | 매출 YoY | 분기 매출 YoY | GM | OM | 순이익 YoY | 영업CF/순이익 | FCF 마진 | SBC/매출 | 발생액/자산 | DSO 변화 | ROIC | 순차입/EBITDA | 주식 YoY | 12항목 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ft.iterrows():
        nd = "순현금" if (pd.notna(r["netdebt"]) and r["netdebt"] <= 0) else _fmt_x(r["nd_ebitda"])
        L.append(f"| {r['fiscal_date'].date()} | {_fmt_p(r['q_rev'])} | {_fmt_p(r['q_rev_yoy'])} | {_fmt_p(r['q_gm'])} | {_fmt_p(r['q_om'])} | {_fmt_p(r['q_ni'])} | {_fmt_x(r['q_ocfni'])} | {_fmt_p(r['q_fcfm'])} | {_fmt_p(r['sbc_rev'])} | {_fmt_p(r['accruals'])} | {_fmt_p(r['dso_chg'])} | {_fmt_p(r['roic'])} | {nd} | {_fmt_p(r['sh_chg'], 1)} | {r['q_score']:.0f}/12 ({int(r['q_avail'])}) |")
    last = f.iloc[-1]; px = _price_on(close, as_of)
    sh = last["shares"]; mcap = px * sh if pd.notna(sh) else np.nan
    per = mcap / last["ni_ttm"] if (pd.notna(mcap) and pd.notna(last["ni_ttm"]) and last["ni_ttm"] > 0) else np.nan
    ps = mcap / last["rev_ttm"] if pd.notna(mcap) and pd.notna(last["rev_ttm"]) and last["rev_ttm"] > 0 else np.nan
    ev = mcap + (last["netdebt"] if pd.notna(last["netdebt"]) else 0)
    ev_ebitda = ev / last["ebitda_ttm"] if pd.notna(last["ebitda_ttm"]) and last["ebitda_ttm"] > 0 else np.nan
    pfcf = mcap / last["fcf_ttm"] if pd.notna(last["fcf_ttm"]) and last["fcf_ttm"] > 0 else np.nan
    c1y = close[(close.index <= as_of) & (close.index > as_of - pd.Timedelta(days=365))]
    pos52 = (px - c1y.min()) / (c1y.max() - c1y.min()) if len(c1y) > 20 and c1y.max() > c1y.min() else np.nan
    r1y = px / c1y.iloc[0] - 1 if len(c1y) > 20 else np.nan
    L += ["", f"**밸류에이션 (기준일 종가)**: 시총 ${_fmt_m(mcap)}M · PER(TTM) {_fmt_x(per)} · P/S {_fmt_x(ps)} · EV/EBITDA {_fmt_x(ev_ebitda)} · P/FCF {_fmt_x(pfcf)} · 52주 범위 내 위치 {_fmt_p(pos52)} · 최근 1년 주가 {_fmt_p(r1y)}", ""]
    return "\n".join(L)


def cmd_sample(cfg: dict) -> None:
    """무작위 (종목, 기준일) 표본 → 블라인드 카드 md + 정답표 json + 가격 parquet."""
    from data.adapters.yfinance_bulk import download_many
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    start, end = pd.Timestamp(cfg.get("as_of_start", "2016-06-01")), pd.Timestamp(cfg.get("as_of_end", "2024-10-31"))
    picks = []; cards = []; key = {}
    for grp, prefix, small in (("large", "A", False), ("small", "B", True)):
        st = pd.read_parquet(cfg[f"{grp}_statements"]); ft = pd.read_parquet(cfg[f"{grp}_features"])
        for c in ("fiscal_date", "report_date"):
            st[c] = pd.to_datetime(st[c])
        ft["effective"] = pd.to_datetime(ft["effective"]); ft["fiscal_date"] = pd.to_datetime(ft["fiscal_date"])
        n = int(cfg.get("n_per_group", 20))
        syms = sorted(ft["symbol"].unique()); rng.shuffle(syms)
        chosen = []
        for t in syms:
            if len(chosen) >= n * 3:
                break
            g = ft[ft.symbol == t]
            d0 = g["effective"].sort_values().iloc[7] if len(g) >= 8 else None    # 8번째 분기부터
            if d0 is None or d0 > end:
                continue
            lo = max(d0, start)
            as_of = lo + pd.Timedelta(days=int(rng.integers(0, max(1, (end - lo).days))))
            chosen.append((t, as_of))
        tickers = [t for t, _ in chosen]
        bench = "IWM" if small else "QQQ"
        px = download_many(tickers + [bench], pd.Timestamp("2015-01-01").date(), pd.Timestamp(cfg.get("price_end", "2026-10-24")).date())
        bclose = px[bench]["Close"].astype(float)
        k = 0
        for t, as_of in chosen:
            if k >= n:
                break
            if t not in px or px[t].empty:
                continue
            close = px[t]["Close"].astype(float).dropna()
            if _price_on(close, as_of) != _price_on(close, as_of) or close.index[0] > as_of - pd.Timedelta(days=200):
                continue
            sec = ft[ft.symbol == t]["sector"].dropna()
            card = _blind_card(f"{prefix}{k + 1:02d}", st[st.symbol == t], ft[ft.symbol == t], as_of, close, sec.iloc[0] if len(sec) else None, small)
            if not card:
                continue
            k += 1; cid = f"{prefix}{k:02d}"
            cards.append(card)
            fwd = {}
            for m in (6, 12, 24):
                d2 = as_of + pd.Timedelta(days=int(m * 30.4))
                p0, p1 = _price_on(close, as_of), _price_on(close, d2); b0, b1 = _price_on(bclose, as_of), _price_on(bclose, d2)
                if d2 <= close.index[-1]:
                    fwd[f"r{m}"] = p1 / p0 - 1; fwd[f"b{m}"] = b1 / b0 - 1
            key[cid] = {"ticker": t, "as_of": str(as_of.date()), "group": grp, "bench": bench, **fwd}
            picks.append(cid)
    with open(os.path.join(out_dir, "blind_cards.md"), "w") as fh:
        fh.write(f"# 블라인드 재무제표 카드 — {len(cards)} 종목 (종목명 가림, 기준일까지의 자료만)\n\n" + "\n\n".join(cards))
    with open(os.path.join(out_dir, "answer_key.json"), "w") as fh:
        json.dump(key, fh, ensure_ascii=False, indent=1)
    log(f"표본 {len(picks)} (대형 {sum(1 for c in picks if c.startswith('A'))} · 중소형 {sum(1 for c in picks if c.startswith('B'))}) → {out_dir}/blind_cards.md · answer_key.json")


def cmd_settle(cfg: dict) -> None:
    """판정 json(id → verdict/conviction/thesis/recognized) + 정답표 → 봉인 해제 정산 md."""
    key = json.load(open(cfg["answer_key"])); jd = json.load(open(cfg["judgments"]))
    rows = []
    for cid, k in key.items():
        j = jd.get(cid, {})
        rows.append({"id": cid, "ticker": k["ticker"], "as_of": k["as_of"], "group": k["group"], "bench": k["bench"], "verdict": j.get("verdict", "?"), "conv": j.get("conviction"),
                     "recognized": bool(j.get("recognized")), **{m: k.get(m) for m in ("r6", "b6", "r12", "b12", "r24", "b24")}, "thesis": j.get("thesis", "")})
    df = pd.DataFrame(rows)
    for m in (6, 12, 24):
        df[f"x{m}"] = df[f"r{m}"] - df[f"b{m}"]
    L = [f"# 블라인드 수동 분석 정산 — 표본 {len(df)} (대형 {int((df.group == 'large').sum())} · 중소형 {int((df.group == 'small').sum())})", "",
         "판정은 종목명을 가린 채 기준일까지의 재무제표·밸류에이션만으로 썼다(`blind_cards.md`, `judgments.json`). 아래는 봉인 해제 뒤 앞 6/12/24개월 수익과 벤치(대형 QQQ·중소형 IWM) 대비.", ""]
    for grp_name, sub in (("전체", df), ("대형", df[df.group == "large"]), ("중소형", df[df.group == "small"])):
        L += [f"## {grp_name}", "", "| 판정 | n | 12개월 수익 중앙 | 12개월 벤치 대비 중앙 | 벤치 이김 | 24개월 벤치 대비 중앙 | 6개월 벤치 대비 중앙 |", "|---|---|---|---|---|---|---|"]
        for v in ("매수", "보류", "회피"):
            s = sub[sub.verdict == v]
            if s.empty:
                continue
            L.append(f"| {v} | {len(s)} | {s.r12.median():+.1%} | {s.x12.median():+.1%} | {(s.x12 > 0).mean():.0%} | {s.x24.median() if s.x24.notna().any() else float('nan'):+.1%} | {s.x6.median():+.1%} |")
        L.append("")
    L += ["## 종목별", "", "| id | 티커 | 기준일 | 판정 | 확신 | 알아봄 | 6m | 12m | 24m | 12m 벤치 대비 | 24m 벤치 대비 | 논지(요약) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.0%}"
    for _, r in df.sort_values(["group", "id"]).iterrows():
        L.append(f"| {r.id} | {r.ticker} | {r.as_of} | {r.verdict} | {r.conv or ''} | {'예' if r.recognized else ''} | {f(r.r6)} | {f(r.r12)} | {f(r.r24)} | {f(r.x12)} | {f(r.x24)} | {str(r.thesis)[:90]} |")
    with open(cfg["out_md"], "w") as fh:
        fh.write("\n".join(L))
    df.to_json(cfg["out_md"].replace(".md", ".json"), orient="records", force_ascii=False, indent=1)
    log(f"저장 {cfg['out_md']}")


def cmd_winners(cfg: dict) -> None:
    """최고 3년 구간 수익 상위 승자 + 같은 섹터 대조의 T0 정렬 재무 궤적."""
    from data.adapters.yfinance_bulk import download_many
    ft = pd.read_parquet(cfg["features"]); ft["effective"] = pd.to_datetime(ft["effective"]); ft["fiscal_date"] = pd.to_datetime(ft["fiscal_date"])
    tickers = sorted(ft.symbol.unique())
    px = download_many(tickers + ["QQQ"], pd.Timestamp("2015-06-01").date(), pd.Timestamp(cfg.get("end", "2026-10-24")).date())
    win = int(cfg.get("window_days", 756)); start = pd.Timestamp(cfg.get("start", "2016-01-04"))
    best = []
    for t in tickers:
        if t not in px or px[t].empty:
            continue
        c = px[t]["Close"].astype(float).dropna(); c = c[c.index >= start]
        if len(c) < win + 20:
            continue
        r = c.shift(-win) / c - 1
        i = r.idxmax()
        best.append({"ticker": t, "t0": i, "ret3y": float(r.max()), "ret_total": float(c.iloc[-1] / c.iloc[0] - 1), "sector": ft[ft.symbol == t]["sector"].dropna().iloc[0] if ft[ft.symbol == t]["sector"].notna().any() else None})
    b = pd.DataFrame(best).sort_values("ret3y", ascending=False)
    top = b.head(int(cfg.get("n_winners", 15)))
    # 대조: 같은 섹터, 전체 수익 하위 — 승자 수만큼
    ctrl = b[~b.ticker.isin(top.ticker)].sort_values("ret_total").groupby("sector").head(3).head(len(top))
    metrics = ["q_rev", "q_rev_accel", "q_gm", "q_dgm", "q_om", "q_dom", "q_ocfni", "q_fcfm", "sbc_rev", "q_rnd", "sh_chg", "roic", "inc_roic", "accruals", "q_score", "a_pe", "a_ps"]
    names = {"q_rev": "매출 TTM YoY", "q_rev_accel": "분기 매출 가속(%p)", "q_gm": "GM", "q_dgm": "ΔGM(1년)", "q_om": "OM", "q_dom": "ΔOM(1년)", "q_ocfni": "영업CF/순이익", "q_fcfm": "FCF 마진",
             "sbc_rev": "SBC/매출", "q_rnd": "R&D/매출", "sh_chg": "주식 YoY", "roic": "ROIC", "inc_roic": "증분 ROIC", "accruals": "발생액/자산", "q_score": "12항목 점수", "a_pe": "PER(TTM)", "a_ps": "P/S"}
    offsets = [-8, -4, -2, 0, 2, 4, 8, 12]

    def traj(t: str, t0: pd.Timestamp) -> dict:
        g = ft[ft.symbol == t].sort_values("effective").reset_index(drop=True)
        c = px[t]["Close"].astype(float).dropna()
        k0 = int((g["effective"] <= t0).sum()) - 1
        out = {}
        for o in offsets:
            k = k0 + o
            if 0 <= k < len(g):
                r = g.iloc[k]; d = r["effective"]; p = _price_on(c, d)
                mc = p * r["shares"] if pd.notna(r["shares"]) else np.nan
                vals = {m: r[m] for m in metrics if m in g.columns}
                vals["a_pe"] = mc / r["ni_ttm"] if (pd.notna(mc) and pd.notna(r["ni_ttm"]) and r["ni_ttm"] > 0) else np.nan
                vals["a_ps"] = mc / r["rev_ttm"] if (pd.notna(mc) and pd.notna(r["rev_ttm"]) and r["rev_ttm"] > 0) else np.nan
                vals["px_rel"] = p / _price_on(c, t0) - 1
                vals["date"] = d
                out[o] = vals
        return out

    L = [f"# 크게 성장한 기업의 시점별 재무 특징 — 승자 {len(top)} (최고 3년 구간 수익) vs 같은 섹터 대조 {len(ctrl)} · {start.date()}~{cfg.get('end', '2026-10-24')}", "",
         "T0 = 그 종목의 주가가 가장 많이 오른 3년 구간의 **시작일**. 분기 오프셋은 발표일 기준(T−8 = 상승 시작 2년 전 발표 분기). 승자는 결과를 보고 고른 것이라 **서술적**이다(예측 아님).", "",
         "## 승자 목록", "", "| 티커 | 섹터 | T0 | 3년 수익 | 2016~ 총수익 |", "|---|---|---|---|---|"]
    for _, r in top.iterrows():
        L.append(f"| {r.ticker} | {r.sector} | {r.t0.date()} | {r.ret3y:+.0%} | {r.ret_total:+.0%} |")
    L += ["", "## 대조(같은 섹터, 총수익 하위)", "", "| 티커 | 섹터 | 최고 3년 구간 | 3년 수익 | 총수익 |", "|---|---|---|---|---|"]
    for _, r in ctrl.iterrows():
        L.append(f"| {r.ticker} | {r.sector} | {r.t0.date()} | {r.ret3y:+.0%} | {r.ret_total:+.0%} |")
    TW = {r.ticker: traj(r.ticker, r.t0) for _, r in top.iterrows()}
    TC = {r.ticker: traj(r.ticker, r.t0) for _, r in ctrl.iterrows()}
    pct_metrics = {"q_rev", "q_rev_accel", "q_gm", "q_dgm", "q_om", "q_dom", "q_fcfm", "sbc_rev", "q_rnd", "sh_chg", "roic", "inc_roic", "accruals", "px_rel"}

    def med_table(T: dict, title: str) -> list[str]:
        out = [f"## {title} — 지표별 중앙값 (열 = T0 기준 분기 오프셋)", "", "| 지표 | " + " | ".join(f"T{o:+d}" for o in offsets) + " |", "|---|" + "---|" * len(offsets)]
        for m in metrics + ["px_rel"]:
            vals = []
            for o in offsets:
                xs = [T[t][o][m] for t in T if o in T[t] and m in T[t][o] and pd.notna(T[t][o][m])]
                if not xs:
                    vals.append("—"); continue
                v = float(np.median(xs))
                vals.append(f"{v:+.0%}" if m in pct_metrics else (f"{v:.1f}" if m in ("q_ocfni", "a_pe", "a_ps") else f"{v:.1f}"))
            out.append(f"| {names.get(m, m)} | " + " | ".join(vals) + " |")
        return out + [""]
    L += med_table(TW, "승자") + med_table(TC, "대조")
    L += ["## 승자 개별 궤적 (T−8 → T0 → T+8, 발표일 기준)", ""]
    for _, r in top.iterrows():
        T = TW[r.ticker]
        L += [f"### {r.ticker} ({r.sector}) — T0 {r.t0.date()}, 3년 {r.ret3y:+.0%}", "", "| 오프셋 | 분기(유효일) | 매출 YoY | 가속 | GM | OM | ΔOM | 영업CF/NI | FCF 마진 | SBC/매출 | R&D/매출 | 주식 YoY | ROIC | 점수 | PER | P/S | 주가(T0 대비) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for o in offsets:
            if o not in T:
                continue
            v = T[o]
            L.append(f"| T{o:+d} | {v['date'].date()} | {_fmt_p(v.get('q_rev'))} | {_fmt_p(v.get('q_rev_accel'))} | {_fmt_p(v.get('q_gm'))} | {_fmt_p(v.get('q_om'))} | {_fmt_p(v.get('q_dom'), 1)} | {_fmt_x(v.get('q_ocfni'))} | {_fmt_p(v.get('q_fcfm'))} | {_fmt_p(v.get('sbc_rev'))} | {_fmt_p(v.get('q_rnd'))} | {_fmt_p(v.get('sh_chg'), 1)} | {_fmt_p(v.get('roic'))} | {v.get('q_score', float('nan')):.0f} | {_fmt_x(v.get('a_pe'))} | {_fmt_x(v.get('a_ps'))} | {_fmt_p(v.get('px_rel'))} |")
        L.append("")
    with open(cfg["out_md"], "w") as fh:
        fh.write("\n".join(L))
    json.dump({"winners": top.assign(t0=top.t0.astype(str)).to_dict("records"), "controls": ctrl.assign(t0=ctrl.t0.astype(str)).to_dict("records"),
               "traj_winners": {t: {str(o): {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in d.items()} for o, d in T.items()} for t, T in TW.items()},
               "traj_controls": {t: {str(o): {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in d.items()} for o, d in T.items()} for t, T in TC.items()}},
              open(cfg["out_md"].replace(".md", ".json"), "w"), ensure_ascii=False, indent=1, default=str)
    log(f"저장 {cfg['out_md']}")


if __name__ == "__main__":
    cmd, cfg = sys.argv[1], json.loads(sys.argv[2])
    {"fetch": cmd_fetch, "features": cmd_features, "study": cmd_study, "cards": cmd_cards, "sample": cmd_sample, "settle": cmd_settle, "winners": cmd_winners}[cmd](cfg)
