"""성장주 보유 전략 그리드 배치 (설계서 docs/strategy_notes/성장주_보유_전략_설계_2026-10.md §5).

    docker compose exec backtester python growth_hold_cli.py '{
      "universe": "sp500+nasdaq100", "start": "2016-01-04", "end": "2026-10-24", "pad_days": 420,
      "grid": {"max_names": [20, 40, 0], "crash_switch": ["off", "full", "half"], "trend_filter": [null, "sma200+20"], "rebalance": ["M", "Q"], "tilt": [1.0, 2.0]},
      "base": {"rank": "per_discount", "cheap_def": "all", "exit_weeks": 8, "band": 0.3},
      "placebo_seeds": 5, "out": "...json", "report_md": "...md"}'

플라세보: 같은 조합에서 rank="random", tilt=1 (리스트에서 무작위 N 종목 동일비중) seed 별. max_names=0 이면 플라세보 = 동일비중과 같으므로 생략.
벤치: QQQ·SPY 보유(세전, 마지막에 한 번 과세한 세후), 리스트 동일비중 매일 리밸런스·무비용(list_hold, 기존 숫자).
집중 변형(설계서 §9): grid 에 "rank": ["momentum","mom6",…], "rerank": [false,true]; base "name_cap": 0. cfg "exclude_listed_after": "2016-01-04" 는 그 뒤 첫 가격이
나오는(= 뒤늦게 상장한) 종목을 리스트에서 뺀다(생존 편향 점검), "list_all": true 는 성장 리스트 없이 유니버스 전체를 리스트로 쓴다(대조).
행에 "플라세보 백분위"(전략 세후가 무작위 seed 중 몇 % 를 이기나)·"QQQ 이긴 해"·"고유 종목 수" 가 붙는다.
재무제표 판(설계서 §11): cfg "quality_features": growth_quality_cli.py features parquet → aux q_score(0~12)·q_rev·q_fcfm·q_roe·q_netcash·q_nodil…,
rank "quality"(점수+백분위 보조)·"fcf"·"roe"·"revgrowth"·"combo:q_score+mom6", require "q_score>=8" (조건은 플라세보에도 적용).
저평가 혼합(설계서 §11.5): aux per(TTM PER)·peg(PER ÷ 매출 TTM 성장%)·dd(52주 고점 대비 낙폭)·per_disc(자기 250일 PER 중앙 대비) 로 require "q_score>=11,dd>=0.1",
entry_require "per<=30"(새로 살 때만), rank per_low|per_discount|drawdown|peg, tilt·cheap_def(싸진 종목 가중).
서프라이즈 판(설계서 §15): aux nbeat4(최근 4발표 beat 수)·surp_avg4·beat_streak·q_plus1(=q_score+1[surp>0])·q_plus(+1[nbeat4==4]) 로 require "q_plus>=13", rank qplus|combo:q_score+surp.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_bulk import download_many
from growth_dip_scalp_cli import load_snapshots, log
from strategy.adapters.growth_dip_scalp import ScalpRules, build_arrays, buy_hold, list_hold, list_mask, per_columns, precompute, weekly_lists
from strategy.adapters.growth_hold_portfolio import HoldRules, expand_grid, run_hold, beat_history
from strategy.adapters.minervini_screener import evaluate_growth


def lever_stats(equity: pd.Series, L: float, borrow: float = 0.03, fee: float = 0.0095) -> dict:
    """전략 일별 자산 곡선에 일일 재조정 레버 L 을 사후 적용(설계서 §14.3). 세금은 벤치처럼 마지막 한 번 22%."""
    r = equity.pct_change().fillna(0.0).to_numpy()
    rl = L * r - (L - 1) * borrow / 252.0 - fee / 252.0
    eq = np.cumprod(1.0 + np.maximum(rl, -0.999))
    e = pd.Series(eq, index=equity.index)
    tot = float(e.iloc[-1] - 1); after = tot - max(0.0, tot - 0.025) * 0.22
    mdd = float((e / e.cummax() - 1).min())
    yearly = (e.resample("YE").last() / e.resample("YE").first() - 1)
    first = float(e[e.index <= "2020-12-31"].iloc[-1] - 1) if (e.index <= "2020-12-31").any() else None
    second = float(e.iloc[-1] / e[e.index <= "2020-12-31"].iloc[-1] - 1) if (e.index <= "2020-12-31").any() else None
    return {"수익률": tot, "세후": after, "MDD": mdd, "전반": first, "후반": second, "연도별": {str(k.year): float(v) for k, v in yearly.items()}}


def main(cfg: dict) -> None:
    prov = default_universe_provider()
    uni = cfg.get("universe", "sp500+nasdaq100")
    if isinstance(uni, dict):          # {"include": ["nasdaq_full"], "exclude": ["sp500", "nasdaq100"]} — 중소형(설계서 §12)
        inc = [t for u in uni.get("include", []) for t in prov.get_tickers(u)]
        exc = {t for u in uni.get("exclude", []) for t in prov.get_tickers(u)}
        tickers = [t for t in dict.fromkeys(inc) if t not in exc]
    else:
        tickers = list(dict.fromkeys(t for u in uni.split("+") for t in prov.get_tickers(u.strip()))) if "+" in uni else prov.get_tickers(uni)
    if cfg.get("tickers"):             # 명시 종목 목록(예비 검사 표본 — 프로토콜 §2-1) 이 유니버스를 대신한다
        tickers = list(dict.fromkeys(t.strip().upper() for t in cfg["tickers"]))
    if cfg.get("max_tickers"):
        tickers = tickers[: int(cfg["max_tickers"])]
    start, end = pd.Timestamp(cfg.get("start", "2016-01-04")), pd.Timestamp(cfg.get("end", date.today().isoformat()))
    pad = int(cfg.get("pad_days", 420))
    t0 = time.time()
    prices = download_many(tickers + ["QQQ", "SPY", "TQQQ", "IWM"], (start - pd.Timedelta(days=pad)).date(), end.date())
    qqq, spy = prices.pop("QQQ"), prices.pop("SPY"); tqqq = prices.pop("TQQQ", None); iwm = prices.pop("IWM", None)
    cal = pd.to_datetime(qqq.index)
    data = precompute({t: d for t, d in prices.items() if t in tickers}, cal)
    log(f"universe {len(data)} · {start.date()} ~ {end.date()} · 가격 {time.time() - t0:.0f}s")
    snaps, industry = load_snapshots(list(data))
    eps_tbl = {}; est_tbl = {}
    for t, qs in snaps.items():
        rows = sorted(((pd.Timestamp(q.effective_report_date), q.eps_actual, q.eps_estimate) for q in qs if q.eps_actual is not None), key=lambda r: r[0])
        if len(rows) >= 4:
            eps_tbl[t] = (np.array([r[0].value for r in rows]), np.array([r[1] for r in rows], dtype=float))
            est_tbl[t] = np.array([np.nan if r[2] is None else r[2] for r in rows], dtype=float)
    for t, d in data.items():
        if t in eps_tbl:
            per_columns(d, eps_tbl[t][0], eps_tbl[t][1], 250)
            tb = eps_tbl[t]; day_ns = d.index.values.astype("datetime64[ns]").astype(np.int64)
            k = np.searchsorted(tb[0], day_ns, side="right")
            yoy = np.full(len(tb[1]), np.nan)
            for q in range(4, len(tb[1])):
                prev = tb[1][q - 4]; yoy[q] = tb[1][q] / abs(prev) - 1 if prev else np.nan
            d["last_yoy"] = np.where(k >= 1, yoy[np.maximum(k - 1, 0)], np.nan)
            est = est_tbl[t]
            with np.errstate(invalid="ignore", divide="ignore"):
                surp = np.where(np.abs(est) > 0.01, (tb[1] - est) / np.abs(est), np.nan)
            d["surp"] = np.where(k >= 1, surp[np.maximum(k - 1, 0)], np.nan)          # 최근 발표 EPS 서프라이즈 (발표일부터)
            for kk, v in beat_history(surp, k).items():                                 # 서프라이즈 이력 (설계서 §15.2)
                d[kk] = v
            c = d["Close"].astype(float).to_numpy(); react = np.full(len(d), np.nan)
            rd_idx = np.searchsorted(day_ns, tb[0], side="left")                     # 발표일(또는 그 다음 거래일) 위치
            for q, ri in enumerate(rd_idx):
                if 1 <= ri < len(c) - 1 and not np.isnan(c[ri - 1]) and c[ri - 1] > 0 and not np.isnan(c[ri + 1]):
                    nxt = rd_idx[q + 1] if q + 1 < len(rd_idx) else len(c)
                    react[ri + 1: max(ri + 1, nxt)] = c[ri + 1] / c[ri - 1] - 1          # 발표 전날 종가 → 발표 다음날 종가, 다음 발표 전까지 유지
            d["ereact"] = react
        d["hi252"] = d["High"].astype(float).rolling(252, min_periods=20).max()
        c = d["Close"].astype(float)
        d["mom12"] = c.shift(21) / c.shift(252) - 1                 # 12-1 개월 모멘텀 (최근 1달 제외)
        d["mom6"] = c / c.shift(126) - 1
    arr = build_arrays(data, cal, None, start, end)
    hi252 = np.column_stack([data[t]["hi252"].reindex(arr.days).to_numpy(dtype=float) for t in arr.tickers])
    aux = {k: np.column_stack([(data[t][k] if k in data[t] else pd.Series(np.nan, index=arr.days)).reindex(arr.days).to_numpy(dtype=float) for t in arr.tickers]) for k in ("mom12", "mom6", "surp", "ereact", "nbeat4", "surp_avg4", "beat_streak")}
    if cfg.get("news_features"):                                      # growth_news_cli.py 출력 (long parquet)
        nf = pd.read_parquet(cfg["news_features"]).rename(columns={"Date": "date"}); nf["date"] = pd.to_datetime(nf["date"])
        for k in ("nsent30", "nvol30", "nvolrel", "nbull30", "nneg30"):
            piv = nf.pivot(index="date", columns="ticker", values=k).reindex(index=arr.days, columns=arr.tickers)
            aux[k] = piv.to_numpy(dtype=float)
        aux["nvol30"] = np.nan_to_num(aux["nvol30"]); aux["nneg30"] = np.nan_to_num(aux["nneg30"])   # 기사 없음 = 0 건
        log(f"뉴스 특징 {cfg['news_features']}: 종목 {nf.ticker.nunique()}, 감성 있는 (일,종목) {int(np.isfinite(aux['nsent30']).sum())}")
    if cfg.get("quality_features"):                                   # growth_quality_cli.py features 출력 (분기별 체크리스트, 설계서 §11)
        from strategy.adapters.fundamental_quality import AUX_COLS, daily_aux
        from strategy.adapters.growth_hold_portfolio import xrank
        qf = pd.read_parquet(cfg["quality_features"]); qf["effective"] = pd.to_datetime(qf["effective"])
        aux.update(daily_aux(qf, arr.days, list(arr.tickers), AUX_COLS, min_available=int(cfg.get("quality_min_available", 8))))
        tie = np.nanmean(np.stack([xrank(aux["q_rev"]), xrank(aux["q_fcfm"]), xrank(aux["q_roe"])]), axis=0)   # 같은 점수 안 순서: 매출 성장·FCF 마진·ROE 백분위 평균
        aux["q_qual"] = aux["q_score"] + 0.99 * np.nan_to_num(tie, nan=0.5)
        sp, nb, qs0 = aux["surp"], aux["nbeat4"], aux["q_score"]                                 # 서프라이즈 보탠 점수(설계서 §15.2): 데이터 없는 종목은 NaN
        aux["q_plus1"] = np.where(np.isfinite(qs0) & np.isfinite(sp), qs0 + (sp > 0), np.nan)
        aux["q_plus"] = np.where(np.isfinite(aux["q_plus1"]) & np.isfinite(nb), aux["q_plus1"] + (nb >= 4), np.nan)
        log(f"서프라이즈: surp 있는 (일,종목) {int(np.isfinite(sp).sum())}, q_score 있는데 surp 없는 {int((np.isfinite(qs0) & ~np.isfinite(sp)).sum())}, "
            f"q_score>=11 중 surp>0 비율 {float(np.nanmean(sp[qs0 >= 11] > 0)) if np.any(qs0 >= 11) else float('nan'):.0%} · nbeat4==4 비율 {float(np.nanmean(nb[qs0 >= 11] >= 4)) if np.any(qs0 >= 11) else float('nan'):.0%}")
        qs = aux["q_score"]
        log(f"재무제표 특징 {cfg['quality_features']}: 종목 {qf.symbol.nunique()}, 점수 있는 (일,종목) {int(np.isfinite(qs).sum())}/{qs.size}, "
            f"점수 ≥ 8 비율 {float(np.nanmean(qs >= 8)):.0%} · ≥ 10 {float(np.nanmean(qs >= 10)):.0%}")
        if "sector" in qf.columns:                                     # 고급 판 A(업종 상대, 설계서 §12.2) + Mohanram G
            from strategy.adapters.fundamental_quality import advanced_scores
            sec_of = qf.dropna(subset=["sector"]).groupby("symbol")["sector"].first().to_dict()
            sectors = np.asarray([sec_of.get(t) for t in arr.tickers], dtype=object)
            adv = advanced_scores(aux, sectors, min_available=int(cfg.get("quality_min_available", 8)))
            aux.update(adv)
            a_, g_ = aux["a_score"], aux["g_score"]
            log(f"고급 판: 섹터 있는 종목 {int(sum(x is not None for x in sectors))}, A 점수 있는 (일,종목) {int(np.isfinite(a_).sum())}, A ≥ 10 {float(np.nanmean(a_ >= 10)):.0%} · ≥ 11 {float(np.nanmean(a_ >= 11)):.0%}, "
                f"G ≥ 6 {float(np.nanmean(g_ >= 6)):.0%}, F ≥ 8 {float(np.nanmean(aux['f_score'] >= 8)):.0%}")
    with np.errstate(invalid="ignore", divide="ignore"):                  # 저평가 열(설계서 §11.5): per(TTM PER, EPS>0), peg = PER ÷ (매출 TTM 성장 ×100)
        aux["per"] = np.where((arr.EPS is not None) & (np.nan_to_num(arr.EPS, nan=0.0) > 0), arr.C / arr.EPS, np.nan) if arr.EPS is not None else np.full(arr.C.shape, np.nan)
        qrev = aux.get("q_rev", np.full(arr.C.shape, np.nan))
        aux["peg"] = np.where(qrev > 0, aux["per"] / (qrev * 100.0), np.nan)
        aux["peg_inv"] = -aux["peg"]
    qqq_close = qqq["Close"].astype(float)
    qqq_in = qqq_close.reindex(arr.days)
    qqq_yearly = {str(k.year): float(v) for k, v in (qqq_in.resample("YE").last() / qqq_in.resample("YE").first() - 1).items()}
    # 생존 편향 점검: 이 날짜 뒤에 첫 가격이 나오는(= 그 뒤 상장) 종목은 리스트에서 뺀다
    late = np.zeros(len(arr.tickers), dtype=bool)
    if cfg.get("exclude_listed_after"):
        cut = pd.Timestamp(cfg["exclude_listed_after"])
        if cut < start - pd.Timedelta(days=pad):                      # 가격 창이 cut 뒤에 시작하면 cut 앞뒤 가격을 따로 받아 상장 여부를 본다
            early = download_many(list(arr.tickers), (cut - pd.Timedelta(days=30)).date(), cut.date())
            fv_of = {t: (early[t]["Close"].first_valid_index() if t in early else None) for t in arr.tickers}
        else:
            fv_of = {t: data[t]["Close"].first_valid_index() for t in arr.tickers}
        for j, t in enumerate(arr.tickers):
            fv = fv_of[t]
            late[j] = fv is None or pd.Timestamp(fv) > cut
        log(f"상장 {cut.date()} 이후 종목 {int(late.sum())}/{len(late)} 제외: {', '.join(t for t, l in zip(arr.tickers, late) if l)[:400]}")

    base = HoldRules(**cfg.get("base", {}))
    combos = expand_grid(base, cfg.get("grid", {})) if cfg.get("grid") else [base]
    log(f"그리드 {len(combos)} 조합")
    gcache: dict = {}

    def growth_fn(t: str, as_of: date) -> bool:
        k = (t, as_of)
        if k not in gcache:
            g = evaluate_growth(snaps.get(t, []), as_of, eps_min_yoy=0.25, rev_min_yoy=0.25, n_required=3)
            gcache[k] = bool(g.data_available and g.passed)
        return gcache[k]

    lcache: dict = {}
    rows = []; dumps: list = []; bench = {"QQQ": buy_hold(qqq, start, end), "SPY": buy_hold(spy, start, end)}
    bench["QQQ 세후"] = bench["QQQ"] - max(0.0, bench["QQQ"] - 0.025) * 0.22
    bench["SPY 세후"] = bench["SPY"] - max(0.0, bench["SPY"] - 0.025) * 0.22
    if iwm is not None and len(iwm) > 0:                                   # 소형주 대조(러셀2000, 설계서 §12)
        bench["IWM"] = buy_hold(iwm, start, end); bench["IWM 세후"] = bench["IWM"] - max(0.0, bench["IWM"] - 0.025) * 0.22
        iw = iwm["Close"].astype(float).reindex(arr.days).ffill(); bench["IWM MDD"] = float((iw / iw.cummax() - 1).min())
    if tqqq is not None and len(tqqq) > 0:                                 # 3배 레버리지 대조 (2026-11-09 사용자 요청)
        bench["TQQQ"] = buy_hold(tqqq, start, end); bench["TQQQ 세후"] = bench["TQQQ"] - max(0.0, bench["TQQQ"] - 0.025) * 0.22
        tq = tqqq["Close"].astype(float).reindex(arr.days).ffill(); bench["TQQQ MDD"] = float((tq / tq.cummax() - 1).min())
        if (tq.index <= "2020-12-31").any():
            bench["TQQQ 전반"] = float(tq[tq.index <= "2020-12-31"].iloc[-1] / tq.iloc[0] - 1); bench["TQQQ 후반"] = float(tq.iloc[-1] / tq[tq.index <= "2020-12-31"].iloc[-1] - 1)
    if (qqq_in.index <= "2020-12-31").any():
        bench["QQQ 전반"] = float(qqq_in[qqq_in.index <= "2020-12-31"].iloc[-1] / qqq_in.iloc[0] - 1)
        bench["QQQ 후반"] = float(qqq_in.iloc[-1] / qqq_in[qqq_in.index <= "2020-12-31"].iloc[-1] - 1)
    t0 = time.time()
    for i, ru in enumerate(combos):
        tf = ru.trend_filter
        if tf not in lcache:
            sr = ScalpRules(trend_filter=tf, max_vol_ratio=None, min_days_to_earnings=0)
            lists = weekly_lists(list(data), cal, growth_fn, data, sr, start, end)
            lm0 = list_mask(arr, lists)
            if cfg.get("list_all"):                               # 대조: 성장 리스트 없이 유니버스 전체를 리스트로 — 유동성(주가 ≥ $10·20일 거래대금 ≥ $20M)은 리스트와 같게 건다(2026-11-09, 중소형 판부터)
                with np.errstate(invalid="ignore"):
                    lm0 = ~np.isnan(arr.C) & (np.nan_to_num(arr.DV, nan=0.0) >= sr.min_dollar_vol_m * 1e6) & (np.nan_to_num(arr.C, nan=0.0) >= sr.min_price)
            lm0[:, late] = False
            lcache[tf] = (lists, lm0, list_hold(data, lists, cal, start, end))
            log(f"  리스트 trend={tf}: 평균 {np.mean([len(v) for v in lists.values()]):.0f} 종목/주, 무비용 매일 동일비중 {lcache[tf][2]:+.0%}")
        lists, lm, lh = lcache[tf]
        res = run_hold(arr, lm, ru, qqq_close, hi252, record=bool(cfg.get("dump")), aux=aux)
        if cfg.get("dump"):
            dumps.append({"rules": asdict(ru), "equity": {str(d.date()): float(v) for d, v in res.equity.items()},
                          "after_tax": res.stats.pop("세후 곡선"), "cash": res.stats.pop("현금 비율"), "holdings": res.stats.pop("보유 이력"),
                          "trades": res.trades, "list_size": {str(k.date()): len(v) for k, v in lists.items()}})
        pl = []
        if ru.max_names > 0 and ru.rank != "random":
            for s in range(int(cfg.get("placebo_seeds", 5))):
                pr = HoldRules(**{**asdict(ru), "rank": "random", "tilt": 1.0, "seed": 100 + s})
                pl.append(run_hold(arr, lm, pr, qqq_close, hi252, aux=aux).stats)
        yearly = (res.equity.resample("YE").last() / res.equity.resample("YE").first() - 1).round(4)
        lev = {f"레버{L}": lever_stats(res.equity, float(L), float(cfg.get("borrow_rate", 0.03)), float(cfg.get("lever_fee", 0.0095))) for L in cfg.get("leverage", [])}
        row = {"rules": asdict(ru), **res.stats, "리스트 무비용 보유": lh, "레버": lev,
               "플라세보 세후": [p["세후"] for p in pl], "플라세보 세후 중앙": float(np.median([p["세후"] for p in pl])) if pl else None,
               "플라세보 세후 90p": float(np.percentile([p["세후"] for p in pl], 90)) if pl else None,
               "플라세보 백분위": float(np.mean([p["세후"] < res.stats["세후"] for p in pl])) if pl else None,
               "플라세보 MDD 중앙": float(np.median([p["MDD"] for p in pl])) if pl else None,
               "연도별": {str(k.year): float(v) for k, v in yearly.items()},
               "QQQ 이긴 해": int(sum(float(v) > qqq_yearly.get(str(k.year), np.inf) for k, v in yearly.items())),
               "전반(~2020)": float(res.equity[res.equity.index <= "2020-12-31"].iloc[-1] - 1) if (res.equity.index <= "2020-12-31").any() else None,
               "후반(2021~)": float(res.equity.iloc[-1] / res.equity[res.equity.index <= "2020-12-31"].iloc[-1] - 1) if (res.equity.index <= "2020-12-31").any() else None}
        rows.append(row)
        log(f"[{i + 1}/{len(combos)}] N{ru.max_names} {ru.rank}{(' req[' + ru.require + ']') if ru.require else ''} tilt{ru.tilt} {ru.cheap_def} {ru.rebalance} band{ru.band} exit{ru.exit_weeks}w crash={ru.crash_switch} trend={tf or '-'} "
            f"{'rerank ' if ru.rerank else ''}→ {res.stats['수익률']:+.0%} 세후 {res.stats['세후']:+.0%} MDD {res.stats['MDD']:+.0%} 매도 {res.stats['매도 건수']} 고유 {res.stats['고유 종목 수']} QQQ이긴해 {row['QQQ 이긴 해']}/{len(yearly)} 세금 {res.stats['세금 총액']:.2f}"
            + (f" 플라세보 중앙 {row['플라세보 세후 중앙']:+.0%} 90p {row['플라세보 세후 90p']:+.0%} 백분위 {row['플라세보 백분위']:.0%}" if pl else "")
            + "".join(f" | {k} 세후 {v['세후']:+.0%} MDD {v['MDD']:+.0%}" for k, v in lev.items()) + f" ({time.time() - t0:.0f}s)")
    out = {"config": cfg, "base": asdict(base), "start": str(start.date()), "end": str(end.date()), "universe_n": len(data), "benchmark": bench, "qqq_yearly": qqq_yearly,
           "excluded_late": [t for t, l in zip(arr.tickers, late) if l], "rows": rows}
    with open(cfg["out"], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    write_md(out, cfg["report_md"])
    if cfg.get("dump"):
        ind = {t: (industry.get(t) or "?") for t in arr.tickers}
        days_in = arr.days
        with open(cfg["dump"], "w") as f:
            json.dump({"combos": dumps, "qqq": {str(d.date()): float(v) for d, v in qqq["Close"].reindex(days_in).items()},
                       "spy": {str(d.date()): float(v) for d, v in spy["Close"].reindex(days_in).items()},
                       "industry": ind, "close": {t: [None if np.isnan(v) else float(v) for v in arr.C[:, j]] for j, t in enumerate(arr.tickers)},
                       "days": [str(d.date()) for d in days_in]}, f, ensure_ascii=False, default=str)
        log(f"덤프 {cfg['dump']}")
    log(f"저장 {cfg['out']} · {cfg['report_md']}")


def write_md(out: dict, path: str) -> None:
    rows = out["rows"]; b = out["benchmark"]
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.0%}"
    L = [f"# 성장주 보유 전략 그리드 — {out['start']} ~ {out['end']} · 유니버스 {out['universe_n']}", "",
         f"기준: QQQ {f(b['QQQ'])} (세후 {f(b['QQQ 세후'])}) · SPY {f(b['SPY'])} (세후 {f(b['SPY 세후'])})"
         + (f" · TQQQ {f(b['TQQQ'])} (세후 {f(b['TQQQ 세후'])}, MDD {f(b.get('TQQQ MDD'))})" if "TQQQ" in b else "")
         + (f" · IWM {f(b['IWM'])} (세후 {f(b['IWM 세후'])}, MDD {f(b.get('IWM MDD'))})" if "IWM" in b else "")
         + f" · 리스트 무비용 매일 동일비중 {f(rows[0]['리스트 무비용 보유']) if rows else '—'}", "",
         "| # | N | 랭킹 | 재선정 | tilt | cheap | 리밸 | 밴드 | 편출 | 스위치 | 추세 | 수익률 | 세후 | CAGR | MDD | 매도 수 | 고유 종목 | QQQ 이긴 해 | 세금 | 전반(~2020) | 후반(2021~) | 플라세보 세후 중앙 / 90p / 백분위 | 레버 (세후 / MDD) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["세후"])
    ny = len(rows[0]["연도별"]) if rows else 0
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    for rank, i in enumerate(order, 1):
        r = rows[i]; ru = r["rules"]
        L.append(f"| {rank} | {ru['max_names'] or '전부'} | {ru['rank']}{(' [' + ru['require'] + ']') if ru.get('require') else ''} | {'예' if ru.get('rerank') else '-'} | {ru['tilt']} | {ru['cheap_def']} | {ru['rebalance']} | {ru['band']} | {ru['exit_weeks']}w | {ru['crash_switch']} | {ru['trend_filter'] or '-'} | "
                 f"{f(r['수익률'])} | {f(r['세후'])} | {f(r['CAGR'])} | {f(r['MDD'])} | {r['매도 건수']} | {r.get('고유 종목 수', '—')} | {r.get('QQQ 이긴 해', '—')}/{ny} | {r['세금 총액']:.2f} | {f(r['전반(~2020)'])} | {f(r['후반(2021~)'])} | "
                 f"{f(r['플라세보 세후 중앙'])} / {f(r['플라세보 세후 90p'])} / {pct(r.get('플라세보 백분위'))} | "
                 + (" · ".join(f"{k} {f(v['세후'])} / {f(v['MDD'])}" for k, v in r.get("레버", {}).items()) or "—") + " |")
    df = pd.DataFrame([{**r["rules"], "세후": r["세후"], "MDD": r["MDD"], "백분위": r.get("플라세보 백분위"), "전반": r.get("전반(~2020)"), "QQQ승": r.get("QQQ 이긴 해")} for r in rows])
    L += ["", "## 요약", "", f"- 조합 {len(rows)}. 세후 중앙값 {f(df['세후'].median())}, 최고 {f(df['세후'].max())}, 최저 {f(df['세후'].min())}. QQQ 세후({f(b['QQQ 세후'])}) 를 넘는 조합 {int((df['세후'] > b['QQQ 세후']).sum())}/{len(df)}."
          + (f" 플라세보 백분위 ≥ 90% 조합 {int((df['백분위'] >= 0.9).sum())}/{int(df['백분위'].notna().sum())}, 전반(~2020) QQQ 세전 초과 조합 {int((df['전반'] > b.get('QQQ 전반', np.inf)).sum()) if 'QQQ 전반' in b else '—'}." if df['백분위'].notna().any() else "")]
    if out.get("excluded_late"):
        L.append(f"- 상장 뒤늦은 종목 {len(out['excluded_late'])} 제외(exclude_listed_after={out['config'].get('exclude_listed_after')}): {', '.join(out['excluded_late'])}")
    for col in ("max_names", "rank", "require", "rerank", "tilt", "cheap_def", "rebalance", "band", "exit_weeks", "crash_switch", "trend_filter", "tilt_sticky"):
        if col in df and df[col].astype(str).nunique() > 1:
            grp = df.groupby(df[col].astype(str)).agg(세후=("세후", "median"), MDD=("MDD", "median"), 백분위=("백분위", "median"), QQQ승=("QQQ승", "median"))
            L.append(f"- {col} 별 세후/MDD/플라세보 백분위/QQQ 이긴 해 중앙값: " + " · ".join(f"{k} {f(v.세후)}/{f(v.MDD)}/{pct(None if pd.isna(v.백분위) else v.백분위)}/{v.QQQ승:.0f}" for k, v in grp.iterrows()))
    with open(path, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "md":
        _o = json.load(open(sys.argv[2])); write_md(_o, sys.argv[3] if len(sys.argv) > 3 else _o["config"]["report_md"])
    else:
        main(json.loads(sys.argv[1]))
