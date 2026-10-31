"""성장주 당일 눌림 +1% 매매 — 파라미터 그리드 배치.

    docker compose exec backtester python growth_dip_scalp_cli.py '{
        "universe": "sp500+nasdaq100", "start": "2016-01-04", "end": "2026-10-24",
        "grid": {"x_pct": [2, 3, 4], "tp_pct": [0.7, 1.0, 1.5, 2.0], "sl_pct": [0.5, 1.0, 1.5, null],
                 "fill": ["optimistic", "neutral", "pessimistic"], "index_filter": [true, false], "trend_filter": [null, "sma50"]},
        "base": {"growth_mode": "minervini", "growth_required": 3},
        "placebo_seeds": 3, "out": "/app/sweep_results/growth_dip_grid.json", "report_md": "/app/sweep_results/growth_dip_grid.md"}'

cfg "intraday":
  {"source": "alphavantage", "interval": "5m", "months": 36, "seed": 7}  — Alpha Vantage 월별 분봉(2000~). 레짐별로 고르게 months 개 달을
      표본으로 뽑아, 그 달에 리스트 종목이 (가장 느슨한 X 로) 트리거된 (종목, 월) 쌍만 받는다(분당 75콜 페이싱, parquet 캐시).
      "months": "all" 이면 전 구간(콜 수 수천~만). 당일 체결은 분봉으로 확정하고 보유 구간(swing)은 일봉으로 이어진다.
  {"source": "yfinance", "interval": "5m", "days": 59}                    — 최근 60일만.
결과 rows 에 "분봉" 블록(표본 달에서의 확정 성과)이 붙는다. hold_mode 기본은 swing(설계서 §2.4 2026-10-30 변경).

설계서 docs/strategy_notes/성장주_당일눌림_1퍼센트_설계_2026-10.md §5. 엔진 strategy/adapters/growth_dip_scalp.py (일봉 근사 — 세 체결 경계).
펀더멘털은 pages._shared.growth_sources(Alpha Vantage 분기·발표일, 디스크 캐시) → 미너비니 evaluate_growth 를 point-in-time 으로.
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
from data.adapters.yfinance_bulk import download_intraday, download_many
from pages._shared.growth_sources import build_growth_adapters
from strategy.adapters.evidence_backtest import classify_regimes
from strategy.adapters.growth_dip_scalp import (
    ScalpRules, build_arrays, build_bars3d, buy_hold, daily_trigger_mask, earnings_mask, expand_grid, list_for_day, list_hold, list_mask,
    next_earnings_exit, per_columns, precompute, run_sim_fast, run_sim_intraday_fast, run_swing, weekly_lists,
)
from strategy.adapters.minervini_screener import evaluate_growth


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_snapshots(tickers: list[str]) -> tuple[dict[str, list], dict[str, str | None]]:
    adapters = build_growth_adapters()
    if not adapters:
        raise SystemExit("성장 펀더멘털 어댑터 없음 (ALPHAVANTAGE_API_KEY / EODHD 키 확인)")
    out: dict[str, list] = {}; ind: dict[str, str | None] = {}
    t0 = time.time()
    for i, t in enumerate(tickers):
        q = []; industry = None
        for a in adapters:
            try:
                s = a.fetch(t)
            except Exception:  # noqa: BLE001
                s = None
            if s is not None and s.quarters:
                q = s.quarters; industry = getattr(s, "industry", None)
                break
        out[t] = q; ind[t] = (str(industry).upper() if industry else None)
        if (i + 1) % 50 == 0:
            log(f"  펀더멘털 {i + 1}/{len(tickers)} ({time.time() - t0:.0f}s)")
    return out, ind


def main(cfg: dict) -> None:
    prov = default_universe_provider()
    uni = cfg.get("universe", "sp500+nasdaq100")
    if isinstance(uni, dict):      # {"include": ["nasdaq_full"], "exclude": ["sp500", "nasdaq100"]} — 예: S&P·나스닥100 밖 소형주
        inc = [t for u in uni.get("include", []) for t in prov.get_tickers(u)]
        exc = {t for u in uni.get("exclude", []) for t in prov.get_tickers(u)}
        tickers = [t for t in dict.fromkeys(inc) if t not in exc]
    elif "+" in uni:
        tickers = list(dict.fromkeys(t for u in uni.split("+") for t in prov.get_tickers(u.strip())))
    else:
        tickers = prov.get_tickers(uni)
    if cfg.get("max_tickers"):
        tickers = tickers[: int(cfg["max_tickers"])]
    start, end = pd.Timestamp(cfg.get("start", "2016-01-04")), pd.Timestamp(cfg.get("end", date.today().isoformat()))
    log(f"universe {len(tickers)} · {start.date()} ~ {end.date()}")

    t0 = time.time()
    pad_days = int(cfg.get("pad_days", 120))      # 지표 선행 구간. sma200·자기 PER(per_lookback 250) 을 처음부터 쓰려면 420 이상
    prices = download_many(tickers + ["QQQ", "SPY"], (start - pd.Timedelta(days=pad_days)).date(), end.date())
    qqq, spy = prices.pop("QQQ"), prices.pop("SPY")
    cal = pd.to_datetime(qqq.index)
    data = precompute({t: d for t, d in prices.items() if t in tickers}, cal)
    index_df = precompute({"QQQ": qqq}, cal)["QQQ"]
    log(f"가격 {len(data)} 종목 {time.time() - t0:.0f}s")

    snaps, industry_of = load_snapshots(list(data))
    base = ScalpRules(**cfg.get("base", {}))

    # ---- TTM EPS → PER, 산업군 중앙값 대비 (point-in-time: 발표일 ≤ 기준일 분기 4개 합)
    eps_tbl: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t, qs in snaps.items():
        rows = sorted(((pd.Timestamp(q.effective_report_date), q.eps_actual) for q in qs if q.eps_actual is not None), key=lambda r: r[0])
        if len(rows) >= 4:
            eps_tbl[t] = (np.array([r[0].value for r in rows]), np.array([r[1] for r in rows], dtype=float))

    # 자기 PER 열 (설계서 §2.7): ttm_eps / per / per_ref — per_own 트리거·hold_mode=per 가 쓴다. per_lookback 은 base 값 하나.
    for t, d in data.items():
        if t in eps_tbl:
            per_columns(d, eps_tbl[t][0], eps_tbl[t][1], base.per_lookback)
    # 최근 발표 분기 EPS YoY(발표 순서 기준 4개 전 분기 대비) · 마지막 발표 뒤 거래일 — §3-2 Q-4/Q-5 (point-in-time)
    for t, d in data.items():
        tb = eps_tbl.get(t)
        if tb is None:
            continue
        day_ns = d.index.values.astype("datetime64[ns]").astype(np.int64)
        k = np.searchsorted(tb[0], day_ns, side="right")
        yoy = np.full(len(tb[1]), np.nan)
        for q in range(4, len(tb[1])):
            prev = tb[1][q - 4]
            yoy[q] = tb[1][q] / abs(prev) - 1 if prev else np.nan
        d["last_yoy"] = np.where(k >= 1, yoy[np.maximum(k - 1, 0)], np.nan)
        rep_days = np.searchsorted(cal.values.astype("datetime64[ns]").astype(np.int64), tb[0], side="left")
        pos = np.arange(len(cal)); kk = np.searchsorted(rep_days, pos, side="right")
        dsr = np.where(kk >= 1, pos - rep_days[np.maximum(kk - 1, 0)], np.nan)
        d["days_since_report"] = pd.Series(dsr, index=cal).reindex(d.index).to_numpy()
    log(f"자기 PER 열: {sum(1 for d in data.values() if 'per_ref' in d and d['per_ref'].notna().any())} 종목에 원래 PER 있음 (창 {base.per_lookback}일)")

    def ttm_eps(t: str, as_of: pd.Timestamp) -> float | None:
        tb = eps_tbl.get(t)
        if tb is None:
            return None
        k = int(np.searchsorted(tb[0], as_of.value, side="right"))
        return float(tb[1][k - 4:k].sum()) if k >= 4 else None

    per_cache: dict[pd.Timestamp, dict[str, float]] = {}

    def per_table(f: pd.Timestamp) -> dict[str, float]:
        if f not in per_cache:
            tbl = {}
            for t, d in data.items():
                if f in d.index and not np.isnan(float(d.at[f, "Close"])):
                    e = ttm_eps(t, f)
                    if e and e > 0:
                        tbl[t] = float(d.at[f, "Close"]) / e
            per_cache[f] = tbl
        return per_cache[f]

    def per_fn_factory(rules: ScalpRules):
        if rules.per_filter != "industry_low":
            return None
        med_cache: dict[tuple[pd.Timestamp, str], float | None] = {}

        def _fn(t: str, as_of: date) -> bool:
            f = pd.Timestamp(as_of); tbl = per_table(f); ind = industry_of.get(t)
            if t not in tbl or not ind:
                return False
            key = (f, ind)
            if key not in med_cache:
                peers = [v for u, v in tbl.items() if industry_of.get(u) == ind and u != t]
                med_cache[key] = float(np.median(peers)) if len(peers) >= rules.per_min_peers else None
            m = med_cache[key]
            return m is not None and tbl[t] < m * rules.per_ratio
        return _fn

    def growth_fn_factory(rules: ScalpRules):
        cache: dict[tuple[str, date], bool] = {}

        def _fn(t: str, as_of: date) -> bool:
            key = (t, as_of)
            if key not in cache:
                g = evaluate_growth(snaps.get(t, []), as_of, eps_min_yoy=rules.eps_min, rev_min_yoy=rules.rev_min,
                                    n_required=rules.growth_required)
                if not g.data_available:
                    cache[key] = False
                elif rules.growth_mode == "any":
                    cache[key] = bool(g.checks.get("2-1 EPS YoY") or g.checks.get("2-3 매출"))
                else:
                    cache[key] = bool(g.passed)
            return cache[key]
        return _fn

    # 발표일(당일 + 이후 n 거래일) 제외 집합
    def earnings_days(n: int) -> dict[str, set[pd.Timestamp]]:
        out: dict[str, set[pd.Timestamp]] = {}
        for t, qs in snaps.items():
            s = set()
            for q in qs:
                rd = q.report_date
                if rd is None:
                    continue
                pos = cal.searchsorted(pd.Timestamp(rd))
                for k in range(0, n + 1):
                    if pos + k < len(cal):
                        s.add(cal[pos + k])
            out[t] = s
        return out

    arr = build_arrays(data, cal, index_df, start, end)
    log(f"배열 {arr.O.shape} (일 × 종목)")
    intraday = None; intra_days = None; intra_day_ok = None
    if cfg.get("intraday"):
        ic = cfg["intraday"]; src = ic.get("source", "yfinance"); interval = ic.get("interval", "5m")
        t0 = time.time()
        if src == "alphavantage":
            from data.adapters.alphavantage_intraday_bulk import fetch_months
            months_cfg = ic.get("months", 36)
            all_months = sorted({(d.year, d.month) for d in cal[(cal >= start) & (cal <= end)]})
            if months_cfg == "all":
                months = all_months
            else:
                regimes_m = classify_regimes(qqq); reg_of = {(d.year, d.month): r for d, r in regimes_m.regime.items()}
                rng_m = np.random.default_rng(int(ic.get("seed", 7)))
                per = max(1, int(months_cfg) // 3); months = []
                for reg in ("상승장", "평범", "하락장"):
                    pool = [m for m in all_months if reg_of.get(m) == reg]
                    months += [pool[k] for k in sorted(rng_m.choice(len(pool), size=min(per, len(pool)), replace=False))] if pool else []
                months = sorted(set(months))
            # 범위: 리스트에 한 번이라도 든 종목 × 그 달에 가장 느슨한 X 로 트리거가 있는 달
            x_min = min([float(v) for v in cfg.get("grid", {}).get("x_pct", [base.x_pct])] + [base.x_pct])
            # 가장 느슨한 트리거(x 최소, 교차 조건 없음)로 일봉 트리거 마스크 — 트리거 종류(open/prev_close/per_own…)와 무관하게 같은 식
            loose = ScalpRules(**{**asdict(base), "x_pct": x_min, **({"trigger": "per_own_any"} if base.trigger.startswith("per_own") else {})})
            trig_loose = daily_trigger_mask(arr, loose)
            tpos = {t: j for j, t in enumerate(arr.tickers)}
            lists_for_pairs = {}
            for tf in {base.trend_filter, *cfg.get("grid", {}).get("trend_filter", [base.trend_filter])}:
                r_ = ScalpRules(**{**asdict(base), "trend_filter": tf})
                lists_for_pairs[tf] = weekly_lists(list(data), cal, growth_fn_factory(r_), data, r_, start, end, extra_fn=per_fn_factory(r_))
            pairs = []; n_members = 0
            for (y, m) in months:
                mdays = cal[(cal.year == y) & (cal.month == m) & (cal >= start) & (cal <= end)]
                members_m = set()
                for d in mdays:
                    for bl in lists_for_pairs.values():
                        members_m |= list_for_day(bl, d)          # 그 달에 (어느 추세 필터로든) 리스트에 있던 종목만
                n_members += len(members_m)
                mi = arr.days.searchsorted(mdays)
                for t in sorted(members_m):
                    j = tpos.get(t)
                    if j is not None and len(mi) and trig_loose[mi, j].any():
                        pairs.append((t, f"{y:04d}-{m:02d}"))
            log(f"AV 분봉: 표본 {len(months)} 달 · 달마다 리스트 종목 평균 {n_members / max(1, len(months)):.0f} → (종목,월) {len(pairs)} 쌍 (캐시 미스 1콜씩, 분당 ≈70)")
            intraday = fetch_months(pairs, interval=interval, progress=lambda a, b, c: log(f"  AV {a}/{b} (콜 {c}, {time.time() - t0:.0f}s)"))
            month_set = set(months)
            intra_day_ok = np.array([(d.year, d.month) in month_set for d in cal[(cal >= start) & (cal <= end)]])
        else:
            bars = download_intraday(list(data), interval, int(ic.get("days", 59)))
            intraday = {t: {d: g for d, g in df.groupby(df.index.date)} for t, df in bars.items()}
        all_days = sorted({d for v in intraday.values() for d in v})
        intra_days = (pd.Timestamp(all_days[0]), pd.Timestamp(all_days[-1])) if all_days else None
        log(f"분봉 {len(intraday)} 종목 {len(all_days)} 일 {time.time() - t0:.0f}s")

    grid = cfg.get("grid", {})
    combos = expand_grid(base, grid) if grid else [base]
    cap = int(cfg.get("max_combos", 600))
    combos = combos[:cap]
    log(f"그리드 {len(combos)} 조합")
    regimes = classify_regimes(qqq)
    reg_by_month = {(d.year, d.month): r for d, r in regimes.regime.items()}

    # 리스트는 (growth_mode, growth_required, eps_min, rev_min, trend_filter, min_dollar_vol_m, min_price) 에만 의존 → 캐시
    list_cache: dict[tuple, dict] = {}
    mask_cache: dict[tuple, np.ndarray] = {}
    earn_cache: dict[int, dict] = {}
    emask_cache: dict[int, np.ndarray] = {}
    b3 = None
    if intraday is not None:
        b3 = build_bars3d(arr, intraday, cfg["intraday"].get("interval", "5m"))
        log(f"분봉 배열 {b3.O.shape} (분봉 있는 날 × 종목 × 봉) · 분봉 있는 (일,종목) {int(b3.has.sum())} · 일봉과 가격 기준 불일치로 버림 {b3.dropped}")
        if intra_day_ok is None:
            intra_day_ok = b3.has.any(axis=1)
    earn_dates = {t: [pd.Timestamp(q.report_date) for q in qs if q.report_date] for t, qs in snaps.items()}
    earn_exit_cache: dict[int, np.ndarray] = {}
    rows = []
    intra_cache: dict[tuple, dict] = {}
    t0 = time.time()
    trades_dump: list[dict] = []     # cfg "trades_out": 조합별 거래 목록 json (진단용)
    for i, rules in enumerate(combos):
        lk = (rules.growth_mode, rules.growth_required, rules.eps_min, rules.rev_min, rules.trend_filter, rules.min_dollar_vol_m, rules.min_price,
              rules.per_filter, rules.per_ratio, rules.per_min_peers)
        if lk not in list_cache:
            t1 = time.time()
            list_cache[lk] = weekly_lists(list(data), cal, growth_fn_factory(rules), data, rules, start, end, extra_fn=per_fn_factory(rules))
            mask_cache[lk] = list_mask(arr, list_cache[lk])
            log(f"  리스트 {lk}: 평균 {np.mean([len(v) for v in list_cache[lk].values()]):.0f} 종목/주 ({time.time() - t1:.0f}s)")
        lists = list_cache[lk]; lm = mask_cache[lk]
        ek = (rules.earnings_exclude_days, rules.min_days_to_earnings)
        if ek not in emask_cache:
            if rules.earnings_exclude_days not in earn_cache:
                earn_cache[rules.earnings_exclude_days] = earnings_days(rules.earnings_exclude_days)
            em_ = earnings_mask(arr, earn_cache[rules.earnings_exclude_days])
            if rules.min_days_to_earnings > 0:
                # 다음 발표일까지 거래일 < min 이면 제외 (발표일 모르는 종목은 그대로) — 설계서 §2.6
                nxt = next_earnings_exit(arr, earn_dates, 0)
                dte = nxt - np.arange(len(arr.days))[:, None]
                em_ = em_ | ((nxt < len(arr.days)) & (dte < rules.min_days_to_earnings))
            emask_cache[ek] = em_
        earn = earn_cache[rules.earnings_exclude_days]; em = emask_cache[ek]
        def _run(mode="strategy", rng=None, use_bars=False):
            if rules.hold_mode in ("swing", "per"):
                nb = rules.earnings_exit_days_before
                if nb not in earn_exit_cache:
                    earn_exit_cache[nb] = next_earnings_exit(arr, earn_dates, nb)
                return run_swing(arr, lm, em, rules, earn_exit_cache[nb], mode=mode, rng=rng,
                                 day_ok=intra_day_ok if use_bars else None, b3=b3 if use_bars else None)
            if use_bars:
                return run_sim_intraday_fast(arr, b3, lm, em, rules, mode=mode, rng=rng, day_ok=intra_day_ok)
            return run_sim_fast(arr, lm, em, rules, mode=mode, rng=rng)

        res = _run()
        if cfg.get("trades_out"):
            trades_dump.append({"combo": i, "rules": asdict(rules), "kind": "daily", "trades": res.trades})
        pa, pb = [], []
        for s in range(int(cfg.get("placebo_seeds", 3))):
            pa.append(_run("placebo_a", np.random.default_rng(100 + s)).stats)
            pb.append(_run("placebo_b", np.random.default_rng(200 + s)).stats)
        intra_block = None
        ik = (rules.x_pct, rules.tp_pct, rules.sl_pct, rules.index_filter, rules.index_drop_pct, rules.trend_filter, rules.trigger, rules.skip_gap_below, lk,
              rules.earnings_exclude_days, rules.hold_mode, rules.tp2_pct, rules.max_hold_days, rules.earnings_exit_days_before,
              rules.min_days_to_earnings, rules.max_vol_ratio, rules.close_pos_stop, rules.per_exit_pct,
              rules.px_below_med_pct, rules.below_sma200_pct, rules.per_ref_max_ratio, rules.max_last_yoy, rules.min_days_since_report)
        if b3 is not None and ik in intra_cache:
            intra_block = intra_cache[ik]
        elif b3 is not None and all(asdict(rules).get(k) == v for k, v in cfg.get("intraday_subgrid", {}).items()):
            # 분봉은 fill 과 무관 → 같은 ik 는 캐시로 한 번만, intraday_subgrid 로 범위를 더 줄일 수 있다
            ri = _run(use_bars=True)
            if cfg.get("trades_out"):
                trades_dump.append({"combo": i, "rules": asdict(rules), "kind": "intraday", "trades": ri.trades})
            pa_i = [_run("placebo_a", np.random.default_rng(300 + s_), use_bars=True).stats for s_ in range(int(cfg.get("placebo_seeds", 3)))]
            pb_i = [_run("placebo_b", np.random.default_rng(400 + s_), use_bars=True).stats for s_ in range(int(cfg.get("placebo_seeds", 3)))]
            i0, i1 = intra_days
            intra_block = {**ri.stats, "구간": [str(i0.date()), str(i1.date())],
                           "플라세보A 거래당": float(np.nanmean([x["거래당 순손익"] if x["거래당 순손익"] is not None else np.nan for x in pa_i])),
                           "플라세보B 거래당": float(np.nanmean([x["거래당 순손익"] if x["거래당 순손익"] is not None else np.nan for x in pb_i]))}
            intra_cache[ik] = intra_block
        # 레짐별 거래당 손익
        by_reg: dict[str, list[float]] = {}
        for tr in res.trades:
            r = reg_by_month.get((tr["day"].year, tr["day"].month), "?")
            by_reg.setdefault(r, []).append(tr["ret"])
        by_year = {}
        for y, g in pd.Series({tr["day"]: tr["ret"] for tr in res.trades}).groupby(lambda d: d.year) if res.trades else []:
            by_year[int(y)] = {"거래": int(len(g)), "승률": float((g > 0).mean()), "거래당": float(g.mean())}
        row = {"rules": asdict(rules), **res.stats, "분봉": intra_block,
               "리스트 평균 종목 수": float(np.mean([len(v) for v in lists.values()])) if lists else 0.0,
               "플라세보A 거래당": float(np.nanmean([x["거래당 순손익"] if x["거래당 순손익"] is not None else np.nan for x in pa])),
               "플라세보A 수익률": float(np.mean([x["수익률"] for x in pa])),
               "플라세보B 거래당": float(np.nanmean([x["거래당 순손익"] if x["거래당 순손익"] is not None else np.nan for x in pb])),
               "플라세보B 수익률": float(np.mean([x["수익률"] for x in pb])),
               "레짐별 거래당": {k: float(np.mean(v)) for k, v in by_reg.items()},
               "레짐별 거래 수": {k: len(v) for k, v in by_reg.items()},
               "연도별": by_year}
        rows.append(row)
        log(f"[{i + 1}/{len(combos)}] {rules.trigger}{'+PER' if rules.per_filter else ''} x{rules.x_pct} tp{rules.tp_pct}/{rules.tp2_pct} {rules.max_hold_days}d sl{rules.sl_pct} {rules.fill} idx{int(rules.index_filter)} {rules.trend_filter or '-'} "
            f"→ {res.stats['수익률']:+.1%} 세후 {res.stats['세후']:+.1%} 거래 {res.stats['거래 수']} 승률 {(res.stats['승률'] or 0):.0%} "
            f"거래당 {(res.stats['거래당 순손익'] or 0):+.3%} 플A {row['플라세보A 거래당']:+.3%} 플B {row['플라세보B 거래당']:+.3%}"
            + (f" | 분봉 거래 {intra_block['거래 수']} 승률 {(intra_block['승률'] or 0):.0%} 거래당 {(intra_block['거래당 순손익'] or 0):+.3%} 플A {intra_block['플라세보A 거래당']:+.3%}" if intra_block else "")
            + (f" 보유 {res.stats.get('평균 보유일', 0) or 0:.1f}일 당일익절 {(res.stats.get('당일 익절 비율') or 0):.0%}" if rules.hold_mode in ("swing", "per") else "")
            + f" ({time.time() - t0:.0f}s)")

    bench = {"QQQ": buy_hold(qqq, start, end), "SPY": buy_hold(spy, start, end)}
    if list_cache:
        first = next(iter(list_cache.values()))
        bench["리스트 동일비중 보유"] = list_hold(data, first, cal, start, end)
    out = {"config": cfg, "base": asdict(base), "start": str(start.date()), "end": str(end.date()), "universe_n": len(data),
           "benchmark": bench, "rows": rows}
    with open(cfg["out"], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    if cfg.get("trades_out"):
        with open(cfg["trades_out"], "w") as f:
            json.dump(trades_dump, f, ensure_ascii=False, default=str)
    write_md(out, cfg["report_md"])
    log(f"저장 {cfg['out']} · {cfg['report_md']}")


def write_md(out: dict, path: str) -> None:
    rows = out["rows"]
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"
    g = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3%}"
    L = [f"# 성장주 당일 눌림 +1% 그리드 — {out['start']} ~ {out['end']} · 유니버스 {out['universe_n']}", "",
         f"기준: QQQ {f(out['benchmark']['QQQ'])} · SPY {f(out['benchmark']['SPY'])} · 리스트 동일비중 보유 {f(out['benchmark'].get('리스트 동일비중 보유'))}", "",
         "| # | 트리거 | X% | 익절 | 손절 | 보유 | 체결 | 지수필터 | 추세 | 성장 | 수익률 | 세후 | CAGR | MDD | 거래 | 연 거래 | 승률 | 거래당 | 중앙값 | 평균보유일 | 당일익절 | 하루 최악 | 플라세보A 거래당 | 플라세보B 거래당 | 상승장/평범/하락장 거래당 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(range(len(rows)), key=lambda i: -(rows[i]["세후"] if rows[i]["세후"] is not None else -9))
    for rank, i in enumerate(order, 1):
        r = rows[i]; ru = r["rules"]; rg = r["레짐별 거래당"]
        hold = (f"swing tp2 {ru['tp2_pct']}% / {ru['max_hold_days']}일 / 실적−{ru['earnings_exit_days_before']}" if ru['hold_mode'] == 'swing'
                else f"PER 회복(−{ru.get('per_exit_pct', 0)}%) / {ru['max_hold_days'] or '무기한'}" if ru['hold_mode'] == 'per' else 'day')
        filt = " ".join(x for x in (f"실적≥{ru['min_days_to_earnings']}" if ru.get('min_days_to_earnings') else "", f"거래량≤{ru['max_vol_ratio']}" if ru.get('max_vol_ratio') else "",
                                   f"종가손절<{ru['close_pos_stop']}" if ru.get('close_pos_stop') else "",
                                   f"중앙−{ru['px_below_med_pct']}%" if ru.get('px_below_med_pct') is not None else "", f"200선−{ru['below_sma200_pct']}%" if ru.get('below_sma200_pct') is not None else "",
                                   f"PER오염<{ru['per_ref_max_ratio']}" if ru.get('per_ref_max_ratio') is not None else "", f"YoY≤{ru['max_last_yoy']}" if ru.get('max_last_yoy') is not None else "",
                                   f"발표후≥{ru['min_days_since_report']}" if ru.get('min_days_since_report') else "") if x) or "-"
        hold = hold + (f" · {filt}" if filt != "-" else "")
        L.append(f"| {rank} | {ru['trigger']}{' +저PER' if ru.get('per_filter') else ''} | {ru['x_pct']} | {ru['tp_pct']} | {ru['sl_pct'] if ru['sl_pct'] is not None else '없음'} | {hold} | {ru['fill']} | {'O' if ru['index_filter'] else '-'} | {ru['trend_filter'] or '-'} | {ru['growth_mode']}{ru['growth_required'] if ru['growth_mode'] == 'minervini' else ''} | "
                 f"{f(r['수익률'])} | {f(r['세후'])} | {f(r['CAGR'])} | {f(r['MDD'])} | {r['거래 수']} | {r['연 거래']:.0f} | {'—' if r['승률'] is None else format(r['승률'], '.0%')} | {g(r['거래당 순손익'])} | {g(r['거래당 중앙값'])} | "
                 f"{'—' if r.get('평균 보유일') is None else format(r['평균 보유일'], '.1f')} | {'—' if r.get('당일 익절 비율') is None else format(r['당일 익절 비율'], '.0%')} | {f(r['하루 최악'])} | "
                 f"{g(r['플라세보A 거래당'])} | {g(r['플라세보B 거래당'])} | {g(rg.get('상승장'))} / {g(rg.get('평범'))} / {g(rg.get('하락장'))} |")
    first_intra = next((r["분봉"] for r in rows if r.get("분봉")), None)
    if first_intra:
        src = (out.get("config", {}).get("intraday") or {}).get("source", "yfinance")
        L += ["", f"## 분봉 확정 체결 ({src} 5분봉, {first_intra['구간'][0]} ~ {first_intra['구간'][1]}) — 일봉 경계와 비교", "",
              "| # | 트리거 | X% | 익절 | 손절 | 보유 | 지수필터 | 추세 | 분봉 수익률 | 분봉 거래 | 승률 | 거래당 | 중앙값 | 플라세보A | 플라세보B | 청산 사유 | 일봉 거래당(같은 규칙·긴 구간) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        seen = set()
        for r in rows:
            b = r.get("분봉"); ru = r["rules"]
            key = (ru["x_pct"], ru["tp_pct"], ru["sl_pct"], ru["index_filter"], ru["trend_filter"], ru["trigger"], ru["hold_mode"], ru.get("per_exit_pct"),
                   ru["max_hold_days"], ru.get("min_days_to_earnings"), ru.get("max_vol_ratio"), ru.get("close_pos_stop"),
                   ru.get("px_below_med_pct"), ru.get("below_sma200_pct"), ru.get("per_ref_max_ratio"), ru.get("max_last_yoy"), ru.get("min_days_since_report"))
            if not b or key in seen:
                continue
            seen.add(key)
            hold_b = (f"PER 회복(−{ru.get('per_exit_pct', 0)}%)/{ru['max_hold_days'] or '무기한'}" if ru['hold_mode'] == 'per'
                      else f"swing {ru['max_hold_days']}일" if ru['hold_mode'] == 'swing' else 'day')
            L.append(f"| {len(seen)} | {ru['trigger']} | {ru['x_pct']} | {ru['tp_pct']} | {ru['sl_pct'] if ru['sl_pct'] is not None else '없음'} | {hold_b} | {'O' if ru['index_filter'] else '-'} | {ru['trend_filter'] or '-'} | {f(b['수익률'])} | {b['거래 수']} | "
                     f"{'—' if b['승률'] is None else format(b['승률'], '.0%')} | {g(b['거래당 순손익'])} | {g(b['거래당 중앙값'])} | {g(b['플라세보A 거래당'])} | {g(b['플라세보B 거래당'])} | {b['청산 사유']} | {g(r['거래당 순손익'])} ({ru['fill']}) |")
    df = pd.DataFrame([{**r["rules"], "세후": r["세후"], "거래당": r["거래당 순손익"], "승률": r["승률"], "플A": r["플라세보A 거래당"], "플B": r["플라세보B 거래당"]} for r in rows])
    L += ["", "## 요약", "",
          f"- 조합 {len(rows)}개. 세후 수익률 중앙값 {f(df['세후'].median())}, 최고 {f(df['세후'].max())}, 최저 {f(df['세후'].min())}. 거래당 순손익 중앙값 {g(df['거래당'].median())}.",
          f"- 플라세보 A(리스트 밖 같은 트리거)를 거래당 +0.2%p 이상 이기는 조합: {int(((df['거래당'] - df['플A']) >= 0.002).sum())}/{len(df)} · 플라세보 B(리스트 종목 무작위 날): {int(((df['거래당'] - df['플B']) >= 0.002).sum())}/{len(df)}.",
          f"- 세후로 QQQ 보유({f(out['benchmark']['QQQ'])})를 넘는 조합: {int((df['세후'] > out['benchmark']['QQQ']).sum())}/{len(df)}."]
    for col in ("trigger", "fill", "x_pct", "tp_pct", "sl_pct", "tp2_pct", "max_hold_days", "index_filter", "index_drop_pct", "trend_filter", "per_filter",
                "min_days_to_earnings", "max_vol_ratio", "close_pos_stop", "hold_mode", "per_exit_pct",
                "px_below_med_pct", "below_sma200_pct", "per_ref_max_ratio", "max_last_yoy", "min_days_since_report", "max_positions"):
        if col in df and df[col].nunique() > 1:
            grp = df.groupby(df[col].astype(str))["세후"].median()
            L.append(f"- {col} 별 세후 중앙값: " + " · ".join(f"{k} {f(v)}" for k, v in grp.items()))
    with open(path, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "md":  # 저장된 json 으로 md 만 다시 쓰기: python growth_dip_scalp_cli.py md <json> [<md>]
        _out = json.load(open(sys.argv[2]))
        write_md(_out, sys.argv[3] if len(sys.argv) > 3 else _out["config"]["report_md"])
    else:
        main(json.loads(sys.argv[1]))
