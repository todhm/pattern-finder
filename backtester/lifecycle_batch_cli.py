"""«The Lifecycle Trade» 배치 검증 — IPO 코호트(또는 지정 티커)에 6장 통계 재현 + 4매수지점 × 5매도규칙 시뮬.

    # 코호트: Finnhub IPO 캘린더(→ AV LISTING_STATUS 폴백)에서 첫 거래일이 구간 안인 IPO
    docker compose exec backtester python lifecycle_batch_cli.py \\
        '{"mode": "cohort", "ipo_from": "2016-01-01", "ipo_to": "2022-12-31", "as_of": "2026-10-10",
          "out": "/tmp/lifecycle_cohort.json", "report_md": "/tmp/lifecycle_report.md"}'
    # 지정 티커 (책의 iconic 예시 등) — {티커: 첫 거래일}
    docker compose exec backtester python lifecycle_batch_cli.py \\
        '{"mode": "tickers", "tickers": {"PANW": "2012-07-20", "FB": "2012-05-18"}, "out": "/tmp/lifecycle_iconic.json"}'
    # 저장된 결과 재분석
    docker compose exec backtester python lifecycle_batch_cli.py '{"analyze": "/tmp/lifecycle_cohort.json", "report_md": "..."}'

옵션: exchanges(["NASDAQ","NYSE"]), exclude_spac(true), min_offer(5), max_tickers(0=전체), workers(4),
max_days(756 = 3년), rules, buy_kinds, params(LifecycleParams 필드 덮어쓰기), revenue(false — 7장 스크린의
매출 50% 판정에 성장 펀더멘털 호출; 코호트 전체면 느리다).
진행 로그 stderr, 요약 표 stdout. 결과 JSON 은 티커마다 바로바로 덮어쓴다(중단해도 부분 결과 남음).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, timedelta

import pandas as pd

from data.adapters.composed_ipo_calendar import build_default_ipo_calendar
from data.adapters.composed_market_data import build_default_market_data
from data.domain.models import IpoEvent
from strategy.adapters.lifecycle_trade import (
    BUY_KINDS,
    RULES,
    IpoStats,
    LifecycleParams,
    analyze_ticker,
    cohort_tables,
    fetch_many,
    ipo_cohort,
    kind_summary,
    rule_pattern_table,
    rule_summary,
)

ICONIC = {   # 책의 사례 + 저자 거래 (티커: (첫 거래일, 공모가)) — 패턴 라벨 비교용. 공모가 = IPO-AF 바닥선
    "PANW": ("2012-07-20", 42), "FB": ("2012-05-18", 38), "GOOGL": ("2004-08-19", 85), "SHAK": ("2015-01-30", 21),
    "TSLA": ("2010-06-29", 17), "NFLX": ("2002-05-23", 15), "TWLO": ("2016-06-23", 15), "ACIA": ("2016-05-13", 23),
    "GRUB": ("2014-04-04", 26), "SNAP": ("2017-03-02", 17), "NOW": ("2012-06-29", 18), "BABA": ("2014-09-19", 68),
    "FSLR": ("2006-11-17", 20), "ZG": ("2011-07-20", 20), "LULU": ("2007-07-30", 18), "V": ("2008-03-19", 44),
    "GPRO": ("2014-06-26", 24), "FIT": ("2015-06-18", 20), "TWTR": ("2013-11-07", 26), "ETSY": ("2015-04-16", 16),
    "PBPB": ("2013-10-04", 14), "LC": ("2014-12-11", 15), "CBOE": ("2010-06-15", 29), "LNKD": ("2011-05-19", 45),
    "MBLY": ("2014-08-01", 25), "LOCO": ("2014-07-25", 15), "ZNGA": ("2011-12-16", 10), "KORS": ("2011-12-15", 20),
    "FEYE": ("2013-09-20", 20), "SHOP": ("2015-05-21", 17), "MA": ("2006-05-25", 39), "CMG": ("2006-01-26", 22),
    "YELP": ("2012-03-02", 15), "TLRY": ("2018-07-19", 17), "RDFN": ("2017-07-28", 15),
}
BOOK_LABEL = {"PANW": "Late Bloomer", "FB": "Pump and Dump", "GOOGL": "Stair Stepper", "SHAK": "One-Hit Wonder(Plus)",
              "TSLA": "Late Bloomer", "NFLX": "Pump and Dump", "TWLO": "One-Hit Wonder Plus", "ACIA": "Rocket Ship",
              "SNAP": "Disappointment", "NOW": "Stair Stepper", "BABA": "One-Hit Wonder Plus", "FSLR": "Rocket Ship",
              "ZG": "Pump and Dump", "LULU": "Late Bloomer", "V": "Late Bloomer", "GPRO": "One-Hit Wonder",
              "FIT": "One-Hit Wonder", "TWTR": "One-Hit Wonder Plus", "ETSY": "Disappointment→I-AP", "PBPB": "Disappointment",
              "LC": "Disappointment", "CBOE": "Pump and Dump", "LNKD": "Pump and Dump", "MBLY": "One-Hit Wonder Plus",
              "LOCO": "One-Hit Wonder", "ZNGA": "One-Hit Wonder", "KORS": "Rocket Ship", "FEYE": "Rocket Ship",
              "SHOP": "Late Bloomer", "MA": "Stair Stepper", "CMG": "Late Bloomer", "YELP": "Late Bloomer"}


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _rec(res) -> dict:
    lc = res.lifecycle
    return {
        "symbol": res.symbol, "ipo": res.ipo.model_dump(mode="json") if res.ipo else None,
        "pattern": lc.pattern, "pattern_note": lc.pattern_note, "phase_now": lc.phase_now,
        "ap_gain": lc.ap_gain, "ap_exists": lc.ap_exists, "gray": lc.gray, "weeks": int(len(res.frames.w)),
        "has_day1": res.frames.has_day1, "ipo_base": bool(lc.ipo_base), "iap_start": (res.frames.w.index[lc.iap_start_i].date().isoformat()
                                                                           if lc.iap_start_i is not None else None),
        "buy_points": [{"kind": b.kind, "date": b.signal_date.date().isoformat(), "level": b.level, "note": b.note} for b in res.buy_points],
        "trades": [{**asdict(t), "sells": [(d_.isoformat(), px, f, why) for d_, px, f, why in t.sells]} for t in res.trades],
        "stats": asdict(res.stats) if res.stats else None,
        "screen": asdict(res.screen) if res.screen else None,
    }


def run(cfg: dict) -> list[dict]:
    md = build_default_market_data()
    p = LifecycleParams(**cfg.get("params", {}))
    as_of = date.fromisoformat(cfg["as_of"]) if cfg.get("as_of") else date.today()
    rules = tuple(cfg.get("rules", RULES))
    kinds = tuple(cfg.get("buy_kinds", BUY_KINDS))
    max_days = int(cfg.get("max_days", 756))
    if cfg.get("mode", "cohort") == "tickers":
        tick = cfg.get("tickers") or ICONIC
        events = []
        for s_, v in tick.items():      # "YYYY-MM-DD" 또는 ["YYYY-MM-DD", 공모가]
            d_, offer = (v, None) if isinstance(v, str) else (v[0], v[1] if len(v) > 1 else None)
            events.append(IpoEvent(symbol=s_, ipo_date=date.fromisoformat(d_), offer_price=offer, source="manual"))
    else:
        cal = build_default_ipo_calendar()
        if cal is None:
            raise SystemExit("IPO 캘린더 소스 없음 (FINNHUB_API_KEY / ALPHAVANTAGE_API_KEY)")
        raw = cal.fetch_ipos(date.fromisoformat(cfg["ipo_from"]), date.fromisoformat(cfg["ipo_to"]))
        _log(f"IPO 캘린더 {cal.last_source}: {len(raw)}건")
        events = ipo_cohort(raw, tuple(cfg.get("exchanges", ("NASDAQ", "NYSE"))), bool(cfg.get("exclude_spac", True)),
                            cfg.get("min_offer", 5.0))
        _log(f"코호트 필터 후 {len(events)}건")
    if cfg.get("max_tickers"):
        events = events[: int(cfg["max_tickers"])]
    growth = None
    if cfg.get("revenue"):
        from pages._shared.growth_sources import build_growth_adapters
        growth = build_growth_adapters()
    by_sym = {e.symbol: e for e in events}
    start = min(e.ipo_date for e in events) - timedelta(days=7)
    frames, errors = fetch_many(md, list(by_sym), start, as_of, int(cfg.get("workers", 4)), _log)
    _log(f"일봉 {len(frames)}/{len(events)} (실패 {len(errors)})")
    out_path = cfg.get("out", "/tmp/lifecycle_batch.json")
    results: list[dict] = []
    for i, (sym, df) in enumerate(frames.items()):
        ev = by_sym[sym]
        rev = None
        if growth:
            from strategy.adapters.lifecycle_trade import revenue_yoy_asof
            for a in growth:
                try:
                    snap = a.fetch(sym)
                except Exception:  # noqa: BLE001
                    continue
                if snap.quarters:
                    rev = revenue_yoy_asof(snap.quarters, as_of)
                    break
        try:
            res = analyze_ticker(sym, df, ev, p, as_of, rules, kinds, max_days, rev)
        except Exception as e:  # noqa: BLE001
            _log(f"  {sym} 분석 실패 {type(e).__name__}: {e}")
            continue
        if res is None:
            continue
        results.append(_rec(res))
        if (i + 1) % 25 == 0 or i == len(frames) - 1:
            _log(f"  분석 {i + 1}/{len(frames)}")
            with open(out_path, "w") as f:
                json.dump({"config": cfg, "errors": errors, "results": results}, f, ensure_ascii=False, indent=1, default=str)
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "errors": errors, "results": results}, f, ensure_ascii=False, indent=1, default=str)
    return results


# --------------------------------------------------------------------------- #
# 분석·리포트
# --------------------------------------------------------------------------- #
def _trades_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for t in r["trades"]:
            rows.append({"symbol": r["symbol"], "pattern": r["pattern"], "kind": t["kind"], "rule": t["rule"],
                         "entry_date": t["entry_date"], "pct_gain": t["pct_gain"], "drawdown_k": t["drawdown_k"],
                         "peak_retained": t["peak_retained"], "days": t["days"], "reason": t["reason"], "open": t["open_position"]})
    return pd.DataFrame(rows)


def _table(d: pd.DataFrame, index: bool = False) -> str:
    """tabulate 없이 마크다운 표."""
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    rows = [[("" if v is None else str(v)) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def _md(df: pd.DataFrame, pct_cols: tuple[str, ...] = ()) -> str:
    if df is None or df.empty:
        return "_(없음)_"
    d = df.copy()
    for c in d.columns:
        if c in pct_cols or (d[c].dtype.kind == "f" and (c.startswith(("비율", "승률", "중앙값 수익", "평균 수익", "책 중앙값 수익", "책(", "책 비율", "책 조정", "이후 52주", "코호트 비율", "중앙값 피크", "책 피크", "상위10%", "최대", "pct_gain")) or c in ("값", "책"))):
            d[c] = d[c].map(lambda x: "—" if pd.isna(x) else (f"{x:+.1%}" if abs(x) < 50 else f"{x:.1f}"))
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "—" if pd.isna(x) else f"{x:,.1f}")
    return _table(d)


def report(results: list[dict], title: str = "라이프사이클 트레이드 배치 검증") -> str:
    lines = [f"# {title}", "", f"종목 {len(results)}건 · 첫 거래일 데이터 보유 {sum(bool(r['has_day1']) for r in results)}건", ""]
    stats = [IpoStats(**r["stats"]) for r in results if r.get("stats")]
    tabs = cohort_tables(stats)
    if tabs:
        lines += ["## 6장 연구 재현 (책 1982~2017 수치와 비교)", "", f"대상: 첫 거래일부터 데이터가 있는 {len(stats)}건", ""]
        for key, name in (("table_6_1", "Table 6.1 — 첫날 종가 대비 +100% 달성 비율"), ("fig_6_1", "Fig 6.1 — 첫날 저가 이탈"),
                          ("fig_6_2", "Fig 6.2 — 첫날 저가 −10% 손절 발동"), ("table_6_2", "Table 6.2 — 빠른 2배 후 −50% 조정"),
                          ("summary", "80쪽 요약 — 수익 분포·유동성 텔")):
            lines += [f"### {name}", "", _md(tabs[key]), ""]
    pat = pd.Series([r["pattern"] for r in results]).value_counts().rename_axis("패턴").reset_index(name="n")
    pat["비율"] = pat["n"] / len(results)
    lines += ["## 패턴 분포 (사후 라벨)", "", _md(pat), ""]
    tf = _trades_df(results)
    if not tf.empty:
        lines += ["## 매도 규칙별 성과 ($100K, 책 Table 5.2 와 비교)", "", _md(rule_summary(tf)), "",
                  "## 규칙 × 패턴 — 중앙값 수익 (책 Table 5.1)", "", _table(rule_pattern_table(tf).map(lambda x: "—" if pd.isna(x) else f"{x:+.0%}"), index=True), "",
                  "## 규칙 × 패턴 — 중앙값 DD $K", "", _table(rule_pattern_table(tf, "drawdown_k").map(lambda x: "—" if pd.isna(x) else f"{x:.0f}"), index=True), "",
                  "## 매수 지점 × 규칙", "", _md(kind_summary(tf)), ""]
        reasons = tf.groupby(["rule", "reason"]).size().reset_index(name="n").sort_values(["rule", "n"], ascending=[True, False])
        lines += ["## 청산 사유", "", _table(reasons), ""]
    if not tf.empty:
        lines += liquidity_sections(results, tf)
    if any(r["symbol"] in BOOK_LABEL for r in results):
        rows = [{"티커": r["symbol"], "책 라벨": BOOK_LABEL.get(r["symbol"], ""), "코드 라벨": r["pattern"], "메모": r["pattern_note"],
                 "IPO-AP": f"{r['ap_gain']:+.0%}", "매수지점": ", ".join(b["kind"] for b in r["buy_points"])}
                for r in results if r["symbol"] in BOOK_LABEL]
        lines += ["## 책 사례 라벨 대조", "", _table(pd.DataFrame(rows)), ""]
    return "\n".join(lines)


def _summ(g: pd.DataFrame) -> pd.Series:
    r = g["pct_gain"]
    return pd.Series({"n": len(r), "승률": float((r > 0).mean()), "중앙값 수익": float(r.median()), "평균 수익": float(r.mean()),
                      "상위10% 수익": float(r.quantile(0.9)), "최대": float(r.max()), "중앙값 DD $K": float(g["drawdown_k"].median()),
                      "중앙값 보유일": float(g["days"].median())})


def liquidity_sections(results: list[dict], tf: pd.DataFrame) -> list[str]:
    """책의 두 주장 — '유동성 텔($20M)' 과 '90일 내 2배' — 로 거래를 갈라 본다 (문서 §E.4 대조군)."""
    st = {r["symbol"]: (r.get("stats") or {}) for r in results}
    t = tf.copy()
    t["유동($20M↑)"] = t["symbol"].map(lambda s_: (st[s_].get("dollar_vol") or 0) >= 20e6)
    t["90일2배"] = t["symbol"].map(lambda s_: (st[s_].get("weeks_to_double") or 1e9) <= 13)
    t["진입연도"] = t["entry_date"].astype(str).str[:4]
    out = ["## 대조군 — 유동성 텔 · 90일 2배 · 진입 연도 (MCP 제외)", ""]
    sub = t[t["rule"] != "MCP"]
    out += ["### 규칙 × 유동성 (전 기간 평균 거래대금 ≥ $20M)", "", _md(sub.groupby(["rule", "유동($20M↑)"]).apply(_summ, include_groups=False).reset_index()), ""]
    fm = sub[sub["kind"] == "first mature base"]
    out += ["### first mature base × 유동성", "", _md(fm.groupby(["rule", "유동($20M↑)"]).apply(_summ, include_groups=False).reset_index()), ""]
    out += ["### 규칙 × 90일 내 2배 (Rare Jewels 가격 조건)", "", _md(sub.groupby(["rule", "90일2배"]).apply(_summ, include_groups=False).reset_index()), ""]
    liq40 = sub[(sub["rule"] == "40-Week") & sub["유동($20M↑)"]]
    out += ["### 40-Week · 유동 종목 — 진입 연도별", "", _md(liq40.groupby("진입연도").apply(_summ, include_groups=False).reset_index()), ""]
    top = sub[sub["rule"] == "40-Week"].nlargest(12, "pct_gain")[["symbol", "pattern", "kind", "entry_date", "pct_gain", "drawdown_k", "days", "reason"]]
    out += ["### 40-Week 상위 12 거래", "", _md(top), ""]
    return out


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if cfg.get("analyze"):
        with open(cfg["analyze"]) as f:
            results = json.load(f)["results"]
    else:
        results = run(cfg)
    md = report(results, cfg.get("title", "라이프사이클 트레이드 배치 검증"))
    print(md)
    if cfg.get("report_md"):
        with open(cfg["report_md"], "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
