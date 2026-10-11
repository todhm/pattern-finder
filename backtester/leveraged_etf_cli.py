"""레버리지 ETF 전략 랩 배치 — 페이지 47 과 같은 엔진으로 여러 ETF·기간·파라미터를 돌려 JSON + 마크다운 보고서.

    docker compose exec backtester python leveraged_etf_cli.py '{"etf": "TQQQ", "signal": "QQQ", "start": "1999-06-01"}'
    docker compose exec backtester python leveraged_etf_cli.py '{"runs": [{...}, {...}], "report_md": "/tmp/lev.md"}'

config 키: etf, signal, start, end, synthetic(bool), params{sma_n, sma_enter, sma_exit, sma_weekly, vol_target,
vol_lookback, vol_max, sig_stock_weight, sig_growth}, costs{tax(bool), exec_lag, rebalance_band}, grid(bool: SMA/변동성 민감도)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.leveraged_etf_data import load_lab_prices
from strategy.adapters.short_term_lab import SHORT_TERM_ORDER, ShortTermParams, run_short_term, trade_stats
from strategy.adapters.leveraged_etf_lab import (
    STRATEGY_ORDER,
    CostParams,
    LabParams,
    crisis_table,
    current_signal,
    param_grid,
    run_lab,
    summary_table,
    yearly_returns,
)


def _costs(cfg: dict) -> CostParams:
    c = cfg.get("costs", {})
    return CostParams(tax_rate=0.22 if c.get("tax", True) else 0.0, exec_lag=int(c.get("exec_lag", 0)),
                      rebalance_band=float(c.get("rebalance_band", 0.02)))


def run_one(md, cfg: dict) -> dict:
    etf, signal = cfg.get("etf", "TQQQ"), cfg.get("signal", "QQQ")
    start = date.fromisoformat(cfg.get("start", "1999-06-01"))
    end = date.fromisoformat(cfg["end"]) if cfg.get("end") else date.today()
    p = LabParams(**cfg.get("params", {}))
    costs = _costs(cfg)
    print(f"[lev] {etf}/{signal} {start}~{end} synthetic={cfg.get('synthetic', True)}", file=sys.stderr)
    prices = load_lab_prices(md, etf, signal, start, end, synthetic=cfg.get("synthetic", True))
    res = run_lab(prices.lev_close, prices.signal_close, p, costs, prices.cash_rate, start)
    trade_rows = {}
    if cfg.get("short_term", True) and prices.signal_df is not None:
        sp = ShortTermParams(**cfg.get("short_params", {}))
        st_res = run_short_term(prices.lev_close, prices.signal_df, sp, costs, prices.cash_rate, start)
        res.update(st_res)
        trade_rows = {k: trade_stats(r, prices.lev_close) for k, r in st_res.items()}
    out = {
        "etf": etf, "signal": signal, "start": start.isoformat(), "end": end.isoformat(),
        "source": prices.source_note, "params": asdict(p), "costs": asdict(costs),
        "summary": summary_table(res).to_dict(orient="records"),
        "trade_stats": trade_rows,
        "yearly": yearly_returns(res).rename_axis("year").reset_index().to_dict(orient="records"),
        "crisis": crisis_table(res).assign(시작=lambda d: d["시작"].astype(str), 끝=lambda d: d["끝"].astype(str))
                  .to_dict(orient="records"),
        "now": {k: (v.isoformat() if isinstance(v, date) else v)
                for k, v in current_signal(prices.lev_close, prices.signal_close, p).items()},
        "equity_monthly": {k: r.equity.resample("ME").last().round(0).to_dict() for k, r in res.items()},
    }
    if cfg.get("grid", True):
        sma_grid = [LabParams(sma_n=n, sma_enter=e, sma_exit=x, sma_weekly=w)
                    for n in (100, 150, 200, 250) for (e, x) in ((0.0, 0.0), (0.03, 0.03), (0.05, 0.03))
                    for w in (False, True)]
        vol_grid = [LabParams(vol_target=t, vol_lookback=lb) for t in (0.30, 0.45, 0.60) for lb in (10, 20, 40)]
        out["grid_sma"] = param_grid(prices.lev_close, prices.signal_close, sma_grid, costs, prices.cash_rate, start,
                                     STRATEGY_ORDER[1]).to_dict(orient="records")
        out["grid_vol"] = param_grid(prices.lev_close, prices.signal_close, vol_grid, costs, prices.cash_rate, start,
                                     STRATEGY_ORDER[2]).to_dict(orient="records")
        out["grid_combo"] = param_grid(prices.lev_close, prices.signal_close,
                                       [LabParams(sma_n=n, vol_target=t) for n in (150, 200) for t in (0.30, 0.45, 0.60)],
                                       costs, prices.cash_rate, start, STRATEGY_ORDER[3]).to_dict(orient="records")
    # 월별 평가액의 키를 문자열로
    out["equity_monthly"] = {k: {ts.strftime("%Y-%m"): v for ts, v in d.items()} for k, d in out["equity_monthly"].items()}
    return out


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:+.1f}%"


def report_md(runs: list[dict]) -> str:
    L = ["# 레버리지 ETF 사고팔기 전략 — 장기보유 대비 검증 (자동 생성)", ""]
    for r in runs:
        L += [f"## {r['etf']} (신호 {r['signal']}) · {r['start']} ~ {r['end']}", f"> {r['source']}", "",
              "| 전략 | 최종 배수 | CAGR | MDD | 최악 연도 | 샤프 | 매매 | 시장 노출 | 수수료+세금 | 청산 후 배수 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in r["summary"]:
            L.append(f"| {s['전략']} | {s['최종 배수']:.1f}x | {_pct(s['CAGR'])} | {_pct(s['MDD'])} | {_pct(s['최악 연도'])} | "
                     f"{s['샤프']:.2f} | {s['매매 횟수']} | {s['시장 노출'] * 100:.0f}% | {s['수수료+세금'] * 100:.0f}% | "
                     f"{s['청산 후 배수']:.1f}x |")
        cols = [s["전략"] for s in r["summary"]]
        if r.get("trade_stats"):
            L += ["", "### 단기 전략 거래 통계 (가격 기준, 비용 전)", "", "| 전략 | 거래 | 연간 | 승률 | 평균 보유일 | 평균 | 중앙값 | 최악 | 최고 |", "|---|---|---|---|---|---|---|---|---|"]
            for k, s in r["trade_stats"].items():
                L.append(f"| {k} | {s['trades']} | {s['per_year']:.1f} | {_pct(s.get('win_rate'))} | {s.get('avg_days', float('nan')):.1f} | "
                         f"{_pct(s.get('avg_ret'))} | {_pct(s.get('median_ret'))} | {_pct(s.get('worst'))} | {_pct(s.get('best'))} |")
        L += ["", "### 위기 구간 (고점→저점)", "", "| 구간 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for c in r["crisis"]:
            L.append(f"| {c['구간']} {c['시작']}~{c['끝']} | " + " | ".join(_pct(c.get(k)) for k in cols) + " |")
        L += ["", "### 연도별", "", "| 연도 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for y in r["yearly"]:
            year = y.get("year", next((v for k, v in y.items() if k not in cols), "?"))
            L.append(f"| {year} | " + " | ".join(_pct(y.get(k)) for k in cols) + " |")
        n = r["now"]
        L += ["", f"### 지금 신호 ({n['as_of']})", "",
              f"- 신호 지수 {n['signal_close']:.2f} vs SMA {n['sma']:.2f} ({_pct(n['dist_to_sma'])}) → "
              f"필터 {'보유' if n['filter_on'] else '현금'}",
              f"- 실현 변동성 {n['realized_vol'] * 100:.0f}% → 타기팅 비중 {n['vol_weight'] * 100:.0f}%, "
              f"결합 비중 {n['combined_weight'] * 100:.0f}%", ""]
        for key, title in (("grid_sma", "SMA 필터 민감도"), ("grid_vol", "변동성 타기팅 민감도"), ("grid_combo", "결합 민감도")):
            if key not in r:
                continue
            df = pd.DataFrame(r[key])
            L += [f"### {title}", ""]
            cols = [c for c in ("sma_n", "enter", "exit", "weekly", "vol_target", "lookback") if df[c].nunique() > 1]
            L += ["| " + " | ".join(cols + ["CAGR", "MDD", "샤프", "매매", "최종 배수"]) + " |", "|---|" + "---|" * (len(cols) + 4)]
            for _, g in df.iterrows():
                L.append("| " + " | ".join(str(g[c]) for c in cols) +
                         f" | {_pct(g['CAGR'])} | {_pct(g['MDD'])} | {g['샤프']:.2f} | {int(g['매매'])} | {g['최종 배수']:.1f}x |")
            L.append("")
    return "\n".join(L)


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if "analyze" in cfg:
        runs = json.load(open(cfg["analyze"]))
        print(report_md(runs))
        return
    md = build_default_market_data()
    runs = cfg.get("runs") or [cfg]
    results = [run_one(md, c) for c in runs]
    if cfg.get("out"):
        json.dump(results, open(cfg["out"], "w"), ensure_ascii=False, default=str)
    if cfg.get("report_md"):
        open(cfg["report_md"], "w").write(report_md(results))
        print(f"[lev] report → {cfg['report_md']}", file=sys.stderr)
    if not cfg.get("out") and not cfg.get("report_md"):
        print(json.dumps(results, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
