"""⑩ 급락 회피 — 시작 시점별(롤링) 세후 성과 배치.

    docker compose exec backtester python deep_correction_cli.py '{"etf": "TQQQ", "signal": "QQQ", "years": [1,3,5,10], "out": "/tmp/dc.json"}'
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from datetime import date

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.deep_correction_lab import (
    DeepParams,
    MonthlySmaParams,
    signal_monthly_sma,
    avoided_segments,
    current_state,
    param_grid,
    rolling_study,
    rolling_summary,
    run_pair,
    signal,
)
from strategy.adapters.leveraged_etf_data import UNDERLYING_START, ETF_SPECS, load_lab_prices
from strategy.adapters.leveraged_etf_lab import CostParams, crisis_table, yearly_returns


def run(cfg: dict) -> dict:
    etf, sig_sym = cfg.get("etf", "TQQQ"), cfg.get("signal", "QQQ")
    spec = ETF_SPECS[etf]
    early = date.fromisoformat(cfg["early_start"]) if cfg.get("early_start") else max(UNDERLYING_START.get(spec.underlying, spec.inception), date(1999, 12, 1))
    end = date.today()
    rule = cfg.get("rule", "deep")
    p = MonthlySmaParams(**cfg.get("params", {})) if rule == "monthly_sma" else DeepParams(**cfg.get("params", {}))
    costs = CostParams(tax_rate=0.22 if cfg.get("tax", True) else 0.0, exec_lag=int(cfg.get("exec_lag", 0)), rebalance_band=0.02)
    md = build_default_market_data()
    pr = load_lab_prices(md, etf, sig_sym, early, end, synthetic=True)
    s = signal_monthly_sma(pr.signal_close, p) if rule == "monthly_sma" else signal(pr.signal_close, p)
    out = {"etf": etf, "signal": sig_sym, "rule": rule, "params": asdict(p), "costs": asdict(costs), "source": pr.source_note, "full": {}, "rolling": {}}
    for label, start in (("전체(합성 포함)", early), ("실데이터", spec.inception)):
        hold, strat = run_pair(pr.lev_close, s, start, None, costs, pr.cash_rate)
        out["full"][label] = {
            "start": str(hold.equity.index[0].date()), "hold_mult": hold.equity.iloc[-1] / hold.equity.iloc[0],
            "strat_mult": strat.equity.iloc[-1] / strat.equity.iloc[0], "hold": hold.metrics, "strat": strat.metrics, "trades": strat.trades,
            "yearly": yearly_returns({"⑩": strat, "보유": hold}).rename_axis("year").reset_index().to_dict(orient="records"),
            "crisis": crisis_table({"⑩": strat, "보유": hold}).assign(시작=lambda d: d["시작"].astype(str), 끝=lambda d: d["끝"].astype(str)).to_dict(orient="records"),
            "segments": avoided_segments(strat, pr.lev_close, pr.signal_close).to_dict(orient="records"),
        }
        print(f"[dc] {etf} {label} 보유 {out['full'][label]['hold_mult']:.1f}x ⑩ {out['full'][label]['strat_mult']:.1f}x", file=sys.stderr)
    for y in cfg.get("years", [1, 3, 5, 10]):
        df = rolling_study(pr.lev_close, s, y, costs, pr.cash_rate, step_months=int(cfg.get("step_months", 3)))
        out["rolling"][str(y)] = {"summary": rolling_summary(df), "rows": df.to_dict(orient="records")}
        sm = out["rolling"][str(y)]["summary"]
        print(f"[dc] {y}년 창 {sm.get('windows')}개 승률 {sm.get('win_rate', 0):.0%} 중앙 비율 {sm.get('ratio_median', 0):.2f}", file=sys.stderr)
    if rule == "monthly_sma":
        from strategy.adapters.deep_correction_lab import decision_monthly, param_grid_monthly
        out["grid"] = param_grid_monthly(pr.lev_close, pr.signal_close, spec.inception, costs, pr.cash_rate).to_dict(orient="records")
        out["now"] = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in decision_monthly(pr.signal_close, p).items()}
    else:
        out["grid"] = param_grid(pr.lev_close, pr.signal_close, spec.inception, costs, pr.cash_rate).to_dict(orient="records")
        out["now"] = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in current_state(pr.signal_close, p).items()}
    return out


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    runs = cfg.get("runs") or [cfg]
    res = [run(c) for c in runs]
    if cfg.get("out"):
        json.dump(res, open(cfg["out"], "w"), ensure_ascii=False, default=str)
    else:
        print(json.dumps(res, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
