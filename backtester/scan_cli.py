"""Command-line scan runner — mirrors the Multi Wedgepop Streamlit page.

Used for iterative strategy tuning without Chrome/Streamlit round-trips.
Reads a small YAML-ish dict of knobs from argv JSON, runs the full
multi-ticker wedge-pop scan, prints key metrics + trade breakdown as
JSON on stdout so the caller can diff iterations.

Run inside the docker container:
    docker compose exec backtester python scan_cli.py '{"max_tickers": 150, ...}'
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date, timedelta

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.multi_wedgepop_strategy import MultiWedgepopStrategy
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule


def run(knobs: dict) -> dict:
    today = date.today()
    start_date = knobs.get("start_date") or (today - timedelta(days=365))
    end_date = knobs.get("end_date") or today
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    universe_provider = WikipediaUniverseAdapter()

    market_regime_df = None
    if knobs.get("enable_market_regime_filter", True):
        fetch_start = start_date - timedelta(days=400)
        market_regime_df = market_data.fetch_ohlcv("SPY", fetch_start, end_date)

    detector = WedgePopDetector(
        lookback=int(knobs.get("detect_lookback", 10)),
        ema_fast=int(knobs.get("ema_fast", 10)),
        ema_slow=int(knobs.get("ema_slow", 20)),
        consolidation_pct=float(knobs.get("consolidation_pct", 30.0)) / 100.0,
        max_consolidation_pct=(
            float(knobs["max_consolidation_pct"]) / 100.0
            if knobs.get("enable_max_cp") and "max_consolidation_pct" in knobs
            else None
        ),
        breakout_atr_mult=float(knobs.get("breakout_atr_mult", 0.005)),
        max_breakout_atr_mult=(
            float(knobs["max_breakout_atr_mult"])
            if knobs.get("enable_max_bp") and "max_breakout_atr_mult" in knobs
            else None
        ),
        slope_lookback=int(knobs.get("slope_lookback", 10)),
        cooldown_bars=int(knobs.get("cooldown_bars", 0)),
        require_above_long_smas=bool(knobs.get("require_above_long_smas", True)),
        late_entry_bars=int(knobs.get("late_entry_bars", 0)),
    )

    exit_detector = None
    if knobs.get("enable_exh_exit", True):
        exit_detector = ExhaustionExtensionTopDetector(
            extension_atr_mult=float(knobs.get("exh_exit_extension_atr", 1.9)),
            min_slow_slope=float(knobs.get("exh_exit_min_slope", 0.005)),
            max_close_location=float(knobs.get("exh_exit_max_close_loc", 0.5)),
            min_sell_dominance=float(knobs.get("exh_exit_min_sell_dom", 1.5)),
            enable_rejection_override=bool(
                knobs.get("exh_exit_rejection_override", True)
            ),
            ema_fast=int(knobs.get("ema_fast", 10)),
            ema_slow=int(knobs.get("ema_slow", 20)),
            slope_lookback=int(knobs.get("slope_lookback", 10)),
        )

    strategy = WedgepopStrategy(
        market_data=market_data,
        detector=detector,
        ema_trail=int(knobs.get("ema_fast", 10)),
        ema_slow=int(knobs.get("ema_slow", 20)),
        max_entry_ema_extension_atr=(
            float(knobs.get("max_entry_ema_extension_atr", 1.5))
            if knobs.get("enable_entry_ema_filter", True)
            else None
        ),
        max_ema_slope_decline=None,
        min_ema_slow_slope=(
            float(knobs.get("min_ema_slow_slope", 0.005))
            if knobs.get("enable_min_slope", True)
            else None
        ),
        max_ema_slow_slope=(
            float(knobs.get("max_ema_slow_slope", 0.30))
            if knobs.get("enable_max_slope", False)
            else None
        ),
        require_gap_up=bool(knobs.get("require_gap_up", False)),
        use_smart_trail=bool(knobs.get("use_smart_trail", False)),
        exit_detector=exit_detector,
        enable_swing_resistance_filter=bool(knobs.get("enable_swing_resistance", True)),
        swing_pivot_left=int(knobs.get("swing_pivot_left", 2)),
        swing_pivot_right=int(knobs.get("swing_pivot_right", 2)),
        swing_pivot_lookback=int(knobs.get("swing_pivot_lookback", 60)),
        swing_resistance_tolerance_atr=float(
            knobs.get("swing_resistance_tol_atr", 0.5)
        ),
        enable_trendline_exit=bool(knobs.get("enable_trendline_exit", True)),
        trendline_max_pivots=int(knobs.get("trendline_max_pivots", 3)),
        enable_resistance_break_exit=bool(
            knobs.get("enable_resistance_break_exit", True)
        ),
        resistance_break_pierce_buffer_atr=float(
            knobs.get("resistance_break_pierce_buffer_atr", 0.5)
        ),
        resistance_break_confirm_buffer_atr=float(
            knobs.get("resistance_break_confirm_buffer_atr", 0.1)
        ),
        enable_market_regime_filter=bool(
            knobs.get("enable_market_regime_filter", True)
        ),
        market_regime_df=market_regime_df,
        enable_breakeven_stop=bool(knobs.get("enable_breakeven_stop", True)),
        enable_gap_down_rejection=bool(knobs.get("enable_gap_down_rejection", True)),
        enable_signal_close_strength_filter=bool(
            knobs.get("enable_signal_close_strength", False)
        ),
        min_signal_close_location=float(knobs.get("min_signal_close_location", 0.5)),
        enable_swing_breakout_filter=bool(knobs.get("enable_swing_breakout", False)),
        swing_breakout_buffer_atr=float(knobs.get("swing_breakout_buffer_atr", 0.0)),
        max_signal_bar_gain_atr=(
            float(knobs.get("max_signal_bar_gain_atr", 2.5))
            if knobs.get("enable_euphoria_cap", True)
            else None
        ),
        breakeven_arm_r_multiple=float(knobs.get("breakeven_arm_r_multiple", 1.0)),
        structural_exit_grace_bars=int(knobs.get("structural_exit_grace_bars", 0)),
        breakeven_exit_offset_r=float(knobs.get("breakeven_exit_offset_r", 0.0)),
        structural_exit_close_confirm=bool(
            knobs.get("structural_exit_close_confirm", False)
        ),
    )

    runner = MultiWedgepopStrategy(
        market_data=market_data,
        universe_provider=universe_provider,
        detector=detector,
        strategy=strategy,
        max_workers=int(knobs.get("max_workers", 8)),
    )

    fee_schedule = TossFeeSchedule(
        buy_commission_pct=float(knobs.get("buy_fee_pct", 0.10)) / 100.0,
        sell_commission_pct=float(knobs.get("sell_fee_pct", 0.10)) / 100.0,
        sec_fee_pct=float(knobs.get("sec_fee_pct", 0.00229)) / 100.0,
    )

    config = MultiStrategyConfig(
        universe=knobs.get("universe", "sp500"),
        start_date=start_date,
        end_date=end_date,
        pattern_name="wedge_pop",
        initial_capital=float(knobs.get("initial_capital", 100_000)),
        risk_per_trade=float(knobs.get("risk_pct", 5.0)) / 100.0,
        max_holding_days=int(knobs.get("max_holding_days", 60)),
        max_tickers=(
            int(knobs["max_tickers"]) if knobs.get("max_tickers", 0) > 0 else None
        ),
        fee_schedule=fee_schedule,
    )

    result = runner.run(config)

    exit_reasons = Counter(t.exit_reason for t in result.trades)
    pnls = [t.pnl_pct for t in result.trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    trade_rows = [
        {
            "ticker": t.ticker,
            "entry": t.entry_date.isoformat(),
            "exit": t.exit_date.isoformat(),
            "days": (t.exit_date - t.entry_date).days,
            "reason": t.exit_reason,
            "pnl_pct": round(t.pnl_pct * 100, 2),
            "gross_pnl": round(t.gross_pnl, 2),
            "net_pnl": round(t.pnl, 2),
        }
        for t in result.trades
    ]

    return {
        "tickers_scanned": result.tickers_scanned,
        "total_signals": result.total_signals,
        "trades_taken": result.trades_taken,
        "total_return_pct": round(result.total_return_pct * 100, 2),
        "win_rate_pct": round(result.win_rate * 100, 2),
        "max_dd_pct": round(result.max_drawdown_pct * 100, 2),
        "final_capital": round(result.final_capital, 2),
        "gross_pnl": round(sum(t.gross_pnl for t in result.trades), 2),
        "net_pnl": round(sum(t.pnl for t in result.trades), 2),
        "commission": round(result.total_commission, 2),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win_pct": round(sum(wins) / len(wins) * 100, 2) if wins else 0,
        "avg_loss_pct": round(sum(losses) / len(losses) * 100, 2) if losses else 0,
        "exit_reasons": dict(exit_reasons),
        "trades": trade_rows,
    }


if __name__ == "__main__":
    # argv[1] may be a single knobs dict OR a list of {"label": ..., "knobs": ...}
    # entries. Batched form avoids paying docker-exec startup overhead per variant.
    arg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if isinstance(arg, list):
        out = []
        for entry in arg:
            label = entry.get("label", "")
            knobs = entry.get("knobs", {})
            try:
                r = run(knobs)
                r["label"] = label
            except Exception as e:
                r = {"label": label, "error": str(e)}
            out.append(r)
        print(json.dumps(out, default=str))
    else:
        print(json.dumps(run(arg), default=str))
