"""Scraface ORB long-only backtest runner.

Pairs with :class:`ScrafaceORBDetector`. One position at a time,
bar-by-bar 1-minute simulation. Per signal:

  - Enter at signal's ``entry_price`` on its ``entry_ts``.
  - Exit when EITHER the bar's low touches ``stop_loss`` (filled at
    stop, ``stop_loss``) OR the bar's high reaches ``take_profit``
    (filled at target). Stop is checked first — same-bar tie goes to
    the conservative side.
  - If neither fires by session end, force-close at the session's
    last 1-minute bar's close (intraday day-trade hygiene).

Position sizing: classic risk-based — ``shares = floor(risk_dollars /
(entry_price - stop_loss))``. ``risk_dollars`` = current equity ×
``risk_per_trade``.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pattern.adapters.scraface_orb import (
    ScrafaceORBDetector,
    ScrafaceORBSignal,
)
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    Trade,
)


class ScrafaceORBStrategy:
    """Single-ticker long-only ORB retest backtester."""

    def __init__(self, detector: ScrafaceORBDetector) -> None:
        self.detector = detector

    def run(
        self,
        df_1m: pd.DataFrame,
        df_daily: pd.DataFrame,
        config: StrategyConfig,
    ) -> StrategyResult:
        signals = self.detector.detect(df_1m, df_daily)
        trades, equity_curve = self._simulate(df_1m, signals, config)
        perf = self._performance(trades, config)
        return StrategyResult(
            config=config,
            performance=perf,
            equity_curve=equity_curve,
        )

    # ---- internals --------------------------------------------------

    def _simulate(
        self,
        df_1m: pd.DataFrame,
        signals: list[ScrafaceORBSignal],
        config: StrategyConfig,
    ) -> tuple[list[Trade], list[EquityPoint]]:
        # Index by entry_ts so the bar loop can dispatch in O(1).
        sig_by_ts = {s.entry_ts: s for s in signals}
        # Pre-compute session_date per bar so the force-close check
        # at session boundary is just a date compare.
        if df_1m.empty:
            return [], [
                EquityPoint(date=config.start_date, equity=config.initial_capital)
            ]
        local_dates = self._local_dates(df_1m.index)

        equity = float(config.initial_capital)
        trades: list[Trade] = []
        # Equity curve: one point per session (close of last bar).
        equity_curve: list[EquityPoint] = []

        open_pos: dict | None = None
        idx = df_1m.index
        highs = df_1m["High"].to_numpy(dtype=float)
        lows = df_1m["Low"].to_numpy(dtype=float)
        closes = df_1m["Close"].to_numpy(dtype=float)

        for i in range(len(df_1m)):
            ts = idx[i]
            sess = local_dates[i]
            is_last_in_session = (
                i == len(df_1m) - 1 or local_dates[i + 1] != sess
            )

            if open_pos is None:
                sig = sig_by_ts.get(ts)
                if sig is not None:
                    risk_per_share = sig.entry_price - sig.stop_loss
                    if risk_per_share <= 0:
                        continue
                    risk_dollars = equity * config.risk_per_trade
                    shares = int(risk_dollars // risk_per_share)
                    if shares <= 0:
                        continue
                    open_pos = {
                        "signal": sig,
                        "entry_ts": ts,
                        "entry_price": sig.entry_price,
                        "stop": sig.stop_loss,
                        "tp": sig.take_profit,
                        "shares": shares,
                    }
                # Even on entry bar we still check session-end below
                # (rare, but a signal triggering on the very last bar
                # of the session would otherwise leave the trade open
                # past the close).
            if open_pos is None:
                # No-op bar — skip session-end logic, advance.
                continue

            # Same-bar exit checks. Stop first (conservative).
            exit_price: float | None = None
            exit_reason: str | None = None
            # Don't fire stop/tp on the entry bar itself when entry
            # already equals close — the bar's low/high happened
            # *before* our entry. Use a strict ts > entry_ts gate.
            on_entry_bar = ts == open_pos["entry_ts"]
            if not on_entry_bar:
                if lows[i] <= open_pos["stop"]:
                    exit_price = open_pos["stop"]
                    exit_reason = "stop_loss"
                elif highs[i] >= open_pos["tp"]:
                    exit_price = open_pos["tp"]
                    exit_reason = "take_profit"

            if exit_price is None and is_last_in_session:
                exit_price = float(closes[i])
                exit_reason = "session_close"

            if exit_price is not None:
                shares = open_pos["shares"]
                pnl = (exit_price - open_pos["entry_price"]) * shares
                pnl_pct = (
                    (exit_price - open_pos["entry_price"])
                    / open_pos["entry_price"]
                )
                trades.append(
                    Trade(
                        pattern_name=config.pattern_name,
                        entry_date=open_pos["entry_ts"].date(),
                        exit_date=ts.date(),
                        entry_price=open_pos["entry_price"],
                        exit_price=exit_price,
                        stop_loss=open_pos["stop"],
                        shares=shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason or "end_of_data",
                        entry_ts=_to_naive_dt(open_pos["entry_ts"]),
                        exit_ts=_to_naive_dt(ts),
                    )
                )
                equity += pnl
                open_pos = None

            if is_last_in_session:
                equity_curve.append(
                    EquityPoint(date=sess, equity=equity)
                )

        # Trailing equity point if we ended mid-session with no entry.
        if not equity_curve:
            equity_curve.append(
                EquityPoint(date=config.start_date, equity=equity)
            )
        return trades, equity_curve

    @staticmethod
    def _local_dates(idx: pd.DatetimeIndex):
        # Bar timestamps from EODHD intraday come back tz-aware
        # (America/New_York). ``ts.date()`` already reads off the
        # local-tz calendar date — no extra conversion needed.
        return [ts.date() for ts in idx]

    @staticmethod
    def _performance(
        trades: list[Trade], config: StrategyConfig
    ) -> StrategyPerformance:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        final_capital = config.initial_capital + total_pnl
        avg_win_pct = (
            sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss_pct = (
            sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
        )
        # Max drawdown from the running equity curve.
        max_dd = 0.0
        peak = config.initial_capital
        running = config.initial_capital
        for t in trades:
            running += t.pnl
            if running > peak:
                peak = running
            dd = (peak - running) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return StrategyPerformance(
            initial_capital=config.initial_capital,
            final_capital=final_capital,
            total_return_pct=(
                (final_capital - config.initial_capital) / config.initial_capital
                if config.initial_capital > 0
                else 0.0
            ),
            total_trades=len(trades),
            win_rate=(len(wins) / len(trades) if trades else 0.0),
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            max_drawdown_pct=max_dd,
            trades=trades,
        )


def _to_naive_dt(ts: pd.Timestamp) -> datetime:
    """Strip tz for the Trade model (pydantic stores naive datetimes)."""
    if ts.tzinfo is not None:
        return ts.tz_convert(ts.tz).tz_localize(None).to_pydatetime()
    return ts.to_pydatetime()
