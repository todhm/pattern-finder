"""Trade Sharp ORB long-only backtest runner.

Pairs with :class:`TradeSharpORBDetector`. One position at a time,
bar-by-bar simulation on the 5-minute frame. Per signal:

  - Enter at signal's ``entry_price`` on its ``entry_ts``. Buy-stop
    semantics — the order was parked at ``rejection_high`` and got
    hit when the entry bar's high crossed it.
  - Exit when EITHER the bar's low touches ``stop_loss`` (filled at
    stop) OR the bar's high reaches ``take_profit`` (filled at
    target). Stop is checked first — same-bar tie goes conservative.
  - If neither fires by the session's last bar, force-flat at the
    last bar's close (intraday day-trade hygiene; the video
    scenarios all close intraday).

Position sizing: classic risk-based — ``shares = floor(risk_dollars
/ (entry_price - stop_loss))``. Capped at
``max_position_pct_of_equity`` × equity to keep notional in line
with capital and prevent commission blow-up on tight stops.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pattern.adapters.tradesharp_orb import (
    TradeSharpORBDetector,
    TradeSharpORBSignal,
)
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    Trade,
)


class TradeSharpORBStrategy:
    """Single-ticker long-only Trade Sharp ORB backtester."""

    def __init__(
        self,
        detector: TradeSharpORBDetector,
        max_position_pct_of_equity: float = 0.30,
    ) -> None:
        self.detector = detector
        self.max_position_pct_of_equity = max_position_pct_of_equity

    def run(
        self,
        df_5m: pd.DataFrame,
        df_daily: pd.DataFrame,
        config: StrategyConfig,
    ) -> StrategyResult:
        signals = self.detector.detect(df_5m, df_daily)
        trades, equity_curve = self._simulate(df_5m, signals, config)
        perf = self._performance(trades, config)
        return StrategyResult(
            config=config,
            performance=perf,
            equity_curve=equity_curve,
        )

    # ---- internals -------------------------------------------------

    def _simulate(
        self,
        df_5m: pd.DataFrame,
        signals: list[TradeSharpORBSignal],
        config: StrategyConfig,
    ) -> tuple[list[Trade], list[EquityPoint]]:
        sig_by_ts = {s.entry_ts: s for s in signals}
        if df_5m.empty:
            return [], [
                EquityPoint(date=config.start_date, equity=config.initial_capital)
            ]
        local_dates = [ts.date() for ts in df_5m.index]

        equity = float(config.initial_capital)
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        open_pos: dict | None = None

        idx = df_5m.index
        highs = df_5m["High"].to_numpy(dtype=float)
        lows = df_5m["Low"].to_numpy(dtype=float)
        closes = df_5m["Close"].to_numpy(dtype=float)

        for i in range(len(df_5m)):
            ts = idx[i]
            sess = local_dates[i]
            is_last_in_session = (
                i == len(df_5m) - 1 or local_dates[i + 1] != sess
            )

            if open_pos is None:
                sig = sig_by_ts.get(ts)
                if sig is not None:
                    risk_per_share = sig.entry_price - sig.stop_loss
                    if risk_per_share <= 0:
                        continue
                    risk_dollars = equity * config.risk_per_trade
                    risk_shares = int(risk_dollars // risk_per_share)
                    if (
                        self.max_position_pct_of_equity > 0
                        and sig.entry_price > 0
                    ):
                        max_notional = (
                            equity * self.max_position_pct_of_equity
                        )
                        cap_shares = int(max_notional // sig.entry_price)
                        shares = min(risk_shares, cap_shares)
                    else:
                        shares = risk_shares
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
            if open_pos is None:
                continue

            exit_price: float | None = None
            exit_reason: str | None = None
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
                equity_curve.append(EquityPoint(date=sess, equity=equity))

        if not equity_curve:
            equity_curve.append(
                EquityPoint(date=config.start_date, equity=equity)
            )
        return trades, equity_curve

    @staticmethod
    def _performance(
        trades: list[Trade], config: StrategyConfig
    ) -> StrategyPerformance:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        final = config.initial_capital + total_pnl
        avg_win_pct = (
            sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss_pct = (
            sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
        )
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
            final_capital=final,
            total_return_pct=(
                (final - config.initial_capital) / config.initial_capital
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
    if ts.tzinfo is not None:
        return ts.tz_convert(ts.tz).tz_localize(None).to_pydatetime()
    return ts.to_pydatetime()
