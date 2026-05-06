"""First Candle Rule long-only backtest runner.

Pairs with :class:`FirstCandleRuleDetector`. Single position, bar-by-
bar 1-minute simulation. Entry on the engulfing-bar close, exit on
stop or TP (3:1 R/R per the video), force-flat at session close.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pattern.adapters.first_candle_rule import (
    FirstCandleRuleDetector,
    FirstCandleRuleSignal,
)
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    Trade,
)


class FirstCandleRuleStrategy:
    """Single-ticker long-only First Candle Rule backtester."""

    def __init__(
        self,
        detector: FirstCandleRuleDetector,
        max_position_pct_of_equity: float = 0.30,
        max_below_stop_strikes: int = 3,
        breakeven_after_minutes: int = 0,
        breakeven_after_r_multiple: float = 0.0,
    ) -> None:
        self.detector = detector
        self.max_position_pct_of_equity = max_position_pct_of_equity
        # **Tolerant stop — N-strike rule.**
        self.max_below_stop_strikes = max_below_stop_strikes
        # **Break-even stop arming (Crabel rule).** Once one of these
        # triggers fires, the stop level is raised from its initial
        # value (``fvg_pre_low − tick``) up to entry price. From
        # then on a tag of entry price closes the trade at $0 PnL
        # net of fees instead of the original -1R loss.
        #
        # Why this matters for FCR specifically: 5-month NASDAQ100
        # backtest showed 34/85 trades end as ``session_close`` with
        # avg -0.99R — i.e. they drift down all afternoon without
        # hitting either TP or initial stop. Arming BE converts a
        # chunk of those -1R drift losses into ~0R scratches.
        #
        # Two arming triggers (whichever fires first wins):
        #   - ``breakeven_after_minutes``: wall-clock minutes since
        #     entry. Crabel: "the ideal trade shows profit
        #     instantaneously; the longer it stays flat, the more
        #     vulnerable." Default 0 = disabled.
        #   - ``breakeven_after_r_multiple``: bar high reaches
        #     ``entry + R × initial_risk``. Default 0 = disabled.
        self.breakeven_after_minutes = breakeven_after_minutes
        self.breakeven_after_r_multiple = breakeven_after_r_multiple

    def run(
        self,
        df_intraday: pd.DataFrame,
        df_daily: pd.DataFrame,
        config: StrategyConfig,
    ) -> StrategyResult:
        signals = self.detector.detect(df_intraday, df_daily)
        trades, equity_curve = self._simulate(df_intraday, signals, config)
        perf = self._performance(trades, config)
        return StrategyResult(
            config=config,
            performance=perf,
            equity_curve=equity_curve,
        )

    def _simulate(
        self,
        df: pd.DataFrame,
        signals: list[FirstCandleRuleSignal],
        config: StrategyConfig,
    ) -> tuple[list[Trade], list[EquityPoint]]:
        sig_by_ts = {s.entry_ts: s for s in signals}
        if df.empty:
            return [], [
                EquityPoint(date=config.start_date, equity=config.initial_capital)
            ]
        local_dates = [ts.date() for ts in df.index]

        equity = float(config.initial_capital)
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        open_pos: dict | None = None

        idx = df.index
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)

        for i in range(len(df)):
            ts = idx[i]
            sess = local_dates[i]
            is_last = (
                i == len(df) - 1 or local_dates[i + 1] != sess
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
                        "entry_ts": ts,
                        "entry_price": sig.entry_price,
                        "initial_stop": sig.stop_loss,
                        "stop": sig.stop_loss,
                        "tp": sig.take_profit,
                        "shares": shares,
                        "initial_risk": risk_per_share,
                        # N-strike stop bookkeeping: ``below_streak``
                        # is True while the current run of bars has
                        # been continuously below the stop; rolls a
                        # single strike per excursion.
                        "below_stop_streak": False,
                        "below_stop_strikes": 0,
                        # BE arming flag — once True the stop level
                        # is raised to entry price and stays there.
                        "be_armed": False,
                    }
            if open_pos is None:
                continue

            exit_price: float | None = None
            exit_reason: str | None = None
            on_entry_bar = ts == open_pos["entry_ts"]
            # ---- Break-even arming (Crabel rule) ----
            # Two arming triggers, whichever fires first:
            #   - ``breakeven_after_r_multiple > 0``: bar high
            #     reaches entry + R × initial_risk
            #   - ``breakeven_after_minutes > 0``: wall-clock minutes
            #     since entry exceed the threshold
            # On arm: raise stop from initial level to entry price.
            # Idempotent — guarded by ``be_armed``.
            if not on_entry_bar and not open_pos["be_armed"]:
                arm = False
                if self.breakeven_after_r_multiple > 0:
                    be_target = (
                        open_pos["entry_price"]
                        + self.breakeven_after_r_multiple
                        * open_pos["initial_risk"]
                    )
                    if highs[i] >= be_target:
                        arm = True
                if not arm and self.breakeven_after_minutes > 0:
                    elapsed = (ts - open_pos["entry_ts"]).total_seconds() / 60
                    if elapsed >= self.breakeven_after_minutes:
                        arm = True
                if arm:
                    open_pos["stop"] = open_pos["entry_price"]
                    open_pos["be_armed"] = True

            # Stop bookkeeping — runs every bar after entry.
            #
            # Two modes:
            #   1. **Initial stop** (be_armed=False): N-strike tolerant.
            #      Each new excursion below the stop counts as one
            #      strike; the Nth strike fires the exit. Lets the
            #      first N-1 wicks slide as liquidity grabs.
            #   2. **BE-armed stop** (be_armed=True): single-tap.
            #      Once the stop is at entry price, any touch closes
            #      the trade at $0 PnL net of fees. The N-strike
            #      tolerance was designed for the *initial* stop
            #      below the FVG-pre-low; the BE level represents a
            #      "scratch the trade" decision and should fire on
            #      the first tag, not the Nth excursion. Matches
            #      the comment above: "a tag of entry price closes
            #      the trade at $0 PnL".
            if not on_entry_bar:
                if lows[i] < open_pos["stop"]:
                    if open_pos["be_armed"]:
                        exit_price = open_pos["stop"]
                        exit_reason = "breakeven_stop"
                    else:
                        if not open_pos["below_stop_streak"]:
                            open_pos["below_stop_strikes"] += 1
                            open_pos["below_stop_streak"] = True
                        if (
                            self.max_below_stop_strikes > 0
                            and open_pos["below_stop_strikes"]
                            >= self.max_below_stop_strikes
                        ):
                            exit_price = open_pos["stop"]
                            exit_reason = "stop_loss"
                else:
                    open_pos["below_stop_streak"] = False

                # TP only checked if stop didn't fire this bar.
                if exit_price is None and highs[i] >= open_pos["tp"]:
                    exit_price = open_pos["tp"]
                    exit_reason = "take_profit"

            if exit_price is None and is_last:
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
                        entry_ts=_to_naive(open_pos["entry_ts"]),
                        exit_ts=_to_naive(ts),
                    )
                )
                equity += pnl
                open_pos = None

            if is_last:
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
            peak = max(peak, running)
            dd = (peak - running) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
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


def _to_naive(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is not None:
        return ts.tz_convert(ts.tz).tz_localize(None).to_pydatetime()
    return ts.to_pydatetime()
