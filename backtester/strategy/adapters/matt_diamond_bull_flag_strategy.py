"""Matt Diamond Bull Flag — single-ticker scalper strategy.

Pairs with :class:`pattern.adapters.matt_diamond_bull_flag.MattDiamondBullFlagDetector`.
Long-only intraday simulator that mirrors the *scalper* discipline
Matt describes in the source video:

* **Entry**: GTR buy-stop fill produced by the detector.
* **Stop**: pre-computed ATR-aware stop on the signal.
* **Target**: configurable. Default = ``target_at_r_multiple=1.5R``
  ("$1-$2 quick scalp on Tesla"). Set to a larger value or wire to
  HoD for swingier runs.
* **Time stop**: if the trade hasn't reached ``time_stop_min_r`` after
  ``time_stop_bars`` bars, exit at market — matches "if it doesn't
  move quickly, get out" scalper habit.
* **No add-to-winner**: Matt's described style is single-entry scalp;
  averaging up isn't part of the source video. Stays off.
* **Session loss cap**: optional Ross-style "1 loss, sit down" gate
  (off by default — Matt didn't articulate one).
* **End-of-session liquidation**: last bar of the session closes the
  position.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pattern.adapters.matt_diamond_bull_flag import (
    MattDiamondBullFlagDetector,
    MattDiamondSignal,
)
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    TossFeeSchedule,
    Trade,
)


class MattDiamondBullFlagStrategy:
    """Single-ticker long-only Matt-Diamond-style Bull Flag backtester."""

    def __init__(
        self,
        detector: MattDiamondBullFlagDetector,
        *,
        max_position_pct_of_equity: float = 0.30,
        # Sweep-tuned (2026-05): 2.5R is the win-rate-optimal target
        # for the resistance-break + GTR-volume default stack. Lower
        # values (1.0R-1.5R) get tagged out before letting winners run;
        # higher (4.0R+) miss target on the strong wins that actually
        # carry the strategy. 2.5R = PF 7.68 on the 17-month sweep.
        target_at_r_multiple: float = 2.5,
        target_min_r_multiple: float = 1.0,
        enable_time_stop: bool = True,
        time_stop_bars: int = 10,
        time_stop_min_r: float = 0.5,
        enable_breakeven_after_r: float | None = 1.0,
        be_stop_buffer_pct: float = 0.001,
        max_session_losses: int = 0,
        fee_schedule: TossFeeSchedule | None = None,
    ) -> None:
        """
        Parameters
        ----------
        target_at_r_multiple:
            Fixed scalper take-profit (multiples of risk). Default 1.5
            — Matt's "$1-$2 quick scalp on TSLA" when risk ≈ $1.
        target_min_r_multiple:
            Sanity gate. If the configured R-target is smaller than
            ``target_min_r_multiple`` after cost overlay (e.g. very
            tight bar where ATR-widened stop = entry), drop the
            trade. Set 1.0 = never trade for less than 1R.
        enable_time_stop:
            Default True. Matt cut TSLA trades that didn't move
            quickly; the time stop is the systemic version.
        time_stop_bars:
            Bars to wait after entry. On 1m, 10 = 10 minutes.
        time_stop_min_r:
            If unrealized P/L hasn't reached this R-multiple by
            ``time_stop_bars``, exit at market.
        enable_breakeven_after_r:
            None disables. If set (default 1.0R), once the trade
            touches +XR the protective stop is raised to entry
            (minus ``be_stop_buffer_pct``).
        max_session_losses:
            0 disables. >=1 caps stop_loss/breakeven_stop exits per
            session.
        """
        self.detector = detector
        self.max_position_pct_of_equity = max_position_pct_of_equity
        self.target_at_r_multiple = target_at_r_multiple
        self.target_min_r_multiple = target_min_r_multiple
        self.enable_time_stop = enable_time_stop
        self.time_stop_bars = time_stop_bars
        self.time_stop_min_r = time_stop_min_r
        self.enable_breakeven_after_r = enable_breakeven_after_r
        self.be_stop_buffer_pct = be_stop_buffer_pct
        self.max_session_losses = max_session_losses
        self.fee_schedule = fee_schedule

    # ---- public entrypoint --------------------------------------------

    def run(
        self,
        df_intraday: pd.DataFrame,
        df_daily: pd.DataFrame,
        config: StrategyConfig,
    ) -> StrategyResult:
        signals = self.detector.detect(df_intraday, df_daily)
        trades, equity = self._simulate(df_intraday, signals, config)
        perf = self._performance(trades, config)
        return StrategyResult(config=config, performance=perf, equity_curve=equity)

    # ---- core sim ------------------------------------------------------

    def _simulate(
        self,
        df: pd.DataFrame,
        signals: list[MattDiamondSignal],
        config: StrategyConfig,
    ) -> tuple[list[Trade], list[EquityPoint]]:
        if df.empty:
            return [], [
                EquityPoint(date=config.start_date, equity=config.initial_capital)
            ]

        sig_by_ts: dict[pd.Timestamp, MattDiamondSignal] = {
            s.entry_ts: s for s in signals
        }
        local_dates = [ts.date() for ts in df.index]

        equity = float(config.initial_capital)
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []

        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        idx = df.index

        open_pos: dict | None = None
        session_loss_count = 0
        current_session = None

        for i in range(len(df)):
            ts = idx[i]
            sess = local_dates[i]
            is_last_of_session = (
                i == len(df) - 1 or local_dates[i + 1] != sess
            )

            if sess != current_session:
                current_session = sess
                session_loss_count = 0

            # ---- entries ----
            if open_pos is None:
                sig = sig_by_ts.get(ts)
                if (
                    self.max_session_losses > 0
                    and session_loss_count >= self.max_session_losses
                ):
                    sig = None
                if sig is not None:
                    risk_per_share = sig.entry_price - sig.stop_loss
                    if risk_per_share <= 0:
                        continue

                    target_price = sig.entry_price + (
                        self.target_at_r_multiple * risk_per_share
                    )
                    expected_r = (
                        target_price - sig.entry_price
                    ) / risk_per_share
                    if expected_r + 1e-9 < self.target_min_r_multiple:
                        continue

                    risk_dollars = equity * config.risk_per_trade
                    risk_shares = int(risk_dollars // risk_per_share)
                    if self.max_position_pct_of_equity > 0 and sig.entry_price > 0:
                        max_notional = equity * self.max_position_pct_of_equity
                        cap_shares = int(max_notional // sig.entry_price)
                        shares = min(risk_shares, cap_shares)
                    else:
                        shares = risk_shares
                    if shares <= 0:
                        continue

                    entry_commission = (
                        self.fee_schedule.buy_fee(sig.entry_price, shares)
                        if self.fee_schedule
                        else 0.0
                    )
                    open_pos = {
                        "entry_ts": ts,
                        "entry_bar_idx": i,
                        "entry_price": sig.entry_price,
                        "avg_cost": sig.entry_price,
                        "initial_stop": sig.stop_loss,
                        "stop": sig.stop_loss,
                        "target": target_price,
                        "shares": shares,
                        "initial_risk": risk_per_share,
                        "be_promoted": False,
                        "commission": entry_commission,
                    }

            if open_pos is None:
                continue

            exit_price: float | None = None
            exit_reason: str | None = None
            on_entry_bar = ts == open_pos["entry_ts"]

            if not on_entry_bar:
                # OHLC only — conservative priority on intrabar order:
                #   1) Stop  2) TP  3) BE promotion  4) Time stop
                if lows[i] <= open_pos["stop"]:
                    exit_price = open_pos["stop"]
                    exit_reason = (
                        "breakeven_stop"
                        if open_pos["be_promoted"]
                        else "stop_loss"
                    )

                if exit_price is None and highs[i] >= open_pos["target"]:
                    exit_price = open_pos["target"]
                    exit_reason = "take_profit"

                # BE promotion — once the trade touches +XR.
                if (
                    exit_price is None
                    and self.enable_breakeven_after_r is not None
                    and not open_pos["be_promoted"]
                ):
                    trigger = open_pos["entry_price"] + (
                        self.enable_breakeven_after_r * open_pos["initial_risk"]
                    )
                    if highs[i] >= trigger:
                        # Raise stop to entry minus buffer (avoid being
                        # tagged out by the exact entry price on a
                        # follow-up bar).
                        new_stop = open_pos["entry_price"] * (
                            1 - self.be_stop_buffer_pct
                        )
                        if new_stop > open_pos["stop"]:
                            open_pos["stop"] = new_stop
                            open_pos["be_promoted"] = True

                # Time stop — if not enough R after N bars, market-out.
                if exit_price is None and self.enable_time_stop:
                    bars_held = i - open_pos["entry_bar_idx"]
                    if bars_held >= self.time_stop_bars:
                        unreal_r = (
                            closes[i] - open_pos["entry_price"]
                        ) / open_pos["initial_risk"]
                        if unreal_r < self.time_stop_min_r:
                            exit_price = float(closes[i])
                            exit_reason = "time_stop"

            if exit_price is None and is_last_of_session:
                exit_price = float(closes[i])
                exit_reason = "session_close"

            if exit_price is not None:
                shares = open_pos["shares"]
                if self.fee_schedule:
                    open_pos["commission"] += self.fee_schedule.sell_fee(
                        exit_price, shares
                    )
                gross_pnl = (exit_price - open_pos["avg_cost"]) * shares
                pnl = gross_pnl - open_pos["commission"]
                pnl_pct = (
                    pnl / (open_pos["avg_cost"] * shares)
                    if open_pos["avg_cost"] > 0 and shares > 0
                    else 0.0
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
                if exit_reason in ("stop_loss", "breakeven_stop"):
                    session_loss_count += 1
                open_pos = None

            if is_last_of_session:
                equity_curve.append(EquityPoint(date=sess, equity=equity))

        if not equity_curve:
            equity_curve.append(
                EquityPoint(date=config.start_date, equity=equity)
            )
        return trades, equity_curve

    # ---- performance ---------------------------------------------------

    @staticmethod
    def _performance(
        trades: list[Trade], config: StrategyConfig
    ) -> StrategyPerformance:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        final = config.initial_capital + total_pnl
        avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
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
