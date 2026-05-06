"""Universe-wide First Candle Rule strategy.

Single-portfolio runner that scans an NY-equity universe for FCR
opening-range FVG entries and walks them chronologically by session
date, taking at most one trade per day.

Lifecycle (mirrors Multi Scraface / Multi FVG):

    Phase 1 — **scan** (parallel)
        For each ticker: fetch RTH-only 1m intraday + daily, run the
        :class:`FirstCandleRuleDetector` on the pair (with whatever
        gap / volume / SMA filters the user picked).

    Phase 2a — **collect**
        Group signals by ``session_date``. For each, compute the
        psychology-fit score:
            score = fvg_height_pct × entry_bar_rvol × daily_rvol
        — proxies for the three institutional-conviction signals
        the FCR pattern relies on (displacement size, entry-time
        institutional volume, day-level institutional participation).

    Phase 2b — **walk** (sequential)
        For each session date in order: pick the highest-scoring
        candidate, delegate to :class:`FirstCandleRuleStrategy` for
        the single-trade simulation, apply Toss fees post-hoc.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_t, time, timedelta
from typing import Any

import pandas as pd

from data.domain.market_calendar import MarketCalendar, NY
from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.adapters.first_candle_rule import (
    FirstCandleRuleDetector,
    FirstCandleRuleSignal,
)
from strategy.adapters.first_candle_rule_strategy import (
    FirstCandleRuleStrategy,
)
from strategy.domain.models import (
    EquityPoint,
    MultiStrategyConfig,
    MultiStrategyResult,
    MultiTrade,
    StrategyConfig,
)


class MultiFirstCandleRuleStrategy:
    """Single-portfolio FCR strategy across a ticker universe."""

    _warmup_days: int = 5

    def __init__(
        self,
        market_data: MarketDataPort,
        daily_market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: FirstCandleRuleDetector,
        market: MarketCalendar = NY,
        max_workers: int = 4,
        min_bars: int = 60,
        interval: str = "1m",
        max_position_pct_of_equity: float = 1.0,
        max_below_stop_strikes: int = 3,
        breakeven_after_minutes: int = 0,
        breakeven_after_r_multiple: float = 0.0,
        # Universe-quality pre-filters (None = disabled). Applied at
        # the *ticker* level (vs the detector's day-level filters).
        # Cheap to compute from the per-ticker daily frame and they
        # cut the SP500 picker problem cleanly: low-ATR tickers
        # can't reach 2R intraday; low-dollar-volume tickers don't
        # have the institutional flow the strategy depends on.
        min_daily_atr_pct: float | None = None,
        min_avg_dollar_volume: float | None = None,
        atr_period: int = 14,
        dollar_volume_lookback: int = 20,
    ) -> None:
        self._market_data = market_data
        self._daily_market_data = daily_market_data
        self._universe_provider = universe_provider
        self._detector = detector
        self._market = market
        self._max_workers = max_workers
        self._min_bars = min_bars
        self._interval = interval
        self._strategy = FirstCandleRuleStrategy(
            detector,
            max_position_pct_of_equity=max_position_pct_of_equity,
            max_below_stop_strikes=max_below_stop_strikes,
            breakeven_after_minutes=breakeven_after_minutes,
            breakeven_after_r_multiple=breakeven_after_r_multiple,
        )
        self._min_daily_atr_pct = min_daily_atr_pct
        self._min_avg_dollar_volume = min_avg_dollar_volume
        self._atr_period = atr_period
        self._dollar_volume_lookback = dollar_volume_lookback

    # ---- public API ----

    def run(self, config: MultiStrategyConfig) -> MultiStrategyResult:
        tickers = self._universe_provider.get_tickers(config.universe)
        if config.max_tickers is not None:
            tickers = tickers[: config.max_tickers]

        ticker_state, failed = self._scan_universe(tickers, config)
        signals_by_date, total_signals = self._collect_signals(
            ticker_state, config
        )
        trades, curve, final_capital, max_dd = self._walk_signals(
            signals_by_date, ticker_state, config
        )
        return self._build_result(
            config=config,
            tickers_scanned=len(tickers),
            total_signals=total_signals,
            trades=trades,
            equity_curve=curve,
            final_capital=final_capital,
            max_dd=max_dd,
            failed=failed,
        )

    # ---- phase 1: scan ----

    def _scan_universe(
        self, tickers: list[str], config: MultiStrategyConfig
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        ticker_state: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        # Capture the most recent failure reason — surfaces to the
        # page UI so users can tell "API quota exhausted" from
        # "ticker simply has no signals". Without this, the failed-
        # tickers expander reads as silent fail.
        self._last_fetch_error: str | None = None
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(self._scan_ticker, t, config): t for t in tickers}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    state = fut.result()
                except Exception as exc:
                    self._last_fetch_error = f"{type(exc).__name__}: {exc}"
                    failed.append(t)
                    continue
                if state is None:
                    failed.append(t)
                    continue
                ticker_state[t] = state
        return ticker_state, failed

    def _scan_ticker(
        self, ticker: str, config: MultiStrategyConfig
    ) -> dict[str, Any] | None:
        # Daily: pad ~280 calendar days for SMA200 / 20-day vol avg.
        daily_start = config.start_date - timedelta(days=400)
        intraday_start = config.start_date - timedelta(days=self._warmup_days)
        try:
            df_intraday = self._market_data.fetch_ohlcv(
                ticker,
                intraday_start,
                config.end_date,
                interval=self._interval,
            )
            df_daily = self._daily_market_data.fetch_ohlcv(
                ticker, daily_start, config.end_date
            )
        except Exception as exc:
            # Re-raise so ``_scan_universe`` can capture the message.
            # The catch there records the most recent error so the
            # page can show a real reason.
            raise
        if df_intraday is None or df_intraday.empty or len(df_intraday) < self._min_bars:
            return None
        if df_daily is None or df_daily.empty:
            return None

        # Universe-quality pre-filter on the daily frame before any
        # signal scanning. Reject tickers whose volatility (ATR%)
        # can't reach 2R intraday OR whose liquidity (avg $-volume)
        # is too thin to attract institutional flow. Both filters
        # are computed over a trailing window so they reflect the
        # ticker's recent character, not just one outlier day.
        if (
            self._min_daily_atr_pct is not None
            or self._min_avg_dollar_volume is not None
        ):
            recent = df_daily.tail(
                max(self._atr_period, self._dollar_volume_lookback) + 5
            )
            if self._min_daily_atr_pct is not None:
                tr = (recent["High"] - recent["Low"])
                # Use simple H-L range / Close as a cheap ATR proxy
                # (full Wilder ATR adds complexity without changing
                # the decision threshold).
                atr_pct = (
                    tr.tail(self._atr_period).mean()
                    / float(recent["Close"].iloc[-1])
                )
                if atr_pct < self._min_daily_atr_pct:
                    return None
            if self._min_avg_dollar_volume is not None:
                dv = (recent["Close"] * recent["Volume"]).tail(
                    self._dollar_volume_lookback
                ).mean()
                if dv < self._min_avg_dollar_volume:
                    return None

        # tz-convert intraday to market-local once.
        if (
            df_intraday.index.tz is not None
            and str(df_intraday.index.tz) != self._market.tz
        ):
            df_intraday = df_intraday.copy()
            df_intraday.index = df_intraday.index.tz_convert(self._market.tz)

        signals = self._detector.detect(df_intraday, df_daily)
        # Pre-compute D-RVOL by date for tie-break scoring.
        avg_vol = df_daily["Volume"].rolling(20).mean()
        drvol = df_daily["Volume"] / avg_vol
        drvol_by_date = {
            ts.date(): float(v)
            for ts, v in drvol.items()
            if pd.notna(v)
        }
        # Daily ATR ($) for ATR-normalized FVG sizing — kills the
        # small-cap bias the original ``fvg_pct`` had (since fvg_pct
        # divides by entry_price, low-priced tickers automatically
        # got higher scores even with smaller absolute moves).
        daily_atr = (
            df_daily["High"] - df_daily["Low"]
        ).rolling(self._atr_period).mean()
        atr_by_date = {
            ts.date(): float(v)
            for ts, v in daily_atr.items()
            if pd.notna(v) and v > 0
        }
        # Average daily $-volume — a separate quality term that
        # penalizes low-liquidity picks even after the ATR-norm
        # makes FVG size comparable across price levels.
        avg_dv = (df_daily["Close"] * df_daily["Volume"]).rolling(
            self._dollar_volume_lookback
        ).mean()
        avg_dv_by_date = {
            ts.date(): float(v)
            for ts, v in avg_dv.items()
            if pd.notna(v) and v > 0
        }
        return {
            "df_intraday": df_intraday,
            "df_daily": df_daily,
            "signals": signals,
            "drvol_by_date": drvol_by_date,
            "atr_by_date": atr_by_date,
            "avg_dv_by_date": avg_dv_by_date,
        }

    # ---- phase 2a: collect ----

    def _collect_signals(
        self,
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[dict[date_t, list[dict[str, Any]]], int]:
        """Group signals by session date with the fit-score tie-break."""
        by_date: dict[date_t, list[dict[str, Any]]] = defaultdict(list)
        import math
        total = 0
        for ticker, state in ticker_state.items():
            df_intraday: pd.DataFrame = state["df_intraday"]
            drvol_by_date: dict[date_t, float] = state["drvol_by_date"]
            atr_by_date: dict[date_t, float] = state["atr_by_date"]
            avg_dv_by_date: dict[date_t, float] = state["avg_dv_by_date"]
            for sig in state["signals"]:
                if (
                    sig.session_date < config.start_date
                    or sig.session_date > config.end_date
                ):
                    continue
                # **ATR-normalized FVG size** — kills the small-cap
                # bias the prior ``fvg_pct`` had. A $1 FVG on a stock
                # with $50 daily ATR is small (0.02); the same $1 FVG
                # on a stock with $5 ATR is huge (0.20). This compares
                # FVG magnitude in volatility units, which is what
                # institutional flow looks like across price scales.
                atr_dollars = atr_by_date.get(sig.session_date)
                if atr_dollars is None or atr_dollars <= 0:
                    continue  # no ATR → can't normalize → skip score
                fvg_height = sig.fvg_high - sig.fvg_low
                fvg_atr = fvg_height / atr_dollars
                # Entry-bar RVOL — unchanged, already a ratio.
                entry_rvol = self._entry_bar_rvol(
                    df_intraday, sig.entry_ts, sig.session_date
                )
                # Daily RVOL — already a ratio.
                daily_rvol = drvol_by_date.get(sig.session_date, 1.0)
                # **Liquidity term** — log of average daily $-volume.
                # Penalizes low-liquidity tickers without dominating
                # the score. log10($1M) = 6, log10($1B) = 9 → mega-cap
                # gets ~50% score boost over mid-cap, not 1000x.
                avg_dv = avg_dv_by_date.get(sig.session_date, 1.0)
                liq_term = (
                    math.log10(avg_dv) if avg_dv > 0 else 0.0
                )
                score = fvg_atr * entry_rvol * daily_rvol * liq_term
                by_date[sig.session_date].append({
                    "ticker": ticker,
                    "signal": sig,
                    "fvg_pct": fvg_atr,  # repurpose to mean ATR-normalized
                    "entry_rvol": entry_rvol,
                    "daily_rvol": daily_rvol,
                    "score": score,
                })
                total += 1
        return by_date, total

    @staticmethod
    def _entry_bar_rvol(
        df_intraday: pd.DataFrame,
        entry_ts: pd.Timestamp,
        session_date: date_t,
    ) -> float:
        """RVOL of the entry bar vs the first 30 bars of the session.

        Returns 1.0 if anything's missing (neutral score contribution).
        """
        if "Volume" not in df_intraday.columns:
            return 1.0
        try:
            entry_vol = float(df_intraday.loc[entry_ts, "Volume"])
        except KeyError:
            return 1.0
        sess = df_intraday[df_intraday.index.date == session_date]
        if sess.empty:
            return 1.0
        early = sess.head(30)
        avg = float(early["Volume"].mean())
        if not (avg > 0):
            return 1.0
        return entry_vol / avg

    # ---- phase 2b: walk ----

    def _walk_signals(
        self,
        signals_by_date: dict[date_t, list[dict[str, Any]]],
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[list[MultiTrade], list[EquityPoint], float, float]:
        capital = config.initial_capital
        peak = capital
        max_dd = 0.0
        trades: list[MultiTrade] = []
        curve: list[EquityPoint] = [
            EquityPoint(date=config.start_date, equity=capital)
        ]
        for sess_date in sorted(signals_by_date.keys()):
            best = max(
                signals_by_date[sess_date], key=lambda c: c["score"]
            )
            ticker = best["ticker"]
            sig: FirstCandleRuleSignal = best["signal"]
            multi_trade = self._execute_one(
                ticker=ticker,
                state=ticker_state[ticker],
                signal=sig,
                fit=best,
                capital=capital,
                config=config,
            )
            if multi_trade is None:
                continue
            capital += multi_trade.pnl
            peak = max(peak, capital)
            if peak > 0:
                max_dd = max(max_dd, (peak - capital) / peak)
            trades.append(multi_trade)
            curve.append(
                EquityPoint(date=multi_trade.exit_date, equity=round(capital, 2))
            )
        return trades, curve, capital, max_dd

    def _execute_one(
        self,
        ticker: str,
        state: dict[str, Any],
        signal: FirstCandleRuleSignal,
        fit: dict[str, Any],
        capital: float,
        config: MultiStrategyConfig,
    ) -> MultiTrade | None:
        per_config = StrategyConfig(
            ticker=ticker,
            start_date=config.start_date,
            end_date=config.end_date,
            pattern_name=config.pattern_name,
            initial_capital=capital,
            risk_per_trade=config.risk_per_trade,
            max_holding_days=config.max_holding_days,
        )
        sim_trades, _curve = self._strategy._simulate(
            df=state["df_intraday"],
            signals=[signal],
            config=per_config,
        )
        if not sim_trades:
            return None
        trade = sim_trades[0]
        commission = config.fee_schedule.round_trip(
            trade.entry_price, trade.exit_price, trade.shares
        )
        gross_pnl = trade.pnl
        net_pnl = round(gross_pnl - commission, 2)
        cost_basis = trade.entry_price * trade.shares
        net_pnl_pct = (
            round(net_pnl / cost_basis, 4) if cost_basis > 0 else 0.0
        )
        # Repurpose MultiTrade.signal_* fields to carry the FCR fit-
        # score components — visible in the trade table.
        return MultiTrade(
            ticker=ticker,
            signal_volume=float(fit["entry_rvol"]),
            signal_buy_volume=float(fit["fvg_pct"]),
            signal_sell_volume=float(fit["daily_rvol"]),
            signal_buy_sell_ratio=round(fit["score"], 6),
            commission=round(commission, 2),
            gross_pnl=round(gross_pnl, 2),
            pattern_name=trade.pattern_name,
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_loss=trade.stop_loss,
            shares=trade.shares,
            pnl=net_pnl,
            pnl_pct=net_pnl_pct,
            exit_reason=trade.exit_reason,
            entry_ts=trade.entry_ts,
            exit_ts=trade.exit_ts,
        )

    # ---- result ----

    @staticmethod
    def _build_result(
        config: MultiStrategyConfig,
        tickers_scanned: int,
        total_signals: int,
        trades: list[MultiTrade],
        equity_curve: list[EquityPoint],
        final_capital: float,
        max_dd: float,
        failed: list[str],
    ) -> MultiStrategyResult:
        wins = [t for t in trades if t.pnl > 0]
        win_rate = round(len(wins) / len(trades), 4) if trades else 0.0
        total_return = (
            round(
                (final_capital - config.initial_capital)
                / config.initial_capital,
                4,
            )
            if config.initial_capital > 0
            else 0.0
        )
        total_commission = round(sum(t.commission for t in trades), 2)
        return MultiStrategyResult(
            config=config,
            tickers_scanned=tickers_scanned,
            total_signals=total_signals,
            trades_taken=len(trades),
            win_rate=win_rate,
            total_return_pct=total_return,
            initial_capital=config.initial_capital,
            final_capital=round(final_capital, 2),
            max_drawdown_pct=round(max_dd, 4),
            total_commission=total_commission,
            trades=trades,
            equity_curve=equity_curve,
            failed_tickers=failed,
        )
