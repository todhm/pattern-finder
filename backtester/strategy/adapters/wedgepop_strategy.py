from datetime import date

import numpy as np
import pandas as pd

from data.domain.ports import MarketDataPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from pattern.helpers.pivots import (
    find_swing_highs,
    find_swing_lows,
    fit_lower_trendline,
    last_n_swing_highs,
    last_swing_high,
    recent_swing_high,
)
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    Trade,
)
from strategy.domain.ports import StrategyRunnerPort


class WedgepopStrategy(StrategyRunnerPort):
    """TraderLion 'Wedge Pop' (Oliver Kell - The Money Pattern) trading strategy.

    Self-contained strategy adapter: depends only on the strategy domain
    plus inbound ports (`MarketDataPort`, `PatternDetector`). Holds no
    reference to the backtest domain — execution, sizing, exits, and
    performance accounting all live inside the strategy boundary.

    All distance-based thresholds (exhaustion, entry extension, climax)
    are expressed in **ATR multiples** rather than fixed percentages.
    ATR automatically scales with each stock's volatility, so "2.5 ATR
    above EMA" represents a comparable visual distance on a quiet $500
    blue-chip and a jumpy $20 growth name — eliminating the "1% is huge
    on stock A but invisible on stock B" problem that plagued the
    old fixed-% gates.

    Lifecycle (one position at a time)

    Entry
        - The injected `PatternDetector` (expected to be a wedge-pop
          detector) signals the first close above both EMAs after a
          consolidation. Buy at the NEXT bar's OPEN — gives a clean fill
          and avoids paying the breakout candle's close.

    Initial stop ("logical pivot stop")
        - Just below the consolidation low (provided as `signal.stop_loss`).
          This is the level the doc recommends for failed-breakout protection.

    Position size
        - Fixed-fractional risk: ``shares = capital * risk_per_trade /
          (entry - stop)``. Caps downside at `risk_per_trade` per trade.

    Exit — only four opt-in conditions, each independent:

        1. ``exit_detector`` fires   — Exhaustion Extension Top exit
                                       (inject ``ExhaustionExtensionTopDetector``).
        2. ``enable_trendline_exit`` — bar's LOW pierces the higher-low
                                       trendline fitted through recent swing
                                       lows. LOW-based limit-order fill: the
                                       sell fills at the line (or at the open
                                       on a gap-down).
        3. ``use_smart_trail``       — Chandelier exit:
                                       ``highest_high - N × ATR``, N widens with
                                       R-profit (<2R→3, 2-4R→4, >4R→5). Arms
                                       after 3 bars. LOW-based (wick-proof).
        4. ``enable_resistance_break_exit`` — false-breakout rule. Entry-time
                                       swing resistance is stored; once close
                                       confirms above it, a subsequent low
                                       piercing back below fires an exit at
                                       the resistance line (gap-down → open).

        If none is active (or none fires), the position is held to the
        last bar of the dataframe. There is no hard stop, EMA trail,
        exhaustion-line, climax-bar, or time stop.

    Optional extras (toggle-only, no tuning):

        - ``enable_market_regime_filter`` (Entry): reject entries when
          the seeded ``market_regime_df`` (typically SPY) closes at or
          below its 200 SMA on the signal bar's date. Wedge pops in
          weak markets fail disproportionately.
        - ``enable_breakeven_stop`` (Exit): once any close during the
          trade reaches ``entry + 1R``, a subsequent low returning to
          the entry price exits at entry (zero P&L before fees). Kills
          the -3..-10% loss tail.
        - ``enable_gap_down_rejection`` (Entry): reject entries whose
          entry-bar open gaps below the signal bar's low — a
          structural break of the pop setup before we can buy.

    **Pattern-based exit** (opt-in via ``exit_detector``)
        When any `PatternDetector` is injected, any bar inside the
        holding period that fires a signal from that detector triggers
        an immediate exit at that bar's close. Typical use: inject an
        `ExhaustionExtensionTopDetector` so that a long exits once the
        market prints a euphoric blow-off top. All conditions those
        detectors check (extensions, slopes, candle shapes, rolling
        volume ratios) are end-of-bar quantities, so exiting at the
        signal bar's close uses no future data — every input to the
        decision is known by the time the bar finishes. Signals are
        pre-computed once on the full dataframe in ``execute()``;
        pandas' EWM/rolling indicators are causal, so each signal's
        date maps to a bar whose detection used only data up to and
        including that bar. The entry bar itself is skipped
        regardless of whether it fires — a same-bar exit against our
        own entry is nonsensical.

    The strategy's exit logic is tied to the wedge-pop lifecycle (Wedge
    Pop -> EMA Crossback -> Base n' Break -> Exhaustion Extension), so the
    injected detector must be a wedge-pop detector for the rules to make
    sense. The dependency is typed against the port (`PatternDetector`)
    to keep the adapter free of concrete-adapter coupling; composition is
    the responsibility of the caller.
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        detector: PatternDetector,
        ema_trail: int = 10,
        ema_slow: int = 20,
        atr_period: int = 14,
        extension_atr_mult: float = 2.5,
        climax_atr_mult: float = 1.5,
        max_entry_ema_extension_atr: float | None = None,
        max_ema_slope_decline: float | None = 0.01,
        min_ema_slow_slope: float | None = None,
        max_ema_slow_slope: float | None = None,
        require_gap_up: bool = False,
        use_smart_trail: bool = False,
        exit_detector: PatternDetector | None = None,
        enable_swing_resistance_filter: bool = False,
        swing_pivot_left: int = 2,
        swing_pivot_right: int = 2,
        swing_pivot_lookback: int = 60,
        swing_resistance_tolerance_atr: float = 0.5,
        enable_trendline_exit: bool = False,
        trendline_max_pivots: int = 3,
        trendline_min_pivots: int = 2,
        enable_resistance_break_exit: bool = False,
        resistance_break_pierce_buffer_atr: float = 0.5,
        resistance_break_confirm_buffer_atr: float = 0.1,
        enable_market_regime_filter: bool = False,
        market_regime_df: pd.DataFrame | None = None,
        enable_breakeven_stop: bool = False,
        enable_gap_down_rejection: bool = False,
        enable_signal_close_strength_filter: bool = False,
        min_signal_close_location: float = 0.5,
        enable_swing_breakout_filter: bool = False,
        swing_breakout_buffer_atr: float = 0.0,
        max_signal_bar_gain_atr: float | None = None,
        breakeven_arm_r_multiple: float = 1.0,
        structural_exit_grace_bars: int = 0,
        breakeven_exit_offset_r: float = 0.0,
        structural_exit_close_confirm: bool = False,
    ):
        self._market_data = market_data
        self._detector = detector
        self._exit_detector = exit_detector
        self.ema_trail = ema_trail
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.extension_atr_mult = extension_atr_mult
        self.climax_atr_mult = climax_atr_mult
        self.max_entry_ema_extension_atr = max_entry_ema_extension_atr
        # Slope entry filter as a RANGE ``[min, max]``. The legacy
        # ``max_ema_slope_decline`` param still works as a
        # backwards-compatible shim: setting it to e.g. 0.01 is
        # equivalent to ``min_ema_slow_slope=-0.01``. The new
        # ``min_ema_slow_slope`` / ``max_ema_slow_slope`` take
        # precedence when both are set — the old shim only fills
        # in the min bound if it's ``None``. This lets users
        # require a POSITIVE slope (e.g. ``min=0.05`` → "ema20
        # must be rising by at least +5% over the slope_lookback
        # window"), not just reject steep declines.
        self.max_ema_slope_decline = max_ema_slope_decline
        if min_ema_slow_slope is None and max_ema_slope_decline is not None:
            min_ema_slow_slope = -max_ema_slope_decline
        self.min_ema_slow_slope = min_ema_slow_slope
        self.max_ema_slow_slope = max_ema_slow_slope
        self.require_gap_up = require_gap_up
        self.use_smart_trail = use_smart_trail
        self.enable_swing_resistance_filter = enable_swing_resistance_filter
        self.swing_pivot_left = swing_pivot_left
        self.swing_pivot_right = swing_pivot_right
        self.swing_pivot_lookback = swing_pivot_lookback
        self.swing_resistance_tolerance_atr = swing_resistance_tolerance_atr
        self.enable_trendline_exit = enable_trendline_exit
        self.trendline_max_pivots = trendline_max_pivots
        self.trendline_min_pivots = trendline_min_pivots
        self.enable_resistance_break_exit = enable_resistance_break_exit
        self.resistance_break_pierce_buffer_atr = resistance_break_pierce_buffer_atr
        self.resistance_break_confirm_buffer_atr = resistance_break_confirm_buffer_atr
        self.enable_market_regime_filter = enable_market_regime_filter
        # Pre-compute SPY 200-SMA and store a date → close-above-sma
        # lookup so entry checks are O(1). The comparison is strict
        # ``close > sma200`` (both at signal bar's date). NaN SMAs
        # (< 200 bars of history) resolve to False → rejection.
        self._market_regime_lookup: dict[date, bool] = {}
        if enable_market_regime_filter and market_regime_df is not None:
            m = market_regime_df.copy()
            if m.index.tz is not None:
                m.index = m.index.tz_localize(None)
            sma200 = m["Close"].rolling(window=200, min_periods=200).mean()
            ok = (m["Close"] > sma200).fillna(False)
            self._market_regime_lookup = {idx.date(): bool(val) for idx, val in ok.items()}
        self.enable_breakeven_stop = enable_breakeven_stop
        self.enable_gap_down_rejection = enable_gap_down_rejection
        # Signal-bar close-strength filter. Rejects entries whose
        # signal bar closed in the lower part of its intraday range
        # (a pump-and-fade candle), because wedge pops that close
        # weak within the day tend to reverse on the next bar.
        # ``close_location = (close - low) / (high - low)`` — 1.0 =
        # closed exactly at the high, 0.0 = at the low. Degenerate
        # bars (high == low) bypass the filter.
        self.enable_signal_close_strength_filter = enable_signal_close_strength_filter
        self.min_signal_close_location = min_signal_close_location
        # Structural-breakout filter. Requires the signal bar's HIGH to
        # clear the most recent confirmable swing high by at least
        # ``swing_breakout_buffer_atr × ATR``. Addresses the failure mode
        # where the wedge pop fires while price is still under near-term
        # overhead supply — those setups get stopped by resistance-break
        # / trendline exits on the next bar or two. Complementary to the
        # existing ``enable_swing_resistance_filter`` which checks the
        # entry open vs resistance; this one gates at the signal bar's
        # high instead, so a wide-range breakout candle that pokes above
        # the pivot intraday passes even if the open is still below.
        self.enable_swing_breakout_filter = enable_swing_breakout_filter
        self.swing_breakout_buffer_atr = swing_breakout_buffer_atr
        # Euphoria cap — reject entries whose signal bar's single-day
        # gain ``(close - open) / ATR`` exceeds this multiple. Targets
        # the adverse-selection failure of multi-ticker scans: the
        # ranking metric (buy/sell ratio, close-near-high) prefers
        # bars that closed strong intraday, which at universe scale
        # surfaces euphoric gap-and-go candles. Those tend to
        # mean-revert the next bar. Distinct from the detector's
        # ``max_breakout_atr_mult`` — that one caps the
        # ``max(ema_distance, daily_move)`` magnitude (gap from prior
        # close or distance above EMA), this one caps the intrabar
        # open-to-close body in ATR units. ``None`` = filter off.
        self.max_signal_bar_gain_atr = max_signal_bar_gain_atr
        # Breakeven-stop arming threshold in R-multiples. Default 1.0
        # preserves the original behavior ("any close at ≥ +1R flips
        # the armed flag"). Raising to 1.5 or 2.0 delays arming so a
        # trade has to prove itself further before the exit tightens
        # to entry price. Observed failure mode from full-universe
        # scans: trades that barely tag +1R, then revisit entry on
        # intraday noise, fire a breakeven stop and lock in a small
        # net loss from fees. A higher arm threshold turns those
        # into trades that either keep running (benefit from trend)
        # or exit later via trendline / exhaustion / trail.
        self.breakeven_arm_r_multiple = breakeven_arm_r_multiple
        # Grace-period bar count for the structural exits (trendline
        # break, resistance break). For the first
        # ``structural_exit_grace_bars`` bars *after* the entry bar,
        # those two LOW-based stops are suppressed, letting initial
        # chop play out before fast-exit logic fires. Does NOT affect
        # breakeven, exhaustion, or smart-trail exits — those still
        # follow their own cadence. Default 0 preserves the original
        # no-grace behavior.
        self.structural_exit_grace_bars = structural_exit_grace_bars
        # Breakeven exit offset in R-multiples. Default 0.0 preserves
        # the original "exit at exact entry price" behavior. Setting
        # to 0.3 means once the breakeven is armed, the exit triggers
        # when ``low ≤ entry + 0.3R`` and fills at that level — locks
        # in a small profit instead of zero. Observed failure mode at
        # scale: trades that hit +1R, then pull back to entry, exit
        # at 0 gross pnl and print ~-0.5% net after commissions,
        # producing a loss cluster at -0.6% to -1%. A small positive
        # offset converts those breakevens into +0.1~+0.3% net wins
        # (on a 5%-risk sizing: +0.3R × 5% = +1.5% gross per trade).
        # Cost: clips winners that pulled back through the offset but
        # would have run further. Winners in the sample are rare and
        # usually clean runs, so the tradeoff is expected to favor
        # the offset.
        self.breakeven_exit_offset_r = breakeven_exit_offset_r
        # When True, trendline-break + resistance-break exits fire on
        # CLOSE below level instead of LOW touching. Observed failure
        # mode: 1-day gap-down / wick crashes hit the LOW trigger and
        # exit at the worst intraday price; many bars close back
        # ABOVE the level the same day. CLOSE-based filters those
        # wick stops and only fires on bars that actually closed
        # broken. Default False preserves the original LOW-based
        # limit-order fill model.
        self.structural_exit_close_confirm = structural_exit_close_confirm

    # ---- public API ----

    def run(self, config: StrategyConfig) -> StrategyResult:
        """Fetch data via the market-data port and execute the strategy."""
        df = self._market_data.fetch_ohlcv(config.ticker, config.start_date, config.end_date)
        return self.execute(df, config)

    def execute(self, df: pd.DataFrame, config: StrategyConfig) -> StrategyResult:
        """Run the strategy on a pre-fetched OHLCV DataFrame.

        Useful when the caller already holds the market data (e.g. a UI
        page rendering both the chart and the strategy from a single
        fetch). The DataFrame must have the same shape as
        `MarketDataPort.fetch_ohlcv` returns.

        The DataFrame may contain bars before ``config.start_date``
        (warmup for indicator convergence — e.g. 200 extra bars so
        the 200 SMA is valid from day 1). Signals that fire during
        the warmup period are silently discarded.
        """
        df = self._with_indicators(df)
        signals = self._detector.detect(df)
        # Discard warmup-period signals outside the user's date range.
        signals = [s for s in signals if config.start_date <= s.date <= config.end_date]

        # Pre-compute pattern-based exit dates once. Safe against
        # look-ahead: the detector's conditions (extensions, slopes,
        # candle shapes, rolling volume) are all causal — EWM/rolling
        # indicators only read past+current values — so each signal's
        # date corresponds to a bar that could have been detected at
        # its own close in real time.
        exit_dates: set[date] = set()
        if self._exit_detector is not None:
            exit_dates = {s.date for s in self._exit_detector.detect(df)}

        performance, equity_curve = self._run_signals(df, signals, config, exit_dates)

        return StrategyResult(
            config=config,
            performance=performance,
            equity_curve=equity_curve,
        )

    # ---- indicators ----

    def _with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_trail"] = df["Close"].ewm(span=self.ema_trail, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=self.ema_slow, adjust=False).mean()
        df["atr"] = self._atr(df, self.atr_period)
        if (
            self.enable_swing_resistance_filter
            or self.enable_trendline_exit
            or self.enable_resistance_break_exit
            or self.enable_swing_breakout_filter
        ):
            df["swing_high"] = find_swing_highs(
                df, left=self.swing_pivot_left, right=self.swing_pivot_right
            )
            df["swing_low"] = find_swing_lows(df, left=self.swing_pivot_left, right=self.swing_pivot_right)
        return df

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        prev_close = df["Close"].shift(1)
        true_range = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()

    # ---- execution loop ----

    def _run_signals(
        self,
        df: pd.DataFrame,
        signals: list[PatternSignal],
        config: StrategyConfig,
        exit_dates: set[date],
    ) -> tuple[StrategyPerformance, list[EquityPoint]]:
        capital = config.initial_capital
        peak = capital
        max_dd = 0.0
        trades: list[Trade] = []
        curve: list[EquityPoint] = [EquityPoint(date=config.start_date, equity=capital)]

        next_open_idx = 0  # blocks new entries while a position is open
        for signal in signals:
            entry_idx = self._next_open_index(df, signal.date)
            if entry_idx is None or entry_idx < next_open_idx:
                continue

            trade, exit_idx = self._execute_trade(df, signal, entry_idx, capital, config, exit_dates)
            if trade is None:
                continue

            capital += trade.pnl
            peak = max(peak, capital)
            max_dd = max(max_dd, (peak - capital) / peak if peak > 0 else 0.0)
            trades.append(trade)
            curve.append(EquityPoint(date=trade.exit_date, equity=round(capital, 2)))
            next_open_idx = exit_idx + 1

        performance = self._build_performance(config.initial_capital, capital, max_dd, trades)
        return performance, curve

    @staticmethod
    def _next_open_index(df: pd.DataFrame, signal_date) -> int | None:
        # `signal_date` is a plain `date`. yfinance returns a tz-aware
        # DatetimeIndex (America/New_York), so a naive Timestamp would
        # raise on comparison. Localize the Timestamp to the index tz
        # whenever the index has one.
        ts = pd.Timestamp(signal_date)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        after = df.index[df.index > ts]
        if len(after) == 0:
            return None
        return df.index.get_loc(after[0])

    def _execute_trade(
        self,
        df: pd.DataFrame,
        signal: PatternSignal,
        entry_idx: int,
        capital: float,
        config: StrategyConfig,
        exit_dates: set[date] | None = None,
    ) -> tuple[Trade | None, int]:
        if exit_dates is None:
            exit_dates = set()
        entry_price = float(df["Open"].iloc[entry_idx])

        # (A) Market-regime filter — reject entries taken while SPY
        # (or whatever index the caller seeded) sits at or below its
        # 200 SMA. Wedge pops in bear/neutral markets fail at far
        # higher rates than in confirmed uptrends. Lookup is O(1).
        if self.enable_market_regime_filter:
            signal_date = df.index[entry_idx - 1].date() if entry_idx > 0 else None
            if signal_date is not None and not self._market_regime_lookup.get(signal_date, False):
                return None, entry_idx

        # (C) Gap-down open rejection — reject entries whose entry-bar
        # open gaps below the signal bar's low. A bar that opens
        # below yesterday's low already broke the pop structure
        # overnight; holding is structurally dead-setup chasing.
        if self.enable_gap_down_rejection and entry_idx > 0:
            signal_low = float(df["Low"].iloc[entry_idx - 1])
            if entry_price < signal_low:
                return None, entry_idx

        # (D) Signal-bar close-strength filter — reject when the
        # signal bar's close sits in the lower part of its intraday
        # range (pump-and-fade candle). Wedge pops whose breakout
        # candle fails to close near the high tend to reverse on the
        # next bar; requiring close in the upper portion of the day's
        # range filters out weak follow-through setups driving the
        # bulk of 1-2 day losers in multi-ticker scans.
        if self.enable_signal_close_strength_filter and entry_idx > 0:
            sig_idx = entry_idx - 1
            sig_high = float(df["High"].iloc[sig_idx])
            sig_low = float(df["Low"].iloc[sig_idx])
            sig_close = float(df["Close"].iloc[sig_idx])
            bar_range = sig_high - sig_low
            if bar_range > 0:
                close_location = (sig_close - sig_low) / bar_range
                if close_location < self.min_signal_close_location:
                    return None, entry_idx

        # (E) Swing-breakout filter — require the signal bar's HIGH
        # to clear the most recent confirmable swing high by at
        # least ``swing_breakout_buffer_atr × ATR``. Setups that
        # fire while price still sits under near-term overhead
        # supply run into resistance almost immediately (the 7/7
        # trendline-break losses in the scan report all came from
        # entries where the wedge pop never cleared structure).
        if self.enable_swing_breakout_filter and entry_idx > 0 and "swing_high" in df.columns:
            sig_idx = entry_idx - 1
            last = last_swing_high(
                df["swing_high"],
                upto_idx=sig_idx,
                lookback=self.swing_pivot_lookback,
                right=self.swing_pivot_right,
            )
            if last is not None:
                _, pivot_price = last
                sig_high = float(df["High"].iloc[sig_idx])
                atr_at_signal = float(df["atr"].iloc[sig_idx])
                buffer = self.swing_breakout_buffer_atr * atr_at_signal if atr_at_signal > 0 else 0.0
                if sig_high <= pivot_price + buffer:
                    return None, entry_idx

        # (F) Euphoria cap — reject entries whose signal bar printed
        # an excessively large intrabar body in ATR units. The multi-
        # ticker ranker prefers bars that closed near the high, which
        # at scale surfaces euphoric spikes that mean-revert. Capping
        # ``(close - open) / ATR`` at the signal bar filters those
        # out without tightening the volume or structural gates.
        if self.max_signal_bar_gain_atr is not None and entry_idx > 0:
            sig_idx = entry_idx - 1
            sig_open = float(df["Open"].iloc[sig_idx])
            sig_close = float(df["Close"].iloc[sig_idx])
            atr_at_signal = float(df["atr"].iloc[sig_idx])
            if atr_at_signal > 0:
                gain_atr = (sig_close - sig_open) / atr_at_signal
                if gain_atr > self.max_signal_bar_gain_atr:
                    return None, entry_idx

        # Reject entry when the open gaps BELOW both EMAs — if we're
        # buying a long and the open is already under the 10 EMA and
        # 20 EMA (computed at the signal bar, the last known values
        # at market open), the breakout has failed overnight and
        # entering is chasing a dead setup.
        if entry_idx > 0:
            sig_ema_fast = float(df["ema_trail"].iloc[entry_idx - 1])
            sig_ema_slow = float(df["ema_slow"].iloc[entry_idx - 1])
            if entry_price < sig_ema_fast and entry_price < sig_ema_slow:
                return None, entry_idx

        # Optional gap-up confirmation (TraderLion: "Most Wedge Pops that
        # start new trends often include unfilled gaps with strong volume").
        # If enabled, the next bar's open must clear the breakout bar's
        # close — otherwise the breakout failed to follow through and we
        # skip this signal entirely.
        if self.require_gap_up and entry_idx > 0:
            prev_close = float(df["Close"].iloc[entry_idx - 1])
            if entry_price <= prev_close:
                return None, entry_idx

        # "EMA-extension" entry filter (ATR-scaled): reject entries
        # where the entry-bar open is more than
        # ``max_entry_ema_extension_atr × ATR`` above
        # ``max(ema_fast, ema_slow)`` at the signal bar. Using ATR
        # as the yardstick means the same multiplier (e.g. 1.5)
        # works across high- and low-volatility regimes — no need
        # to retune a fixed % per stock. Setting to ``None``
        # disables the filter entirely.
        if self.max_entry_ema_extension_atr is not None and entry_idx > 0:
            sig_fast = float(df["ema_trail"].iloc[entry_idx - 1])
            sig_slow = float(df["ema_slow"].iloc[entry_idx - 1])
            ref_ema = max(sig_fast, sig_slow)
            atr_at_signal = float(df["atr"].iloc[entry_idx - 1])
            if ref_ema > 0 and atr_at_signal > 0:
                extension_atr = (entry_price - ref_ema) / atr_at_signal
                if extension_atr > self.max_entry_ema_extension_atr:
                    return None, entry_idx

        # Slope range filter: reject entries where the signal's
        # medium-term EMA slope (metadata variable set by the
        # detector) sits outside
        # ``[min_ema_slow_slope, max_ema_slow_slope]``. The MIN
        # bound catches dead-cat-bounce setups inside rolling-over
        # trends (e.g. AMD 2025-01-06 with ema20_slope20 = -10.5%)
        # OR can be used to require a confirmed uptrend (e.g.
        # ``min_ema_slow_slope = 0.05`` → "only take wedge pops
        # where the 20 EMA has risen at least 5% over the last N
        # bars"). The MAX bound filters out pops that are already
        # inside a parabolic move (e.g. cap at 0.30 to skip names
        # that are up 30%+ on the slope window).
        slow_slope = signal.metadata.get("ema_slow_slope")
        if slow_slope is not None:
            if self.min_ema_slow_slope is not None and slow_slope < self.min_ema_slow_slope:
                return None, entry_idx
            if self.max_ema_slow_slope is not None and slow_slope > self.max_ema_slow_slope:
                return None, entry_idx

        # Swing-high resistance filter: reject if the entry-bar open
        # sits just beneath a recent confirmed swing high. The rule
        # is BYPASSED when the last two swing highs form a higher
        # high (uptrend structure) — in that regime, the pattern of
        # pushing through overhead supply is the norm and blocking
        # near-resistance entries kills too many winners. Only when
        # the last two swing highs are equal or descending (flat or
        # lower-high structure) is the filter applied.
        if self.enable_swing_resistance_filter and entry_idx > 0 and "swing_high" in df.columns:
            signal_idx = entry_idx - 1
            recent = last_n_swing_highs(
                df["swing_high"],
                upto_idx=signal_idx,
                lookback=self.swing_pivot_lookback,
                right=self.swing_pivot_right,
                n=2,
            )
            making_higher_highs = len(recent) >= 2 and recent[-1][1] > recent[-2][1]
            if not making_higher_highs:
                res = recent_swing_high(
                    df["swing_high"],
                    upto_idx=signal_idx,
                    lookback=self.swing_pivot_lookback,
                    right=self.swing_pivot_right,
                )
                if res is not None:
                    _, pivot_price = res
                    atr_at_signal = float(df["atr"].iloc[signal_idx])
                    if atr_at_signal > 0:
                        gap = pivot_price - entry_price
                        if gap > 0 and gap < self.swing_resistance_tolerance_atr * atr_at_signal:
                            return None, entry_idx

        stop = float(signal.stop_loss)
        risk_per_share = entry_price - stop
        if risk_per_share <= 0:
            return None, entry_idx

        risk_amount = capital * config.risk_per_trade
        shares = max(1, int(risk_amount / risk_per_share))

        # No-leverage cap: position notional must not exceed available
        # capital. Tight stops (e.g. consolidation-low just under
        # entry) otherwise inflate ``shares`` so that the position is
        # 3-5× capital, and a 1R adverse move in the absence of a
        # hard stop realizes 3-5× the intended loss — see ICE
        # 2018-02-02 in the 10y export: stop at ~1.3% below entry
        # sized the position at ~$550K on ~$130K capital; the
        # resistance-break exit fired ~3.4% below entry and the dollar
        # loss was ~$20K (≈15% of capital) instead of the 5% risk the
        # sizing formula intended. Capping shares at
        # ``capital / entry_price`` forces ``shares × entry ≤ capital``
        # so only cash you actually have gets deployed.
        max_shares_by_capital = int(capital / entry_price)
        if max_shares_by_capital < 1:
            return None, entry_idx
        shares = min(shares, max_shares_by_capital)

        # Collect every swing high in the lookback window — each one
        # is an independent stop level for the resistance-break exit.
        # Levels already below entry are "confirmed supports" (price
        # has proven it can sit above them); levels above entry stay
        # pending until close confirms a break-through. This matches
        # the trader mental model where *each* drawn pivot on the
        # chart is a potential stop, not just one.
        entry_resistances: list[float] = []
        if self.enable_resistance_break_exit and entry_idx > 0 and "swing_high" in df.columns:
            cutoff = entry_idx - 1 - self.swing_pivot_right
            if cutoff >= 0:
                start = max(0, cutoff - self.swing_pivot_lookback + 1)
                window = df["swing_high"].iloc[start : cutoff + 1]
                entry_resistances = [float(v) for v in window.dropna().tolist()]

        exit_price, exit_idx, exit_reason = self._find_exit(
            df,
            entry_idx,
            entry_price,
            stop,
            config.max_holding_days,
            exit_dates,
            entry_resistances,
        )

        pnl = (exit_price - entry_price) * shares
        trade = Trade(
            pattern_name=signal.pattern_name,
            entry_date=df.index[entry_idx].date(),
            exit_date=df.index[exit_idx].date(),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            stop_loss=round(stop, 2),
            shares=shares,
            pnl=round(pnl, 2),
            pnl_pct=round((exit_price - entry_price) / entry_price, 4),
            exit_reason=exit_reason,
        )
        return trade, exit_idx

    def _find_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop: float,
        max_holding_days: int,
        exit_dates: set[date] | None = None,
        entry_resistances: list[float] | None = None,
    ) -> tuple[float, int, str]:
        if exit_dates is None:
            exit_dates = set()
        if entry_resistances is None:
            entry_resistances = []
        last_idx = len(df) - 1

        # Smart trail state — Chandelier exit with profit-tier widening.
        # ``initial_risk`` is the 1R unit for R-based trail widening.
        # ``stop`` is no longer checked as a hard stop (removed per
        # spec), but it still defines the risk basis computed by
        # ``_execute_trade`` for position sizing.
        #
        # Break-even stop state: once any close reaches entry + 1R,
        # ``breakeven_armed`` flips True and a subsequent low touching
        # the entry price fires an exit at the entry price (zero P&L
        # before commissions). Protects realized profits from giving
        # back and kills the -3..-10% loss tail.
        breakeven_armed = False
        highest_high = 0.0
        initial_risk = entry_price - stop

        # Per-level confirmation state for the resistance-break exit.
        # A level is "confirmed" once a close has proven it with a
        # buffer on top (``confirm_buffer × ATR``) — tagging the level
        # by a hair doesn't count. Entries that already sit at or
        # above the level count as confirmed from the start (the
        # level is already below the current bar, no confirmation
        # dance needed).
        confirmed_levels = [entry_price >= r for r in entry_resistances]

        # Hoist OHLCV + ATR columns to numpy arrays once — the per-bar
        # loop below runs tens to hundreds of iterations per trade and
        # each ``df["X"].iloc[i]`` call pays pandas-label-dispatch +
        # scalar-boxing overhead. Numpy indexing is ~10-20× cheaper
        # and semantically identical (same dtype, same float values).
        # ``atr`` NaNs are pre-collapsed to 0.0 here rather than
        # inside the loop so behavior still matches the original
        # ``pd.isna`` check.
        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        atrs = df["atr"].to_numpy(dtype=float)
        atrs = np.where(np.isnan(atrs), 0.0, atrs)
        index = df.index

        # Only three exit paths are active:
        #   (1) ``exit_detector`` firing (Exhaustion Extension Top)
        #   (2) higher-low trendline break
        #   (3) Smart Trail (Chandelier)
        # The entry bar itself is skipped for (1) and (2) — a same-bar
        # exit against our own entry is nonsense. (3) also effectively
        # waits ≥ 3 bars via its arming guard. If no condition fires,
        # the position holds until the final bar of the dataframe.
        for i in range(entry_idx, last_idx + 1):
            open_bar = float(opens[i])
            high_bar = float(highs[i])
            low_bar = float(lows[i])
            close = float(closes[i])
            atr = float(atrs[i])

            if high_bar > highest_high:
                highest_high = high_bar

            # (B) Break-even stop — fires before any other exit once
            #     the trade has proven itself (close ≥ entry + 1R)
            #     and the stock subsequently revisits entry. Pierce
            #     check uses LAST bar's arm state; arming flip uses
            #     THIS bar's close, so same-bar arm + pierce cannot
            #     fire (analogous to resistance-break ordering).
            if self.enable_breakeven_stop:
                breakeven_exit_price = entry_price + self.breakeven_exit_offset_r * initial_risk
                if breakeven_armed and i > entry_idx and low_bar <= breakeven_exit_price:
                    return (
                        min(open_bar, breakeven_exit_price),
                        i,
                        "breakeven_stop",
                    )
                if (
                    not breakeven_armed
                    and initial_risk > 0
                    and close >= entry_price + self.breakeven_arm_r_multiple * initial_risk
                ):
                    breakeven_armed = True

            # (1) Exhaustion Extension Top exit — fires when the
            #     injected ``exit_detector`` signals on this bar.
            #     Detection was pre-run causally, so firing on this
            #     bar's close uses no future data.
            if i > entry_idx and df.index[i].date() in exit_dates:
                return close, i, "exhaustion_exit"

            # (2) Higher-low trendline break — LOW-based limit-order
            #     model. The instant the bar's low pierces the trendline
            #     fit through recent confirmable swing lows, the sell
            #     limit at the line fills. Gap-downs fill at the open
            #     (realistic slippage) — symmetric to the smart-trail
            #     fill model. Causality: only pivots with
            #     ``idx ≤ i - right`` contribute, so the trendline at
            #     bar i uses no future data.
            # Structural exits (trendline break + resistance break)
            # honor a post-entry grace window. Bars held from entry_idx
            # that fall inside the window suppress both structural
            # stops, keeping initial chop from firing fast exits while
            # leaving breakeven / exhaustion / smart-trail active.
            structural_active = (i - entry_idx) > self.structural_exit_grace_bars

            if (
                self.enable_trendline_exit
                and i > entry_idx
                and structural_active
                and "swing_low" in df.columns
            ):
                tl = fit_lower_trendline(
                    df["swing_low"],
                    upto_idx=i,
                    lookback=self.swing_pivot_lookback,
                    right=self.swing_pivot_right,
                    max_points=self.trendline_max_pivots,
                    min_points=self.trendline_min_pivots,
                )
                if tl is not None:
                    slope, intercept, _ = tl
                    if slope > 0:
                        trendline_y = slope * i + intercept
                        trigger_hit = (
                            close <= trendline_y
                            if self.structural_exit_close_confirm
                            else low_bar <= trendline_y
                        )
                        if trigger_hit:
                            fill = (
                                close if self.structural_exit_close_confirm else min(open_bar, trendline_y)
                            )
                            return (fill, i, "trendline_break")

            # (4) Resistance-break exit — every swing high in the
            #     lookback window acts as an independent stop.
            #     Order of operations per bar is critical:
            #       (a) check pierces against levels that were
            #           ALREADY confirmed *before* this bar. A
            #           level that gets confirmed on this same
            #           bar (via high touching above the level)
            #           cannot fire an exit on the same bar's
            #           low — intraday order of high vs. low is
            #           ambiguous.
            #       (b) THEN flip newly-confirmed levels based on
            #           this bar's HIGH, so they're active from
            #           the next bar onward. High-based lets a
            #           fast intraday break-through arm the stop
            #           even if the bar pulls back to close below.
            #     When a bar's low pierces any already-active
            #     level, exit at the HIGHEST pierced level — the
            #     tightest fill for the trader — with gap-down
            #     fills at open.
            if self.enable_resistance_break_exit and entry_resistances:
                pierce_buffer = self.resistance_break_pierce_buffer_atr * atr if atr > 0 else 0.0
                confirm_buffer = self.resistance_break_confirm_buffer_atr * atr if atr > 0 else 0.0
                if i > entry_idx and structural_active:
                    pierced_trigger: float | None = None
                    for k, r in enumerate(entry_resistances):
                        trigger = r - pierce_buffer
                        pierce_hit = (
                            close <= trigger if self.structural_exit_close_confirm else low_bar <= trigger
                        )
                        if confirmed_levels[k] and pierce_hit:
                            if pierced_trigger is None or trigger > pierced_trigger:
                                pierced_trigger = trigger
                    if pierced_trigger is not None:
                        fill = (
                            close if self.structural_exit_close_confirm else min(open_bar, pierced_trigger)
                        )
                        return (fill, i, "resistance_break")
                # Intraday confirmation — bar's HIGH touching the
                # level by ``confirm_buffer × ATR`` counts. Switched
                # from close-based so an intraday break-through with
                # a pullback close still arms the stop; waiting for
                # close missed fast moves that confirmed and pulled
                # back on the same bar. Activation is still deferred
                # one bar by the pierce-check ordering above, so a
                # same-bar high-confirm + low-pierce does NOT fire.
                for k, r in enumerate(entry_resistances):
                    if not confirmed_levels[k] and high_bar >= r + confirm_buffer:
                        confirmed_levels[k] = True

            # (3) Smart Trail (Chandelier with profit-tier widening).
            if self.use_smart_trail:
                bars_held = i - entry_idx
                if bars_held >= 3 and atr > 0 and initial_risk > 0:
                    unrealized_r = (close - entry_price) / initial_risk
                    if unrealized_r >= 4.0:
                        eff_mult = 5.0
                    elif unrealized_r >= 2.0:
                        eff_mult = 4.0
                    else:
                        eff_mult = 3.0
                    trail_level = highest_high - eff_mult * atr
                    if low_bar <= trail_level:
                        return min(open_bar, trail_level), i, "smart_trail"

        # Fallback — no exit fired. Hold to the last bar of data.
        return float(df["Close"].iloc[last_idx]), last_idx, "end_of_data"

    # ---- performance builder ----

    @staticmethod
    def _build_performance(
        initial_capital: float,
        final_capital: float,
        max_drawdown: float,
        trades: list[Trade],
    ) -> StrategyPerformance:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        return StrategyPerformance(
            initial_capital=initial_capital,
            final_capital=round(final_capital, 2),
            total_return_pct=round((final_capital - initial_capital) / initial_capital, 4),
            total_trades=len(trades),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
            avg_win_pct=round(sum(t.pnl_pct for t in wins) / len(wins), 4) if wins else 0.0,
            avg_loss_pct=round(sum(t.pnl_pct for t in losses) / len(losses), 4) if losses else 0.0,
            max_drawdown_pct=round(max_drawdown, 4),
            trades=trades,
        )
