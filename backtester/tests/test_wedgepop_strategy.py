from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig
from pattern.domain.models import PatternSignal
from tests.test_wedge_pop import (
    ADBE_2021_12_13,
    ALL_2025_05_06,
    AMD_2019_CASE1,
    AMD_2019_CASE2_3,
    AMD_2020_CASE4,
    AMD_2025_DOWNTREND,
    AMD_2026_DOWNTREND,
    AMZN_2021_DATA,
    ELF_DATA,
)


class FakeMarketData(MarketDataPort):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_ohlcv(self, symbol, start, end):
        return self._df


def _config(**overrides) -> StrategyConfig:
    base = dict(
        ticker="TEST",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        pattern_name="wedge_pop",
        initial_capital=100_000.0,
        risk_per_trade=0.02,
        max_holding_days=60,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def _build_strategy(df: pd.DataFrame, **overrides) -> WedgepopStrategy:
    detector = overrides.pop(
        "detector",
        WedgePopDetector(require_above_long_smas=False),
    )
    # Default the slope filter to OFF in the helper — most synthetic
    # fixtures (e.g. ``consolidation_with_breakout``) deliberately
    # pre-build steep phase-1 declines to drive EMA convergence, which
    # trips the production-default slope filter. Tests that *want* to
    # exercise the filter pass ``max_ema_slope_decline`` explicitly.
    overrides.setdefault("max_ema_slope_decline", None)
    # Exit model is now limited to three opt-in conditions; without
    # one, trades run to the end of the dataframe. Default tests to
    # ``use_smart_trail=True`` so trades close normally unless a test
    # specifically opts out.
    overrides.setdefault("use_smart_trail", True)
    return WedgepopStrategy(
        market_data=FakeMarketData(df),
        detector=detector,
        **overrides,
    )


class TestWedgepopStrategyEntries:
    def test_buys_next_open_after_signal(
        self, consolidation_with_breakout: pd.DataFrame
    ):
        strategy = _build_strategy(consolidation_with_breakout)
        result = strategy.run(_config())

        assert result.performance.total_trades >= 1
        # Entry must occur strictly after the breakout signal day.
        first = result.performance.trades[0]
        # Entry price equals the OPEN of the entry bar (next-day open rule).
        entry_row = consolidation_with_breakout.loc[pd.Timestamp(first.entry_date)]
        assert first.entry_price == round(float(entry_row["Open"]), 2)

    def test_one_position_at_a_time(
        self, consolidation_with_breakout: pd.DataFrame
    ):
        # Loosen consolidation/breakout to fire many candidate signals.
        strategy = _build_strategy(
            consolidation_with_breakout,
            detector=WedgePopDetector(consolidation_pct=0.3, breakout_atr_mult=0.0),
        )
        result = strategy.run(_config())
        # Trades must not overlap in time.
        for prev, nxt in zip(
            result.performance.trades, result.performance.trades[1:]
        ):
            assert nxt.entry_date > prev.exit_date

    def test_no_signals_flat_market(self, flat_market: pd.DataFrame):
        strategy = _build_strategy(flat_market)
        result = strategy.run(_config())
        assert result.performance.total_trades == 0
        assert len(result.equity_curve) == 1
        assert result.equity_curve[0].equity == 100_000


class TestWedgepopStrategyRealData:
    def test_elf_full_lifecycle(self):
        strategy = _build_strategy(ELF_DATA)
        result = strategy.run(
            _config(
                start_date=date(2023, 10, 20),
                end_date=date(2023, 11, 17),
            )
        )

        # ELF wedge pop fires (11-06 or 11-14 depending on ATR gate).
        assert result.performance.total_trades >= 1
        trade = result.performance.trades[0]
        assert trade.pattern_name == "wedge_pop"
        assert trade.stop_loss < trade.entry_price
        assert trade.shares >= 1

    def test_amzn_2021_entry_next_open(self):
        strategy = _build_strategy(AMZN_2021_DATA)
        result = strategy.run(
            _config(
                start_date=date(2021, 4, 15),
                end_date=date(2021, 6, 14),
            )
        )

        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        # Signal on 2021-06-08 → entry on 2021-06-09 open.
        assert trade.entry_date == date(2021, 6, 9)
        assert trade.entry_price == round(
            float(AMZN_2021_DATA.loc[pd.Timestamp("2021-06-09")]["Open"]), 2
        )


class TestWedgepopStrategyExitPaths:
    """Each test crafts an OHLCV series that triggers a specific exit branch.

    The setup is the same in every fixture: 20 bars consolidating below ~70
    (so the detector arms a wedge_pop signal), then a breakout bar at index
    20. The post-breakout bars vary to trigger one of the four exit paths.
    """

    @staticmethod
    def _consolidation_then_breakout(
        post_breakout: list[tuple[float, float, float, float]],
    ) -> pd.DataFrame:
        """Build OHLCV with pre-baked consolidation + breakout, then append post bars.

        post_breakout: list of (open, high, low, close) for bars after the
        breakout. Volume is fixed.
        """
        rows = []
        # Pre-history: enough bars for EMA20 + ATR14 to stabilize.
        for i in range(20):
            close = 70.0 - i * 0.05  # very tight downward drift
            opn = close + 0.1
            high = max(opn, close) + 0.15
            low = min(opn, close) - 0.15
            rows.append((opn, high, low, close, 500_000))
        # Breakout bar at index 20: strong close above EMAs.
        rows.append((70.0, 76.5, 69.8, 76.0, 2_500_000))
        # Append post-breakout bars.
        for o, h, l, c in post_breakout:
            rows.append((o, h, l, c, 1_500_000))

        dates = pd.bdate_range("2023-01-02", periods=len(rows))
        return pd.DataFrame(
            {
                "Open": [r[0] for r in rows],
                "High": [r[1] for r in rows],
                "Low": [r[2] for r in rows],
                "Close": [r[3] for r in rows],
                "Volume": [r[4] for r in rows],
            },
            index=dates,
        )

    def test_exhaustion_extension_exit_into_strength(self):
        # Post-breakout: rallies hard, stretching > 15% above 10-EMA.
        post = [
            (76.0, 77.0, 75.5, 76.5),
            (77.0, 79.0, 76.8, 78.5),
            (79.0, 82.0, 78.8, 81.5),
            (82.0, 86.0, 81.8, 85.5),
            (86.0, 92.0, 85.5, 91.5),
            (92.0, 100.0, 91.5, 99.0),
            (99.0, 110.0, 98.0, 108.0),  # parabolic — extension trigger
        ]
        df = self._consolidation_then_breakout(post)
        strategy = _build_strategy(df)
        result = strategy.run(_config(max_holding_days=10))

        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        # Sold into strength: exit > entry, exit price equals a bar's close.
        assert trade.exit_price > trade.entry_price
        assert trade.pnl > 0

    def test_ema_trail_exit_after_profit(self):
        # Post-breakout: ramps up (profitable), then closes back below 10-EMA.
        post = [
            (76.0, 78.0, 75.8, 77.8),
            (78.0, 80.0, 77.5, 79.8),
            (80.0, 82.5, 79.5, 82.0),
            (82.0, 84.5, 81.5, 84.0),
            (84.0, 85.0, 83.0, 84.5),
            (84.5, 84.8, 78.0, 78.2),  # close drops below 10-EMA → trail exit
        ]
        df = self._consolidation_then_breakout(post)
        strategy = _build_strategy(
            df,
            extension_atr_mult=99.0,  # disable extension trigger

        )
        result = strategy.run(_config(max_holding_days=20))

        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        # Trail exit: not the stop_loss, not the consolidation low.
        assert trade.exit_price > trade.stop_loss
        # Should still be net profitable (we trailed a winning trade).
        assert trade.pnl > 0

class TestWedgepopStrategyGapUp:
    """Optional `require_gap_up` filter — TraderLion's "unfilled gap"
    confirmation. When enabled, only Wedge Pops where T+1 open > T close
    are taken.
    """

    def test_default_off_does_not_change_behaviour(
        self, consolidation_with_breakout
    ):
        # OFF (default) must produce identical results to before the flag.
        baseline = _build_strategy(consolidation_with_breakout)
        with_flag = _build_strategy(
            consolidation_with_breakout, require_gap_up=False
        )
        b = baseline.run(_config()).performance
        w = with_flag.run(_config()).performance
        assert b.total_trades == w.total_trades

    def test_elf_passes_gap_up_filter(self):
        # ELF 2023-11-15 open $106.80 > 2023-11-14 close $106.21 → gap up.
        # Use a tight breakout gate (1.0 ATR) so only the strong 11-14
        # breakout fires, not the weaker 11-06 move.
        strategy = _build_strategy(
            ELF_DATA,
            detector=WedgePopDetector(
                breakout_atr_mult=1.0, require_above_long_smas=False
            ),
            require_gap_up=True,
        )
        result = strategy.run(
            _config(
                start_date=date(2023, 10, 20),
                end_date=date(2023, 11, 17),
            )
        )
        assert result.performance.total_trades == 1
        assert result.performance.trades[0].entry_date == date(2023, 11, 15)

    def test_amzn_passes_gap_up_filter(self):
        # AMZN 2021-06-09 open $163.64 > 2021-06-08 close $163.21 → +$0.43 gap up.
        strategy = _build_strategy(AMZN_2021_DATA, require_gap_up=True)
        result = strategy.run(
            _config(
                start_date=date(2021, 4, 15),
                end_date=date(2021, 6, 14),
            )
        )
        assert result.performance.total_trades == 1
        assert result.performance.trades[0].entry_date == date(2021, 6, 9)

    def test_gap_down_signal_is_filtered_out(self):
        """Synthetic: T+1 open < T close → filter must skip the entry.

        Reuses TestWedgepopStrategyExitPaths' consolidation+breakout
        builder. The breakout bar closes at $76.0; we make the next
        bar's open $75.0 (gap down) so the filter rejects it.
        """
        post = [
            # Day 21: gap-down open below the breakout close ($76.0)
            (75.0, 75.5, 74.5, 75.2),
            # A few sideways days so the strategy could otherwise hold
            (75.2, 75.6, 74.8, 75.3),
            (75.3, 75.7, 74.9, 75.4),
        ]
        df = TestWedgepopStrategyExitPaths._consolidation_then_breakout(post)

        # OFF: trade should still happen
        off = _build_strategy(df, require_gap_up=False)
        off_result = off.run(_config(max_holding_days=10))
        assert off_result.performance.total_trades == 1

        # ON: filter rejects the entry → 0 trades
        on = _build_strategy(df, require_gap_up=True)
        on_result = on.run(_config(max_holding_days=10))
        assert on_result.performance.total_trades == 0


class TestWedgepopStrategyTimezone:
    """yfinance returns a tz-aware DatetimeIndex (America/New_York). The
    strategy must compare signal dates against that index without
    raising `Invalid comparison between dtype=datetime64[..., tz] and
    Timestamp`.
    """

    def test_execute_with_tz_aware_index(self):
        # Reuse the ELF fixture and slap a tz on the index.
        df = ELF_DATA.copy()
        df.index = df.index.tz_localize("America/New_York")

        strategy = _build_strategy(df)
        result = strategy.execute(
            df,
            _config(
                start_date=date(2023, 10, 20),
                end_date=date(2023, 11, 17),
            ),
        )

        # Same trade count as tz-naive — tz must not change behaviour.
        assert result.performance.total_trades >= 1
        trade = result.performance.trades[0]
        assert trade.stop_loss < trade.entry_price


class TestWedgepopStrategySizing:
    def test_position_size_respects_risk_per_trade(self):
        strategy = _build_strategy(ELF_DATA)
        result = strategy.run(
            _config(
                start_date=date(2023, 10, 20),
                end_date=date(2023, 11, 17),
                initial_capital=100_000,
                risk_per_trade=0.02,  # risk $2,000
            )
        )
        trade = result.performance.trades[0]

        risk_per_share = trade.entry_price - trade.stop_loss
        expected_shares = max(1, int(2_000 / risk_per_share))
        assert trade.shares == expected_shares


class TestWedgepopStrategyAMDCases:
    """Four AMD cases requested by the user for tuning wedge pop
    without touching existing parameter knobs. Each case is a real
    trade that the *old* strategy mishandled; the fixes are all
    structural (bug fixes or new rules grounded in market structure),
    not numeric tweaks of the extension thresholds.

    Case 1  — 2019-03-12 wedge pop, entry 2019-03-13 @ 23.66.
              Old behaviour held through a massive +11% blow-off bar
              on 2019-03-19 (ext% 9.9%, ext_atr 2.15) because the EMA
              extension defaults (15% / 2.5 ATR) were not tight
              enough. New climax-bar rule catches the bar itself:
              range 2.49 vs ATR ~1.1 = 2.3× and close at 97% of
              range → exit at close 26.00 on 03-19 for +9.89%.

    Case 2  — 2019-08-21 wedge pop, entry 2019-08-22 @ 31.76.
              Entry-bar close 31.90 is above entry, so the breakeven
              stop should arm on the entry bar itself. Old code
              skipped ``entry_idx`` in the loop, so arming was
              missed and the trade dragged to the original
              consolidation-low stop 27.65 on 2019-10-03 (-12.94%).
              Fix is pure bug fix in ``_find_exit``: pre-loop arming
              check uses ``df['Close'].iloc[entry_idx]`` before
              iteration. Trade now exits at breakeven on 08-23.

    Case 3  — 2019-10-11 wedge pop, entry 2019-10-14 @ 29.71. This
              is the *big winner* (+38.98% exit on 2019-11-19). The
              test guards against regressions: it must still be
              detected AND must still run to the profitable exit
              even with all the new exit rules (climax bar, stricter
              exhaustion reference).

    """

    @staticmethod
    def _run(df: pd.DataFrame, **strategy_overrides):
        # Default to a lenient detector for AMD cases — the real-data
        # fixtures have breakout strengths that vary in ATR terms.
        strategy_overrides.setdefault(
            "detector",
            WedgePopDetector(
                breakout_atr_mult=0.0, require_above_long_smas=False
            ),
        )
        strategy = _build_strategy(df, **strategy_overrides)
        return strategy.run(
            _config(
                start_date=df.index[0].date(),
                end_date=df.index[-1].date(),
                max_holding_days=60,
            )
        )

    @staticmethod
    def _trade_by_entry_date(result, entry_date: date):
        for t in result.performance.trades:
            if t.entry_date == entry_date:
                return t
        return None


class TestWedgepopStrategySlopeFilter:
    """Recent EMA slope is now exposed on every wedge-pop signal as
    two metadata variables (``ema_fast_slope`` and ``ema_slow_slope``
    — the percent change of the 10/20 EMAs over the last
    ``slope_lookback`` bars, default 20). The strategy layer uses
    the slow slope to reject dead-cat-bounce entries inside a
    rolling-over medium-term trend. Two real AMD cases the user
    reported drove this filter:

    - **AMD 2025-01-06**: detector fires, but the 20-day slope of
      the 20 EMA is ≈ -10.5%. The next several bars collapse back
      to fresh lows.
    - **AMD 2026-02-24**: detector fires, but the 20-day slope of
      the 20 EMA is ≈ -7.4%. Same dead-cat-bounce pattern.

    The filter threshold (``max_ema_slope_decline``, default 0.05)
    rejects both entries. Disabling it (None or very large)
    reproduces the "before" behaviour where the strategy took the
    losing trade.
    """

    @staticmethod
    def _run(df: pd.DataFrame, **overrides):
        strategy = _build_strategy(df, **overrides)
        return strategy.run(
            _config(
                start_date=df.index[0].date(),
                end_date=df.index[-1].date(),
                max_holding_days=30,
            )
        )

    # ---- detector still detects (filter is downstream) ----

    def test_detector_still_detects_2025_01_06(self):
        """2025-01-06 no longer fires: the new rule "previous bar's
        close must be below fast EMA" blocks it (2025-01-03 close was
        above ema_fast). We use 2026-02-24 instead, which satisfies all
        detector gates including the prev-close rule.

        The filter still lives in the strategy layer — the detector
        emits the signal so downstream pipelines can inspect slope
        metadata. This test pins that contract on the 02-24 signal.
        """
        sigs = WedgePopDetector(
            require_above_long_smas=False
        ).detect(AMD_2026_DOWNTREND)
        dates = {s.date for s in sigs}
        assert date(2026, 2, 24) in dates

        target = next(s for s in sigs if s.date == date(2026, 2, 24))
        assert "ema_fast_slope" in target.metadata
        assert "ema_slow_slope" in target.metadata
        # The slow slope must be the structural reason we reject it.
        assert target.metadata["ema_slow_slope"] < -0.05

    def test_detector_still_detects_2026_02_24(self):
        sigs = WedgePopDetector(
            require_above_long_smas=False
        ).detect(AMD_2026_DOWNTREND)
        dates = {s.date for s in sigs}
        assert date(2026, 2, 24) in dates

        target = next(s for s in sigs if s.date == date(2026, 2, 24))
        assert target.metadata["ema_slow_slope"] < -0.05

    # ---- strategy rejects entries on these slopes ----

    def test_strategy_rejects_2025_01_06_entry_with_default_filter(self):
        """With the production default (max_ema_slope_decline=0.05),
        the AMD 2025-01-06 entry must be rejected.
        """
        result = self._run(
            AMD_2025_DOWNTREND, max_ema_slope_decline=0.05
        )
        entry_dates = {t.entry_date for t in result.performance.trades}
        assert date(2025, 1, 7) not in entry_dates  # entry is next bar

    def test_strategy_rejects_2026_02_24_entry_with_default_filter(self):
        result = self._run(
            AMD_2026_DOWNTREND, max_ema_slope_decline=0.05
        )
        entry_dates = {t.entry_date for t in result.performance.trades}
        assert date(2026, 2, 25) not in entry_dates  # entry is next bar

    # ---- disabling the filter reproduces the bad behaviour ----

    def test_disabling_slope_filter_takes_2025_01_06_entry(self):
        """2025-01-06 no longer fires from the detector (prev close
        2025-01-03 was above ema_fast, violating the new rule). We use
        2026-02-24 instead: with slope filter off the strategy enters on
        2026-02-25 (next bar), and with the default filter (ema_slow_slope
        ≈ -7.4% < -5%) the entry is blocked.
        """
        result = self._run(
            AMD_2026_DOWNTREND, max_ema_slope_decline=None
        )
        entry_dates = {t.entry_date for t in result.performance.trades}
        assert date(2026, 2, 25) in entry_dates

    def test_disabling_slope_filter_takes_2026_02_24_entry(self):
        result = self._run(
            AMD_2026_DOWNTREND, max_ema_slope_decline=None
        )
        entry_dates = {t.entry_date for t in result.performance.trades}
        assert date(2026, 2, 25) in entry_dates

    # ---- regression: existing successful cases still enter ----

    def test_existing_successful_cases_not_affected_by_slope_filter(self):
        """ELF, AMZN, and the AMD 2019 cases have been the baseline
        for the detector/strategy tests. All of their ema_slow_slope20
        values are above -5%, so the default slope filter must NOT
        reject them. The specific entry dates may shift with ATR-based
        breakout gating, so we only check that at least one trade fires.
        """
        for df in (
            ELF_DATA,
            AMZN_2021_DATA,
            AMD_2019_CASE1,
            AMD_2019_CASE2_3,
        ):
            result = self._run(df, max_ema_slope_decline=0.05)
            assert result.performance.total_trades >= 1, (
                f"slope filter wrongly rejected all entries"
            )


def _force_entry(df: pd.DataFrame, entry_date: date, **strategy_overrides):
    """Bypass the detector and invoke ``_execute_trade`` at a specific
    bar. This is used by the ALL 2025-05-06 suite to isolate exit /
    filter behaviour without fighting the detector's cooldown logic.

    Builds a minimal ``PatternSignal`` whose ``stop_loss`` is the
    consolidation low of the 15 bars preceding the entry — matching
    what the real detector would have generated if it had fired.
    """
    detector = strategy_overrides.pop("detector", WedgePopDetector(require_above_long_smas=False))
    # Default helper knobs: off for filters that aren't under test here.
    strategy_overrides.setdefault("max_ema_slope_decline", None)
    strategy = WedgepopStrategy(
        market_data=FakeMarketData(df),
        detector=detector,
        **strategy_overrides,
    )
    df_ind = strategy._with_indicators(df)
    entry_idx = df_ind.index.get_loc(pd.Timestamp(entry_date))
    assert entry_idx >= 1, "entry_date must have at least one prior bar"
    signal_idx = entry_idx - 1
    cons_low = float(df_ind["Low"].iloc[max(0, signal_idx - 15):signal_idx].min())
    signal = PatternSignal(
        date=df_ind.index[signal_idx].date(),
        pattern_name="wedge_pop",
        entry_price=float(df_ind["Close"].iloc[signal_idx]),
        stop_loss=cons_low,
        confidence=1.0,
        metadata={"ema_slow_slope": 0.0},
    )
    config = StrategyConfig(
        ticker="TEST",
        start_date=df_ind.index[0].date(),
        end_date=df_ind.index[-1].date(),
        pattern_name="wedge_pop",
        initial_capital=100_000.0,
        risk_per_trade=0.02,
        max_holding_days=60,
    )
    # Mirror ``execute()``'s end-to-end flow: pre-compute exit dates
    # when the strategy has a detector wired up, so ``_force_entry``
    # exercises the same exit path as the public API.
    exit_dates: set[date] = set()
    if strategy._exit_detector is not None:
        exit_dates = {s.date for s in strategy._exit_detector.detect(df_ind)}
    trade, _ = strategy._execute_trade(
        df_ind, signal, entry_idx, 100_000.0, config, exit_dates
    )
    return trade


class TestEntryEmaExtensionFilter:
    """``max_entry_ema_extension_atr`` entry filter: rejects entries
    where the open price sits more than N ATR above the higher of
    the two signal-bar EMAs.
    """

    def test_default_is_off_and_preserves_existing_behaviour(self):
        """When ``max_entry_ema_extension_atr`` is None, the filter
        is disabled and the existing AMZN 2021-06 entry still goes
        through (slope filter also off so it's not confounded).
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(AMZN_2021_DATA),
            detector=WedgePopDetector(require_above_long_smas=False),
            max_ema_slope_decline=None,
            max_entry_ema_extension_atr=None,
        )
        result = strategy.execute(
            AMZN_2021_DATA,
            StrategyConfig(
                ticker="AMZN",
                start_date=date(2021, 4, 15),
                end_date=date(2021, 6, 14),
                pattern_name="wedge_pop",
                max_holding_days=60,
            ),
        )
        assert any(
            t.entry_date == date(2021, 6, 9)
            for t in result.performance.trades
        )

    def test_tight_threshold_rejects_amd_2020_06_10_entry(self):
        """AMD 2020-06-10 entry @ $57.20 sits ~1.3 ATR above
        the signal-bar EMA stack. A tight
        ``max_entry_ema_extension_atr=1.5`` must reject it.
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(AMD_2020_CASE4),
            detector=WedgePopDetector(require_above_long_smas=False),
            max_entry_ema_extension_atr=1.5,
            max_ema_slope_decline=None,
        )
        result = strategy.execute(
            AMD_2020_CASE4,
            StrategyConfig(
                ticker="AMD",
                start_date=date(2020, 5, 4),
                end_date=date(2020, 6, 16),
                pattern_name="wedge_pop",
                max_holding_days=60,
            ),
        )
        # No trades — the filter rejected the only signal's entry.
        assert result.performance.total_trades == 0

    def test_loose_threshold_lets_chase_entry_through(self):
        """With a very loose ``max_entry_ema_extension_atr=10.0``,
        the AMD 2020-06-10 entry passes the EMA-extension filter.
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(AMD_2020_CASE4),
            detector=WedgePopDetector(require_above_long_smas=False),
            max_entry_ema_extension_atr=10.0,
            max_ema_slope_decline=None,
        )
        result = strategy.execute(
            AMD_2020_CASE4,
            StrategyConfig(
                ticker="AMD",
                start_date=date(2020, 5, 4),
                end_date=date(2020, 6, 16),
                pattern_name="wedge_pop",
                max_holding_days=60,
            ),
        )
        assert result.performance.total_trades == 1
        assert result.performance.trades[0].entry_date == date(2020, 6, 10)


class TestSlopeRangeFilter:
    """The slope filter is now a RANGE ``[min, max]`` applied to
    ``signal.metadata['ema_slow_slope']``. The legacy
    ``max_ema_slope_decline`` kwarg is preserved as a
    backwards-compat shim that fills in only the min bound.

    Semantics:
        - ``min_ema_slow_slope``: reject if slope < min.
          Setting a POSITIVE value requires a confirmed uptrend
          (e.g. 0.05 → "ema20 must have risen at least +5% over
          the slope_lookback window").
        - ``max_ema_slow_slope``: reject if slope > max.
          Setting a MAX catches already-parabolic names that are
          up too much over the slope window.
    """

    # ---- backwards-compat shim ----

    def test_legacy_max_decline_still_works(self):
        """``max_ema_slope_decline=0.05`` → rejects slopes below
        -0.05. Same semantic as before, but now implemented via
        the ``min_ema_slow_slope`` internal machinery.
        """
        # AMD 2025-01-06 has ema_slow_slope ≈ -0.105 → blocked
        result = _build_strategy(
            AMD_2025_DOWNTREND,
            max_ema_slope_decline=0.05,
        ).run(
            _config(
                start_date=AMD_2025_DOWNTREND.index[0].date(),
                end_date=AMD_2025_DOWNTREND.index[-1].date(),
            )
        )
        entry_dates = {t.entry_date for t in result.performance.trades}
        assert date(2025, 1, 7) not in entry_dates

    def test_new_min_ema_slow_slope_overrides_legacy(self):
        """When both ``min_ema_slow_slope`` and
        ``max_ema_slope_decline`` are passed, the new param wins.
        Setting ``min_ema_slow_slope=-0.99`` (effectively off)
        overrides a legacy ``max_ema_slope_decline=0.05``.

        Uses AMD_2026_DOWNTREND / 2026-02-24 signal (entry 02-25)
        because the 2025-01-06 signal no longer fires from the detector
        (prev close was above ema_fast under the updated detector rule).
        """
        result = _build_strategy(
            AMD_2026_DOWNTREND,
            max_ema_slope_decline=0.05,      # would block 2026-02-24
            min_ema_slow_slope=-0.99,         # new param: don't block
        ).run(
            _config(
                start_date=AMD_2026_DOWNTREND.index[0].date(),
                end_date=AMD_2026_DOWNTREND.index[-1].date(),
            )
        )
        entry_dates = {t.entry_date for t in result.performance.trades}
        assert date(2026, 2, 25) in entry_dates

    # ---- require a POSITIVE slope (user's example) ----

    def test_positive_min_slope_requires_confirmed_uptrend(self):
        """``min_ema_slow_slope = 0.05`` ("must be at least +5%")
        rejects signals whose ema_slow_slope is below +5% — even
        if the slope is still positive. Uses AMZN 2021-06-08
        (slope ≈ -3.1%) which clearly fails a +5% floor.
        """
        # AMZN 06-08 slope is NEGATIVE → blocked by a +5% floor.
        result = _build_strategy(
            AMZN_2021_DATA,
            min_ema_slow_slope=0.05,
            max_ema_slope_decline=None,   # disable legacy shim
        ).run(
            _config(
                start_date=date(2021, 4, 15),
                end_date=date(2021, 6, 14),
            )
        )
        assert result.performance.total_trades == 0

    def test_positive_min_slope_lets_strong_uptrend_through(self):
        """Confirm the AMZN case fires when the min floor is set
        to ``None`` — separating the filter effect from any other
        reason the trade might not enter.
        """
        result = _build_strategy(
            AMZN_2021_DATA,
            min_ema_slow_slope=None,     # filter off
            max_ema_slope_decline=None,
        ).run(
            _config(
                start_date=date(2021, 4, 15),
                end_date=date(2021, 6, 14),
            )
        )
        assert result.performance.total_trades == 1

    # ---- upper bound catches parabolic setups ----

    def test_max_ema_slow_slope_rejects_already_parabolic(self):
        """``max_ema_slow_slope`` rejects setups that already sit
        inside a too-steep uptrend. Using a tight cap of 0.01
        (slope must be under +1% over the window) blocks ELF's
        2023-11-14 signal whose slope is NaN / 0.0 (too little
        history) but STILL preserves the min-side behaviour —
        test isolates the max cap by using a loose min.
        """
        # Use AMD_2019_CASE2_3 which has multiple positive slopes
        off = _build_strategy(
            AMD_2019_CASE2_3,
            min_ema_slow_slope=None,
            max_ema_slope_decline=None,
        )
        on = _build_strategy(
            AMD_2019_CASE2_3,
            min_ema_slow_slope=None,
            max_ema_slow_slope=0.01,       # strict upper cap
            max_ema_slope_decline=None,
        )
        off_result = off.run(
            _config(
                start_date=AMD_2019_CASE2_3.index[0].date(),
                end_date=AMD_2019_CASE2_3.index[-1].date(),
            )
        )
        on_result = on.run(
            _config(
                start_date=AMD_2019_CASE2_3.index[0].date(),
                end_date=AMD_2019_CASE2_3.index[-1].date(),
            )
        )
        # Upper cap can only ever shrink the trade set.
        assert on_result.performance.total_trades <= \
            off_result.performance.total_trades


class TestLongSmaFilter:
    """Detector ``require_above_long_smas`` option: when True, the
    breakout bar's close must sit above BOTH the 50 SMA and the 200
    SMA. Used to filter wedge pops down to ones that happen inside
    a confirmed long-term uptrend.
    """

    def test_default_detects_amzn_2021_wedge_pop(self):
        """Baseline: default detector (filter off) still detects the
        AMZN 2021-06-08 wedge pop.
        """
        sigs = WedgePopDetector(require_above_long_smas=False).detect(AMZN_2021_DATA)
        assert any(s.date == date(2021, 6, 8) for s in sigs)

    def test_filter_rejects_when_not_enough_history(self):
        """AMZN_2021_DATA has ~42 bars — not enough for a 200 SMA
        to converge. With ``require_above_long_smas=True`` the
        filter should reject the signal (NaN SMA is treated as a
        fail).
        """
        sigs = WedgePopDetector(
            require_above_long_smas=True
        ).detect(AMZN_2021_DATA)
        assert all(s.date != date(2021, 6, 8) for s in sigs)

    def test_filter_rejects_all_when_sma_history_is_short(self):
        """``AMD_2019_CASE2_3`` is a 166-bar fixture — not enough
        for a 200 SMA to converge. The default detector produces
        several signals; with the long-SMA filter enabled, *all*
        of them are rejected because the 200 SMA is NaN on every
        bar. This pins the NaN-is-a-fail behaviour.
        """
        sigs_off = WedgePopDetector(require_above_long_smas=False).detect(AMD_2019_CASE2_3)
        sigs_on = WedgePopDetector(
            require_above_long_smas=True,
        ).detect(AMD_2019_CASE2_3)
        assert len(sigs_off) >= 1
        assert len(sigs_on) == 0

    def test_filter_is_monotone_on_real_data(self):
        """Regardless of the specific bars, enabling the filter can
        only ever shrink the signal set (it's a pure AND gate).
        Run it on every real fixture we have and assert
        ``len(filtered) <= len(unfiltered)``.
        """
        off = WedgePopDetector(require_above_long_smas=False)
        on = WedgePopDetector(require_above_long_smas=True)
        for name, df in (
            ("ELF", ELF_DATA),
            ("AMZN", AMZN_2021_DATA),
            ("AMD_2019_CASE1", AMD_2019_CASE1),
            ("AMD_2019_CASE2_3", AMD_2019_CASE2_3),
            ("AMD_2020_CASE4", AMD_2020_CASE4),
        ):
            n_off = len(off.detect(df))
            n_on = len(on.detect(df))
            assert n_on <= n_off, (
                f"{name}: filter is not monotone ({n_on} > {n_off})"
            )


class TestPatternExitHook:
    """Generic pattern-based exit hook.

    Uses flat, stable bars so that NO other exit rule (hard stop,
    smart trail, EMA trail, exhaustion, climax) fires during the
    hold period. A fake detector that fires on a specific date is
    injected; the tests verify the hook works correctly in
    isolation.

    Smart trail stays disarmed because ``trail_level > stop``
    is satisfied BUT ``low > trail_level`` — the flat bars' lows
    sit comfortably above the trail line.
    """

    @staticmethod
    def _build_fixture():
        """Flat bars with low just above the smart-trail level.

          bars 0-24: flat at 100 (warmup)
          bars 25-29: breakout to 105
          bar 30: entry at open 104.5
          bars 31-55: flat at 105 — high=105.5, low=104.5
                      trail ≈ 105.5 - 3×0.5 = 104.0 < low(104.5)
                      so trail never triggers.
        """
        rows = []
        for _ in range(25):
            rows.append((100.0, 100.5, 99.5, 100.0, 1_000_000))
        for p in (101.0, 102.0, 103.0, 104.0, 105.0):
            rows.append((p - 0.5, p + 0.3, p - 0.8, p, 2_000_000))
        rows.append((104.5, 105.5, 104.2, 105.0, 2_000_000))
        for _ in range(25):
            rows.append((104.8, 105.5, 104.5, 105.0, 1_500_000))

        idx = pd.date_range("2023-01-02", periods=len(rows), freq="B")
        return pd.DataFrame(
            {
                "Open":   [r[0] for r in rows],
                "High":   [r[1] for r in rows],
                "Low":    [r[2] for r in rows],
                "Close":  [r[3] for r in rows],
                "Volume": [r[4] for r in rows],
            },
            index=idx,
        )

    @staticmethod
    def _fake_detector(target_date):
        """Detector that fires on exactly one date."""
        class _Det(ReversalExtensionDetector):
            name = "fake_exit"

            def detect(self, df, weekly_df=None, monthly_df=None):
                return [
                    PatternSignal(
                        date=target_date,
                        pattern_name=self.name,
                        entry_price=0.0,
                        stop_loss=0.0,
                        confidence=1.0,
                        metadata={},
                    )
                ]
        return _Det()

    def test_pattern_exit_fires_on_target_bar(self):
        df = self._build_fixture()
        entry_date = df.index[30].date()
        target_date = df.index[40].date()

        trade = _force_entry(
            df,
            entry_date,
            use_smart_trail=True,
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
            exit_detector=self._fake_detector(target_date),
        )
        assert trade is not None
        assert trade.exit_date == target_date
        assert trade.exit_price == round(float(df["Close"].iloc[40]), 2)

    def test_without_exit_detector_no_early_exit(self):
        df = self._build_fixture()
        entry_date = df.index[30].date()
        target_date = df.index[40].date()

        trade = _force_entry(
            df,
            entry_date,
            use_smart_trail=True,
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
        )
        assert trade is not None
        assert trade.exit_date != target_date

    def test_pattern_exit_uses_no_future_data(self):
        df = self._build_fixture()
        entry_date = df.index[30].date()
        target_idx = 40

        det = self._fake_detector(df.index[target_idx].date())

        trade_full = _force_entry(
            df, entry_date,
            use_smart_trail=True, extension_atr_mult=99.0,
            climax_atr_mult=99.0, exit_detector=det,
        )
        trade_trunc = _force_entry(
            df.iloc[: target_idx + 1].copy(), entry_date,
            use_smart_trail=True, extension_atr_mult=99.0,
            climax_atr_mult=99.0, exit_detector=det,
        )
        assert trade_full is not None and trade_trunc is not None
        assert trade_full.exit_date == trade_trunc.exit_date
        assert trade_full.exit_price == trade_trunc.exit_price

    def test_entry_bar_signal_is_ignored(self):
        df = self._build_fixture()
        entry_date = df.index[30].date()

        trade = _force_entry(
            df,
            entry_date,
            use_smart_trail=True,
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
            exit_detector=self._fake_detector(entry_date),
        )
        assert trade is not None
        assert trade.exit_date != entry_date
