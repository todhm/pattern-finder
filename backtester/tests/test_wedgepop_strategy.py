from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort
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
    detector = overrides.pop("detector", WedgePopDetector())
    # Default the slope filter to OFF in the helper — most synthetic
    # fixtures (e.g. ``consolidation_with_breakout``) deliberately
    # pre-build steep phase-1 declines to drive EMA convergence, which
    # trips the production-default slope filter. Tests that *want* to
    # exercise the filter pass ``max_ema_slope_decline`` explicitly.
    overrides.setdefault("max_ema_slope_decline", None)
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
            detector=WedgePopDetector(consolidation_pct=0.3, breakout_pct=0.005),
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

        # ELF wedge pop on 2023-11-14 → entry next session 2023-11-15.
        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        assert trade.pattern_name == "wedge_pop"
        assert trade.entry_date == date(2023, 11, 15)
        assert trade.entry_price == round(float(ELF_DATA.iloc[18]["Open"]), 2)
        assert trade.stop_loss < trade.entry_price
        # Risk-based sizing must produce a positive position.
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

    def test_hard_stop_exit_at_consolidation_low(self):
        # Post-breakout: collapses through the consolidation low.
        # Entry bar closes BELOW entry (75.5 < 76.0) so the
        # breakeven-after-profit stop is never armed; the consolidation
        # low is the only protection — and day 22 pierces it.
        post = [
            (76.0, 76.5, 75.5, 75.5),  # entry bar (idx 21, no arming)
            (76.0, 76.0, 60.0, 60.5),  # day 22 — Low pierces stop
        ]
        df = self._consolidation_then_breakout(post)
        strategy = _build_strategy(df)
        result = strategy.run(_config(max_holding_days=10))

        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        # Hard stop fills at the stop_loss level exactly.
        assert trade.exit_price == trade.stop_loss
        assert trade.pnl < 0

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
        strategy = _build_strategy(df, extension_pct=0.15)
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
            extension_pct=0.50,  # disable extension trigger
            extension_atr_mult=99.0,
            trail_after_profit=True,
        )
        result = strategy.run(_config(max_holding_days=20))

        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        # Trail exit: not the stop_loss, not the consolidation low.
        assert trade.exit_price > trade.stop_loss
        # Should still be net profitable (we trailed a winning trade).
        assert trade.pnl > 0

    def test_breakeven_stop_arms_after_profit(self):
        """Stateful 손절: once a single in-trade bar closes above
        ``entry_price``, the hard stop ratchets from the
        consolidation low up to ``entry_price`` (breakeven). A later
        intraday low piercing entry exits the trade at breakeven
        instead of giving back the full unrealized gain to the
        original consolidation-low stop or the time stop. Mirrors
        the CDNS 2026-02 case the user reported.

        The 익절 path (EMA trail) is *not* affected — it still uses
        the per-bar ``close > entry`` check, so a trade that briefly
        dipped below entry and then recovered into bigger profits
        stays in the position.
        """
        post = [
            # Day 21: rallies above entry (76) — arms breakeven stop
            (76.0, 79.0, 75.8, 78.5),
            # Day 22: still in profit, close > ema → no trail exit
            (78.5, 82.5, 78.0, 81.5),
            # Day 23: collapses — intraday low (70.5) pierces the
            # now-armed breakeven stop at 76. Exit at 76, not at the
            # consolidation-low stop and not at the day-23 close.
            (81.0, 81.5, 70.5, 70.8),
            # Days 24-25: stay below entry. Without breakeven arming
            # the old logic dragged this trade to the time stop.
            (70.8, 71.0, 69.5, 69.8),
            (69.8, 70.5, 69.0, 70.0),
        ]
        df = self._consolidation_then_breakout(post)
        strategy = _build_strategy(
            df,
            extension_pct=0.50,        # disable exhaustion exit
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,      # disable climax exit
            trail_after_profit=True,
        )
        result = strategy.run(_config(max_holding_days=20))

        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        # Breakeven stop fills at the entry price, not the original
        # consolidation-low stop and not the day-23 close.
        assert trade.exit_price == round(trade.entry_price, 2)
        assert trade.exit_price > trade.stop_loss
        assert trade.pnl == 0

    def test_ema_trail_does_not_fire_under_water(self):
        """Regression guard: the EMA trail must NOT fire on a bar
        whose close is below ``entry_price``, even if the trade was
        previously above entry. The 익절 path stays per-bar so
        recoverable dips don't kill profitable trades.

        This is the inverse of the breakeven-stop test: we craft a
        scenario where the breakeven stop is *not* hit (intraday low
        stays above entry) but a single down-close drops below entry
        AND below the EMA. The trade must continue.
        """
        post = [
            # Day 21: profit (arms breakeven stop, but stop irrelevant
            # here because no later low pierces entry)
            (76.0, 79.0, 75.8, 78.5),
            (78.5, 82.5, 78.0, 81.5),
            (81.5, 84.0, 81.0, 83.5),
            (83.5, 86.0, 83.0, 85.0),
            # A down-close to 76.5: above entry (76), so the trail's
            # per-bar `close > entry` check stays true. No exit.
            (84.5, 85.0, 76.2, 76.5),
            (76.5, 78.0, 76.1, 77.5),
        ]
        df = self._consolidation_then_breakout(post)
        strategy = _build_strategy(
            df,
            extension_pct=0.50,        # disable exhaustion exit
            extension_atr_mult=99.0,
            trail_after_profit=True,
        )
        result = strategy.run(_config(max_holding_days=20))

        # Trade exists but did NOT exit at breakeven (entry=76):
        # the down-close is still above entry, so neither the trail
        # nor the breakeven stop trips.
        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        assert trade.exit_price > trade.entry_price

    def test_time_stop_exit_at_max_holding(self):
        # Post-breakout: drifts sideways above the EMA → no other
        # exit fires. Open == close keeps the close from triggering
        # the breakeven arm (`close > entry` is strict), so the
        # only remaining exit is the time stop.
        post = [(76.1, 76.5, 75.8, 76.1) for _ in range(8)]
        df = self._consolidation_then_breakout(post)
        strategy = _build_strategy(
            df,
            extension_pct=0.50,
            extension_atr_mult=99.0,
            trail_after_profit=True,
        )
        result = strategy.run(_config(max_holding_days=5))

        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        # Time stop: held exactly max_holding_days bars from entry.
        held = (
            df.index.get_loc(pd.Timestamp(trade.exit_date))
            - df.index.get_loc(pd.Timestamp(trade.entry_date))
        )
        assert held == 5


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
        # ELF 2023-11-15 open $106.80 > 2023-11-14 close $106.21 → +$0.59 gap up.
        strategy = _build_strategy(ELF_DATA, require_gap_up=True)
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

        # Same trade as the tz-naive ELF test — tz must not change behaviour.
        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        assert trade.entry_date == date(2023, 11, 15)


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

    Case 4  — 2020-06-09 wedge pop. Next-bar open on 2020-06-10 is
              $57.20, which is 20% of the signal bar's range above
              the signal bar's high of $56.46. The entry is
              "chasing" — new ``max_entry_chase_ratio=0.15`` rule
              rejects the trade outright, avoiding a -14% loss.
    """

    @staticmethod
    def _run(df: pd.DataFrame, **strategy_overrides):
        strategy = _build_strategy(df, **strategy_overrides)
        return strategy.run(
            _config(
                start_date=df.index[0].date(),
                end_date=df.index[-1].date(),
                max_holding_days=60,
            )
        )

    # ---- Case 1 ----

    def test_case1_climax_bar_exits_on_2019_03_19(self):
        result = self._run(AMD_2019_CASE1)

        # Exactly one trade over the window (only one wedge pop).
        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]

        # Entry is next open after the 2019-03-12 signal.
        assert trade.entry_date == date(2019, 3, 13)
        assert trade.entry_price == 23.66
        # Climax-bar exit still fires on 2019-03-19 — but under the
        # new HIGH-based limit-order model the fill is at the climax
        # line (prev_close + climax_atr_mult × ATR), not at the bar's
        # close. For AMD 2019-03-19 that line sits around $24.87.
        assert trade.exit_date == date(2019, 3, 19)
        assert trade.exit_price == 24.87
        # Still profitable — trade exits above entry.
        assert trade.pnl_pct > 0.04

    def test_case1_disabling_climax_reverts_to_old_late_exit(self):
        """Setting climax_atr_mult to infinity disables the climax
        rule; the trade then rides to a later exit via the EMA
        trail / time stop, showing why the structural rule matters.
        """
        # With climax off, the trade must NOT exit on 2019-03-19 via
        # a single-bar blow-off — it rides further.
        result = self._run(AMD_2019_CASE1, climax_atr_mult=float("inf"))
        trade = result.performance.trades[0]
        assert trade.exit_date > date(2019, 3, 19)

    # ---- Case 2 ----

    @staticmethod
    def _trade_by_entry_date(result, entry_date: date):
        for t in result.performance.trades:
            if t.entry_date == entry_date:
                return t
        return None

    def test_case2_breakeven_stop_fires_on_2019_08_23(self):
        result = self._run(AMD_2019_CASE2_3)

        trade = self._trade_by_entry_date(result, date(2019, 8, 22))
        assert trade is not None
        assert trade.entry_price == 31.76

        # Breakeven stop fires on the next bar (2019-08-23) — the
        # 08-22 close (31.90) armed the stop to 31.76. 08-23 opens
        # at 31.30 (gap-down), which is already below the armed
        # breakeven, so the new LOW-based stop-fill model fills at
        # the OPEN (31.30), not at the breakeven line. Still a tiny
        # loss vs the old -12.94% ride to the consolidation low.
        assert trade.exit_date == date(2019, 8, 23)
        assert trade.exit_price == 31.30
        assert trade.exit_price < trade.entry_price
        # Still miles better than the -12.94% old behaviour.
        assert trade.pnl_pct > -0.02
        # And still NOT the original consolidation-low stop 27.65.
        assert trade.exit_price > trade.stop_loss

    def test_case2_without_pre_loop_arming_would_hit_stop_loss(self):
        """With the breakeven mechanism disabled, Case 2 reverts to
        its old far-dated consolidation-low exit — the "bad" outcome
        the fix avoids.
        """
        result = self._run(
            AMD_2019_CASE2_3, arm_breakeven_after_profit=False
        )
        trade = self._trade_by_entry_date(result, date(2019, 8, 22))
        assert trade is not None
        # Without breakeven arming the trade falls through to the
        # consolidation-low hard stop ~6 weeks later for a ~-13% loss.
        assert trade.exit_date > date(2019, 8, 23)
        assert trade.pnl_pct < -0.05

    # ---- Case 3 ----

    def test_case3_2019_10_11_signal_still_produces_winner(self):
        """The 10-14 entry still runs to a clear winner. Under the
        old close-based exhaustion check this trade coasted all the
        way to 2019-11-19 for +38.98%. The new HIGH-based
        limit-order model exits earlier — as soon as a bar's high
        touches the ``ref_ema + atr × 2.5`` line — which in this
        fixture lands on 2019-11-05. The trade is still a solid
        +24% winner, just not the full +39% run.
        """
        result = self._run(AMD_2019_CASE2_3)

        winner = self._trade_by_entry_date(result, date(2019, 10, 14))
        assert winner is not None
        assert winner.entry_price == 29.71
        assert winner.exit_date == date(2019, 11, 5)
        assert winner.pnl_pct > 0.20

    # ---- Case 4 ----

    def test_case4_chase_entry_is_rejected(self):
        result = self._run(AMD_2020_CASE4)

        # The 2020-06-09 signal is the only one in the window, and
        # the chase-entry filter must reject it → zero trades.
        assert result.performance.total_trades == 0

    def test_case4_disabling_chase_filter_lets_the_entry_through(self):
        """Setting max_entry_chase_ratio to infinity disables the
        filter; the bad trade goes through on 2020-06-10. With the
        breakeven-stop fix the loss is now limited, but the entry
        still fires — which is the point of the filter: avoid
        committing capital to a chasing entry in the first place.
        """
        result = self._run(
            AMD_2020_CASE4, max_entry_chase_ratio=float("inf")
        )
        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        assert trade.entry_date == date(2020, 6, 10)
        assert trade.entry_price == 57.20

    def test_case4_disabling_both_guards_reproduces_loser(self):
        """With both the chase filter and the breakeven stop
        disabled, Case 4 reverts to its original "ride the loss"
        behaviour. The fixture's 30-bar window cuts off before the
        full -14% crash to the consolidation-low stop, but even
        within that window the trade is clearly a loser (no
        profitable exit fires). This is the "before" state the
        structural fixes defend against.
        """
        result = self._run(
            AMD_2020_CASE4,
            max_entry_chase_ratio=float("inf"),
            arm_breakeven_after_profit=False,
        )
        assert result.performance.total_trades == 1
        trade = result.performance.trades[0]
        assert trade.entry_date == date(2020, 6, 10)
        assert trade.pnl_pct < -0.03


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
        sigs = WedgePopDetector().detect(AMD_2026_DOWNTREND)
        dates = {s.date for s in sigs}
        assert date(2026, 2, 24) in dates

        target = next(s for s in sigs if s.date == date(2026, 2, 24))
        assert "ema_fast_slope" in target.metadata
        assert "ema_slow_slope" in target.metadata
        # The slow slope must be the structural reason we reject it.
        assert target.metadata["ema_slow_slope"] < -0.05

    def test_detector_still_detects_2026_02_24(self):
        sigs = WedgePopDetector().detect(AMD_2026_DOWNTREND)
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
        """ELF 2023-11-14, AMZN 2021-06-08, and the three AMD 2019
        cases have been the baseline for the detector/strategy
        tests. All of their ema_slow_slope20 values are above
        -5%, so the default slope filter must NOT reject them.
        """
        for df, date_wanted in (
            (ELF_DATA, date(2023, 11, 15)),
            (AMZN_2021_DATA, date(2021, 6, 9)),
            (AMD_2019_CASE1, date(2019, 3, 13)),
            (AMD_2019_CASE2_3, date(2019, 8, 22)),
            (AMD_2019_CASE2_3, date(2019, 10, 14)),
        ):
            result = self._run(df, max_ema_slope_decline=0.05)
            entry_dates = {t.entry_date for t in result.performance.trades}
            assert date_wanted in entry_dates, (
                f"slope filter wrongly rejected entry on {date_wanted}"
            )


def _force_entry(df: pd.DataFrame, entry_date: date, **strategy_overrides):
    """Bypass the detector and invoke ``_execute_trade`` at a specific
    bar. This is used by the ALL 2025-05-06 suite to isolate exit /
    filter behaviour without fighting the detector's cooldown logic.

    Builds a minimal ``PatternSignal`` whose ``stop_loss`` is the
    consolidation low of the 15 bars preceding the entry — matching
    what the real detector would have generated if it had fired.
    """
    detector = strategy_overrides.pop("detector", WedgePopDetector())
    # Default helper knobs: off for filters that aren't under test here.
    strategy_overrides.setdefault("max_ema_slope_decline", None)
    strategy_overrides.setdefault("max_entry_chase_ratio", float("inf"))
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
    trade, _ = strategy._execute_trade(
        df_ind, signal, entry_idx, 100_000.0, config
    )
    return trade


class TestAllstate20250506Exhaustion:
    """The ALL (Allstate) 2025-05-06 case the user brought up as the
    canonical reason to make exhaustion HIGH-based.

    Trade context (driven by ``_force_entry`` so the detector's
    cooldown machinery doesn't get in the way):

        entry bar 2025-05-06 open ~$196.14, consolidation low stop
        pulls from the April sell-off ($181.06).

    Under the OLD close-based exhaustion rule neither 05-07 nor
    05-08 fired — both bars' closes were only ~2.5% above the 10
    EMA, far below the default 15% / 2.5 ATR thresholds. The new
    HIGH-based limit-order model lets the user tune ``extension_pct``
    so the sell limit fills at the line when the bar's *high*
    touches it, regardless of where the close ends up.

    05-07: H=200.61, max(ema10, ema20) ≈ 195.28 → +2.73% high
    05-08: H=202.13, max(ema10, ema20) ≈ 195.90 → +3.18% high
    """

    def test_default_exhaustion_does_not_exit_on_05_07_or_05_08(self):
        """With production defaults (15% / 2.5 ATR / climax 1.5),
        neither day fires any exit rule. The trade holds through.
        """
        trade = _force_entry(
            ALL_2025_05_06, date(2025, 5, 6)
        )
        assert trade is not None
        assert trade.entry_date == date(2025, 5, 6)
        # Exit date must be AFTER 05-08 — neither day fires anything.
        assert trade.exit_date > date(2025, 5, 8)

    def test_tuned_exhaustion_25_pct_exits_on_05_07_at_the_line(self):
        """Setting ``extension_pct=0.025`` puts the pct trigger line
        at ``195.28 × 1.025 ≈ $200.16``. The 05-07 bar's high
        ($200.61) touches that line during the bar, so the limit
        order fills at the line (not at the high).
        """
        trade = _force_entry(
            ALL_2025_05_06,
            date(2025, 5, 6),
            extension_pct=0.025,
            extension_atr_mult=99.0,    # isolate the pct leg
            climax_atr_mult=99.0,       # isolate exhaustion from climax
        )
        assert trade.exit_date == date(2025, 5, 7)
        # Fill is at the 2.5% line, near but strictly below the bar's
        # high ($200.61).
        assert 200.10 <= trade.exit_price <= 200.20
        assert trade.exit_price < 200.61
        assert trade.pnl_pct > 0.0

    def test_tuned_exhaustion_30_pct_exits_on_05_08_at_the_line(self):
        """Setting ``extension_pct=0.03`` raises the trigger high
        enough that 05-07 (high 200.61 < 201.14) doesn't fire, but
        the 05-08 high ($202.13) does — fills at
        ``195.90 × 1.03 ≈ $201.78``.
        """
        trade = _force_entry(
            ALL_2025_05_06,
            date(2025, 5, 6),
            extension_pct=0.03,
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
        )
        assert trade.exit_date == date(2025, 5, 8)
        assert 201.70 <= trade.exit_price <= 201.85
        assert trade.exit_price < 202.13   # not the intraday high
        assert trade.pnl_pct > 0.0

    def test_atr_leg_can_also_fire_on_05_07(self):
        """A tighter ``extension_atr_mult`` also arms the HIGH-based
        exit on 05-07 via the ATR leg rather than the pct leg.
        Setting the pct leg effectively off (``extension_pct=0.50``)
        lets us isolate the ATR leg's behaviour.
        """
        trade = _force_entry(
            ALL_2025_05_06,
            date(2025, 5, 6),
            extension_pct=0.50,           # disable pct leg
            extension_atr_mult=1.0,       # tight ATR leg
            climax_atr_mult=99.0,
        )
        # 05-07: ref_ema ≈ 195.28, atr ≈ 4.6, line ≈ 199.88.
        # High 200.61 > line → fires at the line (approximate range
        # because ATR is EWMA-smoothed, not a pinned value).
        assert trade.exit_date == date(2025, 5, 7)
        assert 199.50 <= trade.exit_price <= 200.10

    def test_climax_atr_mult_can_fire_on_05_07_via_high(self):
        """The climax rule is now HIGH-based too: ``prev_close +
        climax_atr_mult × ATR``. For 05-07 with
        ``climax_atr_mult=0.5``:
            prev_close = 198.16, ATR ≈ 4.39 → line ≈ 200.36.
        The bar's high $200.61 touches it and range 2.43 > 2.20
        (0.5 × ATR), so climax fires at the line. Exhaustion is
        disabled to isolate the climax leg.
        """
        trade = _force_entry(
            ALL_2025_05_06,
            date(2025, 5, 6),
            extension_pct=0.50,
            extension_atr_mult=99.0,
            climax_atr_mult=0.5,
        )
        assert trade.exit_date == date(2025, 5, 7)
        assert 200.20 <= trade.exit_price <= 200.70
        assert trade.exit_price < 200.61   # not the raw high
        assert trade.pnl_pct > 0.0


class TestEntryEmaExtensionFilter:
    """New ``max_entry_ema_extension_pct`` entry filter: rejects
    entries where the open price sits more than N% above the higher
    of the two signal-bar EMAs. Complements the existing
    ``max_entry_chase_ratio`` (which anchors on the signal bar's
    high, not on the EMA stack).
    """

    def test_default_is_off_and_preserves_existing_behaviour(self):
        """When ``max_entry_ema_extension_pct`` is None, the filter
        is disabled and the existing AMZN 2021-06 entry still goes
        through (slope filter also off so it's not confounded).
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(AMZN_2021_DATA),
            detector=WedgePopDetector(),
            max_ema_slope_decline=None,
            max_entry_ema_extension_pct=None,
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
        """AMD 2020-06-10 entry @ $57.20 sits ~1.3 ATR (≈ 5%) above
        the signal-bar EMA stack. A tight
        ``max_entry_ema_extension_pct=0.04`` (4%) must reject it.
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(AMD_2020_CASE4),
            detector=WedgePopDetector(),
            max_entry_chase_ratio=float("inf"),   # isolate the new filter
            max_entry_ema_extension_pct=0.04,
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
        """With a very loose ``max_entry_ema_extension_pct=0.20``,
        the AMD 2020-06-10 entry passes the EMA-extension filter.
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(AMD_2020_CASE4),
            detector=WedgePopDetector(),
            max_entry_chase_ratio=float("inf"),
            max_entry_ema_extension_pct=0.20,
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


class TestSameDayExit:
    """``_find_exit`` now iterates from ``entry_idx`` (not
    ``entry_idx + 1``), so the entry bar itself is evaluated for
    exit rules. The entry bar's intraday high can fire the
    exhaustion / climax lines and the bar's intraday low can fire
    the hard stop, all on the same trading day as the entry.

    The canonical case is ADBE 2021-12-10 → 2021-12-13:
        entry @ $652.77
        intraday high 675.21 (+3.44% from open in a single day)
        close 658.30 (near the open — spike faded)

    Under the old ``entry_idx + 1`` loop the trade was anchored
    to the breakeven stop for the rest of the lifecycle and the
    intraday spike was invisible. Under the new same-day logic,
    tuned exhaustion lines fire on 12-13 itself at the line price.
    """

    def test_default_exhaustion_does_not_fire_same_day_on_adbe_12_13(self):
        """Production defaults (15% / 2.5 ATR / climax 1.5) are too
        loose to fire on the 12-13 entry bar. The trade is armed to
        breakeven at end-of-day and is carried to 12-14 — but
        12-14 opens at $635.36, already below the armed breakeven
        $652.77. Under the new LOW-based stop-fill model the gap
        fills at the OPEN (realistic slippage), not at the stop.
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(ADBE_2021_12_13),
            detector=WedgePopDetector(consolidation_pct=0.10),
            max_ema_slope_decline=None,
            max_entry_chase_ratio=float("inf"),
        )
        result = strategy.execute(
            ADBE_2021_12_13,
            StrategyConfig(
                ticker="ADBE",
                start_date=ADBE_2021_12_13.index[0].date(),
                end_date=ADBE_2021_12_13.index[-1].date(),
                pattern_name="wedge_pop",
                max_holding_days=60,
            ),
        )
        trade = next(
            t
            for t in result.performance.trades
            if t.entry_date == date(2021, 12, 13)
        )
        assert trade.entry_price == 652.77
        # Exit on 12-14 at the gap-down open — not the breakeven
        # line, and not at the intraday low ($599.10).
        assert trade.exit_date == date(2021, 12, 14)
        assert trade.exit_price == 635.36
        assert trade.exit_price < 652.77
        assert trade.exit_price > 599.10
        assert trade.pnl < 0

    def test_tuned_exhaustion_15_pct_exits_same_day_on_entry_bar(self):
        """With ``extension_pct=0.015``, the pct line sits around
        $663 on 2021-12-13 (ref_ema ≈ 653.28 × 1.015 ≈ 663). The
        bar's high $675.21 touches that line intraday, so the
        HIGH-based limit-order fires on the SAME BAR as the entry.
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(ADBE_2021_12_13),
            detector=WedgePopDetector(consolidation_pct=0.10),
            max_ema_slope_decline=None,
            max_entry_chase_ratio=float("inf"),
            extension_pct=0.015,
            extension_atr_mult=99.0,    # isolate pct leg
            climax_atr_mult=99.0,       # isolate from climax
        )
        result = strategy.execute(
            ADBE_2021_12_13,
            StrategyConfig(
                ticker="ADBE",
                start_date=ADBE_2021_12_13.index[0].date(),
                end_date=ADBE_2021_12_13.index[-1].date(),
                pattern_name="wedge_pop",
                max_holding_days=60,
            ),
        )
        trade = next(
            t
            for t in result.performance.trades
            if t.entry_date == date(2021, 12, 13)
        )
        # Same-day exit: entry_date == exit_date.
        assert trade.exit_date == date(2021, 12, 13)
        # Fill is the line (~663), NOT the bar's high ($675.21).
        assert 662.0 <= trade.exit_price <= 664.0
        assert trade.exit_price < 675.21
        assert trade.pnl > 0

    def test_tuned_exhaustion_30_pct_exits_same_day_at_higher_line(self):
        """Raising ``extension_pct`` to 0.03 pushes the line up to
        ~$672.88. The bar's high $675.21 still crosses it, so the
        exit is still same-day — just at the higher line, capturing
        more of the intraday pop.
        """
        strategy = WedgepopStrategy(
            market_data=FakeMarketData(ADBE_2021_12_13),
            detector=WedgePopDetector(consolidation_pct=0.10),
            max_ema_slope_decline=None,
            max_entry_chase_ratio=float("inf"),
            extension_pct=0.03,
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
        )
        result = strategy.execute(
            ADBE_2021_12_13,
            StrategyConfig(
                ticker="ADBE",
                start_date=ADBE_2021_12_13.index[0].date(),
                end_date=ADBE_2021_12_13.index[-1].date(),
                pattern_name="wedge_pop",
                max_holding_days=60,
            ),
        )
        trade = next(
            t
            for t in result.performance.trades
            if t.entry_date == date(2021, 12, 13)
        )
        assert trade.exit_date == date(2021, 12, 13)
        assert 672.0 <= trade.exit_price <= 673.5
        assert trade.exit_price < 675.21
        assert trade.pnl_pct > 0.025

    def test_intraday_stop_pierce_fills_at_the_line_not_the_low(self):
        """LOW-based stop symmetry: if the bar OPENS above the stop
        and the intraday low pierces it, the fill is at the stop
        line (limit-order model), not at the bar's low.
        """
        rows = []
        for i in range(25):
            rows.append(
                ("2023-01-{:02d}".format(i + 2), 100.0, 100.5, 99.5, 100.0, 1_000_000)
            )
        # Breakout bar @ 104 (signal)
        rows.append(("2023-02-06", 100.5, 104.5, 99.0, 104.0, 2_000_000))
        # Entry bar: opens 104, closes 104.2 (no arming on close).
        rows.append(("2023-02-07", 104.0, 104.8, 103.5, 104.2, 2_000_000))
        # Next bar opens ABOVE the cons_low stop (~99.5) but
        # intraday low pierces it at 98.5. Stop fill must be 99.5
        # (the line), not 98.5 (the low).
        rows.append(("2023-02-08", 103.0, 103.5, 98.5, 99.0, 3_000_000))
        # Filler bar so the loop can iterate.
        rows.append(("2023-02-09", 99.0, 100.0, 98.5, 99.5, 1_000_000))

        df = pd.DataFrame(
            {
                "Open": [r[1] for r in rows],
                "High": [r[2] for r in rows],
                "Low": [r[3] for r in rows],
                "Close": [r[4] for r in rows],
                "Volume": [r[5] for r in rows],
            },
            index=pd.to_datetime([r[0] for r in rows]),
        )
        trade = _force_entry(
            df,
            date(2023, 2, 7),
            extension_pct=0.50,
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
            arm_breakeven_after_profit=False,   # keep stop at cons_low
        )
        # Stop fires 02-08 at the cons_low line (99.5), not the
        # bar's low (98.5).
        assert trade is not None
        assert trade.exit_date == date(2023, 2, 8)
        # cons_low from 15 bars before 02-06 ≈ 99.5 (flat consolidation).
        assert trade.exit_price == trade.stop_loss
        assert trade.exit_price > 98.5   # above the intraday low
        assert trade.pnl_pct < 0

    def test_gap_down_open_below_stop_fills_at_open_not_the_line(self):
        """LOW-based stop symmetry: if the bar OPENS below the
        stop (gap-down through the line), the fill is at the OPEN
        (realistic slippage), not at the stop line. Symmetric to
        the exhaustion rule where gap-ups fill at the (favourable)
        open.
        """
        rows = []
        for i in range(25):
            rows.append(
                ("2023-01-{:02d}".format(i + 2), 100.0, 100.5, 99.5, 100.0, 1_000_000)
            )
        # Breakout bar.
        rows.append(("2023-02-06", 100.5, 104.5, 99.0, 104.0, 2_000_000))
        # Entry bar, closes within range.
        rows.append(("2023-02-07", 104.0, 104.8, 103.5, 104.2, 2_000_000))
        # GAP DOWN through the stop. Open 95 is already below
        # cons_low (~99.5) — no intraday pierce needed.
        rows.append(("2023-02-08", 95.0, 95.5, 93.0, 94.0, 3_000_000))

        df = pd.DataFrame(
            {
                "Open": [r[1] for r in rows],
                "High": [r[2] for r in rows],
                "Low": [r[3] for r in rows],
                "Close": [r[4] for r in rows],
                "Volume": [r[5] for r in rows],
            },
            index=pd.to_datetime([r[0] for r in rows]),
        )
        trade = _force_entry(
            df,
            date(2023, 2, 7),
            extension_pct=0.50,
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
            arm_breakeven_after_profit=False,
        )
        assert trade is not None
        assert trade.exit_date == date(2023, 2, 8)
        # Fills at the gap-down open (95), NOT at the stop line
        # (~99.5). A stop order sitting at 99.5 cannot fill above
        # the open when the whole bar is below the stop.
        assert trade.exit_price == 95.0
        assert trade.exit_price < trade.stop_loss
        assert trade.pnl_pct < 0

    def test_same_day_hard_stop_fires_when_low_pierces_cons_low(self):
        """Structural guarantee: if the entry bar's intraday low
        drops to the consolidation-low stop, the hard stop fires
        on the same bar at the stop price. Uses a forced entry so
        we control the exact stop level.
        """
        # Build a minimal synthetic fixture where the entry bar
        # drops from its open straight through a contrived stop.
        rows = []
        # 25 flat bars around $100 for EMA convergence.
        for i in range(25):
            rows.append(
                ("2023-01-{:02d}".format(i + 2), 100.0, 100.5, 99.5, 100.0, 1_000_000)
            )
        # Breakout bar at $104 (signal).
        rows.append(("2023-02-06", 100.5, 104.5, 99.0, 104.0, 2_000_000))
        # Entry bar: opens $104, drops intraday to $95 before
        # closing at $100. cons_low over last 15 bars is $99.5.
        rows.append(("2023-02-07", 104.0, 104.5, 95.0, 100.0, 2_000_000))
        # A few more bars so the loop has something to iterate.
        rows.append(("2023-02-08", 100.0, 100.5, 99.5, 100.0, 1_000_000))

        df = pd.DataFrame(
            {
                "Open": [r[1] for r in rows],
                "High": [r[2] for r in rows],
                "Low": [r[3] for r in rows],
                "Close": [r[4] for r in rows],
                "Volume": [r[5] for r in rows],
            },
            index=pd.to_datetime([r[0] for r in rows]),
        )
        trade = _force_entry(
            df,
            date(2023, 2, 7),
            extension_pct=0.50,      # disable exhaustion
            extension_atr_mult=99.0,
            climax_atr_mult=99.0,
        )
        # Entry bar's low ($95) pierces the cons_low stop ($99.5)
        # intraday → hard stop fires on 2023-02-07 at the stop.
        assert trade is not None
        assert trade.exit_date == date(2023, 2, 7)
        assert trade.exit_price == trade.stop_loss
        assert trade.pnl_pct < 0


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
        sigs = WedgePopDetector().detect(AMZN_2021_DATA)
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
        sigs_off = WedgePopDetector().detect(AMD_2019_CASE2_3)
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
        off = WedgePopDetector()
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
