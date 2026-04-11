from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig
from tests.test_wedge_pop import AMZN_2021_DATA, ELF_DATA


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
        post = [
            (76.0, 76.5, 75.5, 76.2),  # entry bar (idx 21)
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

    def test_time_stop_exit_at_max_holding(self):
        # Post-breakout: drifts sideways above the EMA → no other exit fires.
        post = [(76.0, 76.5, 75.6, 76.1) for _ in range(8)]
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
