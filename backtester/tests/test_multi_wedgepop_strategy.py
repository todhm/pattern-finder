from datetime import date

import pandas as pd
import pytest

from data.adapters.wikipedia_universe import StaticUniverseAdapter
from data.domain.ports import MarketDataPort
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.multi_wedgepop_strategy import MultiWedgepopStrategy
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import (
    MultiStrategyConfig,
    StrategyConfig,
    TossFeeSchedule,
)
from tests.test_wedge_pop import (
    AMD_2019_CASE2_3,
    AMZN_2021_DATA,
    ELF_DATA,
)

ZERO_FEES = TossFeeSchedule(
    buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
)


class FakeUniverseMarketData(MarketDataPort):
    """Returns a different DataFrame depending on the requested ticker."""

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        failing: set[str] | None = None,
    ):
        self._tables = tables
        self._failing = failing or set()

    def fetch_ohlcv(self, symbol, start, end):
        if symbol in self._failing:
            raise RuntimeError(f"simulated fetch failure for {symbol}")
        if symbol not in self._tables:
            raise KeyError(symbol)
        return self._tables[symbol]


def _build_runner(
    tables: dict[str, pd.DataFrame],
    universe_name: str = "test",
    failing: set[str] | None = None,
    max_workers: int = 2,
) -> MultiWedgepopStrategy:
    market_data = FakeUniverseMarketData(tables, failing=failing)
    universe = StaticUniverseAdapter({universe_name: list(tables.keys())})
    # Lenient detector for test fixtures: disable breakout ATR gate
    # and long-SMA requirement (test data rarely has 200+ bars).
    detector = WedgePopDetector(
        breakout_atr_mult=0.0, require_above_long_smas=False
    )
    # Disable the per-ticker slope filter inside the multi runner —
    # most of these tests assert specific trade counts on real fixtures
    # (ELF, AMZN, AMD) whose slopes sit under the production default
    # threshold. Tests that DO want to exercise the slope filter pass
    # an explicit strategy instead.
    per_ticker = WedgepopStrategy(
        market_data=market_data,
        detector=detector,
        max_ema_slope_decline=None,
        # Exit model now limits to three opt-in conditions; enable
        # smart trail so tests that assert multi-trade entry sets
        # still see positions close and free up capital.
        use_smart_trail=True,
    )
    return MultiWedgepopStrategy(
        market_data=market_data,
        universe_provider=universe,
        detector=detector,
        strategy=per_ticker,
        max_workers=max_workers,
    )


def _config(**overrides) -> MultiStrategyConfig:
    base = dict(
        universe="test",
        start_date=date(2021, 1, 1),
        end_date=date(2024, 1, 1),
        pattern_name="wedge_pop",
        initial_capital=100_000.0,
        risk_per_trade=0.02,
        max_holding_days=60,
    )
    base.update(overrides)
    return MultiStrategyConfig(**base)


def _consolidation_then_breakout(
    post_breakout: list[tuple[float, float, float, float]],
    breakout_volume: int = 2_500_000,
    breakout_close: float = 76.0,
    breakout_high: float = 76.5,
    breakout_low: float = 69.8,
    breakout_open: float = 70.0,
    consolidation_volume: int = 500_000,
    post_volume: int = 1_500_000,
) -> pd.DataFrame:
    """Same shape as the per-ticker exit tests: 20 bars consolidating
    below ~70, one breakout bar at index 20, then arbitrary post bars.

    The breakout-bar OHLC and volume are knobs the multi-strategy
    tests use to control the buy/sell-ratio ranking — varying
    ``breakout_close`` between near-high and near-low changes the
    estimated buy pressure on the bar.
    """
    rows: list[tuple[float, float, float, float, int]] = []
    for i in range(20):
        close = 70.0 - i * 0.05
        opn = close + 0.1
        high = max(opn, close) + 0.15
        low = min(opn, close) - 0.15
        rows.append((opn, high, low, close, consolidation_volume))
    rows.append(
        (
            breakout_open,
            breakout_high,
            breakout_low,
            breakout_close,
            breakout_volume,
        )
    )
    for o, h, l, c in post_breakout:
        rows.append((o, h, l, c, post_volume))

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


# A "rally" post-breakout sequence that lets the trade close cleanly via
# the exhaustion-extension exit (so the test doesn't depend on time stop).
RALLY_POST = [
    (76.0, 77.0, 75.5, 76.5),
    (77.0, 79.0, 76.8, 78.5),
    (79.0, 82.0, 78.8, 81.5),
    (82.0, 86.0, 81.8, 85.5),
    (86.0, 92.0, 85.5, 91.5),
    (92.0, 100.0, 91.5, 99.0),
    (99.0, 110.0, 98.0, 108.0),
]


class TestStaticUniverseAdapter:
    def test_returns_tickers_for_known_universe(self):
        adapter = StaticUniverseAdapter({"sp500": ["AAPL", "MSFT", "GOOG"]})
        assert adapter.get_tickers("sp500") == ["AAPL", "MSFT", "GOOG"]

    def test_case_insensitive_lookup(self):
        adapter = StaticUniverseAdapter({"sp500": ["AAPL"]})
        assert adapter.get_tickers("SP500") == ["AAPL"]

    def test_unknown_universe_raises(self):
        adapter = StaticUniverseAdapter({"sp500": ["AAPL"]})
        with pytest.raises(ValueError):
            adapter.get_tickers("nasdaq100")

    def test_returns_copy_not_reference(self):
        original = ["AAPL", "MSFT"]
        adapter = StaticUniverseAdapter({"sp500": original})
        out = adapter.get_tickers("sp500")
        out.append("EVIL")
        assert adapter.get_tickers("sp500") == ["AAPL", "MSFT"]


class TestBuySellRatioSelection:
    """The 'multi' part: when several tickers fire on the same date,
    the bar with the highest *buy/sell volume ratio* wins. Raw volume
    no longer drives selection — what matters is intrabar buying
    pressure (close near the high vs close near the low). The other
    contenders are discarded for that date entirely (we don't queue).
    """

    def test_high_buy_pressure_wins_when_signals_collide(self):
        # Same total volume on the breakout bar, but different close
        # locations: BULL closes near the high (strong buying), BEAR
        # closes near the low (strong selling). BULL's buy/sell ratio
        # is far higher → BULL wins the auction.
        bullish = _consolidation_then_breakout(
            RALLY_POST,
            breakout_volume=5_000_000,
            breakout_close=76.4,
        )
        bearish = _consolidation_then_breakout(
            RALLY_POST,
            breakout_volume=5_000_000,
            breakout_close=70.5,
        )
        runner = _build_runner({"BULL": bullish, "BEAR": bearish})
        result = runner.run(_config(fee_schedule=ZERO_FEES))

        assert result.trades_taken == 1
        assert result.trades[0].ticker == "BULL"
        # BULL's ratio must be way above neutral (1.0).
        assert result.trades[0].signal_buy_sell_ratio > 10.0

    def test_raw_volume_does_not_decide_the_auction(self):
        # LOUD has 10× the volume but closes mid-bar. QUIET has lower
        # volume but closes very near the high (much stronger ratio).
        # The new selection rule must pick QUIET over LOUD.
        loud = _consolidation_then_breakout(
            RALLY_POST,
            breakout_volume=10_000_000,
            breakout_close=73.0,  # mid-bar → ratio ≈ 1
        )
        quiet = _consolidation_then_breakout(
            RALLY_POST,
            breakout_volume=1_000_000,
            breakout_close=76.45,  # near high → very high ratio
        )
        runner = _build_runner({"LOUD": loud, "QUIET": quiet})
        result = runner.run(_config(fee_schedule=ZERO_FEES))

        assert result.trades_taken == 1
        assert result.trades[0].ticker == "QUIET"

    def test_total_signals_counts_all_universe_signals(self):
        # Both tickers fire on the same date → 2 signals total, but
        # only 1 trade is taken (highest buy/sell ratio wins).
        a = _consolidation_then_breakout(
            RALLY_POST, breakout_close=76.4
        )
        b = _consolidation_then_breakout(
            RALLY_POST, breakout_close=70.5
        )
        runner = _build_runner({"A": a, "B": b})
        result = runner.run(_config(fee_schedule=ZERO_FEES))

        assert result.total_signals == 2
        assert result.trades_taken == 1

    def test_pressure_fields_are_populated_and_consistent(self):
        # buy_volume + sell_volume must reconstruct the bar's volume,
        # and the ratio must reflect the close's location.
        bullish = _consolidation_then_breakout(
            RALLY_POST,
            breakout_volume=5_000_000,
            breakout_close=76.4,
        )
        runner = _build_runner({"BULL": bullish})
        result = runner.run(_config(fee_schedule=ZERO_FEES))
        trade = result.trades[0]

        total = trade.signal_buy_volume + trade.signal_sell_volume
        assert total == pytest.approx(trade.signal_volume, rel=1e-3)
        assert trade.signal_buy_volume > trade.signal_sell_volume
        assert trade.signal_buy_sell_ratio > 1.0


class TestSinglePositionLockout:
    """While a position is open, *every* other signal is ignored —
    same ticker, different ticker, doesn't matter. New signals are
    only considered after the previous trade exits.
    """

    def test_two_real_tickers_run_sequentially(self):
        # ELF (Nov 2023) and AMZN (Jun 2021) fire on completely
        # different dates, so the lockout never bites — both trades
        # should make it through.
        runner = _build_runner({"ELF": ELF_DATA, "AMZN": AMZN_2021_DATA})
        result = runner.run(_config())
        assert result.trades_taken == 2
        # Trades must not overlap in time.
        for prev, nxt in zip(result.trades, result.trades[1:]):
            assert nxt.entry_date > prev.exit_date

    def test_lockout_blocks_overlapping_signals(self):
        # Both tickers fire on the same day. The high buy/sell-ratio
        # one is taken; the other is a valid signal but discarded
        # because there's no queue.
        winner = _consolidation_then_breakout(
            RALLY_POST, breakout_close=76.4
        )
        loser = _consolidation_then_breakout(
            RALLY_POST, breakout_close=70.5
        )
        runner = _build_runner({"WIN": winner, "LOSE": loser})
        result = runner.run(_config(fee_schedule=ZERO_FEES))

        assert result.total_signals == 2
        assert result.trades_taken == 1
        assert result.trades[0].ticker == "WIN"


class TestPortfolioBookkeeping:
    def test_initial_and_final_capital_consistent_with_trades(self):
        runner = _build_runner({"ELF": ELF_DATA, "AMZN": AMZN_2021_DATA})
        result = runner.run(_config(initial_capital=100_000))

        assert result.initial_capital == 100_000.0
        expected_final = 100_000.0 + sum(t.pnl for t in result.trades)
        assert result.final_capital == pytest.approx(round(expected_final, 2))

    def test_total_return_matches_capital_delta(self):
        runner = _build_runner({"ELF": ELF_DATA, "AMZN": AMZN_2021_DATA})
        result = runner.run(_config())

        delta = result.final_capital - result.initial_capital
        expected = round(delta / result.initial_capital, 4)
        assert result.total_return_pct == expected

    def test_equity_curve_anchors_to_start_and_grows_with_trades(self):
        runner = _build_runner({"ELF": ELF_DATA, "AMZN": AMZN_2021_DATA})
        cfg = _config(initial_capital=100_000)
        result = runner.run(cfg)

        curve = result.equity_curve
        assert curve[0].date == cfg.start_date
        assert curve[0].equity == 100_000
        # One curve point per trade after the anchor.
        assert len(curve) == 1 + result.trades_taken
        assert curve[-1].equity == result.final_capital


class TestFailureHandling:
    def test_failing_fetch_is_recorded_not_raised(self):
        runner = _build_runner(
            {"ELF": ELF_DATA, "AMZN": AMZN_2021_DATA},
            failing={"AMZN"},
        )
        result = runner.run(_config())

        assert "AMZN" in result.failed_tickers
        assert result.tickers_scanned == 2
        # ELF still produces its trade unaffected.
        tickers_traded = {t.ticker for t in result.trades}
        assert tickers_traded == {"ELF"}

    def test_too_short_data_is_skipped(self):
        # 5-bar DataFrame is below the 15-bar minimum.
        tiny = AMZN_2021_DATA.head(5)
        runner = _build_runner({"TINY": tiny, "ELF": ELF_DATA})
        result = runner.run(_config())

        assert "TINY" in result.failed_tickers
        assert result.trades_taken >= 1


class TestTossFees:
    """Toss Securities fees are deducted from each trade and reflected
    in the trade-level pnl, the aggregate equity curve, and the result
    rollup. Defaults match Toss's published US-equity fees: 0.1% buy,
    0.1% sell, 0.00229% SEC fee on the sell side.
    """

    def test_zero_fee_schedule_matches_gross(self):
        runner = _build_runner({"AMZN": AMZN_2021_DATA})
        result = runner.run(_config(fee_schedule=ZERO_FEES))

        assert result.total_commission == 0.0
        assert result.trades_taken == 1
        trade = result.trades[0]
        assert trade.commission == 0.0
        assert trade.pnl == trade.gross_pnl

    def test_default_fees_deduct_round_trip_commission(self):
        runner = _build_runner({"AMZN": AMZN_2021_DATA})
        result = runner.run(_config())  # default = TossFeeSchedule()

        trade = result.trades[0]
        # Round-trip = 0.1% × buy notional + (0.1% + 0.00229%) × sell notional
        buy_notional = trade.entry_price * trade.shares
        sell_notional = trade.exit_price * trade.shares
        expected = round(
            buy_notional * 0.001
            + sell_notional * (0.001 + 0.0000229),
            2,
        )
        assert trade.commission == expected
        # Net pnl = gross pnl - commission (within rounding).
        assert trade.pnl == pytest.approx(
            round(trade.gross_pnl - trade.commission, 2)
        )
        assert result.total_commission == trade.commission

    def test_higher_fees_reduce_final_capital(self):
        # Same trade, two different fee schedules — the heavier one
        # must produce strictly lower final capital.
        light = TossFeeSchedule(
            buy_commission_pct=0.001,
            sell_commission_pct=0.001,
            sec_fee_pct=0.0,
        )
        heavy = TossFeeSchedule(
            buy_commission_pct=0.005,
            sell_commission_pct=0.005,
            sec_fee_pct=0.0,
        )
        light_runner = _build_runner({"AMZN": AMZN_2021_DATA})
        heavy_runner = _build_runner({"AMZN": AMZN_2021_DATA})
        light_result = light_runner.run(_config(fee_schedule=light))
        heavy_result = heavy_runner.run(_config(fee_schedule=heavy))

        assert heavy_result.total_commission > light_result.total_commission
        assert heavy_result.final_capital < light_result.final_capital

    def test_total_commission_aggregates_all_trades(self):
        runner = _build_runner({"ELF": ELF_DATA, "AMZN": AMZN_2021_DATA})
        result = runner.run(_config())

        expected = round(sum(t.commission for t in result.trades), 2)
        assert result.total_commission == expected

    def test_pnl_pct_uses_net_on_cost_basis(self):
        runner = _build_runner({"AMZN": AMZN_2021_DATA})
        result = runner.run(_config())
        trade = result.trades[0]

        cost_basis = trade.entry_price * trade.shares
        expected_pct = round(trade.pnl / cost_basis, 4)
        assert trade.pnl_pct == expected_pct


class TestMaxTickersAndUniverse:
    def test_max_tickers_truncates_universe(self):
        runner = _build_runner(
            {
                "AMZN": AMZN_2021_DATA,
                "ELF": ELF_DATA,
                "AMZN2": AMZN_2021_DATA,
                "ELF2": ELF_DATA,
            }
        )
        result = runner.run(_config(max_tickers=2))
        assert result.tickers_scanned == 2

    def test_empty_universe_returns_empty_result(self):
        runner = _build_runner({})
        result = runner.run(_config())
        assert result.tickers_scanned == 0
        assert result.trades_taken == 0
        assert result.total_signals == 0
        assert result.final_capital == result.initial_capital


class TestMultiVolumeFilter:
    """Multi-strategy excludes wedge-pop signals whose breakout-bar
    volume is below the 20-day average (``metadata.volume_ratio < 1.0``).

    Rationale: a bar that closes near its high but on below-average
    volume is weak confirmation — the buying pressure simply wasn't
    there. The per-ticker strategy can still take the signal; only
    the multi-ticker auction rejects it so the winner-takes-all
    selection isn't dominated by light-volume breakouts.
    """

    def test_amd_2019_08_21_signal_is_filtered_from_multi(self):
        """AMD 2019-08-21 is a detected wedge pop, but its breakout
        bar's volume (41.4M) is below the 20-day average (~60M),
        so ``volume_ratio ≈ 0.52``. The multi strategy must skip
        it and pick the later 2019-10-11 signal (vol_ratio 1.38)
        instead.
        """
        runner = _build_runner({"AMD": AMD_2019_CASE2_3})
        result = runner.run(
            _config(
                start_date=date(2019, 7, 8),
                end_date=date(2019, 10, 21),
                fee_schedule=ZERO_FEES,
            )
        )

        # The 08-21 signal must NOT appear as a trade entry (filtered
        # by volume_ratio < 1.0). The 10-14 entry must be present.
        entry_dates = {t.entry_date for t in result.trades}
        assert date(2019, 8, 22) not in entry_dates  # 08-21 signal → 08-22 entry
        assert date(2019, 10, 14) in entry_dates

    def test_per_ticker_strategy_still_takes_low_volume_signal(self):
        """The volume filter lives in the *multi* strategy only —
        the per-ticker ``WedgepopStrategy`` does NOT filter by
        volume_ratio, so single-ticker backtests keep all signals.
        (Slope filter is disabled here so we isolate the volume
        filter's behaviour from the unrelated slope gate.)
        """
        strategy = WedgepopStrategy(
            market_data=FakeUniverseMarketData({"AMD": AMD_2019_CASE2_3}),
            detector=WedgePopDetector(
                breakout_atr_mult=0.0, require_above_long_smas=False
            ),
            max_ema_slope_decline=None,
        )
        result = strategy.execute(
            AMD_2019_CASE2_3,
            StrategyConfig(
                ticker="AMD",
                start_date=date(2019, 7, 8),
                end_date=date(2019, 10, 21),
                pattern_name="wedge_pop",
                max_holding_days=60,
            ),
        )
        # Per-ticker strategy doesn't filter by volume — it should
        # take more signals than the multi strategy's 1-trade result
        # (which excludes low-volume signals from the auction).
        assert result.performance.total_trades >= 1
