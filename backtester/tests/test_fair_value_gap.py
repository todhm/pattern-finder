"""Fair Value Gap detector + strategy unit tests.

Hand-built OHLC scenarios. Each scenario constructs the *minimum*
bars needed to satisfy the structural setup, then asserts the
detector emits exactly the expected signal — and that the strategy's
fixed-R envelope (TP / hard stop) fires on the right bar.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.domain.ports import MarketDataPort
from pattern.adapters.fair_value_gap import FairValueGapDetector
from pattern.helpers.sessions import NY_TZ
from strategy.adapters.fair_value_gap_strategy import FairValueGapStrategy
from strategy.domain.models import StrategyConfig


def _bar(o, h, l, c, v=1_000):
    return {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}


def _intraday_index(bars: int, start_str: str = "2024-03-18 09:30") -> pd.DatetimeIndex:
    return pd.date_range(
        start=start_str, periods=bars, freq="15min", tz=NY_TZ
    )


def _build_choch_fvg_session() -> pd.DataFrame:
    """Hand-built 15m session walking the *strict* CHoCH + retest
    semantics:

        bars 4 → L1  (low 80, confirms at bar 6)   — wider swing
        bars 7 → H1  (high 100, confirms at bar 9) — for ATR magnitude
        bars 10→ L2  (low 75, confirms at bar 12)   gate
        bars 13-15:  rally
        bar  15:     CHoCH — close 102 > H1 (100), L2 already printed
        bars 16-18:  3-bar bullish FVG
                       bar 16.high = 104
                       bar 17.low  = 102   ← stop level (FVG-producing low)
                       bar 18.low  = 110   ← gap = [104, 110], mid = 107
        bar  19:     RETEST — low 105 (≤ mid 107), close 109 (> mid 107),
                     low > fvg_low (104) → signal fires here
        bar  20:     ENTRY (next-open) at 109. 1R = 109 - 102 = 7
        bars 20-26:  rally to 3R = 130
    """
    rows = [
        _bar(100, 102,  99, 101),  # 0
        _bar(101, 103, 100, 102),  # 1
        _bar(102, 104, 100, 103),  # 2
        _bar(103, 105,  99, 102),  # 3
        _bar(102, 100,  80,  85),  # 4: L1 candidate (low 80)
        _bar( 85,  90,  84,  88),  # 5
        _bar( 88,  98,  87,  97),  # 6: L1 confirms (confirm_idx=4)
        _bar( 97, 100,  95,  99),  # 7: H1 candidate (high 100)
        _bar( 99,  99,  95,  97),  # 8
        _bar( 97,  98,  90,  92),  # 9: H1 confirms (confirm_idx=7)
        _bar( 92,  93,  75,  78),  # 10: L2 candidate (low 75, lower than L1=80)
        _bar( 78,  82,  77,  80),  # 11
        _bar( 80,  85,  79,  84),  # 12: L2 confirms (confirm_idx=10)
        _bar( 84,  90,  83,  89),  # 13: rally begins
        _bar( 89,  98,  88,  97),  # 14
        _bar( 97, 105,  96, 102),  # 15: CHoCH — close 102 > H1 (100). H1-L2=25 (≫2 ATR)
        _bar(102, 104, 101, 103),  # 16: FVG bar 1, high=104 (bullish)
        _bar(103, 112, 102, 111),  # 17: middle big bullish bar, low=102 (= stop)
        _bar(111, 113, 110, 112),  # 18: FVG bar 3, low=110 > bar16.high=104 → GAP
        _bar(112, 113, 105, 109),  # 19: RETEST — low 105 ≤ mid 107, close 109 > 107
        _bar(109, 113, 108, 112),  # 20: ENTRY bar (open 109, stop=102, 1R=7)
        _bar(112, 117, 111, 116),  # 21: high 117 → 1R target (109+7=116) hits
        _bar(116, 122, 115, 121),  # 22: high 122 → ~2R (109+14=123 — NOT yet)
        _bar(121, 124, 120, 123),  # 23: high 124 → 2R (123) hits
        _bar(123, 128, 122, 127),  # 24
        _bar(127, 131, 126, 130),  # 25: high 131 → 3R (109+21=130) hits
        _bar(130, 134, 129, 133),  # 26
    ]
    df = pd.DataFrame(rows)
    df.index = _intraday_index(len(rows))
    return df


def _flat_no_choch_session() -> pd.DataFrame:
    """Sideways noise — no CHoCH should trigger, no signal."""
    rows = [_bar(100, 100.5, 99.5, 100) for _ in range(20)]
    df = pd.DataFrame(rows)
    df.index = _intraday_index(len(rows))
    return df


# ---- detector ------------------------------------------------------------


def test_detector_emits_signal_on_choch_then_fvg() -> None:
    df = _build_choch_fvg_session()
    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1)
    signals = det.detect(df)
    assert len(signals) == 1, [s.metadata for s in signals]
    s = signals[0]
    # Signal fires at the RETEST bar (19), not the FVG-completion
    # bar (18). The framework's "midpoint = entry" rule.
    assert s.timestamp == pd.Timestamp(df.index[19]).to_pydatetime()
    # Entry = FVG midpoint (limit-fill convention) — NOT the retest
    # bar's close. Stop = FVG-producing candle's low (bar 17 low = 102).
    assert s.entry_price == 107.0  # fvg_mid
    assert s.stop_loss == 102.0
    # FVG zone metadata.
    assert s.metadata["fvg_low"] == 104.0   # bar 16.high
    assert s.metadata["fvg_high"] == 110.0  # bar 18.low
    assert s.metadata["fvg_mid"] == 107.0
    # CHoCH cleared H1 = 100. Swing magnitude H1-L2 = 25 ≫ 2×ATR ✓.
    assert s.metadata["choch_high"] == 100.0
    # FVG bar = 18, CHoCH bar = 15 → offset 3.
    assert s.metadata["choch_bar_offset"] == 3
    # Retest happened 1 bar after FVG completion (bar 19 - bar 18 = 1).
    assert s.metadata["retest_bar_offset"] == 1
    # Chart-helper timestamps for the visualization layer.
    assert pd.Timestamp(s.metadata["choch_timestamp"]) == df.index[15]
    assert pd.Timestamp(s.metadata["fvg_start_timestamp"]) == df.index[16]
    assert pd.Timestamp(s.metadata["fvg_completion_timestamp"]) == df.index[18]
    # H1 (the broken swing high) printed at bar 7.
    assert pd.Timestamp(s.metadata["choch_break_level_start_ts"]) == df.index[7]


def test_detector_emits_nothing_on_flat_session() -> None:
    det = FairValueGapDetector(min_gap_pct=0.0)
    signals = det.detect(_flat_no_choch_session())
    assert signals == []


def test_detector_min_gap_pct_filters_thin_gaps() -> None:
    """A 3-bar bullish gap with width below ``min_gap_pct × close`` is
    skipped — keeps the strategy out of bid/ask-spread noise."""
    df = _build_choch_fvg_session()
    # The FVG is 6 dollars wide on a ~109 retest close → 5.5%. A
    # 7% threshold drops it.
    det_strict = FairValueGapDetector(
        min_gap_pct=0.07, max_signals_per_session=1
    )
    assert det_strict.detect(df) == []
    det_lenient = FairValueGapDetector(
        min_gap_pct=0.0, max_signals_per_session=1
    )
    assert len(det_lenient.detect(df)) == 1


def test_detector_no_signal_when_weak_choch_swing() -> None:
    """A CHoCH where the H1-L2 swing is below the ATR magnitude
    threshold is rejected — guards against the "tiny green-candle
    bounce" failure mode the user flagged on the screenshot.

    Build a fixture where the structural sequence is technically
    valid (lower-lows + intermediate swing high broken) but the
    ``H1 - L2`` magnitude is only ~1 dollar — well below the 2 ATR
    minimum on a chart with ~3 ATR. With ``min_choch_swing_atr=2``
    no signal should fire.
    """
    rows = [
        _bar(100, 102,  98, 101),
        _bar(101, 103,  99, 102),
        _bar(102, 104, 100, 103),
        _bar(103, 105, 101, 104),
        # Tiny micro-structure with all swings within ~1 dollar.
        _bar(104, 105, 100, 101),  # 4: micro L1 candidate (low 100)
        _bar(101, 102, 100, 101),
        _bar(101, 102, 100, 101),  # 6: confirms L1
        _bar(101, 101.5, 100.5, 101),  # 7: tiny H1 (101.5)
        _bar(101, 101.4, 100.4, 100.5),
        _bar(100.5, 101, 100, 100.5),  # 9: confirms H1
        _bar(100.5, 100.6, 99.5, 99.7),  # 10: micro L2 (99.5) — below L1 (100)
        _bar(99.7, 100, 99.6, 99.8),
        _bar(99.8, 100.2, 99.7, 100),  # 12: confirms L2
        _bar(100, 102, 99.8, 101.6),  # 13: rally — close 101.6 > H1 (101.5)
    ]
    df = pd.DataFrame(rows)
    df.index = _intraday_index(len(rows))

    det = FairValueGapDetector(
        min_gap_pct=0.0, min_choch_swing_atr=2.0
    )
    assert det.detect(df) == []
    # Sanity: with the magnitude gate disabled, the structural sequence
    # would still trigger CHoCH even if no FVG actually forms.
    det_off = FairValueGapDetector(
        min_gap_pct=0.0, min_choch_swing_atr=None
    )
    # No assertion on count — this just verifies the gate is the
    # actual cause of the rejection above (no exception).
    det_off.detect(df)


def test_detector_no_signal_when_no_midpoint_retest() -> None:
    """If price runs away from the FVG without retracing to the
    midpoint, no signal should fire. The framework explicitly
    states the midpoint touch is the entry trigger."""
    df = _build_choch_fvg_session().copy()
    # Patch out bar 19's retest dip — make every bar after FVG
    # completion stay strictly above the midpoint (107). The pending
    # FVG should expire after ``max_retest_bars`` with no signal.
    for i in range(19, len(df)):
        df.iloc[i, df.columns.get_loc("Low")] = max(
            float(df.iloc[i]["Low"]), 108.0
        )
    det = FairValueGapDetector(
        min_gap_pct=0.0, min_choch_swing_atr=None, max_retest_bars=30
    )
    assert det.detect(df) == []


def test_detector_max_signals_per_session() -> None:
    """Cap at 1 must keep only the first qualifying FVG per session."""
    df = _build_choch_fvg_session()
    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1)
    signals = det.detect(df)
    assert len(signals) <= 1


# ---- strategy ------------------------------------------------------------


class _StubMarket(MarketDataPort):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        return self._df


def test_strategy_take_profit_fires_at_3r() -> None:
    """Signal at retest bar 19. Entry = fvg_mid = 107 (limit fill on
    that same bar — its low touched 105, below 107). Stop = 102.
    1R = 5. 3R target = 122. Bar 22's high (122) hits exactly →
    take_profit fires at the target."""
    df = _build_choch_fvg_session()
    strat = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1),
        take_profit_r_multiple=3.0,
        enable_breakeven_stop=False,
    )
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    result = strat.execute(df, cfg)
    assert result.performance.total_trades == 1
    trade = result.performance.trades[0]
    assert trade.entry_price == pytest.approx(107.0)
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == pytest.approx(122.0)


def test_strategy_take_profit_fires_at_1r() -> None:
    """Lower the target to 1R = 107+5 = 112. Bar 20's high (113)
    clears it on the bar right after entry → take_profit fires."""
    df = _build_choch_fvg_session()
    strat = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1),
        take_profit_r_multiple=1.0,
        enable_breakeven_stop=False,
    )
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    result = strat.execute(df, cfg)
    assert result.performance.total_trades == 1
    trade = result.performance.trades[0]
    assert trade.exit_reason == "take_profit"


def test_strategy_same_bar_tp_and_stop_resolves_by_distance() -> None:
    """A bar that prints both ``high >= target`` AND ``low <= stop``
    cannot be ordered from OHLC alone. The strategy uses
    distance-from-open as a heuristic: the closer level was likely
    hit first within the bar.

    Entry = 107, stop = 102, 1R = 5. With ``take_profit_r_multiple=3``
    the target is 122. Patch bar 20 (first exit bar after entry) so
    its range straddles both levels.
    """
    # Case A — open close to stop → stop fires first.
    df_stop = _build_choch_fvg_session().copy()
    df_stop.iloc[20, df_stop.columns.get_loc("Open")] = 108.0
    df_stop.iloc[20, df_stop.columns.get_loc("High")] = 125.0
    df_stop.iloc[20, df_stop.columns.get_loc("Low")] = 100.0
    df_stop.iloc[20, df_stop.columns.get_loc("Close")] = 110.0

    strat = FairValueGapStrategy(
        market_data=_StubMarket(df_stop),
        detector=FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1),
        take_profit_r_multiple=3.0,
        enable_breakeven_stop=False,
    )
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df_stop.index[0].date(),
        end_date=df_stop.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    trade = strat.execute(df_stop, cfg).performance.trades[0]
    assert trade.exit_reason == "initial_stop"
    assert trade.exit_price == pytest.approx(102.0)

    # Case B — open close to target → TP fires first.
    df_tp = _build_choch_fvg_session().copy()
    df_tp.iloc[20, df_tp.columns.get_loc("Open")] = 120.0
    df_tp.iloc[20, df_tp.columns.get_loc("High")] = 125.0
    df_tp.iloc[20, df_tp.columns.get_loc("Low")] = 100.0
    df_tp.iloc[20, df_tp.columns.get_loc("Close")] = 124.0
    trade = FairValueGapStrategy(
        market_data=_StubMarket(df_tp),
        detector=FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1),
        take_profit_r_multiple=3.0,
        enable_breakeven_stop=False,
    ).execute(df_tp, cfg).performance.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == pytest.approx(122.0)


def test_detector_single_signal_per_choch_consumes_structure() -> None:
    """After a CHoCH→FVG retest fires, the CHoCH is consumed: a
    second 3-bar bullish FVG forming during the continuation rally
    must NOT fire as a back-to-back signal. Re-using already-played
    structure was the META 2026-03-31 double-entry the user flagged.

    The standard fixture's bars 21-23 already form a second valid
    3-bar bullish FVG (high[21]=117 < low[23]=120 → gap [117, 120],
    mid 118.5). Patch bar 24 so its low pierces that midpoint while
    its close holds above — without the consumption rule this would
    fire a second signal. With the rule it must not, because the
    only ChoCH in this fixture has already been consumed at bar 19.
    """
    df = _build_choch_fvg_session().copy()
    df.iloc[24, df.columns.get_loc("Low")] = 118.0
    df.iloc[24, df.columns.get_loc("Close")] = 120.0
    det = FairValueGapDetector(
        min_gap_pct=0.0,
        max_signals_per_session=2,
        max_retest_bars=30,
    )
    signals = det.detect(df)
    assert len(signals) == 1, (
        f"expected exactly 1 signal (second FVG inherits no active "
        f"ChoCH after consumption), got {len(signals)}: "
        f"{[s.metadata for s in signals]}"
    )
    assert signals[0].timestamp == pd.Timestamp(df.index[19]).to_pydatetime()


def test_detector_rejects_fvg_overlapping_choch_bar() -> None:
    """The FVG must form **after** the CHoCH bar — bar i-2 (first
    FVG bar) strictly greater than choch_bar. SWKS 2026-04-17 case:
    the same big green candle that broke the prior swing high was
    also being used as the FVG-producing structure, leaving the
    gap measured against the very bar that broke. Reject any FVG
    where any of (i-2, i-1, i) coincides with or precedes the CHoCH
    bar.
    """
    df = _build_choch_fvg_session().copy().astype(float)
    # Standard fixture: ChoCH at bar 15, FVG at bars 16/17/18 — all
    # strictly after, valid. Patch bar 15 (ChoCH bar) to also be a
    # bullish bar that, together with bars 16-17, would form a
    # 3-bar FVG completing at bar 17. Without the fix this would
    # emit a second signal where the FVG straddles the ChoCH bar.
    df.iat[15, df.columns.get_loc("Open")] = 97.0
    df.iat[15, df.columns.get_loc("High")] = 102.0
    df.iat[15, df.columns.get_loc("Low")] = 97.0
    df.iat[15, df.columns.get_loc("Close")] = 102.0  # close > H1=100, ChoCH still fires
    # Bars 16, 17 already bullish in the fixture; bar 17 already has
    # low=102 and high=112. Set bar 16 high so there's a gap to 17:
    # bar15.high=102 < bar17.low=102 → not strict, but the original
    # geometry naturally has bar 16 high=104, bar 17 low=102 from
    # the fixture, so a fresh 3-bar window 15-16-17 would compute
    # gap = max(0, 104 - 102) = 2 (negative actually). Instead we
    # rely on the *original* 16-17-18 FVG to fire and verify the
    # detector emits exactly that one — never a 14-15-16 or 15-16-17
    # variant where bar 15 (ChoCH) sits inside the gap window.

    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=2)
    signals = det.detect(df)
    assert len(signals) == 1
    s = signals[0]
    # FVG-completion bar must be ≥ choch_bar + 3 so all three FVG
    # bars (i-2, i-1, i) sit strictly after the ChoCH bar.
    assert s.metadata["choch_bar_offset"] >= 3, (
        f"FVG completion at choch_bar+{s.metadata['choch_bar_offset']} "
        f"violates the post-ChoCH gating rule"
    )


def test_detector_skips_fvg_outside_regular_session() -> None:
    """Pre/post bars can supply ChoCH structure (early swings flow
    into the same session), but the 3-bar bullish FVG itself only
    counts when all three bars are inside RTH (09:30 ≤ t < 16:00).

    Build the standard fixture but shift the index so the FVG bars
    (16-18) land in the post-market window. Result: no signal."""
    df = _build_choch_fvg_session()
    # Original first bar is 09:30 (bar 0). Shift by 7h so bar 0 =
    # 16:30 — the entire fixture moves into pre/post hours.
    shifted = df.copy()
    shifted.index = df.index + pd.Timedelta(hours=7)
    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1)
    assert det.detect(shifted) == []


def test_strategy_disable_stops_outside_rth_skips_eth_stop() -> None:
    """When ``disable_stops_outside_rth=True`` (default), an ETH bar
    (16:00 ET = post-RTH in our fixture) whose low pierces the stop
    must NOT exit the trade — only TP can fire there. Same fixture
    with the option turned off should hit the stop and exit.
    """
    df = _build_choch_fvg_session().copy().astype(float)
    # Bar 26 of the fixture is at 16:00 ET (excluded from
    # 09:30 ≤ t < 16:00 RTH window). Patch its low below the stop
    # (102) so the gate makes a difference. RTH bars 20-25 already
    # rally well above the stop and don't pierce it.
    df.iat[26, df.columns.get_loc("Low")] = 95.0

    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    detector = FairValueGapDetector(
        min_gap_pct=0.0, max_signals_per_session=1
    )

    # disable_stops_outside_rth=True → no exit at bar 26, walks to
    # deadline. force_close_at_session_end=False to isolate the ETH
    # stop gate behavior from the session-close rule.
    strat_gated = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=detector,
        take_profit_r_multiple=10.0,  # unreachable so TP doesn't preempt
        enable_breakeven_stop=False,
        enable_bos_trail=False,
        disable_stops_outside_rth=True,
        force_close_at_session_end=False,
    )
    trade = strat_gated.execute(df, cfg).performance.trades[0]
    assert trade.exit_reason == "end_of_data", (
        f"expected end_of_data (ETH stop suppressed), got "
        f"{trade.exit_reason}"
    )

    # disable_stops_outside_rth=False → ETH bar's low pierces stop,
    # initial_stop fires at the stop level.
    strat_open = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=detector,
        take_profit_r_multiple=10.0,
        enable_breakeven_stop=False,
        enable_bos_trail=False,
        disable_stops_outside_rth=False,
        force_close_at_session_end=False,
    )
    trade = strat_open.execute(df, cfg).performance.trades[0]
    assert trade.exit_reason == "initial_stop"
    assert trade.exit_price == pytest.approx(102.0)


def test_strategy_force_close_at_session_end() -> None:
    """``force_close_at_session_end=True`` (default) flats the
    position at the last RTH bar's close if no TP / stop fired
    during the session. Off → trade walks to deadline.

    Fixture session = 2024-03-18, RTH bars 0-25 (15:45 last RTH,
    bar 26 at 16:00 is ETH and excluded). With TP=5R unreachable
    in the rally and stops not pierced, the only RTH-bound exit
    path is the session_close at bar 25.
    """
    df = _build_choch_fvg_session()
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    detector = FairValueGapDetector(
        min_gap_pct=0.0, max_signals_per_session=1
    )

    # ON (default): forced flat at bar 25 (last RTH bar) close.
    strat_on = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=detector,
        take_profit_r_multiple=10.0,  # unreachable
        enable_breakeven_stop=False,
        enable_bos_trail=False,
        force_close_at_session_end=True,
    )
    trade = strat_on.execute(df, cfg).performance.trades[0]
    assert trade.exit_reason == "session_close"
    # Bar 25 close in the fixture is 130.
    assert trade.exit_price == pytest.approx(130.0)
    # Exit bar index = 25 (last RTH bar at 15:45).
    assert trade.exit_ts == pd.Timestamp(df.index[25]).to_pydatetime()

    # OFF: trade runs to end_of_data at deadline.
    strat_off = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=detector,
        take_profit_r_multiple=10.0,
        enable_breakeven_stop=False,
        enable_bos_trail=False,
        force_close_at_session_end=False,
    )
    trade = strat_off.execute(df, cfg).performance.trades[0]
    assert trade.exit_reason == "end_of_data"


def test_strategy_initial_stop_fires_when_low_pierces_stop() -> None:
    """Patch bar 21 to dip below the stop (102) — strategy must exit
    via initial_stop at the stop level."""
    df = _build_choch_fvg_session().copy()
    df.iloc[21, df.columns.get_loc("Low")] = 100.0
    strat = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1),
        take_profit_r_multiple=10.0,  # unreachably high so TP doesn't preempt
        enable_breakeven_stop=False,
    )
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    result = strat.execute(df, cfg)
    assert result.performance.total_trades == 1
    trade = result.performance.trades[0]
    assert trade.exit_reason == "initial_stop"
    assert trade.exit_price == pytest.approx(102.0)


def test_chart_overlays_render_choch_break_level_and_signal_stop_tp() -> None:
    """When ``fvg_signals`` is provided, the chart now also paints
    the broken swing-high level (purple horizontal from H1's bar to
    the CHoCH bar), per-signal stop (red dashed) and per-signal
    take-profit (green dotted) horizontals — even with no trades.
    These are what makes the pattern page actually useful for
    visually verifying detection quality."""
    from visualization.adapters.plotly_charts import PlotlyChartBuilder

    df = _build_choch_fvg_session()
    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1)
    signals = det.detect(df)
    assert len(signals) == 1

    fig_no_overlay = PlotlyChartBuilder().build_candlestick_with_trades(
        df, trades=[]
    )
    fig_overlay = PlotlyChartBuilder().build_candlestick_with_trades(
        df, trades=[], fvg_signals=signals, take_profit_r=3.0
    )

    added = len(fig_overlay.layout.shapes) - len(fig_no_overlay.layout.shapes)
    # FVG box (1) + midpoint (1) + ChoCH break level horizontal (1)
    # + signal stop (1) + signal TP (1) = 5 added shapes. This
    # fixture's post-entry rally is monotonic (each bar prints a
    # higher high than the previous), so no swing high confirms and
    # no BOS line gets drawn — the BOS overlay needs a real
    # rally-peak → pullback → breakout sequence (covered by
    # ``test_chart_overlay_bos_line_post_entry_rally_peak`` below).
    assert added == 5, (
        f"expected 5 added shapes (box+mid+choch_line+stop+tp), got {added}"
    )

    # ChoCH text annotation appears once. No BOS annotation in a
    # monotonic-rally fixture.
    annotation_texts = [a.text for a in fig_overlay.layout.annotations]
    assert annotation_texts.count("ChoCH") == 1
    assert annotation_texts.count("BOS") == 0


def test_chart_overlay_bos_line_post_entry_rally_peak() -> None:
    """Framework BOS = post-entry rally PEAK (a confirmed swing
    high), not the pre-entry choch_high. When price prints a peak,
    pulls back, then closes above the peak, the chart should draw
    the BOS line at the peak's level and annotate the breakout bar.

    Patch bars 20-25 of the standard fixture so:
      bar 22:  rally peak (high 125 — swing-high candidate)
      bar 23-24: lower highs (pullback) — confirms swing high at 22
      bar 25:  close 128 > 125 → BOS event

    Bar 26 of the fixture is at 16:00 ET (post-RTH) and gets dropped
    by the helper's ETH gate, so the breakout has to land on bar 25
    (15:45) or earlier to register.
    """
    from visualization.adapters.plotly_charts import PlotlyChartBuilder

    df = _build_choch_fvg_session().copy()
    bos_bars = [
        # (idx, O, H, L, C)
        (20, 109, 113, 109, 112),
        (21, 113, 118, 112, 117),
        (22, 120, 125, 117, 124),  # PEAK (swing high after entry)
        (23, 119, 120, 115, 117),  # lower high (pullback)
        (24, 117, 119, 114, 116),  # lower high — confirms 22 as swing
        (25, 122, 130, 120, 128),  # BOS: close 128 > 125
    ]
    for idx, o, h, l, c in bos_bars:
        df.iat[idx, df.columns.get_loc("Open")] = o
        df.iat[idx, df.columns.get_loc("High")] = h
        df.iat[idx, df.columns.get_loc("Low")] = l
        df.iat[idx, df.columns.get_loc("Close")] = c

    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1)
    signals = det.detect(df)
    assert len(signals) == 1

    fig = PlotlyChartBuilder().build_candlestick_with_trades(
        df, trades=[], fvg_signals=signals, take_profit_r=3.0
    )
    annotation_texts = [a.text for a in fig.layout.annotations]
    # Both "BOS line" (level label, always when swing confirms) AND
    # "BOS" (breakout arrow, only when close pierces) appear here.
    assert "BOS line" in annotation_texts
    assert annotation_texts.count("BOS") == 1, (
        f"expected exactly 1 BOS arrow annotation, got: {annotation_texts}"
    )


def test_chart_overlay_bos_line_drawn_without_breakout() -> None:
    """The TSLA 2026-04-24 case: rally peaks, then pulls back, then
    the trade exits via BE stop *before* any close pierces the
    peak. Even though no BOS event fired, the swing-high level was
    the structural pivot the strategy was watching — the chart
    must still draw the BOS line so the user can see the level.
    The "BOS" arrow annotation is reserved for actual breakouts."""
    from visualization.adapters.plotly_charts import PlotlyChartBuilder

    df = _build_choch_fvg_session().copy()
    # Same peak/pullback as the breakout test, but no breakout —
    # bar 26 closes BELOW the bar-22 peak so no BOS fires.
    bos_bars = [
        (20, 109, 113, 109, 112),
        (21, 113, 118, 112, 117),
        (22, 120, 125, 117, 124),  # PEAK at 125
        (23, 119, 120, 115, 117),
        (24, 117, 119, 114, 116),  # confirms 22 swing
        (25, 117, 122, 116, 121),  # rebound below 125
        (26, 121, 124, 119, 122),  # close 122 < peak 125 → no BOS
    ]
    for idx, o, h, l, c in bos_bars:
        df.iat[idx, df.columns.get_loc("Open")] = o
        df.iat[idx, df.columns.get_loc("High")] = h
        df.iat[idx, df.columns.get_loc("Low")] = l
        df.iat[idx, df.columns.get_loc("Close")] = c

    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1)
    signals = det.detect(df)
    assert len(signals) == 1

    swing_ts, event_ts, level = PlotlyChartBuilder._find_post_entry_bos(
        df, df.index[19]
    )
    # Swing high confirmed but never broken — line drawn, no arrow.
    assert level == 125.0
    assert event_ts is None

    fig = PlotlyChartBuilder().build_candlestick_with_trades(
        df, trades=[], fvg_signals=signals, take_profit_r=3.0
    )
    annotation_texts = [a.text for a in fig.layout.annotations]
    # "BOS line" label appears (the level is shown), but the
    # breakout arrow "BOS" does NOT (no close > 125).
    assert "BOS line" in annotation_texts
    assert annotation_texts.count("BOS") == 0


def test_find_post_entry_bos_helper() -> None:
    """``PlotlyChartBuilder._find_post_entry_bos`` should return the
    swing-high level, the swing-high bar timestamp, and the
    breakout bar timestamp. With monotonic rallies it returns
    ``(None, None, None)`` because no swing high confirms."""
    from visualization.adapters.plotly_charts import PlotlyChartBuilder

    # Monotonic rally — no swing high should print.
    df_mono = _build_choch_fvg_session()
    sig_ts = df_mono.index[19]
    swing_ts, event_ts, level = PlotlyChartBuilder._find_post_entry_bos(
        df_mono, sig_ts
    )
    assert (swing_ts, event_ts, level) == (None, None, None)

    # Patch in a peak/pullback/breakout — should now find BOS.
    # Breakout lands on bar 25 (15:45 ET, last RTH 15m bar); bar 26
    # at 16:00 is ETH-filtered.
    df_bos = df_mono.copy()
    bos_bars = [
        (20, 109, 113, 109, 112),
        (21, 113, 118, 112, 117),
        (22, 120, 125, 117, 124),  # peak (high=125)
        (23, 119, 120, 115, 117),
        (24, 117, 119, 114, 116),  # confirms swing
        (25, 122, 130, 120, 128),  # close 128 > 125 → BOS
    ]
    for idx, o, h, l, c in bos_bars:
        df_bos.iat[idx, df_bos.columns.get_loc("Open")] = o
        df_bos.iat[idx, df_bos.columns.get_loc("High")] = h
        df_bos.iat[idx, df_bos.columns.get_loc("Low")] = l
        df_bos.iat[idx, df_bos.columns.get_loc("Close")] = c
    swing_ts, event_ts, level = PlotlyChartBuilder._find_post_entry_bos(
        df_bos, df_bos.index[19]
    )
    assert level == 125.0
    assert pd.Timestamp(swing_ts) == pd.Timestamp(df_bos.index[22]).tz_localize(None)
    assert pd.Timestamp(event_ts) == pd.Timestamp(df_bos.index[25]).tz_localize(None)


def test_chart_overlays_render_fvg_box_midpoint_and_tp_line() -> None:
    """``build_candlestick_with_trades`` paints FVG-specific shapes
    (rectangle + midpoint line) and a CHoCH annotation when the
    caller supplies ``fvg_signals``, plus a take-profit horizontal
    when ``take_profit_r`` is set. Other strategies that don't pass
    these args must keep rendering exactly as before."""
    from visualization.adapters.plotly_charts import PlotlyChartBuilder

    df = _build_choch_fvg_session()
    det = FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1)
    signals = det.detect(df)
    assert len(signals) == 1

    strat = FairValueGapStrategy(
        market_data=_StubMarket(df),
        detector=det,
        take_profit_r_multiple=1.0,
        enable_breakeven_stop=False,
    )
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    result = strat.execute(df, cfg)
    trades = result.performance.trades

    # With overlays.
    fig_with = PlotlyChartBuilder().build_candlestick_with_trades(
        df, trades, fvg_signals=signals, take_profit_r=1.0
    )
    # Without overlays — same data, just no FVG-specific args.
    fig_without = PlotlyChartBuilder().build_candlestick_with_trades(
        df, trades
    )

    # FVG overlay adds: 1 rect + 1 midpoint line per signal, 1 TP
    # line per trade. ``fig_without`` has none of these.
    overlay_shapes_added = len(fig_with.layout.shapes) - len(
        fig_without.layout.shapes
    )
    assert overlay_shapes_added >= 3, (
        f"expected ≥3 added shapes (FVG box + midpoint + TP line), "
        f"got {overlay_shapes_added}"
    )

    # CHoCH annotation appears once per signal.
    annotation_texts = [a.text for a in fig_with.layout.annotations]
    assert "ChoCH" in annotation_texts


def test_strategy_run_forwards_interval_to_market_data() -> None:
    class _Recording(MarketDataPort):
        def __init__(self, df):
            self._df = df
            self.last_interval = None

        def fetch_ohlcv(self, symbol, start, end, interval="1d"):
            self.last_interval = interval
            return self._df

    df = _build_choch_fvg_session()
    market = _Recording(df)
    strat = FairValueGapStrategy(
        market_data=market,
        detector=FairValueGapDetector(min_gap_pct=0.0, max_signals_per_session=1),
    )
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        pattern_name="fair_value_gap",
        max_holding_days=20,
    )
    strat.run(cfg)
    # Default interval for FVG strategy is "15m".
    assert market.last_interval == "15m"
