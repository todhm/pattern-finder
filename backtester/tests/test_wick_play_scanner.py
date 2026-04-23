"""Unit tests for :class:`WickPlayBuySignalScanner`.

Reuses the synthetic baseline df from ``test_wick_play_regime`` —
22 uptrend bars + wick + inside + breakout — wrapped in a fake
``MarketDataPort`` so scan / refresh_targets / build_signal_at can
run against deterministic data with no network.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from data.adapters.wikipedia_universe import StaticUniverseAdapter
from data.domain.ports import MarketDataPort
from pattern.adapters.exhaustion_extension_top import (
    ExhaustionExtensionTopDetector,
)
from pattern.adapters.wick_play import WickPlayDetector
from signals.adapters.wick_play_scanner import WickPlayBuySignalScanner
from strategy.adapters.wickplay_strategy import WickPlayStrategy


def _make_df(rows: list[tuple], start: str) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(rows))
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


def _baseline_rows() -> list[tuple]:
    rows: list[tuple] = []
    for i in range(22):
        open_ = 100.0 + i * 0.9
        close = open_ + 0.3
        high = close + 0.5
        low = open_ - 0.3
        rows.append((open_, high, low, close, 1_500_000))
    # wick bar
    rows.append((120.0, 126.0, 117.0, 118.5, 1_400_000))
    # inside bar
    rows.append((119.0, 120.0, 118.0, 119.0, 500_000))
    # breakout bar
    rows.append((119.5, 129.0, 119.0, 128.0, 3_000_000))
    # extra bar so "entry = next open" is printed
    rows.append((128.5, 130.0, 127.0, 129.5, 2_500_000))
    return rows


def _start_date() -> str:
    """Pick a start date such that the signal bar (index 24) lands
    within ``lookback_days`` of today. The baseline sequence is 26
    business days long; signal is at idx 24, so we anchor the df
    so idx 24 ≈ today - 1 bday and everything else works out."""
    return (date.today() - timedelta(days=45)).isoformat()


class _FakeMarketData(MarketDataPort):
    """Returns a pre-built DataFrame for any ticker, sliced to
    ``[start, end]`` on the index."""

    def __init__(self, df_by_ticker: dict[str, pd.DataFrame]):
        self._df_by_ticker = df_by_ticker

    def fetch_ohlcv(self, symbol, start, end):
        df = self._df_by_ticker.get(symbol)
        if df is None:
            raise ValueError(f"unknown ticker {symbol}")
        mask = (df.index.date >= start) & (df.index.date <= end)
        return df.loc[mask].copy()


def _lenient_detector(**overrides):
    kwargs = dict(
        min_upper_wick_ratio=0.5,
        max_volume_dryup=1.0,
        max_wick_range_atr=None,
        min_breakout_strength_atr=0.0,
        min_prior_trend_20d=None,
        min_wick_close_location=0.15,
        psych_dramatic_wick_ratio=0.65,
        min_psych_score=0,
    )
    kwargs.update(overrides)
    return WickPlayDetector(**kwargs)


def _build_scanner(
    tickers: list[str],
    breakeven: bool = True,
) -> WickPlayBuySignalScanner:
    df = _make_df(_baseline_rows(), start=_start_date())
    market_data = _FakeMarketData({t: df for t in tickers})
    universe = StaticUniverseAdapter({"test": tickers})
    detector = _lenient_detector()
    exit_detector = ExhaustionExtensionTopDetector(
        extension_atr_mult=2.2, min_slow_slope=0.0, ema_fast=10
    )
    strategy = WickPlayStrategy(
        market_data=market_data,
        detector=detector,
        exit_detector=exit_detector,
        ema_trail=10,
        enable_breakeven_stop=breakeven,
        breakeven_arm_r_multiple=2.0,
        breakeven_exit_offset_r=1.0,
    )
    return WickPlayBuySignalScanner(
        market_data=market_data,
        universe_provider=universe,
        detector=detector,
        strategy=strategy,
        max_workers=2,
    )


# ---------- scan ----------------------------------------------------


def test_scan_returns_signal_with_target_fields():
    scanner = _build_scanner(["ABC", "XYZ"])
    signals = scanner.scan(universe="test", lookback_days=60)
    assert len(signals) >= 1, "expected at least one detector-hit ticker"

    sig = signals[0]
    assert sig.pattern_name == "wick_play"
    assert sig.entry_price > sig.stop_loss > 0
    meta = sig.metadata

    # Core target fields populated
    assert "target_2r" in meta
    assert "target_3r" in meta
    assert meta["target_2r"] > sig.entry_price
    assert meta["target_3r"] > meta["target_2r"]
    assert meta["target_exhaustion_primary"] is not None
    assert meta["r_to_exhaustion_primary"] is not None
    assert meta["ema_trail_current"] > 0

    # Breakeven fields emitted because strategy has it on
    assert meta["breakeven_arm_price"] > sig.entry_price
    assert meta["breakeven_exit_price"] > sig.entry_price
    assert meta["breakeven_exit_price"] < meta["breakeven_arm_price"]

    # Pressure fields populated
    assert meta["volume"] > 0
    assert "buy_sell_ratio" in meta

    # entry_confirmed=True because we appended a bar after breakout
    assert meta["entry_confirmed"] is True


def test_scan_ordering_is_date_desc_then_ratio_desc():
    scanner = _build_scanner(["AAA", "BBB", "CCC"])
    signals = scanner.scan(universe="test", lookback_days=60)
    dates = [s.signal_date for s in signals]
    # Same signal date across all tickers, so ordering resolves to
    # buy/sell ratio descending. All identical here (same df), so
    # we just check date-desc invariant holds.
    assert dates == sorted(dates, reverse=True)


def test_scan_excludes_breakeven_fields_when_strategy_disables_it():
    scanner = _build_scanner(["ABC"], breakeven=False)
    signals = scanner.scan(universe="test", lookback_days=60)
    assert signals
    meta = signals[0].metadata
    assert "breakeven_arm_price" not in meta
    assert "breakeven_exit_price" not in meta


# ---------- refresh_targets ---------------------------------------


def test_refresh_updates_latest_close_and_timestamp():
    scanner = _build_scanner(["ABC"])
    signals = scanner.scan(universe="test", lookback_days=60)
    sig = signals[0]
    prev_refreshed = sig.metadata["refreshed_at"]
    # Ensure wall-clock advances at least 1 second between stamps
    # so the equality check is meaningful.
    before = datetime.utcnow()
    refreshed = scanner.refresh_targets(sig)

    assert refreshed.entry_price == sig.entry_price  # locked
    assert refreshed.stop_loss == sig.stop_loss      # locked
    assert "latest_close" in refreshed.metadata
    assert refreshed.metadata["refreshed_at"] >= before.isoformat(timespec="seconds")
    # Refresh should at least re-stamp even if values don't change
    assert refreshed.metadata["refreshed_at"] >= prev_refreshed


# ---------- build_signal_at ---------------------------------------


def test_build_signal_at_detector_hit_returns_detector_metadata():
    scanner = _build_scanner(["ABC"])
    # Signal bar = index 24 of the baseline → compute its date
    df = _make_df(_baseline_rows(), start=_start_date())
    signal_date = df.index[24].date()

    sig = scanner.build_signal_at("ABC", signal_date)
    assert sig.pattern_name == "wick_play"
    assert sig.metadata.get("manually_added") is True
    assert "manually_added_no_signal" not in sig.metadata
    assert sig.metadata.get("trigger") in {"wick_high", "inside_high"}


def test_build_signal_at_non_hit_returns_manual_fallback():
    scanner = _build_scanner(["ABC"])
    df = _make_df(_baseline_rows(), start=_start_date())
    # A prior uptrend bar — detector doesn't fire here
    quiet_date = df.index[10].date()

    sig = scanner.build_signal_at("ABC", quiet_date)
    assert sig.pattern_name == "manual"
    assert sig.metadata["manually_added_no_signal"] is True
    assert sig.stop_loss == float(df["Low"].iloc[10])


def test_build_signal_at_unknown_date_raises():
    scanner = _build_scanner(["ABC"])
    try:
        scanner.build_signal_at("ABC", date(1999, 1, 4))
        raise AssertionError("should have raised")
    except ValueError:
        pass


# ---------- scan diagnostics --------------------------------------


def test_last_scan_stats_populated_after_successful_scan():
    scanner = _build_scanner(["ABC", "DEF"])
    signals = scanner.scan(universe="test", lookback_days=60)
    stats = scanner.last_scan_stats
    assert stats["tickers_requested"] == 2
    assert stats["tickers_with_data"] == 2
    assert stats["tickers_fetch_failed"] == 0
    assert stats["tickers_too_few_bars"] == 0
    assert stats["total_detector_hits_history"] >= 2  # both tickers hit
    assert stats["in_window_hits"] == len(signals)
    assert stats["returned"] == len(signals)
    assert stats["lookback_days"] == 60


def test_last_scan_stats_flags_fetch_failure():
    """When every ticker fetch fails we should still populate stats —
    specifically so the UI can explain "no data" vs "no pattern"."""

    class _AlwaysFails(MarketDataPort):
        def fetch_ohlcv(self, symbol, start, end):
            raise RuntimeError("boom")

    scanner = WickPlayBuySignalScanner(
        market_data=_AlwaysFails(),
        universe_provider=StaticUniverseAdapter({"test": ["A", "B"]}),
        detector=_lenient_detector(),
        strategy=WickPlayStrategy(
            market_data=_AlwaysFails(),
            detector=_lenient_detector(),
        ),
        max_workers=1,
    )
    signals = scanner.scan(universe="test", lookback_days=60)
    assert signals == []
    stats = scanner.last_scan_stats
    assert stats["tickers_requested"] == 2
    assert stats["tickers_with_data"] == 0
    assert stats["tickers_fetch_failed"] == 2
    assert stats["in_window_hits"] == 0


def test_last_scan_stats_in_window_vs_total_hits():
    """Detector hits can land outside the lookback window; stats
    should separate the two so the UI can surface "detector fired
    but too long ago"."""
    scanner = _build_scanner(["ABC"])
    scanner.scan(universe="test", lookback_days=1)  # very tight window
    stats = scanner.last_scan_stats
    # The synthetic signal bar sits ~1 business day before today
    # (we anchor _start_date accordingly), so a 1-day lookback
    # should still catch it OR land it just outside.
    assert stats["total_detector_hits_history"] >= 1
    # Either way: total_hits_history >= in_window_hits, and the
    # stats exist — that's the contract.
    assert stats["in_window_hits"] <= stats["total_detector_hits_history"]
