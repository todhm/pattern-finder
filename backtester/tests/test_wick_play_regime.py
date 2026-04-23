"""Unit tests for the regime + blackout gates on WickPlayDetector.

Synthetic fixture: a clean uptrend → wick bar → inside bar → breakout
bar sequence that fires exactly one signal under a "lenient" detector
(most value-based gates disabled). Each test then toggles ONE gate
and checks the expected pass/reject outcome.
"""

from datetime import date

import pandas as pd
import pytest

from pattern.adapters.wick_play import WickPlayDetector


def _make_df(rows: list[tuple], start: str = "2024-01-01") -> pd.DataFrame:
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


def _baseline_df() -> pd.DataFrame:
    """22 prior uptrend bars + wick + inside + breakout. 25 bars total.

    - Prior: gentle uptrend so 20-day return at wick bar is ~+18% (>-1%).
    - Wick bar (index 22): Open 120, High 126, Low 117, Close 118.5
      → upper wick ratio 0.667, close_location 0.167, both gates pass.
    - Inside bar (index 23): fully inside wick range, volume dries to 1/3.
    - Breakout bar (index 24): close 128 > wick high 126 with 2× volume
      expansion and bullish body.
    """
    rows: list[tuple] = []
    for i in range(22):
        open_ = 100.0 + i * 0.9
        close = open_ + 0.3
        high = close + 0.5
        low = open_ - 0.3
        rows.append((open_, high, low, close, 1_500_000))
    # wick bar — prominent upper wick, close near the low
    rows.append((120.0, 126.0, 117.0, 118.5, 1_400_000))
    # inside bar — volume dryup, range inside wick
    rows.append((119.0, 120.0, 118.0, 119.0, 500_000))
    # breakout bar — bullish, close clears wick high
    rows.append((119.5, 129.0, 119.0, 128.0, 3_000_000))
    return _make_df(rows)


def _lenient_detector(**overrides) -> WickPlayDetector:
    """Detector with structural gates on but value-based gates off
    so the baseline df reliably fires exactly one signal."""
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


# ---------- Baseline -----------------------------------------------


def test_baseline_signal_fires():
    df = _baseline_df()
    signals = _lenient_detector().detect(df)
    assert len(signals) == 1
    assert signals[0].date == df.index[-1].date()


# ---------- Blackout gate ------------------------------------------


def test_blackout_on_signal_date_rejects():
    df = _baseline_df()
    signal_date = df.index[-1].date()
    detector = _lenient_detector(blackout_dates={signal_date})
    assert detector.detect(df) == []


def test_blackout_on_unrelated_date_ignored():
    df = _baseline_df()
    detector = _lenient_detector(blackout_dates={date(1999, 1, 1)})
    assert len(detector.detect(df)) == 1


def test_blackout_on_entry_date_rejects():
    """When blackout hits i+1 (the next-day entry bar), the signal
    at bar i must also be rejected. Constructed by appending one
    extra bar past the breakout so signal at index 24 has an entry
    candidate at index 25."""
    df = _baseline_df()
    # Append one more bar so index 25 exists as a potential entry.
    extra = pd.DataFrame(
        {
            "Open": [128.5], "High": [130.0], "Low": [127.0],
            "Close": [129.5], "Volume": [2_500_000],
        },
        index=pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(days=1), periods=1
        ),
    )
    df2 = pd.concat([df, extra])
    entry_date = df2.index[-1].date()  # entry = signal + 1
    detector = _lenient_detector(blackout_dates={entry_date})
    assert detector.detect(df2) == []


# ---------- Regime gate — SMA -------------------------------------


def _regime_df_rising(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(df)):
        c = 200.0 + i * 1.0
        rows.append((c, c + 0.5, c - 0.5, c, 1_000_000))
    return _make_df(rows, start=df.index[0].strftime("%Y-%m-%d"))


def _regime_df_falling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(df)):
        c = 200.0 - i * 0.5
        rows.append((c, c + 0.5, c - 0.5, c, 1_000_000))
    return _make_df(rows, start=df.index[0].strftime("%Y-%m-%d"))


def test_regime_above_sma_passes():
    df = _baseline_df()
    detector = _lenient_detector(
        regime_df=_regime_df_rising(df), regime_min_above_sma=10
    )
    assert len(detector.detect(df)) == 1


def test_regime_below_sma_rejects():
    df = _baseline_df()
    detector = _lenient_detector(
        regime_df=_regime_df_falling(df), regime_min_above_sma=10
    )
    assert detector.detect(df) == []


# ---------- Regime gate — N-day drawdown --------------------------


def test_regime_n_day_drawdown_rejects():
    df = _baseline_df()
    # Flat index except a sharp −5% crash on the signal bar.
    rows = [(100.0, 100.5, 99.5, 100.0, 1_000_000)] * (len(df) - 1)
    rows.append((100.0, 100.0, 94.0, 95.0, 1_000_000))
    regime_df = _make_df(rows, start=df.index[0].strftime("%Y-%m-%d"))
    detector = _lenient_detector(
        regime_df=regime_df, regime_max_n_day_drawdown=(5, 0.02)
    )
    assert detector.detect(df) == []


def test_regime_n_day_flat_passes():
    df = _baseline_df()
    rows = [(100.0, 100.5, 99.5, 100.0, 1_000_000)] * len(df)
    regime_df = _make_df(rows, start=df.index[0].strftime("%Y-%m-%d"))
    detector = _lenient_detector(
        regime_df=regime_df, regime_max_n_day_drawdown=(5, 0.02)
    )
    assert len(detector.detect(df)) == 1


# ---------- Misconfiguration --------------------------------------


def test_regime_gate_without_regime_df_raises():
    df = _baseline_df()
    detector = _lenient_detector(
        regime_df=None, regime_min_above_sma=10
    )
    with pytest.raises(ValueError):
        detector.detect(df)


def test_invalid_drawdown_tuple_raises():
    with pytest.raises(ValueError):
        WickPlayDetector(regime_max_n_day_drawdown=(0, 0.02))
    with pytest.raises(ValueError):
        WickPlayDetector(regime_max_n_day_drawdown=(5, -0.01))
