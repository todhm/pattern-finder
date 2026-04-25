"""Session-helper unit tests.

Fixtures build a small synthetic 15m frame spanning two trading days so
we can exercise date grouping, previous-session lookup, and mid-session
accumulation without hitting yfinance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pattern.helpers.sessions import (
    NY_TZ,
    SessionOHLC,
    ensure_ny_tz,
    prev_session_ohlc,
    regular_session_only,
    resample_to_daily,
    session_dates,
    session_ohlc_so_far,
)


def _two_session_frame() -> pd.DataFrame:
    """Day 1: 09:30–15:45 (26 bars), Day 2: 09:30–15:45 (26 bars). NY-tz."""
    bars = []
    for day in ["2024-03-18", "2024-03-19"]:
        idx = pd.date_range(
            start=f"{day} 09:30", end=f"{day} 15:45", freq="15min", tz=NY_TZ
        )
        # Day 1 range: 100..125; Day 2 range: 130..155 (strictly higher)
        base = 100 if day == "2024-03-18" else 130
        for i, ts in enumerate(idx):
            price = base + i
            bars.append(
                {
                    "ts": ts,
                    "Open": price,
                    "High": price + 0.5,
                    "Low": price - 0.5,
                    "Close": price + 0.25,
                    "Volume": 1_000 + i,
                }
            )
    df = pd.DataFrame(bars).set_index("ts")
    return df


def test_ensure_ny_tz_localizes_naive() -> None:
    idx = pd.date_range("2024-03-18 09:30", periods=3, freq="15min")
    df = pd.DataFrame({"Open": [1, 2, 3]}, index=idx)
    out = ensure_ny_tz(df)
    assert str(out.index.tz) == NY_TZ


def test_ensure_ny_tz_converts_utc() -> None:
    idx = pd.date_range("2024-03-18 13:30", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({"Open": [1, 2, 3]}, index=idx)
    out = ensure_ny_tz(df)
    assert str(out.index.tz) == NY_TZ
    # 13:30 UTC on 2024-03-18 is 09:30 NY (EDT is UTC-4 after DST).
    assert out.index[0].strftime("%H:%M") == "09:30"


def test_ensure_ny_tz_noop_when_already_ny() -> None:
    idx = pd.date_range("2024-03-18 09:30", periods=3, freq="15min", tz=NY_TZ)
    df = pd.DataFrame({"Open": [1, 2, 3]}, index=idx)
    out = ensure_ny_tz(df)
    assert out is df  # no copy


def test_regular_session_only_drops_pre_and_post() -> None:
    idx = pd.date_range(
        "2024-03-18 04:00", "2024-03-18 19:45", freq="15min", tz=NY_TZ
    )
    df = pd.DataFrame({"Open": range(len(idx))}, index=idx)
    out = regular_session_only(df)
    assert out.index.min().strftime("%H:%M") == "09:30"
    assert out.index.max().strftime("%H:%M") == "15:45"


def test_session_dates_groups_by_ny_calendar() -> None:
    df = _two_session_frame()
    dates = session_dates(df)
    assert len(set(dates)) == 2
    # Both sessions have all 26 bars.
    counts = pd.Series(dates).value_counts()
    assert set(counts.tolist()) == {26}


def test_prev_session_ohlc_returns_day1_from_day2() -> None:
    df = _two_session_frame()
    at = df.index[26]  # first bar of day 2 (09:30)
    prev = prev_session_ohlc(df, at)
    assert prev is not None
    # Day 1 prices 100..125; open=100, close=125.25, high=125.5, low=99.5
    assert prev == SessionOHLC(open=100.0, high=125.5, low=99.5, close=125.25)


def test_prev_session_ohlc_none_on_first_session() -> None:
    df = _two_session_frame()
    at = df.index[0]  # first bar of day 1
    assert prev_session_ohlc(df, at) is None


def test_prev_session_ohlc_accepts_mid_day_timestamp() -> None:
    df = _two_session_frame()
    # Pass a timestamp that doesn't match any bar exactly; session
    # date lookup still works.
    at = pd.Timestamp("2024-03-19 11:07", tz=NY_TZ)
    prev = prev_session_ohlc(df, at)
    assert prev is not None
    assert prev.close == 125.25


def test_session_ohlc_so_far_includes_at_ts() -> None:
    df = _two_session_frame()
    # Third bar of day 2 → session so far spans the first three bars.
    at = df.index[28]
    snap = session_ohlc_so_far(df, at)
    assert snap is not None
    # Day 2 prices 130..132; open=130, high=132.5, low=129.5, close=132.25
    assert snap == SessionOHLC(open=130.0, high=132.5, low=129.5, close=132.25)


def test_session_ohlc_so_far_none_before_first_bar() -> None:
    df = _two_session_frame()
    at = pd.Timestamp("2024-03-17 15:00", tz=NY_TZ)  # before any data
    assert session_ohlc_so_far(df, at) is None


def test_resample_to_daily_produces_one_row_per_session() -> None:
    df = _two_session_frame()
    daily = resample_to_daily(df)
    assert len(daily) == 2
    assert daily.index.tz is None
    # Day 1: 100..125, Day 2: 130..155
    assert daily.iloc[0]["Open"] == 100.0
    assert daily.iloc[0]["Close"] == 125.25
    assert daily.iloc[1]["High"] == 155.5
    assert daily.iloc[1]["Low"] == 129.5
    # Volume sums across 26 bars per session.
    expected_vol = sum(1_000 + i for i in range(26))
    assert daily.iloc[0]["Volume"] == expected_vol


def test_ensure_ny_tz_rejects_non_datetime_index() -> None:
    df = pd.DataFrame({"Open": [1, 2, 3]}, index=[0, 1, 2])
    with pytest.raises(TypeError):
        ensure_ny_tz(df)
