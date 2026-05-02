"""Unit tests for the MarketCalendar abstraction."""

from datetime import time

from data.domain.market_calendar import (
    CRYPTO,
    KR,
    NY,
    MarketCalendar,
    market_for_ticker,
)


def test_ny_calendar_describes_us_cash_session() -> None:
    assert NY.name == "NY"
    assert NY.tz == "America/New_York"
    assert NY.rth_open == time(9, 30)
    assert NY.rth_close == time(16, 0)
    assert NY.zoneinfo.key == "America/New_York"


def test_kr_calendar_describes_korean_continuous_session() -> None:
    assert KR.name == "KR"
    assert KR.tz == "Asia/Seoul"
    assert KR.rth_open == time(9, 0)
    # 15:00 (not 15:30) — matches the depth EODHD's intraday feed
    # actually carries; everything past 15:00 is the closing-auction
    # window that EODHD only surfaces as a placeholder bar.
    assert KR.rth_close == time(15, 0)
    assert KR.zoneinfo.key == "Asia/Seoul"


def test_market_calendar_is_hashable_and_frozen() -> None:
    """Frozen so callers can use it as a dict key or set member —
    e.g., a cache keyed by (ticker, calendar)."""
    cal = MarketCalendar(
        name="X", tz="UTC", rth_open=time(0, 0), rth_close=time(23, 59)
    )
    {cal: 1}  # would raise TypeError if not hashable
    try:
        cal.name = "Y"  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("MarketCalendar must be frozen")


def test_market_for_ticker_resolves_kospi_suffix() -> None:
    assert market_for_ticker("005930.KS") == KR  # Samsung
    assert market_for_ticker("000660.KS") == KR  # SK Hynix


def test_market_for_ticker_resolves_kosdaq_suffix() -> None:
    assert market_for_ticker("035720.KQ") == KR  # KOSDAQ-listed example
    assert market_for_ticker("293490.KQ") == KR


def test_market_for_ticker_is_case_insensitive() -> None:
    assert market_for_ticker("005930.ks") == KR
    assert market_for_ticker("AaPl") == NY


def test_market_for_ticker_defaults_to_ny_for_us_symbols() -> None:
    assert market_for_ticker("AAPL") == NY
    assert market_for_ticker("META") == NY
    assert market_for_ticker("BRK-B") == NY
    assert market_for_ticker("SPY") == NY


def test_crypto_calendar_is_24_7() -> None:
    assert CRYPTO.name == "CRYPTO"
    assert CRYPTO.tz == "UTC"
    assert CRYPTO.is_24_7 is True
    assert CRYPTO.zoneinfo.key == "UTC"


def test_ny_and_kr_calendars_are_not_24_7() -> None:
    """is_24_7 defaults to False for session-bound markets."""
    assert NY.is_24_7 is False
    assert KR.is_24_7 is False


def test_market_for_ticker_resolves_cc_suffix() -> None:
    assert market_for_ticker("BTC-USD.CC") == CRYPTO
    assert market_for_ticker("ETH-USD.CC") == CRYPTO
    assert market_for_ticker("btc-usd.cc") == CRYPTO
