"""Unit tests for EODHDAdapter."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.adapters.eodhd_adapter import EODHDAdapter


class _StubResp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = "" if status_code == 200 else str(payload)

    def json(self):
        return self._payload


class _StubClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return self._responses.pop(0)


_INTRADAY_BAR = {
    "timestamp": int(
        pd.Timestamp("2024-06-03 09:30", tz="America/New_York").timestamp()
    ),
    "gmtoffset": 0,
    "datetime": "2024-06-03 13:30:00",
    "open": 100.0,
    "high": 101.5,
    "low": 99.5,
    "close": 100.75,
    "volume": 12345,
}

_EOD_BAR = {
    "date": "2024-06-03",
    "open": 100.0,
    "high": 101.5,
    "low": 99.5,
    "close": 100.75,
    "adjusted_close": 100.75,
    "volume": 12345,
}


# ---- symbol translation -----------------------------------------------------


def test_normalize_symbol_us_bare_ticker_gets_us_suffix() -> None:
    assert EODHDAdapter._normalize_symbol("AAPL") == "AAPL.US"
    assert EODHDAdapter._normalize_symbol("aapl") == "AAPL.US"
    assert EODHDAdapter._normalize_symbol("BRK-B") == "BRK-B.US"


def test_normalize_symbol_kospi_yfinance_to_eodhd() -> None:
    """yfinance uses ``.KS`` for KOSPI; EODHD uses ``.KO`` (MIC XKRX)."""
    assert EODHDAdapter._normalize_symbol("005930.KS") == "005930.KO"
    assert EODHDAdapter._normalize_symbol("000660.ks") == "000660.KO"


def test_normalize_symbol_kosdaq_passthrough() -> None:
    """KOSDAQ ``.KQ`` matches between yfinance and EODHD."""
    assert EODHDAdapter._normalize_symbol("035720.KQ") == "035720.KQ"


def test_normalize_symbol_other_suffixes_passthrough() -> None:
    """Hand-written EODHD-native symbols stay verbatim."""
    assert EODHDAdapter._normalize_symbol("BMW.XETRA") == "BMW.XETRA"
    assert EODHDAdapter._normalize_symbol("7203.TSE") == "7203.TSE"


# ---- intraday fetch ---------------------------------------------------------


def test_intraday_uses_intraday_endpoint_with_unix_timestamps() -> None:
    client = _StubClient([_StubResp(200, [_INTRADAY_BAR])])
    a = EODHDAdapter(api_key="k", base_url="https://x", http_client=client)
    a.fetch_ohlcv("AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="15m")
    url, params = client.calls[0]
    assert url == "https://x/intraday/AAPL.US"
    assert params["api_token"] == "k"
    assert params["interval"] == "15m"
    assert params["fmt"] == "json"
    # ``from`` / ``to`` are unix epochs (seconds), to is +1 day
    # past the user's end so the last calendar day is fully
    # included by EODHD's exclusive boundary.
    assert isinstance(params["from"], int)
    assert isinstance(params["to"], int)
    assert params["to"] - params["from"] == 86400  # exact 1-day window


def test_intraday_returns_ohlcv_in_ny_tz() -> None:
    client = _StubClient([_StubResp(200, [_INTRADAY_BAR])])
    a = EODHDAdapter(api_key="k", http_client=client)
    df = a.fetch_ohlcv(
        "AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="15m"
    )
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.iloc[0]["Open"] == 100.0
    assert df.iloc[0]["Close"] == 100.75
    assert str(df.index.tz) == "America/New_York"


def test_intraday_translates_kospi_symbol() -> None:
    """Yfinance ``005930.KS`` → EODHD ``005930.KO``."""
    client = _StubClient([_StubResp(200, [_INTRADAY_BAR])])
    a = EODHDAdapter(api_key="k", base_url="https://x", http_client=client)
    a.fetch_ohlcv("005930.KS", date(2024, 6, 3), date(2024, 6, 3), interval="15m")
    url, _ = client.calls[0]
    assert url == "https://x/intraday/005930.KO"


def test_unsupported_intraday_interval_rejected() -> None:
    a = EODHDAdapter(api_key="k", http_client=_StubClient([]))
    # ``10m`` is in the FVG strategy's old ask list but EODHD's
    # API rejects it with HTTP 422 — we surface it as a clear
    # ValueError before the network call instead of letting the
    # remote complain.
    with pytest.raises(ValueError, match="does not map interval"):
        a.fetch_ohlcv("AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="10m")
    with pytest.raises(ValueError, match="does not map interval"):
        a.fetch_ohlcv("AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="2m")


def test_intraday_drops_kr_close_auction_placeholder_bars() -> None:
    """KR (and similar) sessions emit a single close-auction reference
    bar with ``V=None`` and OHLC all equal — for Samsung 005930 on a
    typical day this is the 15:00 KST placeholder. Detector logic
    treats it as a flat green/red candle and it breaks all-bullish
    FVG checks + chart hover. The adapter must drop those rows."""
    real_bar = dict(_INTRADAY_BAR)
    placeholder = {
        "timestamp": int(
            pd.Timestamp("2024-06-03 09:45", tz="America/New_York").timestamp()
        ),
        "gmtoffset": 0,
        "datetime": "2024-06-03 13:45:00",
        "open": 101.0,
        "high": 101.0,
        "low": 101.0,
        "close": 101.0,
        "volume": None,
    }
    client = _StubClient([_StubResp(200, [real_bar, placeholder])])
    a = EODHDAdapter(api_key="k", http_client=client)
    df = a.fetch_ohlcv(
        "AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="15m"
    )
    # Real bar kept, placeholder dropped.
    assert len(df) == 1
    assert df.iloc[0]["Open"] == 100.0


def test_intraday_keeps_real_low_volume_bar() -> None:
    """A bar with a real OHLC range but tiny volume (early pre-market
    on a thin US name) must NOT be filtered as a placeholder — only
    the OHLC-flat + V-null pair is the synthetic signature."""
    thin_bar = {
        "timestamp": int(
            pd.Timestamp("2024-06-03 04:15", tz="America/New_York").timestamp()
        ),
        "gmtoffset": 0,
        "datetime": "2024-06-03 08:15:00",
        "open": 100.0,
        "high": 100.05,
        "low": 99.95,
        "close": 100.02,
        "volume": 17,
    }
    client = _StubClient([_StubResp(200, [thin_bar])])
    a = EODHDAdapter(api_key="k", http_client=client)
    df = a.fetch_ohlcv(
        "AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="15m"
    )
    assert len(df) == 1


def test_intraday_empty_response_raises() -> None:
    client = _StubClient([_StubResp(200, [])])
    a = EODHDAdapter(api_key="k", http_client=client)
    with pytest.raises(ValueError, match="No data found"):
        a.fetch_ohlcv("AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="15m")


def test_intraday_http_error_surfaces() -> None:
    client = _StubClient([_StubResp(403, "Only EOD data allowed for free users")])
    a = EODHDAdapter(api_key="k", http_client=client)
    with pytest.raises(ValueError, match="HTTP 403"):
        a.fetch_ohlcv("AAPL", date(2024, 6, 3), date(2024, 6, 3), interval="15m")


# ---- daily fetch ------------------------------------------------------------


def test_daily_uses_eod_endpoint_and_returns_naive_index() -> None:
    client = _StubClient([_StubResp(200, [_EOD_BAR])])
    a = EODHDAdapter(api_key="k", base_url="https://x", http_client=client)
    df = a.fetch_ohlcv("AAPL", date(2024, 6, 1), date(2024, 6, 5), interval="1d")
    url, params = client.calls[0]
    assert url == "https://x/eod/AAPL.US"
    assert params["period"] == "d"
    assert params["from"] == "2024-06-01"
    assert params["to"] == "2024-06-05"
    # Daily index stays tz-naive to match yfinance's daily frame
    # (intraday detectors check tz to know whether RTH gating
    # applies; daily must be naive so they treat it as session-
    # boundary-only).
    assert df.index.tz is None
    assert df.iloc[0]["Open"] == 100.0


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="EODHD_API_KEY"):
        EODHDAdapter()
