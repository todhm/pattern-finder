"""Unit tests for MassiveAdapter (Polygon-compatible)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.adapters.massive_adapter import MassiveAdapter


class _StubResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _StubClient:
    """Plays back a pre-baked sequence of payloads keyed by request count."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None):
        self.calls.append((url, dict(params or {})))
        return _StubResponse(self._responses.pop(0))


# Compute a real ms-epoch timestamp for the test bar so it lands
# inside the test's date window regardless of clock drift.
_SAMPLE_TS = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
SAMPLE_BAR = {
    "v": 1234,
    "o": 100.0,
    "h": 101.5,
    "l": 99.5,
    "c": 100.75,
    "t": int(_SAMPLE_TS.timestamp() * 1000),
}


def _payload(rows: list[dict], next_url: str | None = None) -> dict:
    out = {"results": rows, "status": "OK"}
    if next_url:
        out["next_url"] = next_url
    return out


def test_fetch_ohlcv_maps_polygon_columns_and_tz() -> None:
    client = _StubClient([_payload([SAMPLE_BAR])])
    adapter = MassiveAdapter(api_key="test", http_client=client)
    df = adapter.fetch_ohlcv(
        "AAPL", date(2024, 1, 1), date(2024, 1, 2), interval="15m"
    )
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.iloc[0]["Open"] == 100.0
    assert df.iloc[0]["Close"] == 100.75
    assert df.iloc[0]["Volume"] == 1234
    # tz contract: NY local (matches yfinance adapter).
    assert str(df.index.tz) == "America/New_York"


def test_fetch_ohlcv_url_encodes_interval() -> None:
    """15m → multiplier=15, timespan=minute. URL must match Polygon spec."""
    client = _StubClient([_payload([SAMPLE_BAR])])
    adapter = MassiveAdapter(
        api_key="k", base_url="https://api.example.com", http_client=client
    )
    adapter.fetch_ohlcv(
        "AAPL", date(2024, 1, 1), date(2024, 1, 2), interval="15m"
    )
    url, params = client.calls[0]
    assert url == (
        "https://api.example.com/v2/aggs/ticker/AAPL/range/15/minute/"
        "2024-01-01/2024-01-03"  # end + 1 day pad (trimmed in df)
    )
    assert params["apiKey"] == "k"
    assert params["adjusted"] == "true"
    assert params["sort"] == "asc"


def test_fetch_ohlcv_paginates_via_next_url() -> None:
    bar2 = dict(SAMPLE_BAR, t=SAMPLE_BAR["t"] + 900_000, c=200.0)  # +15min
    client = _StubClient(
        [
            _payload([SAMPLE_BAR], next_url="https://api.x/next?cursor=abc"),
            _payload([bar2]),
        ]
    )
    adapter = MassiveAdapter(api_key="k", http_client=client)
    df = adapter.fetch_ohlcv(
        "AAPL", date(2024, 1, 1), date(2024, 1, 2), interval="15m"
    )
    assert len(df) == 2
    assert df.iloc[1]["Close"] == 200.0
    # Second call uses the cursor URL with only the apiKey appended.
    assert client.calls[1][0] == "https://api.x/next?cursor=abc"
    assert client.calls[1][1] == {"apiKey": "k"}


def test_fetch_ohlcv_raises_on_empty_results() -> None:
    client = _StubClient([_payload([])])
    adapter = MassiveAdapter(api_key="k", http_client=client)
    with pytest.raises(ValueError, match="No data found"):
        adapter.fetch_ohlcv(
            "AAPL", date(2024, 1, 1), date(2024, 1, 2), interval="15m"
        )


def test_unsupported_interval_raises() -> None:
    adapter = MassiveAdapter(api_key="k", http_client=_StubClient([]))
    with pytest.raises(ValueError, match="does not map interval"):
        adapter.fetch_ohlcv(
            "AAPL", date(2024, 1, 1), date(2024, 1, 2), interval="3m"
        )


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
        MassiveAdapter()
