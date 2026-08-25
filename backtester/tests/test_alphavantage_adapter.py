"""Unit tests for AlphaVantageAdapter."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.adapters.alphavantage_adapter import AlphaVantageAdapter


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
        self.calls: list[dict] = []

    def get(self, url, params=None):
        self.calls.append(dict(params or {}))
        return self._responses.pop(0)


def _month_payload(ts_prices: dict[str, float]) -> dict:
    return {
        "Meta Data": {},
        "Time Series (15min)": {
            ts: {
                "1. open": str(p),
                "2. high": str(p + 1),
                "3. low": str(p - 1),
                "4. close": str(p + 0.5),
                "5. volume": "1000",
            }
            for ts, p in ts_prices.items()
        },
    }


def _adapter(responses, **kw) -> tuple[AlphaVantageAdapter, _StubClient]:
    client = _StubClient(responses)
    return (
        AlphaVantageAdapter(
            api_key="test", http_client=client, rate_retry_sleep=0.0, **kw
        ),
        client,
    )


class TestSetup:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AlphaVantageAdapter()

    def test_rejects_daily_interval(self):
        adapter, _ = _adapter([])
        with pytest.raises(ValueError):
            adapter.fetch_ohlcv("TQQQ", date(2024, 1, 1), date(2024, 1, 31))


class TestFetch:
    def test_one_call_per_month_and_et_index(self):
        adapter, client = _adapter(
            [
                _StubResp(200, _month_payload({"2024-01-16 09:30:00": 100.0})),
                _StubResp(200, _month_payload({"2024-02-01 10:00:00": 110.0})),
            ]
        )
        df = adapter.fetch_ohlcv(
            "TQQQ", date(2024, 1, 10), date(2024, 2, 10), interval="15m"
        )
        assert len(client.calls) == 2
        assert [c["month"] for c in client.calls] == ["2024-01", "2024-02"]
        assert client.calls[0]["interval"] == "15min"
        assert len(df) == 2
        assert str(df.index.tz) == "America/New_York"
        assert df.iloc[0]["Close"] == pytest.approx(100.5)

    def test_slices_to_requested_range(self):
        adapter, _ = _adapter(
            [
                _StubResp(
                    200,
                    _month_payload(
                        {
                            "2024-01-05 09:30:00": 90.0,  # start 이전 — 제외
                            "2024-01-16 09:30:00": 100.0,
                        }
                    ),
                )
            ]
        )
        df = adapter.fetch_ohlcv(
            "TQQQ", date(2024, 1, 10), date(2024, 1, 31), interval="15m"
        )
        assert len(df) == 1

    def test_rate_limit_retries_then_succeeds(self):
        adapter, client = _adapter(
            [
                _StubResp(200, {"Information": "API rate limit exceeded"}),
                _StubResp(200, _month_payload({"2024-01-16 09:30:00": 100.0})),
            ]
        )
        df = adapter.fetch_ohlcv(
            "TQQQ", date(2024, 1, 10), date(2024, 1, 31), interval="15m"
        )
        assert len(client.calls) == 2
        assert len(df) == 1

    def test_rate_limit_exhausted_raises(self):
        adapter, _ = _adapter(
            [_StubResp(200, {"Note": "rate limit"})] * 3,
            max_rate_retries=2,
        )
        with pytest.raises(RuntimeError):
            adapter.fetch_ohlcv(
                "TQQQ", date(2024, 1, 10), date(2024, 1, 31), interval="15m"
            )

    def test_empty_month_returns_empty(self):
        # 상장 이전 달 — 마커 없는 빈 응답은 조용히 빈 프레임.
        adapter, _ = _adapter([_StubResp(200, {"Meta Data": {}})])
        df = adapter.fetch_ohlcv(
            "TQQQ", date(2024, 1, 10), date(2024, 1, 31), interval="15m"
        )
        assert df.empty

    def test_error_message_raises(self):
        adapter, _ = _adapter(
            [_StubResp(200, {"Error Message": "Invalid API call"})]
        )
        with pytest.raises(ValueError):
            adapter.fetch_ohlcv(
                "TQQQ", date(2024, 1, 10), date(2024, 1, 31), interval="15m"
            )
