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

    def test_rejects_unknown_interval(self):
        adapter, _ = _adapter([])
        with pytest.raises(ValueError):
            adapter.fetch_ohlcv("TQQQ", date(2024, 1, 1), date(2024, 1, 31), interval="1wk")


def _daily_payload(rows: dict[str, tuple]) -> dict:
    """{날짜: (close, adjusted_close, volume, split_coefficient)}"""
    return {
        "Meta Data": {},
        "Time Series (Daily)": {
            d: {"1. open": str(c), "2. high": str(c + 2), "3. low": str(c - 2),
                "4. close": str(c), "5. adjusted close": str(adj),
                "6. volume": str(v), "7. dividend amount": "0.0",
                "8. split coefficient": str(sp)}
            for d, (c, adj, v, sp) in rows.items()
        },
    }


class TestDaily:
    def test_one_call_full_history_split_adjusted(self):
        # 2024-06-10 에 2:1 분할 — 이전 날의 raw 가격은 2배, 거래량은 절반
        adapter, client = _adapter([_StubResp(200, _daily_payload({
            "2024-06-06": (200.0, 100.0, 1000, 1.0),
            "2024-06-07": (202.0, 101.0, 1000, 1.0),
            "2024-06-10": (102.0, 102.0, 2000, 2.0),
            "2024-06-11": (103.0, 103.0, 2000, 1.0),
        }))])
        df = adapter.fetch_ohlcv("NVDA", date(2024, 6, 7), date(2024, 6, 10))
        assert len(client.calls) == 1
        assert client.calls[0]["function"] == "TIME_SERIES_DAILY_ADJUSTED"
        assert client.calls[0]["outputsize"] == "full"
        assert list(df.index.date) == [date(2024, 6, 7), date(2024, 6, 10)]
        assert str(df.index.tz) == "America/New_York"
        row = df.iloc[0]
        assert row["Close"] == pytest.approx(101.0)
        assert row["High"] == pytest.approx(204.0 * 0.5)       # OHLC 도 조정
        assert row["Volume"] == pytest.approx(2000.0)          # 분할 전 거래량 × 2
        assert df.iloc[1]["Volume"] == pytest.approx(2000.0)   # 분할 당일은 그대로

    def test_unmarked_information_note_is_retried(self):
        adapter, client = _adapter([
            _StubResp(200, {"Information": "Thank you for using Alpha Vantage!"}),
            _StubResp(200, _daily_payload({"2024-06-06": (100.0, 100.0, 1000, 1.0)})),
        ])
        df = adapter.fetch_ohlcv("X", date(2024, 6, 1), date(2024, 6, 30))
        assert len(client.calls) == 2 and len(df) == 1

    def test_empty_range_raises(self):
        adapter, _ = _adapter([_StubResp(200, _daily_payload({
            "2024-06-06": (100.0, 100.0, 1000, 1.0)}))])
        with pytest.raises(ValueError):
            adapter.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 31))


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
