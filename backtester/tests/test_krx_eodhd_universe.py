"""Unit tests for KrxEodhdUniverseAdapter."""

from __future__ import annotations

import pytest

from data.adapters.wikipedia_universe import KrxEodhdUniverseAdapter


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


_KOSPI_PAGE = [
    {"Code": "005930", "Name": "Samsung Electronics", "Type": "Common Stock"},
    {"Code": "000660", "Name": "SK Hynix", "Type": "Common Stock"},
    # Mixed-in ETFs / preferred should be filtered out.
    {"Code": "069500", "Name": "KODEX 200 ETF", "Type": "ETF"},
    {"Code": "005935", "Name": "Samsung Pref", "Type": "Preferred Stock"},
    {"Code": "035720", "Name": "Kakao", "Type": "Common Stock"},
]

_KOSDAQ_PAGE = [
    {"Code": "247540", "Name": "Ecopro BM", "Type": "Common Stock"},
    {"Code": "086520", "Name": "Ecopro", "Type": "Common Stock"},
]


# ---- alias resolution -------------------------------------------------------


def test_kospi_full_returns_common_stock_only_with_ks_suffix() -> None:
    client = _StubClient([_StubResp(200, _KOSPI_PAGE)])
    a = KrxEodhdUniverseAdapter(api_key="k", http_client=client)
    tickers = a.get_tickers("kospi_full")
    assert tickers == ["005930.KS", "000660.KS", "035720.KS"]
    # ETFs / preferred dropped.
    assert "069500.KS" not in tickers
    assert "005935.KS" not in tickers


def test_kosdaq_full_uses_kq_suffix() -> None:
    client = _StubClient([_StubResp(200, _KOSDAQ_PAGE)])
    a = KrxEodhdUniverseAdapter(api_key="k", http_client=client)
    tickers = a.get_tickers("kosdaq_full")
    assert tickers == ["247540.KQ", "086520.KQ"]


def test_krx_all_combines_kospi_and_kosdaq() -> None:
    """Two API calls in order; KOSPI first then KOSDAQ."""
    client = _StubClient(
        [_StubResp(200, _KOSPI_PAGE), _StubResp(200, _KOSDAQ_PAGE)]
    )
    a = KrxEodhdUniverseAdapter(api_key="k", http_client=client)
    tickers = a.get_tickers("krx_all")
    assert tickers[:3] == ["005930.KS", "000660.KS", "035720.KS"]
    assert tickers[-2:] == ["247540.KQ", "086520.KQ"]


def test_aliases_are_normalized_case_insensitive() -> None:
    client = _StubClient([_StubResp(200, _KOSPI_PAGE)])
    a = KrxEodhdUniverseAdapter(api_key="k", http_client=client)
    # Spelled differently but resolves to the same alias bucket.
    tickers = a.get_tickers("KOSPI-FULL")
    assert "005930.KS" in tickers


def test_unknown_universe_raises() -> None:
    a = KrxEodhdUniverseAdapter(api_key="k", http_client=_StubClient([]))
    with pytest.raises(ValueError, match="Unknown universe"):
        a.get_tickers("sp500")


def test_handles_classmethod() -> None:
    assert KrxEodhdUniverseAdapter.handles("kospi_full")
    assert KrxEodhdUniverseAdapter.handles("KOSDAQ_full")
    assert KrxEodhdUniverseAdapter.handles("krx_all")
    assert not KrxEodhdUniverseAdapter.handles("kospi200")  # → wiki adapter
    assert not KrxEodhdUniverseAdapter.handles("sp500")


def test_url_path_uses_exchange_code() -> None:
    client = _StubClient([_StubResp(200, _KOSPI_PAGE)])
    a = KrxEodhdUniverseAdapter(api_key="k", base_url="https://x", http_client=client)
    a.get_tickers("kospi_full")
    url, params = client.calls[0]
    assert url == "https://x/exchange-symbol-list/KO"
    assert params["api_token"] == "k"
    assert params["fmt"] == "json"


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="EODHD_API_KEY"):
        KrxEodhdUniverseAdapter()


def test_http_error_surfaces() -> None:
    client = _StubClient([_StubResp(403, "API key required")])
    a = KrxEodhdUniverseAdapter(api_key="k", http_client=client)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        a.get_tickers("kospi_full")
