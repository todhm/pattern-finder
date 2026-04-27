"""Unit tests for KrxWikipediaUniverseAdapter."""

import pytest

from data.adapters.wikipedia_universe import KrxWikipediaUniverseAdapter


SAMPLE_HTML = """
<html><body>
<table class="wikitable">
  <tr><th>Year</th><th>Closing level</th><th>Change in index</th></tr>
  <tr><td>2024</td><td>2,400</td><td>+5%</td></tr>
</table>
<table class="wikitable">
  <tr><th>Company</th><th>Symbol</th><th>GICSSector</th></tr>
  <tr><td>Samsung Electronics</td><td>005930</td><td>Information Technology</td></tr>
  <tr><td>SK Hynix</td><td>000660</td><td>Information Technology</td></tr>
  <tr><td>NAVER</td><td>035420</td><td>Communication Services</td></tr>
  <tr><td>Footnote row</td><td>123456[1]</td><td>Sector</td></tr>
  <tr><td>Bad row</td><td>NOTANUM</td><td>Sector</td></tr>
</table>
</body></html>
"""


class _StubResp:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        pass


class _StubClient:
    def __init__(self, text: str):
        self._text = text
        self.last_url: str | None = None

    def get(self, url: str):
        self.last_url = url
        return _StubResp(self._text)


def test_kospi200_extracts_six_digit_symbols_with_ks_suffix() -> None:
    adapter = KrxWikipediaUniverseAdapter(http_client=_StubClient(SAMPLE_HTML))
    tickers = adapter.get_tickers("kospi200")
    # Three valid symbols + the footnote row (which still has a 6-digit code).
    assert "005930.KS" in tickers
    assert "000660.KS" in tickers
    assert "035420.KS" in tickers
    assert "123456.KS" in tickers
    # Non-numeric junk dropped.
    assert "NOTANUM.KS" not in tickers
    # No prefix-stripped duplicates.
    assert all(t.endswith(".KS") for t in tickers)


def test_aliases_route_to_kospi() -> None:
    adapter = KrxWikipediaUniverseAdapter(http_client=_StubClient(SAMPLE_HTML))
    for alias in ["kospi", "kospi200", "kospi_200", "KOSPI-200"]:
        tickers = adapter.get_tickers(alias)
        assert tickers
        assert tickers[0].endswith(".KS")


def test_unknown_universe_raises() -> None:
    adapter = KrxWikipediaUniverseAdapter(http_client=_StubClient(SAMPLE_HTML))
    with pytest.raises(ValueError):
        adapter.get_tickers("kosdaq150")
    with pytest.raises(ValueError):
        adapter.get_tickers("sp500")


def test_first_wikitable_is_skipped_in_favor_of_constituents() -> None:
    """The page has an annual-levels table BEFORE the constituents
    table. The adapter must skip past the levels table (which has no
    ``Symbol`` header) and only return tickers from the proper
    constituents table."""
    adapter = KrxWikipediaUniverseAdapter(http_client=_StubClient(SAMPLE_HTML))
    tickers = adapter.get_tickers("kospi200")
    # ``2024`` from the levels table must NOT leak into the result.
    assert "2024.KS" not in tickers
    assert "2,400.KS" not in tickers
