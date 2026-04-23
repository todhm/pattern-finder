"""Unit tests for NasdaqTraderUniverseAdapter parser + composite."""

import pytest

from data.adapters.wikipedia_universe import (
    CompositeUniverseAdapter,
    NasdaqTraderUniverseAdapter,
    StaticUniverseAdapter,
)


SAMPLE = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
    "Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n"
    "QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\n"
    "ZTEST|Test Ticker - Common Stock|G|Y|N|100|N|N\n"
    "BABA|Alibaba Group - American Depositary Shares|Q|N|N|100|N|N\n"
    "BRK.B|Berkshire Hathaway - Common Stock|Q|N|N|100|N|N\n"
    "File Creation Time: 0423202609:00|||||||\n"
)


def test_parse_common_only_filters_etfs_tests_adrs():
    tickers = NasdaqTraderUniverseAdapter._parse(SAMPLE, common_only=True)
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "QQQ" not in tickers       # ETF filtered
    assert "ZTEST" not in tickers     # test issue filtered
    assert "BABA" not in tickers      # ADR (not "Common Stock") filtered
    assert "BRK-B" in tickers         # dot-normalized


def test_parse_composite_keeps_adrs_but_drops_etfs_and_tests():
    tickers = NasdaqTraderUniverseAdapter._parse(SAMPLE, common_only=False)
    assert "AAPL" in tickers
    assert "BABA" in tickers          # ADR kept
    assert "QQQ" not in tickers       # ETF still filtered
    assert "ZTEST" not in tickers     # test still filtered


def test_handles_recognises_aliases():
    assert NasdaqTraderUniverseAdapter.handles("nasdaq_full")
    assert NasdaqTraderUniverseAdapter.handles("NASDAQ-ALL")
    assert NasdaqTraderUniverseAdapter.handles("nasdaq composite")
    assert NasdaqTraderUniverseAdapter.handles("nasdaq_common")
    assert not NasdaqTraderUniverseAdapter.handles("sp500")
    assert not NasdaqTraderUniverseAdapter.handles("nasdaq100")


def test_get_tickers_raises_for_unknown_key():
    adapter = NasdaqTraderUniverseAdapter()
    with pytest.raises(ValueError):
        adapter.get_tickers("sp500")


def test_composite_falls_through():
    static_nasdaq = StaticUniverseAdapter({"nasdaq_full": ["AAA", "BBB"]})
    static_sp = StaticUniverseAdapter({"sp500": ["XXX", "YYY", "ZZZ"]})
    comp = CompositeUniverseAdapter([static_nasdaq, static_sp])
    assert comp.get_tickers("nasdaq_full") == ["AAA", "BBB"]
    assert comp.get_tickers("sp500") == ["XXX", "YYY", "ZZZ"]


def test_composite_raises_when_no_adapter_handles():
    static = StaticUniverseAdapter({"sp500": ["A"]})
    comp = CompositeUniverseAdapter([static])
    with pytest.raises(ValueError):
        comp.get_tickers("mystery_universe")
