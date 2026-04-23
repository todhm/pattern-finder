import httpx
from bs4 import BeautifulSoup

from data.domain.ports import UniverseProviderPort


class WikipediaUniverseAdapter(UniverseProviderPort):
    """Resolves index universes by scraping Wikipedia constituent tables.

    Supports:
        - ``sp500`` / ``s&p500`` — S&P 500 constituents (~500 tickers).
        - ``nasdaq100`` / ``nasdaq`` — Nasdaq-100 constituents (~100 tickers).

    Implementation: HTTP fetch via ``httpx`` + table parse via
    ``beautifulsoup4`` with the stdlib ``html.parser`` backend. This
    deliberately avoids ``pandas.read_html`` (and therefore the
    ``lxml`` / ``html5lib`` dependency) so the adapter works on any
    Python install with just the existing project deps.

    Wikipedia normalizes some tickers with dots (BRK.B, BF.B), but
    yfinance expects dashes (BRK-B, BF-B), so we normalize on the way
    out.
    """

    SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

    SP500_ALIASES = {"sp500", "s&p500", "sp_500", "s&p 500"}
    NASDAQ_ALIASES = {"nasdaq", "nasdaq100", "nasdaq_100", "nasdaq-100"}

    USER_AGENT = "Mozilla/5.0 (compatible; pattern-finder/1.0)"

    def __init__(self, http_client: httpx.Client | None = None):
        self._client = http_client or httpx.Client(
            headers={"User-Agent": self.USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )

    def get_tickers(self, universe: str) -> list[str]:
        key = universe.strip().lower()
        if key in self.SP500_ALIASES:
            return self._fetch_table(self.SP500_URL, ["Symbol", "Ticker"])
        if key in self.NASDAQ_ALIASES:
            return self._fetch_table(self.NASDAQ100_URL, ["Ticker", "Symbol"])
        raise ValueError(
            f"Unknown universe: {universe!r}. Expected one of "
            f"{sorted(self.SP500_ALIASES | self.NASDAQ_ALIASES)}"
        )

    # ---- internals ----

    def _fetch_table(
        self, url: str, header_candidates: list[str]
    ) -> list[str]:
        resp = self._client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for table in soup.find_all("table", class_="wikitable"):
            tickers = self._extract_column(table, header_candidates)
            if tickers:
                return [self._normalize(t) for t in tickers]
        raise RuntimeError(
            f"Could not locate a ticker column on {url}. "
            f"Tried headers={header_candidates}"
        )

    @staticmethod
    def _extract_column(table, header_candidates: list[str]) -> list[str]:
        """Pull the first column whose header matches a candidate name.

        Wikipedia constituent tables put the ticker in either the first
        or second column depending on the page, so we search by header
        text rather than hard-coding an index.
        """
        header_row = table.find("tr")
        if header_row is None:
            return []
        header_cells = header_row.find_all(["th", "td"])
        headers = [c.get_text(strip=True) for c in header_cells]

        col_idx: int | None = None
        for candidate in header_candidates:
            if candidate in headers:
                col_idx = headers.index(candidate)
                break
        if col_idx is None:
            return []

        tickers: list[str] = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if col_idx >= len(cells):
                continue
            text = cells[col_idx].get_text(strip=True)
            # Strip Wikipedia footnote markers like "AAPL[1]" or "BRK.B[a]"
            text = text.split("[", 1)[0].strip()
            if text and all(
                ch.isalnum() or ch in {".", "-"} for ch in text
            ):
                tickers.append(text)
        return tickers

    @staticmethod
    def _normalize(symbol: str) -> str:
        return symbol.replace(".", "-").strip().upper()


class NasdaqTraderUniverseAdapter(UniverseProviderPort):
    """Resolves Nasdaq-listed universes from nasdaqtrader.com.

    Data source: ``https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt``
    — official Nasdaq-published daily listing file. Pipe-delimited with
    columns:
        Symbol | Security Name | Market Category | Test Issue |
        Financial Status | Round Lot Size | ETF | NextShares

    Unlike Wikipedia (static index constituents), this covers the FULL
    Nasdaq cash-equity universe (~5,400 rows → ~2,200 common stocks
    after filtering) so downstream multi-ticker strategies can scan
    the broader Nasdaq tape, not just the Nasdaq-100.

    Aliases
        - ``nasdaq_full`` / ``nasdaq_all`` — all non-test, non-ETF
          securities, default filtered to common stock only.
        - ``nasdaq_common`` — explicit common-stock-only (same as
          default full).
        - ``nasdaq_composite`` — every non-test, non-ETF listing,
          including ADRs / units / warrants / preferreds. Noisier,
          but gives the full breadth.

    Filters applied
        - Drop rows with ``Test Issue = Y``.
        - Drop ETFs (``ETF = Y``) — those are price-index products,
          not the kind of single-name setups Wick Play targets.
        - ``common_stock_only`` (default True for _full/_all/_common)
          additionally requires ``"Common Stock"`` in the Security
          Name to exclude ADRs, depositary receipts, units, etc.
    """

    NASDAQ_LISTED_URL = (
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    )
    ALIASES_FULL = {"nasdaq_full", "nasdaq_all", "nasdaqfull", "nasdaqall"}
    ALIASES_COMMON = {"nasdaq_common", "nasdaqcommon"}
    ALIASES_COMPOSITE = {"nasdaq_composite", "nasdaqcomposite"}
    USER_AGENT = "Mozilla/5.0 (compatible; pattern-finder/1.0)"

    def __init__(self, http_client: httpx.Client | None = None):
        self._client = http_client or httpx.Client(
            headers={"User-Agent": self.USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )

    @classmethod
    def handles(cls, universe: str) -> bool:
        key = cls._normalize_key(universe)
        return (
            key in cls.ALIASES_FULL
            or key in cls.ALIASES_COMMON
            or key in cls.ALIASES_COMPOSITE
        )

    def get_tickers(self, universe: str) -> list[str]:
        key = self._normalize_key(universe)
        if key in self.ALIASES_COMPOSITE:
            return self._fetch(common_only=False)
        if key in self.ALIASES_FULL or key in self.ALIASES_COMMON:
            return self._fetch(common_only=True)
        raise ValueError(
            f"Unknown universe: {universe!r}. Expected one of "
            f"{sorted(self.ALIASES_FULL | self.ALIASES_COMMON | self.ALIASES_COMPOSITE)}"
        )

    # ---- internals ----

    @staticmethod
    def _normalize_key(universe: str) -> str:
        return (
            universe.strip().lower().replace("-", "_").replace(" ", "_")
        )

    def _fetch(self, common_only: bool) -> list[str]:
        resp = self._client.get(self.NASDAQ_LISTED_URL)
        resp.raise_for_status()
        return self._parse(resp.text, common_only=common_only)

    @staticmethod
    def _parse(text: str, common_only: bool) -> list[str]:
        """Parse the pipe-delimited listing file.

        The file ends with a ``File Creation Time`` footer that must
        be skipped. Exposed as a classmethod-style staticmethod so
        the parse logic can be unit-tested with a synthetic payload
        instead of hitting the network.
        """
        lines = text.strip().split("\n")
        out: list[str] = []
        for line in lines[1:]:  # skip header
            if line.startswith("File Creation Time"):
                break
            parts = line.split("|")
            if len(parts) < 8:
                continue
            symbol = parts[0].strip()
            name = parts[1].strip()
            test_issue = parts[3].strip()
            etf_flag = parts[6].strip()
            if not symbol:
                continue
            if test_issue == "Y":
                continue
            if etf_flag == "Y":
                continue
            if common_only and "Common Stock" not in name:
                continue
            # yfinance ticker convention: dots → dashes (BRK.B → BRK-B)
            out.append(symbol.replace(".", "-").upper())
        return out


class CompositeUniverseAdapter(UniverseProviderPort):
    """Tries each constituent adapter in order until one handles
    the requested universe. Pattern: declarative multi-source
    resolver without the caller having to know which adapter owns
    which universe key.
    """

    def __init__(self, adapters: list[UniverseProviderPort]):
        if not adapters:
            raise ValueError("CompositeUniverseAdapter needs at least one adapter")
        self._adapters = list(adapters)

    def get_tickers(self, universe: str) -> list[str]:
        last_err: Exception | None = None
        for adapter in self._adapters:
            try:
                return adapter.get_tickers(universe)
            except ValueError as e:
                last_err = e
                continue
        raise ValueError(
            f"No adapter accepted universe={universe!r}. "
            f"Last error: {last_err}"
        )


def default_universe_provider() -> UniverseProviderPort:
    """Factory: composite of NasdaqTrader + Wikipedia.

    NasdaqTrader is tried first so ``nasdaq_full`` / ``nasdaq_all``
    resolve there; everything else falls through to Wikipedia
    (``sp500``, ``nasdaq100``).
    """
    return CompositeUniverseAdapter(
        [NasdaqTraderUniverseAdapter(), WikipediaUniverseAdapter()]
    )


class StaticUniverseAdapter(UniverseProviderPort):
    """In-memory universe provider for tests and ad-hoc ticker lists.

    Maps universe names to fixed ticker lists supplied at construction.
    """

    def __init__(self, mapping: dict[str, list[str]]):
        self._mapping = {k.lower(): list(v) for k, v in mapping.items()}

    def get_tickers(self, universe: str) -> list[str]:
        key = universe.strip().lower()
        if key not in self._mapping:
            raise ValueError(
                f"Unknown universe: {universe!r}. Known: {sorted(self._mapping)}"
            )
        return list(self._mapping[key])
