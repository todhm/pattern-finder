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
