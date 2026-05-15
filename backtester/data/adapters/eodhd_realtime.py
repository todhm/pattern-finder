"""EODHD ``/api/real-time/`` adapter — live current-session snapshot.

Distinct from :mod:`data.adapters.eodhd_adapter` (`/api/intraday/` →
historical 1m/5m bars) because the real-time endpoint:

    - Returns a **single quote**, not a DataFrame of bars.
    - Is **available on this tier today** (intraday is lagged ~1 day
      on the current EODHD subscription).
    - Supports a **multi-symbol bulk fetch** via the ``s=`` query
      param — one HTTP per ~100 tickers, ideal for the Bull Flag
      live monitor where the watchlist is small.

Used **only** by Bull Flag pages (21 / 22 / 23). Not wired into the
composed market-data factory so other pages keep going through the
bars-oriented routing they expect.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx

from data.adapters.eodhd_adapter import EODHDAdapter
from data.domain.ports import RealtimeQuote, RealtimeQuotePort


class EODHDRealtimeAdapter(RealtimeQuotePort):
    """Live snapshot fetcher backed by EODHD ``/api/real-time/{symbol}``.

    Single-symbol GET returns a JSON object; multi-symbol via ``s=``
    returns a JSON array (one element per ticker, including the
    primary in the URL path). We normalize both shapes here so the
    caller always gets ``dict[symbol → RealtimeQuote]``.
    """

    DEFAULT_BASE_URL = "https://eodhd.com/api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("EODHD_API_KEY")
        if not self._api_key:
            raise ValueError(
                "EODHDRealtimeAdapter requires the EODHD_API_KEY env var "
                "(loaded from secrets/secret.env) or an explicit api_key."
            )
        self._base_url = (
            base_url
            or os.environ.get("EODHD_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=timeout_seconds)

    # ---- port API --------------------------------------------------

    def fetch_quote(self, symbol: str) -> RealtimeQuote:
        eodhd_sym = EODHDAdapter._normalize_symbol(symbol)
        url = f"{self._base_url}/real-time/{eodhd_sym}"
        resp = self._client.get(
            url, params={"api_token": self._api_key, "fmt": "json"}
        )
        if resp.status_code != 200:
            raise ValueError(
                f"EODHD real-time {symbol}: "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )
        return self._parse_row(resp.json(), symbol)

    def fetch_quotes(
        self, symbols: list[str]
    ) -> dict[str, RealtimeQuote]:
        """Bulk fetch using ``/real-time/{primary}?s=rest`` so a
        watchlist of N tickers costs only ⌈N/MAX_BULK⌉ HTTPs.

        EODHD's docs cap the ``s=`` list at ~100 symbols per call;
        we chunk above that. Symbols that fail to parse / aren't
        returned by EODHD are silently dropped from the result —
        callers get a partial dict, not an exception.
        """
        if not symbols:
            return {}
        out: dict[str, RealtimeQuote] = {}
        # Pre-build symbol-mapping so ``code`` field on EODHD response
        # rows can be reverse-mapped back to user's input casing.
        sym_map = {
            EODHDAdapter._normalize_symbol(s): s for s in symbols
        }
        chunk_size = 50  # well under EODHD's 100-symbol limit
        chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        for chunk in chunks:
            try:
                rows = self._fetch_chunk(chunk)
            except Exception:
                # Whole-chunk failure — leave those symbols missing.
                continue
            for row in rows:
                code = row.get("code")
                user_sym = sym_map.get(code)
                if not user_sym:
                    continue
                try:
                    out[user_sym] = self._parse_row(row, user_sym)
                except Exception:
                    continue
        return out

    # ---- internals --------------------------------------------------

    def _fetch_chunk(self, symbols: list[str]) -> list[dict]:
        """Single HTTP for up to ~100 symbols via ``/real-time/{p}?s=...``."""
        eodhd_syms = [EODHDAdapter._normalize_symbol(s) for s in symbols]
        primary, rest = eodhd_syms[0], eodhd_syms[1:]
        url = f"{self._base_url}/real-time/{primary}"
        params: dict[str, str | int] = {
            "api_token": self._api_key,
            "fmt": "json",
        }
        if rest:
            params["s"] = ",".join(rest)
        resp = self._client.get(url, params=params)
        if resp.status_code != 200:
            raise ValueError(
                f"EODHD real-time bulk: HTTP {resp.status_code} "
                f"{resp.text[:200]}"
            )
        payload = resp.json()
        # Single-symbol path returns a dict; multi returns a list.
        if isinstance(payload, dict):
            return [payload]
        return payload if isinstance(payload, list) else []

    @staticmethod
    def _parse_row(row: dict, user_symbol: str) -> RealtimeQuote:
        """EODHD field → :class:`RealtimeQuote`. Skips quotes with
        zero ``timestamp`` (= "N/A" stub EODHD returns for tickers
        without same-day prints)."""
        ts = row.get("timestamp")
        if not isinstance(ts, (int, float)) or ts <= 0:
            raise ValueError(f"{user_symbol}: no timestamp in EODHD response")
        # EODHD sometimes returns string "NA" for prices on closed/halted
        # tickers — coerce defensively, raise on non-numeric.
        def _num(key: str, default: float = 0.0) -> float:
            v = row.get(key, default)
            if isinstance(v, str):
                if v.upper() in ("NA", "N/A", ""):
                    return default
                v = float(v)
            return float(v)

        return RealtimeQuote(
            symbol=user_symbol,
            timestamp=datetime.fromtimestamp(int(ts), tz=timezone.utc),
            open=_num("open"),
            high=_num("high"),
            low=_num("low"),
            last=_num("close"),  # /real-time/'s "close" = last print
            volume=int(row.get("volume") or 0),
            previous_close=_num("previousClose"),
            change=_num("change"),
            change_pct=_num("change_p"),
        )
