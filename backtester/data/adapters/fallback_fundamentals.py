"""Primary→fallback chain for FundamentalsPort.

Mirrors :class:`FallbackMarketDataAdapter` for OHLCV: try the primary,
on failure / missing-data response transparently route to the
fallback. Failure-mode signal is the *empty* TickerFundamentals
(both fields None) — different from raising so a partial primary
response (e.g. float ok but splits missing) still counts as success.
"""

from __future__ import annotations

import logging

from data.domain.ports import FundamentalsPort, TickerFundamentals

log = logging.getLogger(__name__)


class FallbackFundamentalsAdapter(FundamentalsPort):
    def __init__(
        self,
        primary: FundamentalsPort,
        fallback: FundamentalsPort,
        primary_label: str = "primary",
        fallback_label: str = "fallback",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_label = primary_label
        self._fallback_label = fallback_label

    def fetch(self, symbol: str) -> TickerFundamentals:
        try:
            primary_result = self._primary.fetch(symbol)
        except Exception as exc:
            log.warning(
                "Fundamentals %s primary (%s) raised: %s — using fallback",
                symbol, self._primary_label, exc,
            )
            primary_result = TickerFundamentals(
                symbol=symbol, float_shares=None, splits=None
            )

        # If primary returned nothing useful, hit the fallback.
        if primary_result.float_shares is None and primary_result.splits is None:
            try:
                return self._fallback.fetch(symbol)
            except Exception as exc:
                log.warning(
                    "Fundamentals %s fallback (%s) raised: %s",
                    symbol, self._fallback_label, exc,
                )
                return primary_result  # both empty

        # Primary partially succeeded. Fill missing pieces from
        # fallback so float-only responses still get splits.
        if primary_result.float_shares is None or primary_result.splits is None:
            try:
                fallback_result = self._fallback.fetch(symbol)
            except Exception:
                return primary_result
            return TickerFundamentals(
                symbol=symbol,
                float_shares=(
                    primary_result.float_shares
                    if primary_result.float_shares is not None
                    else fallback_result.float_shares
                ),
                splits=(
                    primary_result.splits
                    if primary_result.splits is not None
                    else fallback_result.splits
                ),
            )
        return primary_result
