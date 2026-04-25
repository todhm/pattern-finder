from __future__ import annotations

from abc import ABC, abstractmethod

from signals.domain.models import BuySignal, SignalStatus


class SignalRepositoryPort(ABC):
    """Port: persist and manage buy signals.

    Adapters: in-memory (MVP), Postgres (planned), JSON file, etc.
    Keeping the port minimal — save / list / get / status-update /
    notes / delete — lets the UI layer stay agnostic to storage.
    """

    @abstractmethod
    def save(self, signal: BuySignal) -> BuySignal:
        """Persist a signal. Idempotent on ``signal.id``."""

    @abstractmethod
    def list(
        self,
        status: SignalStatus | None = None,
        interval: str | None = None,
    ) -> list[BuySignal]:
        """Return signals, optionally filtered by status and bar
        interval, newest first.

        ``interval=None`` returns every cadence (back-compat for
        pages that pre-date the 15m work). Pass e.g. ``"1d"`` to
        show only daily signals on the legacy watchlist, ``"15m"``
        on intraday pages.
        """

    @abstractmethod
    def get(self, signal_id: str) -> BuySignal | None:
        """Return signal by id or ``None`` if missing."""

    @abstractmethod
    def update_status(self, signal_id: str, status: SignalStatus) -> None:
        """Mutate status (no-op if id missing)."""

    @abstractmethod
    def update_notes(self, signal_id: str, notes: str) -> None:
        """Replace notes (no-op if id missing)."""

    @abstractmethod
    def delete(self, signal_id: str) -> None:
        """Remove signal (no-op if id missing)."""


class SignalScannerPort(ABC):
    """Port: discover current buy signals across a ticker universe.

    A scanner runs the pattern detector over recent bars of every
    ticker in a named universe and returns the signals that survive
    the configured entry-time filters (those evaluable at signal-bar
    close — filters requiring tomorrow's open are deferred to
    execution time).
    """

    @abstractmethod
    def scan(
        self,
        universe: str,
        lookback_days: int,
        max_tickers: int | None = None,
    ) -> list[BuySignal]:
        """Return recent buy signals, newest first."""
