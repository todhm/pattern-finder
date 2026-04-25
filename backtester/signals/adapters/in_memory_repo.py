from __future__ import annotations

from copy import deepcopy

from signals.domain.models import BuySignal, SignalStatus
from signals.domain.ports import SignalRepositoryPort


class InMemorySignalRepo(SignalRepositoryPort):
    """Process-memory signal store — placeholder for a real DB adapter.

    Useful for the MVP so the UI can save / list / annotate without
    setting up Postgres. Data does NOT persist across container
    restarts or Streamlit reloads. When a durable adapter is added
    (e.g. ``PostgresSignalRepo``), swap the instance in the composition
    root — no UI code changes required.
    """

    def __init__(self) -> None:
        self._store: dict[str, BuySignal] = {}

    def save(self, signal: BuySignal) -> BuySignal:
        self._store[signal.id] = deepcopy(signal)
        return signal

    def list(
        self,
        status: SignalStatus | None = None,
        interval: str | None = None,
    ) -> list[BuySignal]:
        items = list(self._store.values())
        if status is not None:
            items = [s for s in items if s.status == status]
        if interval is not None:
            items = [s for s in items if s.interval == interval]
        items.sort(key=lambda s: (s.signal_date, s.created_at), reverse=True)
        return items

    def get(self, signal_id: str) -> BuySignal | None:
        found = self._store.get(signal_id)
        return deepcopy(found) if found else None

    def update_status(self, signal_id: str, status: SignalStatus) -> None:
        if signal_id in self._store:
            self._store[signal_id].status = status

    def update_notes(self, signal_id: str, notes: str) -> None:
        if signal_id in self._store:
            self._store[signal_id].notes = notes

    def delete(self, signal_id: str) -> None:
        self._store.pop(signal_id, None)
