"""Postgres-backed :class:`SignalRepositoryPort` via SQLAlchemy.

Schema changes go through Alembic (``alembic revision --autogenerate``
reads ``signals.adapters.orm.BuySignalRow`` via the shared
``db.base.Base`` metadata). The adapter itself is pure persistence —
no DDL, no migrations.
"""

from __future__ import annotations

from sqlalchemy import select

from db.session import session_scope
from signals.adapters.orm import BuySignalRow
from signals.domain.models import BuySignal, SignalStatus
from signals.domain.ports import SignalRepositoryPort


class PostgresSignalRepo(SignalRepositoryPort):
    """Durable signal repository. Connection pool is module-level in
    ``db.session`` so repeated instantiation is cheap. Each method
    opens a short-lived session — no long-running transactions."""

    # ---- port implementation ----

    def save(self, signal: BuySignal) -> BuySignal:
        """Insert-or-update by ``signal.id``."""
        row = BuySignalRow.from_domain(signal)
        with session_scope() as s:
            s.merge(row)
            s.commit()
        return signal

    def list(self, status: SignalStatus | None = None) -> list[BuySignal]:
        stmt = select(BuySignalRow)
        if status is not None:
            stmt = stmt.where(BuySignalRow.status == status.value)
        stmt = stmt.order_by(
            BuySignalRow.signal_date.desc(),
            BuySignalRow.created_at.desc(),
        )
        with session_scope() as s:
            rows = s.scalars(stmt).all()
            return [r.to_domain() for r in rows]

    def get(self, signal_id: str) -> BuySignal | None:
        with session_scope() as s:
            row = s.get(BuySignalRow, signal_id)
            return row.to_domain() if row else None

    def update_status(self, signal_id: str, status: SignalStatus) -> None:
        with session_scope() as s:
            row = s.get(BuySignalRow, signal_id)
            if row is not None:
                row.status = status.value
                s.commit()

    def update_notes(self, signal_id: str, notes: str) -> None:
        with session_scope() as s:
            row = s.get(BuySignalRow, signal_id)
            if row is not None:
                row.notes = notes or ""
                s.commit()

    def delete(self, signal_id: str) -> None:
        with session_scope() as s:
            row = s.get(BuySignalRow, signal_id)
            if row is not None:
                s.delete(row)
                s.commit()
