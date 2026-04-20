"""SQLAlchemy ORM mapping for :class:`BuySignal`.

Kept separate from the domain dataclass so the domain stays
framework-agnostic. Conversion helpers live on the ORM row class.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Date, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base
from signals.domain.models import BuySignal, SignalStatus


class BuySignalRow(Base):
    """``buy_signals`` table row — maps 1:1 to :class:`BuySignal`."""

    __tablename__ = "buy_signals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    pattern_name: Mapped[str] = mapped_column(String, nullable=False)
    entry_price: Mapped[float] = mapped_column(nullable=False)
    stop_loss: Mapped[float] = mapped_column(nullable=False)
    # Postgres JSONB so arbitrary metadata keys persist without
    # column churn — filter gates, stop/TP levels, refreshed_at.
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("idx_buy_signals_signal_date", "signal_date"),
        Index("idx_buy_signals_status", "status"),
        Index("idx_buy_signals_ticker", "ticker"),
    )

    # ---- conversion ----

    def to_domain(self) -> BuySignal:
        return BuySignal(
            id=self.id,
            ticker=self.ticker,
            signal_date=self.signal_date,
            pattern_name=self.pattern_name,
            entry_price=float(self.entry_price),
            stop_loss=float(self.stop_loss),
            metadata=self.metadata_json or {},
            status=SignalStatus(self.status),
            notes=self.notes or "",
            created_at=self.created_at,
        )

    @classmethod
    def from_domain(cls, sig: BuySignal) -> "BuySignalRow":
        return cls(
            id=sig.id,
            ticker=sig.ticker,
            signal_date=sig.signal_date,
            pattern_name=sig.pattern_name,
            entry_price=sig.entry_price,
            stop_loss=sig.stop_loss,
            metadata_json=sig.metadata or {},
            status=sig.status.value,
            notes=sig.notes or "",
            created_at=sig.created_at or datetime.utcnow(),
        )
