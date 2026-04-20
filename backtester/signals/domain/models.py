from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class SignalStatus(str, Enum):
    """Lifecycle states for a scanned buy signal.

    ``pending`` — just discovered, not yet acted on.
    ``taken`` — user opened an actual position based on it.
    ``rejected`` — user manually dismissed (poor setup, wrong fit).
    ``expired`` — entry window passed (e.g. signal older than N bars).
    """

    PENDING = "pending"
    TAKEN = "taken"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class BuySignal:
    """A scanner-discovered buy opportunity the user may act on.

    ``metadata`` carries the detector + filter rationale
    (breakout strength, slopes, volume, filter-gate pass/fail map)
    so the UI can explain *why* this ticker appeared and the user can
    annotate/store the evidence alongside the signal.
    """

    ticker: str
    signal_date: date
    pattern_name: str
    entry_price: float
    stop_loss: float
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: SignalStatus = SignalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""
