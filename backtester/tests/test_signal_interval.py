"""Interval-aware signal persistence.

Covers the Phase-4 contract on the in-memory repo (the Postgres
adapter shares the filter logic — its DB integration is out of scope
for unit tests):

1. ``BuySignal`` defaults preserve pre-existing behavior — unset
   ``interval`` stays ``"1d"`` and ``signal_datetime`` stays ``None``.
2. ``repo.list(interval=...)`` filters by cadence so daily and
   intraday watchlists stay separated in the same table.
3. ORM round-trip (domain → row → domain) preserves ``interval``
   and ``signal_datetime`` end-to-end.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from signals.adapters.in_memory_repo import InMemorySignalRepo
from signals.adapters.orm import BuySignalRow
from signals.domain.models import BuySignal, SignalStatus


def _daily_signal(ticker: str = "AAPL") -> BuySignal:
    return BuySignal(
        ticker=ticker,
        signal_date=date(2024, 3, 18),
        pattern_name="wedge_pop",
        entry_price=100.0,
        stop_loss=99.0,
    )


def _intraday_signal(ticker: str = "AAPL") -> BuySignal:
    return BuySignal(
        ticker=ticker,
        signal_date=date(2024, 3, 18),
        pattern_name="wedge_pop",
        entry_price=100.0,
        stop_loss=99.0,
        interval="15m",
        signal_datetime=datetime(2024, 3, 18, 10, 15, tzinfo=timezone.utc),
    )


def test_daily_signal_defaults_remain_unchanged() -> None:
    """Untouched callers must keep producing 1d signals — this is the
    whole safety net behind adding ``interval`` as a default-valued
    field rather than a required one."""
    s = _daily_signal()
    assert s.interval == "1d"
    assert s.signal_datetime is None


def test_repo_filters_by_interval() -> None:
    repo = InMemorySignalRepo()
    repo.save(_daily_signal("AAPL"))
    repo.save(_intraday_signal("AAPL"))
    repo.save(_intraday_signal("MSFT"))

    assert {s.ticker for s in repo.list(interval="1d")} == {"AAPL"}
    assert {s.ticker for s in repo.list(interval="15m")} == {"AAPL", "MSFT"}
    # No filter → both cadences returned (back-compat for legacy pages).
    assert len(repo.list()) == 3


def test_repo_combines_status_and_interval_filters() -> None:
    repo = InMemorySignalRepo()
    pending_daily = _daily_signal("AAPL")
    taken_intraday = _intraday_signal("MSFT")
    taken_intraday.status = SignalStatus.TAKEN
    repo.save(pending_daily)
    repo.save(taken_intraday)

    pending_intraday = repo.list(status=SignalStatus.PENDING, interval="15m")
    assert pending_intraday == []

    taken_intraday_only = repo.list(status=SignalStatus.TAKEN, interval="15m")
    assert [s.ticker for s in taken_intraday_only] == ["MSFT"]


def test_orm_round_trip_preserves_interval_fields() -> None:
    intraday = _intraday_signal()
    row = BuySignalRow.from_domain(intraday)
    back = row.to_domain()
    assert back.interval == "15m"
    assert back.signal_datetime == intraday.signal_datetime

    daily = _daily_signal()
    row = BuySignalRow.from_domain(daily)
    back = row.to_domain()
    assert back.interval == "1d"
    assert back.signal_datetime is None
