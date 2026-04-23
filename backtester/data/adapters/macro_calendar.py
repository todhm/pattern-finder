"""Hardcoded US macro-event calendar.

Entry-gate support for strategies that want to **blackout** days on
which index-wide macro prints routinely dominate idiosyncratic setup
behaviour (FOMC rate decisions, CPI releases).

Rationale — Wick Play post-mortem (2024-06 → 2026-04 live trades):
    - SHOP 2024-12-09 → 2024-12-18 (−$5,330): stopped out on the
      FOMC "hawkish cut" day. SPY −3% intraday on the Dot Plot
      surprise; the Wick Play structure was fine, the macro tape
      killed it.
    - Not all losses come from this — REGN 2025-02-21 breakeven
      was a broad-market regime issue, not an event — but FOMC/CPI
      days are the highest-leverage single-day blackouts.

Dates are the **announcement day** (end of 2-day FOMC meetings, day
CPI prints pre-market). The caller decides whether to extend the
window (e.g. ``±1 day``) in its own filter logic.

No external API — curated static lists. Update this file when new
years are added. Range currently covers 2017-01 through 2026-12
(2017-2022 added 2026-04 after discovering multi-year S&P 500
backtests had zero blackout coverage prior to 2023).
"""

from datetime import date, timedelta


# FOMC rate-decision dates (announcement day = end of 2-day meeting)
_FOMC_DATES: tuple[date, ...] = (
    # 2017
    date(2017, 2, 1),
    date(2017, 3, 15),
    date(2017, 5, 3),
    date(2017, 6, 14),
    date(2017, 7, 26),
    date(2017, 9, 20),
    date(2017, 11, 1),
    date(2017, 12, 13),
    # 2018
    date(2018, 1, 31),
    date(2018, 3, 21),
    date(2018, 5, 2),
    date(2018, 6, 13),
    date(2018, 8, 1),
    date(2018, 9, 26),
    date(2018, 11, 8),
    date(2018, 12, 19),
    # 2019
    date(2019, 1, 30),
    date(2019, 3, 20),
    date(2019, 5, 1),
    date(2019, 6, 19),
    date(2019, 7, 31),
    date(2019, 9, 18),
    date(2019, 10, 30),
    date(2019, 12, 11),
    # 2020
    date(2020, 1, 29),
    date(2020, 3, 3),   # emergency cut #1 (intermeeting)
    date(2020, 3, 15),  # emergency cut #2 (intermeeting)
    date(2020, 4, 29),
    date(2020, 6, 10),
    date(2020, 7, 29),
    date(2020, 9, 16),
    date(2020, 11, 5),
    date(2020, 12, 16),
    # 2021
    date(2021, 1, 27),
    date(2021, 3, 17),
    date(2021, 4, 28),
    date(2021, 6, 16),
    date(2021, 7, 28),
    date(2021, 9, 22),
    date(2021, 11, 3),
    date(2021, 12, 15),
    # 2022
    date(2022, 1, 26),
    date(2022, 3, 16),
    date(2022, 5, 4),
    date(2022, 6, 15),
    date(2022, 7, 27),
    date(2022, 9, 21),
    date(2022, 11, 2),
    date(2022, 12, 14),
    # 2023
    date(2023, 2, 1),
    date(2023, 3, 22),
    date(2023, 5, 3),
    date(2023, 6, 14),
    date(2023, 7, 26),
    date(2023, 9, 20),
    date(2023, 11, 1),
    date(2023, 12, 13),
    # 2024
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
)


# BLS CPI release dates (pre-market, typically 8:30 AM ET)
_CPI_DATES: tuple[date, ...] = (
    # 2017
    date(2017, 1, 18), date(2017, 2, 15), date(2017, 3, 15),
    date(2017, 4, 14), date(2017, 5, 12), date(2017, 6, 14),
    date(2017, 7, 14), date(2017, 8, 11), date(2017, 9, 14),
    date(2017, 10, 13), date(2017, 11, 15), date(2017, 12, 13),
    # 2018
    date(2018, 1, 12), date(2018, 2, 14), date(2018, 3, 13),
    date(2018, 4, 11), date(2018, 5, 10), date(2018, 6, 12),
    date(2018, 7, 12), date(2018, 8, 10), date(2018, 9, 13),
    date(2018, 10, 11), date(2018, 11, 14), date(2018, 12, 12),
    # 2019
    date(2019, 2, 13), date(2019, 3, 12),
    date(2019, 4, 10), date(2019, 5, 10), date(2019, 6, 12),
    date(2019, 7, 11), date(2019, 8, 13), date(2019, 9, 12),
    date(2019, 10, 10), date(2019, 11, 13), date(2019, 12, 11),
    # 2020
    date(2020, 1, 14), date(2020, 2, 13), date(2020, 3, 11),
    date(2020, 4, 10), date(2020, 5, 12), date(2020, 6, 10),
    date(2020, 7, 14), date(2020, 8, 12), date(2020, 9, 11),
    date(2020, 10, 13), date(2020, 11, 12), date(2020, 12, 10),
    # 2021
    date(2021, 1, 13), date(2021, 2, 10), date(2021, 3, 10),
    date(2021, 4, 13), date(2021, 5, 12), date(2021, 6, 10),
    date(2021, 7, 13), date(2021, 8, 11), date(2021, 9, 14),
    date(2021, 10, 13), date(2021, 11, 10), date(2021, 12, 10),
    # 2022
    date(2022, 1, 12), date(2022, 2, 10), date(2022, 3, 10),
    date(2022, 4, 12), date(2022, 5, 11), date(2022, 6, 10),
    date(2022, 7, 13), date(2022, 8, 10), date(2022, 9, 13),
    date(2022, 10, 13), date(2022, 11, 10), date(2022, 12, 13),
    # 2023
    date(2023, 1, 12), date(2023, 2, 14), date(2023, 3, 14),
    date(2023, 4, 12), date(2023, 5, 10), date(2023, 6, 13),
    date(2023, 7, 12), date(2023, 8, 10), date(2023, 9, 13),
    date(2023, 10, 12), date(2023, 11, 14), date(2023, 12, 12),
    # 2024
    date(2024, 1, 11), date(2024, 2, 13), date(2024, 3, 12),
    date(2024, 4, 10), date(2024, 5, 15), date(2024, 6, 12),
    date(2024, 7, 11), date(2024, 8, 14), date(2024, 9, 11),
    date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
    # 2025
    date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
    date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
    date(2025, 7, 15), date(2025, 8, 12), date(2025, 9, 11),
    date(2025, 10, 15), date(2025, 11, 13), date(2025, 12, 10),
    # 2026
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 14), date(2026, 5, 13), date(2026, 6, 10),
    date(2026, 7, 15), date(2026, 8, 12), date(2026, 9, 10),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 9),
)


def fomc_dates() -> tuple[date, ...]:
    """Known FOMC rate-decision announcement days."""
    return _FOMC_DATES


def cpi_dates() -> tuple[date, ...]:
    """Known US CPI release days (BLS, 8:30 AM ET)."""
    return _CPI_DATES


def blackout_dates(
    start: date,
    end: date,
    include_fomc: bool = True,
    include_cpi: bool = False,
    window_days: int = 0,
) -> set[date]:
    """Build a blackout date set covering [start, end].

    Args:
        start, end: inclusive date window.
        include_fomc: include FOMC announcement days.
        include_cpi: include CPI release days.
        window_days: expand each event ± this many days. ``0`` = event
            day only. ``1`` = event day plus the trading day before
            and after (calendar-day padding; callers typically only
            compare against trading-day indices, so this is fine).

    Returns:
        Set of dates. Empty set if no events configured or window
        excludes all.
    """
    events: list[date] = []
    if include_fomc:
        events.extend(_FOMC_DATES)
    if include_cpi:
        events.extend(_CPI_DATES)

    out: set[date] = set()
    for d in events:
        if d < start or d > end:
            continue
        if window_days <= 0:
            out.add(d)
            continue
        for offset in range(-window_days, window_days + 1):
            out.add(d + timedelta(days=offset))
    return out
