"""Pin the chart's intraday-aware trade marker / hover behavior.

Two guarantees:
1. ``build_candlestick_with_trades`` resolves a 15m trade's
   ``entry_ts``/``exit_ts`` onto the **exact** intraday bar — not the
   first bar of that session, which is what the legacy date-based
   resolver did.
2. The BUY/SELL hover text shows ``YYYY-MM-DD HH:MM`` for intraday
   trades and falls back to the date-only format for daily trades
   (so existing daily pages display unchanged).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from pattern.helpers.sessions import NY_TZ
from strategy.domain.models import Trade
from visualization.adapters.plotly_charts import PlotlyChartBuilder


def _intraday_df() -> pd.DataFrame:
    day = pd.Timestamp("2024-03-18", tz=NY_TZ)
    idx = pd.date_range(
        start=day.replace(hour=9, minute=30), periods=26, freq="15min", tz=NY_TZ
    )
    closes = [100 + i * 0.3 for i in range(26)]
    return pd.DataFrame(
        {
            "Open": [c - 0.05 for c in closes],
            "High": [c + 0.2 for c in closes],
            "Low": [c - 0.2 for c in closes],
            "Close": closes,
            "Volume": [1_000] * 26,
        },
        index=idx,
    )


def _daily_df() -> pd.DataFrame:
    idx = pd.date_range("2024-03-15", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Open": list(range(100, 110)),
            "High": [x + 1 for x in range(100, 110)],
            "Low": [x - 1 for x in range(100, 110)],
            "Close": [x + 0.5 for x in range(100, 110)],
            "Volume": [1_000] * 10,
        },
        index=idx,
    )


def _intraday_trade(entry_hh_mm, exit_hh_mm) -> Trade:
    tz = pd.Timestamp("now", tz=NY_TZ).tzinfo
    return Trade(
        pattern_name="wp",
        entry_date=date(2024, 3, 18),
        exit_date=date(2024, 3, 18),
        entry_price=100.5,
        exit_price=105.0,
        stop_loss=99.0,
        shares=100,
        pnl=450.0,
        pnl_pct=0.045,
        exit_reason="smart_trail",
        entry_ts=datetime(2024, 3, 18, *entry_hh_mm, tzinfo=tz),
        exit_ts=datetime(2024, 3, 18, *exit_hh_mm, tzinfo=tz),
    )


def _hover_for(fig, name: str) -> str:
    for tr in fig.data:
        if tr.name == name:
            return tr.hovertext[0]
    raise AssertionError(f"no {name} trace on figure")


def _x_for(fig, name: str):
    for tr in fig.data:
        if tr.name == name:
            return tr.x[0]
    raise AssertionError(f"no {name} trace on figure")


def test_intraday_trade_hover_includes_time() -> None:
    df = _intraday_df()
    trade = _intraday_trade((10, 30), (14, 45))
    fig = PlotlyChartBuilder().build_candlestick_with_trades(df, [trade])

    assert "2024-03-18 10:30" in _hover_for(fig, "Buy")
    assert "2024-03-18 14:45" in _hover_for(fig, "Sell")


def test_intraday_marker_lands_on_exact_bar() -> None:
    """Legacy date-based resolver collapsed all intraday bars of a
    session to the LAST bar of that day. The new ts-aware resolver
    must put each marker on its specific 15m bar instead."""
    df = _intraday_df()
    trade = _intraday_trade((10, 30), (14, 45))
    fig = PlotlyChartBuilder().build_candlestick_with_trades(df, [trade])

    buy_x = pd.Timestamp(_x_for(fig, "Buy"))
    sell_x = pd.Timestamp(_x_for(fig, "Sell"))
    # Compare wall-clock strings — plotly may strip the tz from the
    # stored x value, but the H:MM components must match the bar.
    assert buy_x.strftime("%H:%M") == "10:30"
    assert sell_x.strftime("%H:%M") == "14:45"


def test_daily_trade_hover_unchanged() -> None:
    """Daily trades have midnight ``entry_ts`` (set by the strategy
    layer); the chart helper's hover formatter must collapse those
    to date-only so existing daily pages keep their old display."""
    df = _daily_df()
    trade = Trade(
        pattern_name="wp",
        entry_date=date(2024, 3, 18),
        exit_date=date(2024, 3, 22),
        entry_price=103.5,
        exit_price=106.5,
        stop_loss=100.0,
        shares=100,
        pnl=300.0,
        pnl_pct=0.029,
        exit_reason="smart_trail",
        entry_ts=datetime(2024, 3, 18, 0, 0),
        exit_ts=datetime(2024, 3, 22, 0, 0),
    )
    fig = PlotlyChartBuilder().build_candlestick_with_trades(df, [trade])

    buy_hover = _hover_for(fig, "Buy")
    assert "2024-03-18" in buy_hover
    # No HH:MM substring for midnight-only trades.
    assert "00:00" not in buy_hover


def test_intraday_chart_uses_hour_rangebreaks() -> None:
    """The intraday chart must hide non-session hours and weekends
    on the x-axis. Without this rangebreak setup the 16:00 → next-
    day 09:30 overnight gap renders as a wide empty stretch with
    EMA/SMA lines drawn flat across it — confusing on a 15m chart."""
    df = _intraday_df()
    fig = PlotlyChartBuilder().build_candlestick_with_trades(df, [])

    breaks = list(fig.layout.xaxis.rangebreaks)
    # Two breaks expected: weekend collapse + non-session hour collapse.
    assert len(breaks) == 2
    bounds = [tuple(b.bounds) for b in breaks]
    assert ("sat", "mon") in bounds
    assert (16, 9.5) in bounds


def test_extended_hours_chart_uses_wider_bounds() -> None:
    """When the df includes pre-market or after-hours bars, the
    rangebreak bounds switch to ``[20, 4]`` so the full 04:00–20:00
    extended-hours window stays visible. FVG users need this so the
    morning CHoCH that printed during pre-market doesn't disappear
    from the x-axis."""
    # 04:00–20:00 extended-hours session (1 calendar day).
    idx = pd.date_range(
        "2024-03-18 04:00", "2024-03-18 19:45", freq="15min", tz=NY_TZ
    )
    n = len(idx)
    df = pd.DataFrame(
        {
            "Open": range(n),
            "High": [x + 1 for x in range(n)],
            "Low": [x - 1 for x in range(n)],
            "Close": [x + 0.5 for x in range(n)],
            "Volume": [1_000] * n,
        },
        index=idx,
    )
    fig = PlotlyChartBuilder().build_candlestick_with_trades(df, [])
    breaks = list(fig.layout.xaxis.rangebreaks)
    bounds = [tuple(b.bounds) for b in breaks]
    assert (20, 4) in bounds
    assert (16, 9.5) not in bounds


def test_daily_chart_uses_missing_dates_rangebreaks() -> None:
    """Daily charts keep the legacy ``values=missing_dates`` config
    — pre-existing pages render exactly as before."""
    df = _daily_df()
    fig = PlotlyChartBuilder().build_candlestick_with_trades(df, [])
    breaks = list(fig.layout.xaxis.rangebreaks)
    assert len(breaks) == 1
    # ``values`` is set; ``bounds`` should not be.
    assert breaks[0].values is not None


def test_legacy_trade_without_ts_still_renders() -> None:
    """Hand-built ``Trade`` objects (e.g. fixtures, exports) may have
    ``entry_ts=None``; the chart must fall back to date-based
    resolution rather than raise."""
    df = _daily_df()
    trade = Trade(
        pattern_name="wp",
        entry_date=date(2024, 3, 18),
        exit_date=date(2024, 3, 22),
        entry_price=103.5,
        exit_price=106.5,
        stop_loss=100.0,
        shares=100,
        pnl=300.0,
        pnl_pct=0.029,
        exit_reason="smart_trail",
    )
    fig = PlotlyChartBuilder().build_candlestick_with_trades(df, [trade])
    assert "2024-03-18" in _hover_for(fig, "Buy")
