"""Shared result-rendering helpers for Multi-Wedgepop Streamlit pages.

Both the daily (``3_Multi_Wedgepop.py``) and 15m
(``8_Multi_Wedgepop_15m.py``) pages finish with the same block of
UI — headline metrics, equity curve, top-trades bar, ticker
contribution, trade table, per-trade candlestick charts. Keeping
that code in one place means adding a new cadence (1h, 5m, …) later
only requires a new page wrapper, not another copy of this view.

The helpers are deliberately thin: they take a
``MultiStrategyResult`` (plus the chart-builder / market-data port
they need) and call ``streamlit`` directly. No decisions about what
to render live here — the calling page composes them.

Pages that pre-date these helpers (notably ``3_Multi_Wedgepop.py``)
still inline the same logic; the intention is that any future
touch-up to that page pulls in these helpers too, at which point
the duplication disappears. Until then, any change to this module
must also be mirrored in the legacy page's inline block.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from data.domain.ports import MarketDataPort
from strategy.domain.models import (
    MultiStrategyResult,
    StrategyPerformance,
    TossFeeSchedule,
    Trade,
)
from visualization.adapters.plotly_charts import PlotlyChartBuilder

EXIT_REASON_LABELS = {
    "take_profit": "Take Profit (R-target)",
    "initial_stop": "Initial Stop (consolidation low)",
    "exhaustion_exit": "Exhaustion Extension Top",
    "trendline_break": "Higher-Low Trendline Break",
    "smart_trail": "Smart Trail (Chandelier)",
    "resistance_break": "Resistance Break (false breakout)",
    "breakeven_stop": "Break-even Stop (≥1R unrealized)",
    "bos_trail_stop": "BOS Trail (FVG midpoint)",
    "end_of_data": "End of Data (no exit fired)",
}


def render_headline_metrics(
    result: MultiStrategyResult,
    *,
    universe_label: str,
) -> None:
    """Three rows of 4 metrics: scan stats, return stats, fee stats."""
    st.subheader(f"Universe — {universe_label}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tickers Scanned", result.tickers_scanned)
    m2.metric("Total Signals", result.total_signals)
    m3.metric("Trades Taken", result.trades_taken)
    m4.metric("Failed Tickers", len(result.failed_tickers))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Total Return (net)", f"{result.total_return_pct:.2%}")
    m6.metric("Win Rate", f"{result.win_rate:.0%}" if result.trades_taken else "—")
    m7.metric("Final Capital", f"${result.final_capital:,.0f}")
    m8.metric("Max Drawdown", f"{result.max_drawdown_pct:.2%}")

    m9, m10, m11, m12 = st.columns(4)
    gross_pnl = sum(t.gross_pnl for t in result.trades)
    m9.metric("Total Commission", f"${result.total_commission:,.2f}")
    m10.metric("Gross P&L", f"${gross_pnl:,.2f}")
    m11.metric("Net P&L", f"${gross_pnl - result.total_commission:,.2f}")
    m12.metric(
        "Fees as % of Gross",
        f"{(result.total_commission / gross_pnl):.2%}" if gross_pnl else "—",
    )


def render_equity_curve(
    chart_builder: PlotlyChartBuilder,
    result: MultiStrategyResult,
    *,
    title: str,
) -> None:
    st.subheader("Portfolio Equity Curve")
    fig = chart_builder.build_equity_curve(result.equity_curve, title=title)
    st.plotly_chart(fig, use_container_width=True)


def render_top_trades(
    chart_builder: PlotlyChartBuilder,
    result: MultiStrategyResult,
    *,
    title: str = "Top trades by % return",
) -> None:
    st.subheader("Best Trades")
    fig = chart_builder.build_top_trades_bar(result.trades, title=title)
    st.plotly_chart(fig, use_container_width=True)


def render_ticker_contribution(
    chart_builder: PlotlyChartBuilder,
    result: MultiStrategyResult,
    *,
    title: str = "Total P&L contribution per ticker",
) -> None:
    st.subheader("P&L by Ticker")
    fig = chart_builder.build_ticker_contribution_bar(result.trades, title=title)
    st.plotly_chart(fig, use_container_width=True)


def render_trade_table(
    result: MultiStrategyResult, market_tz: str | None = None
) -> None:
    """Detail table — same columns the daily page has always shown.

    Uses ``entry_ts`` / ``exit_ts`` when populated (always true after
    the strategy generalization) so intraday trades display the exact
    bar time; falls back to ``entry_date`` / ``exit_date`` for rows
    coming from old code paths.

    ``market_tz`` lets KR pages render times in KST instead of the
    default NY clock.
    """
    st.subheader("Trades — Details")
    rows = []
    for t in result.trades:
        entry_label = _format_trade_time(t.entry_ts, t.entry_date, market_tz)
        exit_label = _format_trade_time(t.exit_ts, t.exit_date, market_tz)
        rows.append(
            {
                "Ticker": t.ticker,
                "Entry": entry_label,
                "Exit": exit_label,
                "Exit Reason": EXIT_REASON_LABELS.get(
                    t.exit_reason, t.exit_reason
                ),
                "Entry Price": f"${t.entry_price:,.2f}",
                "Exit Price": f"${t.exit_price:,.2f}",
                "Stop": f"${t.stop_loss:,.2f}",
                "Shares": t.shares,
                "Sig Vol": f"{t.signal_volume:,.0f}",
                "Buy Vol": f"{t.signal_buy_volume:,.0f}",
                "Sell Vol": f"{t.signal_sell_volume:,.0f}",
                "Buy/Sell": f"{t.signal_buy_sell_ratio:.2f}",
                "Gross P&L": f"${t.gross_pnl:,.2f}",
                "Commission": f"${t.commission:,.2f}",
                "Net P&L ($)": f"${t.pnl:,.2f}",
                "Net P&L (%)": f"{t.pnl_pct:.2%}",
            }
        )
    st.dataframe(rows, use_container_width=True)


def render_per_trade_charts(
    chart_builder: PlotlyChartBuilder,
    market_data: MarketDataPort,
    result: MultiStrategyResult,
    *,
    interval: str = "1d",
    context_before_days: int = 60,
    context_after_days: int = 30,
    warmup_days: int = 400,
    default_expanded: int = 3,
    market=None,
    market_tz: str | None = None,
) -> None:
    """Render one candlestick chart per trade.

    ``interval`` is forwarded to ``market_data.fetch_ohlcv`` so the
    15m page fetches intraday context and the daily page fetches
    daily bars. ``warmup_days`` buys extra history so any MAs the
    chart draws converge before the trade window; for 15m pages
    this should be dropped to stay inside yfinance's 60-day cap.
    """
    st.subheader("Trade Charts")
    for i, t in enumerate(result.trades):
        pnl_sign = "+" if t.pnl >= 0 else ""
        entry_label = _format_trade_time(t.entry_ts, t.entry_date, market_tz)
        exit_label = _format_trade_time(t.exit_ts, t.exit_date, market_tz)
        label = (
            f"{t.ticker} — {entry_label} → {exit_label}  "
            f"({pnl_sign}{t.pnl_pct:.2%})"
        )
        with st.expander(label, expanded=(i < default_expanded)):
            chart_start = t.entry_date - timedelta(days=context_before_days)
            chart_end = t.exit_date + timedelta(days=context_after_days)
            fetch_start = chart_start - timedelta(days=warmup_days)
            try:
                ticker_df = market_data.fetch_ohlcv(
                    t.ticker, fetch_start, chart_end, interval=interval
                )
            except Exception as exc:
                st.warning(f"Failed to fetch data for {t.ticker}: {exc}")
                continue
            if ticker_df is None or ticker_df.empty:
                st.warning(f"No data for {t.ticker}")
                continue

            chart_kwargs: dict = {}
            if market is not None:
                chart_kwargs["market"] = market
            fig = chart_builder.build_candlestick_with_trades(
                ticker_df,
                [t],
                title=f"{t.ticker} — {entry_label} → {exit_label}",
                **chart_kwargs,
            )
            fig.update_xaxes(range=[str(chart_start), str(chart_end)])
            st.plotly_chart(fig, use_container_width=True)


def render_failed_tickers(result: MultiStrategyResult) -> None:
    if result.failed_tickers:
        with st.expander(f"Failed tickers ({len(result.failed_tickers)})"):
            st.write(", ".join(result.failed_tickers))


# ---- single-ticker fee accounting --------------------------------------


def render_toss_fee_inputs(
    *,
    header: str = "Fees (Toss Securities)",
    caption: str | None = (
        "토스증권 미국주식 기본 수수료. 매수/매도 각 0.1% + SEC fee 0.00229% (매도)."
    ),
    key_prefix: str = "",
) -> TossFeeSchedule:
    """Render Buy / Sell / SEC commission widgets and return a
    :class:`TossFeeSchedule`.

    Must be called from inside ``with st.sidebar:`` on the calling
    page so the widgets land in the sidebar. ``key_prefix`` lets a
    page host several fee blocks (e.g. one per strategy) without
    widget-key collisions.
    """
    st.header(header)
    if caption:
        st.caption(caption)
    buy_fee_pct = st.number_input(
        "Buy commission (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
        key=f"{key_prefix}buy_fee_pct",
    )
    sell_fee_pct = st.number_input(
        "Sell commission (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
        key=f"{key_prefix}sell_fee_pct",
    )
    sec_fee_pct = st.number_input(
        "SEC fee (%) — sell only",
        value=0.00229,
        min_value=0.0,
        max_value=1.0,
        step=0.0001,
        format="%.5f",
        key=f"{key_prefix}sec_fee_pct",
        help="미국 SEC Section 31 규정 수수료. 매도 거래대금에 부과.",
    )
    return TossFeeSchedule(
        buy_commission_pct=buy_fee_pct / 100.0,
        sell_commission_pct=sell_fee_pct / 100.0,
        sec_fee_pct=sec_fee_pct / 100.0,
    )


def apply_fees_to_trades(
    trades: list[Trade],
    fee_schedule: TossFeeSchedule,
) -> list[dict]:
    """Augment each :class:`Trade` with Toss commission and net-of-
    fees PnL. Returned dicts are ready for ``st.dataframe`` — the
    caller just picks which columns to display. The original
    ``Trade`` objects are left untouched so upstream consumers
    (charts, exports) keep working with the gross numbers the
    strategy produced.

    ``net_pnl_pct`` is reported on cost basis (entry_price × shares)
    so it's directly comparable to ``pnl_pct`` on the gross side;
    both are fraction-of-notional returns.
    """
    rows: list[dict] = []
    for t in trades:
        commission = fee_schedule.round_trip(
            t.entry_price, t.exit_price, t.shares
        )
        gross = t.pnl
        net = gross - commission
        cost_basis = t.entry_price * t.shares
        net_pct = net / cost_basis if cost_basis > 0 else 0.0
        rows.append(
            {
                "trade": t,
                "commission": commission,
                "gross_pnl": gross,
                "net_pnl": net,
                "net_pnl_pct": net_pct,
            }
        )
    return rows


def render_single_ticker_headline_metrics(
    perf: StrategyPerformance,
    fee_rows: list[dict],
    *,
    subtitle: str,
) -> None:
    """Headline metrics for the single-ticker 15m pages — parallels
    ``render_headline_metrics`` for multi but consumes
    :class:`StrategyPerformance` + fee-augmented rows. Shows both
    gross (strategy-native) and net-of-fees totals so the fee impact
    is visible at a glance.
    """
    total_commission = sum(r["commission"] for r in fee_rows)
    total_gross = sum(r["gross_pnl"] for r in fee_rows)
    total_net = sum(r["net_pnl"] for r in fee_rows)
    net_return = (
        total_net / perf.initial_capital if perf.initial_capital > 0 else 0.0
    )

    st.subheader(subtitle)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return (gross)", f"{perf.total_return_pct:.2%}")
    c2.metric("Total Return (net)", f"{net_return:.2%}")
    c3.metric("Trades", perf.total_trades)
    c4.metric("Win Rate", f"{perf.win_rate:.0%}" if perf.total_trades else "—")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Final Capital (gross)", f"${perf.final_capital:,.0f}")
    c6.metric(
        "Final Capital (net)",
        f"${perf.initial_capital + total_net:,.0f}",
    )
    c7.metric("Max DD", f"{perf.max_drawdown_pct:.2%}")
    c8.metric("Initial", f"${perf.initial_capital:,.0f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Total Commission", f"${total_commission:,.2f}")
    c10.metric("Gross P&L", f"${total_gross:,.2f}")
    c11.metric("Net P&L", f"${total_net:,.2f}")
    c12.metric(
        "Fees as % of Gross",
        f"{(total_commission / total_gross):.2%}" if total_gross else "—",
    )


def render_single_ticker_trade_table(fee_rows: list[dict]) -> None:
    """Trade table with commission / gross / net columns, using the
    fee-augmented rows from :func:`apply_fees_to_trades`."""
    st.subheader("Trades")
    rendered: list[dict] = []
    for r in fee_rows:
        t: Trade = r["trade"]
        entry_label = _format_trade_time(t.entry_ts, t.entry_date)
        exit_label = _format_trade_time(t.exit_ts, t.exit_date)
        rendered.append(
            {
                "Entry": entry_label,
                "Exit": exit_label,
                "Exit Reason": EXIT_REASON_LABELS.get(
                    t.exit_reason, t.exit_reason
                ),
                "Entry Price": f"${t.entry_price:,.2f}",
                "Exit Price": f"${t.exit_price:,.2f}",
                "Stop": f"${t.stop_loss:,.2f}",
                "Shares": t.shares,
                "Gross P&L": f"${r['gross_pnl']:,.2f}",
                "Commission": f"${r['commission']:,.2f}",
                "Net P&L ($)": f"${r['net_pnl']:,.2f}",
                "Net P&L (%)": f"{r['net_pnl_pct']:.2%}",
            }
        )
    st.dataframe(rendered, use_container_width=True)


def _format_trade_time(
    ts, fallback_date: date, market_tz: str | None = None
) -> str:
    """Prefer the exact bar timestamp when the strategy populated it
    (always the case after the interval generalization), fall back to
    the session date for legacy rows.

    ``market_tz`` (e.g. ``"Asia/Seoul"``) converts a tz-aware NY
    timestamp into the market's local clock before formatting, so
    KR trades surface as 15:15 KST rather than 01:15 NY (same
    moment, two clocks). When ``None`` we strip tz keeping the wall
    clock intact — the original behavior for US tickers.
    """
    if ts is None:
        return fallback_date.isoformat()
    stamp = pd.Timestamp(ts)
    if market_tz is not None and stamp.tz is not None:
        stamp = stamp.tz_convert(market_tz).tz_localize(None)
    if stamp.time().hour == 0 and stamp.time().minute == 0:
        return stamp.date().isoformat()
    return stamp.strftime("%Y-%m-%d %H:%M")
