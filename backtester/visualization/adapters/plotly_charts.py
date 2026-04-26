from datetime import time as _dt_time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pattern.domain.models import PatternSignal
from pattern.helpers.pivots import find_swing_highs
from strategy.domain.models import EquityPoint, MultiTrade, Trade
from visualization.domain.ports import ChartBuilderPort

# Regular session window for BOS reference filtering. After-hours
# bars print on thin liquidity and routinely spike above the actual
# RTH structural pivot — using them as the BOS reference would
# anchor the line off-screen and make the visualization useless.
_RTH_OPEN = _dt_time(9, 30)
_RTH_CLOSE = _dt_time(16, 0)


class PlotlyChartBuilder(ChartBuilderPort):
    def build_candlestick_with_signals(
        self,
        df: pd.DataFrame,
        signals: list[PatternSignal],
        title: str = "",
    ) -> go.Figure:
        # Strip timezone so dates, rangebreaks, and signal markers all align
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
                decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
                whiskerwidth=0.8,
            ),
            row=1,
            col=1,
        )

        # Moving averages
        ema10 = df["Close"].ewm(span=10, adjust=False).mean()
        ema20 = df["Close"].ewm(span=20, adjust=False).mean()
        sma50 = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200).mean()

        for ma, name, color in [
            (ema10, "10 EMA", "#ef5350"),
            (ema20, "20 EMA", "#2196F3"),
            (sma50, "50 SMA", "#9C27B0"),
            (sma200, "200 SMA", "#00BCD4"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.2),
                ),
                row=1,
                col=1,
            )

        # Volume bars
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=colors,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Signal markers — match dates to exact DataFrame index values
        if signals:
            date_map = {idx.date(): idx for idx in df.index}
            matched = [
                (date_map[s.date], s)
                for s in signals
                if s.date in date_map
            ]
            signal_dates = [m[0] for m in matched]
            signal_prices = [m[1].entry_price for m in matched]
            stop_losses = [m[1].stop_loss for m in matched]
            signals = [m[1] for m in matched]
            hover_texts = [
                f"Pattern: {s.pattern_name}<br>"
                f"Entry: {s.entry_price:.2f}<br>"
                f"Stop: {s.stop_loss:.2f}<br>"
                f"Confidence: {s.confidence:.2f}"
                for s in signals
            ]

            fig.add_trace(
                go.Scatter(
                    x=signal_dates,
                    y=signal_prices,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="#2196F3",
                        line=dict(width=1, color="white"),
                    ),
                    name="Entry Signal",
                    hovertext=hover_texts,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=signal_dates,
                    y=stop_losses,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color="#F44336",
                        line=dict(width=1, color="white"),
                    ),
                    name="Stop Loss",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700,
            margin=dict(l=50, r=50, t=50, b=30),
            xaxis=dict(
                rangebreaks=self._rangebreaks(df),
            ),
            yaxis=dict(autorange=True, fixedrange=False),
            yaxis2=dict(autorange=True, fixedrange=False),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    def build_candlestick_with_trades(
        self,
        df: pd.DataFrame,
        trades: list[Trade],
        title: str = "",
        fvg_signals: list | None = None,
        take_profit_r: float | None = None,
    ) -> go.Figure:
        """Candlestick chart annotated with executed trades.

        For each Trade we render:
          - A blue triangle-up at (entry_date, entry_price) — "BUY".
          - An exit marker at (exit_date, exit_price). If the exit price
            equals the stop_loss, it's drawn as a red X ("STOP"), else as
            a green triangle-down ("SELL") for profitable exits or an
            orange diamond for losing-but-not-stopped exits.
          - A dashed red horizontal segment at the stop_loss level for
            the trade's duration, so the risk envelope is visible.
          - A dotted connector between entry and exit, colored by P&L.

        Hover text on every marker shows price, share count, $ value,
        and (on exits) realized P&L in dollars and percent.

        ``fvg_signals`` (Fair Value Gap detector output) and
        ``take_profit_r`` (e.g. 3.0 for 1:3 R/R) are optional. When
        provided we also paint the structural setup behind each
        trade: a green box for the FVG zone, a dashed midpoint line,
        a CHoCH annotation, plus stop / take-profit horizontal
        extensions so the risk-reward envelope is visible at a
        glance — matching the framework's step-3/4 illustrations.
        Other strategies (wedgepop, wickplay) leave both arguments
        as ``None`` and render exactly as before.
        """
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing=dict(
                    line=dict(color="#26a69a", width=1), fillcolor="#26a69a"
                ),
                decreasing=dict(
                    line=dict(color="#ef5350", width=1), fillcolor="#ef5350"
                ),
                whiskerwidth=0.8,
            ),
            row=1,
            col=1,
        )

        # Moving averages — 10/20 EMA drive the strategy's trail/exit
        # logic; 50/200 SMA give a longer-term trend context so you can
        # visually confirm whether a trade sits in an uptrend or not.
        ema10 = df["Close"].ewm(span=10, adjust=False).mean()
        ema20 = df["Close"].ewm(span=20, adjust=False).mean()
        sma50 = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200).mean()
        for ma, name, color in [
            (ema10, "10 EMA", "#ef5350"),
            (ema20, "20 EMA", "#2196F3"),
            (sma50, "50 SMA", "#9C27B0"),
            (sma200, "200 SMA", "#00BCD4"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.2),
                ),
                row=1,
                col=1,
            )

        # Volume
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=vol_colors,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Per-trade overlay: stop line, connector, then aggregated marker traces.
        #
        # Two resolution paths:
        #   - ``_resolve_ts(trade_ts)`` — exact bar match when the trade
        #     carries an ``entry_ts``/``exit_ts`` (populated on every
        #     run since the interval generalization). Intraday charts
        #     need this so a 15m BUY marker lands on the *specific*
        #     bar the strategy entered on, not the last bar of that
        #     session.
        #   - ``_resolve_date(d)`` — legacy path for daily / resampled
        #     (weekly/monthly) charts. Snaps a date to the first bar
        #     at or after that calendar day. Kept as a fallback for
        #     hand-built Trade objects where the ts fields are None.
        date_map = {idx.date(): idx for idx in df.index}

        def _resolve_date(d):
            if d in date_map:
                return date_map[d]
            ts = pd.Timestamp(d)
            after = df.index[df.index >= ts]
            return after[0] if len(after) > 0 else None

        def _resolve_ts(ts):
            """Map a trade Timestamp onto the chart's df.index, picking
            the bar at-or-before the ts so 15m trades land on the exact
            entry/exit bar.

            tz alignment rules (picked so wall-clock time is preserved,
            not shifted to UTC):

            - df naive + ts tz-aware  → strip ts's tz keeping wall clock
              (``tz_localize(None)``). Daily df on a ts that came from
              a tz-aware intraday fetch would fall into this branch.
            - df tz-aware + ts naive  → localize ts into df's tz.
            - both tz-aware, different zones → convert ts into df's tz.
            """
            if ts is None:
                return None
            stamp = pd.Timestamp(ts)
            if df.index.tz is None and stamp.tz is not None:
                stamp = stamp.tz_localize(None)
            elif df.index.tz is not None and stamp.tz is None:
                stamp = stamp.tz_localize(df.index.tz)
            elif (
                df.index.tz is not None
                and stamp.tz is not None
                and str(stamp.tz) != str(df.index.tz)
            ):
                stamp = stamp.tz_convert(df.index.tz)
            if stamp in df.index:
                return stamp
            before = df.index[df.index <= stamp]
            return before[-1] if len(before) > 0 else None

        def _resolve_trade(ts, d):
            """Prefer exact-bar resolution via ts, fall back to date-
            based resolution. Returns the DatetimeIndex value or None."""
            return _resolve_ts(ts) if ts is not None else _resolve_date(d)

        def _hover_when(ts, d):
            """Display label — ``YYYY-MM-DD HH:MM`` if intraday ts is
            available, else ISO date. Keeps daily-trade hovers exactly
            as before while 15m trades surface the bar time."""
            if ts is not None:
                stamp = pd.Timestamp(ts)
                if stamp.time().hour != 0 or stamp.time().minute != 0:
                    return stamp.strftime("%Y-%m-%d %H:%M")
            return str(d)

        def _strip_tz(ts):
            """The chart's df.index has been tz-stripped at the top of
            the function; FVG-overlay timestamps coming from detector
            metadata may still carry tz info. Normalize to tz-naive so
            ``add_shape`` / ``add_annotation`` x-values land on the
            right candle."""
            stamp = pd.Timestamp(ts)
            if stamp.tz is not None:
                stamp = stamp.tz_localize(None)
            return stamp

        exit_reason_labels = {
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

        entry_x: list = []
        entry_y: list[float] = []
        entry_hover: list[str] = []
        sell_x: list = []
        sell_y: list[float] = []
        sell_hover: list[str] = []

        for t in trades:
            entry_marker_x = _resolve_trade(
                getattr(t, "entry_ts", None), t.entry_date
            )
            exit_marker_x = _resolve_trade(
                getattr(t, "exit_ts", None), t.exit_date
            )
            if entry_marker_x is None or exit_marker_x is None:
                continue

            entry_value = t.entry_price * t.shares
            exit_value = t.exit_price * t.shares
            pnl_sign = "+" if t.pnl >= 0 else ""
            exit_reason = getattr(t, "exit_reason", "end_of_data")
            reason_label = exit_reason_labels.get(exit_reason, exit_reason)
            entry_when = _hover_when(
                getattr(t, "entry_ts", None), t.entry_date
            )
            exit_when = _hover_when(
                getattr(t, "exit_ts", None), t.exit_date
            )

            entry_x.append(entry_marker_x)
            entry_y.append(t.entry_price)
            entry_hover.append(
                "<b>BUY</b><br>"
                f"Date: {entry_when}<br>"
                f"Price: ${t.entry_price:,.2f}<br>"
                f"Shares: {t.shares}<br>"
                f"Value: ${entry_value:,.0f}<br>"
                f"Stop: ${t.stop_loss:,.2f}"
            )

            sell_x.append(exit_marker_x)
            sell_y.append(t.exit_price)
            sell_hover.append(
                "<b>SELL</b><br>"
                f"Date: {exit_when}<br>"
                f"Price: ${t.exit_price:,.2f}<br>"
                f"Shares: {t.shares}<br>"
                f"Value: ${exit_value:,.0f}<br>"
                f"P&L: {pnl_sign}${t.pnl:,.2f} "
                f"({pnl_sign}{t.pnl_pct:.2%})<br>"
                f"Exit: {reason_label}"
            )

            # Stop-loss envelope segment
            fig.add_trace(
                go.Scatter(
                    x=[entry_marker_x, exit_marker_x],
                    y=[t.stop_loss, t.stop_loss],
                    mode="lines",
                    line=dict(color="#F44336", width=1, dash="dash"),
                    name="Stop level",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

            # Entry → exit connector, colored by outcome
            connector_color = "#26a69a" if t.pnl > 0 else "#ef5350"
            fig.add_trace(
                go.Scatter(
                    x=[entry_marker_x, exit_marker_x],
                    y=[t.entry_price, t.exit_price],
                    mode="lines",
                    line=dict(color=connector_color, width=1.2, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        if entry_x:
            fig.add_trace(
                go.Scatter(
                    x=entry_x,
                    y=entry_y,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=16,
                        color="#2196F3",
                        line=dict(width=1, color="white"),
                    ),
                    name="Buy",
                    hovertext=entry_hover,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
        if sell_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=16,
                        color="#26a69a",
                        line=dict(width=1, color="white"),
                    ),
                    name="Sell",
                    hovertext=sell_hover,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

        # ---- FVG structural overlay -----------------------------
        # When the caller hands us the detector's signals, paint
        # each setup's structural anchors directly on the chart:
        #   - light-green rectangle = FVG zone
        #   - dashed mid-line       = midpoint hold/break level
        #   - CHoCH text annotation = where the structural break printed
        # Other strategies leave ``fvg_signals=None`` and skip this
        # whole block, so existing wedgepop/wickplay charts render
        # exactly as before.
        if fvg_signals:
            for sig in fvg_signals:
                meta = sig.metadata or {}
                if "fvg_low" not in meta or "fvg_high" not in meta:
                    continue
                fvg_low = float(meta["fvg_low"])
                fvg_high = float(meta["fvg_high"])
                fvg_mid = float(
                    meta.get("fvg_mid", (fvg_low + fvg_high) / 2.0)
                )

                fvg_start_iso = meta.get("fvg_start_timestamp")
                fvg_start = (
                    _strip_tz(pd.Timestamp(fvg_start_iso))
                    if fvg_start_iso
                    else _strip_tz(pd.Timestamp(sig.timestamp))
                )
                sig_ts = _strip_tz(pd.Timestamp(sig.timestamp))

                # Box spans the FVG's *active* window — from the
                # first bar of the gap (fvg_start) up through the
                # retest entry bar (sig_ts), where the gap got
                # consumed. After entry the box becomes irrelevant;
                # stop/TP lines below carry forward instead.
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=fvg_start,
                    x1=sig_ts,
                    y0=fvg_low,
                    y1=fvg_high,
                    fillcolor="rgba(76, 175, 80, 0.18)",
                    line=dict(color="rgba(76, 175, 80, 0.55)", width=1),
                    row=1,
                    col=1,
                )
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=fvg_start,
                    x1=sig_ts,
                    y0=fvg_mid,
                    y1=fvg_mid,
                    line=dict(color="#4caf50", width=1, dash="dash"),
                    row=1,
                    col=1,
                )

                choch_iso = meta.get("choch_timestamp")
                choch_high_meta = meta.get("choch_high")
                if choch_iso and choch_high_meta is not None:
                    choch_ts = _strip_tz(pd.Timestamp(choch_iso))
                    # Horizontal at the broken swing-high level. Spans
                    # from the swing-high bar (where the level first
                    # printed) to the CHoCH break bar — so the user
                    # *sees* exactly which prior structure got
                    # invalidated. Without this line the ChoCH text
                    # alone hovers at an arbitrary y-coordinate and
                    # the "break point" is ambiguous.
                    break_start_iso = meta.get("choch_break_level_start_ts")
                    line_x0 = (
                        _strip_tz(pd.Timestamp(break_start_iso))
                        if break_start_iso
                        else choch_ts
                    )
                    fig.add_shape(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=line_x0,
                        x1=choch_ts,
                        y0=float(choch_high_meta),
                        y1=float(choch_high_meta),
                        line=dict(
                            color="#ab47bc", width=1.5, dash="dashdot"
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_annotation(
                        x=choch_ts,
                        y=float(choch_high_meta),
                        text="ChoCH",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor="#ab47bc",
                        font=dict(color="#ab47bc", size=11),
                        row=1,
                        col=1,
                    )

                    # ---- BOS (Break of Structure) overlay ---------
                    # Per the framework: after the FVG entry, price
                    # rallies and prints a *new* swing high — that
                    # rally peak (NOT choch_high) is the BOS line.
                    # When a later close pierces that swing-high
                    # level, BOS fires and the strategy's optional
                    # ``enable_bos_trail`` lifts the stop to the FVG
                    # midpoint. The line we draw here:
                    #
                    #   - sits at the level of the most recent
                    #     confirmed post-entry swing high (running
                    #     max — once a new higher peak prints it
                    #     replaces the prior reference)
                    #   - extends from that swing-high bar forward
                    #     to the breakout bar
                    #
                    # Drawn unconditionally (not gated on
                    # enable_bos_trail) — it's diagnostic info that
                    # tells the user *where* the structural pivot
                    # lives even when no trail is configured.
                    bos_swing_ts, bos_event_ts, bos_level = (
                        self._find_post_entry_bos(df, sig_ts)
                    )
                    if bos_level is not None and bos_swing_ts is not None:
                        # Extent: if the level got broken, stop the
                        # line at the breakout bar (the level was
                        # taken out there). If still intact, extend
                        # ~30 bars forward from the swing-high bar
                        # so the user can see *where* the line sits
                        # while the trade is still live, even when
                        # the trade exits via BE/stop/TP before any
                        # breakout. Without this, a trade that BE-
                        # exited on a pullback (TSLA 2026-04-24
                        # case) would draw nothing at all and the
                        # user couldn't tell what level was being
                        # watched.
                        if bos_event_ts is not None:
                            line_end = bos_event_ts
                        else:
                            after_swing = df.index[df.index > bos_swing_ts]
                            if len(after_swing) > 0:
                                line_end = _strip_tz(
                                    pd.Timestamp(
                                        after_swing[
                                            min(30, len(after_swing) - 1)
                                        ]
                                    )
                                )
                            else:
                                line_end = bos_swing_ts
                        fig.add_shape(
                            type="line",
                            xref="x",
                            yref="y",
                            x0=bos_swing_ts,
                            x1=line_end,
                            y0=bos_level,
                            y1=bos_level,
                            line=dict(
                                color="#ff9800", width=1.8, dash="dot"
                            ),
                            row=1,
                            col=1,
                        )
                        # Small "BOS line" label at the right end so
                        # the line is identifiable even when not
                        # broken.
                        fig.add_annotation(
                            x=line_end,
                            y=bos_level,
                            text="BOS line",
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            font=dict(color="#ff9800", size=10),
                            row=1,
                            col=1,
                        )
                        # The actual "BOS" arrow only fires on
                        # breakout — its presence on the chart is
                        # the user's signal that the structural
                        # break happened.
                        if bos_event_ts is not None:
                            fig.add_annotation(
                                x=bos_event_ts,
                                y=bos_level,
                                text="BOS",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=1.5,
                                arrowcolor="#ff9800",
                                ax=0,
                                ay=-30,
                                font=dict(
                                    color="#ff9800",
                                    size=11,
                                    family="Arial Black",
                                ),
                                row=1,
                                col=1,
                            )

                # ---- per-signal stop / take-profit lines ----------
                # Even when no trade was taken on this signal (e.g.
                # the strategy was already in a position), draw the
                # *plan*: where the stop sits, where the R-target
                # sits. Lines extend ~30 bars forward from the
                # retest entry bar so they're visible without
                # flooding the whole future of the chart. Each line
                # carries a small "STOP" / "TP +Nx" label so the
                # user can read the levels at a glance.
                sig_entry = float(sig.entry_price)
                sig_stop = float(sig.stop_loss)
                risk = sig_entry - sig_stop
                if risk > 0:
                    after_sig_full = df.index[df.index > sig_ts]
                    sig_extend_end = (
                        after_sig_full[min(30, len(after_sig_full) - 1)]
                        if len(after_sig_full) > 0
                        else sig_ts
                    )
                    fig.add_shape(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=sig_ts,
                        x1=sig_extend_end,
                        y0=sig_stop,
                        y1=sig_stop,
                        line=dict(
                            color="#ef5350", width=1.8, dash="dash"
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_annotation(
                        x=sig_extend_end,
                        y=sig_stop,
                        text="STOP",
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(color="#ef5350", size=10),
                        bgcolor="rgba(0,0,0,0.0)",
                        row=1,
                        col=1,
                    )
                    if take_profit_r is not None:
                        sig_target = sig_entry + take_profit_r * risk
                        fig.add_shape(
                            type="line",
                            xref="x",
                            yref="y",
                            x0=sig_ts,
                            x1=sig_extend_end,
                            y0=sig_target,
                            y1=sig_target,
                            line=dict(
                                color="#26a69a", width=1.8, dash="dot"
                            ),
                            row=1,
                            col=1,
                        )
                        fig.add_annotation(
                            x=sig_extend_end,
                            y=sig_target,
                            text=f"TP +{take_profit_r:g}R",
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            font=dict(color="#26a69a", size=10),
                            row=1,
                            col=1,
                        )

        # ---- Take-profit envelope -------------------------------
        # Per-trade horizontal at ``entry + R × initial_risk``. The
        # stop line is already drawn during the trade loop above, so
        # this layer just adds the upside half of the R/R box.
        if take_profit_r is not None:
            for t in trades:
                entry_x_t = _resolve_trade(
                    getattr(t, "entry_ts", None), t.entry_date
                )
                exit_x_t = _resolve_trade(
                    getattr(t, "exit_ts", None), t.exit_date
                )
                if entry_x_t is None or exit_x_t is None:
                    continue
                init_risk = t.entry_price - t.stop_loss
                if init_risk <= 0:
                    continue
                tp_price = t.entry_price + take_profit_r * init_risk
                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=entry_x_t,
                    x1=exit_x_t,
                    y0=tp_price,
                    y1=tp_price,
                    line=dict(color="#26a69a", width=1.5, dash="dot"),
                    row=1,
                    col=1,
                )

        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700,
            margin=dict(l=50, r=50, t=50, b=30),
            xaxis=dict(
                rangebreaks=self._rangebreaks(df),
            ),
            yaxis=dict(autorange=True, fixedrange=False),
            yaxis2=dict(autorange=True, fixedrange=False),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    def build_simple_candlestick(
        self,
        df: pd.DataFrame,
        title: str = "",
        height: int = 400,
        with_mas: bool = False,
    ) -> go.Figure:
        """Minimal candlestick chart for context.

        Used to render higher-timeframe (weekly, yearly) context next
        to the primary daily chart. Resampled dataframes often span
        many years; we skip the `rangebreaks` logic that the daily
        chart needs because non-trading days are irrelevant at the
        weekly/yearly resolution.

        ``with_mas=True`` overlays the same four moving averages as
        the daily chart (10/20 EMA + 50/200 SMA), computed on the
        resampled timeframe so the weekly "10 EMA" is 10 weeks etc.
        """
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing=dict(
                    line=dict(color="#26a69a", width=1), fillcolor="#26a69a"
                ),
                decreasing=dict(
                    line=dict(color="#ef5350", width=1), fillcolor="#ef5350"
                ),
                whiskerwidth=0.8,
            )
        )

        if with_mas:
            ema10 = df["Close"].ewm(span=10, adjust=False).mean()
            ema20 = df["Close"].ewm(span=20, adjust=False).mean()
            sma50 = df["Close"].rolling(50).mean()
            sma200 = df["Close"].rolling(200).mean()
            for ma, name, color in [
                (ema10, "10 EMA", "#ef5350"),
                (ema20, "20 EMA", "#2196F3"),
                (sma50, "50 SMA", "#9C27B0"),
                (sma200, "200 SMA", "#00BCD4"),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ma,
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.2),
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=height,
            margin=dict(l=50, r=50, t=50, b=30),
            yaxis=dict(autorange=True, fixedrange=False),
        )
        fig.update_yaxes(title_text="Price")
        return fig

    @staticmethod
    def _missing_dates(df: pd.DataFrame) -> list[str]:
        """Return dates with no data between first and last trading day."""
        all_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
        missing = all_days.difference(df.index)
        return [d.strftime("%Y-%m-%d") for d in missing]

    @classmethod
    def _rangebreaks(cls, df: pd.DataFrame) -> list[dict]:
        """Plotly ``xaxis.rangebreaks`` config, picked from the index.

        Daily frames get a list of missing calendar dates so weekends
        / holidays collapse on the x-axis. Intraday frames use
        plotly's pattern-based breaks instead — without them the
        16:00 → next-day 09:30 overnight gap renders as a wide empty
        stretch and indicator lines (EMA / SMA) appear as flat
        horizontal segments crossing it.

        Three regimes:
        - **Daily**: ``values=missing_dates`` — collapses weekends /
          holidays.
        - **Intraday RTH only** (no bar before 09:30 or after 16:00):
          ``bounds=[16, 9.5]`` hides each day's overnight 16:00–09:30
          gap. Same as the original 1d-vs-15m split.
        - **Intraday with extended hours** (any pre-market / after-
          hours bar present): ``bounds=[20, 4]`` instead, so the
          chart shows the full 04:00–20:00 trading window and only
          collapses the true overnight 20:00–04:00 idle. Picked
          automatically from the time-of-day distribution; FVG
          callers that need pre-market CHoCH context get the right
          x-axis without any extra config.
        """
        if len(df.index) == 0:
            return [dict(values=cls._missing_dates(df))]
        is_intraday = (
            df.index.tz is not None
            or df.index.normalize().value_counts().max() > 1
        )
        if not is_intraday:
            return [dict(values=cls._missing_dates(df))]

        # Detect extended-hours: any bar's time-of-day falls outside
        # the 09:30–16:00 regular session window. ``hour + minute/60``
        # makes the comparison exact at quarter-hour boundaries.
        tods = pd.Series(
            [t.hour + t.minute / 60.0 for t in df.index]
        )
        has_pre_post = bool(((tods < 9.5) | (tods >= 16.0)).any())
        if has_pre_post:
            return [
                dict(bounds=["sat", "mon"]),
                dict(bounds=[20, 4], pattern="hour"),
            ]
        return [
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour"),
        ]

    @staticmethod
    def _find_post_entry_bos(
        df: pd.DataFrame,
        sig_ts: pd.Timestamp,
        swing_left: int = 1,
        swing_right: int = 1,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None, float | None]:
        """Locate the FVG strategy's BOS reference (swing high) and
        break event, if any.

        Per the framework's step-4 picture, the BOS reference is
        **the FIRST post-entry rally peak** — the singular
        "한참 올라갔던 상승선" that the strategy waits for price to
        pierce. Implementation: find the first confirmed 2/2 swing
        high *strictly after* ``sig_ts`` and freeze it as the
        reference. Don't promote to later (higher) peaks; that
        would shift the line away from the level the user actually
        sees getting broken on the chart.

        Returns ``(swing_high_bar_ts, bos_event_bar_ts, level)``:

        - ``level`` / ``swing_high_bar_ts`` — the first confirmed
          post-entry swing high.
        - ``bos_event_bar_ts`` — the first bar (after the swing
          high) whose CLOSE pierces ``level``. ``None`` while the
          peak is still intact.

        Returns ``(None, None, None)`` when no swing high confirms
        in the post-entry RTH window.

        ETH gate: bars whose start falls outside 09:30–16:00 NY
        local are excluded from the post-entry window. Without this,
        a thin after-hours print regularly anchors the line off-
        chart (TSLA 2026-04-24 had a 17:50 spike at $391 well
        above the visible RTH peak at ~$382).

        Pivot half-widths default to **1/1** (a bar whose high is
        strictly higher than the immediately neighboring bar on
        each side). The detector's CHoCH logic uses 2/2 because it
        needs a stricter "real swing" gate, but for the BOS
        reference the framework just talks about the rally's *peak*
        — and on 1m bars 2/2 is so strict that obvious chart peaks
        like the TSLA 2026-04-24 09:46 high get overridden two bars
        later by a marginally-higher print, leaving the line
        anchored at the wrong level.
        """
        post = df[df.index > sig_ts]
        # Drop ETH bars before swing-high detection. After-hours
        # liquidity routinely prints isolated spikes that aren't
        # real structural pivots — TSLA 2026-04-24 had an 17:50
        # post-market high at $391 well above the visible RTH peak
        # at ~$382, which set ``bos_level`` off-chart and rendered
        # the line invisible. Daily-resolution frames (all bars at
        # midnight) are exempt from the time-of-day gate.
        times = [t.time() for t in post.index]
        if times and not all(t == _dt_time(0, 0) for t in times):
            rth_mask = np.array(
                [(_RTH_OPEN <= t < _RTH_CLOSE) for t in times], dtype=bool
            )
            post = post[rth_mask]
        if len(post) < swing_left + swing_right + 1:
            return None, None, None
        swing_arr = find_swing_highs(
            post, left=swing_left, right=swing_right
        ).to_numpy()
        closes_arr = post["Close"].to_numpy(dtype=float)
        post_index = post.index

        bos_level: float | None = None
        bos_swing_idx: int | None = None
        bos_event_idx: int | None = None
        for j in range(len(post_index)):
            confirm = j - swing_right
            if confirm >= 0 and bos_level is None:
                # Freeze the FIRST confirmed swing high as the BOS
                # reference. Do not promote to later (higher) peaks
                # — the user expects the line at the first rally
                # pivot they can see, not whatever turns out to be
                # the session's all-time high.
                sh_val = swing_arr[confirm]
                if not np.isnan(sh_val):
                    bos_level = float(sh_val)
                    bos_swing_idx = confirm
            if (
                bos_level is not None
                and bos_event_idx is None
                and closes_arr[j] > bos_level
            ):
                bos_event_idx = j
                break
        if bos_level is None or bos_swing_idx is None:
            return None, None, None

        def _strip(ts):
            if hasattr(ts, "tz") and ts.tz is not None:
                ts = ts.tz_localize(None)
            return pd.Timestamp(ts)

        swing_ts = _strip(post_index[bos_swing_idx])
        event_ts = (
            _strip(post_index[bos_event_idx])
            if bos_event_idx is not None
            else None
        )
        return swing_ts, event_ts, bos_level

    def build_equity_curve(
        self,
        equity_points: list[EquityPoint],
        title: str = "",
    ) -> go.Figure:
        dates = [p.date for p in equity_points]
        equities = [p.equity for p in equity_points]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equities,
                mode="lines+markers",
                name="Equity",
                line=dict(color="#2196F3", width=2),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(33, 150, 243, 0.1)",
            )
        )

        if len(equities) >= 2:
            initial = equities[0]
            fig.add_hline(
                y=initial,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Initial: ${initial:,.0f}",
            )

        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=400,
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            margin=dict(l=50, r=50, t=50, b=30),
        )

        return fig

    # ---- multi-ticker scan visualizations ----

    def build_top_trades_bar(
        self,
        trades: list[MultiTrade],
        title: str = "",
    ) -> go.Figure:
        """Horizontal bar of the best individual trades the scan took.

        Each bar is one trade, labelled ``TICKER · entry_date``, sized
        by ``pnl_pct`` and coloured by sign. The hover surfaces the
        signal-day volume that won the daily auction so the user can
        see *why* the multi-strategy picked this ticker.
        """
        ordered = sorted(trades, key=lambda t: t.pnl_pct, reverse=True)
        # Reverse so the largest bar lands at the top of the chart.
        ordered = list(reversed(ordered))
        labels = [
            f"{t.ticker} · {t.entry_date.isoformat()}" for t in ordered
        ]
        returns = [t.pnl_pct * 100 for t in ordered]
        colors = ["#26a69a" if r >= 0 else "#ef5350" for r in returns]
        hover = [
            f"<b>{t.ticker}</b><br>"
            f"Entry: {t.entry_date} @ ${t.entry_price:,.2f}<br>"
            f"Exit:  {t.exit_date} @ ${t.exit_price:,.2f}<br>"
            f"Stop:  ${t.stop_loss:,.2f}<br>"
            f"Shares: {t.shares}<br>"
            f"P&L: ${t.pnl:,.2f} ({t.pnl_pct:.2%})<br>"
            f"Signal Vol: {t.signal_volume:,.0f}<br>"
            f"Buy / Sell Vol: {t.signal_buy_volume:,.0f} / "
            f"{t.signal_sell_volume:,.0f}<br>"
            f"Buy/Sell Ratio: {t.signal_buy_sell_ratio:.2f}"
            for t in ordered
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=returns,
                y=labels,
                orientation="h",
                marker_color=colors,
                hovertext=hover,
                hoverinfo="text",
                text=[f"{r:.1f}%" for r in returns],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=max(320, 26 * len(trades) + 120),
            xaxis_title="Trade Return (%)",
            yaxis_title="Trade",
            margin=dict(l=180, r=50, t=50, b=40),
        )
        return fig

    def build_ticker_contribution_bar(
        self,
        trades: list[MultiTrade],
        title: str = "",
    ) -> go.Figure:
        """Per-ticker total P&L contribution.

        Sums dollar P&L by ticker so the user can see which names did
        the heavy lifting. Hover surfaces trade count and win rate per
        ticker — important context because a ticker that contributed
        $5k from one giant winner reads very differently from one that
        ground out $5k across ten trades.
        """
        agg: dict[str, dict[str, float]] = {}
        for t in trades:
            slot = agg.setdefault(
                t.ticker, {"pnl": 0.0, "trades": 0, "wins": 0}
            )
            slot["pnl"] += t.pnl
            slot["trades"] += 1
            if t.pnl > 0:
                slot["wins"] += 1

        rows = sorted(agg.items(), key=lambda kv: kv[1]["pnl"])
        tickers = [k for k, _ in rows]
        pnls = [v["pnl"] for _, v in rows]
        colors = ["#26a69a" if p >= 0 else "#ef5350" for p in pnls]
        hover = [
            f"<b>{k}</b><br>"
            f"P&L: ${v['pnl']:,.2f}<br>"
            f"Trades: {int(v['trades'])}<br>"
            f"Win Rate: "
            f"{(v['wins'] / v['trades']) if v['trades'] else 0:.0%}"
            for k, v in rows
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=pnls,
                y=tickers,
                orientation="h",
                marker_color=colors,
                hovertext=hover,
                hoverinfo="text",
                text=[f"${p:,.0f}" for p in pnls],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=max(320, 24 * len(rows) + 120),
            xaxis_title="P&L ($)",
            yaxis_title="Ticker",
            margin=dict(l=80, r=50, t=50, b=40),
        )
        return fig
