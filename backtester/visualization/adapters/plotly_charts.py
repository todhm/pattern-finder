import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pattern.domain.models import PatternSignal
from strategy.domain.models import EquityPoint, MultiTrade, Trade
from visualization.domain.ports import ChartBuilderPort


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
                rangebreaks=[
                    dict(values=self._missing_dates(df)),
                ],
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

        # Per-trade overlay: stop line, connector, then aggregated marker traces
        date_map = {idx.date(): idx for idx in df.index}

        entry_x: list = []
        entry_y: list[float] = []
        entry_hover: list[str] = []
        sell_x: list = []
        sell_y: list[float] = []
        sell_hover: list[str] = []
        stop_x: list = []
        stop_y: list[float] = []
        stop_hover: list[str] = []

        for t in trades:
            if t.entry_date not in date_map or t.exit_date not in date_map:
                continue
            entry_ts = date_map[t.entry_date]
            exit_ts = date_map[t.exit_date]

            entry_value = t.entry_price * t.shares
            exit_value = t.exit_price * t.shares
            pnl_sign = "+" if t.pnl >= 0 else ""
            is_stop = abs(t.exit_price - t.stop_loss) < 1e-6

            entry_x.append(entry_ts)
            entry_y.append(t.entry_price)
            entry_hover.append(
                "<b>BUY</b><br>"
                f"Date: {t.entry_date}<br>"
                f"Price: ${t.entry_price:,.2f}<br>"
                f"Shares: {t.shares}<br>"
                f"Value: ${entry_value:,.0f}<br>"
                f"Stop: ${t.stop_loss:,.2f}"
            )

            exit_label = "STOP" if is_stop else "SELL"
            exit_hover_text = (
                f"<b>{exit_label}</b><br>"
                f"Date: {t.exit_date}<br>"
                f"Price: ${t.exit_price:,.2f}<br>"
                f"Shares: {t.shares}<br>"
                f"Value: ${exit_value:,.0f}<br>"
                f"P&L: {pnl_sign}${t.pnl:,.2f} "
                f"({pnl_sign}{t.pnl_pct:.2%})"
            )
            if is_stop:
                stop_x.append(exit_ts)
                stop_y.append(t.exit_price)
                stop_hover.append(exit_hover_text)
            else:
                sell_x.append(exit_ts)
                sell_y.append(t.exit_price)
                sell_hover.append(exit_hover_text)

            # Stop-loss envelope segment
            fig.add_trace(
                go.Scatter(
                    x=[entry_ts, exit_ts],
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
                    x=[entry_ts, exit_ts],
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
        if stop_x:
            fig.add_trace(
                go.Scatter(
                    x=stop_x,
                    y=stop_y,
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=14,
                        color="#F44336",
                        line=dict(width=2, color="white"),
                    ),
                    name="Stop",
                    hovertext=stop_hover,
                    hoverinfo="text",
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
                rangebreaks=[dict(values=self._missing_dates(df))],
            ),
            yaxis=dict(autorange=True, fixedrange=False),
            yaxis2=dict(autorange=True, fixedrange=False),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    @staticmethod
    def _missing_dates(df: pd.DataFrame) -> list[str]:
        """Return dates with no data between first and last trading day."""
        all_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
        missing = all_days.difference(df.index)
        return [d.strftime("%Y-%m-%d") for d in missing]

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
