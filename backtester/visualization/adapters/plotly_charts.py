import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pattern.domain.models import PatternSignal
from strategy.domain.models import EquityPoint
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
