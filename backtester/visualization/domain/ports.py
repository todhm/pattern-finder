from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go

from pattern.domain.models import PatternSignal
from strategy.domain.models import EquityPoint, Trade


class ChartBuilderPort(ABC):
    """Port: chart building interface."""

    @abstractmethod
    def build_candlestick_with_signals(
        self,
        df: pd.DataFrame,
        signals: list[PatternSignal],
        title: str = "",
    ) -> go.Figure: ...

    @abstractmethod
    def build_candlestick_with_trades(
        self,
        df: pd.DataFrame,
        trades: list[Trade],
        title: str = "",
    ) -> go.Figure: ...

    @abstractmethod
    def build_equity_curve(
        self,
        equity_points: list[EquityPoint],
        title: str = "",
    ) -> go.Figure: ...
