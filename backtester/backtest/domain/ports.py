from abc import ABC, abstractmethod

import pandas as pd

from backtest.domain.models import BacktestResult
from pattern.domain.models import PatternSignal


class BacktestEnginePort(ABC):
    """Port: backtesting engine interface."""

    @abstractmethod
    def run(
        self, df: pd.DataFrame, signals: list[PatternSignal]
    ) -> BacktestResult: ...
