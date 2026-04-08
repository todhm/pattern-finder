from abc import ABC, abstractmethod

from strategy.domain.models import StrategyConfig, StrategyResult


class StrategyRunnerPort(ABC):
    """Port: strategy execution interface."""

    @abstractmethod
    def run(self, config: StrategyConfig) -> StrategyResult: ...
