"""15-minute Wedge Pop strategy.

Thin subclass of :class:`WedgepopStrategy` that swaps in:

1. **Intraday bar-identity hooks** — ``_signal_match_ts``,
   ``_signal_key``, ``_bar_key``, ``_trade_exit_key`` all resolve to
   the exact 15m timestamp instead of the session date. This is the
   only behavioral change needed for the inherited entry / exit /
   sizing logic to run correctly on 15m bars: every distance
   threshold in the parent class is already ATR-scaled, so the
   filters and stops stay meaningful when the bar cadence shrinks.

2. **A default parameter set tuned to a 26-bars-per-session cadence.**
   The daily defaults were calibrated for a 1-bar-per-day view where
   ``ema_trail=10`` means "last two trading weeks"; on 15m bars that
   same period is a meaningless ~2.5 hours. We rescale indicator
   periods to session-count units so each parameter's *intent* —
   "one session of momentum", "~one hour fractals" — carries over.

   Cadence reference::

       1 session (6.5h regular hours)   =   26 × 15m bars
       1 week   (5 sessions)            =  130 × 15m bars
       1 month  (~21 sessions)          =  546 × 15m bars

Parameters intentionally **NOT** rescaled:

- ATR multiples (``extension_atr_mult``, ``climax_atr_mult``, swing
  tolerance, resistance buffers). These are dimensionless statistical
  thresholds ("2.5 ATRs above EMA is euphoric") and translate without
  change.
- R-multiples for breakeven arming. Same reasoning.
- ``max_holding_days`` on ``StrategyConfig`` — passed per-call by the
  caller. Note its semantic is "max BARS held" regardless of interval,
  so daily callers pass session counts and 15m callers should pass
  bar counts (e.g. 520 ≈ one month of 15m bars for a swing hold,
  26 for intraday-only).

Data requirements: yfinance caps 15m history at 60 calendar days, so
back-tests past that window require a different ``MarketDataPort``
adapter (Alpaca / Polygon / …). The strategy itself is source-
agnostic — the port contract from ``data/domain/ports.py`` is all it
needs.
"""

from __future__ import annotations

import pandas as pd

from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from strategy.adapters.multi_wedgepop_strategy import MultiWedgepopStrategy
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import Trade


class Wedgepop15mStrategy(WedgepopStrategy):
    """Wedge Pop strategy wired for 15-minute bars.

    Inherits every entry filter, position-sizing rule, and exit path
    from :class:`WedgepopStrategy` verbatim. Only the four
    bar-identity hooks are overridden (to key off
    ``PatternSignal.timestamp`` instead of ``date``) and the default
    indicator periods are rescaled from a 1-bar-per-day cadence to a
    26-bar-per-session cadence.

    All ``**overrides`` forward to the parent initializer, so callers
    can still toggle individual filters (``enable_swing_resistance_filter``,
    ``enable_breakeven_stop``, ATR multiples, …) without patching this
    class.
    """

    # Forwarded to ``MarketDataPort.fetch_ohlcv`` by the inherited
    # ``run()`` path so the same entry point pulls 15m bars.
    _interval: str = "15m"

    def __init__(
        self,
        market_data: MarketDataPort,
        detector: PatternDetector,
        # --- intraday period defaults (reasoned in the module docstring) ---
        ema_trail: int = 26,            # 1 session
        ema_slow: int = 78,             # 3 sessions
        atr_period: int = 26,           # 1 session
        swing_pivot_left: int = 4,      # ~1-hour fractal
        swing_pivot_right: int = 4,
        swing_pivot_lookback: int = 520,  # 20 sessions of pivots
        # --- filters that don't make sense by default on 15m ---
        # The 1d default (``-0.01`` over 20 bars = "worse than -1%/month")
        # is a rollover-trend guard. On 15m, 78 bars ≈ 3 sessions and a
        # -1% threshold is common intraday noise; leave the slope
        # filter off by default and let callers opt in with a
        # bar-appropriate value.
        max_ema_slope_decline: float | None = None,
        min_ema_slow_slope: float | None = None,
        **overrides,
    ) -> None:
        super().__init__(
            market_data=market_data,
            detector=detector,
            ema_trail=ema_trail,
            ema_slow=ema_slow,
            atr_period=atr_period,
            swing_pivot_left=swing_pivot_left,
            swing_pivot_right=swing_pivot_right,
            swing_pivot_lookback=swing_pivot_lookback,
            max_ema_slope_decline=max_ema_slope_decline,
            min_ema_slow_slope=min_ema_slow_slope,
            **overrides,
        )

    # ---- intraday bar-identity hooks ----
    #
    # Default ``WedgepopStrategy`` keys by session ``date`` (midnight
    # Timestamp), which is fine when one bar == one session. For 15m
    # we need to distinguish bars within the same session, so every
    # hook returns the exact ``DatetimeIndex`` value instead.

    @staticmethod
    def _signal_match_ts(signal: PatternSignal) -> pd.Timestamp:
        if signal.timestamp is None:
            raise ValueError(
                "Wedgepop15mStrategy requires PatternSignal.timestamp "
                "(got None). Intraday detectors must populate the "
                f"timestamp field on every signal; session {signal.date} "
                "is not a unique bar identifier at 15m resolution."
            )
        return pd.Timestamp(signal.timestamp)

    @staticmethod
    def _signal_key(signal: PatternSignal) -> pd.Timestamp:
        if signal.timestamp is None:
            raise ValueError(
                "Wedgepop15mStrategy requires PatternSignal.timestamp "
                f"(got None on session {signal.date})."
            )
        return pd.Timestamp(signal.timestamp)

    @staticmethod
    def _bar_key(df: pd.DataFrame, i: int) -> pd.Timestamp:
        return pd.Timestamp(df.index[i])

    @staticmethod
    def _trade_exit_key(trade: Trade) -> pd.Timestamp:
        if trade.exit_ts is None:
            raise ValueError(
                "Wedgepop15mStrategy expects populated exit_ts on every "
                "trade. The generalized WedgepopStrategy always sets it "
                "(including on the 1d path), so a missing value indicates "
                "a hand-constructed Trade bypassing the strategy runner."
            )
        return pd.Timestamp(trade.exit_ts)


class MultiWedgepop15mStrategy(MultiWedgepopStrategy):
    """15m universe-scanner version.

    Same walk algorithm as :class:`MultiWedgepopStrategy` — the only
    overrides are:

    - Default injected ``strategy`` is a :class:`Wedgepop15mStrategy`,
      so the inherited bar-identity hooks resolve to the 15m
      timestamp semantics throughout the scan → walk pipeline.
    - ``_warmup_days`` is dropped from 400 to 30 — yfinance caps 15m
      history at 60 calendar days, so anything larger would exceed
      the upstream fetch window on the first tick.
    - ``min_bars`` default raised so that the slower EMA (78 bars by
      default in the injected 15m strategy) has enough history to
      converge before the first signal check.
    """

    # 30 calendar days ≈ 20 sessions ≈ 520 15m bars — sufficient
    # warmup for the 78-bar slow EMA and 26-bar ATR used by
    # ``Wedgepop15mStrategy`` defaults, and well within yfinance's
    # 60-day intraday cap.
    _warmup_days: int = 30

    def __init__(
        self,
        market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: PatternDetector,
        strategy: WedgepopStrategy | None = None,
        max_workers: int = 8,
        min_bars: int = 100,
    ) -> None:
        super().__init__(
            market_data=market_data,
            universe_provider=universe_provider,
            detector=detector,
            strategy=strategy or Wedgepop15mStrategy(
                market_data=market_data, detector=detector
            ),
            max_workers=max_workers,
            min_bars=min_bars,
        )
