"""15-minute Wick Play strategy.

Mirror of :mod:`wedgepop_15m_strategy` but for the capitulation-
reversal Wick Play setup. The pattern is the same B-plan approach:
a thin subclass of the daily strategy that overrides only the
bar-identity hooks and the default parameter set, inheriting every
entry / exit / sizing rule unchanged.

Cadence reference::

    1 session (6.5h regular hours)   =   26 × 15m bars
    1 week   (5 sessions)            =  130 × 15m bars
    1 month  (~21 sessions)          =  546 × 15m bars

A note on the detector: ``WickPlayDetector`` bakes several lookback
windows in daily units (``psych_vol_lookback=20``,
``prior_trend_lookback=20``, ``pct_high_lookback=252``). Those params
accept any bar count at construction time, so
``build_wickplay_15m_detector()`` below rescales them to session-
count units and is the recommended default when wiring up the 15m
strategy from a Streamlit page.

Data constraint: yfinance caps 15m history at 60 calendar days, so
long back-tests require a different ``MarketDataPort`` adapter. The
strategy and detector are source-agnostic — the port contract alone
is enough.
"""

from __future__ import annotations

import pandas as pd

from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.adapters.wick_play import WickPlayDetector
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from strategy.adapters.multi_wickplay_strategy import MultiWickPlayStrategy
from strategy.adapters.wickplay_strategy import WickPlayStrategy
from strategy.domain.models import Trade


class Wickplay15mStrategy(WickPlayStrategy):
    """Wick Play strategy wired for 15-minute bars.

    Inherits every stop, trail, and exit rule from
    :class:`WickPlayStrategy`. Changes:

    - ``_interval = "15m"`` on the class so the inherited ``run()``
      path pulls 15m bars.
    - ``_strip_index_tz = False`` so the NY-tz intraday index stays
      tz-aware (required for ``_next_open_index`` to match against
      ``signal.timestamp``).
    - The four interval-identity hooks resolve to the exact
      ``DatetimeIndex`` value rather than the session date.
    - Default indicator periods rescaled to session-count units:
      ``ema_trail=26`` (≈ 1 session) and ``atr_period=26``.
      ``min_trail_bars=8`` (≈ 2 hours) so the EMA trail doesn't fire
      on the entry bar's immediate pullback.
    """

    _interval: str = "15m"
    _strip_index_tz: bool = False

    def __init__(
        self,
        market_data: MarketDataPort,
        detector: PatternDetector,
        exit_detector: PatternDetector | None = None,
        ema_trail: int = 26,          # 1 session
        atr_period: int = 26,         # 1 session
        min_trail_bars: int = 8,      # ~2 hours of settle-in after entry
        enable_same_day_reversal_exit: bool = False,
        max_same_day_close_location: float = 0.3,
        enable_gap_down_rejection: bool = True,
        max_entry_gap_down: float = 0.005,
        enable_breakeven_stop: bool = False,
        breakeven_arm_r_multiple: float = 1.5,
        breakeven_exit_offset_r: float = 0.5,
    ) -> None:
        super().__init__(
            market_data=market_data,
            detector=detector,
            exit_detector=exit_detector,
            ema_trail=ema_trail,
            atr_period=atr_period,
            min_trail_bars=min_trail_bars,
            enable_same_day_reversal_exit=enable_same_day_reversal_exit,
            max_same_day_close_location=max_same_day_close_location,
            enable_gap_down_rejection=enable_gap_down_rejection,
            max_entry_gap_down=max_entry_gap_down,
            enable_breakeven_stop=enable_breakeven_stop,
            breakeven_arm_r_multiple=breakeven_arm_r_multiple,
            breakeven_exit_offset_r=breakeven_exit_offset_r,
        )

    # ---- intraday bar-identity hooks ----

    @staticmethod
    def _signal_match_ts(signal: PatternSignal) -> pd.Timestamp:
        if signal.timestamp is None:
            raise ValueError(
                "Wickplay15mStrategy requires PatternSignal.timestamp "
                f"(got None on session {signal.date})."
            )
        return pd.Timestamp(signal.timestamp)

    @staticmethod
    def _signal_key(signal: PatternSignal) -> pd.Timestamp:
        if signal.timestamp is None:
            raise ValueError(
                "Wickplay15mStrategy requires PatternSignal.timestamp "
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
                "Wickplay15mStrategy expects populated exit_ts on every trade."
            )
        return pd.Timestamp(trade.exit_ts)


class MultiWickplay15mStrategy(MultiWickPlayStrategy):
    """15m universe-scanner version of Wick Play.

    Same walk algorithm as :class:`MultiWickPlayStrategy`; differences
    are limited to:

    - Injected per-ticker strategy defaults to
      :class:`Wickplay15mStrategy` so bar-identity hooks resolve to
      intraday timestamps.
    - ``_warmup_days`` drops to 30 so the universe scan stays inside
      yfinance's 60-day 15m cap.
    - ``min_bars`` raised so the slower indicators in the detector /
      strategy have enough warmup to converge before signals fire.
    """

    _warmup_days: int = 30

    def __init__(
        self,
        market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: PatternDetector,
        strategy: WickPlayStrategy | None = None,
        max_workers: int = 8,
        min_bars: int = 260,  # ≥10 sessions of warmup before the first signal
    ) -> None:
        super().__init__(
            market_data=market_data,
            universe_provider=universe_provider,
            detector=detector,
            strategy=strategy or Wickplay15mStrategy(
                market_data=market_data, detector=detector
            ),
            max_workers=max_workers,
            min_bars=min_bars,
        )


def build_wickplay_15m_detector(
    *,
    min_psych_score: int = 3,
    **overrides,
) -> WickPlayDetector:
    """Convenience factory: ``WickPlayDetector`` with 15m-appropriate
    defaults.

    The detector's daily-named parameters (``psych_vol_lookback=20``,
    ``prior_trend_lookback=20``, ``pct_high_lookback=252``,
    ``atr_period=14``, ``cooldown_bars=5``) are rescaled to session-
    count units so each carries its *intent* over to 15m bars:

    - ``psych_vol_lookback=520`` — 20 sessions of volume average
      (instead of 20 daily bars).
    - ``prior_trend_lookback=520`` — 20-session return gate.
    - ``pct_high_lookback=520`` — "recent high" within ~1 month of
      intraday bars (52-week high at 1-session-per-day resolution
      would need 6552 bars, far past yfinance's 15m cap).
    - ``atr_period=26`` — one session of ATR.
    - ``cooldown_bars=26`` — no back-to-back signals within a session.

    ``**overrides`` lets callers patch any default; everything else
    flows to ``WickPlayDetector.__init__`` unchanged.
    """
    defaults = dict(
        psych_vol_lookback=520,
        prior_trend_lookback=520,
        pct_high_lookback=520,
        sma_period=520,   # even when off, the default should match cadence
        atr_period=26,
        cooldown_bars=26,
        min_psych_score=min_psych_score,
    )
    defaults.update(overrides)
    return WickPlayDetector(**defaults)
