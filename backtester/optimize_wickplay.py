"""Ablation sweep for Wick Play strategy defaults.

Runs :class:`MultiWickPlayStrategy` against a sample of the Nasdaq
universe (default: first 500 common-stock tickers alphabetically),
varying the index-regime and FOMC-blackout gates that were added in
response to the Multi-Wick Play post-mortem. Other knobs stay at
their docstring-documented defaults so the comparison isolates the
effect of the new filters.

Runs from inside the container:

    docker compose exec backtester python optimize_wickplay.py

Output: a console table sorted by net return. First invocation primes
the parquet cache (~1-2 min for 500 tickers); subsequent runs reuse
the cache so variant configs finish in seconds each.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.macro_calendar import blackout_dates as build_blackout_dates
from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import (
    ExhaustionExtensionTopDetector,
)
from pattern.adapters.wick_play import WickPlayDetector
from strategy.adapters.multi_wickplay_strategy import MultiWickPlayStrategy
from strategy.adapters.wickplay_strategy import WickPlayStrategy
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule


# ---- Sweep parameters ------------------------------------------------

START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 4, 22)
MAX_TICKERS = 500
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
MAX_HOLDING_DAYS = 40
REGIME_SYMBOL = "^GSPC"


@dataclass
class SweepConfig:
    """One row in the ablation table — a named set of detector overrides."""
    name: str
    detector_overrides: dict[str, Any]

    def __str__(self) -> str:
        return self.name


def build_sweep() -> list[SweepConfig]:
    """Ablation grid. Each config layers exactly one dimension on top
    of the baseline so the marginal effect of every gate is readable
    in isolation, then a final "all-on" row shows the combined effect."""
    base: dict[str, Any] = {}
    return [
        SweepConfig("baseline", dict(base)),
        SweepConfig(
            "+fomc_blackout",
            {**base, "__blackout_fomc": True},
        ),
        SweepConfig(
            "+regime_sma50",
            {**base, "regime_min_above_sma": 50},
        ),
        SweepConfig(
            "+regime_sma20",
            {**base, "regime_min_above_sma": 20},
        ),
        SweepConfig(
            "+regime_dd5_2pct",
            {**base, "regime_max_n_day_drawdown": (5, 0.02)},
        ),
        SweepConfig(
            "+regime_dd5_3pct",
            {**base, "regime_max_n_day_drawdown": (5, 0.03)},
        ),
        SweepConfig(
            "+fomc_plus_sma50",
            {
                **base,
                "__blackout_fomc": True,
                "regime_min_above_sma": 50,
            },
        ),
        SweepConfig(
            "+fomc_plus_dd5_2pct",
            {
                **base,
                "__blackout_fomc": True,
                "regime_max_n_day_drawdown": (5, 0.02),
            },
        ),
        SweepConfig(
            "+all_gates",
            {
                **base,
                "__blackout_fomc": True,
                "regime_min_above_sma": 50,
                "regime_max_n_day_drawdown": (5, 0.02),
            },
        ),
        SweepConfig(
            "+all_gates_cpi",
            {
                **base,
                "__blackout_fomc": True,
                "__blackout_cpi": True,
                "regime_min_above_sma": 50,
                "regime_max_n_day_drawdown": (5, 0.02),
            },
        ),
    ]


# ---- Detector / strategy construction --------------------------------


def _detector_defaults(
    regime_df,
    blackout: set[date] | None,
    **overrides,
) -> WickPlayDetector:
    """Build a WickPlayDetector with sensible defaults + per-config
    overrides. Defaults mirror the Streamlit page values so comparisons
    match what a user would see interactively.
    """
    kwargs: dict[str, Any] = dict(
        min_upper_wick_ratio=0.5,
        max_volume_dryup=1.0,
        breakout_trigger="wick_high",
        stop_mode="wick_low",
        max_wick_range_atr=2.5,
        cooldown_bars=5,
        psych_vol_lookback=20,
        psych_wick_vol_exhaustion_mult=1.0,
        psych_breakout_vol_expansion_mult=1.0,
        psych_prior_red_streak=2,
        psych_dramatic_wick_ratio=0.65,
        min_psych_score=3,
        min_breakout_strength_atr=0.3,
        min_prior_trend_20d=-0.01,
        max_prior_trend_20d=None,
        min_wick_close_location=0.15,
        min_breakout_close_location=None,
        require_above_sma200=False,
        min_pct_of_52w_high=None,
    )
    # Internal sweep flags start with __ and are stripped before the
    # real constructor call — they control adapter-level plumbing
    # (which regime_df / blackout to pass), not detector internals.
    sweep_only = {k: v for k, v in overrides.items() if k.startswith("__")}
    detector_only = {k: v for k, v in overrides.items() if not k.startswith("__")}
    kwargs.update(detector_only)
    # Regime data is only meaningful when at least one regime gate
    # is configured; otherwise pass None to avoid pointless work.
    has_regime_gate = (
        kwargs.get("regime_min_above_sma") is not None
        or kwargs.get("regime_min_above_ema") is not None
        or kwargs.get("regime_max_n_day_drawdown") is not None
    )
    kwargs["regime_df"] = regime_df if has_regime_gate else None
    kwargs["blackout_dates"] = blackout if sweep_only.get("__blackout_fomc") else None
    return WickPlayDetector(**kwargs)


# ---- Result plumbing -------------------------------------------------


@dataclass
class SweepResult:
    name: str
    trades: int
    win_rate: float
    total_return_pct: float
    final_capital: float
    max_drawdown_pct: float
    total_commission: float
    runtime_s: float


def _fmt_row(r: SweepResult) -> str:
    return (
        f"{r.name:<24} trades={r.trades:>4}  win={r.win_rate:>6.1%}  "
        f"ret={r.total_return_pct:>+7.2%}  final=${r.final_capital:>11,.0f}  "
        f"maxDD={r.max_drawdown_pct:>6.2%}  fees=${r.total_commission:>8,.0f}  "
        f"t={r.runtime_s:>5.1f}s"
    )


def main() -> None:
    print(
        f"[sweep] window={START_DATE} → {END_DATE}  "
        f"max_tickers={MAX_TICKERS}  capital=${INITIAL_CAPITAL:,.0f}  "
        f"risk={RISK_PER_TRADE:.0%}"
    )

    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    universe_provider = default_universe_provider()

    tickers = universe_provider.get_tickers("nasdaq_full")
    print(f"[sweep] nasdaq_full returned {len(tickers)} tickers — sampling first {MAX_TICKERS}")
    tickers = sorted(tickers)[:MAX_TICKERS]

    # --- Fetch regime benchmark once ---
    print(f"[sweep] fetching regime symbol {REGIME_SYMBOL}...")
    regime_df = market_data.fetch_ohlcv(
        REGIME_SYMBOL,
        START_DATE - timedelta(days=400),
        END_DATE,
    )
    print(f"[sweep] regime_df rows={len(regime_df)}")

    # --- Build blackout sets ---
    blackout_fomc = build_blackout_dates(
        start=START_DATE - timedelta(days=400),
        end=END_DATE,
        include_fomc=True,
        include_cpi=False,
        window_days=0,
    )
    blackout_fomc_cpi = build_blackout_dates(
        start=START_DATE - timedelta(days=400),
        end=END_DATE,
        include_fomc=True,
        include_cpi=True,
        window_days=0,
    )
    print(
        f"[sweep] FOMC dates in window: {sum(1 for d in blackout_fomc if START_DATE <= d <= END_DATE)}  "
        f"FOMC+CPI: {sum(1 for d in blackout_fomc_cpi if START_DATE <= d <= END_DATE)}"
    )

    fee_schedule = TossFeeSchedule(
        buy_commission_pct=0.001, sell_commission_pct=0.001, sec_fee_pct=0.0000229
    )

    base_cfg = MultiStrategyConfig(
        universe="nasdaq_full",
        start_date=START_DATE,
        end_date=END_DATE,
        pattern_name="wick_play",
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        max_holding_days=MAX_HOLDING_DAYS,
        max_tickers=MAX_TICKERS,
        fee_schedule=fee_schedule,
    )

    results: list[SweepResult] = []
    for sweep in build_sweep():
        overrides = dict(sweep.detector_overrides)
        use_cpi = overrides.pop("__blackout_cpi", False)
        blackout = (
            blackout_fomc_cpi if use_cpi else blackout_fomc
        )
        detector = _detector_defaults(
            regime_df=regime_df,
            blackout=blackout,
            **overrides,
        )
        exit_detector = ExhaustionExtensionTopDetector(
            extension_atr_mult=2.5,
            min_slow_slope=0.005,
            ema_fast=10,
        )
        per_ticker = WickPlayStrategy(
            market_data=market_data,
            detector=detector,
            exit_detector=exit_detector,
            ema_trail=10,
            min_trail_bars=2,
            enable_same_day_reversal_exit=False,
            enable_gap_down_rejection=True,
            max_entry_gap_down=0.005,
            enable_breakeven_stop=False,
        )
        runner = MultiWickPlayStrategy(
            market_data=market_data,
            universe_provider=_StaticList(tickers),
            detector=detector,
            strategy=per_ticker,
            max_workers=8,
        )

        t0 = time.monotonic()
        result = runner.run(base_cfg)
        dt = time.monotonic() - t0
        sr = SweepResult(
            name=sweep.name,
            trades=result.trades_taken,
            win_rate=result.win_rate,
            total_return_pct=result.total_return_pct,
            final_capital=result.final_capital,
            max_drawdown_pct=result.max_drawdown_pct,
            total_commission=result.total_commission,
            runtime_s=dt,
        )
        results.append(sr)
        print(_fmt_row(sr))

    # --- Summary table sorted by return ---
    print()
    print("=" * 120)
    print("SORTED BY NET RETURN (descending)")
    print("=" * 120)
    for r in sorted(results, key=lambda x: x.total_return_pct, reverse=True):
        print(_fmt_row(r))


class _StaticList:
    """Tiny UniverseProviderPort impl so we don't re-fetch the Nasdaq
    list for each sweep row. Holds a fixed ticker list."""

    def __init__(self, tickers: list[str]):
        self._tickers = list(tickers)

    def get_tickers(self, universe: str) -> list[str]:
        return list(self._tickers)


if __name__ == "__main__":
    main()
