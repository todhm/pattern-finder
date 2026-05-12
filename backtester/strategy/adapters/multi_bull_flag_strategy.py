"""Universe-wide Bull Flag strategy.

Single-portfolio runner that walks an NY-equity universe day-by-day
and, when multiple tickers fire on the same session, picks the one
with the strongest Ross-style stock-selection profile:

    1차 정렬:  RVOL 내림차순  (오늘의 runner — 거래량이 몰린 쪽이 follow-through ↑)
    2차 정렬:  float 오름차순  (작은 float = 같은 매수 압력에 더 큰 % 이동)
    3차 정렬:  entry_ts 오름차순  (영상 sweet spot = 09:30~10:30 첫 풀백)

Lifecycle (mirrors Multi FCR / Multi Wedgepop):

    Phase 1 — **scan** (parallel)
        For each ticker: fetch 1m + 5m + daily, lookup float via
        yfinance (throttled), run :class:`BullFlagDetector`.
    Phase 2a — **collect**
        Group signals by ``session_date``.
    Phase 2b — **walk** (sequential)
        For each session date: sort candidates by (RVOL desc, float asc,
        entry_ts asc), pick the top one, delegate to
        :class:`BullFlagStrategy` for the single-trade simulation.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_t, time, timedelta
from typing import Any

import pandas as pd


class _DailyFiltered:
    """Sentinel — ticker was successfully fetched + checked but had
    no qualifying session in this chunk's window. Distinguishes
    early-skipped tickers from actual data-fetch failures so
    ``failed_tickers`` doesn't include them.
    """


DAILY_FILTERED = _DailyFiltered()

from data.domain.market_calendar import MarketCalendar, NY
from data.domain.ports import (
    FundamentalsPort,
    MarketDataPort,
    UniverseProviderPort,
)
from pattern.adapters.bull_flag import BullFlagDetector, BullFlagSignal
from strategy.adapters.bull_flag_strategy import BullFlagStrategy
from strategy.domain.models import (
    EquityPoint,
    MultiStrategyConfig,
    MultiStrategyResult,
    MultiTrade,
    StrategyConfig,
)


class MultiBullFlagStrategy:
    """Single-portfolio Bull Flag strategy across a ticker universe.

    Tiebreaker for same-day signals: ``(RVOL desc, float asc, entry_ts asc)``.
    See module docstring for rationale.
    """

    _warmup_days: int = 5

    def __init__(
        self,
        market_data: MarketDataPort,
        market_data_5m: MarketDataPort,
        daily_market_data: MarketDataPort,
        fundamentals: FundamentalsPort,
        universe_provider: UniverseProviderPort,
        # Detector / strategy params injected via factory args so the
        # page can wire user inputs without re-instantiating the runner.
        detector_factory,
        strategy_factory,
        market: MarketCalendar = NY,
        max_workers: int = 4,
        min_bars: int = 60,
        require_float_filter: bool = True,
        chunk_months: int = 1,
        # Unfiltered intraday source — page wires the composed market
        # data WITHOUT the RegularSessionFilterAdapter so PM bars
        # (04:00–09:29 ET) are visible. None disables the PM high
        # filter (detector skips the gate).
        raw_market_data: MarketDataPort | None = None,
    ) -> None:
        self._market_data = market_data
        self._market_data_5m = market_data_5m
        self._daily_market_data = daily_market_data
        self._fundamentals = fundamentals
        self._universe_provider = universe_provider
        # Factories take ``float_shares`` + ``splits`` and return
        # configured detector / strategy. This keeps detector params
        # owned by the page.
        self._detector_factory = detector_factory
        self._strategy_factory = strategy_factory
        self._market = market
        self._max_workers = max_workers
        self._min_bars = min_bars
        self._require_float_filter = require_float_filter
        self._raw_market_data = raw_market_data
        # Long windows (e.g. 16 months × 2,262 NASDAQ tickers × 1m
        # bars ≈ 25GB) OOM-kill the container. ``chunk_months`` splits
        # the request window into smaller passes so the in-memory
        # ``ticker_state`` for each pass fits comfortably; Mongo cache
        # absorbs the re-fetch across chunk boundaries without an
        # upstream network call.
        self._chunk_months = chunk_months

    # ---- public API ----

    def prefetch(
        self,
        tickers: list[str],
        start_date: date_t,
        end_date: date_t,
    ) -> None:
        """**Phase 1** — ingest the full window for ``tickers`` into Mongo.

        Walks each ticker once, fetching (1m, 5m, daily, fundamentals)
        through the cache stack so the day-chunked Mongo collection
        is populated. Per-ticker DataFrames are dropped immediately
        after the cache write — memory high-water is one ticker's
        worth of data (~10MB for a 1-year 1m window) × ``max_workers``.

        Idempotent: re-running ``prefetch`` on already-cached data is
        a no-op aside from Mongo lookups (no upstream calls), so a
        sweep can call it once per session and re-use the populated
        cache across every iteration's :meth:`simulate` call.

        ``tickers`` is passed explicitly rather than resolved through
        ``universe_provider`` so the caller controls (a) the universe
        selection and (b) the slice (e.g. ``[:max_tickers]``).
        """
        daily_start = start_date - timedelta(days=120)
        intraday_start = start_date - timedelta(days=self._warmup_days)

        def _ingest(ticker: str) -> None:
            try:
                self._market_data.fetch_ohlcv(
                    ticker, intraday_start, end_date, interval="1m"
                )
            except Exception:
                pass
            try:
                self._market_data_5m.fetch_ohlcv(
                    ticker, intraday_start, end_date, interval="5m"
                )
            except Exception:
                pass
            try:
                self._daily_market_data.fetch_ohlcv(
                    ticker, daily_start, end_date
                )
            except Exception:
                pass
            try:
                self._fundamentals.fetch(ticker)
            except Exception:
                pass
            # No state retained — return value discarded after Mongo
            # write. Local DataFrames go out of scope here.

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = [ex.submit(_ingest, t) for t in tickers]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    pass
        gc.collect()

    def simulate(
        self,
        config: MultiStrategyConfig,
        tickers: list[str] | None = None,
    ) -> MultiStrategyResult:
        """**Phase 2** — run the chunked simulation on already-cached data.

        Assumes :meth:`prefetch` has populated Mongo for the same
        (tickers × window). If the cache misses for some ticker /
        day, the wrapped market-data stack will route through its
        normal upstream path — so a cold call to ``simulate`` still
        works, it just pays the upstream cost lazily per chunk.

        ``tickers`` defaults to the configured universe slice
        (``universe_provider.get_tickers(config.universe)[:max_tickers]``)
        so existing callers don't break — pass explicitly when the
        caller already has a list (e.g. resolved once for ``prefetch``).
        """
        if tickers is None:
            tickers = self._universe_provider.get_tickers(config.universe)
            if config.max_tickers is not None:
                tickers = tickers[: config.max_tickers]

        chunks = self._split_window(
            config.start_date, config.end_date, self._chunk_months
        )

        all_trades: list[MultiTrade] = []
        all_failed: set[str] = set()
        total_signals = 0
        capital = config.initial_capital
        peak = capital
        max_dd = 0.0
        equity_curve: list[EquityPoint] = [
            EquityPoint(date=config.start_date, equity=capital)
        ]

        for chunk_idx, (c_start, c_end) in enumerate(chunks):
            chunk_cfg = config.model_copy(update={
                "start_date": c_start, "end_date": c_end,
                "initial_capital": capital,
            })

            ticker_state, failed = self._scan_universe(tickers, chunk_cfg)
            all_failed.update(failed)
            signals_by_date, sig_count = self._collect_signals(
                ticker_state, chunk_cfg
            )
            total_signals += sig_count
            trades, curve, capital, chunk_dd = self._walk_signals(
                signals_by_date, ticker_state, chunk_cfg
            )
            all_trades.extend(trades)
            # Merge equity curve — drop the chunk's seed point (= prior
            # chunk's final capital) to avoid a duplicate at the boundary.
            equity_curve.extend(curve[1:])
            # Track max-DD globally — each chunk reports its own DD but
            # cross-chunk peaks need running comparison too.
            peak = max(peak, capital)
            if peak > 0:
                running_dd = (peak - capital) / peak
                max_dd = max(max_dd, max(running_dd, chunk_dd))

            del ticker_state, signals_by_date, trades, curve
            gc.collect()

        return self._build_result(
            config=config,
            tickers_scanned=len(tickers),
            total_signals=total_signals,
            trades=all_trades,
            equity_curve=equity_curve,
            final_capital=capital,
            max_dd=max_dd,
            failed=sorted(all_failed),
        )

    def run(self, config: MultiStrategyConfig) -> MultiStrategyResult:
        """Convenience: :meth:`prefetch` then :meth:`simulate`.

        Used by pages that don't separate ingestion from simulation.
        Sweep / batch callers should invoke prefetch once + simulate
        many times for efficiency.
        """
        tickers = self._universe_provider.get_tickers(config.universe)
        if config.max_tickers is not None:
            tickers = tickers[: config.max_tickers]
        self.prefetch(tickers, config.start_date, config.end_date)
        return self.simulate(config, tickers=tickers)

    @staticmethod
    def _split_window(
        start: date_t, end: date_t, chunk_months: int
    ) -> list[tuple[date_t, date_t]]:
        """Split [start, end] into adjacent windows of ``chunk_months``.

        Last chunk may be shorter. Each (start_i, end_i) pair stays
        within the original window — no overlap, no gap.
        """
        if chunk_months <= 0:
            return [(start, end)]
        chunks: list[tuple[date_t, date_t]] = []
        cursor = start
        while cursor <= end:
            # Advance by chunk_months calendar months, clamp to end.
            year = cursor.year + (cursor.month - 1 + chunk_months) // 12
            month = ((cursor.month - 1 + chunk_months) % 12) + 1
            try:
                next_cursor = date_t(year, month, cursor.day)
            except ValueError:
                # day-of-month doesn't exist next month — clamp to 1st.
                next_cursor = date_t(year, month, 1)
            chunk_end = min(next_cursor - timedelta(days=1), end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)
        return chunks

    # ---- phase 1: scan ----

    def _scan_universe(
        self, tickers: list[str], config: MultiStrategyConfig
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Run per-ticker scan in parallel. Returns (state_dict, failed_list).

        ``failed_list`` only contains tickers with **actual** data-fetch
        problems — exceptions or missing daily data. Tickers that were
        successfully checked but didn't pass the daily gate
        (``DAILY_FILTERED`` sentinel) are silently dropped from both
        the state dict and the failed list.
        """
        ticker_state: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        self._last_fetch_error: str | None = None
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(self._scan_ticker, t, config): t for t in tickers}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    state = fut.result()
                except Exception as exc:
                    self._last_fetch_error = f"{type(exc).__name__}: {exc}"
                    failed.append(t)
                    continue
                if isinstance(state, _DailyFiltered):
                    # Verified but no qualifying session — not failed.
                    continue
                if state is None:
                    failed.append(t)
                    continue
                ticker_state[t] = state
        return ticker_state, failed

    def _scan_ticker(
        self, ticker: str, config: MultiStrategyConfig
    ) -> dict[str, Any] | None | _DailyFiltered:
        # yfinance daily history padded for RVOL / split-blackout calcs.
        daily_start = config.start_date - timedelta(days=120)
        intraday_start = config.start_date - timedelta(days=self._warmup_days)

        # ============ Phase A — cheap pre-filter ============
        # Fetch daily + fundamentals FIRST. ~95% of NASDAQ tickers
        # never pass the 4-criteria gate on a given chunk, and the
        # intraday/PM fetches dominate Mongo I/O cost. Running the
        # daily gate up front (same logic as the detector's
        # ``_qualifying_sessions``) lets us short-circuit those
        # tickers before paying for 1m/5m/PM reads.
        try:
            df_daily = self._daily_market_data.fetch_ohlcv(
                ticker, daily_start, config.end_date
            )
        except Exception:
            raise
        if df_daily is None or df_daily.empty:
            return None

        fundamentals = self._fundamentals.fetch(ticker)
        float_shares = fundamentals.float_shares
        splits = fundamentals.splits
        if float_shares is None and self._require_float_filter:
            return None

        # Build a probe detector to reuse ``_qualifying_sessions``
        # logic with the exact same params (max_rvol, max_gap_pct,
        # daily_trend SMA, splits/price_floor — all the daily-level
        # gates). pm_high_by_date isn't needed for daily gates.
        probe = self._detector_factory(
            float_shares=float_shares,
            splits=splits,
            pm_high_by_date={},
        )
        qualifying = probe._qualifying_sessions(df_daily)
        # Trim to chunk window — qualifying dict spans full daily
        # history (120-day padding), but only signals inside [start, end]
        # can produce trades for this chunk.
        in_window = {
            d: m for d, m in qualifying.items()
            if config.start_date <= d <= config.end_date
        }
        if not in_window:
            # No qualifying session — short-circuit. NOT a fetch
            # failure; the daily gate just didn't open. Return the
            # sentinel so ``_scan_universe`` can keep these out of
            # the ``failed_tickers`` bucket.
            return DAILY_FILTERED  # type: ignore[return-value]

        # ============ Phase B — full fetch (qualifying tickers only) ============
        try:
            df_intraday = self._market_data.fetch_ohlcv(
                ticker, intraday_start, config.end_date, interval="1m"
            )
            df_5m = self._market_data_5m.fetch_ohlcv(
                ticker, intraday_start, config.end_date, interval="5m"
            )
        except Exception:
            raise
        if df_intraday is None or df_intraday.empty or len(df_intraday) < self._min_bars:
            return None

        # tz-convert intraday once.
        if (
            df_intraday.index.tz is not None
            and str(df_intraday.index.tz) != self._market.tz
        ):
            df_intraday = df_intraday.copy()
            df_intraday.index = df_intraday.index.tz_convert(self._market.tz)
        if (
            df_5m is not None
            and not df_5m.empty
            and df_5m.index.tz is not None
            and str(df_5m.index.tz) != self._market.tz
        ):
            df_5m = df_5m.copy()
            df_5m.index = df_5m.index.tz_convert(self._market.tz)
        if df_5m is not None and df_5m.empty:
            df_5m = None

        # Pre-market high per session (Ross: "PM high = breakout
        # reference"). Only fetched when ``raw_market_data`` is wired
        # — the same composed adapter but without the RegularSession
        # filter so 04:00–09:29 bars are visible.
        pm_high_by_date: dict[date_t, float] = {}
        if self._raw_market_data is not None:
            try:
                raw_df = self._raw_market_data.fetch_ohlcv(
                    ticker, intraday_start, config.end_date, interval="1m"
                )
                if raw_df is not None and not raw_df.empty:
                    if raw_df.index.tz is None:
                        raw_df = raw_df.copy()
                        raw_df.index = raw_df.index.tz_localize(self._market.tz)
                    elif str(raw_df.index.tz) != self._market.tz:
                        raw_df = raw_df.copy()
                        raw_df.index = raw_df.index.tz_convert(self._market.tz)
                    pm = raw_df.between_time("04:00", "09:29")
                    if not pm.empty:
                        for d, group in pm.groupby(pm.index.date):
                            pm_high_by_date[d] = float(group["High"].max())
            except Exception:
                # PM fetch failure → just disable the gate for this
                # ticker, don't bail the whole scan.
                pm_high_by_date = {}

        # Rebuild detector with the now-populated pm_high_by_date.
        # The probe detector with empty PM dict was only used for the
        # cheap qualifying check; its other gates (RVOL, gap, daily
        # trend, splits, price-floor) match what ``detect`` will run
        # again — slight duplication but each path is identical.
        detector = self._detector_factory(
            float_shares=float_shares,
            splits=splits,
            pm_high_by_date=pm_high_by_date,
        )
        signals = detector.detect(df_intraday, df_daily, df_5m=df_5m)

        # Daily RVOL by date — used for tiebreaker.
        avg_vol = df_daily["Volume"].rolling(50, min_periods=20).mean()
        drvol = df_daily["Volume"] / avg_vol
        drvol_by_date = {
            ts.date(): float(v)
            for ts, v in drvol.items()
            if pd.notna(v)
        }
        return {
            "df_intraday": df_intraday,
            "df_5m": df_5m,
            "df_daily": df_daily,
            "signals": signals,
            "drvol_by_date": drvol_by_date,
            "float_shares": float_shares,
            "splits": splits,
            "pm_high_by_date": pm_high_by_date,
        }

    # ---- phase 2a: collect ----

    def _collect_signals(
        self,
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[dict[date_t, list[dict[str, Any]]], int]:
        by_date: dict[date_t, list[dict[str, Any]]] = defaultdict(list)
        total = 0
        for ticker, state in ticker_state.items():
            drvol_by_date: dict[date_t, float] = state["drvol_by_date"]
            float_shares = state["float_shares"]
            for sig in state["signals"]:
                if (
                    sig.session_date < config.start_date
                    or sig.session_date > config.end_date
                ):
                    continue
                rvol = drvol_by_date.get(sig.session_date, sig.rvol)
                by_date[sig.session_date].append({
                    "ticker": ticker,
                    "signal": sig,
                    "rvol": float(rvol),
                    "float_shares": (
                        float(float_shares) if float_shares else float("inf")
                    ),
                    "entry_ts": pd.Timestamp(sig.entry_ts),
                })
                total += 1
        return by_date, total

    # ---- phase 2b: walk ----

    def _walk_signals(
        self,
        signals_by_date: dict[date_t, list[dict[str, Any]]],
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[list[MultiTrade], list[EquityPoint], float, float]:
        capital = config.initial_capital
        peak = capital
        max_dd = 0.0
        trades: list[MultiTrade] = []
        curve: list[EquityPoint] = [
            EquityPoint(date=config.start_date, equity=capital)
        ]
        for sess_date in sorted(signals_by_date.keys()):
            # Tiebreaker: RVOL desc → float asc → entry_ts asc.
            ranked = sorted(
                signals_by_date[sess_date],
                key=lambda c: (-c["rvol"], c["float_shares"], c["entry_ts"]),
            )
            best = ranked[0]
            ticker = best["ticker"]
            sig: BullFlagSignal = best["signal"]
            multi_trade = self._execute_one(
                ticker=ticker,
                state=ticker_state[ticker],
                signal=sig,
                fit=best,
                capital=capital,
                config=config,
            )
            if multi_trade is None:
                continue
            capital += multi_trade.pnl
            peak = max(peak, capital)
            if peak > 0:
                max_dd = max(max_dd, (peak - capital) / peak)
            trades.append(multi_trade)
            curve.append(
                EquityPoint(date=multi_trade.exit_date, equity=round(capital, 2))
            )
        return trades, curve, capital, max_dd

    def _execute_one(
        self,
        ticker: str,
        state: dict[str, Any],
        signal: BullFlagSignal,
        fit: dict[str, Any],
        capital: float,
        config: MultiStrategyConfig,
    ) -> MultiTrade | None:
        per_config = StrategyConfig(
            ticker=ticker,
            start_date=config.start_date,
            end_date=config.end_date,
            pattern_name=config.pattern_name,
            initial_capital=capital,
            risk_per_trade=config.risk_per_trade,
            max_holding_days=config.max_holding_days,
        )
        # Build a fresh strategy per ticker — the factory injects the
        # *same* detector that produced this signal so adds / TP /
        # stops behave identically to the single-ticker page.
        detector = self._detector_factory(
            float_shares=state["float_shares"],
            splits=state["splits"],
            pm_high_by_date=state.get("pm_high_by_date", {}),
        )
        strategy = self._strategy_factory(detector=detector)
        sim_trades, _curve = strategy._simulate(
            df=state["df_intraday"],
            signals=[signal],
            config=per_config,
        )
        if not sim_trades:
            return None
        trade = sim_trades[0]
        # Toss fees already applied inside ``BullFlagStrategy._simulate``
        # (it carries ``fee_schedule``). For Multi reporting we surface
        # gross + commission separately so the table matches other
        # multi pages. Recompute commission for display from the same
        # schedule used by the strategy.
        commission = config.fee_schedule.round_trip(
            trade.entry_price, trade.exit_price, trade.shares
        )
        # Note: trade.pnl already nets commission (BullFlagStrategy
        # tracks commission through the position lifecycle including
        # adds). Use it directly as net_pnl.
        net_pnl = trade.pnl
        gross_pnl = net_pnl + commission
        cost_basis = trade.entry_price * trade.shares
        net_pnl_pct = (
            round(net_pnl / cost_basis, 4) if cost_basis > 0 else 0.0
        )
        # Repurpose MultiTrade.signal_* fields to surface tiebreaker
        # values in the table.
        return MultiTrade(
            ticker=ticker,
            signal_volume=float(fit["rvol"]),  # daily RVOL
            signal_buy_volume=float(fit["float_shares"]) if fit["float_shares"] != float("inf") else 0.0,
            signal_sell_volume=0.0,
            signal_buy_sell_ratio=round(fit["rvol"], 2),  # display = RVOL
            commission=round(commission, 2),
            gross_pnl=round(gross_pnl, 2),
            pattern_name=trade.pattern_name,
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_loss=trade.stop_loss,
            shares=trade.shares,
            pnl=round(net_pnl, 2),
            pnl_pct=net_pnl_pct,
            exit_reason=trade.exit_reason,
            entry_ts=trade.entry_ts,
            exit_ts=trade.exit_ts,
        )

    # ---- result ----

    @staticmethod
    def _build_result(
        config: MultiStrategyConfig,
        tickers_scanned: int,
        total_signals: int,
        trades: list[MultiTrade],
        equity_curve: list[EquityPoint],
        final_capital: float,
        max_dd: float,
        failed: list[str],
    ) -> MultiStrategyResult:
        wins = [t for t in trades if t.pnl > 0]
        win_rate = round(len(wins) / len(trades), 4) if trades else 0.0
        total_return = (
            round(
                (final_capital - config.initial_capital)
                / config.initial_capital,
                4,
            )
            if config.initial_capital > 0
            else 0.0
        )
        total_commission = round(sum(t.commission for t in trades), 2)
        return MultiStrategyResult(
            config=config,
            tickers_scanned=tickers_scanned,
            total_signals=total_signals,
            trades_taken=len(trades),
            win_rate=win_rate,
            total_return_pct=total_return,
            initial_capital=config.initial_capital,
            final_capital=round(final_capital, 2),
            max_drawdown_pct=round(max_dd, 4),
            total_commission=total_commission,
            trades=trades,
            equity_curve=equity_curve,
            failed_tickers=failed,
        )
