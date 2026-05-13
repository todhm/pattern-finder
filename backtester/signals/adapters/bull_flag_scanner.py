"""Buy-signal scanner for Bull Flag (Ross Cameron) intraday setups.

Companion to :class:`UniverseBuySignalScanner` (wedgepop) and
:class:`WickPlayBuySignalScanner` (wick_play) — but operates on **1m
intraday** bars instead of daily. Each emitted :class:`BuySignal`
captures the exact entry / stop / target levels :class:`BullFlagStrategy`
would act on at the moment the breakout candle printed, plus enough
live context (current HoD, latest close, 9 EMA) so the user can decide
whether the setup is still actionable when they look at it.

Pipeline mirrors :class:`MultiBullFlagStrategy._scan_ticker`:
    1. Fetch daily (120-day padding for RVOL / split-blackout / trend).
    2. Fetch fundamentals (float + split history).
    3. Cheap 4-criteria gate via the detector's ``_qualifying_sessions``.
       ~95% of tickers fail here → short-circuit before paying for 1m/5m.
    4. Fetch 1m intraday + 5m intraday + (optional) raw 1m for PM high.
    5. Run :class:`BullFlagDetector.detect` and keep signals where
       ``signal.session_date >= today − lookback_days``.
    6. For each signal compute current target / latest_close / current
       9 EMA / current HoD so the watchlist row stays honest as the
       session progresses.
    7. Sort by ``(session_date desc, RVOL desc, float asc, entry_ts asc)``
       — same tiebreaker the :class:`MultiBullFlagStrategy` walker
       uses to pick a single ticker per session.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timedelta

import pandas as pd

from data.domain.market_calendar import NY, MarketCalendar
from data.domain.ports import (
    FundamentalsPort,
    MarketDataPort,
    UniverseProviderPort,
)
from pattern.adapters.bull_flag import BullFlagDetector, BullFlagSignal
from signals.domain.models import BuySignal
from signals.domain.ports import SignalScannerPort
from strategy.adapters.bull_flag_strategy import BullFlagStrategy


class BullFlagBuySignalScanner(SignalScannerPort):
    """Discover active Bull Flag buy signals across a NY equity universe.

    Detector + strategy params come in through factories so the
    Streamlit page can wire user inputs without re-instantiating the
    scanner. The factories receive per-ticker fundamentals (float,
    splits, PM high) so each ticker gets a fully-configured detector.
    """

    _warmup_days: int = 5

    def __init__(
        self,
        market_data: MarketDataPort,
        market_data_5m: MarketDataPort,
        daily_market_data: MarketDataPort,
        fundamentals: FundamentalsPort,
        universe_provider: UniverseProviderPort,
        detector_factory,
        strategy_factory,
        market: MarketCalendar = NY,
        max_workers: int = 8,
        require_float_filter: bool = True,
        raw_market_data: MarketDataPort | None = None,
        min_bars: int = 60,
    ) -> None:
        self._market_data = market_data
        self._market_data_5m = market_data_5m
        self._daily_market_data = daily_market_data
        self._fundamentals = fundamentals
        self._universe_provider = universe_provider
        self._detector_factory = detector_factory
        self._strategy_factory = strategy_factory
        self._market = market
        self._max_workers = max_workers
        self._require_float_filter = require_float_filter
        self._raw_market_data = raw_market_data
        self._min_bars = min_bars
        # Populated by ``scan()`` for the UI's "why zero signals?" panel.
        self.last_scan_stats: dict = {}

    # ---- public API --------------------------------------------------

    def screen_premarket(
        self,
        universe: str,
        target_date: date | None = None,
        max_tickers: int | None = None,
    ) -> list[dict]:
        """Daily-only 4-criteria gate for ``target_date`` (today by default).

        Skips intraday + PM + 5m fetches — only daily OHLCV +
        fundamentals are pulled per ticker, so even a 2,200-NASDAQ
        scan completes in seconds (and ~free after the first call
        once the daily cache is warm). Returns a sorted list of
        ``{ticker, gap_pct, rvol, open_price, float_shares}`` so the
        UI can build "today's watchlist" before market opens.

        Tie-breaker matches :class:`MultiBullFlagStrategy` 's walker:
        ``RVOL desc → float asc``.
        """
        target = target_date or date.today()
        tickers = self._universe_provider.get_tickers(universe)
        if max_tickers is not None:
            tickers = tickers[:max_tickers]
        out: list[dict] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {
                ex.submit(self._screen_one, t, target): t for t in tickers
            }
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception:
                    continue
                if row is not None:
                    out.append(row)
        out.sort(
            key=lambda r: (
                -r["rvol"],
                r.get("float_shares") or float("inf"),
            )
        )
        return out

    def _screen_one(self, ticker: str, target: date) -> dict | None:
        """Daily-only screen for a single ticker. None when not qualifying."""
        daily_start = target - timedelta(days=120)
        try:
            df_daily = self._daily_market_data.fetch_ohlcv(
                ticker, daily_start, target
            )
        except Exception:
            return None
        if df_daily is None or df_daily.empty:
            return None
        fundamentals = self._fundamentals.fetch(ticker)
        float_shares = fundamentals.float_shares
        if float_shares is None and self._require_float_filter:
            return None
        probe = self._detector_factory(
            float_shares=float_shares,
            splits=fundamentals.splits,
            pm_high_by_date={},
        )
        qualifying = probe._qualifying_sessions(df_daily)
        meta = qualifying.get(target)
        if meta is None:
            return None
        return {
            "ticker": ticker,
            "gap_pct": float(meta["gap_pct"]),
            "rvol": float(meta["rvol"]),
            "open_price": float(meta["open_price"]),
            "float_shares": (
                float(float_shares) if float_shares is not None else None
            ),
        }

    def monitor_tickers(
        self,
        tickers: list[str],
        target_date: date | None = None,
    ) -> list[BuySignal]:
        """Intraday scan for an *explicit* ticker list (vs ``scan`` 's
        universe iteration). Use after :meth:`screen_premarket` has
        narrowed the universe to a handful of tickers — monitoring
        e.g. 10 tickers is fast enough for a 30-60s refresh cadence.

        Returns same-day signals only (``signal.session_date ==
        target_date``). Sorted newest entry first.
        """
        today = target_date or date.today()
        # cutoff == today means _scan_ticker keeps signals with
        # session_date >= today → only today's signals.
        out: list[BuySignal] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {
                ex.submit(self._scan_ticker, t, today, today): t
                for t in tickers
            }
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception:
                    continue
                if result.get("status") == "ok":
                    out.extend(result["signals"])
        out.sort(
            key=lambda s: s.metadata.get("entry_ts", ""),
            reverse=True,
        )
        return out

    def scan(
        self,
        universe: str,
        lookback_days: int = 3,
        max_tickers: int | None = None,
    ) -> list[BuySignal]:
        today = date.today()
        cutoff = today - timedelta(days=lookback_days)
        tickers = self._universe_provider.get_tickers(universe)
        if max_tickers is not None:
            tickers = tickers[:max_tickers]

        stats = {
            "tickers_requested": len(tickers),
            "tickers_fetch_failed": 0,
            "tickers_daily_filtered": 0,
            "tickers_intraday_failed": 0,
            "tickers_with_data": 0,
            "total_detector_hits_history": 0,
            "in_window_hits": 0,
            "returned": 0,
            "universe": universe,
            "lookback_days": int(lookback_days),
            "cutoff": cutoff.isoformat(),
        }

        all_signals: list[BuySignal] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {
                ex.submit(self._scan_ticker, t, today, cutoff): t
                for t in tickers
            }
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception:
                    stats["tickers_fetch_failed"] += 1
                    continue
                status = result["status"]
                if status == "fetch_failed":
                    stats["tickers_fetch_failed"] += 1
                    continue
                if status == "daily_filtered":
                    stats["tickers_daily_filtered"] += 1
                    continue
                if status == "intraday_failed":
                    stats["tickers_intraday_failed"] += 1
                    continue
                stats["tickers_with_data"] += 1
                stats["total_detector_hits_history"] += result["total_hits"]
                stats["in_window_hits"] += len(result["signals"])
                all_signals.extend(result["signals"])

        # Same tiebreaker as MultiBullFlagStrategy._walk_signals so the
        # row order here matches what the multi-ticker walker would pick.
        all_signals.sort(
            key=lambda s: (
                s.signal_date,
                -float(s.metadata.get("rvol", 0.0)),
                float(s.metadata.get("float_shares") or float("inf")),
                s.metadata.get("entry_ts", ""),
            ),
        )
        # Newest date first.
        all_signals.sort(
            key=lambda s: s.signal_date,
            reverse=True,
        )
        stats["returned"] = len(all_signals)
        self.last_scan_stats = stats
        return all_signals

    def build_signal_at(
        self,
        ticker: str,
        target_date: date,
    ) -> BuySignal:
        """Build a :class:`BuySignal` pinned to ``target_date``.

        Detector-backed when the bull-flag detector fires that day;
        manual-fallback otherwise — entry = session open + a 1R
        guesstimate stop (open × 0.95) so the row is still usable as a
        watchlist marker.
        """
        today = date.today()
        end = max(today, target_date)
        state = self._fetch_ticker_state(ticker, end)
        if state is None:
            raise ValueError(f"No data for {ticker}")
        df_intra = state["df_intraday"]
        df_daily = state["df_daily"]
        df_5m = state["df_5m"]

        detector = self._detector_factory(
            float_shares=state["float_shares"],
            splits=state["splits"],
            pm_high_by_date=state["pm_high_by_date"],
        )
        signals = detector.detect(df_intra, df_daily, df_5m=df_5m)
        match = next(
            (s for s in signals if s.session_date == target_date), None
        )
        if match is not None:
            return self._signal_to_buysignal(
                ticker=ticker,
                signal=match,
                state=state,
                detector=detector,
            )

        # Fallback: manual marker on the requested session.
        session_df = df_intra[df_intra.index.date == target_date]
        if session_df.empty:
            raise ValueError(
                f"{target_date.isoformat()} has no intraday bars for {ticker}."
            )
        open_price = float(session_df["Open"].iloc[0])
        session_low = float(session_df["Low"].min())
        stop = max(session_low, open_price * 0.95)
        risk = open_price - stop
        latest_idx = len(df_intra) - 1
        latest_close = float(df_intra["Close"].iloc[latest_idx])
        latest_ts = df_intra.index[latest_idx]
        return BuySignal(
            ticker=ticker,
            signal_date=target_date,
            pattern_name="manual",
            entry_price=open_price,
            stop_loss=stop,
            interval="1m",
            signal_datetime=session_df.index[0].to_pydatetime(),
            metadata={
                "manually_added": True,
                "manually_added_no_signal": True,
                "trigger": "manual",
                "open_price": open_price,
                "risk_per_share": round(risk, 4) if risk > 0 else None,
                "latest_close": round(latest_close, 4),
                "latest_ts": latest_ts.isoformat(),
                "latest_date": latest_ts.date().isoformat(),
                "current_hod": float(session_df["High"].max()),
                "refreshed_at": datetime.utcnow().isoformat(timespec="seconds"),
            },
        )

    def refresh_targets(self, signal: BuySignal) -> BuySignal:
        """Re-price live targets against the latest intraday bars.

        Entry / stop are locked at scan time (the breakout candle has
        printed) but the current HoD, 9 EMA, and latest close all
        evolve as the session progresses. Recomputing these keeps the
        row honest mid-session.
        """
        today = date.today()
        state = self._fetch_ticker_state(signal.ticker, today)
        if state is None:
            return signal
        df_intra = state["df_intraday"]
        session_df = df_intra[df_intra.index.date == signal.signal_date]
        if session_df.empty:
            return signal
        live = self._live_metrics(
            session_df=session_df,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
        )
        new_meta = {
            **signal.metadata,
            **live,
            "refreshed_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        return replace(signal, metadata=new_meta)

    # ---- per-ticker pipeline ----------------------------------------

    def _scan_ticker(
        self,
        ticker: str,
        today: date,
        cutoff: date,
    ) -> dict:
        try:
            state = self._fetch_ticker_state(ticker, today)
        except Exception:
            return {"status": "fetch_failed", "signals": [], "total_hits": 0}
        if state is None:
            return {"status": "fetch_failed", "signals": [], "total_hits": 0}
        if state.get("status") == "daily_filtered":
            return {"status": "daily_filtered", "signals": [], "total_hits": 0}
        if state.get("status") == "intraday_failed":
            return {"status": "intraday_failed", "signals": [], "total_hits": 0}

        detector = self._detector_factory(
            float_shares=state["float_shares"],
            splits=state["splits"],
            pm_high_by_date=state["pm_high_by_date"],
        )
        signals = detector.detect(
            state["df_intraday"],
            state["df_daily"],
            df_5m=state["df_5m"],
        )

        out: list[BuySignal] = []
        for sig in signals:
            if sig.session_date < cutoff:
                continue
            out.append(
                self._signal_to_buysignal(
                    ticker=ticker,
                    signal=sig,
                    state=state,
                    detector=detector,
                )
            )
        return {
            "status": "ok",
            "signals": out,
            "total_hits": len(signals),
        }

    def _fetch_ticker_state(self, ticker: str, end: date) -> dict | None:
        """Fetch + tz-normalize 1m / 5m / daily / fundamentals / PM high.

        Returns ``None`` on hard data-fetch failure. Returns dict with
        ``status="daily_filtered"`` (no qualifying session) or
        ``status="intraday_failed"`` (qualified but intraday missing)
        so the caller can keep diagnostic counters honest.
        """
        daily_start = end - timedelta(days=120)
        intraday_start = end - timedelta(days=self._warmup_days)
        try:
            df_daily = self._daily_market_data.fetch_ohlcv(
                ticker, daily_start, end
            )
        except Exception:
            return None
        if df_daily is None or df_daily.empty:
            return None
        fundamentals = self._fundamentals.fetch(ticker)
        float_shares = fundamentals.float_shares
        splits = fundamentals.splits
        if float_shares is None and self._require_float_filter:
            return {"status": "daily_filtered"}

        # Cheap daily-gate short-circuit — reuses the detector's
        # ``_qualifying_sessions`` so the same 4-criteria logic decides.
        probe = self._detector_factory(
            float_shares=float_shares, splits=splits, pm_high_by_date={}
        )
        qualifying = probe._qualifying_sessions(df_daily)
        if not qualifying:
            return {"status": "daily_filtered"}

        try:
            df_intra = self._market_data.fetch_ohlcv(
                ticker, intraday_start, end, interval="1m"
            )
            df_5m = self._market_data_5m.fetch_ohlcv(
                ticker, intraday_start, end, interval="5m"
            )
        except Exception:
            return {"status": "intraday_failed"}
        if (
            df_intra is None
            or df_intra.empty
            or len(df_intra) < self._min_bars
        ):
            return {"status": "intraday_failed"}

        df_intra = self._tz_normalize(df_intra)
        if df_5m is not None and not df_5m.empty:
            df_5m = self._tz_normalize(df_5m)
        else:
            df_5m = None

        pm_high_by_date: dict[date, float] = {}
        if self._raw_market_data is not None:
            try:
                raw_df = self._raw_market_data.fetch_ohlcv(
                    ticker, intraday_start, end, interval="1m"
                )
                if raw_df is not None and not raw_df.empty:
                    raw_df = self._tz_normalize(raw_df)
                    pm = raw_df.between_time("04:00", "09:29")
                    if not pm.empty:
                        for d, group in pm.groupby(pm.index.date):
                            pm_high_by_date[d] = float(group["High"].max())
            except Exception:
                pm_high_by_date = {}

        avg_vol = df_daily["Volume"].rolling(50, min_periods=20).mean()
        drvol = df_daily["Volume"] / avg_vol
        drvol_by_date = {
            ts.date(): float(v) for ts, v in drvol.items() if pd.notna(v)
        }

        return {
            "status": "ok",
            "df_intraday": df_intra,
            "df_5m": df_5m,
            "df_daily": df_daily,
            "float_shares": float_shares,
            "splits": splits,
            "pm_high_by_date": pm_high_by_date,
            "drvol_by_date": drvol_by_date,
        }

    def _tz_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tz is None:
            out = df.copy()
            out.index = out.index.tz_localize(self._market.tz)
            return out
        if str(df.index.tz) != self._market.tz:
            out = df.copy()
            out.index = out.index.tz_convert(self._market.tz)
            return out
        return df

    # ---- signal/target conversion ----------------------------------

    def _signal_to_buysignal(
        self,
        ticker: str,
        signal: BullFlagSignal,
        state: dict,
        detector: BullFlagDetector,
    ) -> BuySignal:
        df_intra = state["df_intraday"]
        session_df = df_intra[df_intra.index.date == signal.session_date]
        risk = signal.entry_price - signal.stop_loss
        # Surface fixed-R targets + the strategy-configured target so
        # the UI can show "what BullFlagStrategy._initial_target would
        # have picked" alongside the canonical HoD reference.
        strategy: BullFlagStrategy = self._strategy_factory(detector=detector)
        target_r = strategy.target_at_r_multiple
        if target_r is not None and risk > 0:
            target_strategy = signal.entry_price + target_r * risk
            target_label = f"{target_r:.1f}R"
        else:
            target_strategy = signal.hod_at_entry
            target_label = "HoD"

        add_at_r_trigger = (
            signal.entry_price + strategy.add_at_r * risk
            if strategy.enable_add_to_winner and risk > 0
            else None
        )
        be_stop_after_add = (
            signal.entry_price * (1.0 - strategy.be_stop_buffer_pct)
            if strategy.enable_add_to_winner
            else None
        )

        live = self._live_metrics(
            session_df=session_df,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
        )

        rvol_today = state["drvol_by_date"].get(signal.session_date, signal.rvol)
        pm_high = state["pm_high_by_date"].get(signal.session_date)

        metadata = {
            "trigger": "bull_flag",
            "gap_pct": signal.gap_pct,
            "rvol": float(rvol_today),
            "open_price": signal.open_price,
            "float_shares": signal.float_shares,
            "pole_start_ts": pd.Timestamp(signal.pole_start_ts).isoformat(),
            "pole_start_price": signal.pole_start_price,
            "pole_end_ts": pd.Timestamp(signal.pole_end_ts).isoformat(),
            "pole_end_price": signal.pole_end_price,
            "flag_low": signal.flag_low,
            "flag_low_ts": pd.Timestamp(signal.flag_low_ts).isoformat(),
            "entry_ts": pd.Timestamp(signal.entry_ts).isoformat(),
            "hod_at_entry": signal.hod_at_entry,
            "r_multiple_to_hod": signal.r_multiple_to_hod,
            "risk_per_share": round(risk, 4) if risk > 0 else None,
            "target_strategy": round(target_strategy, 4),
            "target_strategy_label": target_label,
            "target_hod": round(signal.hod_at_entry, 4),
            "premarket_high": pm_high,
            "entry_confirmed": True,
            "refreshed_at": datetime.utcnow().isoformat(timespec="seconds"),
            **({"add_at_r_trigger": round(add_at_r_trigger, 4)} if add_at_r_trigger else {}),
            **({"be_stop_after_add": round(be_stop_after_add, 4)} if be_stop_after_add else {}),
            **({"add_at_r": strategy.add_at_r} if strategy.enable_add_to_winner else {}),
        }
        if risk > 0:
            metadata.update({
                "target_1r": round(signal.entry_price + 1.0 * risk, 4),
                "target_2r": round(signal.entry_price + 2.0 * risk, 4),
                "target_3r": round(signal.entry_price + 3.0 * risk, 4),
            })
        metadata.update(live)

        return BuySignal(
            ticker=ticker,
            signal_date=signal.session_date,
            pattern_name=detector.name,
            entry_price=float(signal.entry_price),
            stop_loss=float(signal.stop_loss),
            interval="1m",
            signal_datetime=pd.Timestamp(signal.entry_ts).to_pydatetime(),
            metadata=metadata,
        )

    @staticmethod
    def _live_metrics(
        session_df: pd.DataFrame,
        entry_price: float,
        stop_loss: float,
    ) -> dict:
        """Snapshot of the session-so-far: current HoD, latest close,
        9 EMA at the latest bar, and exit-state flags so the watchlist
        row can show "stop已tripped" or "target hit" mid-session.
        """
        if session_df.empty:
            return {}
        latest_idx = len(session_df) - 1
        latest_close = float(session_df["Close"].iloc[latest_idx])
        latest_ts = session_df.index[latest_idx]
        current_hod = float(session_df["High"].max())
        session_low_since_entry = float(session_df["Low"].min())
        # 9 EMA at the latest bar (Ross "9 EMA on every TF").
        ema9 = (
            session_df["Close"].ewm(span=9, adjust=False).mean().iloc[-1]
        )
        risk = entry_price - stop_loss
        return {
            "latest_close": round(latest_close, 4),
            "latest_high": float(session_df["High"].iloc[latest_idx]),
            "latest_ts": latest_ts.isoformat(),
            "latest_date": latest_ts.date().isoformat(),
            "current_hod": round(current_hod, 4),
            "current_ema9": round(float(ema9), 4),
            "session_low_since_entry": round(session_low_since_entry, 4),
            "stop_tripped": session_low_since_entry <= stop_loss,
            "hod_reached_since_entry": current_hod
            > float(session_df["High"].iloc[0]),
            "unrealized_r": (
                round((latest_close - entry_price) / risk, 2)
                if risk > 0
                else None
            ),
        }
