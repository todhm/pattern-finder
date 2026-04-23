"""Buy-signal scanner for Wick Play (Oliver Kell's 3-bar setup).

Companion to :class:`UniverseBuySignalScanner` (wedgepop): runs the
Wick Play detector across a ticker universe, surfaces recent buy
opportunities, and annotates each with the exact stop / TP / breakeven
reference levels the paired :class:`WickPlayStrategy` would act on.

Differences from the wedgepop scanner
-------------------------------------

1. **No post-detection filter gates** — Wick Play's detector already
   bakes in psych-score, regime, macro-blackout, prior-trend and
   breakout-strength checks. Anything the detector emits has already
   survived them, so the scanner doesn't re-evaluate.
2. **Simpler target surface** — Wick Play exits are:
       ``wick_low`` (hard stop)
       ``exhaustion_extension_top`` (paired TP detector)
       ``ema_trail`` (continuous trail, not a static level)
       ``breakeven_stop`` (opt-in, arm at +N R → move stop to +M R)
   Only the first two + breakeven levels are pre-computable as prices;
   the trail is bar-by-bar so we surface the current EMA instead.
3. **Buy/sell ratio is informational**, not a hard filter. Unlike the
   wedgepop ``volume_ratio >= 1.0`` gate, Wick Play has no pre-exec
   volume threshold — sorting by buy/sell ratio still matches the
   :class:`MultiWickPlayStrategy` daily-auction tiebreaker.

All three public entry points (``scan`` / ``build_signal_at`` /
``refresh_targets``) mirror the wedgepop scanner's semantics so the UI
layer can reuse the same code paths.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timedelta

import pandas as pd

from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.adapters.wick_play import WickPlayDetector
from pattern.domain.models import PatternSignal
from signals.domain.models import BuySignal
from signals.domain.ports import SignalScannerPort
from strategy.adapters.wickplay_strategy import WickPlayStrategy


class WickPlayBuySignalScanner(SignalScannerPort):
    """Discover active Wick Play buy signals across a ticker universe.

    Pipeline
        1. Per-ticker (parallel): fetch OHLCV → decorate with
           :meth:`WickPlayStrategy._with_indicators` → run
           :meth:`WickPlayDetector.detect`, keep signals inside the
           lookback window.
        2. Compute buy/sell pressure for the daily-auction tiebreaker.
        3. Compute stop/TP/breakeven levels off the signal bar.
        4. Sort by ``(signal_date desc, buy_sell_ratio desc)``.

    Each surviving signal becomes a :class:`BuySignal` carrying the
    detector's metadata + buy/sell split + target levels +
    ``entry_confirmed`` flag (True iff the next bar's open has already
    printed).
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: WickPlayDetector,
        strategy: WickPlayStrategy,
        max_workers: int = 8,
        min_bars: int = 15,
        warmup_days: int = 400,
    ) -> None:
        self._market_data = market_data
        self._universe_provider = universe_provider
        self._detector = detector
        self._strategy = strategy
        self._max_workers = max_workers
        self._min_bars = min_bars
        self._warmup_days = warmup_days
        # Populated by the most recent ``scan()`` call so the UI can
        # distinguish "no tickers fetched" from "fetched but detector
        # didn't fire" from "detector fired outside the lookback
        # window". Untouched by ``build_signal_at`` / ``refresh_targets``.
        self.last_scan_stats: dict = {}

    # ---- public API --------------------------------------------------

    def scan(
        self,
        universe: str,
        lookback_days: int = 5,
        max_tickers: int | None = None,
    ) -> list[BuySignal]:
        today = date.today()
        cutoff = today - timedelta(days=lookback_days)

        tickers = self._universe_provider.get_tickers(universe)
        if max_tickers is not None:
            tickers = tickers[:max_tickers]

        # Aggregate diagnostics so the UI can answer "why zero
        # signals?" concretely — data fetch failures vs. detector
        # silence vs. all-hits-before-cutoff are three different
        # problems with three different fixes.
        stats = {
            "tickers_requested": len(tickers),
            "tickers_with_data": 0,
            "tickers_fetch_failed": 0,
            "tickers_too_few_bars": 0,
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
                if result["status"] == "fetch_failed":
                    stats["tickers_fetch_failed"] += 1
                    continue
                if result["status"] == "too_few_bars":
                    stats["tickers_too_few_bars"] += 1
                    continue
                stats["tickers_with_data"] += 1
                stats["total_detector_hits_history"] += result["total_hits"]
                stats["in_window_hits"] += len(result["signals"])
                all_signals.extend(result["signals"])

        all_signals.sort(
            key=lambda s: (
                s.signal_date,
                s.metadata.get("buy_sell_ratio", 0.0),
            ),
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

        Detector-backed when the wick-play detector fires on
        ``target_date``; manual-fallback otherwise (entry = next bar's
        open or same-bar close if not printed yet, stop = bar low,
        pattern_name = ``"manual"``).
        """
        today = date.today()
        fetch_start = today - timedelta(days=self._warmup_days)
        end = max(today, target_date)
        df = self._market_data.fetch_ohlcv(ticker, fetch_start, end)
        if df is None or df.empty:
            raise ValueError(f"No data for {ticker}")

        df_ind = self._strategy._with_indicators(df)
        idx = self._locate_signal_idx(df_ind, target_date)
        if idx is None:
            raise ValueError(
                f"{target_date.isoformat()} not a trading day for {ticker}."
            )

        signals = self._detector.detect(df_ind)
        target = next((s for s in signals if s.date == target_date), None)

        last_idx = len(df_ind) - 1
        latest_close = float(df_ind["Close"].iloc[last_idx])
        latest_date = df_ind.index[last_idx].date()

        if idx + 1 <= last_idx:
            entry_price = float(df_ind["Open"].iloc[idx + 1])
            entry_date = df_ind.index[idx + 1].date()
            entry_confirmed = True
        else:
            entry_price = float(df_ind["Close"].iloc[idx])
            entry_date = target_date
            entry_confirmed = False

        pressure = self._signal_pressure(df_ind, idx)

        if target is not None:
            pattern_name = target.pattern_name
            stop_loss = float(target.stop_loss)
            signal_close = round(float(target.entry_price), 4)
            extra_meta = dict(target.metadata)
            manual_flags = {"manually_added": True}
        else:
            pattern_name = "manual"
            stop_loss = float(df_ind["Low"].iloc[idx])
            signal_close = round(float(df_ind["Close"].iloc[idx]), 4)
            extra_meta = {"trigger": "manual"}
            manual_flags = {
                "manually_added": True,
                "manually_added_no_signal": True,
            }

        targets = self._compute_targets(
            entry_price=entry_price,
            stop_loss=stop_loss,
            df=df_ind,
            ref_idx=last_idx,
        )

        return BuySignal(
            ticker=ticker,
            signal_date=target_date,
            pattern_name=pattern_name,
            entry_price=entry_price,
            stop_loss=stop_loss,
            metadata={
                **extra_meta,
                **pressure,
                "entry_confirmed": entry_confirmed,
                "entry_date": entry_date.isoformat(),
                "signal_close": signal_close,
                "latest_close": round(latest_close, 4),
                "latest_date": latest_date.isoformat(),
                "refreshed_at": datetime.utcnow().isoformat(timespec="seconds"),
                **manual_flags,
                **targets,
            },
        )

    def refresh_targets(self, signal: BuySignal) -> BuySignal:
        """Re-price stop / TP / breakeven levels against today's bar.

        Entry and stop are historical (locked once taken), but the
        exhaustion threshold (``ema_fast + ext × ATR``) and latest
        close shift with every new bar. Recomputing keeps displayed
        levels aligned with what ``WickPlayStrategy._find_exit`` would
        fire on today.
        """
        today = date.today()
        fetch_start = today - timedelta(days=self._warmup_days)
        df = self._market_data.fetch_ohlcv(signal.ticker, fetch_start, today)
        df_ind = self._strategy._with_indicators(df)
        last_idx = len(df_ind) - 1
        latest_close = float(df_ind["Close"].iloc[last_idx])
        latest_date = df_ind.index[last_idx].date()

        targets = self._compute_targets(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            df=df_ind,
            ref_idx=last_idx,
        )
        new_meta = {
            **signal.metadata,
            **targets,
            "latest_close": round(latest_close, 4),
            "latest_date": latest_date.isoformat(),
            "refreshed_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        return replace(signal, metadata=new_meta)

    # ---- per-ticker worker ------------------------------------------

    def _scan_ticker(
        self,
        ticker: str,
        today: date,
        cutoff: date,
    ) -> dict:
        """Scan one ticker. Returns a dict with ``status`` +
        (on success) the list of in-window :class:`BuySignal`
        objects and the total detector-hit count across the full
        fetched history. The caller aggregates these for diagnostics.
        """
        fetch_start = today - timedelta(days=self._warmup_days)
        try:
            df = self._market_data.fetch_ohlcv(ticker, fetch_start, today)
        except Exception:
            return {"status": "fetch_failed", "signals": [], "total_hits": 0}
        if df is None or df.empty:
            return {"status": "fetch_failed", "signals": [], "total_hits": 0}
        if len(df) < self._min_bars:
            return {"status": "too_few_bars", "signals": [], "total_hits": 0}

        df_ind = self._strategy._with_indicators(df)
        signals = self._detector.detect(df_ind)
        last_idx = len(df_ind) - 1
        latest_close = float(df_ind["Close"].iloc[last_idx])
        latest_date = df_ind.index[last_idx].date()

        out: list[BuySignal] = []
        for signal in signals:
            if signal.date < cutoff:
                continue
            idx = self._locate_signal_idx(df_ind, signal.date)
            if idx is None:
                continue
            pressure = self._signal_pressure(df_ind, idx)

            # Entry = next-bar open when printed (matches the
            # backtest's fill convention); provisional = signal close
            # until that bar arrives.
            if idx + 1 <= last_idx:
                entry_price = float(df_ind["Open"].iloc[idx + 1])
                entry_date = df_ind.index[idx + 1].date()
                entry_confirmed = True
            else:
                entry_price = float(signal.entry_price)
                entry_date = signal.date
                entry_confirmed = False

            stop_loss = float(signal.stop_loss)
            targets = self._compute_targets(
                entry_price=entry_price,
                stop_loss=stop_loss,
                df=df_ind,
                ref_idx=idx,
            )
            out.append(
                BuySignal(
                    ticker=ticker,
                    signal_date=signal.date,
                    pattern_name=signal.pattern_name,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    metadata={
                        **signal.metadata,
                        **pressure,
                        "entry_confirmed": entry_confirmed,
                        "entry_date": entry_date.isoformat(),
                        "signal_close": round(float(signal.entry_price), 4),
                        "latest_close": round(latest_close, 4),
                        "latest_date": latest_date.isoformat(),
                        "refreshed_at": datetime.utcnow().isoformat(
                            timespec="seconds"
                        ),
                        **targets,
                    },
                )
            )
        return {
            "status": "ok",
            "signals": out,
            "total_hits": len(signals),
        }

    # ---- target computation -----------------------------------------

    def _compute_targets(
        self,
        entry_price: float,
        stop_loss: float,
        df: pd.DataFrame,
        ref_idx: int,
    ) -> dict:
        """Surface the exit prices :class:`WickPlayStrategy` actually
        acts on:

            - ``target_exhaustion_primary`` — ``ema_fast + ext × ATR``
              at ``ref_idx`` (paired exit detector's trigger price).
            - ``breakeven_arm_price`` / ``breakeven_exit_price`` — the
              arm + lock levels computed from the strategy's configured
              ``arm_r`` / ``offset_r`` multiples. Only emitted when
              ``enable_breakeven_stop`` is on.
            - ``target_2r`` / ``target_3r`` — reward-framing reference
              lines (no exit mechanic, shown for context).
            - ``ema_trail_current`` — today's fast-EMA level. Purely
              informational: the trail exit fires intra-bar so it
              can't be pre-computed, but surfacing the current EMA
              lets the user see "how far below current price is the
              trail right now".
        """
        strat = self._strategy
        risk = entry_price - stop_loss
        out: dict = {
            "risk_per_share": round(risk, 4) if risk > 0 else None,
        }
        if risk > 0:
            out["target_2r"] = round(entry_price + 2.0 * risk, 4)
            out["target_3r"] = round(entry_price + 3.0 * risk, 4)

        atr_at_ref = (
            float(df["atr"].iloc[ref_idx])
            if not pd.isna(df["atr"].iloc[ref_idx])
            else 0.0
        )
        ema_at_ref = (
            float(df["ema_trail"].iloc[ref_idx])
            if not pd.isna(df["ema_trail"].iloc[ref_idx])
            else 0.0
        )
        out["ema_trail_current"] = round(ema_at_ref, 4) if ema_at_ref > 0 else None

        # --- Exhaustion Extension Top primary price ---------------
        exit_det = getattr(strat, "exit_detector", None)
        if exit_det is not None and atr_at_ref > 0 and ema_at_ref > 0:
            ext_mult = float(exit_det.extension_atr_mult)
            primary = ema_at_ref + ext_mult * atr_at_ref
            out["target_exhaustion_primary"] = round(primary, 4)
            if risk > 0:
                out["r_to_exhaustion_primary"] = round(
                    (primary - entry_price) / risk, 2
                )

        # --- Breakeven arm / exit prices --------------------------
        # Surfaced only when the strategy is configured to act on
        # them — keeps the UI honest about what will actually fire.
        if getattr(strat, "enable_breakeven_stop", False) and risk > 0:
            arm_r = float(strat.breakeven_arm_r_multiple)
            offset_r = float(strat.breakeven_exit_offset_r)
            out["breakeven_arm_price"] = round(entry_price + arm_r * risk, 4)
            out["breakeven_exit_price"] = round(entry_price + offset_r * risk, 4)
            out["breakeven_arm_r"] = arm_r
            out["breakeven_offset_r"] = offset_r
        return out

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _signal_pressure(df: pd.DataFrame, idx: int) -> dict:
        """Close-location × volume split. Same formula as
        :meth:`MultiWickPlayStrategy._signal_pressure` so the per-date
        ordering here matches the walker's tiebreaker.
        """
        high = float(df["High"].iloc[idx])
        low = float(df["Low"].iloc[idx])
        close = float(df["Close"].iloc[idx])
        volume = float(df["Volume"].iloc[idx])
        bar_range = high - low
        if bar_range <= 0 or volume <= 0:
            half = volume / 2.0
            return {
                "volume": volume,
                "buy_volume": round(half, 2),
                "sell_volume": round(half, 2),
                "buy_sell_ratio": 1.0,
            }
        buy_vol = (close - low) / bar_range * volume
        sell_vol = (high - close) / bar_range * volume
        return {
            "volume": volume,
            "buy_volume": round(buy_vol, 2),
            "sell_volume": round(sell_vol, 2),
            "buy_sell_ratio": round(buy_vol / (sell_vol + 1.0), 4),
        }

    @staticmethod
    def _locate_signal_idx(df: pd.DataFrame, signal_date: date) -> int | None:
        ts = pd.Timestamp(signal_date)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        if ts not in df.index:
            return None
        return int(df.index.get_loc(ts))
