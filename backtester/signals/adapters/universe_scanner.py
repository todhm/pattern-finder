from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timedelta

import pandas as pd

from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from pattern.helpers.pivots import (
    fit_lower_trendline,
    last_swing_high,
    recent_swing_high,
)
from signals.domain.models import BuySignal
from signals.domain.ports import SignalScannerPort
from strategy.adapters.wedgepop_strategy import WedgepopStrategy


class UniverseBuySignalScanner(SignalScannerPort):
    """Discover active buy signals across a ticker universe.

    Returns every signal inside the lookback window that passes the
    filter gates, ordered **by date (newest first) then by
    ``buy/sell`` ratio (highest first)** — the exact precedence
    ``MultiWedgepopStrategy`` uses when choosing which ticker to buy
    on a given day. So the item at the top of each date's block is
    the one the backtest walker would have taken first; if that
    trade is already closed, the next item is what it would have
    bought next, and so on. A ``volume_ratio >= 1.0`` hard filter
    is applied upstream to match
    ``MultiWedgepopStrategy._collect_signals``.

    Pipeline:

    1. Per-ticker, in parallel: fetch OHLCV, decorate with
       ``WedgepopStrategy._with_indicators``, run
       ``detector.detect(df_ind)``, keep signals whose date falls
       inside the ``lookback_days`` window.
    2. Apply hard filter ``metadata.volume_ratio >= 1.0``.
    3. Compute ``buy_sell_ratio`` for each remaining signal using the
       same close-location × volume formula as the backtest walker.
    4. Apply signal-bar-close filters (market regime, slope range,
       euphoria cap, close strength, swing breakout). Filters
       requiring the *next* bar's open (gap-down, EMA-extension,
       gap-up, swing resistance) are deferred to actual execution
       time — they live in ``WedgepopStrategy._execute_trade`` on
       the backtest path.
    5. Sort by ``(signal_date desc, buy_sell_ratio desc)``.

    Each surviving signal becomes a :class:`BuySignal` with metadata
    carrying the detector rationale + buy/sell volume split + a
    ``filter_gates`` sub-dict showing which evaluated gates passed.
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: PatternDetector,
        strategy: WedgepopStrategy,
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

    # ---- public API ----

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

        # Phase 1 — collect candidates per ticker in parallel. Each
        # carries the signal, its indicator-decorated frame, and the
        # precomputed pressure stats so phase 2 doesn't re-fetch rows.
        all_candidates: list[dict] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {
                ex.submit(self._scan_ticker, t, today, cutoff): t
                for t in tickers
            }
            for fut in as_completed(futures):
                try:
                    all_candidates.extend(fut.result())
                except Exception:
                    continue

        # Phase 2 — apply signal-bar gates to each candidate (no
        # per-date dedup: a given day can surface multiple tickers).
        kept: list[BuySignal] = []
        for cand in all_candidates:
            gates, passes = self._evaluate_signal_gates(
                cand["df"], cand["signal"], cand["idx"]
            )
            if not passes:
                continue
            signal = cand["signal"]
            entry_price = cand["entry_price"]
            stop_loss = float(signal.stop_loss)
            targets = self._compute_targets(
                entry_price=entry_price,
                stop_loss=stop_loss,
                df=cand["df"],
                signal_idx=cand["idx"],
            )
            kept.append(
                BuySignal(
                    ticker=cand["ticker"],
                    signal_date=cand["signal_date"],
                    pattern_name=signal.pattern_name,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    metadata={
                        **signal.metadata,
                        "volume": cand["volume"],
                        "buy_volume": round(cand["buy_volume"], 2),
                        "sell_volume": round(cand["sell_volume"], 2),
                        "buy_sell_ratio": round(cand["buy_sell_ratio"], 4),
                        "filter_gates": gates,
                        "entry_confirmed": cand["entry_confirmed"],
                        "entry_date": cand["entry_date"].isoformat(),
                        "signal_close": round(float(signal.entry_price), 4),
                        "latest_close": round(cand["latest_close"], 4),
                        "latest_date": cand["latest_date"].isoformat(),
                        **targets,
                    },
                )
            )

        # Sort: newest date first, then within-date by buy/sell ratio
        # descending — the very precedence MultiWedgepopStrategy uses
        # when picking which ticker to buy on a given day.
        kept.sort(
            key=lambda s: (
                s.signal_date,
                s.metadata.get("buy_sell_ratio", 0.0),
            ),
            reverse=True,
        )
        return kept

    # ---- manual build (ticker + date -> BuySignal) ----

    def build_signal_at(
        self,
        ticker: str,
        target_date: date,
    ) -> BuySignal:
        """Build a :class:`BuySignal` for ``ticker`` pinned to
        ``target_date``. Used by the manual "add to watchlist"
        flow: the user supplies a ticker + date.

        Two execution paths:

        * **Detector-backed**: if the wedge-pop detector fires on
          ``target_date`` under the current settings, use its
          ``entry_price`` / ``stop_loss`` / metadata. Identical to a
          scan-surfaced signal.
        * **Manual fallback**: detector silent on ``target_date``.
          Still register the ticker — ``entry_price`` = next bar's
          open if printed (else target bar's close), ``stop_loss`` =
          target bar's low, ``pattern_name`` = ``"manual"``. Metadata
          flags ``manually_added_no_signal=True`` so downstream
          tooling can tell the two apart, and the stop/TP reference
          levels are still computed off the latest bar so HL Trendline /
          Resistance / Exhaustion displays work.

        Raises only when the ticker has no data at all or the date
        sits outside the fetched range — legitimate "can't
        register" cases rather than filter misses.
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

        # Entry: next bar's open if printed, else same-bar close.
        if idx + 1 <= last_idx:
            entry_price = float(df_ind["Open"].iloc[idx + 1])
            entry_date = df_ind.index[idx + 1].date()
            entry_confirmed = True
        else:
            entry_price = float(df_ind["Close"].iloc[idx])
            entry_date = target_date
            entry_confirmed = False

        pressure = self._signal_pressure(df_ind, idx) or {
            "volume": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "buy_sell_ratio": 1.0,
        }

        if target is not None:
            # Detector-backed path — match scanner semantics exactly.
            pattern_name = target.pattern_name
            stop_loss = float(target.stop_loss)
            signal_close = round(float(target.entry_price), 4)
            extra_meta = dict(target.metadata)
            gates, _ = self._evaluate_signal_gates(df_ind, target, idx)
            manual_flags = {"manually_added": True}
        else:
            # Manual fallback — no detector signal at target_date.
            pattern_name = "manual"
            stop_loss = float(df_ind["Low"].iloc[idx])
            signal_close = round(float(df_ind["Close"].iloc[idx]), 4)
            extra_meta = {
                "trigger": "manual",
                "consolidation_low": stop_loss,
            }
            gates = {}
            manual_flags = {
                "manually_added": True,
                "manually_added_no_signal": True,
            }

        targets = self._compute_targets(
            entry_price=entry_price,
            stop_loss=stop_loss,
            df=df_ind,
            signal_idx=last_idx,  # refresh-style: latest bar view
        )

        return BuySignal(
            ticker=ticker,
            signal_date=target_date,
            pattern_name=pattern_name,
            entry_price=entry_price,
            stop_loss=stop_loss,
            metadata={
                **extra_meta,
                "volume": pressure["volume"],
                "buy_volume": round(pressure["buy_volume"], 2),
                "sell_volume": round(pressure["sell_volume"], 2),
                "buy_sell_ratio": round(pressure["buy_sell_ratio"], 4),
                "filter_gates": gates,
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

    # ---- refresh for held/watchlist signals ----

    def refresh_targets(self, signal: BuySignal) -> BuySignal:
        """Return a copy of ``signal`` with stop/TP metadata recomputed
        against the latest OHLCV for ``signal.ticker``.

        Used for positions already in the watchlist: ``entry_price``
        and ``stop_loss`` are historical (locked once the trade was
        taken), but the structural stops (HL trendline, swing
        resistance supports/hurdles) and exhaustion threshold drift
        over time — a trendline's slope moves the level each bar,
        new swing highs can appear, the fast EMA and ATR both
        update. Calling this every time the watchlist is viewed
        keeps the displayed lines aligned with the actual exit
        rules as they'd fire on today's bar.

        Fetches fresh data (cache bypassed for end >= today via
        ``CachedMarketDataAdapter``), re-prepares indicators through
        ``WedgepopStrategy._with_indicators``, then evaluates
        ``_compute_targets`` at the *latest* bar index so every
        level reflects current structure — not the stale view from
        when the signal was saved.
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
            signal_idx=last_idx,
        )
        new_meta = {
            **signal.metadata,
            **targets,
            "latest_close": round(latest_close, 4),
            "latest_date": latest_date.isoformat(),
            "refreshed_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        return replace(signal, metadata=new_meta)

    # ---- per-ticker worker ----

    def _scan_ticker(
        self,
        ticker: str,
        today: date,
        cutoff: date,
    ) -> list[dict]:
        fetch_start = today - timedelta(days=self._warmup_days)
        try:
            df = self._market_data.fetch_ohlcv(ticker, fetch_start, today)
        except Exception:
            return []
        if df is None or df.empty or len(df) < self._min_bars:
            return []

        df_ind = self._strategy._with_indicators(df)
        signals = self._detector.detect(df_ind)
        last_idx = len(df_ind) - 1
        latest_close = float(df_ind["Close"].iloc[last_idx])
        latest_date = df_ind.index[last_idx].date()

        candidates: list[dict] = []
        for signal in signals:
            if signal.date < cutoff:
                continue
            # Hard volume-ratio filter — same as
            # MultiWedgepopStrategy._collect_signals.
            if signal.metadata.get("volume_ratio", 1.0) < 1.0:
                continue
            idx = self._locate_signal_idx(df_ind, signal.date)
            if idx is None:
                continue
            pressure = self._signal_pressure(df_ind, idx)
            if pressure is None:
                continue

            # If the entry bar (signal_idx + 1) has already printed,
            # use its actual open as the entry price — the backtest
            # fills at next-bar open, so the scanner's quoted entry
            # should match once that bar is available. Otherwise
            # fall back to the signal-bar close as a provisional
            # reference (entry_confirmed=False).
            if idx + 1 <= last_idx:
                entry_price = float(df_ind["Open"].iloc[idx + 1])
                entry_date = df_ind.index[idx + 1].date()
                entry_confirmed = True
            else:
                entry_price = float(signal.entry_price)
                entry_date = signal.date
                entry_confirmed = False

            candidates.append(
                {
                    "ticker": ticker,
                    "signal_date": signal.date,
                    "signal": signal,
                    "df": df_ind,
                    "idx": idx,
                    "entry_price": entry_price,
                    "entry_date": entry_date,
                    "entry_confirmed": entry_confirmed,
                    "latest_close": latest_close,
                    "latest_date": latest_date,
                    **pressure,
                }
            )
        return candidates

    def _compute_targets(
        self,
        entry_price: float,
        stop_loss: float,
        df: pd.DataFrame,
        signal_idx: int,
    ) -> dict:
        """Derive stop-loss & take-profit reference levels that match
        the *actual* exit rules inside ``WedgepopStrategy._find_exit``.

        The strategy has no single "TP price" — multiple structural
        stops can fire on the downside and the nearest overhead swing
        high functions as a natural first-target. This helper
        surfaces them all so the UI can display exactly what the
        backtest would act on:

        - **Consolidation-low** stop (``stop_consolidation_low``):
          the sizing / 1R basis. Backtest doesn't enforce it as a
          hard stop but it is the "intended" risk level.
        - **Higher-low trendline** (``stop_trendline_at_entry``,
          ``stop_trendline_slope``) exit: line fit through recent
          confirmable swing lows, evaluated at the entry bar. The
          exit fires when a future bar's low pierces ``slope × i +
          intercept``. Slope is reported per-bar (per trading day).
        - **Swing-resistance break-exit supports**
          (``stop_resistance_supports``): all swing highs inside the
          pivot lookback that sit **below entry**. Strategy marks
          these confirmed-from-start; once a future bar's low
          pierces ``level − pierce_buffer × ATR`` the exit fires
          at the level. These act as staged supports under price.
        - **Swing-resistance overhead hurdles**
          (``target_resistance_hurdles``): swing highs **above
          entry** in the same window. The strategy's
          ``resistance_break`` exit flips these to active once a
          bar's high touches ``level + confirm_buffer × ATR`` — if
          price later fails back through, exit fires at the level.
          So each level is simultaneously a "target to clear" and
          a "retest stop if failed".
        - **R-multiples** (``target_2r``, ``target_3r``):
          ``entry + N × (entry − stop)``. Purely reward-framing,
          unrelated to exit mechanics — useful to gauge whether the
          nearest hurdle gives ≥ 2R of headroom.
        """
        strat = self._strategy
        lookback = strat.swing_pivot_lookback
        right = strat.swing_pivot_right

        risk = entry_price - stop_loss
        out: dict = {
            "stop_consolidation_low": round(stop_loss, 4),
            "risk_per_share": round(risk, 4) if risk > 0 else None,
        }
        if risk > 0:
            out["target_2r"] = round(entry_price + 2.0 * risk, 4)
            out["target_3r"] = round(entry_price + 3.0 * risk, 4)

        atr_at_signal = float(df["atr"].iloc[signal_idx]) if not pd.isna(
            df["atr"].iloc[signal_idx]
        ) else 0.0

        # --- HL Trendline (trendline_break exit) ---------------
        # Fit through recent confirmable swing lows up to the entry
        # bar. ``y = slope × bar_idx + intercept``; at a future bar
        # i the exit fires when ``low ≤ y(i)``. Gate on the same
        # flag the strategy uses so disabling the exit hides the
        # level.
        if strat.enable_trendline_exit and "swing_low" in df.columns:
            tl = fit_lower_trendline(
                df["swing_low"],
                upto_idx=signal_idx,
                lookback=lookback,
                right=right,
                max_points=strat.trendline_max_pivots,
                min_points=strat.trendline_min_pivots,
            )
            if tl is not None:
                slope, intercept, _ = tl
                if slope > 0:
                    out["stop_trendline_at_entry"] = round(
                        slope * signal_idx + intercept, 4
                    )
                    # slope is per-bar; expose so UI can project a
                    # few bars out.
                    out["stop_trendline_slope"] = round(slope, 4)

        # --- Swing-resistance break-exit levels ----------------
        # Mirrors ``WedgepopStrategy._execute_trade`` collection:
        # swing highs in the lookback window, partitioned by
        # position relative to entry. Gated on the exit flag so
        # turning off ``resistance_break`` hides the supports /
        # hurdles / next-resistance targets entirely.
        if strat.enable_resistance_break_exit and "swing_high" in df.columns:
            cutoff = signal_idx - right
            if cutoff >= 0:
                start = max(0, cutoff - lookback + 1)
                levels = [
                    float(v)
                    for v in df["swing_high"].iloc[start : cutoff + 1].dropna().tolist()
                ]
                pierce_buf = (
                    strat.resistance_break_pierce_buffer_atr * atr_at_signal
                    if atr_at_signal > 0
                    else 0.0
                )
                confirm_buf = (
                    strat.resistance_break_confirm_buffer_atr * atr_at_signal
                    if atr_at_signal > 0
                    else 0.0
                )
                supports = sorted(
                    [lv for lv in levels if lv <= entry_price], reverse=True
                )
                hurdles = sorted([lv for lv in levels if lv > entry_price])
                if supports:
                    out["stop_resistance_supports"] = [
                        {
                            "level": round(lv, 4),
                            "pierce_trigger": round(lv - pierce_buf, 4),
                        }
                        for lv in supports
                    ]
                if hurdles:
                    out["target_resistance_hurdles"] = [
                        {
                            "level": round(lv, 4),
                            "confirm_trigger": round(lv + confirm_buf, 4),
                            "r_multiple": round((lv - entry_price) / risk, 2)
                            if risk > 0
                            else None,
                        }
                        for lv in hurdles
                    ]

        # --- Exhaustion Extension Top price level -------------
        # Uses the exit detector's ``extension_atr_mult`` threshold:
        # bar's HIGH must exceed ``ema_fast + extension_atr_mult ×
        # ATR`` for the primary path to fire (additional confirms —
        # slope, close-location, sell-dominance — are dynamic and
        # can't be precomputed). The rejection-override path
        # relaxes the extension to ``× rejection_leniency`` so we
        # show that as a secondary, earlier trigger.
        exit_det = getattr(strat, "_exit_detector", None)
        if exit_det is not None and atr_at_signal > 0:
            # ``WedgepopStrategy._with_indicators`` stores the fast
            # EMA under ``ema_trail`` (same period as the exit
            # detector's ``ema_fast``), so read from there.
            ema_fast_sig = float(df["ema_trail"].iloc[signal_idx])
            ext_mult = float(exit_det.extension_atr_mult)
            primary = ema_fast_sig + ext_mult * atr_at_signal
            out["target_exhaustion_primary"] = round(primary, 4)
            if risk > 0:
                out["r_to_exhaustion_primary"] = round(
                    (primary - entry_price) / risk, 2
                )
            if getattr(exit_det, "enable_rejection_override", False):
                lenient = float(
                    getattr(exit_det, "rejection_leniency", 0.9)
                )
                reject = ema_fast_sig + ext_mult * lenient * atr_at_signal
                out["target_exhaustion_rejection"] = round(reject, 4)
                if risk > 0:
                    out["r_to_exhaustion_rejection"] = round(
                        (reject - entry_price) / risk, 2
                    )

        # --- Nearest overhead swing high (legacy flat field;
        # also a ``resistance_break`` upside exit candidate, so
        # gated on the same flag).
        if strat.enable_resistance_break_exit and "swing_high" in df.columns:
            res = recent_swing_high(
                df["swing_high"],
                upto_idx=signal_idx,
                lookback=lookback,
                right=right,
            )
            if res is not None:
                _, pivot_price = res
                if pivot_price > entry_price:
                    out["target_next_resistance"] = round(pivot_price, 4)
                    if risk > 0:
                        out["r_to_next_resistance"] = round(
                            (pivot_price - entry_price) / risk, 2
                        )
        return out

    @staticmethod
    def _signal_pressure(df: pd.DataFrame, idx: int) -> dict | None:
        """Same close-location × volume split used by
        ``MultiWedgepopStrategy._signal_pressure`` so the per-date
        winner here matches the one the backtest walker would pick.
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
                "buy_volume": half,
                "sell_volume": half,
                "buy_sell_ratio": 1.0,
            }
        buy_vol = (close - low) / bar_range * volume
        sell_vol = (high - close) / bar_range * volume
        return {
            "volume": volume,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "buy_sell_ratio": buy_vol / (sell_vol + 1.0),
        }

    # ---- filter evaluation ----

    def _evaluate_signal_gates(
        self,
        df: pd.DataFrame,
        signal: PatternSignal,
        idx: int,
    ) -> tuple[dict[str, bool], bool]:
        """Run only the signal-bar-close filters. Returns a
        ``{gate_name: passed}`` map plus an overall boolean.
        Filters needing the next bar's open are intentionally skipped
        here — they will apply at execution time, not at scan time.
        """
        strat = self._strategy
        gates: dict[str, bool] = {}

        # Market regime — SPY > 200 SMA at signal_date.
        if strat.enable_market_regime_filter:
            gates["market_regime"] = bool(
                strat._market_regime_lookup.get(signal.date, False)
            )

        # EMA slow slope range.
        slope = signal.metadata.get("ema_slow_slope")
        if slope is not None and (
            strat.min_ema_slow_slope is not None
            or strat.max_ema_slow_slope is not None
        ):
            ok = True
            if strat.min_ema_slow_slope is not None:
                ok = ok and slope >= strat.min_ema_slow_slope
            if strat.max_ema_slow_slope is not None:
                ok = ok and slope <= strat.max_ema_slow_slope
            gates["slope_range"] = ok

        # Euphoria cap.
        if strat.max_signal_bar_gain_atr is not None:
            sig_open = float(df["Open"].iloc[idx])
            sig_close = float(df["Close"].iloc[idx])
            atr = float(df["atr"].iloc[idx])
            gain_atr = (sig_close - sig_open) / atr if atr > 0 else 0.0
            gates["euphoria_cap"] = gain_atr <= strat.max_signal_bar_gain_atr

        # Signal close-strength.
        if strat.enable_signal_close_strength_filter:
            sig_high = float(df["High"].iloc[idx])
            sig_low = float(df["Low"].iloc[idx])
            sig_close = float(df["Close"].iloc[idx])
            bar_range = sig_high - sig_low
            if bar_range > 0:
                loc = (sig_close - sig_low) / bar_range
                gates["close_strength"] = (
                    loc >= strat.min_signal_close_location
                )
            else:
                gates["close_strength"] = True

        # Swing breakout.
        if strat.enable_swing_breakout_filter and "swing_high" in df.columns:
            res = last_swing_high(
                df["swing_high"],
                upto_idx=idx,
                lookback=strat.swing_pivot_lookback,
                right=strat.swing_pivot_right,
            )
            if res is not None:
                _, pivot_price = res
                atr = float(df["atr"].iloc[idx])
                buffer = strat.swing_breakout_buffer_atr * atr if atr > 0 else 0.0
                sig_high = float(df["High"].iloc[idx])
                gates["swing_breakout"] = sig_high > pivot_price + buffer
            else:
                gates["swing_breakout"] = True

        all_pass = all(gates.values())
        return gates, all_pass

    @staticmethod
    def _locate_signal_idx(df: pd.DataFrame, signal_date: date) -> int | None:
        ts = pd.Timestamp(signal_date)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        if ts not in df.index:
            return None
        return int(df.index.get_loc(ts))
