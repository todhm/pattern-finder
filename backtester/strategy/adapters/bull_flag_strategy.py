"""Bull Flag long-only intraday strategy runner.

Pairs with :class:`BullFlagDetector`. 단일 포지션 1m/5m bar-by-bar
시뮬레이션. Ross Cameron의 영상 핵심 룰을 충실히 구현:

    - **Entry**: breakout 캔들 close.
    - **Stop**: ``flag_low - tick``.
    - **1차 익절(target)**: HoD 재돌파 — 영상의 "first target = retest
      of high of day". HoD에 처음 닿는 봉의 high에서 체결.
    - **R/R 게이트**: ``hod_at_entry - entry < target_min_r_multiple ×
      risk`` 이면 trade 자체를 거른다(영상의 "if I don't think I
      can get 326 I won't take the trade").
    - **Add-to-winner (옵션)**: 진입 후 +``add_at_r``R 도달 시 동일
      수량을 추가 매수 + 손절을 entry로 끌어올림 (break-even).
    - **End-of-session 청산**: 세션 마지막 봉 종가에 강제 청산.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pattern.adapters.bull_flag import BullFlagDetector, BullFlagSignal
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    TossFeeSchedule,
    Trade,
)


class BullFlagStrategy:
    """Single-ticker long-only Bull Flag backtester."""

    def __init__(
        self,
        detector: BullFlagDetector,
        max_position_pct_of_equity: float = 0.30,
        target_min_r_multiple: float = 2.0,
        target_at_r_multiple: float | None = None,
        enable_add_to_winner: bool = True,
        add_at_r: float = 1.5,
        add_confirm_on_close: bool = True,
        max_session_losses: int = 1,
        be_stop_buffer_pct: float = 0.003,
        fee_schedule: TossFeeSchedule | None = None,
    ) -> None:
        """
        Parameters
        ----------
        target_min_r_multiple:
            진입 시점 (HoD - entry) / risk 가 이 값 미만이면
            trade 자체를 거른다. 영상의 2:1 게이트.
        target_at_r_multiple:
            None(기본)이면 익절가 = HoD. 값이 주어지면
            ``entry + R × risk`` 고정 R-multiple로 익절.
        enable_add_to_winner:
            영상의 "double the position at +20¢" 메커니즘. 진입 후
            ``add_at_r``R 도달 시 같은 수량 추가 + 손절을 entry로
            올림(BE).
        add_at_r:
            추가 매수를 발동시킬 R 배수. 기본 1.5 (= +1.5R) — SPRC
            테스트에서 +1R(default 이전)은 진입 직후 wick 한 번에
            트리거되어 BE stop 즉시 발화 → 손실 케이스. 1.5R로
            올리면 더 확실히 추세 살아있을 때만 add.
        add_confirm_on_close:
            True(기본)이면 ``add_at_r`` 도달 여부를 **봉 close 가격**
            으로 판정. False면 high(=wick) 기준. wick 한 번에
            트리거되는 fake-add 차단.
        max_session_losses:
            한 세션 안에서 손실(stop_loss / breakeven_stop) 누적이
            이 값에 도달하면 그 세션의 추가 진입을 차단. 기본 1 —
            영상의 "loss 발생 시 사이즈 다운, 30분 미체결 시 quit"
            룰의 백테스트 근사. UGRO 케이스에서 같은 세션 3회 진입
            → 3회 LOSS 같은 패턴을 차단.
        be_stop_buffer_pct:
            BE stop이 평단가와 정확히 같아져 즉시 발화하는 것을
            막기 위한 버퍼 (% of avg entry price). 기본 0.1% =
            \$10 종목에서 1¢. ``avg_entry × (1 - buffer_pct)`` 가
            실제 BE stop 가격이 됨.
        """
        self.detector = detector
        self.max_position_pct_of_equity = max_position_pct_of_equity
        self.target_min_r_multiple = target_min_r_multiple
        self.target_at_r_multiple = target_at_r_multiple
        self.enable_add_to_winner = enable_add_to_winner
        self.add_at_r = add_at_r
        self.add_confirm_on_close = add_confirm_on_close
        self.max_session_losses = max_session_losses
        self.be_stop_buffer_pct = be_stop_buffer_pct
        # Toss 증권 수수료 (None이면 수수료 0). entry/add/exit 마다 적용.
        self.fee_schedule = fee_schedule

    def run(
        self,
        df_intraday: pd.DataFrame,
        df_daily: pd.DataFrame,
        config: StrategyConfig,
        df_5m: pd.DataFrame | None = None,
    ) -> StrategyResult:
        """
        df_5m: optional. detector의 ``enable_mtf_check=True``와 함께
        쓰면 5m alignment 안 되는 신호를 자동으로 거른다.
        """
        signals = self.detector.detect(df_intraday, df_daily, df_5m=df_5m)
        trades, equity_curve = self._simulate(df_intraday, signals, config)
        perf = self._performance(trades, config)
        return StrategyResult(
            config=config,
            performance=perf,
            equity_curve=equity_curve,
        )

    # ---- core sim --------------------------------------------------

    def _simulate(
        self,
        df: pd.DataFrame,
        signals: list[BullFlagSignal],
        config: StrategyConfig,
    ) -> tuple[list[Trade], list[EquityPoint]]:
        sig_by_ts = {s.entry_ts: s for s in signals}

        if df.empty:
            return [], [EquityPoint(date=config.start_date, equity=config.initial_capital)]
        local_dates = [ts.date() for ts in df.index]

        equity = float(config.initial_capital)
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        open_pos: dict | None = None
        # 세션 단위 loss 카운터 — ``max_session_losses`` 도달 시 그
        # 세션의 추가 진입 차단. 새 세션 시작 시 0으로 리셋.
        session_loss_count = 0
        current_session = None
        # 직전 trade의 exit 시각 — "second pullback" 게이트에 사용.
        # 새 trade의 pole_start_ts가 이 시각 이전이면 직전 trade의
        # up-move를 재활용하는 것 → reject (영상 정통 룰).
        last_exit_ts: pd.Timestamp | None = None

        idx = df.index
        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)

        for i in range(len(df)):
            ts = idx[i]
            sess = local_dates[i]
            is_last_of_session = i == len(df) - 1 or local_dates[i + 1] != sess

            # 새 세션 진입 시 loss 카운터 + last_exit_ts 리셋
            # (second-pullback 게이트는 같은 세션 안에서만 의미).
            if sess != current_session:
                current_session = sess
                session_loss_count = 0
                last_exit_ts = None

            if open_pos is None:
                sig = sig_by_ts.get(ts)
                # max_session_losses 초과 시 신규 진입 차단.
                if self.max_session_losses > 0 and session_loss_count >= self.max_session_losses:
                    sig = None
                # 영상 정통: "second pullback" = 첫 익절 후 NEW pole + NEW pullback.
                # 폴이 직전 trade hold 기간을 잡으면(pole_start_ts ≤ last_exit_ts)
                # 같은 up-move를 재활용 — 영상 룰 위반. ALGS 4/16에서 첫 trade가
                # 10:48-10:51 hold + TP, 두 번째 신호의 pole이 10:49(=hold 중)
                # 부터 잡혀서 10:56에 잘못 fire하는 케이스 차단.
                if sig is not None and last_exit_ts is not None:
                    sig_pole_start = pd.Timestamp(sig.pole_start_ts)
                    cmp_exit = pd.Timestamp(last_exit_ts)
                    # tz 정합
                    if sig_pole_start.tzinfo is None and cmp_exit.tzinfo is not None:
                        sig_pole_start = sig_pole_start.tz_localize(cmp_exit.tz)
                    elif sig_pole_start.tzinfo is not None and cmp_exit.tzinfo is None:
                        cmp_exit = cmp_exit.tz_localize(sig_pole_start.tz)
                    if sig_pole_start <= cmp_exit:
                        sig = None
                if sig is not None:
                    risk_per_share = sig.entry_price - sig.stop_loss
                    if risk_per_share <= 0:
                        continue

                    # R/R 게이트 — 영상의 "I won't take the trade if
                    # I don't think I can get 2:1".
                    # epsilon 1e-9: FP 오차로 R/R=2.0을 살짝 밑돌게 계산되는
                    # 케이스(예: 7.97-7.85=0.119999... → R/R=1.99999... < 2.0)를
                    # 정상 통과시키기 위함. target_at_r_multiple로 fixed target
                    # 잡으면 expected_r은 이론상 정확히 target_at_r_multiple이지만
                    # FP 부정확성으로 미세하게 작아질 수 있음.
                    target_price = self._initial_target(sig, risk_per_share)
                    expected_r = (target_price - sig.entry_price) / risk_per_share
                    if expected_r + 1e-9 < self.target_min_r_multiple:
                        continue

                    # 사이즈: risk_per_trade × equity / risk_per_share,
                    # notional 캡 적용.
                    risk_dollars = equity * config.risk_per_trade
                    risk_shares = int(risk_dollars // risk_per_share)
                    if self.max_position_pct_of_equity > 0 and sig.entry_price > 0:
                        max_notional = equity * self.max_position_pct_of_equity
                        cap_shares = int(max_notional // sig.entry_price)
                        shares = min(risk_shares, cap_shares)
                    else:
                        shares = risk_shares
                    if shares <= 0:
                        continue

                    # Toss 수수료 적용 — entry buy fee
                    entry_commission = (
                        self.fee_schedule.buy_fee(sig.entry_price, shares)
                        if self.fee_schedule else 0.0
                    )
                    open_pos = {
                        "entry_ts": ts,
                        # 표시/BE-stop 기준이 되는 *최초* entry 가격. add 후에도 절대 변하지 않음.
                        "entry_price": sig.entry_price,
                        # PnL 계산용 가중평균. add 발화 시 weighted avg로 업데이트.
                        "avg_cost": sig.entry_price,
                        "initial_stop": sig.stop_loss,
                        "stop": sig.stop_loss,
                        "target": target_price,
                        "shares": shares,
                        "initial_risk": risk_per_share,
                        # add-to-winner 상태
                        "added": False,
                        # 누적 수수료 — exit 시 sell fee 더해서 PnL에서 차감
                        "commission": entry_commission,
                    }
            if open_pos is None:
                continue

            exit_price: float | None = None
            exit_reason: str | None = None
            on_entry_bar = ts == open_pos["entry_ts"]

            if not on_entry_bar:
                # OHLC만 알고 intrabar 순서를 모르므로 보수적 우선순위:
                #   1) Stop (현재 stop) — 손절 우선
                #   2) TP (현재 target) — 손절 안 닿았으면 익절
                #   3) Add — 둘 다 안 닿은 경우만 doubling
                # (이전 순서: add → stop → TP. add 발화로 BE stop이 같은 봉
                #  low에 즉시 발사되면서 봉 high가 이미 TP를 넘었어도 무시되는
                #  버그. 사용자 ALGS 4/16 케이스에서 같은 캔들 entry+stop 표시.)

                # ---- 1. Stop check (현재 stop 기준) ----
                if lows[i] <= open_pos["stop"]:
                    exit_price = open_pos["stop"]
                    exit_reason = "breakeven_stop" if open_pos["added"] else "stop_loss"

                # ---- 2. Take-profit check (HoD or fixed R) ----
                if exit_price is None and highs[i] >= open_pos["target"]:
                    exit_price = open_pos["target"]
                    exit_reason = "take_profit"

                # ---- 3. Add-to-winner (이번 봉에서 stop/TP 안 닿았을 때만) ----
                if (
                    exit_price is None
                    and self.enable_add_to_winner
                    and not open_pos["added"]
                ):
                    add_trigger = open_pos["entry_price"] + self.add_at_r * open_pos["initial_risk"]
                    # close 확정 모드(default)면 봉 close가 트리거 위로
                    # 닫혀야 add 발화. 1m wick 한 번에 트리거되는 fake-add 차단.
                    trigger_price = closes[i] if self.add_confirm_on_close else highs[i]
                    if trigger_price >= add_trigger:
                        # 더블링 — Ross 영상 정통 룰:
                        #   1) 같은 수량 추가 → shares 2x
                        #   2) 평단가(avg_cost)는 PnL 계산용으로만 업데이트
                        #   3) BE stop은 **최초 entry 가격**으로 끌어올림
                        #      (avg가 아님! avg-buffer로 두면 add 직후 wick에
                        #      즉시 BE stop 발화하는 ALGS 4/16 LOSS 케이스 발생.
                        #      Ross의 영상 표현: "stop at original entry —
                        #      worst case I'm flat on initial size".)
                        added_shares = open_pos["shares"]
                        new_shares = open_pos["shares"] * 2
                        avg_price = (open_pos["entry_price"] + add_trigger) / 2
                        if self.fee_schedule:
                            open_pos["commission"] += self.fee_schedule.buy_fee(
                                add_trigger, added_shares
                            )
                        open_pos["shares"] = new_shares
                        open_pos["avg_cost"] = avg_price
                        # 최초 entry 가격으로 BE stop. be_stop_buffer_pct는
                        # 살짝 아래로 (= entry × (1 - buffer))로 두는 옵션 — 진입가
                        # 정확히 한 번 tag 시 즉시 발사 방지용. 기본 0.3% = $7.94 → $7.92.
                        open_pos["stop"] = open_pos["entry_price"] * (1 - self.be_stop_buffer_pct)
                        open_pos["added"] = True
                        # 익절가는 그대로 유지 — 영상은 "let it run".

            # 세션 마지막 봉 — 강제 청산
            if exit_price is None and is_last_of_session:
                exit_price = float(closes[i])
                exit_reason = "session_close"

            if exit_price is not None:
                shares = open_pos["shares"]
                # Toss 수수료 — exit sell fee + SEC fee
                if self.fee_schedule:
                    open_pos["commission"] += self.fee_schedule.sell_fee(
                        exit_price, shares
                    )
                # PnL은 weighted avg cost 기준 (add 했으면 avg, 아니면 = entry).
                gross_pnl = (exit_price - open_pos["avg_cost"]) * shares
                pnl = gross_pnl - open_pos["commission"]
                pnl_pct = (
                    pnl / (open_pos["avg_cost"] * shares)
                    if open_pos["avg_cost"] > 0 and shares > 0
                    else 0.0
                )
                trades.append(
                    Trade(
                        pattern_name=config.pattern_name,
                        entry_date=open_pos["entry_ts"].date(),
                        exit_date=ts.date(),
                        # 표시는 *최초* entry 가격 — add 후 avg로 덮어쓰면 사용자가
                        # "$7.94에 진입했는데 왜 $8.00으로 표시?" 헷갈림.
                        entry_price=open_pos["entry_price"],
                        exit_price=exit_price,
                        stop_loss=open_pos["stop"],
                        shares=shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason or "end_of_data",
                        entry_ts=_to_naive(open_pos["entry_ts"]),
                        exit_ts=_to_naive(ts),
                    )
                )
                equity += pnl
                # 세션 loss 카운트 — stop_loss / breakeven_stop 모두
                # "체결됐는데 trade가 잘 안 풀린 케이스"라 손실로 카운트.
                # take_profit / session_close 는 카운트 X.
                if exit_reason in ("stop_loss", "breakeven_stop"):
                    session_loss_count += 1
                # second-pullback 게이트용 — 직전 trade exit 시각 기록.
                last_exit_ts = ts
                open_pos = None

            if is_last_of_session:
                equity_curve.append(EquityPoint(date=sess, equity=equity))

        if not equity_curve:
            equity_curve.append(EquityPoint(date=config.start_date, equity=equity))
        return trades, equity_curve

    # ---- target helpers -------------------------------------------

    def _initial_target(self, sig: BullFlagSignal, risk_per_share: float) -> float:
        """초기 target가 — 옵션에 따라 HoD 또는 고정 R-multiple."""
        if self.target_at_r_multiple is not None:
            return sig.entry_price + (self.target_at_r_multiple * risk_per_share)
        return sig.hod_at_entry

    # ---- performance ----------------------------------------------

    @staticmethod
    def _performance(trades: list[Trade], config: StrategyConfig) -> StrategyPerformance:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        final = config.initial_capital + total_pnl
        avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss_pct = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
        max_dd = 0.0
        peak = config.initial_capital
        running = config.initial_capital
        for t in trades:
            running += t.pnl
            peak = max(peak, running)
            dd = (peak - running) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return StrategyPerformance(
            initial_capital=config.initial_capital,
            final_capital=final,
            total_return_pct=(
                (final - config.initial_capital) / config.initial_capital
                if config.initial_capital > 0
                else 0.0
            ),
            total_trades=len(trades),
            win_rate=(len(wins) / len(trades) if trades else 0.0),
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            max_drawdown_pct=max_dd,
            trades=trades,
        )


def _to_naive(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is not None:
        return ts.tz_convert(ts.tz).tz_localize(None).to_pydatetime()
    return ts.to_pydatetime()
