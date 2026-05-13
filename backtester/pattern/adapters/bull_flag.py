"""Bull Flag pattern detector — Ross Cameron / Warrior Trading style.

Adapted from "How I Made $1,000,000 in 51 Days of Day Trading"
(YouTube ``m5zu_X-_51I``). The setup is intraday-only and requires
the underlying stock to satisfy four hard supply/demand filters
(영상의 5 criteria 중 News를 제외한 4가지) before any pattern
geometry is even checked:

    1. **Gap-up ≥ +2% pre-market** (open vs prior daily close).
       영상은 +10%를 이상적이라고 하지만 백테스트 가용성을 위해
       하한은 +2% (영상의 minimum threshold).
    2. **5x Relative Volume** vs. 50-day average daily volume.
    3. **Price between $2 and $20** at session open.
    4. **Float < 10M shares** outstanding (외부 입력으로 주입).

이 4가지를 모두 만족하지 않으면 그날은 아예 스캔 대상에서 제외.
잔여 News 필터(5번)는 yfinance만으로는 못 받아서 빠져 있음 — 추후
news catalyst 데이터 소스가 붙으면 ``news_filter`` hook 추가.

세션 단위 게이트를 통과한 뒤에는 인트라데이 봉에서 Bull Flag
geometry를 잡는다:

    - **Pole**: 직전 ``pole_lookback`` 봉 안에서 누적 +``pole_min_pct``
      이상 상승 + 그린 캔들이 ``pole_min_green_bars`` 이상.
    - **Flag (풀백)**: pole 직후 1~``flag_max_bars`` 봉. 풀백 저점이
      pole 시작점 + 50% retrace 안쪽이어야 bullish.
    - **Breakout**: 풀백 끝난 뒤 직전 캔들의 high를 close로 돌파한
      첫 번째 캔들.

**시간 프레임 (영상 46:17~47:02)**:
영상에서 Ross가 직접 언급한 사용 가능 timeframe은 ``10초 / 1분 / 5분 / 15분``.
**디폴트 시연 = 1분봉**. 가장 잘 통하는 셋업은 **lower timeframe
(10초·1분)의 첫 풀백** ("first pullbacks on the lower time frames
like 10-second, one minute usually work well"). 5분/15분 첫 풀백도
양호. 본 detector는 인트라데이 OHLCV에 generic하게 동작 — 호출자가
1m/5m/15m 어떤 frame이든 ``df_intraday``로 주입하면 됨. 실제 영상
디폴트와 본 백테스트 default는 ``1m``.

**영상에서 직접 거론된 ticker 사례**:
- **OSR** — 영상 촬영 당일 +120% (\$2.50 → \$5.50, 20분), \$12,227 수익
- **ATNF** — +564% gap-up, 5800만주, \$98,754 수익
- **MLGO** — Float ~800K / 거래량 3억주 / +430% (저-Float 대표)
- **IMTE** — 비슷한 저-Float 케이스
- **Ford (F)** — counter-example (라지캡 횡보 — 절대 트레이드 X)

Long-only. 영상은 short side는 다루지 않음.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BullFlagSignal:
    """Single intraday Bull Flag long entry.

    필드 의미:
      - ``pole_start_ts / pole_start_price``: 풀백 직전 가장 낮았던
        스윙 저점 (= 폴 시작 기준점). stop은 이 가격이 아니라 풀백
        저점에 둔다.
      - ``pole_end_ts / pole_end_price``: 풀백 시작 직전 캔들의 high.
      - ``flag_low / flag_low_ts``: 풀백 안에서의 가장 낮은 low — 손절가.
      - ``entry_ts / entry_price``: breakout 캔들의 close.
      - ``stop_loss``: ``flag_low - tick``.
      - ``hod_at_entry``: 진입 시점까지의 일중 최고가. 1차 익절 후보.
      - ``r_multiple_to_hod``: ``(hod - entry) / (entry - stop)``.
      - ``gap_pct / rvol_5x / price_in_range``: 통과한 일별 필터의 값
        (디버그/리포트용).
    """

    session_date: date
    direction: str  # always "long"

    # --- pattern geometry ---
    pole_start_ts: pd.Timestamp
    pole_start_price: float
    pole_end_ts: pd.Timestamp
    pole_end_price: float
    flag_low: float
    flag_low_ts: pd.Timestamp

    # --- entry ---
    entry_ts: pd.Timestamp
    entry_price: float
    stop_loss: float
    hod_at_entry: float
    r_multiple_to_hod: float

    # --- session-level filter values (debug/report) ---
    gap_pct: float
    rvol: float
    open_price: float
    float_shares: float | None


class BullFlagDetector:
    """Long-only Bull Flag detector with hard supply/demand gate.

    Parameters
    ----------
    float_shares:
        주식 발행량(float) — 외부에서 주입. None이면 float 필터를
        통과시키지 않음(=항상 reject) — 영상의 5 criteria 중
        가장 핵심이라 None을 허용하면 비-A-quality 종목을 통과시킬
        위험이 큼. 명시적으로 ``require_float_filter=False``로
        끄기 전엔 강제.
    require_float_filter:
        float 데이터 없을 때 필터를 스킵할지. 기본 True.
    min_gap_pct:
        세션 시초가가 전일 종가 대비 최소 ``min_gap_pct`` 이상
        gap-up 했을 때만 통과. 영상은 "최소 +2%, 이상적 +10%".
        백테스트에서 +2%면 너무 많은 후보가 통과해서 수익 분산
        효과가 떨어진다면 +5~10%로 조여서 시험.
    min_rvol:
        오늘 거래량 / 50일 평균 거래량 의 최소값. 영상 기본 5.0.
    min_price / max_price:
        세션 시초가 기준 가격 범위. 영상 기본 $2 / $20.
    max_float_shares:
        영상 기본 10M (10_000_000). 더 빡빡하게 가려면 5M으로.
    pole_lookback:
        breakout 후보 봉 직전 N봉 안에서 폴 검출.
    pole_min_pct:
        폴 시작점부터 폴 끝점(=풀백 시작 직전)까지 최소 상승률.
        기본 +5%. 영상은 명시적 수치는 없지만 "5~7개의 그린 캔들"이
        대략 +5~10% 상승에 해당.
    pole_min_green_bars:
        폴 구간에 포함돼야 하는 bullish (close>open) 봉의 최소 수.
    flag_max_bars:
        풀백을 구성하는 캔들 수의 상한. 기본 3 — 영상 표현 "1~3개의
        풀백 캔들".
    flag_max_retrace:
        flag_low가 폴 상승분의 50% 이상을 지켜야 bullish.
        ``(pole_end - flag_low) / (pole_end - pole_start) <= 0.5``.
    latest_entry_local:
        세션 내 진입 가능 마감 시각 — 기본 11:30 ET. 영상의
        "sweet spot 시간대"는 개장 후 1~2시간.
    stop_tick_buffer:
        flag_low 아래로 한 틱 더 깎아 stop 위치 결정. 기본 $0.01.
    """

    name = "bull_flag"

    def __init__(
        self,
        float_shares: float | None = None,
        require_float_filter: bool = True,
        min_gap_pct: float = 0.02,
        min_rvol: float = 5.0,
        min_price: float = 2.0,
        max_price: float = 20.0,
        max_float_shares: float = 10_000_000.0,
        pole_lookback: int = 7,
        pole_min_pct: float = 0.08,
        pole_min_green_bars: int = 3,
        flag_max_bars: int = 4,
        flag_max_retrace: float = 0.5,
        latest_entry_local: time = time(12, 0),
        stop_tick_buffer: float = 0.01,
        # ---- Bar-quality / data-continuity guards ----
        # 영상의 화면엔 doji 봉/빈 분봉이 없음 — Ross가 보는 화면은 항상
        # 거래가 활발한 high-volume runner의 dense 분봉. 우리 데이터(yfinance)
        # 는 sparse일 수 있으므로 "영상의 시각적 quality"를 확보하기 위해
        # 데이터 품질 게이트를 추가.
        # 1) 봉의 high-low 가 이 값 이하면 zero-range(무거래)로 간주하고 거름.
        min_bar_range: float = 0.001,
        # 2) pole+flag+breakout 윈도우 안의 인접 봉 시간차가 이 값을
        #    초과하면 데이터 갭으로 보고 그 윈도우 reject. 1m chart에서
        #    90s = 한 봉 갭까지 허용. 60s = 엄격.
        max_bar_gap_seconds: int = 90,
        # ---- Volume profile (DP4 영상 P0) ----
        # Bull Flag의 핵심: pole은 高 거래량, flag는 低 거래량,
        # breakout은 다시 高 거래량. Ross가 영상에서 가장 강조한 룰.
        enable_volume_profile: bool = True,
        # flag 평균 거래량 / pole 평균 거래량 의 max 허용 비율.
        # 기본 0.7 = flag 거래량이 pole의 70% 이하여야 함.
        pullback_volume_ratio: float = 0.7,
        # breakout 봉 거래량 / **flag 평균 거래량** 의 min 허용 비율.
        # 영상은 "even higher volume" 만 qualitative로 언급, 수치적 임계값
        # 명시 X. 기본 0.0 = 비활성 (영상 정통). 사용자가 특정 시장에서
        # fake breakout 차단 위해 0.5~1.0으로 옵트인 가능.
        # 비교 baseline은 풀백 평균 ("fresh buyers return" 의미에 충실).
        breakout_volume_ratio: float = 0.0,
        # ---- Topping tail reject (DP4 영상 P1) ----
        # 폴 마지막 봉의 upper wick / total range 가 이 값을 넘으면
        # "위에서 거부됨" → bull flag reject. 기본 0.5 = 위꼬리가
        # 전체 봉의 절반 이상이면 안 됨.
        max_pole_topping_tail_ratio: float = 0.5,
        # ---- Multi-timeframe alignment (DP4 영상 33:30~35:00 정통 룰) ----
        # 영상: "both 1m and 5m giving the same signal" — 즉 두 분봉 모두
        # bull flag 패턴을 감지해야 alignment 성립. VVPR 사례에서 Ross는
        # 1m이 풀백 형성 중 + 5m이 막 돌파 중 → 둘 다 buy 신호.
        # 구현: 같은 BullFlagDetector를 5m 데이터에도 적용해서, 1m 신호
        # 시점 ±``mtf_tolerance_seconds`` 안에 5m 신호가 있는지 확인.
        enable_mtf_check: bool = False,
        mtf_tolerance_seconds: int = 600,
        # ---- Reverse-split / data-continuity guard ----
        # ``splits`` 는 yfinance 의 ``Ticker.splits`` Series — index가
        # split 발생일, 값이 비율 (예: 1:15 reverse split = 0.0667).
        # 이 날짜로부터 ±``split_blackout_days`` 거래일 안의 세션은
        # 전부 거름. UGRO처럼 reverse split 직전후의 sub-\$1 데이터에
        # 5 criteria 통과 + bull flag fake-fire 이슈를 차단.
        splits: pd.Series | None = None,
        split_blackout_days: int = 30,
        # 직전 ``price_floor_lookback_days`` 거래일 동안 daily low가
        # ``min_price`` 미만으로 한 번이라도 내려간 적이 있으면 거름.
        # split-adjusted price가 시간에 따라 점프하는 종목(sub-\$1
        # → \$7+ 같은) 을 catch.
        price_floor_lookback_days: int = 30,
        # ---- CSV 결과 분석 기반 새 필터 (2026-05 export 분석) ----
        # 단일 변수 분석에서 가장 임팩트 큰 3가지 — 이 값들을 넘으면
        # 손실 가능성 대폭 증가 (export CSV 34건 분석 결과 win-rate가
        # 36%로 떨어짐).
        #
        # ``max_rvol`` — RVOL 상한. 영상 정통은 "≥ 5x" 만 명시. 그러나
        # ≥ 30x는 이미 over-extended → setup played out. 기본 30.0.
        max_rvol: float | None = 30.0,
        # ``max_gap_pct`` — pre-market gap 상한. 영상은 "≥ +2%, 이상적
        # +10%" 만 명시. > 50% gap도 trade 통과 시 win-rate 75% 유지
        # (2026-05-13 sweep 결과 → 0.30 대비 +2.98%p return, P/L↑).
        # 기본 0.50.
        max_gap_pct: float | None = 0.50,
        # ``max_stop_distance_pct`` — entry → stop 거리 (risk %)의 상한.
        # > 5%이면 entry가 실제 지지선(flag_low)에서 너무 멀어진 셋업 →
        # wide stop → poor R/R. 기본 0.05.
        max_stop_distance_pct: float = 0.05,
        # ---- Ross 영상 정통 추가 룰 ----
        # ``require_9ema_support`` — "I use 9 EMA on every timeframe"
        # (Ross). 풀백 저점이 9 EMA에서 받쳐주는 셋업이 정통. 풀백
        # 저점 봉의 low가 ``ema9_tolerance_pct`` 안에 있어야 통과.
        require_9ema_support: bool = True,
        # 2.5% — sweep 결과 0.015 → 0.025로 늘려도 win-rate 75% 유지
        # (오히려 0.015 대비 +1 trade). 영상의 "9 EMA hold"가 tick 정확
        # 일치 아니라는 점 반영. 0.03 까지 늘리면 +3 trade인데 P/L 하락.
        ema9_tolerance_pct: float = 0.025,
        # ``also_accept_20ema_support`` — sweep-tunable. False가 영상
        # 정통(Ross는 9 EMA만 강조). True로 켜면 9 EMA 못 닿더라도
        # 20 EMA 근방이면 통과 (Brett Burgett / Nathan Michaud 룰).
        # 기본 False — 데이터로 효과 검증 후 default 변경 여부 결정.
        also_accept_20ema_support: bool = False,
        # ``require_daily_trend`` — Ross 영상엔 daily SMA 명시 X.
        # Sweep 결과 (2026-05-13): True/False 비교 시 False가 +1 trade,
        # win 75% 유지, return +1.93%p. Ross 영상에 없는 보조 게이트
        # 였고 데이터 상 도움 안 되어 default OFF. 보수적 운영 원하면
        # True로 켜기.
        require_daily_trend: bool = False,
        daily_trend_sma_period: int = 50,
        # ``max_nth_pullback`` — Ross: "1st/2nd pullback work well,
        # 3rd start to be cautious". 1 = 첫만, 2 = 첫·둘째, 3 = 셋째까지.
        # 기본 2 (영상 정통 보수적 해석).
        max_nth_pullback: int = 2,
        # ``premarket_high_by_date`` — Ross는 PM high를 저항/돌파 기준으로 봄.
        # 페이지/composition root에서 RegularSessionFilter 적용 전 raw
        # intraday로 (start, end) 동안 세션별 PM high 계산해서 주입.
        # 진입가 < PM high면 reject. None이면 게이트 비활성 (영상 정통은
        # PM high를 명시적으로 룰화하지 않으니 기본 None).
        premarket_high_by_date: dict[date, float] | None = None,
    ) -> None:
        if pole_lookback < 2:
            raise ValueError("pole_lookback must be >= 2")
        if not 0 < flag_max_retrace <= 1:
            raise ValueError("flag_max_retrace must be in (0, 1]")
        self.float_shares = float_shares
        self.require_float_filter = require_float_filter
        self.min_gap_pct = min_gap_pct
        self.min_rvol = min_rvol
        self.min_price = min_price
        self.max_price = max_price
        self.max_float_shares = max_float_shares
        self.pole_lookback = pole_lookback
        self.pole_min_pct = pole_min_pct
        self.pole_min_green_bars = pole_min_green_bars
        self.flag_max_bars = flag_max_bars
        self.flag_max_retrace = flag_max_retrace
        self.latest_entry_local = latest_entry_local
        self.stop_tick_buffer = stop_tick_buffer
        self.enable_volume_profile = enable_volume_profile
        self.pullback_volume_ratio = pullback_volume_ratio
        self.breakout_volume_ratio = breakout_volume_ratio
        self.max_pole_topping_tail_ratio = max_pole_topping_tail_ratio
        self.enable_mtf_check = enable_mtf_check
        self.mtf_tolerance_seconds = mtf_tolerance_seconds
        self.min_bar_range = min_bar_range
        self.max_bar_gap_seconds = max_bar_gap_seconds
        self.splits = splits
        self.split_blackout_days = split_blackout_days
        self.price_floor_lookback_days = price_floor_lookback_days
        # New filters
        self.max_rvol = max_rvol
        self.max_gap_pct = max_gap_pct
        self.max_stop_distance_pct = max_stop_distance_pct
        self.require_9ema_support = require_9ema_support
        self.ema9_tolerance_pct = ema9_tolerance_pct
        self.also_accept_20ema_support = also_accept_20ema_support
        self.require_daily_trend = require_daily_trend
        self.daily_trend_sma_period = daily_trend_sma_period
        self.max_nth_pullback = max_nth_pullback
        self.premarket_high_by_date = premarket_high_by_date or {}

    # ---- public API ------------------------------------------------

    def detect(
        self,
        df_intraday: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_5m: pd.DataFrame | None = None,
    ) -> list[BullFlagSignal]:
        """
        df_5m: optional. ``enable_mtf_check=True`` 일 때만 사용. 1m 신호
        시점의 5m 차트 정합성을 추가 검증해서 alignment 안 되는 신호는
        걸러낸다. None이면 MTF 게이트는 통과로 간주.
        """
        if df_intraday.empty or df_daily.empty:
            return []

        # ---- Stock Selection: 5 criteria 중 4개 게이트 ----
        # 1) Float < max_float_shares — 종목 단위 단발성 체크.
        if self.require_float_filter:
            if self.float_shares is None:
                return []  # data 없으면 보수적으로 reject
            if self.float_shares >= self.max_float_shares:
                return []

        # 2~4) 일별 게이트(gap %, RVOL, price 범위) — 통과 세션 set 계산.
        qualifying = self._qualifying_sessions(df_daily)

        signals: list[BullFlagSignal] = []
        local_dates = self._local_dates(df_intraday.index)
        for sess_date, sess in self._iter_sessions(df_intraday, local_dates):
            if sess_date not in qualifying:
                continue
            sess_signals = self._scan_session(
                sess_date, sess, qualifying[sess_date]
            )
            signals.extend(sess_signals)

        # ---- Post-filter: Multi-timeframe alignment (DP4 영상 정통 룰) ----
        # 영상: "both 1m and 5m giving the same signal".
        # 같은 detector를 5m에도 돌려서 1m 신호 시점 ±tolerance 안에 5m
        # 신호가 있으면 alignment 성립. 없으면 1m 신호 거름.
        if self.enable_mtf_check and df_5m is not None and len(df_5m) > 0:
            sigs_5m = self._detect_5m_signals(df_5m, df_daily)
            signals = [
                s for s in signals if self._mtf_aligned(s.entry_ts, sigs_5m)
            ]

        return signals

    def _detect_5m_signals(
        self, df_5m: pd.DataFrame, df_daily: pd.DataFrame
    ) -> list["BullFlagSignal"]:
        """5m 데이터에 같은 BullFlagDetector를 적용해서 신호 추출.

        Recursion 방지: ``enable_mtf_check`` 잠시 False로 토글하고 detect.
        5m bar gap은 약 5분(300s)이므로 ``max_bar_gap_seconds`` 도 5m
        스케일로 일시 조정 (300 + 100 slack = 400s).
        """
        saved_mtf = self.enable_mtf_check
        saved_gap = self.max_bar_gap_seconds
        try:
            self.enable_mtf_check = False
            self.max_bar_gap_seconds = max(self.max_bar_gap_seconds, 400)
            return self.detect(df_5m, df_daily, df_5m=None)
        finally:
            self.enable_mtf_check = saved_mtf
            self.max_bar_gap_seconds = saved_gap

    def _mtf_aligned(
        self,
        signal_ts: pd.Timestamp,
        sigs_5m: list["BullFlagSignal"],
    ) -> bool:
        """1m signal 시점 ±tolerance 안에 5m bull flag 신호가 있는지 확인."""
        if not sigs_5m:
            return False
        sig_ts = pd.Timestamp(signal_ts)
        for s5 in sigs_5m:
            if s5.session_date != sig_ts.date():
                continue
            s5_ts = pd.Timestamp(s5.entry_ts)
            # tz 정합
            if sig_ts.tzinfo is None and s5_ts.tzinfo is not None:
                cmp_sig = sig_ts.tz_localize(s5_ts.tz)
                cmp_s5 = s5_ts
            elif sig_ts.tzinfo is not None and s5_ts.tzinfo is None:
                cmp_sig = sig_ts
                cmp_s5 = s5_ts.tz_localize(sig_ts.tz)
            else:
                cmp_sig = sig_ts
                cmp_s5 = s5_ts
            gap = abs((cmp_sig - cmp_s5).total_seconds())
            if gap <= self.mtf_tolerance_seconds:
                return True
        return False

    # ---- session-level filter --------------------------------------

    def _qualifying_sessions(
        self, df_daily: pd.DataFrame
    ) -> dict[date, dict]:
        """Return {session_date: {gap_pct, rvol, open_price}} for sessions
        passing all gates: (gap_pct, rvol, price range, no recent split,
        price floor over lookback).
        """
        if "Open" not in df_daily.columns or "Close" not in df_daily.columns:
            return {}

        prev_close = df_daily["Close"].shift(1)
        gap_pct = (df_daily["Open"] - prev_close) / prev_close
        avg_vol = df_daily["Volume"].rolling(50, min_periods=20).mean()
        rvol = df_daily["Volume"] / avg_vol

        # Price-floor lookback: 최근 N거래일 동안 daily Low가 한 번이라도
        # min_price 아래로 내려갔다면 그 세션은 거른다. UGRO처럼
        # \$0.41 → \$7.35 점프(reverse split)는 직전 30일 lookback에서
        # 반드시 sub-min_price 봉을 포함하므로 여기서 자연스럽게 차단됨.
        # price_floor_lookback_days=0 → 게이트 비활성화 (전 세션 통과)
        if self.price_floor_lookback_days <= 0:
            floor_ok = pd.Series(True, index=df_daily.index)
        else:
            rolling_min_low = (
                df_daily["Low"]
                .rolling(
                    self.price_floor_lookback_days,
                    min_periods=min(5, self.price_floor_lookback_days),
                )
                .min()
            )
            floor_ok = rolling_min_low >= self.min_price

        gate = (
            (gap_pct >= self.min_gap_pct)
            & (rvol >= self.min_rvol)
            & (df_daily["Open"] >= self.min_price)
            & (df_daily["Open"] <= self.max_price)
            & floor_ok
        )
        # RVOL upper cap — over-extended momentum (CSV 분석: ≥30x 시
        # 36% win rate). None이면 비활성.
        if self.max_rvol is not None:
            gate = gate & (rvol <= self.max_rvol)
        # Gap upper cap — too-far gap = mean-reversion risk
        # (CSV 분석: >30% gap에서 36% win rate).
        if self.max_gap_pct is not None:
            gate = gate & (gap_pct <= self.max_gap_pct)
        # Daily trend filter — Ross: "daily uptrend 상에서만 진입".
        # 진입일 close > SMA{period} 인지 확인. SMA 계산용 충분한
        # 히스토리가 없으면 (rolling NaN) 보수적으로 False.
        if self.require_daily_trend:
            sma = (
                df_daily["Close"]
                .rolling(
                    self.daily_trend_sma_period,
                    min_periods=max(10, self.daily_trend_sma_period // 4),
                )
                .mean()
            )
            trend_ok = (df_daily["Close"] > sma).fillna(False)
            gate = gate & trend_ok

        # Split blackout — split 이벤트 전후 ±N거래일 거름.
        # ``Ticker.splits`` 는 split 발생일을 인덱스로, 비율(>1=forward,
        # <1=reverse)을 값으로 가짐. 둘 다(forward + reverse) 모두
        # 데이터 불연속을 만들므로 동일하게 거른다.
        if self.splits is not None and len(self.splits) > 0:
            blackout_dates: set[date] = set()
            n = self.split_blackout_days
            daily_idx_dates = [ts.date() for ts in df_daily.index]
            daily_dates_arr = np.array(daily_idx_dates)
            for split_ts in self.splits.index:
                split_date = (
                    split_ts.date() if hasattr(split_ts, "date") else split_ts
                )
                # 가장 가까운 거래일 위치 찾기
                if split_date < daily_idx_dates[0] or split_date > daily_idx_dates[-1]:
                    # split이 daily 윈도우 밖에 있어도 인접 N일은 차단해야
                    # 하므로 가까운 boundary 처리.
                    if split_date < daily_idx_dates[0]:
                        center_loc = 0
                    else:
                        center_loc = len(daily_idx_dates) - 1
                else:
                    center_loc = int(np.searchsorted(daily_dates_arr, split_date))
                lo = max(0, center_loc - n)
                hi = min(len(daily_idx_dates), center_loc + n + 1)
                for d in daily_idx_dates[lo:hi]:
                    blackout_dates.add(d)
        else:
            blackout_dates = set()

        out: dict[date, dict] = {}
        for ts, ok in gate.items():
            if not bool(ok):
                continue
            d = ts.date()
            if d in blackout_dates:
                continue
            out[d] = {
                "gap_pct": float(gap_pct.loc[ts]),
                "rvol": float(rvol.loc[ts]),
                "open_price": float(df_daily.loc[ts, "Open"]),
            }
        return out

    # ---- intraday helpers -----------------------------------------

    @staticmethod
    def _local_dates(idx: pd.DatetimeIndex) -> np.ndarray:
        return np.array([ts.date() for ts in idx])

    @staticmethod
    def _iter_sessions(
        df: pd.DataFrame, local_dates: np.ndarray
    ) -> Iterable[tuple[date, pd.DataFrame]]:
        if len(df) == 0:
            return
        change = np.r_[True, local_dates[1:] != local_dates[:-1]]
        boundaries = np.flatnonzero(change)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(df)
            yield local_dates[start], df.iloc[start:end]

    # ---- core pattern logic ---------------------------------------

    def _scan_session(
        self,
        sess_date: date,
        sess: pd.DataFrame,
        session_meta: dict,
    ) -> list[BullFlagSignal]:
        """Scan a single qualified session for Bull Flag entries.

        한 세션 안에서 여러 풀백이 발생할 수 있다 — Ross Cameron은
        영상에서 첫 번째와 두 번째 풀백을 trade하지만 세 번째부턴
        보수적으로 가라고 함. 여기서는 ``flag_max_bars`` 안의 모든
        breakout을 후보로 잡고, 같은 폴(=같은 pole_start_ts)에 대해
        최대 2개의 신호까지 허용.
        """
        idx = sess.index
        local_time = idx if idx.tz is None else idx.tz_convert(idx.tz)
        cutoff_minutes = (
            self.latest_entry_local.hour * 60
            + self.latest_entry_local.minute
        )
        bar_minutes = np.array(
            [t.hour * 60 + t.minute for t in local_time.time]
        )

        opens = sess["Open"].to_numpy(dtype=float)
        highs = sess["High"].to_numpy(dtype=float)
        lows = sess["Low"].to_numpy(dtype=float)
        closes = sess["Close"].to_numpy(dtype=float)
        volumes = (
            sess["Volume"].to_numpy(dtype=float)
            if "Volume" in sess.columns
            else np.zeros(len(sess), dtype=float)
        )
        timestamps = sess.index

        n = len(sess)
        if n < self.pole_lookback + self.flag_max_bars + 1:
            return []

        # 9 EMA — Ross's "I use this on every time frame". Computed
        # once per session (instead of per-iteration) for speed.
        ema9 = (
            pd.Series(closes).ewm(span=9, adjust=False).mean().to_numpy()
        )
        # 20 EMA — fallback support level used by many bull-flag
        # gurus (Brett Burgett, Nathan Michaud) and Ross himself when
        # 9 EMA gets violated but 20 EMA holds. Computed only if the
        # fallback gate is enabled.
        ema20 = (
            pd.Series(closes).ewm(span=20, adjust=False).mean().to_numpy()
            if self.also_accept_20ema_support
            else None
        )
        # Pre-market high for this session_date — Ross uses PM high
        # as resistance/breakout reference. 0.0 = no PM data injected
        # → the check skips (graceful degradation).
        pm_high = float(self.premarket_high_by_date.get(sess_date, 0.0))

        signals: list[BullFlagSignal] = []
        used_pole_starts: set[int] = set()
        # N-th pullback counter — Ross: "1st/2nd work well, 3rd start
        # to be cautious". Increment on every fired signal in this
        # session; reject when count > max_nth_pullback.
        pullback_count = 0
        i = self.pole_lookback  # earliest possible pole_end index

        while i < n - 1:
            # 시간대 컷오프 — 11:30 ET 이후엔 신규 진입 안 함.
            if bar_minutes[i] >= cutoff_minutes:
                break

            # ---- 1. 폴 검출 ----
            # 직전 ``pole_lookback`` 봉 안에서 누적 상승률이
            # ``pole_min_pct`` 이상이고, bullish 봉 수가
            # ``pole_min_green_bars`` 이상일 때 폴 인정.
            window_start = max(0, i - self.pole_lookback + 1)
            window_highs = highs[window_start : i + 1]
            window_lows = lows[window_start : i + 1]

            # 영상 정통: pole_end_price = 윈도우 내 PEAK 가격.
            # ``highs[i]`` 만 쓰면 풀백 중간 봉이 pole_end로 잡혀서
            # 풀백 high를 못 깬 봉이 잘못 진입하는 버그.
            # ALGS 4/16: i=10:52 (high 8.19) 일 때 윈도우 안에 10:51
            # 의 high 8.25가 진짜 폴 peak. pole_end_price=8.25로 두면
            # 10:53/10:55 close가 8.25를 못 깨서 진입 안 되고, 10:56
            # close 8.285만 진짜 돌파로 인정.
            pole_end_offset = int(np.argmax(window_highs))
            pole_end_idx = window_start + pole_end_offset
            pole_end_price = float(window_highs[pole_end_offset])

            pole_start_idx = window_start + int(np.argmin(window_lows))
            pole_start_price = float(lows[pole_start_idx])

            if pole_start_price <= 0:
                i += 1
                continue
            pole_pct = (pole_end_price - pole_start_price) / pole_start_price
            if pole_pct < self.pole_min_pct:
                i += 1
                continue

            green_bars = int(
                np.sum(
                    closes[pole_start_idx : i + 1]
                    > opens[pole_start_idx : i + 1]
                )
            )
            if green_bars < self.pole_min_green_bars:
                i += 1
                continue

            # ---- 1a. Bar-quality + 시간 연속성 (데이터 품질) ----
            # 폴 윈도우 내 무거래 봉(zero-range) / 시간 갭이 있으면 reject.
            # 영상의 시각적 quality 보장 — Ross 화면엔 doji/빈 분봉 없음.
            pole_bars_ranges = highs[pole_start_idx : i + 1] - lows[pole_start_idx : i + 1]
            if np.any(pole_bars_ranges <= self.min_bar_range):
                i += 1
                continue
            # 인접 봉 시간차 체크 (timestamps[k] - timestamps[k-1] <= max_gap)
            pole_gaps_ok = True
            for k in range(pole_start_idx + 1, i + 1):
                dt = (timestamps[k] - timestamps[k - 1]).total_seconds()
                if dt > self.max_bar_gap_seconds:
                    pole_gaps_ok = False
                    break
            if not pole_gaps_ok:
                i += 1
                continue
            # 폴 봉 거래량 비교는 ``enable_volume_profile`` 게이트가 담당
            # (pole_avg vs flag_avg). 영상 정통.

            # ---- 1b. 폴 마지막 봉 topping-tail 체크 (DP4 영상 P1) ----
            # 마지막 폴 봉(= peak 봉, pole_end_idx)이 긴 위꼬리로 끝나면
            # "위에서 거부됨" 패턴 → bull flag 진입 자체를 거부.
            if self.max_pole_topping_tail_ratio < 1.0:
                pole_end_range = highs[pole_end_idx] - lows[pole_end_idx]
                if pole_end_range > 0:
                    body_top = max(opens[pole_end_idx], closes[pole_end_idx])
                    upper_wick = highs[pole_end_idx] - body_top
                    if upper_wick / pole_end_range > self.max_pole_topping_tail_ratio:
                        i += 1
                        continue

            # ---- 2. 풀백 + breakout 검출 ----
            # ``i`` 직후 ``flag_max_bars`` 봉 동안 풀백을 추적,
            # 그 사이 breakout 캔들(close > 직전 캔들 high) 발생 시 진입.
            flag_low = pole_end_price
            flag_low_idx = i
            fired = False
            # 영상 정통: 돌파 봉 *전*에 최소 1개 pullback 봉이 있어야 함.
            # "1-3 candles of pullback" — 같은 봉이 풀백+돌파 동시에 발생하면
            # 사실상 0-bar flag → bull flag 아님. 사용자 ALGS 10:53 케이스에서
            # 이 룰 누락으로 같은 봉에 진입하는 버그.
            has_prior_pullback = False
            for k in range(1, self.flag_max_bars + 1):
                j = i + k
                if j >= n:
                    break

                # ---- 2a. Bar-quality + 시간 연속성 (flag/breakout 봉) ----
                # 무거래 봉 / 폴-풀백 시간 갭이 있으면 패턴 무효화.
                if (highs[j] - lows[j]) <= self.min_bar_range:
                    break  # zero-range 봉이 풀백에 끼면 reject
                if (
                    timestamps[j] - timestamps[j - 1]
                ).total_seconds() > self.max_bar_gap_seconds:
                    break  # 데이터 갭 → 풀백 무효

                # 50% retrace 안쪽인지 확인 (현재까지의 flag_low 기준)
                pole_height = pole_end_price - pole_start_price
                if pole_height <= 0:
                    break
                retrace_check_low = min(flag_low, lows[j])
                retrace = (pole_end_price - retrace_check_low) / pole_height
                if retrace > self.flag_max_retrace:
                    break  # 폴 무효화 — 다음 폴 후보로

                # ── Breakout 체크 (영상 정통) ──
                # 1. high > pole_end (영상 "first candle to make a new high")
                # 2. close > pole_end (확정)
                # 3. close > open (양봉 — 돌파 commitment)
                # 4. 직전에 적어도 1개 pullback 봉 있어야 (0-bar flag 방지)
                is_green = closes[j] > opens[j]
                is_breakout_candidate = (
                    highs[j] > pole_end_price
                    and closes[j] > pole_end_price
                    and is_green
                )
                if is_breakout_candidate and has_prior_pullback:
                    # ---- Volume profile gate (DP4 영상 P0) ----
                    # 1. 폴 평균 거래량 기준 계산 (pole_start_idx ~ i,
                    #    green 봉만)
                    # 2. 풀백 평균 거래량 / 폴 평균 ≤ pullback_volume_ratio
                    # 3. 돌파봉 거래량 / 폴 평균 ≥ breakout_volume_ratio
                    if self.enable_volume_profile and j > i:
                        pole_green_mask = (
                            closes[pole_start_idx : i + 1]
                            > opens[pole_start_idx : i + 1]
                        )
                        pole_vols = volumes[pole_start_idx : i + 1][pole_green_mask]
                        flag_vols = volumes[i + 1 : j]  # 풀백 봉들
                        if len(pole_vols) > 0:
                            pole_vol_avg = float(np.mean(pole_vols))
                            if pole_vol_avg > 0:
                                # 풀백 거래량 게이트 — 풀백 평균 / 폴 평균 ≤ 0.7
                                # ("light volume on red candles")
                                if len(flag_vols) > 0:
                                    flag_vol_avg = float(np.mean(flag_vols))
                                    if (
                                        flag_vol_avg / pole_vol_avg
                                        > self.pullback_volume_ratio
                                    ):
                                        break  # 풀백 거래량이 너무 큼 → fake bull flag
                                    # 돌파봉 거래량 게이트 — 돌파봉 / 풀백 평균 ≥ 1.0
                                    # ("fresh round of buyers"). 폴 평균이 아닌
                                    # **풀백 평균** 기준이라 textbook entry 살림.
                                    if (
                                        flag_vol_avg > 0
                                        and volumes[j] / flag_vol_avg
                                        < self.breakout_volume_ratio
                                    ):
                                        break  # 돌파 거래량이 풀백 수준 이하 → 약한 돌파

                    # 진입 = pole_end_high (= 폴 윈도우 최고점, idealized).
                    # 손절 = flag_low (= 풀백 저점, idealized).
                    # tick buffer 안 씀 — 영상은 정확한 textbook 가격으로 표시.
                    stop = float(flag_low)
                    entry_price = float(pole_end_price)
                    if entry_price - stop <= 0:
                        break

                    # ---- Stop distance cap (CSV 분석: >5% 시 25% win) ----
                    risk_pct = (entry_price - stop) / entry_price
                    if risk_pct > self.max_stop_distance_pct:
                        break  # wide stop = poor entry vs support

                    # ---- 9 EMA (또는 20 EMA fallback) support check ----
                    # Ross: "9 EMA on every TF". 풀백 저점이 9 EMA
                    # tolerance 안이면 통과. 9 EMA 못 받쳐도
                    # ``also_accept_20ema_support=True`` 일 때 20 EMA
                    # tolerance 안이면 통과 (Brett Burgett 룰 + Ross도
                    # 영상에서 longer-term EMA 언급).
                    if self.require_9ema_support and flag_low_idx < len(ema9):
                        ema_pass = False
                        ema9_at_low = float(ema9[flag_low_idx])
                        if ema9_at_low > 0:
                            ema9_dev = abs(flag_low - ema9_at_low) / ema9_at_low
                            if ema9_dev <= self.ema9_tolerance_pct:
                                ema_pass = True
                        if (
                            not ema_pass
                            and ema20 is not None
                            and flag_low_idx < len(ema20)
                        ):
                            ema20_at_low = float(ema20[flag_low_idx])
                            if ema20_at_low > 0:
                                ema20_dev = (
                                    abs(flag_low - ema20_at_low) / ema20_at_low
                                )
                                # 20 EMA tolerance — 9 EMA 보다 살짝
                                # 넓게 (20 EMA가 본질적으로 변동성이
                                # 더 큰 distance라 같은 % 이내라도 OK).
                                if ema20_dev <= self.ema9_tolerance_pct * 1.5:
                                    ema_pass = True
                        if not ema_pass:
                            break  # 9도 20도 받쳐주지 않음

                    # ---- Pre-market high check ----
                    # 진입가가 PM high보다 위여야 함 (Ross: "breaking PM
                    # high = confirmation"). pm_high=0 이면 데이터 없음
                    # → 게이트 skip.
                    if pm_high > 0 and entry_price <= pm_high:
                        break

                    # ---- N-th pullback counter (Ross: 1st/2nd OK) ----
                    if pullback_count >= self.max_nth_pullback:
                        break  # 3번째 이후 풀백 reject

                    # HoD = 진입 시점까지의 일중 최고가
                    hod = float(np.max(highs[: j + 1]))
                    risk_per_share = entry_price - stop
                    r_to_hod = (
                        (hod - entry_price) / risk_per_share
                        if risk_per_share > 0
                        else 0.0
                    )

                    if pole_start_idx not in used_pole_starts:
                        signals.append(
                            BullFlagSignal(
                                session_date=sess_date,
                                direction="long",
                                pole_start_ts=timestamps[pole_start_idx],
                                pole_start_price=pole_start_price,
                                pole_end_ts=timestamps[pole_end_idx],
                                pole_end_price=pole_end_price,
                                flag_low=flag_low,
                                flag_low_ts=timestamps[flag_low_idx],
                                entry_ts=timestamps[j],
                                entry_price=entry_price,
                                stop_loss=stop,
                                hod_at_entry=hod,
                                r_multiple_to_hod=r_to_hod,
                                gap_pct=session_meta["gap_pct"],
                                rvol=session_meta["rvol"],
                                open_price=session_meta["open_price"],
                                float_shares=self.float_shares,
                            )
                        )
                        used_pole_starts.add(pole_start_idx)
                    # Increment per-session pullback counter so the
                    # next signal in this session is checked against
                    # ``max_nth_pullback`` even if its candidate breaks
                    # off a different pole.
                    pullback_count += 1
                    fired = True
                    i = j  # 이 breakout 다음부터 새 폴 검색
                    break

                # ── 이번 봉이 breakout 발사 안 됐으면, 풀백 상태 업데이트 ──
                # 같은 봉이 풀백 + breakout 후보를 다 가지더라도 has_prior_pullback
                # 게이트로 막혔을 것. 이 봉의 low를 풀백으로 등록해서 다음 봉이
                # breakout 후보일 때 가능하게 함.
                if lows[j] < pole_end_price:
                    has_prior_pullback = True
                    if lows[j] < flag_low:
                        flag_low = float(lows[j])
                        flag_low_idx = j

            if not fired:
                i += 1

        return signals
