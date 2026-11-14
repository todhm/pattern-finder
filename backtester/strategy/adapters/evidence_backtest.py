"""근거중심매매(온휘) 규칙을 날마다 다시 판정하는 포트폴리오 시뮬레이터.

규칙(설계서 §1.4·§1.5·§2.10, 2026-10-29 책 재대조판 = SimRules 기본값):
  지수  QQQ 주봉을 먼저 본다 — 하락채널+정배열(최악 4분면)이면 그 주 신규 진입 없음 (index_gate="worst"; "ema20" 은 주봉 20주선 아래도 금지).
  진입  주봉 5대 매매 근거 '예' ≥ 4 (금요일 종가), 주봉 하락 다이버전스 2회(고점 판별) 종목 제외 → 그 주 안에 일봉 캔들 상승 신호(③, 20MA 위 마감)
        가 나온 날 종가로 확정 → 다음날 시가 1차(자산의 10%). 추격 금지: 확정일 종가가 일봉 20MA 보다 5% 이상 위면 기다린다.
  2차  진입가 아래 0.618 이상 깊은 피보나치 레벨(0.618→0.786→1.0)에 **지정가**로 걸어 두고 닿는 날 그 가격에 나머지 10% (기간 제한 없음).
  손절 2차 레벨 아래 레벨(근거선)을 종가(몸통)로 이탈 → 다음날 시가 전량. 꼬리 이탈은 무시.
  익절 1:1 = 추세기반 3점 확장(상승 시작 A → 고점 B → 되돌림 저점 C, 목표 C+(B−A)) 종가 도달 → 다음날 시가 75% 매도, 손절선을 진입가로(본전 스탑).
        하락 리버스 매수는 최소 기술적 반등 0.618 레벨. 잔여는 1.618 확장(리버스는 0.382 레벨) 도달 시 청산.
  매도 계획에 이동평균선은 쓰지 않는다(4요소) — 주봉 20주선 이탈 청산·최대 보유일은 기본 꺼짐.
  동시 보유 ≤ 5, 포지션당 자본의 20%(1차 10% + 2차 10%).
매일 종가로 보유 종목을 다시 보고, 체결은 지정가 2차를 빼면 전부 다음날 시가. 수수료 편도 0.15%, 세후 22% 는 결과에서 따로 계산.

legacy_rules() = 2026-10-28 첫 실행판(1:1 = 직전 고점, 2차 = 첫 아래 레벨 ±1.5% 터치 다음날 시가, 지수 게이트 없음, 20주선 청산·120일 보유 제한).

플라세보: 같은 날 같은 개수만큼 무작위 종목을 같은 규칙으로 사는 'sel'(종목 선택 효과), 매주 빈 자리를 무작위로 채우는
'full'(시장 타이밍 효과) 두 가지. 순수 pandas/numpy, 데이터 포트 없음 (CLI 가 조립).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from strategy.adapters.evidence_checklist import (
    EvidenceParams, TimeframeReport, add_indicators, candle_state, channel_state, divergence_state,
    fib_state, five_checks, fvg_list, ma_state, supply_state, supply_zones, swing_points, to_weekly, Check,
)

RETRACE_ORDER = (0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.272, 1.382, 1.618)


@dataclass(frozen=True)
class SimRules:
    horizon_days: int = 130
    max_positions: int = 5
    weekly_min_yes: int = 4
    confirm_window: int = 5          # 주봉 신호 뒤 일봉 ③ 확인을 기다리는 거래일
    require_daily_candle: bool = True
    second_window: int = 0           # 2차 매수 대기 거래일 (0 = 제한 없음)
    second_min_ratio: float = 0.618  # 2차 레벨은 이 비율 이상 깊은 레벨만 (0 = 진입가 아래 첫 레벨)
    second_fill: str = "limit"       # limit: 터치일 min(시가, 레벨) 체결 / next_open: 터치 다음날 시가
    second_tol: float = 0.0          # 터치 허용치 (legacy 0.015)
    target_mode: str = "extension"   # extension: 3점 추세기반 1:1·1.618 / swing_high: 직전 고점·ext1.272
    index_gate: str = "worst"        # none / worst(지수 주봉 최악 4분면) / ema20(최악 또는 주봉 20주선 아래)
    max_ext_above_ema20: float | None = 0.05   # 확정일 종가가 일봉 EMA20 보다 이만큼 위면 추격으로 보고 대기
    exclude_double_bear_div: bool = True       # 주봉 하락 다이버전스 2회 = 고점 → 제외
    candle_low_stop: bool = False    # 확정 캔들 저점을 손절선으로 끌어올림 (책 "저점 모닝스타 진입이면 그 저점")
    tp1_frac: float = 0.75
    final_ext: float = 1.618
    max_hold_days: int = 0           # 0 = 제한 없음
    weekly_trend_exit: bool = False  # 책의 매도 계획은 MA 제외 4요소
    # ---- 베팅(사이징·청산) 실험 손잡이 (2026-11-13, 기본값 = 책 그대로)
    second_buy: bool = True          # False 면 2차 없이 1차에 position_frac 전부
    breakeven_on_tp1: bool = True    # 1:1 도달 시 손절을 본전으로 (tp1_frac 0 이면 매도 없이 스탑만 올림)
    use_final: bool = True           # False 면 최종 목표 청산 없음(손절·추세 이탈·구간 종료까지 보유)
    stop_mode: str = "daily"         # daily: 매일 종가 근거선 이탈 / weekly: 주 마지막 거래일 종가만 / none: 가격 손절 없음
    trend_exit_any: bool = False     # 금요일 종가 < 주봉 20주선이면 1:1 여부와 무관하게 청산
    exclude_worst_quadrant: bool = True
    fee: float = 0.0015              # 편도 수수료+슬리피지
    position_frac: float = 0.2
    daily_bars: int = 400
    weekly_bars: int = 260


def legacy_rules(**over) -> SimRules:
    """2026-10-28 첫 실행판 규칙."""
    base = dict(second_window=15, second_min_ratio=0.0, second_fill="next_open", second_tol=0.015, target_mode="swing_high",
                index_gate="none", max_ext_above_ema20=None, exclude_double_bear_div=False, candle_low_stop=False,
                final_ext=1.272, max_hold_days=120, weekly_trend_exit=True)
    base.update(over)
    return SimRules(**base)


@dataclass
class Trade:
    ticker: str
    signal: str
    entry: str
    entry_px: float
    qty: float
    stop: float
    second_lvl: float | None
    target1: float
    final_target: float
    exits: list = field(default_factory=list)   # (date, px, qty, reason)
    second_done: bool = False
    second_px: float | None = None
    tp1_done: bool = False
    cost: float = 0.0
    proceeds: float = 0.0

    @property
    def open_qty(self) -> float:
        return self.qty - sum(q for _, _, q, _ in self.exits)

    @property
    def closed(self) -> bool:
        return self.open_qty <= 1e-12

    def pnl(self) -> float:
        return self.proceeds - self.cost


# ---------------------------------------------------------------------------
# 전처리 + 빠른 리포트 (add_indicators 를 한 번만)
# ---------------------------------------------------------------------------

def precompute(prices: dict[str, pd.DataFrame], p: EvidenceParams = EvidenceParams()) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out = {}
    for t, df in prices.items():
        if len(df) < 300:
            continue
        d = add_indicators(df, p)
        w = add_indicators(to_weekly(df), p)
        out[t] = (d, w)
    return out


def fast_report(label: str, df: pd.DataFrame, weekly_ma, p: EvidenceParams, lr: int) -> TimeframeReport:
    sw = swing_points(df, lr)
    ma = ma_state(df, p)
    ch = channel_state(df, sw, ma.alignment, p)
    fb = fib_state(df, sw, p)
    boxes = supply_zones(df, p)
    sup = Check(*supply_state(df, boxes, p))
    fv = fvg_list(df, p.fvg_lookback)
    cs = candle_state(df, p)
    dv = divergence_state(df, sw, p)
    rep = TimeframeReport(label, df, sw, ma, ch, fb, boxes, sup, fv, cs, dv)
    rep.five = five_checks(rep, weekly_ma or ma, p)
    return rep


def reports_at(d: pd.DataFrame, w: pd.DataFrame, as_of: pd.Timestamp, p: EvidenceParams, r: SimRules
               ) -> tuple[TimeframeReport, TimeframeReport] | None:
    dd = d.loc[:as_of].tail(r.daily_bars)
    ww = w.loc[:as_of].tail(r.weekly_bars)       # 라벨(금요일) ≤ as_of 인 완성 주봉만
    if len(dd) < 200 or len(ww) < 60 or dd.index[-1] != as_of:
        return None
    wr = fast_report("주봉", ww, None, p, max(3, p.pivot_lr // 2))
    dr = fast_report("일봉", dd, wr.ma, p, p.pivot_lr)
    return wr, dr


# ---------------------------------------------------------------------------
# 레벨 계산
# ---------------------------------------------------------------------------

def entry_levels(wr: TimeframeReport, entry_px: float, rules: SimRules = SimRules(), pullback_low: float | None = None
                 ) -> tuple[float, float | None, float, float] | None:
    """(손절, 2차 레벨, 1:1 목표, 최종 목표).

    2차 = 진입가 아래 피보나치 레벨 중 비율 ≥ second_min_ratio 인 첫 번째(책: 0.618 이하·채널 하단), 손절 = 그 아래 레벨(근거선).
    1:1 = 상승 되돌림이면 3점 추세기반 확장 C+(B−A) (A=상승 시작 저점, B=고점, C=되돌림 저점), 최종 = C+final_ext·(B−A).
         하락 리버스면 최소 기술적 반등 0.618 레벨, 최종 0.382 레벨.  target_mode="swing_high" 는 첫 실행판(직전 고점·ext1.272).
    """
    fb = wr.fib
    if fb is None:
        return None
    lv = [(r, fb.levels[r]) for r in RETRACE_ORDER if r in fb.levels]
    below = [(r, v) for r, v in lv if v < entry_px * 0.995]
    if not below:
        return None
    second = None
    stop = None
    for k, (r, v) in enumerate(below):
        if r >= rules.second_min_ratio:
            second = v
            stop = below[k + 1][1] if k + 1 < len(below) else None
            break
    if second is None or stop is None:
        # 충분히 깊은 레벨이 없으면 2차 없이 가장 가까운 아래 레벨을 근거선으로
        second, stop = None, below[0][1]
    if stop < entry_px * 0.80:          # 근거선이 너무 멀면(−20% 초과) 책의 "−30% 노출" 구간 → 제외
        return None
    reverse = fb.kind.startswith("하락")
    if rules.target_mode == "extension" and not reverse:
        a, b = fb.lo, fb.hi
        c = min(entry_px, pullback_low) if pullback_low is not None else entry_px   # C = 지지가 나온 곳(되돌림 저점)
        target1 = c + (b - a)
        final = c + rules.final_ext * (b - a)
    elif reverse:
        target1 = fb.levels.get(0.618, fb.hi)
        final = fb.levels.get(0.382, target1 * 1.08)
    else:                                # swing_high (legacy)
        target1 = fb.hi
        final = fb.levels.get("ext1.272", target1 * 1.1)
    if target1 < entry_px * 1.02:
        ext = fb.levels.get("ext1.272")
        if ext and ext > entry_px * 1.02:
            target1 = ext
        else:
            return None
    if final <= target1:
        final = target1 * 1.08
    return float(stop), (float(second) if second else None), float(target1), float(final)


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    start: str
    end: str
    equity: pd.Series
    trades: list[Trade]
    signals: dict[str, list[str]]        # 금요일 → 후보
    stats: dict
    cash: pd.Series | None = None        # 일별 현금 비중(자산 대비) — 남는 현금 운용 대조용


def index_gate_open(index_data, day: pd.Timestamp, p: EvidenceParams, rules: SimRules) -> bool:
    """지수를 먼저 본다(14장): 주봉 최악 4분면(하락채널+정배열)이면 신규 진입 없음. ema20 모드는 주봉 20주선 아래도 금지."""
    if rules.index_gate == "none" or index_data is None:
        return True
    d, w = index_data
    rep = reports_at(d, w, day, p, rules)
    if rep is None:
        return True
    wr, _ = rep
    if "최악" in wr.channel.quadrant:
        return False
    if rules.index_gate == "ema20" and not wr.ma.above_ema20:
        return False
    return True


def _fridays(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return [d for d in idx if d.weekday() == 4]


def run_sim(data: dict[str, tuple[pd.DataFrame, pd.DataFrame]], calendar: pd.DatetimeIndex, start: pd.Timestamp,
            rules: SimRules = SimRules(), p: EvidenceParams = EvidenceParams(),
            picker: Callable[[pd.Timestamp, list[str], int], list[str]] | None = None,
            seed: int | None = None, screen_cache: dict | None = None,
            index_data: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> SimResult:
    """calendar = 거래일 인덱스(QQQ). picker 가 있으면 후보 대신 picker(금요일, 후보, 빈 자리 수) 가 돌려주는 종목을 산다(플라세보).
    index_data = (지수 일봉, 지수 주봉) precompute 결과 — rules.index_gate 가 none 이 아니면 금요일마다 지수 주봉을 먼저 본다."""
    rng = np.random.default_rng(seed)
    days = calendar[calendar >= start][: rules.horizon_days + 1]
    if len(days) < 30:
        raise ValueError("구간이 너무 짧음")
    cash = 1.0
    positions: dict[str, Trade] = {}
    pending_buy: list[tuple[str, float, Trade | None, dict]] = []   # (ticker, 금액, 기존 트레이드(2차) or None, 레벨)
    pending_sell: list[tuple[str, float, str]] = []                   # (ticker, 수량, 사유)
    watch: dict[str, tuple[pd.Timestamp, TimeframeReport]] = {}      # 주봉 신호 뒤 일봉 확인 대기
    equity = []
    cash_hist = []
    signals: dict[str, list[str]] = {}
    trades: list[Trade] = []
    tickers = list(data)

    def px(t: str, day: pd.Timestamp, col: str) -> float | None:
        d = data[t][0]
        if day in d.index:
            return float(d.at[day, col])
        return None

    for i, day in enumerate(days):
        # ---- 1) 전날 결정 체결 (오늘 시가)
        for t, qty, reason in pending_sell:
            tr = positions.get(t)
            o = px(t, day, "Open")
            if tr is None or o is None:
                continue
            qty = min(qty, tr.open_qty)
            tr.exits.append((day.date().isoformat(), o, qty, reason))
            tr.proceeds += qty * o * (1 - rules.fee)
            cash += qty * o * (1 - rules.fee)
            if tr.closed:
                del positions[t]
        pending_sell = []
        for t, amt, tr, lv in pending_buy:
            o = px(t, day, "Open")
            if o is None or amt <= 0 or cash < amt:
                continue
            qty = amt * (1 - rules.fee) / o
            cash -= amt
            if tr is None:
                if len(positions) >= rules.max_positions or t in positions:
                    cash += amt
                    continue
                tr = Trade(t, lv["signal"], day.date().isoformat(), o, qty, lv["stop"], lv["second"], lv["target1"], lv["final"], cost=amt)
                positions[t] = tr
                trades.append(tr)
            else:
                tr.entry_px = (tr.entry_px * tr.qty + o * qty) / (tr.qty + qty)
                tr.qty += qty
                tr.cost += amt
                tr.second_done = True
                tr.second_px = o
        pending_buy = []

        # ---- 2) 오늘 종가 기준 보유 종목 재판정
        for t, tr in list(positions.items()):
            c = px(t, day, "Close")
            if c is None:
                continue
            held = (day - pd.Timestamp(tr.entry)).days
            # 2차 지정가: 레벨에 닿은 날 min(시가, 레벨) 로 체결 (장중 체결이므로 손절 판정보다 먼저)
            if (rules.second_fill == "limit" and (not tr.second_done) and tr.second_lvl and day.date().isoformat() != tr.entry
                    and (rules.second_window <= 0 or held <= rules.second_window)):
                lo = px(t, day, "Low"); o = px(t, day, "Open")
                if lo is not None and lo <= tr.second_lvl * (1 + rules.second_tol):
                    fill = min(o, tr.second_lvl) if o is not None else tr.second_lvl
                    amt = tr.cost
                    if cash >= amt and fill > 0:
                        q2 = amt * (1 - rules.fee) / fill
                        cash -= amt
                        tr.entry_px = (tr.entry_px * tr.qty + fill * q2) / (tr.qty + q2)
                        tr.qty += q2; tr.cost += amt; tr.second_done = True; tr.second_px = fill
            week_end = i + 1 >= len(days) or days[i + 1].isocalendar()[:2] != day.isocalendar()[:2]
            stop_on = rules.stop_mode == "daily" or (rules.stop_mode == "weekly" and week_end)
            if stop_on and c <= tr.stop:
                pending_sell.append((t, tr.open_qty, "손절(근거선 몸통 이탈)" if not tr.tp1_done else "본전 스탑"))
                continue
            if not tr.tp1_done and c >= tr.target1:
                tr.tp1_done = True
                if rules.breakeven_on_tp1:
                    tr.stop = max(tr.stop, tr.entry_px)
                if rules.tp1_frac > 0:
                    pending_sell.append((t, tr.open_qty * rules.tp1_frac, "1:1 익절 75%"))
                continue
            if rules.use_final and tr.tp1_done and c >= tr.final_target:
                pending_sell.append((t, tr.open_qty, f"최종 목표 {rules.final_ext}"))
                continue
            if rules.max_hold_days > 0 and held >= rules.max_hold_days:
                pending_sell.append((t, tr.open_qty, "최대 보유일"))
                continue
            if (rules.weekly_trend_exit and tr.tp1_done or rules.trend_exit_any) and day.weekday() == 4:
                w = data[t][1]
                if day in w.index and c < float(w.at[day, "EMA20"]):
                    pending_sell.append((t, tr.open_qty, "주봉 20주선 이탈"))
                    continue
            if (rules.second_fill != "limit" and (not tr.second_done) and tr.second_lvl
                    and (rules.second_window <= 0 or held <= rules.second_window)):
                lo = px(t, day, "Low")
                if lo is not None and lo <= tr.second_lvl * (1 + rules.second_tol) and c > tr.stop:
                    pending_buy.append((t, tr.cost, tr, {}))

        # ---- 3) 일봉 확인 대기 종목 (주봉 4/5 뒤 5일 안에 ③)
        for t, (sig_day, wr) in list(watch.items()):
            if (day - sig_day).days > rules.confirm_window * 1.6 or t in positions:
                del watch[t]
                continue
            if len(positions) + sum(1 for b in pending_buy if b[2] is None) >= rules.max_positions:
                continue
            rep = reports_at(data[t][0], data[t][1], day, p, rules)
            if rep is None:
                continue
            _, dr = rep
            ok = (not rules.require_daily_candle) or (dr.candle.bullish and dr.candle.above_ema20)
            if ok:
                c = px(t, day, "Close")
                if rules.max_ext_above_ema20 is not None:
                    e20 = float(data[t][0].at[day, "EMA20"])
                    if c > e20 * (1 + rules.max_ext_above_ema20):
                        continue                       # 추격 금지 — 다음 근거 자리(창 안)까지 기다린다
                pb_low = float(wr.df["Low"].iloc[int(wr.fib.hi_pos):].min()) if wr.fib is not None else None
                lv = entry_levels(wr, c, rules, pb_low)
                del watch[t]
                if lv is None:
                    continue
                stop, second, t1, fin = lv
                if not rules.second_buy:
                    second = None
                if rules.candle_low_stop:
                    cl = px(t, day, "Low")
                    if cl is not None and cl * 0.995 > stop:
                        stop = cl * 0.995
                eq_now = cash + sum(positions[q].open_qty * (px(q, day, "Close") or 0) for q in positions)
                amt = eq_now * rules.position_frac * (0.5 if second else 1.0)
                pending_buy.append((t, amt, None, {"signal": sig_day.date().isoformat(), "stop": stop, "second": second, "target1": t1, "final": fin}))

        # ---- 4) 금요일: 주봉 스크리닝
        if day.weekday() == 4 and i < len(days) - 10:
            free = rules.max_positions - len(positions) - len(watch)
            gate_ok = index_gate_open(index_data, day, p, rules)
            # 전 종목 주봉 스크리닝은 보유 여부와 무관하므로 날짜별로 캐시해 플라세보 런과 공유한다
            if screen_cache is not None and day in screen_cache:
                all_c = screen_cache[day]
            else:
                all_c = []
                for t in tickers:
                    rep = reports_at(data[t][0], data[t][1], day, p, rules)
                    if rep is None:
                        continue
                    wr, _ = rep
                    if wr.yes_count >= rules.weekly_min_yes:
                        all_c.append((t, "최악" in wr.channel.quadrant, wr.div.bearish_count >= 2))
                if screen_cache is not None:
                    screen_cache[day] = all_c
            cands = [t for t, worst, bear2 in all_c
                     if not (rules.exclude_worst_quadrant and worst) and not (rules.exclude_double_bear_div and bear2)
                     and t not in positions and t not in watch] if (free > 0 and gate_ok) else []
            signals[day.date().isoformat()] = cands if gate_ok else ["(지수 게이트)"]
            chosen = cands if picker is None else (picker(day, cands, free) if gate_ok else [])
            # 후보가 빈 자리보다 많으면 가중치(자동) 높은 순 — 여기선 주봉 예 개수 동률이라 종가 대비 20MA 거리 작은 순(덜 확장된 것)
            def _ext(t):
                d = data[t][0]
                return abs(float(d.at[day, "Close"]) / float(d.at[day, "EMA20"]) - 1) if day in d.index else 9
            chosen = sorted(chosen, key=_ext)[: max(free, 0)]
            for t in chosen:
                rep = reports_at(data[t][0], data[t][1], day, p, rules)
                if rep:
                    watch[t] = (day, rep[0])

        # ---- 5) 자산 기록
        eq = cash + sum(tr.open_qty * (px(t, day, "Close") or tr.entry_px) for t, tr in positions.items())
        equity.append((day, eq))
        cash_hist.append((day, cash / eq if eq > 0 else 0.0))

    # 종료: 남은 포지션 마지막 종가로 평가 청산
    last = days[-1]
    for t, tr in list(positions.items()):
        c = px(t, last, "Close") or tr.entry_px
        q = tr.open_qty
        tr.exits.append((last.date().isoformat(), c, q, "구간 종료 평가"))
        tr.proceeds += q * c * (1 - rules.fee)
    eq_s = pd.Series(dict(equity))
    stats = summarize(eq_s, trades, rules)
    return SimResult(days[0].date().isoformat(), last.date().isoformat(), eq_s, trades, signals, stats, pd.Series(dict(cash_hist)))


def summarize(eq: pd.Series, trades: list[Trade], rules: SimRules) -> dict:
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    pnls = [tr.pnl() for tr in trades]
    wins = [x for x in pnls if x > 0]
    realized = sum(pnls)
    tax = max(0.0, realized - 0.025) * 0.22          # 자본 1 = 1억 원 가정, 공제 250만 = 0.025
    hold = []
    for tr in trades:
        if tr.exits:
            hold.append((pd.Timestamp(tr.exits[-1][0]) - pd.Timestamp(tr.entry)).days)
    reasons: dict[str, int] = {}
    for tr in trades:
        for _, _, _, rs in tr.exits:
            reasons[rs] = reasons.get(rs, 0) + 1
    exposure = float(np.mean([1.0 if v < 1 else 0.0 for v in eq])) if len(eq) else 0.0
    return {
        "수익률": ret, "세후 수익률": ret - tax, "MDD": mdd, "거래 수": len(trades), "승률": (len(wins) / len(pnls)) if pnls else None,
        "평균 손익(자본 대비)": (float(np.mean(pnls)) if pnls else None), "평균 보유일": (float(np.mean(hold)) if hold else None),
        "청산 사유": reasons, "2차 체결": sum(1 for tr in trades if tr.second_done), "1:1 도달": sum(1 for tr in trades if tr.tp1_done),
    }


# ---------------------------------------------------------------------------
# 레짐 분류 · 플라세보
# ---------------------------------------------------------------------------

def classify_regimes(qqq: pd.DataFrame) -> pd.DataFrame:
    """월말마다 QQQ 로 상승장/하락장/평범 라벨."""
    c = qqq["Close"].astype(float)
    sma200 = c.rolling(200).mean()
    hi252 = c.rolling(252).max()
    df = pd.DataFrame({"close": c, "sma200": sma200, "sma200_prev": sma200.shift(21), "hi252": hi252})
    m = df.groupby([df.index.year, df.index.month]).tail(1).dropna()
    lab = []
    for _, r in m.iterrows():
        if r.close < r.sma200:
            lab.append("하락장")
        elif r.sma200 > r.sma200_prev and r.close >= r.hi252 * 0.95:
            lab.append("상승장")
        else:
            lab.append("평범")
    m = m.assign(regime=lab)
    return m


def pick_dates(regimes: pd.DataFrame, n_each: int, seed: int, lo: str, hi: str) -> list[tuple[str, pd.Timestamp]]:
    rng = np.random.default_rng(seed)
    out = []
    sub = regimes[(regimes.index >= lo) & (regimes.index <= hi)]
    for reg in ("상승장", "평범", "하락장"):
        idx = sub[sub.regime == reg].index
        if len(idx) == 0:
            continue
        for d in rng.choice(idx, size=min(n_each, len(idx)), replace=False):
            out.append((reg, pd.Timestamp(d)))
    return sorted(out, key=lambda x: x[1])


def placebo_sel(rng: np.random.Generator, universe: list[str]):
    """같은 금요일에 후보 개수만큼 무작위 종목 (종목 선택 효과 검증)."""
    def _pick(day, cands, free):
        k = min(len(cands), free)
        if k <= 0:
            return []
        return list(rng.choice(universe, size=k, replace=False))
    return _pick


def placebo_full(rng: np.random.Generator, universe: list[str]):
    """매주 빈 자리를 무작위로 채움 (타이밍 효과 검증)."""
    def _pick(day, cands, free):
        if free <= 0:
            return []
        return list(rng.choice(universe, size=free, replace=False))
    return _pick


def buy_hold(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    c = df["Close"].loc[start:end]
    return float(c.iloc[-1] / c.iloc[0] - 1) if len(c) > 1 else float("nan")
