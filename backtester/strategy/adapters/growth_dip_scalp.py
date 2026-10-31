"""성장주 당일 눌림 +1% 매매 — 그리드 백테스트 엔진 (순수 pandas/numpy, 데이터 포트 없음).

설계서: docs/strategy_notes/성장주_당일눌림_1퍼센트_설계_2026-10.md
  [A] 리스트  미너비니 STAGE 2 성장(evaluate_growth, point-in-time) 통과 종목을 매주 금요일 종가로 고정해 다음 주에 쓴다.
  [B] 트리거  당일 저가가 **당일 시가 −X%** (open, 기본 — "당일 데이터로 당일 떨어진 종목") / 전일 종가 −X% (prev_close) / 20일선 터치 (ma20)
              / 장중 고점 −X% (intraday_high, 분봉 전용). 지수 필터: QQQ 저가가 QQQ 시가(open 모드) 또는 전일 종가 −idx% 이하.
  [C] 진입    일봉 근사: 트리거 가격(갭 하락이면 시가). 분봉: 트리거 봉 다음 봉 시가.
  [D] 청산    hold_mode="day": +tp% 지정가 / −sl% 손절 / 당일 종가(15:55) — 먼저 오는 것.
              hold_mode="swing"(기본, 2026-10-30 사용자 변경): 당일 +tp% 도달이면 그날 청산, 아니면 **보유** →
              이후 일봉 고가 ≥ +tp2%(갭이면 시가) / 다음 실적 발표 n일 전 종가 / 기한(max_hold_days) 종가 / (sl_pct 가 있으면 종가 기준 손절).
      일봉 근사는 장중 순서를 모르므로 세 경계를 같이 낸다. 주의: 당일 **저가**는 트리거 터치 뒤에 찍히므로 손절 판정에 쓸 수 있지만,
      당일 **고가**는 트리거 전에 찍혔을 수 있어 익절 판정에 쓰면 낙관 편향이 생긴다.
        optimistic  고가 ≥ 익절가면 익절(손절보다 우선), 아니면 손절, 아니면 종가        — 상한
        neutral     손절 먼저(저가 ≤ 손절가), 아니면 고가 ≥ 익절가면 익절, 아니면 종가  — 익절 쪽은 여전히 낙관
        pessimistic 손절 먼저, 익절은 **종가 ≥ 익절가**일 때만, 아니면 종가              — 하한
      확정값은 분봉(run_sim 에 intraday_trade_fn)으로만 나온다. 일봉 그리드는 "경계가 비용 벽을 넘는가"만 본다.
  비용 편도 fee_side, 양도세 22%(연 단위, 자본 1 = 1억 가정 → 공제 0.025, 손실 이월 없음).
  [E] 자기 PER 변형(설계서 §2.7, 2026-10-31): trigger="per_own"(저가 PER 이 원래 PER(250일 중앙, per_columns) × (1−X%) 아래로 처음 내려온 날)
              / "per_own_any"(그 아래인 날 전부). hold_mode="per": 당일 청산 없이 종가 PER ≥ 진입 시 원래 PER × (1−per_exit_pct%) 이면 청산("PER 회복"),
              EPS ≤ 0 이면 청산("PER 소멸"), max_hold_days 0 = 기한 없음. Arrays.EPS/PREF 는 CLI 가 per_columns 로 만든 ttm_eps/per_ref 열.
플라세보 A: 같은 날 같은 트리거에 걸린 리스트 밖 종목을 같은 수만큼 무작위.
플라세보 B: 리스트 종목을 트리거 없는 무작위 날에 시가 매수, 같은 청산 규칙.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
import itertools
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalpRules:
    trigger: str = "open"                # open(당일 시가 대비 −X%, 2026-10-30 사용자 변경) | prev_close(전일 종가 대비) | ma20(20일선 터치) | intraday_high(분봉 전용, 장중 고점 대비)
    x_pct: float = 3.0                   # 전일 종가 대비 하락 폭 (prev_close 트리거)
    index_filter: bool = True
    index_drop_pct: float = 0.5          # QQQ 저가가 전일 종가 −이만큼 이하일 때만
    tp_pct: float = 1.0
    sl_pct: float | None = None          # None = 가격 손절 없음 (기본, 2026-10-30: 보유 중 청산은 +tp2 / 실적 직전 / 기한). day 모드면 당일 종가 청산
    fill: str = "neutral"                # optimistic | neutral | pessimistic (일봉 근사 전용)
    position_frac: float = 0.2
    max_positions: int = 5
    fee_side: float = 0.0015
    min_dollar_vol_m: float = 20.0
    min_price: float = 10.0
    trend_filter: str | None = "sma200+20"  # None | sma50 | sma200 | sma200+20 (리스트 고정 시 종가 > 50일선 / 200일선 / 200일선×1.2). 기본 sma200+20 (10/31 필터판)
    growth_mode: str = "minervini"       # minervini: 4개 중 growth_required / any: 2-1 EPS 또는 2-3 매출
    growth_required: int = 3
    eps_min: float = 0.25
    rev_min: float = 0.25
    earnings_exclude_days: int = 1       # 발표일 당일 + 이후 n 거래일 제외
    skip_gap_below: bool = False         # 시가가 이미 트리거 아래(갭 하락)면 건너뜀
    per_filter: str | None = None        # None | industry_low: TTM PER 이 같은 산업군 중앙값 × per_ratio 보다 낮을 때만 (2026-10-30 사용자 추가)
    per_ratio: float = 0.8
    per_min_peers: int = 3               # 산업군 비교에 필요한 최소 종목 수
    hold_mode: str = "swing"             # day: 당일 청산(초기 설계) / swing: 당일 +tp 미도달이면 보유 → +tp2 / 다음 실적 직전 / 기한
    tp2_pct: float = 10.0                # 보유 중 2차 목표 (고가 도달 시 그 가격, 갭 상승이면 시가)
    max_hold_days: int = 60              # 기한(거래일). 0 = 실적 직전까지만
    earnings_exit_days_before: int = 1   # 다음 발표일 n 거래일 전 종가 청산
    # ---- 손실 방지 필터 (2026-10-31, 설계서 §2.6 — 손실 거래 특징에서)
    min_days_to_earnings: int = 20       # 다음 실적까지 거래일이 이보다 적으면 진입 안 함 (CLI 가 emask 에 접음). 기본 20 (10/31 필터판 결과)
    max_vol_ratio: float | None = 1.5    # 거래량/20일 평균이 이보다 크면 진입 안 함 (일봉: 당일 총거래량, 분봉: 트리거 봉까지 누적/경과 비율). 기본 1.5
    close_pos_stop: float | None = None  # 진입일 (종가−저가)/(고가−저가) < 값이면 그날 종가 손절 (swing, 당일 익절 안 됐을 때)
    # ---- 자기 PER 기준 (2026-10-31 사용자 요청, 설계서 §2.7): trigger="per_own"(교차)/"per_own_any", hold_mode="per"
    per_lookback: int = 250              # "원래 PER" = 전날까지 이 거래일 수의 일간 PER 중앙값 (CLI 가 per_ref 열로 만든다, 그리드 불가)
    per_exit_pct: float = 0.0            # hold_mode="per": 종가 PER ≥ 진입 시 원래 PER × (1 − 이 값 %) 이면 종가 청산
    # ---- 데이터판 체크리스트 필터 (설계서 §3-2, 검증 문서 §16 — CLI 가 extra_ok 마스크로 접음, 플라세보 A 에도 같이 적용)
    px_below_med_pct: float | None = None    # 전일 종가 ≤ 250일 가격 중앙값 × (1 − v%) 일 때만 (Q-1 가격 하락형)
    below_sma200_pct: float | None = None    # 전일 종가 ≤ 200일선 × (1 − v%) 일 때만 (Q-2)
    per_ref_max_ratio: float | None = None   # 250일 PER 창 최대 ÷ 원래 PER < v 일 때만 (Q-3 적자 탈출 오염 제외)
    max_last_yoy: float | None = None        # 최근 발표 분기 EPS YoY ≤ v (1.0 = +100%) 일 때만 (Q-4)
    min_days_since_report: int = 0           # 마지막 발표 뒤 거래일 ≥ v 일 때만 (Q-5)


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------

def precompute(prices: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """티커별로 거래일 캘린더에 맞춘 OHLC + 전일 종가·SMA20·SMA50·20일 거래대금."""
    out = {}
    for t, df in prices.items():
        d = df.reindex(calendar)
        c = d["Close"].astype(float)
        d["prev_close"] = c.shift(1)
        d["sma20"] = c.rolling(20).mean()
        d["sma50"] = c.rolling(50).mean()
        d["sma200"] = c.rolling(200).mean()
        d["vol20"] = d["Volume"].astype(float).rolling(20).mean()
        d["dvol20"] = (c * d["Volume"].astype(float)).rolling(20).mean()
        out[t] = d
    return out


def per_columns(df: pd.DataFrame, report_ns: np.ndarray, eps_q: np.ndarray, lookback: int = 250) -> pd.DataFrame:
    """자기 PER 열 (설계서 §2.7). report_ns: 발표일(ns 정수, 오름차순), eps_q: 그 분기 EPS. point-in-time TTM EPS = 발표일 ≤ 그날 최근 4개 합.
    ttm_eps(≤0 이면 NaN), per(종가 ÷ ttm_eps), per_ref(전날까지 lookback 일 PER 중앙값, 최소 절반)."""
    day_ns = df.index.values.astype("datetime64[ns]").astype(np.int64)       # pandas 2 는 인덱스 단위가 ns 가 아닐 수 있다
    k = np.searchsorted(np.asarray(report_ns, dtype=np.int64), day_ns, side="right")
    cs = np.r_[0.0, np.cumsum(eps_q)]
    ttm = np.where(k >= 4, cs[k] - cs[np.maximum(k - 4, 0)], np.nan)
    ttm = np.where(ttm > 0, ttm, np.nan)
    df["ttm_eps"] = ttm
    per = df["Close"].astype(float) / df["ttm_eps"]
    df["per"] = per
    df["per_ref"] = per.rolling(lookback, min_periods=max(10, lookback // 2)).median().shift(1)
    df["per_win_max"] = per.rolling(lookback, min_periods=max(10, lookback // 2)).max().shift(1)
    df["px_med"] = df["Close"].astype(float).rolling(lookback, min_periods=max(10, lookback // 2)).median().shift(1)
    return df


def _trend_ok(row: pd.Series, tf: str) -> bool:
    """sma50 / sma200 / sma200+20 (200일선 × 1.2 위) — 값이 없으면 불통과."""
    if tf.startswith("sma200"):
        base = row.get("sma200"); mult = 1 + float(tf.split("+")[1]) / 100 if "+" in tf else 1.0
    else:
        base = row.get("sma50"); mult = 1.0
    return pd.notna(base) and row["Close"] > base * mult


def weekly_lists(tickers: list[str], calendar: pd.DatetimeIndex, growth_fn: Callable[[str, date], bool],
                 data: dict[str, pd.DataFrame], rules: ScalpRules, start: pd.Timestamp, end: pd.Timestamp,
                 extra_fn: Callable[[str, date], bool] | None = None) -> dict[pd.Timestamp, set[str]]:
    """금요일(또는 그 주 마지막 거래일)마다 리스트를 만들어 다음 주에 쓴다. growth_fn(ticker, as_of) 은 point-in-time 통과 여부.
    extra_fn 은 추가 필터(예: 산업군 대비 저PER) — None 이면 무시."""
    days = calendar[(calendar >= start - pd.Timedelta(days=10)) & (calendar <= end)]
    fri = [d for i, d in enumerate(days) if i + 1 == len(days) or days[i + 1].weekday() < d.weekday()]
    lists: dict[pd.Timestamp, set[str]] = {}
    for f in fri:
        s = set()
        for t in tickers:
            d = data.get(t)
            if d is None or f not in d.index:
                continue
            row = d.loc[f]
            if pd.isna(row.get("Close")) or pd.isna(row.get("dvol20")):
                continue
            if row["dvol20"] < rules.min_dollar_vol_m * 1e6 or row["Close"] < rules.min_price:
                continue
            if rules.trend_filter and not _trend_ok(row, rules.trend_filter):
                continue
            if growth_fn(t, f.date()) and (extra_fn is None or extra_fn(t, f.date())):
                s.add(t)
        lists[f] = s
    return lists


def list_for_day(lists: dict[pd.Timestamp, set[str]], day: pd.Timestamp) -> set[str]:
    keys = [k for k in lists if k < day]
    return lists[max(keys)] if keys else set()


# ---------------------------------------------------------------------------
# 일봉 근사 체결
# ---------------------------------------------------------------------------

def day_trade(row: pd.Series, rules: ScalpRules, forced_entry: float | None = None) -> tuple[float, str] | None:
    """한 종목·하루. (순수익률, 청산 사유) 또는 None(트리거 없음). forced_entry 가 있으면 트리거 없이 그 가격에 진입(플라세보 B)."""
    o, h, l, c, pc = (float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["prev_close"]))
    if any(np.isnan(v) for v in (o, h, l, c, pc)):
        return None
    if forced_entry is not None:
        entry = forced_entry
    else:
        if rules.trigger == "open":
            level = o * (1 - rules.x_pct / 100)
        elif rules.trigger == "prev_close":
            level = pc * (1 - rules.x_pct / 100)
        else:                                       # ma20: 전일 종가는 20일선 위, 오늘 저가가 20일선 터치
            s20 = float(row["sma20"])
            if np.isnan(s20) or pc <= s20:
                return None
            level = s20
        if l > level:
            return None
        if o <= level:
            if rules.skip_gap_below:
                return None
            entry = o
        else:
            entry = level
    tp = entry * (1 + rules.tp_pct / 100)
    sl = entry * (1 - rules.sl_pct / 100) if rules.sl_pct else None
    hit_tp = h >= tp
    hit_sl = sl is not None and l <= sl
    if rules.fill == "optimistic":
        if hit_tp:
            px, why = tp, "익절"
        elif hit_sl:
            px, why = sl, "손절"
        else:
            px, why = c, "종가"
    elif rules.fill == "neutral":
        if hit_sl:
            px, why = sl, "손절"
        elif hit_tp:
            px, why = tp, "익절"
        else:
            px, why = c, "종가"
    else:                                            # pessimistic
        if hit_sl:
            px, why = sl, "손절"
        elif c >= tp:
            px, why = tp, "익절"
        else:
            px, why = c, "종가"
    ret = px / entry - 1 - 2 * rules.fee_side
    return ret, why


def day_trade_intraday(bars: pd.DataFrame, prev_close: float, rules: ScalpRules,
                       min_time: str = "10:00", cutoff_time: str = "14:30", eod_time: str = "15:55",
                       forced_entry: bool = False) -> tuple[float, str] | None:
    """분봉(정규장, tz America/New_York 또는 naive 09:30~16:00)으로 한 종목·하루를 그대로 재생.
    트리거 봉에서 min(봉 시가, 레벨) 로 지정가 체결 → 이후 봉마다 손절(저가) 우선, 익절(고가), 마감 봉 시가 청산.
    forced_entry: 트리거 없이 min_time 첫 봉 시가 매수(플라세보 B)."""
    if bars is None or len(bars) < 10 or np.isnan(prev_close):
        return None
    tm = bars.index.strftime("%H:%M")
    hi_run = bars["High"].cummax()
    entry = None; k0 = None
    for k in range(len(bars)):
        if tm[k] < min_time:
            continue
        if tm[k] > cutoff_time:
            break
        o, l = float(bars["Open"].iloc[k]), float(bars["Low"].iloc[k])
        if forced_entry:
            entry, k0 = o, k
            break
        if rules.trigger == "prev_close":
            level = prev_close * (1 - rules.x_pct / 100)
        elif rules.trigger == "open":
            level = float(bars["Open"].iloc[0]) * (1 - rules.x_pct / 100)
        else:
            level = float(hi_run.iloc[k]) * (1 - rules.x_pct / 100)
        if l <= level:
            if o <= level and rules.skip_gap_below:
                return None
            entry, k0 = min(o, level), k
            break
    if entry is None:
        return None
    tp = entry * (1 + rules.tp_pct / 100)
    sl = entry * (1 - rules.sl_pct / 100) if rules.sl_pct else None
    px, why = None, None
    for k in range(k0 + 1, len(bars)):
        h, l, o = float(bars["High"].iloc[k]), float(bars["Low"].iloc[k]), float(bars["Open"].iloc[k])
        if tm[k] >= eod_time:
            px, why = o, "종가"
            break
        if sl is not None and l <= sl:
            px, why = sl, "손절"
            break
        if h >= tp:
            px, why = tp, "익절"
            break
    if px is None:
        px, why = float(bars["Close"].iloc[-1]), "종가"
    return px / entry - 1 - 2 * rules.fee_side, why


# ---------------------------------------------------------------------------
# 포트폴리오 시뮬레이션
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    equity: pd.Series
    trades: list[dict]
    stats: dict


def _tax_equity(equity: pd.Series, pnl_by_day: pd.Series) -> pd.Series:
    """연말마다 그 해 실현손익(자본 대비)에 22% (공제 0.025, 이월 없음) 를 뺀 세후 곡선."""
    eq = equity.copy()
    tax_total = 0.0
    for y, g in pnl_by_day.groupby(pnl_by_day.index.year):
        realized = float(g.sum())
        tax = max(0.0, realized - 0.025) * 0.22
        last = g.index[-1]
        eq.loc[eq.index >= last] -= tax
        tax_total += tax
    return eq


def run_sim(data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, index_df: pd.DataFrame | None,
            lists: dict[pd.Timestamp, set[str]], rules: ScalpRules, start: pd.Timestamp, end: pd.Timestamp,
            earnings: dict[str, set[pd.Timestamp]] | None = None,
            picker: Callable[[pd.Timestamp, list[str], set[str]], list[str]] | None = None,
            placebo_b: np.random.Generator | None = None,
            intraday: dict[str, dict[date, pd.DataFrame]] | None = None) -> SimResult:
    """매일: 리스트 종목 중 트리거 → 하락 폭 큰 순으로 max_positions 까지 → 같은 날 청산.
    picker(day, triggered_in_list, list) 가 있으면 그 결과를 대신 산다(플라세보 A). placebo_b 가 있으면 트리거 대신
    리스트 종목의 무작위 날(트리거 없는 날)에 시가 매수한다. intraday = {ticker: {date: 분봉}} 이면 일봉 근사 대신 분봉으로 체결."""

    def _trade(t: str, day: pd.Timestamp, forced: bool = False):
        row = data[t].loc[day]
        if intraday is not None:
            bars = intraday.get(t, {}).get(day.date())
            if bars is None:
                return None
            return day_trade_intraday(bars, float(row["prev_close"]), rules, forced_entry=forced)
        return day_trade(row, rules, forced_entry=float(row["Open"]) if forced else None)
    days = calendar[(calendar >= start) & (calendar <= end)]
    eq = 1.0
    equity, pnl_days, trades = [], [], []
    for day in days:
        lst = list_for_day(lists, day)
        idx_ok = True
        if rules.index_filter and index_df is not None and day in index_df.index:
            r = index_df.loc[day]
            ref = r["Open"] if rules.trigger == "open" else r["prev_close"]
            if pd.notna(ref) and pd.notna(r["Low"]):
                idx_ok = float(r["Low"]) <= float(ref) * (1 - rules.index_drop_pct / 100)
            else:
                idx_ok = False
        day_pnl = 0.0
        if idx_ok and lst:
            if placebo_b is not None:
                # 리스트 종목 중 무작위로 뽑아 트리거와 무관하게 시가 매수
                pool = [t for t in lst if t in data and day in data[t].index and not np.isnan(float(data[t].at[day, "Open"]))]
                chosen = list(placebo_b.choice(pool, size=min(rules.max_positions, len(pool)), replace=False)) if pool else []
                for t in chosen:
                    row = data[t].loc[day]
                    if earnings and day in earnings.get(t, ()):
                        continue
                    res = _trade(t, day, forced=True)
                    if res:
                        day_pnl += rules.position_frac * res[0]
                        trades.append({"day": day, "ticker": t, "ret": res[0], "why": res[1]})
            else:
                cands = []
                for t in lst:
                    d = data.get(t)
                    if d is None or day not in d.index:
                        continue
                    if earnings and day in earnings.get(t, ()):
                        continue
                    row = d.loc[day]
                    res = _trade(t, day)
                    if res is not None:
                        depth = float(row["Low"]) / float(row["prev_close"]) - 1
                        cands.append((depth, t, res))
                cands.sort()
                chosen_t = [t for _, t, _ in cands]
                if picker is not None:
                    chosen_t = picker(day, chosen_t, lst)
                    cands = []
                    for t in chosen_t:
                        d = data.get(t)
                        if d is None or day not in d.index:
                            continue
                        res = _trade(t, day)
                        if res is not None:
                            cands.append((0.0, t, res))
                for _, t, res in cands[: rules.max_positions]:
                    day_pnl += rules.position_frac * res[0]
                    trades.append({"day": day, "ticker": t, "ret": res[0], "why": res[1]})
        eq *= 1 + day_pnl
        equity.append((day, eq)); pnl_days.append((day, day_pnl * (eq / (1 + day_pnl))))
    eq_s = pd.Series(dict(equity)); pnl_s = pd.Series(dict(pnl_days))
    after_tax = _tax_equity(eq_s, pnl_s) if len(eq_s) else eq_s
    return SimResult(eq_s, trades, summarize(eq_s, after_tax, trades, days))


def summarize(eq: pd.Series, eq_after_tax: pd.Series, trades: list[dict], days: pd.DatetimeIndex) -> dict:
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    years = max(1e-9, (days[-1] - days[0]).days / 365.25) if len(days) else 1
    total = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    total_at = float(eq_after_tax.iloc[-1] - 1) if len(eq_after_tax) else 0.0
    why = pd.Series([t["why"] for t in trades]).value_counts().to_dict() if trades else {}
    daily = eq.pct_change().dropna()
    return {
        "수익률": total, "세후": total_at,
        "CAGR": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "세후 CAGR": float((1 + total_at) ** (1 / years) - 1) if total_at > -1 else -1.0,
        "MDD": float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0,
        "거래 수": int(len(rets)), "연 거래": float(len(rets) / years),
        "승률": float((rets > 0).mean()) if len(rets) else None,
        "거래당 순손익": float(rets.mean()) if len(rets) else None,
        "거래당 중앙값": float(np.median(rets)) if len(rets) else None,
        "최악 거래": float(rets.min()) if len(rets) else None,
        "하루 최악": float(daily.min()) if len(daily) else 0.0,
        "청산 사유": why,
        "시장 노출(거래일 비율)": float(len({t["day"] for t in trades}) / max(1, len(days))),
    }


# ---------------------------------------------------------------------------
# 플라세보·비교
# ---------------------------------------------------------------------------

def placebo_a_picker(rng: np.random.Generator, data: dict[str, pd.DataFrame], rules: ScalpRules):
    """같은 날 같은 트리거에 걸린 **리스트 밖** 종목을 같은 수만큼 무작위."""
    def _pick(day: pd.Timestamp, triggered_in_list: list[str], lst: set[str]) -> list[str]:
        k = min(len(triggered_in_list), rules.max_positions)
        if k <= 0:
            return []
        pool = []
        for t, d in data.items():
            if t in lst or day not in d.index:
                continue
            row = d.loc[day]
            if pd.isna(row.get("dvol20")) or row["dvol20"] < rules.min_dollar_vol_m * 1e6 or row["Close"] < rules.min_price:
                continue
            if day_trade(row, rules) is not None:
                pool.append(t)
        if not pool:
            return []
        return list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return _pick


def buy_hold(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    c = df["Close"].loc[start:end].dropna()
    return float(c.iloc[-1] / c.iloc[0] - 1) if len(c) > 1 else float("nan")


def list_hold(data: dict[str, pd.DataFrame], lists: dict[pd.Timestamp, set[str]], calendar: pd.DatetimeIndex,
              start: pd.Timestamp, end: pd.Timestamp) -> float:
    """리스트 종목을 동일비중으로 그냥 들고 있었다면 (매일 그 주 리스트의 평균 수익률)."""
    days = calendar[(calendar >= start) & (calendar <= end)]
    eq = 1.0
    for day in days:
        lst = list_for_day(lists, day)
        rs = []
        for t in lst:
            d = data.get(t)
            if d is not None and day in d.index and pd.notna(d.at[day, "prev_close"]) and pd.notna(d.at[day, "Close"]):
                rs.append(float(d.at[day, "Close"]) / float(d.at[day, "prev_close"]) - 1)
        if rs:
            eq *= 1 + float(np.mean(rs))
    return eq - 1


# ---------------------------------------------------------------------------
# 그리드
# ---------------------------------------------------------------------------

def expand_grid(base: ScalpRules, grid: dict[str, list]) -> list[ScalpRules]:
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        over = dict(zip(keys, combo))
        out.append(ScalpRules(**{**asdict(base), **over}))
    return out

# ---------------------------------------------------------------------------
# 벡터화 일봉 시뮬레이션 (그리드용 — run_sim 과 같은 규칙, numpy)
# ---------------------------------------------------------------------------

@dataclass
class Arrays:
    days: pd.DatetimeIndex
    tickers: list[str]
    O: np.ndarray; H: np.ndarray; L: np.ndarray; C: np.ndarray; PC: np.ndarray
    S20: np.ndarray; S50: np.ndarray; DV: np.ndarray
    idx_low: np.ndarray; idx_pc: np.ndarray
    idx_open: np.ndarray | None = None
    V: np.ndarray | None = None; V20: np.ndarray | None = None; S200: np.ndarray | None = None
    EPS: np.ndarray | None = None; PREF: np.ndarray | None = None   # TTM EPS, 원래 PER (per_columns, 설계서 §2.7)
    PWMAX: np.ndarray | None = None; PXMED: np.ndarray | None = None  # PER 창 최대, 250일 가격 중앙 (per_columns)
    LYOY: np.ndarray | None = None; DSR: np.ndarray | None = None      # 최근 분기 EPS YoY, 발표 뒤 거래일 (CLI 열 last_yoy / days_since_report)


def build_arrays(data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, index_df: pd.DataFrame | None,
                 start: pd.Timestamp, end: pd.Timestamp) -> Arrays:
    days = calendar[(calendar >= start) & (calendar <= end)]
    tk = list(data)
    def col(name):
        return np.column_stack([(data[t][name] if name in data[t] else pd.Series(np.nan, index=days)).reindex(days).to_numpy(dtype=float) for t in tk])
    O, H, L, C, PC, S20, S50, DV = (col(n) for n in ("Open", "High", "Low", "Close", "prev_close", "sma20", "sma50", "dvol20"))
    V, V20, S200 = col("Volume"), col("vol20"), col("sma200")
    EPS, PREF = col("ttm_eps"), col("per_ref")
    PWMAX, PXMED, LYOY, DSR = col("per_win_max"), col("px_med"), col("last_yoy"), col("days_since_report")
    if index_df is not None:
        il = index_df["Low"].reindex(days).to_numpy(dtype=float); ip = index_df["prev_close"].reindex(days).to_numpy(dtype=float)
        io_ = index_df["Open"].reindex(days).to_numpy(dtype=float)
    else:
        il = np.full(len(days), np.nan); ip = np.full(len(days), np.nan); io_ = np.full(len(days), np.nan)
    return Arrays(days, tk, O, H, L, C, PC, S20, S50, DV, il, ip, io_, V, V20, S200, EPS, PREF, PWMAX, PXMED, LYOY, DSR)


def list_mask(arr: Arrays, lists: dict[pd.Timestamp, set[str]]) -> np.ndarray:
    pos = {t: j for j, t in enumerate(arr.tickers)}
    m = np.zeros((len(arr.days), len(arr.tickers)), dtype=bool)
    keys = sorted(lists)
    kp = 0
    cur: set[str] = set()
    for i, d in enumerate(arr.days):
        while kp < len(keys) and keys[kp] < d:
            cur = lists[keys[kp]]; kp += 1
        for t in cur:
            j = pos.get(t)
            if j is not None:
                m[i, j] = True
    return m


def earnings_mask(arr: Arrays, earnings: dict[str, set[pd.Timestamp]] | None) -> np.ndarray:
    m = np.zeros((len(arr.days), len(arr.tickers)), dtype=bool)
    if not earnings:
        return m
    di = {d: i for i, d in enumerate(arr.days)}
    for j, t in enumerate(arr.tickers):
        for d in earnings.get(t, ()):
            i = di.get(d)
            if i is not None:
                m[i, j] = True
    return m


def _outcome(entry: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, rules: ScalpRules) -> tuple[np.ndarray, np.ndarray]:
    """진입가 행렬 → (순수익률, 사유 코드 0 종가 / 1 익절 / 2 손절) — day_trade 와 같은 규칙."""
    tp = entry * (1 + rules.tp_pct / 100)
    hit_tp = H >= tp
    if rules.sl_pct:
        sl = entry * (1 - rules.sl_pct / 100); hit_sl = L <= sl
    else:
        sl = np.full_like(entry, np.nan); hit_sl = np.zeros_like(hit_tp)
    if rules.fill == "optimistic":
        px = np.where(hit_tp, tp, np.where(hit_sl, sl, C)); why = np.where(hit_tp, 1, np.where(hit_sl, 2, 0))
    elif rules.fill == "neutral":
        px = np.where(hit_sl, sl, np.where(hit_tp, tp, C)); why = np.where(hit_sl, 2, np.where(hit_tp, 1, 0))
    else:
        ctp = C >= tp
        px = np.where(hit_sl, sl, np.where(ctp, tp, C)); why = np.where(hit_sl, 2, np.where(ctp, 1, 0))
    return px / entry - 1 - 2 * rules.fee_side, why


_WHY = {0: "종가", 1: "익절", 2: "손절"}


def _book(arr: Arrays, trig: np.ndarray, ret: np.ndarray, why: np.ndarray, ret_open: np.ndarray, why_open: np.ndarray,
          depth: np.ndarray, elig: np.ndarray, lmask: np.ndarray, emask: np.ndarray, idx_ok: np.ndarray, liquid: np.ndarray,
          valid: np.ndarray, rules: ScalpRules, mode: str, rng: np.random.Generator | None, day_ok: np.ndarray | None) -> SimResult:
    """일별 선택·장부 — 일봉 근사와 분봉 확정이 같은 선택 규칙을 쓴다. day_ok: 시뮬에 포함할 날(표본 달 등), None = 전부."""
    T = len(arr.days)
    eq = 1.0; equity = np.empty(T); pnl = np.empty(T); trades = []
    for i in range(T):
        sel: list[int] = []
        if day_ok is None or day_ok[i]:
            if mode == "strategy":
                row = np.where(elig[i], depth[i], np.inf)
                order = np.argsort(row)[: rules.max_positions]
                sel = [j for j in order if np.isfinite(row[j])]
            elif mode == "placebo_a":
                k = min(int(elig[i].sum()), rules.max_positions)
                if k > 0:
                    pool = np.flatnonzero(trig[i] & ~lmask[i] & liquid[i] & idx_ok[i])
                    if len(pool):
                        sel = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
            else:  # placebo_b
                if idx_ok[i]:
                    pool = np.flatnonzero(lmask[i] & ~emask[i] & valid[i])
                    if len(pool):
                        sel = list(rng.choice(pool, size=min(rules.max_positions, len(pool)), replace=False))
        r_src, w_src = (ret_open, why_open) if mode == "placebo_b" else (ret, why)
        day_pnl = 0.0
        for j in sel:
            rj = float(r_src[i, j])
            if np.isnan(rj):
                continue
            day_pnl += rules.position_frac * rj
            trades.append({"day": arr.days[i], "ticker": arr.tickers[j], "ret": rj, "why": _WHY[int(w_src[i, j])]})
        pnl[i] = eq * day_pnl
        eq *= 1 + day_pnl
        equity[i] = eq
    eq_s = pd.Series(equity, index=arr.days); pnl_s = pd.Series(pnl, index=arr.days)
    days_used = arr.days if day_ok is None else arr.days[day_ok]
    return SimResult(eq_s, trades, summarize(eq_s, _tax_equity(eq_s, pnl_s), trades, days_used if len(days_used) else arr.days))


def checklist_ok(arr: Arrays, rules: ScalpRules) -> np.ndarray:
    """(T,N) 데이터판 체크리스트 필터(설계서 §3-2). 조건이 None 이면 통과, 필요한 열이 없으면 불통과."""
    ok = np.ones(arr.O.shape, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        if rules.px_below_med_pct is not None:
            ok &= (arr.PXMED is not None) and (arr.PC <= arr.PXMED * (1 - rules.px_below_med_pct / 100))
        if rules.below_sma200_pct is not None:
            ok &= (arr.S200 is not None) and (arr.PC <= arr.S200 * (1 - rules.below_sma200_pct / 100))
        if rules.per_ref_max_ratio is not None:
            ok &= (arr.PWMAX is not None and arr.PREF is not None) and (arr.PWMAX / arr.PREF < rules.per_ref_max_ratio)
        if rules.max_last_yoy is not None:
            ok &= (arr.LYOY is not None) and (arr.LYOY <= rules.max_last_yoy)
        if rules.min_days_since_report > 0:
            ok &= (arr.DSR is not None) and (arr.DSR >= rules.min_days_since_report)
    return ok


def _common_masks(arr: Arrays, rules: ScalpRules):
    O, H, L, C, PC = arr.O, arr.H, arr.L, arr.C, arr.PC
    valid = ~(np.isnan(O) | np.isnan(H) | np.isnan(L) | np.isnan(C) | np.isnan(PC))
    idx_ok = np.ones(len(arr.days), dtype=bool)
    if rules.index_filter:
        ref = arr.idx_open if (rules.trigger == "open" and arr.idx_open is not None) else arr.idx_pc
        idx_ok = (~np.isnan(arr.idx_low)) & (~np.isnan(ref)) & (arr.idx_low <= ref * (1 - rules.index_drop_pct / 100))
    liquid = (~np.isnan(arr.DV)) & (arr.DV >= rules.min_dollar_vol_m * 1e6) & (C >= rules.min_price)
    if rules.max_vol_ratio is not None and arr.V is not None and arr.V20 is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            liquid &= (arr.V / arr.V20) <= rules.max_vol_ratio          # 일봉 근사: 당일 총거래량 (장중엔 모름 → 약간 낙관)
    liquid &= checklist_ok(arr, rules)                                   # §3-2 필터 — 전략·플라세보 A 공통
    return valid, idx_ok, liquid


def per_level(arr: Arrays, rules: ScalpRules) -> np.ndarray:
    """(T,N) 자기 PER 트리거의 가격 레벨 = 원래 PER × (1 − x%) × TTM EPS. EPS·원래 PER 없으면 NaN."""
    if arr.EPS is None or arr.PREF is None:
        return np.full(arr.O.shape, np.nan)
    return arr.PREF * (1 - rules.x_pct / 100) * arr.EPS


def daily_trigger_mask(arr: Arrays, rules: ScalpRules) -> np.ndarray:
    """(T,N) 일봉 근사 트리거(유동성·지수·리스트 무관) — CLI 의 분봉 표본 (종목, 월) 선택용."""
    valid, _, _ = _common_masks(arr, ScalpRules(**{**asdict(rules), "max_vol_ratio": None}))
    return _daily_level(arr, rules, valid)[1]


def _daily_level(arr: Arrays, rules: ScalpRules, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """일봉 근사 트리거 레벨 (T,N) 과 트리거 마스크."""
    O, L, PC = arr.O, arr.L, arr.PC
    if rules.trigger == "open":
        level = O * (1 - rules.x_pct / 100); trig = valid & (L <= level)
    elif rules.trigger == "prev_close":
        level = PC * (1 - rules.x_pct / 100); trig = valid & (L <= level)
    elif rules.trigger == "ma20":
        level = arr.S20; trig = valid & ~np.isnan(level) & (PC > level) & (L <= level)
    elif rules.trigger in ("per_own", "per_own_any"):   # 설계서 §2.7: 레벨 = 원래 PER × (1 − X%) × TTM EPS
        level = per_level(arr, rules)
        trig = valid & ~np.isnan(level) & (L <= level)
        if rules.trigger == "per_own":
            trig &= PC > level                           # 교차: 전일 종가는 레벨 위
    else:                                   # intraday_high 는 일봉으로는 당일 고점 대비 — 순서를 모르므로 open 과 같이 근사
        level = O * (1 - rules.x_pct / 100); trig = valid & (L <= level)
    return level, trig


def run_sim_fast(arr: Arrays, lmask: np.ndarray, emask: np.ndarray, rules: ScalpRules,
                 mode: str = "strategy", rng: np.random.Generator | None = None, day_ok: np.ndarray | None = None) -> SimResult:
    """일봉 근사. mode: strategy / placebo_a (리스트 밖 같은 트리거 무작위) / placebo_b (리스트 종목 무작위 날 시가)."""
    O, H, L, C, PC = arr.O, arr.H, arr.L, arr.C, arr.PC
    valid, idx_ok, liquid = _common_masks(arr, rules)
    level, trig = _daily_level(arr, rules, valid)
    gap = O <= level
    if rules.skip_gap_below:
        trig &= ~gap
    entry = np.where(gap, O, level)
    ret, why = _outcome(entry, H, L, C, rules)
    ret_open, why_open = _outcome(O, H, L, C, rules)
    depth = np.where(trig, L / PC - 1, np.inf)
    elig = trig & lmask & ~emask & idx_ok[:, None] & liquid
    return _book(arr, trig, ret, why, ret_open, why_open, depth, elig, lmask, emask, idx_ok, liquid, valid, rules, mode, rng, day_ok)


# ---------------------------------------------------------------------------
# 분봉 확정 (벡터화) — bars3d: (일 × 종목 × 봉) Open/High/Low, 정규장 고정 그리드
# ---------------------------------------------------------------------------

def bar_grid(interval: str = "5m") -> list[str]:
    step = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}[interval]
    out, t = [], 9 * 60 + 30
    while t < 16 * 60:
        out.append(f"{t // 60:02d}:{t % 60:02d}"); t += step
    return out


@dataclass
class Bars3D:
    grid: list[str]
    days_idx: np.ndarray                                   # (Td,) 분봉이 하나라도 있는 날의 arr.days 인덱스
    O: np.ndarray; H: np.ndarray; L: np.ndarray            # (Td, N, B) float32, NaN = 봉 없음
    has: np.ndarray                                        # (T, N) bool — 그 날 분봉이 있는가 (전체 날 기준)
    dropped: int = 0                                       # 일봉과 가격 기준이 어긋나 버린 (일, 종목) 수
    V: np.ndarray | None = None                            # (Td, N, B) 봉 거래량 (max_vol_ratio 용, 없으면 None)


def build_bars3d(arr: Arrays, intraday: dict[str, dict[date, pd.DataFrame]], interval: str = "5m",
                 max_basis_dev: float = 0.05) -> Bars3D:
    """분봉을 (분봉 있는 날, 종목, 봉) 배열로 — 표본 달만 있을 때 전체 날로 잡으면 2,000 종목에 5GB 가 넘어 날을 압축한다.
    **가격 기준 검증**: 그 날 분봉의 첫 시가·최고가가 일봉 시가·고가와 ``max_basis_dev`` 이상 어긋나면 버린다(``has=False``).
    티커가 재사용된 종목(GEN=옛 SYMC, WTW=옛 WLTW 등)은 Alpha Vantage 분봉이 다른 회사 가격으로 와서 가짜 +1,700% 갭 익절을
    만들었다(2026-10-31). 정상 데이터의 배당 조정 차이는 2% 이내."""
    grid = bar_grid(interval); gi = {g: k for k, g in enumerate(grid)}
    T, N, B = len(arr.days), len(arr.tickers), len(grid)
    di = {d.date(): i for i, d in enumerate(arr.days)}
    items: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    dropped = 0
    for j, t in enumerate(arr.tickers):
        for d, bars in intraday.get(t, {}).items():
            i = di.get(d)
            if i is None or bars is None or len(bars) < 10:
                continue
            ks = [gi.get(x) for x in bars.index.strftime("%H:%M")]
            ok = [n for n, k in enumerate(ks) if k is not None]
            if len(ok) < 10:
                continue
            d_open, d_high = float(arr.O[i, j]), float(arr.H[i, j])
            b_open, b_high = float(bars["Open"].iloc[ok[0]]), float(bars["High"].iloc[ok].max())
            if np.isnan(d_open) or np.isnan(d_high) or abs(b_open / d_open - 1) > max_basis_dev or abs(b_high / d_high - 1) > max_basis_dev:
                dropped += 1
                continue
            kk = np.array([ks[n] for n in ok])
            items.append((i, j, kk, bars["Open"].to_numpy(dtype=np.float32)[ok], bars["High"].to_numpy(dtype=np.float32)[ok],
                          bars["Low"].to_numpy(dtype=np.float32)[ok],
                          bars["Volume"].to_numpy(dtype=np.float32)[ok] if "Volume" in bars else None))
    days_idx = np.array(sorted({i for i, *_ in items}), dtype=int)
    pos = {i: k for k, i in enumerate(days_idx)}
    Td = len(days_idx)
    O = np.full((Td, N, B), np.nan, dtype=np.float32); H = O.copy(); L = O.copy()
    has_vol = any(v is not None for *_, v in items)
    V = O.copy() if has_vol else None
    has = np.zeros((T, N), dtype=bool)
    for i, j, kk, o, h, l, v in items:
        r = pos[i]
        O[r, j, kk] = o; H[r, j, kk] = h; L[r, j, kk] = l
        if V is not None and v is not None:
            V[r, j, kk] = v
        has[i, j] = True
    return Bars3D(grid, days_idx, O, H, L, has, dropped, V)


def _first_true(mask: np.ndarray) -> np.ndarray:
    """(T,N,B) bool → (T,N) 첫 True 의 봉 인덱스, 없으면 B."""
    B = mask.shape[-1]
    any_ = mask.any(axis=-1)
    k = mask.argmax(axis=-1)
    return np.where(any_, k, B)


def intraday_outcomes(b3: Bars3D, prev_close: np.ndarray, rules: ScalpRules, min_time: str = "10:00",
                      cutoff_time: str = "14:30", eod_time: str = "15:55", forced: bool = False,
                      vol20: np.ndarray | None = None, day_level: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """day_trade_intraday 의 벡터판. 분봉 있는 날(b3.days_idx)만 계산하고 (T,N) 으로 펼쳐 반환 (trig, entry, ret, why).
    day_level: (T,N) 일 단위로 정해진 가격 레벨(per_own 트리거) — 장 시작 전에 아는 값."""
    grid = np.array(b3.grid); B = len(grid)
    win = (grid >= min_time) & (grid <= cutoff_time)
    k_eod = int(np.argmax(grid >= eod_time)) if (grid >= eod_time).any() else B - 1
    O, H, L = b3.O, b3.H, b3.L
    T, N = prev_close.shape
    sub = b3.days_idx
    pc = prev_close[sub]; has = b3.has[sub]
    if forced:
        k0 = np.full((len(sub), N), int(np.argmax(win)))
        entry = np.take_along_axis(O, k0[:, :, None], axis=2)[:, :, 0].astype(float)
        trig = has & ~np.isnan(entry)
    else:
        if rules.trigger == "prev_close":
            level = (pc * (1 - rules.x_pct / 100)).astype(np.float32)[:, :, None]
        elif rules.trigger == "open":
            k_first = _first_true(~np.isnan(O))
            day_open = np.take_along_axis(O, np.minimum(k_first, B - 1)[:, :, None], axis=2)
            level = (day_open * (1 - rules.x_pct / 100)).astype(np.float32)
        elif rules.trigger == "ma20":
            level = np.full((len(sub), N, 1), np.nan, dtype=np.float32)   # 분봉에선 미지원
        elif rules.trigger in ("per_own", "per_own_any"):
            lv = day_level[sub] if day_level is not None else np.full((len(sub), N), np.nan)
            if rules.trigger == "per_own":
                lv = np.where(pc > lv, lv, np.nan)                          # 교차만: 전일 종가가 레벨 위
            level = lv.astype(np.float32)[:, :, None]
        else:
            hi_run = np.fmax.accumulate(np.nan_to_num(H, nan=-np.inf), axis=2)
            level = (hi_run * (1 - rules.x_pct / 100)).astype(np.float32)
        touch = (L <= level) & win[None, None, :]
        k0 = _first_true(touch)
        trig = (k0 < B) & has & ~np.isnan(pc)
        k0c = np.minimum(k0, B - 1)
        o0 = np.take_along_axis(O, k0c[:, :, None], axis=2)[:, :, 0].astype(float)
        lv0 = np.take_along_axis(np.broadcast_to(level, L.shape), k0c[:, :, None], axis=2)[:, :, 0].astype(float)
        if rules.skip_gap_below:
            trig &= ~(o0 <= lv0)
        entry = np.minimum(o0, lv0)
        if rules.max_vol_ratio is not None and b3.V is not None and vol20 is not None:
            # 트리거 봉까지 누적 거래량 ÷ (20일 평균 거래량 × 경과 시간 비율) — 장중에 알 수 있는 값
            cum = np.nancumsum(b3.V, axis=2)
            cum0 = np.take_along_axis(cum, k0c[:, :, None], axis=2)[:, :, 0].astype(float)
            elapsed = (k0c + 1) / B
            with np.errstate(invalid="ignore", divide="ignore"):
                vr = cum0 / (vol20[sub] * elapsed)
            trig &= ~(vr > rules.max_vol_ratio)
    tp = entry * (1 + rules.tp_pct / 100); sl = entry * (1 - rules.sl_pct / 100) if rules.sl_pct else None
    after = np.arange(B)[None, None, :] > k0[:, :, None]
    k_tp = _first_true((H >= tp[:, :, None]) & after)
    k_sl = _first_true((L <= sl[:, :, None]) & after) if sl is not None else np.full((len(sub), N), B)
    k_exit = np.minimum(np.minimum(k_tp, k_sl), k_eod)
    eod_px = O[:, :, k_eod].astype(float)
    px = np.where(k_sl <= k_exit, sl if sl is not None else np.nan, np.where(k_tp <= k_exit, tp, eod_px))
    why = np.where(k_sl <= k_exit, 2, np.where(k_tp <= k_exit, 1, 0))
    ret = px / entry - 1 - 2 * rules.fee_side
    ret = np.where(trig & ~np.isnan(ret), ret, np.nan)
    # (T,N) 으로 펼치기 — 분봉 없는 날은 트리거 없음
    trig_f = np.zeros((T, N), dtype=bool); trig_f[sub] = trig
    entry_f = np.full((T, N), np.nan); entry_f[sub] = entry
    ret_f = np.full((T, N), np.nan); ret_f[sub] = ret
    why_f = np.zeros((T, N), dtype=int); why_f[sub] = why
    return trig_f, entry_f, ret_f, why_f


def run_sim_intraday_fast(arr: Arrays, b3: Bars3D, lmask: np.ndarray, emask: np.ndarray, rules: ScalpRules,
                          mode: str = "strategy", rng: np.random.Generator | None = None, day_ok: np.ndarray | None = None) -> SimResult:
    valid, idx_ok, liquid = _common_masks(arr, rules)
    dl = per_level(arr, rules) if rules.trigger.startswith("per_own") else None
    trig, entry, ret, why = intraday_outcomes(b3, arr.PC, rules, vol20=arr.V20, day_level=dl)
    _, _, ret_open, why_open = intraday_outcomes(b3, arr.PC, rules, forced=True)
    depth = np.where(trig, arr.L / arr.PC - 1, np.inf)
    elig = trig & lmask & ~emask & idx_ok[:, None] & liquid
    if day_ok is None:
        day_ok = b3.has.any(axis=1)
    return _book(arr, trig, ret, why, ret_open, why_open, depth, elig, lmask, emask, idx_ok, liquid, valid & b3.has, rules, mode, rng, day_ok)

# ---------------------------------------------------------------------------
# 스윙 보유 모드 — 당일 +tp 미도달이면 보유: +tp2 / 다음 실적 직전 / 기한 / (종가 손절)
# ---------------------------------------------------------------------------

def next_earnings_exit(arr: Arrays, earnings_dates: dict[str, list[pd.Timestamp]], days_before: int) -> np.ndarray:
    """(T,N) int: i 일에 진입했을 때 '다음 발표일 − days_before' 의 날 인덱스 (없으면 T, 즉 기한만 적용)."""
    T, N = len(arr.days), len(arr.tickers)
    out = np.full((T, N), T, dtype=int)
    for j, t in enumerate(arr.tickers):
        rds = sorted(set(arr.days.searchsorted(pd.Timestamp(d)) for d in earnings_dates.get(t, [])))
        rds = [r for r in rds if 0 < r < T]
        if not rds:
            continue
        ptr = 0
        for i in range(T):
            while ptr < len(rds) and rds[ptr] <= i:
                ptr += 1
            if ptr < len(rds):
                out[i, j] = max(i, rds[ptr] - days_before)
    return out


def run_swing(arr: Arrays, lmask: np.ndarray, emask: np.ndarray, rules: ScalpRules, earn_exit: np.ndarray,
              mode: str = "strategy", rng: np.random.Generator | None = None, day_ok: np.ndarray | None = None,
              b3: "Bars3D | None" = None) -> SimResult:
    """당일 +tp 체결은 일봉 근사(fill 경계) 또는 분봉(b3)으로, 미도달 시 보유. 포지션은 여러 날 걸치므로 현금·주식 장부로 계산."""
    O, H, L, C, PC = arr.O, arr.H, arr.L, arr.C, arr.PC
    valid, idx_ok, liquid = _common_masks(arr, rules)
    day_rules = ScalpRules(**{**asdict(rules), "sl_pct": None})      # 당일은 손절 없이 +tp 만 본다
    is_per = rules.trigger.startswith("per_own")
    dl = per_level(arr, rules) if is_per else None
    if b3 is not None:
        trig, entry, ret0, why0 = intraday_outcomes(b3, PC, day_rules, vol20=arr.V20, day_level=dl)
        _, entry_o, ret0_o, why0_o = intraday_outcomes(b3, PC, day_rules, forced=True)
        valid = valid & b3.has
        if day_ok is None:
            day_ok = b3.has.any(axis=1)
    else:
        level, trig = _daily_level(arr, rules, valid)
        gap = O <= level
        if rules.skip_gap_below:
            trig &= ~gap
        entry = np.where(gap, O, level)
        ret0, why0 = _outcome(entry, H, L, C, day_rules)
        entry_o = O; ret0_o, why0_o = _outcome(O, H, L, C, day_rules)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(trig, (L / dl - 1) if is_per else (L / PC - 1), np.inf)     # per: 레벨 대비 더 깊이 내려온 순
        close_pos = np.where(H > L, (C - L) / (H - L), np.nan)
    elig = trig & lmask & ~emask & idx_ok[:, None] & liquid
    per_hold = rules.hold_mode == "per"
    if per_hold:
        if arr.EPS is None or arr.PREF is None:
            raise ValueError("hold_mode='per' 는 Arrays.EPS/PREF (per_columns) 가 필요")
        valid = valid & ~np.isnan(arr.PREF) & ~np.isnan(arr.EPS)         # 플라세보 B 도 원래 PER 이 있는 날만
    T = len(arr.days)
    cash = 1.0; positions: dict[int, dict] = {}
    equity = np.empty(T); realized = np.zeros(T); trades = []
    fee = rules.fee_side
    for i in range(T):
        # ---- 보유 종목 청산 판정 (오늘 일봉)
        for j, pos in list(positions.items()):
            e = pos["entry"]; o, h, c = float(O[i, j]), float(H[i, j]), float(C[i, j])
            if np.isnan(c):
                continue
            tp2 = e * (1 + rules.tp2_pct / 100)
            px = why = None
            if per_hold:
                # 설계서 §2.7: PER 이 낮은 동안 보유 — 종가 PER 이 진입 시 원래 PER × (1 − per_exit) 로 돌아오면 청산, EPS 소멸이면 청산
                eps_now = float(arr.EPS[i, j])
                if np.isnan(eps_now) or eps_now <= 0:
                    px, why = c, "PER 소멸"
                elif c / eps_now >= pos["per_target"]:
                    px, why = c, "PER 회복"
                elif rules.sl_pct and c <= e * (1 - rules.sl_pct / 100):
                    px, why = c, "손절(종가)"
                elif i >= pos["limit_i"]:
                    px, why = c, "기한"
                elif i == T - 1:
                    px, why = c, "구간 종료"
            elif o >= tp2:
                px, why = o, "익절2(갭)"
            elif h >= tp2:
                px, why = tp2, "익절2"
            elif rules.sl_pct and c <= e * (1 - rules.sl_pct / 100):
                px, why = c, "손절(종가)"
            elif i >= pos["exit_i"]:
                px, why = c, "실적 전" if pos["exit_i"] < pos["limit_i"] else "기한"
            elif i == T - 1:
                px, why = c, "구간 종료"
            if px is not None:
                proceeds = pos["shares"] * px * (1 - fee)
                cash += proceeds; realized[i] += proceeds - pos["amount"]
                trades.append({"day": arr.days[pos["i"]], "ticker": arr.tickers[j], "ret": proceeds / pos["amount"] - 1, "why": why,
                               "hold": i - pos["i"], "exit": arr.days[i]})
                del positions[j]
        # ---- 신규 진입
        sel: list[int] = []
        free = rules.max_positions - len(positions)
        if free > 0 and (day_ok is None or day_ok[i]):
            if mode == "strategy":
                row = np.where(elig[i], depth[i], np.inf)
                order = np.argsort(row)[:free]
                sel = [j for j in order if np.isfinite(row[j]) and j not in positions]
            elif mode == "placebo_a":
                k = min(int(elig[i].sum()), free)
                if k > 0:
                    pool = np.flatnonzero(trig[i] & ~lmask[i] & liquid[i] & idx_ok[i])
                    pool = [j for j in pool if j not in positions]
                    if pool:
                        sel = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
            else:
                if idx_ok[i]:
                    pool = [j for j in np.flatnonzero(lmask[i] & ~emask[i] & valid[i]) if j not in positions]
                    if pool:
                        sel = list(rng.choice(pool, size=min(free, len(pool)), replace=False))
        e_src, r_src, w_src = (entry_o, ret0_o, why0_o) if mode == "placebo_b" else (entry, ret0, why0)
        eq_now = cash + sum(p["shares"] * float(C[i, jj]) for jj, p in positions.items() if not np.isnan(C[i, jj]))
        for j in sel:
            e = float(e_src[i, j]); r0 = float(r_src[i, j])
            if np.isnan(e) or np.isnan(r0):
                continue
            amount = rules.position_frac * eq_now
            if amount > cash:
                break
            if per_hold:
                cash -= amount
                limit_i = i + rules.max_hold_days if rules.max_hold_days > 0 else T
                positions[j] = {"i": i, "entry": e, "amount": amount, "shares": amount * (1 - fee) / e, "exit_i": limit_i, "limit_i": limit_i,
                                "per_target": float(arr.PREF[i, j]) * (1 - rules.per_exit_pct / 100)}
            elif int(w_src[i, j]) == 1 or rules.hold_mode == "day":
                # 당일 청산 (익절 또는 day 모드의 종가)
                cash += amount * r0; realized[i] += amount * r0          # 당일 왕복: 원금은 그대로, 손익만 반영
                trades.append({"day": arr.days[i], "ticker": arr.tickers[j], "ret": r0, "why": "익절1" if int(w_src[i, j]) == 1 else "종가(당일)", "hold": 0, "exit": arr.days[i]})
            elif rules.close_pos_stop is not None and not np.isnan(close_pos[i, j]) and close_pos[i, j] < rules.close_pos_stop:
                # 종가가 당일 범위 하위(close_pos_stop)로 마감 → 그날 종가 손절 (설계서 §2.6)
                rc = float(C[i, j]) / e - 1 - 2 * fee
                cash += amount * rc; realized[i] += amount * rc
                trades.append({"day": arr.days[i], "ticker": arr.tickers[j], "ret": rc, "why": "종가 손절", "hold": 0, "exit": arr.days[i]})
            else:
                cash -= amount
                limit_i = i + rules.max_hold_days if rules.max_hold_days > 0 else T
                exit_i = min(int(earn_exit[i, j]), limit_i)
                positions[j] = {"i": i, "entry": e, "amount": amount, "shares": amount * (1 - fee) / e, "exit_i": exit_i, "limit_i": limit_i}
        equity[i] = cash + sum(p["shares"] * float(C[i, jj]) for jj, p in positions.items() if not np.isnan(C[i, jj]))
    eq_s = pd.Series(equity, index=arr.days); pnl_s = pd.Series(realized, index=arr.days)
    days_used = arr.days if day_ok is None else arr.days[day_ok]
    st = summarize(eq_s, _tax_equity(eq_s, pnl_s), trades, arr.days)
    hs = [t["hold"] for t in trades]
    st["평균 보유일"] = float(np.mean(hs)) if hs else None
    st["당일 익절 비율"] = float(np.mean([t["hold"] == 0 for t in trades])) if trades else None
    return SimResult(eq_s, trades, st)
