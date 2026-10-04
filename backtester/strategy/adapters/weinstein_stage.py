"""Stan Weinstein «최적의 타이밍» — 4단계(Stage) 분석 + 매수 체크리스트 평가 헬퍼.

문서: ``docs/strategy_notes/와인스타인_최적의타이밍_체크리스트_2026_10.md``
(항목 번호 0-1 … 8-6 은 그 문서의 STAGE 테이블과 일치한다.)

모든 함수는 **주봉(금요일 종가)** DataFrame 을 기준으로 동작한다. 일봉 → 주봉 변환은
:func:`to_weekly`. 책에는 수치 임계값이 없으므로 여기의 기본값은 문서 §6 의 "제안값"이며
:class:`StageParams` 로 바꿀 수 있다.

순수 pandas/numpy 만 사용 — 데이터 소스(yfinance 등)는 호출자가 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 파라미터
# --------------------------------------------------------------------------- #
@dataclass
class StageParams:
    ma_weeks: int = 30
    slope_weeks: int = 4
    flat_threshold: float = 0.005      # |4주 기울기| < 0.5% → "평평"
    stage_lookback: int = 26           # 1/3단계 판정 시 직전 추세 탐색 범위(주)
    base_min_weeks: int = 12           # 2-6: 베이스 최소 길이
    base_max_weeks: int = 104
    base_width: float = 0.30           # 2-5: 베이스 고저 폭 상한 (3x ETF면 완화)
    touch_tolerance: float = 0.03      # 저항선 ±3% 안이면 "터치"
    near_resistance_pct: float = 0.15  # 3-3: 돌파가 위 15% 이내 저항 탐색
    near_resistance_min_weeks: int = 6 # 3-3: 그 구간에 종가가 이만큼 이상 머물렀으면 '의미 있는 매물대'
    mid_term_weeks: int = 130          # 3-4: 2.5년
    long_term_weeks: int = 520         # 3-5: 10년
    vol_pass: float = 2.0              # 4-1: 대어 기준
    vol_aplus: float = 3.0
    vol_fail: float = 1.2
    rs_window: int = 52                # 맨스필드 RS 제로선 = 52주 평균
    rs_slope_weeks: int = 13
    late_stage_from_breakout: float = 0.25  # 2-10: 돌파가 대비 +25% 넘으면 후반
    late_stage_ma_ext: float = 0.15         # 2-10: MA 이격 15% 넘으면 후반
    leverage: float = 1.0              # 3x ETF 등: 폭/이격 임계값에 곱한다
    vol_required: bool = True          # False: 4-1 을 정보 항목으로 (ETF 처럼 거래량 규칙이 안 맞는 자산)

    def scaled(self, x: float) -> float:
        return x * self.leverage


# --------------------------------------------------------------------------- #
# 주봉 변환 / MA / 단계
# --------------------------------------------------------------------------- #
def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """일봉 OHLCV → 금요일 기준 주봉. 책의 30주 MA 는 금요일 종가 30개 평균."""
    df = daily.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    w = (
        df.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )
    return w


def add_ma(weekly: pd.DataFrame, p: StageParams = StageParams()) -> pd.DataFrame:
    w = weekly.copy()
    w["MA"] = w["Close"].rolling(p.ma_weeks).mean()
    w["slope"] = w["MA"] / w["MA"].shift(p.slope_weeks) - 1
    w["above"] = w["Close"] > w["MA"]
    w["ext"] = w["Close"] / w["MA"] - 1      # MA 이격
    return w


def classify_stages(weekly_ma: pd.DataFrame, p: StageParams = StageParams()) -> pd.Series:
    """문서 §6 규칙.

    2 = 주가 > MA & MA 상승, 4 = 주가 < MA & MA 하락,
    평평(|slope| < flat) 이면 직전 추세가 4 → 1단계, 2 → 3단계.
    그 외(혼합)는 직전 단계 유지. 0 = 데이터 부족.
    """
    w = weekly_ma
    stages = np.zeros(len(w), dtype=int)
    last_trend = 0
    prev = 0
    for i in range(len(w)):
        s = w["slope"].iat[i]
        ab = bool(w["above"].iat[i])
        if np.isnan(s):
            stages[i] = 0
            continue
        if ab and s >= p.flat_threshold:
            st = 2
        elif (not ab) and s <= -p.flat_threshold:
            st = 4
        elif abs(s) < p.flat_threshold:
            if last_trend == 4:
                st = 1
            elif last_trend == 2:
                st = 3
            else:
                st = prev
        else:
            st = prev
        if st in (2, 4):
            last_trend = st
        stages[i] = st
        prev = st
    return pd.Series(stages, index=w.index, name="stage")


def mansfield_rs(close: pd.Series, index_close: pd.Series, window: int = 52) -> pd.Series:
    """맨스필드 상보강도: (종목/지수 비율) / 그 비율의 52주 평균 − 1, ×100. 0 = 제로선."""
    ratio = (close / index_close.reindex(close.index).ffill()).dropna()
    rs = (ratio / ratio.rolling(window).mean() - 1) * 100
    return rs.reindex(close.index)


# --------------------------------------------------------------------------- #
# 베이스(트레이딩 범위) / 돌파
# --------------------------------------------------------------------------- #
@dataclass
class Base:
    start: pd.Timestamp
    end: pd.Timestamp          # 베이스 마지막 주 (돌파 주 직전)
    weeks: int
    high: float                # 저항선
    low: float                 # 지지선
    width: float
    touches: int
    ma_crosses: int
    avg_volume: float
    prebreak_rise: float       # 2-9: high / low − 1 ... (베이스 저점 → 저항)


def find_base(w: pd.DataFrame, i: int, p: StageParams = StageParams()) -> Base | None:
    """주 i 직전까지 끝나는 가장 긴 트레이딩 범위를 찾는다.

    범위 폭(max High − min Low)/min Low 는 창을 늘릴수록 단조 증가하므로
    처음 폭 상한을 넘는 지점에서 멈춘다.
    """
    width_cap = p.scaled(p.base_width)
    best = None
    for L in range(p.base_min_weeks // 2, p.base_max_weeks + 1):
        if i - L < 0:
            break
        seg = w.iloc[i - L : i]
        hi, lo = float(seg["High"].max()), float(seg["Low"].min())
        if lo <= 0:
            break
        width = (hi - lo) / lo
        if width > width_cap:
            break
        best = (L, hi, lo, width, seg)
    if best is None:
        return None
    L, hi, lo, width, seg = best
    touches = int((seg["High"] >= hi * (1 - p.scaled(p.touch_tolerance))).sum())
    above = (seg["Close"] > seg["MA"]).astype(int)
    ma_crosses = int(above.diff().abs().fillna(0).sum())
    return Base(
        start=seg.index[0],
        end=seg.index[-1],
        weeks=L,
        high=hi,
        low=lo,
        width=width,
        touches=touches,
        ma_crosses=ma_crosses,
        avg_volume=float(seg["Volume"].mean()),
        prebreak_rise=float(seg["Close"].iloc[-1] / lo - 1),
    )


def overhead_resistance(w: pd.DataFrame, i: int, base: Base, price: float, p: StageParams) -> dict:
    """3-3 ~ 3-5: 베이스 시작 이전 구간에서 돌파가 위 저항을 찾는다."""
    base_start_pos = w.index.get_loc(base.start)
    hist_mid = w.iloc[max(0, base_start_pos - p.mid_term_weeks) : base_start_pos]
    hist_long = w.iloc[max(0, base_start_pos - p.long_term_weeks) : base_start_pos]
    near_cap = price * (1 + p.scaled(p.near_resistance_pct))
    near = hist_mid[(hist_mid["Close"] > price) & (hist_mid["Close"] <= near_cap)]
    return {
        "mid_max_high": float(hist_mid["High"].max()) if len(hist_mid) else np.nan,
        "long_max_high": float(hist_long["High"].max()) if len(hist_long) else np.nan,
        "near_weeks": int(len(near)),
        "near_volume": float(near["Volume"].sum()) if len(near) else 0.0,
        "mid_weeks_available": int(len(hist_mid)),
        "long_weeks_available": int(len(hist_long)),
    }


# --------------------------------------------------------------------------- #
# 체크리스트 평가
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    code: str
    name: str
    value: str
    passed: bool | None        # None = 정보성(합격/불합격 없음)
    required: bool = False
    note: str = ""


def _fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x*100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def evaluate_breakout(
    w: pd.DataFrame,
    stages: pd.Series,
    rs: pd.Series,
    i: int,
    base: Base,
    p: StageParams,
    market: dict | None = None,
    sector: dict | None = None,
) -> list[Check]:
    """돌파 주 i 에서 STAGE 2~5 항목을 평가한다. market/sector 는 호출자가 미리 계산해 넘긴다."""
    row = w.iloc[i]
    prev_stage = int(stages.iloc[max(0, i - 8) : i].mode().iat[0]) if i > 0 else 0
    was_stage1 = (stages.iloc[max(0, i - 8) : i] == 1).any()
    checks: list[Check] = []

    # ----- STAGE 0 / 1 (외부 계산값) -----
    for src, prefix in ((market, "0"), (sector, "1")):
        if src:
            for c in src.get("checks", []):
                checks.append(c)

    # ----- STAGE 2 -----
    checks.append(Check("2-1", "주가 > 30주 MA", f"close {row.Close:.2f} vs MA {row.MA:.2f}", bool(row.above), True))
    slope_ok = row.slope >= 0
    checks.append(Check("2-2", "30주 MA 방향(4주 기울기 ≥ 0)", _fmt(row.slope, True), bool(slope_ok), True,
                        "상승(+0.5%↑)" if row.slope >= p.flat_threshold else "평평" if slope_ok else "하락 → 절대 매수 금지"))
    cur_stage = int(stages.iloc[i])
    checks.append(Check("2-3", "단계 전이 1→2 (또는 2단계 지속)", f"직전8주 최빈 {prev_stage} → 현재 {cur_stage}",
                        cur_stage == 2 and (was_stage1 or prev_stage in (1, 2)), True,
                        "1→2 전환(투자가 최적)" if was_stage1 else "2단계 지속 매수(continuation)"))
    pre = stages.iloc[max(0, w.index.get_loc(base.start) - 52) : w.index.get_loc(base.start)]
    checks.append(Check("2-4", "베이스 전 52주 내 4단계(하락) 존재", f"{int((pre == 4).sum())}주", bool((pre == 4).any()), False,
                        "진짜 바닥(1단계)인지 vs 상승 중 쉬어가기인지"))
    checks.append(Check("2-5", "트레이딩 범위 존재", f"{base.weeks}주, 폭 {_fmt(base.width, True)}, 상단 터치 {base.touches}회",
                        base.touches >= 2, True))
    checks.append(Check("2-6", "베이스 길이", f"{base.weeks}주",
                        base.weeks >= p.base_min_weeks, False,
                        "≥52 최상" if base.weeks >= 52 else "≥26 우량" if base.weeks >= 26 else "≥12 통과" if base.weeks >= 12 else "짧음"))
    v4 = w["Volume"].iloc[i - 4 : i].mean()
    checks.append(Check("2-7", "1단계 후반 거래량 증가(돌파 전 4주 > 베이스 평균)", f"{v4/base.avg_volume:.2f}x",
                        bool(v4 > base.avg_volume), False))
    checks.append(Check("2-8", "베이스 중 MA 교차 ≥ 2회", f"{base.ma_crosses}회", base.ma_crosses >= 2, False))
    checks.append(Check("2-9", "돌파 전 베이스 내 상승 ≥ 40% (대어 ③)", _fmt(base.prebreak_rise, True),
                        base.prebreak_rise >= 0.4, False, "가점 항목"))
    ext_cap = p.scaled(p.late_stage_ma_ext)
    checks.append(Check("2-10", "상승 후반 아님 (MA 이격)", f"이격 {_fmt(row.ext, True)} (상한 {_fmt(ext_cap, True)})",
                        bool(row.ext <= ext_cap), True))

    # ----- STAGE 3 -----
    res = base.high
    oh = overhead_resistance(w, i, base, res, p)
    checks.append(Check("3-1", "저항선(돌파 기준가)", f"{res:.2f} (베이스 {base.start.date()}~{base.end.date()})", None))
    checks.append(Check("3-2", "저항선 + MA 동시 돌파 (주봉 종가)", f"close {row.Close:.2f} > res {res:.2f} & > MA {row.MA:.2f}",
                        bool(row.Close > res and row.Close > row.MA), True))
    checks.append(Check("3-3", f"돌파가 위 +{_fmt(p.scaled(p.near_resistance_pct), True, 0)} 내 과거 매물 없음",
                        f"{oh['near_weeks']}주 체류 (기준 <{p.near_resistance_min_weeks}주)", oh["near_weeks"] < p.near_resistance_min_weeks, True,
                        "의미 있는 매물대면 B- 강등 (이전 지지대 = 다음 저항)"))
    mid_ok = np.isnan(oh["mid_max_high"]) or oh["mid_max_high"] <= res
    long_ok = np.isnan(oh["long_max_high"]) or oh["long_max_high"] <= res
    checks.append(Check("3-4", "2.5년 내 위쪽 저항 없음", f"130주 최고 {_fmt(oh['mid_max_high'])} vs {res:.2f} ({oh['mid_weeks_available']}주 데이터)",
                        bool(mid_ok), False))
    checks.append(Check("3-5", "10년 내 위쪽 저항 없음 = A+", f"520주 최고 {_fmt(oh['long_max_high'])} vs {res:.2f} ({oh['long_weeks_available']}주 데이터)",
                        bool(long_ok), False, "A+ 플래그"))

    # ----- STAGE 4 -----
    vol_ratio = row.Volume / v4 if v4 > 0 else np.nan
    checks.append(Check("4-1", "돌파 주 거래량 / 직전 4주 평균", f"{vol_ratio:.2f}x",
                        bool(vol_ratio >= p.vol_pass), p.vol_required,
                        "A+ (≥3x)" if vol_ratio >= p.vol_aplus else "통과(≥2x)" if vol_ratio >= p.vol_pass
                        else "미달 — 빈약(<1.2x)" if vol_ratio < p.vol_fail else "미달(1.2~2x)"))
    v_prev3 = w["Volume"].iloc[i - 3 : i].values
    pre_slope = np.polyfit(range(3), v_prev3, 1)[0] if len(v_prev3) == 3 else np.nan
    checks.append(Check("4-3", "돌파 전 1~3주 거래량 선행 증가", "증가" if pre_slope > 0 else "감소", bool(pre_slope > 0), False, "가점"))
    # 4-2, 4-4, 4-5 는 돌파 이후 데이터라 시뮬레이션 쪽에서 평가

    # ----- STAGE 5 -----
    rs_now = rs.iloc[i]
    rs_prev = rs.iloc[i - p.rs_slope_weeks] if i >= p.rs_slope_weeks else np.nan
    checks.append(Check("5-1", f"RS선 {p.rs_slope_weeks}주 기울기 > 0", f"{_fmt(rs_prev)} → {_fmt(rs_now)}",
                        bool(rs_now > rs_prev) if not np.isnan(rs_prev) else None, True))
    checks.append(Check("5-2", "돌파 주 맨스필드 RS > 0 (플러스 영역)", _fmt(rs_now), bool(rs_now > 0), False))
    checks.append(Check("5-3", "제로선 전환(13주 전 ≤ 0 → 지금 > 0) = 대어 ②",
                        f"{_fmt(rs_prev)} → {_fmt(rs_now)}", bool(rs_now > 0 and rs_prev <= 0) if not np.isnan(rs_prev) else None, False, "가점"))
    return checks


def grade(checks: list[Check]) -> str:
    """문서 §7 등급표."""
    c = {k.code: k for k in checks}

    def ok(code):
        k = c.get(code)
        return bool(k and k.passed)

    required_fail = [k.code for k in checks if k.required and k.passed is False]
    if required_fail:
        # 저항(3-3)만 실패면 B-, 그 외 필수 실패는 탈락
        if required_fail == ["3-3"]:
            return "B-"
        return "탈락(" + ",".join(required_fail) + ")"
    vol = c.get("4-1")
    vol_x = float(vol.value.rstrip("x")) if vol else 0
    base_weeks = int(c["2-6"].value.rstrip("주")) if "2-6" in c else 0
    vol_ok = (vol_x >= 2) if (vol and vol.required) else True
    if ok("3-5") and vol_ok and ok("5-2") and base_weeks >= 26:
        return "A+"
    if ok("3-4") and vol_ok and ok("5-2"):
        return "A"
    if vol_ok:
        return "B"
    return "B-"


def checks_to_frame(checks: Iterable[Check]) -> pd.DataFrame:
    rows = []
    for k in checks:
        rows.append({
            "code": k.code, "항목": k.name, "값": k.value,
            "판정": "—" if k.passed is None else ("✅" if k.passed else "❌"),
            "필수": "필수" if k.required else "",
            "비고": k.note,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 시장 폭 지표 (유니버스 일봉 dict → 주간 시계열)
# --------------------------------------------------------------------------- #
def breadth_from_universe(dailies: dict[str, pd.DataFrame], p: StageParams = StageParams()) -> pd.DataFrame:
    """0-3 1·2단계 비율, 0-4 A/D 라인, 0-5 MI(200일), 0-6 주간 신고−신저."""
    closes = pd.DataFrame({t: d["Close"] for t, d in dailies.items()}).sort_index()
    closes.index = pd.to_datetime(closes.index)
    if getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_localize(None)
    chg = closes.diff()
    adv = (chg > 0).sum(axis=1)
    dec = (chg < 0).sum(axis=1)
    net = adv - dec
    ad_line = net.cumsum()
    mi = net.rolling(200).mean()

    weekly_close = closes.resample("W-FRI").last()
    hi52 = weekly_close.rolling(52).max()
    lo52 = weekly_close.rolling(52).min()
    new_high = (weekly_close >= hi52).sum(axis=1)
    new_low = (weekly_close <= lo52).sum(axis=1)

    stage_cols = {}
    for t, d in dailies.items():
        try:
            w = add_ma(to_weekly(d), p)
            stage_cols[t] = classify_stages(w, p)
        except Exception:
            continue
    stages = pd.DataFrame(stage_cols).reindex(weekly_close.index).ffill()
    valid = (stages > 0).sum(axis=1)
    pct12 = ((stages == 1) | (stages == 2)).sum(axis=1) / valid.replace(0, np.nan)
    pct2 = (stages == 2).sum(axis=1) / valid.replace(0, np.nan)

    out = pd.DataFrame({
        "pct_stage12": pct12,
        "pct_stage2": pct2,
        "ad_line": ad_line.resample("W-FRI").last(),
        "mi": mi.resample("W-FRI").last(),
        "nh": new_high, "nl": new_low, "nh_nl": new_high - new_low,
        "n_valid": valid,
    })
    return out
