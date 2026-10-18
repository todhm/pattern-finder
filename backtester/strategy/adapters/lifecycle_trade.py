"""«The Lifecycle Trade» (Boboch·Donnelly·Krull·Daill, 2019) — IPO 생애주기 트레이드 엔진 (Streamlit 비의존).

문서 ``docs/strategy_notes/라이프사이클_트레이드_체크리스트_2026_10.md`` 의 [자동]/[반자동] 항목을 코드로 옮긴 것.

    IPO 캘린더(첫 거래일·공모가) → 일봉 → 주봉(10주·40주 "modified" SMA)
    → 베이스 탐지 → 생애주기 단계(IPO-BASE / IPO-AP / IPO-AF / I-DDP / I-AP) + 6패턴 분류
    → 4종 매수 지점(IPO base · initial base · breakaway gap · first mature base)
    → 4종 매도 규칙(Ascender · Midterm · 40-Week · Everest) + 10% 손절 + MCP 스톱 — $100K 시뮬
    → 6장 연구 재현(Table 6.1/6.2, Fig 6.1/6.2, 유동성 텔) · 7장 3종 스크린(IPO Alert / Rare Jewels / Liquidity Matters)

책의 수치는 TradeStation 으로 손으로 고른 매수 지점에 규칙을 적용한 것이고, 여기서는 베이스·단계를
규칙으로 근사한다 — 패턴 분류는 사후(hindsight) 라벨이며 스크린(기준일)에서는 '지금까지의 패턴'만 쓴다.
데이터 소스는 전부 주입 — composition root(페이지·CLI)가 고른다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from data.domain.models import IpoEvent
from strategy.adapters.minervini_scan import fetch_with_retry

PATTERNS = ("Rocket Ship", "Stair Stepper", "Late Bloomer", "Pump and Dump", "One-Hit Wonder",
            "One-Hit Wonder Plus", "Disappointment")
RULES = ("Ascender", "Midterm", "40-Week", "Everest", "MCP")
BUY_KINDS = ("IPO base", "initial base", "breakaway gap", "first mature base")
LIQUIDITY_USD = 20e6


@dataclass(frozen=True)
class LifecycleParams:
    ipo_base_max_weeks: int = 7        # 5장: IPO base = IPO 후 7주 안에 시작하는 짧은 횡보
    ipo_base_max_len: int = 10         # IPO base 최대 길이(주) — 그보다 길면 initial base / I-DDP 베이스
    base_min_weeks: int = 3            # 구글 18일·이베이 21일 → 3주
    base_max_depth: float = 0.35       # 베이스 폭(고점 대비). 초기 IPO 는 변동성이 커 35%
    base_max_drift: float = 0.20       # 베이스 안 종가 순변화 한도 — 추세 구간을 베이스로 오인하지 않게
    ap_min_gain: float = 0.20          # IPO-AP 성립: 기준가(IPO base 돌파 종가 / 첫 주 종가) 대비 +20% (Stair Stepper 정의)
    ap_max_weeks: int = 20             # 3장: IPO-AP 는 보통 20주 미만
    ap_fail_drawdown: float = 0.40     # 3장: IPO-AP 종료 = 고점 −40% 또는 IPO base 아래
    big_ap_gain: float = 0.60          # One-Hit Wonder 류: 큰 IPO-AP (SHAK 2배, FIT/LOCO/ZNGA +60~70%) 뒤 전체 되돌림
    big_ap_weeks: int = 20             # IPO-AP 길이 안(20주)이면 '빠른' 것으로 본다
    rocket_gain: float = 1.00          # Rocket Ship: IPO-AP 100%+ (block diagram: >100~500%)
    rocket_weeks: int = 13
    mature_base_min_weeks: int = 7     # first mature base — initial base(7주 초과) 와 같은 길이 기준
    iddp_min_weeks: int = 13           # 실패 후 최소 1분기 지나야 '성숙 베이스'
    late_bloomer_min_weeks: int = 39   # 4장: Late Bloomer 의 I-DDP "close to a year or more"
    gap_pct: float = 0.25              # 5장: breakaway gap — 전일 종가 대비 시가·종가 +25% (FB +30%)
    gap_vol_mult: float = 1.5
    stop_pct: float = 0.10             # 5장 연구: 매수가 −10% 손절
    mcp: float = 0.45                  # 8장: 피크 미실현 이익의 45% 보전
    mcp_arm_gain: float = 0.10         # MCP 스톱은 이익이 이만큼(Eric: +10%) 난 뒤에만 켠다 — 그 전엔 10% 손절만
    slip: float = 0.005
    everest_ext: float = 0.12          # Everest '포물선' 근사: 종가/21일 EMA −1 ≥ 12% & 60일 신고가 & 10일 +20%
    everest_run: float = 0.20
    everest_run_days: int = 10
    ascender_max_days: int = 378       # 18개월 ≈ 378 거래일
    midterm_long_days: int = 252       # 1년 보유 후 장기 트리거
    ema_days: int = 21
    sma_days: int = 50


# --------------------------------------------------------------------------- #
# 데이터 준비
# --------------------------------------------------------------------------- #
@dataclass
class Frames:
    d: pd.DataFrame            # 일봉 + EMA21/SMA50 + 주 라벨/주말 플래그 + (주말 행) SMA10w/SMA40w
    w: pd.DataFrame            # 주봉 (W-FRI) + SMA10/SMA40 (min_periods=1 — 책의 'modified' MA)
    ipo_date: date | None
    has_day1: bool             # 첫 봉이 IPO 첫 거래일(±10일)인가 — 6장 통계 자격


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[(out["Close"] > 0) & out["Close"].notna()]
    return out[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def to_weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                 "Volume": "sum"}).dropna(subset=["Close"])
    # 책 각주 9: 상장 초기엔 40주가 차기 전까지 '있는 만큼의 평균' (rolling average)
    w["SMA10"] = w["Close"].rolling(10, min_periods=1).mean()
    w["SMA40"] = w["Close"].rolling(40, min_periods=1).mean()
    return w


def prepare(daily: pd.DataFrame, ipo_date: date | None, as_of: date | None = None,
            p: LifecycleParams = LifecycleParams()) -> Frames | None:
    if daily is None or len(daily) < 5:
        return None
    d = normalize_daily(daily)
    if ipo_date is not None:
        d = d[d.index >= pd.Timestamp(ipo_date) - pd.Timedelta(days=3)]
    if as_of is not None:
        d = d[d.index <= pd.Timestamp(as_of)]
    if len(d) < 5:
        return None
    d["EMA21"] = d["Close"].ewm(span=p.ema_days, adjust=False).mean()
    d["SMA50"] = d["Close"].rolling(p.sma_days, min_periods=1).mean()
    d["DVOL20"] = (d["Close"] * d["Volume"]).rolling(20, min_periods=1).mean()
    w = to_weekly(d)
    wk_label = d.index.to_period("W-FRI")
    d["wk_end"] = pd.Series(wk_label, index=d.index).ne(pd.Series(wk_label, index=d.index).shift(-1)).values
    fri = pd.Series(wk_label, index=d.index).dt.end_time.dt.normalize()
    # 주말 행에 그 주의 SMA10/SMA40 (그 주 종가 포함 — 금요일 마감 시점 값)
    d["SMA10w"] = fri.map(w["SMA10"]).values
    d["SMA40w"] = fri.map(w["SMA40"]).values
    has_day1 = ipo_date is not None and abs((d.index[0].date() - ipo_date).days) <= 10
    return Frames(d=d, w=w, ipo_date=ipo_date, has_day1=bool(has_day1))


# --------------------------------------------------------------------------- #
# 베이스 (주봉 횡보) 탐지
# --------------------------------------------------------------------------- #
@dataclass
class Base:
    start_i: int
    end_i: int
    high: float
    low: float
    weeks: int
    depth: float
    breakout_i: int | None     # 베이스 고점 위로 주봉 종가가 올라선 주 (없으면 None = 형성 중/실패)

    def kind_at(self, p: LifecycleParams) -> str:
        return "IPO base" if self.start_i <= p.ipo_base_max_weeks else "base"


def find_bases(w: pd.DataFrame, start_i: int, end_i: int, p: LifecycleParams,
               min_weeks: int | None = None) -> list[Base]:
    """[start_i, end_i] 주봉 구간에서 '폭 ≤ depth & 순변화 ≤ drift' 횡보를 탐욕적으로 자른다.

    ``min_weeks`` 가 찬 뒤 주봉 종가가 베이스 고점(꼬리 포함) 위로 마감하면 그 주가 돌파.
    폭이 넘치면(위/아래 이탈) 돌파 없이 끝난 베이스. 돌파/종료 다음 주부터 다시 탐색.
    """
    min_weeks = min_weeks or p.base_min_weeks
    H, L, C = w["High"].values, w["Low"].values, w["Close"].values
    out: list[Base] = []
    s = start_i
    while s + min_weeks - 1 <= end_i:
        hi, lo, j, bo = H[s], L[s], s, None
        while j + 1 <= end_i:
            k = j + 1
            if (k - s) >= min_weeks and C[k] > hi:
                bo = k
                break
            nh, nl = max(hi, H[k]), min(lo, L[k])
            if (nh - nl) / nh > p.base_max_depth:
                break
            hi, lo, j = nh, nl, k
        weeks = j - s + 1
        if weeks >= min_weeks and abs(C[j] / C[s] - 1) <= p.base_max_drift:
            out.append(Base(s, j, float(hi), float(lo), weeks, float((hi - lo) / hi), bo))
            s = (bo if bo is not None else j) + 1
        else:
            s += 1
    return out


def breakout_day(d: pd.DataFrame, level: float, after: pd.Timestamp) -> pd.Timestamp | None:
    """``after``(베이스 마지막 주 금요일) 이후 첫 일봉 종가 > level."""
    seg = d[(d.index > after) & (d["Close"] > level)]
    return seg.index[0] if len(seg) else None


# --------------------------------------------------------------------------- #
# 생애주기 단계·패턴
# --------------------------------------------------------------------------- #
@dataclass
class Lifecycle:
    gray: float                         # 첫 주 종가 (책의 회색선)
    offer: float | None
    ipo_base: Base | None
    ap_start_i: int
    ap_ref: float                       # IPO-AP 기준가
    ap_peak_i: int
    ap_peak: float
    ap_gain: float
    ap_exists: bool
    af_i: int | None                    # IPO-AF 확정 주 (고점 −40% / 기준가·베이스·공모가 아래 마감)
    undercut_i: int | None              # IPO-AP 전체 되돌림(기준가·베이스 저가 아래 마감) 주
    iddp_start_i: int | None
    mature_base: Base | None
    iap_start_i: int | None
    turbulence: float
    turbulence_cleared_i: int | None
    pattern: str
    pattern_note: str
    phases: pd.Series                   # 주봉 인덱스 → "IPO-BASE"/"IPO-AP"/"IPO-AF"/"I-DDP"/"I-AP"
    bases: list[Base]

    @property
    def phase_now(self) -> str:
        return str(self.phases.iloc[-1]) if len(self.phases) else "?"


def classify_lifecycle(w: pd.DataFrame, p: LifecycleParams = LifecycleParams(),
                       offer_price: float | None = None) -> Lifecycle:
    n = len(w)
    H, L, C = w["High"].values, w["Low"].values, w["Close"].values
    S40 = w["SMA40"].values
    gray = float(C[0])
    # 공모가는 분할 미조정 숫자다. 일봉이 분할 조정돼 있으면(MA 10:1, GOOGL 20:1) 공모가가 첫 주 종가의 몇 배가 되므로
    # 그 경우(비율 > 1.6 또는 < 0.15) 공모가 바닥선을 쓰지 않는다 — 상장 첫 주에 공모가 −37% 는 극히 드물다.
    if offer_price is not None and not (0.15 <= offer_price / gray <= 1.6):
        offer_price = None
    bases = find_bases(w, 0, n - 1, p)
    ipo_base = next((b for b in bases if b.start_i <= p.ipo_base_max_weeks and b.weeks <= p.ipo_base_max_len), None)

    # --- IPO-AP ---
    # 시작 = IPO base 돌파 주(베이스가 있으면) / 첫 주. 상승률은 책의 회색선(첫 주 종가) 기준.
    # 바닥선(floor) = IPO base 저가 또는 공모가 — 첫 주 저가는 55% 가 3주 안에 깨므로(6장) 쓰지 않는다.
    # IPO-AP 종료(IPO-AF) = 고점 −40% 마감 또는 바닥선 아래 마감. 첫 주(ap_start) 고가는 '상승' 에 넣지 않는다
    # (상장일 스파이크를 IPO-AP 로 오인하지 않게) — 이후 주가 그 위로 올라가야 상승이다.
    if ipo_base is not None and ipo_base.breakout_i is not None:
        ap_start, floor = ipo_base.breakout_i, ipo_base.low
    else:
        ap_start, floor = 0, (float(offer_price) if offer_price else -np.inf)
    ap_ref = gray
    runmax, peak_i, af_i = float(C[ap_start]), ap_start, None
    for i in range(ap_start + 1, n):
        if i <= ap_start + p.ap_max_weeks and H[i] > runmax:
            runmax, peak_i = float(H[i]), i
        if C[i] < runmax * (1 - p.ap_fail_drawdown) or C[i] < floor:
            af_i = i
            break
    ap_gain = runmax / ap_ref - 1
    ap_exists = ap_gain >= p.ap_min_gain and peak_i > ap_start
    struct_low = ipo_base.low if ipo_base is not None else gray      # '구조 전체' 이탈 기준
    hit = np.where(C[peak_i + 1:] < struct_low)[0] if peak_i + 1 < n else np.array([])
    undercut_i = int(peak_i + 1 + hit[0]) if len(hit) else None

    # --- I-DDP → first mature base → I-AP ---
    if ap_exists:
        iddp_start = af_i
    else:
        iddp_start = ap_start   # IPO-AF: 처음부터 실사 구간
    mature, iap_start = None, None
    if iddp_start is not None:
        for b in bases:
            # 베이스는 실패 확정 주보다 먼저 시작할 수 있다(하락 초입부터 횡보) — 끝이 I-DDP 안에 있으면 된다
            if (b.end_i >= iddp_start and b.weeks >= p.mature_base_min_weeks and b.breakout_i is not None
                    and b.breakout_i - iddp_start >= p.iddp_min_weeks and C[b.breakout_i] > S40[b.breakout_i]):
                mature, iap_start = b, b.breakout_i
                break
    turbulence = runmax if ap_exists else float(max(H[: ap_start + 1]))
    cleared = None
    if iap_start is not None:
        hit = np.where(C[iap_start:] > turbulence)[0]
        cleared = int(iap_start + hit[0]) if len(hit) else None

    # --- 패턴 (지금까지의 데이터로 본 라벨) ---
    # Rocket Ship / Stair Stepper 는 IPO-AP 의 모양(크기·속도)과 '베이스 이탈 없음' 으로, One-Hit Wonder 는
    # 빠른 급등 뒤 IPO-AP 전체 되돌림(기준가 아래 마감) 으로, Late Bloomer 는 그 사이(정체·−40%) 로 가른다.
    fast = ap_exists and (peak_i - ap_start) <= p.big_ap_weeks
    fast_big = ap_exists and ap_gain >= p.big_ap_gain and fast
    rocket = ap_exists and ap_gain >= p.rocket_gain and (peak_i - ap_start) <= p.rocket_weeks
    if not ap_exists:
        if iap_start is not None:
            pattern, note = "Pump and Dump", f"IPO-AF → I-DDP {iap_start - iddp_start}주 → 성숙 베이스 돌파"
        elif n >= 52:
            pattern, note = "Disappointment", "IPO-AP 없음, 1년+ 동안 성숙 베이스 돌파 없음"
        else:
            pattern, note = "미분류(IPO-AF 진행 중)", f"상장 {n}주, IPO-AP 없음"
    elif rocket and undercut_i is None:
        pattern, note = "Rocket Ship", f"IPO-AP {ap_gain:+.0%} / {peak_i - ap_start}주, 구조 이탈 없음" + (
            f" (이후 −40% 조정 {af_i}주차)" if af_i is not None else "")
    elif af_i is None:
        pattern, note = "Stair Stepper", f"IPO-AP {ap_gain:+.0%}, 베이스 이탈 없이 {len(bases)}개 베이스"
    else:
        iddp_len = (iap_start if iap_start is not None else n - 1) - af_i
        round_trip = undercut_i is not None
        if fast_big and round_trip:
            if iap_start is not None:
                pattern, note = "One-Hit Wonder Plus", f"IPO-AP {ap_gain:+.0%} / {peak_i - ap_start}주 후 전체 되돌림, I-DDP {iddp_len}주 → 성숙 베이스 돌파"
            else:
                pattern, note = "One-Hit Wonder", f"IPO-AP {ap_gain:+.0%} / {peak_i - ap_start}주 후 전체 되돌림, I-DDP {iddp_len}주 진행 중"
        elif iap_start is not None:
            pattern, note = "Late Bloomer", (f"IPO-AP {ap_gain:+.0%} → I-DDP {iddp_len}주 → 성숙 베이스 돌파"
                                           + ("" if iddp_len >= p.late_bloomer_min_weeks else " (책 기준 1년보다 짧음)"))
        else:
            pattern, note = "Late Bloomer 후보(I-DDP 진행 중)", f"IPO-AP {ap_gain:+.0%} 후 실패, I-DDP {iddp_len}주 진행 중"

    # --- 주별 단계 ---
    ph = np.array(["I-AP"] * n, dtype=object)
    if ipo_base is not None:
        ph[ipo_base.start_i: ipo_base.end_i + 1] = "IPO-BASE"
    if ap_exists:
        ph[ap_start: peak_i + 1] = "IPO-AP"
        if af_i is not None:
            ph[peak_i + 1: af_i + 1] = "IPO-AF"
            ph[af_i + 1:] = "I-DDP"
    else:
        ph[ap_start:] = "IPO-AF"
        ph[min(n, ap_start + 8):] = "I-DDP"
    if iap_start is not None:
        ph[iap_start:] = "I-AP"
    phases = pd.Series(ph, index=w.index)
    return Lifecycle(gray=gray, offer=offer_price, ipo_base=ipo_base, ap_start_i=ap_start, ap_ref=ap_ref,
                     ap_peak_i=peak_i, ap_peak=runmax, ap_gain=float(ap_gain), ap_exists=bool(ap_exists), af_i=af_i, undercut_i=undercut_i,
                     iddp_start_i=iddp_start if (ap_exists and af_i is not None) or not ap_exists else None,
                     mature_base=mature, iap_start_i=iap_start, turbulence=float(turbulence),
                     turbulence_cleared_i=cleared, pattern=pattern, pattern_note=note, phases=phases, bases=bases)


# --------------------------------------------------------------------------- #
# 매수 지점 (5장 4종)
# --------------------------------------------------------------------------- #
@dataclass
class BuyPoint:
    kind: str
    signal_date: pd.Timestamp      # 돌파 확인 일봉(종가) — 진입은 다음 거래일 시가
    level: float                   # 돌파 기준가(베이스 고점) 또는 갭 전일 종가
    base: Base | None
    note: str = ""


def _base_buy(fr: Frames, b: Base | None, kind: str) -> BuyPoint | None:
    if b is None or b.breakout_i is None:
        return None
    day = breakout_day(fr.d, b.high, fr.w.index[b.end_i])
    if day is None:
        return None
    return BuyPoint(kind, day, b.high, b, f"{b.weeks}주 베이스 폭 {b.depth:.0%}, 돌파가 {b.high:.2f}")


def find_buy_points(fr: Frames, lc: Lifecycle, p: LifecycleParams = LifecycleParams()) -> list[BuyPoint]:
    out: list[BuyPoint] = []
    bp = _base_buy(fr, lc.ipo_base, "IPO base")
    if bp:
        out.append(bp)
    limit = lc.iddp_start_i if lc.iddp_start_i is not None else len(fr.w)
    initial = next((b for b in lc.bases if b is not lc.ipo_base and b.breakout_i is not None and b.end_i < limit
                    and (b.start_i > p.ipo_base_max_weeks or b.weeks > p.ipo_base_max_len)
                    and b is not lc.mature_base), None)
    bp = _base_buy(fr, initial, "initial base")
    if bp:
        out.append(bp)
    bp = _base_buy(fr, lc.mature_base, "first mature base")
    if bp:
        out.append(bp)
    # breakaway gap: 전일 종가 대비 시가·종가 +gap, 거래량 20일 평균 ×1.5, 최근 베이스 고점 위
    d = fr.d
    prev_c = d["Close"].shift(1)
    vol20 = d["Volume"].rolling(20, min_periods=5).mean().shift(1)
    gap = (d["Open"] / prev_c - 1 >= p.gap_pct) & (d["Close"] / prev_c - 1 >= p.gap_pct) & (d["Volume"] >= p.gap_vol_mult * vol20)
    for day in d.index[gap.fillna(False).values]:
        recent = [b for b in lc.bases if fr.w.index[b.end_i] < day]
        if not recent:
            continue
        b = recent[-1]
        if float(d.at[day, "Close"]) > b.high:
            out.append(BuyPoint("breakaway gap", day, float(prev_c[day]), b,
                                f"갭 {d.at[day, 'Open'] / prev_c[day] - 1:+.0%}, 베이스 고점 {b.high:.2f} 위 마감"))
            break
    return out


# --------------------------------------------------------------------------- #
# 매도 규칙 시뮬레이션 ($100K, 책 5장 방식)
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    rule: str
    kind: str
    entry_date: date
    entry_px: float
    exit_date: date | None
    avg_exit: float | None
    pct_gain: float
    drawdown_k: float          # 보유 중 계좌 피크 대비 최대 하락 ($K, $100K 기준)
    peak_retained: float | None
    days: int
    reason: str
    sells: list[tuple[date, float, float, str]] = field(default_factory=list)   # (날짜, 가격, 비중, 사유)
    open_position: bool = False


def simulate_rule(fr: Frames, entry_signal: pd.Timestamp, rule: str, kind: str,
                  p: LifecycleParams = LifecycleParams(), max_days: int = 756, capital: float = 100_000.0,
                  stop_pct: float | None = None) -> Trade | None:
    d = fr.d
    pos = d.index.searchsorted(entry_signal, side="right")   # 신호일 다음 봉
    if pos >= len(d):
        return None
    stop_pct = p.stop_pct if stop_pct is None else stop_pct
    O, H, L, C, V = (d[c].values for c in ("Open", "High", "Low", "Close", "Volume"))
    E21, S50, WK, S10, S40 = d["EMA21"].values, d["SMA50"].values, d["wk_end"].values, d["SMA10w"].values, d["SMA40w"].values
    i0 = pos
    entry_px = float(O[i0]) * (1 + p.slip)
    shares = capital / entry_px
    remaining, cash = 1.0, 0.0
    peak_eq, max_dd = capital, 0.0
    pending: list[tuple[float, str]] = []
    sells: list[tuple[date, float, float, str]] = []
    peak_close = entry_px
    asc_half = asc_q = False
    mid_below, mid_level = 0, None
    week_peak_close = entry_px
    armed = False
    last_i = i0
    end = min(len(d), i0 + max_days)
    exit_reason = ""

    def _sell(i: int, frac: float, px: float, why: str) -> None:
        nonlocal remaining, cash
        frac = min(frac, remaining)
        if frac <= 1e-9:
            return
        cash += shares * frac * px
        remaining -= frac
        sells.append((d.index[i].date(), px, frac, why))

    for i in range(i0, end):
        last_i = i
        # 1) 전일 종가 신호 → 오늘 시가 체결
        if pending and i > i0:
            for frac, why in pending:
                _sell(i, frac, float(O[i]) * (1 - p.slip), why)
            pending = []
            if remaining <= 1e-9:
                exit_reason = sells[-1][3]
                break
        # 2) 장중 트리거 (Midterm 중기 레벨, MCP 스톱)
        if rule == "Midterm" and mid_level is not None and L[i] <= mid_level:
            _sell(i, 1.0, mid_level * (1 - p.slip), "Midterm 중기: 10주선 2주 하향 후 저점 −6%")
        if rule == "MCP" and peak_close >= entry_px * (1 + p.mcp_arm_gain):
            lvl = entry_px + (peak_close - entry_px) * p.mcp
            if L[i] <= lvl and i > i0:
                _sell(i, 1.0, lvl * (1 - p.slip), f"MCP {p.mcp:.0%} 스톱")
        if remaining <= 1e-9:
            exit_reason = sells[-1][3]
            break
        # 3) 종가 평가
        eq = cash + shares * remaining * float(C[i])
        peak_eq = max(peak_eq, eq)
        max_dd = max(max_dd, peak_eq - eq)
        peak_close = max(peak_close, float(C[i]))
        # 4) 종가 신호
        if C[i] < entry_px * (1 - stop_pct) and not sells:
            pending.append((1.0, f"손절 −{stop_pct:.0%}"))
            continue
        held = i - i0
        if rule == "Ascender":
            if not asc_half and C[i] < E21[i] * 0.97:
                asc_half = True
                pending.append((0.5, "Ascender ½: 종가 < 21일 EMA −3%"))
            if not asc_q and C[i] < S50[i] * 0.97:
                asc_q = True
                pending.append((0.25, "Ascender ¼: 종가 < 50일 SMA −3%"))
            if C[i] < entry_px and (asc_half or asc_q):
                pending.append((1.0, "Ascender 마지막 ¼: 매수가 이탈"))
            elif C[i] >= entry_px * 6:
                pending.append((1.0, "Ascender 마지막 ¼: +500%"))
            elif held >= p.ascender_max_days:
                pending.append((1.0, "Ascender: 18개월 보유 종료"))
        elif rule == "Midterm" and WK[i]:
            week_peak_close = max(week_peak_close, float(C[i]))
            if C[i] < S10[i]:
                mid_below += 1
                if mid_below >= 2 and mid_level is None:
                    lows = L[max(i0, i - 9): i + 1]
                    mid_level = float(np.min(lows[-10:])) * 0.94
            else:
                mid_below, mid_level = 0, None
            if held >= p.midterm_long_days and C[i] < S10[i]:
                pending.append((1.0, "Midterm 장기: 1년 보유 후 주봉 종가 < 10주선"))
            elif C[i] <= week_peak_close * 0.70:
                pending.append((1.0, "Midterm: 고점 −30% 주봉 마감"))
            elif C[i] < S40[i]:
                pending.append((1.0, "Midterm: 주봉 종가 < 40주선"))
        elif rule == "40-Week" and WK[i]:
            if C[i] < S40[i] * 0.99:
                pending.append((1.0, "40-Week: 금요일 종가 < 40주선 −1%"))
        elif rule == "Everest":
            if not armed and i >= i0 + p.everest_run_days:
                hi60 = float(np.max(C[max(0, i - 60): i + 1]))
                if (C[i] / E21[i] - 1 >= p.everest_ext and C[i] >= hi60
                        and C[i] / C[i - p.everest_run_days] - 1 >= p.everest_run):
                    armed = True
            if armed and i >= 2:
                gapped_prev = O[i - 1] > H[i - 2]
                ref_low = L[i - 1] if gapped_prev else L[i - 2]
                if C[i] < ref_low:
                    pending.append((1.0, "Everest: 포물선 후 2일 전 저가 아래 마감" + (" (갭일 저가)" if gapped_prev else "")))
    # 마감 처리
    open_pos = remaining > 1e-9
    if open_pos:
        px_last = float(C[last_i])
        final = cash + shares * remaining * px_last
        exit_date = d.index[last_i].date()
        if not exit_reason:
            exit_reason = "open(데이터 끝)" if last_i >= len(d) - 1 else "time(최대 보유)"
    else:
        final = cash
        exit_date = sells[-1][0]
    profit = final - capital
    peak_profit = peak_eq - capital
    avg_exit = (cash / (shares * (1 - remaining))) if (1 - remaining) > 1e-9 else None
    return Trade(rule=rule, kind=kind, entry_date=d.index[i0].date(), entry_px=round(entry_px, 4),
                 exit_date=exit_date, avg_exit=round(avg_exit, 4) if avg_exit else None,
                 pct_gain=profit / capital, drawdown_k=max_dd / 1000.0,
                 peak_retained=(profit / peak_profit) if peak_profit > 0 else None,
                 days=(exit_date - d.index[i0].date()).days, reason=exit_reason, sells=sells, open_position=open_pos)


# --------------------------------------------------------------------------- #
# 6장 연구 재현 — IPO 1,679건 통계
# --------------------------------------------------------------------------- #
@dataclass
class IpoStats:
    symbol: str
    ipo_date: date | None
    n_weeks: float
    day1_close: float
    day1_low: float
    weeks_to_double: float | None
    undercut_close_before_double: bool | None
    weeks_undercut_low: float | None
    weeks_stop10: float | None
    corrected50_after_double: bool | None   # 26주 내 2배 달성 종목이 이후 52주 안에 고점 −50%
    max_gain: float
    ath_within_10d_never_higher: bool
    liquid: bool                            # 전 기간 평균 거래대금 ≥ $20M
    dollar_vol: float


def ipo_stats(fr: Frames, symbol: str) -> IpoStats | None:
    if not fr.has_day1:
        return None
    d = fr.d
    c0, l0 = float(d["Close"].iloc[0]), float(d["Low"].iloc[0])
    wk = (d.index - d.index[0]).days / 7.0
    C, Lo = d["Close"].values, d["Low"].values
    dbl = np.where(C[1:] >= 2 * c0)[0]
    w_dbl = float(wk[dbl[0] + 1]) if len(dbl) else None
    under_before = bool((C[1: dbl[0] + 1] < c0).any()) if len(dbl) else None
    uc = np.where(Lo[1:] < l0)[0]
    st = np.where(Lo[1:] < 0.9 * l0)[0]
    corrected = None
    if w_dbl is not None and w_dbl <= 26:
        j = dbl[0] + 1
        seg = C[j: j + 253]
        corrected = bool((seg <= 0.5 * np.maximum.accumulate(seg)).any())
    ath_i = int(np.argmax(C))
    dvol = float((d["Close"] * d["Volume"]).mean())
    return IpoStats(symbol=symbol, ipo_date=fr.ipo_date, n_weeks=float(wk[-1]), day1_close=c0, day1_low=l0,
                    weeks_to_double=w_dbl, undercut_close_before_double=under_before,
                    weeks_undercut_low=float(wk[uc[0] + 1]) if len(uc) else None,
                    weeks_stop10=float(wk[st[0] + 1]) if len(st) else None, corrected50_after_double=corrected,
                    max_gain=float(C.max() / c0 - 1), ath_within_10d_never_higher=bool(ath_i <= 10 and len(C) > 10),
                    liquid=dvol >= LIQUIDITY_USD, dollar_vol=dvol)


def cohort_tables(stats: list[IpoStats]) -> dict[str, pd.DataFrame]:
    """책 Table 6.1 / Fig 6.1 / Fig 6.2 / Table 6.2 / 유동성 텔 / 80쪽 요약을 같은 형식으로 — '책' 열은 1982~2017 원문 수치."""
    if not stats:
        return {}
    df = pd.DataFrame([s.__dict__ for s in stats])
    n = len(df)

    def _share(mask: pd.Series, elig: pd.Series) -> tuple[int, float]:
        k = int(elig.sum())
        return k, (float((mask & elig).sum()) / k if k else float("nan"))

    rows = []
    for label, h, book in (("4주", 4, 0.013), ("13주", 13, 0.06), ("52주", 52, 0.20), ("4년", 208, 0.40), ("10년", 520, 0.51)):
        k, v = _share(df["weeks_to_double"].fillna(1e9) <= h, df["n_weeks"] >= h)
        rows.append({"첫날 종가 대비 +100% 달성": label, "대상 n": k, "비율": v, "책(1982~2017)": book})
    t61 = pd.DataFrame(rows)
    rows = []
    for label, h, book in (("3주", 3, 0.55), ("39주", 39, 0.82), ("208주+", 208, 0.91)):
        k, v = _share(df["weeks_undercut_low"].fillna(1e9) <= h, df["n_weeks"] >= h)
        rows.append({"첫날 저가 이탈": label, "대상 n": k, "비율": v, "책": book})
    f61 = pd.DataFrame(rows)
    rows = []
    for label, h, book in (("2주", 2, 0.27), ("9주", 9, 0.50), ("208주+", 208, 0.86)):
        k, v = _share(df["weeks_stop10"].fillna(1e9) <= h, df["n_weeks"] >= h)
        rows.append({"첫날 저가 −10% 손절 발동": label, "대상 n": k, "비율": v, "책": book})
    f62 = pd.DataFrame(rows)
    rows = []
    elig = df["n_weeks"] >= 78
    for label, lo, hi, book_pct, book_corr in (("≤8주", 0, 8, 0.033, 0.86), ("8~13주", 8, 13, 0.027, 0.64),
                                               ("13~26주", 13, 26, 0.056, 0.67)):
        m = (df["weeks_to_double"].fillna(1e9) > lo) & (df["weeks_to_double"].fillna(1e9) <= hi) & elig
        k = int(m.sum())
        corr = df.loc[m, "corrected50_after_double"]
        rows.append({"2배 달성 시점": label, "종목 수": k, "코호트 비율": k / max(int(elig.sum()), 1), "책 비율": book_pct,
                     "이후 52주 내 −50% 조정": float(corr.fillna(False).mean()) if k else float("nan"), "책 조정 비율": book_corr})
    m = (df["weeks_to_double"].fillna(1e9) <= 26) & elig
    k = int(m.sum())
    corr = df.loc[m, "corrected50_after_double"]
    rows.append({"2배 달성 시점": "합계 ≤26주", "종목 수": k, "코호트 비율": k / max(int(elig.sum()), 1), "책 비율": 0.116,
                 "이후 52주 내 −50% 조정": float(corr.fillna(False).mean()) if k else float("nan"), "책 조정 비율": 0.72})
    t62 = pd.DataFrame(rows)
    liq = df["liquid"]
    g = df["max_gain"]
    fast = df["weeks_to_double"].fillna(1e9) <= 13
    summary = pd.DataFrame([
        {"항목": "상장 후 1년 내 +100%", "값": t61.iloc[2]["비율"], "책": 0.20},
        {"항목": "13주 내 +100%", "값": t61.iloc[1]["비율"], "책": 0.06},
        {"항목": "상장 10일 안에 ATH 찍고 다시는 못 넘음", "값": float(df["ath_within_10d_never_higher"].mean()), "책": 0.11},
        {"항목": "≥ +50% (ATH)", "값": float((g >= 0.5).mean()), "책": 0.69},
        {"항목": "≥ +100% (ATH)", "값": float((g >= 1.0).mean()), "책": 0.51},
        {"항목": "≥ +200% (ATH)", "값": float((g >= 2.0).mean()), "책": 0.33},
        {"항목": "≥ +400% (ATH)", "값": float((g >= 4.0).mean()), "책": 0.195},
        {"항목": "≥ +900% (ATH)", "값": float((g >= 9.0).mean()), "책": 0.10},
        {"항목": "≥ +2000% (ATH)", "값": float((g >= 20.0).mean()), "책": 0.052},
        {"항목": "거래대금 ≥ $20M/일 (전 기간 평균)", "값": float(liq.mean()), "책": 0.28},
        {"항목": "  유동 종목 중 ≥ +100%", "값": float((g[liq] >= 1.0).mean()) if liq.any() else float("nan"), "책": 0.75},
        {"항목": "  ≥ +200% 종목 중 유동 비율", "값": float(liq[g >= 2.0].mean()) if (g >= 2.0).any() else float("nan"), "책": 0.58},
        {"항목": "  P(≥+200% | 유동) / P(≥+200%)", "값": (float((g[liq] >= 2.0).mean()) / float((g >= 2.0).mean()))
            if liq.any() and (g >= 2.0).any() else float("nan"), "책": 2.0},
        {"항목": "90일 내 +100% 종목 중 ≥ +500%", "값": float((g[fast] >= 5.0).mean()) if fast.any() else float("nan"), "책": None},
        {"항목": "  그 외 종목 중 ≥ +500%", "값": float((g[~fast] >= 5.0).mean()) if (~fast).any() else float("nan"), "책": None},
        {"항목": "  배수 (책: 2배)", "값": (float((g[fast] >= 5.0).mean()) / float((g[~fast] >= 5.0).mean()))
            if fast.any() and (~fast).any() and (g[~fast] >= 5.0).any() else float("nan"), "책": 2.0},
        {"항목": "결국 2배 된 종목 중 그 전에 첫날 종가 아래로 안 간 비율", "값": float((~df.loc[df["weeks_to_double"].notna(), "undercut_close_before_double"].astype(bool)).mean())
            if df["weeks_to_double"].notna().any() else float("nan"), "책": 0.032},
    ])
    return {"table_6_1": t61, "fig_6_1": f61, "fig_6_2": f62, "table_6_2": t62, "summary": summary, "n": pd.DataFrame([{"n": n}])}


# --------------------------------------------------------------------------- #
# 7장 스크린 (기준일)
# --------------------------------------------------------------------------- #
@dataclass
class ScreenRow:
    symbol: str
    ipo_date: date | None
    age_weeks: int
    close: float
    dollar_vol_m: float
    above_52w_low: float
    doubled_in_90d: bool
    revenue_yoy: float | None
    ipo_alert: bool
    rare_jewel: bool | None
    liquidity_matters: bool | None
    phase: str
    pattern: str
    base_status: str
    breakout_this_week: bool
    dist_to_base_high: float | None
    vs_gray: float
    vs_sma10: float
    vs_sma40: float


def screen_ticker(symbol: str, fr: Frames, lc: Lifecycle, revenue_yoy: float | None,
                  p: LifecycleParams = LifecycleParams()) -> ScreenRow:
    d, w = fr.d, fr.w
    close = float(d["Close"].iloc[-1])
    dvol = float(d["DVOL20"].iloc[-1])
    low52 = float(d["Low"].iloc[-252:].min())
    above = close / low52 - 1
    age = len(w)
    C = d["Close"].values
    doubled = False
    if fr.has_day1:
        first = C[: 64]
        doubled = bool(first.max() >= 2 * C[0])
    if not doubled and len(C) > 63:
        r63 = pd.Series(C).pct_change(63)
        doubled = bool((r63 >= 1.0).any())
    liquid = dvol >= LIQUIDITY_USD
    young = age <= 156
    ipo_alert = young and above >= 0.25 and liquid
    rev_ok = None if revenue_yoy is None else revenue_yoy >= 0.5
    rare = (doubled and liquid and rev_ok) if rev_ok is not None else (None if (doubled and liquid) else False)
    liqm = (young and liquid and rev_ok) if rev_ok is not None else (None if (young and liquid) else False)
    last = lc.bases[-1] if lc.bases else None
    n = len(w)
    bo_week = last is not None and last.breakout_i == n - 1
    forming = last is not None and last.breakout_i is None and last.end_i >= n - 2
    if bo_week:
        status = f"이번 주 돌파 ({last.weeks}주 베이스, 폭 {last.depth:.0%})"
    elif forming:
        status = f"베이스 형성 중 {last.weeks}주, 폭 {last.depth:.0%}, 고점 {last.high:.2f}"
    else:
        status = "베이스 없음"
    dist = (last.high / close - 1) if (last is not None and forming) else None
    return ScreenRow(symbol=symbol, ipo_date=fr.ipo_date, age_weeks=age, close=close, dollar_vol_m=dvol / 1e6,
                     above_52w_low=above, doubled_in_90d=doubled, revenue_yoy=revenue_yoy, ipo_alert=ipo_alert,
                     rare_jewel=rare, liquidity_matters=liqm, phase=lc.phase_now, pattern=lc.pattern,
                     base_status=status, breakout_this_week=bool(bo_week), dist_to_base_high=dist,
                     vs_gray=close / lc.gray - 1, vs_sma10=close / float(w["SMA10"].iloc[-1]) - 1,
                     vs_sma40=close / float(w["SMA40"].iloc[-1]) - 1)


# --------------------------------------------------------------------------- #
# 집계 (Table 5.1 / 5.2)
# --------------------------------------------------------------------------- #
BOOK_TABLE_5_2 = pd.DataFrame([
    {"rule": "Ascender", "책 중앙값 수익": 0.51, "책 중앙값 DD $K": 57, "책 피크이익 보전": 0.48, "책 보유일": 503},
    {"rule": "Midterm", "책 중앙값 수익": 0.70, "책 중앙값 DD $K": 66, "책 피크이익 보전": 0.45, "책 보유일": 201},
    {"rule": "40-Week", "책 중앙값 수익": 0.81, "책 중앙값 DD $K": 102, "책 피크이익 보전": 0.41, "책 보유일": 374},
    {"rule": "Everest", "책 중앙값 수익": 0.52, "책 중앙값 DD $K": 33, "책 피크이익 보전": 0.63, "책 보유일": 87},
])


def trades_frame(results: list["TickerResult"]) -> pd.DataFrame:
    rows = []
    for r in results:
        for t in r.trades:
            rows.append({"symbol": r.symbol, "ipo_date": r.ipo.ipo_date if r.ipo else None, "pattern": r.lifecycle.pattern,
                         "kind": t.kind, "rule": t.rule, "entry_date": t.entry_date, "entry_px": t.entry_px,
                         "exit_date": t.exit_date, "pct_gain": t.pct_gain, "drawdown_k": t.drawdown_k,
                         "peak_retained": t.peak_retained, "days": t.days, "reason": t.reason, "open": t.open_position})
    return pd.DataFrame(rows)


def rule_summary(tf: pd.DataFrame) -> pd.DataFrame:
    if tf.empty:
        return tf
    g = tf.groupby("rule")
    out = pd.DataFrame({
        "거래 n": g.size(), "승률": g["pct_gain"].apply(lambda s: float((s > 0).mean())),
        "중앙값 수익": g["pct_gain"].median(), "평균 수익": g["pct_gain"].mean(),
        "중앙값 DD $K": g["drawdown_k"].median(), "중앙값 피크이익 보전": g["peak_retained"].median(),
        "중앙값 보유일": g["days"].median(), "미청산": g["open"].sum(),
    }).reset_index()
    return out.merge(BOOK_TABLE_5_2, on="rule", how="left")


def rule_pattern_table(tf: pd.DataFrame, value: str = "pct_gain") -> pd.DataFrame:
    if tf.empty:
        return tf
    return tf.pivot_table(index="rule", columns="pattern", values=value, aggfunc="median")


def kind_summary(tf: pd.DataFrame) -> pd.DataFrame:
    if tf.empty:
        return tf
    g = tf.groupby(["kind", "rule"])
    return pd.DataFrame({"n": g.size(), "승률": g["pct_gain"].apply(lambda s: float((s > 0).mean())),
                         "중앙값 수익": g["pct_gain"].median(), "중앙값 DD $K": g["drawdown_k"].median(),
                         "중앙값 보유일": g["days"].median()}).reset_index()


# --------------------------------------------------------------------------- #
# 오케스트레이션
# --------------------------------------------------------------------------- #
@dataclass
class TickerResult:
    symbol: str
    ipo: IpoEvent | None
    frames: Frames                 # 전체 히스토리 (시뮬·통계용)
    lifecycle: Lifecycle           # 전체 히스토리 기준 (사후 라벨)
    buy_points: list[BuyPoint]
    trades: list[Trade]
    stats: IpoStats | None
    frames_asof: Frames | None     # 기준일까지 (스크린용)
    lifecycle_asof: Lifecycle | None
    screen: ScreenRow | None


def analyze_ticker(symbol: str, daily: pd.DataFrame, ipo: IpoEvent | None, p: LifecycleParams, as_of: date | None,
                   rules: tuple[str, ...] = RULES, buy_kinds: tuple[str, ...] = BUY_KINDS, max_days: int = 756,
                   revenue_yoy: float | None = None, run_sim: bool = True) -> TickerResult | None:
    ipo_date = ipo.ipo_date if ipo else None
    fr = prepare(daily, ipo_date, None, p)
    if fr is None or len(fr.w) < 4:
        return None
    lc = classify_lifecycle(fr.w, p, ipo.offer_price if ipo else None)
    bps = find_buy_points(fr, lc, p)
    trades: list[Trade] = []
    if run_sim:
        for bp in bps:
            if bp.kind not in buy_kinds:
                continue
            for rule in rules:
                t = simulate_rule(fr, bp.signal_date, rule, bp.kind, p, max_days=max_days)
                if t is not None:
                    trades.append(t)
    st = ipo_stats(fr, symbol)
    fr_a = lc_a = sc = None
    if as_of is not None:
        fr_a = prepare(daily, ipo_date, as_of, p)
        if fr_a is not None and len(fr_a.w) >= 2:
            lc_a = classify_lifecycle(fr_a.w, p, ipo.offer_price if ipo else None)
            sc = screen_ticker(symbol, fr_a, lc_a, revenue_yoy, p)
    return TickerResult(symbol=symbol, ipo=ipo, frames=fr, lifecycle=lc, buy_points=bps, trades=trades, stats=st,
                        frames_asof=fr_a, lifecycle_asof=lc_a, screen=sc)


def fetch_many(market_data, symbols: list[str], start: date, end: date, max_workers: int = 4,
               log: Callable[[str], None] = lambda _m: None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    errors: list[str] = []

    def _fetch(sym: str) -> pd.DataFrame | None:
        try:
            df = fetch_with_retry(market_data, sym, start, end)
            return df if df is not None and len(df) else None
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sym}: {type(e).__name__}")
            return None

    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, (s, df) in enumerate(zip(symbols, ex.map(_fetch, symbols))):
            if df is not None:
                out[s] = df
            if (i + 1) % 50 == 0:
                log(f"  일봉 {i + 1}/{len(symbols)}")
    return out, errors


def ipo_cohort(events: list[IpoEvent], exchanges: tuple[str, ...] = ("NASDAQ", "NYSE"), exclude_spac: bool = True,
               min_offer_price: float | None = 5.0) -> list[IpoEvent]:
    out = []
    for e in events:
        if exchanges and e.exchange_group not in exchanges:
            continue
        if exclude_spac and e.looks_like_spac:
            continue
        if min_offer_price is not None and e.offer_price is not None and e.offer_price < min_offer_price:
            continue
        if e.symbol.endswith(("-W", "-R", "-U", "W", "U")) and len(e.symbol) > 4 and e.symbol[-1] in "WU" and e.symbol[:-1].isalpha():
            continue   # 워런트/유닛 (ABCW, ABCU)
        out.append(e)
    return out


def revenue_yoy_asof(quarters, as_of: date) -> float | None:
    """GrowthSnapshot.quarters → 발표일 ≤ 기준일 최신 분기 매출 YoY (7장 Rare Jewels/Liquidity Matters 의 '매출 50%')."""
    usable = sorted((q for q in quarters if q.effective_report_date <= as_of), key=lambda q: q.fiscal_date)
    if len(usable) < 5:
        return None
    cur, prev = usable[-1].revenue, usable[-5].revenue
    if cur is None or prev in (None, 0):
        return None
    return (cur - prev) / abs(prev)


def cohort_window(as_of: date, years_back: int, min_age_weeks: int) -> tuple[date, date]:
    return as_of - timedelta(days=365 * years_back), as_of - timedelta(weeks=min_age_weeks)
