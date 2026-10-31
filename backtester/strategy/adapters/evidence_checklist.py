"""근거중심매매(온휘) 체크리스트 — 자동 판정 엔진.

책의 규칙 중 가격·거래량만으로 셀 수 있는 것을 계산한다. 채널·피보나치는 스윙 피벗으로
그린 **근사치**라서 화면에서 사람이 확정한다. 설계서:
``docs/strategy_notes/근거중심매매_검증_체크리스트_2026-10.md`` (§2 파라미터, §4 자동 항목, §8 도구 설계).

순수 pandas/numpy. streamlit·데이터 소스에 의존하지 않는다 (페이지가 조립).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from pattern.helpers.pivots import find_swing_highs, find_swing_lows


# ---------------------------------------------------------------------------
# 파라미터 (책이 정한 값 + [가정])
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceParams:
    ema_periods: tuple[int, ...] = (20, 50, 100, 200)   # EMA
    sma_long: int = 365                                   # SMA
    rsi_n: int = 14                                       # [가정] 책 미명시
    overbought: float = 70.0
    oversold: float = 30.0
    pivot_lr: int = 10            # 스윙 피벗 좌우 봉 수 (일봉). 주봉은 절반
    channel_tol: float = 0.03     # [가정] 채널선 접촉 ±3%
    ma_tol: float = 0.02          # 이동평균선 접촉 ±2%
    level_tol: float = 0.015      # 피보나치 레벨 ±1.5%
    box_window: int = 30          # 매물대 후보 창 (봉)
    box_max_range: float = 0.15   # [가정] 박스 높이(몸통 기준) ≤ 15%
    box_min_crosses: int = 3      # [가정] 20MA 교차 ≥ 3회
    box_tol: float = 0.02
    vol_avg_n: int = 20
    strong_wick: float = 2.0      # 꼬리/몸통 ≥ 2 = 강한 신호 (1:2~3)
    fvg_lookback: int = 60


BUY_RETRACE = (0.382, 0.5, 0.618)
REVERSE_BUY = (1.236, 1.272, 1.382, 1.618)
SELL_EXT = (1.272, 1.382, 1.618, 2.618)
ALL_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.236, 1.272, 1.382, 1.618, 2.618)

# 근거가중치 (14장)
WEIGHTS = {"채널": 1.0, "피보나치": 1.25, "주봉MA": 1.0, "다이버전스": 1.0,
           "매물대": 0.75, "거래량": 0.5, "파동": 0.5, "차트패턴": 0.5, "트라이앵글": 0.75}


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI."""
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(ad > 0, 100.0).where(au + ad > 0, 50.0)


def add_indicators(df: pd.DataFrame, p: EvidenceParams = EvidenceParams()) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    c = out["Close"].astype(float)
    for n in p.ema_periods:
        out[f"EMA{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    out[f"SMA{p.sma_long}"] = c.rolling(p.sma_long, min_periods=p.sma_long).mean()
    out["RSI"] = rsi(c, p.rsi_n)
    out["VolAvg"] = out["Volume"].rolling(p.vol_avg_n, min_periods=5).mean()
    return out


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return (df.resample("W-FRI")
              .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
              .dropna(subset=["Close"]))


def to_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return (df.resample("4h")
              .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
              .dropna(subset=["Close"]))


# ---------------------------------------------------------------------------
# 이동평균선 상태 (1장·14장)
# ---------------------------------------------------------------------------

@dataclass
class MAState:
    alignment: str                 # 정배열 / 역배열 / 혼합
    above_ema20: bool
    touching: list[str]            # ±ma_tol 안에 있는 선
    recent_cross: list[str]        # 최근 10봉 내 골든/데드크로스
    below_200: bool
    below_365: bool | None
    converged: bool                # 5선 폭/종가 < 3%
    text: str


def ma_state(df: pd.DataFrame, p: EvidenceParams = EvidenceParams(), lookback: int = 10) -> MAState:
    last = df.iloc[-1]
    c = float(last["Close"])
    names = [f"EMA{n}" for n in p.ema_periods]
    vals = [last.get(n, np.nan) for n in names]
    if any(pd.isna(v) for v in vals):
        align = "데이터 부족"
    elif all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        align = "정배열"
    elif all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        align = "역배열"
    else:
        align = "혼합"
    touching = []
    for n in names + [f"SMA{p.sma_long}"]:
        v = last.get(n, np.nan)
        if not pd.isna(v) and abs(c / v - 1) <= p.ma_tol:
            touching.append(n)
    crosses = []
    tail = df.tail(lookback + 1)
    for a, b in ((20, 50), (20, 100), (50, 100), (50, 200), (100, 200)):
        s = (tail[f"EMA{a}"] > tail[f"EMA{b}"]).astype(int).diff().dropna()
        if (s > 0).any():
            crosses.append(f"골든 {a}/{b}")
        if (s < 0).any():
            crosses.append(f"데드 {a}/{b}")
    s365 = last.get(f"SMA{p.sma_long}", np.nan)
    allv = [v for v in vals + [s365] if not pd.isna(v)]
    converged = bool(allv) and (max(allv) - min(allv)) / c < 0.03
    text = (f"{align} · 종가 {'>' if c > last['EMA20'] else '<'} 20MA"
            + (f" · 접촉 {', '.join(touching)}" if touching else " · 접촉 없음")
            + (f" · 최근 {', '.join(crosses)}" if crosses else "")
            + (" · 5선 수렴" if converged else ""))
    return MAState(align, bool(c > last["EMA20"]), touching, crosses,
                   bool(c < last["EMA200"]) if not pd.isna(last["EMA200"]) else False,
                   None if pd.isna(s365) else bool(c < s365), converged, text)


# ---------------------------------------------------------------------------
# 스윙·채널·피보나치 (2장·4장) — 근사
# ---------------------------------------------------------------------------

def swing_points(df: pd.DataFrame, lr: int) -> pd.DataFrame:
    """피벗 표: columns = pos(정수 위치), price, kind('H'/'L')."""
    sh = find_swing_highs(df, lr, lr)
    sl = find_swing_lows(df, lr, lr)
    rows = []
    for i, (h, l) in enumerate(zip(sh.to_numpy(), sl.to_numpy())):
        if not np.isnan(h):
            rows.append((i, float(h), "H"))
        if not np.isnan(l):
            rows.append((i, float(l), "L"))
    return pd.DataFrame(rows, columns=["pos", "price", "kind"])


@dataclass
class Channel:
    kind: str                      # 고고저 / 저저고
    slope: float                   # 로그가격/봉
    base_pos: int
    base_log: float                # 기준선(고고저=저항선, 저저고=지지선)의 base_pos 에서 로그값
    width_log: float               # 평행선까지의 로그 거리 (양수)
    direction: str                 # 상승 채널 / 하락 채널 / 횡보

    def lines_at(self, pos: int) -> tuple[float, float, float]:
        """(upper, mid, lower) 가격."""
        base = self.base_log + self.slope * (pos - self.base_pos)
        if self.kind == "고고저":
            upper, lower = base, base - self.width_log
        else:
            lower, upper = base, base + self.width_log
        return float(np.exp(upper)), float(np.exp((upper + lower) / 2)), float(np.exp(lower))

    def series(self, n: int) -> pd.DataFrame:
        arr = np.array([self.lines_at(i) for i in range(n)])
        return pd.DataFrame(arr, columns=["upper", "mid", "lower"])


def fit_channel(df: pd.DataFrame, sw: pd.DataFrame, kind: str) -> Channel | None:
    """마지막 두 피벗으로 기준선, 그 사이·이후의 반대쪽 극값으로 평행선. 로그 가격."""
    hs = sw[sw.kind == "H"]
    ls = sw[sw.kind == "L"]
    a, b = (hs, ls) if kind == "고고저" else (ls, hs)
    if len(a) < 2 or len(b) < 1:
        return None
    p1, p2 = a.iloc[-2], a.iloc[-1]
    if p2.pos == p1.pos:
        return None
    y1, y2 = np.log(p1.price), np.log(p2.price)
    slope = (y2 - y1) / (p2.pos - p1.pos)
    span = b[b.pos >= p1.pos]
    if span.empty:
        return None
    # 기준선에서 가장 먼 반대쪽 피벗으로 폭 결정
    dists = [np.log(r.price) - (y1 + slope * (r.pos - p1.pos)) for r in span.itertuples()]
    width = max(-d for d in dists) if kind == "고고저" else max(dists)
    if width <= 0:
        return None
    direction = "상승 채널" if slope > 1e-4 else ("하락 채널" if slope < -1e-4 else "횡보")
    return Channel(kind, float(slope), int(p1.pos), float(y1), float(width), direction)


@dataclass
class ChannelState:
    channels: list[Channel]
    touch: str | None              # 상단/중단/하단/없음 (어느 채널이든)
    position: float | None         # 0=하단 1=상단 (첫 채널 기준)
    direction: str
    quadrant: str
    text: str


def channel_state(df: pd.DataFrame, sw: pd.DataFrame, align: str,
                  p: EvidenceParams = EvidenceParams()) -> ChannelState:
    chans = [c for c in (fit_channel(df, sw, "고고저"), fit_channel(df, sw, "저저고")) if c]
    n = len(df) - 1
    c = float(df["Close"].iloc[-1])
    touch, pos_val, direction = None, None, "채널 없음"
    parts = []
    for ch in chans:
        up, mid, lo = ch.lines_at(n)
        for name, v in (("하단", lo), ("중단", mid), ("상단", up)):
            if abs(c / v - 1) <= p.channel_tol and touch is None:
                touch = name
        if pos_val is None and up > lo:
            pos_val = float((np.log(c) - np.log(lo)) / (np.log(up) - np.log(lo)))
            direction = ch.direction
        parts.append(f"{ch.kind}({ch.direction}) 상 {up:,.2f} / 중 {mid:,.2f} / 하 {lo:,.2f}")
    if direction.startswith("상승") and align == "역배열":
        quad = "상승채널+역배열 (최선)"
    elif direction.startswith("하락") and align == "정배열":
        quad = "하락채널+정배열 (최악)"
    elif direction == "채널 없음":
        quad = "—"
    else:
        quad = f"{direction}+{align}"
    text = (f"접촉 {touch or '없음'}"
            + (f" · 위치 {pos_val:.0%}" if pos_val is not None else "")
            + f" · 4분면 {quad} · " + " | ".join(parts))
    return ChannelState(chans, touch, pos_val, direction, quad, text)


@dataclass
class FibState:
    kind: str                      # 상승 되돌림 / 하락 리버스
    hi: float
    lo: float
    hi_pos: int
    lo_pos: int
    ratio: float | None            # 현재 종가의 되돌림 비율 (0=고점, 1=저점)
    zone: str                      # 정석 매수(0.382~0.618) / 리버스 매수(1.236~1.618) / 매도 레벨 / 없음
    nearest: tuple[float, float] | None   # (비율, 가격)
    levels: dict[float, float]
    text: str


def fib_state(df: pd.DataFrame, sw: pd.DataFrame, p: EvidenceParams = EvidenceParams()) -> FibState | None:
    hs = sw[sw.kind == "H"]
    ls = sw[sw.kind == "L"]
    highs, lows = df["High"].to_numpy(), df["Low"].to_numpy()
    if hs.empty and ls.empty:
        return None
    # 한쪽 피벗이 없으면 그 앞 구간의 극값으로 대신한다
    if hs.empty:
        l = ls.iloc[-1]
        hp = int(np.argmax(highs[: int(l.pos) + 1]))
        hs = pd.DataFrame([(hp, float(highs[hp]), "H")], columns=["pos", "price", "kind"])
    if ls.empty:
        h = hs.iloc[-1]
        lp = int(np.argmin(lows[: int(h.pos) + 1]))
        ls = pd.DataFrame([(lp, float(lows[lp]), "L")], columns=["pos", "price", "kind"])
    h, l = hs.iloc[-1], ls.iloc[-1]
    if l.pos < h.pos:
        # 마지막 피벗이 고점: 그 앞 저점 → 고점 상승 뒤 되돌림 중 (저점이 고점보다 높으면 하락 진행 중)
        before = ls[ls.pos < h.pos]
        if before.empty:
            lp0 = int(np.argmin(lows[: int(h.pos) + 1]))
            before = pd.DataFrame([(lp0, float(lows[lp0]), "L")], columns=["pos", "price", "kind"])
        lo_row = before.iloc[-1]
        if lo_row.price < h.price:
            hi, lo, hp, lp, kind = h.price, lo_row.price, h.pos, lo_row.pos, "상승 되돌림"
        else:
            lp = int(np.argmin(lows[int(h.pos):]) + h.pos)
            hi, lo, hp, kind = h.price, float(lows[lp]), h.pos, "하락 리버스(진행 중)"
    else:
        # 마지막 피벗이 저점: 그 앞 고점 → 저점 하락 (리버스). 고점이 저점보다 낮으면 상승 진행 중
        before = hs[hs.pos < l.pos]
        if before.empty:
            hp0 = int(np.argmax(highs[: int(l.pos) + 1]))
            before = pd.DataFrame([(hp0, float(highs[hp0]), "H")], columns=["pos", "price", "kind"])
        hi_row = before.iloc[-1]
        if hi_row.price > l.price:
            hi, lo, hp, lp, kind = hi_row.price, l.price, hi_row.pos, l.pos, "하락 리버스"
        else:
            hp = int(np.argmax(highs[int(l.pos):]) + l.pos)
            hi, lo, lp, kind = float(highs[hp]), l.price, l.pos, "상승 되돌림(진행 중)"
    if hi <= lo:
        return None
    c = float(df["Close"].iloc[-1])
    rng = hi - lo
    levels = {r: hi - r * rng for r in ALL_RATIOS}           # 되돌림·리버스 (고점=0)
    ext = {r: lo + r * rng for r in SELL_EXT}                 # 위로 확장 (저점=0, 고점=1)
    ratio = (hi - c) / rng
    zone = "없음"
    nearest = None
    if c > hi:
        cand = [(r, v) for r, v in ext.items() if abs(c / v - 1) <= p.level_tol]
        zone = "매도 레벨" if cand else "고점 위(확장 중)"
        nearest = cand[0] if cand else min(ext.items(), key=lambda kv: abs(c - kv[1]))
    else:
        near = [(r, v) for r, v in levels.items() if abs(c / v - 1) <= p.level_tol]
        nearest = near[0] if near else min(levels.items(), key=lambda kv: abs(c - kv[1]))
        if near and near[0][0] in BUY_RETRACE:
            zone = "정석 매수(0.382~0.618)"
        elif near and near[0][0] in REVERSE_BUY:
            zone = "리버스 매수(1.236~1.618)"
        elif near and near[0][0] == 0.236:
            zone = "0.236(초강세장 전용)"
        elif near:
            zone = f"{near[0][0]} 레벨"
    text = (f"{kind}: 고 {hi:,.2f} → 저 {lo:,.2f} · 현재 되돌림 {ratio:.3f} · {zone}"
            + (f" · 가까운 레벨 {nearest[0]} = {nearest[1]:,.2f}" if nearest else ""))
    return FibState(kind, float(hi), float(lo), int(hp), int(lp), float(ratio), zone, nearest,
                    {**levels, **{f"ext{r}": v for r, v in ext.items()}}, text)


# ---------------------------------------------------------------------------
# 매물대 (5장) · FVG (6장)
# ---------------------------------------------------------------------------

@dataclass
class Box:
    start: int
    end: int
    top: float
    bottom: float


def supply_zones(df: pd.DataFrame, p: EvidenceParams = EvidenceParams()) -> list[Box]:
    """몸통 기준 횡보 박스: 창 안 몸통 고저 범위 ≤ box_max_range & 20MA 교차 ≥ box_min_crosses."""
    body_hi = df[["Open", "Close"]].max(axis=1).to_numpy()
    body_lo = df[["Open", "Close"]].min(axis=1).to_numpy()
    close = df["Close"].to_numpy()
    ema = df["EMA20"].to_numpy() if "EMA20" in df else np.full(len(df), np.nan)
    boxes: list[Box] = []
    w = p.box_window
    i = 0
    while i + w <= len(df):
        top, bot = body_hi[i:i + w].max(), body_lo[i:i + w].min()
        seg_c, seg_e = close[i:i + w], ema[i:i + w]
        ok_ema = ~np.isnan(seg_e)
        crosses = int(np.abs(np.diff((seg_c[ok_ema] > seg_e[ok_ema]).astype(int))).sum()) if ok_ema.sum() > 2 else 0
        if bot > 0 and top / bot - 1 <= p.box_max_range and crosses >= p.box_min_crosses:
            if boxes and i <= boxes[-1].end and abs(boxes[-1].top / top - 1) < 0.05:
                boxes[-1] = Box(boxes[-1].start, i + w - 1, max(boxes[-1].top, top), min(boxes[-1].bottom, bot))
            else:
                boxes.append(Box(i, i + w - 1, float(top), float(bot)))
            i += w // 2
        else:
            i += max(1, w // 6)
    return boxes


def supply_state(df: pd.DataFrame, boxes: list[Box], p: EvidenceParams = EvidenceParams()) -> tuple[bool | None, str]:
    """현재 종가가 과거 박스 가장자리(±box_tol)에 있는가. 박스 안(이전에 돌파했던)이면 지지 후보."""
    if not boxes:
        return None, "매물대 후보 없음"
    c = float(df["Close"].iloc[-1])
    n = len(df) - 1
    hits = []
    for b in boxes:
        if b.end >= n - 2:
            continue
        if abs(c / b.top - 1) <= p.box_tol:
            hits.append(f"박스 상단 {b.top:,.2f} 접촉")
        elif abs(c / b.bottom - 1) <= p.box_tol:
            hits.append(f"박스 하단 {b.bottom:,.2f} 접촉")
        elif b.bottom <= c <= b.top:
            hits.append(f"박스 안 {b.bottom:,.2f}~{b.top:,.2f}")
    return (bool(hits), " · ".join(hits)) if hits else (False, f"박스 {len(boxes)}개, 현재가와 무관")


@dataclass
class Fvg:
    pos: int
    bullish: bool
    top: float
    bottom: float
    fill: float    # 0~1


def fvg_list(df: pd.DataFrame, lookback: int = 60) -> list[Fvg]:
    h, l = df["High"].to_numpy(), df["Low"].to_numpy()
    out = []
    n = len(df)
    for i in range(max(2, n - lookback), n):
        if l[i] > h[i - 2]:          # 상승 FVG
            top, bot = l[i], h[i - 2]
            after = l[i + 1:]
            fill = 0.0 if after.size == 0 else float(np.clip((top - after.min()) / (top - bot), 0, 1))
            out.append(Fvg(i - 1, True, float(top), float(bot), fill))
        elif h[i] < l[i - 2]:        # 하락 FVG
            top, bot = l[i - 2], h[i]
            after = h[i + 1:]
            fill = 0.0 if after.size == 0 else float(np.clip((after.max() - bot) / (top - bot), 0, 1))
            out.append(Fvg(i - 1, False, float(top), float(bot), fill))
    return out


def fvg_text(fvgs: list[Fvg]) -> str:
    if not fvgs:
        return "최근 FVG 없음"
    open_ = [f for f in fvgs if f.fill < 1.0]
    parts = [f"FVG {len(fvgs)}개, 미채움 {len(open_)}개"]
    for f in fvgs[-3:]:
        parts.append(f"{'상승' if f.bullish else '하락'} {f.bottom:,.2f}~{f.top:,.2f} 채움 {f.fill:.0%}"
                     + (" ⚠️ 상승 FVG 완전 채움(추세 전환)" if f.bullish and f.fill >= 1.0 else ""))
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# 캔들 (7장)
# ---------------------------------------------------------------------------

@dataclass
class CandleState:
    patterns: list[str]
    bullish: bool
    bearish: bool
    lower_wick_ratio: float
    upper_wick_ratio: float
    above_ema20: bool
    vol_ratio: float | None
    text: str


def _bar(df: pd.DataFrame, i: int) -> dict:
    r = df.iloc[i]
    o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
    body = abs(c - o)
    return dict(o=o, h=h, l=l, c=c, body=body, up=h - max(o, c), dn=min(o, c) - l,
                bull=c > o, rng=h - l)


def candle_state(df: pd.DataFrame, p: EvidenceParams = EvidenceParams()) -> CandleState:
    n = len(df)
    b0 = _bar(df, n - 1)
    b1 = _bar(df, n - 2) if n >= 2 else None
    b2 = _bar(df, n - 3) if n >= 3 else None
    eps = max(b0["rng"], 1e-9) * 0.1
    pats: list[str] = []
    is_doji = lambda b: b["body"] <= 0.1 * max(b["rng"], 1e-9) + 1e-9
    lw = b0["dn"] / b0["body"] if b0["body"] > 0 else (b0["dn"] / eps if b0["dn"] > 0 else 0.0)
    uw = b0["up"] / b0["body"] if b0["body"] > 0 else (b0["up"] / eps if b0["up"] > 0 else 0.0)
    if is_doji(b0):
        pats.append("도지(힘싸움)")
    if lw >= p.strong_wick and b0["dn"] > 2 * b0["up"]:
        pats.append("아래꼬리 강(1:2+)")
    elif lw >= 1.0 and b0["dn"] > 2 * b0["up"]:
        pats.append("아래꼬리 약(1:1)")
    if uw >= p.strong_wick and b0["up"] > 2 * b0["dn"]:
        pats.append("윗꼬리 강(역망치/유성)")
    if lw >= 1.5 and uw >= 1.5:
        pats.append("위아래 꼬리(반전·힘싸움)")
    if b1:
        if b0["bull"] and not b1["bull"] and b0["c"] >= b1["o"] and b0["o"] <= b1["c"] and b0["body"] > b1["body"]:
            pats.append("상승장악형")
        if not b0["bull"] and b1["bull"] and b0["o"] >= b1["c"] and b0["c"] <= b1["o"] and b0["body"] > b1["body"]:
            pats.append("하락장악형")
        if b0["body"] > 0 and b1["body"] > 0 and b0["bull"] != b1["bull"] and 0.8 <= b0["body"] / b1["body"] <= 1.25:
            pats.append("불리쉬 레일로드트랙" if b0["bull"] else "베어리쉬 레일로드트랙")
        if is_doji(b0) and b1["body"] > 0:
            # 장악형 뒤 도지 = 다람쥐
            if b2 and b1["bull"] and not b2["bull"] and b1["c"] >= b2["o"] and b1["o"] <= b2["c"]:
                pats.append("상승다람쥐")
            if b2 and not b1["bull"] and b2["bull"] and b1["o"] >= b2["c"] and b1["c"] <= b2["o"]:
                pats.append("하락다람쥐")
    if b1 and b2:
        if not b2["bull"] and is_doji(b1) and b0["bull"] and b0["c"] > (b2["o"] + b2["c"]) / 2:
            pats.append("모닝스타")
        if b2["bull"] and is_doji(b1) and not b0["bull"] and b0["c"] < (b2["o"] + b2["c"]) / 2:
            pats.append("이브닝스타")
        if all(b["bull"] for b in (b0, b1, b2)) and b0["c"] > b1["c"] > b2["c"]:
            pats.append("백삼병")
        if all(not b["bull"] for b in (b0, b1, b2)) and b0["c"] < b1["c"] < b2["c"]:
            pats.append("흑삼병")
    bull_names = ("아래꼬리 강", "상승장악형", "불리쉬 레일로드트랙", "상승다람쥐", "모닝스타", "백삼병")
    bear_names = ("윗꼬리 강", "하락장악형", "베어리쉬 레일로드트랙", "하락다람쥐", "이브닝스타", "흑삼병")
    bullish = any(pt.startswith(bn) for pt in pats for bn in bull_names)
    bearish = any(pt.startswith(bn) for pt in pats for bn in bear_names)
    ema20 = df["EMA20"].iloc[-1] if "EMA20" in df else np.nan
    above = bool(b0["c"] > ema20) if not pd.isna(ema20) else False
    va = df["VolAvg"].iloc[-1] if "VolAvg" in df else np.nan
    vr = None if pd.isna(va) or va <= 0 else float(df["Volume"].iloc[-1] / va)
    text = (", ".join(pats) if pats else "특이 캔들 없음")
    text += f" · 꼬리/몸통 아래 {lw:.1f} 위 {uw:.1f} · 20MA {'위' if above else '아래'} 마감"
    if vr is not None:
        text += f" · 거래량 {vr:.1f}×"
    if bullish and not above:
        text += " ⚠️ 반전 캔들인데 20MA 아래 마감(신뢰 낮음)"
    return CandleState(pats, bullish, bearish, float(lw), float(uw), above, vr, text)


# ---------------------------------------------------------------------------
# 다이버전스 (3장)
# ---------------------------------------------------------------------------

@dataclass
class DivState:
    rsi: float
    overbought: bool
    oversold: bool
    bearish_count: int        # 연속 하락 다이버전스 수 (2 = 고점 판별 기준)
    bullish: bool
    text: str


def divergence_state(df: pd.DataFrame, sw: pd.DataFrame, p: EvidenceParams = EvidenceParams()) -> DivState:
    r = df["RSI"].to_numpy()
    hs = sw[sw.kind == "H"].tail(3)
    ls = sw[sw.kind == "L"].tail(2)
    bear = 0
    hp = list(hs.itertuples())
    for a, b in zip(hp, hp[1:]):
        if b.price > a.price and r[b.pos] < r[a.pos]:
            bear += 1
        else:
            bear = 0
    bull = False
    if len(ls) == 2:
        a, b = ls.iloc[0], ls.iloc[1]
        bull = bool(b.price < a.price and r[int(b.pos)] > r[int(a.pos)])
    cur = float(r[-1])
    text = f"RSI {cur:.0f}" + (" 과매수" if cur >= p.overbought else "") + (" 과매도" if cur <= p.oversold else "")
    if bear:
        text += f" · 하락 다이버전스 {bear}회 연속" + (" (고점 판별 기준 충족)" if bear >= 2 else " (첫 신호, 경고)")
    if bull:
        text += " · 상승 다이버전스"
    return DivState(cur, cur >= p.overbought, cur <= p.oversold, bear, bull, text)


# ---------------------------------------------------------------------------
# 5대 매매 근거 체크리스트 (14장) + 근거가중치
# ---------------------------------------------------------------------------

@dataclass
class Check:
    ok: bool | None
    why: str


@dataclass
class TimeframeReport:
    label: str
    df: pd.DataFrame
    swings: pd.DataFrame
    ma: MAState
    channel: ChannelState
    fib: FibState | None
    boxes: list[Box]
    supply: Check
    fvgs: list[Fvg]
    candle: CandleState
    div: DivState
    five: dict[str, Check] = field(default_factory=dict)

    @property
    def yes_count(self) -> int:
        return sum(1 for c in self.five.values() if c.ok)


def five_checks(tf: "TimeframeReport", weekly_ma: MAState, p: EvidenceParams = EvidenceParams()) -> dict[str, Check]:
    ch = tf.channel
    c1 = Check(ch.touch in ("중단", "하단") if ch.channels else None,
               f"채널 {ch.touch or '미접촉'} (근사 작도, 사람이 확정)")
    c2 = Check(bool(weekly_ma.touching), "주봉 " + (", ".join(weekly_ma.touching) if weekly_ma.touching else "주요 MA 접촉 없음"))
    cs = tf.candle
    c3 = Check(cs.bullish and cs.above_ema20, cs.text)
    dv = tf.div
    c4 = Check(dv.bullish or dv.oversold, dv.text + (" · 하락 다이버전스 중(불리)" if dv.bearish_count else ""))
    fb = tf.fib
    c5 = Check(None if fb is None else fb.zone.startswith(("정석 매수", "리버스 매수")),
               fb.text if fb else "스윙 부족")
    return {"① 채널 중단·하단 접촉": c1, "② 주봉 주요 이동평균선": c2, "③ 캔들 상승 신호": c3,
            "④ 유리한 다이버전스·과매도": c4, "⑤ 피보나치 주요 구간": c5}


def weighted_score(five: dict[str, Check], supply_ok: bool | None, vol_confirm: bool,
                   pattern_pts: float = 0.0, wave: bool = False) -> tuple[float, list[str]]:
    keys = ["채널", "주봉MA", "다이버전스", "피보나치"]
    oks = [c.ok for c in five.values()]
    # ③ 캔들은 5대 체크리스트에만 들어가고 가중치표에는 없다 (거래량 0.5 가 캔들 보강용)
    pts, parts = 0.0, []
    for k, ok in zip(keys, (oks[0], oks[1], oks[3], oks[4])):
        if ok:
            pts += WEIGHTS[k]
            parts.append(f"{k} {WEIGHTS[k]}")
    if supply_ok:
        pts += WEIGHTS["매물대"]; parts.append("매물대 0.75")
    if vol_confirm and oks[2]:
        pts += WEIGHTS["거래량"]; parts.append("거래량 0.5")
    if pattern_pts:
        pts += pattern_pts; parts.append(f"차트패턴 {pattern_pts}")
    if wave:
        pts += WEIGHTS["파동"]; parts.append("파동 0.5")
    return pts, parts


def verdict(weekly_yes: int, lower_yes: int | None, score: float) -> str:
    if weekly_yes >= 4 or score >= 4:
        return "베팅"
    if lower_yes is not None and lower_yes >= 4:
        return "단기(청산 목적)"
    if weekly_yes >= 2:
        return "보류(근거 쌓는 중)"
    return "매매 안 함"


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------

def build_report(label: str, raw: pd.DataFrame, weekly_ma: MAState | None,
                 p: EvidenceParams = EvidenceParams(), lr: int | None = None) -> TimeframeReport:
    df = add_indicators(raw, p)
    lr = lr or p.pivot_lr
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


def evaluate(daily: pd.DataFrame, as_of: date, hourly: pd.DataFrame | None = None,
             p: EvidenceParams = EvidenceParams()) -> dict[str, TimeframeReport]:
    """기준일까지의 일봉(+시간봉)으로 주봉·일봉·4시간봉 리포트. 미래 봉은 자른다."""
    d = daily.copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx
    d = d[d.index.date <= as_of]
    if len(d) < 60:
        raise ValueError(f"기준일까지 일봉이 {len(d)}개뿐 — 최소 60개 필요")
    w = to_weekly(d)
    out: dict[str, TimeframeReport] = {}
    out["주봉"] = build_report("주봉", w, None, p, lr=max(3, p.pivot_lr // 2))
    wma = out["주봉"].ma
    out["일봉"] = build_report("일봉", d, wma, p)
    if hourly is not None and len(hourly):
        h = hourly.copy()
        hidx = pd.to_datetime(h.index)
        if getattr(hidx, "tz", None) is not None:
            hidx = hidx.tz_localize(None)
        h.index = hidx
        h = h[h.index.date <= as_of]
        h4 = to_4h(h)
        if len(h4) >= 100:
            out["4시간봉"] = build_report("4시간봉", h4, wma, p)
    return out


def forward_result(daily: pd.DataFrame, as_of: date, weeks: tuple[int, ...] = (4, 13, 26)) -> dict[str, float | None]:
    """봉인 해제용: 기준일 종가 대비 N주 뒤 수익률과 그 사이 MDD."""
    d = daily.copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx
    past = d[d.index.date <= as_of]
    if past.empty:
        return {}
    base = float(past["Close"].iloc[-1])
    fut = d[d.index.date > as_of]
    out: dict[str, float | None] = {}
    for wk in weeks:
        seg = fut.iloc[: wk * 5]
        if len(seg) < wk * 5 * 0.8:
            out[f"{wk}주 수익"] = None
            out[f"{wk}주 MDD"] = None
            continue
        out[f"{wk}주 수익"] = float(seg["Close"].iloc[-1] / base - 1)
        eq = seg["Close"] / base
        out[f"{wk}주 MDD"] = float((eq / eq.cummax() - 1).min())
    return out
