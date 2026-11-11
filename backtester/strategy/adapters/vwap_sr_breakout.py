"""VWAP 지지·저항 돌파 (15분봉) — 강의 규칙을 그대로 기계화한 순수 pandas/numpy 엔진.

규칙 출처: docs/strategy_notes/images/vwapcourse/vwapcourse.pdf (Udemy VWAP 강의 화면 + 사용자 메모).
설계서: docs/strategy_notes/VWAP_지지저항_돌파_15분봉_설계_2026-11.md

핵심 규칙
- VWAP 은 **세션(거래일)마다 리셋**되는 누적 VWAP (차트의 흰 선, 날짜가 바뀌면 점프).
- 지지·저항은 두 가지 모양(``sr_mode``): **수평 박스**(직전 ``box_bars`` 봉의 고가 최대·저가 최소) 와 **추세선**(창 안 최근 두 스윙 고점을 이은
  저항선 — 평평하거나 내려오는 것만, 두 스윙 저점을 이은 지지선 — 평평하거나 올라가는 것만; 강의의 하락 쐐기·상승 삼각형).
- 진입(롱): 구조가 **좁고**(높이 ≤ ATR×``max_box_atr``) 저항을 ``min_touches`` 번 이상 두드렸으며, 저항이 VWAP 위에 있고 직전 종가가 VWAP 가까이
  (≤ ATR×``max_vwap_dist_atr``) 있을 때 **종가가 저항 위로 마감**하면 진입. "한 번에 뻥 뚫는" 큰 봉(범위 > ATR×``max_entry_bar_atr``)은 추격 금지. 숏은 대칭.
- 손절: VWAP 아래(``stop_mode="vwap"``) 또는 지지(박스 아랫선·지지 추세선) 아래(``"box"``) 또는 둘 중 더 먼 곳(``"far"``) − ATR×``stop_buffer_atr``.
  강의대로 **종가가 손절선을 넘어 마감할 때만** 손절(윅은 무시, ``stop_on_close``).
- 익절: VWAP 에서 ATR×``tp_min_dist_atr`` 이상 멀어진 뒤 "나갈 이유"가 처음 나오는 봉 마감에 청산 —
  거래량 스파이크(≥ 평균×``tp_volume_mult``) · 첫 역캔들 · 반대쪽 긴 윅(≥ 봉 범위×``tp_wick_frac``) · 큰 봉(≥ ATR×``tp_big_bar_atr``).
- 그 밖에 장 마감 봉 청산(``session_flat``), 최대 보유 봉 수. 한 번에 한 포지션, 청산 뒤 새 돌파에 재진입.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


@dataclass
class VwapSrParams:
    box_bars: int = 16               # 구조(박스·추세선) 탐색 창(봉 수). 15분봉 16 = 4시간
    sr_mode: str = "both"            # box(수평 박스) | trend(스윙 피벗 추세선) | both(둘 중 먼저 뚫리는 것)
    sr_price: str = "body"           # body(캔들 몸통 = max/min(시가,종가)) | wick(고가/저가) — 피벗·박스·터치·포락선에 쓰는 가격. 긴 꼬리 하나가 선을 망치지 않게 몸통이 기본
    pivot_lr: int = 2                # 스윙 고점·저점 좌우 봉 수 (추세선용)
    trend_bars: int = 78             # 추세선 피벗 탐색 창(봉 수) — 박스 창과 별개. 길수록 긴 선이 나온다
    trend_max_pivots: int = 12       # 창 안에서 고려할 최근 피벗 수 (쌍 수 = n(n−1)/2, 속도용)
    max_box_atr: float = 3.0         # 구조 높이 ≤ ATR × 값 (좁을수록 좋다; 16봉 박스는 보통 ATR×3~4 라 1.5 면 거래가 거의 없다)
    min_touches: int = 2             # 저항(지지)선 터치 봉 수
    touch_tol_atr: float = 0.15      # 터치 판정 폭 (ATR 배)
    max_vwap_dist_atr: float = 1.5   # 직전 종가 ↔ VWAP 거리 ≤ ATR × 값 ("VWAP 에 최대한 가까이 진입")
    max_entry_bar_atr: float = 2.0   # 돌파 봉 범위 ≤ ATR × 값 (한 번에 뻥 뚫는 봉은 추격 금지)
    breakout_volume_mult: float = 1.0  # 돌파 봉 거래량 ≥ 평균 × 값 (0 = 미사용)
    atr_bars: int = 20
    vol_avg_bars: int = 20
    allow_long: bool = True
    allow_short: bool = True
    stop_mode: str = "far"           # vwap | box(지지 구조 밖) | far(둘 중 더 먼 곳 = 지지선과 VWAP 모두 아래; 강의 "지지가 있으면 지지 밑, 없으면 VWAP 밑")
    stop_buffer_atr: float = 0.1
    stop_on_close: bool = True       # 종가 이탈 시만 손절 (윅 무시)
    trail_vwap: bool = False         # 손절선을 VWAP 따라 끌어올림
    tp_min_dist_atr: float = 1.5     # VWAP 에서 이만큼 멀어진 뒤부터 익절 신호 탐색 (ATR 기준)
    tp_min_dist_pct: float = 0.3     # 같은 조건의 가격 % 기준 — 둘 중 큰 쪽. 시간외 봉은 ATR 이 작아져 2달러만 위여도 "멀어짐"이 되던 것을 막는다
    tp_min_bars: int = 2             # 진입 뒤 이만큼 지나야 익절 신호를 본다
    tp_opp_body_atr: float = 0.5     # 역캔들은 몸통 ≥ ATR × 값 인 것만 (작은 음봉 하나에 나가지 않게)
    tp_volume_mult: float = 2.5      # 거래량 스파이크
    tp_opposite_candle: bool = True  # 첫 역캔들
    tp_wick_frac: float = 0.5        # 반대쪽 윅 ≥ 봉 범위 × 값 (0 = 미사용)
    tp_big_bar_atr: float = 2.5      # 큰 봉 (0 = 미사용)
    session_flat: str = "weekend"    # off | daily(매일 마지막 봉) | weekend(다음 봉까지 하루 넘게 비는 주말·휴일 전 마지막 봉만). bool 도 받는다(True=daily)
    max_hold_bars: int = 0           # 0 = 제한 없음
    fill: str = "close"              # close(신호 봉 종가) | next_open(다음 봉 시가)
    fee_bps: float = 5.0             # 편도 비용 bp (수수료+슬리피지)
    risk_pct: float = 1.0            # 1거래 위험 = 자본 × 값 % (0 = 전액 투입)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VwapSrResult:
    frame: pd.DataFrame                 # 입력 + vwap/atr/box·trend 열/long_sig/short_sig/equity
    trades: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def session_vwap(df: pd.DataFrame) -> pd.Series:
    """거래일마다 리셋되는 누적 VWAP (typical price × 거래량)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    day = _day_key(df.index)
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return (pv / vv).ffill()


def _day_key(idx: pd.DatetimeIndex) -> np.ndarray:
    loc = idx.tz_convert("America/New_York") if idx.tz is not None else idx
    return loc.normalize().values


def atr_series(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def _touches(x: np.ndarray, level: np.ndarray, tol: np.ndarray, n: int, side: int) -> np.ndarray:
    """각 봉 t 에 대해 창 x[t-n+1..t] 중 level[t] 에 tol 안으로 닿은 봉 수 (side +1: 고가가 level−tol 이상, −1: 저가가 level+tol 이하)."""
    T = len(x); out = np.zeros(T)
    for t in range(n - 1, T):
        lv, tl = level[t], tol[t]
        if np.isnan(lv) or np.isnan(tl):
            continue
        w = x[t - n + 1:t + 1]
        out[t] = np.sum(w >= lv - tl) if side > 0 else np.sum(w <= lv + tl)
    return out


def _swing_mask(x: np.ndarray, lr: int, is_high: bool) -> np.ndarray:
    """엄격한 국소 극값 (좌우 lr 봉보다 크거나/작거나)."""
    T = len(x); m = np.zeros(T, dtype=bool)
    if T < 2 * lr + 1:
        return m
    core = np.ones(T - 2 * lr, dtype=bool)
    c = x[lr:T - lr]
    for k in range(1, lr + 1):
        a, b = x[lr - k:T - lr - k], x[lr + k:T - lr + k]
        core &= (c > a) & (c > b) if is_high else (c < a) & (c < b)
    m[lr:T - lr] = core
    return m


def trend_lines(H: np.ndarray, L: np.ndarray, atr_prev: np.ndarray, p: VwapSrParams) -> dict[str, np.ndarray]:
    """각 봉 t 에 대해 직전 창 [t−trend_bars, t−1] 안의 **확정된** 스윙 고점 쌍 중 가장 좋은 저항선, 스윙 저점 쌍 중 가장 좋은 지지선의 t 에서의 값.

    후보 = 최근 ``trend_max_pivots`` 개 피벗의 모든 쌍 (j1<j2). 저항선은 평평하거나 내려오는 것만(두 피벗 사이 상승 ≤ tol), 지지선은 평평하거나
    올라가는 것만. 첫 피벗부터 t−1 까지 고가가 선 위로 tol 이상 튀어나오면 무효(상단 포락선). 유효한 쌍 중 터치 ≥ ``min_touches`` 인 것 가운데
    **지금 가격에 가장 가까운 선**(저항은 t 에서의 값이 가장 낮은 것, 지지는 가장 높은 것)을 고르되, 그 선과 **겹치는**(t 에서의 값 차이 ≤ tol) 선이 여럿이면
    그 중 **리테스트(터치)가 가장 많은 선**, 같으면 더 긴 선을 고른다 — 봉이 실제로 먼저 부딪히는 "바로 위 저항" 중 가장 많이 확인된 선. (처음엔 최근 두 피벗만 이어 16봉마다 끊겼고, 그 다음엔 터치 최대를 골라 멀리 있는 완만한 선이 뽑혀 돌파를 놓쳤다.)
    반환: res(값), res_slope(봉당), res_i(첫 피벗 index), res_v(첫 피벗 값), res_touch, sup/… 동일.
    """
    T, W, lr = len(H), p.trend_bars, p.pivot_lr
    sh, sl = _swing_mask(H, lr, True), _swing_mask(L, lr, False)
    out = {k: np.full(T, np.nan) for k in ("res", "res_slope", "res_i", "res_v", "res_touch", "sup", "sup_slope", "sup_i", "sup_v", "sup_touch")}
    for t in range(2 * lr + 2, T):
        tol = p.touch_tol_atr * atr_prev[t]
        if np.isnan(tol):
            continue
        lo = max(0, t - W)
        for key, mask, x, sign in (("res", sh, H, +1), ("sup", sl, L, -1)):
            piv = (np.flatnonzero(mask[lo:t - lr]) + lo)[-p.trend_max_pivots:]      # 확정: j + lr ≤ t − 1
            if len(piv) < 2:
                continue
            cands = []
            for a in range(len(piv) - 1):
                j1 = piv[a]; seg = x[j1:t]; k = np.arange(j1, t)
                for b in range(a + 1, len(piv)):
                    j2 = piv[b]
                    slope = (x[j2] - x[j1]) / (j2 - j1)
                    if sign * slope * (t - j1) > tol:                          # 저항은 (지금까지 연장해도) 올라가면 안 되고, 지지는 내려가면 안 된다
                        continue
                    line = x[j1] + slope * (k - j1)
                    if sign > 0:
                        if np.any(seg > line + tol):
                            continue
                        touch = int(np.sum(seg >= line - tol))
                    else:
                        if np.any(seg < line - tol):
                            continue
                        touch = int(np.sum(seg <= line + tol))
                    if touch < p.min_touches:
                        continue
                    cands.append((x[j1] + slope * (t - j1), touch, t - j1, j1, slope))
            if not cands:
                continue
            near = min(c[0] for c in cands) if sign > 0 else max(c[0] for c in cands)       # 저항: 가장 낮은 선, 지지: 가장 높은 선
            group = [c for c in cands if abs(c[0] - near) <= tol]                             # 겹치는 선들
            _, touch, _, j1, slope = max(group, key=lambda c: (c[1], c[2]))                   # 리테스트 최다 → 더 긴 선
            out[key][t] = x[j1] + slope * (t - j1)
            out[key + "_slope"][t] = slope; out[key + "_i"][t] = j1; out[key + "_v"][t] = x[j1]; out[key + "_touch"][t] = touch
    return out


def clean_wicks(df: pd.DataFrame, max_wick_pct: float = 1.0) -> tuple[pd.DataFrame, int]:
    """몸통(시가·종가)에서 ``max_wick_pct`` % 넘게 벗어난 꼬리를 몸통 끝으로 자른다. 시간외 소량 체결의 bad tick(QQQ 2026-10-06 17:30 저가 610 등)이
    ATR·윅 익절·차트를 망치는 것을 막는다. 반환 (정리된 프레임, 잘린 봉 수). 0 이하면 그대로."""
    if max_wick_pct <= 0 or df.empty:
        return df, 0
    out = df.copy()
    body_hi, body_lo = np.maximum(out["Open"], out["Close"]), np.minimum(out["Open"], out["Close"])
    lim = max_wick_pct / 100.0 * out["Close"]
    bad_hi, bad_lo = out["High"] > body_hi + lim, out["Low"] < body_lo - lim
    out.loc[bad_hi, "High"] = body_hi[bad_hi]
    out.loc[bad_lo, "Low"] = body_lo[bad_lo]
    return out, int((bad_hi | bad_lo).sum())


def signal_frame(df: pd.DataFrame, p: VwapSrParams) -> pd.DataFrame:
    """지표·박스·추세선·진입 신호를 열로 붙인 프레임. 구조는 **직전** ``box_bars`` 봉(현재 봉 제외)으로 만든다."""
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy().astype(float)
    out["vwap"] = session_vwap(out)
    out["atr"] = atr_series(out, p.atr_bars)
    atr_prev = out["atr"].shift(1)
    out["vol_avg"] = out["Volume"].rolling(p.vol_avg_bars, min_periods=max(2, p.vol_avg_bars // 2)).mean().shift(1)
    if p.sr_price == "body":
        hi_src, lo_src = np.maximum(out["Open"], out["Close"]), np.minimum(out["Open"], out["Close"])
    else:
        hi_src, lo_src = out["High"], out["Low"]
    out["sr_hi"], out["sr_lo"] = hi_src, lo_src
    hi_prev, lo_prev = hi_src.shift(1), lo_src.shift(1)
    out["box_high"] = hi_prev.rolling(p.box_bars, min_periods=p.box_bars).max()
    out["box_low"] = lo_prev.rolling(p.box_bars, min_periods=p.box_bars).min()
    tol = p.touch_tol_atr * atr_prev
    out["touch_high"] = _touches(hi_prev.to_numpy(), out["box_high"].to_numpy(), tol.to_numpy(), p.box_bars, side=+1)
    out["touch_low"] = _touches(lo_prev.to_numpy(), out["box_low"].to_numpy(), tol.to_numpy(), p.box_bars, side=-1)
    tl = trend_lines(hi_src.to_numpy(dtype=float), lo_src.to_numpy(dtype=float), atr_prev.to_numpy(), p)
    for k, v in tl.items():
        out[k] = v
    c_prev, v_prev = out["Close"].shift(1), out["vwap"].shift(1)
    near_vwap = (c_prev - v_prev).abs() <= p.max_vwap_dist_atr * atr_prev
    bar_range = out["High"] - out["Low"]
    not_huge = bar_range <= p.max_entry_bar_atr * atr_prev
    vol_ok = (out["Volume"] >= p.breakout_volume_mult * out["vol_avg"]) if p.breakout_volume_mult > 0 else pd.Series(True, index=out.index)
    base = near_vwap & not_huge & vol_ok & atr_prev.notna()
    maxh = p.max_box_atr * atr_prev
    out["c_near_vwap"], out["c_not_huge"], out["c_vol_ok"] = near_vwap, not_huge, vol_ok
    out["c_close_above_vwap"] = out["Close"] > out["vwap"]
    use_box, use_trend = p.sr_mode in ("box", "both"), p.sr_mode in ("trend", "both")
    # 수평 박스
    box_ok = (out["box_high"] - out["box_low"] <= maxh) & out["box_high"].notna()
    out["tight"] = box_ok
    box_long = box_ok & (out["touch_high"] >= p.min_touches) & (out["Close"] > out["box_high"]) & (out["box_high"] >= v_prev)
    box_short = box_ok & (out["touch_low"] >= p.min_touches) & (out["Close"] < out["box_low"]) & (out["box_low"] <= v_prev)
    # 추세선: 저항선 아래쪽 경계는 지지선이 있으면 그것, 없으면 박스 아랫선 (높이 판정용)
    res_lo = out["sup"].fillna(out["box_low"]); sup_hi = out["res"].fillna(out["box_high"])
    out["tr_break"] = out["res"].notna() & (out["Close"] > out["res"]); out["box_break"] = out["box_high"].notna() & (out["Close"] > out["box_high"])
    out["c_tr_height"] = out["res"] - res_lo <= maxh; out["c_box_height"] = box_ok
    out["c_tr_res_above_vwap"] = out["res"] >= v_prev; out["c_box_res_above_vwap"] = out["box_high"] >= v_prev
    out["c_box_touch"] = out["touch_high"] >= p.min_touches
    tr_long = out["res"].notna() & (out["res"] - res_lo <= maxh) & (out["res_touch"] >= p.min_touches) & (out["Close"] > out["res"]) & (out["res"] >= v_prev)
    tr_short = out["sup"].notna() & (sup_hi - out["sup"] <= maxh) & (out["sup_touch"] >= p.min_touches) & (out["Close"] < out["sup"]) & (out["sup"] <= v_prev)
    long_sig = base & ((box_long if use_box else False) | (tr_long if use_trend else False)) if p.allow_long else pd.Series(False, index=out.index)
    short_sig = base & ((box_short if use_box else False) | (tr_short if use_trend else False)) if p.allow_short else pd.Series(False, index=out.index)
    out["long_sig"] = long_sig.fillna(False).astype(bool)
    out["short_sig"] = short_sig.fillna(False).astype(bool)
    # 어떤 구조가 뚫렸나 (둘 다면 추세선 우선 — 강의가 그린 선에 가깝다)
    out["long_kind"] = np.where(use_trend & tr_long.fillna(False), "추세선", np.where(use_box & box_long.fillna(False), "박스", ""))
    out["short_kind"] = np.where(use_trend & tr_short.fillna(False), "추세선", np.where(use_box & box_short.fillna(False), "박스", ""))
    return out


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------
def _levels_at(f: pd.DataFrame, i: int, side: int, kind: str, p: VwapSrParams) -> dict:
    """신호 봉 i 에서 뚫린 구조의 저항·지지 기하 (그리기용: 시작 index·값·봉당 기울기)."""
    r = f.iloc[i]; W = p.box_bars
    g = {"kind": kind}
    if kind == "추세선":
        # 뚫린 쪽은 추세선, 반대쪽은 추세선이 있으면 그것, 없으면 박스
        for key, bk in (("res", "box_high"), ("sup", "box_low")):
            if not np.isnan(r[key]):
                g[key] = dict(i0=int(r[key + "_i"]), v0=float(r[key + "_v"]), slope=float(r[key + "_slope"]), at=float(r[key]))
            else:
                g[key] = dict(i0=max(0, i - W), v0=float(r[bk]), slope=0.0, at=float(r[bk]))
    else:
        g["res"] = dict(i0=max(0, i - W), v0=float(r["box_high"]), slope=0.0, at=float(r["box_high"]))
        g["sup"] = dict(i0=max(0, i - W), v0=float(r["box_low"]), slope=0.0, at=float(r["box_low"]))
    return g


def run_vwap_sr(df: pd.DataFrame, p: VwapSrParams | None = None, capital: float = 1.0) -> VwapSrResult:
    p = p or VwapSrParams()
    f = signal_frame(df, p)
    O, H, L, C, V = (f[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    vwap, atr, vol_avg = f["vwap"].to_numpy(), f["atr"].to_numpy(), f["vol_avg"].to_numpy()
    lsig, ssig = f["long_sig"].to_numpy(), f["short_sig"].to_numpy()
    lkind, skind = f["long_kind"].to_numpy(), f["short_kind"].to_numpy()
    day = _day_key(f.index)
    last_of_day = np.r_[day[1:] != day[:-1], True]
    mode = {True: "daily", False: "off"}.get(p.session_flat, p.session_flat)
    if mode == "daily":
        flat_here = last_of_day
    elif mode == "weekend":                                   # 다음 봉까지 하루 넘게 비면(주말·휴일) 그 전 마지막 봉에서 청산
        gap = np.r_[(day[1:] - day[:-1]) > np.timedelta64(1, "D"), False]
        flat_here = last_of_day & gap
    else:
        flat_here = np.zeros(len(f), dtype=bool)
    T = len(f); fee = p.fee_bps / 1e4
    equity = capital; eq_curve = np.full(T, np.nan)
    trades: list[dict] = []
    pos = None
    i = 0
    while i < T:
        if pos is None:
            eq_curve[i] = equity
            if lsig[i] or ssig[i]:
                side = 1 if lsig[i] else -1
                geo = _levels_at(f, i, side, lkind[i] if side > 0 else skind[i], p)
                if p.fill == "next_open" and i + 1 < T:
                    entry_i, entry = i + 1, O[i + 1]
                else:
                    entry_i, entry = i, C[i]
                stop = _initial_stop(side, vwap[i], geo["res"]["at"], geo["sup"]["at"], atr[i], p)
                dist = abs(entry - stop) / entry
                frac = 1.0 if p.risk_pct <= 0 or dist <= 0 else min(1.0, p.risk_pct / 100.0 / dist)
                pos = dict(side=side, sig_i=i, entry_i=entry_i, entry=float(entry), stop=float(stop), stop0=float(stop),
                           geo=geo, vwap0=float(vwap[i]), frac=frac, extended=False)
                if entry_i == i and p.fill == "close":
                    i += 1; continue
                i = entry_i; eq_curve[i] = equity; i += 1; continue
            i += 1; continue
        s, e = pos["side"], pos["entry"]
        if p.trail_vwap and not np.isnan(vwap[i]):
            cand = vwap[i] - s * p.stop_buffer_atr * atr[i]
            pos["stop"] = max(pos["stop"], cand) if s > 0 else min(pos["stop"], cand)
        exit_px = None; why = None
        if p.stop_on_close:
            if (s > 0 and C[i] < pos["stop"]) or (s < 0 and C[i] > pos["stop"]):
                exit_px, why = C[i], "손절"
        else:
            if s > 0 and L[i] <= pos["stop"]:
                exit_px, why = min(O[i], pos["stop"]), "손절"
            elif s < 0 and H[i] >= pos["stop"]:
                exit_px, why = max(O[i], pos["stop"]), "손절"
        if exit_px is None and not np.isnan(atr[i]) and i - pos["entry_i"] >= p.tp_min_bars:
            far_th = max(p.tp_min_dist_atr * atr[i], p.tp_min_dist_pct / 100.0 * C[i])
            if s * (C[i] - vwap[i]) >= far_th:
                pos["extended"] = True
            if pos["extended"]:
                rng = H[i] - L[i]
                if p.tp_volume_mult > 0 and not np.isnan(vol_avg[i]) and V[i] >= p.tp_volume_mult * vol_avg[i]:
                    why = "익절·거래량 스파이크"
                elif p.tp_opposite_candle and s * (C[i] - O[i]) < 0 and abs(C[i] - O[i]) >= p.tp_opp_body_atr * atr[i]:
                    why = "익절·역캔들"
                elif p.tp_wick_frac > 0 and rng > 0 and ((H[i] - max(O[i], C[i])) if s > 0 else (min(O[i], C[i]) - L[i])) >= p.tp_wick_frac * rng:
                    why = "익절·윅"
                elif p.tp_big_bar_atr > 0 and rng >= p.tp_big_bar_atr * atr[i]:
                    why = "익절·큰 봉"
                if why:
                    exit_px = C[i]
        if exit_px is None and flat_here[i]:
            exit_px, why = C[i], "주말 전 청산" if mode == "weekend" else "장 마감"
        if exit_px is None and p.max_hold_bars > 0 and i - pos["entry_i"] >= p.max_hold_bars:
            exit_px, why = C[i], "보유 한도"
        if exit_px is not None and p.fill == "next_open" and why != "장 마감" and i + 1 < T and not last_of_day[i]:
            eq_curve[i] = equity * (1 + pos["frac"] * s * (C[i] / e - 1))
            i += 1; exit_px = O[i]
        if exit_px is None:
            eq_curve[i] = equity * (1 + pos["frac"] * s * (C[i] / e - 1)); i += 1; continue
        ret = s * (exit_px / e - 1) - 2 * fee
        pnl = pos["frac"] * ret
        equity *= 1 + pnl; eq_curve[i] = equity
        dist = abs(e - pos["stop0"]) / e
        g = pos["geo"]
        trades.append({"side": "롱" if s > 0 else "숏", "kind": g["kind"], "signal_time": f.index[pos["sig_i"]], "entry_time": f.index[pos["entry_i"]],
                       "exit_time": f.index[i], "exit_i": int(i), "entry": e, "exit": float(exit_px), "stop": pos["stop0"], "stop_last": pos["stop"],
                       "box_high": g["res"]["at"], "box_low": g["sup"]["at"], "res_geo": g["res"], "sup_geo": g["sup"], "vwap_entry": pos["vwap0"], "why": why,
                       "ret": ret, "pnl": pnl, "frac": pos["frac"], "R": (ret / dist) if dist > 0 else np.nan,
                       "bars": int(i - pos["entry_i"]), "equity": equity})
        pos = None; i += 1
    if pos is not None:                                   # 기간 끝에 열려 있는 포지션 → 마지막 종가로 "미청산" 기록
        s, e, i = pos["side"], pos["entry"], T - 1
        ret = s * (C[i] / e - 1) - 2 * fee; pnl = pos["frac"] * ret
        equity *= 1 + pnl; eq_curve[i] = equity
        dist = abs(e - pos["stop0"]) / e; g = pos["geo"]
        trades.append({"side": "롱" if s > 0 else "숏", "kind": g["kind"], "signal_time": f.index[pos["sig_i"]], "entry_time": f.index[pos["entry_i"]],
                       "exit_time": f.index[i], "exit_i": int(i), "entry": e, "exit": float(C[i]), "stop": pos["stop0"], "stop_last": pos["stop"],
                       "box_high": g["res"]["at"], "box_low": g["sup"]["at"], "res_geo": g["res"], "sup_geo": g["sup"], "vwap_entry": pos["vwap0"],
                       "why": "미청산(기간 끝)", "ret": ret, "pnl": pnl, "frac": pos["frac"], "R": (ret / dist) if dist > 0 else np.nan,
                       "bars": int(i - pos["entry_i"]), "equity": equity})
    f["equity"] = pd.Series(eq_curve, index=f.index).ffill().fillna(capital)
    return VwapSrResult(frame=f, trades=trades, stats=_stats(f, trades, capital))


def _initial_stop(side: int, vwap0: float, res_at: float, sup_at: float, atr: float, p: VwapSrParams) -> float:
    buf = p.stop_buffer_atr * (0.0 if np.isnan(atr) else atr)
    if side > 0:
        cands = {"vwap": vwap0 - buf, "box": sup_at - buf}
        cands["far"] = min(cands["vwap"], cands["box"])
    else:
        cands = {"vwap": vwap0 + buf, "box": res_at + buf}
        cands["far"] = max(cands["vwap"], cands["box"])
    return cands.get(p.stop_mode, cands["vwap"])


def _stats(f: pd.DataFrame, trades: list[dict], capital: float) -> dict:
    eq = f["equity"]
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    bh = float(f["Close"].iloc[-1] / f["Close"].iloc[0] - 1) if len(f) > 1 else 0.0
    out = {"거래 수": len(trades), "총 수익률": float(eq.iloc[-1] / capital - 1) if len(eq) else 0.0, "MDD": float(dd),
           "단순 보유": bh, "봉 수": int(len(f)), "거래일 수": int(pd.Series(_day_key(f.index)).nunique())}
    if trades:
        r = np.array([t["ret"] for t in trades]); pn = np.array([t["pnl"] for t in trades])
        gp, gl = pn[pn > 0].sum(), -pn[pn < 0].sum()
        out.update({"승률": float((r > 0).mean()), "거래당 평균": float(r.mean()), "거래당 중앙": float(np.median(r)),
                    "평균 R": float(np.nanmean([t["R"] for t in trades])), "손익비(PF)": float(gp / gl) if gl > 0 else float("inf"),
                    "최악 거래": float(r.min()), "최고 거래": float(r.max()), "평균 보유 봉": float(np.mean([t["bars"] for t in trades])),
                    "롱 수": int(sum(t["side"] == "롱" for t in trades)), "숏 수": int(sum(t["side"] == "숏" for t in trades)),
                    "구조": pd.Series([t["kind"] for t in trades]).value_counts().to_dict(),
                    "청산 사유": pd.Series([t["why"] for t in trades]).value_counts().to_dict()})
    return out


def structure_segments(f: pd.DataFrame, p: VwapSrParams) -> list[dict]:
    """진입 여부와 상관없이 **감지된 모든 구조**를 그리기용 선분으로 묶는다.

    같은 피벗 쌍(res_i·기울기)이 연속되는 봉들은 하나의 추세선 선분(피벗 → 마지막 유효 봉), 같은 (box_high, box_low) 가 연속되는
    '좁은 박스' 봉들은 하나의 박스 선분(창 시작 → 마지막 봉). 반환 dict: kind(res|sup|box), i0, i1, y0, y1, (box: y_low), touches, ok(터치 ≥ min).
    """
    segs: list[dict] = []
    T = len(f)
    for key in ("res", "sup"):
        line, slope, i0s, v0s, touch = (f[k].to_numpy() for k in (key, key + "_slope", key + "_i", key + "_v", key + "_touch"))
        cur = None
        for t in range(T):
            if np.isnan(line[t]):
                k = None
            else:
                k = (int(i0s[t]), round(float(slope[t]), 9))
            if k != (cur["key"] if cur else None):
                if cur:
                    segs.append(cur)
                cur = None if k is None else {"kind": key, "key": k, "i0": k[0], "y0": float(v0s[t]), "slope": k[1], "i1": t, "y1": float(line[t]), "touches": int(touch[t])}
            elif cur:
                cur["i1"] = t; cur["y1"] = float(line[t]); cur["touches"] = max(cur["touches"], int(touch[t]))
        if cur:
            segs.append(cur)
    bh, bl, tight, th, tl = (f[k].to_numpy() for k in ("box_high", "box_low", "tight", "touch_high", "touch_low"))
    cur = None
    for t in range(T):
        k = (round(float(bh[t]), 6), round(float(bl[t]), 6)) if (tight[t] and not np.isnan(bh[t])) else None
        if k != (cur["key"] if cur else None):
            if cur:
                segs.append(cur)
            cur = None if k is None else {"kind": "box", "key": k, "i0": max(0, t - p.box_bars), "y0": k[0], "y_low": k[1], "slope": 0.0, "i1": t, "y1": k[0],
                                           "touches": int(max(th[t], tl[t]))}
        elif cur:
            cur["i1"] = t; cur["touches"] = max(cur["touches"], int(max(th[t], tl[t])))
    if cur:
        segs.append(cur)
    for sg in segs:
        sg["ok"] = sg["touches"] >= p.min_touches
        sg.pop("key", None)
    return segs


_REASONS = [("c_close_above_vwap", "종가가 VWAP 아래"), ("c_near_vwap", "직전 종가가 VWAP 에서 멂"), ("c_not_huge", "돌파 봉이 너무 큼(추격 금지)"),
            ("c_vol_ok", "돌파 봉 거래량 부족")]


def missed_breakouts(f: pd.DataFrame, p: VwapSrParams) -> pd.DataFrame:
    """종가가 (선택된) 저항 추세선이나 수평 박스 윗선을 넘었는데 롱 진입이 안 난 봉과 **막은 조건**. 페이지 점검표용."""
    rows = []
    use_box, use_trend = p.sr_mode in ("box", "both"), p.sr_mode in ("trend", "both")
    for ts, r in f[(f["tr_break"] | f["box_break"]) & ~f["long_sig"]].iterrows():
        why = [nm for col, nm in _REASONS if not bool(r[col])]
        if use_trend and r["tr_break"]:
            why += [nm for col, nm in (("c_tr_height", "구조 높이 > ATR×max"), ("c_tr_res_above_vwap", "저항선이 VWAP 아래")) if not bool(r[col])]
        elif use_box and r["box_break"]:
            why += [nm for col, nm in (("c_box_height", "박스 높이 > ATR×max"), ("c_box_touch", "저항 터치 수 부족"), ("c_box_res_above_vwap", "박스 윗선이 VWAP 아래")) if not bool(r[col])]
        if not why:
            why = ["(롱 꺼짐)" if not p.allow_long else "(포지션 보유 중이거나 구조 모드 밖)"]
        rows.append({"시각": ts, "구조": "추세선" if r["tr_break"] else "박스", "종가": r["Close"], "저항": r["res"] if r["tr_break"] else r["box_high"], "VWAP": r["vwap"],
                     "ATR": r["atr"], "막은 조건": " · ".join(why)})
    return pd.DataFrame(rows)


def current_setup(f: pd.DataFrame) -> dict:
    """마지막 봉 기준 박스·추세선·VWAP 상태 (실시간 참고용)."""
    if f.empty:
        return {}
    r = f.iloc[-1]
    return {"시각": f.index[-1], "종가": float(r["Close"]), "VWAP": float(r["vwap"]), "저항(박스 위)": float(r["box_high"]),
            "지지(박스 아래)": float(r["box_low"]), "좁은 박스": bool(r["tight"]), "저항 터치": int(r["touch_high"]), "지지 터치": int(r["touch_low"]),
            "저항 추세선": float(r["res"]) if not np.isnan(r["res"]) else None, "지지 추세선": float(r["sup"]) if not np.isnan(r["sup"]) else None,
            "ATR": float(r["atr"]), "롱 신호": bool(r["long_sig"]), "숏 신호": bool(r["short_sig"])}
