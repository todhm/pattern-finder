"""참사 조기경보 — 가격 밖 데이터로 ⑩ 급락 회피의 이탈을 앞당길 수 있는가.

로드맵 `TQQQ_초과수익_로드맵_2026-10.md` 기둥 2 의 첫 실험. ⑩ 은 "200일선 아래 & 52주 고점 대비 −18%" 에서야
끊겨 참사 5년에 평균 −45% 를 맞는다. 여기서는 거시·신용·변동성 신호가 켜져 있을 때 **더 얕은 낙폭(dd_alarm)** 에서
끊는 변형을 만들고, (1) 신호별 혼동행렬(⑩ 이탈 대비 선행 일수, 거짓 경보), (2) 세후 성과를 잰다.

신호는 전부 "그날 알 수 있었던 값" 으로 정렬한다 (발표 지연: UNRATE +35일, ICSA +5일). 체결은 다음날(exec_lag 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- 신호 (히스테리시스 → 0/1 에피소드)

def _hysteresis(on: pd.Series, off: pd.Series) -> pd.Series:
    """on 이 True 면 1, off 가 True 면 0, 그 외 유지. 시작 0."""
    state = np.zeros(len(on)); cur = 0.0
    onv, offv = on.fillna(False).values, off.fillna(False).values
    for i in range(len(on)):
        if cur == 0.0 and onv[i]:
            cur = 1.0
        elif cur == 1.0 and offv[i]:
            cur = 0.0
        state[i] = cur
    return pd.Series(state, index=on.index)


def align_daily(s: pd.Series, index: pd.DatetimeIndex, lag_days: int = 0) -> pd.Series:
    """저빈도/다른 달력 시리즈를 일봉 인덱스로 (발표 지연 lag_days 뒤에 알 수 있었던 값으로 ffill)."""
    s = s.dropna().copy()
    s.index = pd.DatetimeIndex(s.index).normalize() + pd.Timedelta(days=lag_days)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def sig_term_spread(t10y3m: pd.Series, index) -> pd.Series:
    x = align_daily(t10y3m, index)
    return _hysteresis(x < 0.0, x > 0.2).rename("기간스프레드 역전")


def sig_credit_spread(baa10y: pd.Series, index, rise: float = 0.5, fall: float = 0.25, lookback: int = 126) -> pd.Series:
    x = align_daily(baa10y, index)
    low = x.rolling(lookback, min_periods=20).min()
    return _hysteresis(x >= low + rise, x < low + fall).rename("신용스프레드 급등")


def sig_sahm(unrate: pd.Series, index) -> pd.Series:
    u = unrate.dropna().copy(); u.index = pd.DatetimeIndex(u.index)
    m3 = u.rolling(3).mean()
    sahm = m3 - m3.rolling(12).min()
    s = align_daily(sahm, index, lag_days=35)
    return _hysteresis(s >= 0.5, s < 0.3).rename("실업률 Sahm")


def sig_claims(icsa: pd.Series, index, up: float = 1.2, down: float = 1.1) -> pd.Series:
    c = icsa.dropna().copy(); c.index = pd.DatetimeIndex(c.index)
    avg4 = c.rolling(4).mean()
    ratio = avg4 / avg4.rolling(52).min()
    r = align_daily(ratio, index, lag_days=5)
    return _hysteresis(r >= up, r < down).rename("실업수당 청구 급증")


def sig_vix_level(vix: pd.Series, index, on: float = 30.0, off: float = 25.0) -> pd.Series:
    v = align_daily(vix, index)
    return _hysteresis(v > on, v < off).rename("VIX > 30")


def sig_vix_term(vix: pd.Series, vix3m: pd.Series, index, days_on: int = 3, days_off: int = 5) -> pd.Series:
    v, v3 = align_daily(vix, index), align_daily(vix3m, index)
    inv = (v > v3)
    on = inv.rolling(days_on).sum() >= days_on
    off = (~inv).rolling(days_off).sum() >= days_off
    s = _hysteresis(on, off)
    s[v3.isna()] = np.nan
    return s.rename("VIX 기간구조 역전")


def sig_trend_12m(close: pd.Series) -> pd.Series:
    """월말 종가 < 12개월 SMA → 1 (월 1회 판정, 다음 달 내내 유지)."""
    me = close.groupby(close.index.to_period("M")).apply(lambda x: x.index[-1])
    m = close.loc[me.values]
    below = (m < m.rolling(12).mean()).astype(float)
    return below.reindex(close.index, method="ffill").fillna(0.0).rename("12개월 추세 하회")


# ----------------------------------------------------------------------------- 변형 ⑩

def early_exit_filter(signal_close: pd.Series, alarm: pd.Series | None, n: int = 200, dd: float = 0.18,
                      dd_alarm: float = 0.08, reenter_n: int = 20, require_below_sma: bool = True) -> pd.Series:
    """⑩ 에 조기 이탈 조건 추가: 200일선 아래 & (낙폭 > dd  OR  (경보 ON & 낙폭 > dd_alarm)). 재진입 20일선 위.

    alarm 이 None 이면 원래 ⑩ 과 동일. require_below_sma=False 면 경보 ON 일 때 200일선 조건 없이 dd_alarm 만 본다.
    """
    c = signal_close.dropna()
    sma = c.rolling(n).mean()
    drawdown = c / c.rolling(252, min_periods=20).max() - 1.0
    below = c < sma
    a = alarm.reindex(c.index).ffill().fillna(0.0).astype(bool) if alarm is not None else pd.Series(False, index=c.index)
    deep = below & (drawdown < -dd)
    early = a & (drawdown < -dd_alarm) & (below if require_below_sma else True)
    off = (deep | early).values
    on = (c > c.rolling(reenter_n).mean()).values
    state = np.ones(len(c)); cur = 1.0
    for i in range(len(c)):
        if np.isnan(sma.iloc[i]):
            state[i] = 0.0; continue
        if off[i]:
            cur = 0.0
        elif on[i]:
            cur = 1.0
        state[i] = cur
    return pd.Series(state, index=c.index)


# ----------------------------------------------------------------------------- 혼동행렬

@dataclass
class Episode:
    exit_date: pd.Timestamp
    reentry_date: pd.Timestamp | None
    hold_ret: float          # 이탈~재진입 동안 레버리지 ETF 수익 (음수 = 진짜 회피)
    kind: str                # "참사" (ETF −40% 이하) / "조정" / "휩소" (ETF 상승)


def episodes(base: pd.Series, lev_close: pd.Series) -> list[Episode]:
    """⑩ 의 이탈 에피소드와 그 구간의 ETF 수익으로 참사/조정/휩소 라벨."""
    s = base.dropna(); out = []
    prev = 1.0
    exit_at = None
    for ts, v in s.items():
        if prev == 1.0 and v == 0.0:
            exit_at = ts
        elif prev == 0.0 and v == 1.0 and exit_at is not None:
            seg = lev_close[(lev_close.index >= exit_at) & (lev_close.index <= ts)]
            r = float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) > 1 else 0.0
            out.append(Episode(exit_at, ts, r, "참사" if r < -0.40 else ("조정" if r < 0 else "휩소")))
            exit_at = None
        prev = v
    if exit_at is not None:
        seg = lev_close[lev_close.index >= exit_at]
        r = float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) > 1 else 0.0
        out.append(Episode(exit_at, None, r, "진행중"))
    return out


def alarm_episodes(sig: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s = sig.dropna(); out = []; start = None; prev = 0.0
    for ts, v in s.items():
        if prev == 0.0 and v == 1.0:
            start = ts
        elif prev == 1.0 and v == 0.0 and start is not None:
            out.append((start, ts)); start = None
        prev = v
    if start is not None:
        out.append((start, s.index[-1]))
    return out


def confusion(sig: pd.Series, eps: list[Episode], lookback_bdays: int = 120) -> dict:
    """에피소드별 선행 일수(⑩ 이탈 − 첫 경보, 영업일) 와 거짓 경보 수.

    거짓 경보 = 경보 에피소드 중 어떤 ⑩ 이탈의 [이탈 −lookback, 재진입] 창과도 겹치지 않는 것.
    신호가 없는 구간(NaN) 의 에피소드는 '측정 불가'.
    """
    s = sig.dropna()
    rows = []
    for e in eps:
        if s.empty or e.exit_date < s.index[0]:
            rows.append({"exit": e.exit_date.date(), "kind": e.kind, "lead_bdays": None, "measurable": False}); continue
        idx = s.index.get_indexer([e.exit_date], method="pad")[0]
        lo = max(0, idx - lookback_bdays)
        window = s.iloc[lo: idx + 1]
        first = window[window == 1.0]
        lead = int(idx - s.index.get_loc(first.index[0])) if len(first) else None
        rows.append({"exit": e.exit_date.date(), "kind": e.kind, "lead_bdays": lead, "measurable": True})
    windows = []
    for e in eps:
        if s.empty or e.exit_date < s.index[0]:
            continue
        idx = s.index.get_indexer([e.exit_date], method="pad")[0]
        lo = s.index[max(0, idx - lookback_bdays)]
        hi = e.reentry_date if e.reentry_date is not None else s.index[-1]
        windows.append((lo, hi))
    false_alarms = []
    for a, b in alarm_episodes(s):
        if not any(a <= hi and b >= lo for lo, hi in windows):
            false_alarms.append((a.date(), b.date()))
    crash_leads = [r["lead_bdays"] for r in rows if r["kind"] == "참사" and r["measurable"]]
    return {"rows": rows, "false_alarms": false_alarms, "n_false": len(false_alarms),
            "crash_caught": sum(1 for x in crash_leads if x is not None), "crash_measurable": len(crash_leads),
            "crash_lead_mean": float(np.mean([x for x in crash_leads if x is not None])) if any(x is not None for x in crash_leads) else None,
            "coverage_start": str(s.index[0].date()) if not s.empty else None}


# ----------------------------------------------------------------------------- 재진입 게이트 (이탈 규칙은 ⑩ 그대로)

def reentry_gate_filter(signal_close: pd.Series, alarm: pd.Series | None, n: int = 200, dd: float = 0.18,
                        reenter_n: int = 20, reenter_n2: int | None = None, gate_mode: str = "block",
                        partial: float = 0.5) -> pd.Series:
    """⑩ 의 이탈(200일선 아래 & 낙폭 > dd)은 그대로, **재진입만** 바꾼다.

    - gate_mode="block": 경보 ON 동안 재진입 금지. OFF 되면 reenter_n 일선 위에서 재진입.
    - gate_mode="partial": 경보 ON 이면 ``partial`` 만 재진입, OFF 되면 전량.
    - gate_mode="price": 경보 무시. 종가 > reenter_n 일선 **그리고** > reenter_n2 일선 일 때 재진입.
    반환은 목표 비중(0 / partial / 1).
    """
    c = signal_close.dropna()
    sma = c.rolling(n).mean()
    drawdown = c / c.rolling(252, min_periods=20).max() - 1.0
    off = ((c < sma) & (drawdown < -dd)).values
    above1 = (c > c.rolling(reenter_n).mean())
    above2 = (c > c.rolling(reenter_n2).mean()) if reenter_n2 else pd.Series(True, index=c.index)
    on = (above1 & above2).values
    a = alarm.reindex(c.index).ffill().fillna(0.0).astype(bool).values if alarm is not None else np.zeros(len(c), bool)
    state = np.ones(len(c)); cur = 1.0
    for i in range(len(c)):
        if np.isnan(sma.iloc[i]):
            state[i] = 0.0; continue
        if off[i]:
            cur = 0.0
        elif on[i]:
            if gate_mode == "block":
                if not a[i]:
                    cur = 1.0
            elif gate_mode == "partial":
                cur = 1.0 if not a[i] else max(cur, partial)
            else:
                cur = 1.0
        elif cur == partial and gate_mode == "partial" and not a[i] and above1.iloc[i]:
            cur = 1.0  # 경보 해제 후 전량
        state[i] = cur
    return pd.Series(state, index=c.index)
