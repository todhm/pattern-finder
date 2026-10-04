"""William O'Neil «How to Make Money in Stocks» — 'M = Market Direction' 시장 타이밍 상태기계 + 매도 규칙.

문서: ``docs/strategy_notes/오닐_최고의주식_최적의타이밍_체크리스트_2026_10.md`` (항목 번호가 이 코드와 일치)

입력은 지수(QQQ) **일봉** OHLCV. 매매 대상(TQQQ)은 호출자가 노출(exposure) 시계열을 받아 적용한다.

상태:
    UP        확인된 상승장 → 노출 100%
    PRESSURE  분산일 누적으로 압박받는 상승장 → 노출 50%
    CORR      조정(분산일 과다 또는 고점 대비 낙폭) → 노출 0%, 랠리 시도 일수 카운트
책의 규칙(2판, 1995) + IBD 가 뒤에 수치화한 관행(25일 창, 5~6회, +1.25%, 4~7일차)을 파라미터로 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ONeilParams:
    # --- 분산일 (ch.7 "heavy volume without further price progress") ---
    dist_drop: float = -0.002          # 종가 −0.2% 이하 & 거래량 > 전일 → 분산일
    stall_enabled: bool = True         # 정체일: 소폭 상승(전일 상승폭보다 작고 ≤ +0.4%) & 거래량↑ & 종가가 일중범위 하단 절반
    stall_max_gain: float = 0.004
    dist_window: int = 25              # 분산일 유효 기간(거래일)
    dist_expire_gain: float = 0.05     # 지수가 분산일 종가 대비 +5% 오르면 그 분산일 소멸
    pressure_count: int = 4            # 이 이상이면 PRESSURE (노출 50%)
    correction_count: int = 6          # 이 이상이면 CORR (노출 0%)
    # --- 고점 대비 낙폭으로도 조정 판정 (ch.7 "intermediate tops usually 8%~12% declines") ---
    drawdown_corr: float = 0.08        # 현재 상승장(마지막 FTD 이후) 최고 종가 대비 −8% → CORR (0 이면 비활성)
    # --- 팔로스루데이 (ch.7 "Wait for a second confirmation") ---
    ftd_min_day: int = 4               # 랠리 4일차부터
    ftd_max_day: int = 0               # >0 이면 그 일수 안에 FTD 가 없을 때 랠리 시도를 리셋(책: "10일 이후 FTD 는 약함"). 0 = IBD 현대 관행(4일차 이후 아무 날)
    ftd_gain: float = 0.0125           # +1.25% 이상 (책: Dow 1%+, IBD 관행 1.25~1.7%)
    ftd_day3_gain: float = 0.02        # 3일차는 +2% 이상이면 허용 ("occasionally as early as the third day if very powerful")
    ftd_need_volume: bool = True       # 거래량 조건 사용
    ftd_volume_mode: str = "prev"      # prev = 전일보다 많음(책) / avg50 = 50일 평균보다 많음 / either
    undercut_fail: bool = True         # FTD 뒤 랠리 저점 하향 이탈 → 실패, CORR 복귀
    # --- 매도 규칙 (ch.9·10) — 매매 대상(TQQQ) 가격에 적용 ---
    loss_cut: float = 0.08             # 진입가 대비 −8% (레버리지 배수 곱함)
    leverage: float = 3.0
    rule34_extended_200: float = 0.0   # TQQQ 가 200일선 위 +70%↑ 면 절반 (0 = 비활성)
    rule35_ma200_down: bool = False    # QQQ 200일선 20일 기울기 < 0 → 노출 0
    # --- 수동 항목 대체(프록시) ---
    breadth_divergence: bool = False   # 지수 52주고점 −2% 이내 & Nasdaq-100 50일선 위 비율 < 50% → PRESSURE 로 강등
    defensive_rotation: bool = False   # XLU/QQQ 20일 비율 변화 > +5% → PRESSURE 로 강등
    exposure_pressure: float = 0.5
    reentry_above_ma50: bool = False   # (판단) FTD 에 더해 QQQ 종가 > 50일선이어야 복귀 — 베어마켓 랠리 FTD 필터


def distribution_flags(q: pd.DataFrame, p: ONeilParams, volume: pd.Series | None = None) -> pd.DataFrame:
    """일별 분산일/정체일 플래그. volume 을 주면 QQQ ETF 거래량 대신 그것(예: 구성종목 거래량 합)을 쓴다."""
    ret = q["Close"].pct_change()
    v = (volume.reindex(q.index) if volume is not None else q["Volume"]).astype(float)
    v = v.where(v > 0).ffill()
    vol_up = v > v.shift(1)
    vol_avg = v > v.rolling(50).mean()
    if p.ftd_volume_mode == "avg50":
        ftd_vol = vol_avg
    elif p.ftd_volume_mode == "either":
        ftd_vol = vol_up | vol_avg
    else:
        ftd_vol = vol_up
    dist = (ret <= p.dist_drop) & vol_up
    rng = (q["High"] - q["Low"]).replace(0, np.nan)
    pos_in_range = (q["Close"] - q["Low"]) / rng
    stall = (
        (ret > 0) & (ret <= p.stall_max_gain) & (ret < ret.shift(1)) & vol_up & (pos_in_range < 0.5)
    ) if p.stall_enabled else pd.Series(False, index=q.index)
    return pd.DataFrame({"ret": ret, "dist": dist.fillna(False), "stall": stall.fillna(False), "vol_up": vol_up, "ftd_vol": ftd_vol.fillna(False)})


def run_state_machine(
    q: pd.DataFrame,
    p: ONeilParams,
    breadth_pct50: pd.Series | None = None,
    xlu_ratio: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> pd.DataFrame:
    """지수 일봉 → 상태·분산일 수·랠리 일수·노출 시계열 (당일 종가 기준, 적용은 다음 날)."""
    f = distribution_flags(q, p, volume)
    close = q["Close"]
    hi52 = close.rolling(252, min_periods=60).max()
    ma200 = close.rolling(200).mean()
    ma200_slope = ma200 - ma200.shift(20)
    ma50 = close.rolling(50).mean()

    states, counts, rally_days, exposures, events = [], [], [], [], []
    state = "UP"
    dist_list: list[tuple[pd.Timestamp, float]] = []   # (날짜, 분산일 종가)
    rally_low = np.nan
    rally_day = 0
    ftd_low = np.nan
    run_high = float(q["Close"].iloc[0])
    for i, (d, row) in enumerate(q.iterrows()):
        c = float(row["Close"])
        ev = ""
        if state in ("UP", "PRESSURE"):
            run_high = max(run_high, c)
        # 분산일 갱신
        dist_list = [(dd, dc) for dd, dc in dist_list if (i - q.index.get_loc(dd)) < p.dist_window and c < dc * (1 + p.dist_expire_gain)]
        if f["dist"].iat[i] or f["stall"].iat[i]:
            dist_list.append((d, c))
        n = len(dist_list)

        if state in ("UP", "PRESSURE"):
            dd52 = c / hi52.iat[i] - 1 if not np.isnan(hi52.iat[i]) else 0.0
            dd_run = c / run_high - 1
            if n >= p.correction_count:
                state, ev = "CORR", f"분산일 {n}개 → 조정"
            elif p.drawdown_corr and dd_run <= -p.drawdown_corr:
                state, ev = "CORR", f"상승장 고점 대비 {dd_run:.1%} → 조정"
            elif p.undercut_fail and not np.isnan(ftd_low) and c < ftd_low:
                state, ev = "CORR", f"FTD 뒤 랠리 저점 {ftd_low:.2f} 하향 이탈 → 실패"
            elif p.rule35_ma200_down and not np.isnan(ma200_slope.iat[i]) and ma200_slope.iat[i] < 0 and c < ma200.iat[i]:
                state, ev = "CORR", "200일선 하락 전환 & 종가 < 200일선 (규칙 35)"
            else:
                new = "PRESSURE" if n >= p.pressure_count else "UP"
                if p.breadth_divergence and breadth_pct50 is not None:
                    b = breadth_pct50.get(d, np.nan)
                    if not np.isnan(b) and dd52 > -0.02 and b < 0.5:
                        new, ev = "PRESSURE", f"폭 다이버전스(50일선 위 {b:.0%})"
                if p.defensive_rotation and xlu_ratio is not None:
                    x = xlu_ratio.get(d, np.nan)
                    if not np.isnan(x) and x > 0.05:
                        new, ev = "PRESSURE", f"방어주 로테이션(XLU/QQQ 20일 +{x:.0%})"
                if new != state:
                    ev = ev or (f"분산일 {n}개 → 압박" if new == "PRESSURE" else f"분산일 {n}개 → 상승장 복귀")
                state = new
            if state == "CORR":
                rally_low, rally_day, ftd_low = c, 0, np.nan
                dist_list = []
        else:  # CORR: 랠리 시도 추적
            if np.isnan(rally_low) or c < rally_low:
                rally_low, rally_day = c, 0
            elif rally_day == 0 and c > float(q["Close"].iat[i - 1]):
                rally_day = 1
            elif rally_day > 0:
                rally_day += 1
                if p.ftd_max_day and rally_day > p.ftd_max_day:
                    rally_low, rally_day = c, 0          # 10일 안에 FTD 없음 → 약한 랠리, 새 시도 대기
                    states.append(state); counts.append(n); rally_days.append(0); exposures.append(0.0); events.append("")
                    continue
                r = f["ret"].iat[i]
                vol_ok = (not p.ftd_need_volume) or bool(f["ftd_vol"].iat[i])
                within = rally_day >= p.ftd_min_day and (p.ftd_max_day == 0 or rally_day <= p.ftd_max_day)
                day3 = rally_day == 3 and r >= p.ftd_day3_gain
                ma_ok = (not p.reentry_above_ma50) or (not np.isnan(ma50.iat[i]) and c > ma50.iat[i])
                if vol_ok and ma_ok and ((within and r >= p.ftd_gain) or day3):
                    state, ev = "UP", f"팔로스루데이 {rally_day}일차 {r:+.2%}"
                    ftd_low = rally_low
                    run_high = c
                    dist_list = []
        states.append(state)
        counts.append(n)
        rally_days.append(rally_day if state == "CORR" else 0)
        exposures.append({"UP": 1.0, "PRESSURE": p.exposure_pressure, "CORR": 0.0}[state])
        events.append(ev)
    out = pd.DataFrame({"state": states, "dist_count": counts, "rally_day": rally_days, "exposure": exposures, "event": events,
                        "dist_day": f["dist"].values, "stall_day": f["stall"].values}, index=q.index)
    return out


def apply_sell_rules(exposure: pd.Series, trd: pd.DataFrame, p: ONeilParams) -> tuple[pd.Series, list[dict]]:
    """노출 시계열에 TQQQ 가격 기반 매도 규칙을 덧씌운다 (손절 7~8%×lev, 규칙 34)."""
    exp = exposure.copy()
    close = trd["Close"].reindex(exp.index).ffill()
    ma200 = close.rolling(200).mean()
    entry_px = np.nan
    stopped = False
    log = []
    prev = 0.0
    for i, d in enumerate(exp.index):
        e = exp.iat[i]
        c = float(close.iat[i])
        if prev == 0.0 and e > 0:          # 신규 진입
            entry_px, stopped = c, False
        if e == 0.0:
            stopped = False
        if stopped:
            exp.iat[i] = 0.0
        elif e > 0 and not np.isnan(entry_px) and p.loss_cut and c <= entry_px * (1 - p.loss_cut * p.leverage):
            stopped = True
            exp.iat[i] = 0.0
            log.append({"date": d.date(), "rule": f"손절 −{p.loss_cut*p.leverage:.0%} (진입 {entry_px:.2f} → {c:.2f})"})
        elif e > 0 and p.rule34_extended_200 and not np.isnan(ma200.iat[i]) and c / ma200.iat[i] - 1 >= p.rule34_extended_200:
            exp.iat[i] = min(e, 0.5)
            if prev > 0.5:
                log.append({"date": d.date(), "rule": f"규칙 34: 200일선 위 {c/ma200.iat[i]-1:.0%} → 절반"})
        prev = exp.iat[i]
    return exp, log


def simulate_exposure(exposure: pd.Series, trd: pd.DataFrame, slip: float = 0.0005, lag: int = 1) -> pd.Series:
    """노출(0~1)을 다음 날 시가부터 적용한 자본곡선. 노출 변경일에 편도 slip."""
    tc = trd["Close"].reindex(exposure.index).ffill()
    to = trd["Open"].reindex(exposure.index).ffill()
    pos = exposure.shift(lag).fillna(0.0)
    # 일간 수익: 노출 변경이 있는 날은 시가 체결 → (close/open −1) 만큼만 노출 반영, 그 외는 close/close
    r_cc = tc.pct_change().fillna(0.0)
    r_oc = (tc / to - 1).fillna(0.0)
    r_co = (to / tc.shift(1) - 1).fillna(0.0)
    prev_pos = pos.shift(1).fillna(0.0)
    changed = (pos != prev_pos)
    daily = np.where(changed, prev_pos * r_co + pos * r_oc, pos * r_cc)
    cost = np.where(changed, (pos - prev_pos).abs() * slip, 0.0)
    eq = pd.Series((1 + daily - cost), index=exposure.index).cumprod()
    return eq
