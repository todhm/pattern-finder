"""미너비니 '수동' 체크리스트 중 가격 데이터로 판정 가능한 항목의 자동화.

``pages/_shared/minervini_manual_checks.py`` 의 항목 키와 같은 이름을 쓴다.
모두 point-in-time — 인자로 받는 일봉은 기준일까지로 잘려 있어야 한다.

자동화한 것 (규칙은 각 함수 docstring):
    0-4 새 강세장 초입 · 3-1 지수보다 먼저 바닥 · 3-2 지수 저점일 신고가 근접
    · 3-3 조정 깊이 ≤ 지수×2.5 · 3-4 산업군 내 RS 1~2위(스크린 생존자 한정)
    · 4-1 베이스 이전 선행 상승 +30% · 4-10 추격 아님(피봇 +5% 이내)

여전히 사람 몫: 2-8 가이던스, 3-6 신선함, 4-8 털어내기, 4-9 손잡이 위치,
VCP 최종 눈판독, 5-2·5-3(계좌 상태).

판정값: True/False, 측정 불가는 None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from strategy.adapters.minervini_screener import PivotLevels


@dataclass(frozen=True)
class Correction:
    """지수의 최근 1년 최대 조정 구간 (고점 → 저점)."""

    peak_date: date
    trough_date: date
    depth: float                 # 0.20 = −20%
    weeks_since_trough: float
    rally_from_trough: float     # 기준일 종가 / 저점 − 1


@dataclass(frozen=True)
class ManualAuto:
    checks: dict[str, bool | None]
    notes: dict[str, str]


def index_correction(index_df: pd.DataFrame, lookback: int = 252) -> Correction | None:
    """최근 ``lookback`` 봉 안의 최대 낙폭 구간. 봉 부족이면 None."""
    c = index_df["Close"].tail(lookback)
    if len(c) < 60:
        return None
    dd = c / c.cummax() - 1.0
    trough_ts = dd.idxmin()
    peak_ts = c.loc[:trough_ts].idxmax()
    trough_pos = c.index.get_loc(trough_ts)
    return Correction(
        peak_date=peak_ts.date(), trough_date=trough_ts.date(),
        depth=float(-dd.min()),
        weeks_since_trough=(len(c) - 1 - trough_pos) / 5.0,
        rally_from_trough=float(c.iloc[-1] / c.loc[trough_ts] - 1.0),
    )


def new_bull_window(corr: Correction | None, min_depth: float = 0.10,
                    max_weeks: float = 8.0) -> bool | None:
    """0-4 — 지수가 −10%↑ 조정 뒤 저점에서 8주 이내 & 저점 대비 반등 중."""
    if corr is None:
        return None
    return (corr.depth >= min_depth and corr.weeks_since_trough <= max_weeks
            and corr.rally_from_trough > 0.05)


def leadership_checks(
    stock_df: pd.DataFrame, corr: Correction | None,
    near_high: float = 0.10, max_ratio: float = 2.5, min_index_depth: float = 0.05,
) -> dict[str, tuple[bool | None, str]]:
    """3-1·3-2·3-3 — 지수의 최근 조정 구간에서 종목이 어떻게 버텼나.

    - 3-1: 지수 고점~기준일 사이 종목 최저 종가일이 지수 저점일보다 **앞섬**.
    - 3-2: 지수 저점일에 종목 종가가 그날까지의 52주 고가 −10% 이내.
    - 3-3: 같은 구간 종목 최대 낙폭 ≤ 지수 낙폭 × 2.5.
    지수 조정이 −5% 미만이면 비교 의미가 없어 None.
    """
    na = "지수 조정 −5% 미만 — 비교할 조정 구간 없음"
    if corr is None or corr.depth < min_index_depth:
        return {k: (None, na) for k in ("m3_1", "m3_2", "m3_3")}
    c = stock_df["Close"]
    peak, trough = pd.Timestamp(corr.peak_date), pd.Timestamp(corr.trough_date)
    window = c.loc[peak:]
    if len(window) < 5 or trough not in c.index:
        return {k: (None, "종목 데이터가 조정 구간을 덮지 못함") for k in ("m3_1", "m3_2", "m3_3")}
    low_ts = window.idxmin()
    r31 = low_ts < trough
    hi_at_trough = float(stock_df["High"].loc[:trough].tail(252).max())
    gap = float(c.loc[trough]) / hi_at_trough - 1.0
    r32 = gap >= -near_high
    seg = c.loc[peak:]
    s_dd = float(-(seg / seg.cummax() - 1.0).min())
    ratio = s_dd / corr.depth
    return {
        "m3_1": (r31, f"종목 저점 {low_ts.date()} vs 지수 저점 {corr.trough_date}"),
        "m3_2": (r32, f"지수 저점일 종목 52주 고가 대비 {gap:+.0%} (기준 −10% 이내)"),
        "m3_3": (ratio <= max_ratio,
                 f"종목 낙폭 −{s_dd:.0%} ÷ 지수 −{corr.depth:.0%} = {ratio:.1f}배 (기준 ≤ 2.5배)"),
    }


def prior_uptrend(stock_df: pd.DataFrame, pl: PivotLevels,
                  lookback: int = 126, min_gain: float = 0.30) -> tuple[bool | None, str]:
    """4-1 — 베이스 천장 이전 ``lookback``봉 최저가 → 베이스 천장 상승폭 ≥ +30%."""
    if pl.base_weeks is None:
        return None, "베이스 천장 측정 불가"
    n_back = int(round(pl.base_weeks * 5))
    if n_back >= len(stock_df):
        return None, "봉 부족"
    head = stock_df.iloc[: len(stock_df) - n_back]
    top = float(head["High"].tail(10).max())
    low = float(head["Low"].tail(lookback).min())
    gain = top / low - 1.0
    return gain >= min_gain, f"베이스 이전 6개월 저점 ${low:,.2f} → 천장 ${top:,.2f} = {gain:+.0%}"


def not_chasing(pl: PivotLevels, max_ext: float = 0.05) -> tuple[bool, str]:
    """4-10 — 현재가가 피봇 +5% 초과로 확장됐으면 추격."""
    return (pl.dist_to_pivot <= max_ext,
            f"피봇 대비 {pl.dist_to_pivot:+.1%} · {pl.status} (기준 ≤ +5%)")


def industry_rank(cands: list[tuple[str, str, float | None]]) -> dict[str, tuple[bool | None, str]]:
    """3-4 — (티커, 산업군, RS) 목록에서 같은 산업군 내 RS 순위 ≤ 2.

    스크린 생존자끼리만 비교하므로 '산업군 전체 1~2위'의 근사치다.
    """
    out: dict[str, tuple[bool | None, str]] = {}
    df = pd.DataFrame(cands, columns=["t", "ind", "rs"])
    for ind, g in df.groupby("ind"):
        g = g.sort_values("rs", ascending=False).reset_index(drop=True)
        for i, row in g.iterrows():
            if ind == "미분류":
                out[row.t] = (None, "산업군 미분류")
            else:
                out[row.t] = (i < 2, f"{ind} 생존자 {len(g)}개 중 RS {i + 1}위")
    return out


def manual_auto(
    stock_df: pd.DataFrame, index_df: pd.DataFrame, pl: PivotLevels,
) -> ManualAuto:
    """종목 1개의 자동화 가능한 수동 항목 일괄 판정 (3-4 제외 — 후보군 필요)."""
    corr = index_correction(index_df)
    checks: dict[str, bool | None] = {}
    notes: dict[str, str] = {}
    checks["m0_4"] = new_bull_window(corr)
    notes["m0_4"] = (f"지수 최근 1년 최대 조정 −{corr.depth:.0%} ({corr.peak_date}→{corr.trough_date}), "
                     f"저점 후 {corr.weeks_since_trough:.0f}주, 반등 {corr.rally_from_trough:+.0%}"
                     if corr else "지수 봉 부족")
    for k, (v, n) in leadership_checks(stock_df, corr).items():
        checks[k], notes[k] = v, n
    checks["m4_1"], notes["m4_1"] = prior_uptrend(stock_df, pl)
    checks["m4_10"], notes["m4_10"] = not_chasing(pl)
    return ManualAuto(checks=checks, notes=notes)
