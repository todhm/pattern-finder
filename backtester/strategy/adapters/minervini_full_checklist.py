"""미너비니 체크리스트 **전 단계** 판정 + 워치 → 돌파 진입 재현.

43번 페이지(``pages/43_Minervini_Checklist.py``)의 STAGE 정의를 그대로 따른다:
단계 통과 = 필수(guard) 전부 + 핵심(core) ``min_core``개 이상 (None 이면 core 전부).
종합 판정: 전 단계 통과 = 매수, STAGE 4 만 실패 = 워치, 그 외 = 탈락.

사람 눈이 필요했던 항목의 코드 근사 (각 함수 docstring 에 기준):
    2-8 가이던스 하향 → 기준일 이전 120일 헤드라인 키워드
    3-5 주도 그룹 → 이번 스크린 RS 통과 종목이 많이 몰린 상위 산업군
    4-6 최종 축소 → 마지막 축소 깊이 (없으면 10일 레인지)
    4-9 손잡이 위치 → 피봇이 베이스 레인지 상단 1/3
    4-10 추격 → 최근 10일 +15% 미만 & 피봇 +5% 이내
    4-12 돌파 거래량 → 돌파일 거래량 ≥ 50일 평균 × 1.4

STAGE 0(시장)은 :mod:`minervini_scan` 게이트가 담당한다 (여기선 받지 않음).
STAGE 5 는 고정 파라미터(손절 8%·비중 20%·익절 22%)라 항상 통과 — 숫자만 기록.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from strategy.adapters.minervini_manual_auto import prior_uptrend
from strategy.adapters.minervini_screener import (
    PivotLevels,
    evaluate_trend_template,
    pivot_levels,
)

_GUIDANCE_CUT = re.compile(
    r"(lower|lowers|lowered|lowering|cut|cuts|slash|slashes|reduce|reduces|reduced|trim|trims|withdraw|withdraws)"
    r"\b[^.]{0,40}\b(guidance|outlook|forecast|full[- ]year)"
    r"|(guidance|outlook|forecast)\b[^.]{0,20}\b(cut|lowered|reduced|below expectations|disappoint)",
    re.I,
)


@dataclass(frozen=True)
class StageResult:
    ok: bool
    items: dict[str, bool | None]
    notes: dict[str, str] = field(default_factory=dict)


def _stage(items: dict[str, bool | None], guard: set[str], bonus: set[str],
           min_core: int | None, notes: dict[str, str]) -> StageResult:
    """43 페이지 규칙. None(측정 불가)은 guard 에선 통과로, core 에선 미통과로 센다.

    값은 파이썬 bool 로 정규화한다 — numpy.bool_ 는 ``is True``/``is not False``
    identity 비교가 항상 어긋나 guard 실패가 통과로, core 통과가 미통과로 세졌다.
    """
    items = {k: (None if v is None else bool(v)) for k, v in items.items()}
    core = [k for k in items if k not in guard and k not in bonus]
    guard_ok = all(items[k] is not False for k in guard if k in items)
    core_pass = sum(items[k] is True for k in core)
    need = len(core) if min_core is None else min_core
    return StageResult(ok=guard_ok and core_pass >= need, items=items, notes=notes)


def guidance_cut(titles: list[str]) -> bool:
    """2-8 — 헤드라인 중 가이던스/전망 하향 표현이 하나라도 있으면 True."""
    return any(_GUIDANCE_CUT.search(t or "") for t in titles)


def stage2(growth, annual, surprise, balance, guidance_titles: list[str] | None) -> StageResult:
    """2-1~2-4·2-7 중 3개 이상 + 경보(2-8·2-9·2-10) 없음. 분기 데이터 미확인 = 탈락."""
    notes: dict[str, str] = {}
    if growth is None or not growth.data_available:
        return StageResult(False, {"s2_1": None}, {"s2_1": "분기 펀더멘털 미확인 — 검증 불가라 탈락"})
    c = growth.checks
    items: dict[str, bool | None] = {
        "s2_1": c["2-1 EPS YoY"], "s2_2": c["2-2 EPS 가속"],
        "s2_3": c["2-3 매출"], "s2_4": c["2-4 마진 개선"],
        "s2_7": surprise.latest_beat if surprise is not None and surprise.data_available else None,
    }
    if guidance_titles is None:
        items["s2_8"], notes["s2_8"] = None, "뉴스 소스 없음"
    else:
        cut = guidance_cut(guidance_titles)
        items["s2_8"] = not cut
        notes["s2_8"] = f"헤드라인 {len(guidance_titles)}건" + (" — 하향 표현 발견" if cut else "")
    items["s2_9"] = balance.passed if balance is not None and balance.data_available else None
    items["s2_10"] = (None if annual is None or not annual.data_available or annual.decel_warning is None
                      else not annual.decel_warning)
    return _stage(items, guard={"s2_8", "s2_9", "s2_10"}, bonus=set(), min_core=3, notes=notes)


def stage3(manual: dict[str, bool | None], industry: str, leading_industries: set[str],
           liquidity_ok: bool) -> StageResult:
    """3-1·3-2·3-4·3-5 중 3개 이상 + 3-3·3-7 필수.

    3-5 주도 그룹 = 이번 스크린 RS 통과 종목 수 상위 산업군(``leading_industries``)에 포함.
    """
    items = {
        "s3_1": manual.get("m3_1"), "s3_2": manual.get("m3_2"),
        "s3_3": manual.get("m3_3"), "s3_4": manual.get("m3_4"),
        "s3_5": None if industry == "미분류" else industry in leading_industries,
        "s3_7": liquidity_ok,
    }
    return _stage(items, guard={"s3_3", "s3_7"}, bonus=set(), min_core=3, notes={})


def _base_range(df: pd.DataFrame, base_lookback: int = 120) -> tuple[float, float]:
    """pivot_levels 와 같은 정의의 베이스 천장(최근 5일 제외)·이후 최저가."""
    window = df.tail(base_lookback)
    struct = window["High"].iloc[:-5] if len(window) > 10 else window["High"]
    pos = int(struct.to_numpy().argmax())
    return float(struct.max()), float(window["Low"].iloc[pos:].min())


def stage4(df: pd.DataFrame, pl: PivotLevels, dryup_max: float = 0.8,
           chase_10d: float = 0.15, min_core: int | None = None) -> StageResult:
    """STAGE 4 — guard(4-3·4-10·4-11) 전부 + core ``min_core``개 (None = 전부, 43 페이지 원칙).

    2026-10 샘플 검증: core 7개를 코드 근사로 **전부** 요구하면 194종목 × 30거래일
    동안 통과가 0건이었다 (4% zigzag 축소 체감 등은 눈판독의 근사일 뿐). 그래서
    백테스트는 ``min_core`` 로 완화해 후보를 올리고 VCP 최종 판정은 사람 눈으로 한다.
    4-8·4-13 보너스, 4-12 는 돌파 시점에 판정."""
    up_ok, up_note = prior_uptrend(df, pl)
    last_c = pl.contractions[-1] if pl.contractions else pl.tightness_10d
    hi, lo = _base_range(df)
    handle_pos = (pl.pivot - lo) / (hi - lo) if hi > lo else None
    ret10 = float(df["Close"].iloc[-1] / df["Close"].iloc[-11] - 1.0) if len(df) > 11 else 0.0
    items: dict[str, bool | None] = {
        "s4_1": up_ok,
        "s4_2": None if pl.base_weeks is None else 3.0 <= pl.base_weeks <= 60.0,
        "s4_3": None if pl.base_depth is None else 0.10 <= pl.base_depth <= 0.35,
        "s4_4": 2 <= len(pl.contractions) <= 6,
        "s4_5": pl.contraction_ok,
        "s4_6": last_c < 0.10,
        "s4_7": pl.volume_dryup < dryup_max,
        "s4_9": None if handle_pos is None else handle_pos >= 2 / 3,
        "s4_10": ret10 < chase_10d and pl.dist_to_pivot <= 0.05,
        "s4_11": pl.method != "recent-high",
    }
    notes = {
        "s4_1": up_note,
        "s4_3": f"깊이 {pl.base_depth:.0%}" if pl.base_depth is not None else "",
        "s4_4": " → ".join(f"−{c:.0%}" for c in pl.contractions) or "축소 없음",
        "s4_6": f"마지막 축소 {last_c:.1%}",
        "s4_7": f"10일/50일 거래량 {pl.volume_dryup:.2f}",
        "s4_9": f"피봇이 베이스 레인지의 {handle_pos:.0%} 높이" if handle_pos is not None else "",
        "s4_10": f"10일 수익률 {ret10:+.0%}, 피봇 대비 {pl.dist_to_pivot:+.1%}",
        "s4_11": f"피봇 {pl.pivot:.2f} [{pl.method}]",
    }
    return _stage(items, guard={"s4_3", "s4_10", "s4_11"}, bonus=set(), min_core=min_core, notes=notes)


@dataclass(frozen=True)
class Entry:
    setup_date: date          # STAGE 4 가 완성된 날 (그날 종가까지의 데이터로 판정)
    entry_date: date          # 돌파일
    entry_price: float
    pivot: float
    breakout_volume_ratio: float
    stage4: StageResult


def find_watch_entry(
    df_full: pd.DataFrame, as_of: date, window: int = 30, vol_mult: float = 1.4,
    max_gap: float = 0.05, min_core4: int | None = None,
) -> Entry | None:
    """워치 → 진입. 기준일부터 ``window`` 거래일 동안 매일 그날까지의 데이터로
    추세 템플릿 + STAGE 4 를 다시 판정하고, 통과한 다음 날 피봇 돌파 +
    거래량 ≥ 50일 평균 × ``vol_mult`` (4-12) 이면 진입한다.

    - 체결가 = max(피봇, 시가). 시가가 피봇 +``max_gap`` 넘게 갭 상승이면 추격이라 건너뜀.
    - 돌파일 거래량은 그날 장 마감 기준 — 실전에선 장중 예상 거래량으로 판단하는 근사.
    """
    idx = df_full.index
    start = int(idx.searchsorted(pd.Timestamp(as_of), side="right"))  # as_of 다음 거래일
    for d in range(start, min(start + window, len(df_full))):
        hist = df_full.iloc[:d]                       # 돌파 후보일 직전까지
        tt = evaluate_trend_template(hist)
        if tt is None or not tt.passed:
            continue
        pl = pivot_levels(hist)
        if pl is None:
            continue
        s4 = stage4(hist, pl, min_core=min_core4)
        if not s4.ok:
            continue
        bar = df_full.iloc[d]
        vol50 = float(hist["Volume"].tail(50).mean())
        ratio = float(bar["Volume"]) / vol50 if vol50 > 0 else 0.0
        if float(bar["High"]) < pl.pivot or ratio < vol_mult:
            continue
        if float(bar["Open"]) > pl.pivot * (1 + max_gap):
            continue
        return Entry(setup_date=hist.index[-1].date(), entry_date=idx[d].date(),
                     entry_price=max(pl.pivot, float(bar["Open"])), pivot=pl.pivot,
                     breakout_volume_ratio=ratio, stage4=s4)
    return None


@dataclass(frozen=True)
class TradeResult:
    exit_reason: str          # stop | target | open
    exit_date: date | None
    pnl: float                # 청산 손익 (open 이면 마지막 종가 기준)
    days_held: int


def simulate_trade(df_full: pd.DataFrame, entry: Entry, stop_pct: float = 0.08,
                   target_pct: float = 0.22, max_days: int = 126) -> TradeResult:
    """진입일부터 손절/익절 선착 (같은 날 둘 다면 손절). ``max_days`` 넘으면 그날 종가 청산."""
    pos = int(df_full.index.searchsorted(pd.Timestamp(entry.entry_date)))
    held = df_full.iloc[pos: pos + max_days + 1]
    stop, target = entry.entry_price * (1 - stop_pct), entry.entry_price * (1 + target_pct)
    for i, (ts, row) in enumerate(held.iterrows()):
        if float(row["Low"]) <= stop:
            return TradeResult("stop", ts.date(), -stop_pct, i)
        if float(row["High"]) >= target:
            return TradeResult("target", ts.date(), target_pct, i)
    last = held.iloc[-1]
    reason = "time" if len(held) > max_days else "open"
    return TradeResult(reason, held.index[-1].date(),
                       float(last["Close"]) / entry.entry_price - 1.0, len(held) - 1)
