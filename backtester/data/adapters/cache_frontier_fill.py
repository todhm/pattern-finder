"""캐시 앞단 채우기 — 날짜 단위 캐시의 최근 거래일을 전 종목 1콜로 채운다.

왜 필요한가 (2026-10 실측): ``MongoDayCacheAdapter`` 는 (종목, 날짜) 단위라
기준일이 거래일 하루만 앞으로 가도 **모든** 종목에 빠진 날이 생기고, 그 하루를
채우려고 종목마다 Alpha Vantage 전체 히스토리 1콜(분당 75콜 상한)을 다시 한다.
nasdaq_full 5,000 종목 = 64분. 이 모듈은 그 전에 ``BulkDailyBarsPort`` 로
빠진 거래일을 날짜당 1콜로 채워 종목별 루프를 전부 캐시 적중으로 만든다.

규칙
- 오늘(컨테이너 UTC 기준)은 부분 봉일 수 있어 채우지 않는다.
- 최신 거래일은 항상 1콜 받아 그날 프레임에 있는 유니버스 심볼 수를 분모로 삼고,
  나머지 날은 확정 success 문서가 그 분모의 ``min_coverage`` 이상이면 건너뛴다.
- 빈 응답(휴장)은 같은 실행에서 다른 날이 정상 응답했을 때만 휴장으로 기록한다
  (소스 장애를 휴장으로 영구 기록하는 사고 방지).
- 구간 안에 분할이 발효된 종목은 분할 전에 받은 문서가 분할 전 가격이므로
  캐시를 비워 종목별 소스(조정가)로 다시 받게 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import pandas as pd

from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.domain.ports import BulkDailyBarsPort


@dataclass
class FillReport:
    as_of: date
    days_checked: list[date] = field(default_factory=list)
    days_filled: list[date] = field(default_factory=list)
    days_holiday: list[date] = field(default_factory=list)
    days_skipped_covered: list[date] = field(default_factory=list)
    days_deferred: list[date] = field(default_factory=list)  # 빈 응답인데 건강 확인 불가
    docs_written: int = 0
    splits_dropped: dict[str, int] = field(default_factory=dict)
    dead_marked: int = 0  # 폐지 심볼에 쓴 nodata 마커 수 (심볼×날짜×캐시)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"확인 {len(self.days_checked)}일",
                 f"채움 {len(self.days_filled)}일 ({self.docs_written:,} 문서)"]
        if self.days_holiday:
            parts.append(f"휴장 {len(self.days_holiday)}일")
        if self.days_deferred:
            parts.append(f"보류 {len(self.days_deferred)}일")
        if self.dead_marked:
            parts.append(f"폐지 마커 {self.dead_marked:,}")
        if self.splits_dropped:
            parts.append(f"분할 재조회 {len(self.splits_dropped)}종목 "
                         f"({', '.join(sorted(self.splits_dropped)[:8])})")
        if self.errors:
            parts.append(f"오류 {len(self.errors)}건: {self.errors[0][:120]}")
        return " · ".join(parts)


def fill_recent_days(
    cache: MongoDayCacheAdapter,
    bulk: BulkDailyBarsPort,
    symbols: list[str],
    as_of: date,
    *,
    lookback_bdays: int = 15,
    min_coverage: float = 0.95,
    today: date | None = None,
    marker_caches: list[MongoDayCacheAdapter] | None = None,
    log: Callable[[str], None] | None = None,
) -> FillReport:
    """``as_of`` 까지 최근 ``lookback_bdays`` 거래일의 캐시 앞단을 채운다.

    ``marker_caches``: 일봉 폴백 체인의 캐시 전부(AV·yfinance·EODHD·Massive).
    grouped 에 없는 심볼 중 최근 ``lookback_bdays`` 안에 어느 소스가 이미
    'No data found' 를 기록한 심볼(= 폐지)은 그날 nodata 마커를 모든 캐시에
    써서 네 소스를 차례로 두드리는 비용(심볼당 3~4콜)을 없앤다. None 이면
    ``cache`` 에만 쓴다.
    """
    today = today or date.today()
    symbols = list(dict.fromkeys(s for s in symbols if s))
    rep = FillReport(as_of=as_of)
    if not symbols:
        return rep
    end = min(as_of, today - timedelta(days=1))
    days = [ts.date() for ts in pd.bdate_range(end=end, periods=lookback_bdays)]
    rep.days_checked = days
    need = max(1, math.ceil(len(symbols) * min_coverage))

    # 1) 분할: 구간 안에 발효된 종목은 캐시 무효화 (grouped 봉은 분할 후 가격)
    try:
        split_syms = set(bulk.split_symbols(days[0], max(as_of, days[-1])))
    except Exception as exc:  # noqa: BLE001
        split_syms = set()
        rep.errors.append(f"splits: {type(exc).__name__}: {exc}")
    for sym in sorted(split_syms & set(symbols)):
        # 분할 발효일 00:00 UTC 이전 문서가 있으면 전부 삭제 → 종목별 소스가 조정가로 재조회
        before = datetime.combine(days[0], datetime.min.time(), tzinfo=timezone.utc)
        n = cache.drop_symbol_if_fetched_before(sym, before)
        if n:
            rep.splits_dropped[sym] = n

    # 2) 날짜별 채우기 — 최신일부터. 유니버스(@기준일)에는 두 소스 모두 모르는
    #    폐지·비상장 심볼이 섞여 있어 "유니버스 대비 95%" 는 영원히 못 채운다.
    #    그래서 처음 받은 거래일 프레임에 있는 심볼 수(alive)를 분모로 쓴다.
    healthy = False  # 이번 실행에서 비어 있지 않은 응답을 한 번이라도 받았나
    need: int | None = None
    pending_empty: list[date] = []
    for d in reversed(days):
        if need is not None and cache.coverage(d, symbols) >= need:
            rep.days_skipped_covered.append(d)
            continue
        if log:
            log(f"캐시 채우기 {d} (전 종목 1콜)")
        try:
            frame = bulk.fetch_day(d)
        except Exception as exc:  # noqa: BLE001
            rep.errors.append(f"{d}: {type(exc).__name__}: {exc}")
            continue
        if frame is None or frame.empty:
            if healthy:
                rep.docs_written += cache.fill_day(d, frame, symbols)
                rep.days_holiday.append(d)
            else:
                pending_empty.append(d)
            continue
        if need is None:
            alive = len(set(symbols) & set(frame.index))
            need = max(1, math.ceil(alive * min_coverage))
            if cache.coverage(d, symbols) >= need:
                # 최신일이 이미 차 있으면 이 날은 다시 쓰지 않는다 (분모만 얻은 셈).
                # 폐지 마커는 그래도 쓴다 — 어제 종목별 경로가 죽었다고 기록한
                # 심볼이 오늘 네 소스를 다시 두드리지 않게.
                healthy = True
                rep.days_skipped_covered.append(d)
                _mark_dead(cache, marker_caches or [cache], d, days, symbols, frame, rep)
                continue
        healthy = True
        rep.docs_written += cache.fill_day(d, frame, symbols)
        rep.days_filled.append(d)
        _mark_dead(cache, marker_caches or [cache], d, days, symbols, frame, rep)
        # 먼저 나온(더 최신인) 빈 날들은 이제 휴장으로 확정
        for pe in pending_empty:
            rep.docs_written += cache.fill_day(pe, pd.DataFrame(), symbols)
            rep.days_holiday.append(pe)
        pending_empty = []
    for lst in (rep.days_filled, rep.days_holiday, rep.days_skipped_covered):
        lst.sort()
    rep.days_deferred = sorted(pending_empty)
    return rep


def _mark_dead(
    cache: MongoDayCacheAdapter,
    marker_caches: list[MongoDayCacheAdapter],
    d: date,
    days: list[date],
    symbols: list[str],
    frame: pd.DataFrame,
    rep: FillReport,
) -> None:
    """거래일 ``d`` 의 grouped 프레임에 없는 심볼 중 '이미 죽었다고 기록된' 심볼에
    nodata 마커를 쓴다. 증거 두 가지가 겹칠 때만: ① 전 시장 1콜에 봉이 없다,
    ② 최근 창 안에 어느 캐시든 No data found 마커가 있다 (처음 보는 심볼은
    종목별 경로가 한 번은 직접 확인하도록 남긴다)."""
    absent = [s for s in symbols if s not in frame.index]
    if not absent:
        return
    dead: set[str] = set()
    for mc in marker_caches:
        dead |= mc.recent_nodata_symbols(days, absent)
    if not dead:
        return
    dead_list = sorted(dead)
    for mc in marker_caches:
        try:
            rep.dead_marked += mc.mark_nodata_days([d], dead_list)
        except Exception as exc:  # noqa: BLE001
            rep.errors.append(f"nodata marker {d}: {type(exc).__name__}: {exc}")
