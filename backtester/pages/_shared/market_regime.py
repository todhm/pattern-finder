"""STAGE 0 시장 환경 — SPY + QQQ 이중 판정 (44 스크리너 · 45 리플레이 공용).

지수 일봉을 받아 :func:`evaluate_market` 로 지수별 0-1·0-2 를 판정하고
:func:`combine_market` 로 합친다. 순수 로직은 어댑터에, 여기는 fetch 와
Streamlit 표시 문자열만.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from strategy.adapters.minervini_screener import (
    MarketVerdict,
    combine_market,
    evaluate_market,
)

INDEX_SYMBOLS = ("SPY", "QQQ")


def fetch_market_verdicts(
    market_data, as_of: date, symbols: tuple[str, ...] = INDEX_SYMBOLS,
) -> tuple[list[MarketVerdict], list[str]]:
    """지수별 판정 리스트 + 조회 실패 메시지. 실패한 지수는 건너뛴다."""
    verdicts: list[MarketVerdict] = []
    errors: list[str] = []
    start = as_of - timedelta(days=460)
    for sym in symbols:
        try:
            df = market_data.fetch_ohlcv(sym, start, as_of)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[df.index <= pd.Timestamp(as_of)]
            v = evaluate_market(df, symbol=sym)
            if v is None:
                errors.append(f"{sym}: 봉 부족 ({len(df)})")
            else:
                verdicts.append(v)
        except Exception as e:  # noqa: BLE001 — 지수 하나 실패해도 나머지로 판정
            errors.append(f"{sym} 조회 실패: {e}")
    return verdicts, errors


def verdicts_to_state(verdicts: list[MarketVerdict]) -> list[dict]:
    """세션 저장용 직렬화 (dataclass → dict)."""
    return [
        {"symbol": v.symbol, "close": v.close, "sma200": v.sma200,
         "sma200_prev21": v.sma200_prev21, "trend_ok": v.trend_ok,
         "dist_days": v.dist_days, "dist_ok": v.dist_ok, "note": v.note}
        for v in verdicts
    ]


def stage0_lines(rows: list[dict]) -> tuple[list[str], bool, bool]:
    """표시용 — (지수별 0-1/0-2 줄, 0-1 전부 통과, 0-2 전부 통과)."""
    lines: list[str] = []
    trend_all = all(r["trend_ok"] for r in rows) if rows else False
    dist_all = all(r["dist_ok"] is not False for r in rows) if rows else False
    for r in rows:
        arrow = "상승" if r["sma200"] > r["sma200_prev21"] else "하락"
        lines.append(
            f"{'✅' if r['trend_ok'] else '❌'} **0-1 {r['symbol']} 추세** — "
            f"{r['close']:,.0f} vs 200일선 {r['sma200']:,.0f} ({arrow} 중)"
        )
    for r in rows:
        dd = r["dist_days"]
        mark = "ℹ️" if dd is None else ("✅" if r["dist_ok"] else "❌")
        lines.append(
            f"{mark} **0-2 {r['symbol']} 분산일** — 최근 25거래일 중 "
            f"{'?' if dd is None else dd}회 (하락 −0.2%↑ + 거래량 증가일. 5회↑ = 기관 매도 경고)"
        )
    return lines, trend_all, dist_all


def combine_rows(rows: list[dict]) -> tuple[bool | None, list[str]]:
    """state dict 리스트에 대해 combine_market 과 같은 규칙."""
    vs = [
        MarketVerdict(symbol=r["symbol"], close=r["close"], sma200=r["sma200"],
                      sma200_prev21=r["sma200_prev21"], trend_ok=r["trend_ok"],
                      dist_days=r["dist_days"], dist_ok=r["dist_ok"])
        for r in rows
    ]
    return combine_market(vs)
