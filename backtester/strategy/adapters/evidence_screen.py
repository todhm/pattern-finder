"""근거중심매매 스크리너 — 여러 종목에 체크리스트 엔진을 돌려 '매수할 만한' 종목만 추린다.

판정은 ``evidence_checklist.evaluate`` 의 자동 결과만 쓴다(채널·피보나치는 근사). 그래서 여기서 뽑힌 종목은
"후보"이고, 확정은 53 페이지에서 사람이 한다. 순수 함수 + 데이터 포트 주입 (streamlit 없음).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from strategy.adapters.evidence_checklist import EvidenceParams, TimeframeReport, evaluate, weighted_score
from strategy.adapters.minervini_scan import fetch_with_retry


@dataclass(frozen=True)
class ScreenFilter:
    min_weekly_yes: int = 3          # 주봉 5대 체크 '예' 최소 개수 (책: 4 베팅, 3 은 "근거 쌓는 중" 후보)
    min_score: float = 0.0           # 자동 가중치 합 최소
    exclude_worst_quadrant: bool = True   # 하락채널+정배열(최악) 제외
    require_above_ema20_weekly: bool = False
    years: int = 5                   # 조회 기간 (주봉 EMA200 = 약 4년)


def summarize(reports: dict[str, TimeframeReport]) -> dict:
    """evaluate 결과 → 표 한 줄."""
    w, d = reports["주봉"], reports["일봉"]
    score, parts = weighted_score(w.five, bool(w.supply.ok or d.supply.ok),
                                  bool(w.candle.vol_ratio and w.candle.vol_ratio >= 1.0))
    row = {"주봉 예": w.yes_count, "일봉 예": d.yes_count, "자동 가중치": round(score, 2)}
    for k, c in w.five.items():
        row[k[:1]] = "✅" if c.ok else ("❌" if c.ok is False else "—")
    row.update({
        "4분면(주봉)": w.channel.quadrant,
        "배열(주봉)": w.ma.alignment,
        "주봉 20MA 위": w.ma.above_ema20,
        "주봉 MA 접촉": ", ".join(w.ma.touching) or "—",
        "피보나치(주봉)": w.fib.zone if w.fib else "—",
        "피보나치(일봉)": d.fib.zone if d.fib else "—",
        "캔들(주봉)": ", ".join(w.candle.patterns) or "—",
        "캔들(일봉)": ", ".join(d.candle.patterns) or "—",
        "RSI(주봉)": round(w.div.rsi, 0),
        "다이버전스(주봉)": ("상승 " if w.div.bullish else "") + (f"하락×{w.div.bearish_count}" if w.div.bearish_count else ""),
        "매물대": "✅" if (w.supply.ok or d.supply.ok) else "—",
        "종가": round(float(d.df["Close"].iloc[-1]), 2),
        "가중치 구성": ", ".join(parts),
    })
    return row


def passes(row: dict, f: ScreenFilter) -> bool:
    if row["주봉 예"] < f.min_weekly_yes or row["자동 가중치"] < f.min_score:
        return False
    if f.exclude_worst_quadrant and "최악" in row["4분면(주봉)"]:
        return False
    if f.require_above_ema20_weekly and not row["주봉 20MA 위"]:
        return False
    return True


def apply_filter(all_df: pd.DataFrame, f: ScreenFilter) -> pd.DataFrame:
    """전체 표 → 통과 표 (정렬). 필터는 표에 다시 적용할 수 있으므로 스캔을 다시 돌릴 필요가 없다."""
    if all_df.empty or "주봉 예" not in all_df:
        return all_df.iloc[0:0]
    ok = all_df[all_df["오류"].isna()] if "오류" in all_df else all_df
    mask = ok.apply(lambda r: passes(r, f), axis=1) if len(ok) else pd.Series(dtype=bool)
    hit = ok[mask].sort_values(["주봉 예", "자동 가중치", "일봉 예"], ascending=False) if len(ok) else ok
    return hit.reset_index(drop=True)


def _row_from_df(ticker: str, df: pd.DataFrame, as_of: date, p: EvidenceParams) -> dict:
    try:
        return {"티커": ticker, **summarize(evaluate(df, as_of, None, p))}
    except Exception as e:  # noqa: BLE001
        return {"티커": ticker, "오류": f"{type(e).__name__}: {str(e)[:80]}"}


def screen_one(market_data, ticker: str, as_of: date, p: EvidenceParams, years: int) -> dict:
    df = fetch_with_retry(market_data, ticker, as_of - timedelta(days=365 * years), as_of)
    return _row_from_df(ticker, df, as_of, p)


def compute_rows(prices: dict[str, pd.DataFrame], as_of: date, p: EvidenceParams = EvidenceParams(),
                 max_workers: int = 4, progress: Callable[[int, int, str], None] | None = None) -> pd.DataFrame:
    """이미 받아 둔 {티커: 일봉} 으로 계산만 (종목당 약 0.03초)."""
    items = list(prices.items())
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, row in enumerate(ex.map(lambda kv: _row_from_df(kv[0], kv[1], as_of, p), items), 1):
            rows.append(row)
            if progress:
                progress(i, len(items), row["티커"])
    return pd.DataFrame(rows)


def run_screen(market_data, tickers: list[str], as_of: date, p: EvidenceParams = EvidenceParams(),
               f: ScreenFilter = ScreenFilter(), max_workers: int = 2,
               progress: Callable[[int, int, str], None] | None = None,
               prices: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(통과 표, 전체 표). ``prices`` 가 있으면 조회 없이 계산만, 없으면 캐시 스택으로 종목별 조회."""
    if prices is not None:
        missing = [t for t in tickers if t not in prices]
        all_df = compute_rows({t: prices[t] for t in tickers if t in prices}, as_of, p, max(max_workers, 4), progress)
        if missing:
            all_df = pd.concat([all_df, pd.DataFrame([{"티커": t, "오류": "데이터 없음"} for t in missing])], ignore_index=True)
        return apply_filter(all_df, f), all_df
    rows: list[dict] = []
    done = 0

    def _one(t: str) -> dict:
        try:
            return screen_one(market_data, t, as_of, p, f.years)
        except Exception as e:  # noqa: BLE001
            return {"티커": t, "오류": f"{type(e).__name__}: {str(e)[:80]}"}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in ex.map(_one, tickers):
            rows.append(row)
            done += 1
            if progress:
                progress(done, len(tickers), row["티커"])
    all_df = pd.DataFrame(rows)
    return apply_filter(all_df, f), all_df
