"""Alpha Vantage growth-fundamentals adapter — 분기 EPS·매출 + 섹터.

REST endpoint: ``GET https://www.alphavantage.co/query`` with::

    function=EARNINGS          → quarterlyEarnings
        {fiscalDateEnding, reportedDate, reportedEPS, estimatedEPS}
        — 1996년까지 풀 히스토리 + **발표일** (point-in-time의 핵심)
    function=INCOME_STATEMENT  → quarterlyReports
        {fiscalDateEnding, totalRevenue, netIncome} — ~2006년부터
    (같은 두 응답의 annualEarnings / annualReports 도 함께 저장 →
     연간 EPS·매출·순이익률: 미너비니 2-5 코드 33 · 2-6 · 2-10 판정)
    function=OVERVIEW          → Sector / Industry (산업군 집계용)

이 레포의 EODHD 플랜에는 Fundamentals API가 없어서(HTTP 403,
2026-09 확인) 성장 펀더멘털의 **기본 소스는 이 어댑터**다. 키는
``ALPHAVANTAGE_API_KEY`` (secrets — alphafolio 파이프라인과 공유).

티커당 3 HTTP 콜 → 반드시 가격 스크린 통과 종목에만 호출하고
디스크 캐시(기본 7일)한다. 레이트리밋 응답(JSON에 "Note"/
"Information"만 있음)은 잠깐 대기 후 재시도 — 소진 시 그 시점까지
모은 데이터로 반환 (Failure contract: raise 대신 quarters=[]).
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from data.domain.models import (
    AnnualFinancials,
    GrowthSnapshot,
    QuarterlyBalance,
    QuarterlyFinancials,
)

# 캐시 포맷 버전. 연간(annualEarnings/annualReports) 섹션이 추가된 v2부터
# 파일명에 붙는다 — 구버전 캐시(분기만)는 자연히 무시되고 재조회된다.
CACHE_VERSION = "v2"
from data.domain.ports import BalanceSheetPort, GrowthFundamentalsPort


def _parse_date(v: Any) -> date | None:
    try:
        return date.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None


def _parse_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None  # "None" 문자열 포함


class AlphaVantageGrowthFundamentalsAdapter(GrowthFundamentalsPort, BalanceSheetPort):
    """Alpha Vantage 기반 분기·연간 실적 + 재무상태표 fetcher + 디스크 캐시."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        cache_dir: str | os.PathLike = "/tmp/pattern-finder-cache/growth-fundamentals-av",
        cache_ttl_days: float = 7.0,
        include_profile: bool = True,
        rate_limit_retries: int = 3,
        rate_limit_wait_s: float = 15.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "AlphaVantageGrowthFundamentalsAdapter requires the "
                "ALPHAVANTAGE_API_KEY env var or an explicit api_key."
            )
        self._client = http_client or httpx.Client(timeout=60.0)
        self._cache_dir = Path(cache_dir)
        self._cache_ttl_s = cache_ttl_days * 86400.0
        self._include_profile = include_profile
        self._retries = rate_limit_retries
        self._wait_s = rate_limit_wait_s

    def fetch(self, symbol: str) -> GrowthSnapshot:
        raw = self._fetch_raw(symbol)
        if raw is None:
            return GrowthSnapshot(symbol=symbol)
        return self._parse(symbol, raw)

    # ---- 재무상태표 (2-9) — 별도 1콜, 별도 캐시 ------------------------

    def fetch_balance_sheet(self, symbol: str) -> list[QuarterlyBalance]:
        """``BALANCE_SHEET`` quarterlyReports → 재고·매출채권. 발표일은
        같은 분기의 EARNINGS reportedDate(캐시된 스냅샷)에서 채운다."""
        path = self._cache_dir / f"{symbol.replace('/', '_').replace('.', '_')}.bs.v1.json"
        data = None
        try:
            if path.exists() and (time.time() - path.stat().st_mtime) < self._cache_ttl_s:
                data = json.loads(path.read_text())
        except Exception:
            data = None
        if data is None:
            raw = self._query("BALANCE_SHEET", symbol)
            if raw is None:
                return []
            data = {"quarterlyReports": raw.get("quarterlyReports") or []}
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            except Exception:
                pass
        report_dates = {
            q.fiscal_date: q.report_date for q in self.fetch(symbol).quarters
        }
        out: list[QuarterlyBalance] = []
        for row in data.get("quarterlyReports") or []:
            if not isinstance(row, dict):
                continue
            fiscal = _parse_date(row.get("fiscalDateEnding"))
            if fiscal is None:
                continue
            out.append(QuarterlyBalance(
                symbol=symbol, fiscal_date=fiscal,
                report_date=report_dates.get(fiscal),
                inventory=_parse_float(row.get("inventory")),
                receivables=_parse_float(row.get("currentNetReceivables")),
            ))
        return sorted(out, key=lambda b: b.fiscal_date)

    # ---- raw + disk cache -------------------------------------------

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace(".", "_")
        return self._cache_dir / f"{safe}.{CACHE_VERSION}.json"

    def _query(self, function: str, symbol: str) -> dict | None:
        """단일 function 호출. 레이트리밋이면 대기-재시도, 실패 시 None."""
        for attempt in range(self._retries + 1):
            try:
                resp = self._client.get(
                    self.BASE_URL,
                    params={
                        "function": function,
                        "symbol": symbol,
                        "apikey": self._api_key,
                    },
                )
            except Exception:
                return None
            if resp.status_code != 200:
                return None
            try:
                data = resp.json()
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            # 레이트리밋/오류 응답은 데이터 키 없이 Note/Information/
            # Error Message만 담겨 온다.
            if any(k in data for k in ("Note", "Information", "Error Message")):
                if attempt < self._retries and (
                    "Note" in data or "Information" in data
                ):
                    time.sleep(self._wait_s)
                    continue
                return None
            return data
        return None

    def _fetch_raw(self, symbol: str) -> dict | None:
        path = self._cache_path(symbol)
        try:
            if path.exists() and (time.time() - path.stat().st_mtime) < self._cache_ttl_s:
                return json.loads(path.read_text())
        except Exception:
            pass
        earnings = self._query("EARNINGS", symbol)
        income = self._query("INCOME_STATEMENT", symbol)
        if earnings is None and income is None:
            return None
        overview = (
            self._query("OVERVIEW", symbol) if self._include_profile else None
        )
        slim = {
            "quarterlyEarnings": (earnings or {}).get("quarterlyEarnings") or [],
            "quarterlyReports": (income or {}).get("quarterlyReports") or [],
            "annualEarnings": (earnings or {}).get("annualEarnings") or [],
            "annualReports": (income or {}).get("annualReports") or [],
            "Sector": (overview or {}).get("Sector"),
            "Industry": (overview or {}).get("Industry"),
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(slim))
        except Exception:
            pass
        return slim

    # ---- parsing ----------------------------------------------------

    def _parse(self, symbol: str, data: dict) -> GrowthSnapshot:
        by_fiscal: dict[date, dict[str, Any]] = {}
        for row in data.get("quarterlyEarnings") or []:
            if not isinstance(row, dict):
                continue
            fiscal = _parse_date(row.get("fiscalDateEnding"))
            if fiscal is None:
                continue
            by_fiscal.setdefault(fiscal, {}).update(
                report_date=_parse_date(row.get("reportedDate")),
                eps_actual=_parse_float(row.get("reportedEPS")),
                eps_estimate=_parse_float(row.get("estimatedEPS")),
            )
        for row in data.get("quarterlyReports") or []:
            if not isinstance(row, dict):
                continue
            fiscal = _parse_date(row.get("fiscalDateEnding"))
            if fiscal is None:
                continue
            by_fiscal.setdefault(fiscal, {}).update(
                revenue=_parse_float(row.get("totalRevenue")),
                net_income=_parse_float(row.get("netIncome")),
            )
        # --- 연간: annualEarnings(EPS) + annualReports(매출·순이익) ---
        # AV는 연간 발표일을 주지 않는다 → 같은 회계연도 말 Q4 분기의
        # reportedDate를 연간 발표일로 쓴다 (연간 실적은 Q4와 함께 발표).
        by_year: dict[date, dict[str, Any]] = {}
        for row in data.get("annualEarnings") or []:
            if not isinstance(row, dict):
                continue
            fiscal = _parse_date(row.get("fiscalDateEnding"))
            if fiscal is None:
                continue
            by_year.setdefault(fiscal, {}).update(
                eps=_parse_float(row.get("reportedEPS")),
            )
        for row in data.get("annualReports") or []:
            if not isinstance(row, dict):
                continue
            fiscal = _parse_date(row.get("fiscalDateEnding"))
            if fiscal is None:
                continue
            by_year.setdefault(fiscal, {}).update(
                revenue=_parse_float(row.get("totalRevenue")),
                net_income=_parse_float(row.get("netIncome")),
            )
        for fiscal, fields in by_year.items():
            q4 = by_fiscal.get(fiscal)
            fields["report_date"] = q4.get("report_date") if q4 else None

        sector = data.get("Sector")
        industry = data.get("Industry")
        return GrowthSnapshot(
            symbol=symbol,
            sector=None if sector in (None, "None", "") else str(sector),
            industry=None if industry in (None, "None", "") else str(industry),
            quarters=[
                QuarterlyFinancials(symbol=symbol, fiscal_date=f, **fields)
                for f, fields in sorted(by_fiscal.items())
            ],
            annuals=[
                AnnualFinancials(symbol=symbol, fiscal_date=f, **fields)
                for f, fields in sorted(by_year.items())
            ],
        )
