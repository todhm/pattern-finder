"""EODHD growth-fundamentals adapter — 분기 EPS·매출 + 섹터.

REST endpoint::

    GET /api/fundamentals/{symbol}.US?api_token=...

⚠️ 이 엔드포인트는 EODHD 쿼터를 **요청 1건당 10 API 콜**로 계산한다.
그래서 (1) 반드시 가격 스크린을 통과한 소수 종목에만 호출하고,
(2) 결과를 디스크(JSON)에 캐시한다 — 과거 분기 실적은 불변이므로
TTL을 길게 잡아도 안전하다 (기본 7일).

payload에서 파싱하는 경로:

    General.Sector / General.Industry
    Earnings::History        → {fiscal: {reportDate, epsActual, epsEstimate}}
    Financials::Income_Statement::quarterly
                             → {fiscal: {filing_date, totalRevenue, netIncome}}

두 소스를 fiscal date(분기 말일) 기준으로 병합한다. 발표일은
Earnings History의 ``reportDate``를 우선, 없으면 Income Statement의
``filing_date`` — 둘 다 없으면 모델이 fiscal+90일로 보수 가정.

Failure contract: HTTP/파싱 실패 시 raise 대신 ``quarters=[]`` 반환
(:class:`GrowthFundamentalsPort` 계약) — 스크리너가 '미확인' 처리.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from data.domain.models import GrowthSnapshot, QuarterlyFinancials
from data.domain.ports import GrowthFundamentalsPort

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_date(v: Any) -> date | None:
    if isinstance(v, str) and _DATE_RE.match(v):
        try:
            return date.fromisoformat(v)
        except ValueError:
            return None
    return None


def _parse_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class EODHDGrowthFundamentalsAdapter(GrowthFundamentalsPort):
    """EODHD ``/fundamentals`` 기반 분기 실적 fetcher + 디스크 캐시."""

    DEFAULT_BASE_URL = "https://eodhd.com/api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        cache_dir: str | os.PathLike = "/tmp/pattern-finder-cache/growth-fundamentals",
        cache_ttl_days: float = 7.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("EODHD_API_KEY")
        if not self._api_key:
            raise ValueError(
                "EODHDGrowthFundamentalsAdapter requires the EODHD_API_KEY "
                "env var (loaded from secrets/secret.env) or an explicit "
                "api_key argument."
            )
        self._base_url = (
            base_url
            or os.environ.get("EODHD_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)
        self._cache_dir = Path(cache_dir)
        self._cache_ttl_s = cache_ttl_days * 86400.0

    def fetch(self, symbol: str) -> GrowthSnapshot:
        raw = self._fetch_raw(symbol)
        if raw is None:
            return GrowthSnapshot(symbol=symbol)
        return self._parse(symbol, raw)

    # ---- raw payload + disk cache ----------------------------------

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace(".", "_")
        return self._cache_dir / f"{safe}.json"

    def _fetch_raw(self, symbol: str) -> dict | None:
        path = self._cache_path(symbol)
        try:
            if path.exists() and (time.time() - path.stat().st_mtime) < self._cache_ttl_s:
                return json.loads(path.read_text())
        except Exception:
            pass  # 캐시 손상 → 원격 재조회
        eodhd_symbol = symbol if "." in symbol else f"{symbol}.US"
        url = f"{self._base_url}/fundamentals/{eodhd_symbol}"
        try:
            resp = self._client.get(
                url, params={"api_token": self._api_key, "fmt": "json"}
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
        # 캐시엔 필요한 블록만 저장 — 전체 payload는 티커당 수 MB까지 간다.
        slim = {
            "General": {
                k: (data.get("General") or {}).get(k)
                for k in ("Sector", "Industry")
            },
            "Earnings": {"History": (data.get("Earnings") or {}).get("History")},
            "Financials": {
                "Income_Statement": {
                    "quarterly": (
                        (data.get("Financials") or {}).get("Income_Statement")
                        or {}
                    ).get("quarterly")
                }
            },
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(slim))
        except Exception:
            pass
        return slim

    # ---- parsing ----------------------------------------------------

    def _parse(self, symbol: str, data: dict) -> GrowthSnapshot:
        general = data.get("General") or {}
        earnings_hist = (data.get("Earnings") or {}).get("History") or {}
        income_q = (
            (data.get("Financials") or {}).get("Income_Statement") or {}
        ).get("quarterly") or {}

        by_fiscal: dict[date, dict[str, Any]] = {}
        for key, row in earnings_hist.items():
            if not isinstance(row, dict):
                continue
            fiscal = _parse_date(row.get("date")) or _parse_date(key)
            if fiscal is None:
                continue
            by_fiscal.setdefault(fiscal, {}).update(
                report_date=_parse_date(row.get("reportDate")),
                eps_actual=_parse_float(row.get("epsActual")),
                eps_estimate=_parse_float(row.get("epsEstimate")),
            )
        for key, row in income_q.items():
            if not isinstance(row, dict):
                continue
            fiscal = _parse_date(row.get("date")) or _parse_date(key)
            if fiscal is None:
                continue
            slot = by_fiscal.setdefault(fiscal, {})
            if slot.get("report_date") is None:
                slot["report_date"] = _parse_date(row.get("filing_date"))
            slot.update(
                revenue=_parse_float(row.get("totalRevenue")),
                net_income=_parse_float(row.get("netIncome")),
            )

        quarters = [
            QuarterlyFinancials(symbol=symbol, fiscal_date=fiscal, **fields)
            for fiscal, fields in sorted(by_fiscal.items())
        ]
        return GrowthSnapshot(
            symbol=symbol,
            sector=general.get("Sector") or None,
            industry=general.get("Industry") or None,
            quarters=quarters,
        )
