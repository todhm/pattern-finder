"""Alpha Vantage 재무제표 3종 어댑터 — "재무제표 읽기" 품질 판정용 (``FinancialStatementsPort``).

    function=INCOME_STATEMENT  quarterlyReports  — 매출·매출총이익·영업이익·R&D·SG&A·D&A·이자·세전·순이익
    function=BALANCE_SHEET     quarterlyReports  — 자산·부채·현금·차입·자본·재고·매출채권·이연매출·주식 수
    function=CASH_FLOW         quarterlyReports  — 영업현금흐름·capex·배당·자사주·증자·투자/재무 CF
    발표일은 같은 분기의 EARNINGS reportedDate (성장 어댑터 캐시 ``{sym}.v2.json`` 재사용 → 추가 콜 없음).

캐시: 성장 어댑터와 같은 디렉터리(``/tmp/pattern-finder-cache/growth-fundamentals-av``).
  - 손익·발표일: ``AlphaVantageGrowthFundamentalsAdapter.fetch_raw_statements`` (그 어댑터의 TTL)
  - 재무상태표 ``{sym}.bs.v1.json`` (기존 2-9 캐시와 같은 파일·포맷 ``{"quarterlyReports": [...]}``)
  - 현금흐름표 ``{sym}.cf.v1.json``
  분기 재무제표는 과거가 바뀌지 않으므로 TTL 기본 30일. 레이트리밋(Note/Information)이면 대기 후 재시도.
  ``pace_s`` 는 HTTP 콜마다 잠깐 쉬어 분당 한도(75)를 지키는 용도(배치 CLI 가 준다).

실패 계약: raise 대신 빈 리스트. 회계연도 말일이 엔드포인트마다 하루 이틀 다를 수 있어(52주 결산)
분기 매칭은 ±10일 안의 가장 가까운 fiscalDateEnding 으로 한다.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

from data.adapters.alphavantage_growth_fundamentals import (
    AlphaVantageGrowthFundamentalsAdapter,
    _parse_date,
    _parse_float,
)
from data.domain.models import QuarterlyStatement
from data.domain.ports import FinancialStatementsPort


def _f(row: dict, *keys: str) -> float | None:
    """여러 후보 키 중 처음 값이 있는 것."""
    for k in keys:
        v = _parse_float(row.get(k))
        if v is not None:
            return v
    return None


def _nearest(target: date, pool: dict[date, Any], tol_days: int = 10) -> Any:
    best = None; best_d = tol_days + 1
    for d, v in pool.items():
        dd = abs((d - target).days)
        if dd < best_d:
            best, best_d = v, dd
    return best


class AlphaVantageStatementsAdapter(FinancialStatementsPort):
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        cache_dir: str | os.PathLike = "/tmp/pattern-finder-cache/growth-fundamentals-av",
        cache_ttl_days: float = 30.0,
        growth: AlphaVantageGrowthFundamentalsAdapter | None = None,
        pace_s: float = 0.0,
        rate_limit_retries: int = 3,
        rate_limit_wait_s: float = 15.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self._api_key:
            raise ValueError("AlphaVantageStatementsAdapter requires ALPHAVANTAGE_API_KEY.")
        self._client = http_client or httpx.Client(timeout=60.0)
        self._cache_dir = Path(cache_dir)
        self._ttl_s = cache_ttl_days * 86400.0
        self._growth = growth or AlphaVantageGrowthFundamentalsAdapter(
            api_key=self._api_key, http_client=self._client, cache_dir=cache_dir)
        self._pace = pace_s
        self._retries = rate_limit_retries
        self._wait_s = rate_limit_wait_s

    # ---- HTTP ---------------------------------------------------------

    def _query(self, function: str, symbol: str) -> dict | None:
        for attempt in range(self._retries + 1):
            try:
                resp = self._client.get(self.BASE_URL, params={"function": function, "symbol": symbol, "apikey": self._api_key})
            except Exception:
                return None
            finally:
                if self._pace > 0:
                    time.sleep(self._pace)
            if resp.status_code != 200:
                return None
            try:
                data = resp.json()
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            if any(k in data for k in ("Note", "Information", "Error Message")):
                if attempt < self._retries and ("Note" in data or "Information" in data):
                    time.sleep(self._wait_s); continue
                return None
            return data
        return None

    def _cached_reports(self, symbol: str, function: str, suffix: str) -> list[dict]:
        safe = symbol.replace("/", "_").replace(".", "_")
        path = self._cache_dir / f"{safe}.{suffix}.v1.json"
        try:
            if path.exists() and (time.time() - path.stat().st_mtime) < self._ttl_s:
                return json.loads(path.read_text()).get("quarterlyReports") or []
        except Exception:
            pass
        raw = self._query(function, symbol)
        if raw is None:
            # 오래된 캐시라도 있으면 그걸 쓴다 (레이트리밋·장애)
            try:
                if path.exists():
                    return json.loads(path.read_text()).get("quarterlyReports") or []
            except Exception:
                pass
            return []
        data = {"quarterlyReports": raw.get("quarterlyReports") or []}
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        except Exception:
            pass
        return data["quarterlyReports"]

    # ---- port -----------------------------------------------------------

    def fetch_statements(self, symbol: str) -> list[QuarterlyStatement]:
        raw = self._growth.fetch_raw_statements(symbol) or {}
        income = [r for r in (raw.get("quarterlyReports") or []) if isinstance(r, dict)]
        if not income:
            return []
        rep_dates: dict[date, date] = {}
        for r in raw.get("quarterlyEarnings") or []:
            fd, rd = _parse_date(r.get("fiscalDateEnding")), _parse_date(r.get("reportedDate"))
            if fd and rd:
                rep_dates[fd] = rd
        bs = {d: r for r in self._cached_reports(symbol, "BALANCE_SHEET", "bs") if (d := _parse_date(r.get("fiscalDateEnding")))}
        cf = {d: r for r in self._cached_reports(symbol, "CASH_FLOW", "cf") if (d := _parse_date(r.get("fiscalDateEnding")))}
        out: list[QuarterlyStatement] = []
        for r in income:
            fd = _parse_date(r.get("fiscalDateEnding"))
            if fd is None:
                continue
            b = _nearest(fd, bs) or {}
            c = _nearest(fd, cf) or {}
            cash = _f(b, "cashAndShortTermInvestments")
            if cash is None:
                ce, sti = _f(b, "cashAndCashEquivalentsAtCarryingValue"), _f(b, "shortTermInvestments")
                cash = None if ce is None else ce + (sti or 0.0)
            std = _f(b, "shortTermDebt", "currentDebt", "currentLongTermDebt")
            ltd = _f(b, "longTermDebt", "longTermDebtNoncurrent")
            tdebt = _f(b, "shortLongTermDebtTotal")
            if tdebt is None and (std is not None or ltd is not None):
                tdebt = (std or 0.0) + (ltd or 0.0)
            out.append(QuarterlyStatement(
                symbol=symbol, fiscal_date=fd, report_date=_nearest(fd, rep_dates),
                revenue=_f(r, "totalRevenue"), cost_of_revenue=_f(r, "costOfRevenue", "costofGoodsAndServicesSold"),
                gross_profit=_f(r, "grossProfit"), operating_income=_f(r, "operatingIncome"),
                rnd=_f(r, "researchAndDevelopment"), sga=_f(r, "sellingGeneralAndAdministrative"),
                da=_f(r, "depreciationAndAmortization", "depreciation"), ebit=_f(r, "ebit"), ebitda=_f(r, "ebitda"),
                interest_expense=_f(r, "interestExpense", "interestAndDebtExpense"),
                pretax_income=_f(r, "incomeBeforeTax"), income_tax=_f(r, "incomeTaxExpense"), net_income=_f(r, "netIncome"),
                total_assets=_f(b, "totalAssets"), total_current_assets=_f(b, "totalCurrentAssets"),
                total_current_liabilities=_f(b, "totalCurrentLiabilities"), cash_and_st_investments=cash,
                short_term_debt=std, long_term_debt=ltd, total_debt=tdebt, total_liabilities=_f(b, "totalLiabilities"),
                equity=_f(b, "totalShareholderEquity"), inventory=_f(b, "inventory"), receivables=_f(b, "currentNetReceivables"),
                deferred_revenue=_f(b, "deferredRevenue"), shares_outstanding=_f(b, "commonStockSharesOutstanding"),
                goodwill=_f(b, "goodwill"), retained_earnings=_f(b, "retainedEarnings"),
                operating_cf=_f(c, "operatingCashflow"), capex=_f(c, "capitalExpenditures"), dividends=_f(c, "dividendPayout"),
                buybacks=_f(c, "paymentsForRepurchaseOfCommonStock", "paymentsForRepurchaseOfEquity", "proceedsFromRepurchaseOfEquity"),
                stock_issuance=_f(c, "proceedsFromIssuanceOfCommonStock"), sbc=_f(c, "stockBasedCompensation"),
                investing_cf=_f(c, "cashflowFromInvestment"), financing_cf=_f(c, "cashflowFromFinancing"),
            ))
        return sorted(out, key=lambda q: q.fiscal_date)
