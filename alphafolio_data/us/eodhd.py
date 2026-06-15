"""EODHD Fundamentals collector — `/fundamentals/{sym}.US` 1 콜 → 4 테이블 통합 적재.

응답 구조:
  General, Highlights, Valuation, SharesStats, Technicals, SplitsDividends,
  AnalystRatings, Holders, ESGScores, outstandingShares, Earnings, Financials
  - Highlights/Valuation/SharesStats/Technicals/AnalystRatings/SplitsDividends/General
    → us_stock_basic latest snapshot (source='api', date=today)
  - Financials.Income_Statement.quarterly  → us_income_statement  (분기 시계열, 163 quarters since 1985 for AAPL)
  - Financials.Balance_Sheet.quarterly     → us_balance_sheet
  - Financials.Cash_Flow.quarterly         → us_cash_flow

API quota:
  - paid plan: 100,000 unit/day
  - 1 fundamentals call = 10 unit
  - 분당 limit 1200 unit → ~120 calls/min
  - 6,200 active 종목 × 10 = 62,000 unit/day (62% 사용), 약 52분 소요

각 종목 단위로 4 테이블 atomic transaction.
"""
import asyncio
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import asyncpg

try:
    from collection_logger import CollectionLogger  # type: ignore
except Exception:  # pragma: no cover - test/standalone fallback
    CollectionLogger = None  # type: ignore

logger = logging.getLogger(__name__)


USER_AGENT_HEADERS = {"User-Agent": "alphafolio-data/1.0 (+eodhd)"}


class RateLimitError(Exception):
    """EODHD daily budget 소진 시 raise."""


# ---------------------------------------------------------------------------
# 안전 변환 헬퍼 (AV safe_convert 패턴과 동일)
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "" or v == "None" or v == "-" or v == "N/A":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    f = _to_float(v)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError):
        return None


def _to_bigint(v: Any) -> Optional[int]:
    return _to_int(v)


def _to_date(v: Any) -> Optional[date]:
    if v is None or v == "" or v == "None":
        return None
    if isinstance(v, date):
        return v
    try:
        # "2026-03-31" 또는 "2026-Q1" 같은 양식 처리
        s = str(v)
        if "T" in s:
            s = s.split("T", 1)[0]
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _pct_clamp(v: Any, cap: float = 100.0) -> Optional[float]:
    """Percentage 단위 값. EODHD 가 가끔 100 초과 (MYRG 103.12% 같은 data anomaly)
    → cap 으로 clamp."""
    f = _to_float(v)
    if f is None:
        return None
    return min(f, cap)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class EODHDFundamentalsCollector:
    BASE_URL = "https://eodhd.com/api/fundamentals/{sym}.US"
    USER_URL = "https://eodhd.com/api/user"

    def __init__(
        self,
        api_token: str,
        database_url: str,
        *,
        call_interval: float = 0.06,
        max_concurrent: int = 4,
        daily_unit_budget: int = 100_000,
        target_symbols: Optional[List[str]] = None,
        deadline: Optional[datetime] = None,
    ):
        self.api_token = api_token
        self.database_url = database_url
        self.call_interval = call_interval
        self.max_concurrent = max_concurrent
        self.daily_unit_budget = daily_unit_budget
        self.target_symbols = target_symbols
        self.deadline = deadline

        self.pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 사용량 모니터링
        self.api_requests_used = 0
        self.daily_rate_limit = daily_unit_budget

        # collection_logger (있으면 사용, fallback OK)
        if CollectionLogger is not None:
            log_path = (
                "/app/log/us_eodhd_fundamentals_collected.json"
                if os.getenv("RAILWAY_PROJECT_ID")
                else "log/us_eodhd_fundamentals_collected.json"
            )
            self.collection_logger = CollectionLogger(log_path)
        else:
            self.collection_logger = None

    async def init_pool(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.database_url, min_size=2, max_size=10
            )
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=USER_AGENT_HEADERS)

    async def close_pool(self):
        if self.session:
            await self.session.close()
            self.session = None
        if self.pool:
            await self.pool.close()
            self.pool = None

    # ------------------------------------------------------------------
    # universe / no_data
    # ------------------------------------------------------------------

    async def get_active_symbols(self) -> List[str]:
        """Universe fallback chain:
          1. us_stock_basic 의 is_active=true  (반복 run 시 가장 정확)
          2. (1)이 비어있으면 us_listing_status 의 active stock × NASDAQ/NYSE
             (fresh DB 의 first run 케이스 — TRUNCATE 직후 또는 초기 setup)
          3. (2)도 비어있으면 us_symbol (finnhub_symbol task 의 master list)
        """
        if self.target_symbols:
            return list(self.target_symbols)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT symbol FROM us_stock_basic "
                "WHERE is_active = true ORDER BY symbol"
            )
            if rows:
                return [r["symbol"] for r in rows]
            logger.warning(
                "[EODHD] us_stock_basic empty → fallback to us_listing_status"
            )
            rows = await conn.fetch(
                """SELECT DISTINCT symbol FROM us_listing_status
                   WHERE delisting_date IS NULL
                     AND asset_type = 'Stock'
                     AND exchange IN ('NASDAQ', 'NYSE', 'NYSE MKT', 'NYSE ARCA', 'AMEX', 'BATS')
                   ORDER BY symbol"""
            )
            if rows:
                return [r["symbol"] for r in rows]
            logger.warning(
                "[EODHD] us_listing_status empty → fallback to us_symbol"
            )
            rows = await conn.fetch(
                "SELECT DISTINCT symbol FROM us_symbol ORDER BY symbol"
            )
        return [r["symbol"] for r in rows]

    async def get_no_data_skip(self) -> set:
        """최근 7일 안에 no_data 마킹된 종목 (영구 fail 재시도 방지)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT DISTINCT symbol FROM collection_state
                   WHERE collection_name = 'us_eodhd_fundamentals'
                     AND status = 'no_data'
                     AND date >= CURRENT_DATE - 7"""
            )
        return {r["symbol"] for r in rows}

    async def _mark_no_data(self, symbol: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO collection_state
                   (collection_name, symbol, date, status, updated_at)
                   VALUES ('us_eodhd_fundamentals', $1, CURRENT_DATE, 'no_data', NOW())
                   ON CONFLICT (collection_name, symbol, date) DO UPDATE
                   SET status = 'no_data', updated_at = NOW()""",
                symbol,
            )

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------

    async def _fetch_one(
        self, symbol: str, max_retries: int = 4
    ) -> Optional[Dict[str, Any]]:
        url = self.BASE_URL.format(sym=symbol)
        params = {"api_token": self.api_token, "fmt": "json"}
        for attempt in range(max_retries):
            try:
                async with self.session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    # budget 모니터링 (헤더 없으면 무시)
                    used_hdr = r.headers.get("x-apirequests-used")
                    lim_hdr = r.headers.get("x-apirequests-limit")
                    if used_hdr:
                        self.api_requests_used = int(used_hdr)
                    if lim_hdr:
                        self.daily_rate_limit = int(lim_hdr)
                    remaining = self.daily_rate_limit - self.api_requests_used
                    if r.status == 429:
                        wait = min(5 * (2 ** attempt), 60)
                        logger.warning(
                            f"[EODHD] {symbol} 429 → sleep {wait}s"
                        )
                        await asyncio.sleep(wait)
                        continue
                    if r.status == 402 or (
                        self.daily_rate_limit > 0 and remaining < 20
                    ):
                        raise RateLimitError(
                            f"budget {self.api_requests_used}/"
                            f"{self.daily_rate_limit} — abort"
                        )
                    if r.status == 404:
                        return None
                    if r.status != 200:
                        logger.warning(
                            f"[EODHD] {symbol} HTTP {r.status}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(5 * (2 ** attempt))
                            continue
                        return None
                    data = await r.json()
                    if not data or "General" not in data:
                        return None
                    return data
            except asyncio.TimeoutError:
                logger.warning(f"[EODHD] {symbol} timeout attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                return None
            except RateLimitError:
                raise
            except Exception as e:
                logger.error(f"[EODHD] {symbol} fetch error: {e}")
                return None
        return None

    # ------------------------------------------------------------------
    # transforms
    # ------------------------------------------------------------------

    def transform_stock_basic(
        self, data: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        G = data.get("General") or {}
        H = data.get("Highlights") or {}
        V = data.get("Valuation") or {}
        S = data.get("SharesStats") or {}
        T = data.get("Technicals") or {}
        SD = data.get("SplitsDividends") or {}
        AR = data.get("AnalystRatings") or {}
        return {
            "symbol": symbol,
            "assettype": G.get("Type"),
            "stock_name": G.get("Name"),
            "description": G.get("Description"),
            "cik": G.get("CIK"),
            "exchange": G.get("Exchange"),
            "currency": G.get("CurrencyCode"),
            "country": G.get("CountryName"),
            "sector": G.get("Sector"),
            "industry": G.get("Industry"),
            "address": G.get("Address"),
            "officialsite": G.get("WebURL"),
            "fiscalyearend": G.get("FiscalYearEnd"),
            "latestquarter": _to_date(H.get("MostRecentQuarter")),
            "market_cap": _to_bigint(H.get("MarketCapitalization")),
            "ebitda": _to_bigint(H.get("EBITDA")),
            "per": _to_float(H.get("PERatio")),
            "peg": _to_float(H.get("PEGRatio")),
            "bookvalue": _to_float(H.get("BookValue")),
            "dividendpershare": _to_float(H.get("DividendShare")),
            "dividendyield": _to_float(H.get("DividendYield")),
            "eps": _to_float(H.get("EarningsShare")),
            "revenuepersharettm": _to_float(H.get("RevenuePerShareTTM")),
            "profitmargin": _to_float(H.get("ProfitMargin")),
            "operatingmarginttm": _to_float(H.get("OperatingMarginTTM")),
            "returnonassetsttm": _to_float(H.get("ReturnOnAssetsTTM")),
            "returnonequityttm": _to_float(H.get("ReturnOnEquityTTM")),
            "revenuettm": _to_bigint(H.get("RevenueTTM")),
            "grossprofitttm": _to_bigint(H.get("GrossProfitTTM")),
            "dilutedepsttm": _to_float(H.get("DilutedEpsTTM")),
            "quarterlyearningsgrowthyoy": _to_float(
                H.get("QuarterlyEarningsGrowthYOY")
            ),
            "quarterlyrevenuegrowthyoy": _to_float(
                H.get("QuarterlyRevenueGrowthYOY")
            ),
            "analysttargetprice": _to_float(
                H.get("WallStreetTargetPrice")
            ) or _to_float(AR.get("TargetPrice")),
            "analystratingstrongbuy": _to_int(AR.get("StrongBuy")),
            "analystratingbuy": _to_int(AR.get("Buy")),
            "analystratinghold": _to_int(AR.get("Hold")),
            "analystratingsell": _to_int(AR.get("Sell")),
            "analystratingstrongsell": _to_int(AR.get("StrongSell")),
            "trailingpe": _to_float(V.get("TrailingPE")),
            "forwardpe": _to_float(V.get("ForwardPE")),
            "pricetosalesratiottm": _to_float(V.get("PriceSalesTTM")),
            "pricetobookratio": _to_float(V.get("PriceBookMRQ")),
            "evtorevenue": _to_float(V.get("EnterpriseValueRevenue")),
            "evtoebitda": _to_float(V.get("EnterpriseValueEbitda")),
            "beta": _to_float(T.get("Beta")),
            "week52high": _to_float(T.get("52WeekHigh")),
            "week52low": _to_float(T.get("52WeekLow")),
            "day50movingaverage": _to_float(T.get("50DayMA")),
            "day200movingaverage": _to_float(T.get("200DayMA")),
            "sharesoutstanding": _to_bigint(S.get("SharesOutstanding")),
            "sharesfloat": _to_bigint(S.get("SharesFloat")),
            "percentinsiders": _pct_clamp(S.get("PercentInsiders")),
            "percentinstitutions": _pct_clamp(S.get("PercentInstitutions")),
            "dividenddate": _to_date(SD.get("DividendDate")),
            "exdividenddate": _to_date(SD.get("ExDividendDate")),
            "is_active": True,
            "date": date.today(),
            "source": "api",
            "updated_at": datetime.now(),
        }

    def transform_income_statement(
        self, data: Dict[str, Any], symbol: str
    ) -> List[Dict[str, Any]]:
        qrt = (
            ((data.get("Financials") or {}).get("Income_Statement") or {})
            .get("quarterly") or {}
        )
        out = []
        now = datetime.now()
        for fd_str, q in qrt.items():
            fde = _to_date(q.get("date") or fd_str)
            if fde is None:
                continue
            net_income = _to_bigint(q.get("netIncome"))
            out.append({
                "symbol": symbol,
                "fiscal_date_ending": fde,
                "reported_currency": (q.get("currency_symbol") or "")[:10] or None,
                "gross_profit": _to_bigint(q.get("grossProfit")),
                "total_revenue": _to_bigint(q.get("totalRevenue")),
                "cost_of_revenue": _to_bigint(q.get("costOfRevenue")),
                "cost_of_goods_and_services_sold": _to_bigint(q.get("costOfRevenue")),
                "operating_income": _to_bigint(q.get("operatingIncome")),
                "selling_general_and_administrative": _to_bigint(
                    q.get("sellingGeneralAdministrative")
                ),
                "research_and_development": _to_bigint(q.get("researchDevelopment")),
                "operating_expenses": _to_bigint(q.get("totalOperatingExpenses")),
                "investment_income_net": None,
                "net_interest_income": _to_bigint(q.get("netInterestIncome")),
                "interest_income": _to_bigint(q.get("interestIncome")),
                "interest_expense": _to_bigint(q.get("interestExpense")),
                "non_interest_income": None,
                "other_non_operating_income": _to_bigint(
                    q.get("nonOperatingIncomeNetOther")
                ),
                "depreciation": _to_bigint(q.get("depreciationAndAmortization")),
                "depreciation_and_amortization": _to_bigint(
                    q.get("depreciationAndAmortization")
                ),
                "income_before_tax": _to_bigint(q.get("incomeBeforeTax")),
                "income_tax_expense": _to_bigint(
                    q.get("incomeTaxExpense") or q.get("taxProvision")
                ),
                "interest_and_debt_expense": _to_bigint(q.get("interestExpense")),
                "net_income_from_continuing_operations": _to_bigint(
                    q.get("netIncomeFromContinuingOps")
                ),
                "comprehensive_income_net_of_tax": net_income,
                "ebit": _to_bigint(q.get("ebit")),
                "ebitda": _to_bigint(q.get("ebitda")),
                "net_income": net_income,
                "created_at": now,
                "filing_date": _to_date(q.get("filing_date")),
            })
        return out

    def transform_balance_sheet(
        self, data: Dict[str, Any], symbol: str
    ) -> List[Dict[str, Any]]:
        qrt = (
            ((data.get("Financials") or {}).get("Balance_Sheet") or {})
            .get("quarterly") or {}
        )
        out = []
        now = datetime.now()
        for fd_str, q in qrt.items():
            fde = _to_date(q.get("date") or fd_str)
            if fde is None:
                continue
            intangible = _to_bigint(q.get("intangibleAssets"))
            goodwill = _to_bigint(q.get("goodWill"))
            intangible_excl = None
            if intangible is not None:
                intangible_excl = intangible - (goodwill or 0)
            out.append({
                "symbol": symbol,
                "fiscal_date_ending": fde,
                "reported_currency": (q.get("currency_symbol") or "")[:10] or None,
                "total_assets": _to_bigint(q.get("totalAssets")),
                "total_current_assets": _to_bigint(q.get("totalCurrentAssets")),
                "cash_and_cash_equivalents_at_carrying_value": _to_bigint(
                    q.get("cash")
                ),
                "cash_and_short_term_investments": _to_bigint(
                    q.get("cashAndShortTermInvestments")
                ),
                "inventory": _to_bigint(q.get("inventory")),
                "current_net_receivables": _to_bigint(q.get("netReceivables")),
                "total_non_current_assets": _to_bigint(q.get("nonCurrentAssetsTotal")),
                "property_plant_equipment": _to_bigint(
                    q.get("propertyPlantAndEquipmentNet")
                    or q.get("propertyPlantEquipment")
                ),
                "accumulated_depreciation_amortization_ppe": _to_bigint(
                    q.get("accumulatedDepreciation")
                ),
                "intangible_assets": intangible,
                "intangible_assets_excluding_goodwill": intangible_excl,
                "goodwill": goodwill,
                "investments": _to_bigint(q.get("longTermInvestments")),
                "long_term_investments": _to_bigint(q.get("longTermInvestments")),
                "short_term_investments": _to_bigint(q.get("shortTermInvestments")),
                "other_current_assets": _to_bigint(q.get("otherCurrentAssets")),
                "other_non_current_assets": _to_bigint(
                    q.get("otherAssets") or q.get("nonCurrrentAssetsOther")
                ),
                "total_liabilities": _to_bigint(q.get("totalLiab")),
                "total_current_liabilities": _to_bigint(
                    q.get("totalCurrentLiabilities")
                ),
                "current_accounts_payable": _to_bigint(q.get("accountsPayable")),
                "deferred_revenue": _to_bigint(
                    q.get("currentDeferredRevenue") or q.get("deferredLongTermLiab")
                ),
                "current_debt": _to_bigint(q.get("shortTermDebt")),
                "short_term_debt": _to_bigint(q.get("shortTermDebt")),
                "total_non_current_liabilities": _to_bigint(
                    q.get("nonCurrentLiabilitiesTotal")
                ),
                "capital_lease_obligations": _to_bigint(
                    q.get("capitalLeaseObligations")
                ),
                "long_term_debt": _to_bigint(q.get("longTermDebt")),
                "current_long_term_debt": _to_bigint(q.get("shortLongTermDebt")),
                "long_term_debt_noncurrent": _to_bigint(q.get("longTermDebt")),
                "short_long_term_debt_total": _to_bigint(
                    q.get("shortLongTermDebtTotal")
                ),
                "other_current_liabilities": _to_bigint(q.get("otherCurrentLiab")),
                "other_non_current_liabilities": _to_bigint(
                    q.get("otherLiab") or q.get("nonCurrentLiabilitiesOther")
                ),
                "total_shareholder_equity": _to_bigint(
                    q.get("totalStockholderEquity")
                ),
                "treasury_stock": _to_bigint(q.get("treasuryStock")),
                "retained_earnings": _to_bigint(q.get("retainedEarnings")),
                "common_stock": _to_bigint(q.get("commonStock")),
                "common_stock_shares_outstanding": _to_bigint(
                    q.get("commonStockSharesOutstanding")
                ),
                "created_at": now,
                "filing_date": _to_date(q.get("filing_date")),
            })
        return out

    def transform_cash_flow(
        self, data: Dict[str, Any], symbol: str
    ) -> List[Dict[str, Any]]:
        qrt = (
            ((data.get("Financials") or {}).get("Cash_Flow") or {})
            .get("quarterly") or {}
        )
        out = []
        now = datetime.now()
        for fd_str, q in qrt.items():
            fde = _to_date(q.get("date") or fd_str)
            if fde is None:
                continue
            net_income = _to_bigint(q.get("netIncome"))
            dividends_paid = _to_bigint(q.get("dividendsPaid"))
            out.append({
                "symbol": symbol,
                "fiscal_date_ending": fde,
                "reported_currency": (q.get("currency_symbol") or "")[:10] or None,
                "operating_cashflow": _to_bigint(
                    q.get("totalCashFromOperatingActivities")
                ),
                "payments_for_operating_activities": None,
                "proceeds_from_operating_activities": None,
                "change_in_operating_liabilities": _to_bigint(
                    q.get("changeToLiabilities")
                ),
                "change_in_operating_assets": _to_bigint(
                    q.get("changeToOperatingActivities")
                ),
                "depreciation_depletion_and_amortization": _to_bigint(
                    q.get("depreciation")
                ),
                "capital_expenditures": _to_bigint(q.get("capitalExpenditures")),
                "change_in_receivables": _to_bigint(
                    q.get("changeToAccountReceivables")
                ),
                "change_in_inventory": _to_bigint(q.get("changeToInventory")),
                "profit_loss": net_income,
                "cashflow_from_investment": _to_bigint(
                    q.get("totalCashflowsFromInvestingActivities")
                ),
                "cashflow_from_financing": _to_bigint(
                    q.get("totalCashFromFinancingActivities")
                ),
                "proceeds_from_repayments_of_short_term_debt": _to_bigint(
                    q.get("netBorrowings")
                ),
                "payments_for_repurchase_of_common_stock": _to_bigint(
                    q.get("salePurchaseOfStock")
                ),
                "payments_for_repurchase_of_equity": None,
                "payments_for_repurchase_of_preferred_stock": None,
                "dividend_payout": dividends_paid,
                "dividend_payout_common_stock": dividends_paid,
                "dividend_payout_preferred_stock": None,
                "proceeds_from_issuance_of_common_stock": _to_bigint(
                    q.get("issuanceOfCapitalStock")
                ),
                "proceeds_from_issuance_of_long_term_debt_and_capital_securities": None,
                "proceeds_from_issuance_of_preferred_stock": None,
                "proceeds_from_repurchase_of_equity": None,
                "proceeds_from_sale_of_treasury_stock": None,
                "change_in_cash_and_cash_equivalents": _to_bigint(
                    q.get("changeInCash")
                ),
                "change_in_exchange_rate": _to_bigint(q.get("exchangeRateChanges")),
                "net_income": net_income,
                "created_at": now,
                "filing_date": _to_date(q.get("filing_date")),
            })
        return out

    # ------------------------------------------------------------------
    # save (asyncpg INSERT)
    # ------------------------------------------------------------------

    async def save_stock_basic(self, conn, row: Dict[str, Any]) -> None:
        fields = list(row.keys())
        placeholders = ", ".join(f"${i + 1}" for i in range(len(fields)))
        updates = ", ".join(
            f"{f} = EXCLUDED.{f}"
            for f in fields
            if f not in ("symbol", "date", "source")
        )
        sql = f"""
            INSERT INTO us_stock_basic ({', '.join(fields)})
            VALUES ({placeholders})
            ON CONFLICT (symbol, date, source) DO UPDATE SET {updates}
        """
        await conn.execute(sql, *[row[f] for f in fields])

    async def save_income_statement(self, conn, rows: List[Dict[str, Any]]) -> int:
        return await self._save_financial(
            conn, "us_income_statement", rows, _INCOME_COLS
        )

    async def save_balance_sheet(self, conn, rows: List[Dict[str, Any]]) -> int:
        return await self._save_financial(
            conn, "us_balance_sheet", rows, _BALANCE_COLS
        )

    async def save_cash_flow(self, conn, rows: List[Dict[str, Any]]) -> int:
        return await self._save_financial(
            conn, "us_cash_flow", rows, _CASHFLOW_COLS
        )

    async def _save_financial(
        self, conn, table: str, rows: List[Dict[str, Any]], col_list: List[str]
    ) -> int:
        """공통 — table 별 col_list 사용. available_at 은 EODHD filing_date
        > us_earnings_history.reported_date > fiscal+45d 순으로 fallback.

        성능: asyncpg `executemany` 로 prepared statement 재사용
        (per-row execute 대비 종목당 ~500 row × 3 테이블 = 1500 statement
        plan 비용 회피, AAPL 163 quarter 기준 ~30s → ~1s)."""
        if not rows:
            return 0
        upd_cols = [c for c in col_list if c not in ("symbol", "fiscal_date_ending")]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in upd_cols) + \
                        ", available_at = EXCLUDED.available_at"
        n_cols = len(col_list)
        col_ph = ", ".join(f"${i + 1}" for i in range(n_cols))
        sql = f"""
            INSERT INTO {table} ({', '.join(col_list)}, available_at)
            VALUES ({col_ph}, COALESCE(
                ${n_cols + 1}::date,
                (SELECT reported_date FROM us_earnings_history
                 WHERE symbol = $1::varchar AND fiscal_date_ending = $2),
                ($2::date + INTERVAL '45 days')::date
            ))
            ON CONFLICT (symbol, fiscal_date_ending) DO UPDATE SET {update_clause}
        """
        args_list = [
            tuple([r.get(c) for c in col_list] + [r.get("filing_date")])
            for r in rows
        ]
        # 실패 시 절대 삼키지 않음 — 예외를 그대로 raise 해서 호출부의
        # conn.transaction() 이 깨끗하게 롤백하고, _process_symbol → run_collection
        # 이 그 종목을 (가짜 "ok" 가 아니라) 실제 error 로 집계하게 한다.
        # (과거 per-row fallback 은 executemany 실패로 이미 abort 된 트랜잭션에서
        #  무의미하게 재시도하다 모든 실패를 삼켜, stock_basic 까지 롤백된 종목을
        #  "ok" 로 카운트하는 silent data-loss 를 만들었다.)
        await conn.executemany(sql, args_list)
        return len(rows)

    # ------------------------------------------------------------------
    # process_symbol
    # ------------------------------------------------------------------

    async def _process_symbol(self, symbol: str) -> Dict[str, Any]:
        async with self.semaphore:
            data = await self._fetch_one(symbol)
            await asyncio.sleep(self.call_interval)
            if not data:
                await self._mark_no_data(symbol)
                return {"symbol": symbol, "status": "no_data"}

            sb = self.transform_stock_basic(data, symbol)
            inc = self.transform_income_statement(data, symbol)
            bal = self.transform_balance_sheet(data, symbol)
            cf = self.transform_cash_flow(data, symbol)

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await self.save_stock_basic(conn, sb)
                    inc_n = await self.save_income_statement(conn, inc)
                    bal_n = await self.save_balance_sheet(conn, bal)
                    cf_n = await self.save_cash_flow(conn, cf)

            if self.collection_logger is not None:
                try:
                    self.collection_logger.mark_collected(
                        "us_eodhd_fundamentals",
                        symbol,
                        [date.today().isoformat()],
                        records_count=1 + inc_n + bal_n + cf_n,
                    )
                    self.collection_logger.save_log()
                except Exception:
                    pass

            return {
                "symbol": symbol,
                "status": "ok",
                "inc": inc_n,
                "bal": bal_n,
                "cf": cf_n,
            }

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run_collection(self) -> Dict[str, Any]:
        symbols = await self.get_active_symbols()
        skip = await self.get_no_data_skip()
        targets = [s for s in symbols if s not in skip]
        logger.info(
            f"[EODHD] active={len(symbols)} skip(no_data 7d)={len(skip)} "
            f"targets={len(targets)}"
        )

        ok = err = no_data = 0
        rate_limited = False
        tasks = [self._process_symbol(s) for s in targets]
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                res = await coro
            except RateLimitError as e:
                logger.warning(f"[EODHD] rate limit reached: {e}")
                rate_limited = True
                break
            except Exception as e:
                logger.error(f"[EODHD] process error: {e}")
                err += 1
                continue
            results.append(res)
            status = res.get("status")
            if status == "ok":
                ok += 1
            elif status == "no_data":
                no_data += 1
            if (i + 1) % 200 == 0:
                logger.info(
                    f"[EODHD] progress {i + 1}/{len(targets)} "
                    f"ok={ok} no_data={no_data} err={err} "
                    f"used={self.api_requests_used}/{self.daily_rate_limit}"
                )
            if self.deadline and datetime.now() >= self.deadline:
                logger.warning("[EODHD] deadline reached, stopping")
                break

        return {
            "status": "rate_limited" if rate_limited else "completed",
            "targets": len(targets),
            "ok": ok,
            "no_data": no_data,
            "errors": err,
            "api_requests_used": self.api_requests_used,
            "daily_rate_limit": self.daily_rate_limit,
        }


class EODHDDividendsCollector:
    """EODHD `/api/div/{sym}.US` → us_dividends 전체 배당 이력 적재.

    fundamentals 와 별개 엔드포인트(1 unit/call, fundamentals 10 unit 대비 저렴).
    응답: [{date(=ex-date), declarationDate, recordDate, paymentDate, value, ...}]
    전체 이력 제공(2017 이전까지). #7b stock_basic_compute 가 이 테이블을 as-of
    merge 해 시점별 exdividenddate/dividenddate 를 채운다.
    """

    DIV_URL = "https://eodhd.com/api/div/{sym}.US"

    def __init__(
        self,
        api_token: str,
        database_url: str,
        *,
        call_interval: float = 0.05,
        max_concurrent: int = 6,
        from_date: str = "2015-01-01",
        target_symbols: Optional[List[str]] = None,
        deadline: Optional[datetime] = None,
    ):
        self.api_token = api_token
        self.database_url = database_url
        self.call_interval = call_interval
        self.max_concurrent = max_concurrent
        self.from_date = from_date
        self.target_symbols = target_symbols
        self.deadline = deadline
        self.pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.api_requests_used = 0
        self.daily_rate_limit = 100_000

    async def init_pool(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=USER_AGENT_HEADERS)

    async def close_pool(self):
        if self.session:
            await self.session.close()
            self.session = None
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def get_active_symbols(self) -> List[str]:
        if self.target_symbols:
            return list(self.target_symbols)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT symbol FROM us_stock_basic "
                "WHERE is_active = true ORDER BY symbol"
            )
        return [r["symbol"] for r in rows]

    async def _fetch_one(self, symbol: str, max_retries: int = 4):
        url = self.DIV_URL.format(sym=symbol)
        params = {"api_token": self.api_token, "fmt": "json", "from": self.from_date}
        for attempt in range(max_retries):
            try:
                async with self.session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    used = r.headers.get("x-apirequests-used")
                    lim = r.headers.get("x-apirequests-limit")
                    if used:
                        self.api_requests_used = int(used)
                    if lim:
                        self.daily_rate_limit = int(lim)
                    if r.status == 429:
                        await asyncio.sleep(min(5 * (2 ** attempt), 60))
                        continue
                    if r.status == 402:
                        raise RateLimitError(
                            f"budget {self.api_requests_used}/{self.daily_rate_limit}"
                        )
                    if r.status == 404:
                        return []
                    if r.status != 200:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(5 * (2 ** attempt))
                            continue
                        return None
                    data = await r.json()
                    return data if isinstance(data, list) else []
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                return None
            except RateLimitError:
                raise
            except Exception as e:
                logger.error(f"[EODHD div] {symbol} fetch error: {e}")
                return None
        return None

    def transform(self, rows, symbol: str) -> List[tuple]:
        out = []
        for d in rows or []:
            ex = _to_date(d.get("date"))
            if ex is None:
                continue
            out.append((
                symbol, ex,
                _to_date(d.get("declarationDate")),
                _to_date(d.get("recordDate")),
                _to_date(d.get("paymentDate")),
                _to_float(d.get("value")),
            ))
        return out

    async def _process_symbol(self, symbol: str) -> Dict[str, Any]:
        async with self.semaphore:
            rows = await self._fetch_one(symbol)
            await asyncio.sleep(self.call_interval)
            if rows is None:
                return {"symbol": symbol, "status": "error", "n": 0}
            recs = self.transform(rows, symbol)
            if not recs:
                return {"symbol": symbol, "status": "no_data", "n": 0}
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        """INSERT INTO us_dividends
                             (symbol, ex_dividend_date, declaration_date,
                              record_date, payment_date, amount, created_at)
                           VALUES ($1,$2,$3,$4,$5,$6, NOW())
                           ON CONFLICT (symbol, ex_dividend_date) DO UPDATE SET
                             declaration_date = EXCLUDED.declaration_date,
                             record_date = EXCLUDED.record_date,
                             payment_date = EXCLUDED.payment_date,
                             amount = EXCLUDED.amount""",
                        recs,
                    )
            return {"symbol": symbol, "status": "ok", "n": len(recs)}

    async def run_collection(self) -> Dict[str, Any]:
        symbols = await self.get_active_symbols()
        logger.info(f"[EODHD div] targets={len(symbols)}")
        ok = err = no_data = total = 0
        rate_limited = False
        tasks = [self._process_symbol(s) for s in symbols]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                res = await coro
            except RateLimitError as e:
                logger.warning(f"[EODHD div] rate limit: {e}")
                rate_limited = True
                break
            except Exception as e:
                logger.error(f"[EODHD div] process error: {e}")
                err += 1
                continue
            st = res.get("status")
            if st == "ok":
                ok += 1
                total += res["n"]
            elif st == "no_data":
                no_data += 1
            else:
                err += 1
            if (i + 1) % 500 == 0:
                logger.info(
                    f"[EODHD div] {i + 1}/{len(symbols)} ok={ok} "
                    f"no_data={no_data} err={err} rows={total}"
                )
            if self.deadline and datetime.now() >= self.deadline:
                logger.warning("[EODHD div] deadline reached")
                break
        return {
            "status": "rate_limited" if rate_limited else "completed",
            "symbols": len(symbols), "ok": ok, "no_data": no_data,
            "errors": err, "rows": total,
            "api_requests_used": self.api_requests_used,
        }


# ---------------------------------------------------------------------------
# 컬럼 순서 정의 (PK 가 항상 앞 2개)
# ---------------------------------------------------------------------------

_INCOME_COLS = [
    "symbol", "fiscal_date_ending", "reported_currency", "gross_profit",
    "total_revenue", "cost_of_revenue", "cost_of_goods_and_services_sold",
    "operating_income", "selling_general_and_administrative",
    "research_and_development", "operating_expenses", "investment_income_net",
    "net_interest_income", "interest_income", "interest_expense",
    "non_interest_income", "other_non_operating_income", "depreciation",
    "depreciation_and_amortization", "income_before_tax", "income_tax_expense",
    "interest_and_debt_expense", "net_income_from_continuing_operations",
    "comprehensive_income_net_of_tax", "ebit", "ebitda", "net_income",
    "created_at",
]

_BALANCE_COLS = [
    "symbol", "fiscal_date_ending", "reported_currency", "total_assets",
    "total_current_assets", "cash_and_cash_equivalents_at_carrying_value",
    "cash_and_short_term_investments", "inventory", "current_net_receivables",
    "total_non_current_assets", "property_plant_equipment",
    "accumulated_depreciation_amortization_ppe", "intangible_assets",
    "intangible_assets_excluding_goodwill", "goodwill", "investments",
    "long_term_investments", "short_term_investments", "other_current_assets",
    "other_non_current_assets", "total_liabilities", "total_current_liabilities",
    "current_accounts_payable", "deferred_revenue", "current_debt",
    "short_term_debt", "total_non_current_liabilities",
    "capital_lease_obligations", "long_term_debt", "current_long_term_debt",
    "long_term_debt_noncurrent", "short_long_term_debt_total",
    "other_current_liabilities", "other_non_current_liabilities",
    "total_shareholder_equity", "treasury_stock", "retained_earnings",
    "common_stock", "common_stock_shares_outstanding", "created_at",
]

_CASHFLOW_COLS = [
    "symbol", "fiscal_date_ending", "reported_currency", "operating_cashflow",
    "payments_for_operating_activities", "proceeds_from_operating_activities",
    "change_in_operating_liabilities", "change_in_operating_assets",
    "depreciation_depletion_and_amortization", "capital_expenditures",
    "change_in_receivables", "change_in_inventory", "profit_loss",
    "cashflow_from_investment", "cashflow_from_financing",
    "proceeds_from_repayments_of_short_term_debt",
    "payments_for_repurchase_of_common_stock",
    "payments_for_repurchase_of_equity",
    "payments_for_repurchase_of_preferred_stock", "dividend_payout",
    "dividend_payout_common_stock", "dividend_payout_preferred_stock",
    "proceeds_from_issuance_of_common_stock",
    "proceeds_from_issuance_of_long_term_debt_and_capital_securities",
    "proceeds_from_issuance_of_preferred_stock",
    "proceeds_from_repurchase_of_equity",
    "proceeds_from_sale_of_treasury_stock",
    "change_in_cash_and_cash_equivalents", "change_in_exchange_rate",
    "net_income", "created_at",
]
