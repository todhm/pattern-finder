"""
Data collector module
Collects raw data from PostgreSQL tables with appropriate collection periods
No LLM involved - pure Python data retrieval
"""
import io
import csv
from typing import Optional
from datetime import date

from kr.db.queries import (
    get_stock_basic,
    get_stock_detail,
    get_stock_grade,
    get_stock_indicators,
    get_intraday_total,
    get_individual_investor_daily_trading,
    get_foreign_ownership,
    get_financial_position,
    get_research_reports,
    get_blocktrades,
    get_dividends,
    get_largest_shareholder,
    get_market_index,
    get_economic_indicators,
    get_exchange_rate,
    # US economic indicators (for macro analysis)
    get_us_fed_funds_rate,
    get_us_treasury_yield,
    get_us_cpi,
    get_us_unemployment_rate,
    get_us_gdp,
    get_us_pmi,
    get_us_vix,
    get_us_dollar_index,
)


# Collection periods per table (in days)
COLLECTION_PERIODS = {
    # Korean stock data
    "kr_stock_grade": 1,            # Latest 1 record
    "kr_indicators": 90,            # 90 days (3 months)
    "kr_intraday_total": 90,        # 90 days
    "kr_individual_investor_daily_trading": 60,  # 60 days (2 months)
    "kr_foreign_ownership": 60,     # 60 days
    "kr_financial_position": 730,   # 2 years
    "kr_research_reports": 180,     # 6 months
    "kr_blocktrades": 30,           # 30 days
    "kr_dividends": 1095,           # 3 years
    "kr_largest_shareholder": 365,  # 1 year
    # Korean market/economic data
    "market_index": 60,             # 60 days
    "bok_economic_indicators": 365, # 1 year
    "exchange_rate": 90,            # 90 days
    # US economic indicators (for macro analysis)
    "us_fed_funds_rate": 365,       # 1 year - US interest rate policy
    "us_treasury_yield": 365,       # 1 year - US bond yield trend
    "us_cpi": 365,                  # 1 year - US inflation trend
    "us_unemployment_rate": 365,    # 1 year - US employment situation
    "us_gdp": 365,                  # 1 year - US economic growth
    "us_pmi": 365,                  # 1 year - US manufacturing sentiment
    "us_vix": 90,                   # 90 days - Market volatility/fear index
    "us_dollar_index": 90,          # 90 days - Dollar strength/weakness
}


async def collect_stock_data(symbol: str, target_date: Optional[date] = None) -> dict:
    """
    Collect all relevant data for a stock from PostgreSQL tables.

    Args:
        symbol: Stock code (e.g., "005930")
        target_date: Analysis reference date (default: today)

    Returns:
        dict containing all collected data:
        - stock_basic: Basic stock info
        - stock_detail: Detailed stock info
        - stock_grade: Quant analysis result
        - indicators: Technical indicators (time series)
        - prices: Daily price data (time series)
        - investor_trading: Investor trading data (time series)
        - foreign_ownership: Foreign ownership data (time series)
        - financials: Financial statements
        - research_reports: Securities research reports
        - blocktrades: Block trade data
        - dividends: Dividend info
        - largest_shareholder: Largest shareholder info
        - market_index_kospi: KOSPI index (time series)
        - market_index_kosdaq: KOSDAQ index (time series)
        - economic_indicators: BOK economic indicators
        - exchange_rate: Exchange rate data (time series)
        - us_fed_funds_rate: US federal funds rate (time series)
        - us_treasury_yield: US treasury yield (time series)
        - us_cpi: US consumer price index (time series)
        - us_unemployment_rate: US unemployment rate (time series)
        - us_gdp: US GDP (time series)
        - us_pmi: US PMI (time series)
        - us_vix: VIX index (time series)
        - us_dollar_index: US dollar index (time series)
    """
    data = {
        # Single record (JSON format for agent)
        "stock_basic": await get_stock_basic(symbol),
        "stock_detail": await get_stock_detail(symbol),
        "stock_grade": await get_stock_grade(symbol, target_date),

        # Time series data (list format, will be converted to CSV for token efficiency)
        "indicators": await get_stock_indicators(
            symbol,
            days=COLLECTION_PERIODS["kr_indicators"],
            end_date=target_date
        ),
        "prices": await get_intraday_total(
            symbol,
            days=COLLECTION_PERIODS["kr_intraday_total"],
            end_date=target_date
        ),
        "investor_trading": await get_individual_investor_daily_trading(
            symbol,
            days=COLLECTION_PERIODS["kr_individual_investor_daily_trading"],
            end_date=target_date
        ),
        "foreign_ownership": await get_foreign_ownership(
            symbol,
            days=COLLECTION_PERIODS["kr_foreign_ownership"],
            end_date=target_date
        ),
        "financials": await get_financial_position(symbol),
        "research_reports": await get_research_reports(
            symbol,
            days=COLLECTION_PERIODS["kr_research_reports"]
        ),
        "blocktrades": await get_blocktrades(
            symbol,
            days=COLLECTION_PERIODS["kr_blocktrades"],
            end_date=target_date
        ),
        "dividends": await get_dividends(symbol),
        "largest_shareholder": await get_largest_shareholder(symbol),

        # Market data
        "market_index_kospi": await get_market_index(
            exchange="KOSPI",
            days=COLLECTION_PERIODS["market_index"],
            end_date=target_date
        ),
        "market_index_kosdaq": await get_market_index(
            exchange="KOSDAQ",
            days=COLLECTION_PERIODS["market_index"],
            end_date=target_date
        ),
        "economic_indicators": await get_economic_indicators(end_date=target_date),
        "exchange_rate": await get_exchange_rate(
            currency="원/미국달러",
            days=COLLECTION_PERIODS["exchange_rate"],
            end_date=target_date
        ),

        # US economic indicators (for macro analysis)
        "us_fed_funds_rate": await get_us_fed_funds_rate(
            days=COLLECTION_PERIODS["us_fed_funds_rate"],
            end_date=target_date
        ),
        "us_treasury_yield": await get_us_treasury_yield(
            days=COLLECTION_PERIODS["us_treasury_yield"],
            end_date=target_date
        ),
        "us_cpi": await get_us_cpi(
            days=COLLECTION_PERIODS["us_cpi"],
            end_date=target_date
        ),
        "us_unemployment_rate": await get_us_unemployment_rate(
            days=COLLECTION_PERIODS["us_unemployment_rate"],
            end_date=target_date
        ),
        "us_gdp": await get_us_gdp(
            days=COLLECTION_PERIODS["us_gdp"],
            end_date=target_date
        ),
        "us_pmi": await get_us_pmi(
            days=COLLECTION_PERIODS["us_pmi"],
            end_date=target_date
        ),
        "us_vix": await get_us_vix(
            days=COLLECTION_PERIODS["us_vix"],
            end_date=target_date
        ),
        "us_dollar_index": await get_us_dollar_index(
            days=COLLECTION_PERIODS["us_dollar_index"],
            end_date=target_date
        ),
    }

    return data


def to_csv(data: list[dict], columns: list[str] = None) -> str:
    """
    Convert list of dicts to CSV string for token efficiency.

    Token efficiency comparison (e.g., kr_indicators 10 days):
    - JSON: ~2,500 tokens
    - CSV: ~800 tokens (about 3x more efficient)

    Args:
        data: List of dictionaries to convert
        columns: Specific columns to include (default: all columns)

    Returns:
        CSV formatted string
    """
    if not data:
        return ""

    if columns is None:
        columns = list(data[0].keys())

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)

    return output.getvalue()


def get_collection_period(table_name: str) -> int:
    """
    Get the collection period for a specific table.

    Args:
        table_name: Name of the table

    Returns:
        Collection period in days
    """
    return COLLECTION_PERIODS.get(table_name, 30)
