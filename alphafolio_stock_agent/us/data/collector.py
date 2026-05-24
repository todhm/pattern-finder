"""
Data collector module for US stocks
Collects raw data from PostgreSQL tables with appropriate collection periods
No LLM involved - pure Python data retrieval
"""
import io
import csv
from typing import Optional
from datetime import date

from us.db.queries import (
    # Stock data
    get_stock_basic,
    get_stock_grade,
    get_daily_prices,
    get_weekly_prices,
    get_stock_indicators,
    # Options data
    get_option_daily_summary,
    # Financial data
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_earnings_estimates,
    get_dividends,
    # News and insider
    get_stock_news,
    get_market_news,
    get_insider_transactions,
    # US economic indicators
    get_us_fed_funds_rate,
    get_us_treasury_yield,
    get_us_cpi,
    get_us_unemployment_rate,
    get_us_gdp,
    get_us_pmi,
    # Market indicators
    get_us_vix,
    get_us_move_index,
    get_us_dollar_index,
    get_us_credit_spread,
    get_us_fed_rrp,
    # Market index and ETFs
    get_market_index,
    get_etf_prices,
    get_us_market_regime,
    # Helpers
    get_stock_name,
)


# Collection periods per table (in days)
COLLECTION_PERIODS = {
    # Stock data
    "us_stock_grade": 1,              # Latest 1 record
    "us_stock_basic": 1,              # Latest record (static info)
    "us_daily": 90,                   # 90 days
    "us_weekly": 52,                  # 52 weeks (1 year)
    "us_indicators": 90,              # 90 days

    # Options data (252 days for IV percentile - 1 trading year)
    "us_option_daily_summary": 252,   # 252 trading days for IV percentile
    "us_option": 7,                   # 7 days (detailed chain, limited)

    # Financial data
    "us_income_statement": 8,         # 8 quarters (2 years)
    "us_balance_sheet": 8,            # 8 quarters
    "us_cash_flow": 8,                # 8 quarters
    "us_earnings_estimates": 4,       # Latest 4 estimates
    "us_dividends": 365,              # 1 year

    # News and insider
    "us_news_stock": 30,              # 30 days for stock news
    "us_news_market": 7,              # 7 days for market news
    "us_insider_transactions": 90,    # 90 days

    # US economic indicators
    "us_fed_funds_rate": 365,         # 1 year
    "us_treasury_yield": 365,         # 1 year
    "us_cpi": 365,                    # 1 year
    "us_unemployment_rate": 365,      # 1 year
    "us_gdp": 365,                    # 1 year
    "us_pmi": 365,                    # 1 year

    # Market indicators
    "us_vix": 90,                     # 90 days
    "us_move_index": 90,              # 90 days
    "us_dollar_index": 90,            # 90 days
    "us_credit_spread": 90,           # 90 days
    "us_fed_rrp": 90,                 # 90 days

    # Market index and ETFs
    "market_index": 60,               # 60 days
    "us_daily_etf": 90,               # 90 days
    "us_market_regime": 1,            # Latest 1 (for caching)
}


async def collect_stock_data(symbol: str, target_date: Optional[date] = None) -> dict:
    """
    Collect all relevant data for a US stock from PostgreSQL tables.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        target_date: Analysis reference date (default: today)

    Returns:
        dict containing all collected data:
        - stock_basic: Basic stock info (sector, industry, 200-day MA)
        - stock_grade: Quant analysis result (scores, scenarios)
        - daily_prices: Daily OHLCV data (time series)
        - weekly_prices: Weekly OHLCV data (time series)
        - indicators: Technical indicators (time series)
        - option_summary: Options daily summary (Put/Call, IV, GEX)
        - income_statement: Income statement (8 quarters)
        - balance_sheet: Balance sheet (8 quarters)
        - cash_flow: Cash flow statement (8 quarters)
        - earnings_estimates: Earnings estimates
        - dividends: Dividend info
        - stock_news: Stock-specific news with sentiment
        - market_news: Market-wide news
        - insider_transactions: Insider trading data
        - us_fed_funds_rate: Fed funds rate (time series)
        - us_treasury_yield: Treasury yield (time series)
        - us_cpi: CPI data (time series)
        - us_unemployment_rate: Unemployment rate (time series)
        - us_gdp: GDP data (time series)
        - us_pmi: PMI data (time series)
        - us_vix: VIX index (time series)
        - us_move_index: MOVE index (time series)
        - us_dollar_index: Dollar index (time series)
        - us_credit_spread: Credit spread (time series)
        - us_fed_rrp: Fed RRP (time series)
        - market_index_sp500: S&P 500 index (time series)
        - market_index_nasdaq: NASDAQ index (time series)
        - etf_prices: Safe haven ETF prices (GLD, TLT, etc.)
        - market_regime: Cached market regime analysis
    """
    data = {
        # =================================================================
        # Stock-specific data (Single record - JSON format)
        # =================================================================
        "stock_basic": await get_stock_basic(symbol),
        "stock_grade": await get_stock_grade(symbol, target_date),

        # =================================================================
        # Stock-specific time series data (list format -> CSV for efficiency)
        # =================================================================
        "daily_prices": await get_daily_prices(
            symbol,
            days=COLLECTION_PERIODS["us_daily"],
            end_date=target_date
        ),
        "weekly_prices": await get_weekly_prices(
            symbol,
            weeks=COLLECTION_PERIODS["us_weekly"],
            end_date=target_date
        ),
        "indicators": await get_stock_indicators(
            symbol,
            days=COLLECTION_PERIODS["us_indicators"],
            end_date=target_date
        ),

        # =================================================================
        # Options data (252 days for IV percentile calculation)
        # =================================================================
        "option_summary": await get_option_daily_summary(
            symbol,
            days=COLLECTION_PERIODS["us_option_daily_summary"],
            end_date=target_date
        ),

        # =================================================================
        # Financial statements (8 quarters = 2 years)
        # =================================================================
        "income_statement": await get_income_statement(
            symbol,
            quarters=COLLECTION_PERIODS["us_income_statement"]
        ),
        "balance_sheet": await get_balance_sheet(
            symbol,
            quarters=COLLECTION_PERIODS["us_balance_sheet"]
        ),
        "cash_flow": await get_cash_flow(
            symbol,
            quarters=COLLECTION_PERIODS["us_cash_flow"]
        ),
        "earnings_estimates": await get_earnings_estimates(symbol),
        "dividends": await get_dividends(
            symbol,
            days=COLLECTION_PERIODS["us_dividends"]
        ),

        # =================================================================
        # News and insider trading
        # =================================================================
        "stock_news": await get_stock_news(
            symbol,
            days=COLLECTION_PERIODS["us_news_stock"],
            end_date=target_date
        ),
        "market_news": await get_market_news(
            days=COLLECTION_PERIODS["us_news_market"],
            end_date=target_date
        ),
        "insider_transactions": await get_insider_transactions(
            symbol,
            days=COLLECTION_PERIODS["us_insider_transactions"],
            end_date=target_date
        ),

        # =================================================================
        # US economic indicators (365 days)
        # =================================================================
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

        # =================================================================
        # Market indicators (90 days)
        # =================================================================
        "us_vix": await get_us_vix(
            days=COLLECTION_PERIODS["us_vix"],
            end_date=target_date
        ),
        "us_move_index": await get_us_move_index(
            days=COLLECTION_PERIODS["us_move_index"],
            end_date=target_date
        ),
        "us_dollar_index": await get_us_dollar_index(
            days=COLLECTION_PERIODS["us_dollar_index"],
            end_date=target_date
        ),
        "us_credit_spread": await get_us_credit_spread(
            days=COLLECTION_PERIODS["us_credit_spread"],
            end_date=target_date
        ),
        "us_fed_rrp": await get_us_fed_rrp(
            days=COLLECTION_PERIODS["us_fed_rrp"],
            end_date=target_date
        ),

        # =================================================================
        # Market indices (60 days)
        # =================================================================
        "market_index_sp500": await get_market_index(
            exchange="SP500",
            days=COLLECTION_PERIODS["market_index"],
            end_date=target_date
        ),
        "market_index_nasdaq": await get_market_index(
            exchange="NASDAQ",
            days=COLLECTION_PERIODS["market_index"],
            end_date=target_date
        ),

        # =================================================================
        # Safe haven ETF prices (90 days)
        # =================================================================
        "etf_prices": await get_etf_prices(
            symbols=["GLD", "TLT", "SHY", "HYG", "AGG"],
            days=COLLECTION_PERIODS["us_daily_etf"],
            end_date=target_date
        ),

        # =================================================================
        # Market regime (cached)
        # =================================================================
        "market_regime": await get_us_market_regime(
            days=COLLECTION_PERIODS["us_market_regime"]
        ),
    }

    return data


def to_csv(data: list[dict], columns: list[str] = None) -> str:
    """
    Convert list of dicts to CSV string for token efficiency.

    Token efficiency comparison (e.g., us_indicators 10 days):
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
        Collection period in days (or quarters for financial statements)
    """
    return COLLECTION_PERIODS.get(table_name, 30)


async def get_stock_info(symbol: str) -> dict:
    """
    Get basic stock info for display purposes.

    Args:
        symbol: Stock ticker

    Returns:
        dict with symbol and name
    """
    name = await get_stock_name(symbol)
    return {
        "symbol": symbol,
        "name": name or symbol
    }
