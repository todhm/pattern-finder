"""
PostgreSQL query functions module
Table-specific query functions with appropriate collection periods
"""
import json
from typing import Optional
from datetime import date, timedelta
from kr.db.connection import Database


# =============================================================================
# Stock Basic Information
# =============================================================================

async def get_stock_basic(symbol: str) -> Optional[dict]:
    """Get stock basic info from kr_stock_basic table"""
    query = """
        SELECT * FROM kr_stock_basic
        WHERE symbol = $1
    """
    return await Database.fetchrow(query, symbol)


async def get_stock_detail(symbol: str) -> Optional[dict]:
    """Get stock detail info from kr_stock_detail table"""
    query = """
        SELECT * FROM kr_stock_detail
        WHERE symbol = $1
    """
    return await Database.fetchrow(query, symbol)


# =============================================================================
# Quant Analysis Results
# =============================================================================

async def get_stock_grade(symbol: str, target_date: Optional[date] = None) -> Optional[dict]:
    """Get quant analysis result from kr_stock_grade table (latest 1 record)"""
    if target_date:
        query = """
            SELECT * FROM kr_stock_grade
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 1
        """
        return await Database.fetchrow(query, symbol, target_date)
    else:
        query = """
            SELECT * FROM kr_stock_grade
            WHERE symbol = $1
            ORDER BY date DESC
            LIMIT 1
        """
        return await Database.fetchrow(query, symbol)


async def save_strategy(symbol: str, strategy: dict, target_date: date = None) -> bool:
    """Save strategy result to kr_stock_grade.strategy column (JSON)

    Args:
        symbol: Stock code
        strategy: Strategy result dict
        target_date: Analysis target date (saves to this date's record).
                     If None, updates the latest row for the symbol.
    """
    strategy_json = json.dumps(strategy, ensure_ascii=False)
    if target_date is None:
        # No date specified: update the latest row for this symbol
        query = """
            UPDATE kr_stock_grade
            SET strategy = $2::jsonb
            WHERE symbol = $1 AND date = (
                SELECT MAX(date) FROM kr_stock_grade WHERE symbol = $1
            )
        """
        await Database.execute(query, symbol, strategy_json)
    else:
        query = """
            UPDATE kr_stock_grade
            SET strategy = $2::jsonb
            WHERE symbol = $1 AND date = $3
        """
        await Database.execute(query, symbol, strategy_json, target_date)
    return True


# =============================================================================
# Technical Indicators (90 days)
# =============================================================================

async def get_stock_indicators(
    symbol: str,
    days: int = 90,
    end_date: date = None
) -> list[dict]:
    """
    Get technical indicators from kr_indicators table

    Args:
        symbol: Stock code
        days: Number of days to retrieve
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM kr_indicators
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


# =============================================================================
# Price Data (90 days)
# =============================================================================

async def get_intraday_total(
    symbol: str,
    days: int = 90,
    end_date: date = None
) -> list[dict]:
    """
    Get daily price data from kr_intraday_total table

    Args:
        symbol: Stock code
        days: Number of days to retrieve
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM kr_intraday_total
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


# =============================================================================
# Investor Trading (60 days)
# =============================================================================

async def get_individual_investor_daily_trading(
    symbol: str,
    days: int = 60,
    end_date: date = None
) -> list[dict]:
    """
    Get investor trading data from kr_individual_investor_daily_trading table

    Args:
        symbol: Stock code
        days: Number of days to retrieve
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM kr_individual_investor_daily_trading
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


# =============================================================================
# Foreign Ownership (60 days)
# =============================================================================

async def get_foreign_ownership(
    symbol: str,
    days: int = 60,
    end_date: date = None
) -> list[dict]:
    """
    Get foreign ownership data from kr_foreign_ownership table

    Args:
        symbol: Stock code
        days: Number of days to retrieve
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM kr_foreign_ownership
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


# =============================================================================
# Financial Position (2 years)
# =============================================================================

async def get_financial_position(symbol: str, years: int = 2) -> list[dict]:
    """Get financial statement data from kr_financial_position table"""
    cutoff_date = date.today() - timedelta(days=years * 365)
    query = """
        SELECT * FROM kr_financial_position
        WHERE symbol = $1 AND rcept_dt >= $2
        ORDER BY rcept_dt DESC, bsns_year DESC
    """
    return await Database.fetch(query, symbol, cutoff_date)


# =============================================================================
# Research Reports (6 months)
# =============================================================================

async def get_research_reports(symbol: str, days: int = 180) -> list[dict]:
    """Get securities research reports from kr_research_reports table"""
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT * FROM kr_research_reports
        WHERE symbol = $1 AND date >= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, symbol, cutoff_date)


# =============================================================================
# Block Trades (30 days)
# =============================================================================

async def get_blocktrades(
    symbol: str,
    days: int = 30,
    end_date: date = None
) -> list[dict]:
    """
    Get block trade data from kr_blocktrades table

    Args:
        symbol: Stock code
        days: Number of days to retrieve
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM kr_blocktrades
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


# =============================================================================
# Dividends (3 years)
# =============================================================================

async def get_dividends(symbol: str, years: int = 3) -> list[dict]:
    """Get dividend info from kr_dividends table"""
    cutoff_date = date.today() - timedelta(days=years * 365)
    query = """
        SELECT * FROM kr_dividends
        WHERE symbol = $1 AND stlm_dt >= $2
        ORDER BY stlm_dt DESC
    """
    return await Database.fetch(query, symbol, cutoff_date)


# =============================================================================
# Largest Shareholder (1 year)
# =============================================================================

async def get_largest_shareholder(symbol: str, years: int = 1) -> list[dict]:
    """Get largest shareholder info from kr_largest_shareholder table"""
    cutoff_date = date.today() - timedelta(days=years * 365)
    query = """
        SELECT * FROM kr_largest_shareholder
        WHERE symbol = $1 AND stlm_dt >= $2
        ORDER BY stlm_dt DESC
    """
    return await Database.fetch(query, symbol, cutoff_date)


# =============================================================================
# Market Index (60 days)
# =============================================================================

async def get_market_index(
    exchange: str = "KOSPI",
    days: int = 60,
    end_date: date = None
) -> list[dict]:
    """
    Get market index data from market_index table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM market_index
        WHERE exchange = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, exchange, end_date, days)


async def get_market_indices(exchanges: list[str] = None, days: int = 60) -> dict[str, list[dict]]:
    """Get multiple market indices at once"""
    if exchanges is None:
        exchanges = ["KOSPI", "KOSDAQ"]

    result = {}
    for exchange in exchanges:
        result[exchange] = await get_market_index(exchange, days)

    return result


# =============================================================================
# Economic Indicators (1 year)
# =============================================================================

async def get_economic_indicators(days: int = 365, end_date: date = None) -> list[dict]:
    """
    Get latest economic indicators from bok_economic_indicators table (latest per indicator)

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT DISTINCT ON (stat_name) *
        FROM bok_economic_indicators
        WHERE time_value <= $1
        ORDER BY stat_name, time_value DESC
    """
    return await Database.fetch(query, end_date)


async def get_economic_indicators_history(stat_name: str, days: int = 365) -> list[dict]:
    """Get time series data for a specific economic indicator"""
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT * FROM bok_economic_indicators
        WHERE stat_name LIKE $1 AND time_value >= $2
        ORDER BY time_value DESC
    """
    return await Database.fetch(query, f"%{stat_name}%", cutoff_date)


# =============================================================================
# Exchange Rate (90 days)
# =============================================================================

async def get_exchange_rate(
    currency: str = "원/미국달러",
    days: int = 90,
    end_date: date = None
) -> list[dict]:
    """
    Get exchange rate data from exchange_rate table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM exchange_rate
        WHERE item_name1 LIKE $1 AND time_value >= $2 AND time_value <= $3
        ORDER BY time_value DESC
    """
    return await Database.fetch(query, f"%{currency}%", cutoff_date, end_date)


# =============================================================================
# Executives
# =============================================================================

async def get_executives(symbol: str) -> list[dict]:
    """Get executive info from kr_executive table"""
    query = """
        SELECT * FROM kr_executive
        WHERE symbol = $1
        ORDER BY stlm_dt DESC
    """
    return await Database.fetch(query, symbol)


# =============================================================================
# Stock Acquisition/Disposal
# =============================================================================

async def get_stock_acquisition_disposal(symbol: str) -> list[dict]:
    """Get treasury stock info from kr_stockacquisitiondisposal table"""
    query = """
        SELECT * FROM kr_stockacquisitiondisposal
        WHERE symbol = $1
        ORDER BY stlm_dt DESC
        LIMIT 10
    """
    return await Database.fetch(query, symbol)


# =============================================================================
# US Economic Indicators (for Korean stock macro analysis)
# =============================================================================

async def get_us_fed_funds_rate(days: int = 365, end_date: date = None) -> list[dict]:
    """
    Get US federal funds rate from us_fed_funds_rate table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_fed_funds_rate
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_treasury_yield(days: int = 365, end_date: date = None) -> list[dict]:
    """
    Get US treasury yield from us_treasury_yield table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_treasury_yield
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_cpi(days: int = 365, end_date: date = None) -> list[dict]:
    """
    Get US CPI from us_cpi table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_cpi
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_unemployment_rate(days: int = 365, end_date: date = None) -> list[dict]:
    """
    Get US unemployment rate from us_unemployment_rate table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_unemployment_rate
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_gdp(days: int = 365, end_date: date = None) -> list[dict]:
    """
    Get US GDP from us_gdp table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_gdp
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_pmi(days: int = 365, end_date: date = None) -> list[dict]:
    """
    Get US PMI from us_pmi table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_pmi
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_vix(days: int = 90, end_date: date = None) -> list[dict]:
    """
    Get VIX index from us_vix table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_vix
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_dollar_index(days: int = 90, end_date: date = None) -> list[dict]:
    """
    Get US dollar index from us_dollar_index table

    Args:
        end_date: Only retrieve data on or before this date (default: today)
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_dollar_index
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


# =============================================================================
# Market Regime Data (for global market analysis)
# =============================================================================

async def get_us_market_regime(days: int = 30) -> list[dict]:
    """Get US market regime data from us_market_regime table"""
    query = """
        SELECT * FROM us_market_regime
        ORDER BY created_at DESC
        LIMIT $1
    """
    return await Database.fetch(query, days)


async def get_us_credit_spread(days: int = 90) -> list[dict]:
    """Get US credit spread from us_credit_spread table"""
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT * FROM us_credit_spread
        WHERE date >= $1
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date)


async def get_us_move_index(days: int = 90) -> list[dict]:
    """Get MOVE index (bond volatility) from us_move_index table"""
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT * FROM us_move_index
        WHERE date >= $1
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date)


async def get_us_etf_prices(symbols: list[str] = None, days: int = 30) -> dict[str, list[dict]]:
    """
    Get ETF prices from us_daily_etf table

    Default symbols:
    - GLD: Gold ETF (safe haven)
    - TLT: 20+ Year Treasury Bond ETF
    - SHY: 1-3 Year Treasury Bond ETF
    - HYG: High Yield Corporate Bond ETF
    - AGG: Aggregate Bond ETF
    """
    if symbols is None:
        symbols = ["GLD", "TLT", "SHY", "HYG", "AGG"]

    cutoff_date = date.today() - timedelta(days=days)
    result = {}

    for symbol in symbols:
        query = """
            SELECT symbol, date, open, high, low, close, volume
            FROM us_daily_etf
            WHERE symbol = $1 AND date >= $2
            ORDER BY date DESC
        """
        result[symbol] = await Database.fetch(query, symbol, cutoff_date)

    return result


# =============================================================================
# Korean Benchmark Index (kr_benchmark_index)
# =============================================================================

async def get_vkospi(days: int = 90) -> list[dict]:
    """
    Get VKOSPI (Korea VIX) from kr_benchmark_index table

    VKOSPI index_name: '코스피 200 변동성지수'
    Typical value range: 15-45 (similar to US VIX)
    """
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT date, index_name, close, change_amount, change_rate, open, high, low
        FROM kr_benchmark_index
        WHERE index_category = 'volatility'
          AND index_name = '코스피 200 변동성지수'
          AND date >= $1
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date)


async def get_benchmark_index(
    index_category: str = None,
    index_name: str = None,
    days: int = 90
) -> list[dict]:
    """
    Get benchmark index data from kr_benchmark_index table

    Args:
        index_category: 'kospi', 'kosdaq', 'volatility', 'futures', 'gold'
        index_name: Specific index name to filter (partial match)
        days: Number of days to retrieve

    Returns:
        List of index data records
    """
    cutoff_date = date.today() - timedelta(days=days)

    if index_category and index_name:
        query = """
            SELECT date, index_category, index_name, close, change_amount, change_rate,
                   open, high, low, volume, trading_value, market_cap
            FROM kr_benchmark_index
            WHERE index_category = $1
              AND index_name LIKE $2
              AND date >= $3
            ORDER BY date DESC
        """
        return await Database.fetch(query, index_category, f"%{index_name}%", cutoff_date)
    elif index_category:
        query = """
            SELECT date, index_category, index_name, close, change_amount, change_rate,
                   open, high, low, volume, trading_value, market_cap
            FROM kr_benchmark_index
            WHERE index_category = $1 AND date >= $2
            ORDER BY date DESC
        """
        return await Database.fetch(query, index_category, cutoff_date)
    elif index_name:
        query = """
            SELECT date, index_category, index_name, close, change_amount, change_rate,
                   open, high, low, volume, trading_value, market_cap
            FROM kr_benchmark_index
            WHERE index_name LIKE $1 AND date >= $2
            ORDER BY date DESC
        """
        return await Database.fetch(query, f"%{index_name}%", cutoff_date)
    else:
        query = """
            SELECT date, index_category, index_name, close, change_amount, change_rate,
                   open, high, low, volume, trading_value, market_cap
            FROM kr_benchmark_index
            WHERE date >= $1
            ORDER BY date DESC
        """
        return await Database.fetch(query, cutoff_date)


async def get_kospi200_index(days: int = 90) -> list[dict]:
    """Get KOSPI 200 index data"""
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT date, index_name, close, change_amount, change_rate, open, high, low,
               volume, trading_value, market_cap
        FROM kr_benchmark_index
        WHERE index_category = 'kospi'
          AND index_name LIKE '%코스피 200%'
          AND date >= $1
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date)


async def get_gold_index(days: int = 90) -> list[dict]:
    """Get KRX Gold Spot index data"""
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT date, index_name, close, change_amount, change_rate, open, high, low
        FROM kr_benchmark_index
        WHERE index_category = 'gold'
          AND date >= $1
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date)
