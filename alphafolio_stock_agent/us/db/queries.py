"""
PostgreSQL query functions module for US stocks
Table-specific query functions with appropriate collection periods
"""
import json
from typing import Optional
from datetime import date, timedelta
from us.db.connection import Database


# =============================================================================
# Stock Basic Information
# =============================================================================

async def get_stock_basic(symbol: str) -> Optional[dict]:
    """
    Get stock basic info from us_stock_basic table.
    Includes: sector, industry, description, marketCap, 200-day MA, etc.
    """
    query = """
        SELECT * FROM us_stock_basic
        WHERE symbol = $1
    """
    return await Database.fetchrow(query, symbol)


# =============================================================================
# Quant Analysis Results
# =============================================================================

async def get_stock_grade(symbol: str, target_date: Optional[date] = None) -> Optional[dict]:
    """
    Get quant analysis result from us_stock_grade table (latest 1 record).
    Contains: final_score, final_grade, scenario probabilities, stop_loss_pct, take_profit_pct
    """
    if target_date:
        query = """
            SELECT * FROM us_stock_grade
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 1
        """
        return await Database.fetchrow(query, symbol, target_date)
    else:
        query = """
            SELECT * FROM us_stock_grade
            WHERE symbol = $1
            ORDER BY date DESC
            LIMIT 1
        """
        return await Database.fetchrow(query, symbol)


async def save_strategy(symbol: str, strategy: dict, target_date: date = None) -> bool:
    """
    Save strategy result to us_stock_grade.strategy column (JSON).

    Args:
        symbol: Stock ticker
        strategy: Strategy result dict
        target_date: Analysis target date (saves to this date's record).
                     If None, updates the latest row for the symbol.
    """
    strategy_json = json.dumps(strategy, ensure_ascii=False)
    if target_date is None:
        # No date specified: update the latest row for this symbol
        query = """
            UPDATE us_stock_grade
            SET strategy = $2::jsonb
            WHERE symbol = $1 AND date = (
                SELECT MAX(date) FROM us_stock_grade WHERE symbol = $1
            )
        """
        await Database.execute(query, symbol, strategy_json)
    else:
        query = """
            UPDATE us_stock_grade
            SET strategy = $2::jsonb
            WHERE symbol = $1 AND date = $3
        """
        await Database.execute(query, symbol, strategy_json, target_date)
    return True
    return True


# =============================================================================
# Daily Price Data (90 days)
# =============================================================================

async def get_daily_prices(
    symbol: str,
    days: int = 90,
    end_date: date = None
) -> list[dict]:
    """
    Get daily price data from us_daily table.
    Includes: open, high, low, close, volume, PE, PBR, etc.
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM us_daily
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


async def get_weekly_prices(
    symbol: str,
    weeks: int = 52,
    end_date: date = None
) -> list[dict]:
    """
    Get weekly price data from us_weekly table.
    52 weeks = 1 year of weekly data.
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM us_weekly
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, weeks)


# =============================================================================
# Technical Indicators (90 days)
# =============================================================================

async def get_stock_indicators(
    symbol: str,
    days: int = 90,
    end_date: date = None
) -> list[dict]:
    """
    Get technical indicators from us_indicators table.
    Includes: RSI, MACD, Bollinger Bands, ADX, Stochastic, etc.
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM us_indicators
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


# =============================================================================
# Options Data (252 days for IV percentile)
# =============================================================================

async def get_option_daily_summary(
    symbol: str,
    days: int = 252,
    end_date: date = None
) -> list[dict]:
    """
    Get options daily summary from us_option_daily_summary table.
    252 days needed for IV percentile calculation (1 trading year).

    Includes: implied_volatility, put_volume, call_volume, put_call_ratio,
              net_gex, gamma_flip_level, etc.
    """
    if end_date is None:
        end_date = date.today()
    query = """
        SELECT * FROM us_option_daily_summary
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT $3
    """
    return await Database.fetch(query, symbol, end_date, days)


async def get_option_chain(
    symbol: str,
    days: int = 7,
    end_date: date = None
) -> list[dict]:
    """
    Get detailed option chain data from us_option table.
    Limited to recent data due to large table size.

    Includes: strike, expiration, put/call, delta, gamma, theta, vega, IV, etc.
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_option
        WHERE underlying = $1 AND date >= $2 AND date <= $3
        ORDER BY date DESC, expiration, strike
    """
    return await Database.fetch(query, symbol, cutoff_date, end_date)


# =============================================================================
# Financial Statements (2 years / 8 quarters)
# =============================================================================

async def get_income_statement(symbol: str, quarters: int = 8) -> list[dict]:
    """Get income statement data from us_income_statement table."""
    query = """
        SELECT * FROM us_income_statement
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT $2
    """
    return await Database.fetch(query, symbol, quarters)


async def get_balance_sheet(symbol: str, quarters: int = 8) -> list[dict]:
    """Get balance sheet data from us_balance_sheet table."""
    query = """
        SELECT * FROM us_balance_sheet
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT $2
    """
    return await Database.fetch(query, symbol, quarters)


async def get_cash_flow(symbol: str, quarters: int = 8) -> list[dict]:
    """Get cash flow statement data from us_cash_flow table."""
    query = """
        SELECT * FROM us_cash_flow
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT $2
    """
    return await Database.fetch(query, symbol, quarters)


async def get_earnings_estimates(symbol: str) -> list[dict]:
    """Get earnings estimates from us_earnings_estimates table."""
    query = """
        SELECT * FROM us_earnings_estimates
        WHERE symbol = $1
        ORDER BY estimate_date DESC
        LIMIT 4
    """
    return await Database.fetch(query, symbol)


# =============================================================================
# Dividends (365 days)
# =============================================================================

async def get_dividends(symbol: str, days: int = 365) -> list[dict]:
    """Get dividend info from us_dividends table."""
    cutoff_date = date.today() - timedelta(days=days)
    query = """
        SELECT * FROM us_dividends
        WHERE symbol = $1 AND ex_dividend_date >= $2
        ORDER BY ex_dividend_date DESC
    """
    return await Database.fetch(query, symbol, cutoff_date)


# =============================================================================
# News with Sentiment (30 days for stock, 7 days for market)
# =============================================================================

async def get_stock_news(
    symbol: str,
    days: int = 30,
    end_date: date = None
) -> list[dict]:
    """
    Get stock-specific news from us_news table.
    Includes: title, summary, source, overall_sentiment_score, ticker_sentiment_score, etc.
    Note: symbol is stored in symbol_sentiment JSONB array as {"ticker": "AAPL", ...}
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_news
        WHERE symbol_sentiment @> ('[{"ticker": "' || $1 || '"}]')::jsonb
          AND time_published >= $2 AND time_published <= $3
        ORDER BY time_published DESC
        LIMIT 50
    """
    return await Database.fetch(query, symbol, cutoff_date, end_date)


async def get_market_news(
    days: int = 7,
    end_date: date = None
) -> list[dict]:
    """
    Get market-wide news from us_news table.
    Filter for news without specific ticker (market news).
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_news
        WHERE (symbol_sentiment IS NULL OR symbol_sentiment = '[]'::jsonb)
          AND time_published >= $1 AND time_published <= $2
        ORDER BY time_published DESC
        LIMIT 50
    """
    return await Database.fetch(query, cutoff_date, end_date)


# =============================================================================
# Insider Transactions (90 days)
# =============================================================================

async def get_insider_transactions(
    symbol: str,
    days: int = 90,
    end_date: date = None
) -> list[dict]:
    """
    Get insider trading data from us_insider_transactions table.
    Includes: executive, executive_title, shares, share_price, etc.
    """
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_insider_transactions
        WHERE symbol = $1 AND date >= $2 AND date <= $3
        ORDER BY date DESC
    """
    return await Database.fetch(query, symbol, cutoff_date, end_date)


# =============================================================================
# US Economic Indicators (365 days)
# =============================================================================

async def get_us_fed_funds_rate(days: int = 365, end_date: date = None) -> list[dict]:
    """Get US federal funds rate from us_fed_funds_rate table."""
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
    """Get US treasury yield from us_treasury_yield table."""
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
    """Get US CPI from us_cpi table."""
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
    """Get US unemployment rate from us_unemployment_rate table."""
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
    """Get US GDP from us_gdp table."""
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
    """Get US PMI from us_pmi table."""
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_pmi
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


# =============================================================================
# Market Indicators (90 days)
# =============================================================================

async def get_us_vix(days: int = 90, end_date: date = None) -> list[dict]:
    """Get VIX index from us_vix table."""
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_vix
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_move_index(days: int = 90, end_date: date = None) -> list[dict]:
    """Get MOVE index (bond volatility) from us_move_index table."""
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_move_index
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_dollar_index(days: int = 90, end_date: date = None) -> list[dict]:
    """Get US dollar index from us_dollar_index table."""
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_dollar_index
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_credit_spread(days: int = 90, end_date: date = None) -> list[dict]:
    """Get US credit spread from us_credit_spread table."""
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_credit_spread
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


async def get_us_fed_rrp(days: int = 90, end_date: date = None) -> list[dict]:
    """Get Fed reverse repo from us_fed_rrp table."""
    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    query = """
        SELECT * FROM us_fed_rrp
        WHERE date >= $1 AND date <= $2
        ORDER BY date DESC
    """
    return await Database.fetch(query, cutoff_date, end_date)


# =============================================================================
# Market Index (60 days)
# =============================================================================

async def get_market_index(
    exchange: str = "SP500",
    days: int = 60,
    end_date: date = None
) -> list[dict]:
    """
    Get market index data from market_index table.

    Args:
        exchange: "SP500" or "NASDAQ"
        days: Number of days to retrieve
        end_date: Only retrieve data on or before this date
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
    """Get multiple market indices at once."""
    if exchanges is None:
        exchanges = ["SP500", "NASDAQ"]

    result = {}
    for exchange in exchanges:
        result[exchange] = await get_market_index(exchange, days)

    return result


# =============================================================================
# Market Regime (Caching)
# =============================================================================

async def get_us_market_regime(days: int = 1) -> list[dict]:
    """
    Get US market regime data from us_market_regime table.
    Used for caching market regime analysis results.
    """
    query = """
        SELECT * FROM us_market_regime
        ORDER BY created_at DESC
        LIMIT $1
    """
    return await Database.fetch(query, days)


async def save_market_regime(regime_data: dict) -> bool:
    """Save market regime analysis result to us_market_regime table."""
    query = """
        INSERT INTO us_market_regime (
            regime, confidence, analysis_date, data
        ) VALUES ($1, $2, $3, $4::json)
    """
    await Database.execute(
        query,
        regime_data.get('regime'),
        regime_data.get('confidence'),
        regime_data.get('analysis_date'),
        json.dumps(regime_data, ensure_ascii=False)
    )
    return True


# =============================================================================
# Safe Haven ETFs (90 days)
# =============================================================================

async def get_etf_prices(
    symbols: list[str] = None,
    days: int = 90,
    end_date: date = None
) -> dict[str, list[dict]]:
    """
    Get ETF prices from us_daily_etf table.

    Default symbols for safe haven analysis:
    - GLD: Gold ETF
    - TLT: 20+ Year Treasury Bond ETF
    - SHY: 1-3 Year Treasury Bond ETF
    - HYG: High Yield Corporate Bond ETF
    - AGG: Aggregate Bond ETF
    """
    if symbols is None:
        symbols = ["GLD", "TLT", "SHY", "HYG", "AGG"]

    if end_date is None:
        end_date = date.today()
    cutoff_date = end_date - timedelta(days=days)
    result = {}

    for symbol in symbols:
        query = """
            SELECT symbol, date, open, high, low, close, volume
            FROM us_daily_etf
            WHERE symbol = $1 AND date >= $2 AND date <= $3
            ORDER BY date DESC
        """
        result[symbol] = await Database.fetch(query, symbol, cutoff_date, end_date)

    return result


# =============================================================================
# Sector Benchmarks (if available)
# =============================================================================

async def get_sector_benchmarks(sector: str = None) -> list[dict]:
    """
    Get sector benchmark data from us_sector_benchmarks table.

    Note: This table may be empty. Check data availability before using.
    """
    if sector:
        query = """
            SELECT * FROM us_sector_benchmarks
            WHERE sector = $1
            ORDER BY date DESC
            LIMIT 1
        """
        row = await Database.fetchrow(query, sector)
        return [row] if row else []
    else:
        query = """
            SELECT DISTINCT ON (sector) *
            FROM us_sector_benchmarks
            ORDER BY sector, date DESC
        """
        return await Database.fetch(query)


# =============================================================================
# Helper Functions
# =============================================================================

async def check_data_freshness(symbol: str) -> dict:
    """
    Check data freshness for a given symbol.
    Returns latest dates for key tables.
    """
    freshness = {}

    # Check us_stock_grade
    grade_query = """
        SELECT MAX(date) as latest_date FROM us_stock_grade WHERE symbol = $1
    """
    result = await Database.fetchrow(grade_query, symbol)
    freshness['us_stock_grade'] = str(result['latest_date']) if result and result['latest_date'] else None

    # Check us_daily
    daily_query = """
        SELECT MAX(date) as latest_date FROM us_daily WHERE symbol = $1
    """
    result = await Database.fetchrow(daily_query, symbol)
    freshness['us_daily'] = str(result['latest_date']) if result and result['latest_date'] else None

    # Check us_indicators
    indicators_query = """
        SELECT MAX(date) as latest_date FROM us_indicators WHERE symbol = $1
    """
    result = await Database.fetchrow(indicators_query, symbol)
    freshness['us_indicators'] = str(result['latest_date']) if result and result['latest_date'] else None

    # Check us_option_daily_summary
    options_query = """
        SELECT MAX(date) as latest_date FROM us_option_daily_summary WHERE symbol = $1
    """
    result = await Database.fetchrow(options_query, symbol)
    freshness['us_option_daily_summary'] = str(result['latest_date']) if result and result['latest_date'] else None

    return freshness


async def get_stock_name(symbol: str) -> Optional[str]:
    """Get company name for a given symbol."""
    query = """
        SELECT name FROM us_stock_basic WHERE symbol = $1
    """
    result = await Database.fetchrow(query, symbol)
    return result['name'] if result else None
