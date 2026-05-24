"""
Research data collector module for US Stock Research Agent
Collects qualitative data from multiple sources

Data Sources:
- Tier 1: SEC Filings (search), Analyst Reports (DB + search)
- Tier 2: News (Serper), Insider Transactions (DB)
"""
import asyncio
from typing import Optional
from datetime import date, timedelta

from us.db import queries
from us.tools.search_tools import (
    search_serper,
    search_stock_news,
    search_sector_news,
    search_sec_filings,
    search_analyst_reports
)


# =============================================================================
# Constants
# =============================================================================

# Collection periods (in days)
COLLECTION_PERIODS = {
    "sec_filings": 90,
    "analyst_reports": 90,
    "insider_transactions": 90,
    "news": 30
}

# Market cap thresholds (in USD)
MARKET_CAP_THRESHOLDS = {
    "large": 10_000_000_000,   # $10B+ = Large cap
    "mid": 2_000_000_000,       # $2B+ = Mid cap
    # Below $2B = Small cap
}

# News collection strategy by stock size
NEWS_STRATEGY = {
    "large": {"days": 7, "max_articles": 30},
    "mid": {"days": 14, "max_articles": 25},
    "small": {"days": 30, "max_articles": 20},
}


# =============================================================================
# Stock Size Classification
# =============================================================================

def determine_stock_size(market_cap: float) -> str:
    """
    Determine stock size based on market capitalization.

    Args:
        market_cap: Market capitalization in USD

    Returns:
        "large", "mid", or "small"
    """
    if not market_cap or market_cap <= 0:
        return "mid"  # Default to mid if no market cap data

    if market_cap >= MARKET_CAP_THRESHOLDS["large"]:
        return "large"
    elif market_cap >= MARKET_CAP_THRESHOLDS["mid"]:
        return "mid"
    return "small"


def get_news_strategy(market_cap: float) -> dict:
    """
    Get news collection strategy based on market capitalization.

    Args:
        market_cap: Market capitalization in USD

    Returns:
        Strategy dict with days, max_articles
    """
    size = determine_stock_size(market_cap)
    return NEWS_STRATEGY[size]


# =============================================================================
# SEC Filings Collection
# =============================================================================

async def collect_sec_filings(stock_name: str, symbol: str) -> list[dict]:
    """
    Collect SEC filings for a stock via search.

    Args:
        stock_name: Company name
        symbol: Stock ticker

    Returns:
        List of SEC filing items
    """
    results = await search_sec_filings(stock_name, symbol)

    filings = []
    for item in results:
        filings.append({
            "type": "sec_filing",
            "source": "SEC",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
            "date": None,  # Date parsing from search results is unreliable
        })

    return filings


# =============================================================================
# Analyst Reports Collection
# =============================================================================

async def collect_analyst_reports(
    symbol: str,
    stock_name: str,
    days: int = 90
) -> list[dict]:
    """
    Collect analyst reports from search.

    Args:
        symbol: Stock ticker
        stock_name: Company name
        days: Number of days to collect

    Returns:
        List of analyst reports
    """
    reports = []

    # Search for analyst reports
    search_results = await search_analyst_reports(stock_name, symbol)

    for item in search_results:
        reports.append({
            "type": "analyst_report",
            "source": "Search",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
            "date": item.get("date"),
            "firm": None,  # Cannot extract from search
            "target_price": None,
            "rating": None,
        })

    return reports


# =============================================================================
# Insider Transactions Collection
# =============================================================================

async def collect_insider_transactions(symbol: str, days: int = 90) -> list[dict]:
    """
    Collect insider transactions from database.

    Args:
        symbol: Stock ticker
        days: Number of days to collect

    Returns:
        List of insider transactions
    """
    try:
        transactions = await queries.get_insider_transactions(symbol, days=days)
        return transactions if transactions else []
    except Exception:
        return []


# =============================================================================
# News Collection
# =============================================================================

async def collect_stock_news_data(
    stock_name: str,
    symbol: str,
    market_cap: float
) -> list[dict]:
    """
    Collect stock-specific news.

    Args:
        stock_name: Company name
        symbol: Stock ticker
        market_cap: Market cap for strategy selection

    Returns:
        List of news items
    """
    strategy = get_news_strategy(market_cap)
    news = await search_stock_news(stock_name, symbol, num=strategy["max_articles"])
    return news if news else []


async def collect_sector_news_data(sector: str, market_cap: float) -> list[dict]:
    """
    Collect sector-related news.

    Args:
        sector: Sector name
        market_cap: Market cap for strategy selection

    Returns:
        List of sector news items
    """
    if not sector:
        return []

    strategy = get_news_strategy(market_cap)
    news = await search_sector_news(sector, num=min(10, strategy["max_articles"]))

    # Mark as sector news
    for item in news:
        item["news_type"] = "sector"

    return news


# =============================================================================
# Integrated Collection Function
# =============================================================================

async def collect_research_data(
    symbol: str,
    stock_name: str,
    sector: str,
    market_cap: float
) -> dict:
    """
    Collect all research data for US Stock Research Agent.

    Collects:
    - Analyst reports (search)
    - SEC filings (search)
    - Insider transactions (DB)
    - Stock news (search)
    - Sector news (search)

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        stock_name: Company name (e.g., "Apple Inc")
        sector: Sector from us_stock_basic.sector
        market_cap: Market capitalization in USD

    Returns:
        Dictionary containing all collected data:
        - analyst_reports: List of analyst reports
        - sec_filings: List of SEC filing items
        - insider_transactions: List of insider transactions
        - stock_news: List of stock-specific news
        - sector_news: List of sector-related news
        - metadata: Collection metadata
    """
    stock_size = determine_stock_size(market_cap)
    news_strategy = get_news_strategy(market_cap)

    # Collect all data in parallel
    analyst_task = collect_analyst_reports(
        symbol, stock_name,
        days=COLLECTION_PERIODS["analyst_reports"]
    )
    sec_task = collect_sec_filings(stock_name, symbol)
    insider_task = collect_insider_transactions(
        symbol,
        days=COLLECTION_PERIODS["insider_transactions"]
    )
    stock_news_task = collect_stock_news_data(stock_name, symbol, market_cap)
    sector_news_task = collect_sector_news_data(sector, market_cap)

    results = await asyncio.gather(
        analyst_task,
        sec_task,
        insider_task,
        stock_news_task,
        sector_news_task,
        return_exceptions=True
    )

    # Handle results (convert exceptions to empty lists)
    analyst_reports = results[0] if isinstance(results[0], list) else []
    sec_filings = results[1] if isinstance(results[1], list) else []
    insider_transactions = results[2] if isinstance(results[2], list) else []
    stock_news = results[3] if isinstance(results[3], list) else []
    sector_news = results[4] if isinstance(results[4], list) else []

    return {
        "analyst_reports": analyst_reports,
        "sec_filings": sec_filings,
        "insider_transactions": insider_transactions,
        "stock_news": stock_news,
        "sector_news": sector_news,
        "metadata": {
            "symbol": symbol,
            "stock_name": stock_name,
            "sector": sector,
            "market_cap": market_cap,
            "stock_size": stock_size,
            "news_strategy": news_strategy,
            "collection_date": date.today().isoformat(),
            "collection_periods": COLLECTION_PERIODS,
            "data_counts": {
                "analyst_reports": len(analyst_reports),
                "sec_filings": len(sec_filings),
                "insider_transactions": len(insider_transactions),
                "stock_news": len(stock_news),
                "sector_news": len(sector_news),
            }
        }
    }
