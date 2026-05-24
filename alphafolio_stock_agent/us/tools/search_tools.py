"""
Search tools for US market regime analysis
Provides news and web search capabilities for sentiment analysis
Uses Serper API only (no Naver for US market)
"""
import aiohttp
import asyncio
from typing import Optional
from us.config import settings


# =============================================================================
# Serper Web Search (Primary search tool for US)
# =============================================================================

async def search_serper(query: str, num: int = 5, search_type: str = "search") -> list[dict]:
    """
    Search using Serper API (Google search results)

    Args:
        query: Search query
        num: Number of results
        search_type: "search" for web, "news" for news

    Returns:
        List of search results with title, snippet, link
    """
    if not settings.SERPER_API_KEY:
        return []

    url = f"https://google.serper.dev/{search_type}"
    headers = {
        "X-API-KEY": settings.SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num,
        "gl": "us",  # US region
        "hl": "en"   # English
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if search_type == "news":
                        return data.get("news", [])
                    else:
                        return data.get("organic", [])
                else:
                    return []
    except Exception:
        return []


# =============================================================================
# Market Regime Search (US-focused)
# =============================================================================

# Search queries for US market regime analysis
MARKET_REGIME_QUERIES = {
    # US market sentiment
    "us_market": [
        "S&P 500 market outlook",
        "NASDAQ stock market forecast",
        "US stock market sentiment",
    ],
    # Global macro affecting US
    "global_macro": [
        "Fed interest rate policy outlook",
        "US Treasury yields forecast",
        "Global economic outlook 2025",
    ],
    # Fed policy specific
    "fed_policy": [
        "Federal Reserve monetary policy",
        "Fed rate decision outlook",
        "FOMC meeting expectations",
    ],
}


async def search_market_news() -> dict:
    """
    Search for US market regime related news from Serper

    Returns:
        Dictionary with categorized search results:
        - us_market: US market news
        - global_macro: Global macro news
        - fed_policy: Fed policy news
    """
    results = {
        "us_market": [],
        "global_macro": [],
        "fed_policy": []
    }

    # Create all search tasks
    tasks = []

    # US market news
    for query in MARKET_REGIME_QUERIES["us_market"]:
        tasks.append(("us_market", search_serper(query, num=3, search_type="news")))

    # Global macro news
    for query in MARKET_REGIME_QUERIES["global_macro"]:
        tasks.append(("global_macro", search_serper(query, num=3, search_type="news")))

    # Fed policy news
    for query in MARKET_REGIME_QUERIES["fed_policy"]:
        tasks.append(("fed_policy", search_serper(query, num=3, search_type="news")))

    # Execute all searches in parallel
    search_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

    # Organize results by category
    for i, (category, _) in enumerate(tasks):
        result = search_results[i]
        if isinstance(result, list):
            results[category].extend(result)

    # Remove duplicates by link
    for category in results:
        seen_links = set()
        unique_items = []
        for item in results[category]:
            link = item.get("link") or item.get("originalUrl", "")
            if link and link not in seen_links:
                seen_links.add(link)
                unique_items.append(item)
        results[category] = unique_items

    return results


def format_news_for_prompt(news_results: dict) -> tuple[str, str, str]:
    """
    Format search results into prompt-friendly text.

    Args:
        news_results: Output from search_market_news()

    Returns:
        Tuple of (us_news, global_news, fed_news) formatted strings
    """
    # US market news
    us_lines = []
    for item in news_results.get("us_market", [])[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if title:
            us_lines.append(f"- {title}")
            if snippet:
                us_lines.append(f"  {snippet[:200]}")
    us_news = "\n".join(us_lines) if us_lines else "No relevant news"

    # Global macro news
    global_lines = []
    for item in news_results.get("global_macro", [])[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if title:
            global_lines.append(f"- {title}")
            if snippet:
                global_lines.append(f"  {snippet[:200]}")
    global_news = "\n".join(global_lines) if global_lines else "No relevant news"

    # Fed policy news
    fed_lines = []
    for item in news_results.get("fed_policy", [])[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if title:
            fed_lines.append(f"- {title}")
            if snippet:
                fed_lines.append(f"  {snippet[:200]}")
    fed_news = "\n".join(fed_lines) if fed_lines else "No relevant news"

    return us_news, global_news, fed_news


# =============================================================================
# Stock Research Search
# =============================================================================

async def search_stock_news(stock_name: str, symbol: str, num: int = 10) -> list[dict]:
    """
    Search for stock-specific news.

    Args:
        stock_name: Company name
        symbol: Stock ticker
        num: Number of results

    Returns:
        List of news items
    """
    query = f"{symbol} {stock_name} stock news"
    return await search_serper(query, num=num, search_type="news")


async def search_sector_news(sector: str, num: int = 5) -> list[dict]:
    """
    Search for sector-related news.

    Args:
        sector: Sector name
        num: Number of results

    Returns:
        List of news items
    """
    if not sector:
        return []

    query = f"{sector} sector stock market news"
    return await search_serper(query, num=num, search_type="news")


async def search_sec_filings(stock_name: str, symbol: str) -> list[dict]:
    """
    Search for SEC filings related to a stock.

    Args:
        stock_name: Company name
        symbol: Stock ticker

    Returns:
        List of SEC filing results
    """
    query = f"site:sec.gov {symbol} {stock_name} 10-K 10-Q 8-K"
    return await search_serper(query, num=10, search_type="search")


async def search_analyst_reports(stock_name: str, symbol: str) -> list[dict]:
    """
    Search for analyst reports and ratings.

    Args:
        stock_name: Company name
        symbol: Stock ticker

    Returns:
        List of analyst report results
    """
    query = f"{symbol} {stock_name} analyst rating target price"
    return await search_serper(query, num=10, search_type="news")


# =============================================================================
# Helper Functions
# =============================================================================

def summarize_search_results(results: dict) -> str:
    """
    Summarize search results into a text format for LLM consumption

    Args:
        results: Output from search_market_news()

    Returns:
        Formatted string summary of news
    """
    lines = []

    # US market news
    if results.get("us_market"):
        lines.append("## US Market News")
        for item in results["us_market"][:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title:
                lines.append(f"- {title}")
                if snippet:
                    lines.append(f"  {snippet[:150]}...")
        lines.append("")

    # Global macro news
    if results.get("global_macro"):
        lines.append("## Global Macro News")
        for item in results["global_macro"][:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title:
                lines.append(f"- {title}")
                if snippet:
                    lines.append(f"  {snippet[:150]}...")
        lines.append("")

    # Fed policy news
    if results.get("fed_policy"):
        lines.append("## Fed Policy News")
        for item in results["fed_policy"][:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title:
                lines.append(f"- {title}")
                if snippet:
                    lines.append(f"  {snippet[:150]}...")
        lines.append("")

    return "\n".join(lines) if lines else "No news data available."
