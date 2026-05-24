"""
Search tools for market regime analysis
Provides news and web search capabilities for sentiment analysis
"""
import aiohttp
import asyncio
from typing import Optional
from kr.config import settings


# =============================================================================
# Naver News Search (Korean market sentiment)
# =============================================================================

async def search_naver_news(query: str, display: int = 5) -> list[dict]:
    """
    Search Naver News API for Korean market news

    Args:
        query: Search query (Korean)
        display: Number of results (max 100)

    Returns:
        List of news items with title, description, link, pubDate
    """
    if not settings.NAVER_CLIENT_ID or not settings.NAVER_CLIENT_SECRET:
        return []

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display,
        "sort": "date"  # Latest first
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    items = data.get("items", [])
                    # Clean up HTML tags from title and description
                    for item in items:
                        item["title"] = _strip_html(item.get("title", ""))
                        item["description"] = _strip_html(item.get("description", ""))
                    return items
                else:
                    return []
    except Exception:
        return []


# =============================================================================
# Serper Web Search (Global news, English search)
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
        "num": num
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
# Google Custom Search (Backup)
# =============================================================================

async def search_google(query: str, num: int = 5) -> list[dict]:
    """
    Search using Google Custom Search API

    Args:
        query: Search query
        num: Number of results (max 10 per request)

    Returns:
        List of search results with title, snippet, link
    """
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_SEARCH_ENGINE_ID:
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": settings.GOOGLE_API_KEY,
        "cx": settings.GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": min(num, 10)
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    items = data.get("items", [])
                    return [
                        {
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", "")
                        }
                        for item in items
                    ]
                else:
                    return []
    except Exception:
        return []


# =============================================================================
# Market Regime Search (Integrated)
# =============================================================================

# Search queries for market regime analysis
MARKET_REGIME_QUERIES = {
    # Korean market sentiment (Naver)
    "korea_market": [
        "코스피 전망",
        "증시 외국인 매매",
    ],
    # Global macro (Serper - English)
    "global_macro": [
        "Fed interest rate policy outlook",
        "US stock market forecast",
    ],
    # Foreign view on Korea (Serper - English)
    "korea_foreign_view": [
        "Korea stock market outlook",
        "KOSPI foreign investors",
    ],
}


async def search_market_news() -> dict:
    """
    Search for market regime related news from multiple sources

    Returns:
        Dictionary with categorized search results:
        - korea_market: Korean market news (Naver)
        - global_macro: Global macro news (Serper)
        - korea_foreign_view: Foreign view on Korea (Serper)
    """
    results = {
        "korea_market": [],
        "global_macro": [],
        "korea_foreign_view": []
    }

    # Create all search tasks
    tasks = []

    # Korean market news (Naver)
    for query in MARKET_REGIME_QUERIES["korea_market"]:
        tasks.append(("korea_market", search_naver_news(query, display=3)))

    # Global macro news (Serper)
    for query in MARKET_REGIME_QUERIES["global_macro"]:
        tasks.append(("global_macro", search_serper(query, num=3, search_type="news")))

    # Foreign view on Korea (Serper)
    for query in MARKET_REGIME_QUERIES["korea_foreign_view"]:
        tasks.append(("korea_foreign_view", search_serper(query, num=3, search_type="news")))

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


def summarize_search_results(results: dict) -> str:
    """
    Summarize search results into a text format for LLM consumption

    Args:
        results: Output from search_market_news()

    Returns:
        Formatted string summary of news
    """
    lines = []

    # Korean market news
    if results.get("korea_market"):
        lines.append("## Korean Market News (Naver)")
        for item in results["korea_market"][:5]:
            title = item.get("title", "")
            desc = item.get("description", "")
            if title:
                lines.append(f"- {title}")
                if desc:
                    lines.append(f"  {desc[:150]}...")
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

    # Foreign view on Korea
    if results.get("korea_foreign_view"):
        lines.append("## Foreign View on Korea Market")
        for item in results["korea_foreign_view"][:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title:
                lines.append(f"- {title}")
                if snippet:
                    lines.append(f"  {snippet[:150]}...")
        lines.append("")

    return "\n".join(lines) if lines else "No news data available."


# =============================================================================
# Helper Functions
# =============================================================================

def _strip_html(text: str) -> str:
    """Remove HTML tags from text"""
    import re
    clean = re.sub(r'<[^>]+>', '', text)
    clean = clean.replace("&quot;", '"')
    clean = clean.replace("&amp;", '&')
    clean = clean.replace("&lt;", '<')
    clean = clean.replace("&gt;", '>')
    clean = clean.replace("&apos;", "'")
    return clean.strip()
