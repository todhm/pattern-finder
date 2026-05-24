"""
Research data collector module for Stock Research Agent
Collects qualitative data from multiple sources (Tier 1-2 only)
No LLM involved - pure Python data retrieval

Data Sources:
- Tier 1: Disclosures (DART + KIND search), Research Reports (DB + Naver Finance)
- Tier 2: News (Naver API)
- Excluded: SNS, Community (Tier 3 - Phase 2)
"""
import asyncio
from typing import Optional
from datetime import date, timedelta

from kr.db.queries import get_research_reports
from kr.tools.search_tools import search_naver_news, search_serper


# =============================================================================
# Constants
# =============================================================================

# Collection periods (in days)
COLLECTION_PERIODS = {
    "disclosures": 90,       # Disclosure search period
    "research_reports": 90,  # Research report period
}

# Market cap thresholds (in KRW)
MARKET_CAP_THRESHOLDS = {
    "large": 10_000_000_000_000,   # 10 trillion+ = Large cap
    "medium": 1_000_000_000_000,   # 1 trillion+ = Mid cap
    # Below 1 trillion = Small cap
}

# News collection strategy by stock size
NEWS_STRATEGY = {
    "large": {"days": 7, "use_keyword_filter": True, "max_articles": 30},
    "medium": {"days": 14, "use_keyword_filter": False, "max_articles": 30},
    "small": {"days": 30, "use_keyword_filter": False, "max_articles": 30},
}

# Large cap keyword filters
LARGE_CAP_INCLUDE_KEYWORDS = [
    "실적", "전망", "목표가", "매수", "매도", "주가",
    "상승", "하락", "급등", "급락", "공시", "배당", "자사주"
]

LARGE_CAP_EXCLUDE_KEYWORDS = [
    "광고", "이벤트", "채용", "스포츠", "연예"
]


# =============================================================================
# Stock Size Classification
# =============================================================================

def determine_stock_size(market_cap: float) -> str:
    """
    Determine stock size based on market capitalization.

    Args:
        market_cap: Market capitalization in KRW

    Returns:
        "large", "medium", or "small"
    """
    if not market_cap or market_cap <= 0:
        return "small"  # Default to small if no market cap data

    if market_cap >= MARKET_CAP_THRESHOLDS["large"]:
        return "large"
    elif market_cap >= MARKET_CAP_THRESHOLDS["medium"]:
        return "medium"
    return "small"


def get_news_strategy(market_cap: float) -> dict:
    """
    Get news collection strategy based on market capitalization.

    Args:
        market_cap: Market capitalization in KRW

    Returns:
        Strategy dict with days, use_keyword_filter, max_articles
    """
    size = determine_stock_size(market_cap)
    return NEWS_STRATEGY[size]


# =============================================================================
# Disclosure Search (DART + KIND)
# =============================================================================

async def search_dart_disclosures(stock_name: str, days: int = 90) -> list[dict]:
    """
    Search DART (Financial Supervisory Service) for disclosures.

    Args:
        stock_name: Stock name to search
        days: Number of days to search (for date filtering in results)

    Returns:
        List of disclosure items from DART
    """
    query = f"site:dart.fss.or.kr {stock_name}"
    results = await search_serper(query, num=10, search_type="search")

    disclosures = []
    for item in results:
        disclosures.append({
            "type": "disclosure",
            "source": "DART",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
            "date": None,  # Date parsing from search results is unreliable
        })

    return disclosures


async def search_kind_disclosures(stock_name: str, days: int = 90) -> list[dict]:
    """
    Search KIND (Korea Exchange) for disclosures.

    Args:
        stock_name: Stock name to search
        days: Number of days to search (for date filtering in results)

    Returns:
        List of disclosure items from KIND
    """
    query = f"site:kind.krx.co.kr {stock_name}"
    results = await search_serper(query, num=10, search_type="search")

    disclosures = []
    for item in results:
        disclosures.append({
            "type": "disclosure",
            "source": "KIND",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
            "date": None,
        })

    return disclosures


async def collect_disclosures(stock_name: str, days: int = 90) -> list[dict]:
    """
    Collect disclosures from DART and KIND.

    Args:
        stock_name: Stock name to search
        days: Number of days to search

    Returns:
        Combined list of disclosures from both sources
    """
    # Search both sources in parallel
    dart_task = search_dart_disclosures(stock_name, days)
    kind_task = search_kind_disclosures(stock_name, days)

    dart_results, kind_results = await asyncio.gather(
        dart_task, kind_task, return_exceptions=True
    )

    disclosures = []

    if isinstance(dart_results, list):
        disclosures.extend(dart_results)

    if isinstance(kind_results, list):
        disclosures.extend(kind_results)

    # Remove duplicates by URL
    seen_urls = set()
    unique_disclosures = []
    for item in disclosures:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_disclosures.append(item)

    return unique_disclosures


# =============================================================================
# Research Reports (DB + Naver Finance)
# =============================================================================

async def search_naver_research_reports(stock_name: str) -> list[dict]:
    """
    Search Naver Finance for research reports.
    Source: https://finance.naver.com/research/company_list.naver

    Args:
        stock_name: Stock name to search

    Returns:
        List of research report items
    """
    query = f"site:finance.naver.com/research {stock_name} 리포트"
    results = await search_serper(query, num=10, search_type="search")

    reports = []
    for item in results:
        reports.append({
            "type": "research_report",
            "source": "Naver Finance",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
            "date": None,
            "securities_firm": None,  # Cannot extract from search
            "target_price": None,
            "investment_opinion": None,
        })

    return reports


async def collect_research_reports(
    symbol: str,
    stock_name: str,
    days: int = 90
) -> list[dict]:
    """
    Collect research reports from DB and Naver Finance.

    Priority:
    1. DB (kr_research_reports) - has structured data since 2025.11
    2. Naver Finance search - supplement if DB data insufficient

    Args:
        symbol: Stock symbol
        stock_name: Stock name
        days: Number of days to collect

    Returns:
        List of research reports
    """
    reports = []

    # 1. Get reports from DB
    db_reports = await get_research_reports(symbol, days=days)

    if db_reports:
        for report in db_reports:
            reports.append({
                "type": "research_report",
                "source": "DB",
                "title": report.get("title", ""),
                "snippet": report.get("summary", ""),
                "url": report.get("pdf_url", ""),
                "date": str(report.get("date", "")) if report.get("date") else None,
                "securities_firm": report.get("securities_firm", ""),
                "target_price": float(report.get("target_price", 0)) if report.get("target_price") else None,
                "investment_opinion": report.get("investment_opinion", ""),
            })

    # 2. If DB reports are insufficient (< 3), supplement with Naver search
    if len(reports) < 3:
        naver_reports = await search_naver_research_reports(stock_name)
        # Add only non-duplicate reports
        existing_titles = {r.get("title", "").lower() for r in reports}
        for report in naver_reports:
            title_lower = report.get("title", "").lower()
            if title_lower and title_lower not in existing_titles:
                reports.append(report)
                existing_titles.add(title_lower)

    return reports


# =============================================================================
# News Collection (Adaptive Strategy)
# =============================================================================

def filter_news_by_keywords(news: list[dict]) -> list[dict]:
    """
    Filter news for large-cap stocks using keyword rules.

    Include news that contains at least one include keyword
    and does not contain any exclude keywords.

    Args:
        news: List of news items

    Returns:
        Filtered list of news items
    """
    filtered = []

    for item in news:
        title = item.get("title", "").lower()
        description = item.get("description", "").lower()
        text = title + " " + description

        # Check exclude keywords first
        has_exclude = any(kw in text for kw in LARGE_CAP_EXCLUDE_KEYWORDS)
        if has_exclude:
            continue

        # Check include keywords
        has_include = any(kw in text for kw in LARGE_CAP_INCLUDE_KEYWORDS)
        if has_include:
            filtered.append(item)

    return filtered


async def collect_stock_news(stock_name: str, market_cap: float) -> list[dict]:
    """
    Collect stock-specific news with adaptive strategy based on market cap.

    Args:
        stock_name: Stock name to search
        market_cap: Market capitalization in KRW

    Returns:
        List of news items
    """
    strategy = get_news_strategy(market_cap)

    # For large caps, use multiple keyword-focused queries
    if strategy["use_keyword_filter"]:
        # Search with focused queries for large caps
        queries = [
            f"{stock_name} 실적",
            f"{stock_name} 전망",
            f"{stock_name} 주가",
        ]

        all_news = []
        tasks = [search_naver_news(q, display=10) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_news.extend(result)

        # Remove duplicates by link
        seen_links = set()
        unique_news = []
        for item in all_news:
            link = item.get("link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                unique_news.append(item)

        # Apply keyword filter
        filtered_news = filter_news_by_keywords(unique_news)

        # Limit to max articles
        return filtered_news[:strategy["max_articles"]]

    else:
        # For medium/small caps, simple search
        news = await search_naver_news(stock_name, display=strategy["max_articles"])
        return news


async def collect_sector_news(sector: str, market_cap: float) -> list[dict]:
    """
    Collect sector-related news separately.

    Args:
        sector: Sector to search
        market_cap: Market capitalization (for determining period)

    Returns:
        List of sector news items
    """
    if not sector:
        return []

    strategy = get_news_strategy(market_cap)

    # Search sector news
    news = await search_naver_news(sector, display=strategy["max_articles"])

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
    Collect all research data for Stock Research Agent.

    Collects:
    - Research reports (DB + Naver Finance)
    - Disclosures (DART + KIND search)
    - Stock news (adaptive strategy based on market cap)
    - Sector news (separate collection)

    Args:
        symbol: Stock symbol (e.g., "005930")
        stock_name: Stock name (e.g., "Samsung Electronics")
        sector: Sector from kr_stock_detail.theme
        market_cap: Market capitalization in KRW

    Returns:
        Dictionary containing all collected data:
        - research_reports: List of research reports
        - disclosures: List of disclosure items
        - stock_news: List of stock-specific news
        - sector_news: List of sector-related news
        - metadata: Collection metadata
    """
    stock_size = determine_stock_size(market_cap)
    news_strategy = get_news_strategy(market_cap)

    # Collect all data in parallel
    reports_task = collect_research_reports(
        symbol, stock_name,
        days=COLLECTION_PERIODS["research_reports"]
    )
    disclosures_task = collect_disclosures(
        stock_name,
        days=COLLECTION_PERIODS["disclosures"]
    )
    stock_news_task = collect_stock_news(stock_name, market_cap)
    sector_news_task = collect_sector_news(sector, market_cap)

    results = await asyncio.gather(
        reports_task,
        disclosures_task,
        stock_news_task,
        sector_news_task,
        return_exceptions=True
    )

    # Handle results (convert exceptions to empty lists)
    research_reports = results[0] if isinstance(results[0], list) else []
    disclosures = results[1] if isinstance(results[1], list) else []
    stock_news = results[2] if isinstance(results[2], list) else []
    sector_news = results[3] if isinstance(results[3], list) else []

    return {
        "research_reports": research_reports,
        "disclosures": disclosures,
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
                "research_reports": len(research_reports),
                "disclosures": len(disclosures),
                "stock_news": len(stock_news),
                "sector_news": len(sector_news),
            }
        }
    }
