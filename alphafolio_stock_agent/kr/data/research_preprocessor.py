"""
Research data preprocessor module for Stock Research Agent
Transforms raw collected data into structured input for LLM
No LLM involved - pure Python data processing

Purpose:
- Normalize and clean text data
- Remove duplicates
- Filter by relevance
- Structure data for efficient LLM consumption
"""
import re
from typing import Optional
from datetime import date, datetime


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text by removing HTML tags, extra whitespace, and special characters.

    Args:
        text: Raw text to normalize

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", '&')
    text = text.replace("&lt;", '<')
    text = text.replace("&gt;", '>')
    text = text.replace("&apos;", "'")
    text = text.replace("&nbsp;", ' ')

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


# =============================================================================
# Deduplication
# =============================================================================

def deduplicate_items(items: list[dict], key: str = "title") -> list[dict]:
    """
    Remove duplicate items based on a key field.

    Args:
        items: List of items to deduplicate
        key: Field to use for deduplication

    Returns:
        Deduplicated list
    """
    if not items:
        return []

    seen = set()
    unique_items = []

    for item in items:
        value = item.get(key, "")
        if not value:
            continue

        # Normalize for comparison
        normalized = normalize_text(value).lower()

        if normalized not in seen:
            seen.add(normalized)
            unique_items.append(item)

    return unique_items


def deduplicate_by_similarity(items: list[dict], key: str = "title", threshold: float = 0.8) -> list[dict]:
    """
    Remove items with similar titles (simple word overlap check).

    Args:
        items: List of items to deduplicate
        key: Field to use for comparison
        threshold: Similarity threshold (0-1)

    Returns:
        Deduplicated list
    """
    if not items:
        return []

    def get_words(text: str) -> set:
        return set(normalize_text(text).lower().split())

    unique_items = []

    for item in items:
        text = item.get(key, "")
        if not text:
            continue

        words = get_words(text)
        is_duplicate = False

        for existing in unique_items:
            existing_words = get_words(existing.get(key, ""))
            if not existing_words:
                continue

            # Calculate Jaccard similarity
            intersection = len(words & existing_words)
            union = len(words | existing_words)
            similarity = intersection / union if union > 0 else 0

            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_items.append(item)

    return unique_items


# =============================================================================
# Research Reports Summary
# =============================================================================

def summarize_research_reports(reports: list[dict]) -> dict:
    """
    Summarize research reports for LLM consumption.

    Extracts:
    - Latest target prices and investment opinions
    - Report titles and summaries
    - Securities firm information

    Args:
        reports: List of research report items

    Returns:
        Structured summary dict
    """
    if not reports:
        return {
            "count": 0,
            "latest_opinions": [],
            "reports": [],
        }

    # Deduplicate by title
    unique_reports = deduplicate_items(reports, key="title")

    # Extract target prices and opinions
    latest_opinions = []
    for report in unique_reports[:5]:  # Top 5 most recent
        opinion = {
            "title": truncate_text(normalize_text(report.get("title", "")), 100),
            "date": report.get("date"),
            "securities_firm": report.get("securities_firm"),
        }

        if report.get("target_price"):
            opinion["target_price"] = report["target_price"]
        if report.get("investment_opinion"):
            opinion["investment_opinion"] = report["investment_opinion"]

        latest_opinions.append(opinion)

    # Format all reports for reference
    formatted_reports = []
    for report in unique_reports[:10]:  # Max 10 reports
        formatted = {
            "title": truncate_text(normalize_text(report.get("title", "")), 150),
            "source": report.get("source", "Unknown"),
        }

        if report.get("snippet"):
            formatted["summary"] = truncate_text(normalize_text(report.get("snippet", "")), 200)
        if report.get("date"):
            formatted["date"] = report["date"]
        if report.get("securities_firm"):
            formatted["securities_firm"] = report["securities_firm"]

        formatted_reports.append(formatted)

    return {
        "count": len(unique_reports),
        "latest_opinions": latest_opinions,
        "reports": formatted_reports,
    }


# =============================================================================
# Disclosures Summary
# =============================================================================

# Disclosure type keywords for classification
DISCLOSURE_TYPES = {
    "earnings": ["실적", "매출", "영업이익", "순이익", "분기보고서", "사업보고서"],
    "shareholder": ["지분", "주식", "최대주주", "자사주", "취득", "처분"],
    "executive": ["임원", "대표이사", "이사회", "선임", "해임", "사임"],
    "dividend": ["배당", "주당배당금", "배당금"],
    "investment": ["투자", "인수", "합병", "M&A", "계약"],
    "regulatory": ["공정거래", "제재", "과징금", "소송", "판결"],
}


def classify_disclosure(title: str) -> str:
    """
    Classify disclosure type based on title keywords.

    Args:
        title: Disclosure title

    Returns:
        Disclosure type string
    """
    title_lower = title.lower()

    for dtype, keywords in DISCLOSURE_TYPES.items():
        for keyword in keywords:
            if keyword in title_lower:
                return dtype

    return "other"


def summarize_disclosures(disclosures: list[dict]) -> dict:
    """
    Summarize disclosure data for LLM consumption.

    Classifies and groups disclosures by type.

    Args:
        disclosures: List of disclosure items

    Returns:
        Structured summary dict
    """
    if not disclosures:
        return {
            "count": 0,
            "by_type": {},
            "significant_items": [],
        }

    # Deduplicate
    unique_disclosures = deduplicate_by_similarity(disclosures, key="title", threshold=0.7)

    # Classify by type
    by_type = {}
    for disclosure in unique_disclosures:
        title = normalize_text(disclosure.get("title", ""))
        dtype = classify_disclosure(title)

        if dtype not in by_type:
            by_type[dtype] = []

        by_type[dtype].append({
            "title": truncate_text(title, 150),
            "source": disclosure.get("source", "Unknown"),
            "url": disclosure.get("url"),
        })

    # Identify significant items (earnings, major events)
    significant_types = ["earnings", "shareholder", "executive", "investment"]
    significant_items = []

    for dtype in significant_types:
        if dtype in by_type:
            for item in by_type[dtype][:2]:  # Max 2 per type
                significant_items.append({
                    "type": dtype,
                    "title": item["title"],
                    "source": item["source"],
                })

    return {
        "count": len(unique_disclosures),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "significant_items": significant_items[:10],  # Max 10 significant items
    }


# =============================================================================
# News Summary
# =============================================================================

def summarize_news(stock_news: list[dict], sector_news: list[dict]) -> dict:
    """
    Summarize news data for LLM consumption.

    Args:
        stock_news: List of stock-specific news items
        sector_news: List of sector-related news items

    Returns:
        Structured summary dict
    """
    # Process stock news
    unique_stock_news = deduplicate_by_similarity(stock_news, key="title", threshold=0.7)
    stock_headlines = []

    for news in unique_stock_news[:15]:  # Max 15 headlines
        headline = {
            "title": truncate_text(normalize_text(news.get("title", "")), 100),
        }
        if news.get("description"):
            headline["summary"] = truncate_text(normalize_text(news.get("description", "")), 150)
        if news.get("pubDate"):
            headline["date"] = news["pubDate"]

        stock_headlines.append(headline)

    # Process sector news
    unique_sector_news = deduplicate_by_similarity(sector_news, key="title", threshold=0.7)
    sector_headlines = []

    for news in unique_sector_news[:10]:  # Max 10 sector headlines
        headline = {
            "title": truncate_text(normalize_text(news.get("title", "")), 100),
        }
        if news.get("description"):
            headline["summary"] = truncate_text(normalize_text(news.get("description", "")), 150)

        sector_headlines.append(headline)

    return {
        "stock_news": {
            "count": len(unique_stock_news),
            "headlines": stock_headlines,
        },
        "sector_news": {
            "count": len(unique_sector_news),
            "headlines": sector_headlines,
        },
    }


# =============================================================================
# LLM Input Formatting
# =============================================================================

def format_for_llm(preprocessed_data: dict, metadata: dict) -> str:
    """
    Format preprocessed data as text for LLM consumption.
    Optimized for token efficiency.

    Args:
        preprocessed_data: Output from preprocess_research_data
        metadata: Collection metadata

    Returns:
        Formatted text string
    """
    lines = []

    # Header
    lines.append(f"## Stock Research Data: {metadata.get('stock_name', 'Unknown')}")
    lines.append(f"- Symbol: {metadata.get('symbol', 'Unknown')}")
    lines.append(f"- Sector: {metadata.get('sector', 'Unknown')}")
    lines.append(f"- Stock Size: {metadata.get('stock_size', 'Unknown')}")
    lines.append("")

    # Research Reports
    reports = preprocessed_data.get("reports_summary", {})
    if reports.get("count", 0) > 0:
        lines.append("### Research Reports")
        lines.append(f"Total: {reports['count']} reports")
        lines.append("")

        # Latest opinions with target prices
        if reports.get("latest_opinions"):
            lines.append("**Latest Opinions:**")
            for op in reports["latest_opinions"]:
                line = f"- {op.get('title', 'N/A')}"
                if op.get("securities_firm"):
                    line += f" ({op['securities_firm']})"
                if op.get("target_price"):
                    line += f" - Target: {op['target_price']:,.0f}won"
                if op.get("investment_opinion"):
                    line += f" [{op['investment_opinion']}]"
                lines.append(line)
            lines.append("")

    # Disclosures
    disclosures = preprocessed_data.get("disclosures_summary", {})
    if disclosures.get("count", 0) > 0:
        lines.append("### Disclosures")
        lines.append(f"Total: {disclosures['count']} disclosures")

        if disclosures.get("by_type"):
            type_str = ", ".join([f"{k}: {v}" for k, v in disclosures["by_type"].items()])
            lines.append(f"By Type: {type_str}")
        lines.append("")

        if disclosures.get("significant_items"):
            lines.append("**Significant Items:**")
            for item in disclosures["significant_items"]:
                lines.append(f"- [{item.get('type', 'other')}] {item.get('title', 'N/A')}")
            lines.append("")

    # News
    news = preprocessed_data.get("news_summary", {})

    # Stock News
    stock_news = news.get("stock_news", {})
    if stock_news.get("count", 0) > 0:
        lines.append("### Stock News")
        lines.append(f"Total: {stock_news['count']} articles")
        lines.append("")
        lines.append("**Headlines:**")
        for headline in stock_news.get("headlines", [])[:10]:
            lines.append(f"- {headline.get('title', 'N/A')}")
            if headline.get("summary"):
                lines.append(f"  {headline['summary']}")
        lines.append("")

    # Sector News
    sector_news = news.get("sector_news", {})
    if sector_news.get("count", 0) > 0:
        lines.append(f"### Sector News ({metadata.get('sector', 'Unknown')})")
        lines.append(f"Total: {sector_news['count']} articles")
        lines.append("")
        lines.append("**Headlines:**")
        for headline in sector_news.get("headlines", [])[:7]:
            lines.append(f"- {headline.get('title', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Preprocessing Function
# =============================================================================

def preprocess_research_data(collected_data: dict) -> dict:
    """
    Preprocess all collected research data for LLM consumption.

    Takes output from collect_research_data() and transforms it into
    a structured format optimized for the Stock Research Agent LLM.

    Args:
        collected_data: Output from research_collector.collect_research_data()

    Returns:
        Preprocessed data dict:
        - reports_summary: Summarized research reports
        - disclosures_summary: Summarized disclosures
        - news_summary: Summarized news
        - formatted_text: Token-efficient text for LLM
        - metadata: Original metadata with processing info
    """
    metadata = collected_data.get("metadata", {})

    # Summarize each data type
    reports_summary = summarize_research_reports(
        collected_data.get("research_reports", [])
    )

    disclosures_summary = summarize_disclosures(
        collected_data.get("disclosures", [])
    )

    news_summary = summarize_news(
        collected_data.get("stock_news", []),
        collected_data.get("sector_news", [])
    )

    # Build preprocessed data structure
    preprocessed = {
        "reports_summary": reports_summary,
        "disclosures_summary": disclosures_summary,
        "news_summary": news_summary,
        "metadata": {
            **metadata,
            "preprocessing_date": date.today().isoformat(),
            "processed_counts": {
                "research_reports": reports_summary.get("count", 0),
                "disclosures": disclosures_summary.get("count", 0),
                "stock_news": news_summary.get("stock_news", {}).get("count", 0),
                "sector_news": news_summary.get("sector_news", {}).get("count", 0),
            }
        }
    }

    # Generate formatted text for LLM
    preprocessed["formatted_text"] = format_for_llm(preprocessed, metadata)

    return preprocessed
