"""
Research data preprocessor module for US Stock Research Agent
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
# Analyst Reports Summary
# =============================================================================

def summarize_analyst_reports(reports: list[dict]) -> dict:
    """
    Summarize analyst reports for LLM consumption.

    Extracts:
    - Latest target prices and ratings
    - Report titles and summaries
    - Firm information

    Args:
        reports: List of analyst report items

    Returns:
        Structured summary dict
    """
    if not reports:
        return {
            "count": 0,
            "latest_ratings": [],
            "reports": [],
        }

    # Deduplicate by title
    unique_reports = deduplicate_items(reports, key="title")

    # Extract target prices and ratings
    latest_ratings = []
    for report in unique_reports[:5]:  # Top 5 most recent
        rating = {
            "title": truncate_text(normalize_text(report.get("title", "")), 100),
            "date": report.get("date"),
            "firm": report.get("firm"),
        }

        if report.get("target_price"):
            rating["target_price"] = report["target_price"]
        if report.get("rating"):
            rating["rating"] = report["rating"]

        latest_ratings.append(rating)

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
        if report.get("firm"):
            formatted["firm"] = report["firm"]

        formatted_reports.append(formatted)

    return {
        "count": len(unique_reports),
        "latest_ratings": latest_ratings,
        "reports": formatted_reports,
    }


# =============================================================================
# SEC Filings Summary
# =============================================================================

# SEC filing type classifications
SEC_FILING_TYPES = {
    "earnings": ["10-K", "10-Q", "8-K"],
    "proxy": ["DEF 14A", "DEFA14A"],
    "insider": ["Form 4", "Form 3", "Form 5"],
    "registration": ["S-1", "S-3", "424B"],
    "other": []
}


def classify_sec_filing(title: str) -> str:
    """
    Classify SEC filing type based on title.

    Args:
        title: Filing title

    Returns:
        Filing type string
    """
    title_upper = title.upper()

    for filing_type, patterns in SEC_FILING_TYPES.items():
        for pattern in patterns:
            if pattern in title_upper:
                return filing_type

    return "other"


def summarize_sec_filings(filings: list[dict]) -> dict:
    """
    Summarize SEC filing data for LLM consumption.

    Args:
        filings: List of SEC filing items

    Returns:
        Structured summary dict
    """
    if not filings:
        return {
            "count": 0,
            "by_type": {},
            "significant_items": [],
        }

    # Deduplicate
    unique_filings = deduplicate_by_similarity(filings, key="title", threshold=0.7)

    # Classify by type
    by_type = {}
    for filing in unique_filings:
        title = normalize_text(filing.get("title", ""))
        ftype = classify_sec_filing(title)

        if ftype not in by_type:
            by_type[ftype] = []

        by_type[ftype].append({
            "title": truncate_text(title, 150),
            "source": filing.get("source", "SEC"),
            "url": filing.get("url"),
        })

    # Identify significant items (earnings filings)
    significant_types = ["earnings", "proxy", "insider"]
    significant_items = []

    for ftype in significant_types:
        if ftype in by_type:
            for item in by_type[ftype][:2]:  # Max 2 per type
                significant_items.append({
                    "type": ftype,
                    "title": item["title"],
                    "source": item["source"],
                })

    return {
        "count": len(unique_filings),
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
        if news.get("snippet"):
            headline["summary"] = truncate_text(normalize_text(news.get("snippet", "")), 150)
        if news.get("date"):
            headline["date"] = news["date"]

        stock_headlines.append(headline)

    # Process sector news
    unique_sector_news = deduplicate_by_similarity(sector_news, key="title", threshold=0.7)
    sector_headlines = []

    for news in unique_sector_news[:10]:  # Max 10 sector headlines
        headline = {
            "title": truncate_text(normalize_text(news.get("title", "")), 100),
        }
        if news.get("snippet"):
            headline["summary"] = truncate_text(normalize_text(news.get("snippet", "")), 150)

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
# Insider Transactions Summary
# =============================================================================

def summarize_insider_transactions(transactions: list[dict]) -> dict:
    """
    Summarize insider transaction data for LLM.

    Args:
        transactions: List of insider transactions from DB

    Returns:
        Structured summary dict
    """
    if not transactions:
        return {
            "count": 0,
            "net_shares": 0,
            "net_value": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "notable_transactions": []
        }

    buy_shares = 0
    sell_shares = 0
    buy_value = 0.0
    sell_value = 0.0
    buy_count = 0
    sell_count = 0
    notable = []

    # Executive titles for highlighting
    executive_titles = ["CEO", "CFO", "COO", "President", "Director", "Chief"]

    for tx in transactions:
        tx_type = tx.get("transaction_type", "").upper()
        shares = tx.get("shares", 0) or 0
        price = tx.get("price", 0) or 0
        value = shares * price
        title = tx.get("position", "")
        name = tx.get("insider_name", "Unknown")

        is_buy = "BUY" in tx_type or "PURCHASE" in tx_type or "P" == tx_type
        is_sell = "SELL" in tx_type or "SALE" in tx_type or "S" == tx_type

        if is_buy:
            buy_shares += shares
            buy_value += value
            buy_count += 1
        elif is_sell:
            sell_shares += shares
            sell_value += value
            sell_count += 1

        # Track notable (executive) transactions
        is_executive = any(et in title for et in executive_titles)
        if is_executive and value > 100000:  # Over $100K
            notable.append({
                "name": name,
                "position": title,
                "type": "Buy" if is_buy else "Sell" if is_sell else "Other",
                "shares": shares,
                "value": round(value, 2)
            })

    return {
        "count": len(transactions),
        "net_shares": buy_shares - sell_shares,
        "net_value": round(buy_value - sell_value, 2),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "notable_transactions": notable[:5]  # Top 5 notable
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
    lines.append(f"- Market Cap Tier: {metadata.get('stock_size', 'Unknown')}")
    lines.append("")

    # Analyst Reports
    reports = preprocessed_data.get("analyst_summary", {})
    if reports.get("count", 0) > 0:
        lines.append("### Analyst Reports")
        lines.append(f"Total: {reports['count']} reports")
        lines.append("")

        # Latest ratings with target prices
        if reports.get("latest_ratings"):
            lines.append("**Latest Ratings:**")
            for rating in reports["latest_ratings"]:
                line = f"- {rating.get('title', 'N/A')}"
                if rating.get("firm"):
                    line += f" ({rating['firm']})"
                if rating.get("target_price"):
                    line += f" - Target: ${rating['target_price']:,.2f}"
                if rating.get("rating"):
                    line += f" [{rating['rating']}]"
                lines.append(line)
            lines.append("")

    # SEC Filings
    filings = preprocessed_data.get("sec_filings_summary", {})
    if filings.get("count", 0) > 0:
        lines.append("### SEC Filings")
        lines.append(f"Total: {filings['count']} filings")

        if filings.get("by_type"):
            type_str = ", ".join([f"{k}: {v}" for k, v in filings["by_type"].items()])
            lines.append(f"By Type: {type_str}")
        lines.append("")

        if filings.get("significant_items"):
            lines.append("**Significant Items:**")
            for item in filings["significant_items"]:
                lines.append(f"- [{item.get('type', 'other')}] {item.get('title', 'N/A')}")
            lines.append("")

    # Insider Transactions
    insider = preprocessed_data.get("insider_summary", {})
    if insider.get("count", 0) > 0:
        lines.append("### Insider Transactions")
        lines.append(f"Total: {insider['count']} transactions in 90 days")
        lines.append(f"Net Shares: {insider['net_shares']:,}")
        lines.append(f"Net Value: ${insider['net_value']:,.2f}")
        lines.append(f"Buys: {insider['buy_count']}, Sells: {insider['sell_count']}")
        lines.append("")

        if insider.get("notable_transactions"):
            lines.append("**Notable Executive Transactions:**")
            for tx in insider["notable_transactions"]:
                lines.append(f"- {tx['name']} ({tx['position']}): {tx['type']} {tx['shares']:,} shares (${tx['value']:,.2f})")
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
        - analyst_summary: Summarized analyst reports
        - sec_filings_summary: Summarized SEC filings
        - insider_summary: Summarized insider transactions
        - news_summary: Summarized news
        - formatted_text: Token-efficient text for LLM
        - metadata: Original metadata with processing info
    """
    metadata = collected_data.get("metadata", {})

    # Summarize each data type
    analyst_summary = summarize_analyst_reports(
        collected_data.get("analyst_reports", [])
    )

    sec_filings_summary = summarize_sec_filings(
        collected_data.get("sec_filings", [])
    )

    insider_summary = summarize_insider_transactions(
        collected_data.get("insider_transactions", [])
    )

    news_summary = summarize_news(
        collected_data.get("stock_news", []),
        collected_data.get("sector_news", [])
    )

    # Build preprocessed data structure
    preprocessed = {
        "analyst_summary": analyst_summary,
        "sec_filings_summary": sec_filings_summary,
        "insider_summary": insider_summary,
        "news_summary": news_summary,
        "metadata": {
            **metadata,
            "preprocessing_date": date.today().isoformat(),
            "processed_counts": {
                "analyst_reports": analyst_summary.get("count", 0),
                "sec_filings": sec_filings_summary.get("count", 0),
                "insider_transactions": insider_summary.get("count", 0),
                "stock_news": news_summary.get("stock_news", {}).get("count", 0),
                "sector_news": news_summary.get("sector_news", {}).get("count", 0),
            }
        }
    }

    # Generate formatted text for LLM
    preprocessed["formatted_text"] = format_for_llm(preprocessed, metadata)

    return preprocessed
