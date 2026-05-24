"""
Stock Research Agent prompt templates for US stocks
Extracts qualitative data: earnings_outlook, risk_issues, sector_momentum, insider_activity

Design principles:
1. LLM extracts only 4 items: earnings_outlook, risk_issues, sector_momentum, insider_activity
2. No fabrication - only use INPUT_DATA
3. Source mapping required for all summaries
4. Task-driven approach (not Role-driven)
5. Context addition only - does not modify quant results
"""
import json
from datetime import date


# =============================================================================
# Output Schema (for prompt reference)
# =============================================================================

OUTPUT_SCHEMA = """{
  "symbol": "Stock ticker",
  "stock_name": "Company name",
  "analysis_date": "YYYY-MM-DD",
  "sector": "Sector name",

  "overall_sentiment": "positive | neutral | negative",
  "confidence": 0.0-1.0,

  "earnings_outlook": {
    "direction": "positive | neutral | negative",
    "summary": "Earnings outlook summary (1-2 sentences)",
    "key_points": ["Key point 1", "Key point 2"],
    "sources": [
      {"type": "analyst_report | news | sec_filing", "title": "Title", "publisher": "Source", "date": "Date"}
    ]
  },

  "risk_issues": [
    {
      "category": "regulatory | market | operational | financial | other",
      "severity": "high | medium | low",
      "title": "Risk title",
      "summary": "Risk description (1-2 sentences)",
      "sources": [{"type": "...", "title": "...", "publisher": "...", "date": "..."}]
    }
  ],

  "sector_momentum": {
    "sector": "Sector name",
    "direction": "positive | neutral | negative",
    "summary": "Sector momentum summary (1-2 sentences)",
    "related_news_count": news_count,
    "sources": [{"type": "...", "title": "...", "publisher": "...", "date": "..."}]
  },

  "insider_activity": {
    "signal": "STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL",
    "net_shares": net_shares_count,
    "net_value": net_dollar_value,
    "summary": "Insider activity summary (1-2 sentences)",
    "sources": [{"type": "insider_transaction", "title": "...", "date": "..."}]
  },

  "metadata": {
    "data_collection": {
      "news_articles": article_count,
      "insider_transactions": transaction_count,
      "sec_filings": filing_count
    },
    "market_cap_tier": "large | mid | small",
    "collection_date": "YYYY-MM-DD"
  }
}"""


# =============================================================================
# System Prompt
# =============================================================================

STOCK_RESEARCH_SYSTEM_PROMPT = """You are a Stock Research Data Extraction Module.

## Purpose
Extract information from the provided data and transform it into a structured format.
The extracted information is used to confirm the direction of quant analysis results and enrich explanations.

## Core Principles
1. **Data-based**: Use only information from INPUT_DATA
2. **Extraction only**: Do not generate new analysis or predictions
3. **Source required**: All summaries must specify sources
4. **Focus on 4 items**: Extract only earnings_outlook, risk_issues, sector_momentum, insider_activity

## Extraction Guidelines

### earnings_outlook (Earnings Outlook)
- Extract analyst target prices, ratings, earnings forecasts from reports
- Extract earnings-related content from news
- direction judgment criteria:
  - positive: Earnings improvement, target price raised, buy ratings dominant
  - neutral: Mixed opinions, earnings maintenance forecast
  - negative: Earnings deterioration, target price lowered, sell ratings dominant

### risk_issues (Risk Issues)
- Extract regulatory, market, operational, financial risks
- category classification:
  - regulatory: Government regulation, legal issues, sanctions
  - market: Competition intensification, demand decrease, price decline
  - operational: Production issues, quality issues, labor problems
  - financial: Debt, liquidity, currency risk
  - other: Other risks
- severity judgment:
  - high: Immediate action needed, direct impact on earnings/stock price
  - medium: Watch needed, potential impact
  - low: Reference information

### sector_momentum (Sector Momentum)
- Extract industry/sector trends from sector-related news
- Include related stocks, policies, market trends
- direction judgment criteria:
  - positive: Sector growth, policy support, investment expansion
  - neutral: No significant change
  - negative: Sector decline, regulation strengthening, investment reduction

### insider_activity (Insider Trading Activity)
- Summarize insider buy/sell transactions
- Weight executive (CEO/CFO) transactions higher
- signal judgment:
  - STRONG_BUY: Large net buying by executives
  - BUY: Net buying
  - NEUTRAL: Balanced or no activity
  - SELL: Net selling
  - STRONG_SELL: Large net selling by executives

## Constraints
- Do not generate information not in INPUT_DATA
- No speculation or assumptions
- Do not include content without sources
- Output JSON exactly matching OUTPUT_SCHEMA
"""


# =============================================================================
# User Prompt Template
# =============================================================================

STOCK_RESEARCH_USER_PROMPT = """## TASK
Extract qualitative research data for US stock

## TARGET_STOCK
- Company Name: {stock_name}
- Ticker: {symbol}
- Sector: {sector}
- Market Cap Tier: {stock_size}
- Analysis Date: {analysis_date}

## INPUT_DATA
{formatted_text}

## DATA_STATISTICS
- News Articles: {news_count}
- Insider Transactions: {insider_count}
- SEC Filings: {filing_count}

## EXTRACTION_INSTRUCTIONS

### Step 1: News Analysis
- Extract earnings/risk-related content from stock news
- Extract industry trends from sector news
- Identify positive/negative sentiment ratio

### Step 2: Insider Transaction Analysis
- Identify buy vs sell transactions
- Weight executive transactions (CEO, CFO) higher
- Calculate net shares and net value

### Step 3: Sector Analysis
- Extract sector-wide trends and themes
- Identify policy impacts and competitive dynamics

### Step 4: Overall Sentiment Determination
- Combine earnings_outlook, risk_issues, sector_momentum, insider_activity
- Determine overall_sentiment and confidence
- confidence criteria:
  - 0.8+: Sufficient data, clear direction
  - 0.5-0.8: Moderate data, direction exists
  - Below 0.5: Insufficient data or mixed signals

## OUTPUT_SCHEMA
```json
{output_schema}
```

## OUTPUT
Output only JSON exactly matching OUTPUT_SCHEMA above. No other text.
If INPUT_DATA lacks information, fill all schema fields anyway.
Use empty lists ([]) or "No information available" for missing data.
"""


# =============================================================================
# Validation Retry Prompt
# =============================================================================

STOCK_RESEARCH_RETRY_PROMPT = """## VALIDATION FAILED - Correction Required

The following errors were found in the previous output:

### Error List
{validation_errors}

### Previous Output
{previous_output}

### Request
Correct the above errors and output JSON that fully conforms to OUTPUT_SCHEMA.

Check the following:
1. overall_sentiment must be one of "positive", "neutral", "negative"
2. confidence must be a number between 0.0 and 1.0
3. earnings_outlook.direction, sector_momentum.direction follow same rules
4. risk_issues items must have category, severity, title, summary
5. insider_activity.signal must be one of STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
6. All summaries must include sources array (can be empty)

Output only the corrected JSON.
"""


# =============================================================================
# Helper Functions
# =============================================================================

def format_stock_research_prompt(
    symbol: str,
    stock_name: str,
    sector: str,
    stock_size: str,
    analysis_date: str,
    formatted_text: str,
    news_count: int,
    insider_count: int,
    filing_count: int
) -> str:
    """
    Format the stock research user prompt with data.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        stock_name: Company name
        sector: Sector from us_stock_basic.sector
        stock_size: "large" | "mid" | "small"
        analysis_date: Analysis date (YYYY-MM-DD)
        formatted_text: Preprocessed text from research_preprocessor
        news_count: Number of news articles
        insider_count: Number of insider transactions
        filing_count: Number of SEC filings

    Returns:
        Formatted prompt string
    """
    return STOCK_RESEARCH_USER_PROMPT.format(
        symbol=symbol,
        stock_name=stock_name,
        sector=sector if sector else "Unknown",
        stock_size=stock_size,
        analysis_date=analysis_date,
        formatted_text=formatted_text if formatted_text else "No data available",
        news_count=news_count,
        insider_count=insider_count,
        filing_count=filing_count,
        output_schema=OUTPUT_SCHEMA
    )


def format_research_retry_prompt(
    validation_errors: list[str],
    previous_output: dict
) -> str:
    """
    Format the validation retry prompt.

    Args:
        validation_errors: List of validation error messages
        previous_output: Previous LLM output that failed validation

    Returns:
        Formatted retry prompt string
    """
    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    return STOCK_RESEARCH_RETRY_PROMPT.format(
        validation_errors=errors_text,
        previous_output=json.dumps(previous_output, ensure_ascii=False, indent=2)
    )


def get_output_schema() -> str:
    """
    Get the output schema string for reference.

    Returns:
        Output schema JSON string
    """
    return OUTPUT_SCHEMA


def validate_research_output(output: dict) -> list[str]:
    """
    Validate stock research output and return list of errors.

    Args:
        output: LLM output dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required top-level fields
    required_fields = ["symbol", "stock_name", "analysis_date", "sector",
                       "overall_sentiment", "confidence", "earnings_outlook",
                       "sector_momentum"]

    for field in required_fields:
        if field not in output:
            errors.append(f"Missing required field: {field}")

    # Check overall_sentiment value
    if output.get("overall_sentiment") not in ["positive", "neutral", "negative"]:
        errors.append("overall_sentiment must be 'positive', 'neutral', or 'negative'")

    # Check confidence range
    confidence = output.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            errors.append("confidence must be a number between 0 and 1")

    # Check earnings_outlook
    earnings = output.get("earnings_outlook", {})
    if earnings:
        if earnings.get("direction") not in ["positive", "neutral", "negative"]:
            errors.append("earnings_outlook.direction must be 'positive', 'neutral', or 'negative'")
        if not earnings.get("summary"):
            errors.append("earnings_outlook.summary is required")

    # Check sector_momentum
    sector_mom = output.get("sector_momentum", {})
    if sector_mom:
        if sector_mom.get("direction") not in ["positive", "neutral", "negative"]:
            errors.append("sector_momentum.direction must be 'positive', 'neutral', or 'negative'")
        if not sector_mom.get("summary"):
            errors.append("sector_momentum.summary is required")

    # Check insider_activity
    insider = output.get("insider_activity", {})
    if insider:
        valid_signals = ["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]
        if insider.get("signal") not in valid_signals:
            errors.append(f"insider_activity.signal must be one of {valid_signals}")

    # Check risk_issues structure
    risk_issues = output.get("risk_issues", [])
    valid_categories = ["regulatory", "market", "operational", "financial", "other"]
    valid_severities = ["high", "medium", "low"]

    for i, risk in enumerate(risk_issues):
        if risk.get("category") not in valid_categories:
            errors.append(f"risk_issues[{i}].category must be one of {valid_categories}")
        if risk.get("severity") not in valid_severities:
            errors.append(f"risk_issues[{i}].severity must be one of {valid_severities}")
        if not risk.get("title"):
            errors.append(f"risk_issues[{i}].title is required")
        if not risk.get("summary"):
            errors.append(f"risk_issues[{i}].summary is required")

    return errors
