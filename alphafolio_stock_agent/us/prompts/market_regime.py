"""
Market Regime Agent prompt templates for US stocks
Analyzes US market conditions to determine regime (Risk-On/Off/Neutral)

Design principles:
1. Quantitative data (DB signals) = main driver
2. Qualitative data (news/search) = context/why
3. LLM combines signals + qualitative analysis -> final judgment
4. Task-driven approach (not Role-driven)
"""

# =============================================================================
# System Prompt
# =============================================================================

MARKET_REGIME_SYSTEM_PROMPT = """You are a Market Regime Analysis Module.

## Purpose
Analyze US and global market conditions to determine the current market regime (Risk-On/Off/Neutral).
Provide convincing analysis that individual investors can understand "why the market is in this state".

## Analysis Method
1. **Quantitative Signals (Main)**: QUANTITATIVE_SIGNALS determine the core direction
2. **Qualitative Analysis (Supporting)**: NEWS_DATA explains the background and context
3. **Combined Judgment**: Combine quantitative + qualitative for final regime determination

## Judgment Principles
- If quantitative signals clearly point to a direction (risk_on/risk_off): Follow that direction
- If quantitative signals are mixed: Use news analysis to determine tilt or judge as neutral
- Qualitative analysis does not override quantitative signals; it adjusts confidence and intensity

## Output Requirements
- All output must be in English
- Structure: What (conclusion) → Why (evidence) → How (investment implications)
- Use concrete numbers and fact-based explanations instead of abstract expressions
- Example: "VIX 18.5, credit spread stable, put/call ratio 0.85 → Risk-On"

## Constraints
- Use only the provided data (QUANTITATIVE_SIGNALS, NEWS_DATA)
- No direct DB queries, external searches, or new assumptions
- Output JSON exactly matching OUTPUT_SCHEMA
"""

# =============================================================================
# User Prompt Template
# =============================================================================

MARKET_REGIME_USER_PROMPT = """## TASK
Analyze US market environment and determine market regime

## QUANTITATIVE_SIGNALS (Main Driver)
{quantitative_signals}

## NEWS_DATA (Context and Why)

### US Market News
{us_news}

### Global Macro News
{global_news}

### Fed Policy News
{fed_news}

## CONTEXT
- Analysis Date: {analysis_date}
- Signal Summary: {signal_summary}

## ANALYSIS STEPS
Perform analysis in the following order:

### Step 1: Interpret Quantitative Signals
- Check each signal in QUANTITATIVE_SIGNALS (global_signals, us_market_signals)
- Identify direction (risk_on/risk_off/neutral) for each signal
- Check for consistency/inconsistency between signals

### Step 2: Analyze Qualitative Data
- US Market News: Investor sentiment, major issues
- Global News: Fed policy, geopolitical risks, economic data
- Fed Policy: Interest rate outlook, balance sheet policy

### Step 3: Global Impact Analysis
- Analyze global factors affecting US market
- Transmission channels: Dollar strength, credit conditions, liquidity
- Impact timing and intensity

### Step 4: Final Regime Determination
- Quantitative signal direction + Qualitative analysis → Final judgment
- Determine confidence score
- Summarize rationale in What→Why format

### Step 5: Investment Implications
- Investment stance for current regime (aggressive/moderate/conservative/defensive)
- Preferred/avoided sectors
- Key monitoring points

## OUTPUT_SCHEMA
```json
{{
  "regime": "risk_on | risk_off | neutral",
  "confidence": 0.0-1.0,
  "regime_rationale": "Core rationale for regime determination (2-3 sentences, What→Why format)",

  "global_market": {{
    "us_market_regime": "risk_on | risk_off | neutral",
    "vix_level": VIX_value,
    "vix_signal": "low_fear | moderate | high_fear | extreme_fear",
    "credit_spread_signal": "tight | normal | wide | very_wide",
    "yield_curve_signal": "normal | flat | inverted",
    "dollar_trend": "strengthening | stable | weakening",
    "safe_haven_flow": "Safe haven asset flow description",
    "key_indicators": [
      {{"name": "Indicator name", "value": value, "change": "change", "signal": "signal", "interpretation": "interpretation"}}
    ],
    "summary": "Global market summary (2-3 sentences)"
  }},

  "us_market": {{
    "sp500_trend": "uptrend | sideways | downtrend",
    "nasdaq_trend": "uptrend | sideways | downtrend",
    "market_breadth": "strong | moderate | weak",
    "sector_rotation": "Current sector rotation pattern",
    "put_call_ratio": put_call_ratio_value,
    "market_sentiment": "bullish | neutral | bearish",
    "key_events": ["Key event 1", "Key event 2"],
    "summary": "US market summary (2-3 sentences)"
  }},

  "news_sentiment": {{
    "market_news_sentiment": "positive | neutral | negative",
    "market_news_highlights": ["Key headline 1", "Key headline 2"],
    "fed_policy_sentiment": "hawkish | neutral | dovish",
    "fed_highlights": ["Key Fed point 1", "Key Fed point 2"]
  }},

  "investment_implications": {{
    "risk_appetite": "aggressive | moderate | conservative | defensive",
    "position_sizing": "Position sizing guidance",
    "sector_preference": ["Preferred sector 1", "Preferred sector 2"],
    "sectors_to_avoid": ["Sector to avoid 1", "Sector to avoid 2"],
    "hedging_recommendation": "Hedging strategy recommendation",
    "time_horizon": "Recommended investment horizon",
    "key_risks": ["Key risk 1", "Key risk 2"],
    "action_triggers": ["Regime change trigger 1", "Regime change trigger 2"]
  }},

  "evidence": [
    {{"type": "quantitative", "source": "Data source", "fact": "Specific fact", "interpretation": "Interpretation"}},
    {{"type": "qualitative", "source": "News source", "fact": "Specific fact", "interpretation": "Interpretation"}}
  ],
  "data_sources": ["us_vix", "us_credit_spread", "serper_news", ...],
  "analysis_timestamp": "ISO format timestamp"
}}
```

## OUTPUT
Output only JSON exactly matching OUTPUT_SCHEMA above. No other text.
"""

# =============================================================================
# Validation Retry Prompt
# =============================================================================

MARKET_REGIME_RETRY_PROMPT = """## VALIDATION FAILED - Correction Required

The following errors were found in the previous output:

### Error List
{validation_errors}

### Previous Output
{previous_output}

### Request
Correct the above errors and output JSON that fully conforms to OUTPUT_SCHEMA.

Check the following:
1. regime must be one of "risk_on", "risk_off", "neutral"
2. confidence must be a number between 0.0 and 1.0
3. All required fields must be filled
4. evidence must contain at least 2 quantitative and qualitative items each

Output only the corrected JSON.
"""

# =============================================================================
# Helper Functions
# =============================================================================

def format_market_regime_prompt(
    quantitative_signals: dict,
    us_news: str,
    global_news: str,
    fed_news: str,
    analysis_date: str,
    signal_summary: str
) -> str:
    """
    Format the market regime user prompt with data.

    Args:
        quantitative_signals: Output from regime_preprocessor.calculate_regime_signals()
        us_news: US market news text (from Serper)
        global_news: Global macro news text (from Serper)
        fed_news: Fed policy news text (from Serper)
        analysis_date: Analysis reference date (YYYY-MM-DD)
        signal_summary: Brief summary of quantitative signals

    Returns:
        Formatted prompt string
    """
    import json

    return MARKET_REGIME_USER_PROMPT.format(
        quantitative_signals=json.dumps(quantitative_signals, ensure_ascii=False, indent=2),
        us_news=us_news if us_news else "No data available",
        global_news=global_news if global_news else "No data available",
        fed_news=fed_news if fed_news else "No data available",
        analysis_date=analysis_date,
        signal_summary=signal_summary if signal_summary else "No signal summary available"
    )


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


def format_validation_retry_prompt(
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
    import json

    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    return MARKET_REGIME_RETRY_PROMPT.format(
        validation_errors=errors_text,
        previous_output=json.dumps(previous_output, ensure_ascii=False, indent=2)
    )


def generate_signal_summary(quantitative_signals: dict) -> str:
    """
    Generate a brief summary of quantitative signals for prompt context.

    Args:
        quantitative_signals: Output from regime_preprocessor.calculate_regime_signals()

    Returns:
        Brief summary string
    """
    summary_parts = []

    if "signal_summary" in quantitative_signals:
        sig = quantitative_signals["signal_summary"]

        # Global signals summary
        global_ro = sig.get("global_risk_on_count", 0)
        global_rf = sig.get("global_risk_off_count", 0)
        if global_ro > global_rf:
            summary_parts.append(f"Global: Risk-On dominant ({global_ro}:{global_rf})")
        elif global_rf > global_ro:
            summary_parts.append(f"Global: Risk-Off dominant ({global_rf}:{global_ro})")
        else:
            summary_parts.append(f"Global: Mixed ({global_ro}:{global_rf})")

        # US signals summary
        us_ro = sig.get("us_risk_on_count", 0)
        us_rf = sig.get("us_risk_off_count", 0)
        if us_ro > us_rf:
            summary_parts.append(f"US: Risk-On dominant ({us_ro}:{us_rf})")
        elif us_rf > us_ro:
            summary_parts.append(f"US: Risk-Off dominant ({us_rf}:{us_ro})")
        else:
            summary_parts.append(f"US: Mixed ({us_ro}:{us_rf})")

    return " | ".join(summary_parts) if summary_parts else "No signal analysis available"
