"""
Market Regime Agent module for US stocks
Analyzes US market conditions and determines market regime (Risk-On/Off/Neutral)

Design principles:
1. System-controlled structure (not LLM self-deciding)
2. Task-driven approach (not Role-driven)
3. Judgment + Validation loop
4. Quantitative data = main driver, Qualitative data = context/why
"""
import json
import re
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from us.config import settings
from us.state import AgentState
from us.db import queries
from us.data.regime_preprocessor import (
    calculate_regime_signals,
    generate_signal_summary
)
from us.tools.search_tools import search_market_news, format_news_for_prompt
from us.prompts.market_regime import (
    MARKET_REGIME_SYSTEM_PROMPT,
    format_market_regime_prompt,
    format_validation_retry_prompt
)
from us.utils.debug_saver import save_market_regime


# =============================================================================
# Data Collection for Market Regime
# =============================================================================

async def collect_regime_data() -> dict:
    """
    Collect all data needed for US market regime analysis.

    Sources:
    - PostgreSQL: VIX, credit spread, dollar index, Treasury yields,
                  MOVE index, US ETF prices, market indices
    - Search APIs: Serper (US market news, global macro, Fed policy)

    Returns:
        dict with 'quantitative' and 'qualitative' data
    """
    data = {
        "quantitative": {},
        "qualitative": {}
    }

    # === Quantitative Data (DB) ===

    # US VIX
    try:
        vix_data = await queries.get_us_vix(days=30)
        data["quantitative"]["vix"] = vix_data
    except Exception:
        data["quantitative"]["vix"] = []

    # US Credit Spread
    try:
        credit_spread_data = await queries.get_us_credit_spread(days=30)
        data["quantitative"]["credit_spread"] = credit_spread_data
    except Exception:
        data["quantitative"]["credit_spread"] = []

    # US Dollar Index
    try:
        dollar_index_data = await queries.get_us_dollar_index(days=30)
        data["quantitative"]["dollar_index"] = dollar_index_data
    except Exception:
        data["quantitative"]["dollar_index"] = []

    # US Treasury Yield
    try:
        treasury_yield_data = await queries.get_us_treasury_yield(days=30)
        data["quantitative"]["treasury_yield"] = treasury_yield_data
    except Exception:
        data["quantitative"]["treasury_yield"] = []

    # MOVE Index
    try:
        move_index_data = await queries.get_us_move_index(days=30)
        data["quantitative"]["move_index"] = move_index_data
    except Exception:
        data["quantitative"]["move_index"] = []

    # US ETF Prices (safe haven and risk assets)
    try:
        us_etf_data = await queries.get_us_etf_prices(
            symbols=["GLD", "TLT", "SHY", "HYG", "SPY", "QQQ"],
            days=30
        )
        data["quantitative"]["us_etf"] = us_etf_data
    except Exception:
        data["quantitative"]["us_etf"] = {}

    # Market Indices (S&P 500, NASDAQ)
    try:
        sp500_data = await queries.get_market_index("SPX", days=30)
        data["quantitative"]["sp500"] = sp500_data
    except Exception:
        data["quantitative"]["sp500"] = []

    try:
        nasdaq_data = await queries.get_market_index("NDX", days=30)
        data["quantitative"]["nasdaq"] = nasdaq_data
    except Exception:
        data["quantitative"]["nasdaq"] = []

    # === Qualitative Data (Search) ===

    try:
        news_results = await search_market_news()
        data["qualitative"]["news"] = news_results
    except Exception:
        data["qualitative"]["news"] = {
            "us_market": [],
            "global_macro": [],
            "fed_policy": []
        }

    return data


# =============================================================================
# Node: Collect Market Regime Data
# =============================================================================

async def collect_market_regime_data(state: AgentState) -> dict:
    """
    Collect data for market regime analysis.

    This node:
    1. Fetches quantitative data from PostgreSQL
    2. Fetches qualitative data from search APIs

    Args:
        state: Current workflow state

    Returns:
        Updated state with regime_raw_data
    """
    try:
        regime_data = await collect_regime_data()

        # Store in state's raw_data if it exists, or create new structure
        raw_data = state.get("raw_data") or {}
        raw_data["regime_data"] = regime_data

        return {
            "raw_data": raw_data
        }

    except Exception as e:
        return {
            "error": f"Market regime data collection failed: {str(e)}"
        }


# =============================================================================
# Node: Analyze Market Regime
# =============================================================================

async def analyze_market_regime(state: AgentState) -> dict:
    """
    Analyze market regime using quantitative signals + qualitative context.

    This node:
    1. Preprocesses quantitative data into signals (Python)
    2. Formats qualitative data for context
    3. Calls LLM to combine signals and generate regime judgment

    Args:
        state: Current workflow state (must have regime_data in raw_data)

    Returns:
        Updated state with market_regime result
    """
    try:
        analysis_date = state.get("target_date", datetime.now().strftime("%Y-%m-%d"))

        raw_data = state.get("raw_data", {})
        regime_data = raw_data.get("regime_data", {})

        if not regime_data:
            return {
                "error": "No regime data available for analysis"
            }

        quant_data = regime_data.get("quantitative", {})
        qual_data = regime_data.get("qualitative", {})

        # === Step 1: Calculate quantitative signals ===

        # Extract price lists from market indices
        sp500_prices = []
        nasdaq_prices = []

        if quant_data.get("sp500"):
            sp500_prices = [float(r.get("close", 0)) for r in quant_data["sp500"]]
        if quant_data.get("nasdaq"):
            nasdaq_prices = [float(r.get("close", 0)) for r in quant_data["nasdaq"]]

        # Calculate all signals using regime_preprocessor
        quantitative_signals = calculate_regime_signals(
            vix_data=quant_data.get("vix"),
            credit_spread_data=quant_data.get("credit_spread"),
            dollar_index_data=quant_data.get("dollar_index"),
            treasury_yield_data=quant_data.get("treasury_yield"),
            move_index_data=quant_data.get("move_index"),
            us_etf_data=quant_data.get("us_etf"),
            sp500_prices=sp500_prices,
            nasdaq_prices=nasdaq_prices
        )

        # === Step 2: Format qualitative data ===

        news_data = qual_data.get("news", {})
        us_news, global_news, fed_news = format_news_for_prompt(news_data)

        # === Step 3: Generate signal summary ===

        signal_summary = generate_signal_summary(quantitative_signals)

        # === Step 4: Call LLM ===

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )

        user_prompt = format_market_regime_prompt(
            quantitative_signals=quantitative_signals,
            us_news=us_news,
            global_news=global_news,
            fed_news=fed_news,
            analysis_date=analysis_date,
            signal_summary=signal_summary
        )

        messages = [
            SystemMessage(content=MARKET_REGIME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await llm.ainvoke(messages)

        # === Step 5: Parse response ===

        market_regime = parse_llm_response(response.content)

        if market_regime is None:
            return {
                "error": "Failed to parse market regime LLM response"
            }

        # Add data sources to result
        market_regime["data_sources"] = get_used_data_sources(quant_data, qual_data)
        market_regime["analysis_timestamp"] = datetime.now().isoformat()

        # Save intermediate result for debugging
        symbol = state.get("symbol", "MARKET")
        try:
            save_market_regime(symbol, market_regime)
        except Exception:
            pass  # Don't fail if debug save fails

        return {
            "market_regime": market_regime
        }

    except Exception as e:
        return {
            "error": f"Market regime analysis failed: {str(e)}"
        }


def parse_llm_response(content: str) -> Optional[dict]:
    """
    Parse LLM response content as JSON.

    Args:
        content: Raw LLM response content

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        # Try direct JSON parse
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in content
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def get_used_data_sources(quant_data: dict, qual_data: dict) -> list[str]:
    """
    Generate list of data sources actually used.

    Args:
        quant_data: Quantitative data dict
        qual_data: Qualitative data dict

    Returns:
        List of data source names
    """
    sources = []

    # Check quantitative sources
    if quant_data.get("vix"):
        sources.append("us_vix")
    if quant_data.get("credit_spread"):
        sources.append("us_credit_spread")
    if quant_data.get("dollar_index"):
        sources.append("us_dollar_index")
    if quant_data.get("treasury_yield"):
        sources.append("us_treasury_yield")
    if quant_data.get("move_index"):
        sources.append("us_move_index")
    if quant_data.get("us_etf"):
        sources.append("us_etf_prices")
    if quant_data.get("sp500"):
        sources.append("sp500_index")
    if quant_data.get("nasdaq"):
        sources.append("nasdaq_index")

    # Check qualitative sources
    news = qual_data.get("news", {})
    if news.get("us_market"):
        sources.append("serper_us_market")
    if news.get("global_macro"):
        sources.append("serper_global")
    if news.get("fed_policy"):
        sources.append("serper_fed_policy")

    return sources


# =============================================================================
# Validation
# =============================================================================

async def validate_market_regime(state: AgentState) -> dict:
    """
    Validate market regime output.

    Args:
        state: Current workflow state

    Returns:
        Updated state with validation errors if any
    """
    market_regime = state.get("market_regime")

    if not market_regime:
        return {
            "validation_errors": ["No market regime output to validate"]
        }

    errors = []

    # Check required fields
    if market_regime.get("regime") not in ["risk_on", "risk_off", "neutral"]:
        errors.append(f"Invalid regime value: {market_regime.get('regime')}")

    confidence = market_regime.get("confidence")
    if confidence is None or not (0 <= confidence <= 1):
        errors.append(f"Invalid confidence value: {confidence}")

    if not market_regime.get("regime_rationale"):
        errors.append("Missing regime_rationale")

    # Check nested structures exist
    if not market_regime.get("global_market"):
        errors.append("Missing global_market section")
    if not market_regime.get("us_market"):
        errors.append("Missing us_market section")
    if not market_regime.get("investment_implications"):
        errors.append("Missing investment_implications section")

    # Check evidence
    evidence = market_regime.get("evidence", [])
    if len(evidence) < 2:
        errors.append("Insufficient evidence (need at least 2 items)")

    return {
        "validation_errors": errors
    }


# =============================================================================
# Combined Node for Parallel Execution
# =============================================================================

async def market_regime_node(state: AgentState) -> dict:
    """
    Combined LangGraph node for market regime analysis.

    This single node performs:
    1. Data collection
    2. Signal calculation
    3. LLM analysis
    4. Validation

    This is useful for parallel execution with stock_research_node.

    Args:
        state: Current workflow state

    Returns:
        Updated state with market_regime result
    """
    # Step 1: Collect data
    collect_result = await collect_market_regime_data(state)

    if collect_result.get("error"):
        return collect_result

    # Merge collection result into state
    updated_state = {**state}
    if collect_result.get("raw_data"):
        updated_state["raw_data"] = collect_result["raw_data"]

    # Step 2: Analyze
    analyze_result = await analyze_market_regime(updated_state)

    # Combine results
    result = {}
    if collect_result.get("raw_data"):
        result["raw_data"] = collect_result["raw_data"]
    if analyze_result.get("market_regime"):
        result["market_regime"] = analyze_result["market_regime"]
    if analyze_result.get("error"):
        result["error"] = analyze_result["error"]

    return result


# =============================================================================
# Standalone Execution
# =============================================================================

async def run_market_regime_analysis(target_date: str = None) -> dict:
    """
    Run market regime analysis standalone (without full workflow).

    This can be used for testing or to get market regime independently.

    Args:
        target_date: Analysis date (YYYY-MM-DD), defaults to today

    Returns:
        Market regime result dict
    """
    from us.state import create_initial_state

    # Create minimal state
    state = create_initial_state(symbol="", target_date=target_date)

    # Collect data
    result = await collect_market_regime_data(state)
    if result.get("error"):
        return {"error": result["error"]}

    state["raw_data"] = result.get("raw_data", {})

    # Analyze
    result = await analyze_market_regime(state)
    if result.get("error"):
        return {"error": result["error"]}

    return result.get("market_regime", {})


# =============================================================================
# CLI Test Entry Point
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("Running US Market Regime Analysis...")
        print("-" * 50)

        result = await run_market_regime_analysis()

        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(main())
