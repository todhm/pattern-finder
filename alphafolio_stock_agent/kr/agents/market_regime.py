"""
Market Regime Agent module
Analyzes global market conditions and determines Korean market regime (Risk-On/Off/Neutral)

Design principles (from 에이전트 작업 계획.md):
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

from kr.config import settings
from kr.state import AgentState
from kr.schemas import MarketRegimeOutput
from kr.db import queries
from kr.data.regime_preprocessor import calculate_regime_signals
from kr.tools.search_tools import search_market_news
from kr.prompts.market_regime import (
    MARKET_REGIME_SYSTEM_PROMPT,
    format_market_regime_prompt,
    format_news_for_prompt,
    generate_signal_summary,
    format_validation_retry_prompt
)
from kr.utils.debug_saver import save_market_regime


# =============================================================================
# Data Collection for Market Regime
# =============================================================================

async def collect_regime_data() -> dict:
    """
    Collect all data needed for market regime analysis.

    Sources:
    - PostgreSQL: VIX, VKOSPI, credit spread, dollar index, exchange rate,
                  MOVE index, US ETF prices, market indices
    - Search APIs: Naver (Korean news), Serper (global news, foreign view)

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

    # VKOSPI (Korea VIX)
    try:
        vkospi_data = await queries.get_vkospi(days=30)
        data["quantitative"]["vkospi"] = vkospi_data
    except Exception:
        data["quantitative"]["vkospi"] = []

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

    # Exchange Rate (USD/KRW)
    try:
        exchange_rate_data = await queries.get_exchange_rate(days=30)
        data["quantitative"]["exchange_rate"] = exchange_rate_data
    except Exception:
        data["quantitative"]["exchange_rate"] = []

    # MOVE Index
    try:
        move_index_data = await queries.get_us_move_index(days=30)
        data["quantitative"]["move_index"] = move_index_data
    except Exception:
        data["quantitative"]["move_index"] = []

    # US ETF Prices (safe haven assets)
    try:
        us_etf_data = await queries.get_us_etf_prices(
            symbols=["GLD", "TLT", "SHY", "HYG", "AGG"],
            days=30
        )
        data["quantitative"]["us_etf"] = us_etf_data
    except Exception:
        data["quantitative"]["us_etf"] = {}

    # Market Indices (KOSPI, KOSDAQ)
    try:
        market_indices = await queries.get_market_indices(
            exchanges=["KOSPI", "KOSDAQ"],
            days=30
        )
        data["quantitative"]["market_indices"] = market_indices
    except Exception:
        data["quantitative"]["market_indices"] = {}

    # Foreign investor flow (calculate from market index data)
    # This would need additional calculation from investor trading data
    data["quantitative"]["foreign_flow_5d"] = None
    data["quantitative"]["foreign_flow_20d"] = None

    # === Qualitative Data (Search) ===

    try:
        news_results = await search_market_news()
        data["qualitative"]["news"] = news_results
    except Exception:
        data["qualitative"]["news"] = {
            "korea_market": [],
            "global_macro": [],
            "korea_foreign_view": []
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
        kospi_prices = []
        kosdaq_prices = []

        market_indices = quant_data.get("market_indices", {})
        if "KOSPI" in market_indices:
            kospi_prices = [float(r.get("close", 0)) for r in market_indices["KOSPI"]]
        if "KOSDAQ" in market_indices:
            kosdaq_prices = [float(r.get("close", 0)) for r in market_indices["KOSDAQ"]]

        # Calculate all signals using regime_preprocessor
        quantitative_signals = calculate_regime_signals(
            vkospi_data=quant_data.get("vkospi"),
            vix_data=quant_data.get("vix"),
            credit_spread_data=quant_data.get("credit_spread"),
            dollar_index_data=quant_data.get("dollar_index"),
            exchange_rate_data=quant_data.get("exchange_rate"),
            move_index_data=quant_data.get("move_index"),
            us_etf_data=quant_data.get("us_etf"),
            foreign_flow_5d=quant_data.get("foreign_flow_5d"),
            foreign_flow_20d=quant_data.get("foreign_flow_20d"),
            kospi_prices=kospi_prices,
            kosdaq_prices=kosdaq_prices
        )

        # === Step 2: Format qualitative data ===

        news_data = qual_data.get("news", {})
        korea_news, global_news, foreign_view_news = format_news_for_prompt(news_data)

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
            korea_news=korea_news,
            global_news=global_news,
            foreign_view_news=foreign_view_news,
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
    if quant_data.get("vkospi"):
        sources.append("kr_benchmark_index (VKOSPI)")
    if quant_data.get("credit_spread"):
        sources.append("us_credit_spread")
    if quant_data.get("dollar_index"):
        sources.append("us_dollar_index")
    if quant_data.get("exchange_rate"):
        sources.append("exchange_rate")
    if quant_data.get("move_index"):
        sources.append("us_move_index")
    if quant_data.get("us_etf"):
        sources.append("us_daily_etf")
    if quant_data.get("market_indices"):
        sources.append("market_index")

    # Check qualitative sources
    news = qual_data.get("news", {})
    if news.get("korea_market"):
        sources.append("naver_news")
    if news.get("global_macro"):
        sources.append("serper_global")
    if news.get("korea_foreign_view"):
        sources.append("serper_korea_view")

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
    if not market_regime.get("korea_market"):
        errors.append("Missing korea_market section")
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
    from kr.state import create_initial_state

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
        print("Running Market Regime Analysis...")
        print("-" * 50)

        result = await run_market_regime_analysis()

        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(main())
