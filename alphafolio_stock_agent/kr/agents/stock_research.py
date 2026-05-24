"""
Stock Research Agent module
Collects and analyzes qualitative data for stock research

Design principles (from 에이전트 작업 계획.md):
1. System-controlled structure (not LLM self-deciding)
2. Task-driven approach (not Role-driven)
3. LLM extracts only 3 items: earnings_outlook, risk_issues, sector_momentum
4. INPUT_DATA only - no fabrication allowed
5. Source mapping required for all summaries
6. Context addition only - does not modify quant results
"""
import json
import re
from datetime import datetime, date
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from kr.config import settings
from kr.state import AgentState
from kr.schemas import StockResearchOutput
from kr.db import queries
from kr.data.research_collector import (
    collect_research_data,
    determine_stock_size
)
from kr.data.research_preprocessor import (
    preprocess_research_data,
    format_for_llm
)
from kr.prompts.stock_research import (
    STOCK_RESEARCH_SYSTEM_PROMPT,
    format_stock_research_prompt,
    format_research_retry_prompt,
    validate_research_output
)
from kr.utils.debug_saver import save_stock_research


# =============================================================================
# Data Collection for Stock Research
# =============================================================================

async def collect_stock_research_data(
    symbol: str,
    stock_name: str,
    sector: str,
    market_cap: float
) -> dict:
    """
    Collect all qualitative research data for a stock.

    Wrapper function for research_collector.collect_research_data()

    Args:
        symbol: Stock code (e.g., "005930")
        stock_name: Stock name (e.g., "삼성전자")
        sector: Sector from kr_stock_detail.theme
        market_cap: Market capitalization in KRW

    Returns:
        Collected data dict from research_collector
    """
    return await collect_research_data(
        symbol=symbol,
        stock_name=stock_name,
        sector=sector,
        market_cap=market_cap
    )


# =============================================================================
# Node: Collect Stock Research Data
# =============================================================================

async def collect_stock_research_node(state: AgentState) -> dict:
    """
    LangGraph node: Collect stock research data.

    This node:
    1. Gets stock detail info (theme, market_cap) from DB
    2. Collects research data (reports, disclosures, news)
    3. Preprocesses data for LLM consumption

    Args:
        state: Current workflow state (must have symbol, stock_name)

    Returns:
        Updated state with research_raw_data in raw_data
    """
    try:
        symbol = state.get("symbol")
        stock_name = state.get("stock_name", "")

        if not symbol:
            return {
                "error": "No symbol provided for stock research"
            }

        # Get stock detail for sector and market_cap
        stock_detail = await queries.get_stock_detail(symbol)

        if not stock_detail:
            # Fallback: try to get basic info
            stock_basic = await queries.get_stock_basic(symbol)
            sector = "Unknown"
            market_cap = 0
            if stock_basic:
                stock_name = stock_name or stock_basic.get("name", "Unknown")
        else:
            sector = stock_detail.get("theme", "Unknown")  # DB column is 'theme'
            market_cap = float(stock_detail.get("market_cap", 0) or 0)
            stock_name = stock_name or stock_detail.get("name", "Unknown")

        # Collect research data
        collected_data = await collect_stock_research_data(
            symbol=symbol,
            stock_name=stock_name,
            sector=sector,
            market_cap=market_cap
        )

        # Preprocess for LLM
        preprocessed = preprocess_research_data(collected_data)

        # Store in raw_data
        raw_data = state.get("raw_data") or {}
        raw_data["research_data"] = {
            "collected": collected_data,
            "preprocessed": preprocessed,
            "stock_info": {
                "symbol": symbol,
                "stock_name": stock_name,
                "sector": sector,
                "market_cap": market_cap,
                "stock_size": determine_stock_size(market_cap)
            }
        }

        return {
            "raw_data": raw_data,
            "stock_name": stock_name  # Update stock_name if not set
        }

    except Exception as e:
        return {
            "error": f"Stock research data collection failed: {str(e)}"
        }


# =============================================================================
# Node: Analyze Stock Research
# =============================================================================

async def analyze_stock_research(state: AgentState) -> dict:
    """
    LangGraph node: Analyze stock research using LLM extraction.

    This node:
    1. Formats preprocessed data for LLM
    2. Calls LLM to extract earnings_outlook, risk_issues, sector_momentum
    3. Validates output and retries if needed (max 2 retries)

    Args:
        state: Current workflow state (must have research_data in raw_data)

    Returns:
        Updated state with stock_research result
    """
    try:
        raw_data = state.get("raw_data", {})
        research_data = raw_data.get("research_data", {})

        if not research_data:
            return {
                "error": "No research data available for analysis"
            }

        preprocessed = research_data.get("preprocessed", {})
        stock_info = research_data.get("stock_info", {})

        if not preprocessed:
            return {
                "error": "Research data preprocessing failed"
            }

        # Get data counts
        metadata = preprocessed.get("metadata", {})
        processed_counts = metadata.get("processed_counts", {})

        # Check if we have any data to analyze
        total_data = sum(processed_counts.values())
        if total_data == 0:
            # Return minimal result if no data
            return {
                "stock_research": create_empty_research_result(
                    stock_info=stock_info,
                    reason="데이터 없음"
                )
            }

        # Format text for LLM
        formatted_text = preprocessed.get("formatted_text", "데이터 없음")

        # Build user prompt
        user_prompt = format_stock_research_prompt(
            symbol=stock_info.get("symbol", ""),
            stock_name=stock_info.get("stock_name", ""),
            sector=stock_info.get("sector", "Unknown"),
            stock_size=stock_info.get("stock_size", "medium"),
            analysis_date=state.get("target_date", date.today().isoformat()),
            formatted_text=formatted_text,
            report_count=processed_counts.get("research_reports", 0),
            news_count=processed_counts.get("stock_news", 0) + processed_counts.get("sector_news", 0),
            disclosure_count=processed_counts.get("disclosures", 0)
        )

        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )

        # First LLM call
        messages = [
            SystemMessage(content=STOCK_RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await llm.ainvoke(messages)
        result = parse_llm_response(response.content)

        if result is None:
            return {
                "error": "Failed to parse stock research LLM response"
            }

        # Validate output
        validation_errors = validate_research_output(result)

        # Retry if validation failed (max 2 retries)
        retry_count = 0
        max_retries = 2

        while validation_errors and retry_count < max_retries:
            retry_count += 1

            retry_prompt = format_research_retry_prompt(
                validation_errors=validation_errors,
                previous_output=result
            )

            messages = [
                SystemMessage(content=STOCK_RESEARCH_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
                HumanMessage(content=retry_prompt)
            ]

            response = await llm.ainvoke(messages)
            result = parse_llm_response(response.content)

            if result is None:
                continue

            validation_errors = validate_research_output(result)

        # If still has errors after retries, log but continue
        if validation_errors:
            result["_validation_warnings"] = validation_errors

        # Enrich result with metadata
        result["symbol"] = stock_info.get("symbol", "")
        result["stock_name"] = stock_info.get("stock_name", "")
        result["analysis_date"] = state.get("target_date", date.today().isoformat())
        result["sector"] = stock_info.get("sector", "Unknown")

        # Add metadata
        result["metadata"] = {
            "data_collection": {
                "research_reports": processed_counts.get("research_reports", 0),
                "news_articles": processed_counts.get("stock_news", 0) + processed_counts.get("sector_news", 0),
                "disclosures": processed_counts.get("disclosures", 0)
            },
            "excluded": {
                "sns": "Tier 3 excluded",
                "community": "Tier 3 excluded"
            },
            "stock_size": stock_info.get("stock_size", "medium"),
            "collection_date": date.today().isoformat(),
            "retry_count": retry_count
        }

        return {
            "stock_research": result
        }

    except Exception as e:
        return {
            "error": f"Stock research analysis failed: {str(e)}"
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


def create_empty_research_result(stock_info: dict, reason: str) -> dict:
    """
    Create empty research result when no data is available.

    Args:
        stock_info: Stock information dict
        reason: Reason for empty result

    Returns:
        Minimal valid research result
    """
    return {
        "symbol": stock_info.get("symbol", ""),
        "stock_name": stock_info.get("stock_name", ""),
        "analysis_date": date.today().isoformat(),
        "sector": stock_info.get("sector", "Unknown"),
        "overall_sentiment": "neutral",
        "confidence": 0.3,
        "earnings_outlook": {
            "direction": "neutral",
            "summary": reason,
            "key_points": [],
            "sources": []
        },
        "risk_issues": [],
        "sector_momentum": {
            "sector": stock_info.get("sector", "Unknown"),
            "direction": "neutral",
            "summary": reason,
            "related_news_count": 0,
            "sources": []
        },
        "disclosure_summary": {
            "recent_count": 0,
            "significant_items": []
        },
        "metadata": {
            "data_collection": {
                "research_reports": 0,
                "news_articles": 0,
                "disclosures": 0
            },
            "excluded": {
                "sns": "Tier 3 excluded",
                "community": "Tier 3 excluded"
            },
            "stock_size": stock_info.get("stock_size", "medium"),
            "collection_date": date.today().isoformat(),
            "empty_reason": reason
        }
    }


# =============================================================================
# Validation
# =============================================================================

async def validate_stock_research(state: AgentState) -> dict:
    """
    Validate stock research output.

    Args:
        state: Current workflow state

    Returns:
        Updated state with validation errors if any
    """
    stock_research = state.get("stock_research")

    if not stock_research:
        return {
            "validation_errors": ["No stock research output to validate"]
        }

    errors = validate_research_output(stock_research)

    return {
        "validation_errors": errors
    }


# =============================================================================
# Combined Node: Collect and Analyze
# =============================================================================

async def stock_research_node(state: AgentState) -> dict:
    """
    Combined LangGraph node for stock research.

    This single node performs:
    1. Data collection (research reports, disclosures, news)
    2. Preprocessing
    3. LLM extraction
    4. Validation

    This is useful for parallel execution with market_regime_node.

    Args:
        state: Current workflow state

    Returns:
        Updated state with stock_research result
    """
    # Step 1: Collect data
    collect_result = await collect_stock_research_node(state)

    if collect_result.get("error"):
        return collect_result

    # Merge collection result into state
    updated_state = {**state}
    if collect_result.get("raw_data"):
        updated_state["raw_data"] = collect_result["raw_data"]
    if collect_result.get("stock_name"):
        updated_state["stock_name"] = collect_result["stock_name"]

    # Step 2: Analyze
    analyze_result = await analyze_stock_research(updated_state)

    # Combine results
    result = {}
    if collect_result.get("raw_data"):
        result["raw_data"] = collect_result["raw_data"]
    if collect_result.get("stock_name"):
        result["stock_name"] = collect_result["stock_name"]
    if analyze_result.get("stock_research"):
        result["stock_research"] = analyze_result["stock_research"]
        # Save intermediate result for debugging
        symbol = state.get("symbol", "UNKNOWN")
        try:
            save_stock_research(symbol, analyze_result["stock_research"])
        except Exception:
            pass  # Don't fail if debug save fails
    if analyze_result.get("error"):
        result["error"] = analyze_result["error"]

    return result


# =============================================================================
# Standalone Execution
# =============================================================================

async def run_stock_research(
    symbol: str,
    target_date: str = None
) -> dict:
    """
    Run stock research analysis standalone (without full workflow).

    This can be used for testing or to get stock research independently.

    Args:
        symbol: Stock code (e.g., "005930")
        target_date: Analysis date (YYYY-MM-DD), defaults to today

    Returns:
        Stock research result dict
    """
    from kr.state import create_initial_state

    # Create minimal state
    state = create_initial_state(symbol=symbol, target_date=target_date)

    # Run combined node
    result = await stock_research_node(state)

    if result.get("error"):
        return {"error": result["error"]}

    return result.get("stock_research", {})


# =============================================================================
# CLI Test Entry Point
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # Test with Samsung Electronics
        test_symbol = "005930"

        print(f"Running Stock Research Analysis for {test_symbol}...")
        print("-" * 50)

        result = await run_stock_research(test_symbol)

        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(main())
