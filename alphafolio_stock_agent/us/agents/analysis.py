"""
Analysis Agent module for US stocks - Main orchestration module
Integrates Market Regime + Stock Research + Quant Data -> Final Strategy Output

Design principles:
1. System-controlled structure (not LLM self-deciding)
2. Task-driven approach (not Role-driven)
3. Data/Narrative separation (V2 architecture)
4. Judgment + Validation loop
"""
import json
import re
import asyncio
from datetime import datetime, date
from typing import Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from us.config import settings
from us.state import AgentState
from us.db import queries
from us.data.collector import collect_stock_data, COLLECTION_PERIODS
from us.data.preprocessor import preprocess_all
from us.data.output_builder import (
    build_output_template,
    merge_llm_narrative_v2,
    validate_data_narrative_consistency,
    extract_system_triggers
)
from us.prompts.analysis import (
    ANALYSIS_SYSTEM_PROMPT,
    format_analysis_prompt,
    format_validation_retry_prompt
)
from us.utils.debug_saver import (
    save_collect_db,
    save_preprocessor,
    save_final_result
)


# =============================================================================
# Node: Collect Stock Data
# =============================================================================

async def collect_node(state: AgentState) -> dict:
    """
    LangGraph node: Collect stock data from database.

    This node:
    1. Fetches stock basic info
    2. Fetches all required time series data (prices, indicators, options, insider, etc.)
    3. Stores raw data in state

    Args:
        state: Current workflow state (must have symbol)

    Returns:
        Updated state with raw_data
    """
    try:
        symbol = state.get("symbol")
        target_date = state.get("target_date")

        if not symbol:
            return {"error": "No symbol provided for data collection"}

        # Convert target_date string to date object if needed
        if isinstance(target_date, str):
            end_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            end_date = target_date or date.today()

        # Collect all stock data
        collected_data = await collect_stock_data(symbol, end_date)

        if not collected_data:
            return {"error": f"No data found for symbol {symbol}"}

        # Get stock name from basic info
        stock_basic = collected_data.get("stock_basic", {})
        stock_name = stock_basic.get("stock_name", "")

        # Save intermediate result for debugging
        try:
            save_collect_db(symbol, collected_data)
        except Exception:
            pass

        # Store in raw_data
        raw_data = state.get("raw_data") or {}
        raw_data["collected_data"] = collected_data

        return {
            "raw_data": raw_data,
            "stock_name": stock_name
        }

    except Exception as e:
        return {"error": f"Data collection failed: {str(e)}"}


# =============================================================================
# Node: Preprocess Data
# =============================================================================

async def preprocess_node(state: AgentState) -> dict:
    """
    LangGraph node: Preprocess collected data.

    This node:
    1. Transforms raw data into interpreted signals
    2. Calculates technical indicators
    3. Analyzes options and insider data
    4. Prepares data for output builder

    Args:
        state: Current workflow state (must have raw_data.collected_data)

    Returns:
        Updated state with preprocessed_data in raw_data
    """
    try:
        raw_data = state.get("raw_data", {})
        collected_data = raw_data.get("collected_data")

        if not collected_data:
            return {"error": "No collected data available for preprocessing"}

        # Preprocess all data
        preprocessed_data = preprocess_all(collected_data)

        # Save intermediate result for debugging
        symbol = state.get("symbol", "UNKNOWN")
        try:
            save_preprocessor(symbol, preprocessed_data)
        except Exception:
            pass

        # Store in raw_data
        raw_data["preprocessed_data"] = preprocessed_data

        return {"raw_data": raw_data}

    except Exception as e:
        return {"error": f"Data preprocessing failed: {str(e)}"}


# =============================================================================
# Node: Run Parallel Agents (Market Regime + Stock Research)
# =============================================================================

async def parallel_agents_node(state: AgentState) -> dict:
    """
    LangGraph node: Run Market Regime and Stock Research agents in parallel.

    This node:
    1. Kicks off Market Regime Agent
    2. Kicks off Stock Research Agent
    3. Waits for both to complete
    4. Merges results into state

    Args:
        state: Current workflow state

    Returns:
        Updated state with market_regime and stock_research
    """
    from us.agents.market_regime import market_regime_node
    from us.agents.stock_research import stock_research_node

    try:
        # Run both agents in parallel
        market_task = market_regime_node(state)
        research_task = stock_research_node(state)

        market_result, research_result = await asyncio.gather(
            market_task,
            research_task,
            return_exceptions=True
        )

        result = {}

        # Handle market regime result
        if isinstance(market_result, dict):
            if market_result.get("market_regime"):
                result["market_regime"] = market_result["market_regime"]
            if market_result.get("raw_data"):
                # Merge raw_data
                raw_data = state.get("raw_data") or {}
                raw_data.update(market_result.get("raw_data", {}))
                result["raw_data"] = raw_data

        # Handle stock research result
        if isinstance(research_result, dict):
            if research_result.get("stock_research"):
                result["stock_research"] = research_result["stock_research"]
            if research_result.get("stock_name"):
                result["stock_name"] = research_result["stock_name"]

        return result

    except Exception as e:
        return {"error": f"Parallel agents failed: {str(e)}"}


# =============================================================================
# Node: Analyze (LLM Call)
# =============================================================================

async def analyze_node(state: AgentState) -> dict:
    """
    LangGraph node: Generate analysis narrative using LLM.

    This node:
    1. Builds output template with data sections filled
    2. Calls LLM to generate narrative sections
    3. Parses and validates LLM response

    Args:
        state: Current workflow state

    Returns:
        Updated state with analysis_output
    """
    try:
        raw_data = state.get("raw_data", {})
        preprocessed_data = raw_data.get("preprocessed_data", {})
        market_regime = state.get("market_regime")
        stock_research = state.get("stock_research")

        if not preprocessed_data:
            return {"error": "No preprocessed data available for analysis"}

        # Build output template with data sections filled
        quant_summary = preprocessed_data.get("quant_summary", {})
        system_triggers = extract_system_triggers(quant_summary)

        pre_filled_template = build_output_template(
            preprocessed_data=preprocessed_data,
            market_regime=market_regime,
            system_triggers=system_triggers
        )

        # Format analysis prompt
        analysis_date = state.get("target_date", date.today().isoformat())
        stock_info = preprocessed_data.get("stock_info", {})

        user_prompt = format_analysis_prompt(
            pre_filled_template=pre_filled_template,
            market_regime=market_regime,
            stock_research=stock_research,
            system_triggers=system_triggers,
            analysis_date=analysis_date,
            exchange=stock_info.get("exchange", "NYSE/NASDAQ"),
            stock_name=stock_info.get("stock_name", state.get("stock_name", "")),
            symbol=state.get("symbol", "")
        )

        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )

        # Call LLM
        messages = [
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await llm.ainvoke(messages)
        llm_output = parse_llm_response(response.content)

        if llm_output is None:
            return {"error": "Failed to parse analysis LLM response"}

        # Store results
        return {
            "analysis_output": {
                "template": pre_filled_template,
                "llm_output": llm_output
            }
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# =============================================================================
# Node: Validate Output
# =============================================================================

async def validate_node(state: AgentState) -> dict:
    """
    LangGraph node: Validate analysis output.

    This node:
    1. Checks required fields
    2. Validates scenario probabilities sum to 100%
    3. Checks for forbidden symbols
    4. Validates bearish strategy has defensive keywords

    Args:
        state: Current workflow state

    Returns:
        Updated state with validation_errors
    """
    try:
        analysis_output = state.get("analysis_output", {})
        llm_output = analysis_output.get("llm_output", {})

        if not llm_output:
            return {"validation_errors": ["No analysis output to validate"]}

        errors = []

        # Check final_grade (7단계 한글 - quant/us와 통일)
        valid_grades = [
            "강력 매수", "매수", "매수 고려", "중립", "매도 고려", "매도", "강력 매도"
        ]
        if llm_output.get("final_grade") not in valid_grades:
            errors.append(f"Invalid final_grade: {llm_output.get('final_grade')}. Must be one of {valid_grades}")

        # Check scenario probabilities sum to 100%
        scenarios = llm_output.get("scenarios_narrative", {})
        template = analysis_output.get("template", {})

        bullish_prob = template.get("scenarios", {}).get("bullish", {}).get("data", {}).get("probability", 0)
        sideways_prob = template.get("scenarios", {}).get("sideways", {}).get("data", {}).get("probability", 0)
        bearish_prob = template.get("scenarios", {}).get("bearish", {}).get("data", {}).get("probability", 0)

        total_prob = bullish_prob + sideways_prob + bearish_prob
        if total_prob != 100:
            errors.append(f"Scenario probabilities sum to {total_prob}%, not 100%")

        # Check for forbidden symbols in narrative
        # Note: '/' is allowed in financial terms (P/E, Put/Call, Risk/Reward, etc.)
        forbidden = ['+', '→', '|']
        narrative_text = json.dumps(llm_output, ensure_ascii=False)

        for symbol in forbidden:
            if symbol in narrative_text:
                errors.append(f"Forbidden symbol '{symbol}' found in narrative")

        # Check bearish strategy has defensive keywords
        bearish_narrative = scenarios.get("bearish", {})
        bearish_strategy = bearish_narrative.get("strategy", "")

        # English + Korean defensive keywords
        defensive_keywords = [
            # English
            "stop loss", "exit", "reduce", "hedge", "cash", "defensive", "protect",
            # Korean (한글)
            "손절", "손절매", "청산", "비중 축소", "축소", "헤지", "현금", "방어적", "방어", "보호"
        ]
        has_defensive = any(kw.lower() in bearish_strategy.lower() for kw in defensive_keywords)

        if bearish_strategy and not has_defensive:
            errors.append("Bearish strategy must contain defensive keywords (stop loss/exit/reduce/hedge/cash)")

        # Check triggers have at least 2 items
        for scenario_name in ["bullish", "sideways", "bearish"]:
            scenario = scenarios.get(scenario_name, {})
            triggers = scenario.get("triggers", [])
            if len(triggers) < 2:
                errors.append(f"{scenario_name} scenario must have at least 2 triggers")

        return {"validation_errors": errors}

    except Exception as e:
        return {"validation_errors": [f"Validation error: {str(e)}"]}


# =============================================================================
# Node: Retry Analysis (if validation failed)
# =============================================================================

async def retry_node(state: AgentState) -> dict:
    """
    LangGraph node: Retry analysis if validation failed.

    This node:
    1. Formats retry prompt with validation errors
    2. Calls LLM to correct the output
    3. Updates analysis_output

    Args:
        state: Current workflow state

    Returns:
        Updated state with corrected analysis_output
    """
    try:
        validation_errors = state.get("validation_errors", [])
        analysis_output = state.get("analysis_output", {})
        llm_output = analysis_output.get("llm_output", {})
        retry_count = state.get("retry_count", 0)

        if not validation_errors or retry_count >= 2:
            # No errors or max retries reached
            return {}

        # Format retry prompt
        retry_prompt = format_validation_retry_prompt(
            validation_errors=validation_errors,
            previous_output=llm_output
        )

        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )

        # Call LLM with retry prompt
        messages = [
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=retry_prompt)
        ]

        response = await llm.ainvoke(messages)
        new_llm_output = parse_llm_response(response.content)

        if new_llm_output is None:
            return {"retry_count": retry_count + 1}

        # Update analysis_output
        analysis_output["llm_output"] = new_llm_output

        return {
            "analysis_output": analysis_output,
            "retry_count": retry_count + 1
        }

    except Exception as e:
        return {"error": f"Retry failed: {str(e)}"}


# =============================================================================
# Node: Finalize Output
# =============================================================================

async def finalize_node(state: AgentState) -> dict:
    """
    LangGraph node: Finalize analysis output.

    This node:
    1. Merges template (data) with LLM output (narrative)
    2. Adds metadata
    3. Saves final result

    Args:
        state: Current workflow state

    Returns:
        Updated state with final_output
    """
    try:
        analysis_output = state.get("analysis_output", {})
        template = analysis_output.get("template", {})
        llm_output = analysis_output.get("llm_output", {})

        if not template or not llm_output:
            return {"error": "Missing template or LLM output for finalization"}

        # Merge template with LLM narrative
        final_output = merge_llm_narrative_v2(template, llm_output)

        # Add analysis date
        final_output["analysis_date"] = state.get("target_date", date.today().isoformat())

        # Update metadata
        final_output["metadata"]["retry_count"] = state.get("retry_count", 0)
        final_output["metadata"]["data_freshness"] = datetime.now().isoformat()

        # Validate data-narrative consistency (soft validation, just warnings)
        warnings = validate_data_narrative_consistency(final_output)
        if warnings:
            final_output["metadata"]["consistency_warnings"] = warnings

        # Save final result
        symbol = state.get("symbol", "UNKNOWN")
        target_date = state.get("target_date", date.today().isoformat())
        try:
            save_final_result(symbol, final_output, target_date)
        except Exception:
            pass

        return {"final_output": final_output}

    except Exception as e:
        return {"error": f"Finalization failed: {str(e)}"}


# =============================================================================
# Helper Functions
# =============================================================================

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


def should_retry(state: AgentState) -> str:
    """
    Conditional edge: Determine if retry is needed.

    Args:
        state: Current workflow state

    Returns:
        "retry" if validation failed and retries available, "finalize" otherwise
    """
    validation_errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)

    if validation_errors and retry_count < 2:
        return "retry"
    return "finalize"


# =============================================================================
# Combined Analysis Function (for non-LangGraph execution)
# =============================================================================

async def run_analysis(
    symbol: str,
    target_date: str = None,
    include_market_regime: bool = True,
    include_stock_research: bool = True,
    progress_callback: Callable[[str], None] = None
) -> dict:
    """
    Run complete analysis for a US stock.

    This function orchestrates the entire analysis pipeline without using
    LangGraph, useful for simpler execution or testing.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        target_date: Analysis date (YYYY-MM-DD), defaults to today
        include_market_regime: Whether to run Market Regime Agent
        include_stock_research: Whether to run Stock Research Agent
        progress_callback: Optional callback for progress updates

    Returns:
        Final analysis result dict
    """
    from us.state import create_initial_state
    from us.agents.market_regime import market_regime_node
    from us.agents.stock_research import stock_research_node

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Initialize state
    state = create_initial_state(symbol=symbol, target_date=target_date)

    # Step 1: Collect data
    log(f"Collecting data for {symbol}...")
    result = await collect_node(state)
    if result.get("error"):
        return {"error": result["error"]}
    state.update(result)

    # Step 2: Preprocess
    log("Preprocessing data...")
    result = await preprocess_node(state)
    if result.get("error"):
        return {"error": result["error"]}
    state.update(result)

    # Step 3: Run parallel agents
    log("Running market regime and stock research agents...")
    tasks = []

    if include_market_regime:
        tasks.append(("market_regime", market_regime_node(state)))
    if include_stock_research:
        tasks.append(("stock_research", stock_research_node(state)))

    if tasks:
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        for i, (name, _) in enumerate(tasks):
            if isinstance(results[i], dict):
                state.update(results[i])

    # Step 4: Analyze
    log("Generating analysis narrative...")
    result = await analyze_node(state)
    if result.get("error"):
        return {"error": result["error"]}
    state.update(result)

    # Step 5: Validate
    result = await validate_node(state)
    state.update(result)

    # Step 6: Retry if needed
    retry_count = 0
    while state.get("validation_errors") and retry_count < 2:
        log(f"Retrying analysis (attempt {retry_count + 1})...")
        state["retry_count"] = retry_count
        result = await retry_node(state)
        state.update(result)

        result = await validate_node(state)
        state.update(result)
        retry_count += 1

    # Step 7: Finalize
    log("Finalizing output...")
    result = await finalize_node(state)
    if result.get("error"):
        return {"error": result["error"]}

    return result.get("final_output", {})


# =============================================================================
# CLI Test Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        # Default to Apple if no symbol provided
        symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

        print(f"Running US Stock Analysis for {symbol}...")
        print("=" * 60)

        def progress(msg):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

        result = await run_analysis(
            symbol=symbol,
            progress_callback=progress
        )

        print("=" * 60)

        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(main())
