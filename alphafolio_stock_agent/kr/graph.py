"""
LangGraph workflow definition
Defines the state machine for the analysis workflow

Workflow stages (FSM from 에이전트 작업 계획.md):
INIT -> RETRIEVE (collect + parallel[market_regime, stock_research]) -> ANALYZE -> VALIDATE -> FINALIZE -> END

v2.0: Market Regime Agent와 Stock Research Agent 병렬 실행
"""
import asyncio
from langgraph.graph import StateGraph, END

from kr.state import AgentState
from kr.agents.analysis import (
    collect_data,
    preprocess_data,
    analyze,
    validate,
    finalize,
    should_retry
)
from kr.agents.market_regime import (
    collect_market_regime_data,
    analyze_market_regime
)
from kr.agents.stock_research import (
    stock_research_node
)


# =============================================================================
# Parallel Execution Node
# =============================================================================

async def parallel_agent_analysis(state: AgentState) -> dict:
    """
    Execute Market Regime Agent and Stock Research Agent in parallel.

    This node runs both agents concurrently to improve performance:
    - Market Regime Agent: Analyzes global/Korean market conditions
    - Stock Research Agent: Collects and analyzes qualitative data for the stock

    Args:
        state: Current workflow state

    Returns:
        Updated state with both market_regime and stock_research results
    """
    # Create tasks for parallel execution
    regime_task = run_regime_analysis(state)
    research_task = stock_research_node(state)

    # Run in parallel
    regime_result, research_result = await asyncio.gather(
        regime_task,
        research_task,
        return_exceptions=True
    )

    # Process results
    result = {}

    # Handle Market Regime result
    if isinstance(regime_result, Exception):
        result["market_regime"] = None
        # Don't set error - continue with degraded mode
    elif isinstance(regime_result, dict):
        if regime_result.get("error"):
            result["market_regime"] = None
        else:
            result["market_regime"] = regime_result.get("market_regime")

    # Handle Stock Research result
    if isinstance(research_result, Exception):
        result["stock_research"] = None
    elif isinstance(research_result, dict):
        if research_result.get("error"):
            result["stock_research"] = None
        else:
            result["stock_research"] = research_result.get("stock_research")
            # Update raw_data if research collected new data
            if research_result.get("raw_data"):
                raw_data = state.get("raw_data") or {}
                raw_data.update(research_result.get("raw_data", {}))
                result["raw_data"] = raw_data
            # Update stock_name if research found it
            if research_result.get("stock_name"):
                result["stock_name"] = research_result["stock_name"]

    return result


async def run_regime_analysis(state: AgentState) -> dict:
    """
    Run the full market regime analysis pipeline.

    Combines collect_market_regime_data and analyze_market_regime into one task.

    Args:
        state: Current workflow state

    Returns:
        Updated state with market_regime result
    """
    # Step 1: Collect regime data
    collect_result = await collect_market_regime_data(state)

    if collect_result.get("error"):
        return collect_result

    # Update state with collected data
    updated_state = {**state}
    if collect_result.get("raw_data"):
        updated_state["raw_data"] = collect_result["raw_data"]

    # Step 2: Analyze regime
    analyze_result = await analyze_market_regime(updated_state)

    return analyze_result


def create_graph():
    """
    Create and compile the LangGraph workflow.

    Workflow (v2.0 with parallel agents):
        INIT -> collect -> parallel_agents -> preprocess -> analyze -> validate
                                                                          |
                                                            PASS -> finalize -> END
                                                            FAIL -> analyze (retry, max 2)
                                                            MAX_RETRY -> END (with error)

    Parallel agents execute concurrently:
    - Market Regime Agent: Global market context
    - Stock Research Agent: Qualitative stock analysis

    Returns:
        Compiled workflow graph
    """
    # Create state graph
    workflow = StateGraph(AgentState)

    # ==========================================================================
    # Add nodes
    # ==========================================================================
    workflow.add_node("collect", collect_data)
    workflow.add_node("parallel_agents", parallel_agent_analysis)
    workflow.add_node("preprocess", preprocess_data)
    workflow.add_node("analyze", analyze)
    workflow.add_node("validate", validate)
    workflow.add_node("finalize", finalize)

    # ==========================================================================
    # Define edges
    # ==========================================================================

    # Entry point
    workflow.set_entry_point("collect")

    # collect -> parallel_agents
    workflow.add_conditional_edges(
        "collect",
        lambda state: "end" if state.get("error") else "parallel_agents",
        {
            "parallel_agents": "parallel_agents",
            "end": END
        }
    )

    # parallel_agents -> preprocess
    # Note: Even if agents fail, we continue with preprocess (degraded mode)
    workflow.add_conditional_edges(
        "parallel_agents",
        lambda state: "preprocess",  # Always continue to preprocess
        {
            "preprocess": "preprocess"
        }
    )

    workflow.add_conditional_edges(
        "preprocess",
        lambda state: "end" if state.get("error") else "analyze",
        {
            "analyze": "analyze",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "analyze",
        lambda state: "end" if state.get("error") else "validate",
        {
            "validate": "validate",
            "end": END
        }
    )

    # Validation conditional edge: retry, finalize, or error
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "analyze": "analyze",   # Retry analysis
            "finalize": "finalize", # Validation passed
            "error": END            # Max retries exceeded
        }
    )

    # Finalize to end
    workflow.add_edge("finalize", END)

    # ==========================================================================
    # Compile and return
    # ==========================================================================
    return workflow.compile()


def create_simple_graph():
    """
    Create a simplified workflow for testing (no validation retry, no parallel agents).

    Workflow:
        collect -> preprocess -> analyze -> finalize -> END

    Returns:
        Compiled workflow graph
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("collect", collect_data)
    workflow.add_node("preprocess", preprocess_data)
    workflow.add_node("analyze", analyze)
    workflow.add_node("finalize", finalize)

    # Sequential edges
    workflow.set_entry_point("collect")
    workflow.add_edge("collect", "preprocess")
    workflow.add_edge("preprocess", "analyze")
    workflow.add_edge("analyze", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def create_full_graph():
    """
    Alias for create_graph() - the full workflow with parallel agents.

    Returns:
        Compiled workflow graph
    """
    return create_graph()
