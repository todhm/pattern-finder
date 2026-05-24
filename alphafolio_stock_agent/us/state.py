"""
LangGraph state definition
Defines the state schema for the US stock workflow
"""
from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    """
    State schema for the LangGraph workflow.

    This state is passed between nodes in the workflow:
    INIT -> COLLECT -> PREPROCESS -> PARALLEL_AGENTS -> ANALYZE -> VALIDATE -> FINALIZE -> END
    """

    # ==========================================================================
    # Input fields
    # ==========================================================================
    symbol: str                     # Stock ticker (e.g., "AAPL")
    stock_name: str                 # Stock name (e.g., "Apple Inc.")
    target_date: str                # Analysis reference date (YYYY-MM-DD)

    # ==========================================================================
    # Data collection fields
    # ==========================================================================
    raw_data: Optional[dict]        # Raw data from collector.collect_stock_data()

    # ==========================================================================
    # Preprocessing fields
    # ==========================================================================
    preprocessed_data: Optional[dict]  # Processed data from preprocessor.preprocess_all()

    # ==========================================================================
    # Agent output fields
    # ==========================================================================
    market_regime: Optional[dict]      # Market Regime Agent result (MarketRegimeOutput)
    stock_research: Optional[dict]     # Stock Research Agent result (StockResearchOutput)

    # ==========================================================================
    # Analysis output fields
    # ==========================================================================
    analysis_output: Optional[dict]    # Analysis output (template + llm_output)

    # ==========================================================================
    # Final output fields
    # ==========================================================================
    final_output: Optional[dict]       # Final merged output (template + narrative)
    strategy: Optional[dict]           # Final strategy output (StrategyOutput)

    # ==========================================================================
    # Workflow control fields
    # ==========================================================================
    validation_errors: list[str]       # List of validation errors from last attempt
    retry_count: int                   # Number of validation retries (max 2)
    error: Optional[str]               # Error message if workflow failed
    start_time: float                  # Workflow start time for execution timing


def create_initial_state(symbol: str, target_date: str = None) -> AgentState:
    """
    Create initial state for workflow execution.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        target_date: Analysis reference date (YYYY-MM-DD), defaults to today

    Returns:
        Initial AgentState with default values
    """
    import time
    from datetime import date

    return AgentState(
        # Input
        symbol=symbol,
        stock_name="",  # Will be populated during collection
        target_date=target_date or date.today().isoformat(),

        # Data
        raw_data=None,
        preprocessed_data=None,

        # Agent outputs
        market_regime=None,
        stock_research=None,

        # Analysis output
        analysis_output=None,

        # Final output
        final_output=None,
        strategy=None,

        # Workflow control
        validation_errors=[],
        retry_count=0,
        error=None,
        start_time=time.time()
    )


def get_execution_time(state: AgentState) -> float:
    """
    Calculate execution time from start.

    Args:
        state: Current agent state

    Returns:
        Execution time in seconds
    """
    import time
    return round(time.time() - state["start_time"], 2)


def should_retry(state: AgentState) -> bool:
    """
    Check if workflow should retry after validation failure.

    Args:
        state: Current agent state

    Returns:
        True if retry is allowed (retry_count < 2 and has validation errors)
    """
    return bool(state["validation_errors"]) and state["retry_count"] < 2


def is_workflow_failed(state: AgentState) -> bool:
    """
    Check if workflow has failed.

    Args:
        state: Current agent state

    Returns:
        True if workflow has error or exceeded retry limit
    """
    if state["error"]:
        return True
    if state["validation_errors"] and state["retry_count"] >= 2:
        return True
    return False


def get_state_summary(state: AgentState) -> dict:
    """
    Get summary of current state for logging/debugging.

    Args:
        state: Current agent state

    Returns:
        Summary dict with key state information
    """
    return {
        "symbol": state["symbol"],
        "stock_name": state["stock_name"],
        "target_date": state["target_date"],
        "has_raw_data": state["raw_data"] is not None,
        "has_preprocessed_data": state["preprocessed_data"] is not None,
        "has_market_regime": state["market_regime"] is not None,
        "has_stock_research": state["stock_research"] is not None,
        "has_strategy": state["strategy"] is not None,
        "validation_errors_count": len(state["validation_errors"]),
        "retry_count": state["retry_count"],
        "has_error": state["error"] is not None,
        "execution_time": get_execution_time(state)
    }
