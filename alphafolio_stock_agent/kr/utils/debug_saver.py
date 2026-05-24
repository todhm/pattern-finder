"""
Debug result saver utility
Saves intermediate results for debugging and improvement analysis
"""
import json
import os
from datetime import datetime, date
from pathlib import Path
from decimal import Decimal


# Result save directory
RESULT_SUB_DIR = Path(__file__).parent.parent / "result" / "sub"


def _json_serializer(obj):
    """Custom JSON serializer for special types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, '__dict__'):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_intermediate_result(
    result_type: str,
    symbol: str,
    data: dict,
    execution_date: datetime = None
) -> str:
    """
    Save intermediate result to JSON file.

    Args:
        result_type: Type of result (e.g., "collect_DB", "Market_Regime_Agent", "Stock_Research_Agent")
        symbol: Stock symbol (e.g., "005930")
        data: Data to save
        execution_date: Execution date (defaults to now)

    Returns:
        Saved file path
    """
    # Ensure directory exists
    RESULT_SUB_DIR.mkdir(parents=True, exist_ok=True)

    # Format date
    if execution_date is None:
        execution_date = datetime.now()
    date_str = execution_date.strftime("%Y%m%d")

    # Build filename
    filename = f"{result_type}_{symbol}_{date_str}.json"
    filepath = RESULT_SUB_DIR / filename

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_serializer)

    return str(filepath)


def save_collect_db(symbol: str, raw_data: dict) -> str:
    """Save collect_data DB result."""
    return save_intermediate_result("collect_DB", symbol, raw_data)


def save_preprocessor(symbol: str, preprocessed_data: dict) -> str:
    """Save preprocessor result."""
    return save_intermediate_result("preprocessor", symbol, preprocessed_data)


def save_market_regime(symbol: str, market_regime: dict) -> str:
    """Save Market Regime Agent result."""
    return save_intermediate_result("Market_Regime_Agent", symbol, market_regime)


def save_stock_research(symbol: str, stock_research: dict) -> str:
    """Save Stock Research Agent result."""
    return save_intermediate_result("Stock_Research_Agent", symbol, stock_research)


# =============================================================================
# DEBUG LOGGING FUNCTIONS - FOR ROLLBACK: Delete this entire section
# =============================================================================

def save_system_triggers(symbol: str, system_triggers: dict) -> str:
    """Save system_triggers for debugging."""
    return save_intermediate_result("DEBUG_system_triggers", symbol, system_triggers)


def save_llm_raw_response(symbol: str, raw_response: str) -> str:
    """Save LLM raw response text for debugging."""
    # Ensure directory exists
    RESULT_SUB_DIR.mkdir(parents=True, exist_ok=True)

    # Format date
    date_str = datetime.now().strftime("%Y%m%d")

    # Build filename
    filename = f"DEBUG_llm_raw_response_{symbol}_{date_str}.txt"
    filepath = RESULT_SUB_DIR / filename

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(raw_response)

    return str(filepath)


def save_llm_parsed_narrative(symbol: str, parsed_narrative: dict) -> str:
    """Save parsed LLM narrative for debugging."""
    return save_intermediate_result("DEBUG_llm_parsed_narrative", symbol, parsed_narrative or {})


def save_merged_strategy(symbol: str, merged_strategy: dict) -> str:
    """Save merged strategy (before validation) for debugging."""
    return save_intermediate_result("DEBUG_merged_strategy", symbol, merged_strategy)


# =============================================================================
# END DEBUG LOGGING FUNCTIONS
# =============================================================================
