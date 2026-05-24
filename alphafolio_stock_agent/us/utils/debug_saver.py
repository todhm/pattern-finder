"""
Debug result saver utility for US stocks
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
        symbol: Stock ticker (e.g., "AAPL")
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


def save_final_result(symbol: str, strategy: dict, target_date: str = None) -> str:
    """
    Save final strategy result to main result folder.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        strategy: Final strategy output
        target_date: Analysis target date (YYYY-MM-DD format)

    Returns:
        Saved file path
    """
    result_dir = Path(__file__).parent.parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    if target_date is None:
        target_date = date.today().isoformat()

    # Format: AAPL_2025_12_19.json
    date_formatted = target_date.replace("-", "_")
    filename = f"{symbol}_{date_formatted}.json"
    filepath = result_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2, default=_json_serializer)

    return str(filepath)
