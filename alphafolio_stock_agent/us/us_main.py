"""
US Stock Investment Strategy Generator
CLI entry point for the multi-agent analysis workflow

Usage:
1. Run this file directly (interactive mode)
2. Run with arguments: python us_main.py --symbol AAPL --date 2025-12-19
"""
import asyncio
import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from us.config import settings
from us.db.connection import Database
from us.db.queries import save_strategy
from us.graph import create_graph
from us.state import create_initial_state


def save_result_to_file(symbol: str, strategy: dict, target_date: str) -> str:
    """
    Save final strategy result to JSON file.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        strategy: Final strategy dict
        target_date: Analysis date (YYYY-MM-DD)

    Returns:
        Saved file path
    """
    result_dir = Path(__file__).parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Format: symbol_YYYY_MM_DD.json
    date_str = target_date.replace("-", "_")
    filename = f"{symbol}_{date_str}.json"
    filepath = result_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2)

    return str(filepath)


async def run(symbol: str, target_date: str = None, save_to_db: bool = True) -> dict:
    """
    Run the analysis workflow for a US stock.

    Args:
        symbol: Stock ticker (e.g., "AAPL")
        target_date: Analysis reference date (YYYY-MM-DD), defaults to today
        save_to_db: Whether to save result to PostgreSQL

    Returns:
        Final strategy dict or error dict
    """
    # Connect to database
    await Database.connect()

    try:
        # Create initial state
        initial_state = create_initial_state(
            symbol=symbol,
            target_date=target_date
        )

        # Create and run workflow
        graph = create_graph()
        result = await graph.ainvoke(initial_state)

        # Check for errors
        if result.get("error"):
            print(f"[ERROR] {result['error']}")
            return {"error": result["error"]}

        # Check for validation failures
        if result.get("validation_errors"):
            print(f"[ERROR] Validation failed after max retries:")
            for err in result["validation_errors"]:
                print(f"  - {err}")
            return {
                "error": "Validation failed",
                "validation_errors": result["validation_errors"]
            }

        # Success
        strategy = result.get("final_output")
        if strategy:
            stock_name = result.get('stock_name', symbol)
            exec_time = strategy.get('metadata', {}).get('execution_time_sec', 0) or 0

            print(f"[SUCCESS] Analysis completed for {stock_name}")
            if exec_time:
                print(f"[INFO] Execution time: {exec_time:.2f}s")

            # Save to PostgreSQL
            if save_to_db:
                try:
                    analysis_date = date.fromisoformat(target_date) if target_date else None
                    await save_strategy(symbol, strategy, analysis_date)
                    print(f"[INFO] Result saved to PostgreSQL (us_stock_grade.strategy, date={analysis_date or 'latest'})")
                except Exception as e:
                    print(f"[WARNING] Failed to save to DB: {e}")

            # Save to local JSON file
            try:
                analysis_date_str = target_date if target_date else date.today().isoformat()
                filepath = save_result_to_file(symbol, strategy, analysis_date_str)
                print(f"[INFO] Result saved to {filepath}")
            except Exception as e:
                print(f"[WARNING] Failed to save to file: {e}")

            return strategy
        else:
            print("[ERROR] No strategy generated")
            return {"error": "No strategy generated"}

    finally:
        # Disconnect from database
        await Database.disconnect()


def run_single_analysis():
    """Run single stock analysis (Option 1)"""
    # 1. Stock ticker input
    symbol = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    if not symbol:
        print("[ERROR] Stock ticker is required.")
        return

    # 2. Date input
    date_input = input("Enter analysis date (YYYY-MM-DD, empty for today): ").strip()

    # Date validation
    if date_input:
        try:
            date.fromisoformat(date_input)
            target_date = date_input
        except ValueError:
            print(f"[ERROR] Invalid date format: {date_input}. Use YYYY-MM-DD format.")
            return
    else:
        target_date = get_auto_target_date()

    # Print settings
    print()
    print("-" * 60)
    print(f"[INFO] Ticker: {symbol}")
    print(f"[INFO] Analysis Date: {target_date or date.today().isoformat()}")
    print(f"[INFO] LLM Model: {settings.OPENAI_MODEL}")
    print("-" * 60)
    print()
    print("[INFO] Starting analysis... (may take 2-4 minutes)")
    print()

    # Run workflow
    result = asyncio.run(run(symbol, target_date, save_to_db=True))

    # Output result
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("=" * 60)
    print("[INFO] Analysis complete. Result saved to PostgreSQL.")
    print("=" * 60)


def get_auto_target_date() -> str:
    """
    Get target date automatically based on current time and weekday.

    Rules (based on US market data availability in KST):
    - Monday (all day): Friday (-3 days)
    - Tuesday 00:00~06:29: Friday (-4 days)
    - Tuesday 06:30~23:59: Monday (-1 day)
    - Wed~Sat 00:00~06:29: 2 days ago (-2 days)
    - Wed~Sat 06:30~23:59: Yesterday (-1 day)
    - Sunday (all day): Friday (-2 days)

    Returns:
        Target date string in YYYY-MM-DD format
    """
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 1=Tue, ..., 6=Sun
    current_time = now.hour * 60 + now.minute  # Convert to minutes
    cutoff_time = 6 * 60 + 30  # 06:30 = 390 minutes

    if weekday == 0:  # Monday -> Friday (-3)
        return (now.date() - timedelta(days=3)).isoformat()
    elif weekday == 1:  # Tuesday
        if current_time < cutoff_time:  # 00:00~06:29 -> Friday (-4)
            return (now.date() - timedelta(days=4)).isoformat()
        else:  # 06:30~23:59 -> Monday (-1)
            return (now.date() - timedelta(days=1)).isoformat()
    elif weekday in [2, 3, 4, 5]:  # Wed~Sat
        if current_time < cutoff_time:  # 00:00~06:29 -> 2 days ago (-2)
            return (now.date() - timedelta(days=2)).isoformat()
        else:  # 06:30~23:59 -> Yesterday (-1)
            return (now.date() - timedelta(days=1)).isoformat()
    else:  # Sunday -> Friday (-2)
        return (now.date() - timedelta(days=2)).isoformat()


def run_auto_analysis():
    """Run stock analysis with auto date detection (Option 2)"""
    # 1. Stock ticker input
    symbol = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    if not symbol:
        print("[ERROR] Stock ticker is required.")
        return

    # 2. Auto date detection
    target_date = get_auto_target_date()

    # Print settings
    print()
    print("-" * 60)
    print(f"[INFO] Ticker: {symbol}")
    print(f"[INFO] Analysis Date: {target_date} (auto-detected)")
    print(f"[INFO] LLM Model: {settings.OPENAI_MODEL}")
    print("-" * 60)
    print()
    print("[INFO] Starting analysis... (may take 2-4 minutes)")
    print()

    # Run workflow
    result = asyncio.run(run(symbol, target_date, save_to_db=True))

    # Output result
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("=" * 60)
    print("[INFO] Analysis complete. Result saved to PostgreSQL.")
    print("=" * 60)


def interactive_main():
    """PyCharm/IDE interactive entry point"""
    print("=" * 60)
    print("US Stock Investment Strategy Generator")
    print("=" * 60)
    print()

    run_single_analysis()


def cli_main():
    """Command line entry point with arguments"""
    parser = argparse.ArgumentParser(
        description="US Stock Investment Strategy Generator"
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Stock ticker (e.g., AAPL)"
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Analysis reference date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (JSON), prints to stdout if not specified"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save result to PostgreSQL"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Uppercase ticker
    symbol = args.symbol.upper()

    # Validate date format if provided
    if args.date:
        try:
            date.fromisoformat(args.date)
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}. Use YYYY-MM-DD.")
            return

    # Print settings if verbose
    if args.verbose:
        print(f"[INFO] Ticker: {symbol}")
        print(f"[INFO] Date: {args.date or date.today().isoformat()}")
        print(f"[INFO] Model: {settings.OPENAI_MODEL}")

    # Run workflow
    save_to_db = not args.no_save
    result = asyncio.run(run(symbol, args.date, save_to_db=save_to_db))

    # Output result
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Result saved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    """
    Main entry point
    - Run directly in PyCharm: interactive mode
    - Run with arguments in CLI: CLI mode
    """
    import sys

    # Arguments present -> CLI mode, otherwise -> interactive mode
    if len(sys.argv) > 1:
        cli_main()
    else:
        interactive_main()


if __name__ == "__main__":
    main()
