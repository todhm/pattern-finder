"""
End-to-End Tests for Stock Investment Strategy Workflow

Tests the complete workflow with actual database connection:
1. Data collection
2. Parallel agent execution (Market Regime + Stock Research)
3. Analysis generation
4. Validation
5. Finalization

Usage:
    python -m kr.tests.test_e2e --symbol 005930
    python -m kr.tests.test_e2e --symbol 005930 --verbose
"""
import asyncio
import argparse
import json
from datetime import date

from kr.config import settings
from kr.db.connection import Database


async def test_stock_research_standalone(symbol: str, verbose: bool = False):
    """
    Test Stock Research Agent in isolation.

    Args:
        symbol: Stock code to test
        verbose: Print detailed output
    """
    from kr.agents.stock_research import run_stock_research

    print(f"\n[TEST] Stock Research Agent for {symbol}")
    print("-" * 50)

    await Database.connect()
    try:
        result = await run_stock_research(symbol)

        if result.get("error"):
            print(f"[ERROR] {result['error']}")
            return False

        # Check required fields
        required_fields = [
            "symbol", "stock_name", "overall_sentiment", "confidence",
            "earnings_outlook", "sector_momentum"
        ]

        missing = [f for f in required_fields if f not in result]
        if missing:
            print(f"[FAIL] Missing required fields: {missing}")
            return False

        print(f"[OK] Symbol: {result.get('symbol')}")
        print(f"[OK] Stock Name: {result.get('stock_name')}")
        print(f"[OK] Sentiment: {result.get('overall_sentiment')} (confidence: {result.get('confidence')})")
        print(f"[OK] Sector: {result.get('sector')}")

        metadata = result.get("metadata", {})
        data_counts = metadata.get("data_collection", {})
        print(f"[OK] Data: {data_counts.get('research_reports', 0)} reports, "
              f"{data_counts.get('news_articles', 0)} news, "
              f"{data_counts.get('disclosures', 0)} disclosures")

        if verbose:
            print("\n[VERBOSE] Full result:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        print("\n[PASS] Stock Research Agent test passed")
        return True

    finally:
        await Database.disconnect()


async def test_full_workflow(symbol: str, verbose: bool = False):
    """
    Test the complete workflow with all agents.

    Args:
        symbol: Stock code to test
        verbose: Print detailed output
    """
    from kr.graph import create_graph
    from kr.state import create_initial_state, get_state_summary

    print(f"\n[TEST] Full Workflow for {symbol}")
    print("-" * 50)

    await Database.connect()
    try:
        # Create initial state
        initial_state = create_initial_state(
            symbol=symbol,
            target_date=date.today().isoformat()
        )

        # Create and run workflow
        graph = create_graph()

        print("[INFO] Starting workflow...")
        result = await graph.ainvoke(initial_state)

        # Print state summary
        if verbose:
            summary = get_state_summary(result)
            print(f"[INFO] State summary: {json.dumps(summary, indent=2)}")

        # Check for errors
        if result.get("error"):
            print(f"[ERROR] Workflow error: {result['error']}")
            return False

        # Check for validation failures
        if result.get("validation_errors"):
            print(f"[WARN] Validation errors: {result['validation_errors']}")
            if len(result["validation_errors"]) > 0:
                print("[FAIL] Workflow failed validation")
                return False

        # Check market regime result
        market_regime = result.get("market_regime")
        if market_regime:
            print(f"[OK] Market Regime: {market_regime.get('regime')} "
                  f"(confidence: {market_regime.get('confidence')})")
        else:
            print("[WARN] Market Regime: No result (may be acceptable in degraded mode)")

        # Check stock research result
        stock_research = result.get("stock_research")
        if stock_research:
            print(f"[OK] Stock Research: {stock_research.get('overall_sentiment')} "
                  f"(confidence: {stock_research.get('confidence')})")
        else:
            print("[WARN] Stock Research: No result (may be acceptable in degraded mode)")

        # Check final strategy
        strategy = result.get("strategy")
        if not strategy:
            print("[FAIL] No strategy generated")
            return False

        print(f"[OK] Final Grade: {strategy.get('final_grade')}")
        print(f"[OK] Analysis Date: {strategy.get('analysis_date')}")

        # Check scenarios
        scenarios = strategy.get("scenarios", {})
        if scenarios:
            bullish = scenarios.get("bullish", {})
            sideways = scenarios.get("sideways", {})
            bearish = scenarios.get("bearish", {})

            prob_sum = (
                bullish.get("probability", 0) +
                sideways.get("probability", 0) +
                bearish.get("probability", 0)
            )
            print(f"[OK] Scenarios: Bullish {bullish.get('probability')}%, "
                  f"Sideways {sideways.get('probability')}%, "
                  f"Bearish {bearish.get('probability')}% (sum: {prob_sum}%)")

            if prob_sum != 100:
                print(f"[WARN] Probability sum should be 100%, got {prob_sum}%")

        # Check metadata
        metadata = strategy.get("metadata", {})
        exec_time = metadata.get("execution_time_sec", 0)
        print(f"[OK] Execution Time: {exec_time:.2f}s")

        if verbose:
            print("\n[VERBOSE] Full strategy:")
            print(json.dumps(strategy, ensure_ascii=False, indent=2))

        print("\n[PASS] Full workflow test passed")
        return True

    finally:
        await Database.disconnect()


async def test_parallel_execution(symbol: str, verbose: bool = False):
    """
    Test that parallel agent execution works correctly.

    Args:
        symbol: Stock code to test
        verbose: Print detailed output
    """
    from kr.graph import parallel_agent_analysis
    from kr.state import create_initial_state
    from kr.data.collector import collect_stock_data

    print(f"\n[TEST] Parallel Agent Execution for {symbol}")
    print("-" * 50)

    await Database.connect()
    try:
        # Create initial state with some raw data
        state = create_initial_state(symbol=symbol)

        # Collect stock data first (required for some agent operations)
        raw_data = await collect_stock_data(symbol, None)
        state["raw_data"] = raw_data

        # Get stock name
        if raw_data.get("stock_detail"):
            state["stock_name"] = raw_data["stock_detail"].get("stock_name", "")

        print(f"[INFO] Stock: {state['stock_name']} ({symbol})")
        print("[INFO] Running parallel agents...")

        import time
        start = time.time()

        # Run parallel agents
        result = await parallel_agent_analysis(state)

        elapsed = time.time() - start
        print(f"[INFO] Parallel execution completed in {elapsed:.2f}s")

        # Check results
        market_regime = result.get("market_regime")
        stock_research = result.get("stock_research")

        if market_regime:
            print(f"[OK] Market Regime Agent returned: {market_regime.get('regime')}")
        else:
            print("[WARN] Market Regime Agent: No result")

        if stock_research:
            print(f"[OK] Stock Research Agent returned: {stock_research.get('overall_sentiment')}")
        else:
            print("[WARN] Stock Research Agent: No result")

        if verbose and market_regime:
            print("\n[VERBOSE] Market Regime:")
            print(json.dumps(market_regime, ensure_ascii=False, indent=2))

        if verbose and stock_research:
            print("\n[VERBOSE] Stock Research:")
            print(json.dumps(stock_research, ensure_ascii=False, indent=2))

        # At least one should succeed
        if market_regime or stock_research:
            print("\n[PASS] Parallel execution test passed")
            return True
        else:
            print("\n[FAIL] Both agents failed to return results")
            return False

    finally:
        await Database.disconnect()


async def run_all_e2e_tests(symbol: str, verbose: bool = False):
    """Run all E2E tests"""
    print("=" * 60)
    print(f"End-to-End Tests for Stock Investment Strategy Workflow")
    print(f"Symbol: {symbol}")
    print(f"Date: {date.today().isoformat()}")
    print(f"Model: {settings.OPENAI_MODEL}")
    print("=" * 60)

    results = {}

    # Test 1: Stock Research Agent standalone
    results["stock_research"] = await test_stock_research_standalone(symbol, verbose)

    # Test 2: Parallel execution
    results["parallel_execution"] = await test_parallel_execution(symbol, verbose)

    # Test 3: Full workflow
    results["full_workflow"] = await test_full_workflow(symbol, verbose)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="E2E Tests for Stock Investment Strategy Workflow"
    )
    parser.add_argument(
        "--symbol", "-s",
        default="005930",
        help="Stock code to test (default: 005930)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--test", "-t",
        choices=["all", "research", "parallel", "workflow"],
        default="all",
        help="Specific test to run (default: all)"
    )

    args = parser.parse_args()

    if args.test == "all":
        success = asyncio.run(run_all_e2e_tests(args.symbol, args.verbose))
    elif args.test == "research":
        success = asyncio.run(test_stock_research_standalone(args.symbol, args.verbose))
    elif args.test == "parallel":
        success = asyncio.run(test_parallel_execution(args.symbol, args.verbose))
    elif args.test == "workflow":
        success = asyncio.run(test_full_workflow(args.symbol, args.verbose))
    else:
        success = False

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
