"""
Korean Stock Investment Strategy Generator
CLI entry point for the multi-agent analysis workflow

PyCharm 실행 방법:
1. 이 파일을 직접 실행
2. 종목코드 입력 (예: 005930)
3. 분석 기준 날짜 입력 (예: 2025-12-13, 빈 값이면 오늘 날짜)
"""
import asyncio
import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from kr.config import settings
from kr.db.connection import Database
from kr.db.queries import save_strategy
from kr.graph import create_graph
from kr.state import create_initial_state, get_state_summary


def save_result_to_file(symbol: str, strategy: dict, target_date: str) -> str:
    """
    Save final strategy result to JSON file.

    Args:
        symbol: Stock code (e.g., "005930")
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
    Run the analysis workflow for a stock.

    Args:
        symbol: Stock code (e.g., "005930")
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
        strategy = result.get("strategy")
        if strategy:
            stock_name = result.get('stock_name', symbol)
            exec_time = strategy.get('metadata', {}).get('execution_time_sec', 0)

            print(f"[SUCCESS] Analysis completed for {stock_name}")
            print(f"[INFO] Execution time: {exec_time:.2f}s")

            # Save to PostgreSQL
            if save_to_db:
                try:
                    analysis_date = date.fromisoformat(target_date) if target_date else None
                    await save_strategy(symbol, strategy, analysis_date)
                    print(f"[INFO] Result saved to PostgreSQL (kr_stock_grade.strategy, date={analysis_date or 'latest'})")
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
    # 1. 종목코드 입력
    symbol = input("종목코드를 입력하세요 (예: 005930): ").strip()
    if not symbol:
        print("[ERROR] 종목코드는 필수입니다.")
        return

    # 2. 날짜 입력
    date_input = input("분석 기준 날짜를 입력하세요 (YYYY-MM-DD, 빈 값이면 오늘): ").strip()

    # 날짜 검증
    target_date = None
    if date_input:
        try:
            date.fromisoformat(date_input)
            target_date = date_input
        except ValueError:
            print(f"[ERROR] 잘못된 날짜 형식입니다: {date_input}. YYYY-MM-DD 형식으로 입력하세요.")
            return

    # 설정 출력
    print()
    print("-" * 60)
    print(f"[INFO] 종목코드: {symbol}")
    print(f"[INFO] 분석 기준일: {target_date or date.today().isoformat()}")
    print(f"[INFO] LLM 모델: {settings.OPENAI_MODEL}")
    print("-" * 60)
    print()
    print("[INFO] 분석을 시작합니다... (약 3-4분 소요)")
    print()

    # 워크플로우 실행
    result = asyncio.run(run(symbol, target_date, save_to_db=True))

    # 결과 출력
    print()
    print("=" * 60)
    print("RESULT (분석 결과)")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("=" * 60)
    print("[INFO] 분석 완료. 결과가 PostgreSQL에 저장되었습니다.")
    print("=" * 60)


def get_auto_target_date() -> str:
    """
    Get target date automatically based on current time and weekday.

    Rules:
    - Tue-Fri 21:00-23:59: Today
    - Tue-Fri 00:00-20:59: Yesterday
    - Sat-Sun: Friday (1-2 days ago)
    - Mon 21:00-23:59: Today
    - Mon 00:00-20:59: Friday (3 days ago)

    Returns:
        Target date string in YYYY-MM-DD format
    """
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 1=Tue, ..., 6=Sun
    hour = now.hour

    if weekday == 0:  # Monday
        if hour >= 21:
            return now.date().isoformat()
        else:
            return (now.date() - timedelta(days=3)).isoformat()
    elif weekday in [1, 2, 3, 4]:  # Tue-Fri
        if hour >= 21:
            return now.date().isoformat()
        else:
            return (now.date() - timedelta(days=1)).isoformat()
    elif weekday == 5:  # Saturday
        return (now.date() - timedelta(days=1)).isoformat()
    else:  # Sunday
        return (now.date() - timedelta(days=2)).isoformat()


def run_auto_analysis():
    """Run stock analysis with auto date detection (Option 2)"""
    # 1. 종목코드 입력
    symbol = input("종목코드를 입력하세요 (예: 005930): ").strip()
    if not symbol:
        print("[ERROR] 종목코드는 필수입니다.")
        return

    # 2. 자동 날짜 결정
    target_date = get_auto_target_date()

    # 설정 출력
    print()
    print("-" * 60)
    print(f"[INFO] 종목코드: {symbol}")
    print(f"[INFO] 분석 기준일: {target_date} (자동 설정)")
    print(f"[INFO] LLM 모델: {settings.OPENAI_MODEL}")
    print("-" * 60)
    print()
    print("[INFO] 분석을 시작합니다... (약 3-4분 소요)")
    print()

    # 워크플로우 실행
    result = asyncio.run(run(symbol, target_date, save_to_db=True))

    # 결과 출력
    print()
    print("=" * 60)
    print("RESULT (분석 결과)")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("=" * 60)
    print("[INFO] 분석 완료. 결과가 PostgreSQL에 저장되었습니다.")
    print("=" * 60)


def interactive_main():
    """PyCharm/IDE interactive entry point"""
    print("=" * 60)
    print("한국 주식 투자 전략 생성기 (Korean Stock Strategy Generator)")
    print("=" * 60)
    print()

    run_single_analysis()


def cli_main():
    """Command line entry point with arguments"""
    parser = argparse.ArgumentParser(
        description="Korean Stock Investment Strategy Generator"
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Stock code (e.g., 005930)"
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

    # Validate date format if provided
    if args.date:
        try:
            date.fromisoformat(args.date)
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}. Use YYYY-MM-DD.")
            return

    # Print settings if verbose
    if args.verbose:
        print(f"[INFO] Symbol: {args.symbol}")
        print(f"[INFO] Date: {args.date or date.today().isoformat()}")
        print(f"[INFO] Model: {settings.OPENAI_MODEL}")

    # Run workflow
    save_to_db = not args.no_save
    result = asyncio.run(run(args.symbol, args.date, save_to_db=save_to_db))

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
    - PyCharm에서 직접 실행: 대화형 모드
    - CLI에서 인자와 함께 실행: CLI 모드
    """
    import sys

    # 인자가 있으면 CLI 모드, 없으면 대화형 모드
    if len(sys.argv) > 1:
        cli_main()
    else:
        interactive_main()


if __name__ == "__main__":
    main()
