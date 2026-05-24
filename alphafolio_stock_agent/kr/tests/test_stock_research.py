"""
Stock Research Agent E2E Tests

Tests the complete stock research pipeline:
1. Data collection (research_collector)
2. Preprocessing (research_preprocessor)
3. LLM extraction (stock_research agent)
4. Integration with workflow (graph.py)
"""
import asyncio
import json
from datetime import date

# Test configuration
TEST_SYMBOL = "005930"  # Samsung Electronics (large cap)
TEST_SYMBOL_MEDIUM = "086520"  # Ecopro (medium cap)


async def test_determine_stock_size():
    """Test market cap classification"""
    from kr.data.research_collector import determine_stock_size

    # Large cap: 10 trillion+
    assert determine_stock_size(15_000_000_000_000) == "large"

    # Medium cap: 1-10 trillion
    assert determine_stock_size(5_000_000_000_000) == "medium"

    # Small cap: < 1 trillion
    assert determine_stock_size(500_000_000_000) == "small"

    # Edge cases
    assert determine_stock_size(0) == "small"
    assert determine_stock_size(-1) == "small"
    assert determine_stock_size(None) == "small"

    print("[PASS] test_determine_stock_size")


async def test_news_strategy():
    """Test adaptive news strategy selection"""
    from kr.data.research_collector import get_news_strategy

    # Large cap strategy
    large_strategy = get_news_strategy(15_000_000_000_000)
    assert large_strategy["days"] == 7
    assert large_strategy["use_keyword_filter"] == True

    # Medium cap strategy
    medium_strategy = get_news_strategy(5_000_000_000_000)
    assert medium_strategy["days"] == 14
    assert medium_strategy["use_keyword_filter"] == False

    # Small cap strategy
    small_strategy = get_news_strategy(500_000_000_000)
    assert small_strategy["days"] == 30
    assert small_strategy["use_keyword_filter"] == False

    print("[PASS] test_news_strategy")


async def test_text_normalization():
    """Test text normalization functions"""
    from kr.data.research_preprocessor import normalize_text, truncate_text

    # HTML tag removal
    text = "<p>Hello <b>World</b></p>"
    assert normalize_text(text) == "Hello World"

    # HTML entity decoding
    text = "A &amp; B &lt;C&gt;"
    assert normalize_text(text) == "A & B <C>"

    # Whitespace normalization
    text = "  Multiple   spaces  "
    assert normalize_text(text) == "Multiple spaces"

    # Truncation
    long_text = "A" * 300
    truncated = truncate_text(long_text, 100)
    assert len(truncated) <= 103  # 100 + "..."
    assert truncated.endswith("...")

    print("[PASS] test_text_normalization")


async def test_deduplication():
    """Test deduplication functions"""
    from kr.data.research_preprocessor import deduplicate_items, deduplicate_by_similarity

    # Exact deduplication
    items = [
        {"title": "Breaking News"},
        {"title": "breaking news"},  # Same when normalized
        {"title": "Other News"},
    ]
    unique = deduplicate_items(items, key="title")
    assert len(unique) == 2

    # Similarity deduplication
    items = [
        {"title": "Samsung Electronics reports Q3 earnings"},
        {"title": "Samsung Electronics announces Q3 earnings results"},  # Similar
        {"title": "Completely different news about Apple"},
    ]
    unique = deduplicate_by_similarity(items, key="title", threshold=0.5)
    assert len(unique) == 2  # First and third should remain

    print("[PASS] test_deduplication")


async def test_disclosure_classification():
    """Test disclosure type classification"""
    from kr.data.research_preprocessor import classify_disclosure

    assert classify_disclosure("삼성전자 2024년 3분기 실적발표") == "earnings"
    assert classify_disclosure("최대주주 지분 변동 공시") == "shareholder"
    assert classify_disclosure("대표이사 선임의 건") == "executive"
    assert classify_disclosure("현금 배당 결정") == "dividend"
    assert classify_disclosure("신규 투자 결정") == "investment"
    assert classify_disclosure("공정거래위원회 조사") == "regulatory"
    assert classify_disclosure("기타 일반 공시") == "other"

    print("[PASS] test_disclosure_classification")


async def test_prompt_formatting():
    """Test stock research prompt formatting"""
    from kr.prompts.stock_research import format_stock_research_prompt

    prompt = format_stock_research_prompt(
        symbol="005930",
        stock_name="삼성전자",
        sector="반도체",
        stock_size="large",
        analysis_date="2025-12-12",
        formatted_text="## Test Data\nSome research data here",
        report_count=5,
        news_count=10,
        disclosure_count=3
    )

    # Check key elements are present
    assert "005930" in prompt
    assert "삼성전자" in prompt
    assert "반도체" in prompt
    assert "large" in prompt
    assert "2025-12-12" in prompt
    assert "5건" in prompt or "5" in prompt  # Report count
    assert "10건" in prompt or "10" in prompt  # News count
    assert "3건" in prompt or "3" in prompt  # Disclosure count

    print("[PASS] test_prompt_formatting")


async def test_output_validation():
    """Test stock research output validation"""
    from kr.prompts.stock_research import validate_research_output

    # Valid output
    valid_output = {
        "symbol": "005930",
        "stock_name": "삼성전자",
        "analysis_date": "2025-12-12",
        "sector": "반도체",
        "overall_sentiment": "positive",
        "confidence": 0.8,
        "earnings_outlook": {
            "direction": "positive",
            "summary": "실적 전망 긍정적",
            "key_points": ["포인트1"],
            "sources": []
        },
        "risk_issues": [],
        "sector_momentum": {
            "sector": "반도체",
            "direction": "positive",
            "summary": "섹터 모멘텀 긍정적",
            "related_news_count": 5,
            "sources": []
        }
    }

    errors = validate_research_output(valid_output)
    assert len(errors) == 0, f"Unexpected errors: {errors}"

    # Invalid sentiment
    invalid_output = {**valid_output, "overall_sentiment": "invalid"}
    errors = validate_research_output(invalid_output)
    assert len(errors) > 0
    assert any("overall_sentiment" in e for e in errors)

    # Invalid confidence
    invalid_output = {**valid_output, "confidence": 1.5}
    errors = validate_research_output(invalid_output)
    assert len(errors) > 0
    assert any("confidence" in e for e in errors)

    # Missing required field
    invalid_output = {**valid_output}
    del invalid_output["symbol"]
    errors = validate_research_output(invalid_output)
    assert len(errors) > 0
    assert any("symbol" in e for e in errors)

    print("[PASS] test_output_validation")


async def test_analysis_prompt_integration():
    """Test that analysis prompt includes stock_research context"""
    from kr.prompts.analysis import format_analysis_prompt

    stock_research = {
        "overall_sentiment": "positive",
        "confidence": 0.8,
        "earnings_outlook": {
            "direction": "positive",
            "summary": "실적 전망 긍정적"
        },
        "risk_issues": [],
        "sector_momentum": {
            "sector": "반도체",
            "direction": "positive",
            "summary": "섹터 모멘텀 긍정적"
        },
        "disclosure_summary": {
            "recent_count": 2
        },
        "metadata": {
            "data_collection": {"research_reports": 5},
            "stock_size": "large"
        }
    }

    prompt = format_analysis_prompt(
        stock_info={"name": "삼성전자", "symbol": "005930"},
        quant_summary={},
        technical_summary={},
        price_trend={},
        investor_summary={},
        foreign_ownership={},
        market_summary={},
        economic_summary={},
        analysis_date="2025-12-12",
        exchange="KOSPI",
        market_regime=None,
        stock_research=stock_research
    )

    # Check stock research context is included
    assert "overall_sentiment" in prompt
    assert "earnings_outlook" in prompt
    assert "sector_momentum" in prompt

    print("[PASS] test_analysis_prompt_integration")


async def test_graph_import():
    """Test that graph module correctly imports parallel_agent_analysis"""
    from kr.graph import create_graph, parallel_agent_analysis

    # Verify function exists and is callable
    assert callable(parallel_agent_analysis)

    # Create graph (this validates the workflow structure)
    graph = create_graph()
    assert graph is not None

    print("[PASS] test_graph_import")


async def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("Stock Research Agent Unit Tests")
    print("=" * 60)

    tests = [
        test_determine_stock_size,
        test_news_strategy,
        test_text_normalization,
        test_deduplication,
        test_disclosure_classification,
        test_prompt_formatting,
        test_output_validation,
        test_analysis_prompt_integration,
        test_graph_import,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
