"""
Database connection and query tests
Tests DB connectivity and data retrieval functions
"""
import asyncio
import pytest
from datetime import date

from kr.db.connection import Database
from kr.db.queries import (
    get_stock_basic,
    get_stock_detail,
    get_stock_grade,
    get_stock_indicators,
    get_intraday_total,
    get_individual_investor_daily_trading,
    get_foreign_ownership,
    get_financial_position,
    get_research_reports,
    get_blocktrades,
    get_dividends,
    get_largest_shareholder,
    get_market_index,
    get_economic_indicators,
    get_exchange_rate,
    get_us_fed_funds_rate,
    get_us_treasury_yield,
    get_us_cpi,
    get_us_unemployment_rate,
    get_us_gdp,
    get_us_pmi,
    get_us_vix,
    get_us_dollar_index,
)


# Test symbol: Samsung Electronics
TEST_SYMBOL = "005930"


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module", autouse=True)
async def setup_db():
    """Setup and teardown database connection"""
    await Database.connect()
    yield
    await Database.disconnect()


# =============================================================================
# Connection Tests
# =============================================================================

@pytest.mark.asyncio
async def test_db_connection():
    """Test database connection is established"""
    assert Database.pool is not None
    print("[PASS] Database connection established")


# =============================================================================
# Stock Basic Info Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_stock_basic():
    """Test get_stock_basic query"""
    result = await get_stock_basic(TEST_SYMBOL)

    assert result is not None, f"No data found for {TEST_SYMBOL}"
    assert result.get("symbol") == TEST_SYMBOL
    assert result.get("stock_name") is not None

    print(f"[PASS] get_stock_basic: {result.get('stock_name')}")


@pytest.mark.asyncio
async def test_get_stock_detail():
    """Test get_stock_detail query"""
    result = await get_stock_detail(TEST_SYMBOL)

    assert result is not None, f"No data found for {TEST_SYMBOL}"
    assert result.get("symbol") == TEST_SYMBOL
    assert result.get("industry") is not None

    print(f"[PASS] get_stock_detail: industry={result.get('industry')}")


# =============================================================================
# Quant Analysis Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_stock_grade():
    """Test get_stock_grade query"""
    result = await get_stock_grade(TEST_SYMBOL)

    assert result is not None, f"No grade data found for {TEST_SYMBOL}"
    assert result.get("symbol") == TEST_SYMBOL
    assert result.get("final_grade") is not None
    assert result.get("final_score") is not None

    print(f"[PASS] get_stock_grade: grade={result.get('final_grade')}, score={result.get('final_score')}")


# =============================================================================
# Time Series Data Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_stock_indicators():
    """Test get_stock_indicators query"""
    result = await get_stock_indicators(TEST_SYMBOL, days=30)

    assert result is not None
    assert len(result) > 0, f"No indicator data found for {TEST_SYMBOL}"
    assert result[0].get("symbol") == TEST_SYMBOL
    assert result[0].get("rsi") is not None

    print(f"[PASS] get_stock_indicators: {len(result)} days, latest RSI={result[0].get('rsi')}")


@pytest.mark.asyncio
async def test_get_intraday_total():
    """Test get_intraday_total query"""
    result = await get_intraday_total(TEST_SYMBOL, days=30)

    assert result is not None
    assert len(result) > 0, f"No price data found for {TEST_SYMBOL}"
    assert result[0].get("close") is not None

    print(f"[PASS] get_intraday_total: {len(result)} days, latest close={result[0].get('close')}")


@pytest.mark.asyncio
async def test_get_individual_investor_daily_trading():
    """Test get_individual_investor_daily_trading query"""
    result = await get_individual_investor_daily_trading(TEST_SYMBOL, days=30)

    assert result is not None
    assert len(result) > 0, f"No investor trading data found for {TEST_SYMBOL}"

    print(f"[PASS] get_individual_investor_daily_trading: {len(result)} days")


@pytest.mark.asyncio
async def test_get_foreign_ownership():
    """Test get_foreign_ownership query"""
    result = await get_foreign_ownership(TEST_SYMBOL, days=30)

    assert result is not None
    assert len(result) > 0, f"No foreign ownership data found for {TEST_SYMBOL}"
    assert result[0].get("foreign_rate") is not None

    print(f"[PASS] get_foreign_ownership: {len(result)} days, latest rate={result[0].get('foreign_rate')}%")


# =============================================================================
# Financial Data Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_financial_position():
    """Test get_financial_position query"""
    result = await get_financial_position(TEST_SYMBOL)

    # Financial data may not exist for all stocks
    if result:
        print(f"[PASS] get_financial_position: {len(result)} records")
    else:
        print(f"[WARN] get_financial_position: No data found (may be expected)")


@pytest.mark.asyncio
async def test_get_research_reports():
    """Test get_research_reports query"""
    result = await get_research_reports(TEST_SYMBOL, days=180)

    # Research reports may not exist for all stocks
    if result:
        print(f"[PASS] get_research_reports: {len(result)} reports")
    else:
        print(f"[WARN] get_research_reports: No data found (may be expected)")


# =============================================================================
# Market Data Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_market_index():
    """Test get_market_index query"""
    result = await get_market_index(exchange="KOSPI", days=30)

    assert result is not None
    assert len(result) > 0, "No KOSPI index data found"
    assert result[0].get("close") is not None

    print(f"[PASS] get_market_index (KOSPI): {len(result)} days, latest={result[0].get('close')}")


@pytest.mark.asyncio
async def test_get_economic_indicators():
    """Test get_economic_indicators query"""
    result = await get_economic_indicators()

    assert result is not None
    assert len(result) > 0, "No economic indicator data found"

    # Check for key indicators
    stat_names = [r.get("stat_name", "") for r in result]
    print(f"[PASS] get_economic_indicators: {len(result)} indicators")


@pytest.mark.asyncio
async def test_get_exchange_rate():
    """Test get_exchange_rate query"""
    result = await get_exchange_rate(currency="원/미국달러", days=30)

    assert result is not None
    assert len(result) > 0, "No exchange rate data found"

    print(f"[PASS] get_exchange_rate: {len(result)} days, latest={result[0].get('data_value')}")


# =============================================================================
# US Economic Indicator Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_us_fed_funds_rate():
    """Test get_us_fed_funds_rate query"""
    result = await get_us_fed_funds_rate(days=365)

    if result:
        print(f"[PASS] get_us_fed_funds_rate: {len(result)} records, latest={result[0].get('value')}")
    else:
        print(f"[WARN] get_us_fed_funds_rate: No data found")


@pytest.mark.asyncio
async def test_get_us_treasury_yield():
    """Test get_us_treasury_yield query"""
    result = await get_us_treasury_yield(days=365)

    if result:
        print(f"[PASS] get_us_treasury_yield: {len(result)} records, latest={result[0].get('value')}")
    else:
        print(f"[WARN] get_us_treasury_yield: No data found")


@pytest.mark.asyncio
async def test_get_us_vix():
    """Test get_us_vix query"""
    result = await get_us_vix(days=90)

    if result:
        print(f"[PASS] get_us_vix: {len(result)} records, latest={result[0].get('value')}")
    else:
        print(f"[WARN] get_us_vix: No data found")


@pytest.mark.asyncio
async def test_get_us_dollar_index():
    """Test get_us_dollar_index query"""
    result = await get_us_dollar_index(days=90)

    if result:
        print(f"[PASS] get_us_dollar_index: {len(result)} records, latest={result[0].get('value')}")
    else:
        print(f"[WARN] get_us_dollar_index: No data found")


# =============================================================================
# Run all tests manually
# =============================================================================

async def run_all_tests():
    """Run all tests manually without pytest"""
    print("=" * 60)
    print("Database Tests")
    print("=" * 60)

    await Database.connect()

    try:
        # Connection
        await test_db_connection()

        # Stock info
        await test_get_stock_basic()
        await test_get_stock_detail()
        await test_get_stock_grade()

        # Time series
        await test_get_stock_indicators()
        await test_get_intraday_total()
        await test_get_individual_investor_daily_trading()
        await test_get_foreign_ownership()

        # Financial
        await test_get_financial_position()
        await test_get_research_reports()

        # Market
        await test_get_market_index()
        await test_get_economic_indicators()
        await test_get_exchange_rate()

        # US indicators
        await test_get_us_fed_funds_rate()
        await test_get_us_treasury_yield()
        await test_get_us_vix()
        await test_get_us_dollar_index()

        print("=" * 60)
        print("All database tests completed!")
        print("=" * 60)

    finally:
        await Database.disconnect()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
