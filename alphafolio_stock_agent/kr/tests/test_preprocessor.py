"""
Data preprocessor tests
Tests all preprocessing functions with sample data
"""
import pytest
from decimal import Decimal

from kr.data.preprocessor import (
    safe_float,
    safe_int,
    calculate_trend,
    count_consecutive_positive,
    count_consecutive_negative,
    format_korean_number,
    calculate_volatility,
    generate_investor_summary,
    analyze_technical_indicators,
    analyze_investor_trends,
    analyze_foreign_ownership_trend,
    calculate_price_trend,
    summarize_quant_result,
    summarize_economic_indicators,
    summarize_market_index,
    preprocess_all
)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestSafeFloat:
    """Test safe_float helper function"""

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0
        assert safe_float(None, 10.0) == 10.0

    def test_decimal_converts(self):
        assert safe_float(Decimal("123.45")) == 123.45

    def test_string_converts(self):
        assert safe_float("123.45") == 123.45
        assert safe_float("invalid", 99.0) == 99.0

    def test_int_converts(self):
        assert safe_float(100) == 100.0

    def test_float_passthrough(self):
        assert safe_float(123.45) == 123.45


class TestSafeInt:
    """Test safe_int helper function"""

    def test_none_returns_default(self):
        assert safe_int(None) == 0
        assert safe_int(None, 10) == 10

    def test_decimal_converts(self):
        assert safe_int(Decimal("123.45")) == 123

    def test_string_converts(self):
        assert safe_int("123") == 123
        assert safe_int("invalid", 99) == 99

    def test_float_converts(self):
        assert safe_int(123.45) == 123


class TestCalculateTrend:
    """Test calculate_trend helper function"""

    def test_rising_trend(self):
        # Most recent first, so values[0] > values[-1] means rising
        assert calculate_trend([100, 90, 80]) == "rising"

    def test_falling_trend(self):
        assert calculate_trend([80, 90, 100]) == "falling"

    def test_flat_trend(self):
        assert calculate_trend([100, 90, 100]) == "flat"

    def test_insufficient_data(self):
        assert calculate_trend([100]) == "unknown"
        assert calculate_trend([]) == "unknown"


class TestConsecutiveCounts:
    """Test consecutive counting functions"""

    def test_consecutive_positive(self):
        assert count_consecutive_positive([100, 50, 30, -10]) == 3
        assert count_consecutive_positive([-10, 50, 30]) == 0
        assert count_consecutive_positive([]) == 0

    def test_consecutive_negative(self):
        assert count_consecutive_negative([-100, -50, -30, 10]) == 3
        assert count_consecutive_negative([10, -50, -30]) == 0
        assert count_consecutive_negative([]) == 0


class TestFormatKoreanNumber:
    """Test Korean number formatting"""

    def test_eok_format(self):
        assert "억" in format_korean_number(500000000)
        assert "억" in format_korean_number(-500000000)

    def test_man_format(self):
        assert "만" in format_korean_number(50000)
        assert "만" in format_korean_number(-50000)

    def test_small_numbers(self):
        result = format_korean_number(1000)
        assert "억" not in result
        assert "만" not in result


class TestCalculateVolatility:
    """Test volatility calculation"""

    def test_normal_volatility(self):
        prices = [100, 98, 102, 99, 101]
        vol = calculate_volatility(prices)
        assert vol >= 0

    def test_insufficient_data(self):
        assert calculate_volatility([100]) == 0.0
        assert calculate_volatility([]) == 0.0

    def test_zero_price_handling(self):
        prices = [100, 0, 100]
        # Should not raise division by zero
        vol = calculate_volatility(prices)
        assert vol >= 0


class TestGenerateInvestorSummary:
    """Test investor summary generation"""

    def test_foreign_buying_streak(self):
        foreign = [100, 200, 300, 400]  # 4 days buying
        inst = [10, 20, -30, 40]
        result = generate_investor_summary(foreign, inst)
        assert "외국인" in result
        assert "순매수" in result

    def test_institutional_selling_streak(self):
        foreign = [10, -20, 30]
        inst = [-100, -200, -300, -400]  # 4 days selling
        result = generate_investor_summary(foreign, inst)
        assert "기관" in result
        assert "순매도" in result

    def test_no_notable_trend(self):
        foreign = [10, -20, 30]
        inst = [10, -20, 30]
        result = generate_investor_summary(foreign, inst)
        assert result == "특이 동향 없음"


# =============================================================================
# Technical Indicators Analysis Tests
# =============================================================================

class TestAnalyzeTechnicalIndicators:
    """Test technical indicators analysis"""

    @pytest.fixture
    def sample_indicators(self):
        """Sample indicator data (most recent first)"""
        return [
            {
                "rsi": 75,
                "macd": 10,
                "macd_signal": 8,
                "macd_hist": 2,
                "real_upper_band": 60000,
                "real_middle_band": 55000,
                "real_lower_band": 50000,
                "slowk": 85,
                "slowd": 80,
                "adx": 30,
                "mfi": 75
            },
            {
                "rsi": 70,
                "macd": 7,
                "macd_signal": 9,
                "macd_hist": -2,
                "slowk": 80,
                "slowd": 75,
                "adx": 28,
                "mfi": 70
            }
        ]

    def test_empty_data(self):
        result = analyze_technical_indicators([])
        assert result == {}

    def test_rsi_overbought(self, sample_indicators):
        result = analyze_technical_indicators(sample_indicators)
        assert result["rsi"]["status"] == "overbought"
        assert result["rsi"]["value"] == 75

    def test_macd_crossover_detection(self, sample_indicators):
        result = analyze_technical_indicators(sample_indicators)
        # Previous: macd(7) < signal(9), Current: macd(10) > signal(8)
        assert result["macd"]["crossover"] == "golden_cross"

    def test_stochastic_overbought(self, sample_indicators):
        result = analyze_technical_indicators(sample_indicators)
        assert result["stochastic"]["status"] == "overbought"

    def test_adx_strong_trend(self, sample_indicators):
        result = analyze_technical_indicators(sample_indicators)
        assert result["adx"]["status"] == "strong_trend"


# =============================================================================
# Investor Trends Analysis Tests
# =============================================================================

class TestAnalyzeInvestorTrends:
    """Test investor trends analysis"""

    @pytest.fixture
    def sample_trading_data(self):
        """Sample trading data (most recent first)"""
        return [
            {"foreign_net_volume": 1000000, "inst_net_volume": -500000, "retail_net_volume": -500000},
            {"foreign_net_volume": 800000, "inst_net_volume": -400000, "retail_net_volume": -400000},
            {"foreign_net_volume": 600000, "inst_net_volume": -300000, "retail_net_volume": -300000},
            {"foreign_net_volume": 400000, "inst_net_volume": 200000, "retail_net_volume": 200000},
            {"foreign_net_volume": 200000, "inst_net_volume": 100000, "retail_net_volume": -300000},
        ]

    def test_empty_data(self):
        result = analyze_investor_trends([])
        assert result == {}

    def test_foreign_buying_trend(self, sample_trading_data):
        result = analyze_investor_trends(sample_trading_data)
        assert result["foreign"]["trend"] == "buying"
        assert result["foreign"]["consecutive_buy_days"] > 0

    def test_net_volume_calculation(self, sample_trading_data):
        result = analyze_investor_trends(sample_trading_data)
        # net_5d should be sum of all 5 days
        expected_foreign_5d = 1000000 + 800000 + 600000 + 400000 + 200000
        assert result["foreign"]["net_5d"] == expected_foreign_5d


# =============================================================================
# Foreign Ownership Trend Tests
# =============================================================================

class TestAnalyzeForeignOwnershipTrend:
    """Test foreign ownership trend analysis"""

    @pytest.fixture
    def sample_ownership_data(self):
        """Sample ownership data (most recent first)"""
        return [{"foreign_rate": 55.0 - i * 0.1, "foreign_rate_limit": 100} for i in range(30)]

    def test_empty_data(self):
        result = analyze_foreign_ownership_trend([])
        assert result == {}

    def test_current_rate(self, sample_ownership_data):
        result = analyze_foreign_ownership_trend(sample_ownership_data)
        assert result["current_rate"] == 55.0

    def test_increasing_trend(self, sample_ownership_data):
        result = analyze_foreign_ownership_trend(sample_ownership_data)
        # current (55.0) > 5d ago (55.4)? No, actually decreasing
        assert result["trend"] == "decreasing"


# =============================================================================
# Price Trend Tests
# =============================================================================

class TestCalculatePriceTrend:
    """Test price trend calculation"""

    @pytest.fixture
    def sample_prices(self):
        """Sample price data (most recent first)"""
        prices = []
        for i in range(60):
            prices.append({
                "close": 55000 + (i * 100),  # Decreasing from past to present
                "volume": 1000000 + i * 10000
            })
        # Reverse to make most recent first (higher price)
        return list(reversed(prices))

    def test_empty_data(self):
        result = calculate_price_trend([])
        assert result == {}

    def test_insufficient_data(self):
        result = calculate_price_trend([{"close": 55000}])
        assert result == {}

    def test_current_price(self, sample_prices):
        result = calculate_price_trend(sample_prices)
        assert result["current_price"] > 0

    def test_moving_averages(self, sample_prices):
        result = calculate_price_trend(sample_prices)
        assert "ma5" in result
        assert "ma20" in result
        assert "ma60" in result

    def test_price_changes(self, sample_prices):
        result = calculate_price_trend(sample_prices)
        assert "change_1d" in result
        assert "change_5d" in result
        assert "change_20d" in result

    def test_volume_analysis(self, sample_prices):
        result = calculate_price_trend(sample_prices)
        assert "volume" in result
        assert "current" in result["volume"]
        assert "avg_20d" in result["volume"]


# =============================================================================
# Quant Result Summary Tests
# =============================================================================

class TestSummarizeQuantResult:
    """Test quant result summarization"""

    @pytest.fixture
    def sample_stock_grade(self):
        """Sample stock grade data"""
        return {
            "final_grade": "A",
            "final_score": 85.5,
            "value_score": 80.0,
            "quality_score": 90.0,
            "momentum_score": 75.0,
            "growth_score": 85.0,
            "confidence_score": 80.0,
            "entry_timing_score": 70.0,
            "scenario_bullish_prob": 40,
            "scenario_sideways_prob": 35,
            "scenario_bearish_prob": 25,
            "scenario_bullish_return": "+15%",
            "scenario_sideways_return": "+3%",
            "scenario_bearish_return": "-10%",
            "scenario_sample_count": 100,
            "sector_momentum": 1.5,
            "sector_rank": 3,
            "sector_percentile": 85.0,
            "var_95": -5.0,
            "cvar_95": -7.0,
            "beta": 1.2,
            "volatility_annual": 25.0,
            "max_drawdown_1y": -15.0,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "stop_loss_pct": -5.0,
            "take_profit_pct": 15.0,
            "risk_reward_ratio": 3.0,
            "position_size_pct": 10.0,
            "atr_pct": 2.5,
            "buy_triggers": ["RSI < 30", "MACD golden cross"],
            "sell_triggers": ["RSI > 70", "MACD dead cross"],
            "hold_triggers": ["Stable trend"],
            "risk_profile_text": "Moderate risk",
            "risk_recommendation": "Suitable for growth investors",
            "time_series_text": "Upward trend",
            "signal_overall": "Buy",
            "rs_value": 1.2,
            "rs_rank": 50
        }

    def test_empty_data(self):
        result = summarize_quant_result({})
        assert result == {}

    def test_scores_extraction(self, sample_stock_grade):
        result = summarize_quant_result(sample_stock_grade)
        assert result["scores"]["final_grade"] == "A"
        assert result["scores"]["final_score"] == 85.5

    def test_scenarios_extraction(self, sample_stock_grade):
        result = summarize_quant_result(sample_stock_grade)
        assert result["scenarios"]["bullish_prob"] == 40
        assert result["scenarios"]["sideways_prob"] == 35
        assert result["scenarios"]["bearish_prob"] == 25

    def test_sector_extraction(self, sample_stock_grade):
        result = summarize_quant_result(sample_stock_grade)
        assert result["sector"]["rank"] == 3
        assert result["sector"]["momentum"] == 1.5

    def test_risk_metrics(self, sample_stock_grade):
        result = summarize_quant_result(sample_stock_grade)
        assert result["risk"]["var_95"] == -5.0
        assert result["risk"]["beta"] == 1.2


# =============================================================================
# Economic Indicators Summary Tests
# =============================================================================

class TestSummarizeEconomicIndicators:
    """Test economic indicators summarization"""

    @pytest.fixture
    def sample_kr_indicators(self):
        """Sample Korean economic indicators"""
        return [
            {"stat_name": "기준금리", "data_value": 3.5},
            {"stat_name": "소비자물가 전년동월비", "data_value": 2.8},
            {"stat_name": "경제심리지수", "data_value": 98}
        ]

    @pytest.fixture
    def sample_us_data(self):
        """Sample US economic data"""
        return {
            "fed_funds_rate": [{"value": 5.5, "date": "2024-12-01"}],
            "vix": [{"value": 25, "date": "2024-12-01"}],
            "dollar_index": [
                {"value": 104, "date": "2024-12-05"},
                {"value": 103, "date": "2024-12-04"},
                {"value": 102, "date": "2024-12-03"},
                {"value": 101, "date": "2024-12-02"},
                {"value": 100, "date": "2024-12-01"},
            ]
        }

    def test_empty_data(self):
        result = summarize_economic_indicators([])
        assert "korea" in result
        assert "us" in result

    def test_korean_base_rate(self, sample_kr_indicators):
        result = summarize_economic_indicators(sample_kr_indicators)
        assert result["korea"]["base_rate"]["value"] == 3.5

    def test_us_fed_funds_rate(self, sample_us_data):
        result = summarize_economic_indicators(
            [],
            us_fed_funds_rate=sample_us_data["fed_funds_rate"]
        )
        assert result["us"]["fed_funds_rate"]["value"] == 5.5

    def test_vix_status(self, sample_us_data):
        result = summarize_economic_indicators(
            [],
            us_vix=sample_us_data["vix"]
        )
        assert result["market_sentiment"]["vix"]["status"] == "elevated"

    def test_dollar_index_trend(self, sample_us_data):
        result = summarize_economic_indicators(
            [],
            us_dollar_index=sample_us_data["dollar_index"]
        )
        assert result["market_sentiment"]["dollar_index"]["trend"] == "strengthening"


# =============================================================================
# Market Index Summary Tests
# =============================================================================

class TestSummarizeMarketIndex:
    """Test market index summarization"""

    @pytest.fixture
    def sample_kospi_data(self):
        """Sample KOSPI data"""
        return [{"close": 2500 + i * 10} for i in range(20)]

    def test_empty_data(self):
        result = summarize_market_index([], [])
        assert result == {}

    def test_kospi_summary(self, sample_kospi_data):
        result = summarize_market_index(sample_kospi_data, [])
        assert "kospi" in result
        assert "current" in result["kospi"]
        assert "change_1d" in result["kospi"]


# =============================================================================
# Preprocess All Integration Test
# =============================================================================

class TestPreprocessAll:
    """Test main preprocessing function"""

    @pytest.fixture
    def sample_collected_data(self):
        """Sample collected data structure"""
        return {
            "stock_detail": {
                "symbol": "005930",
                "stock_name": "삼성전자",
                "industry": "반도체",
                "exchange": "KOSPI",
                "theme": "AI"
            },
            "stock_grade": {
                "final_grade": "A",
                "final_score": 85,
                "scenario_bullish_prob": 40,
                "scenario_sideways_prob": 35,
                "scenario_bearish_prob": 25
            },
            "indicators": [
                {"rsi": 55, "macd": 100, "macd_signal": 90, "slowk": 60, "slowd": 55, "adx": 22, "mfi": 55}
            ],
            "prices": [
                {"close": 55000, "volume": 10000000},
                {"close": 54000, "volume": 9000000}
            ],
            "investor_trading": [
                {"foreign_net_volume": 100000, "inst_net_volume": -50000, "retail_net_volume": -50000}
            ],
            "foreign_ownership": [
                {"foreign_rate": 55.0, "foreign_rate_limit": 100}
            ],
            "market_index_kospi": [
                {"close": 2500}
            ],
            "market_index_kosdaq": [],
            "economic_indicators": [],
            "us_vix": [{"value": 20}]
        }

    def test_returns_all_sections(self, sample_collected_data):
        result = preprocess_all(sample_collected_data)

        assert "stock_info" in result
        assert "quant_summary" in result
        assert "technical_summary" in result
        assert "price_trend" in result
        assert "investor_summary" in result
        assert "foreign_ownership" in result
        assert "market_summary" in result
        assert "economic_summary" in result
        assert "data_availability" in result

    def test_stock_info_populated(self, sample_collected_data):
        result = preprocess_all(sample_collected_data)
        assert result["stock_info"]["symbol"] == "005930"
        assert result["stock_info"]["stock_name"] == "삼성전자"

    def test_data_availability_flags(self, sample_collected_data):
        result = preprocess_all(sample_collected_data)
        assert result["data_availability"]["has_indicators"] is True
        assert result["data_availability"]["has_prices"] is True


# =============================================================================
# Run tests manually
# =============================================================================

def run_all_tests():
    """Run all tests manually without pytest"""
    print("=" * 60)
    print("Preprocessor Tests")
    print("=" * 60)

    # Helper function tests
    print("\n[Testing Helper Functions]")

    # safe_float
    assert safe_float(None) == 0.0
    assert safe_float(Decimal("123.45")) == 123.45
    print("[PASS] safe_float")

    # safe_int
    assert safe_int(None) == 0
    assert safe_int(Decimal("123")) == 123
    print("[PASS] safe_int")

    # calculate_trend
    assert calculate_trend([100, 90, 80]) == "rising"
    assert calculate_trend([80, 90, 100]) == "falling"
    print("[PASS] calculate_trend")

    # count functions
    assert count_consecutive_positive([100, 50, 30, -10]) == 3
    assert count_consecutive_negative([-100, -50, -30, 10]) == 3
    print("[PASS] consecutive counting")

    # format_korean_number
    assert "억" in format_korean_number(500000000)
    assert "만" in format_korean_number(50000)
    print("[PASS] format_korean_number")

    # Technical indicators
    print("\n[Testing Technical Indicators Analysis]")
    indicators = [
        {"rsi": 75, "macd": 10, "macd_signal": 8, "slowk": 85, "slowd": 80, "adx": 30, "mfi": 75},
        {"rsi": 70, "macd": 7, "macd_signal": 9, "slowk": 80, "slowd": 75, "adx": 28, "mfi": 70}
    ]
    result = analyze_technical_indicators(indicators)
    assert result["rsi"]["status"] == "overbought"
    assert result["macd"]["crossover"] == "golden_cross"
    print("[PASS] analyze_technical_indicators")

    # Price trend
    print("\n[Testing Price Trend Analysis]")
    prices = [{"close": 55000 + i * 100, "volume": 1000000} for i in range(30)]
    result = calculate_price_trend(prices)
    assert result["current_price"] > 0
    assert "ma5" in result
    print("[PASS] calculate_price_trend")

    # Quant summary
    print("\n[Testing Quant Result Summary]")
    stock_grade = {
        "final_grade": "A",
        "final_score": 85,
        "scenario_bullish_prob": 40,
        "scenario_sideways_prob": 35,
        "scenario_bearish_prob": 25
    }
    result = summarize_quant_result(stock_grade)
    assert result["scores"]["final_grade"] == "A"
    print("[PASS] summarize_quant_result")

    # Preprocess all
    print("\n[Testing preprocess_all]")
    collected_data = {
        "stock_detail": {"symbol": "005930", "stock_name": "삼성전자"},
        "stock_grade": {"final_grade": "A"},
        "indicators": [{"rsi": 55}],
        "prices": [{"close": 55000, "volume": 1000000}, {"close": 54000, "volume": 900000}],
        "investor_trading": [],
        "foreign_ownership": [],
        "market_index_kospi": [],
        "market_index_kosdaq": [],
        "economic_indicators": []
    }
    result = preprocess_all(collected_data)
    assert "stock_info" in result
    assert "quant_summary" in result
    print("[PASS] preprocess_all")

    print("\n" + "=" * 60)
    print("All preprocessor tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
