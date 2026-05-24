"""
Output Builder for V2 Strategy Output

This module handles Data Injection - filling the 'data' sections of the V2 output template
with values from preprocessed data. The LLM only needs to fill the 'narrative' sections.

Based on: docs/에이전트 결과물 개선 방안_v2.md Section 3.4
"""
import json
import re
from typing import Optional


def extract_system_triggers(quant_summary: dict) -> dict:
    """
    Extract system triggers from quant_summary.

    Args:
        quant_summary: Quant analysis summary containing triggers

    Returns:
        System triggers dictionary with buy, sell, hold categories
    """
    triggers = quant_summary.get("triggers", {})

    def parse_trigger(trigger_value):
        """Parse trigger value which may be JSON string or list"""
        if isinstance(trigger_value, str):
            try:
                return json.loads(trigger_value)
            except json.JSONDecodeError:
                return [trigger_value] if trigger_value else []
        elif isinstance(trigger_value, list):
            return trigger_value
        return []

    return {
        "buy": parse_trigger(triggers.get("buy", [])),
        "sell": parse_trigger(triggers.get("sell", [])),
        "hold": parse_trigger(triggers.get("hold", []))
    }


def build_market_environment_data(
    economic_summary: dict,
    market_regime: dict = None
) -> dict:
    """
    Build the data section for market environment.

    Args:
        economic_summary: Economic indicators from preprocessor
        market_regime: Market Regime Agent result (optional)

    Returns:
        MarketEnvironmentData dictionary
    """
    market_sentiment = economic_summary.get("market_sentiment", {})
    us_data = economic_summary.get("us", {})
    korea_data = economic_summary.get("korea", {})

    # Extract VIX
    vix_data = market_sentiment.get("vix", {})
    vix_value = vix_data.get("value") if isinstance(vix_data, dict) else None
    vix_status = vix_data.get("status") if isinstance(vix_data, dict) else None

    # Extract dollar index
    dollar_data = market_sentiment.get("dollar_index", {})
    dollar_value = dollar_data.get("value") if isinstance(dollar_data, dict) else None

    # Extract rates
    fed_rate_data = us_data.get("fed_funds_rate", {})
    fed_rate = fed_rate_data.get("value") if isinstance(fed_rate_data, dict) else None

    korea_rate_data = korea_data.get("base_rate", {})
    korea_rate = korea_rate_data.get("value") if isinstance(korea_rate_data, dict) else None

    # Determine regime
    regime = "neutral"
    if market_regime:
        regime = market_regime.get("regime", "neutral")

    return {
        "vix": vix_value,
        "vix_status": vix_status,
        "fed_rate": fed_rate,
        "korea_base_rate": korea_rate,
        "dollar_index": dollar_value,
        "usd_krw": None,  # Can be added if available in preprocessor
        "regime": regime
    }


def build_technical_summary_data(
    technical_summary: dict,
    price_trend: dict,
    investor_summary: dict
) -> dict:
    """
    Build the data section for technical summary.

    Args:
        technical_summary: Technical indicators from preprocessor
        price_trend: Price trend data from preprocessor
        investor_summary: Investor trading data from preprocessor

    Returns:
        TechnicalSummaryData dictionary
    """
    # RSI
    rsi_data = technical_summary.get("rsi", {})
    rsi_value = rsi_data.get("value") if isinstance(rsi_data, dict) else None

    # MACD
    macd_data = technical_summary.get("macd", {})
    macd_histogram = macd_data.get("histogram") if isinstance(macd_data, dict) else None

    # ADX
    adx_data = technical_summary.get("adx", {})
    adx_value = adx_data.get("value") if isinstance(adx_data, dict) else None

    # Bollinger Bands
    bollinger_data = technical_summary.get("bollinger", {})
    bollinger_upper = bollinger_data.get("upper") if isinstance(bollinger_data, dict) else None
    bollinger_lower = bollinger_data.get("lower") if isinstance(bollinger_data, dict) else None

    # Price data
    current_price = price_trend.get("current_price")
    ma5 = price_trend.get("ma5")
    ma20 = price_trend.get("ma20")
    ma60 = price_trend.get("ma60")

    # Volume
    volume_data = price_trend.get("volume", {})
    volume_ratio = volume_data.get("ratio") if isinstance(volume_data, dict) else None

    # Investor data
    foreign_data = investor_summary.get("foreign", {})
    foreign_consecutive = foreign_data.get("consecutive_buy_days") if isinstance(foreign_data, dict) else None

    institutional_data = investor_summary.get("institutional", {})
    institutional_consecutive = institutional_data.get("consecutive_buy_days") if isinstance(institutional_data, dict) else None

    return {
        "current_price": _round_to_tick(current_price) if current_price else None,
        "ma5": _round_to_tick(ma5) if ma5 else None,
        "ma20": _round_to_tick(ma20) if ma20 else None,
        "ma60": _round_to_tick(ma60) if ma60 else None,
        "rsi": rsi_value,
        "macd_histogram": macd_histogram,
        "adx": adx_value,
        "bollinger_upper": _round_to_tick(bollinger_upper) if bollinger_upper else None,
        "bollinger_lower": _round_to_tick(bollinger_lower) if bollinger_lower else None,
        "foreign_consecutive_buy": foreign_consecutive,
        "institutional_consecutive_buy": institutional_consecutive,
        "volume_ratio": volume_ratio
    }


def _parse_expected_return(return_str: str) -> tuple[float, float] | None:
    """
    Parse expected return string (VARCHAR(20) format from DB).

    Supports multiple formats with fallback strategies:
    1. "+15~+39%" -> (15.0, 39.0)
    2. "-13~-22%" -> (-22.0, -13.0) sorted
    3. "15~39%" -> (15.0, 39.0)
    4. "+15% ~ +39%" with spaces -> (15.0, 39.0)
    5. Single value "+15%" -> (15.0, 15.0)
    6. Regex fallback: extract all number patterns

    Args:
        return_str: Expected return string (e.g., "+15~+39%", "-13~-22%")

    Returns:
        Tuple of (min_pct, max_pct) sorted, or None if parsing fails
    """
    if not return_str or not isinstance(return_str, str):
        return None

    return_str = return_str.strip()

    # Strategy 1: Standard format with ~ separator
    # Handles: "+15~+39%", "-13~-22%", "15~39%"
    if "~" in return_str:
        parts = return_str.replace("%", "").split("~")
        if len(parts) == 2:
            try:
                val1 = float(parts[0].strip())
                val2 = float(parts[1].strip())
                return (min(val1, val2), max(val1, val2))
            except ValueError:
                pass

    # Strategy 2: Format with spaces and ~ separator
    # Handles: "+15% ~ +39%", "15% ~ 39%"
    space_tilde_match = re.match(r'([+-]?\d+\.?\d*)%?\s*~\s*([+-]?\d+\.?\d*)%?', return_str)
    if space_tilde_match:
        try:
            val1 = float(space_tilde_match.group(1))
            val2 = float(space_tilde_match.group(2))
            return (min(val1, val2), max(val1, val2))
        except ValueError:
            pass

    # Strategy 3: Single value format
    # Handles: "+15%", "-10%", "20%"
    single_match = re.match(r'^([+-]?\d+\.?\d*)%?$', return_str)
    if single_match:
        try:
            val = float(single_match.group(1))
            return (val, val)
        except ValueError:
            pass

    # Strategy 4: Regex fallback - extract all percentage-like numbers
    # Handles any format with numbers
    numbers = re.findall(r'[+-]?\d+\.?\d*', return_str)
    if len(numbers) >= 2:
        try:
            vals = [float(n) for n in numbers[:2]]
            return (min(vals), max(vals))
        except ValueError:
            pass
    elif len(numbers) == 1:
        try:
            val = float(numbers[0])
            return (val, val)
        except ValueError:
            pass

    return None


def _get_tick_size(price: int) -> int:
    """
    Get tick size (호가 단위) based on stock price.

    Korean stock tick sizes:
    - Below 2,000: 1
    - 2,000 ~ 5,000: 5
    - 5,000 ~ 20,000: 10
    - 20,000 ~ 50,000: 50
    - 50,000 ~ 200,000: 100
    - 200,000 ~ 500,000: 500
    - Above 500,000: 1,000

    Args:
        price: Stock price in KRW

    Returns:
        Tick size for the given price
    """
    if price < 2000:
        return 1
    elif price < 5000:
        return 5
    elif price < 20000:
        return 10
    elif price < 50000:
        return 50
    elif price < 200000:
        return 100
    elif price < 500000:
        return 500
    else:
        return 1000


def _round_to_tick(price: float) -> int:
    """
    Round price to the nearest valid tick size (호가 단위).

    Args:
        price: Raw calculated price

    Returns:
        Price rounded to valid tick size
    """
    price_int = int(price)
    tick = _get_tick_size(price_int)
    return round(price_int / tick) * tick


def _format_price_range(price_min: float, price_max: float) -> str:
    """
    Format price range as string with comma separators.
    Prices are rounded to valid tick sizes (호가 단위).

    Args:
        price_min: Minimum price
        price_max: Maximum price

    Returns:
        Formatted range string like "54,700~61,000"
    """
    rounded_min = _round_to_tick(price_min)
    rounded_max = _round_to_tick(price_max)
    return f"{rounded_min:,}~{rounded_max:,}"


def build_scenario_data(
    quant_summary: dict,
    technical_summary: dict,
    price_trend: dict,
    scenario_type: str
) -> dict:
    """
    Build the data section for a scenario.

    take_profit and stop_loss calculation:
    - bullish: take_profit = bullish_return max, stop_loss = stop_loss_pct
    - sideways: take_profit = sideways_return max, stop_loss = sideways_return min
    - bearish: take_profit = None, stop_loss = "1st: bearish_return max, 2nd: bearish_return min"

    Args:
        quant_summary: Quant analysis summary from preprocessor
        technical_summary: Technical indicators from preprocessor
        price_trend: Price trend data from preprocessor
        scenario_type: 'bullish', 'sideways', or 'bearish'

    Returns:
        ScenarioData dictionary with take_profit/stop_loss
    """
    scenarios = quant_summary.get("scenarios", {})
    trading = quant_summary.get("trading", {})

    # Get probability based on scenario type
    prob_key = f"{scenario_type}_prob"
    probability = scenarios.get(prob_key, 0)

    # Get expected returns for all scenarios (needed for cross-scenario calculations)
    bullish_return_str = scenarios.get("bullish_return", "")
    sideways_return_str = scenarios.get("sideways_return", "")
    bearish_return_str = scenarios.get("bearish_return", "")

    # Get stop_loss_pct from trading data
    stop_loss_pct = trading.get("stop_loss_pct", -5.0)  # Default -5% if not available

    # Current scenario's expected return
    return_key = f"{scenario_type}_return"
    expected_return = scenarios.get(return_key, "")

    # Sample count
    sample_count = scenarios.get("sample_count")

    # Bollinger bands for support/resistance
    bollinger_data = technical_summary.get("bollinger", {})
    bollinger_upper = bollinger_data.get("upper") if isinstance(bollinger_data, dict) else None
    bollinger_lower = bollinger_data.get("lower") if isinstance(bollinger_data, dict) else None

    # Current price for calculations
    current_price = price_trend.get("current_price", 0)

    # Parse expected returns
    bullish_parsed = _parse_expected_return(bullish_return_str)
    sideways_parsed = _parse_expected_return(sideways_return_str)
    bearish_parsed = _parse_expected_return(bearish_return_str)

    take_profit = None
    stop_loss = None

    if current_price:
        if scenario_type == "bullish" and bullish_parsed:
            # Bullish scenario:
            # take_profit: bullish_return max (single value)
            # stop_loss: stop_loss_pct (single value)
            bull_min, bull_max = bullish_parsed

            take_profit = _round_to_tick(current_price * (1 + bull_max / 100))
            stop_loss = _round_to_tick(current_price * (1 + stop_loss_pct / 100))

        elif scenario_type == "sideways" and sideways_parsed:
            # Sideways scenario:
            # take_profit: sideways_return max (single value)
            # stop_loss: sideways_return min (single value)
            side_min, side_max = sideways_parsed

            take_profit = _round_to_tick(current_price * (1 + side_max / 100))
            stop_loss = _round_to_tick(current_price * (1 + side_min / 100))

        elif scenario_type == "bearish" and bearish_parsed:
            # Bearish scenario:
            # take_profit: None (no profit target in bearish)
            # stop_loss: "1st: bearish_return max, 2nd: bearish_return min"
            bear_min, bear_max = bearish_parsed

            take_profit = None

            sl_1st = _round_to_tick(current_price * (1 + bear_max / 100))
            sl_2nd = _round_to_tick(current_price * (1 + bear_min / 100))
            stop_loss = f"1차: {sl_1st:,}원, 2차: {sl_2nd:,}원"

        else:
            # Fallback: Use Bollinger bands if parsing fails
            if bollinger_upper and bollinger_lower:
                take_profit = _round_to_tick(bollinger_upper)
                stop_loss = _round_to_tick(bollinger_lower)

    # Support/resistance levels from Bollinger bands (rounded to tick size)
    support_level = _round_to_tick(bollinger_lower) if bollinger_lower else None
    resistance_level = _round_to_tick(bollinger_upper) if bollinger_upper else None

    return {
        "probability": probability,
        "support_level": support_level,
        "resistance_level": resistance_level,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "expected_return": expected_return,
        "sample_count": sample_count
    }


def build_output_template(
    preprocessed_data: dict,
    market_regime: dict = None,
    system_triggers: dict = None
) -> dict:
    """
    Build the complete V2 output template with data sections filled.
    LLM only needs to fill the narrative sections.

    Args:
        preprocessed_data: Result from preprocessor.preprocess_all()
        market_regime: Market Regime Agent result (optional)
        system_triggers: System triggers (extracted from quant_summary if not provided)

    Returns:
        Output template with data sections filled, narrative sections empty
    """
    # Extract components from preprocessed data
    stock_info = preprocessed_data.get("stock_info", {})
    quant_summary = preprocessed_data.get("quant_summary", {})
    technical_summary = preprocessed_data.get("technical_summary", {})
    price_trend = preprocessed_data.get("price_trend", {})
    investor_summary = preprocessed_data.get("investor_summary", {})
    economic_summary = preprocessed_data.get("economic_summary", {})

    # Extract system triggers if not provided
    if system_triggers is None:
        system_triggers = extract_system_triggers(quant_summary)

    # Build data sections
    market_env_data = build_market_environment_data(economic_summary, market_regime)
    tech_summary_data = build_technical_summary_data(technical_summary, price_trend, investor_summary)
    bullish_data = build_scenario_data(quant_summary, technical_summary, price_trend, "bullish")
    sideways_data = build_scenario_data(quant_summary, technical_summary, price_trend, "sideways")
    bearish_data = build_scenario_data(quant_summary, technical_summary, price_trend, "bearish")

    return {
        "stock_name": stock_info.get("stock_name", ""),
        "symbol": stock_info.get("symbol", ""),
        "analysis_date": "",  # To be filled by caller
        "final_grade": "",  # To be determined by LLM based on probabilities

        "market_environment": {
            "data": market_env_data,
            "narrative": {
                "global_env": "",  # LLM fills
                "domestic": "",  # LLM fills
                "sector": "",  # LLM fills
                "regime_interpretation": ""  # LLM fills
            }
        },

        "technical_summary": {
            "data": tech_summary_data,
            "narrative": {
                "price_trend": "",  # LLM fills
                "indicators": "",  # LLM fills
                "investor_flow": "",  # LLM fills
                "volume_analysis": ""  # LLM fills
            }
        },

        "scenarios": {
            "bullish": {
                "data": bullish_data,
                "narrative": {
                    "title": "",  # LLM fills
                    "probability_explanation": "",  # LLM fills
                    "confidence_rationale": "",  # LLM fills
                    "strategy": "",  # LLM fills
                    "triggers": [],  # LLM fills based on system_triggers.buy
                    "monitoring_points": [],  # LLM fills
                    "risk_factors": []  # LLM fills
                }
            },
            "sideways": {
                "data": sideways_data,
                "narrative": {
                    "title": "",
                    "probability_explanation": "",
                    "confidence_rationale": "",
                    "strategy": "",
                    "triggers": [],  # LLM fills based on system_triggers.hold
                    "monitoring_points": [],
                    "risk_factors": []
                }
            },
            "bearish": {
                "data": bearish_data,
                "narrative": {
                    "title": "",
                    "probability_explanation": "",
                    "confidence_rationale": "",
                    "strategy": "",  # Must contain defensive keywords
                    "triggers": [],  # LLM fills based on system_triggers.sell
                    "monitoring_points": [],
                    "risk_factors": []
                }
            }
        },

        "system_triggers": system_triggers,  # Passed to LLM for reference

        "metadata": {
            "agent_version": "2.0.0",
            "execution_time_sec": None,
            "retry_count": 0,
            "data_freshness": None
        }
    }


def merge_llm_narrative(template: dict, llm_output: dict) -> dict:
    """
    Merge LLM-generated narrative into the template.
    Preserves system-filled data sections and only updates narrative sections.

    Args:
        template: Output template with data sections filled
        llm_output: LLM output containing narrative sections

    Returns:
        Merged output with both data and narrative
    """
    result = template.copy()

    # Update final_grade from LLM
    if "final_grade" in llm_output:
        result["final_grade"] = llm_output["final_grade"]

    # Update analysis_date if provided
    if "analysis_date" in llm_output:
        result["analysis_date"] = llm_output["analysis_date"]

    # Merge market_environment narrative
    if "market_environment" in llm_output:
        me_output = llm_output["market_environment"]
        if "narrative" in me_output:
            result["market_environment"]["narrative"] = me_output["narrative"]

    # Merge technical_summary narrative
    if "technical_summary" in llm_output:
        ts_output = llm_output["technical_summary"]
        if "narrative" in ts_output:
            result["technical_summary"]["narrative"] = ts_output["narrative"]

    # Merge scenario narratives
    if "scenarios" in llm_output:
        for scenario_name in ["bullish", "sideways", "bearish"]:
            if scenario_name in llm_output["scenarios"]:
                scenario_output = llm_output["scenarios"][scenario_name]
                if "narrative" in scenario_output:
                    result["scenarios"][scenario_name]["narrative"] = scenario_output["narrative"]

    # Update metadata
    if "metadata" in llm_output:
        for key in ["execution_time_sec", "retry_count"]:
            if key in llm_output["metadata"]:
                result["metadata"][key] = llm_output["metadata"][key]

    # Remove system_triggers from final output (internal use only)
    if "system_triggers" in result:
        del result["system_triggers"]

    return result


def merge_llm_narrative_v2(template: dict, llm_output: dict) -> dict:
    """
    Merge LLM-generated narrative (v2 format) into the template.
    Preserves system-filled data sections and only updates narrative sections.

    V2 format: LLM outputs flat narrative structure (market_environment_narrative, etc.)
    This function maps it to the nested structure (market_environment.narrative, etc.)

    Args:
        template: Output template with data sections filled (from build_output_template)
        llm_output: LLM output containing narrative sections only

    Returns:
        Merged output with both data and narrative
    """
    result = template.copy()

    # Update final_grade from LLM
    if "final_grade" in llm_output:
        result["final_grade"] = llm_output["final_grade"]

    # Merge market_environment narrative (v2 format: flat -> nested)
    if "market_environment_narrative" in llm_output:
        result["market_environment"]["narrative"] = llm_output["market_environment_narrative"]

    # Merge technical_summary narrative (v2 format: flat -> nested)
    if "technical_summary_narrative" in llm_output:
        result["technical_summary"]["narrative"] = llm_output["technical_summary_narrative"]

    # Merge scenario narratives (v2 format: flat -> nested)
    if "scenarios_narrative" in llm_output:
        for scenario_name in ["bullish", "sideways", "bearish"]:
            if scenario_name in llm_output["scenarios_narrative"]:
                result["scenarios"][scenario_name]["narrative"] = llm_output["scenarios_narrative"][scenario_name]

    # Remove system_triggers from final output (internal use only)
    if "system_triggers" in result:
        del result["system_triggers"]

    return result


def validate_data_narrative_consistency(output: dict) -> list[str]:
    """
    Validate that narrative sections reference data values correctly.

    This is a helper function to check if the LLM properly used the
    data values when writing narratives.

    Args:
        output: Merged output with both data and narrative

    Returns:
        List of warning messages (not errors, for soft validation)
    """
    warnings = []

    # Check technical summary
    tech_data = output.get("technical_summary", {}).get("data", {})
    tech_narrative = output.get("technical_summary", {}).get("narrative", {})

    # Check if RSI value appears in indicators narrative
    rsi = tech_data.get("rsi")
    if rsi and tech_narrative.get("indicators"):
        rsi_str = str(round(rsi, 2))
        if rsi_str not in tech_narrative["indicators"]:
            warnings.append(f"RSI value {rsi} not found in indicators narrative")

    # Check if current price appears in price_trend narrative
    price = tech_data.get("current_price")
    if price and tech_narrative.get("price_trend"):
        price_str = str(price)
        price_formatted = f"{price:,}"
        if price_str not in tech_narrative["price_trend"] and price_formatted not in tech_narrative["price_trend"]:
            warnings.append(f"Current price {price} not found in price_trend narrative")

    # Check scenario probabilities
    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario = output.get("scenarios", {}).get(scenario_name, {})
        data = scenario.get("data", {})
        narrative = scenario.get("narrative", {})

        prob = data.get("probability")
        prob_exp = narrative.get("probability_explanation", "")

        if prob and prob_exp:
            if str(prob) not in prob_exp:
                warnings.append(f"{scenario_name} probability {prob}% not found in probability_explanation")

    return warnings
