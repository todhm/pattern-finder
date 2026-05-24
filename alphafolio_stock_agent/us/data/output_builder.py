"""
Output Builder for V2 Strategy Output (US Stocks)

This module handles Data Injection - filling the 'data' sections of the V2 output template
with values from preprocessed data. The LLM only needs to fill the 'narrative' sections.

Key differences from Korean version:
- Price fields use float (USD with 2 decimals)
- Tick size: $0.01 for >= $1, $0.0001 for < $1
- Stop loss format: "1st: $xxx, 2nd: $xxx" instead of "1차: xxx원, 2차: xxx원"
- Added US-specific fields: iv_percentile, put_call_ratio, insider_signal
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

    # Extract VIX
    vix_data = market_sentiment.get("vix", {})
    vix_value = vix_data.get("value") if isinstance(vix_data, dict) else None
    vix_status = vix_data.get("status") if isinstance(vix_data, dict) else None

    # Extract dollar index
    dollar_data = market_sentiment.get("dollar_index", {})
    dollar_value = dollar_data.get("value") if isinstance(dollar_data, dict) else None

    # Extract credit spread
    credit_data = market_sentiment.get("credit_spread", {})
    credit_value = credit_data.get("value") if isinstance(credit_data, dict) else None

    # Extract rates
    fed_rate_data = us_data.get("fed_funds_rate", {})
    fed_rate = fed_rate_data.get("value") if isinstance(fed_rate_data, dict) else None

    treasury_data = us_data.get("treasury_yield_10y", {})
    treasury_10y = treasury_data.get("value") if isinstance(treasury_data, dict) else None

    # Determine regime
    regime = "neutral"
    if market_regime:
        regime = market_regime.get("regime", "neutral")

    return {
        "vix": vix_value,
        "vix_status": vix_status,
        "fed_rate": fed_rate,
        "treasury_10y": treasury_10y,
        "dollar_index": dollar_value,
        "credit_spread": credit_value,
        "put_call_ratio": None,  # Will be filled from options data if available
        "regime": regime
    }


def build_technical_summary_data(
    technical_summary: dict,
    price_trend: dict,
    options_summary: dict = None,
    insider_summary: dict = None
) -> dict:
    """
    Build the data section for technical summary.

    Args:
        technical_summary: Technical indicators from preprocessor
        price_trend: Price trend data from preprocessor
        options_summary: Options data from preprocessor (US-specific)
        insider_summary: Insider trading data from preprocessor (US-specific)

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
    ma50 = price_trend.get("ma50")
    ma200 = price_trend.get("ma200")

    # Volume
    volume_data = price_trend.get("volume", {})
    volume_ratio = volume_data.get("ratio") if isinstance(volume_data, dict) else None

    # Options data (US-specific)
    iv_percentile = None
    put_call_ratio = None
    if options_summary:
        iv_data = options_summary.get("implied_volatility", {})
        iv_percentile = iv_data.get("percentile") if isinstance(iv_data, dict) else None

        pc_data = options_summary.get("put_call_ratio", {})
        put_call_ratio = pc_data.get("value") if isinstance(pc_data, dict) else None

    # Insider signal (US-specific)
    insider_signal = None
    if insider_summary:
        insider_signal = insider_summary.get("signal")

    return {
        "current_price": round(current_price, 2) if current_price else None,
        "ma5": round(ma5, 2) if ma5 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma50": round(ma50, 2) if ma50 else None,
        "ma200": round(ma200, 2) if ma200 else None,
        "rsi": rsi_value,
        "macd_histogram": macd_histogram,
        "adx": adx_value,
        "bollinger_upper": round(bollinger_upper, 2) if bollinger_upper else None,
        "bollinger_lower": round(bollinger_lower, 2) if bollinger_lower else None,
        "iv_percentile": iv_percentile,
        "put_call_ratio": put_call_ratio,
        "insider_signal": insider_signal,
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
    space_tilde_match = re.match(r'([+-]?\d+\.?\d*)%?\s*~\s*([+-]?\d+\.?\d*)%?', return_str)
    if space_tilde_match:
        try:
            val1 = float(space_tilde_match.group(1))
            val2 = float(space_tilde_match.group(2))
            return (min(val1, val2), max(val1, val2))
        except ValueError:
            pass

    # Strategy 3: Single value format
    single_match = re.match(r'^([+-]?\d+\.?\d*)%?$', return_str)
    if single_match:
        try:
            val = float(single_match.group(1))
            return (val, val)
        except ValueError:
            pass

    # Strategy 4: Regex fallback
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


def round_price(price: float) -> float:
    """
    Round price to US stock tick size.

    US Decimalization:
    - $1.00 and above: $0.01 (2 decimal places)
    - Below $1.00: $0.0001 (4 decimal places, penny stocks)

    Args:
        price: Raw calculated price

    Returns:
        Price rounded to valid tick size
    """
    if price >= 1.0:
        return round(price, 2)
    else:
        return round(price, 4)


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
    - bearish: take_profit = None, stop_loss = "1st: $xxx, 2nd: $xxx"

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

    # Get expected returns for all scenarios
    bullish_return_str = scenarios.get("bullish_return", "")
    sideways_return_str = scenarios.get("sideways_return", "")
    bearish_return_str = scenarios.get("bearish_return", "")

    # Get stop_loss_pct from trading data
    stop_loss_pct = trading.get("stop_loss_pct", -5.0)

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

            take_profit = round_price(current_price * (1 + bull_max / 100))
            stop_loss = round_price(current_price * (1 + stop_loss_pct / 100))

        elif scenario_type == "sideways" and sideways_parsed:
            # Sideways scenario:
            # take_profit: sideways_return max (single value)
            # stop_loss: sideways_return min (single value)
            side_min, side_max = sideways_parsed

            take_profit = round_price(current_price * (1 + side_max / 100))
            stop_loss = round_price(current_price * (1 + side_min / 100))

        elif scenario_type == "bearish" and bearish_parsed:
            # Bearish scenario:
            # take_profit: None (no profit target in bearish)
            # stop_loss: "1st: $xxx, 2nd: $xxx"
            bear_min, bear_max = bearish_parsed

            take_profit = None

            sl_1st = round_price(current_price * (1 + bear_max / 100))
            sl_2nd = round_price(current_price * (1 + bear_min / 100))
            stop_loss = f"1st: ${sl_1st:,.2f}, 2nd: ${sl_2nd:,.2f}"

        else:
            # Fallback: Use Bollinger bands if parsing fails
            if bollinger_upper and bollinger_lower:
                take_profit = round_price(bollinger_upper)
                stop_loss = round_price(bollinger_lower)

    # Support/resistance levels from Bollinger bands
    support_level = round_price(bollinger_lower) if bollinger_lower else None
    resistance_level = round_price(bollinger_upper) if bollinger_upper else None

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
    options_summary = preprocessed_data.get("options_summary", {})
    insider_summary = preprocessed_data.get("insider_summary", {})
    economic_summary = preprocessed_data.get("economic_summary", {})

    # Extract system triggers if not provided
    if system_triggers is None:
        system_triggers = extract_system_triggers(quant_summary)

    # Build data sections
    market_env_data = build_market_environment_data(economic_summary, market_regime)

    # Add put/call ratio from options data to market environment
    if options_summary:
        pc_data = options_summary.get("put_call_ratio", {})
        if isinstance(pc_data, dict):
            market_env_data["put_call_ratio"] = pc_data.get("value")

    tech_summary_data = build_technical_summary_data(
        technical_summary, price_trend, options_summary, insider_summary
    )
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
                "fed_policy": "",  # LLM fills (US-specific: replaces 'domestic')
                "sector": "",  # LLM fills
                "regime_interpretation": ""  # LLM fills
            }
        },

        "technical_summary": {
            "data": tech_summary_data,
            "narrative": {
                "price_trend": "",  # LLM fills
                "indicators": "",  # LLM fills
                "options_flow": "",  # LLM fills (US-specific: replaces 'investor_flow')
                "insider_activity": ""  # LLM fills (US-specific: replaces 'volume_analysis')
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
        price_str = f"${price:,.2f}"
        price_simple = f"${price}"
        if price_str not in tech_narrative["price_trend"] and price_simple not in tech_narrative["price_trend"]:
            warnings.append(f"Current price ${price} not found in price_trend narrative")

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
