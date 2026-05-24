"""
Analysis Agent module
Contains workflow node functions for the LangGraph workflow
"""
import json
import re
from datetime import date
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from kr.config import settings
from kr.state import AgentState, get_execution_time
from kr.schemas import (
    StrategyOutput, validate_strategy_output,
    StrategyOutputV2, validate_strategy_output_v2,
    validate_forbidden_symbols, validate_triggers_from_system
)
from kr.data.collector import collect_stock_data
from kr.data.output_builder import (
    build_output_template, extract_system_triggers, _parse_expected_return,
    _round_to_tick, merge_llm_narrative, merge_llm_narrative_v2
)
from kr.data.preprocessor import preprocess_all
from kr.db.queries import save_strategy
from kr.prompts.analysis import (
    ANALYSIS_SYSTEM_PROMPT,
    format_analysis_prompt,
    format_validation_retry_prompt
)
from kr.utils.debug_saver import (
    save_collect_db, save_preprocessor,
    # DEBUG LOGGING IMPORTS - FOR ROLLBACK: Remove these 4 lines
    save_system_triggers, save_llm_raw_response,
    save_llm_parsed_narrative, save_merged_strategy
)


# =============================================================================
# Node: Collect Data
# =============================================================================

async def collect_data(state: AgentState) -> dict:
    """
    Collect data from PostgreSQL.

    This is the first node in the workflow.
    Fetches all required data for the stock analysis.

    Args:
        state: Current workflow state

    Returns:
        Updated state fields
    """
    try:
        from datetime import date

        symbol = state["symbol"]
        target_date_str = state["target_date"]

        # Parse target date
        target_date = None
        if target_date_str:
            try:
                target_date = date.fromisoformat(target_date_str)
            except ValueError:
                pass

        # Collect data from all tables
        raw_data = await collect_stock_data(symbol, target_date)

        # Extract stock name from stock_detail
        stock_name = ""
        if raw_data.get("stock_detail"):
            stock_name = raw_data["stock_detail"].get("stock_name", "")

        # Save intermediate result for debugging
        try:
            save_collect_db(symbol, raw_data)
        except Exception:
            pass  # Don't fail if debug save fails

        return {
            "raw_data": raw_data,
            "stock_name": stock_name
        }

    except Exception as e:
        return {
            "error": f"Data collection failed: {str(e)}"
        }


# =============================================================================
# Node: Preprocess Data
# =============================================================================

async def preprocess_data(state: AgentState) -> dict:
    """
    Preprocess collected data.

    Transforms raw data into summarized information for the LLM.

    Args:
        state: Current workflow state

    Returns:
        Updated state fields
    """
    try:
        raw_data = state.get("raw_data")

        if not raw_data:
            return {
                "error": "No raw data available for preprocessing"
            }

        # Preprocess all data
        preprocessed_data = preprocess_all(raw_data)

        # Save intermediate result for debugging
        symbol = state.get("symbol", "UNKNOWN")
        try:
            save_preprocessor(symbol, preprocessed_data)
        except Exception:
            pass  # Don't fail if debug save fails

        return {
            "preprocessed_data": preprocessed_data
        }

    except Exception as e:
        return {
            "error": f"Preprocessing failed: {str(e)}"
        }


# =============================================================================
# Node: Analyze
# =============================================================================

async def analyze(state: AgentState) -> dict:
    """
    Run LLM analysis to generate investment strategy.

    Args:
        state: Current workflow state

    Returns:
        Updated state fields with strategy or validation errors
    """
    try:
        preprocessed_data = state.get("preprocessed_data")

        if not preprocessed_data:
            return {
                "error": "No preprocessed data available for analysis"
            }

        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )

        # Check if this is a retry
        retry_count = state.get("retry_count", 0)
        validation_errors = state.get("validation_errors", [])

        # Get market regime result from state (from Market Regime Agent)
        market_regime = state.get("market_regime")

        # Get stock research result from state (from Stock Research Agent)
        stock_research = state.get("stock_research")

        # Extract stock info
        stock_info = preprocessed_data.get("stock_info", {})
        exchange = stock_info.get("exchange", "KOSPI")
        stock_name = stock_info.get("stock_name", "")
        symbol = stock_info.get("symbol", "")

        # Build pre-filled template with system-calculated data
        quant_summary = preprocessed_data.get("quant_summary", {})
        system_triggers = extract_system_triggers(quant_summary)

        # DEBUG LOGGING - FOR ROLLBACK: Remove this block
        try:
            save_system_triggers(symbol, system_triggers)
        except Exception:
            pass

        output_template = build_output_template(
            preprocessed_data=preprocessed_data,
            market_regime=market_regime,
            system_triggers=system_triggers
        )

        # Set analysis date and stock info in template
        output_template["analysis_date"] = state["target_date"]
        output_template["stock_name"] = stock_name
        output_template["symbol"] = symbol

        if retry_count > 0 and validation_errors and state.get("strategy"):
            # Retry: use validation retry prompt
            user_prompt = format_validation_retry_prompt(
                validation_errors=validation_errors,
                previous_output=state["strategy"]
            )
        else:
            # First attempt: use analysis prompt with pre-filled template
            user_prompt = format_analysis_prompt(
                pre_filled_template=output_template,
                market_regime=market_regime,
                stock_research=stock_research,
                system_triggers=system_triggers,
                analysis_date=state["target_date"],
                exchange=exchange,
                stock_name=stock_name,
                symbol=symbol
            )

        # Call LLM
        messages = [
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await llm.ainvoke(messages)

        # DEBUG LOGGING - FOR ROLLBACK: Remove this block
        try:
            save_llm_raw_response(symbol, response.content)
        except Exception:
            pass

        # Parse JSON response (narrative only)
        llm_narrative = parse_llm_response(response.content)

        # DEBUG LOGGING - FOR ROLLBACK: Remove this block
        try:
            save_llm_parsed_narrative(symbol, llm_narrative)
        except Exception:
            pass

        if llm_narrative is None:
            return {
                "validation_errors": ["Failed to parse LLM response as JSON"],
                "retry_count": retry_count + 1
            }

        # Merge system-calculated data with LLM narrative
        strategy = merge_llm_narrative_v2(output_template, llm_narrative)

        # DEBUG LOGGING - FOR ROLLBACK: Remove this block
        try:
            save_merged_strategy(symbol, strategy)
        except Exception:
            pass

        return {
            "strategy": strategy,
            "validation_errors": []  # Clear previous errors
        }

    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}"
        }


def parse_llm_response(content: str) -> dict | None:
    """
    Parse LLM response content as JSON.

    Args:
        content: Raw LLM response content

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        # Try direct JSON parse
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in content
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Node: Validate
# =============================================================================

async def validate(state: AgentState) -> dict:
    """
    Validate the generated strategy.

    Supports both V1 and V2 output formats.
    V2 format has data/narrative separation and stricter validation.

    Args:
        state: Current workflow state

    Returns:
        Updated state fields with validation results
    """
    try:
        strategy = state.get("strategy")
        preprocessed_data = state.get("preprocessed_data", {})

        if not strategy:
            return {
                "validation_errors": ["No strategy to validate"]
            }

        errors = []

        # Check if V2 format
        use_v2 = is_v2_output(strategy)

        if use_v2:
            # V2 validation: data/narrative consistency
            # Extract system_triggers for validation
            quant_summary = preprocessed_data.get("quant_summary", {})
            system_triggers = extract_system_triggers(quant_summary)

            # 1. Schema validation (V2)
            try:
                strategy_output = StrategyOutputV2(**strategy)
                schema_errors = validate_strategy_output_v2(strategy_output, system_triggers)
                errors.extend(schema_errors)
            except Exception as e:
                errors.append(f"V2 Schema validation failed: {str(e)}")

            # 2. Data/Narrative consistency validation (V2)
            if preprocessed_data:
                consistency_errors = _validate_data_consistency_v2(
                    strategy, preprocessed_data, system_triggers
                )
                errors.extend(consistency_errors)
        else:
            # V1 validation: original logic
            # 1. Schema validation
            try:
                strategy_output = StrategyOutput(**strategy)
                schema_errors = validate_strategy_output(strategy_output)
                errors.extend(schema_errors)
            except Exception as e:
                errors.append(f"Schema validation failed: {str(e)}")

            # 2. Data consistency validation (compare with preprocessed_data)
            if preprocessed_data:
                consistency_errors = _validate_data_consistency(strategy, preprocessed_data)
                errors.extend(consistency_errors)

        if errors:
            return {
                "validation_errors": errors,
                "retry_count": state.get("retry_count", 0) + 1
            }

        return {
            "validation_errors": []
        }

    except Exception as e:
        return {
            "validation_errors": [f"Validation error: {str(e)}"],
            "retry_count": state.get("retry_count", 0) + 1
        }


def _validate_data_consistency(strategy: dict, preprocessed_data: dict) -> list[str]:
    """
    Validate that strategy output is consistent with input data.

    Checks:
    1. RSI value matches preprocessed technical indicators
    2. Scenario probabilities match kr_stock_grade (if available)
    3. Support/Resistance levels match Bollinger bands
    4. Take profit/Stop loss match trading parameters
    5. Triggers include system-provided triggers

    Args:
        strategy: LLM-generated strategy output
        preprocessed_data: System-calculated preprocessed data

    Returns:
        List of error messages (empty if consistent)
    """
    errors = []

    # 1. RSI value consistency check
    technical_summary = preprocessed_data.get("technical_summary", {})
    rsi_data = technical_summary.get("rsi", {})
    system_rsi = rsi_data.get("value") if isinstance(rsi_data, dict) else None

    if system_rsi is not None:
        strategy_technical = strategy.get("technical_summary", {})
        strategy_indicators = strategy_technical.get("indicators", {})
        strategy_rsi = strategy_indicators.get("rsi")

        if strategy_rsi is not None:
            # Allow 1.0 tolerance for rounding differences
            if abs(float(strategy_rsi) - float(system_rsi)) > 1.0:
                errors.append(
                    f"RSI value mismatch: strategy has {strategy_rsi}, "
                    f"but input data has {system_rsi}. Use the input data value."
                )

    # 2. Scenario probabilities consistency check (with kr_stock_grade)
    quant_summary = preprocessed_data.get("quant_summary", {})
    scenarios_data = quant_summary.get("scenarios", {})

    if scenarios_data:
        system_bullish = scenarios_data.get("bullish_prob")
        system_bearish = scenarios_data.get("bearish_prob")

        strategy_scenarios = strategy.get("scenarios", {})

        if system_bullish is not None and strategy_scenarios.get("bullish"):
            strategy_bullish = strategy_scenarios["bullish"].get("probability")
            if strategy_bullish is not None and abs(strategy_bullish - system_bullish) > 10:
                errors.append(
                    f"Bullish probability deviation too large: strategy has {strategy_bullish}%, "
                    f"but kr_stock_grade has {system_bullish}%. "
                    f"Deviation should be within 10%."
                )

        if system_bearish is not None and strategy_scenarios.get("bearish"):
            strategy_bearish = strategy_scenarios["bearish"].get("probability")
            if strategy_bearish is not None and abs(strategy_bearish - system_bearish) > 10:
                errors.append(
                    f"Bearish probability deviation too large: strategy has {strategy_bearish}%, "
                    f"but kr_stock_grade has {system_bearish}%. "
                    f"Deviation should be within 10%."
                )

    # 3. Support/Resistance levels consistency check (with Bollinger bands)
    bollinger = technical_summary.get("bollinger", {})
    system_support = bollinger.get("lower")
    system_resistance = bollinger.get("upper")

    if system_support is not None and system_resistance is not None:
        strategy_scenarios = strategy.get("scenarios", {})

        for scenario_name in ["bullish", "sideways", "bearish"]:
            scenario = strategy_scenarios.get(scenario_name, {})

            strategy_support = scenario.get("support_level")
            strategy_resistance = scenario.get("resistance_level")

            # Allow 3% tolerance for support/resistance levels
            if strategy_support is not None:
                tolerance = system_support * 0.03
                if abs(float(strategy_support) - float(system_support)) > tolerance:
                    errors.append(
                        f"{scenario_name} support_level mismatch: strategy has {strategy_support}, "
                        f"but Bollinger lower band is {system_support}. Use Bollinger band value."
                    )

            if strategy_resistance is not None:
                tolerance = system_resistance * 0.03
                if abs(float(strategy_resistance) - float(system_resistance)) > tolerance:
                    errors.append(
                        f"{scenario_name} resistance_level mismatch: strategy has {strategy_resistance}, "
                        f"but Bollinger upper band is {system_resistance}. Use Bollinger band value."
                    )

    # 4. Take profit/Stop loss consistency check
    # New format:
    # - bullish: take_profit = bullish_return max (int), stop_loss = stop_loss_pct (int)
    # - sideways: take_profit = sideways_return max (int), stop_loss = sideways_return min (int)
    # - bearish: take_profit = null, stop_loss = "1차: xxx원, 2차: xxx원" (str)
    price_trend = preprocessed_data.get("price_trend", {})
    current_price = price_trend.get("current_price")
    trading_data = quant_summary.get("trading", {})
    stop_loss_pct = trading_data.get("stop_loss_pct", -5.0)

    if current_price:
        strategy_scenarios = strategy.get("scenarios", {})

        # Get all expected returns for cross-scenario calculations
        bullish_return_str = scenarios_data.get("bullish_return", "")
        sideways_return_str = scenarios_data.get("sideways_return", "")
        bearish_return_str = scenarios_data.get("bearish_return", "")

        bullish_parsed = _parse_expected_return(bullish_return_str)
        sideways_parsed = _parse_expected_return(sideways_return_str)
        bearish_parsed = _parse_expected_return(bearish_return_str)

        for scenario_name in ["bullish", "sideways", "bearish"]:
            scenario = strategy_scenarios.get(scenario_name, {})
            strategy_tp = scenario.get("take_profit")
            strategy_sl = scenario.get("stop_loss")

            # Calculate expected values based on scenario type
            # All prices are rounded to valid tick sizes
            if scenario_name == "bullish" and bullish_parsed:
                bull_min, bull_max = bullish_parsed

                # take_profit: bullish_return max (single value)
                expected_tp = _round_to_tick(current_price * (1 + bull_max / 100))
                # stop_loss: stop_loss_pct (single value)
                expected_sl = _round_to_tick(current_price * (1 + stop_loss_pct / 100))

                # Validate take_profit (single int)
                if strategy_tp is not None:
                    if not _validate_single_price(strategy_tp, expected_tp, tolerance_pct=0.05):
                        errors.append(
                            f"bullish take_profit mismatch: {strategy_tp}, "
                            f"expected {expected_tp:,} (based on {bullish_return_str} max)"
                        )

                # Validate stop_loss (single int)
                if strategy_sl is not None:
                    if not _validate_single_price(strategy_sl, expected_sl, tolerance_pct=0.05):
                        errors.append(
                            f"bullish stop_loss mismatch: {strategy_sl}, "
                            f"expected {expected_sl:,} (based on stop_loss_pct {stop_loss_pct}%)"
                        )

            elif scenario_name == "sideways" and sideways_parsed:
                side_min, side_max = sideways_parsed

                # take_profit: sideways_return max (single value)
                expected_tp = _round_to_tick(current_price * (1 + side_max / 100))
                # stop_loss: sideways_return min (single value)
                expected_sl = _round_to_tick(current_price * (1 + side_min / 100))

                if strategy_tp is not None:
                    if not _validate_single_price(strategy_tp, expected_tp, tolerance_pct=0.05):
                        errors.append(
                            f"sideways take_profit mismatch: {strategy_tp}, "
                            f"expected {expected_tp:,} (based on {sideways_return_str} max)"
                        )

                if strategy_sl is not None:
                    if not _validate_single_price(strategy_sl, expected_sl, tolerance_pct=0.05):
                        errors.append(
                            f"sideways stop_loss mismatch: {strategy_sl}, "
                            f"expected {expected_sl:,} (based on {sideways_return_str} min)"
                        )

            elif scenario_name == "bearish" and bearish_parsed:
                bear_min, bear_max = bearish_parsed

                # bearish take_profit should be None
                if strategy_tp is not None:
                    errors.append(
                        f"bearish take_profit should be null, but got: {strategy_tp}"
                    )

                # bearish stop_loss: "1차: xxx원, 2차: xxx원" format
                if strategy_sl is not None:
                    expected_sl_1st = _round_to_tick(current_price * (1 + bear_max / 100))
                    expected_sl_2nd = _round_to_tick(current_price * (1 + bear_min / 100))
                    if not _validate_bearish_stop_loss(strategy_sl, expected_sl_1st, expected_sl_2nd, tolerance_pct=0.05):
                        errors.append(
                            f"bearish stop_loss mismatch: {strategy_sl}, "
                            f"expected 1차: {expected_sl_1st:,}원, 2차: {expected_sl_2nd:,}원 (based on {bearish_return_str})"
                        )

    # 5. Triggers consistency check
    triggers_data = quant_summary.get("triggers", {})

    if triggers_data:
        # Parse system triggers (stored as JSON strings)
        try:
            system_buy_triggers = json.loads(triggers_data.get("buy", "[]"))
            system_sell_triggers = json.loads(triggers_data.get("sell", "[]"))
        except (json.JSONDecodeError, TypeError):
            system_buy_triggers = []
            system_sell_triggers = []

        strategy_scenarios = strategy.get("scenarios", {})

        # Check bullish scenario triggers include at least one system buy trigger
        bullish_scenario = strategy_scenarios.get("bullish", {})
        bullish_triggers = bullish_scenario.get("triggers", [])

        if system_buy_triggers and bullish_triggers:
            has_system_trigger = any(
                any(sys_keyword in str(trigger) for sys_keyword in _extract_keywords(sys_trigger))
                for trigger in bullish_triggers
                for sys_trigger in system_buy_triggers
            )
            if not has_system_trigger:
                errors.append(
                    f"Bullish triggers should include system-provided triggers. "
                    f"System buy triggers: {system_buy_triggers}"
                )

        # Check bearish scenario triggers include at least one system sell trigger
        bearish_scenario = strategy_scenarios.get("bearish", {})
        bearish_triggers = bearish_scenario.get("triggers", [])

        if system_sell_triggers and bearish_triggers:
            has_system_trigger = any(
                any(sys_keyword in str(trigger) for sys_keyword in _extract_keywords(sys_trigger))
                for trigger in bearish_triggers
                for sys_trigger in system_sell_triggers
            )
            if not has_system_trigger:
                errors.append(
                    f"Bearish triggers should include system-provided triggers. "
                    f"System sell triggers: {system_sell_triggers}"
                )

    return errors


def _extract_keywords(trigger_text: str) -> list[str]:
    """
    Extract key matching terms from a trigger text.

    Args:
        trigger_text: System trigger text

    Returns:
        List of keywords for matching
    """
    keywords = []

    # Extract percentage patterns like "-3.4%", "+5.2%"
    pct_matches = re.findall(r'[+-]?\d+\.?\d*%', trigger_text)
    keywords.extend(pct_matches)

    # Extract score patterns like "56점", "36점"
    score_matches = re.findall(r'\d+점', trigger_text)
    keywords.extend(score_matches)

    # Extract key terms
    key_terms = ["외국인", "기관", "순매수", "순매도", "연속", "손절", "익절", "섹터"]
    for term in key_terms:
        if term in trigger_text:
            keywords.append(term)

    return keywords


def _parse_price_range(range_str: str) -> tuple[int, int] | None:
    """
    Parse price range string like "54,678~60,987" to (min, max) tuple.

    Args:
        range_str: Price range string with comma separators

    Returns:
        Tuple of (min_price, max_price) or None if parsing fails
    """
    if not range_str or not isinstance(range_str, str):
        return None

    try:
        # Remove commas and split by ~
        parts = range_str.replace(",", "").split("~")
        if len(parts) == 2:
            val1 = int(parts[0].strip())
            val2 = int(parts[1].strip())
            return (min(val1, val2), max(val1, val2))
    except (ValueError, AttributeError):
        pass

    return None


def _validate_price_range(
    strategy_range: str,
    expected_min: int,
    expected_max: int,
    tolerance_pct: float = 0.05
) -> bool:
    """
    Validate that strategy price range matches expected range within tolerance.

    Args:
        strategy_range: Strategy's price range string (e.g., "54,678~60,987")
        expected_min: Expected minimum price
        expected_max: Expected maximum price
        tolerance_pct: Tolerance percentage (default 5%)

    Returns:
        True if valid, False otherwise
    """
    parsed = _parse_price_range(strategy_range)
    if not parsed:
        return False

    strategy_min, strategy_max = parsed

    # Calculate tolerances
    min_tolerance = abs(expected_min * tolerance_pct)
    max_tolerance = abs(expected_max * tolerance_pct)

    # Check if both min and max are within tolerance
    min_valid = abs(strategy_min - expected_min) <= min_tolerance
    max_valid = abs(strategy_max - expected_max) <= max_tolerance

    return min_valid and max_valid


def _validate_single_price(
    strategy_price: int | str,
    expected_price: int,
    tolerance_pct: float = 0.05
) -> bool:
    """
    Validate that strategy price matches expected single price within tolerance.

    Args:
        strategy_price: Strategy's price (int or str that can be converted)
        expected_price: Expected price
        tolerance_pct: Tolerance percentage (default 5%)

    Returns:
        True if valid, False otherwise
    """
    try:
        # Convert to int if string
        if isinstance(strategy_price, str):
            strategy_price = int(strategy_price.replace(",", ""))
        else:
            strategy_price = int(strategy_price)
    except (ValueError, TypeError):
        return False

    tolerance = abs(expected_price * tolerance_pct)
    return abs(strategy_price - expected_price) <= tolerance


def _validate_bearish_stop_loss(
    strategy_sl: str,
    expected_1st: int,
    expected_2nd: int,
    tolerance_pct: float = 0.05
) -> bool:
    """
    Validate bearish stop_loss format: "1차: xxx원, 2차: xxx원".

    Args:
        strategy_sl: Strategy's stop loss string
        expected_1st: Expected 1st stop loss price
        expected_2nd: Expected 2nd stop loss price
        tolerance_pct: Tolerance percentage (default 5%)

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(strategy_sl, str):
        return False

    # Parse "1차: xxx원, 2차: xxx원" format
    match = re.search(r'1차:\s*([\d,]+)원.*2차:\s*([\d,]+)원', strategy_sl)
    if not match:
        return False

    try:
        parsed_1st = int(match.group(1).replace(",", ""))
        parsed_2nd = int(match.group(2).replace(",", ""))
    except (ValueError, TypeError):
        return False

    # Validate both prices within tolerance
    tol_1st = abs(expected_1st * tolerance_pct)
    tol_2nd = abs(expected_2nd * tolerance_pct)

    valid_1st = abs(parsed_1st - expected_1st) <= tol_1st
    valid_2nd = abs(parsed_2nd - expected_2nd) <= tol_2nd

    return valid_1st and valid_2nd


# =============================================================================
# V2 Validation Functions
# =============================================================================

def _validate_data_consistency_v2(
    strategy: dict,
    preprocessed_data: dict,
    system_triggers: dict = None
) -> list[str]:
    """
    Validate V2 strategy output for data/narrative consistency.

    Validates:
    1. Numbers in narrative match data section values
    2. Triggers are based on system_triggers
    3. No forbidden symbols in narratives
    4. Bearish strategy follows rules

    Args:
        strategy: LLM-generated V2 strategy output
        preprocessed_data: System-calculated preprocessed data
        system_triggers: System-provided triggers

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Extract system_triggers if not provided
    if system_triggers is None:
        quant_summary = preprocessed_data.get("quant_summary", {})
        system_triggers = extract_system_triggers(quant_summary)

    # 1. Validate RSI in narrative matches data
    tech_summary = strategy.get("technical_summary", {})
    tech_data = tech_summary.get("data", {})
    tech_narrative = tech_summary.get("narrative", {})

    data_rsi = tech_data.get("rsi")
    indicators_text = tech_narrative.get("indicators", "")

    if data_rsi is not None and indicators_text:
        rsi_str = str(round(data_rsi, 2))
        rsi_int = str(int(round(data_rsi)))
        if rsi_str not in indicators_text and rsi_int not in indicators_text:
            errors.append(
                f"RSI value {data_rsi} not found in indicators narrative. "
                f"Narrative must include the exact data value."
            )

    # 2. Validate current price in narrative matches data
    data_price = tech_data.get("current_price")
    price_trend_text = tech_narrative.get("price_trend", "")

    if data_price is not None and price_trend_text:
        price_str = str(data_price)
        price_formatted = f"{data_price:,}"
        if price_str not in price_trend_text and price_formatted not in price_trend_text:
            errors.append(
                f"Current price {data_price} not found in price_trend narrative. "
                f"Narrative must include the exact data value."
            )

    # 3. Validate scenario probabilities in narrative
    scenarios = strategy.get("scenarios", {})
    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario = scenarios.get(scenario_name, {})
        scenario_data = scenario.get("data", {})
        scenario_narrative = scenario.get("narrative", {})

        data_prob = scenario_data.get("probability")
        prob_explanation = scenario_narrative.get("probability_explanation", "")

        if data_prob is not None and prob_explanation:
            prob_str = str(data_prob)
            if prob_str not in prob_explanation:
                errors.append(
                    f"{scenario_name} probability {data_prob}% not found in probability_explanation. "
                    f"Narrative must include the exact data value."
                )

    # 4. Validate triggers against system_triggers
    if system_triggers:
        # Bullish triggers should use system buy triggers
        bullish_triggers = scenarios.get("bullish", {}).get("narrative", {}).get("triggers", [])
        if bullish_triggers:
            trigger_errors = validate_triggers_from_system(bullish_triggers, system_triggers)
            for err in trigger_errors:
                errors.append(f"bullish: {err}")

        # Sideways triggers should use system hold triggers
        sideways_triggers = scenarios.get("sideways", {}).get("narrative", {}).get("triggers", [])
        if sideways_triggers:
            trigger_errors = validate_triggers_from_system(sideways_triggers, system_triggers)
            for err in trigger_errors:
                errors.append(f"sideways: {err}")

        # Bearish triggers should use system sell triggers
        bearish_triggers = scenarios.get("bearish", {}).get("narrative", {}).get("triggers", [])
        if bearish_triggers:
            trigger_errors = validate_triggers_from_system(bearish_triggers, system_triggers)
            for err in trigger_errors:
                errors.append(f"bearish: {err}")

    # 5. Validate no forbidden symbols in all narratives
    all_narratives = _collect_all_narratives(strategy)
    for narrative_text in all_narratives:
        if narrative_text:
            symbol_errors = validate_forbidden_symbols(narrative_text)
            errors.extend(symbol_errors)

    # 6. Validate bearish strategy
    bearish_strategy = scenarios.get("bearish", {}).get("narrative", {}).get("strategy", "")
    if bearish_strategy:
        # Check for forbidden buy keywords
        buy_keywords = ["매수", "추가 매수", "분할 매수", "저점 매수"]
        for keyword in buy_keywords:
            if keyword in bearish_strategy:
                errors.append(f"Bearish strategy contains forbidden keyword: '{keyword}'")

        # Check for required defensive keywords
        defensive_keywords = ["손절", "관망", "현금"]
        has_defensive = any(kw in bearish_strategy for kw in defensive_keywords)
        if not has_defensive:
            errors.append("Bearish strategy must contain at least one of: 손절, 관망, 현금")

    # 7. Validate expert interpretation quality
    expert_errors = _validate_expert_interpretation(all_narratives)
    errors.extend(expert_errors)

    return errors


def _collect_all_narratives(strategy: dict) -> list[str]:
    """
    Collect all narrative text from V2 strategy output.

    Args:
        strategy: V2 strategy output

    Returns:
        List of all narrative strings
    """
    narratives = []

    # Market environment narratives
    me_narrative = strategy.get("market_environment", {}).get("narrative", {})
    narratives.extend([
        me_narrative.get("global_env", ""),
        me_narrative.get("domestic", ""),
        me_narrative.get("sector", ""),
        me_narrative.get("regime_interpretation", "")
    ])

    # Technical summary narratives
    ts_narrative = strategy.get("technical_summary", {}).get("narrative", {})
    narratives.extend([
        ts_narrative.get("price_trend", ""),
        ts_narrative.get("indicators", ""),
        ts_narrative.get("investor_flow", ""),
        ts_narrative.get("volume_analysis", "")
    ])

    # Scenario narratives
    scenarios = strategy.get("scenarios", {})
    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario_narrative = scenarios.get(scenario_name, {}).get("narrative", {})
        narratives.extend([
            scenario_narrative.get("title", ""),
            scenario_narrative.get("probability_explanation", ""),
            scenario_narrative.get("confidence_rationale", ""),
            scenario_narrative.get("strategy", "")
        ])
        narratives.extend(scenario_narrative.get("triggers", []))
        narratives.extend(scenario_narrative.get("monitoring_points", []))
        narratives.extend(scenario_narrative.get("risk_factors", []))

    return narratives


def _validate_expert_interpretation(narratives: list[str]) -> list[str]:
    """
    Validate expert interpretation quality in narratives.

    Checks:
    1. No simple listing patterns (just stating values without interpretation)
    2. Contains causal connectors (therefore, indicates, means, etc.)
    3. Mentions multiple indicators together (composite analysis)

    Args:
        narratives: List of narrative texts to validate

    Returns:
        List of error messages (empty if quality is acceptable)
    """
    errors = []

    # Combine all narratives for analysis
    combined_text = " ".join(n for n in narratives if n)

    if not combined_text:
        return errors

    # 1. Detect simple listing patterns (warning level)
    simple_patterns = [
        r"[\d.]+로\s*(중립|과매수|과매도)\s*(수준)?입니다\.?\s*$",
        r"[\d]+일\s*연속\s*(순매수|순매도)(하고 있습니다|입니다)\.?\s*$",
        r"[\d,]+원(입니다|으로)\.?\s*$",
    ]

    simple_count = 0
    for pattern in simple_patterns:
        matches = re.findall(pattern, combined_text, re.MULTILINE)
        simple_count += len(matches)

    # If more than 3 simple patterns detected, flag as low quality
    if simple_count > 3:
        errors.append(
            f"Expert interpretation quality low: {simple_count} simple listing patterns detected. "
            f"Use INTERPRETATION_FRAMEWORK to provide composite analysis with causal connections."
        )

    # 2. Check for causal connectors (at least one required)
    causal_words = [
        "따라서", "이로 인해", "시사합니다", "의미합니다",
        "때문에", "결과로", "나타냅니다", "보여줍니다",
        "해석됩니다", "판단됩니다", "예상됩니다", "전망됩니다"
    ]

    has_causal = any(word in combined_text for word in causal_words)
    if not has_causal:
        errors.append(
            "Expert interpretation missing causal connectors. "
            "Include interpretation words like: 시사합니다, 의미합니다, 따라서, 때문에"
        )

    # 3. Check for composite indicator analysis (at least 2 indicators mentioned together)
    indicator_terms = [
        "RSI", "MACD", "ADX", "볼린저", "이동평균",
        "순매수", "순매도", "외국인", "기관", "거래량"
    ]

    mentioned_indicators = [ind for ind in indicator_terms if ind in combined_text]

    if len(mentioned_indicators) < 3:
        errors.append(
            f"Expert interpretation lacks composite analysis. "
            f"Only {len(mentioned_indicators)} indicators mentioned: {mentioned_indicators}. "
            f"Combine at least 3 indicators for expert-level analysis."
        )

    return errors


def is_v2_output(strategy: dict) -> bool:
    """
    Check if strategy output is V2 format (has data/narrative separation).

    Args:
        strategy: Strategy output to check

    Returns:
        True if V2 format, False otherwise
    """
    # V2 format has nested data/narrative structure
    tech_summary = strategy.get("technical_summary", {})
    return "data" in tech_summary and "narrative" in tech_summary


# =============================================================================
# Node: Finalize
# =============================================================================

async def finalize(state: AgentState) -> dict:
    """
    Finalize the workflow and save results.

    Args:
        state: Current workflow state

    Returns:
        Updated state fields with final metadata
    """
    try:
        strategy = state.get("strategy")

        if not strategy:
            return {
                "error": "No strategy to finalize"
            }

        # Update metadata
        execution_time = get_execution_time(state)

        if "metadata" not in strategy:
            strategy["metadata"] = {}

        strategy["metadata"]["execution_time_sec"] = execution_time
        strategy["metadata"]["retry_count"] = state.get("retry_count", 0)

        # Get data freshness from raw data
        raw_data = state.get("raw_data", {})
        data_freshness = {}

        if raw_data.get("stock_grade"):
            grade_date = raw_data["stock_grade"].get("date")
            if grade_date:
                data_freshness["kr_stock_grade"] = str(grade_date)

        if raw_data.get("market_index_kospi"):
            index_date = raw_data["market_index_kospi"][0].get("date") if raw_data["market_index_kospi"] else None
            if index_date:
                data_freshness["market_index"] = str(index_date)

        if raw_data.get("indicators"):
            indicator_date = raw_data["indicators"][0].get("date") if raw_data["indicators"] else None
            if indicator_date:
                data_freshness["indicators"] = str(indicator_date)

        strategy["metadata"]["data_freshness"] = data_freshness

        # Save to database
        symbol = state["symbol"]
        target_date_str = state.get("target_date")
        analysis_date = date.fromisoformat(target_date_str) if target_date_str else date.today()
        await save_strategy(symbol, strategy, analysis_date)

        return {
            "strategy": strategy
        }

    except Exception as e:
        return {
            "error": f"Finalization failed: {str(e)}"
        }


# =============================================================================
# Conditional Edge Functions
# =============================================================================

def should_retry(state: AgentState) -> str:
    """
    Determine next node after validation.

    Args:
        state: Current workflow state

    Returns:
        Next node name: "analyze" (retry), "finalize" (success), or "error" (max retries)
    """
    validation_errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)

    if not validation_errors:
        return "finalize"

    if retry_count < 2:
        return "analyze"

    return "error"


def check_error(state: AgentState) -> str:
    """
    Check if workflow has error and determine next node.

    Args:
        state: Current workflow state

    Returns:
        Next node name or "end" if error
    """
    if state.get("error"):
        return "end"
    return "continue"
