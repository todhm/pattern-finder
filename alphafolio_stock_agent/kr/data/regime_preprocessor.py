"""
Regime Preprocessor Module
Transforms raw market data into interpreted signals for Market Regime Agent.
No LLM involved - pure Python data interpretation.

This module follows the design principle:
- LLM should NOT interpret raw data
- System (Python) interprets data into signals
- LLM only combines signals to make final regime determination
"""
from typing import Optional
from datetime import datetime


# =============================================================================
# Volatility Index Interpretation (VKOSPI, US VIX)
# =============================================================================

def interpret_vkospi(value: float, prev_value: Optional[float] = None) -> dict:
    """
    Interpret VKOSPI (Korea VIX) value into signal.

    VKOSPI typical range: 15-45
    - < 15: Very low fear (excessive optimism, potential correction)
    - 15-20: Low fear (Risk-On environment)
    - 20-25: Moderate (Neutral)
    - 25-30: High fear (Risk-Off signal)
    - > 30: Extreme fear (Strong Risk-Off)

    Args:
        value: Current VKOSPI value
        prev_value: Previous period value for trend calculation

    Returns:
        dict with signal, interpretation, and metadata
    """
    if value is None:
        return {"signal": "no_data", "interpretation": "VKOSPI data unavailable"}

    value = float(value)

    # Determine signal level
    if value < 15:
        signal = "very_low_fear"
        interpretation = "Excessive optimism, potential correction risk"
        regime_implication = "risk_on_caution"
    elif value < 20:
        signal = "low_fear"
        interpretation = "Low fear, Risk-On environment"
        regime_implication = "risk_on"
    elif value < 25:
        signal = "moderate"
        interpretation = "Normal volatility level, Neutral"
        regime_implication = "neutral"
    elif value < 30:
        signal = "high_fear"
        interpretation = "Elevated fear, Risk-Off signal"
        regime_implication = "risk_off"
    else:
        signal = "extreme_fear"
        interpretation = "Extreme fear, Strong Risk-Off"
        regime_implication = "risk_off_strong"

    result = {
        "indicator": "VKOSPI",
        "value": round(value, 2),
        "signal": signal,
        "interpretation": interpretation,
        "regime_implication": regime_implication
    }

    # Add trend if previous value available
    if prev_value is not None:
        prev_value = float(prev_value)
        change = value - prev_value
        change_pct = (change / prev_value) * 100 if prev_value != 0 else 0

        if change_pct > 10:
            trend = "spiking"
            trend_interpretation = "Volatility spiking, fear increasing rapidly"
        elif change_pct > 5:
            trend = "rising"
            trend_interpretation = "Volatility rising, caution warranted"
        elif change_pct < -10:
            trend = "collapsing"
            trend_interpretation = "Volatility collapsing, fear subsiding rapidly"
        elif change_pct < -5:
            trend = "falling"
            trend_interpretation = "Volatility falling, fear subsiding"
        else:
            trend = "stable"
            trend_interpretation = "Volatility stable"

        result["trend"] = trend
        result["trend_interpretation"] = trend_interpretation
        result["change"] = round(change, 2)
        result["change_pct"] = round(change_pct, 2)

    return result


def interpret_vix(value: float, prev_value: Optional[float] = None) -> dict:
    """
    Interpret US VIX value into signal.
    Same logic as VKOSPI - both are volatility indices with similar ranges.
    """
    if value is None:
        return {"signal": "no_data", "interpretation": "VIX data unavailable"}

    value = float(value)

    if value < 15:
        signal = "very_low_fear"
        interpretation = "Excessive optimism in US market"
        regime_implication = "risk_on_caution"
    elif value < 20:
        signal = "low_fear"
        interpretation = "Low fear in US market, Risk-On"
        regime_implication = "risk_on"
    elif value < 25:
        signal = "moderate"
        interpretation = "Normal US volatility, Neutral"
        regime_implication = "neutral"
    elif value < 30:
        signal = "high_fear"
        interpretation = "Elevated US fear, Risk-Off signal"
        regime_implication = "risk_off"
    else:
        signal = "extreme_fear"
        interpretation = "Extreme US fear, Strong Risk-Off"
        regime_implication = "risk_off_strong"

    result = {
        "indicator": "VIX",
        "value": round(value, 2),
        "signal": signal,
        "interpretation": interpretation,
        "regime_implication": regime_implication
    }

    if prev_value is not None:
        prev_value = float(prev_value)
        change = value - prev_value
        change_pct = (change / prev_value) * 100 if prev_value != 0 else 0

        if change_pct > 10:
            trend = "spiking"
        elif change_pct > 5:
            trend = "rising"
        elif change_pct < -10:
            trend = "collapsing"
        elif change_pct < -5:
            trend = "falling"
        else:
            trend = "stable"

        result["trend"] = trend
        result["change"] = round(change, 2)
        result["change_pct"] = round(change_pct, 2)

    return result


# =============================================================================
# Credit Spread Interpretation
# =============================================================================

def interpret_credit_spread(current: float, prev: Optional[float] = None) -> dict:
    """
    Interpret credit spread (HY-IG spread) change.

    Credit spread widening = Risk-Off (credit risk increasing)
    Credit spread tightening = Risk-On (credit risk decreasing)

    Args:
        current: Current spread value (basis points or percentage)
        prev: Previous period spread value
    """
    if current is None:
        return {"signal": "no_data", "interpretation": "Credit spread data unavailable"}

    current = float(current)
    result = {
        "indicator": "Credit Spread",
        "value": round(current, 2)
    }

    if prev is not None:
        prev = float(prev)
        change = current - prev

        if change < -0.1:
            signal = "tightening"
            interpretation = "Spread tightening, credit risk decreasing, Risk-On"
            regime_implication = "risk_on"
        elif change > 0.1:
            signal = "widening"
            interpretation = "Spread widening, credit risk increasing, Risk-Off"
            regime_implication = "risk_off"
        else:
            signal = "stable"
            interpretation = "Spread stable, Neutral"
            regime_implication = "neutral"

        result["signal"] = signal
        result["interpretation"] = interpretation
        result["regime_implication"] = regime_implication
        result["change"] = round(change, 3)
    else:
        result["signal"] = "level_only"
        result["interpretation"] = f"Current spread: {current:.2f}"
        result["regime_implication"] = "neutral"

    return result


# =============================================================================
# Dollar Index Interpretation
# =============================================================================

def interpret_dollar_index(current: float, ma20: Optional[float] = None) -> dict:
    """
    Interpret US Dollar Index trend.

    Dollar strengthening = Risk-Off (flight to safety, EM outflow concern)
    Dollar weakening = Risk-On (EM inflow expectation)

    Args:
        current: Current dollar index value
        ma20: 20-day moving average for trend comparison
    """
    if current is None:
        return {"signal": "no_data", "interpretation": "Dollar index data unavailable"}

    current = float(current)
    result = {
        "indicator": "Dollar Index",
        "value": round(current, 2)
    }

    if ma20 is not None:
        ma20 = float(ma20)
        change_pct = (current - ma20) / ma20 * 100

        if change_pct > 2:
            signal = "strengthening"
            interpretation = "Dollar strengthening, EM outflow concern, Risk-Off for Korea"
            regime_implication = "risk_off"
        elif change_pct < -2:
            signal = "weakening"
            interpretation = "Dollar weakening, EM inflow expectation, Risk-On for Korea"
            regime_implication = "risk_on"
        else:
            signal = "stable"
            interpretation = "Dollar stable, Neutral"
            regime_implication = "neutral"

        result["signal"] = signal
        result["interpretation"] = interpretation
        result["regime_implication"] = regime_implication
        result["vs_ma20_pct"] = round(change_pct, 2)
    else:
        result["signal"] = "level_only"
        result["interpretation"] = f"Current level: {current:.2f}"
        result["regime_implication"] = "neutral"

    return result


# =============================================================================
# Exchange Rate (USD/KRW) Interpretation
# =============================================================================

def interpret_exchange_rate(current: float, ma20: Optional[float] = None) -> dict:
    """
    Interpret USD/KRW exchange rate.

    KRW weakening (rate up) = Foreign outflow concern, Risk-Off for Korea
    KRW strengthening (rate down) = Foreign inflow expectation, Risk-On for Korea

    Args:
        current: Current USD/KRW rate
        ma20: 20-day moving average
    """
    if current is None:
        return {"signal": "no_data", "interpretation": "Exchange rate data unavailable"}

    current = float(current)
    result = {
        "indicator": "USD/KRW",
        "value": round(current, 2)
    }

    if ma20 is not None:
        ma20 = float(ma20)
        change_pct = (current - ma20) / ma20 * 100

        if change_pct > 2:
            signal = "krw_weakening"
            interpretation = "KRW weakening, foreign outflow concern"
            regime_implication = "risk_off"
        elif change_pct < -2:
            signal = "krw_strengthening"
            interpretation = "KRW strengthening, foreign inflow expectation"
            regime_implication = "risk_on"
        else:
            signal = "stable"
            interpretation = "Exchange rate stable"
            regime_implication = "neutral"

        result["signal"] = signal
        result["interpretation"] = interpretation
        result["regime_implication"] = regime_implication
        result["vs_ma20_pct"] = round(change_pct, 2)
    else:
        result["signal"] = "level_only"
        result["interpretation"] = f"Current rate: {current:.2f}"
        result["regime_implication"] = "neutral"

    return result


# =============================================================================
# Safe Haven Flow Interpretation
# =============================================================================

def interpret_safe_haven_flow(
    gold_change_pct: Optional[float] = None,
    bond_change_pct: Optional[float] = None,
    hyg_change_pct: Optional[float] = None
) -> dict:
    """
    Interpret safe haven asset flows.

    Gold (GLD) + Long-term Bond (TLT) rising + High Yield (HYG) falling = Risk-Off
    Gold + Bond falling + High Yield rising = Risk-On

    Args:
        gold_change_pct: Gold price change %
        bond_change_pct: Long-term bond price change %
        hyg_change_pct: High yield bond price change %
    """
    result = {
        "indicator": "Safe Haven Flow"
    }

    # Calculate safe haven score
    # Positive = Risk-Off flow (safe haven buying)
    # Negative = Risk-On flow (risk asset buying)
    score = 0
    components = []

    if gold_change_pct is not None:
        gold_change_pct = float(gold_change_pct)
        score += gold_change_pct
        components.append(f"Gold: {gold_change_pct:+.2f}%")
        result["gold_change_pct"] = round(gold_change_pct, 2)

    if bond_change_pct is not None:
        bond_change_pct = float(bond_change_pct)
        score += bond_change_pct
        components.append(f"Bond: {bond_change_pct:+.2f}%")
        result["bond_change_pct"] = round(bond_change_pct, 2)

    if hyg_change_pct is not None:
        hyg_change_pct = float(hyg_change_pct)
        score -= hyg_change_pct  # HYG rising = Risk-On, so subtract
        components.append(f"HYG: {hyg_change_pct:+.2f}%")
        result["hyg_change_pct"] = round(hyg_change_pct, 2)

    if not components:
        return {"signal": "no_data", "interpretation": "Safe haven flow data unavailable"}

    if score > 3:
        signal = "risk_off_flow"
        interpretation = "Safe haven buying, Risk-Off"
        regime_implication = "risk_off"
    elif score < -3:
        signal = "risk_on_flow"
        interpretation = "Risk asset buying, Risk-On"
        regime_implication = "risk_on"
    else:
        signal = "mixed"
        interpretation = "Mixed flow, direction unclear"
        regime_implication = "neutral"

    result["signal"] = signal
    result["interpretation"] = interpretation
    result["regime_implication"] = regime_implication
    result["safe_haven_score"] = round(score, 2)
    result["components"] = components

    return result


# =============================================================================
# Foreign Investor Flow Interpretation
# =============================================================================

def interpret_foreign_flow(net_5d: int, net_20d: int) -> dict:
    """
    Interpret foreign investor flow in Korean market.

    Args:
        net_5d: 5-day net buying amount (KRW)
        net_20d: 20-day net buying amount (KRW)

    Returns:
        Signal interpretation dict
    """
    if net_5d is None and net_20d is None:
        return {"signal": "no_data", "interpretation": "Foreign flow data unavailable"}

    net_5d = int(net_5d) if net_5d is not None else 0
    net_20d = int(net_20d) if net_20d is not None else 0

    # Convert to billion KRW for readability
    net_5d_bil = net_5d / 1_000_000_000
    net_20d_bil = net_20d / 1_000_000_000

    result = {
        "indicator": "Foreign Flow",
        "net_5d_billion_krw": round(net_5d_bil, 1),
        "net_20d_billion_krw": round(net_20d_bil, 1)
    }

    # Strong buying: both positive and significant
    if net_5d_bil > 500 and net_20d_bil > 1000:
        signal = "strong_buying"
        interpretation = f"Foreign strong net buying: 5D {net_5d_bil:+,.0f}B, 20D {net_20d_bil:+,.0f}B"
        regime_implication = "risk_on"
    elif net_5d_bil > 0 and net_20d_bil > 0:
        signal = "net_buying"
        interpretation = f"Foreign net buying continues: 5D {net_5d_bil:+,.0f}B, 20D {net_20d_bil:+,.0f}B"
        regime_implication = "risk_on"
    # Strong selling: both negative and significant
    elif net_5d_bil < -500 and net_20d_bil < -1000:
        signal = "strong_selling"
        interpretation = f"Foreign strong net selling: 5D {net_5d_bil:+,.0f}B, 20D {net_20d_bil:+,.0f}B"
        regime_implication = "risk_off"
    elif net_5d_bil < 0 and net_20d_bil < 0:
        signal = "net_selling"
        interpretation = f"Foreign net selling continues: 5D {net_5d_bil:+,.0f}B, 20D {net_20d_bil:+,.0f}B"
        regime_implication = "risk_off"
    else:
        signal = "mixed"
        interpretation = f"Foreign flow mixed: 5D {net_5d_bil:+,.0f}B, 20D {net_20d_bil:+,.0f}B"
        regime_implication = "neutral"

    result["signal"] = signal
    result["interpretation"] = interpretation
    result["regime_implication"] = regime_implication

    return result


# =============================================================================
# Market Trend Interpretation
# =============================================================================

def interpret_market_trend(
    prices: list[float],
    index_name: str = "Market"
) -> dict:
    """
    Interpret market index trend.

    Args:
        prices: List of closing prices (most recent first)
        index_name: Name of the index for labeling

    Returns:
        Trend signal dict
    """
    if not prices or len(prices) < 5:
        return {"signal": "insufficient_data", "interpretation": f"{index_name} data insufficient"}

    prices = [float(p) for p in prices]
    current = prices[0]

    # Calculate moving averages
    ma5 = sum(prices[:5]) / 5
    ma20 = sum(prices[:20]) / 20 if len(prices) >= 20 else ma5

    # Calculate changes
    change_5d_pct = (current - prices[4]) / prices[4] * 100 if prices[4] != 0 else 0
    change_20d_pct = (current - prices[19]) / prices[19] * 100 if len(prices) > 19 and prices[19] != 0 else 0

    result = {
        "indicator": f"{index_name} Trend",
        "current": round(current, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "change_5d_pct": round(change_5d_pct, 2),
        "change_20d_pct": round(change_20d_pct, 2)
    }

    # Determine trend
    if change_20d_pct > 5:
        signal = "uptrend"
        interpretation = f"{index_name} in uptrend ({change_20d_pct:+.1f}% over 20D)"
        regime_implication = "risk_on"
    elif change_20d_pct < -5:
        signal = "downtrend"
        interpretation = f"{index_name} in downtrend ({change_20d_pct:+.1f}% over 20D)"
        regime_implication = "risk_off"
    else:
        signal = "sideways"
        interpretation = f"{index_name} trading sideways ({change_20d_pct:+.1f}% over 20D)"
        regime_implication = "neutral"

    result["signal"] = signal
    result["interpretation"] = interpretation
    result["regime_implication"] = regime_implication

    # Add MA crossover signal
    if current > ma5 > ma20:
        result["ma_status"] = "bullish_alignment"
    elif current < ma5 < ma20:
        result["ma_status"] = "bearish_alignment"
    else:
        result["ma_status"] = "mixed"

    return result


# =============================================================================
# MOVE Index Interpretation (Bond Volatility)
# =============================================================================

def interpret_move_index(value: float, prev_value: Optional[float] = None) -> dict:
    """
    Interpret MOVE Index (bond market volatility).

    MOVE typical range: 80-150
    - < 80: Low bond volatility, stable rates
    - 80-100: Normal volatility
    - 100-120: Elevated volatility
    - > 120: High volatility, rate uncertainty

    Args:
        value: Current MOVE index value
        prev_value: Previous period value
    """
    if value is None:
        return {"signal": "no_data", "interpretation": "MOVE index data unavailable"}

    value = float(value)

    if value < 80:
        signal = "low_volatility"
        interpretation = "Low bond volatility, stable rate environment"
        regime_implication = "risk_on"
    elif value < 100:
        signal = "normal"
        interpretation = "Normal bond volatility"
        regime_implication = "neutral"
    elif value < 120:
        signal = "elevated"
        interpretation = "Elevated bond volatility, rate uncertainty"
        regime_implication = "risk_off"
    else:
        signal = "high_volatility"
        interpretation = "High bond volatility, significant rate risk"
        regime_implication = "risk_off_strong"

    result = {
        "indicator": "MOVE Index",
        "value": round(value, 2),
        "signal": signal,
        "interpretation": interpretation,
        "regime_implication": regime_implication
    }

    if prev_value is not None:
        prev_value = float(prev_value)
        change = value - prev_value
        result["change"] = round(change, 2)
        result["trend"] = "rising" if change > 0 else "falling" if change < 0 else "stable"

    return result


# =============================================================================
# Integrated Signal Calculation
# =============================================================================

def calculate_regime_signals(
    vkospi_data: Optional[list[dict]] = None,
    vix_data: Optional[list[dict]] = None,
    credit_spread_data: Optional[list[dict]] = None,
    dollar_index_data: Optional[list[dict]] = None,
    exchange_rate_data: Optional[list[dict]] = None,
    move_index_data: Optional[list[dict]] = None,
    us_etf_data: Optional[dict] = None,
    foreign_flow_5d: Optional[int] = None,
    foreign_flow_20d: Optional[int] = None,
    kospi_prices: Optional[list[float]] = None,
    kosdaq_prices: Optional[list[float]] = None
) -> dict:
    """
    Calculate all regime signals from raw data.

    This is the main entry point that:
    1. Extracts values from raw data
    2. Calls individual interpret_* functions
    3. Aggregates results into a structured output

    Args:
        vkospi_data: VKOSPI time series from DB
        vix_data: US VIX time series from DB
        credit_spread_data: Credit spread time series from DB
        dollar_index_data: Dollar index time series from DB
        exchange_rate_data: USD/KRW time series from DB
        move_index_data: MOVE index time series from DB
        us_etf_data: Dict of ETF data {symbol: [records]}
        foreign_flow_5d: 5-day foreign net buying
        foreign_flow_20d: 20-day foreign net buying
        kospi_prices: KOSPI index prices (most recent first)
        kosdaq_prices: KOSDAQ index prices (most recent first)

    Returns:
        dict with structured signals for Market Regime Agent
    """
    signals = {
        "timestamp": datetime.now().isoformat(),
        "global_signals": {},
        "korea_signals": {},
        "signal_summary": {
            "risk_on_count": 0,
            "risk_off_count": 0,
            "neutral_count": 0,
            "dominant_signal": "neutral"
        }
    }

    risk_on_count = 0
    risk_off_count = 0
    neutral_count = 0

    # === Global Signals ===

    # VIX (US)
    if vix_data and len(vix_data) > 0:
        current_vix = vix_data[0].get("value") or vix_data[0].get("close")
        prev_vix = vix_data[1].get("value") or vix_data[1].get("close") if len(vix_data) > 1 else None
        vix_signal = interpret_vix(current_vix, prev_vix)
        signals["global_signals"]["vix"] = vix_signal

        if "risk_on" in vix_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in vix_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # Credit Spread
    if credit_spread_data and len(credit_spread_data) > 0:
        current_spread = credit_spread_data[0].get("value") or credit_spread_data[0].get("spread")
        prev_spread = credit_spread_data[1].get("value") or credit_spread_data[1].get("spread") if len(credit_spread_data) > 1 else None
        spread_signal = interpret_credit_spread(current_spread, prev_spread)
        signals["global_signals"]["credit_spread"] = spread_signal

        if "risk_on" in spread_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in spread_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # Dollar Index
    if dollar_index_data and len(dollar_index_data) >= 20:
        current_dollar = dollar_index_data[0].get("value") or dollar_index_data[0].get("close")
        dollar_values = [d.get("value") or d.get("close") for d in dollar_index_data[:20]]
        ma20_dollar = sum(float(v) for v in dollar_values if v) / 20
        dollar_signal = interpret_dollar_index(current_dollar, ma20_dollar)
        signals["global_signals"]["dollar_index"] = dollar_signal

        if "risk_on" in dollar_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in dollar_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # MOVE Index
    if move_index_data and len(move_index_data) > 0:
        current_move = move_index_data[0].get("value") or move_index_data[0].get("close")
        prev_move = move_index_data[1].get("value") or move_index_data[1].get("close") if len(move_index_data) > 1 else None
        move_signal = interpret_move_index(current_move, prev_move)
        signals["global_signals"]["move_index"] = move_signal

        if "risk_on" in move_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in move_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # Safe Haven Flow (from US ETF data)
    if us_etf_data:
        gold_change = None
        bond_change = None
        hyg_change = None

        if "GLD" in us_etf_data and len(us_etf_data["GLD"]) >= 5:
            gld = us_etf_data["GLD"]
            gold_change = (float(gld[0].get("close", 0)) - float(gld[4].get("close", 1))) / float(gld[4].get("close", 1)) * 100

        if "TLT" in us_etf_data and len(us_etf_data["TLT"]) >= 5:
            tlt = us_etf_data["TLT"]
            bond_change = (float(tlt[0].get("close", 0)) - float(tlt[4].get("close", 1))) / float(tlt[4].get("close", 1)) * 100

        if "HYG" in us_etf_data and len(us_etf_data["HYG"]) >= 5:
            hyg = us_etf_data["HYG"]
            hyg_change = (float(hyg[0].get("close", 0)) - float(hyg[4].get("close", 1))) / float(hyg[4].get("close", 1)) * 100

        safe_haven_signal = interpret_safe_haven_flow(gold_change, bond_change, hyg_change)
        signals["global_signals"]["safe_haven_flow"] = safe_haven_signal

        if "risk_on" in safe_haven_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in safe_haven_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # === Korea Signals ===

    # VKOSPI
    if vkospi_data and len(vkospi_data) > 0:
        current_vkospi = vkospi_data[0].get("close")
        prev_vkospi = vkospi_data[1].get("close") if len(vkospi_data) > 1 else None
        vkospi_signal = interpret_vkospi(current_vkospi, prev_vkospi)
        signals["korea_signals"]["vkospi"] = vkospi_signal

        if "risk_on" in vkospi_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in vkospi_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # Exchange Rate (USD/KRW)
    if exchange_rate_data and len(exchange_rate_data) >= 20:
        current_rate = exchange_rate_data[0].get("data_value") or exchange_rate_data[0].get("close")
        rate_values = [d.get("data_value") or d.get("close") for d in exchange_rate_data[:20]]
        ma20_rate = sum(float(v) for v in rate_values if v) / 20
        rate_signal = interpret_exchange_rate(current_rate, ma20_rate)
        signals["korea_signals"]["exchange_rate"] = rate_signal

        if "risk_on" in rate_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in rate_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # Foreign Flow
    if foreign_flow_5d is not None or foreign_flow_20d is not None:
        foreign_signal = interpret_foreign_flow(foreign_flow_5d or 0, foreign_flow_20d or 0)
        signals["korea_signals"]["foreign_flow"] = foreign_signal

        if "risk_on" in foreign_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in foreign_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # KOSPI Trend
    if kospi_prices and len(kospi_prices) >= 5:
        kospi_signal = interpret_market_trend(kospi_prices, "KOSPI")
        signals["korea_signals"]["kospi_trend"] = kospi_signal

        if "risk_on" in kospi_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in kospi_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # KOSDAQ Trend
    if kosdaq_prices and len(kosdaq_prices) >= 5:
        kosdaq_signal = interpret_market_trend(kosdaq_prices, "KOSDAQ")
        signals["korea_signals"]["kosdaq_trend"] = kosdaq_signal

        if "risk_on" in kosdaq_signal.get("regime_implication", ""):
            risk_on_count += 1
        elif "risk_off" in kosdaq_signal.get("regime_implication", ""):
            risk_off_count += 1
        else:
            neutral_count += 1

    # === Summary ===
    signals["signal_summary"]["risk_on_count"] = risk_on_count
    signals["signal_summary"]["risk_off_count"] = risk_off_count
    signals["signal_summary"]["neutral_count"] = neutral_count

    # Determine dominant signal
    if risk_on_count > risk_off_count and risk_on_count > neutral_count:
        signals["signal_summary"]["dominant_signal"] = "risk_on"
    elif risk_off_count > risk_on_count and risk_off_count > neutral_count:
        signals["signal_summary"]["dominant_signal"] = "risk_off"
    else:
        signals["signal_summary"]["dominant_signal"] = "neutral"

    return signals
