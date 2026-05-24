"""
Regime Preprocessor Module for US Market
Transforms raw market data into interpreted signals for Market Regime Agent.
No LLM involved - pure Python data interpretation.

US-specific: No VKOSPI, no KRW exchange rate, no Korean foreign flow
Focus on VIX, credit spread, dollar index, Treasury yields, US market indices
"""
from typing import Optional
from datetime import datetime


# =============================================================================
# VIX (US Volatility Index) Interpretation
# =============================================================================

def interpret_vix(value: float, prev_value: Optional[float] = None) -> dict:
    """
    Interpret US VIX value into signal.

    VIX typical range: 10-40
    - < 12: Very low fear (excessive complacency, potential correction)
    - 12-15: Low fear (Risk-On environment)
    - 15-20: Moderate (Neutral)
    - 20-25: Elevated fear (Risk-Off signal)
    - 25-30: High fear (Strong Risk-Off)
    - > 30: Extreme fear (Crisis mode)

    Args:
        value: Current VIX value
        prev_value: Previous period value for trend calculation

    Returns:
        dict with signal, interpretation, and metadata
    """
    if value is None:
        return {"signal": "no_data", "interpretation": "VIX data unavailable"}

    value = float(value)

    if value < 12:
        signal = "very_low_fear"
        interpretation = "Excessive complacency in market, potential correction risk"
        regime_implication = "risk_on_caution"
    elif value < 15:
        signal = "low_fear"
        interpretation = "Low fear, Risk-On environment"
        regime_implication = "risk_on"
    elif value < 20:
        signal = "moderate"
        interpretation = "Normal volatility level, Neutral market"
        regime_implication = "neutral"
    elif value < 25:
        signal = "elevated_fear"
        interpretation = "Elevated fear, Risk-Off signal"
        regime_implication = "risk_off"
    elif value < 30:
        signal = "high_fear"
        interpretation = "High fear, Strong Risk-Off"
        regime_implication = "risk_off_strong"
    else:
        signal = "extreme_fear"
        interpretation = "Extreme fear, Crisis mode - consider defensive positioning"
        regime_implication = "risk_off_crisis"

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

        if change_pct > 15:
            trend = "spiking"
            trend_interpretation = "Volatility spiking, fear increasing rapidly"
        elif change_pct > 8:
            trend = "rising"
            trend_interpretation = "Volatility rising, caution warranted"
        elif change_pct < -15:
            trend = "collapsing"
            trend_interpretation = "Volatility collapsing, fear subsiding rapidly"
        elif change_pct < -8:
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


# =============================================================================
# Credit Spread Interpretation (HY-IG Spread)
# =============================================================================

def interpret_credit_spread(current: float, prev: Optional[float] = None) -> dict:
    """
    Interpret credit spread (HY-IG spread) level and change.

    Credit spread widening = Risk-Off (credit risk increasing)
    Credit spread tightening = Risk-On (credit risk decreasing)

    Typical levels:
    - < 3%: Tight spreads, Risk-On
    - 3-4%: Normal spreads
    - 4-5%: Elevated spreads, Risk-Off signal
    - > 5%: Wide spreads, Strong Risk-Off

    Args:
        current: Current spread value (percentage)
        prev: Previous period spread value
    """
    if current is None:
        return {"signal": "no_data", "interpretation": "Credit spread data unavailable"}

    current = float(current)

    # Level-based signal
    if current < 3:
        level_signal = "tight"
        level_interpretation = "Tight spreads indicate strong risk appetite"
        level_regime = "risk_on"
    elif current < 4:
        level_signal = "normal"
        level_interpretation = "Normal credit spread levels"
        level_regime = "neutral"
    elif current < 5:
        level_signal = "elevated"
        level_interpretation = "Elevated spreads suggest credit stress"
        level_regime = "risk_off"
    else:
        level_signal = "wide"
        level_interpretation = "Wide spreads indicate significant credit stress"
        level_regime = "risk_off_strong"

    result = {
        "indicator": "Credit Spread",
        "value": round(current, 2),
        "level_signal": level_signal,
        "level_interpretation": level_interpretation
    }

    # Change-based signal
    if prev is not None:
        prev = float(prev)
        change = current - prev

        if change < -0.15:
            change_signal = "tightening"
            change_interpretation = "Spread tightening, credit risk decreasing"
            change_regime = "risk_on"
        elif change > 0.15:
            change_signal = "widening"
            change_interpretation = "Spread widening, credit risk increasing"
            change_regime = "risk_off"
        else:
            change_signal = "stable"
            change_interpretation = "Spread stable"
            change_regime = "neutral"

        result["change_signal"] = change_signal
        result["change_interpretation"] = change_interpretation
        result["change"] = round(change, 3)

        # Combined regime implication (level takes precedence if extreme)
        if level_regime in ["risk_off_strong", "risk_on"]:
            result["regime_implication"] = level_regime
        else:
            result["regime_implication"] = change_regime
    else:
        result["regime_implication"] = level_regime

    result["interpretation"] = f"{level_interpretation}. {result.get('change_interpretation', '')}"

    return result


# =============================================================================
# Dollar Index Interpretation
# =============================================================================

def interpret_dollar_index(current: float, ma20: Optional[float] = None) -> dict:
    """
    Interpret US Dollar Index trend.

    Dollar strengthening = Generally Risk-Off (flight to safety)
    Dollar weakening = Generally Risk-On (risk appetite increasing)

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
            interpretation = "Dollar strengthening, flight to safety"
            regime_implication = "risk_off"
        elif change_pct < -2:
            signal = "weakening"
            interpretation = "Dollar weakening, risk appetite increasing"
            regime_implication = "risk_on"
        else:
            signal = "stable"
            interpretation = "Dollar stable"
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
# Treasury Yield Interpretation
# =============================================================================

def interpret_treasury_yield(
    yield_10y: Optional[float] = None,
    yield_2y: Optional[float] = None,
    prev_10y: Optional[float] = None
) -> dict:
    """
    Interpret Treasury yields and yield curve.

    Args:
        yield_10y: 10-year Treasury yield
        yield_2y: 2-year Treasury yield (for curve analysis)
        prev_10y: Previous 10-year yield for trend

    Returns:
        Signal interpretation dict
    """
    result = {
        "indicator": "Treasury Yields"
    }

    if yield_10y is None:
        return {"signal": "no_data", "interpretation": "Treasury yield data unavailable"}

    yield_10y = float(yield_10y)
    result["yield_10y"] = round(yield_10y, 2)

    # Yield curve analysis
    if yield_2y is not None:
        yield_2y = float(yield_2y)
        result["yield_2y"] = round(yield_2y, 2)
        spread = yield_10y - yield_2y
        result["curve_spread"] = round(spread, 2)

        if spread < -0.5:
            curve_signal = "deeply_inverted"
            curve_interpretation = "Deeply inverted curve, recession signal"
            curve_regime = "risk_off_strong"
        elif spread < 0:
            curve_signal = "inverted"
            curve_interpretation = "Inverted curve, caution warranted"
            curve_regime = "risk_off"
        elif spread < 0.5:
            curve_signal = "flat"
            curve_interpretation = "Flat curve, economic uncertainty"
            curve_regime = "neutral"
        else:
            curve_signal = "normal"
            curve_interpretation = "Normal upward sloping curve"
            curve_regime = "risk_on"

        result["curve_signal"] = curve_signal
        result["curve_interpretation"] = curve_interpretation
        result["regime_implication"] = curve_regime

    # Yield level and trend
    if prev_10y is not None:
        prev_10y = float(prev_10y)
        change = yield_10y - prev_10y

        if change > 0.1:
            yield_trend = "rising"
            trend_interpretation = "Yields rising, tightening financial conditions"
        elif change < -0.1:
            yield_trend = "falling"
            trend_interpretation = "Yields falling, easing financial conditions"
        else:
            yield_trend = "stable"
            trend_interpretation = "Yields stable"

        result["yield_trend"] = yield_trend
        result["trend_interpretation"] = trend_interpretation
        result["yield_change"] = round(change, 2)

    result["interpretation"] = result.get("curve_interpretation", f"10Y yield at {yield_10y:.2f}%")

    return result


# =============================================================================
# Safe Haven Flow Interpretation (US ETFs)
# =============================================================================

def interpret_safe_haven_flow(
    gold_change_pct: Optional[float] = None,
    tlt_change_pct: Optional[float] = None,
    hyg_change_pct: Optional[float] = None
) -> dict:
    """
    Interpret safe haven asset flows using US ETFs.

    Gold (GLD) + Long-term Bond (TLT) rising + High Yield (HYG) falling = Risk-Off
    Gold + Bond falling + High Yield rising = Risk-On

    Args:
        gold_change_pct: Gold (GLD) price change %
        tlt_change_pct: Long-term Treasury (TLT) price change %
        hyg_change_pct: High yield bond (HYG) price change %
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

    if tlt_change_pct is not None:
        tlt_change_pct = float(tlt_change_pct)
        score += tlt_change_pct
        components.append(f"TLT: {tlt_change_pct:+.2f}%")
        result["tlt_change_pct"] = round(tlt_change_pct, 2)

    if hyg_change_pct is not None:
        hyg_change_pct = float(hyg_change_pct)
        score -= hyg_change_pct  # HYG rising = Risk-On, so subtract
        components.append(f"HYG: {hyg_change_pct:+.2f}%")
        result["hyg_change_pct"] = round(hyg_change_pct, 2)

    if not components:
        return {"signal": "no_data", "interpretation": "Safe haven flow data unavailable"}

    if score > 3:
        signal = "risk_off_flow"
        interpretation = "Safe haven buying detected, Risk-Off flow"
        regime_implication = "risk_off"
    elif score < -3:
        signal = "risk_on_flow"
        interpretation = "Risk asset buying detected, Risk-On flow"
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
# MOVE Index Interpretation (Bond Volatility)
# =============================================================================

def interpret_move_index(value: float, prev_value: Optional[float] = None) -> dict:
    """
    Interpret MOVE Index (bond market volatility).

    MOVE typical range: 60-150
    - < 70: Low bond volatility, stable rates
    - 70-90: Normal volatility
    - 90-110: Elevated volatility
    - 110-130: High volatility
    - > 130: Extreme volatility, significant rate uncertainty

    Args:
        value: Current MOVE index value
        prev_value: Previous period value
    """
    if value is None:
        return {"signal": "no_data", "interpretation": "MOVE index data unavailable"}

    value = float(value)

    if value < 70:
        signal = "low_volatility"
        interpretation = "Low bond volatility, stable rate environment"
        regime_implication = "risk_on"
    elif value < 90:
        signal = "normal"
        interpretation = "Normal bond market volatility"
        regime_implication = "neutral"
    elif value < 110:
        signal = "elevated"
        interpretation = "Elevated bond volatility, rate uncertainty"
        regime_implication = "risk_off"
    elif value < 130:
        signal = "high"
        interpretation = "High bond volatility, significant rate uncertainty"
        regime_implication = "risk_off_strong"
    else:
        signal = "extreme"
        interpretation = "Extreme bond volatility, crisis-level rate uncertainty"
        regime_implication = "risk_off_crisis"

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
# Market Index Trend Interpretation (S&P 500, NASDAQ)
# =============================================================================

def interpret_market_trend(
    prices: list[float],
    index_name: str = "S&P 500"
) -> dict:
    """
    Interpret US market index trend.

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
    ma50 = sum(prices[:50]) / 50 if len(prices) >= 50 else ma20

    # Calculate changes
    change_5d_pct = (current - prices[4]) / prices[4] * 100 if prices[4] != 0 else 0
    change_20d_pct = (current - prices[19]) / prices[19] * 100 if len(prices) > 19 and prices[19] != 0 else 0

    result = {
        "indicator": f"{index_name} Trend",
        "current": round(current, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2) if len(prices) >= 50 else None,
        "change_5d_pct": round(change_5d_pct, 2),
        "change_20d_pct": round(change_20d_pct, 2)
    }

    # Determine trend
    if change_20d_pct > 5:
        signal = "uptrend"
        interpretation = f"{index_name} in strong uptrend ({change_20d_pct:+.1f}% over 20D)"
        regime_implication = "risk_on"
    elif change_20d_pct > 2:
        signal = "mild_uptrend"
        interpretation = f"{index_name} in mild uptrend ({change_20d_pct:+.1f}% over 20D)"
        regime_implication = "risk_on"
    elif change_20d_pct < -5:
        signal = "downtrend"
        interpretation = f"{index_name} in downtrend ({change_20d_pct:+.1f}% over 20D)"
        regime_implication = "risk_off"
    elif change_20d_pct < -2:
        signal = "mild_downtrend"
        interpretation = f"{index_name} in mild downtrend ({change_20d_pct:+.1f}% over 20D)"
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
# Put/Call Ratio Interpretation
# =============================================================================

def interpret_put_call_ratio(ratio: float, prev_ratio: Optional[float] = None) -> dict:
    """
    Interpret equity put/call ratio for market sentiment.

    Typical range: 0.6 - 1.2
    - < 0.7: Excessive optimism, contrarian bearish signal
    - 0.7-0.85: Bullish sentiment
    - 0.85-1.0: Neutral
    - 1.0-1.2: Bearish sentiment
    - > 1.2: Excessive fear, contrarian bullish signal

    Args:
        ratio: Current put/call ratio
        prev_ratio: Previous ratio for trend
    """
    if ratio is None:
        return {"signal": "no_data", "interpretation": "Put/Call ratio unavailable"}

    ratio = float(ratio)

    if ratio < 0.7:
        signal = "extreme_bullish"
        interpretation = "Excessive optimism, potential contrarian warning"
        regime_implication = "risk_on_caution"
    elif ratio < 0.85:
        signal = "bullish"
        interpretation = "Bullish options sentiment"
        regime_implication = "risk_on"
    elif ratio < 1.0:
        signal = "neutral"
        interpretation = "Neutral options sentiment"
        regime_implication = "neutral"
    elif ratio < 1.2:
        signal = "bearish"
        interpretation = "Bearish options sentiment"
        regime_implication = "risk_off"
    else:
        signal = "extreme_bearish"
        interpretation = "Extreme fear, potential contrarian buy signal"
        regime_implication = "risk_off_contrarian"

    result = {
        "indicator": "Put/Call Ratio",
        "value": round(ratio, 2),
        "signal": signal,
        "interpretation": interpretation,
        "regime_implication": regime_implication
    }

    if prev_ratio is not None:
        prev_ratio = float(prev_ratio)
        change = ratio - prev_ratio
        result["change"] = round(change, 2)
        result["trend"] = "rising" if change > 0.05 else "falling" if change < -0.05 else "stable"

    return result


# =============================================================================
# Integrated Signal Calculation
# =============================================================================

def calculate_regime_signals(
    vix_data: Optional[list[dict]] = None,
    credit_spread_data: Optional[list[dict]] = None,
    dollar_index_data: Optional[list[dict]] = None,
    treasury_yield_data: Optional[list[dict]] = None,
    move_index_data: Optional[list[dict]] = None,
    us_etf_data: Optional[dict] = None,
    sp500_prices: Optional[list[float]] = None,
    nasdaq_prices: Optional[list[float]] = None,
    put_call_ratio: Optional[float] = None
) -> dict:
    """
    Calculate all regime signals from raw data.

    This is the main entry point that:
    1. Extracts values from raw data
    2. Calls individual interpret_* functions
    3. Aggregates results into a structured output

    Args:
        vix_data: US VIX time series from DB
        credit_spread_data: Credit spread time series from DB
        dollar_index_data: Dollar index time series from DB
        treasury_yield_data: Treasury yield time series from DB
        move_index_data: MOVE index time series from DB
        us_etf_data: Dict of ETF data {symbol: [records]}
        sp500_prices: S&P 500 prices (most recent first)
        nasdaq_prices: NASDAQ prices (most recent first)
        put_call_ratio: Current equity put/call ratio

    Returns:
        dict with structured signals for Market Regime Agent
    """
    signals = {
        "timestamp": datetime.now().isoformat(),
        "global_signals": {},
        "us_market_signals": {},
        "signal_summary": {
            "global_risk_on_count": 0,
            "global_risk_off_count": 0,
            "global_neutral_count": 0,
            "us_risk_on_count": 0,
            "us_risk_off_count": 0,
            "us_neutral_count": 0,
            "dominant_signal": "neutral"
        }
    }

    global_risk_on = 0
    global_risk_off = 0
    global_neutral = 0
    us_risk_on = 0
    us_risk_off = 0
    us_neutral = 0

    def count_signal(implication: str, is_global: bool = True):
        nonlocal global_risk_on, global_risk_off, global_neutral
        nonlocal us_risk_on, us_risk_off, us_neutral

        if is_global:
            if "risk_on" in implication:
                global_risk_on += 1
            elif "risk_off" in implication:
                global_risk_off += 1
            else:
                global_neutral += 1
        else:
            if "risk_on" in implication:
                us_risk_on += 1
            elif "risk_off" in implication:
                us_risk_off += 1
            else:
                us_neutral += 1

    # === Global Signals ===

    # VIX
    if vix_data and len(vix_data) > 0:
        current_vix = vix_data[0].get("value") or vix_data[0].get("close")
        prev_vix = vix_data[1].get("value") or vix_data[1].get("close") if len(vix_data) > 1 else None
        vix_signal = interpret_vix(current_vix, prev_vix)
        signals["global_signals"]["vix"] = vix_signal
        count_signal(vix_signal.get("regime_implication", "neutral"), is_global=True)

    # Credit Spread
    if credit_spread_data and len(credit_spread_data) > 0:
        current_spread = credit_spread_data[0].get("value") or credit_spread_data[0].get("spread")
        prev_spread = credit_spread_data[1].get("value") or credit_spread_data[1].get("spread") if len(credit_spread_data) > 1 else None
        spread_signal = interpret_credit_spread(current_spread, prev_spread)
        signals["global_signals"]["credit_spread"] = spread_signal
        count_signal(spread_signal.get("regime_implication", "neutral"), is_global=True)

    # Dollar Index
    if dollar_index_data and len(dollar_index_data) >= 20:
        current_dollar = dollar_index_data[0].get("value") or dollar_index_data[0].get("close")
        dollar_values = [d.get("value") or d.get("close") for d in dollar_index_data[:20]]
        ma20_dollar = sum(float(v) for v in dollar_values if v) / 20
        dollar_signal = interpret_dollar_index(current_dollar, ma20_dollar)
        signals["global_signals"]["dollar_index"] = dollar_signal
        count_signal(dollar_signal.get("regime_implication", "neutral"), is_global=True)

    # Treasury Yields
    if treasury_yield_data and len(treasury_yield_data) > 0:
        yield_10y = treasury_yield_data[0].get("value") or treasury_yield_data[0].get("yield_10y")
        yield_2y = treasury_yield_data[0].get("yield_2y")
        prev_10y = treasury_yield_data[1].get("value") or treasury_yield_data[1].get("yield_10y") if len(treasury_yield_data) > 1 else None
        yield_signal = interpret_treasury_yield(yield_10y, yield_2y, prev_10y)
        signals["global_signals"]["treasury_yield"] = yield_signal
        count_signal(yield_signal.get("regime_implication", "neutral"), is_global=True)

    # MOVE Index
    if move_index_data and len(move_index_data) > 0:
        current_move = move_index_data[0].get("value") or move_index_data[0].get("close")
        prev_move = move_index_data[1].get("value") or move_index_data[1].get("close") if len(move_index_data) > 1 else None
        move_signal = interpret_move_index(current_move, prev_move)
        signals["global_signals"]["move_index"] = move_signal
        count_signal(move_signal.get("regime_implication", "neutral"), is_global=True)

    # Safe Haven Flow (from US ETF data)
    if us_etf_data:
        gold_change = None
        tlt_change = None
        hyg_change = None

        if "GLD" in us_etf_data and len(us_etf_data["GLD"]) >= 5:
            gld = us_etf_data["GLD"]
            gold_change = (float(gld[0].get("close", 0)) - float(gld[4].get("close", 1))) / float(gld[4].get("close", 1)) * 100

        if "TLT" in us_etf_data and len(us_etf_data["TLT"]) >= 5:
            tlt = us_etf_data["TLT"]
            tlt_change = (float(tlt[0].get("close", 0)) - float(tlt[4].get("close", 1))) / float(tlt[4].get("close", 1)) * 100

        if "HYG" in us_etf_data and len(us_etf_data["HYG"]) >= 5:
            hyg = us_etf_data["HYG"]
            hyg_change = (float(hyg[0].get("close", 0)) - float(hyg[4].get("close", 1))) / float(hyg[4].get("close", 1)) * 100

        safe_haven_signal = interpret_safe_haven_flow(gold_change, tlt_change, hyg_change)
        signals["global_signals"]["safe_haven_flow"] = safe_haven_signal
        count_signal(safe_haven_signal.get("regime_implication", "neutral"), is_global=True)

    # === US Market Signals ===

    # S&P 500 Trend
    if sp500_prices and len(sp500_prices) >= 5:
        sp500_signal = interpret_market_trend(sp500_prices, "S&P 500")
        signals["us_market_signals"]["sp500_trend"] = sp500_signal
        count_signal(sp500_signal.get("regime_implication", "neutral"), is_global=False)

    # NASDAQ Trend
    if nasdaq_prices and len(nasdaq_prices) >= 5:
        nasdaq_signal = interpret_market_trend(nasdaq_prices, "NASDAQ")
        signals["us_market_signals"]["nasdaq_trend"] = nasdaq_signal
        count_signal(nasdaq_signal.get("regime_implication", "neutral"), is_global=False)

    # Put/Call Ratio
    if put_call_ratio is not None:
        pc_signal = interpret_put_call_ratio(put_call_ratio)
        signals["us_market_signals"]["put_call_ratio"] = pc_signal
        count_signal(pc_signal.get("regime_implication", "neutral"), is_global=False)

    # === Summary ===
    signals["signal_summary"]["global_risk_on_count"] = global_risk_on
    signals["signal_summary"]["global_risk_off_count"] = global_risk_off
    signals["signal_summary"]["global_neutral_count"] = global_neutral
    signals["signal_summary"]["us_risk_on_count"] = us_risk_on
    signals["signal_summary"]["us_risk_off_count"] = us_risk_off
    signals["signal_summary"]["us_neutral_count"] = us_neutral

    total_risk_on = global_risk_on + us_risk_on
    total_risk_off = global_risk_off + us_risk_off

    # Determine dominant signal
    if total_risk_on > total_risk_off and total_risk_on > (global_neutral + us_neutral):
        signals["signal_summary"]["dominant_signal"] = "risk_on"
    elif total_risk_off > total_risk_on and total_risk_off > (global_neutral + us_neutral):
        signals["signal_summary"]["dominant_signal"] = "risk_off"
    else:
        signals["signal_summary"]["dominant_signal"] = "neutral"

    return signals


def generate_signal_summary(quantitative_signals: dict) -> str:
    """
    Generate a brief summary of quantitative signals for prompt context.

    Args:
        quantitative_signals: Output from calculate_regime_signals()

    Returns:
        Brief summary string
    """
    summary_parts = []

    if "signal_summary" in quantitative_signals:
        sig = quantitative_signals["signal_summary"]

        # Global signals summary
        global_ro = sig.get("global_risk_on_count", 0)
        global_rf = sig.get("global_risk_off_count", 0)
        if global_ro > global_rf:
            summary_parts.append(f"Global: Risk-On dominant ({global_ro}:{global_rf})")
        elif global_rf > global_ro:
            summary_parts.append(f"Global: Risk-Off dominant ({global_rf}:{global_ro})")
        else:
            summary_parts.append(f"Global: Mixed ({global_ro}:{global_rf})")

        # US signals summary
        us_ro = sig.get("us_risk_on_count", 0)
        us_rf = sig.get("us_risk_off_count", 0)
        if us_ro > us_rf:
            summary_parts.append(f"US: Risk-On dominant ({us_ro}:{us_rf})")
        elif us_rf > us_ro:
            summary_parts.append(f"US: Risk-Off dominant ({us_rf}:{us_ro})")
        else:
            summary_parts.append(f"US: Mixed ({us_ro}:{us_rf})")

    return " | ".join(summary_parts) if summary_parts else "No signal analysis available"
