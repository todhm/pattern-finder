"""
Data preprocessor module
Transforms raw data into interpreted information for agents
No LLM involved - pure Python data analysis
"""
from typing import Optional
from decimal import Decimal


# =============================================================================
# Helper Functions
# =============================================================================

def safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float (handles None, Decimal, str)"""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return float(value)


def safe_int(value, default: int = 0) -> int:
    """Safely convert value to int (handles None, Decimal, str)"""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return int(value)


def calculate_trend(values: list[float]) -> str:
    """Calculate trend from value list (most recent first)"""
    if len(values) < 2:
        return "unknown"
    if values[0] > values[-1]:
        return "rising"
    elif values[0] < values[-1]:
        return "falling"
    return "flat"


def count_consecutive_positive(values: list[int]) -> int:
    """Count consecutive positive days (from most recent)"""
    count = 0
    for v in values:
        if v > 0:
            count += 1
        else:
            break
    return count


def count_consecutive_negative(values: list[int]) -> int:
    """Count consecutive negative days (from most recent)"""
    count = 0
    for v in values:
        if v < 0:
            count += 1
        else:
            break
    return count


def format_korean_number(value: int) -> str:
    """Format number in Korean style (억/만 units)"""
    if abs(value) >= 100000000:
        return f"{value / 100000000:+,.1f}억"
    elif abs(value) >= 10000:
        return f"{value / 10000:+,.0f}만"
    return f"{value:+,d}"


def calculate_volatility(prices: list[float]) -> float:
    """Calculate price volatility (standard deviation of returns)"""
    if len(prices) < 2:
        return 0.0
    returns = [(prices[i] - prices[i+1]) / prices[i+1] * 100
               for i in range(len(prices)-1) if prices[i+1] != 0]
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return round((variance ** 0.5), 2)


def generate_investor_summary(foreign: list[int], inst: list[int]) -> str:
    """Generate investor trend summary sentence"""
    parts = []

    foreign_buy = count_consecutive_positive(foreign)
    foreign_sell = count_consecutive_negative(foreign)
    if foreign_buy >= 3:
        parts.append(f"외국인 {foreign_buy}일 연속 순매수")
    elif foreign_sell >= 3:
        parts.append(f"외국인 {foreign_sell}일 연속 순매도")

    inst_buy = count_consecutive_positive(inst)
    inst_sell = count_consecutive_negative(inst)
    if inst_buy >= 3:
        parts.append(f"기관 {inst_buy}일 연속 순매수")
    elif inst_sell >= 3:
        parts.append(f"기관 {inst_sell}일 연속 순매도")

    return ", ".join(parts) if parts else "특이 동향 없음"


# =============================================================================
# Technical Indicators Analysis
# =============================================================================

def analyze_technical_indicators(indicators: list[dict]) -> dict:
    """
    Analyze technical indicators and generate signals.

    Args:
        indicators: Time series data from kr_indicators (most recent first)

    Returns:
        dict with interpreted signals for RSI, MACD, Bollinger, Stochastic, etc.
    """
    if not indicators:
        return {}

    latest = indicators[0]

    # RSI analysis
    rsi_value = safe_float(latest.get("rsi"), 50)
    if rsi_value > 70:
        rsi_status = "overbought"
    elif rsi_value < 30:
        rsi_status = "oversold"
    else:
        rsi_status = "neutral"
    rsi_trend = calculate_trend([safe_float(i.get("rsi"), 50) for i in indicators[:5]])

    # MACD analysis
    macd_value = safe_float(latest.get("macd"), 0)
    macd_signal = safe_float(latest.get("macd_signal"), 0)
    macd_hist = safe_float(latest.get("macd_hist"), 0)
    macd_status = "bullish" if macd_value > macd_signal else "bearish"

    # Check for MACD crossover
    macd_crossover = None
    if len(indicators) >= 2:
        prev_macd = safe_float(indicators[1].get("macd"), 0)
        prev_signal = safe_float(indicators[1].get("macd_signal"), 0)
        if prev_macd <= prev_signal and macd_value > macd_signal:
            macd_crossover = "golden_cross"
        elif prev_macd >= prev_signal and macd_value < macd_signal:
            macd_crossover = "dead_cross"

    # Bollinger Bands analysis
    upper = safe_float(latest.get("real_upper_band"), 0)
    middle = safe_float(latest.get("real_middle_band"), 0)
    lower = safe_float(latest.get("real_lower_band"), 0)

    # Stochastic analysis
    slowk = safe_float(latest.get("slowk"), 50)
    slowd = safe_float(latest.get("slowd"), 50)
    if slowk > 80:
        stoch_status = "overbought"
    elif slowk < 20:
        stoch_status = "oversold"
    else:
        stoch_status = "neutral"

    # ADX (trend strength)
    adx_value = safe_float(latest.get("adx"), 0)
    if adx_value > 25:
        adx_status = "strong_trend"
    elif adx_value > 20:
        adx_status = "moderate_trend"
    else:
        adx_status = "weak_trend"

    # MFI (Money Flow Index)
    mfi_value = safe_float(latest.get("mfi"), 50)
    if mfi_value > 80:
        mfi_status = "overbought"
    elif mfi_value < 20:
        mfi_status = "oversold"
    else:
        mfi_status = "neutral"

    return {
        "rsi": {
            "value": round(rsi_value, 2),
            "status": rsi_status,
            "trend": rsi_trend
        },
        "macd": {
            "value": round(macd_value, 2),
            "signal": round(macd_signal, 2),
            "histogram": round(macd_hist, 2),
            "status": macd_status,
            "crossover": macd_crossover
        },
        "bollinger": {
            "upper": round(upper, 0),
            "middle": round(middle, 0),
            "lower": round(lower, 0)
        },
        "stochastic": {
            "k": round(slowk, 2),
            "d": round(slowd, 2),
            "status": stoch_status
        },
        "adx": {
            "value": round(adx_value, 2),
            "status": adx_status
        },
        "mfi": {
            "value": round(mfi_value, 2),
            "status": mfi_status
        }
    }


# =============================================================================
# Investor Trends Analysis
# =============================================================================

def analyze_investor_trends(trading_data: list[dict]) -> dict:
    """
    Analyze investor trading patterns.

    Args:
        trading_data: Time series from kr_individual_investor_daily_trading

    Returns:
        dict with foreign/institutional/retail trading patterns
    """
    if not trading_data:
        return {}

    foreign_net = [safe_int(d.get("foreign_net_volume"), 0) for d in trading_data]
    inst_net = [safe_int(d.get("inst_net_volume"), 0) for d in trading_data]
    retail_net = [safe_int(d.get("retail_net_volume"), 0) for d in trading_data]

    foreign_net_5d = sum(foreign_net[:5]) if len(foreign_net) >= 5 else sum(foreign_net)
    foreign_net_30d = sum(foreign_net[:30]) if len(foreign_net) >= 30 else sum(foreign_net)
    inst_net_5d = sum(inst_net[:5]) if len(inst_net) >= 5 else sum(inst_net)
    inst_net_30d = sum(inst_net[:30]) if len(inst_net) >= 30 else sum(inst_net)
    retail_net_5d = sum(retail_net[:5]) if len(retail_net) >= 5 else sum(retail_net)
    retail_net_30d = sum(retail_net[:30]) if len(retail_net) >= 30 else sum(retail_net)

    return {
        "foreign": {
            "consecutive_buy_days": count_consecutive_positive(foreign_net),
            "consecutive_sell_days": count_consecutive_negative(foreign_net),
            "net_5d": foreign_net_5d,
            "net_5d_formatted": format_korean_number(foreign_net_5d),
            "net_30d": foreign_net_30d,
            "net_30d_formatted": format_korean_number(foreign_net_30d),
            "trend": "buying" if foreign_net_5d > 0 else "selling"
        },
        "institutional": {
            "consecutive_buy_days": count_consecutive_positive(inst_net),
            "consecutive_sell_days": count_consecutive_negative(inst_net),
            "net_5d": inst_net_5d,
            "net_5d_formatted": format_korean_number(inst_net_5d),
            "net_30d": inst_net_30d,
            "net_30d_formatted": format_korean_number(inst_net_30d),
            "trend": "buying" if inst_net_5d > 0 else "selling"
        },
        "retail": {
            "net_5d": retail_net_5d,
            "net_5d_formatted": format_korean_number(retail_net_5d),
            "net_30d": retail_net_30d,
            "net_30d_formatted": format_korean_number(retail_net_30d),
            "trend": "buying" if retail_net_5d > 0 else "selling"
        },
        "summary": generate_investor_summary(foreign_net, inst_net)
    }


# =============================================================================
# Foreign Ownership Trend Analysis
# =============================================================================

def analyze_foreign_ownership_trend(ownership_data: list[dict]) -> dict:
    """
    Analyze foreign ownership trend.

    Args:
        ownership_data: Time series from kr_foreign_ownership

    Returns:
        dict with foreign ownership rate and trend
    """
    if not ownership_data:
        return {}

    latest = ownership_data[0]
    current_rate = safe_float(latest.get("foreign_rate"), 0)

    # Calculate change over periods
    rate_1d_ago = safe_float(ownership_data[1].get("foreign_rate"), current_rate) if len(ownership_data) > 1 else current_rate
    rate_5d_ago = safe_float(ownership_data[4].get("foreign_rate"), current_rate) if len(ownership_data) > 4 else current_rate
    rate_30d_ago = safe_float(ownership_data[29].get("foreign_rate"), current_rate) if len(ownership_data) > 29 else current_rate

    return {
        "current_rate": round(current_rate, 2),
        "change_1d": round(current_rate - rate_1d_ago, 2),
        "change_5d": round(current_rate - rate_5d_ago, 2),
        "change_30d": round(current_rate - rate_30d_ago, 2),
        "trend": "increasing" if current_rate > rate_5d_ago else "decreasing" if current_rate < rate_5d_ago else "stable",
        "limit_rate": safe_float(latest.get("foreign_rate_limit"), 0)
    }


# =============================================================================
# Price Trend Analysis
# =============================================================================

def calculate_price_trend(prices: list[dict]) -> dict:
    """
    Analyze price trends and momentum.

    Args:
        prices: Time series from kr_intraday_total (most recent first)

    Returns:
        dict with price trends, moving averages, and changes
    """
    if not prices or len(prices) < 2:
        return {}

    closes = [safe_float(p.get("close"), 0) for p in prices]
    current_price = closes[0]

    # Moving averages
    ma5 = sum(closes[:5]) / 5 if len(closes) >= 5 else current_price
    ma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else current_price
    ma60 = sum(closes[:60]) / 60 if len(closes) >= 60 else None

    # Price changes
    change_1d = ((closes[0] - closes[1]) / closes[1] * 100) if len(closes) > 1 and closes[1] != 0 else 0
    change_5d = ((closes[0] - closes[4]) / closes[4] * 100) if len(closes) > 4 and closes[4] != 0 else 0
    change_20d = ((closes[0] - closes[19]) / closes[19] * 100) if len(closes) > 19 and closes[19] != 0 else 0

    # 52-week high/low (approximate with available data)
    high_price = max(closes) if closes else current_price
    low_price = min(closes) if closes else current_price
    price_position = ((current_price - low_price) / (high_price - low_price) * 100) if high_price != low_price else 50

    # Volume analysis
    volumes = [safe_int(p.get("volume"), 0) for p in prices]
    avg_volume_20d = sum(volumes[:20]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes) if volumes else 0
    current_volume = volumes[0] if volumes else 0
    volume_ratio = (current_volume / avg_volume_20d) if avg_volume_20d > 0 else 1

    return {
        "current_price": int(current_price),
        "ma5": int(ma5),
        "ma20": int(ma20),
        "ma60": int(ma60) if ma60 else None,
        "trend_5d": "up" if current_price > ma5 else "down",
        "trend_20d": "up" if current_price > ma20 else "down",
        "trend_60d": "up" if ma60 and current_price > ma60 else "down" if ma60 else "unknown",
        "change_1d": round(change_1d, 2),
        "change_5d": round(change_5d, 2),
        "change_20d": round(change_20d, 2),
        "volatility_20d": calculate_volatility(closes[:20]) if len(closes) >= 20 else 0,
        "price_position": round(price_position, 1),
        "volume": {
            "current": current_volume,
            "avg_20d": int(avg_volume_20d),
            "ratio": round(volume_ratio, 2)
        }
    }


# =============================================================================
# Quant Result Summary
# =============================================================================

def summarize_quant_result(stock_grade: dict) -> dict:
    """
    Summarize key information from kr_stock_grade.

    Args:
        stock_grade: Single record from kr_stock_grade

    Returns:
        dict with key scores, scenarios, sector info, risk metrics, triggers
    """
    if not stock_grade:
        return {}

    return {
        "scores": {
            "final_grade": stock_grade.get("final_grade"),
            "final_score": safe_float(stock_grade.get("final_score")),
            "value_score": safe_float(stock_grade.get("value_score")),
            "quality_score": safe_float(stock_grade.get("quality_score")),
            "momentum_score": safe_float(stock_grade.get("momentum_score")),
            "growth_score": safe_float(stock_grade.get("growth_score")),
            "confidence_score": safe_float(stock_grade.get("confidence_score")),
            "entry_timing_score": safe_float(stock_grade.get("entry_timing_score"))
        },
        "scenarios": {
            "bullish_prob": safe_int(stock_grade.get("scenario_bullish_prob")),
            "sideways_prob": safe_int(stock_grade.get("scenario_sideways_prob")),
            "bearish_prob": safe_int(stock_grade.get("scenario_bearish_prob")),
            "bullish_return": stock_grade.get("scenario_bullish_return"),
            "sideways_return": stock_grade.get("scenario_sideways_return"),
            "bearish_return": stock_grade.get("scenario_bearish_return"),
            "sample_count": safe_int(stock_grade.get("scenario_sample_count"))
        },
        "sector": {
            "momentum": safe_float(stock_grade.get("sector_momentum")),
            "rank": safe_int(stock_grade.get("sector_rank")),
            "percentile": safe_float(stock_grade.get("sector_percentile"))
        },
        "risk": {
            "var_95": safe_float(stock_grade.get("var_95")),
            "cvar_95": safe_float(stock_grade.get("cvar_95")),
            "beta": safe_float(stock_grade.get("beta")),
            "volatility_annual": safe_float(stock_grade.get("volatility_annual")),
            "max_drawdown_1y": safe_float(stock_grade.get("max_drawdown_1y")),
            "sharpe_ratio": safe_float(stock_grade.get("sharpe_ratio")),
            "sortino_ratio": safe_float(stock_grade.get("sortino_ratio"))
        },
        "trading": {
            "stop_loss_pct": safe_float(stock_grade.get("stop_loss_pct")),
            "take_profit_pct": safe_float(stock_grade.get("take_profit_pct")),
            "risk_reward_ratio": safe_float(stock_grade.get("risk_reward_ratio")),
            "position_size_pct": safe_float(stock_grade.get("position_size_pct")),
            "atr_pct": safe_float(stock_grade.get("atr_pct"))
        },
        "triggers": {
            "buy": stock_grade.get("buy_triggers"),
            "sell": stock_grade.get("sell_triggers"),
            "hold": stock_grade.get("hold_triggers")
        },
        "interpretation": {
            "risk_profile": stock_grade.get("risk_profile_text"),
            "risk_recommendation": stock_grade.get("risk_recommendation"),
            "time_series": stock_grade.get("time_series_text"),
            "signal_overall": stock_grade.get("signal_overall")
        },
        "relative_strength": {
            "rs_value": safe_float(stock_grade.get("rs_value")),
            "rs_rank": stock_grade.get("rs_rank")
        }
    }


# =============================================================================
# Economic Indicators Summary
# =============================================================================

def summarize_economic_indicators(
    kr_indicators: list[dict],
    us_fed_funds_rate: list[dict] = None,
    us_treasury_yield: list[dict] = None,
    us_cpi: list[dict] = None,
    us_unemployment_rate: list[dict] = None,
    us_gdp: list[dict] = None,
    us_pmi: list[dict] = None,
    us_vix: list[dict] = None,
    us_dollar_index: list[dict] = None
) -> dict:
    """
    Summarize Korean and US economic indicators.

    Args:
        kr_indicators: BOK economic indicators
        us_*: US economic indicator time series

    Returns:
        dict with summarized economic data
    """
    result = {
        "korea": {},
        "us": {},
        "market_sentiment": {}
    }

    # Korean indicators
    for ind in (kr_indicators or []):
        stat_name = ind.get("stat_name", "")
        value = safe_float(ind.get("data_value"))

        if "기준금리" in stat_name:
            result["korea"]["base_rate"] = {"value": value, "unit": "%"}
        elif "국내총생산" in stat_name or "GDP" in stat_name:
            result["korea"]["gdp_growth"] = {"value": value, "unit": "%"}
        elif "소비자물가" in stat_name:
            result["korea"]["cpi"] = {"value": value, "unit": "%"}
        elif "경제심리" in stat_name:
            result["korea"]["economic_sentiment"] = {"value": value}
        elif "뉴스심리" in stat_name:
            result["korea"]["news_sentiment"] = {"value": value}

    # US Fed Funds Rate
    if us_fed_funds_rate:
        latest = us_fed_funds_rate[0]
        result["us"]["fed_funds_rate"] = {
            "value": safe_float(latest.get("value")),
            "unit": "%",
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # US Treasury Yield
    if us_treasury_yield:
        latest = us_treasury_yield[0]
        result["us"]["treasury_yield_10y"] = {
            "value": safe_float(latest.get("value")),
            "unit": "%",
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # US CPI
    if us_cpi:
        latest = us_cpi[0]
        result["us"]["cpi"] = {
            "value": safe_float(latest.get("value")),
            "unit": "%",
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # US Unemployment Rate
    if us_unemployment_rate:
        latest = us_unemployment_rate[0]
        result["us"]["unemployment_rate"] = {
            "value": safe_float(latest.get("value")),
            "unit": "%",
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # US GDP
    if us_gdp:
        latest = us_gdp[0]
        result["us"]["gdp"] = {
            "value": safe_float(latest.get("value")),
            "unit": "%",
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # US PMI
    if us_pmi:
        latest = us_pmi[0]
        result["us"]["pmi"] = {
            "value": safe_float(latest.get("value")),
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # VIX (Market Fear Index)
    if us_vix:
        latest = us_vix[0]
        vix_value = safe_float(latest.get("value"))
        if vix_value > 30:
            vix_status = "high_fear"
        elif vix_value > 20:
            vix_status = "elevated"
        else:
            vix_status = "low_fear"
        result["market_sentiment"]["vix"] = {
            "value": vix_value,
            "status": vix_status,
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # Dollar Index
    if us_dollar_index:
        latest = us_dollar_index[0]
        prev = us_dollar_index[4] if len(us_dollar_index) > 4 else latest
        current_value = safe_float(latest.get("value"))
        prev_value = safe_float(prev.get("value"))
        result["market_sentiment"]["dollar_index"] = {
            "value": current_value,
            "change_5d": round(current_value - prev_value, 2),
            "trend": "strengthening" if current_value > prev_value else "weakening",
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    return result


# =============================================================================
# Market Index Summary
# =============================================================================

def summarize_market_index(kospi_data: list[dict], kosdaq_data: list[dict]) -> dict:
    """
    Summarize market index trends.

    Args:
        kospi_data: KOSPI index time series
        kosdaq_data: KOSDAQ index time series

    Returns:
        dict with market index summaries
    """
    result = {}

    for name, data in [("kospi", kospi_data), ("kosdaq", kosdaq_data)]:
        if not data:
            continue

        closes = [safe_float(d.get("close"), 0) for d in data]
        current = closes[0] if closes else 0

        change_1d = ((closes[0] - closes[1]) / closes[1] * 100) if len(closes) > 1 and closes[1] != 0 else 0
        change_5d = ((closes[0] - closes[4]) / closes[4] * 100) if len(closes) > 4 and closes[4] != 0 else 0
        change_20d = ((closes[0] - closes[19]) / closes[19] * 100) if len(closes) > 19 and closes[19] != 0 else 0

        ma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else current

        result[name] = {
            "current": round(current, 2),
            "change_1d": round(change_1d, 2),
            "change_5d": round(change_5d, 2),
            "change_20d": round(change_20d, 2),
            "trend_20d": "up" if current > ma20 else "down",
            "ma20": round(ma20, 2)
        }

    return result


# =============================================================================
# Main Preprocessing Function
# =============================================================================

def preprocess_all(collected_data: dict) -> dict:
    """
    Preprocess all collected data for agent consumption.

    Args:
        collected_data: Result from collector.collect_stock_data()

    Returns:
        dict with all preprocessed summaries for the analysis agent
    """
    stock_detail = collected_data.get("stock_detail") or {}
    stock_grade = collected_data.get("stock_grade") or {}

    return {
        # Stock identification
        "stock_info": {
            "symbol": stock_detail.get("symbol"),
            "stock_name": stock_detail.get("stock_name"),
            "exchange": stock_detail.get("exchange"),
            "sector": stock_detail.get("theme")  # DB column is 'theme', but we use 'sector' terminology
        },

        # Quant analysis summary
        "quant_summary": summarize_quant_result(stock_grade),

        # Technical analysis
        "technical_summary": analyze_technical_indicators(
            collected_data.get("indicators", [])
        ),

        # Price trend
        "price_trend": calculate_price_trend(
            collected_data.get("prices", [])
        ),

        # Investor trends
        "investor_summary": analyze_investor_trends(
            collected_data.get("investor_trading", [])
        ),

        # Foreign ownership
        "foreign_ownership": analyze_foreign_ownership_trend(
            collected_data.get("foreign_ownership", [])
        ),

        # Market indices
        "market_summary": summarize_market_index(
            collected_data.get("market_index_kospi", []),
            collected_data.get("market_index_kosdaq", [])
        ),

        # Economic indicators (Korea + US)
        "economic_summary": summarize_economic_indicators(
            kr_indicators=collected_data.get("economic_indicators", []),
            us_fed_funds_rate=collected_data.get("us_fed_funds_rate", []),
            us_treasury_yield=collected_data.get("us_treasury_yield", []),
            us_cpi=collected_data.get("us_cpi", []),
            us_unemployment_rate=collected_data.get("us_unemployment_rate", []),
            us_gdp=collected_data.get("us_gdp", []),
            us_pmi=collected_data.get("us_pmi", []),
            us_vix=collected_data.get("us_vix", []),
            us_dollar_index=collected_data.get("us_dollar_index", [])
        ),

        # Raw data reference (for agent if needed)
        "data_availability": {
            "has_indicators": bool(collected_data.get("indicators")),
            "has_prices": bool(collected_data.get("prices")),
            "has_investor_trading": bool(collected_data.get("investor_trading")),
            "has_foreign_ownership": bool(collected_data.get("foreign_ownership")),
            "has_financials": bool(collected_data.get("financials")),
            "has_research_reports": bool(collected_data.get("research_reports")),
            "indicators_days": len(collected_data.get("indicators", [])),
            "prices_days": len(collected_data.get("prices", []))
        }
    }
