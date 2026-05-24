"""
Data preprocessor module for US stocks
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


def format_usd(value: float) -> str:
    """Format number as USD currency"""
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:,.2f}K"
    return f"${value:,.2f}"


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


def calculate_percentile(value: float, values: list[float]) -> float:
    """Calculate percentile of a value within a list of values"""
    if not values or value is None:
        return 50.0
    sorted_values = sorted(values)
    count_below = sum(1 for v in sorted_values if v < value)
    return round((count_below / len(sorted_values)) * 100, 1)


# =============================================================================
# Technical Indicators Analysis
# =============================================================================

def analyze_technical_indicators(indicators: list[dict]) -> dict:
    """
    Analyze technical indicators and generate signals.

    Args:
        indicators: Time series data from us_indicators (most recent first)

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
            "value": round(macd_value, 4),
            "signal": round(macd_signal, 4),
            "histogram": round(macd_hist, 4),
            "status": macd_status,
            "crossover": macd_crossover
        },
        "bollinger": {
            "upper": round(upper, 2),
            "middle": round(middle, 2),
            "lower": round(lower, 2)
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
# Options Data Analysis (US-specific)
# =============================================================================

def analyze_options_data(option_summary: list[dict]) -> dict:
    """
    Analyze options data for sentiment signals.

    Args:
        option_summary: Time series from us_option_daily_summary (252 days for IV percentile)

    Returns:
        dict with put/call ratio, IV percentile, GEX analysis
    """
    if not option_summary:
        return {}

    latest = option_summary[0]

    # Put/Call Ratio
    put_volume = safe_float(latest.get("put_volume"), 0)
    call_volume = safe_float(latest.get("call_volume"), 0)
    put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0

    if put_call_ratio < 0.7:
        pc_signal = "bullish"
    elif put_call_ratio > 1.0:
        pc_signal = "bearish"
    else:
        pc_signal = "neutral"

    # IV (Implied Volatility) and IV Percentile
    current_iv = safe_float(latest.get("implied_volatility"), 0)
    all_ivs = [safe_float(d.get("implied_volatility"), 0) for d in option_summary if d.get("implied_volatility")]

    iv_percentile = calculate_percentile(current_iv, all_ivs) if all_ivs else 50.0

    if iv_percentile < 20:
        iv_status = "low"
    elif iv_percentile > 80:
        iv_status = "high"
    else:
        iv_status = "moderate"

    # GEX (Gamma Exposure)
    net_gex = safe_float(latest.get("net_gex"), 0)
    gamma_flip = safe_float(latest.get("gamma_flip_level"), 0)

    if net_gex > 0:
        gex_signal = "stable"  # Positive GEX = dealers hedge by buying dips, selling rips
    else:
        gex_signal = "volatile"  # Negative GEX = dealers amplify moves

    return {
        "put_call_ratio": {
            "value": round(put_call_ratio, 2),
            "signal": pc_signal,
            "put_volume": int(put_volume),
            "call_volume": int(call_volume)
        },
        "implied_volatility": {
            "current": round(current_iv, 2),
            "percentile": round(iv_percentile, 1),
            "status": iv_status
        },
        "gamma_exposure": {
            "net_gex": round(net_gex, 2),
            "gamma_flip_level": round(gamma_flip, 2) if gamma_flip else None,
            "signal": gex_signal
        }
    }


# =============================================================================
# Insider Trading Analysis (US-specific)
# =============================================================================

def analyze_insider_transactions(transactions: list[dict]) -> dict:
    """
    Analyze insider trading activity.

    Args:
        transactions: List from us_insider_transactions (90 days)

    Returns:
        dict with insider signal and summary
    """
    if not transactions:
        return {
            "signal": "NEUTRAL",
            "net_shares": 0,
            "net_value": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "summary": "No recent insider transactions"
        }

    buy_shares = 0
    sell_shares = 0
    buy_value = 0.0
    sell_value = 0.0
    buy_count = 0
    sell_count = 0

    # CEO/CFO transactions get 2x weight
    executive_titles = ["CEO", "CFO", "Chief Executive", "Chief Financial"]

    for tx in transactions:
        tx_type = tx.get("transaction_type", "").upper()
        shares = safe_int(tx.get("shares"), 0)
        price = safe_float(tx.get("price"), 0)
        value = shares * price
        title = tx.get("position", "")

        # Apply weight for executives
        weight = 2 if any(et in title for et in executive_titles) else 1

        if "BUY" in tx_type or "PURCHASE" in tx_type:
            buy_shares += shares * weight
            buy_value += value * weight
            buy_count += 1
        elif "SELL" in tx_type or "SALE" in tx_type:
            sell_shares += shares * weight
            sell_value += value * weight
            sell_count += 1

    net_shares = buy_shares - sell_shares
    net_value = buy_value - sell_value

    # Determine signal
    if net_shares > 10000:
        signal = "STRONG_BUY"
    elif net_shares > 0:
        signal = "BUY"
    elif net_shares < -10000:
        signal = "STRONG_SELL"
    elif net_shares < 0:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "net_shares": net_shares,
        "net_value": round(net_value, 2),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "summary": f"{buy_count} buys, {sell_count} sells in 90 days"
    }


# =============================================================================
# News Sentiment Analysis (US-specific)
# =============================================================================

def analyze_news_sentiment(news_data: list[dict]) -> dict:
    """
    Analyze news sentiment scores.

    Args:
        news_data: List from us_news with sentiment scores

    Returns:
        dict with aggregated sentiment analysis
    """
    if not news_data:
        return {
            "overall_sentiment": "neutral",
            "avg_score": 0.0,
            "article_count": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "recent_headlines": []
        }

    scores = []
    bullish_count = 0
    bearish_count = 0
    recent_headlines = []

    for article in news_data[:20]:  # Analyze top 20 articles
        score = safe_float(article.get("ticker_sentiment_score") or article.get("overall_sentiment_score"), 0)
        scores.append(score)

        if score > 0.15:
            bullish_count += 1
        elif score < -0.15:
            bearish_count += 1

        if len(recent_headlines) < 5:
            recent_headlines.append({
                "title": article.get("title", "")[:100],
                "sentiment": "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral",
                "score": round(score, 2)
            })

    avg_score = sum(scores) / len(scores) if scores else 0

    # Interpret average sentiment
    if avg_score > 0.35:
        overall_sentiment = "bullish"
    elif avg_score > 0.15:
        overall_sentiment = "somewhat_bullish"
    elif avg_score < -0.35:
        overall_sentiment = "bearish"
    elif avg_score < -0.15:
        overall_sentiment = "somewhat_bearish"
    else:
        overall_sentiment = "neutral"

    return {
        "overall_sentiment": overall_sentiment,
        "avg_score": round(avg_score, 3),
        "article_count": len(news_data),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "recent_headlines": recent_headlines
    }


# =============================================================================
# Price Trend Analysis
# =============================================================================

def calculate_price_trend(prices: list[dict], stock_basic: dict = None) -> dict:
    """
    Analyze price trends and momentum.

    Args:
        prices: Time series from us_daily (most recent first)
        stock_basic: Basic stock info containing 200-day MA

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
    ma50 = sum(closes[:50]) / 50 if len(closes) >= 50 else current_price

    # Get 200-day MA from stock_basic if available
    ma200 = None
    if stock_basic:
        ma200 = safe_float(stock_basic.get("day200movingaverage"), None)

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
        "current_price": round(current_price, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2) if ma200 else None,
        "trend_5d": "up" if current_price > ma5 else "down",
        "trend_20d": "up" if current_price > ma20 else "down",
        "trend_50d": "up" if current_price > ma50 else "down",
        "trend_200d": "up" if ma200 and current_price > ma200 else "down" if ma200 else "unknown",
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
    Summarize key information from us_stock_grade.

    Args:
        stock_grade: Single record from us_stock_grade

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
    us_fed_funds_rate: list[dict] = None,
    us_treasury_yield: list[dict] = None,
    us_cpi: list[dict] = None,
    us_unemployment_rate: list[dict] = None,
    us_gdp: list[dict] = None,
    us_pmi: list[dict] = None,
    us_vix: list[dict] = None,
    us_dollar_index: list[dict] = None,
    us_credit_spread: list[dict] = None,
    us_move_index: list[dict] = None
) -> dict:
    """
    Summarize US economic indicators.

    Args:
        us_*: US economic indicator time series

    Returns:
        dict with summarized economic data
    """
    result = {
        "us": {},
        "market_sentiment": {}
    }

    # US Fed Funds Rate
    if us_fed_funds_rate:
        latest = us_fed_funds_rate[0]
        result["us"]["fed_funds_rate"] = {
            "value": safe_float(latest.get("value")),
            "unit": "%",
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # US Treasury Yield (10Y)
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
            vix_status = "extreme_fear"
        elif vix_value > 20:
            vix_status = "high_fear"
        elif vix_value > 15:
            vix_status = "moderate"
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

    # Credit Spread (HY-IG)
    if us_credit_spread:
        latest = us_credit_spread[0]
        spread_value = safe_float(latest.get("value"))
        if spread_value > 5:
            spread_status = "very_wide"
        elif spread_value > 4:
            spread_status = "wide"
        elif spread_value > 3:
            spread_status = "normal"
        else:
            spread_status = "tight"
        result["market_sentiment"]["credit_spread"] = {
            "value": spread_value,
            "status": spread_status,
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    # MOVE Index (Bond Volatility)
    if us_move_index:
        latest = us_move_index[0]
        move_value = safe_float(latest.get("value"))
        result["market_sentiment"]["move_index"] = {
            "value": move_value,
            "date": str(latest.get("date")) if latest.get("date") else None
        }

    return result


# =============================================================================
# Market Index Summary
# =============================================================================

def summarize_market_index(sp500_data: list[dict], nasdaq_data: list[dict]) -> dict:
    """
    Summarize market index trends.

    Args:
        sp500_data: S&P 500 index time series
        nasdaq_data: NASDAQ index time series

    Returns:
        dict with market index summaries
    """
    result = {}

    for name, data in [("sp500", sp500_data), ("nasdaq", nasdaq_data)]:
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
    stock_basic = collected_data.get("stock_basic") or {}
    stock_grade = collected_data.get("stock_grade") or {}

    return {
        # Stock identification
        "stock_info": {
            "symbol": stock_basic.get("symbol"),
            "stock_name": stock_basic.get("stock_name"),
            "sector": stock_basic.get("sector"),
            "industry": stock_basic.get("industry"),
            "description": stock_basic.get("description", "")[:500]  # Truncate long descriptions
        },

        # Quant analysis summary
        "quant_summary": summarize_quant_result(stock_grade),

        # Technical analysis
        "technical_summary": analyze_technical_indicators(
            collected_data.get("indicators", [])
        ),

        # Price trend
        "price_trend": calculate_price_trend(
            collected_data.get("daily_prices", []),
            stock_basic
        ),

        # Options analysis (US-specific)
        "options_summary": analyze_options_data(
            collected_data.get("option_summary", [])
        ),

        # Insider trading (US-specific)
        "insider_summary": analyze_insider_transactions(
            collected_data.get("insider_transactions", [])
        ),

        # News sentiment (US-specific)
        "news_sentiment": analyze_news_sentiment(
            collected_data.get("stock_news", [])
        ),

        # Market indices
        "market_summary": summarize_market_index(
            collected_data.get("market_index_sp500", []),
            collected_data.get("market_index_nasdaq", [])
        ),

        # Economic indicators (US)
        "economic_summary": summarize_economic_indicators(
            us_fed_funds_rate=collected_data.get("us_fed_funds_rate", []),
            us_treasury_yield=collected_data.get("us_treasury_yield", []),
            us_cpi=collected_data.get("us_cpi", []),
            us_unemployment_rate=collected_data.get("us_unemployment_rate", []),
            us_gdp=collected_data.get("us_gdp", []),
            us_pmi=collected_data.get("us_pmi", []),
            us_vix=collected_data.get("us_vix", []),
            us_dollar_index=collected_data.get("us_dollar_index", []),
            us_credit_spread=collected_data.get("us_credit_spread", []),
            us_move_index=collected_data.get("us_move_index", [])
        ),

        # Raw data reference (for agent if needed)
        "data_availability": {
            "has_indicators": bool(collected_data.get("indicators")),
            "has_prices": bool(collected_data.get("daily_prices")),
            "has_options": bool(collected_data.get("option_summary")),
            "has_insider": bool(collected_data.get("insider_transactions")),
            "has_news": bool(collected_data.get("stock_news")),
            "has_financials": bool(collected_data.get("income_statement")),
            "indicators_days": len(collected_data.get("indicators", [])),
            "prices_days": len(collected_data.get("daily_prices", [])),
            "options_days": len(collected_data.get("option_summary", []))
        }
    }
