"""
Pydantic schemas for US stock input/output validation
Based on Korean agent schemas with US-specific modifications

Key differences from Korean (kr/schemas.py):
- Removed GlobalToKoreaImpact (not needed for US)
- Added US-specific fields: iv_percentile, insider_signal, put_call_ratio
- Price fields use float instead of int (US uses $0.01 decimals)
- Grade labels in Korean (7 levels): 강력 매수, 매수, 매수 고려, 중립, 매도 고려, 매도, 강력 매도
- Removed Korean-specific fields (VKOSPI, exchange rate, etc.)

V2 schemas with Data/Narrative separation for improved readability and validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Union
from datetime import date
import re


# =============================================================================
# Input Schemas
# =============================================================================

class StockInput(BaseModel):
    """Input schema for US stock analysis request"""
    symbol: str = Field(..., description="Stock ticker (e.g., AAPL)")
    target_date: Optional[date] = Field(None, description="Analysis reference date")


# =============================================================================
# Market Regime Agent Output Schemas (Detailed)
# =============================================================================

class GlobalMarketIndicator(BaseModel):
    """Global market indicator with value and interpretation"""
    name: str = Field(..., description="Indicator name (e.g., VIX, Credit Spread)")
    value: float = Field(..., description="Current value")
    change: Optional[str] = Field(None, description="Change from previous period")
    signal: str = Field(..., description="risk_on | risk_off | neutral")
    interpretation: str = Field(..., description="Brief interpretation of the value")


class GlobalMarketStatus(BaseModel):
    """Global market status analysis"""
    us_market_regime: str = Field(..., description="US market regime: risk_on | risk_off | neutral")
    vix_level: float = Field(..., description="VIX index level")
    vix_signal: str = Field(..., description="low_fear | moderate | high_fear | extreme_fear")
    credit_spread_signal: str = Field(..., description="tight | normal | wide | very_wide")
    yield_curve_signal: str = Field(..., description="normal | flat | inverted")
    dollar_trend: str = Field(..., description="strengthening | stable | weakening")
    safe_haven_flow: str = Field(..., description="Description of gold/bond flow patterns")
    key_indicators: list[GlobalMarketIndicator] = Field(
        default_factory=list, description="Key global indicators"
    )
    summary: str = Field(..., description="Overall global market summary (2-3 sentences)")


class USMarketStatus(BaseModel):
    """US market specific status"""
    sp500_trend: str = Field(..., description="uptrend | sideways | downtrend")
    nasdaq_trend: str = Field(..., description="uptrend | sideways | downtrend")
    market_breadth: str = Field(..., description="strong | moderate | weak")
    sector_rotation: Optional[str] = Field(None, description="Current sector rotation pattern")
    put_call_ratio: Optional[float] = Field(None, description="Overall market put/call ratio")
    market_sentiment: str = Field(..., description="bullish | neutral | bearish")
    key_events: list[str] = Field(
        default_factory=list,
        description="Recent key events affecting US market"
    )
    summary: str = Field(..., description="US market status summary (2-3 sentences)")


class MarketNewsSentiment(BaseModel):
    """News sentiment analysis summary"""
    market_news_sentiment: str = Field(..., description="positive | neutral | negative")
    market_news_highlights: list[str] = Field(
        default_factory=list, description="Key headlines from market news"
    )
    fed_policy_sentiment: str = Field(..., description="hawkish | neutral | dovish")
    fed_highlights: list[str] = Field(
        default_factory=list, description="Key points about Fed policy"
    )


class InvestmentImplications(BaseModel):
    """Investment implications from market regime"""
    risk_appetite: str = Field(..., description="aggressive | moderate | conservative | defensive")
    position_sizing: str = Field(..., description="Recommended position sizing guidance")
    sector_preference: list[str] = Field(
        default_factory=list,
        description="Preferred sectors in current regime"
    )
    sectors_to_avoid: list[str] = Field(
        default_factory=list,
        description="Sectors to avoid or underweight"
    )
    hedging_recommendation: str = Field(..., description="Hedging strategy recommendation")
    time_horizon: str = Field(..., description="Recommended investment time horizon")
    key_risks: list[str] = Field(
        default_factory=list,
        description="Key risks to monitor"
    )
    action_triggers: list[str] = Field(
        default_factory=list,
        description="Conditions that would trigger regime change"
    )


class MarketRegimeOutput(BaseModel):
    """
    Comprehensive output schema for Market Regime Agent.

    This agent analyzes US market conditions and macro environment
    to determine the current market regime (Risk-On/Off/Neutral).
    """
    # Core regime determination
    regime: str = Field(..., description="risk_on | risk_off | neutral")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    regime_rationale: str = Field(..., description="2-3 sentence rationale for regime determination")

    # Detailed analysis components
    global_market: GlobalMarketStatus = Field(..., description="Global market analysis")
    us_market: USMarketStatus = Field(..., description="US market status")
    news_sentiment: MarketNewsSentiment = Field(..., description="News sentiment analysis")
    investment_implications: InvestmentImplications = Field(..., description="Investment implications")

    # Evidence and metadata
    evidence: list[dict] = Field(default_factory=list, description="Supporting evidence with sources")
    data_sources: list[str] = Field(
        default_factory=list,
        description="List of data sources used (DB tables, search APIs)"
    )
    analysis_timestamp: str = Field(..., description="Analysis timestamp (ISO format)")


# =============================================================================
# Stock Research Agent Output Schemas (Detailed)
# =============================================================================

class ResearchSource(BaseModel):
    """Source information for research data"""
    type: str = Field(..., description="analyst_report | news | sec_filing")
    title: str = Field(..., description="Source title")
    publisher: Optional[str] = Field(None, description="Publisher/source")
    date: Optional[str] = Field(None, description="Publication date")
    url: Optional[str] = Field(None, description="Source URL")


class EarningsOutlook(BaseModel):
    """Earnings outlook analysis"""
    direction: str = Field(..., description="positive | neutral | negative")
    summary: str = Field(..., description="1-2 sentence summary of earnings outlook")
    key_points: list[str] = Field(default_factory=list, description="Key points supporting the outlook")
    sources: list[ResearchSource] = Field(default_factory=list, description="Supporting sources")


class RiskIssue(BaseModel):
    """Individual risk issue identified from research"""
    category: str = Field(..., description="regulatory | market | operational | financial | other")
    severity: str = Field(..., description="high | medium | low")
    title: str = Field(..., description="Risk title (short)")
    summary: str = Field(..., description="Risk description (1-2 sentences)")
    sources: list[ResearchSource] = Field(default_factory=list, description="Supporting sources")


class SectorMomentum(BaseModel):
    """Sector momentum analysis"""
    sector: str = Field(..., description="Sector name from us_stock_basic.sector")
    direction: str = Field(..., description="positive | neutral | negative")
    summary: str = Field(..., description="1-2 sentence summary of sector momentum")
    related_news_count: int = Field(default=0, description="Number of related news articles")
    sources: list[ResearchSource] = Field(default_factory=list, description="Supporting sources")


class InsiderActivity(BaseModel):
    """Insider trading activity summary"""
    signal: str = Field(..., description="STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL")
    net_shares: Optional[int] = Field(None, description="Net shares bought/sold (90 days)")
    net_value: Optional[float] = Field(None, description="Net dollar value of transactions")
    summary: str = Field(..., description="1-2 sentence summary of insider activity")


class ResearchDataCollection(BaseModel):
    """Data collection statistics"""
    news_articles: int = Field(default=0, description="Number of news articles collected")
    insider_transactions: int = Field(default=0, description="Number of insider transactions")
    sec_filings: int = Field(default=0, description="Number of SEC filings found")


class ResearchMetadata(BaseModel):
    """Metadata for research data collection"""
    data_collection: ResearchDataCollection = Field(default_factory=ResearchDataCollection)
    market_cap_tier: Optional[str] = Field(None, description="large | mid | small")
    collection_date: Optional[str] = Field(None, description="Data collection date")


class StockResearchOutput(BaseModel):
    """
    Detailed output schema for US Stock Research Agent.

    Purpose: Direction confirmation + enriched explanation
    Note: This does NOT override quant results, only provides context

    Data Sources:
    - us_news (with sentiment scores)
    - us_insider_transactions
    - Serper API (news, analyst reports, SEC filings)
    """
    # Basic info
    symbol: str = Field(..., description="Stock ticker (e.g., AAPL)")
    stock_name: str = Field(..., description="Company name")
    analysis_date: str = Field(..., description="Analysis date (YYYY-MM-DD)")
    sector: str = Field(..., description="Sector from us_stock_basic.sector")

    # Overall sentiment
    overall_sentiment: str = Field(..., description="positive | neutral | negative")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")

    # Detailed analysis
    earnings_outlook: EarningsOutlook = Field(..., description="Earnings outlook analysis")
    risk_issues: list[RiskIssue] = Field(default_factory=list, description="Risk issues identified")
    sector_momentum: SectorMomentum = Field(..., description="Sector momentum analysis")
    insider_activity: InsiderActivity = Field(..., description="Insider trading activity")

    # Metadata
    metadata: ResearchMetadata = Field(
        default_factory=ResearchMetadata,
        description="Data collection metadata"
    )


# =============================================================================
# V2 Schemas: Data/Narrative Separation
# =============================================================================

# -----------------------------------------------------------------------------
# Market Environment V2
# -----------------------------------------------------------------------------

class MarketEnvironmentData(BaseModel):
    """System-filled data for market environment (read-only for LLM)"""
    vix: Optional[float] = Field(None, description="VIX index value")
    vix_status: Optional[str] = Field(None, description="low_fear | moderate | high_fear | extreme_fear")
    fed_rate: Optional[float] = Field(None, description="US Fed Funds Rate")
    treasury_10y: Optional[float] = Field(None, description="10-year Treasury yield")
    dollar_index: Optional[float] = Field(None, description="Dollar index value")
    credit_spread: Optional[float] = Field(None, description="Credit spread (HY-IG)")
    put_call_ratio: Optional[float] = Field(None, description="Market put/call ratio")
    regime: str = Field(..., description="risk_on | risk_off | neutral")


class MarketEnvironmentNarrative(BaseModel):
    """LLM-generated narrative for market environment"""
    global_env: str = Field(..., description="Global market environment narrative (2-3 sentences)")
    fed_policy: str = Field(..., description="Fed policy and interest rate narrative (2-3 sentences)")
    sector: str = Field(..., description="Sector performance narrative (2-3 sentences)")
    regime_interpretation: str = Field(..., description="Regime interpretation narrative (1-2 sentences)")


class MarketEnvironmentV2(BaseModel):
    """Market environment with data/narrative separation"""
    data: MarketEnvironmentData = Field(..., description="System-filled data")
    narrative: MarketEnvironmentNarrative = Field(..., description="LLM-generated narrative")


# -----------------------------------------------------------------------------
# Technical Summary V2
# -----------------------------------------------------------------------------

class TechnicalSummaryData(BaseModel):
    """System-filled data for technical summary (read-only for LLM)"""
    current_price: float = Field(..., description="Current stock price (USD)")
    ma5: Optional[float] = Field(None, description="5-day moving average")
    ma20: Optional[float] = Field(None, description="20-day moving average")
    ma50: Optional[float] = Field(None, description="50-day moving average")
    ma200: Optional[float] = Field(None, description="200-day moving average")
    rsi: float = Field(..., description="RSI value")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram value")
    adx: Optional[float] = Field(None, description="ADX value")
    bollinger_upper: Optional[float] = Field(None, description="Bollinger upper band")
    bollinger_lower: Optional[float] = Field(None, description="Bollinger lower band")
    # US-specific fields
    iv_percentile: Optional[float] = Field(None, description="IV percentile (0-100)")
    put_call_ratio: Optional[float] = Field(None, description="Stock-specific put/call ratio")
    insider_signal: Optional[str] = Field(None, description="STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL")
    volume_ratio: Optional[float] = Field(None, description="Volume ratio vs 20-day average")


class TechnicalSummaryNarrative(BaseModel):
    """LLM-generated narrative for technical summary"""
    price_trend: str = Field(..., description="Price trend narrative (2-3 sentences)")
    indicators: str = Field(..., description="Technical indicators narrative (2-3 sentences)")
    options_flow: str = Field(..., description="Options flow and IV narrative (1-2 sentences)")
    insider_activity: str = Field(..., description="Insider activity narrative (1 sentence)")


class TechnicalSummaryV2(BaseModel):
    """Technical summary with data/narrative separation"""
    data: TechnicalSummaryData = Field(..., description="System-filled data")
    narrative: TechnicalSummaryNarrative = Field(..., description="LLM-generated narrative")


# -----------------------------------------------------------------------------
# Scenario V2
# -----------------------------------------------------------------------------

class ScenarioData(BaseModel):
    """System-filled data for scenario (read-only for LLM)"""
    probability: int = Field(..., ge=0, le=100, description="Probability percentage")
    support_level: Optional[float] = Field(None, description="Support price level (USD)")
    resistance_level: Optional[float] = Field(None, description="Resistance price level (USD)")
    take_profit: Optional[Union[float, str]] = Field(None, description="Take profit target price (float for bullish/sideways, null for bearish)")
    stop_loss: Optional[Union[float, str]] = Field(None, description="Stop loss price (float for bullish/sideways, str for bearish '1st: $xxx, 2nd: $xxx')")
    expected_return: Optional[str] = Field(None, description="Expected return range")
    sample_count: Optional[int] = Field(None, description="Historical sample count for probability")


class ScenarioNarrative(BaseModel):
    """LLM-generated narrative for scenario"""
    title: str = Field(..., description="Scenario title (1 sentence)")
    probability_explanation: str = Field(..., description="Probability explanation with sample count (1 sentence)")
    confidence_rationale: str = Field(..., description="Confidence rationale (1-2 sentences)")
    strategy: str = Field(..., description="Recommended strategy (2-3 sentences)")
    triggers: list[str] = Field(..., min_length=2, description="Trigger conditions (min 2, based on system_triggers)")
    monitoring_points: list[str] = Field(..., min_length=2, description="Monitoring points (min 2)")
    risk_factors: list[str] = Field(..., min_length=2, description="Risk factors (min 2)")


class ScenarioV2(BaseModel):
    """Individual scenario with data/narrative separation"""
    data: ScenarioData = Field(..., description="System-filled data")
    narrative: ScenarioNarrative = Field(..., description="LLM-generated narrative")


class ScenariosV2(BaseModel):
    """Three scenarios with data/narrative separation"""
    bullish: ScenarioV2 = Field(..., description="Bullish scenario")
    sideways: ScenarioV2 = Field(..., description="Sideways scenario")
    bearish: ScenarioV2 = Field(..., description="Bearish scenario")


# -----------------------------------------------------------------------------
# Metadata V2
# -----------------------------------------------------------------------------

class MetadataV2(BaseModel):
    """Analysis metadata for V2"""
    agent_version: str = Field(default="2.0.0", description="Agent version")
    execution_time_sec: Optional[float] = Field(None, description="Execution time in seconds")
    retry_count: int = Field(default=0, description="Number of validation retries")
    data_freshness: Optional[dict] = Field(None, description="Data freshness info")


# -----------------------------------------------------------------------------
# Final Output V2
# -----------------------------------------------------------------------------

class StrategyOutputV2(BaseModel):
    """
    Final strategy output schema with Data/Narrative separation (V2).

    Key differences from Korean version:
    - Price fields use float (USD with 2 decimals)
    - Grade labels in Korean (7 levels): 강력 매수, 매수, 매수 고려, 중립, 매도 고려, 매도, 강력 매도
    - Added US-specific fields: iv_percentile, put_call_ratio, insider_signal
    - Removed Korean-specific fields: VKOSPI, exchange rate, investor flow
    """
    stock_name: str = Field(..., description="Company name")
    symbol: str = Field(..., description="Stock ticker")
    analysis_date: str = Field(..., description="Analysis date (YYYY-MM-DD)")
    final_grade: str = Field(..., description="Investment grade: 강력 매수 | 매수 | 매수 고려 | 중립 | 매도 고려 | 매도 | 강력 매도")

    market_environment: MarketEnvironmentV2 = Field(..., description="Market environment analysis")
    technical_summary: TechnicalSummaryV2 = Field(..., description="Technical analysis summary")
    scenarios: ScenariosV2 = Field(..., description="Three scenarios with strategies")

    metadata: MetadataV2 = Field(default_factory=MetadataV2, description="Analysis metadata")


# =============================================================================
# V2 Validation Functions
# =============================================================================

def validate_scenario_probabilities_v2(scenarios: ScenariosV2) -> bool:
    """Validate that scenario probabilities sum to 100%"""
    total = (
        scenarios.bullish.data.probability +
        scenarios.sideways.data.probability +
        scenarios.bearish.data.probability
    )
    return total == 100


def validate_forbidden_symbols(text: str) -> list[str]:
    """
    Validate that text does not contain forbidden symbols.

    Forbidden: arrow symbols, pipe as separator
    Note: +/- are allowed in numbers (e.g., -3.5%) but not as connectors
    """
    errors = []
    if '->' in text or '→' in text:
        errors.append("Forbidden symbol found: -> or arrow (use complete sentences instead)")
    if ' | ' in text:
        errors.append("Forbidden symbol found: | (use complete sentences instead)")
    return errors


def validate_bearish_strategy_v2(strategy: str) -> list[str]:
    """
    Validate bearish scenario strategy content.

    Rules:
    - Must NOT contain buy keywords
    - Must contain at least one defensive keyword
    """
    errors = []

    # Check for forbidden buy keywords (English + Korean)
    buy_keywords = [
        # English
        "buy", "accumulate", "add position", "average down",
        # Korean (한글)
        "매수", "강력 매수", "추가 매수", "물타기"
    ]
    strategy_lower = strategy.lower()
    for keyword in buy_keywords:
        if keyword in strategy_lower:
            errors.append(f"Bearish strategy contains forbidden keyword: '{keyword}'")

    # Check for required defensive keywords (English + Korean)
    defensive_keywords = [
        # English
        "stop loss", "cut", "reduce", "exit", "hedge", "cash", "protect",
        # Korean (한글)
        "손절", "손절매", "청산", "비중 축소", "축소", "헤지", "현금", "방어적", "방어", "보호"
    ]
    has_defensive = any(kw in strategy_lower for kw in defensive_keywords)
    if not has_defensive:
        errors.append("Bearish strategy must contain at least one defensive keyword: stop loss, cut, reduce, exit, hedge, cash, protect, 손절, 비중 축소, 현금")

    return errors


def validate_strategy_output_v2(
    output: StrategyOutputV2,
    system_triggers: dict = None
) -> list[str]:
    """
    Comprehensive validation for V2 strategy output.

    Validates:
    1. Scenario probabilities sum to 100%
    2. All narrative sections are non-empty
    3. Bearish strategy follows rules
    4. No forbidden symbols in narratives
    5. Required minimum items in lists
    """
    errors = []

    # 1. Check scenario probability sum
    if not validate_scenario_probabilities_v2(output.scenarios):
        total = (
            output.scenarios.bullish.data.probability +
            output.scenarios.sideways.data.probability +
            output.scenarios.bearish.data.probability
        )
        errors.append(f"Scenario probabilities must sum to 100%, got {total}%")

    # 2. Check market environment narratives are non-empty
    me_narrative = output.market_environment.narrative
    if not me_narrative.global_env or len(me_narrative.global_env) < 10:
        errors.append("market_environment.narrative.global_env is empty or too short")
    if not me_narrative.fed_policy or len(me_narrative.fed_policy) < 10:
        errors.append("market_environment.narrative.fed_policy is empty or too short")
    if not me_narrative.sector or len(me_narrative.sector) < 10:
        errors.append("market_environment.narrative.sector is empty or too short")

    # 3. Check technical summary narratives are non-empty
    ts_narrative = output.technical_summary.narrative
    if not ts_narrative.price_trend or len(ts_narrative.price_trend) < 10:
        errors.append("technical_summary.narrative.price_trend is empty or too short")
    if not ts_narrative.indicators or len(ts_narrative.indicators) < 10:
        errors.append("technical_summary.narrative.indicators is empty or too short")

    # 4. Check scenario narratives
    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario = getattr(output.scenarios, scenario_name)
        narrative = scenario.narrative

        # Check required fields
        if not narrative.title:
            errors.append(f"scenarios.{scenario_name}.narrative.title is empty")
        if not narrative.strategy:
            errors.append(f"scenarios.{scenario_name}.narrative.strategy is empty")

        # Check list lengths
        if len(narrative.triggers) < 2:
            errors.append(f"scenarios.{scenario_name}.narrative.triggers must have at least 2 items")
        if len(narrative.monitoring_points) < 2:
            errors.append(f"scenarios.{scenario_name}.narrative.monitoring_points must have at least 2 items")
        if len(narrative.risk_factors) < 2:
            errors.append(f"scenarios.{scenario_name}.narrative.risk_factors must have at least 2 items")

    # 5. Validate bearish strategy
    bearish_errors = validate_bearish_strategy_v2(output.scenarios.bearish.narrative.strategy)
    errors.extend(bearish_errors)

    # 6. Check for forbidden symbols in all narratives
    all_narratives = [
        me_narrative.global_env,
        me_narrative.fed_policy,
        me_narrative.sector,
        me_narrative.regime_interpretation,
        ts_narrative.price_trend,
        ts_narrative.indicators,
        ts_narrative.options_flow,
        ts_narrative.insider_activity,
    ]
    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario = getattr(output.scenarios, scenario_name)
        all_narratives.extend([
            scenario.narrative.title,
            scenario.narrative.probability_explanation,
            scenario.narrative.confidence_rationale,
            scenario.narrative.strategy,
        ])
        all_narratives.extend(scenario.narrative.triggers)
        all_narratives.extend(scenario.narrative.monitoring_points)
        all_narratives.extend(scenario.narrative.risk_factors)

    for narrative_text in all_narratives:
        if narrative_text:
            symbol_errors = validate_forbidden_symbols(narrative_text)
            errors.extend(symbol_errors)

    return errors


# =============================================================================
# Price Rounding Helper (US-specific)
# =============================================================================

def round_price(price: float) -> float:
    """
    Round price to US stock tick size.

    US Decimalization:
    - $1.00 and above: $0.01 (2 decimal places)
    - Below $1.00: $0.0001 (4 decimal places, penny stocks)
    """
    if price >= 1.0:
        return round(price, 2)
    else:
        return round(price, 4)
