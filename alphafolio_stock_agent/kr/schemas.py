"""
Pydantic schemas for input/output validation
Based on the output schema design in 에이전트 작업 계획.md section 9.1

V2 schemas added based on: docs/에이전트 결과물 개선 방안_v2.md
- Data/Narrative separation for improved readability and validation
- Task-driven output structure
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Union
from datetime import date
import re


# =============================================================================
# Input Schemas
# =============================================================================

class StockInput(BaseModel):
    """Input schema for stock analysis request"""
    symbol: str = Field(..., description="Stock code (e.g., 005930)")
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


class GlobalToKoreaImpact(BaseModel):
    """Analysis of how global conditions impact Korean market"""
    impact_level: str = Field(..., description="high | medium | low")
    impact_direction: str = Field(..., description="positive | negative | mixed")
    transmission_channels: list[str] = Field(
        default_factory=list,
        description="Channels through which global factors affect Korea (e.g., foreign capital flow, export demand)"
    )
    key_factors: list[str] = Field(
        default_factory=list,
        description="Key global factors affecting Korea (e.g., USD/KRW, US rates)"
    )
    expected_lag: str = Field(
        default="immediate",
        description="Expected time lag for impact: immediate | 1-2 weeks | 1 month+"
    )
    narrative: str = Field(..., description="Narrative explanation of global→Korea impact")


class KoreaMarketStatus(BaseModel):
    """Korean market specific status"""
    kospi_trend: str = Field(..., description="uptrend | sideways | downtrend")
    kosdaq_trend: str = Field(..., description="uptrend | sideways | downtrend")
    foreign_investor_flow: str = Field(..., description="net_buying | neutral | net_selling")
    foreign_flow_amount: Optional[str] = Field(None, description="Recent foreign net buying amount")
    exchange_rate_pressure: str = Field(..., description="appreciation | stable | depreciation")
    usd_krw_level: Optional[float] = Field(None, description="Current USD/KRW exchange rate")
    domestic_liquidity: str = Field(..., description="abundant | normal | tight")
    market_sentiment: str = Field(..., description="bullish | neutral | bearish")
    key_events: list[str] = Field(
        default_factory=list,
        description="Recent key events affecting Korean market"
    )
    summary: str = Field(..., description="Korean market status summary (2-3 sentences)")


class MarketNewsSentiment(BaseModel):
    """News sentiment analysis summary"""
    korea_news_sentiment: str = Field(..., description="positive | neutral | negative")
    korea_news_highlights: list[str] = Field(
        default_factory=list, description="Key headlines from Korean news"
    )
    global_news_sentiment: str = Field(..., description="positive | neutral | negative")
    global_news_highlights: list[str] = Field(
        default_factory=list, description="Key headlines from global news"
    )
    foreign_view_on_korea: str = Field(..., description="positive | neutral | negative")
    foreign_view_highlights: list[str] = Field(
        default_factory=list, description="Key points from foreign coverage of Korea"
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

    This agent analyzes global market conditions and their impact on Korean market
    to determine the current market regime (Risk-On/Off/Neutral).
    """
    # Core regime determination
    regime: str = Field(..., description="risk_on | risk_off | neutral")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    regime_rationale: str = Field(..., description="2-3 sentence rationale for regime determination")

    # Detailed analysis components
    global_market: GlobalMarketStatus = Field(..., description="Global market analysis")
    global_to_korea_impact: GlobalToKoreaImpact = Field(..., description="Global→Korea impact analysis")
    korea_market: KoreaMarketStatus = Field(..., description="Korean market status")
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
# Agent Output Schemas (Legacy - for backward compatibility)
# =============================================================================

class MarketRegimeOutputSimple(BaseModel):
    """Simplified output schema for Market Regime Agent (legacy)"""
    regime: str = Field(..., description="risk_on | risk_off | neutral")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    evidence: list[dict] = Field(default_factory=list, description="Supporting evidence")


class StockResearchOutputSimple(BaseModel):
    """Simplified output schema for Stock Research Agent (legacy)"""
    sentiment: str = Field(..., description="positive | neutral | negative")
    issues: list[str] = Field(default_factory=list, description="Key issues identified")
    risks: list[str] = Field(default_factory=list, description="Risk factors identified")
    opportunities: list[str] = Field(default_factory=list, description="Opportunities identified")
    sources: list[dict] = Field(default_factory=list, description="Information sources")


# =============================================================================
# Stock Research Agent Output Schemas (Detailed)
# =============================================================================

class ResearchSource(BaseModel):
    """Source information for research data"""
    type: str = Field(..., description="research_report | news | disclosure")
    title: str = Field(..., description="Source title")
    publisher: Optional[str] = Field(None, description="Publisher/securities firm")
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
    sector: str = Field(..., description="Sector name from kr_stock_detail.theme")
    direction: str = Field(..., description="positive | neutral | negative")
    summary: str = Field(..., description="1-2 sentence summary of sector momentum")
    related_news_count: int = Field(default=0, description="Number of related news articles")
    sources: list[ResearchSource] = Field(default_factory=list, description="Supporting sources")


class DisclosureItem(BaseModel):
    """Individual disclosure item"""
    type: str = Field(..., description="earnings | shareholder | executive | dividend | investment | regulatory | other")
    title: str = Field(..., description="Disclosure title")
    date: Optional[str] = Field(None, description="Disclosure date")
    source: str = Field(default="DART", description="DART | KIND")


class DisclosureSummary(BaseModel):
    """Disclosure data summary"""
    recent_count: int = Field(default=0, description="Total number of recent disclosures")
    significant_items: list[DisclosureItem] = Field(default_factory=list, description="Significant disclosure items")


class ResearchDataCollection(BaseModel):
    """Data collection statistics"""
    research_reports: int = Field(default=0, description="Number of research reports collected")
    news_articles: int = Field(default=0, description="Number of news articles collected")
    disclosures: int = Field(default=0, description="Number of disclosures collected")


class ResearchExcluded(BaseModel):
    """Excluded data sources"""
    sns: str = Field(default="Tier 3 excluded", description="SNS exclusion reason")
    community: str = Field(default="Tier 3 excluded", description="Community exclusion reason")


class ResearchMetadata(BaseModel):
    """Metadata for research data collection"""
    data_collection: ResearchDataCollection = Field(default_factory=ResearchDataCollection)
    excluded: ResearchExcluded = Field(default_factory=ResearchExcluded)
    stock_size: Optional[str] = Field(None, description="large | medium | small")
    collection_date: Optional[str] = Field(None, description="Data collection date")


class StockResearchOutput(BaseModel):
    """
    Detailed output schema for Stock Research Agent.

    Purpose: Direction confirmation + enriched explanation
    Note: This does NOT override quant results, only provides context

    Data Sources (Tier 1-2 only):
    - Tier 1: Disclosures (DART + KIND), Research Reports (DB + Naver Finance)
    - Tier 2: News (Naver API)
    - Excluded: SNS, Community (Tier 3 - Phase 2)
    """
    # Basic info
    symbol: str = Field(..., description="Stock code (e.g., 005930)")
    stock_name: str = Field(..., description="Stock name in Korean")
    analysis_date: str = Field(..., description="Analysis date (YYYY-MM-DD)")
    sector: str = Field(..., description="Sector from kr_stock_detail.theme")

    # Overall sentiment
    overall_sentiment: str = Field(..., description="positive | neutral | negative")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")

    # Detailed analysis (3 key areas as per design)
    earnings_outlook: EarningsOutlook = Field(..., description="Earnings outlook analysis")
    risk_issues: list[RiskIssue] = Field(default_factory=list, description="Risk issues identified")
    sector_momentum: SectorMomentum = Field(..., description="Sector momentum analysis")

    # Disclosure summary
    disclosure_summary: DisclosureSummary = Field(
        default_factory=DisclosureSummary,
        description="Summary of recent disclosures"
    )

    # Metadata
    metadata: ResearchMetadata = Field(
        default_factory=ResearchMetadata,
        description="Data collection metadata"
    )


# =============================================================================
# Confidence Evidence Schemas
# =============================================================================

class MacroAdjustment(BaseModel):
    """Macro environment adjustment"""
    factor: str = Field(..., description="Macro factor description")
    impact: str = Field(..., description="Impact description (e.g., +5%)")


class SectorMomentumEvidence(BaseModel):
    """Sector momentum factor"""
    factor: str = Field(..., description="Sector factor description")
    impact: str = Field(..., description="Impact description (e.g., +3%)")


class ConfidenceEvidence(BaseModel):
    """Evidence supporting confidence score"""
    macro_adjustment: Optional[MacroAdjustment] = None
    sector_momentum: Optional[SectorMomentumEvidence] = None


# =============================================================================
# Scenario Schemas
# =============================================================================

class Scenario(BaseModel):
    """Individual scenario (bullish/sideways/bearish)"""
    title: str = Field(..., description="One-line scenario summary")
    probability: int = Field(..., ge=0, le=100, description="Probability percentage")
    confidence: int = Field(..., ge=0, le=100, description="Confidence percentage")
    confidence_evidence: ConfidenceEvidence = Field(..., description="Evidence for confidence")
    support_level: int = Field(..., description="Support price level")
    resistance_level: int = Field(..., description="Resistance price level")
    take_profit: Optional[Union[int, str]] = Field(None, description="Take profit target price (int for bullish/sideways, null for bearish)")
    stop_loss: Optional[Union[int, str]] = Field(None, description="Stop loss price (int for bullish/sideways, str for bearish '1차: xxx원, 2차: xxx원')")
    strategy: Optional[str] = Field(None, description="Recommended strategy for this scenario")
    expected_return: str = Field(..., description="Expected return range (e.g., +10% ~ +25%)")
    triggers: list[str] = Field(default_factory=list, description="Scenario trigger conditions")
    monitoring_points: list[str] = Field(default_factory=list, description="Key monitoring points")
    risk_factors: list[str] = Field(default_factory=list, description="Risk factors for this scenario")


class Scenarios(BaseModel):
    """Three scenarios: bullish, sideways, bearish"""
    bullish: Scenario = Field(..., description="Bullish scenario (profit maximization)")
    sideways: Scenario = Field(..., description="Sideways scenario (position maintenance)")
    bearish: Scenario = Field(..., description="Bearish scenario (loss minimization)")


# =============================================================================
# Market Environment Schema
# =============================================================================

class MarketEnvironment(BaseModel):
    """Market environment summary"""
    macro: str = Field(..., description="Global macro environment summary")
    domestic: str = Field(..., description="Korean domestic economy summary")
    sector: str = Field(..., description="Sector performance summary")
    regime: str = Field(..., description="risk_on | risk_off | neutral")
    regime_evidence: list[dict] = Field(default_factory=list, description="Evidence for regime")


# =============================================================================
# Technical Summary Schema
# =============================================================================

class TechnicalIndicators(BaseModel):
    """Key technical indicators"""
    rsi: float = Field(..., description="RSI value")
    rsi_status: str = Field(..., description="overbought | oversold | neutral")
    macd: str = Field(..., description="MACD status description")
    macd_crossover: Optional[str] = Field(None, description="golden_cross | dead_cross | None")
    stochastic_status: Optional[str] = Field(None, description="Stochastic status")
    adx_status: Optional[str] = Field(None, description="ADX trend strength")


class TechnicalSummary(BaseModel):
    """Technical analysis summary"""
    trend: str = Field(..., description="Overall trend description")
    indicators: TechnicalIndicators = Field(..., description="Key indicator values")
    investor_flow: str = Field(..., description="Investor flow summary")
    volume_status: Optional[str] = Field(None, description="Volume analysis")


# =============================================================================
# Metadata Schema
# =============================================================================

class DataFreshness(BaseModel):
    """Data freshness information"""
    kr_stock_grade: str = Field(..., description="Latest date in kr_stock_grade")
    market_index: str = Field(..., description="Latest date in market_index")
    indicators: Optional[str] = Field(None, description="Latest date in kr_indicators")


class Metadata(BaseModel):
    """Analysis metadata"""
    agent_version: str = Field(default="1.0.0", description="Agent version")
    execution_time_sec: float = Field(..., description="Execution time in seconds")
    llm_cost_usd: float = Field(default=0.0, description="LLM API cost in USD")
    data_freshness: DataFreshness = Field(..., description="Data freshness info")
    retry_count: int = Field(default=0, description="Number of validation retries")


# =============================================================================
# Final Output Schema
# =============================================================================

class StrategyOutput(BaseModel):
    """Final strategy output schema"""
    stock_name: str = Field(..., description="Stock name in Korean")
    symbol: str = Field(..., description="Stock code")
    analysis_date: str = Field(..., description="Analysis date (YYYY-MM-DD)")
    final_grade: str = Field(..., description="Investment grade")

    market_environment: MarketEnvironment = Field(..., description="Market environment analysis")
    technical_summary: TechnicalSummary = Field(..., description="Technical analysis summary")
    scenarios: Scenarios = Field(..., description="Three scenarios with strategies")

    metadata: Metadata = Field(..., description="Analysis metadata")


# =============================================================================
# Validation Helper
# =============================================================================

def validate_scenario_probabilities(scenarios: Scenarios) -> bool:
    """Validate that scenario probabilities sum to 100%"""
    total = (
        scenarios.bullish.probability +
        scenarios.sideways.probability +
        scenarios.bearish.probability
    )
    return total == 100


def validate_strategy_output(output: StrategyOutput) -> list[str]:
    """
    Validate strategy output and return list of errors.

    Returns:
        list of error messages (empty if valid)
    """
    errors = []

    # Check scenario probability sum
    if not validate_scenario_probabilities(output.scenarios):
        total = (
            output.scenarios.bullish.probability +
            output.scenarios.sideways.probability +
            output.scenarios.bearish.probability
        )
        errors.append(f"Scenario probabilities must sum to 100%, got {total}%")

    # Check required fields are not empty
    if not output.market_environment.macro:
        errors.append("market_environment.macro is empty")
    if not output.market_environment.domestic:
        errors.append("market_environment.domestic is empty")
    if not output.market_environment.sector:
        errors.append("market_environment.sector is empty")

    # Check each scenario has required content
    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario = getattr(output.scenarios, scenario_name)
        if not scenario.title:
            errors.append(f"scenarios.{scenario_name}.title is empty")
        if not scenario.triggers:
            errors.append(f"scenarios.{scenario_name}.triggers is empty")
        if not scenario.monitoring_points:
            errors.append(f"scenarios.{scenario_name}.monitoring_points is empty")
        if not scenario.risk_factors:
            errors.append(f"scenarios.{scenario_name}.risk_factors is empty")

    # Check confidence has evidence (macro or sector)
    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario = getattr(output.scenarios, scenario_name)
        has_macro = scenario.confidence_evidence.macro_adjustment is not None
        has_sector = scenario.confidence_evidence.sector_momentum is not None
        if scenario.confidence > 70 and not has_macro and not has_sector:
            errors.append(f"scenarios.{scenario_name} has high confidence but no supporting evidence")

    # [Critical 1-3] Scenario strategy validation
    # 1. Bearish scenario must not contain buy keywords
    bearish_strategy = output.scenarios.bearish.strategy or ""
    buy_keywords = ["매수", "추가 매수", "분할 매수", "저점 매수"]
    for keyword in buy_keywords:
        if keyword in bearish_strategy:
            errors.append(f"bearish scenario strategy contains forbidden keyword: '{keyword}'")

    # 2. Bearish scenario must contain defensive keywords
    defensive_keywords = ["손절", "관망", "현금"]
    has_defensive = any(kw in bearish_strategy for kw in defensive_keywords)
    if bearish_strategy and not has_defensive:
        errors.append("bearish scenario strategy must contain at least one of: 손절, 관망, 현금")

    # Note: grade-probability consistency validation removed (user decision)
    # The validate_grade_probability_consistency() function is kept for potential future use

    # Note: confidence_evidence numeric validation removed (user decision)
    # The validate_confidence_evidence() function is kept for potential future use

    return errors


def validate_grade_probability_consistency(grade: str, bull_prob: int, bear_prob: int) -> list[str]:
    """
    Validate that final_grade is consistent with scenario probabilities.

    Rules:
    - 강력매수: bullish >= 60% AND bearish <= 15%
    - 매수: bullish >= 45% AND bearish <= 25%
    - 중립: default
    - 매도: bearish >= 45% AND bullish <= 25%
    - 강력매도: bearish >= 60% AND bullish <= 15%
    """
    errors = []

    if grade == "강력매수":
        if bull_prob < 60 or bear_prob > 15:
            errors.append(f"강력매수 requires bullish>=60% and bearish<=15%, got bullish={bull_prob}%, bearish={bear_prob}%")
    elif grade == "매수":
        if bull_prob < 45 or bear_prob > 25:
            errors.append(f"매수 requires bullish>=45% and bearish<=25%, got bullish={bull_prob}%, bearish={bear_prob}%")
    elif grade == "매도":
        if bear_prob < 45 or bull_prob > 25:
            errors.append(f"매도 requires bearish>=45% and bullish<=25%, got bullish={bull_prob}%, bearish={bear_prob}%")
    elif grade == "강력매도":
        if bear_prob < 60 or bull_prob > 15:
            errors.append(f"강력매도 requires bearish>=60% and bullish<=15%, got bullish={bull_prob}%, bearish={bear_prob}%")

    return errors


def validate_confidence_evidence(scenarios: Scenarios) -> list[str]:
    """
    Validate that confidence_evidence contains specific, quantitative descriptions.

    Rules:
    - macro_adjustment.factor must contain numeric values (indicators like VIX, rates)
    - sector_momentum.factor must contain numeric values (rankings, percentages)
    """
    import re
    errors = []

    for scenario_name in ["bullish", "sideways", "bearish"]:
        scenario = getattr(scenarios, scenario_name)
        evidence = scenario.confidence_evidence

        # Check macro_adjustment.factor has numeric values
        if evidence.macro_adjustment:
            macro_factor = evidence.macro_adjustment.factor or ""
            has_numeric = bool(re.search(r'\d+(\.\d+)?', macro_factor))
            if macro_factor and not has_numeric:
                errors.append(
                    f"scenarios.{scenario_name}.confidence_evidence.macro_adjustment.factor "
                    f"must contain specific numeric values (e.g., VIX 18.5, rate 4.25%)"
                )

        # Check sector_momentum.factor has numeric values
        if evidence.sector_momentum:
            sector_factor = evidence.sector_momentum.factor or ""
            has_numeric = bool(re.search(r'\d+(\.\d+)?', sector_factor))
            if sector_factor and not has_numeric:
                errors.append(
                    f"scenarios.{scenario_name}.confidence_evidence.sector_momentum.factor "
                    f"must contain specific numeric values (e.g., ranking, percentage)"
                )

    return errors


# =============================================================================
# V2 Schemas: Data/Narrative Separation
# Based on: docs/에이전트 결과물 개선 방안_v2.md
# =============================================================================

# -----------------------------------------------------------------------------
# Market Environment V2
# -----------------------------------------------------------------------------

class MarketEnvironmentData(BaseModel):
    """System-filled data for market environment (read-only for LLM)"""
    vix: Optional[float] = Field(None, description="VIX index value")
    vix_status: Optional[str] = Field(None, description="low_fear | moderate | high_fear | extreme_fear")
    fed_rate: Optional[float] = Field(None, description="US Fed Funds Rate")
    korea_base_rate: Optional[float] = Field(None, description="Korea base rate")
    dollar_index: Optional[float] = Field(None, description="Dollar index value")
    usd_krw: Optional[float] = Field(None, description="USD/KRW exchange rate")
    regime: str = Field(..., description="risk_on | risk_off | neutral")


class MarketEnvironmentNarrative(BaseModel):
    """LLM-generated narrative for market environment"""
    global_env: str = Field(..., description="Global market environment narrative (2-3 sentences)")
    domestic: str = Field(..., description="Korean domestic economy narrative (2-3 sentences)")
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
    current_price: int = Field(..., description="Current stock price")
    ma5: Optional[float] = Field(None, description="5-day moving average")
    ma20: Optional[float] = Field(None, description="20-day moving average")
    ma60: Optional[float] = Field(None, description="60-day moving average")
    rsi: float = Field(..., description="RSI value")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram value")
    adx: Optional[float] = Field(None, description="ADX value")
    bollinger_upper: Optional[float] = Field(None, description="Bollinger upper band")
    bollinger_lower: Optional[float] = Field(None, description="Bollinger lower band")
    foreign_consecutive_buy: Optional[int] = Field(None, description="Foreign consecutive buy days")
    institutional_consecutive_buy: Optional[int] = Field(None, description="Institutional consecutive buy days")
    volume_ratio: Optional[float] = Field(None, description="Volume ratio vs 20-day average")


class TechnicalSummaryNarrative(BaseModel):
    """LLM-generated narrative for technical summary"""
    price_trend: str = Field(..., description="Price trend narrative (2-3 sentences)")
    indicators: str = Field(..., description="Technical indicators narrative (2-3 sentences)")
    investor_flow: str = Field(..., description="Investor flow narrative (1-2 sentences)")
    volume_analysis: str = Field(..., description="Volume analysis narrative (1 sentence)")


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
    support_level: Optional[int] = Field(None, description="Support price level")
    resistance_level: Optional[int] = Field(None, description="Resistance price level")
    take_profit: Optional[Union[int, str]] = Field(None, description="Take profit target price (int for bullish/sideways, null for bearish)")
    stop_loss: Optional[Union[int, str]] = Field(None, description="Stop loss price (int for bullish/sideways, str for bearish '1차: xxx원, 2차: xxx원')")
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

    Key differences from V1:
    - Each section has 'data' (system-filled) and 'narrative' (LLM-generated)
    - LLM can only write narrative sections
    - All numbers in narrative must match data section
    - triggers must be based on system_triggers
    """
    stock_name: str = Field(..., description="Stock name in Korean")
    symbol: str = Field(..., description="Stock code")
    analysis_date: str = Field(..., description="Analysis date (YYYY-MM-DD)")
    final_grade: str = Field(..., description="Investment grade")

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


def validate_narrative_numbers(narrative_text: str, expected_numbers: list) -> list[str]:
    """
    Validate that numbers in narrative text match expected values.

    Args:
        narrative_text: The narrative text to check
        expected_numbers: List of numbers that should appear in the text

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    # Extract numbers from narrative
    found_numbers = re.findall(r'[\d,]+\.?\d*', narrative_text)
    found_numbers = [n.replace(',', '') for n in found_numbers]

    for expected in expected_numbers:
        expected_str = str(expected).replace(',', '')
        # Allow some tolerance for formatting differences
        if expected_str not in found_numbers and str(int(float(expected_str))) not in found_numbers:
            # Check if it's a close match (within formatting tolerance)
            matched = False
            for found in found_numbers:
                try:
                    if abs(float(found) - float(expected_str)) < 0.01:
                        matched = True
                        break
                except ValueError:
                    continue
            if not matched:
                pass  # Don't error for now, this is optional validation

    return errors


def validate_forbidden_symbols(text: str) -> list[str]:
    """
    Validate that text does not contain forbidden symbols.

    Forbidden: +, -, /, |
    Note: These are allowed in numbers (e.g., -3.5%) but not as connectors

    Args:
        text: Text to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    # Check for arrow symbol
    if '→' in text:
        errors.append("Forbidden symbol found: → (use complete sentences instead)")

    # Check for pipe symbol used as separator
    if ' | ' in text:
        errors.append("Forbidden symbol found: | (use complete sentences instead)")

    # Check for + used as connector (not in numbers)
    if re.search(r'[가-힣]\s*\+\s*[가-힣]', text):
        errors.append("Forbidden symbol found: + used as connector (use complete sentences instead)")

    return errors


def validate_bearish_strategy_v2(strategy: str) -> list[str]:
    """
    Validate bearish scenario strategy content.

    Rules:
    - Must NOT contain buy keywords
    - Must contain at least one defensive keyword

    Args:
        strategy: Bearish strategy text

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check for forbidden buy keywords
    buy_keywords = ["매수", "추가 매수", "분할 매수", "저점 매수"]
    for keyword in buy_keywords:
        if keyword in strategy:
            errors.append(f"Bearish strategy contains forbidden keyword: '{keyword}'")

    # Check for required defensive keywords
    defensive_keywords = ["손절", "관망", "현금"]
    has_defensive = any(kw in strategy for kw in defensive_keywords)
    if not has_defensive:
        errors.append("Bearish strategy must contain at least one of: 손절, 관망, 현금")

    return errors


def validate_triggers_from_system(
    narrative_triggers: list[str],
    system_triggers: dict
) -> list[str]:
    """
    Validate that narrative triggers are based on system_triggers.

    Args:
        narrative_triggers: Triggers from narrative output
        system_triggers: System-provided triggers (buy, sell, hold)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Flatten system triggers
    all_system_triggers = []
    for category in ['buy', 'sell', 'hold']:
        triggers = system_triggers.get(category, [])
        if isinstance(triggers, list):
            all_system_triggers.extend(triggers)

    if not all_system_triggers:
        return errors  # No system triggers to validate against

    # Extract key terms from system triggers for matching
    system_keywords = set()
    for trigger in all_system_triggers:
        # Extract numbers and key terms
        numbers = re.findall(r'\d+', trigger)
        system_keywords.update(numbers)
        # Extract key terms
        for term in ['final_score', '외국인', '기관', '섹터', '손절', '익절', '순매수', '순매도']:
            if term in trigger:
                system_keywords.add(term)

    # Check if at least one narrative trigger contains system keywords
    has_match = False
    for narrative_trigger in narrative_triggers:
        for keyword in system_keywords:
            if keyword in narrative_trigger:
                has_match = True
                break
        if has_match:
            break

    if not has_match and system_keywords:
        errors.append(
            "Narrative triggers must be based on system_triggers. "
            f"Expected keywords: {', '.join(list(system_keywords)[:5])}"
        )

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
    5. Triggers are based on system_triggers (if provided)
    6. Required minimum items in lists

    Args:
        output: The V2 strategy output to validate
        system_triggers: Optional system triggers for validation

    Returns:
        List of error messages (empty if valid)
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
    if not me_narrative.domestic or len(me_narrative.domestic) < 10:
        errors.append("market_environment.narrative.domestic is empty or too short")
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
        me_narrative.domestic,
        me_narrative.sector,
        me_narrative.regime_interpretation,
        ts_narrative.price_trend,
        ts_narrative.indicators,
        ts_narrative.investor_flow,
        ts_narrative.volume_analysis,
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

    # 7. Validate triggers against system_triggers
    if system_triggers:
        for scenario_name, trigger_category in [("bullish", "buy"), ("sideways", "hold"), ("bearish", "sell")]:
            scenario = getattr(output.scenarios, scenario_name)
            trigger_errors = validate_triggers_from_system(
                scenario.narrative.triggers,
                system_triggers
            )
            if trigger_errors:
                for err in trigger_errors:
                    errors.append(f"scenarios.{scenario_name}: {err}")

    return errors
