"""
Interpretation and Insights Module
해석 및 통찰 모듈

Provides Why/How insights based on quantitative metrics:
1. Factor Detailed Analysis (Strength/Weakness)
2. Industry Positioning
3. Time Series Trend Analysis
4. Risk Profile
5. Valuation Analysis
6. Signal Light System
"""

import logging
from typing import Dict, List, Tuple, Optional

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InterpretationEngine:
    """
    Generate investment insights from quantitative analysis results
    """

    def __init__(self, symbol: str, factor_scores: Dict, additional_metrics: Dict,
                 weighted_factor_scores: Dict, final_score: float, final_grade: str):
        """
        Initialize interpretation engine

        Args:
            symbol: Stock symbol
            factor_scores: Factor scores from kr_main
            additional_metrics: Additional metrics from kr_additional_metrics
            weighted_factor_scores: Weighted factor scores with contributions
            final_score: Final composite score
            final_grade: Final grade
        """
        self.symbol = symbol
        self.factor_scores = factor_scores
        self.additional_metrics = additional_metrics
        self.weighted_factor_scores = weighted_factor_scores
        self.final_score = final_score
        self.final_grade = final_grade

    def analyze_factor_details(self) -> Dict:
        """
        1. Analyze factor strengths and weaknesses
        Provide Why insights on factor performance

        Returns:
            Dict with strength_factor, weakness_factor, and explanations
        """
        try:
            # Get factor contributions
            factors = []
            for factor_name in ['value', 'quality', 'momentum', 'growth']:
                factor_data = self.factor_scores.get(factor_name, {})
                weighted_result = factor_data.get('weighted_result')
                if weighted_result:
                    factors.append({
                        'name': factor_name,
                        'name_ko': {'value': '가치', 'quality': '품질',
                                   'momentum': '모멘텀', 'growth': '성장'}[factor_name],
                        'score': weighted_result['weighted_score'],
                        'strategies': factor_data.get('strategies', {})
                    })

            # Sort by score
            factors.sort(key=lambda x: x['score'], reverse=True)

            # Identify strength and weakness
            strength_factor = factors[0] if factors else None
            weakness_factor = factors[-1] if factors else None

            # Get top strategies for strength factor
            strength_strategies = []
            if strength_factor:
                strategies = strength_factor['strategies']
                top_3 = sorted(
                    [(k, v) for k, v in strategies.items() if v is not None],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                strength_strategies = top_3

            # Get bottom strategies for weakness factor
            weakness_strategies = []
            if weakness_factor:
                strategies = weakness_factor['strategies']
                bottom_3 = sorted(
                    [(k, v) for k, v in strategies.items() if v is not None],
                    key=lambda x: x[1]
                )[:3]
                weakness_strategies = bottom_3

            # Generate interpretation
            interpretation = ""
            if strength_factor:
                score = strength_factor['score']
                name = strength_factor['name_ko']
                if score >= 80:
                    level = "매우 우수"
                elif score >= 70:
                    level = "우수"
                elif score >= 60:
                    level = "양호"
                else:
                    level = "보통"
                interpretation += f"{name} 팩터 {level} ({score:.1f}점)"

                if strength_strategies:
                    top_strategy = strength_strategies[0]
                    interpretation += f", {top_strategy[0].split('_', 1)[1].replace('_', ' ')} 전략이 강점"

            if weakness_factor:
                score = weakness_factor['score']
                name = weakness_factor['name_ko']
                if score < 40:
                    level = "부진"
                elif score < 50:
                    level = "약세"
                else:
                    level = "보통 이하"
                interpretation += f" / {name} 팩터 {level} ({score:.1f}점)"

            return {
                'strength_factor': strength_factor,
                'weakness_factor': weakness_factor,
                'strength_strategies': strength_strategies,
                'weakness_strategies': weakness_strategies,
                'interpretation': interpretation
            }

        except Exception as e:
            logger.error(f"Factor detail analysis failed: {e}")
            return {}

    def analyze_industry_positioning(self) -> Dict:
        """
        2. Analyze industry positioning
        Provide How insights on competitive position

        Returns:
            Dict with industry rank, percentile, and positioning
        """
        try:
            m = self.additional_metrics
            rank = m.get('industry_rank')
            percentile = m.get('industry_percentile')

            if rank is None or percentile is None:
                return {'positioning': '업종 정보 없음'}

            # Determine positioning
            if percentile >= 90:
                tier = "업종 최상위권"
                desc = "업종 내 압도적 우위"
            elif percentile >= 75:
                tier = "업종 상위권"
                desc = "업종 내 우량주"
            elif percentile >= 50:
                tier = "업종 중상위권"
                desc = "업종 내 평균 이상"
            elif percentile >= 25:
                tier = "업종 중하위권"
                desc = "업종 내 평균 이하"
            else:
                tier = "업종 하위권"
                desc = "업종 내 경쟁력 부족"

            interpretation = f"{tier} (업종 순위 {rank}위, 상위 {100-percentile:.1f}%), {desc}"

            return {
                'rank': rank,
                'percentile': percentile,
                'tier': tier,
                'description': desc,
                'interpretation': interpretation
            }

        except Exception as e:
            logger.error(f"Industry positioning analysis failed: {e}")
            return {}

    def analyze_time_series_trend(self) -> Dict:
        """
        3. Analyze time series trends
        Provide Why insights on momentum and flow

        Returns:
            Dict with trend analysis and interpretation
        """
        try:
            m = self.additional_metrics

            # Factor momentum trends
            factor_trends = []
            for factor_en, factor_ko in [('value', '가치'), ('quality', '품질'),
                                          ('momentum', '모멘텀'), ('growth', '성장')]:
                momentum = m.get(f'{factor_en}_momentum')
                if momentum is not None:
                    try:
                        if momentum > 10:
                            trend = "상승"
                        elif momentum > -10:
                            trend = "중립"
                        else:
                            trend = "하락"
                        factor_trends.append({
                            'factor': factor_ko,
                            'momentum': momentum,
                            'trend': trend
                        })
                    except TypeError:
                        # Skip if momentum is not comparable
                        continue

            # Investor flow
            inst_net = m.get('inst_net_30d', 0)
            foreign_net = m.get('foreign_net_30d', 0)

            inst_trend = "순매수" if inst_net > 0 else "순매도" if inst_net < 0 else "중립"
            foreign_trend = "순매수" if foreign_net > 0 else "순매도" if foreign_net < 0 else "중립"

            # Overall interpretation
            positive_factors = sum(1 for t in factor_trends if t['trend'] == "상승")
            negative_factors = sum(1 for t in factor_trends if t['trend'] == "하락")

            if positive_factors >= 3:
                factor_phase = "팩터 전반적 상승 추세"
            elif negative_factors >= 3:
                factor_phase = "팩터 전반적 하락 추세"
            else:
                factor_phase = "팩터 혼재 양상"

            smart_money = []
            if inst_net > 0:
                smart_money.append("기관 매집")
            if foreign_net > 0:
                smart_money.append("외국인 매집")

            if len(smart_money) == 2:
                flow_phase = "기관/외국인 동반 매수 (긍정적)"
            elif len(smart_money) == 1:
                flow_phase = f"{smart_money[0]} 진행 중"
            else:
                if inst_net < 0 and foreign_net < 0:
                    flow_phase = "기관/외국인 동반 매도 (부정적)"
                else:
                    flow_phase = "수급 중립"

            interpretation = f"{factor_phase}, {flow_phase}"

            return {
                'factor_trends': factor_trends,
                'inst_trend': inst_trend,
                'foreign_trend': foreign_trend,
                'inst_net_30d': inst_net,
                'foreign_net_30d': foreign_net,
                'phase': factor_phase,
                'flow_phase': flow_phase,
                'interpretation': interpretation
            }

        except Exception as e:
            logger.error(f"Time series trend analysis failed: {e}")
            return {}

    def analyze_risk_profile(self) -> Dict:
        """
        4. Analyze risk profile
        Provide How insights on risk characteristics

        Returns:
            Dict with risk assessment and interpretation
        """
        try:
            m = self.additional_metrics

            volatility = m.get('volatility_annual')
            mdd = m.get('max_drawdown_1y')
            var_95 = m.get('var_95')
            beta = m.get('beta')

            risk_factors = []

            # Volatility assessment
            if volatility:
                if volatility < 20:
                    vol_risk = "낮음"
                elif volatility < 30:
                    vol_risk = "보통"
                elif volatility < 40:
                    vol_risk = "높음"
                else:
                    vol_risk = "매우 높음"
                risk_factors.append(f"변동성 {vol_risk} ({volatility:.1f}%)")

            # MDD assessment
            if mdd:
                if mdd > -10:
                    mdd_risk = "낮음"
                elif mdd > -20:
                    mdd_risk = "보통"
                elif mdd > -30:
                    mdd_risk = "높음"
                else:
                    mdd_risk = "매우 높음"
                risk_factors.append(f"낙폭 리스크 {mdd_risk} ({mdd:.1f}%)")

            # Beta assessment
            if beta:
                if beta < 0.8:
                    beta_type = "방어적"
                elif beta < 1.2:
                    beta_type = "시장 평균"
                else:
                    beta_type = "공격적"
                risk_factors.append(f"베타 {beta_type} ({beta:.2f})")

            # Overall risk level
            risk_score = 0
            if volatility:
                if volatility >= 40:
                    risk_score += 3
                elif volatility >= 30:
                    risk_score += 2
                elif volatility >= 20:
                    risk_score += 1

            if mdd:
                if mdd <= -30:
                    risk_score += 3
                elif mdd <= -20:
                    risk_score += 2
                elif mdd <= -10:
                    risk_score += 1

            if risk_score >= 5:
                overall_risk = "고위험"
                recommendation = "공격적 투자자에게 적합"
            elif risk_score >= 3:
                overall_risk = "중위험"
                recommendation = "일반 투자자에게 적합"
            else:
                overall_risk = "저위험"
                recommendation = "보수적 투자자에게 적합"

            interpretation = f"{overall_risk} 프로파일 - {', '.join(risk_factors[:2])}"

            return {
                'volatility': volatility,
                'mdd': mdd,
                'var_95': var_95,
                'beta': beta,
                'risk_factors': risk_factors,
                'overall_risk': overall_risk,
                'recommendation': recommendation,
                'interpretation': interpretation
            }

        except Exception as e:
            logger.error(f"Risk profile analysis failed: {e}")
            return {}

    def analyze_valuation(self) -> Dict:
        """
        5. Analyze valuation
        Provide Why insights on value proposition

        Returns:
            Dict with valuation assessment
        """
        try:
            # Get value factor score and strategies (handle None weighted_result)
            value_factor = self.factor_scores.get('value', {})
            value_result = value_factor.get('weighted_result') or {}
            value_score = value_result.get('weighted_score', 0)
            value_strategies = value_factor.get('strategies', {})

            # Key valuation metrics
            key_metrics = []
            for strategy_name, score in value_strategies.items():
                if score is not None and score >= 70:
                    # Extract metric name
                    metric = strategy_name.split('_', 1)[1] if '_' in strategy_name else strategy_name
                    key_metrics.append((metric, score))

            key_metrics.sort(key=lambda x: x[1], reverse=True)

            # Valuation level
            if value_score >= 70:
                level = "저평가"
                desc = "매력적 가치 투자 기회"
            elif value_score >= 50:
                level = "적정 가치"
                desc = "합리적 가격대"
            elif value_score >= 30:
                level = "다소 고평가"
                desc = "추가 상승 제한적"
            else:
                level = "고평가"
                desc = "밸류에이션 부담 존재"

            interpretation = f"{level} 구간 ({value_score:.1f}점), {desc}"

            return {
                'value_score': value_score,
                'level': level,
                'description': desc,
                'key_metrics': key_metrics[:3],
                'interpretation': interpretation
            }

        except Exception as e:
            logger.error(f"Valuation analysis failed: {e}")
            return {}

    def generate_signal_lights(self) -> Dict:
        """
        6. Generate signal light system
        Provide intuitive Why through color-coded signals

        Signals are generated with priority scores and sorted by importance.
        Only top 3 signals per category are saved to database.

        Returns:
            Dict with buy/neutral/sell signals (sorted by priority)
        """
        try:
            # 신호를 (priority, label, value) 튜플로 저장 (priority 낮을수록 중요)
            buy_signals = []
            neutral_signals = []
            sell_signals = []

            # 1. Final score signal (Priority: 1 - Most important)
            if self.final_score is not None:
                if self.final_score >= 70:
                    buy_signals.append((1, "종합 점수 우수", f"{self.final_score:.1f}점"))
                elif self.final_score >= 50:
                    neutral_signals.append((1, "종합 점수 보통", f"{self.final_score:.1f}점"))
                else:
                    sell_signals.append((1, "종합 점수 부진", f"{self.final_score:.1f}점"))
            else:
                neutral_signals.append((1, "데이터 부족", "평가 불가"))

            # 2. Investor flow signal (Priority: 2 - Very important trading signal)
            m = self.additional_metrics
            inst_net = m.get('inst_net_30d', 0)
            foreign_net = m.get('foreign_net_30d', 0)

            if inst_net > 0 and foreign_net > 0:
                buy_signals.append((2, "기관/외국인 동반 매수", f"합계 {(inst_net+foreign_net)/10000:+,.0f}만주"))
            elif inst_net < 0 and foreign_net < 0:
                sell_signals.append((2, "기관/외국인 동반 매도", f"합계 {(inst_net+foreign_net)/10000:,.0f}만주"))

            # 3. Valuation signal (Priority: 3 - Important for value investing)
            value_result = self.factor_scores.get('value', {}).get('weighted_result') or {}
            value_score = value_result.get('weighted_score', 0)
            if value_score >= 60:
                buy_signals.append((3, "저평가 구간", f"가치 {value_score:.1f}점"))
            elif value_score < 30:
                neutral_signals.append((3, "고평가 우려", f"가치 {value_score:.1f}점"))

            # 4. Factor signals (Priority: 4-7 by importance)
            factor_priorities = {
                'quality': 4,   # Quality most important
                'value': 5,     # Value (if not already covered above)
                'growth': 6,    # Growth
                'momentum': 7   # Momentum least important
            }

            for factor_name, factor_ko in [('quality', '품질'), ('value', '가치'),
                                            ('growth', '성장'), ('momentum', '모멘텀')]:
                factor_data = self.factor_scores.get(factor_name, {})
                weighted_result = factor_data.get('weighted_result')
                if weighted_result:
                    score = weighted_result['weighted_score']
                    priority = factor_priorities[factor_name]
                    if score >= 70:
                        buy_signals.append((priority, f"{factor_ko} 팩터 우수", f"{score:.1f}점"))
                    elif score < 40:
                        sell_signals.append((priority, f"{factor_ko} 팩터 부진", f"{score:.1f}점"))

            # 5. Industry signal (Priority: 8)
            percentile = m.get('industry_percentile')
            if percentile and percentile >= 75:
                buy_signals.append((8, "업종 상위권", f"상위 {100-percentile:.0f}%"))

            # 6. Risk signals (Priority: 9-10 - Neutral warnings)
            volatility = m.get('volatility_annual')
            if volatility and volatility > 40:
                neutral_signals.append((9, "높은 변동성", f"{volatility:.1f}%"))

            mdd = m.get('max_drawdown_1y')
            if mdd and mdd < -25:
                neutral_signals.append((10, "큰 낙폭 경험", f"{mdd:.1f}%"))

            # Sort by priority (lower number = higher priority) and remove priority from tuple
            buy_signals = [(label, value) for _, label, value in sorted(buy_signals, key=lambda x: x[0])]
            neutral_signals = [(label, value) for _, label, value in sorted(neutral_signals, key=lambda x: x[0])]
            sell_signals = [(label, value) for _, label, value in sorted(sell_signals, key=lambda x: x[0])]

            # Overall assessment
            total_signals = len(buy_signals) + len(neutral_signals) + len(sell_signals)
            if len(buy_signals) >= len(sell_signals) + 2:
                overall = "긍정적 신호 우세 - 매수 검토 구간"
            elif len(sell_signals) >= len(buy_signals) + 2:
                overall = "부정적 신호 우세 - 매도 검토 구간"
            else:
                overall = "신호 혼재 - 신중한 접근 필요"

            return {
                'buy_signals': buy_signals,
                'neutral_signals': neutral_signals,
                'sell_signals': sell_signals,
                'buy_count': len(buy_signals),
                'neutral_count': len(neutral_signals),
                'sell_count': len(sell_signals),
                'overall_assessment': overall
            }

        except Exception as e:
            logger.error(f"Signal light generation failed: {e}")
            return {}

    def generate_all_interpretations(self) -> Dict:
        """
        Generate all interpretations

        Returns:
            Dict containing all interpretation results
        """
        logger.info(f"Generating interpretations for {self.symbol}...")

        interpretations = {
            'factor_details': self.analyze_factor_details(),
            'industry_positioning': self.analyze_industry_positioning(),
            'time_series_trend': self.analyze_time_series_trend(),
            'risk_profile': self.analyze_risk_profile(),
            'valuation': self.analyze_valuation(),
            'signal_lights': self.generate_signal_lights()
        }

        logger.info("All interpretations generated successfully")
        return interpretations


if __name__ == '__main__':
    """Test interpretation engine"""
    # Mock data for testing
    mock_factor_scores = {
        'value': {
            'weighted_result': {'weighted_score': 45.0},
            'strategies': {'V1_Low_PER': 60, 'V2_Low_PBR_High_ROE': 80, 'V3_Net_Cash_Flow_Yield': 40}
        },
        'quality': {
            'weighted_result': {'weighted_score': 75.0},
            'strategies': {'Q1_High_ROE': 85, 'Q2_Operating_Margin_Excellence': 90}
        },
        'momentum': {
            'weighted_result': {'weighted_score': 60.0},
            'strategies': {}
        },
        'growth': {
            'weighted_result': {'weighted_score': 55.0},
            'strategies': {}
        }
    }

    mock_additional_metrics = {
        'confidence_score': 95.0,
        'volatility_annual': 32.0,
        'var_95': -2.5,
        'inst_net_30d': 50000000000,
        'foreign_net_30d': -20000000000,
        'value_momentum': -10.0,
        'quality_momentum': 15.0,
        'momentum_momentum': 5.0,
        'growth_momentum': 3.0,
        'industry_rank': 5,
        'industry_percentile': 85.0,
        'max_drawdown_1y': -18.5,
        'beta': 1.2
    }

    mock_weighted_factor_scores = {
        'value': {'weighted_contribution': 15.0},
        'quality': {'weighted_contribution': 35.0},
        'momentum': {'weighted_contribution': 10.0},
        'growth': {'weighted_contribution': 8.0}
    }

    engine = InterpretationEngine(
        symbol='005930',
        factor_scores=mock_factor_scores,
        additional_metrics=mock_additional_metrics,
        weighted_factor_scores=mock_weighted_factor_scores,
        final_score=68.0,
        final_grade='매수 고려'
    )

    interpretations = engine.generate_all_interpretations()

    print("\n" + "="*80)
    print("INTERPRETATION TEST RESULTS")
    print("="*80)

    for key, value in interpretations.items():
        print(f"\n{key}:")
        print(f"  {value.get('interpretation', value)}")
