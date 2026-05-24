# US Quant System - Comprehensive Diagnosis Report

**Date**: 2025-11-25
**Analysis Period**: 2025-01-09 ~ 2025-08-28 (14 dates)
**Sample Size**: 59,452 stock-date observations

---

## Executive Summary

US 퀀트 시스템 IC 분석 결과, **점수-수익률 관계가 완전히 역전**되어 있습니다:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| 전체 IC | -0.053 | 음의 상관관계 (역전) |
| Decile 1 (최저점수) | +125.9% | 가장 높은 수익률 |
| Decile 10 (최고점수) | +2.0% | 가장 낮은 수익률 |
| Long-Short Spread | -123.92% | 완전 역전 |
| 매도 등급 수익률 | +166.3% | 최고 수익 |
| 매수 등급 수익률 | +0.8% | 최저 수익 |

이는 **현재 시스템이 수익률 높은 종목을 "매도"로, 수익률 낮은 종목을 "매수"로 추천**하고 있음을 의미합니다.

**결론: 가중치 조정이나 부분 수정으로는 해결 불가능하며, 시스템 전면 재설계가 필요합니다.**

---

## 1. 현재 시스템 구조 분석

### 1.1 전체 아키텍처

```
Final Score = Base Score × 70% + Interaction Score × 30%

Base Score = Σ(Factor Score × Dynamic Weight)
           = Value × W_v + Quality × W_q + Momentum × W_m + Growth × W_g

where Σ(W) = 1.0 (정규화)
```

### 1.2 등급 기준 (Thresholds)

| Score Range | Grade |
|-------------|-------|
| 85+ | 강력 매수 |
| 65-84 | 매수 |
| 55-64 | 매수 고려 |
| 45-54 | 중립 |
| 35-44 | 매도 고려 |
| 20-34 | 매도 |
| <20 | 강력 매도 |

### 1.3 4 Factor 시스템

| Factor | Strategies | Description |
|--------|------------|-------------|
| **Value** | V1-V15 (15개) | PER, PBR, PSR, PCR, EV/EBITDA, EV/Sales, PEG, PSG, 배당수익률 등 |
| **Quality** | Q1-Q17 (17개) | ROE, ROA, Operating Margin, FCF Yield, Debt Ratio, Piotroski F-Score 등 |
| **Momentum** | M1-M20 (20개) | 가격 모멘텀, RSI, MACD, 이동평균, 거래량 모멘텀, 실적 서프라이즈 등 |
| **Growth** | G1-G18 (18개) | 매출 성장률, EPS 성장률, 이익 가속도, R&D 투자율, TAM 성장성 등 |

### 1.4 11 Conditions 기반 동적 가중치

| Condition | Weight Adjustment Logic |
|-----------|------------------------|
| **Exchange** | NYSE: value↑1.4, quality↑1.3 / NASDAQ: growth↑1.5, momentum↑1.3 |
| **Market Cap** | Mega/Large: quality↑ / Small/Micro: growth↑, momentum↑ |
| **Liquidity** | High: momentum↑ / Low: value↑, growth↑ |
| **Economic Cycle** | Tightening: value↑ / Easing: growth↑ |
| **Sector** | Tech: value↓0.6, growth↑1.5 / Finance: value↑1.4 |
| **Volatility** | High Vol: quality↑ / Low Vol: momentum↑ |
| **Growth Profile** | Hyper Growth: growth↑1.8, value↓0.5 |
| **Profitability** | High Profit: quality↑ / Loss: special handling |
| **Options Positioning** | Put/Call ratio 기반 조정 |
| **Analyst Momentum** | 목표가 변화 기반 |
| **Market Sentiment** | VIX, Fear & Greed 기반 |

---

## 2. 2024년 이후 미국 시장 환경 분석

### 2.1 시장 특성

**2024-2025년 미국 주식시장의 핵심 특징:**

1. **AI 붐과 기술주 랠리**
   - NVDA: +240% (2024), 계속 상승
   - MSFT, META, AMZN: AI 수혜로 고평가 지속
   - "고평가" 기술주가 계속 오르는 환경

2. **Magnificent 7 집중**
   - 7개 종목이 S&P 500 수익률의 대부분 차지
   - 시가총액 상위 집중 현상 심화

3. **Value vs Growth 격차**
   - Russell 1000 Growth >> Russell 1000 Value
   - 전통적 가치 지표가 작동하지 않는 환경

4. **금리 고점 환경**
   - Fed Funds Rate 5.25-5.50% (2024)
   - 고금리에도 기술주 강세 지속 (기존 이론과 반대)

5. **Small Cap 저조**
   - Russell 2000 상대적 부진
   - 대형 성장주로 자금 집중

### 2.2 전통적 퀀트 팩터의 실패

| 전통 팩터 | 2024+ 환경에서의 실패 원인 |
|-----------|----------------------|
| **Value (저PER)** | AI 기술주는 PER 30-100+도 정당화되는 환경 |
| **Value (저PBR)** | 무형자산(IP, 플랫폼) 가치 반영 불가 |
| **Quality (고ROE)** | 성장 투자 중인 기업은 일시적 ROE 저하 |
| **Momentum** | 급등 후 조정보다 추가 급등 빈발 |

---

## 3. 문제점 상세 진단

### 3.1 CRITICAL: 점수-수익률 역전의 근본 원인

**핵심 문제: Value Factor의 설계 철학**

```
현재 로직:
- 낮은 PER = 높은 Value Score = 높은 Final Score = "매수"
- 높은 PER = 낮은 Value Score = 낮은 Final Score = "매도"

2024+ 현실:
- 낮은 PER = 성장 잠재력 낮음 = 저조한 수익률
- 높은 PER = AI/성장 기대 = 높은 수익률

결과: Value Factor가 정반대로 작동
```

**IC 분석 결과 (Sector별):**

| Sector | IC | Interpretation |
|--------|------|----------------|
| Consumer Cyclical | +0.052 | 정상 작동 |
| Real Estate | +0.035 | 정상 작동 |
| Communication Services | +0.006 | 중립 |
| Energy | +0.004 | 중립 |
| **Technology** | **-0.023** | **역전** |
| **Financial Services** | **-0.098** | **심각한 역전** |
| **Basic Materials** | **-0.154** | **심각한 역전** |

### 3.2 가중치 조정의 한계

```
가중치 조정만으로는 근본적 해결 불가

예시:
- Value 가중치 0.6으로 낮춤
- 하지만 "낮은 PER = 높은 점수"라는 로직 자체는 유지
- 결과: 여전히 역효과 발생
```

### 3.3 설계 철학의 근본적 오류

| 현재 시스템 가정 | 2024+ 시장 현실 |
|-----------------|---------------|
| 저평가 = 좋은 투자 | 저평가 = 성장 기대 없음 |
| 고평가 = 위험 | 고평가 = 높은 성장 기대 |
| 고PER = 버블 | 고PER = AI 성장 프리미엄 |
| 저PBR = 저평가 | 저PBR = 무형자산 부족 |
| 고배당 = 안정성 | 고배당 = 성장 포기 신호 |

---

## 4. 시스템 재설계 방향

### 4.1 핵심 원칙

**기존 시스템 폐기, 새로운 패러다임으로 전면 재설계**

1. **절대적 밸류에이션 → 상대적 밸류에이션**
2. **정적 팩터 → 시장 레짐 기반 동적 팩터**
3. **섹터 혼합 평가 → 섹터 Neutral 전략**
4. **규칙 기반 → ML 기반 팩터 선택**

### 4.2 새로운 아키텍처 설계

```
[NEW] Final Score = Regime-Adjusted Factor Score

1. Market Regime Detection
   - AI Bull / Tightening / Recovery / Crisis 등 레짐 감지

2. Dynamic Factor Selection
   - 레짐별 효과적인 팩터 자동 선택/비활성화

3. Sector-Neutral Scoring
   - 섹터 내 상대 순위 사용 (섹터 간 비교 제거)

4. Forward-Looking Metrics
   - 과거 지표 → 미래 예측 지표 중심
```

### 4.3 Factor 재설계

#### A. Value Factor 2.0 (완전 재설계)

**기존 Value 팩터 폐기**, 새로운 "Reasonable Valuation" 팩터로 대체:

```python
# [NEW] Reasonable Valuation Factor
reasonable_valuation_score = (
    # 1. 성장 대비 밸류에이션 (핵심)
    peg_ratio_score * 0.30 +           # PEG (성장 대비 PE)

    # 2. Forward 밸류에이션 (미래 기반)
    forward_pe_score * 0.25 +          # Forward PE (애널리스트 추정)

    # 3. 업종 대비 상대 밸류에이션
    sector_relative_pe_score * 0.20 +  # 섹터 평균 대비 PE

    # 4. Rule of 40 (SaaS/Tech 기업)
    rule_of_40_score * 0.15 +          # 성장률 + 영업이익률 >= 40

    # 5. EV/Revenue to Growth
    ev_revenue_growth_score * 0.10     # EV/Sales ÷ 매출성장률
)
```

#### B. Quality Factor 2.0 (현대화)

```python
# [NEW] Modern Quality Factor
modern_quality_score = (
    # 1. 수익성 (매출총이익률 중심)
    gross_margin_score * 0.20 +        # 매출총이익률 (핵심)

    # 2. 비즈니스 모델 품질
    recurring_revenue_score * 0.20 +   # 반복 매출 비율 (구독모델)
    customer_retention_score * 0.15 +  # 고객 유지율/NRR

    # 3. 자본 효율성
    roic_score * 0.15 +                # ROIC (투자자본수익률)

    # 4. 경영진 실행력
    guidance_beat_score * 0.15 +       # 가이던스 달성률

    # 5. 재무 안정성
    balance_sheet_score * 0.15         # 순현금, 이자보상배율
)
```

#### C. Momentum Factor 2.0 (강화)

```python
# [NEW] Enhanced Momentum Factor
enhanced_momentum_score = (
    # 1. 가격 모멘텀 (기존 유지, 강화)
    price_momentum_12m_score * 0.25 +  # 12개월 모멘텀
    price_momentum_6m_score * 0.15 +   # 6개월 모멘텀

    # 2. 상대 강도
    relative_strength_score * 0.20 +   # 시장 대비 상대 강도

    # 3. 실적 모멘텀 (중요!)
    earnings_revision_score * 0.20 +   # 애널리스트 추정치 상향
    earnings_surprise_score * 0.10 +   # 실적 서프라이즈

    # 4. 기술적 신호
    trend_strength_score * 0.10        # 추세 강도 (ADX 등)
)
```

#### D. Growth Factor 2.0 (확장)

```python
# [NEW] Forward-Looking Growth Factor
forward_growth_score = (
    # 1. 매출 성장 (핵심)
    revenue_growth_score * 0.25 +      # 매출 성장률
    revenue_acceleration_score * 0.15 + # 매출 성장 가속도

    # 2. 이익 성장
    earnings_growth_score * 0.20 +     # EPS 성장률

    # 3. 시장 기회
    tam_expansion_score * 0.15 +       # TAM 확대
    market_share_score * 0.10 +        # 시장점유율 변화

    # 4. 혁신 투자
    rd_intensity_score * 0.10 +        # R&D 투자율

    # 5. AI/테마 노출
    thematic_exposure_score * 0.05     # AI, 클라우드 등 테마 관련성
)
```

### 4.4 시장 레짐 기반 전략 전환

```python
# Market Regime Detection & Strategy Selection
MARKET_REGIME_STRATEGIES = {
    'AI_BULL': {
        'description': 'AI 주도 성장주 랠리',
        'factor_weights': {
            'momentum': 0.35,
            'growth': 0.35,
            'quality': 0.25,
            'value': 0.05  # 거의 비활성화
        },
        'sector_overweight': ['TECHNOLOGY', 'COMMUNICATION SERVICES'],
        'sector_underweight': ['UTILITIES', 'CONSUMER DEFENSIVE']
    },

    'TIGHTENING': {
        'description': '금리 인상기, 긴축 환경',
        'factor_weights': {
            'quality': 0.40,
            'value': 0.30,
            'momentum': 0.20,
            'growth': 0.10
        },
        'sector_overweight': ['HEALTHCARE', 'CONSUMER DEFENSIVE'],
        'sector_underweight': ['TECHNOLOGY', 'REAL ESTATE']
    },

    'RECOVERY': {
        'description': '경기 회복기',
        'factor_weights': {
            'value': 0.30,
            'momentum': 0.30,
            'growth': 0.25,
            'quality': 0.15
        },
        'sector_overweight': ['FINANCIALS', 'INDUSTRIALS'],
        'sector_underweight': ['UTILITIES']
    },

    'CRISIS': {
        'description': '위기/급락장',
        'factor_weights': {
            'quality': 0.50,
            'value': 0.30,
            'momentum': 0.10,
            'growth': 0.10
        },
        'sector_overweight': ['CONSUMER DEFENSIVE', 'HEALTHCARE'],
        'sector_underweight': ['CONSUMER CYCLICAL', 'TECHNOLOGY']
    }
}
```

### 4.5 섹터 Neutral 전략

```python
# Sector-Neutral Scoring
def calculate_sector_neutral_score(symbol, sector):
    """
    섹터 내에서만 상대 순위 계산
    섹터 간 비교 제거 -> 섹터 로테이션은 별도 레이어에서 처리
    """
    # 1. 같은 섹터 종목들의 팩터 점수 조회
    sector_stocks = get_stocks_by_sector(sector)

    # 2. 각 팩터별 섹터 내 percentile 계산
    value_percentile = calculate_percentile(symbol, sector_stocks, 'value')
    quality_percentile = calculate_percentile(symbol, sector_stocks, 'quality')
    momentum_percentile = calculate_percentile(symbol, sector_stocks, 'momentum')
    growth_percentile = calculate_percentile(symbol, sector_stocks, 'growth')

    # 3. 섹터 내 상대 점수 (0-100)
    sector_neutral_score = (
        value_percentile * regime_weights['value'] +
        quality_percentile * regime_weights['quality'] +
        momentum_percentile * regime_weights['momentum'] +
        growth_percentile * regime_weights['growth']
    )

    return sector_neutral_score
```

### 4.6 ML 기반 팩터 성과 예측

```python
# ML-based Factor Performance Prediction
class FactorPerformancePredictor:
    """
    시장 환경 변수를 입력받아 향후 N일간 각 팩터의 성과 예측
    """
    def __init__(self):
        self.model = load_trained_model()  # XGBoost or LSTM

    def predict_factor_returns(self, market_features):
        """
        Input: VIX, 금리, 섹터 모멘텀, 매크로 지표 등
        Output: 각 팩터의 예상 성과 (상대 수익률)
        """
        features = {
            'vix': market_features['vix'],
            'fed_rate': market_features['fed_rate'],
            'yield_curve': market_features['yield_curve'],
            'sector_dispersion': market_features['sector_dispersion'],
            'market_breadth': market_features['market_breadth'],
            ...
        }

        predicted_returns = self.model.predict(features)

        # 예측 성과에 따른 동적 가중치
        dynamic_weights = self.returns_to_weights(predicted_returns)

        return dynamic_weights
```

---

## 5. 실행 계획

### Phase 1: 기존 시스템 분석 완료 및 설계 (1주)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | 기존 팩터 파일 구조 분석 완료 | 구조 문서 |
| 1.2 | 새로운 팩터 정의서 작성 | Factor 2.0 Spec |
| 1.3 | DB 스키마 설계 (신규 테이블) | DDL 스크립트 |
| 1.4 | 시장 레짐 감지 로직 설계 | Regime Detection Spec |

### Phase 2: 핵심 모듈 개발 (2-3주)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | us_value_factor_v2.py 개발 | 신규 Value 팩터 |
| 2.2 | us_quality_factor_v2.py 개발 | 신규 Quality 팩터 |
| 2.3 | us_momentum_factor_v2.py 개발 | 신규 Momentum 팩터 |
| 2.4 | us_growth_factor_v2.py 개발 | 신규 Growth 팩터 |
| 2.5 | us_market_regime.py 개발 | 시장 레짐 감지 |
| 2.6 | us_sector_neutral.py 개발 | 섹터 Neutral 로직 |

### Phase 3: 통합 및 백테스트 (1-2주)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 | us_main_v2.py 통합 | 신규 메인 시스템 |
| 3.2 | 백테스트 프레임워크 구축 | 백테스트 모듈 |
| 3.3 | 2024-2025 데이터 백테스트 | IC 분석 결과 |
| 3.4 | 파라미터 최적화 | 최적 파라미터 |

### Phase 4: 검증 및 배포 (1주)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1 | Out-of-sample 테스트 | 검증 결과 |
| 4.2 | 기존 시스템 대비 성과 비교 | 비교 보고서 |
| 4.3 | Production 배포 | 운영 시스템 |

---

## 6. 성공 기준

### 6.1 정량적 목표

| Metric | 현재 | 목표 |
|--------|------|------|
| 전체 IC | -0.053 | **+0.05 이상** |
| 매수 등급 승률 | 47.4% | **55% 이상** |
| Long-Short Spread | -123.92% | **+10% 이상** |
| Decile 10 vs Decile 1 | -123.92% | **+20% 이상** |

### 6.2 정성적 목표

1. **시장 레짐 적응**: AI Bull 환경에서 성장주 적절히 추천
2. **Value Trap 회피**: 저평가 함정 종목 필터링
3. **섹터 균형**: 특정 섹터 편향 없는 종목 선별
4. **실시간 적응**: 시장 환경 변화에 동적 대응

---

## 7. 결론

현재 US 퀀트 시스템은 **"저평가 = 좋은 투자"라는 전통적 Value 투자 철학**에 기반하여 설계되었으나, 2024년 이후 AI 붐, 성장주 랠리 환경에서 **완전히 역으로 작동**하고 있습니다.

**가중치 조정, 파라미터 튜닝, 이상치 제거 등의 부분 수정으로는 해결 불가능**합니다.

### 재설계 핵심 방향:

1. **Value 팩터 완전 재정의**: 절대적 저평가 → 성장 대비 합리적 밸류에이션
2. **시장 레짐 기반 동적 전략**: AI Bull에서는 Growth/Momentum 중심
3. **섹터 Neutral 접근**: 섹터 내 상대 순위로 평가
4. **Forward-Looking 지표**: 과거 실적 → 미래 예측 중심
5. **ML 기반 팩터 선택**: 시장 환경에 따른 자동 최적화

**즉시 시스템 전면 재설계를 시작합니다.**

---

## Appendix: 분석에 사용된 데이터

### A. IC 분석 결과 파일

| File | Description |
|------|-------------|
| us_ic_analysis.csv | 전체 IC 분석 |
| us_decile_analysis.csv | 10분위 분석 |
| us_sector_analysis.csv | 섹터별 IC |
| us_exchange_analysis.csv | 거래소별 IC |
| us_accuracy_winrate.csv | 등급별 정확도 |
| us_parameter_surface.csv | 파라미터 최적화 |

### B. 기존 코드 파일 (폐기 예정)

| File | Status |
|------|--------|
| us_main.py | → us_main_v2.py로 대체 |
| us_weight.py | → us_market_regime.py로 대체 |
| us_value_factor.py | → us_value_factor_v2.py로 대체 |
| us_quality_factor.py | → us_quality_factor_v2.py로 대체 |
| us_momentum_factor.py | → us_momentum_factor_v2.py로 대체 |
| us_growth_factor.py | → us_growth_factor_v2.py로 대체 |

### C. 신규 개발 파일

| File | Description |
|------|-------------|
| us_main_v2.py | 신규 메인 시스템 |
| us_market_regime.py | 시장 레짐 감지 |
| us_sector_neutral.py | 섹터 Neutral 로직 |
| us_value_factor_v2.py | Reasonable Valuation Factor |
| us_quality_factor_v2.py | Modern Quality Factor |
| us_momentum_factor_v2.py | Enhanced Momentum Factor |
| us_growth_factor_v2.py | Forward-Looking Growth Factor |
| us_factor_ml.py | ML 기반 팩터 예측 |

---

*Report generated by Claude Code - US Quant System Analysis*
*Decision: Immediate Full System Redesign*
