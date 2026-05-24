# US Quant System Phase 3 작업계획

**작성일**: 2025-11-27
**목표**: 장기(3개월~1년) 예측 정확도 향상 및 비선형 관계 반영

---

## 1. Phase 3 작업 개요

### 1.1 배경 (Phase 2 분석 결과)

| 문제점 | 현황 | 목표 |
|--------|------|------|
| Pearson-Spearman IC 괴리 | -0.097 vs +0.174 | 비선형 모델로 해결 |
| Momentum 단기 역작동 | 3일 IC -0.047 | 장기 전략 중심 재설계 |
| 매도 신호 분석 오류 | win_rate 정의 오류 | 방향별 정확도 분리 |
| A+ 승률 | 84.7% (252일) | 유지 및 샘플 수 확대 |

### 1.2 Phase 3 작업 목록

| 순번 | 작업 | 파일 | 우선순위 |
|------|------|------|----------|
| 3-1 | Factor Interaction Model 도입 | `us_factor_interactions.py` (신규) | 높음 |
| 3-2 | Conviction Score 추가 | `us_main_v2.py` 수정 | 높음 |
| 3-3 | Momentum Factor 장기화 재설계 | `us_momentum_factor_v2.py` 수정 | 높음 |
| 3-4 | IC 분석 승률 중심 전환 | `us_ic_analysis.py` 수정 | 중간 |
| 3-5 | Phase 3 검증 분석 | `us_phase3_validation.py` (신규) | 중간 |

---

## 2. Phase 3-1: Factor Interaction Model 도입

### 2.1 도입 근거

```
252일 분석 결과:
- Pearson IC: -0.097 (선형 관계 음수)
- Spearman IC: +0.174 (순위 관계 양수)
- 괴리: 0.271

해석: 점수와 수익률 간 비선형 관계 존재
→ 단순 가중합 방식의 한계
→ 팩터 간 상호작용(Interaction) 효과 반영 필요
```

### 2.2 구현 파일

**파일명**: `us_factor_interactions.py`
**위치**: `C:\project\alpha\quant\us\`

### 2.3 Interaction Terms 설계

| Interaction | 팩터 조합 | 의미 | 가중치 |
|-------------|----------|------|--------|
| **I1** | Growth × Quality | 성장 + 수익성 시너지 | 0.30 |
| **I2** | Growth × Momentum | 성장 + 추세 확인 | 0.25 |
| **I3** | Quality × Value | 고품질 저평가 | 0.20 |
| **I4** | Momentum × Quality | 추세 + 펀더멘탈 뒷받침 | 0.15 |
| **I5** | All-Factor Agreement | 4팩터 일치도 | 0.10 |

### 2.4 구현 코드 설계

```python
"""
US Factor Interactions - Non-linear Factor Combination
======================================================

Phase 2 분석에서 Pearson IC(-0.097)와 Spearman IC(+0.174)의 괴리 발견
→ 비선형 관계 존재 확인
→ Factor Interaction Model로 해결

File: us/us_factor_interactions.py
"""

import numpy as np
from typing import Dict, Optional
from datetime import date

# Interaction Weights
INTERACTION_WEIGHTS = {
    'I1_growth_quality': 0.30,
    'I2_growth_momentum': 0.25,
    'I3_quality_value': 0.20,
    'I4_momentum_quality': 0.15,
    'I5_all_agreement': 0.10
}

class USFactorInteractions:
    """Factor Interaction Calculator"""

    def __init__(self, factor_scores: Dict[str, float]):
        """
        Args:
            factor_scores: {
                'value_score': float,
                'quality_score': float,
                'momentum_score': float,
                'growth_score': float
            }
        """
        self.scores = factor_scores
        self.value = factor_scores.get('value_score', 50)
        self.quality = factor_scores.get('quality_score', 50)
        self.momentum = factor_scores.get('momentum_score', 50)
        self.growth = factor_scores.get('growth_score', 50)

    def calculate(self) -> Dict:
        """
        Interaction 점수 계산

        Returns:
            {
                'interaction_score': float (0-100),
                'interactions': {
                    'I1': {'score': float, 'raw': float},
                    ...
                },
                'conviction_score': float (0-100)
            }
        """
        interactions = {}

        # I1: Growth × Quality
        interactions['I1'] = self._calc_i1_growth_quality()

        # I2: Growth × Momentum
        interactions['I2'] = self._calc_i2_growth_momentum()

        # I3: Quality × Value
        interactions['I3'] = self._calc_i3_quality_value()

        # I4: Momentum × Quality
        interactions['I4'] = self._calc_i4_momentum_quality()

        # I5: All-Factor Agreement (Conviction)
        interactions['I5'] = self._calc_i5_all_agreement()

        # 가중 합산
        interaction_score = self._weighted_sum(interactions)

        return {
            'interaction_score': round(interaction_score, 1),
            'interactions': interactions,
            'conviction_score': interactions['I5']['score']
        }

    def _calc_i1_growth_quality(self) -> Dict:
        """
        I1: Growth × Quality Interaction

        논리: 성장성이 높으면서 수익성도 좋은 기업
        - 둘 다 높음: 시너지 (보너스)
        - 하나만 높음: 중립
        - 둘 다 낮음: 페널티
        """
        # 정규화 (0-1 범위)
        g_norm = self.growth / 100
        q_norm = self.quality / 100

        # 기하평균 기반 (둘 다 높아야 점수 높음)
        geometric_mean = np.sqrt(g_norm * q_norm)

        # 시너지 보너스: 둘 다 70점 이상이면 추가 점수
        synergy_bonus = 0
        if self.growth >= 70 and self.quality >= 70:
            synergy_bonus = 0.1 * min((self.growth - 70) / 30, (self.quality - 70) / 30)

        raw_score = geometric_mean + synergy_bonus
        score = min(100, raw_score * 100)

        return {
            'score': round(score, 1),
            'raw': round(raw_score, 4),
            'growth': self.growth,
            'quality': self.quality
        }

    def _calc_i2_growth_momentum(self) -> Dict:
        """
        I2: Growth × Momentum Interaction

        논리: 성장 기업이면서 주가 모멘텀도 있는 경우
        - 펀더멘탈 성장 + 시장 인식 = 강한 신호
        """
        g_norm = self.growth / 100
        m_norm = self.momentum / 100

        # 조화평균 (균형 중시)
        if g_norm + m_norm > 0:
            harmonic_mean = 2 * g_norm * m_norm / (g_norm + m_norm)
        else:
            harmonic_mean = 0

        score = harmonic_mean * 100

        return {
            'score': round(score, 1),
            'raw': round(harmonic_mean, 4),
            'growth': self.growth,
            'momentum': self.momentum
        }

    def _calc_i3_quality_value(self) -> Dict:
        """
        I3: Quality × Value Interaction

        논리: 고품질이면서 저평가된 기업 (전통적 가치투자)
        """
        q_norm = self.quality / 100
        v_norm = self.value / 100

        # 산술평균 (둘 중 하나만 높아도 의미 있음)
        arithmetic_mean = (q_norm + v_norm) / 2

        # 둘 다 60점 이상이면 보너스
        bonus = 0
        if self.quality >= 60 and self.value >= 60:
            bonus = 0.05

        score = min(100, (arithmetic_mean + bonus) * 100)

        return {
            'score': round(score, 1),
            'raw': round(arithmetic_mean, 4),
            'quality': self.quality,
            'value': self.value
        }

    def _calc_i4_momentum_quality(self) -> Dict:
        """
        I4: Momentum × Quality Interaction

        논리: 주가 상승이 펀더멘탈로 뒷받침되는 경우
        - 모멘텀만 높고 퀄리티 낮음: 위험 (거품 가능성)
        - 모멘텀 + 퀄리티 둘 다 높음: 건강한 상승
        """
        m_norm = self.momentum / 100
        q_norm = self.quality / 100

        # 모멘텀이 높은데 퀄리티 낮으면 페널티
        if self.momentum >= 70 and self.quality < 50:
            penalty = 0.1
        else:
            penalty = 0

        geometric_mean = np.sqrt(m_norm * q_norm)
        score = max(0, (geometric_mean - penalty) * 100)

        return {
            'score': round(score, 1),
            'raw': round(geometric_mean, 4),
            'momentum': self.momentum,
            'quality': self.quality
        }

    def _calc_i5_all_agreement(self) -> Dict:
        """
        I5: All-Factor Agreement (Conviction Score)

        논리: 4개 팩터가 모두 일치하는 정도
        - 표준편차가 낮을수록 확신도 높음
        - 평균 점수도 고려
        """
        scores = [self.value, self.quality, self.momentum, self.growth]

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # 확신도: 표준편차가 낮을수록 높음 (최대 25점 기준)
        # std=0 → conviction=100, std=25 → conviction=0
        conviction_from_std = max(0, 100 - std_score * 4)

        # 평균 점수 반영 (평균이 높을수록 좋은 확신)
        # 평균 50 → 0 보너스, 평균 75 → 25 보너스
        avg_bonus = max(0, (mean_score - 50) / 2)

        # 최종 확신도 점수
        conviction = min(100, conviction_from_std * 0.7 + avg_bonus * 0.3 + mean_score * 0.3)

        return {
            'score': round(conviction, 1),
            'raw': round(std_score, 4),
            'mean': round(mean_score, 1),
            'std': round(std_score, 1),
            'factor_scores': scores
        }

    def _weighted_sum(self, interactions: Dict) -> float:
        """가중 합산"""
        total = 0
        weights = {
            'I1': INTERACTION_WEIGHTS['I1_growth_quality'],
            'I2': INTERACTION_WEIGHTS['I2_growth_momentum'],
            'I3': INTERACTION_WEIGHTS['I3_quality_value'],
            'I4': INTERACTION_WEIGHTS['I4_momentum_quality'],
            'I5': INTERACTION_WEIGHTS['I5_all_agreement']
        }

        for key, weight in weights.items():
            if key in interactions and interactions[key]['score'] is not None:
                total += interactions[key]['score'] * weight

        return total
```

### 2.5 us_main_v2.py 통합 방법

```python
# us_main_v2.py 수정 위치: _analyze_stock() 메서드

from us_factor_interactions import USFactorInteractions

async def _analyze_stock(self, symbol: str, analysis_date: date) -> Optional[Dict]:
    # ... 기존 팩터 계산 ...

    # Factor Interaction 계산 (신규 추가)
    factor_scores = {
        'value_score': value_result['value_score'],
        'quality_score': quality_result['quality_score'],
        'momentum_score': momentum_result['momentum_score'],
        'growth_score': growth_result['growth_score']
    }

    interaction_calc = USFactorInteractions(factor_scores)
    interaction_result = interaction_calc.calculate()

    # Final Score 계산 수정
    # 기존: final_score = weighted_sum(4 factors)
    # 신규: final_score = base_score * 0.7 + interaction_score * 0.3

    base_score = (
        value_result['value_score'] * weights['value'] +
        quality_result['quality_score'] * weights['quality'] +
        momentum_result['momentum_score'] * weights['momentum'] +
        growth_result['growth_score'] * weights['growth']
    )

    final_score = base_score * 0.70 + interaction_result['interaction_score'] * 0.30

    return {
        # ... 기존 필드 ...
        'interaction_score': interaction_result['interaction_score'],
        'conviction_score': interaction_result['conviction_score'],
        'final_score': final_score
    }
```

---

## 3. Phase 3-2: Conviction Score 추가

### 3.1 정의

```
Conviction Score (확신도 점수)
= 4개 팩터 점수의 일치 정도

높은 확신도: 4개 팩터 모두 비슷한 방향 (모두 높거나 모두 낮음)
낮은 확신도: 팩터 간 의견 불일치 (일부 높고 일부 낮음)
```

### 3.2 계산 공식

```python
def calculate_conviction_score(value, quality, momentum, growth):
    """
    Conviction Score 계산

    Args:
        value, quality, momentum, growth: 각 팩터 점수 (0-100)

    Returns:
        conviction_score: 0-100
    """
    scores = [value, quality, momentum, growth]

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # 기본 확신도: 표준편차 기반
    # std=0 → 100점, std=25 → 0점
    base_conviction = max(0, 100 - std_score * 4)

    # 방향 보정: 평균이 극단(높거나 낮음)일수록 보너스
    # 평균 50 → 0, 평균 80 → +15, 평균 20 → +15
    direction_bonus = abs(mean_score - 50) * 0.5

    conviction = min(100, base_conviction * 0.8 + direction_bonus)

    return round(conviction, 1)
```

### 3.3 활용 방법

```python
# 등급 결정 시 확신도 반영
def determine_grade(final_score, conviction_score):
    """
    확신도를 반영한 등급 결정

    기존: final_score만으로 등급 결정
    신규: conviction_score가 높으면 등급 상향 가능
    """
    base_grade = get_base_grade(final_score)

    # 확신도 80 이상 + final_score 75 이상 → A+ 가능
    if conviction_score >= 80 and final_score >= 75:
        if base_grade == 'A':
            return 'A+'

    # 확신도 30 미만 → 등급 하향
    if conviction_score < 30:
        return downgrade(base_grade)

    return base_grade
```

### 3.4 DB 스키마 변경

```sql
-- us_stock_grade 테이블 컬럼 추가
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS interaction_score DECIMAL(5,2);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS conviction_score DECIMAL(5,2);
```

---

## 4. Phase 3-3: Momentum Factor 장기화 재설계

### 4.1 현재 전략 문제점

| 전략 | Lookback | 문제점 |
|------|----------|--------|
| EM1 | 60일 | 단기 변동성에 민감 |
| EM2 | 60일 | 기간 짧음 |
| EM5 | 20일 | **단기 지표 - 제외 대상** |
| EM7 | 7일 | **단기 지표 - 제외 대상** |

### 4.2 재설계 전략

**변경 전 (현재)**:
```python
STRATEGY_WEIGHTS = {
    'EM1': 0.20,  # Risk-Adjusted Momentum (60일)
    'EM2': 0.20,  # Sector Relative Strength (60일)
    'EM3': 0.15,  # EPS Estimate Revision (30일)
    'EM4': 0.15,  # Revenue Estimate Revision (30일)
    'EM5': 0.10,  # Volume Confirmation (20일) ← 제외
    'EM6': 0.10,  # Earnings Momentum (분기)
    'EM7': 0.10   # Analyst Revisions (7일) ← 제외
}
```

**변경 후 (제안)**:
```python
STRATEGY_WEIGHTS_V3 = {
    'EM1': 0.15,  # Risk-Adjusted Momentum → 126일로 변경
    'EM2': 0.15,  # Sector Relative Strength → 126일로 변경
    'EM3': 0.20,  # EPS Estimate Revision (비중 증가)
    'EM4': 0.20,  # Revenue Estimate Revision (비중 증가)
    'EM6': 0.15,  # Earnings Momentum (비중 증가)
    'EM8': 0.15,  # NEW: Long-term Price Momentum (252일)
}
# EM5 (Volume), EM7 (7일 Analyst) 제외
```

### 4.3 EM1 수정: Risk-Adjusted Momentum (126일)

```python
def _calc_em1_risk_adjusted_momentum_v3(self):
    """
    EM1 v3: 6개월 Risk-Adjusted Momentum

    변경점:
    - Lookback: 60일 → 126일
    - Sharpe 계산 기준 연환산
    """
    lookback = 126  # 6개월

    if len(self.price_data) < lookback:
        return {'score': 50, 'raw': None}

    # 일간 수익률 계산
    prices = [d['close'] for d in self.price_data[:lookback]]
    returns = [(prices[i] / prices[i+1] - 1) for i in range(len(prices)-1)]

    avg_return = np.mean(returns)
    volatility = np.std(returns)

    if volatility > 0:
        # 연환산 Sharpe
        sharpe = (avg_return / volatility) * np.sqrt(252)
    else:
        sharpe = 0

    # 정규화: Sharpe -1 ~ +3 → 0 ~ 100
    score = min(100, max(0, (sharpe + 1) / 4 * 100))

    return {
        'score': round(score, 1),
        'raw': round(sharpe, 4),
        'lookback': lookback
    }
```

### 4.4 EM2 수정: Sector Relative Strength (126일)

```python
def _calc_em2_sector_relative_strength_v3(self):
    """
    EM2 v3: 6개월 섹터 대비 상대 강도

    변경점:
    - Lookback: 60일 → 126일
    """
    lookback = 126

    # 종목 수익률 (6개월)
    stock_return = self._get_period_return(lookback)

    # 섹터 평균 수익률 (6개월)
    sector_return = self.sector_returns.get(f'{lookback}d_return', 0)

    if sector_return is None:
        return {'score': 50, 'raw': None}

    # 상대 강도
    relative_strength = stock_return - sector_return

    # 정규화: -30% ~ +30% → 0 ~ 100
    score = min(100, max(0, (relative_strength + 30) / 60 * 100))

    return {
        'score': round(score, 1),
        'raw': round(relative_strength, 2),
        'stock_return': round(stock_return, 2),
        'sector_return': round(sector_return, 2)
    }
```

### 4.5 EM8 신규: Long-term Price Momentum (252일)

```python
def _calc_em8_long_term_momentum(self):
    """
    EM8: Long-term Price Momentum (IBD RS Style)

    공식: RS = 3M수익률*0.4 + 6M수익률*0.2 + 9M수익률*0.2 + 12M수익률*0.2

    논리:
    - 최근 3개월에 가장 높은 가중치 (0.4)
    - 장기 추세도 반영 (6M, 9M, 12M)
    """
    required_days = 252

    if len(self.price_data) < required_days:
        return {'score': 50, 'raw': None}

    current_price = self.price_data[0]['close']

    # 기간별 수익률
    ret_3m = self._get_period_return(63)   # 3개월
    ret_6m = self._get_period_return(126)  # 6개월
    ret_9m = self._get_period_return(189)  # 9개월
    ret_12m = self._get_period_return(252) # 12개월

    if any(r is None for r in [ret_3m, ret_6m, ret_9m, ret_12m]):
        return {'score': 50, 'raw': None}

    # IBD RS Style 가중 평균
    rs_value = ret_3m * 0.4 + ret_6m * 0.2 + ret_9m * 0.2 + ret_12m * 0.2

    # 정규화: -50% ~ +100% → 0 ~ 100
    score = min(100, max(0, (rs_value + 50) / 150 * 100))

    return {
        'score': round(score, 1),
        'raw': round(rs_value, 2),
        'ret_3m': round(ret_3m, 2),
        'ret_6m': round(ret_6m, 2),
        'ret_9m': round(ret_9m, 2),
        'ret_12m': round(ret_12m, 2)
    }

def _get_period_return(self, days: int) -> Optional[float]:
    """기간 수익률 계산"""
    if len(self.price_data) < days:
        return None

    current = self.price_data[0]['close']
    past = self.price_data[days - 1]['close']

    if past and past > 0:
        return (current / past - 1) * 100
    return None
```

### 4.6 us_momentum_factor_v2.py 전체 수정 사항

```python
# 파일: us_momentum_factor_v2.py

# 변경 1: 가중치 수정
STRATEGY_WEIGHTS = {
    'EM1': 0.15,  # Risk-Adjusted (126일)
    'EM2': 0.15,  # Sector Relative (126일)
    'EM3': 0.20,  # EPS Revision
    'EM4': 0.20,  # Revenue Revision
    'EM6': 0.15,  # Earnings Momentum
    'EM8': 0.15,  # Long-term Momentum (신규)
}

# 변경 2: EM5, EM7 제외
# - EM5 (Volume Confirmation): 단기 지표, 장기 예측에 부적합
# - EM7 (Analyst Revisions 7일): 기간 너무 짧음

# 변경 3: EM1, EM2 Lookback 변경
# - 60일 → 126일

# 변경 4: EM8 신규 추가
# - Long-term Price Momentum (252일 기반)

# calculate() 메서드 수정
async def calculate(self) -> Dict:
    strategies = {}
    strategies['EM1'] = self._calc_em1_risk_adjusted_momentum_v3()  # 수정
    strategies['EM2'] = self._calc_em2_sector_relative_strength_v3()  # 수정
    strategies['EM3'] = self._calc_em3_eps_revision()
    strategies['EM4'] = self._calc_em4_revenue_revision()
    # EM5 제외
    strategies['EM6'] = self._calc_em6_earnings_momentum()
    # EM7 제외
    strategies['EM8'] = self._calc_em8_long_term_momentum()  # 신규

    # 가중 평균 계산
    ...
```

---

## 5. Phase 3-4: IC 분석 승률 중심 전환

### 5.1 문제점

현재 Decile Test의 "win_rate" 정의:
```python
# 현재 (문제)
win_rate = (return > 0).mean() * 100  # 모든 구간 동일

# 문제점:
# D1 (매도 신호): return < 0 이어야 정확
# D10 (매수 신호): return > 0 이어야 정확
```

### 5.2 수정 방안

```python
def calculate_directional_accuracy(decile, returns):
    """
    방향별 정확도 계산

    Args:
        decile: 1-10 (D1=매도, D10=매수)
        returns: 수익률 배열

    Returns:
        accuracy: 올바른 예측 비율
    """
    if decile <= 3:  # D1-D3: 매도 신호
        # 음수 수익률이 정확
        return (returns < 0).mean() * 100

    elif decile >= 8:  # D8-D10: 매수 신호
        # 양수 수익률이 정확
        return (returns > 0).mean() * 100

    else:  # D4-D7: 중립
        return 50.0
```

### 5.3 us_ic_analysis.py 수정 위치

```python
# decile_test() 함수 수정

def decile_test(df):
    """STEP 5: Decile Test (수정됨)"""

    for decile in sorted(valid_data['decile'].unique()):
        decile_data = valid_data[valid_data['decile'] == decile]

        # 기존 win_rate (참고용)
        win_rate = (decile_data[ret_col] > 0).mean() * 100

        # 신규: 방향별 정확도
        if decile <= 3:  # 매도 신호
            directional_accuracy = (decile_data[ret_col] < 0).mean() * 100
        elif decile >= 8:  # 매수 신호
            directional_accuracy = (decile_data[ret_col] > 0).mean() * 100
        else:  # 중립
            directional_accuracy = 50.0

        decile_results.append({
            'period': period,
            'decile': decile,
            'count': count,
            'avg_return': avg_return,
            'win_rate': win_rate,  # 기존 (양수 비율)
            'directional_accuracy': directional_accuracy,  # 신규 (방향 정확도)
        })
```

### 5.4 출력 파일 변경

```
기존 출력:
us_decile_test_{DATE}.csv
- period, decile, count, avg_return, win_rate

신규 출력:
us_decile_test_{DATE}.csv
- period, decile, count, avg_return, win_rate, directional_accuracy
                                              ↑ 신규 컬럼
```

---

## 6. Phase 3-5: 검증 분석

### 6.1 검증 스크립트

**파일명**: `us_phase3_validation.py`
**위치**: `C:\project\alpha\quant\us\analysis\`

### 6.2 검증 항목

| 항목 | 기준 | Phase 2 결과 | Phase 3 목표 |
|------|------|--------------|--------------|
| 252일 Final Score IC | Spearman | +0.174 | +0.20 이상 |
| 252일 A+ 승률 | Win Rate | 84.7% | 85% 유지 |
| 252일 D10 방향 정확도 | Accuracy | 71.9% | 75% 이상 |
| 252일 D1 방향 정확도 | Accuracy | 50.9% | 60% 이상 |
| Pearson-Spearman 괴리 | 차이 | 0.271 | 0.15 이하 |

### 6.3 검증 프로세스

```
1단계: 코드 수정 완료
   - us_factor_interactions.py 생성
   - us_main_v2.py 수정 (Interaction, Conviction 추가)
   - us_momentum_factor_v2.py 수정 (EM5, EM7 제외, EM8 추가)
   - us_ic_analysis.py 수정 (방향별 정확도)

2단계: 전체 재계산
   - python us/us_main_v2.py 실행
   - 전체 종목 점수 재계산

3단계: IC 분석 실행
   - python us/analysis/us_ic_analysis.py 실행
   - 신규 CSV 생성

4단계: 결과 비교
   - Phase 2 vs Phase 3 지표 비교
   - 목표 달성 여부 확인

5단계: 보고서 작성
   - us_phase3_분석결과.md 생성
```

---

## 7. 파일 수정 요약

| 파일 | 작업 | 상세 |
|------|------|------|
| `us_factor_interactions.py` | **신규 생성** | Factor Interaction + Conviction Score |
| `us_main_v2.py` | 수정 | Interaction 통합, final_score 계산식 변경 |
| `us_momentum_factor_v2.py` | 수정 | EM5/EM7 제외, EM1/EM2 장기화, EM8 추가 |
| `us_ic_analysis.py` | 수정 | 방향별 정확도 추가 |
| `us_phase3_validation.py` | **신규 생성** | Phase 3 검증 분석 |

---

## 8. 실행 순서

```
Phase 3-1: Factor Interaction Model
├── us_factor_interactions.py 생성
└── us_main_v2.py에 통합

Phase 3-2: Conviction Score
└── Phase 3-1에 포함 (I5 = Conviction)

Phase 3-3: Momentum 재설계
├── us_momentum_factor_v2.py 수정
│   ├── EM5 제외
│   ├── EM7 제외
│   ├── EM1 Lookback 60→126
│   ├── EM2 Lookback 60→126
│   └── EM8 신규 추가
└── 가중치 재조정

Phase 3-4: IC 분석 수정
└── us_ic_analysis.py 방향별 정확도 추가

Phase 3-5: 검증
├── 전체 재계산 실행
├── IC 분석 실행
└── 결과 보고서 작성
```

---

## 9. 예상 효과

| 개선 항목 | 예상 효과 |
|----------|----------|
| Factor Interaction | Pearson-Spearman 괴리 감소, 비선형 관계 포착 |
| Conviction Score | 고확신 신호의 승률 향상, A+ 샘플 증가 |
| Momentum 장기화 | 단기 IC 역작동 해소, 장기 IC 상승 |
| 방향별 정확도 | 매도 신호 정확도 측정 개선, 올바른 성과 평가 |

---

## 10. 다음 세션 시작 방법

```
1. 이 파일 읽기:
   C:\project\alpha\quant\us\result\us_phase3_작업계획.md

2. Phase 3-1부터 순차 진행:
   - us_factor_interactions.py 생성
   - us_main_v2.py 수정
   - us_momentum_factor_v2.py 수정
   - us_ic_analysis.py 수정

3. 전체 재계산 및 검증

4. Phase 3 결과 보고서 작성
```

---

*작성: Claude Code*
*상태: Phase 3 작업계획 완료*
