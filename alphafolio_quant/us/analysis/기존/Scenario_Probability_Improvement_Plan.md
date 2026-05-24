# Scenario Probability Calibration 개선 방안

**작성일**: 2025-12-08
**Phase**: 3.5 (Macro-Enhanced Scenario Engine)
**목표**: 강세/횡보/약세 예측 정확도 향상 (29.6% → 45%+)

---

## 0. 설계 원칙

### 최종 결과물: 3-Scenario 유지

```
┌─────────────────────────────────────────────────────────────┐
│  개인투자자용 최종 출력 (단순/직관적)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   📈 강세 (Bullish):  45%  → "적극 매수, 성장주 비중 확대"    │
│   📊 횡보 (Sideways): 35%  → "관망, 배당주/우량주 중심"       │
│   📉 약세 (Bearish):  20%  → "비중 축소, 현금 비중 확대"      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 내부 계산: 5-Macro Environment 활용

```
┌─────────────────────────────────────────────────────────────┐
│  내부 계산 엔진 (복잡/정교)                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Macro 감지] → [5-Environment 분류] → [3-Scenario 변환]    │
│                                                             │
│  GDP, CPI, Fed Rate, Yield Curve                            │
│       ↓                                                     │
│  SOFT_LANDING / HARD_LANDING / REFLATION / STAGFLATION      │
│       ↓                                                     │
│  강세 확률 ↑↓ / 횡보 확률 ↑↓ / 약세 확률 ↑↓                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**핵심**: 사용자에게는 단순한 3가지, 내부에서는 정교한 5가지로 계산

---

## 1. 현재 문제점 분석

### 1.1 예측 오차 현황 (us_phase3_3_분석결과.md)

| 시나리오 | 예측 구간 | 예측 중앙값 | 실제 발생률 | 오차 | 문제 |
|----------|----------|------------|------------|------|------|
| **Sideways** | 40-60% | 50% | **24.2%** | **-25.8%** | 2.1배 과대추정 |
| **Sideways** | 20-40% | 30% | **17.5%** | **-12.5%** | 1.7배 과대추정 |
| **Bearish** | 0-20% | 10% | **28.1%** | **+18.1%** | 2.8배 과소추정 |
| Bullish | 20-40% | 30% | 39.2% | +9.2% | 소폭 과소추정 |
| Bearish | 20-40% | 30% | 36.5% | +6.5% | 약간 과소 |
| Bearish | 40-60% | 50% | 45.6% | -4.4% | 양호 |

**Overall 최고확률 예측 정확도: 29.6% (목표: 40%+)**

### 1.2 현재 구현의 한계

**us_agent_metrics.py (Line 1003-1117)**:
```python
# 현재: final_score만 기반으로 시나리오 확률 결정
if final_score >= 75:
    raw_bull = 0.55
    raw_bear = 0.15
elif final_score >= 60:
    raw_bull = 0.40
    raw_bear = 0.20
# ...
```

**핵심 문제점**:
1. **Macro 환경 미고려**: 금리, 인플레이션, GDP 성장률 무시
2. **시장 레짐 미연동**: us_market_regime과 분리되어 있음
3. **섹터 특성 미반영**: 섹터별 Macro 민감도 차이 무시
4. **3-시나리오 한계**: Bull/Side/Bear 구분이 지나치게 단순

### 1.3 us_market_regime.py와의 불일치

| 시스템 | 분류 | 고려 요소 |
|--------|------|----------|
| Market Regime | 5개 (AI_BULL, TIGHTENING, RECOVERY, CRISIS, NEUTRAL) | VIX, Fed Rate, CPI, Yield Curve |
| Scenario Prob | 3개 (Bull, Side, Bear) | **final_score만** |

**결론**: Market Regime은 Macro를 보지만 Scenario Probability는 Macro를 무시

---

## 2. 개선 방안: 5-Macro Environment Model

### 2.1 Macro Environment 정의

기존 3-Scenario (Bull/Side/Bear)를 5-Macro Environment로 확장:

| Macro Environment | 설명 | 경제 특성 | 대표 섹터 |
|-------------------|------|----------|----------|
| **SOFT_LANDING** | 연착륙 (Goldilocks) | 성장↓ + 인플레↓ + 금리↓ | Tech, Growth |
| **HARD_LANDING** | 경착륙 (Recession) | 성장↓↓ + 인플레↓ + 금리↓ | Defensive, Bonds |
| **REFLATION** | 리플레이션 | 성장↑ + 인플레↑ + 금리↑ | Cyclicals, Value |
| **STAGFLATION** | 스태그플레이션 | 성장↓ + 인플레↑ + 금리↑ | Commodities, TIPS |
| **DEFLATION** | 디플레이션 | 성장↓↓ + 인플레↓↓ | Cash, Long Duration |

### 2.2 Macro Environment 판정 로직

```
            GDP Growth
              ↑ (High)
              |
  REFLATION   |   SOFT_LANDING
              |
 ←────────────+────────────→ Inflation
   (Low)      |      (High)
              |
  DEFLATION   |   STAGFLATION
              |
              ↓ (Low)
              GDP Growth

+ HARD_LANDING: GDP Growth < -1% 또는 실업률 급등
```

### 2.3 Macro-Scenario Probability Matrix

각 Macro 환경에서의 시나리오 확률 분포:

| Macro Environment | P(Bullish) | P(Sideways) | P(Bearish) | 포트폴리오 수익률 기대 |
|-------------------|------------|-------------|------------|-------------------|
| SOFT_LANDING | 55% | 30% | 15% | +3% (MSCI 추정) |
| HARD_LANDING | 20% | 25% | 55% | 0% (Flat) |
| REFLATION | 45% | 30% | 25% | +2% |
| STAGFLATION | 15% | 25% | 60% | -6% (MSCI 추정) |
| DEFLATION | 25% | 35% | 40% | -2% |

**출처**: [MSCI Macro Scenarios](https://www.msci.com/www/blog-posts/macro-scenarios-soft-hard-or-no/03812879490), [Fed Reserve Framework](https://www.federalreserve.gov/econres/feds/soft-landing-or-stagnation-a-framework-for-estimating-the-probabilities-of-macro-scenarios.htm)

### 2.4 5-Macro → 3-Scenario 변환 로직

**핵심**: Macro Environment를 감지한 후, 강세/횡보/약세 확률로 변환

```python
# 5-Macro Environment → 3-Scenario 변환 테이블
MACRO_TO_SCENARIO = {
    'SOFT_LANDING': {
        'bullish': 0.55,   # 연착륙 = 강세 유리
        'sideways': 0.30,
        'bearish': 0.15
    },
    'HARD_LANDING': {
        'bullish': 0.20,   # 경착륙 = 약세 우세
        'sideways': 0.25,
        'bearish': 0.55
    },
    'REFLATION': {
        'bullish': 0.45,   # 리플레이션 = 강세 but 변동성
        'sideways': 0.30,
        'bearish': 0.25
    },
    'STAGFLATION': {
        'bullish': 0.15,   # 스태그플레이션 = 약세 우세
        'sideways': 0.25,
        'bearish': 0.60
    },
    'DEFLATION': {
        'bullish': 0.25,   # 디플레이션 = 약세 경향
        'sideways': 0.35,
        'bearish': 0.40
    }
}
```

### 2.5 시나리오별 투자 전략 가이드

개인투자자에게 제공할 **최종 메시지**:

| 시나리오 | 확률 예시 | 투자 전략 | 추천 섹터 |
|----------|----------|----------|----------|
| **강세 (Bullish)** | 45%+ | 적극 매수, 성장주 비중 확대, 레버리지 고려 | Technology, Consumer Discretionary |
| **횡보 (Sideways)** | 35%+ | 관망, 배당주/우량주 중심, 옵션 전략 | Healthcare, Consumer Staples |
| **약세 (Bearish)** | 40%+ | 비중 축소, 현금 비중 확대, 방어주 전환 | Utilities, Consumer Staples |

**시나리오별 상세 가이드**:

```
┌────────────────────────────────────────────────────────────────┐
│ 📈 강세 (Bullish) 45%+ 시                                       │
├────────────────────────────────────────────────────────────────┤
│ ✓ 성장주(Growth) 비중 확대                                      │
│ ✓ 모멘텀 전략 적극 활용                                          │
│ ✓ 소형주(Small Cap) 비중 확대 고려                               │
│ ✓ 현금 비중 최소화 (10% 이하)                                    │
│ ✓ 목표 수익률: +15~25%                                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 📊 횡보 (Sideways) 35%+ 시                                      │
├────────────────────────────────────────────────────────────────┤
│ ✓ 배당주 중심 포트폴리오                                         │
│ ✓ 커버드콜 등 옵션 전략 활용                                      │
│ ✓ 대형 우량주(Quality) 중심                                      │
│ ✓ 섹터 분산 강화                                                │
│ ✓ 목표 수익률: +5~10%                                           │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 📉 약세 (Bearish) 40%+ 시                                       │
├────────────────────────────────────────────────────────────────┤
│ ✓ 현금 비중 확대 (30~50%)                                       │
│ ✓ 방어주/필수소비재 중심                                         │
│ ✓ 채권/금 등 안전자산 편입                                       │
│ ✓ 손절 기준 강화 (ATR 2배 이내)                                  │
│ ✓ 목표: 손실 최소화 (-5% 이내)                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Factor Sensitivity Framework

### 3.1 Factor-Macro Sensitivity Matrix

팩터별 Macro 환경 민감도:

| Factor | SOFT_LANDING | HARD_LANDING | REFLATION | STAGFLATION | DEFLATION |
|--------|--------------|--------------|-----------|-------------|-----------|
| **Growth** | +++ | - | + | -- | - |
| **Momentum** | ++ | -- | ++ | -- | - |
| **Value** | + | - | +++ | + | -- |
| **Quality** | + | +++ | - | ++ | ++ |

```
+++ : 최적 환경 (factor weight ×1.5)
++  : 양호 (factor weight ×1.2)
+   : 중립 (factor weight ×1.0)
-   : 부정 (factor weight ×0.8)
--  : 최악 (factor weight ×0.5)
```

### 3.2 Factor Sensitivity Types

| 민감도 유형 | 해당 Factor | 주요 Macro 변수 | 영향 |
|-------------|-------------|----------------|------|
| **Growth-Sensitive** | Growth, Momentum | GDP Growth, EPS Revision | 경기 확장기 유리 |
| **Inflation-Sensitive** | Value | CPI, PPI | 가격 결정력 기업 유리 |
| **Duration-Sensitive** | Quality | 금리, Yield Curve | 금리 인하기 유리 |

### 3.3 새로운 Factor 가중치 계산

```python
# 현재: 정적 가중치
current_weight = REGIME_WEIGHTS[regime][factor]

# 개선: Macro-adjusted 동적 가중치
macro_sensitivity = MACRO_SENSITIVITY_MATRIX[macro_env][factor]
adjusted_weight = current_weight * macro_sensitivity * stock_exposure

# stock_exposure: 개별 종목의 Macro 노출도 (섹터/베타 기반)
```

---

## 4. 종목-Macro 연결 로직

### 4.1 섹터별 Macro Environment 민감도

| Sector | SOFT_LANDING | HARD_LANDING | REFLATION | STAGFLATION |
|--------|--------------|--------------|-----------|-------------|
| Technology | +++ | - | + | -- |
| Healthcare | + | ++ | - | + |
| Financials | + | -- | +++ | + |
| Energy | - | -- | +++ | ++ |
| Consumer Discretionary | ++ | -- | ++ | -- |
| Consumer Staples | + | +++ | - | ++ |
| Utilities | + | ++ | - | + |
| Real Estate | ++ | -- | + | -- |
| Industrials | + | -- | ++ | - |
| Materials | + | -- | +++ | + |
| Communication | ++ | - | + | - |

### 4.2 Stock-Level Macro Exposure 계산

```python
def calculate_macro_exposure(symbol: str, sector: str, beta: float) -> Dict:
    """
    종목별 Macro 민감도 계산

    Components:
    1. Sector Exposure: 섹터별 Macro 민감도
    2. Beta Adjustment: 시장 베타 기반 조정
    3. Size Factor: 대형주 vs 소형주 (소형주가 Macro에 더 민감)
    """
    sector_exposure = SECTOR_MACRO_SENSITIVITY[sector]
    beta_adjustment = min(2.0, max(0.5, beta))

    return {
        'growth_exposure': sector_exposure['growth'] * beta_adjustment,
        'inflation_exposure': sector_exposure['inflation'],
        'duration_exposure': sector_exposure['duration'] / beta_adjustment
    }
```

---

## 5. 신규 Scenario Probability 계산 로직

### 5.1 계산 흐름

```
1. Detect Macro Environment (from us_market_regime.py extended)
   ↓
2. Get Base Scenario Probability (from Macro-Scenario Matrix)
   ↓
3. Adjust by Stock Characteristics
   - Sector Exposure
   - Factor Scores (Growth/Value/Quality/Momentum)
   - Beta
   ↓
4. Apply Historical Calibration
   ↓
5. Output: scenario_bullish_prob, scenario_sideways_prob, scenario_bearish_prob
```

### 5.2 구현 예시

```python
async def calculate_scenario_probability_v2(
    self,
    symbol: str,
    analysis_date: date,
    final_score: float,
    sector: str,
    factor_scores: Dict[str, float],  # 신규: 개별 팩터 점수
    macro_env: str,  # 신규: Macro Environment
    beta: float = 1.0  # 신규: 베타
) -> Dict[str, Any]:
    """
    Phase 3.5: Macro-aware Scenario Probability

    Calculation:
    1. Base from Macro Environment
    2. Adjust by sector exposure
    3. Adjust by factor composition
    4. Apply calibration
    """

    # Step 1: Base probability from Macro Environment
    base_probs = MACRO_SCENARIO_MATRIX[macro_env]  # {bull, side, bear}

    # Step 2: Sector adjustment
    sector_mult = SECTOR_MACRO_SENSITIVITY[sector][macro_env]

    # Step 3: Factor composition adjustment
    # Growth-heavy stocks: more sensitive to macro
    growth_weight = factor_scores.get('growth', 50) / 100
    value_weight = factor_scores.get('value', 50) / 100

    if macro_env in ['SOFT_LANDING', 'REFLATION']:
        # Growth-friendly: boost bull probability
        bull_adj = 0.05 * growth_weight
    elif macro_env in ['STAGFLATION', 'HARD_LANDING']:
        # Value/Quality-friendly: boost bear probability for growth stocks
        bear_adj = 0.10 * growth_weight

    # Step 4: Calculate final probabilities
    bullish_prob = base_probs['bull'] * sector_mult + bull_adj
    bearish_prob = base_probs['bear'] / sector_mult + bear_adj
    sideways_prob = 1.0 - bullish_prob - bearish_prob

    # Step 5: Clamp and normalize
    bullish_prob = max(0.05, min(0.80, bullish_prob))
    bearish_prob = max(0.05, min(0.80, bearish_prob))
    sideways_prob = max(0.05, 1.0 - bullish_prob - bearish_prob)

    # Normalize
    total = bullish_prob + sideways_prob + bearish_prob

    return {
        'scenario_bullish_prob': round(bullish_prob / total * 100),
        'scenario_sideways_prob': round(sideways_prob / total * 100),
        'scenario_bearish_prob': round(bearish_prob / total * 100),
        'macro_environment': macro_env,
        'sector_adjustment': sector_mult
    }
```

---

## 6. 필요한 데이터 소스

### 6.1 신규 필요 데이터

| 데이터 | 테이블 | 현재 상태 | 용도 |
|--------|--------|----------|------|
| GDP Growth | us_gdp | **필요** | Macro 판정 |
| PMI | us_pmi | **필요** | 경기 선행 지표 |
| Unemployment Change | us_unemployment_rate | **있음** | 경착륙 판정 |
| 10Y-2Y Spread | us_treasury_yield | **있음** | Yield Curve |
| Fed Rate | us_fed_funds_rate | **있음** | 금리 환경 |
| CPI YoY | us_cpi | **있음** | 인플레이션 |
| DXY | us_dollar_index | **있음** | 달러 강세 |
| Credit Spread | us_credit_spread | **있음** | 신용 위험 |

### 6.2 us_market_regime.py 확장 필요

현재 5-Regime을 5-Macro Environment와 통합:

```python
MACRO_ENVIRONMENT_MAP = {
    # Current Regime → Macro Environment
    'AI_BULL': 'SOFT_LANDING',     # Low VIX + NASDAQ outperform
    'RECOVERY': 'REFLATION',        # Strong rally + moderate inflation
    'TIGHTENING': 'STAGFLATION',    # High rate + high CPI
    'CRISIS': 'HARD_LANDING',       # VIX spike + market crash
    'NEUTRAL': 'SOFT_LANDING'       # Default assumption
}
```

---

## 7. 기대 효과

### 7.1 예측 오차 개선 목표

| 시나리오 | 현재 오차 | 목표 오차 | 개선률 |
|----------|----------|----------|--------|
| Sideways (40-60%) | -25.8% | -10% | 61% |
| Sideways (20-40%) | -12.5% | -5% | 60% |
| Bearish (0-20%) | +18.1% | +8% | 56% |
| Overall 정확도 | 29.6% | **45%+** | +52% |

### 7.2 포트폴리오 수익률 개선

MSCI 연구 결과 기반:
- Macro 시나리오 인식 시 포트폴리오 드로다운 30% 감소
- Stagflation 조기 인식 시 -6% 손실 → -2% 손실 가능

---

## 8. 구현 로드맵

### Phase 3.5.1: Macro Environment Detection (1주)
- [ ] us_gdp, us_pmi 테이블 생성 및 데이터 수집
- [ ] us_market_regime.py에 `detect_macro_environment()` 추가
- [ ] 5-Macro Environment 판정 로직 구현

### Phase 3.5.2: Factor Sensitivity Matrix (1주)
- [ ] MACRO_SENSITIVITY_MATRIX 상수 정의
- [ ] SECTOR_MACRO_SENSITIVITY 상수 정의
- [ ] calculate_macro_exposure() 함수 구현

### Phase 3.5.3: Scenario Probability v2 (1주)
- [ ] us_agent_metrics.py에 calculate_scenario_probability_v2() 구현
- [ ] us_main_v2.py에서 Macro Environment 연동
- [ ] us_stock_grade 테이블에 macro_environment 컬럼 추가

### Phase 3.5.4: Calibration & Validation (1주)
- [ ] 2024-2025 데이터로 백테스트
- [ ] Calibration 계수 재산출
- [ ] A/B 테스트 (v1 vs v2)

---

## 9. 참고 자료

### Academic/Industry Sources
1. [MSCI Macro Scenarios (2024)](https://www.msci.com/www/blog-posts/macro-scenarios-soft-hard-or-no/03812879490)
2. [Fed Reserve - Soft Landing Framework](https://www.federalreserve.gov/econres/feds/soft-landing-or-stagnation-a-framework-for-estimating-the-probabilities-of-macro-scenarios.htm)
3. [BlackRock 2025 Investment Directions](https://www.blackrock.com/us/financial-professionals/insights/investment-directions-spring-2025)
4. [Wellington - 5 Macro Themes 2025](https://www.wellington.com/en/insights/5-macro-themes-in-2025)
5. [Saxo - Reflation to Stagflation Positioning](https://www.home.saxo/content/articles/bonds/from-reflation-to-stagflation-how-to-position-12052021)
6. [ACM - Augmenting Equity Factor Investing with Global Macro Regimes](https://dl.acm.org/doi/fullHtml/10.1145/3677052.3698620)

### Key Insights
- **MSCI**: Stagflation -6% portfolio loss vs Soft Landing +3% gain
- **Fed Reserve**: Non-Gaussian BEGE-GARCH model for time-varying volatility
- **Wellington**: 2025 will abandon "soft-landing" consensus
- **BlackRock**: Elevated inflation persistence expected
- **Saxo**: Low-volatility factor outperformed during 1970-80s stagflation

---

## 10. 결론

### 10.1 핵심 변경 사항

| 구분 | 현재 (v1) | 개선 (v2) |
|------|----------|----------|
| **출력 형식** | 강세/횡보/약세 3가지 | **동일** (사용자 친화적 유지) |
| **내부 계산** | final_score만 사용 | **5-Macro Environment 활용** |
| **Macro 고려** | 없음 | GDP, CPI, Fed Rate, Yield Curve |
| **섹터 조정** | 없음 | 섹터별 Macro 민감도 적용 |
| **팩터 연동** | 없음 | Growth/Inflation/Duration 민감도 |

### 10.2 개인투자자 최종 결과물

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 AAPL (Apple Inc.) - 2025-12-08 분석 결과                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📈 강세 확률: 52%  ← 가장 높음                                  │
│  📊 횡보 확률: 33%                                              │
│  📉 약세 확률: 15%                                              │
│                                                                 │
│  💡 투자 전략: "성장주 비중 확대, 모멘텀 전략 활용"                │
│  📌 추천 행동: 적극 매수 고려                                    │
│                                                                 │
│  [내부 참고] Macro: SOFT_LANDING / Sector 조정: +8%              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 예상 효과

| 지표 | 현재 | 목표 | 개선률 |
|------|------|------|--------|
| **횡보 과대추정** | 2.1배 | 1.2배 | **43%** |
| **약세 과소추정** | 2.8배 | 1.3배 | **54%** |
| **Overall 정확도** | 29.6% | **45%+** | **+52%** |
| **사용자 신뢰도** | - | 상승 예상 | - |

### 10.4 핵심 메시지

> **"사용자에게는 단순하게, 내부에서는 정교하게"**

- 개인투자자는 **강세/횡보/약세 3가지**만 확인
- 각 시나리오에 맞는 **투자 전략 가이드** 제공
- 내부에서는 **5-Macro Environment**로 정확도 향상
- **섹터/팩터별 조정**으로 종목 특성 반영

---

*작성: Claude Code | 날짜: 2025-12-08*
