# Phase 3.9 한국 퀀트 모델 종합 분석 결과

**분석 일자**: 2025-12-01
**분석 기간**: 2025-08-04 ~ 2025-09-25
**총 샘플 수**: 35,826 ~ 38,564건

---

## 1. Executive Summary

### 핵심 성과 지표
| 지표 | 3일 | 15일 | 30일 | 45일 | 60일 |
|------|-----|------|------|------|------|
| Final Score IC | 0.062 | 0.146 | 0.223 | 0.280 | **0.345** |
| D10 수익률 | 0.97% | 6.05% | 14.38% | 17.84% | **24.81%** |
| D10 승률 | 50.5% | 57.0% | 67.3% | 67.8% | **72.5%** |
| 매수 정확도 | 52.9% | 61.3% | 73.0% | 76.0% | **77.4%** |
| 매도 정확도 | 55.6% | 65.0% | 65.7% | 77.2% | **81.7%** |

### 주요 발견사항
1. **보유기간 60일**에서 최고 성과 달성 (IC 0.345, D10 수익률 24.8%)
2. **Growth Factor**가 가장 강력한 예측력 보유 (IC 0.454)
3. **Dynamic Weight**가 Uniform 대비 6.1%p 우수
4. **Semiconductor 섹터**가 가장 높은 IC (0.528)

---

## 2. Factor IC 상세 분석

### 2.1 팩터별 Spearman IC

| Factor | 3d | 15d | 30d | 45d | 60d | 평가 |
|--------|-----|------|------|------|------|------|
| **Growth Factor** | 0.076 | 0.207 | 0.317 | 0.371 | **0.454** | 최우수 |
| **Final Score** | 0.062 | 0.146 | 0.223 | 0.280 | 0.345 | 우수 |
| Factor Combination | 0.048 | 0.145 | 0.245 | 0.288 | 0.373 | 우수 |
| Sector Rotation | 0.030 | 0.048 | 0.070 | 0.103 | 0.090 | 보통 |
| Value Factor | 0.043 | 0.046 | 0.025 | 0.055 | 0.085 | 약함 |
| Momentum Factor | 0.010 | -0.005 | 0.039 | 0.029 | 0.066 | 약함 |
| Quality Factor | 0.016 | 0.027 | 0.004 | -0.002 | 0.009 | 매우 약함 |

### 2.2 핵심 인사이트
- **Growth Factor의 압도적 우위**: 60일 기준 IC 0.454로 다른 팩터 대비 월등
- **Quality Factor 재검토 필요**: 전 기간에서 IC가 0.03 미만으로 예측력 미흡
- **장기 보유 효과**: 모든 팩터에서 보유기간이 길수록 IC 상승

---

## 3. Decile Test 분석

### 3.1 기간별 Decile 성과

#### 60일 보유 성과 (가장 우수)
| Decile | 점수 범위 | 평균 수익률 | 승률 | 방향성 정확도 | 신호 |
|--------|-----------|-------------|------|---------------|------|
| D1 (최저) | 3.2~17.4 | **-5.30%** | 16.3% | 74.4% | SELL |
| D2 | 17.5~20.1 | -5.73% | 21.9% | 71.7% | SELL |
| D3 | 20.2~22.8 | -6.15% | 21.8% | 73.7% | SELL |
| D4 | 22.9~26.9 | -4.07% | 24.1% | - | HOLD |
| D5 | 27.0~29.2 | 3.37% | 36.6% | - | HOLD |
| D6 | 29.3~31.0 | 6.28% | 44.4% | - | HOLD |
| D7 | 31.1~33.2 | 4.78% | 43.1% | - | HOLD |
| D8 | 33.3~37.8 | 5.92% | 48.8% | 50.2% | BUY |
| D9 | 37.9~42.6 | 10.54% | 57.3% | 57.9% | BUY |
| D10 (최고) | 42.7~59.0 | **24.81%** | **72.5%** | **72.8%** | BUY |

### 3.2 Decile Spread 분석
| 기간 | D10 수익률 | D1 수익률 | Spread | Long-Short 수익 |
|------|-----------|-----------|--------|-----------------|
| 3d | 0.97% | -0.32% | 1.30%p | 잠재 |
| 15d | 6.05% | -1.51% | 7.56%p | 유의미 |
| 30d | 14.38% | -2.35% | 16.73%p | 강력 |
| 45d | 17.84% | -4.29% | 22.13%p | 매우 강력 |
| 60d | **24.81%** | **-5.30%** | **30.11%p** | 최적 |

---

## 4. Weight Comparison (Uniform vs Dynamic)

### 4.1 기간별 비교
| 기간 | Uniform IC | Dynamic IC | 차이 | 우수 방식 |
|------|-----------|------------|------|----------|
| 3d | 0.0625 | 0.0618 | -0.0007 | Uniform |
| 15d | 0.1334 | 0.1463 | **+0.0129** | Dynamic |
| 30d | 0.1835 | 0.2234 | **+0.0398** | Dynamic |
| 45d | 0.2171 | 0.2800 | **+0.0629** | Dynamic |
| 60d | 0.2841 | 0.3450 | **+0.0609** | Dynamic |

---

## 5. 매매 신호 정확도 분석

### 5.1 등급별 성과 (60일 기준)
| 등급 | 건수 | 평균 수익률 | 승률 | 정확도 |
|------|------|-------------|------|--------|
| 매도 | 723 | -10.06% | 11.8% | **81.7%** |
| 매도 고려 | 3,912 | -4.70% | 21.0% | 71.6% |
| 중립 | 11,576 | 2.62% | 38.7% | 50.0% |
| 매수 고려 | 1,935 | 14.05% | 62.4% | 62.4% |
| 매수 | 1,062 | **31.42%** | **77.4%** | **77.4%** |

---

## 6. 테마(섹터) 분석

### 6.1 테마별 성과 순위
| 순위 | 테마 | IC | 샘플수 | 평균 수익률 | 승률 |
|------|------|-----|--------|-------------|------|
| 1 | **Semiconductor** | **0.528** | 945 | 24.04% | 31.8% |
| 2 | Energy | 0.467 | 329 | 7.79% | 26.4% |
| 3 | Electronics | 0.411 | 1,424 | 4.00% | 21.6% |
| ... | ... | ... | ... | ... | ... |
| 16 | IT/Software | 0.211 | 1,148 | -4.70% | 13.5% |
| 17 | **Telecom/Media** | **0.073** | 1,094 | -9.84% | 9.7% |

---

## 7. 옵션 5 프로세스 분석 및 Risk-Adjusted Return 현황

### 7.1 kr_main.py 옵션 5 실행 프로세스

```
analyze_all_stocks_specific_dates(db_manager, date_list)
│
├── Step 1: 섹터 데이터 갱신 (mv_sector_daily_performance)
├── Step 2: 종목 조회 (kr_intraday_total)
├── Step 3: AsyncBatchWeightCalculator → 가중치 일괄 계산
│
└── 각 종목별 analyze_single_stock() 호출:
    │
    ├── Step 1: 조건 분석 및 시장 상태 분류 (ConditionAnalyzer)
    ├── Step 2: 4개 팩터 점수 병렬 계산
    │   ├── ValueFactorCalculator
    │   ├── QualityFactorCalculator
    │   ├── MomentumFactorCalculator
    │   └── GrowthFactorCalculator
    │
    ├── Step 4: 추가 정량 지표 계산 (AdditionalMetricsCalculator)
    │   ├── VaR(95%), CVaR(95%)
    │   ├── 연간 변동성 (volatility_annual)
    │   ├── 최대 낙폭 (max_drawdown_1y)
    │   ├── 베타 (beta)
    │   └── ... (19개 지표)
    │
    ├── Step 5: 최종 점수 계산 (IC 기반 가중치)
    │   final_score = base_factor × 0.36 + sector_rotation × 0.25 + factor_combo × 0.39
    │
    ├── Step 6: 등급 결정 (Phase 3.9 타이밍 필터)
    └── Step 8: 데이터베이스 저장
```

### 7.2 Risk-Adjusted Return 현황

**현재 구현된 리스크 지표:**
| 지표 | 설명 | 파일 위치 |
|------|------|----------|
| VaR(95%) | 95% 확률 하 최대 일일 손실 | kr_additional_metrics.py:227 |
| CVaR(95%) | 꼬리 리스크 평균 손실 | kr_additional_metrics.py:267 |
| volatility_annual | 연간 변동성 % | kr_additional_metrics.py:116 |
| max_drawdown_1y | 1년 최대 낙폭 | kr_additional_metrics.py:846 |
| beta | 시장 대비 민감도 | kr_additional_metrics.py:885 |

**누락된 Risk-Adjusted Return 지표:**
| 지표 | 공식 | 필요성 |
|------|------|--------|
| **Sharpe Ratio** | (Return - Rf) / Volatility | 리스크 대비 수익률 평가 핵심 |
| **Sortino Ratio** | (Return - Rf) / Downside Deviation | 하방 리스크 조정 수익률 |
| **Calmar Ratio** | Return / Max Drawdown | MDD 대비 수익률 |
| **Information Ratio** | (Rp - Rb) / Tracking Error | 벤치마크 대비 초과수익 |

### 7.3 Risk-Adjusted Return 적용 제안

**1. kr_additional_metrics.py에 추가할 함수:**

```python
async def calculate_risk_adjusted_returns(self) -> Dict:
    """
    Risk-Adjusted Return 지표 계산

    Returns:
        Dict: sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio
    """
    # 1. 60일 수익률 계산
    return_query = """
    WITH returns AS (
        SELECT
            (LAST_VALUE(close) OVER (ORDER BY date) -
             FIRST_VALUE(close) OVER (ORDER BY date)) /
            FIRST_VALUE(close) OVER (ORDER BY date) * 100 as return_60d,
            STDDEV(daily_return) as daily_std
        FROM ...
    )
    ...
    """

    # 2. Sharpe Ratio (무위험이자율 3.5% 가정)
    rf_annual = 3.5
    rf_60d = rf_annual * (60/252)
    sharpe = (return_60d - rf_60d) / (volatility * sqrt(60/252))

    # 3. Sortino Ratio (하방 변동성만 사용)
    downside_deviation = ... # 음수 수익률만의 표준편차
    sortino = (return_60d - rf_60d) / downside_deviation

    # 4. Calmar Ratio
    calmar = return_60d / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar
    }
```

**2. 적용 위치:**
- `kr_additional_metrics.py`: `calculate_all_metrics()` 함수에 추가
- `kr_main.py`: `grade_data` 딕셔너리에 새 지표 추가
- `db_async.py`: `save_to_kr_stock_grade()` 함수에 새 컬럼 추가

**3. 데이터베이스 스키마 변경:**
```sql
ALTER TABLE kr_stock_grade ADD COLUMN sharpe_ratio DECIMAL(8,4);
ALTER TABLE kr_stock_grade ADD COLUMN sortino_ratio DECIMAL(8,4);
ALTER TABLE kr_stock_grade ADD COLUMN calmar_ratio DECIMAL(8,4);
```

---

## 8. Bullish/Bearish 시나리오 과소 예측 개선 방안

### 8.1 현재 캘리브레이션 문제 분석

**현재 상태 (phase3_7_scenario_calibration.csv 분석):**
| 시나리오 | 예측 범위 | 예측 중간값 | 실제 발생률 | 오차 |
|----------|-----------|-------------|-------------|------|
| **Bullish 0-20%** | 0-20% | 10% | 15.5% | **+5.5%p 과소** |
| Bullish 20-40% | 20-40% | 30% | 39.6% | +9.6%p 과소 |
| Bullish 40-60% | 40-60% | 50% | 63.8% | +13.8%p 과소 |
| **Bearish 0-20%** | 0-20% | 10% | 30.1% | **+20.1%p 심각한 과소** |
| Bearish 40-60% | 40-60% | 50% | 69.7% | +19.7%p 심각한 과소 |

**핵심 문제:**
- 모든 시나리오에서 실제 발생률이 예측보다 높음
- 특히 **Bearish 시나리오**에서 심각한 과소 예측 (최대 20%p 오차)
- 모델이 전반적으로 **낙관적 편향**을 가짐

### 8.2 Bullish 시나리오 과소 예측 개선 방안

**원인 분석:**
1. 강세장(2025-08~09) 시기의 실제 상승 확률이 과거 평균보다 높음
2. 모델이 과거 데이터 기반으로 보수적 예측

**개선 방안:**

```python
# kr_additional_metrics.py: calculate_scenario_probability() 수정

# 방안 1: 시장 레짐 조정 계수 추가
async def calculate_scenario_probability(self, final_score, industry):
    # 현재 시장 레짐 확인
    market_regime = await self.get_market_regime()

    # 레짐별 조정 계수
    regime_adjustment = {
        'GREED': {'bullish_boost': 1.3, 'bearish_reduce': 0.8},
        'NEUTRAL': {'bullish_boost': 1.0, 'bearish_reduce': 1.0},
        'FEAR': {'bullish_boost': 0.7, 'bearish_reduce': 1.2}
    }

    # 기존 확률에 레짐 조정 적용
    adj = regime_adjustment.get(market_regime, {'bullish_boost': 1.0, 'bearish_reduce': 1.0})
    bullish_prob = min(100, base_bullish_prob * adj['bullish_boost'])
```

```python
# 방안 2: Rolling 윈도우 기반 동적 캘리브레이션
# 최근 60일 데이터를 기준으로 확률 재산정

calibration_query = """
WITH recent_outcomes AS (
    SELECT
        CASE WHEN future_return > 10 THEN 'bullish'
             WHEN future_return < -10 THEN 'bearish'
             ELSE 'sideways' END as outcome,
        COUNT(*) as cnt
    FROM historical_predictions
    WHERE prediction_date >= CURRENT_DATE - INTERVAL '60 days'
    GROUP BY outcome
)
SELECT
    bullish_cnt / total as bullish_base_rate,
    bearish_cnt / total as bearish_base_rate
FROM recent_outcomes
"""
```

### 8.3 Bearish 시나리오 과소 예측 개선 방안

**원인 분석:**
1. **심각한 과소예측 (20%p 오차)**: 하락 확률을 크게 과소평가
2. 고점수 종목 중에서도 급락하는 케이스 미반영
3. 섹터별/테마별 리스크 차이 미반영

**개선 방안:**

```python
# 방안 1: 하락 확률 하한선 도입
def calculate_bearish_probability(self, final_score, theme):
    base_bearish = self._calculate_base_bearish(final_score)

    # 테마별 위험 가산
    high_risk_themes = ['Telecom_Media', 'IT_Software', 'Consumer_Goods']
    theme_risk_premium = 15 if theme in high_risk_themes else 0

    # 점수 구간별 하한선 (과소예측 보정)
    min_bearish = {
        (0, 20): 25,    # 저점수: 최소 25% 하락 확률
        (20, 35): 20,   # 중하위: 최소 20%
        (35, 45): 15,   # 중간: 최소 15%
        (45, 60): 12,   # 중상위: 최소 12%
        (60, 100): 8    # 고점수: 최소 8%
    }

    floor = get_floor(final_score, min_bearish)
    adjusted_bearish = max(base_bearish + theme_risk_premium, floor)

    return adjusted_bearish
```

```python
# 방안 2: Asymmetric Risk Model (비대칭 리스크 모델)
# 상승 시나리오와 하락 시나리오에 다른 모델 적용

async def calculate_asymmetric_scenario(self, final_score, industry):
    # 상승 확률: 기존 모델 (상대적으로 정확)
    bullish_prob = self._standard_bullish_model(final_score, industry)

    # 하락 확률: 보수적 모델 (과소예측 보정)
    bearish_prob = self._conservative_bearish_model(final_score, industry)

    # 정규화
    total = bullish_prob + bearish_prob
    sideways_prob = max(0, 100 - total)

    return {
        'bullish': bullish_prob * 0.9,      # 10% 하향 조정
        'sideways': sideways_prob,
        'bearish': bearish_prob * 1.4       # 40% 상향 조정
    }
```

---

## 9. 캘리브레이션 종합 개선 방안

### 9.1 Platt Scaling 기법 적용

```python
# 확률 캘리브레이션을 위한 Platt Scaling 구현
import numpy as np
from scipy.optimize import minimize

class ProbabilityCalibrator:
    """
    Platt Scaling을 사용한 확률 캘리브레이션
    """
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, predicted_probs, actual_outcomes):
        """
        실제 결과와 예측 확률을 기반으로 캘리브레이션 파라미터 학습
        """
        def log_loss(params):
            a, b = params
            calibrated = 1 / (1 + np.exp(-(a * predicted_probs + b)))
            return -np.mean(actual_outcomes * np.log(calibrated + 1e-10) +
                          (1 - actual_outcomes) * np.log(1 - calibrated + 1e-10))

        result = minimize(log_loss, [1.0, 0.0], method='BFGS')
        self.a, self.b = result.x

    def calibrate(self, raw_prob):
        """
        원시 확률을 캘리브레이션된 확률로 변환
        """
        return 1 / (1 + np.exp(-(self.a * raw_prob + self.b)))
```

### 9.2 시나리오별 캘리브레이션 테이블

```python
# 경험적 캘리브레이션 맵핑 테이블
CALIBRATION_MAP = {
    'bullish': {
        (0, 20): 1.55,    # 10% 예측 → 15.5% 보정
        (20, 40): 1.32,   # 30% 예측 → 39.6% 보정
        (40, 60): 1.28,   # 50% 예측 → 63.8% 보정
        (60, 80): 1.23,   # 70% 예측 → 85.9% 보정
    },
    'bearish': {
        (0, 20): 3.01,    # 10% 예측 → 30.1% 보정 (3배!)
        (20, 40): 1.48,   # 30% 예측 → 44.3% 보정
        (40, 60): 1.39,   # 50% 예측 → 69.7% 보정
        (60, 80): 1.26,   # 70% 예측 → 88.5% 보정
    }
}

def calibrate_probability(scenario, raw_prob):
    """예측 확률을 캘리브레이션된 확률로 변환"""
    for (low, high), multiplier in CALIBRATION_MAP[scenario].items():
        if low <= raw_prob < high:
            return min(100, raw_prob * multiplier)
    return raw_prob
```

### 9.3 구현 위치 및 적용 방법

**1. 새 파일 생성: `kr/kr_probability_calibrator.py`**

```python
class ScenarioCalibrator:
    """시나리오 확률 캘리브레이션 클래스"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.calibration_params = {}

    async def update_calibration(self, lookback_days=90):
        """최근 데이터 기반으로 캘리브레이션 파라미터 업데이트"""
        # 실제 결과와 예측 비교하여 파라미터 갱신
        pass

    def apply_calibration(self, scenario, raw_prob):
        """캘리브레이션 적용"""
        pass
```

**2. `kr_additional_metrics.py` 수정:**

```python
from kr_probability_calibrator import ScenarioCalibrator

async def calculate_scenario_probability(self, final_score, industry):
    # 기존 로직으로 raw probability 계산
    raw_probs = await self._calculate_raw_probabilities(final_score, industry)

    # 캘리브레이션 적용
    calibrator = ScenarioCalibrator(self.db_manager)
    calibrated = {
        'bullish': calibrator.apply_calibration('bullish', raw_probs['bullish']),
        'bearish': calibrator.apply_calibration('bearish', raw_probs['bearish']),
        'sideways': 100 - calibrated_bullish - calibrated_bearish
    }

    return calibrated
```

---

## 10. Telecom/Media 문제 종목 상세 분석

### 10.1 문제 현상

| 날짜 | 종목코드 | 종목명 | 점수 | 30일 수익률 | 문제 |
|------|----------|--------|------|-------------|------|
| 2025-08-22 | 310200 | 애니플러스 | 63.2 | **-20.87%** | 고점수 급락 |
| 2025-09-22 | 023770 | 플레이위드 | 62.9 | **-25.04%** | 고점수 급락 |
| 2025-09-09 | 263700 | 케어랩스 | 62.1 | **-26.52%** | 고점수 급락 |
| 2025-09-12 | 194480 | 데브시스터즈 | 61.7 | **-12.96%** | 고점수 급락 |
| 2025-09-22 | 079160 | CJ CGV | 60.8 | -6.22% | 고점수 하락 |
| 2025-09-16 | 432430 | 와이랩 | 60.7 | -10.73% | 고점수 하락 |
| 2025-09-09 | 047820 | 초록뱀미디어 | 60.2 | -16.09% | 고점수 급락 |
| 2025-08-11 | 035900 | JYP Ent. | 60.2 | -7.34% | 고점수 하락 |
| 2025-08-22 | 040300 | YTN | 59.7 | -6.21% | 고점수 하락 |
| 2025-09-16 | 207760 | 미스터블루 | 59.2 | **-18.86%** | 고점수 급락 |

**통계:**
- **평균 점수**: 60.8점 (D9~D10 수준)
- **평균 수익률**: -15.08% (기대 수익률 D10: +24.8%)
- **역방향 편차**: 약 40%p 역방향

### 10.2 원인 분석

**1. 섹터 특성상 팩터 모델 부적합**

| 팩터 | 일반 섹터 적용 | Telecom/Media 문제점 |
|------|---------------|---------------------|
| Value | PER, PBR 기반 저평가 판단 | 무형자산(IP, 콘텐츠) 가치 미반영 |
| Growth | 매출/이익 성장률 | 히트작 의존, 불규칙한 실적 |
| Quality | ROE, 부채비율 | 콘텐츠 투자비용 → 일시적 수익성 하락 |
| Momentum | 주가/거래량 추세 | 팬덤 기반 과열/급락 변동성 |

**2. 게임/엔터 종목의 특수성**

```
[일반 제조업]                    [게임/엔터]
매출 = 판매량 × 단가              매출 = 히트작 여부 × (비선형)
→ 예측 가능                       → 예측 불가

이익 성장 = 점진적                이익 = 신작 출시 사이클
→ 팩터 모델 적합                  → 이벤트 드리븐
```

**3. 분석 기간 중 악재 발생**
- 2025년 8~9월: 게임/미디어 섹터 전반 약세
- K-콘텐츠 글로벌 경쟁 심화
- 게임사 신작 부진, 광고 시장 위축

### 10.3 별도 모델 개선 근거 및 제안

**A. Telecom/Media 전용 팩터 설계**

```python
class TelecomMediaFactorCalculator:
    """
    Telecom/Media 섹터 전용 팩터 계산기
    """

    async def calculate_content_factor(self):
        """
        콘텐츠 팩터 (40%)
        - IP 가치 평가 (인기 게임/드라마 보유)
        - 콘텐츠 파이프라인 (신작 일정)
        - 글로벌 진출 현황
        """
        pass

    async def calculate_engagement_factor(self):
        """
        사용자 참여 팩터 (30%)
        - DAU/MAU (게임), 시청률 (미디어)
        - 팬덤 크기 (SNS 팔로워, 팬카페)
        - 리텐션율
        """
        pass

    async def calculate_cycle_factor(self):
        """
        사이클 팩터 (20%)
        - 신작 출시 사이클 (3-6개월 선행)
        - 계절성 (광고 시장)
        - 이벤트 일정 (콘서트, 영화 개봉)
        """
        pass

    async def calculate_risk_factor(self):
        """
        리스크 팩터 (10%)
        - 히트작 의존도 (매출 집중도)
        - 경쟁 강도 (신규 게임/드라마 출시)
        - 규제 리스크 (게임 규제)
        """
        pass
```

**B. 이벤트 드리븐 오버레이**

```python
async def get_telecom_media_events(symbol, date):
    """
    게임/미디어 이벤트 조회
    - 신작 출시 예정
    - 대규모 업데이트
    - 콘서트/영화 일정
    """
    events_query = """
    SELECT event_type, event_date, expected_impact
    FROM kr_content_events
    WHERE symbol = $1 AND event_date > $2
    ORDER BY event_date
    """
    # 이벤트 기반 점수 조정
    pass
```

**C. 팬덤/트래픽 데이터 통합**

```python
async def calculate_engagement_score(symbol):
    """
    외부 데이터 기반 engagement 점수
    - 게임: 앱스토어 순위, 스팀 동접
    - 미디어: 유튜브 조회수, 넷플릭스 순위
    - 엔터: 팬카페 회원수, SNS 언급량
    """
    pass
```

**D. 모델 분리 적용 방안**

```python
# kr_main.py 수정
async def analyze_single_stock(symbol, db_manager, ...):
    theme = await get_stock_theme(symbol)

    if theme == 'Telecom_Media':
        # 별도 모델 적용
        from kr_telecom_media_factor import TelecomMediaFactorCalculator
        calculator = TelecomMediaFactorCalculator(symbol, db_manager)
        result = await calculator.calculate_weighted_score()
    else:
        # 기존 모델 적용
        ...
```

---

## 11. V4, V25 전략 역방향 원인 분석

### 11.1 V4_Sustainable_Dividend 분석

**전략 설명:**
- 지속 가능한 배당 평가
- 배당성향, FCF 커버리지, 배당 성장률 평가

**IC 성과:**
| 기간 | Pearson IC | Spearman IC | 상태 |
|------|-----------|-------------|------|
| 15d | +0.009 | **-0.008** | Weak Negative |
| 30d | -0.012 | **-0.032** | Weak Negative |
| 45d | -0.032 | **-0.038** | Weak Negative |
| 60d | -0.039 | **-0.018** | Weak Negative |

**역방향 원인 분석:**

```
[원인 1] 배당주 = 저성장주 연관성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 안정적 배당 기업 → 성숙기 산업 → 성장 모멘텀 부재
• Growth Factor가 IC 0.454로 최강
• 배당 vs 성장 상충 관계

[원인 2] 금리 상승기 배당주 약세
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 2025년 고금리 환경 지속
• 채권 대비 배당주 매력도 하락
• 기관/외국인 배당주 비중 축소

[원인 3] 배당성향 높음 = 투자 여력 부족
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 배당성향 50-80% → 재투자 역량 제한
• R&D/설비 투자 감소 → 경쟁력 약화
```

**개선 방안:**

```python
async def calculate_v4_improved(self):
    """
    V4 개선: 성장-배당 균형 평가
    """
    # 기존 지표
    payout_ratio = ...
    fcf_coverage = ...
    dividend_growth = ...

    # 신규 추가: 성장과의 균형
    growth_score = await self.get_growth_factor_score()

    # 배당주인데 성장도 있는 경우 가산
    if dividend_yield > 3 and growth_score > 50:
        balance_bonus = 15
    else:
        balance_bonus = 0

    # 금리 환경 조정
    interest_rate = await self.get_current_interest_rate()
    if interest_rate > 4.0:  # 고금리
        rate_penalty = 10
    else:
        rate_penalty = 0

    final_score = base_score + balance_bonus - rate_penalty
    return final_score
```

### 11.2 V25_Cash_Rich_Undervalued 분석

**전략 설명:**
- 순현금 / 시가총액 비율로 저평가 판단
- PBR, 유동비율 병행 평가

**IC 성과:**
| 기간 | Pearson IC | Spearman IC | 상태 |
|------|-----------|-------------|------|
| 15d | -0.003 | +0.002 | Neutral |
| 30d | -0.016 | **-0.005** | Weak Negative |
| 45d | -0.023 | **-0.015** | Weak Negative |
| 60d | -0.031 | **-0.012** | Weak Negative |

**역방향 원인 분석:**

```
[원인 1] Value Trap 현상
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 현금 풍부 = 사업 기회 부재의 신호
• 경영진의 자본 배치 역량 부족
• 주주환원 의지 없음

[원인 2] 한국 시장 특수성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 대기업 지배구조 문제
• 현금 쌓아두고 배당 안 하는 문화
• Activist 부재로 가치 실현 지연

[원인 3] 성장 기회비용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 현금 보유 = 투자 안 함 = 성장 포기
• 시장은 성장주에 프리미엄 부여
• 저PBR이 지속되는 구조적 이유
```

**개선 방안:**

```python
async def calculate_v25_improved(self):
    """
    V25 개선: Value Trap 필터링
    """
    # 기존 지표
    net_cash_ratio = ...
    pbr = ...
    current_ratio = ...

    # 신규 추가: Value Trap 필터

    # 1. 주주환원 실적 확인
    buyback_dividend = await self.get_shareholder_return()
    if buyback_dividend < 1:  # 환원 없음
        trap_penalty = 20
    else:
        trap_penalty = 0

    # 2. 최근 투자 활동 확인
    capex_growth = await self.get_capex_growth()
    if capex_growth < -20:  # 투자 감소
        stagnation_penalty = 15
    else:
        stagnation_penalty = 0

    # 3. 현금 보유 기간 확인
    cash_holding_years = await self.get_cash_holding_duration()
    if cash_holding_years > 3:  # 3년 이상 현금 유지
        dormant_penalty = 10
    else:
        dormant_penalty = 0

    final_score = base_score - trap_penalty - stagnation_penalty - dormant_penalty
    return max(0, final_score)
```

### 11.3 V4, V25 전략 개선 종합 제안

**1. 가중치 조정 또는 제외**

```python
# kr_value_factor.py의 calculate_weighted_score()에서
# 역방향 전략 가중치 축소 또는 제외

async def calculate_weighted_score(self):
    strategies = {
        'V2_Magic_Formula': await self.calculate_v2(),
        # 'V4_Sustainable_Dividend': await self.calculate_v4(),  # 제외
        'V13_Magic_Formula_Enhanced': await self.calculate_v13(),
        # 'V25_Cash_Rich_Undervalued': await self.calculate_v25(), # 제외
        ...
    }
```

**2. 조건부 적용**

```python
# 시장 환경에 따라 V4, V25 적용 여부 결정
async def calculate_v4_conditional(self):
    market_regime = await self.get_market_regime()
    interest_rate = await self.get_interest_rate()

    # 저금리 + 약세장에서만 배당주 유효
    if interest_rate < 3.0 and market_regime in ['FEAR', 'PANIC']:
        return await self.calculate_v4()
    else:
        return None  # 적용 안 함
```

**3. 신규 전략으로 대체**

| 기존 전략 | 문제점 | 대체 전략 제안 |
|----------|--------|---------------|
| V4_Sustainable_Dividend | 저성장 함정 | V4_Growth_Dividend (성장+배당 균형) |
| V25_Cash_Rich_Undervalued | Value Trap | V25_Activist_Value (주주환원 실적 포함) |

---

## 12. 결론 및 권고사항

### 12.1 모델 강점
1. **뛰어난 예측력**: 60일 IC 0.345, D10 승률 72.5%
2. **명확한 신호 구분**: 매수/매도 정확도 77~82%
3. **안정적 성과**: Rolling IC 일관성 유지
4. **Dynamic Weight 효과**: Uniform 대비 6.1%p 개선

### 12.2 즉시 개선 필요 영역

| 우선순위 | 영역 | 개선 내용 |
|----------|------|----------|
| 1 | Bearish 캘리브레이션 | 하락 확률 40% 상향 조정 |
| 2 | Telecom/Media | 별도 모델 또는 점수 페널티 확대 |
| 3 | V4, V25 전략 | 제외 또는 조건부 적용 |
| 4 | Risk-Adjusted Return | Sharpe/Sortino Ratio 추가 |

### 12.3 최적 운용 전략
| 항목 | 권고 설정 |
|------|----------|
| **보유 기간** | 45~60일 |
| **매수 점수** | 42.7 이상 (D10) |
| **매도 점수** | 22.8 이하 (D1~D3) |
| **가중치 방식** | Dynamic Weight |
| **선호 섹터** | Semiconductor, Energy, Electronics, Shipbuilding |
| **회피 섹터** | Telecom/Media, IT/Software |
| **포지션 사이즈** | 중소형주(Q1~Q2) 선호 |

---

*본 분석은 2025-08-04 ~ 2025-09-25 기간의 한국 주식시장 데이터를 기반으로 작성되었습니다.*
