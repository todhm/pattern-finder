# US Quant Phase 3.1 상세 작업계획

## 작성일: 2025-11-29
## 목적: Phase 3 분석 결과 기반 시스템 개선

---

## 1. 개선 과제 요약

| 과제 | 현상 | 목표 | 우선순위 |
|------|------|------|----------|
| Mean-Median 괴리 해결 | 평균 +150%, Median -3% | ORS로 이상치 식별/경고 | P1 |
| NASDAQ 예측력 개선 | IC 0.138 (NYSE 0.253) | 거래소별 가중치 최적화 | P1 |
| EM1/EM2/RV2 재설계 | 음의 IC (-0.03 ~ -0.14) | 역방향 활용 또는 재설계 | P2 |
| 섹터 동적 가중치 | 섹터별 IC 편차 큼 | 섹터별 최적 가중치 적용 | P2 |

---

## 2. 설계 원칙: SQL 쿼리 효율성

### 2.1 기존 문제
```python
# BAD: 종목별 개별 쿼리 (N개 종목 = N번 쿼리)
for ticker in tickers:
    weight = get_exchange_weight(ticker)  # 쿼리 1회
    sector_adj = get_sector_adjustment(ticker)  # 쿼리 1회
    ors = calculate_ors(ticker)  # 쿼리 1회
```

### 2.2 개선 설계
```python
# GOOD: 한 번의 배치 쿼리로 전체 처리
# 1) 조정값은 Python 딕셔너리로 관리 (DB 조회 없음)
# 2) SQL CASE WHEN으로 조건부 계산
# 3) 한 번의 UPDATE 쿼리로 전체 종목 처리
```

### 2.3 조정값 관리 방식
```python
# config/weight_adjustments.py (신규 파일)
EXCHANGE_ADJUSTMENTS = {
    'NASDAQ': {'growth': +5, 'momentum': -5, 'quality': 0, 'value': 0},
    'NYSE': {'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0},
    'AMEX': {'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0},
}

SECTOR_ADJUSTMENTS = {
    'HEALTHCARE': {'growth': +10, 'momentum': -5, 'quality': -5, 'value': 0},
    'FINANCIAL SERVICES': {'growth': 0, 'momentum': +10, 'quality': 0, 'value': -5},
    'TECHNOLOGY': {'growth': +5, 'momentum': 0, 'quality': 0, 'value': -5},
    'BASIC MATERIALS': {'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0},  # 균형
    # ... 기타 섹터
}

ORS_THRESHOLDS = {
    'price_threshold': 5.0,      # $5 미만 페니주식
    'volume_threshold': 100000,  # 일평균 거래량
    'volatility_threshold': 100, # 연환산 변동성 %
    'mktcap_threshold': 300,     # 시가총액 $300M 미만
}
```

---

## 3. 과제별 상세 구현 계획

### 3.1 Outlier Risk Score (ORS) 도입

#### 3.1.1 목적
- 5번째 팩터가 아닌 **별도 위험 지표** (0-100점)
- Mean-Median 괴리 원인인 이상치 종목 식별
- position_size 조정 및 경고 표시에 활용

#### 3.1.2 ORS 구성요소

| 요소 | 조건 | 점수 | 가중치 |
|------|------|------|--------|
| Penny Stock | price < $5 | 0-25 | 25% |
| Low Volume | avg_volume < 100K | 0-25 | 25% |
| High Volatility | annual_vol > 100% | 0-25 | 25% |
| Small Cap | mktcap < $300M | 0-25 | 25% |

```python
# ORS 계산 공식
ors = (penny_score * 0.25) + (volume_score * 0.25) +
      (volatility_score * 0.25) + (mktcap_score * 0.25)
```

#### 3.1.3 ORS 적용 방식

```python
# position_size 조정
if ors >= 80:
    position_size *= 0.25  # 75% 축소
    risk_flag = 'EXTREME_RISK'
elif ors >= 60:
    position_size *= 0.50  # 50% 축소
    risk_flag = 'HIGH_RISK'
elif ors >= 40:
    position_size *= 0.75  # 25% 축소
    risk_flag = 'MODERATE_RISK'
else:
    risk_flag = 'NORMAL'
```

#### 3.1.4 구현 파일
- **us_outlier_risk.py** (신규)
  - `calculate_ors_batch()`: 전체 종목 ORS 일괄 계산
  - `apply_ors_adjustment()`: position_size 조정

#### 3.1.5 SQL 효율성 (한 번의 쿼리)
```sql
UPDATE us_stock_grade usg
SET
    outlier_risk_score = (
        -- Penny Stock Score (0-25)
        CASE
            WHEN usp.price < 1 THEN 25
            WHEN usp.price < 3 THEN 20
            WHEN usp.price < 5 THEN 15
            WHEN usp.price < 10 THEN 5
            ELSE 0
        END +
        -- Volume Score (0-25)
        CASE
            WHEN usp.avg_volume < 50000 THEN 25
            WHEN usp.avg_volume < 100000 THEN 20
            WHEN usp.avg_volume < 500000 THEN 10
            ELSE 0
        END +
        -- Volatility Score (0-25)
        CASE
            WHEN usp.volatility_252d > 150 THEN 25
            WHEN usp.volatility_252d > 100 THEN 20
            WHEN usp.volatility_252d > 75 THEN 10
            ELSE 0
        END +
        -- Market Cap Score (0-25)
        CASE
            WHEN usp.market_cap < 100000000 THEN 25
            WHEN usp.market_cap < 300000000 THEN 20
            WHEN usp.market_cap < 1000000000 THEN 10
            ELSE 0
        END
    ),
    risk_flag = CASE
        WHEN outlier_risk_score >= 80 THEN 'EXTREME_RISK'
        WHEN outlier_risk_score >= 60 THEN 'HIGH_RISK'
        WHEN outlier_risk_score >= 40 THEN 'MODERATE_RISK'
        ELSE 'NORMAL'
    END
FROM us_stock_price usp
WHERE usg.ticker = usp.ticker AND usg.base_date = usp.base_date;
```

---

### 3.2 거래소별 가중치 최적화

#### 3.2.1 현황 분석

| 거래소 | final_score IC (30일) | 문제점 |
|--------|----------------------|--------|
| NYSE | 0.050 | 기준 (양호) |
| NASDAQ | -0.008 | 음수 IC (예측력 없음) |
| 차이 | +0.058 | NASDAQ 최적화 필요 |

#### 3.2.2 NASDAQ 팩터별 IC

| Factor | NYSE IC | NASDAQ IC | 조정 방향 |
|--------|---------|-----------|-----------|
| growth | 0.077 | 0.030 | NASDAQ: +5% |
| momentum | 0.035 | 0.003 | NASDAQ: -5% |
| quality | 0.011 | -0.018 | NASDAQ: -3% |
| value | -0.016 | -0.022 | 동일 유지 |

#### 3.2.3 구현 방식 (쿼리 효율적)

```python
# us_exchange_optimizer.py (신규)

EXCHANGE_WEIGHT_ADJUSTMENTS = {
    'NYSE': {'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0},
    'NASDAQ': {'growth': 5, 'momentum': -5, 'quality': -3, 'value': 3},
    'AMEX': {'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0},
}

def get_adjusted_weights(base_weights: dict, exchange: str) -> dict:
    """거래소별 조정된 가중치 반환 (DB 쿼리 없음)"""
    adj = EXCHANGE_WEIGHT_ADJUSTMENTS.get(exchange, {})
    return {
        factor: base_weights[factor] + adj.get(factor, 0)
        for factor in base_weights
    }
```

#### 3.2.4 SQL 통합 (한 번의 쿼리)
```sql
-- final_score 계산 시 거래소 조정 포함
UPDATE us_stock_grade
SET final_score =
    growth_score * (
        CASE exchange
            WHEN 'NASDAQ' THEN (@base_growth + 5) / 100.0
            ELSE @base_growth / 100.0
        END
    ) +
    momentum_score * (
        CASE exchange
            WHEN 'NASDAQ' THEN (@base_momentum - 5) / 100.0
            ELSE @base_momentum / 100.0
        END
    ) +
    quality_score * (
        CASE exchange
            WHEN 'NASDAQ' THEN (@base_quality - 3) / 100.0
            ELSE @base_quality / 100.0
        END
    ) +
    value_score * (
        CASE exchange
            WHEN 'NASDAQ' THEN (@base_value + 3) / 100.0
            ELSE @base_value / 100.0
        END
    )
WHERE base_date = @target_date;
```

---

### 3.3 EM1/EM2/RV2 전략 재설계

#### 3.3.1 현황

| 전략 | 252일 IC | 현재 상태 | 재설계 방향 |
|------|----------|-----------|-------------|
| EM1 | -0.030 | 사용 중 | 역방향 신호로 활용 |
| EM2 | -0.091 | 사용 중 | 역방향 신호로 활용 |
| RV2 | -0.139 | 사용 중 | 역방향 신호로 활용 |

#### 3.3.2 역방향 활용 설계

**개념**: 음의 IC = "점수 높을수록 수익률 낮음" → 점수 역전 시 양의 IC

```python
# us_momentum_factor_v2.py 수정

def calculate_em1_reversed(self, df):
    """EM1 역방향: 원래 점수를 100에서 뺌"""
    original_score = self.calculate_em1(df)
    return 100 - original_score  # 역전

def calculate_em2_reversed(self, df):
    """EM2 역방향"""
    original_score = self.calculate_em2(df)
    return 100 - original_score

# us_value_factor.py 수정
def calculate_rv2_reversed(self, df):
    """RV2 역방향"""
    original_score = self.calculate_rv2(df)
    return 100 - original_score
```

#### 3.3.3 구현 옵션

**Option A: 기존 전략 역전** (권장)
- 장점: 코드 변경 최소화, 검증된 로직 유지
- 단점: 단순 역전이 최적인지 추가 검증 필요

**Option B: 신규 전략 개발**
- 장점: 근본적 개선 가능
- 단점: 개발/검증 시간 소요

#### 3.3.4 검증 방법
```python
# us_ic_analysis.py에 추가
def validate_reversed_strategies():
    """역방향 전략 IC 검증"""
    strategies = ['em1_reversed', 'em2_reversed', 'rv2_reversed']
    for strategy in strategies:
        ic = calculate_ic(f'{strategy}_score', 'return_252d')
        print(f"{strategy}: IC = {ic}")
        # 예상: 양의 IC로 전환
```

---

### 3.4 섹터 동적 가중치

#### 3.4.1 섹터별 최적 팩터 (IC 기준)

| 섹터 | 최고 IC 팩터 | IC 값 | 권장 가중치 조정 |
|------|-------------|-------|-----------------|
| HEALTHCARE | Growth | 0.034 | Growth +10% |
| FINANCIAL SERVICES | EM7 | 0.415 | Momentum +10% |
| TECHNOLOGY | Growth | 0.052 | Growth +5% |
| BASIC MATERIALS | 균형 | 0.06+ | 조정 없음 |
| CONSUMER DEFENSIVE | Quality | 0.169 | Quality +10% |
| ENERGY | Growth | -0.05 | Quality -10% |

#### 3.4.2 섹터 조정 Config

```python
# us_sector_dynamic_weights.py (신규)

SECTOR_WEIGHT_ADJUSTMENTS = {
    'HEALTHCARE': {
        'growth': 10, 'momentum': -5, 'quality': -5, 'value': 0
    },
    'FINANCIAL SERVICES': {
        'growth': 0, 'momentum': 10, 'quality': 0, 'value': -5
    },
    'TECHNOLOGY': {
        'growth': 5, 'momentum': 0, 'quality': 0, 'value': -5
    },
    'CONSUMER DEFENSIVE': {
        'growth': 0, 'momentum': -5, 'quality': 10, 'value': -5
    },
    'ENERGY': {
        'growth': 5, 'momentum': 0, 'quality': -10, 'value': 5
    },
    'BASIC MATERIALS': {
        'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0
    },
    # 기타 섹터는 조정 없음
}

def get_sector_adjusted_weights(base_weights: dict, sector: str) -> dict:
    """섹터별 조정된 가중치 반환 (DB 쿼리 없음)"""
    adj = SECTOR_WEIGHT_ADJUSTMENTS.get(sector, {})
    adjusted = {
        factor: base_weights[factor] + adj.get(factor, 0)
        for factor in base_weights
    }
    # 합계 100으로 정규화
    total = sum(adjusted.values())
    return {k: v * 100 / total for k, v in adjusted.items()}
```

---

## 4. 통합 가중치 계산 플로우

### 4.1 계산 순서
```
1. 레짐 판단 (CRISIS/NORMAL/...) → base_weights 결정
2. 거래소 조정 (NYSE/NASDAQ) → exchange_adjusted_weights
3. 섹터 조정 → sector_adjusted_weights
4. 100% 정규화 → final_weights
5. 점수 계산: final_score = Σ(factor_score × final_weight)
6. ORS 계산 → risk_flag, position_size 조정
```

### 4.2 통합 SQL (한 번의 쿼리로 전체 처리)

```sql
-- us_calculate_final_score.sql
WITH weight_calc AS (
    SELECT
        usg.ticker,
        usg.base_date,
        usg.exchange,
        usg.sector,
        usg.growth_score,
        usg.momentum_score,
        usg.quality_score,
        usg.value_score,

        -- Step 1: 레짐 기본 가중치 (파라미터로 전달)
        @regime_growth AS base_g,
        @regime_momentum AS base_m,
        @regime_quality AS base_q,
        @regime_value AS base_v,

        -- Step 2: 거래소 조정
        CASE usg.exchange
            WHEN 'NASDAQ' THEN @regime_growth + 5
            ELSE @regime_growth
        END AS adj1_g,
        CASE usg.exchange
            WHEN 'NASDAQ' THEN @regime_momentum - 5
            ELSE @regime_momentum
        END AS adj1_m,
        CASE usg.exchange
            WHEN 'NASDAQ' THEN @regime_quality - 3
            ELSE @regime_quality
        END AS adj1_q,
        CASE usg.exchange
            WHEN 'NASDAQ' THEN @regime_value + 3
            ELSE @regime_value
        END AS adj1_v,

        -- Step 3: 섹터 조정 (거래소 조정값에 추가)
        CASE usg.sector
            WHEN 'HEALTHCARE' THEN adj1_g + 10
            WHEN 'TECHNOLOGY' THEN adj1_g + 5
            WHEN 'ENERGY' THEN adj1_g + 5
            ELSE adj1_g
        END AS final_g,
        -- ... (momentum, quality, value도 동일 패턴)

    FROM us_stock_grade usg
    WHERE usg.base_date = @target_date
),
normalized AS (
    SELECT
        *,
        (final_g + final_m + final_q + final_v) AS weight_sum
    FROM weight_calc
)
UPDATE us_stock_grade usg
SET
    final_score = (
        n.growth_score * (n.final_g / n.weight_sum) +
        n.momentum_score * (n.final_m / n.weight_sum) +
        n.quality_score * (n.final_q / n.weight_sum) +
        n.value_score * (n.final_v / n.weight_sum)
    ),
    weight_growth = n.final_g / n.weight_sum * 100,
    weight_momentum = n.final_m / n.weight_sum * 100,
    weight_quality = n.final_q / n.weight_sum * 100,
    weight_value = n.final_v / n.weight_sum * 100
FROM normalized n
WHERE usg.ticker = n.ticker AND usg.base_date = n.base_date;
```

### 4.3 쿼리 수 비교

| 방식 | 종목 수 | 쿼리 수 | 예상 시간 |
|------|---------|---------|-----------|
| 개별 쿼리 (기존) | 5,000 | 15,000+ | 수 분 |
| 배치 쿼리 (개선) | 5,000 | 2-3 | 수 초 |

---

## 5. 파일 구조

### 5.1 신규 파일
```
us/
├── config/
│   └── weight_adjustments.py     # 조정값 config (신규)
├── us_outlier_risk.py            # ORS 계산 (신규)
├── us_exchange_optimizer.py      # 거래소 조정 (신규)
├── us_sector_dynamic_weights.py  # 섹터 조정 (신규)
└── sql/
    └── calculate_final_score.sql # 통합 SQL (신규)
```

### 5.2 수정 파일
```
us/
├── us_momentum_factor_v2.py      # EM1/EM2 역방향 추가
├── us_value_factor.py            # RV2 역방향 추가
├── us_main_v2.py                 # 통합 플로우 수정
└── analysis/
    └── us_ic_analysis.py         # 검증 함수 추가
```

---

## 6. 테이블 스키마 변경

### 6.1 us_stock_grade 컬럼 추가

```sql
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS
    outlier_risk_score DECIMAL(5,2) DEFAULT 0,
    risk_flag VARCHAR(20) DEFAULT 'NORMAL',
    weight_growth DECIMAL(5,2),
    weight_momentum DECIMAL(5,2),
    weight_quality DECIMAL(5,2),
    weight_value DECIMAL(5,2);

CREATE INDEX idx_ors ON us_stock_grade(outlier_risk_score);
CREATE INDEX idx_risk_flag ON us_stock_grade(risk_flag);
```

---

## 7. 구현 순서

### Phase 3.1.1: ORS 도입 (우선)
1. `config/weight_adjustments.py` 생성
2. `us_outlier_risk.py` 구현
3. 테이블 스키마 변경
4. IC 분석으로 ORS 유효성 검증

### Phase 3.1.2: 거래소 최적화
1. `us_exchange_optimizer.py` 구현
2. NASDAQ 종목 대상 A/B 테스트
3. IC 개선 확인

### Phase 3.1.3: 섹터 동적 가중치
1. `us_sector_dynamic_weights.py` 구현
2. 섹터별 IC 변화 모니터링
3. 조정값 튜닝

### Phase 3.1.4: EM1/EM2/RV2 재설계
1. 역방향 전략 구현
2. IC 검증
3. 기존 전략과 성과 비교

### Phase 3.1.5: 통합 및 검증
1. `us_main_v2.py` 통합
2. 전체 파이프라인 테스트
3. IC 분석 리포트 생성

---

## 8. 예상 효과

| 지표 | 현재 | 목표 | 개선율 |
|------|------|------|--------|
| NASDAQ IC (30일) | -0.008 | +0.03 | +475% |
| Mean-Median 괴리 | 153% | 20% | -87% |
| 음의 IC 전략 수 | 3개 | 0개 | -100% |
| 쿼리 수 (배치당) | 15,000+ | 3 | -99.98% |

---

## 9. 검증 체크리스트

- [ ] ORS 계산 정확성 검증
- [ ] ORS-position_size 연동 확인
- [ ] 거래소별 IC 개선 확인
- [ ] 섹터별 IC 변화 모니터링
- [ ] EM1/EM2/RV2 역방향 IC 검증
- [ ] 통합 가중치 합계 100% 확인
- [ ] 배치 쿼리 성능 테스트 (< 10초)
- [ ] 기존 시스템 대비 백테스트

---

*작성: 2025-11-29*
*Phase 3 분석 결과 기반*
