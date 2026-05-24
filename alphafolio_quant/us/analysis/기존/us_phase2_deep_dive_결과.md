# US Phase 2 Deep Dive 분석 결과

**분석일**: 2025-11-26
**데이터**: 137,858건 (2024-01-16 ~ 2025-09-19, 31개 분석일)

---

## 핵심 발견 요약

| 문제 | 근본 원인 | 해결 방안 |
|------|----------|----------|
| Momentum IC -0.056~-0.093 | **EM3, EM4 전략이 반대로 작동** | EM3, EM4 점수 로직 반전 |
| Healthcare IC -0.068 | Growth 팩터만 유효 | Healthcare 전용 Growth 중심 모델 |
| NASDAQ IC -0.029 | Healthcare 집중 (29%) | NYSE 우선 필터 적용 |

---

## STEP 1: Momentum Factor 심층 분석

### 1.1 전략별 IC 분석

| 전략 | 설명 | IC (30일) | IC (60일) | 상태 |
|------|-----|----------|----------|------|
| **EM1** | EPS 수정 | -0.007 | +0.003 | 중립 |
| **EM2** | 매출 수정 | +0.026 | +0.020 | 중립 |
| **EM3** | 1개월 수익률 | **-0.457** | **-0.570** | **역작동** |
| **EM4** | 3개월 수익률 | **-0.322** | **-0.422** | **역작동** |
| **EM5** | 6개월 수익률 | -0.000 | -0.012 | 중립 |
| EM6 | (데이터 없음) | N/A | N/A | - |
| **EM7** | 52주 고점 근접도 | **+0.404** | **+0.546** | **정상 작동** |

### 1.2 핵심 발견: EM3/EM4가 완전히 반대로 작동

**Decile 테스트 결과 (30일 수익률):**

| 분위 | EM3 평균 수익률 | EM3 승률 | EM4 평균 수익률 | EM4 승률 |
|------|----------------|---------|----------------|---------|
| D1 (저점수) | **+27.94%** | **79.5%** | **+31.37%** | **72.8%** |
| D5 (중간) | +4.57% | 62.5% | +2.87% | 60.1% |
| D10 (고점수) | **-5.20%** | **25.4%** | **-3.74%** | **27.7%** |
| 단조성 | **-1.000** | - | **-1.000** | - |

**해석:**
- EM3 (1개월 수익률): 최근 급등한 종목이 **오히려 하락** (평균 회귀 현상)
- EM4 (3개월 수익률): 동일한 패턴 - 최근 수익률이 높을수록 향후 수익률 낮음
- 이것은 **점수 산정 로직의 오류**임 (시장 비효율이 아님)

### 1.3 EM7은 완벽하게 작동

| 분위 | EM7 평균 수익률 | EM7 승률 |
|------|----------------|---------|
| D1 (저점수) | -3.72% | 33.7% |
| D5 (중간) | +2.14% | 58.1% |
| D9 (고점수) | **+18.92%** | **75.1%** |
| 단조성 | **+1.000** | - |

**EM7 (52주 고점 근접도)**는 완벽한 양의 단조성 - 52주 고점에 가까운 종목이 계속 상승.

### 1.4 섹터별 분석

EM3 IC는 **모든 섹터에서 음수**:

| 섹터 | EM3 IC | EM7 IC |
|------|--------|--------|
| Consumer Cyclical | -0.493 | +0.396 |
| Industrials | -0.488 | +0.451 |
| Communication Services | -0.476 | +0.364 |
| Technology | -0.462 | +0.447 |
| Healthcare | -0.431 | +0.334 |

**결론:** EM3/EM4 역작동은 **시장 전체** 현상이며, 특정 섹터 문제가 아님.

### 1.5 Rolling IC 안정성

| 전략 | 평균 IC | 표준편차 | 양수 비율 |
|------|---------|---------|----------|
| EM3 | -0.472 | 0.108 | **0.0%** |
| EM4 | -0.314 | 0.106 | 5.3% |
| EM7 | +0.404 | 0.105 | **100.0%** |

EM3은 분석 기간 동안 **단 한 번도** 양의 IC를 기록한 적 없음. EM7은 **항상** 양의 IC.

---

## STEP 2: Healthcare 섹터 심층 분석

### 2.1 업종별 IC

| 업종 | IC (30일) | 샘플 수 | Healthcare 내 비중 |
|------|----------|---------|-------------------|
| **Biotechnology** | -0.007 | 15,965 | **55.4%** |
| Medical Devices | -0.007 | 3,806 | 13.2% |
| Drug Mfg - Specialty | +0.001 | 2,122 | 7.4% |
| Medical Instruments | -0.063 | 1,584 | 5.5% |
| Health Info Services | -0.015 | 1,464 | 5.1% |
| Diagnostics & Research | -0.039 | 1,444 | 5.0% |
| **Medical Care Facilities** | **+0.090** | 1,192 | 4.1% |
| **Pharmaceutical Retailers** | **+0.399** | 162 | 0.6% |

**발견:** Biotechnology (Healthcare의 55%)가 IC를 0에 가깝게 만듦. 극단적 음의 IC가 아니라 **물량 문제**.

### 2.2 Healthcare 내 팩터 효과

| 팩터 | IC (30일) | IC (60일) | 평가 |
|------|----------|----------|------|
| Value | -0.040 | -0.047 | 작동 안함 |
| Quality | -0.017 | -0.022 | 작동 안함 |
| Momentum | -0.039 | -0.028 | 작동 안함 |
| **Growth** | **+0.023** | **+0.042** | **유일하게 작동** |
| Final Score | -0.025 | +0.023 | 혼재 |

**결론:** Healthcare에서는 **Growth Factor만** 예측력 있음.

### 2.3 시가총액별 분석

| 시가총액 | IC (30일) | 샘플 수 |
|---------|----------|---------|
| Micro (<$300M) | -0.037 | 14,515 |
| Small ($300M-2B) | -0.067 | 7,768 |
| Mid ($2B-10B) | -0.069 | 3,732 |
| Large ($10B-200B) | -0.046 | 2,309 |
| **Mega (>$200B)** | **-0.158** | 279 |

**발견:** 시가총액이 클수록 IC가 악화. 대형 제약사가 가장 예측하기 어려움.

### 2.4 문제 종목 패턴

**고점수인데 대폭 하락:**
- CVAC, RIGL, HALO - 모두 **Biotechnology**
- 패턴: 높은 모멘텀/퀄리티 점수였지만 임상시험 실패

**저점수인데 급등:**
- SINT: +23,100% 수익률 (Medical Devices)
- APLM: +9,650% 수익률 (Biotechnology)
- 패턴: **바이너리 이벤트** 종목 (FDA 승인, 인수합병)

**결론:** Healthcare 점수가 실패하는 이유는 펀더멘털로 예측 불가능한 **바이너리 이벤트** 때문.

---

## STEP 3: 거래소 심층 분석

### 3.1 섹터 분포

| 섹터 | NYSE | NASDAQ | NYSE 비율 |
|------|------|--------|----------|
| Healthcare | 3,405 | **25,365** | 11.8% |
| Technology | 4,668 | 15,160 | 23.5% |
| Financial Services | 7,874 | 15,146 | 34.2% |
| Industrials | 8,680 | 9,201 | 48.5% |

**발견:** NASDAQ에 Healthcare 샘플이 NYSE의 7.5배.

### 3.2 거래소별 팩터 IC

| 팩터 | NYSE IC | NASDAQ IC | 차이 |
|------|---------|-----------|------|
| Value | -0.008 | -0.040 | +0.032 |
| Quality | +0.009 | -0.012 | +0.020 |
| Momentum | -0.068 | -0.039 | -0.028 |
| **Growth** | **+0.075** | **+0.029** | +0.046 |
| **Final Score** | **+0.019** | **-0.016** | **+0.035** |

**발견:** NYSE가 Momentum 제외 모든 팩터에서 NASDAQ보다 우수.

### 3.3 섹터 x 거래소 교차 분석

| 섹터 | NYSE IC | NASDAQ IC | 차이 |
|------|---------|-----------|------|
| **Basic Materials** | **+0.130** | -0.029 | +0.159 |
| Real Estate | +0.009 | -0.044 | +0.053 |
| Financial Services | +0.009 | -0.023 | +0.032 |
| Healthcare | +0.008 | -0.021 | +0.028 |
| Utilities | -0.007 | +0.045 | -0.051 |
| Consumer Cyclical | -0.005 | +0.060 | -0.065 |

**발견:** NYSE가 가치 중심 섹터(Basic Materials, Real Estate, Financials)에서 강세.

---

## 근본 원인 분석

### Momentum Factor IC = -0.056 (Phase 1) 원인

**근본 원인:** EM3 (1개월 수익률)과 EM4 (3개월 수익률)가 **점수를 반대로 계산**하고 있음.

| 전략 | 현재 로직 | 실제 시장 | 수정 방안 |
|------|----------|----------|----------|
| EM3 | 최근 수익률 높음 = 고점수 | 최근 수익률 높음 = **저수익** | **점수 반전** |
| EM4 | 3개월 수익률 높음 = 고점수 | 3개월 수익률 높음 = **저수익** | **점수 반전** |
| EM7 | 52주 고점 근접 = 고점수 | 52주 고점 근접 = 고수익 | 유지 |

### Healthcare IC = -0.068 (Phase 1) 원인

**근본 원인:**
1. 바이너리 이벤트 종목 (Biotech)이 55% 차지
2. Growth Factor만 Healthcare에서 유효
3. 대형주일수록 예측력 감소

### NASDAQ IC = -0.029 (Phase 1) 원인

**근본 원인:** NASDAQ Healthcare 집중 (NASDAQ 전체의 29%)

---

## 권장 수정 사항

### 우선순위 1: Momentum 점수 로직 수정 (긴급)

```python
# us_momentum_factor_v2.py - EM3, EM4 점수 반전

# 현재:
EM3_score = normalize(one_month_return)  # 수익률 높음 = 고점수

# 수정 후:
EM3_score = 100 - normalize(one_month_return)  # 수익률 높음 = 저점수
EM4_score = 100 - normalize(three_month_return)
```

**예상 효과:** Momentum IC가 -0.056에서 약 +0.05로 반전

### 우선순위 2: Healthcare 전용 모델

옵션 A: **Healthcare에 Growth 중심 점수 적용**
```python
if sector == 'HEALTHCARE':
    final_score = growth_score * 0.7 + quality_score * 0.3
```

옵션 B: **Healthcare 분리 처리**
- NYSE Healthcare만 점수화
- Biotechnology는 제외

### 우선순위 3: 거래소 필터

```python
# NYSE 종목 우선
if exchange == 'NYSE':
    priority_boost = 5  # Final score에 가산
```

---

## 수정 후 예상 지표

| 지표 | 현재 | 목표 |
|------|------|------|
| Momentum IC (30일) | -0.056 | > +0.03 |
| Healthcare IC (30일) | -0.068 | > 0.00 |
| NASDAQ IC (30일) | -0.029 | > 0.00 |
| 매수 신호 승률 | 40.2% | > 52% |
| 전체 IC (30일) | +0.011 | > +0.05 |

---

## 생성된 파일

| 파일 | 내용 |
|------|------|
| `us_phase2_momentum_strategies_ic.csv` | EM1-EM7 기간별 IC |
| `us_phase2_momentum_by_sector.csv` | 전략별 섹터별 IC |
| `us_phase2_momentum_rolling.csv` | 날짜별 IC 안정성 |
| `us_phase2_momentum_decile.csv` | 점수-수익률 관계 |
| `us_phase2_healthcare_by_industry.csv` | Healthcare 업종별 IC |
| `us_phase2_healthcare_by_factor.csv` | Healthcare 팩터별 효과 |
| `us_phase2_healthcare_by_marketcap.csv` | 시가총액별 IC |
| `us_phase2_healthcare_problem_stocks.csv` | 문제 종목 예시 |
| `us_phase2_exchange_sector_dist.csv` | 거래소별 섹터 분포 |
| `us_phase2_exchange_factor_ic.csv` | 거래소별 팩터 IC |
| `us_phase2_exchange_cross_analysis.csv` | 섹터 x 거래소 IC 매트릭스 |

---

*분석 완료: 2025-11-26 18:08:38*
*분석 데이터: 137,858건*
