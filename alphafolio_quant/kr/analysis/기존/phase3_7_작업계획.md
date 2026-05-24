# Phase 3.7 - 한국 주식 퀀트 모델 고도화 작업 계획

## 개요

Value 팩터의 음수 IC 문제를 해결하고, 종목별 가중치 시스템을 검증/개선하며, Factor Combo와 동적 가중치 시스템을 구현하는 종합 개선 프로젝트

---

## 현재 상태 진단

### Value 팩터 IC 분석 결과 (30일 기준)

| 전략 | Pearson IC | Spearman IC | 상태 | 판정 |
|------|-----------|-------------|------|------|
| V1_Low_PER | -0.060 | -0.080 | Strong Negative | 제외 |
| **V2_Magic_Formula** | -0.023 | +0.017 | Neutral | **유지** |
| **V3_Low_PBR** | +0.027 | +0.032 | Neutral/Positive | **유지** |
| **V4_High_Dividend** | -0.014 | +0.050 | Neutral | **유지** |
| V5_Low_PSR | N/A | N/A | - | 이미 비활성 |
| V6_High_ROE_Low_PER | -0.079 | -0.088 | Strong Negative | 제외 |
| V7_Low_PCR | -0.116 | -0.159 | Strong Negative | 제외 |
| V8_FCF_Yield | -0.042 | -0.059 | Strong Negative | 제외 |
| V9_Low_EV_Sales | -0.020 | -0.099 | Strong Negative | 제외 |
| V10_Low_Debt_Equity | -0.040 | -0.117 | Strong Negative | 제외 |
| V11_High_Current_Ratio | -0.124 | -0.163 | Strong Negative | 제외 |
| V12_ROE_PBR_Combined | -0.065 | -0.068 | Strong Negative | 제외 |
| **V13_Low_EV_EBITDA** | -0.029 | +0.014 | Neutral | **유지** |
| **V14_Graham_Number** | +0.007 | +0.001 | Neutral | **유지** |
| V15_PEG_Ratio | -0.022 | -0.024 | Weak Negative | 제외 |
| V16_Buffett_Indicator | -0.026 | -0.136 | Strong Negative | 제외 |
| V17_Growth_Adjusted | -0.010 | -0.062 | Strong Negative | 제외 |
| V18_EV_Sales_Growth | -0.005 | -0.036 | Weak Negative | 제외 |
| V19_Value_Momentum | -0.096 | -0.111 | Strong Negative | 제외 |
| V20_ROIC_Value | -0.017 | -0.038 | Weak Negative | 제외 |

### 종목별 가중치 검증 결과

| 가중치 방식 | IC | vs Baseline | 판정 |
|------------|---:|----------:|------|
| Uniform (기준선) | +0.0683 | - | 기준 |
| Market Type Only | +0.0877 | +0.0194 | 개선 |
| Theme Only | +0.0672 | -0.0011 | 효과 없음 |
| Combined | +0.0852 | +0.0169 | 약간 개선 |
| Original Final Score | +0.1115 | +0.0432 | 최고 |

**문제점:**
- KOSPI 가중치: IC -0.0625 악화 (잘못된 설정)
- Theme 가중치: 7개 테마에서 성과 악화

---

## 1단계: 밸류 팩터 업그레이드

### 1.1 신규 전략 설계 (V21~V26)

#### 데이터 소스 (kr_SQL_table_info.csv 기반)

| 테이블 | 주요 컬럼 | 활용 전략 |
|--------|----------|----------|
| kr_intraday_total | close, pbr, per, dividend_yield, market_cap | V21, V22 |
| kr_financial_position | thstrm_amount, account_nm, sj_div | V23, V24, V25 |
| kr_foreign_ownership | foreign_rate, foreign_rate_limit | V26 |
| kr_individual_investor_daily_trading | inst_net_value, foreign_net_value | V26 |
| kr_indicators | rsi, macd, obv, mfi | V22 |

#### V21~V26 전략 정의

| 전략 | 명칭 | 설명 | 핵심 지표 |
|------|-----|------|----------|
| **V21** | Korea_Adjusted_PBR | PBR + 외국인 수급 + 기관 수급 결합 | PBR < 0.8 AND (외국인순매수 > 0 OR 기관순매수 > 0) |
| **V22** | Quality_Dividend | 배당수익률 + 배당성장률 + ROE 결합 | 배당수익률 > 2% AND ROE > 8% AND 배당성장 > 0 |
| **V23** | Asset_Growth_Value | 자산성장률 역발상 (저성장 우선) | 총자산증가율 < 10% AND PBR < 1.5 |
| **V24** | Operating_Leverage | 영업레버리지 기반 가치 | 영업이익/매출 개선 AND 저PER |
| **V25** | Cash_Rich_Undervalued | 순현금/시총 + 저PBR | 순현금비율 > 20% AND PBR < 1.0 |
| **V26** | Smart_Money_Value | 외국인/기관 동시 순매수 + 저평가 | 외국인+기관 순매수 AND PBR < 1.2 |

#### V21 상세 설계: Korea_Adjusted_PBR

```python
"""
한국형 조정 PBR 전략
- 단순 저PBR이 아닌, 수급이 뒷받침되는 저PBR 종목 선별
- Korea Discount 현상을 수급으로 필터링하여 Value Trap 회피

점수 계산:
1. PBR 점수 (40%): PBR이 낮을수록 높은 점수
2. 외국인 수급 점수 (30%): 최근 20일 외국인 순매수 비중
3. 기관 수급 점수 (30%): 최근 20일 기관 순매수 비중

조건:
- PBR < 1.5 (시장 평균 이하)
- 외국인 또는 기관 순매수 > 0
"""
```

#### V22 상세 설계: Quality_Dividend

```python
"""
퀄리티 배당 전략
- 단순 고배당이 아닌, 지속가능한 배당 종목 선별
- ROE와 배당성장률로 배당컷 위험 회피

점수 계산:
1. 배당수익률 점수 (30%): 시장 평균 대비 배당수익률
2. ROE 점수 (35%): 자기자본이익률
3. 배당성장률 점수 (35%): 최근 3년 배당 증가율

조건:
- 배당수익률 > 2%
- ROE > 8%
- 최근 3년 배당 감소 없음
"""
```

#### V23 상세 설계: Asset_Growth_Value

```python
"""
자산성장 역발상 전략
- 자산을 급격히 늘리는 기업은 수익률이 낮은 경향 (Asset Growth Anomaly)
- 안정적으로 성장하는 저평가 기업 선별

점수 계산:
1. 자산성장률 역점수 (50%): 총자산증가율이 낮을수록 높은 점수
2. PBR 점수 (30%): 낮을수록 높은 점수
3. ROE 안정성 (20%): ROE 변동성이 낮을수록 높은 점수

데이터: kr_financial_position (BS - 자산총계)
"""
```

#### V24 상세 설계: Operating_Leverage

```python
"""
영업레버리지 가치 전략
- 매출 증가 시 영업이익이 더 크게 증가하는 구조
- 고정비 비중이 높아 턴어라운드 시 급격한 수익 개선

점수 계산:
1. 영업레버리지 (40%): (영업이익 증가율 / 매출 증가율)
2. 영업이익률 개선 (30%): 전년 대비 영업이익률 변화
3. 저PER 점수 (30%): PER이 낮을수록 높은 점수

데이터: kr_financial_position (CIS - 영업이익, 매출액)
"""
```

#### V25 상세 설계: Cash_Rich_Undervalued

```python
"""
현금부자 저평가 전략
- 순현금(현금-부채)이 시가총액 대비 높은 기업
- 청산가치가 시가총액보다 높은 극단적 저평가 종목

점수 계산:
1. 순현금비율 (50%): (현금및현금성자산 - 총부채) / 시가총액
2. PBR 점수 (30%): 낮을수록 높은 점수
3. 유동비율 (20%): 단기 지급능력

데이터: kr_financial_position (BS - 현금및현금성자산, 부채총계)
"""
```

#### V26 상세 설계: Smart_Money_Value

```python
"""
스마트머니 가치 전략
- 외국인과 기관이 동시에 매수하는 저평가 종목
- 정보우위 투자자들의 집단지성 활용

점수 계산:
1. 외국인 순매수 (35%): 최근 20일 외국인 순매수금액 / 시가총액
2. 기관 순매수 (35%): 최근 20일 기관 순매수금액 / 시가총액
3. PBR 저평가 (30%): PBR이 낮을수록 높은 점수

조건:
- 외국인 순매수 > 0 AND 기관 순매수 > 0
- PBR < 1.2

데이터: kr_individual_investor_daily_trading
"""
```

### 1.2 전략 활성화/비활성화

#### 유지할 전략 (7개)
- V2_Magic_Formula
- V3_Low_PBR (→ V3_Net_Cash_Flow_Yield로 변경됨)
- V4_High_Dividend (→ V4_Sustainable_Dividend)
- V13_Low_EV_EBITDA (→ V13_Magic_Formula)
- V14_Graham_Number (→ V14_Dividend_Growth)
- V21~V26 (신규)

#### 비활성화할 전략 (14개)
```python
# 주석 처리할 전략
V1, V5, V6, V7, V8, V9, V10, V11, V12, V15, V16, V17, V18, V19, V20
```

### 1.3 수정 파일 목록

| 파일 | 수정 내용 |
|------|----------|
| `kr/kr_value_factor.py` | V21~V26 메서드 추가, V1/V5~V12/V15~V20 주석 처리 |
| `kr/weight.py` | VALUE_STRATEGY_WEIGHTS에서 V21~V26 가중치 추가 |
| `kr/batch_weight.py` | VALUE_STRATEGY_WEIGHTS 동기화 |

---

## 2단계: 종목별 가중치 검증 및 분석

### 2.1 실험 설계

#### 실험 A: 종목별 가중치 ON vs OFF

| 설정 | 설명 |
|-----|------|
| Baseline (OFF) | 모든 종목에 동일 가중치 (0.25, 0.25, 0.25, 0.25) |
| Market Type Only | KOSPI/KOSDAQ별 가중치만 적용 |
| Theme Only | 17개 테마별 가중치만 적용 |
| Combined (ON) | Market Type + Theme 결합 |
| Full System | 현재 시스템 전체 (Market + Theme + Cap + Cycle 등) |

#### 실험 B: IC 분석

| 분석 항목 | 설명 |
|----------|------|
| 전체 IC | 전체 종목 대상 IC |
| 시장별 IC | KOSPI vs KOSDAQ |
| 테마별 IC | 17개 테마별 |
| 시가총액별 IC | MEGA/LARGE/MEDIUM/SMALL |
| 경기사이클별 IC | EXPANSION/RECOVERY/SLOWDOWN/RECESSION |

#### 실험 C: 롤링윈도우 검증

```
기간: 2025-08-04 ~ 2025-09-16 (12개 날짜)
윈도우: 5일, 10일, 20일
측정: IC 안정성, 표준편차, 최대 낙폭
```

#### 실험 D: 10분위 테스트 (Decile Test)

```
방법:
1. 각 날짜별로 Final Score 기준 10분위 분류
2. 각 분위별 평균 수익률 계산
3. Top Decile - Bottom Decile = Long-Short Spread
4. 단조성(Monotonicity) 검증
```

#### 실험 E: 파라미터 서피스 분석

```python
# 테스트할 가중치 조합
base_weights = [0.2, 0.3, 0.4, 0.5]
sector_weights = [0.2, 0.3, 0.4]
combo_weights = [0.2, 0.3, 0.4]

# 각 조합에 대해 IC, Hit Rate, Sharpe 측정
# 3D 서피스 시각화
```

### 2.2 Value 팩터 전략별 성능 검증

```
대상: V2, V3, V4, V13, V14, V21~V26 (11개 전략)
측정 지표:
- IC (Pearson, Spearman)
- Hit Rate
- Long-Short Spread
- Decile 수익률 분포
- 테마별/시가총액별 성능 차이
```

### 2.3 출력 파일

| 파일명 | 내용 |
|-------|------|
| phase3_7_weight_on_off_comparison.csv | 가중치 ON/OFF 비교 결과 |
| phase3_7_ic_analysis_by_segment.csv | 세그먼트별 IC 분석 |
| phase3_7_rolling_window_ic.csv | 롤링윈도우 IC |
| phase3_7_decile_test_results.csv | 10분위 테스트 결과 |
| phase3_7_parameter_surface.csv | 파라미터 서피스 분석 |
| phase3_7_value_strategy_performance.csv | Value 전략별 성능 |

---

## 3단계: 팩터콤보 개선

### 3.1 현재 Factor Combo 구조 분석

```
현재 IC: +0.2484 (전체 팩터 중 최고)
구성: Value + Quality + Momentum + Growth 조합 보너스
```

### 3.2 개선 방향

#### 3.2.1 Combo 조건 재설계

| 기존 조건 | 문제점 | 개선안 |
|----------|-------|-------|
| Value High | 음수 IC 전략 포함 | V2,V3,V4,V13,V14,V21~V26만 사용 |
| Growth High | 과도한 의존 | Growth 상한선 설정 |
| 단순 AND 조건 | 유연성 부족 | 점수 기반 가중 합산 |

#### 3.2.2 신규 Combo 패턴

| 패턴 | 조건 | 예상 효과 |
|------|-----|----------|
| Quality_Value_Combo | Quality ≥ 70 AND Value ≥ 60 | 안정적 저평가 |
| Growth_Momentum_Combo | Growth ≥ 70 AND Momentum ≥ 65 | 성장 모멘텀 |
| Dividend_Quality_Combo | V22 ≥ 70 AND Quality ≥ 65 | 퀄리티 배당 |
| Smart_Money_Combo | V26 ≥ 70 AND Momentum ≥ 60 | 수급 기반 |

### 3.3 수정 파일

| 파일 | 수정 내용 |
|------|----------|
| `kr/kr_additional_metrics.py` | Factor Combo 계산 로직 수정 |
| 검증 스크립트 | phase3_7_combo_validation.py |

---

## 4단계: 동적 가중치 시스템 구현

### 4.1 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                  Market Regime Classifier                │
│  (OVERHEATED / GREED / NEUTRAL / FEAR / PANIC)          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Global Factor Weight Adjuster               │
│  - Base Factor Weight (Value/Quality/Momentum/Growth)    │
│  - Sector Rotation Weight                                │
│  - Factor Combo Weight                                   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Stock-Specific Weight Layer                 │
│  (기존 batch_weight.py 로직 유지)                        │
└─────────────────────────────────────────────────────────┘
```

### 4.2 국면별 글로벌 가중치

| 국면 | Value | Quality | Momentum | Growth | Sector | Combo |
|------|-------|---------|----------|--------|--------|-------|
| PANIC | 0.35 | 0.35 | 0.10 | 0.20 | 0.15 | 0.30 |
| FEAR | 0.30 | 0.30 | 0.15 | 0.25 | 0.20 | 0.35 |
| NEUTRAL | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.40 |
| GREED | 0.20 | 0.20 | 0.30 | 0.30 | 0.30 | 0.40 |
| OVERHEATED | 0.25 | 0.25 | 0.20 | 0.30 | 0.25 | 0.35 |

### 4.3 구현 파일

| 파일 | 내용 |
|------|------|
| `kr/regime_weight_adjuster.py` | 국면별 글로벌 가중치 조정 (신규) |
| `kr/batch_weight.py` | 기존 종목별 가중치 유지 |
| `kr/kr_main.py` | 국면 가중치 적용 통합 |

### 4.4 검증 방법

```
1. 과거 데이터로 국면 분류 재현
2. 각 국면에서 가중치별 성과 비교
3. In-Sample vs Out-of-Sample 검증
4. 국면 전환 시점 성과 분석
```

---

## 작업 일정

| 단계 | 세부 작업 | 예상 작업량 |
|------|----------|-----------|
| **1단계** | V21~V26 설계 및 구현 | SQL 쿼리 설계, 메서드 구현, 가중치 설정 |
| | 기존 전략 비활성화 | 주석 처리 및 테스트 |
| **2단계** | 검증 스크립트 작성 | 5개 실험 스크립트 |
| | 분석 실행 및 리포트 | CSV 출력, 결과 해석 |
| **3단계** | Combo 로직 수정 | 조건 재설계, 신규 패턴 추가 |
| | Combo 검증 | IC 분석, 효과 측정 |
| **4단계** | 국면 분류기 구현 | regime_weight_adjuster.py |
| | 통합 테스트 | 전체 시스템 검증 |

---

## 성공 기준

| 지표 | 현재 | 목표 |
|------|-----|------|
| Value Factor IC | -0.1105 | **> 0.00** (최소 Neutral) |
| Final Score IC | +0.1115 | **> 0.12** |
| Hit Rate | 57.6% | **> 60%** |
| Long-Short Spread | 14.24% | **> 15%** |

---

## 참고 파일

### 입력 데이터
- `result test/phase3_6_analysis_data.csv`: 분석용 메인 데이터
- `result test/phase3_6_value_strategies_ic.csv`: Value 전략별 IC
- `kr_SQL_table_info.csv`: SQL 테이블/컬럼 정보

### 기존 코드
- `kr/kr_value_factor.py`: Value 팩터 계산
- `kr/batch_weight.py`: 종목별 가중치 시스템
- `kr/market_classifier.py`: 시장 국면 분류기
- `kr/weight.py`: 기본 가중치 정의

### 분석 결과
- `phase3_6 밸류팩터 분석 결과.md`: Value 팩터 문제점 분석
- `phase3_6_weight_validation_results.csv`: 가중치 검증 결과
- `phase3_6_weight_condition_contribution.csv`: 조건별 기여도
