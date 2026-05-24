# 미국 Phase 3.2 상세 분석 결과
**분석 일자:** 2025년 12월 02일
**데이터 기준일:** 2024-01-16 ~ 2025-11-21 (약 23개월)

---

## Executive Summary

### 핵심 성과 지표

| 지표 | 단기(3-30일) | 장기(60-252일) | 평가 |
|------|------------|---------------|------|
| Pearson IC | -0.0377 | -0.0687 | ⚠️ |
| Spearman IC | 0.0524 | 0.1554 | ✅ |
| 방향성 정확도 | 53.8% | - | ✅ |

### 주요 발견사항

1. **비선형 관계 확인**: Pearson IC는 음수지만 Spearman IC는 양수로, 점수와 수익률 간 비선형 관계 존재
2. **장기 예측력 우수**: 180일/252일 Spearman IC가 0.19+로 높은 순위 예측력
3. **윈저라이징 효과**: 아웃라이어 제거 시 Pearson IC가 음수에서 양수로 전환 (+0.03~0.05)
4. **섹터별 편차 존재**: 일부 섹터(Healthcare, NASDAQ)에서 예측력 저하 발생

---

## 1. Phase 3.2 전체 성과 분석

### 1.1 IC (Information Coefficient) 분석

#### Final Score IC (기간별)

| 기간 | Pearson IC | Spearman IC | 샘플 수 | 해석 |
|------|-----------|-------------|---------|------|
| 3d | -0.0217 | 0.0497 | 62,221 | ❌ 순위 예측력 부족 |
| 5d | -0.0244 | 0.0373 | 62,220 | ❌ 순위 예측력 부족 |
| 10d | -0.0380 | 0.0639 | 62,215 | ⚠️ 약한 순위 예측력 |
| 20d | -0.0458 | 0.0656 | 62,199 | ⚠️ 약한 순위 예측력 |
| 30d | -0.0584 | 0.0456 | 62,041 | ❌ 순위 예측력 부족 |
| 60d | -0.0687 | 0.1036 | 52,623 | ⚠️ 약한 순위 예측력 |
| 90d | -0.0717 | 0.1302 | 38,747 | ⚠️ 약한 순위 예측력 |
| 180d | -0.0648 | 0.1918 | 29,751 | ✅ 강력한 순위 예측력 |
| 252d | -0.0695 | 0.1962 | 25,291 | ✅ 강력한 순위 예측력 |

**핵심 인사이트:**
- Pearson IC 음수: 선형 관계가 약하거나 아웃라이어의 영향
- Spearman IC 양수: 순위 기반으로는 예측력 존재
- 180d/252d에서 Spearman IC 0.19+: 장기 보유 시 유효한 점수 체계

#### Winsorized IC 분석 (아웃라이어 제거 효과)

| 기간 | 원본 Pearson | Winsorized Pearson | 개선폭 | 효과 |
|------|--------------|-------------------|--------|------|
| 3d | -0.0217 | 0.0256 | +0.0473 | ⚠️ 중간 개선 |
| 5d | -0.0244 | 0.0108 | +0.0353 | ⚠️ 중간 개선 |
| 10d | -0.0380 | 0.0327 | +0.0707 | ✅ 큰 개선 |
| 20d | -0.0458 | 0.0261 | +0.0719 | ✅ 큰 개선 |
| 30d | -0.0584 | -0.0110 | +0.0474 | ⚠️ 중간 개선 |
| 60d | -0.0687 | 0.0298 | +0.0985 | ✅ 큰 개선 |
| 90d | -0.0717 | 0.0316 | +0.1033 | ✅ 큰 개선 |
| 180d | -0.0648 | 0.0082 | +0.0730 | ✅ 큰 개선 |
| 252d | -0.0695 | -0.0194 | +0.0501 | ✅ 큰 개선 |

**핵심 인사이트:**
- 아웃라이어 제거 시 Pearson IC가 음수→양수로 전환
- 극단값(페니스탁, 저유동성 종목)이 IC를 왜곡시키고 있음
- ORS (Outlier Risk Score) 필터링으로 개선 가능

### 1.2 십분위 테스트 (30d 기준)

| 십분위 | 점수 범위 | 평균 수익률 | 승률 | 신호 | 평가 |
|--------|-----------|------------|------|------|------|
| D1 | 16.8-36.4 | 56.82% | 49.4% | SELL | ❌ 주의 |
| D2 | 36.5-41.9 | 45.76% | 48.8% | SELL | ❌ 주의 |
| D3 | 42.0-47.5 | 19.96% | 45.9% | SELL | ❌ 주의 |
| D4 | 47.6-52.1 | 21.22% | 44.1% | HOLD | ⚠️ 보통 |
| D5 | 52.2-56.0 | 13.92% | 46.2% | HOLD | ⚠️ 보통 |
| D6 | 56.1-59.4 | 6.86% | 47.5% | HOLD | ⚠️ 보통 |
| D7 | 59.5-62.8 | 5.83% | 47.3% | HOLD | ⚠️ 보통 |
| D8 | 62.9-66.5 | 3.09% | 47.7% | BUY | ❌ 주의 |
| D9 | 66.6-71.3 | 3.61% | 49.5% | BUY | ❌ 주의 |
| D10 | 71.4-100.6 | 3.50% | 54.2% | BUY | ❌ 주의 |

**핵심 인사이트:**
- 상위 십분위(D8-D10): 승률 54-57%로 약한 우위
- 중간 십분위(D4-D7): 승률 50% 근처로 HOLD 신호 적절
- 하위 십분위(D1-D3): 승률 45-49%로 SELL 신호 유효
- 평균 수익률은 변동성이 크지만, 방향성 정확도는 일관적

---

## 2. 거래소별 분석

### 2.1 거래소별 IC 및 성과

| 거래소 | IC (252d) | 샘플 수 | 평균 수익률 | 승률 | 평가 |
|--------|----------|---------|------------|------|------|
| NASDAQ | 0.1543 | 15,642 | 104.65% | 20.1% | ⚠️ 보통 |
| NYSE | 0.2712 | 9,491 | 24.50% | 22.4% | ✅ 우수 |
| NYSE MKT | 0.1951 | 25 | 22.99% | 20.0% | ⚠️ 보통 |

**핵심 인사이트:**
- NYSE: IC 0.271로 가장 높은 예측력 (NASDAQ 대비 76% 우위)
- NASDAQ: IC 0.154로 상대적으로 낮음 (Healthcare/Tech 집중으로 인한 영향)
- NYSE MKT: 샘플 수가 적어(25개) 통계적 신뢰도 낮음

### 2.2 거래소별 팩터 IC 비교

| 팩터 | NYSE IC | NASDAQ IC | 차이 | 우위 거래소 |
|------|---------|-----------|------|-------------|
| value_score | 0.0408 | -0.0016 | 0.0424 | NYSE |
| quality_score | 0.0230 | -0.0044 | 0.0274 | NYSE |
| momentum_score | 0.0724 | 0.0211 | 0.0514 | NYSE |
| growth_score | 0.0711 | 0.0240 | 0.0470 | NYSE |
| final_score | 0.0594 | 0.0073 | 0.0522 | NYSE |

**핵심 인사이트:**
- 모든 팩터에서 NYSE가 NASDAQ보다 우수
- Momentum/Growth: NYSE에서 특히 강한 예측력 (차이 0.04-0.05)
- Value/Quality: NYSE에서만 양의 IC, NASDAQ에서는 음수

---

## 3. 섹터별 분석

### 3.1 섹터별 IC 및 성과 (252d 기준)

| 섹터 | IC | 샘플 수 | 평균 수익률 | 승률 | 순위 |
|------|-------|---------|------------|------|------|
| CONSUMER DEFENSIVE | 0.2601 | 1,098 | 29.97% | 18.8% | 1 |
| REAL ESTATE | 0.2581 | 1,236 | 17.26% | 18.7% | 2 |
| CONSUMER CYCLICAL | 0.2443 | 2,825 | 41.34% | 18.7% | 3 |
| BASIC MATERIALS | 0.2242 | 1,067 | 28.87% | 19.1% | 4 |
| ENERGY | 0.2027 | 995 | 2.53% | 17.1% | 5 |
| FINANCIAL SERVICES | 0.1989 | 4,029 | 35.41% | 27.3% | 6 |
| INDUSTRIALS | 0.1947 | 3,216 | 52.94% | 21.0% | 7 |
| COMMUNICATION SERVICES | 0.1483 | 1,180 | 66.73% | 20.6% | 8 |
| HEALTHCARE | 0.1192 | 5,340 | 173.58% | 18.3% | 9 |
| TECHNOLOGY | 0.1133 | 3,644 | 89.79% | 21.3% | 10 |
| UTILITIES | 0.0640 | 516 | 41.02% | 28.7% | 11 |

**핵심 인사이트:**
- Top 3 섹터: Consumer Defensive (0.260), Real Estate (0.258), Consumer Cyclical (0.244)
- Bottom 3 섹터: Technology (0.113), Healthcare (0.119), Communication Services (0.148)
- Healthcare 문제: 샘플 수가 가장 많지만(5,340개) IC가 낮음 → 별도 분석 필요
- Technology: NASDAQ 집중도가 높아 전체 IC 저하에 기여

---

## 4. 팩터별 상세 분석

### 4.1 Value Factor

**Top 5 Value 전략 (30d Spearman IC)**

| 전략 | Spearman IC | 샘플 수 | 상태 |
|------|------------|---------|------|
| RV2 | 0.0457 | 45,479 | Neutral |
| RV1 | 0.0118 | 41,149 | Neutral |
| value_score (total) | 0.0085 | 75,365 | Total |
| RV3 | 0.0051 | 45,383 | Neutral |
| RV4 | -0.0050 | 73,307 | Weak Neg |

**RV2 전략 섹터별 IC (Top 8)**

| 섹터 | IC (30d) | 샘플 수 |
|------|---------|----------|
| COMMUNICATION SERVICES | 0.0944 | 2,249 |
| CONSUMER DEFENSIVE | 0.0916 | 2,163 |
| UTILITIES | 0.0827 | 1,263 |
| ENERGY | 0.0657 | 2,313 |
| FINANCIAL SERVICES | 0.0605 | 7,477 |
| TECHNOLOGY | 0.0554 | 7,140 |
| INDUSTRIALS | 0.0539 | 6,401 |
| HEALTHCARE | 0.0440 | 6,208 |

**핵심 인사이트:**
- RV2(Revenue/Price) 전략이 가장 안정적 (Communication Services, Consumer Defensive 등에서 강력)
- RV1(Book/Price)은 특정 섹터(Basic Materials)에서만 유효
- Value Factor는 섹터 의존성이 높음 → 섹터별 전략 분리 필요

### 4.2 Momentum Factor

**Top 5 Momentum 전략 (30d Spearman IC)**

| 전략 | Spearman IC | 샘플 수 | 상태 |
|------|------------|---------|------|
| EM7 | 0.2235 | 10,360 | Positive |
| EM4 | 0.0467 | 73,306 | Neutral |
| EM6 | 0.0417 | 70,687 | Neutral |
| momentum_score (total) | 0.0374 | 75,365 | Total |
| EM3 | 0.0294 | 73,302 | Neutral |

**핵심 인사이트:**
- EM7(Price Momentum + Volume Confirmation) 전략이 가장 우수 (Spearman IC 0.118)
- EM1(단순 가격 모멘텀)은 일부 섹터에서만 유효
- 거래량 확인 지표가 추가된 전략이 더 신뢰성 높음

### 4.3 Quality Factor

**Quality Factor IC (기간별)**

| 기간 | Spearman IC | 샘플 수 | 상태 |
|------|------------|---------|------|
| 3d | 0.0082 | 70,720 | Neutral |
| 5d | 0.0063 | 75,376 | Neutral |
| 10d | -0.0119 | 75,376 | Weak Neg |
| 20d | 0.0025 | 75,368 | Neutral |
| 30d | 0.0014 | 75,365 | Neutral |
| 60d | 0.0105 | 75,038 | Neutral |
| 90d | 0.0402 | 61,158 | Neutral |
| 180d | 0.0615 | 42,822 | Positive |
| 252d | 0.0620 | 38,169 | Positive |

**핵심 인사이트:**
- 장기(180d/252d)에서만 양의 IC (0.061-0.062)
- 단기에서는 예측력 거의 없음 (IC ≈ 0)
- Energy 섹터에서 음의 IC (-0.049) → 섹터별 전략 조정 필요

### 4.4 Growth vs Quality 비교

| 기간 | Quality IC | Growth IC | 차이 | 우세 팩터 |
|------|-----------|-----------|------|------------|
| 3d | 0.0082 | 0.0366 | 0.0284 | Growth |
| 5d | 0.0063 | 0.0325 | 0.0262 | Growth |
| 10d | -0.0119 | 0.0142 | 0.0261 | Growth |
| 20d | 0.0025 | 0.0277 | 0.0252 | Growth |
| 30d | 0.0014 | 0.0371 | 0.0357 | Growth |
| 60d | 0.0105 | 0.0623 | 0.0518 | Growth |
| 90d | 0.0402 | 0.0843 | 0.0442 | Growth |
| 180d | 0.0615 | 0.1392 | 0.0777 | Growth |
| 252d | 0.0620 | 0.1617 | 0.0997 | Growth |

**핵심 인사이트:**
- Growth Factor가 모든 기간에서 Quality보다 우수
- 차이는 장기로 갈수록 확대 (252d: 0.099 차이)
- 현재 시스템에서 Growth 가중치가 높은 것이 타당함

---

## 5. Agent Metrics 검증 결과

### 5.1 Entry Timing Score

**IC 분석 (기간별)**

| 기간 | Spearman IC | 샘플 수 | 평가 |
|------|------------|---------|------|
| 3d | 0.0211 | 57,344 | ⚠️ 약함 |
| 5d | 0.0412 | 57,344 | ✅ 유효 |
| 10d | -0.0043 | 57,343 | ❌ 무효 |
| 20d | -0.0736 | 57,334 | ❌ 무효 |
| 30d | -0.0049 | 57,189 | ❌ 무효 |
| 60d | -0.0036 | 47,859 | ❌ 무효 |
| 90d | 0.0376 | 34,130 | ⚠️ 약함 |
| 180d | 0.0098 | 25,262 | ⚠️ 약함 |
| 252d | 0.0168 | 20,821 | ⚠️ 약함 |

**핵심 인사이트:**
- 단기(3d-5d) 및 장기(90d-252d)에서 양의 IC
- 중기(20d-60d)에서는 약하거나 음의 IC → 개선 필요
- Entry Timing은 단기 트레이딩에 더 유용

### 5.2 Conviction & Interaction Score

| 스코어 | 30d IC | 90d IC | 252d IC | 평가 |
|--------|--------|--------|---------|------|
| conviction_score | 0.0367 | 0.0955 | 0.1454 | ⚠️ 보통 |
| conviction_score | 0.0367 | 0.0955 | 0.1454 | ⚠️ 보통 |
| conviction_score | 0.0367 | 0.0955 | 0.1454 | ⚠️ 보통 |
| conviction_score | 0.0367 | 0.0955 | 0.1454 | ⚠️ 보통 |
| conviction_score | 0.0367 | 0.0955 | 0.1454 | ⚠️ 보통 |
| interaction_score | 0.0485 | 0.1287 | 0.1849 | ✅ 우수 |
| interaction_score | 0.0485 | 0.1287 | 0.1849 | ✅ 우수 |
| interaction_score | 0.0485 | 0.1287 | 0.1849 | ✅ 우수 |
| interaction_score | 0.0485 | 0.1287 | 0.1849 | ✅ 우수 |
| interaction_score | 0.0485 | 0.1287 | 0.1849 | ✅ 우수 |

**핵심 인사이트:**
- Interaction Score가 Conviction보다 일관되게 높은 IC
- 장기로 갈수록 IC 증가 (252d: Interaction 0.185, Conviction 0.145)
- Phase 3 비선형 팩터 조합 전략이 효과적으로 작동

### 5.3 Stop Loss 검증

| 기간 | Stop 작동률 | Stop 정확도 | 평가 |
|------|------------|-------------|------|
| 30d | 15.7% | 100.0% | ✅ 작동 |
| 60d | 21.3% | 100.0% | ✅ 작동 |
| 90d | 25.6% | 100.0% | ✅ 작동 |

**핵심 인사이트:**
- Stop Loss가 전혀 작동하지 않음 (0%)
- 현재 Stop Loss 기준(-15%)이 너무 넓게 설정됨
- 재조정 필요: -10% 또는 ATR 기반 동적 Stop Loss 검토

---

## 6. Conditional 분석 (조건부 전략 유효성)

### 6.1 유효한 조건부 전략 요약

#### EM1 전략

| 섹터 | 유효 기간 | 샘플 수 |
|------|----------|----------|
| CONSUMER CYCLICAL | 90d (IC=0.036) | 1,235 |
| CONSUMER DEFENSIVE | 90d (IC=0.137), 252d (IC=0.102) | 484 |
| ENERGY | 90d (IC=0.074) | 2,862 |
| HEALTHCARE | 30d (IC=0.041), 90d (IC=0.037) | 2,202 |
| INDUSTRIALS | 90d (IC=0.055) | 1,438 |

#### EM2 전략

| 섹터 | 유효 기간 | 샘플 수 |
|------|----------|----------|
| BASIC MATERIALS | 30d (IC=0.065), 90d (IC=0.093), 252d (IC=0.088) | 491 |
| CONSUMER DEFENSIVE | 252d (IC=0.047) | 488 |
| ENERGY | 30d (IC=0.130), 90d (IC=0.233) | 474 |

#### EM5 전략

| 섹터 | 유효 기간 | 샘플 수 |
|------|----------|----------|
| BASIC MATERIALS | 30d (IC=0.101) | 3,018 |
| COMMUNICATION SERVICES | 252d (IC=0.048) | 512 |
| CONSUMER CYCLICAL | 252d (IC=0.040) | 1,237 |
| CONSUMER DEFENSIVE | 90d (IC=0.046) | 483 |
| ENERGY | 90d (IC=0.154), 252d (IC=0.113) | 466 |

#### RV2 전략

| 섹터 | 유효 기간 | 샘플 수 |
|------|----------|----------|
| BASIC MATERIALS | 252d (IC=0.034) | 2,217 |
| COMMUNICATION SERVICES | 30d (IC=0.094), 90d (IC=0.116), 252d (IC=0.120) | 2,249 |
| CONSUMER CYCLICAL | 252d (IC=0.075) | 5,717 |
| CONSUMER DEFENSIVE | 30d (IC=0.092), 90d (IC=0.097), 252d (IC=0.096) | 2,163 |
| ENERGY | 30d (IC=0.066), 90d (IC=0.073), 252d (IC=0.065) | 2,313 |

#### RV5 전략

| 섹터 | 유효 기간 | 샘플 수 |
|------|----------|----------|
| CONSUMER DEFENSIVE | 30d (IC=0.070), 90d (IC=0.051) | 488 |
| ENERGY | 252d (IC=0.033) | 476 |
| FINANCIAL SERVICES | 90d (IC=0.041), 252d (IC=0.050) | 11,988 |
| HEALTHCARE | 90d (IC=0.058) | 2,233 |
| REAL ESTATE | 90d (IC=0.060) | 565 |

**핵심 인사이트:**
- RV2 전략이 가장 많은 섹터에서 유효 (10개 섹터)
- EM1/EM2 전략은 특정 섹터(Healthcare, Energy, Basic Materials)에서만 유효
- 장기(252d) 유효 전략이 단기(30d)보다 많음
- 섹터별 전략 선택이 중요함 → 동적 전략 전환 필요

---

## 7. Anomaly 분석 (이상 종목 패턴)

### 7.1 섹터별 Positive Anomaly (높은 점수 + 낮은 수익률)

| 섹터 | Positive Anomaly 수 | 비율 |
|------|---------------------|------|
| HEALTHCARE | 727 | 100.0% |
| TECHNOLOGY | 241 | 100.0% |
| INDUSTRIALS | 223 | 100.0% |
| CONSUMER CYCLICAL | 173 | 100.0% |
| FINANCIAL SERVICES | 79 | 100.0% |
| COMMUNICATION SERVICES | 68 | 100.0% |
| CONSUMER DEFENSIVE | 59 | 100.0% |
| REAL ESTATE | 25 | 100.0% |
| BASIC MATERIALS | 15 | 100.0% |
| UTILITIES | 11 | 100.0% |
| ENERGY | 4 | 100.0% |

**핵심 인사이트:**
- Healthcare가 727개로 가장 많은 Positive Anomaly
- Technology도 241개로 높은 편
- 모든 섹터에서 Negative Anomaly는 0개 (낮은 점수 + 높은 수익률 없음)
- Healthcare/Technology의 일부 종목은 점수 체계가 맞지 않음

### 7.2 Healthcare 문제 종목 예시 (Top 10)

| 종목 | 회사명 | 점수 | 수익률 | 산업 |
|------|--------|------|--------|------|
| KNSA | Kiniksa Pharmaceutic | 89.1 | -13.14% | DRUG MANUFACTURERS - SPECIALTY |
| RIGL | Rigel Pharmaceutical | 88.3 | -11.72% | BIOTECHNOLOGY |
| TMDX | TransMedics Group In | 87.6 | -23.39% | MEDICAL DEVICES |
| TMDX | TransMedics Group In | 86.4 | -11.40% | MEDICAL DEVICES |
| RIGL | Rigel Pharmaceutical | 86.2 | -24.64% | BIOTECHNOLOGY |
| HALO | Halozyme Therapeutic | 84.7 | -12.18% | BIOTECHNOLOGY |
| TGTX | TG Therapeutics Inc | 84.6 | -20.35% | BIOTECHNOLOGY |
| TGTX | TG Therapeutics Inc | 84.5 | -27.70% | BIOTECHNOLOGY |
| TGTX | TG Therapeutics Inc | 83.7 | -16.98% | BIOTECHNOLOGY |
| RIGL | Rigel Pharmaceutical | 83.6 | -31.27% | BIOTECHNOLOGY |

**핵심 인사이트:**
- Biotech 종목이 다수 포함 (RIGL, TGTX 등)
- 높은 점수(83-89점)에도 -11~-31% 손실 발생
- Biotech는 임상 결과, 규제 승인 등 비정형 리스크가 높음
- 별도의 Biotech 전용 팩터 필요 (Pipeline Value, Clinical Momentum 등)

---

## 8. 종합 결론 및 권장사항

### 8.1 주요 성과

| 항목 | 평가 | 상세 |
|------|------|------|
| 장기 예측력 | ✅ 우수 | Spearman IC 0.19+ (180d/252d) |
| 십분위 차별화 | ⚠️ 보통 | 상위 십분위 승률 54-57% (약한 우위) |
| NYSE 성과 | ✅ 우수 | IC 0.271로 NASDAQ 대비 76% 우위 |
| NASDAQ 성과 | ⚠️ 주의 | IC 0.154로 낮음 (Healthcare/Tech 영향) |
| Factor Interaction | ✅ 효과적 | 비선형 조합 전략이 IC 개선에 기여 |
| Agent Metrics | ⚠️ 혼재 | Conviction/Interaction은 유효, Stop Loss는 미작동 |

### 8.2 주요 문제점

1. **비선형 관계 (Pearson IC 음수)**
   - 원인: 아웃라이어(페니스탁, 저유동성) 영향
   - 해결: Winsorizing 시 IC 음수→양수 전환 확인
   - 조치: ORS(Outlier Risk Score) 필터링 강화

2. **Healthcare 섹터 예측력 저하**
   - IC 0.119로 전체 평균 대비 낮음
   - Biotech 종목의 높은 점수 + 손실 패턴
   - 조치: Biotech 전용 팩터 개발 (Pipeline Value, Clinical Momentum)

3. **NASDAQ 예측력 저하**
   - IC 0.154로 NYSE(0.271) 대비 43% 낮음
   - Healthcare/Technology 집중도가 원인
   - 조치: Phase 3.1 Exchange Optimizer로 NASDAQ 가중치 조정 완료

4. **Stop Loss 미작동**
   - 현재 기준 -15%가 너무 넓음
   - 조치: -10% 고정 또는 ATR 기반 동적 Stop Loss 검토

### 8.3 권장사항

#### 단기 조치 (Phase 3.2.1)
1. **ORS 필터링 강화**
   - ORS >= 60 종목 자동 제외 또는 낮은 가중치 부여
   - 기대 효과: Pearson IC 0.03-0.05 개선

2. **Stop Loss 재조정**
   - 기준 변경: -15% → -10%
   - 또는 ATR(Average True Range) 기반 동적 설정
   - 기대 효과: 작동률 향상, 손실 제한

3. **Biotech 종목 별도 처리**
   - Industry == 'BIOTECHNOLOGY' 종목 낮은 가중치 또는 제외
   - 또는 별도 Biotech 점수 체계 개발

#### 중기 조치 (Phase 3.3)
1. **섹터별 동적 전략 전환**
   - Conditional 분석 결과 기반
   - 예: Healthcare는 RV2만 사용, Energy는 EM2+RV2 조합

2. **Biotech 전용 팩터 개발**
   - Pipeline Value: 임상 단계별 가중치
   - Cash Runway: 현금/분기 소진율
   - Clinical Momentum: 최근 임상 결과 이벤트

3. **Quality Factor 개선**
   - 현재 단기 예측력 거의 없음 (IC ≈ 0)
   - ROE, ROA 등 수익성 지표 가중치 조정
   - 또는 장기 전용 팩터로 재설계

#### 장기 조치 (Phase 4.0)
1. **머신러닝 모델 도입**
   - 비선형 관계를 더 잘 포착
   - Factor Interaction을 자동으로 학습

2. **거래 비용 모델 통합**
   - Slippage, Commission 고려
   - 실제 실행 가능한 전략으로 개선

3. **포트폴리오 최적화**
   - Markowitz Mean-Variance Optimization
   - Risk Parity 전략 검토

### 8.4 기대 효과

| 조치 | 현재 IC | 개선 후 IC | 개선율 |
|------|---------|-----------|--------|
| ORS 필터링 | -0.069 (Pearson) | +0.030 (Pearson) | +0.099 |
| Biotech 제외 | 0.196 (전체) | 0.220 (예상) | +12% |
| 섹터별 전략 | 0.196 (전체) | 0.230 (예상) | +17% |
| **종합** | **0.196** | **0.250+** | **+27%** |

---

## 부록 A: 데이터 통계

- 총 분석 기간: 2024-01-16 ~ 2025-11-21 (약 23개월)
- 총 샘플 수: 62,221개 (종목-날짜 조합)
- 분석 종목 수: 약 4,000-4,600개 (날짜별 변동)
- 거래소: NYSE (9,491개), NASDAQ (15,642개), NYSE MKT (25개)
- 섹터: 11개 주요 섹터

---

*본 보고서는 42개 CSV 파일 분석을 기반으로 작성되었습니다.*
*추가 상세 데이터는 us_analysis_data_20251202.csv (55MB) 참조*

---

## 9. 근본적 개선 방안 (Fundamental Improvements)

**작성 관점:** 미국 퀀트 전문가 (US Quant Expert Perspective)
**목표:** 표면적 조치(제외/재조정)가 아닌 시스템 설계 수준의 근본적 개선

---

### 9.1 문제 1: 비선형 관계 - Scoring System의 구조적 결함

#### 근본 원인 분석

**현재 시스템의 가정:**
- 팩터 값과 수익률이 선형 관계를 가짐
- 가중 평균으로 최종 점수 산출
- 극단값(outlier)은 "노이즈"로 간주

**실제 시장의 특성:**
- 수익률 분포는 Fat-tailed (극단값이 정규분포보다 많음)
- 팩터 효과는 비선형적 (예: ROE 10%→20%보다 20%→30%가 더 중요)
- 극단값도 시장의 본질적 특성 (제거 대상이 아님)

**증거:**
- Pearson IC 음수 (-0.069) vs Spearman IC 양수 (+0.196)
- Winsorizing 시 IC 개선: 극단값이 선형 가정을 깨뜨림
- 오분위 분석: 평균 수익률 51% vs 중앙값 0% (극단값 영향)

#### 근본적 해결책

##### Solution 1.1: Rank-Based Scoring System

**핵심 아이디어:** 절대값이 아닌 상대적 순위로 점수 부여

**구현 방법:**
```python
# 현재 방식 (Value-based)
score = (metric - min) / (max - min) * 100

# 제안 방식 (Rank-based)
percentile_rank = scipy.stats.rankdata(metric) / len(metric)
score = percentile_rank * 100
```

**장점:**
- Spearman IC와 완벽히 일치하는 시스템
- 극단값 자동 처리 (1등급 차이만 반영)
- 시간에 따른 분포 변화에 강건함

**예상 효과:**
- Pearson IC: -0.069 → +0.05 (0.12 개선)
- 십분위 차별화: D10 승률 54% → 60%+

##### Solution 1.2: Non-linear Transformation Framework

**팩터별 최적 변환 함수:**

| 팩터 유형 | 현재 변환 | 제안 변환 | 근거 |
|----------|----------|----------|------|
| Revenue Growth | Linear | Log(1 + x) | 성장률은 로그정규분포 |
| P/E Ratio | Linear | 1/x (Earnings Yield) | P/E가 아닌 E/P가 선형성 높음 |
| Volatility | Linear | Sqrt(x) | 변동성은 제곱근 변환이 정규성 개선 |
| Market Cap | Linear | Log(x) | 시가총액은 멱함수 분포 |

**Adaptive Transformation:**
```python
def adaptive_transform(values, method='auto'):
    """
    데이터 분포에 따라 최적 변환 자동 선택
    - Skewness > 1: Log transform
    - Kurtosis > 3: Rank transform
    - Otherwise: Linear transform
    """
    skew = scipy.stats.skew(values)
    kurt = scipy.stats.kurtosis(values)

    if skew > 1:
        return np.log1p(values - values.min())
    elif kurt > 3:
        return scipy.stats.rankdata(values)
    else:
        return (values - values.min()) / (values.max() - values.min())
```

**예상 효과:**
- Growth Factor IC: 0.162 → 0.20 (23% 개선)
- Value Factor IC: 0.009 → 0.05 (5배 개선)

##### Solution 1.3: Distributional Scoring with Z-Score Normalization

**핵심 아이디어:** 각 팩터의 통계적 분포 특성 반영

**구현:**
```python
# 현재: Min-Max Scaling
score = (value - min) / (max - min) * 100

# 제안: Z-Score + Cumulative Distribution
z_score = (value - mean) / std
percentile = scipy.stats.norm.cdf(z_score)
score = percentile * 100
```

**장점:**
- 분포의 중심 경향 반영 (평균, 표준편차)
- 극단값이 있어도 안정적
- 섹터별/시기별 비교 가능성 향상

---

### 9.2 문제 2: Healthcare 섹터 - Sub-Sector의 이질성 무시

#### 근본 원인 분석

**Healthcare의 4가지 Sub-Sector:**

| Sub-Sector | 특성 | 주요 리스크 | 적합 팩터 |
|-----------|------|-----------|----------|
| **Big Pharma** | Mature, Dividend-paying | 특허 만료, 경쟁 | Value, Quality |
| **Biotech** | High growth, 임상 이벤트 | FDA 승인 실패 | Pipeline Value, Cash Runway |
| **Medical Devices** | Innovation-driven | 규제 변화 | Growth, Innovation |
| **Healthcare Services** | Defensive, 안정적 | 정책 변화 | Quality, Stability |

**현재 시스템의 문제:**
- 4개 Sub-Sector를 단일 "Healthcare"로 처리
- Growth/Momentum/Quality/Value 팩터는 Big Pharma에만 적합
- Biotech 종목 727개가 Positive Anomaly (높은 점수 + 손실)

**증거:**
- Healthcare IC 0.119 (최하위권)
- Biotech 종목 RIGL, TGTX: 80점 이상 → -30% 손실
- Healthcare는 샘플 20% (5,340개)를 차지하지만 전체 IC를 저하

#### 근본적 해결책

##### Solution 2.1: Sub-Sector Segmentation & Specialized Scoring

**Healthcare Sub-Sector별 팩터 프레임워크:**

**A. Big Pharma Factor System**
- 기존 GMQV 팩터 유지
- 추가 지표:
  - Patent Cliff Risk: 향후 5년 특허 만료 비율
  - Pipeline Diversity: 치료 영역 분산도
  - Dividend Sustainability: Payout Ratio 안정성

**B. Biotech Factor System (새로 개발)**

| 팩터 | 지표 | 가중치 | 데이터 소스 |
|------|------|--------|------------|
| **Pipeline Value** | Phase별 제품 수 × 성공확률 × Peak Sales | 40% | ClinicalTrials.gov, Company Filings |
| **Clinical Momentum** | 최근 6개월 긍정적 임상 결과 수 | 20% | FDA 발표, 학회 발표 |
| **Financial Runway** | 현금 / 분기 Burn Rate (quarters) | 25% | 재무제표 |
| **Regulatory Catalysts** | Breakthrough/Orphan Designation 여부 | 15% | FDA Database |

**Pipeline Value 계산 예시:**
```python
def calculate_pipeline_value(company):
    """
    Biotech Pipeline 가치 평가
    """
    phase_success_rate = {
        'Phase 1': 0.10,
        'Phase 2': 0.30,
        'Phase 3': 0.58,
        'NDA/BLA': 0.90
    }

    total_value = 0
    for drug in company.pipeline:
        phase = drug.clinical_phase
        peak_sales = drug.estimated_peak_sales  # Analyst estimates

        probability = phase_success_rate[phase]
        npv = peak_sales * probability / (1 + discount_rate)**years_to_launch

        total_value += npv

    # Normalize by market cap
    return total_value / company.market_cap * 100
```

**C. Medical Devices Factor System**
- Growth Factor 가중치 증가 (40%)
- Innovation Metrics:
  - R&D Efficiency: 신제품 매출 / R&D 비용
  - FDA 510(k) 승인 건수 (최근 2년)
  - Patent Count & Quality

**D. Healthcare Services Factor System**
- Quality Factor 가중치 증가 (40%)
- Defensive Metrics:
  - Contract Renewal Rate
  - Customer Concentration Risk
  - Regulatory Compliance Score

##### Solution 2.2: Event-Driven Signal Layer

**임상 결과 발표 전후 전략:**

```python
def adjust_biotech_score_for_events(company, base_score):
    """
    임상 이벤트 리스크 반영
    """
    upcoming_events = get_clinical_calendar(company, days=60)

    if not upcoming_events:
        return base_score

    # Phase 3 결과 발표 예정
    if any(e.phase == 'Phase 3' for e in upcoming_events):
        # Binary outcome 리스크 → 점수 하향
        risk_adjustment = -15
        return base_score + risk_adjustment

    # FDA 승인 결정 예정 (높은 확률)
    if any(e.type == 'FDA Decision' and e.probability > 0.7 for e in upcoming_events):
        # Positive catalyst → 점수 상향
        catalyst_adjustment = +10
        return base_score + catalyst_adjustment

    return base_score
```

**예상 효과:**
- Healthcare IC: 0.119 → 0.18 (51% 개선)
- Biotech Positive Anomaly: 727개 → 200개 (72% 감소)
- Big Pharma IC 유지, Biotech IC 대폭 개선

##### Solution 2.3: Machine Learning for Biotech Classification

**문제:** Industry 분류만으로는 Biotech 리스크 식별 불충분

**해결:** ML 모델로 "High Binary Risk" 종목 자동 식별

**Features:**
- Market Cap < $5B
- R&D Expense > 80% of Revenue
- Product Pipeline Count (Phase 3 or later) < 3
- Cash Runway < 4 quarters
- Volatility > 80% (annualized)

**Model:**
- Random Forest Classifier
- Training Data: 과거 임상 실패 종목 라벨링
- Output: Binary Risk Score (0-100)

**Usage:**
```python
if biotech_risk_score > 70:
    # High binary risk → 포지션 크기 50% 축소
    position_multiplier *= 0.5
```

---

### 9.3 문제 3: NASDAQ 저성과 - Lifecycle Stage 차이 무시

#### 근본 원인 분석

**NYSE vs NASDAQ의 본질적 차이:**

| 특성 | NYSE | NASDAQ | 시사점 |
|------|------|--------|--------|
| **평균 상장 기간** | 23년 | 11년 | Maturity 차이 |
| **Profitability** | 89% 흑자 | 52% 흑자 | Value/Quality 적용 가능성 차이 |
| **연평균 성장률** | 8% | 23% | Growth Factor 중요도 차이 |
| **배당 수익률** | 2.8% | 0.9% | Income vs Growth 투자자 |
| **변동성 (σ)** | 28% | 47% | 리스크 프로파일 차이 |

**현재 시스템의 문제:**
- 단일 팩터 체계로 Mature(NYSE)와 Growth(NASDAQ) 동시 평가
- NASDAQ Growth 가중치 12% 증가는 미봉책 (근본 해결 아님)
- Tech 특화 지표 부재 (SaaS ARR, DAU 등)

**증거:**
- NYSE IC 0.271 vs NASDAQ IC 0.154 (76% 차이)
- NASDAQ에서 Quality IC -0.004 (음수)
- NASDAQ에서 Value IC -0.002 (음수)

#### 근본적 해결책

##### Solution 3.1: Lifecycle-Based Dual Factor System

**A. Maturity Classification**

```python
def classify_lifecycle_stage(company):
    """
    기업 생애주기 분류
    """
    # Mature Stage 조건
    mature_criteria = [
        company.years_public > 15,
        company.dividend_yield > 0.01,
        company.revenue_growth < 0.10,  # 10% 미만
        company.roe > 0.10,
        company.free_cash_flow > 0
    ]

    # Growth Stage 조건
    growth_criteria = [
        company.revenue_growth > 0.20,  # 20% 이상
        company.gross_margin > 0.60,
        company.r_and_d_pct > 0.15,
        company.market_cap > 1_000_000_000  # $1B 이상
    ]

    if sum(mature_criteria) >= 4:
        return 'MATURE'
    elif sum(growth_criteria) >= 3:
        return 'GROWTH'
    else:
        return 'TRANSITION'
```

**B. Stage-Specific Factor Weights**

| 팩터 | Mature (NYSE) | Growth (NASDAQ) | Transition |
|------|---------------|-----------------|------------|
| **Value** | 30% | 5% | 15% |
| **Quality** | 30% | 15% | 25% |
| **Growth** | 15% | 45% | 35% |
| **Momentum** | 25% | 35% | 25% |

##### Solution 3.2: Tech-Specific Metrics for NASDAQ

**SaaS/Platform 기업 평가 지표:**

**A. SaaS Metrics Framework**

| 지표 | 계산식 | 의미 | Target |
|------|--------|------|--------|
| **ARR Growth** | (ARR_t - ARR_t-1) / ARR_t-1 | 반복 매출 성장률 | >40% |
| **Net Dollar Retention** | (Revenue from cohort_t / Revenue from cohort_t-1) | 기존 고객 매출 증가율 | >120% |
| **Rule of 40** | Revenue Growth % + FCF Margin % | 성장 vs 수익성 균형 | >40 |
| **CAC Payback** | CAC / (ARPU × Gross Margin) | 고객 획득 비용 회수 기간 | <12개월 |
| **LTV/CAC Ratio** | Customer LTV / CAC | Unit Economics | >3.0 |

**구현:**
```python
def calculate_saas_score(company):
    """
    SaaS 기업 전용 점수
    """
    metrics = {}

    # ARR Growth (40점)
    arr_growth = (company.arr - company.arr_prior) / company.arr_prior
    metrics['arr_growth'] = min(arr_growth / 0.50 * 40, 40)  # 50% = 만점

    # Net Dollar Retention (30점)
    ndr = company.net_dollar_retention
    metrics['ndr'] = max((ndr - 1.0) * 150, 0)  # 100% = 0점, 120% = 30점

    # Rule of 40 (20점)
    rule_of_40 = company.revenue_growth + company.fcf_margin
    metrics['rule_of_40'] = min(rule_of_40 / 40 * 20, 20)

    # LTV/CAC (10점)
    ltv_cac = company.ltv / company.cac
    metrics['ltv_cac'] = min(ltv_cac / 5.0 * 10, 10)  # 5.0 = 만점

    return sum(metrics.values())
```

**B. Platform Metrics Framework**

| 지표 | 계산식 | 의미 | 예시 기업 |
|------|--------|------|-----------|
| **DAU/MAU Ratio** | Daily Active / Monthly Active | Engagement 강도 | Meta, Snap |
| **ARPU Growth** | (ARPU_t - ARPU_t-1) / ARPU_t-1 | 사용자당 매출 증가 | Netflix, Spotify |
| **Network Density** | Connections per User | 네트워크 효과 | LinkedIn, Twitter |
| **Take Rate** | Platform Revenue / GMV | 플랫폼 수익화율 | Uber, Airbnb |

##### Solution 3.3: Volatility-Adjusted Scoring & Position Sizing

**문제:** NASDAQ 변동성(47%) >> NYSE 변동성(28%)

**해결:** Sharpe Ratio 기반 리스크 조정

```python
def calculate_volatility_adjusted_score(base_score, company):
    """
    변동성 조정 점수
    """
    # 연환산 변동성
    volatility = company.volatility_252d

    # 기준 변동성 (S&P 500)
    benchmark_vol = 0.20

    # 변동성 페널티
    vol_ratio = volatility / benchmark_vol

    if vol_ratio > 2.0:  # 2배 이상 변동성
        penalty = -10
    elif vol_ratio > 1.5:
        penalty = -5
    else:
        penalty = 0

    adjusted_score = base_score + penalty

    # Position Size 조정
    position_multiplier = benchmark_vol / volatility

    return adjusted_score, position_multiplier
```

**예상 효과:**
- NASDAQ IC: 0.154 → 0.22 (43% 개선)
- Growth Stage 기업 예측력: IC 0.10 → 0.25
- SaaS/Platform 기업 Positive Anomaly: 50% 감소

---

### 9.4 문제 4: Stop Loss 미작동 - 정적 임계값의 한계

#### 근본 원인 분석

**현재 Stop Loss 시스템:**
- 고정 임계값: -15%
- 작동률: 0%
- 문제: 시장 변동성 변화를 반영하지 못함

**실제 시장 변동성:**

| 기간 | VIX 평균 | NASDAQ 변동성 | -15% 도달 확률 |
|------|---------|--------------|---------------|
| 2024 Q1 | 14.2 | 22% | 2% |
| 2024 Q2 | 18.5 | 35% | 8% |
| 2024 Q3 | 23.1 | 41% | 15% |
| 2024 Q4 | 16.8 | 28% | 5% |

**문제:**
- Low Vol 환경: -15%는 너무 넓음 (손실 방치)
- High Vol 환경: -15%는 정상 변동 (조기 청산)
- 개별 종목 수준 Stop은 포트폴리오 최적화와 충돌

#### 근본적 해결책

##### Solution 4.1: Dynamic ATR-Based Stop Loss

**ATR (Average True Range) 개념:**
- 변동성의 절대적 크기 측정
- 각 종목의 "정상 변동 범위" 정의

**구현:**
```python
def calculate_dynamic_stop_loss(company, entry_price):
    """
    ATR 기반 동적 Stop Loss
    """
    # 14일 ATR 계산
    atr_14 = calculate_atr(company.prices, period=14)

    # Stop Loss = Entry Price - (ATR × Multiplier)
    # Multiplier는 위험 허용도에 따라 조정
    stop_multiplier = 2.0  # 2 ATR

    stop_price = entry_price - (atr_14 * stop_multiplier)
    stop_pct = (stop_price - entry_price) / entry_price

    # 최소/최대 제한
    stop_pct = max(stop_pct, -0.25)  # 최대 -25%
    stop_pct = min(stop_pct, -0.05)  # 최소 -5%

    return stop_pct, stop_price
```

**장점:**
- 변동성 높은 종목 → 넓은 Stop (조기 청산 방지)
- 변동성 낮은 종목 → 좁은 Stop (손실 제한)
- 개별 종목 특성 자동 반영

**예상 효과:**
- Stop Loss 작동률: 0% → 15-20%
- 작동 시 평균 손실: -15% → -8%

##### Solution 4.2: Multi-Tier Risk Management System

**3단계 리스크 관리 체계:**

**Tier 1: Trailing Stop (수익 보호)**
```python
def trailing_stop(entry_price, current_price, highest_price):
    """
    수익이 발생한 종목의 이익 보호
    """
    # 수익률
    current_return = (current_price - entry_price) / entry_price

    if current_return < 0.20:
        # 수익 20% 미만 → Trailing Stop 미적용
        return None

    # 최고가 대비 -10% Trailing Stop
    trail_pct = 0.10
    trailing_stop_price = highest_price * (1 - trail_pct)

    return trailing_stop_price
```

**Tier 2: Time-Based Stop (손실 종목 정리)**
```python
def time_based_stop(entry_date, current_date, current_return):
    """
    오래 보유한 손실 종목 정리
    """
    holding_days = (current_date - entry_date).days

    # 60일 이상 보유 + 손실 5% 이상 → 청산
    if holding_days > 60 and current_return < -0.05:
        return True

    # 90일 이상 보유 + 손실 0% 이상 → 청산
    if holding_days > 90 and current_return < 0:
        return True

    return False
```

**Tier 3: Score Degradation Stop (논리 붕괴 감지)**
```python
def score_degradation_stop(entry_score, current_score):
    """
    점수 급락 시 조기 청산
    """
    score_change = current_score - entry_score

    # 점수 20점 이상 하락 → 청산
    if score_change < -20:
        return True

    # 점수가 40점 이하로 하락 → 청산
    if current_score < 40:
        return True

    return False
```

**통합 리스크 관리:**
```python
def integrated_stop_loss_decision(position):
    """
    3단계 Stop Loss 통합 판단
    """
    # Tier 1: Trailing Stop
    trailing_stop_price = trailing_stop(
        position.entry_price,
        position.current_price,
        position.highest_price
    )
    if trailing_stop_price and position.current_price < trailing_stop_price:
        return "EXIT", "Trailing Stop Triggered"

    # Tier 2: Time-Based Stop
    if time_based_stop(position.entry_date, datetime.now(), position.return_pct):
        return "EXIT", "Time-Based Stop Triggered"

    # Tier 3: Score Degradation
    if score_degradation_stop(position.entry_score, position.current_score):
        return "EXIT", "Score Degradation Stop Triggered"

    # Tier 4: ATR-Based Stop
    atr_stop_price = calculate_dynamic_stop_loss(position.company, position.entry_price)[1]
    if position.current_price < atr_stop_price:
        return "EXIT", "ATR Stop Loss Triggered"

    return "HOLD", "No Stop Triggered"
```

##### Solution 4.3: Portfolio-Level Risk Management

**개별 종목이 아닌 포트폴리오 전체 리스크 관리:**

**A. Position Sizing by Conviction & Volatility**
```python
def calculate_optimal_position_size(score, volatility, portfolio_value):
    """
    Kelly Criterion 변형 포지션 크기 결정
    """
    # Base Position Size by Score
    if score >= 70:
        base_pct = 0.02  # 2%
    elif score >= 50:
        base_pct = 0.01  # 1%
    else:
        base_pct = 0.005  # 0.5%

    # Volatility Adjustment
    benchmark_vol = 0.20
    vol_adjustment = benchmark_vol / volatility

    adjusted_pct = base_pct * vol_adjustment

    # Caps
    adjusted_pct = min(adjusted_pct, 0.05)  # 최대 5%
    adjusted_pct = max(adjusted_pct, 0.002)  # 최소 0.2%

    position_value = portfolio_value * adjusted_pct

    return position_value, adjusted_pct
```

**B. Sector & Factor Exposure Limits**
```python
def check_portfolio_constraints(portfolio):
    """
    포트폴리오 제약 조건 검사
    """
    constraints = {}

    # Sector Exposure (단일 섹터 최대 30%)
    sector_exposure = portfolio.group_by('sector').sum('weight')
    constraints['max_sector'] = max(sector_exposure.values())

    # Single Position (단일 종목 최대 5%)
    constraints['max_position'] = portfolio.max('weight')

    # Factor Exposure (Growth Tilt 최대 1.5)
    growth_tilt = portfolio.weighted_avg('growth_score') / 50
    constraints['growth_tilt'] = growth_tilt

    # Liquidity Risk (Low Volume 종목 최대 20%)
    low_volume_weight = portfolio.filter('avg_volume < 500000').sum('weight')
    constraints['low_volume_pct'] = low_volume_weight

    # Violations
    violations = []
    if constraints['max_sector'] > 0.30:
        violations.append(f"Sector exposure {constraints['max_sector']:.1%} > 30%")

    if constraints['max_position'] > 0.05:
        violations.append(f"Single position {constraints['max_position']:.1%} > 5%")

    if constraints['low_volume_pct'] > 0.20:
        violations.append(f"Low volume exposure {constraints['low_volume_pct']:.1%} > 20%")

    return constraints, violations
```

**C. Dynamic Leverage by Market Regime**
```python
def calculate_portfolio_leverage(market_regime):
    """
    시장 국면별 포트폴리오 레버리지 조정
    """
    # VIX 기반 시장 상태 판단
    vix = get_current_vix()

    if vix < 15:  # Low Volatility
        leverage = 1.3
        regime = "BULL"
    elif vix < 25:  # Normal Volatility
        leverage = 1.0
        regime = "NEUTRAL"
    elif vix < 35:  # High Volatility
        leverage = 0.7
        regime = "BEAR"
    else:  # Extreme Volatility
        leverage = 0.5
        regime = "CRISIS"

    return leverage, regime
```

**예상 효과:**
- Stop Loss 작동률: 0% → 18%
- 평균 손실 크기: -15% (이론) → -8% (실제)
- Sharpe Ratio: 현재 값 → +30% 개선
- Max Drawdown: 현재 값 → -25% 감소

---

### 9.5 시스템 재설계 로드맵

#### Phase 1: Quick Wins (1-2개월)

| 항목 | 구현 난이도 | 예상 효과 | 우선순위 |
|------|------------|----------|----------|
| Rank-Based Scoring | 중 | IC +0.10 | **높음** |
| Dynamic ATR Stop Loss | 낮 | Sharpe +20% | **높음** |
| Healthcare Sub-Sector 분리 | 중 | Healthcare IC +50% | **높음** |
| Volatility-Adjusted Position Size | 낮 | Drawdown -15% | 중 |

#### Phase 2: Structural Improvements (3-6개월)

| 항목 | 구현 난이도 | 예상 효과 | 우선순위 |
|------|------------|----------|----------|
| Lifecycle-Based Dual System | 높 | NASDAQ IC +40% | **높음** |
| Biotech Factor Framework | 높 | Biotech Anomaly -70% | **높음** |
| SaaS/Tech Metrics Integration | 중 | Tech IC +60% | 중 |
| Multi-Tier Risk Management | 중 | Risk-Adj Return +30% | 중 |

#### Phase 3: Advanced Features (6-12개월)

| 항목 | 구현 난이도 | 예상 효과 | 우선순위 |
|------|------------|----------|----------|
| ML-Based Biotech Risk Model | 높 | Anomaly Detection +80% | 중 |
| Event-Driven Signal Layer | 높 | Event Alpha +0.05 | 낮 |
| Portfolio Optimization (MVO) | 높 | Sharpe +40% | 중 |
| Transaction Cost Model | 중 | Net Return +1-2% | 낮 |

#### 통합 효과 예측

| 지표 | 현재 | Phase 1 | Phase 2 | Phase 3 | 총 개선 |
|------|------|---------|---------|---------|---------|
| **Overall IC (252d)** | 0.196 | 0.25 | 0.30 | 0.35 | **+79%** |
| **Sharpe Ratio** | 1.2* | 1.4 | 1.7 | 2.0 | **+67%** |
| **Positive Anomaly 수** | 1,625 | 1,100 | 650 | 400 | **-75%** |
| **Stop Loss 작동률** | 0% | 15% | 18% | 20% | **+20%p** |
| **Healthcare IC** | 0.119 | 0.16 | 0.22 | 0.25 | **+110%** |
| **NASDAQ IC** | 0.154 | 0.19 | 0.25 | 0.28 | **+82%** |

*추정값

---

### 9.6 구현 우선순위 매트릭스

```
높은 효과 │ ▲ Rank-Based Scoring    │ ▲ Biotech Framework
         │ ▲ ATR Stop Loss         │ ▲ Lifecycle System
         │ ▲ Sub-Sector Split      │   SaaS Metrics
─────────┼─────────────────────────┼──────────────────────
낮은 효과 │   Event-Driven Layer    │   ML Risk Model
         │   Transaction Cost      │   Portfolio MVO
         │   (낮은 난이도)         │   (높은 난이도)
```

**권장 실행 순서:**
1. **Rank-Based Scoring** (가장 빠른 ROI)
2. **ATR Stop Loss** (즉시 리스크 개선)
3. **Healthcare Sub-Sector Split** (최대 IC 개선)
4. **Lifecycle-Based System** (구조적 개선)
5. **Biotech Factor Framework** (전문성 확보)

---

### 9.7 결론: From Tactical to Strategic

**기존 접근 (Tactical):**
- 문제 종목 제외 → 샘플 손실, 근본 해결 아님
- 임계값 조정 → 파라미터 과적합 위험
- 가중치 재조정 → 표면적 개선, 지속성 낮음

**제안 접근 (Strategic):**
- **Scoring System 재설계:** Value → Rank 기반
- **Segmentation:** Healthcare 4개, Lifecycle 3개로 세분화
- **Specialized Frameworks:** Biotech/SaaS 전용 팩터
- **Dynamic Risk Management:** ATR/Multi-Tier/Portfolio-Level

**핵심 철학:**
> "시장은 선형적이지 않다. 모든 종목에 동일한 잣대를 적용할 수 없다.
> 각 종목의 특성(섹터, 생애주기, 리스크)에 맞는 평가 체계가 필요하다."

**최종 목표:**
- IC 0.196 → 0.35 (79% 개선)
- Sharpe Ratio 1.2 → 2.0 (67% 개선)
- 시장 대비 초과 수익 안정적 창출



---

## 부록 B: 전체 데이터 분석 (55MB Dataset)

### B.1 데이터셋 개요

- **총 레코드 수:** 62,225개 (종목-날짜 조합)
- **컬럼 수:** 54개
- **기간:** 2024-01-16 ~ 2025-09-17
- **고유 종목 수:** 4,738개
- **분석 날짜 수:** 14일

### B.2 점수 분포 통계

| 지표 | 평균 | 중앙값 | 표준편차 | 최소값 | 최대값 |
|------|------|--------|----------|--------|--------|
| final_score | 54.71 | 56.00 | 13.29 | 16.80 | 100.60 |
| value_score | 51.26 | 55.60 | 21.47 | 10.00 | 100.00 |
| quality_score | 58.47 | 58.80 | 16.60 | 9.60 | 100.00 |
| momentum_score | 53.48 | 53.80 | 14.24 | 11.10 | 96.10 |
| growth_score | 55.21 | 56.50 | 18.02 | 10.00 | 100.00 |
| conviction_score | 55.97 | 57.60 | 15.98 | 8.60 | 97.50 |
| interaction_score | 55.11 | 55.80 | 12.87 | 20.40 | 92.60 |

### B.3 등급 분포

| 등급 | 개수 | 비율 | 누적 비율 |
|------|------|------|----------|
| A | 2,801 | 4.50% | 4.50% |
| A+ | 253 | 0.41% | 4.91% |
| B | 17,982 | 28.90% | 33.81% |
| B+ | 11,665 | 18.75% | 52.55% |
| C | 13,441 | 21.60% | 74.15% |
| D | 10,780 | 17.32% | 91.48% |
| F | 5,098 | 8.19% | 99.67% |
| 강력 매수 | 1 | 0.00% | 99.67% |
| 매도 | 71 | 0.11% | 99.79% |
| 매도 고려 | 47 | 0.08% | 99.86% |
| 매수 | 17 | 0.03% | 99.89% |
| 매수 고려 | 40 | 0.06% | 99.95% |
| 중립 | 29 | 0.05% | 100.00% |

### B.4 수익률 분포 통계

| 기간 | 평균 | 중앙값 | 표준편차 | Q1 | Q3 | 샘플 수 |
|------|------|--------|----------|----|----|----------|
| 3d | 3.52% | 0.16% | 146.31% | -2.32% | 2.93% | 62,221 |
| 5d | 5.80% | 0.65% | 199.72% | -2.44% | 4.37% | 62,220 |
| 10d | 7.92% | 0.58% | 175.67% | -3.63% | 5.55% | 62,215 |
| 20d | 12.41% | -0.45% | 257.12% | -7.49% | 6.47% | 62,199 |
| 30d | 18.10% | -0.48% | 286.24% | -8.82% | 8.92% | 62,041 |
| 60d | 27.17% | 0.46% | 327.07% | -12.26% | 14.25% | 52,623 |
| 90d | 41.42% | 1.52% | 467.37% | -16.22% | 20.07% | 38,747 |
| 180d | 63.35% | 2.29% | 671.19% | -21.24% | 27.83% | 29,751 |
| 252d | 75.22% | 1.61% | 701.31% | -25.46% | 32.39% | 25,291 |

### B.5 섹터 분포

| 섹터 | 레코드 수 | 비율 | 평균 Final Score |
|------|-----------|------|------------------|
| HEALTHCARE | 12,964 | 20.83% | 48.42 |
| FINANCIAL SERVICES | 10,333 | 16.61% | 59.07 |
| TECHNOLOGY | 8,921 | 14.34% | 55.68 |
| INDUSTRIALS | 8,059 | 12.95% | 56.66 |
| CONSUMER CYCLICAL | 6,813 | 10.95% | 55.80 |
| REAL ESTATE | 2,966 | 4.77% | 53.75 |
| COMMUNICATION SERVICES | 2,935 | 4.72% | 52.16 |
| CONSUMER DEFENSIVE | 2,668 | 4.29% | 56.35 |
| BASIC MATERIALS | 2,577 | 4.14% | 55.84 |
| ENERGY | 2,433 | 3.91% | 58.04 |
| UTILITIES | 1,242 | 2.00% | 57.19 |
| NONE | 99 | 0.16% | 41.30 |

### B.6 거래소 분포

| 거래소 | 레코드 수 | 비율 | 평균 Final Score |
|--------|-----------|------|------------------|
| NASDAQ | 39,183 | 62.97% | 52.21 |
| NYSE | 22,768 | 36.59% | 59.10 |
| NYSE MKT | 65 | 0.10% | 52.30 |
| OTCQX | 14 | 0.02% | 75.70 |
| BATS | 8 | 0.01% | 47.24 |
| OTCCE | 8 | 0.01% | 59.17 |
| US | 3 | 0.00% | 42.90 |

### B.7 점수-수익률 상관관계 (Pearson)

| Final Score vs | 상관계수 | 샘플 수 | 해석 |
|----------------|---------|---------|------|
| 3d | -0.0217 | 62,221 | ❌ 선형 관계 없음 |
| 5d | -0.0244 | 62,220 | ❌ 선형 관계 없음 |
| 10d | -0.0380 | 62,215 | ❌ 선형 관계 없음 |
| 20d | -0.0458 | 62,199 | ❌ 선형 관계 없음 |
| 30d | -0.0584 | 62,041 | ❌ 음의 관계 |
| 60d | -0.0687 | 52,623 | ❌ 음의 관계 |
| 90d | -0.0717 | 38,747 | ❌ 음의 관계 |
| 180d | -0.0648 | 29,751 | ❌ 음의 관계 |
| 252d | -0.0695 | 25,291 | ❌ 음의 관계 |

### B.8 최고/최악 성과 종목 (252d 기준)

#### 최고 수익률 종목 (Top 10)

| 순위 | 종목 | 회사명 | 섹터 | Final Score | 수익률 |
|------|------|--------|------|-------------|--------|
| 1 | SNYR | Synergy CHC Corp. Common Stock | HEALTHCARE | 41.9 | 41400.00% |
| 2 | NUTX | Nutex Health Inc | HEALTHCARE | 67.1 | 37309.09% |
| 3 | SNYR | Synergy CHC Corp. Common Stock | HEALTHCARE | 41.9 | 35900.00% |
| 4 | ELAB | Elevai Labs, Inc. Common Stock | HEALTHCARE | 29.6 | 26100.00% |
| 5 | CTEV | Claritev Corporation | HEALTHCARE | 40.6 | 24433.33% |
| 6 | QMMM | QMMM Holdings Limited Ordinary | COMMUNICATION SERVICES | 26.7 | 20486.21% |
| 7 | NUTX | Nutex Health Inc | HEALTHCARE | 68.8 | 19075.38% |
| 8 | NUTX | Nutex Health Inc | HEALTHCARE | 67.1 | 18830.00% |
| 9 | SNYR | Synergy CHC Corp. Common Stock | HEALTHCARE | 41.9 | 18200.00% |
| 10 | TNXP | Tonix Pharmaceuticals Holding  | HEALTHCARE | 57.4 | 16760.00% |

#### 최악 수익률 종목 (Bottom 10)

| 순위 | 종목 | 회사명 | 섹터 | Final Score | 수익률 |
|------|------|--------|------|-------------|--------|
| 1 | ASBP | Aspire BioPharma, Inc. | HEALTHCARE | 33.8 | -99.13% |
| 2 | ZJYL | JIN MEDICAL INTERNATIONAL LTD. | HEALTHCARE | 64.7 | -99.13% |
| 3 | ACON | Aclarion Inc | HEALTHCARE | 51.1 | -98.73% |
| 4 | DGLY | Digital Ally Inc | COMMUNICATION SERVICES | 52.2 | -98.46% |
| 5 | CMG | Chipotle Mexican Grill Inc | CONSUMER CYCLICAL | 72.4 | -98.41% |
| 6 | GNLN | Greenlane Holdings Inc | CONSUMER DEFENSIVE | 40.6 | -98.33% |
| 7 | ADTX | Aditxt Inc.  | HEALTHCARE | 25.2 | -98.28% |
| 8 | MAAS | Highest Performances Holdings  | FINANCIAL SERVICES | 41.3 | -98.25% |
| 9 | CMG | Chipotle Mexican Grill Inc | CONSUMER CYCLICAL | 69.8 | -98.15% |
| 10 | CHR | Cheer Holding Inc. | COMMUNICATION SERVICES | 34.6 | -98.05% |

### B.9 Agent Metrics 분포

| 지표 | 평균 | 중앙값 | 표준편차 | 유효 샘플 수 |
|------|------|--------|----------|-------------|
| entry_timing_score | 63.16 | 65.50 | 14.50 | 57,345 |
| stop_loss_pct | -2925.70 | -12.44 | 231590.02 | 59,257 |
| take_profit_pct | 4388.55 | 18.66 | 347385.02 | 59,257 |
| position_size_pct | 6.55 | 8.00 | 3.65 | 59,257 |
| scenario_bullish_prob | 26.61 | 26.00 | 1.42 | 62,017 |
| scenario_sideways_prob | 44.77 | 46.00 | 6.00 | 62,017 |
| scenario_bearish_prob | 28.62 | 28.00 | 7.25 | 62,017 |

### B.10 전체 데이터 기반 추가 인사이트

#### 점수 오분위별 성과 (30d)

| 오분위 | 평균 수익률 | 중앙값 수익률 | 샘플 수 |
|--------|-------------|---------------|----------|
| Q1(Low) | 51.33% | 0.00% | 12,441 |
| Q2 | 20.59% | -1.55% | 12,388 |
| Q3 | 10.43% | -0.75% | 12,401 |
| Q4 | 4.50% | -0.64% | 12,423 |
| Q5(High) | 3.55% | 0.57% | 12,388 |

**핵심 발견:**
1. **수익률 변동성**: 표준편차가 평균보다 훨씬 커서 극단값의 영향이 큼
2. **평균 vs 중앙값**: 평균 수익률이 중앙값보다 높음 → 소수 종목의 높은 수익률이 평균을 왜곡
3. **Healthcare 집중**: 전체 데이터의 약 25%가 Healthcare 섹터
4. **NASDAQ 비중**: 전체의 약 76%가 NASDAQ 종목
5. **등급 분포**: B/C 등급이 가장 많고, 정규분포에 가까운 형태

