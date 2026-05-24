# Phase 3.9 작업 계획

## 개요

Phase 3.8 분석 결과를 기반으로 한국 퀀트 시스템 개선 작업

### 분석 결과 요약 (Phase 3.8)
- Growth Factor IC: 0.495 (최고)
- Value/Quality/Momentum IC: 0.004 ~ 0.083 (낮음)
- Top Decile (43.6+): 22.86% 수익률, 71% 승률
- Bottom Decile (~18): -2.07% 수익률, 18.5% 승률

---

## 수정 대상 파일

| 파일 | 수정 내용 |
|------|-----------|
| `kr/weight.py` | base_weights, sentiment_weights |
| `kr/batch_weight.py` | get_market_sentiment_weights |
| `kr/kr_main.py` | regime_thresholds, RSI 필터, 테마 필터 |

---

## 1. Factor 가중치 최적화

### 1.1 weight.py - base_weights (Line 993-997)

```python
# 변경 전
self.base_weights = {
    'value': 0.25,
    'momentum': 0.25,
    'growth': 0.25,
    'quality': 0.25
}

# 변경 후 (IC 분석 결과 기반)
self.base_weights = {
    'value': 0.15,      # IC 0.083 → 15%
    'momentum': 0.20,   # IC 0.041 → 20%
    'growth': 0.40,     # IC 0.495 → 40%
    'quality': 0.25     # IC 0.004 → 25% (방어적)
}
```

**근거**: Growth Factor IC가 0.495로 다른 팩터(0.004~0.083) 대비 압도적으로 높음

---

## 2. 시장 상태별 전략

### 2.1 weight.py - sentiment_weights (Line 646-674)

```python
# 변경 전
sentiment_weights = {
    'OVERHEATED': {'value': 1.3, 'quality': 1.2, 'momentum': 0.6, 'growth': 0.9},
    'GREED': {'momentum': 1.1, 'growth': 1.0, 'value': 0.9, 'quality': 1.0},
    'NEUTRAL': {'all': 1.0},
    'FEAR': {'value': 1.2, 'quality': 1.3, 'momentum': 0.8, 'growth': 0.7},
    'PANIC': {'value': 1.4, 'quality': 1.5, 'momentum': 0.5, 'growth': 0.6}
}

# 변경 후 (시장 상태별 전략)
sentiment_weights = {
    'OVERHEATED': {  # 과열: 보수적
        'quality': 1.3,
        'value': 1.2,
        'growth': 0.8,
        'momentum': 0.7
    },
    'GREED': {  # 상승장: Growth 비중 증가
        'growth': 1.3,
        'momentum': 1.2,
        'value': 0.7,
        'quality': 0.8
    },
    'NEUTRAL': {  # 횡보장: Value 비중 증가
        'value': 1.2,
        'quality': 1.1,
        'growth': 0.9,
        'momentum': 0.8
    },
    'FEAR': {  # 하락장: Quality 비중 증가
        'quality': 1.4,
        'value': 1.1,
        'growth': 0.7,
        'momentum': 0.8
    },
    'PANIC': {  # 공포: Quality 최대
        'quality': 1.5,
        'value': 1.2,
        'growth': 0.6,
        'momentum': 0.7
    }
}
```

### 2.2 batch_weight.py - get_market_sentiment_weights (Line 876-907)

동일한 변경 적용

---

## 3. 승률 개선

### 3.1 kr_main.py - regime_thresholds (Line 296-303)

```python
# 변경 전
regime_thresholds = {
    'PANIC': (85, 50),
    'FEAR': (75, 45),
    'NEUTRAL': (70, 40),
    'GREED': (70, 45),
    'OVERHEATED': (80, 55)
}

# 변경 후 (Decile 분석 기반: Top Decile 43.6+)
regime_thresholds = {
    'PANIC': (50, 25),      # 하락장: BUY 임계값 대폭 상향
    'FEAR': (48, 23),       # 하락장: 보수적
    'NEUTRAL': (45, 21),    # 횡보장: 기본 (Decile 9-10 기준)
    'GREED': (43, 21),      # 상승장: 적극적
    'OVERHEATED': (48, 25)  # 과열: 보수적
}
```

**근거**:
- Decile 10 (점수 43.6+): 22.86% 수익률, 71% 승률
- Decile 9 (점수 40.1~43.5): 10.69% 수익률, 57% 승률
- Decile 1-3 (점수 ~26): 음수 수익률

### 3.2 kr_main.py - RSI 필터 추가 (Line 305-314)

```python
# 변경 전
momentum_pass = momentum_score >= 35
supertrend = additional_metrics.get('supertrend_signal', '보유')
trend_pass = supertrend in ['매수', '보유']
rs_20d = additional_metrics.get('rs_20d', 0) or 0
rs_pass = rs_20d >= -10
filters_passed = sum([momentum_pass, trend_pass, rs_pass])

# 변경 후 (RSI 과매수 필터 추가)
momentum_pass = momentum_score >= 35
supertrend = additional_metrics.get('supertrend_signal', '보유')
trend_pass = supertrend in ['매수', '보유']
rs_20d = additional_metrics.get('rs_20d', 0) or 0
rs_pass = rs_20d >= -10

# NEW: RSI 과매수 필터 (RSI > 70이면 BUY 진입 억제)
rsi = additional_metrics.get('rsi', 50) or 50
rsi_not_overbought = rsi <= 70

filters_passed = sum([momentum_pass, trend_pass, rs_pass, rsi_not_overbought])
```

---

## 4. 테마별 전략 차별화

### 4.1 kr_main.py - 저성과 테마 보수적 진입 (Line 316-327)

```python
# 변경 전
if final_score >= buy_threshold and filters_passed >= 2:
    final_grade = '매수'
elif final_score >= buy_threshold - 10 and filters_passed >= 2:
    final_grade = '매수 고려'
elif final_score < sell_threshold or momentum_score < 25:
    if final_score < sell_threshold - 10:
        final_grade = '매도'
    else:
        final_grade = '매도 고려'
else:
    final_grade = '중립'

# 변경 후 (테마 조건 + 필터 조건 강화)
# 저성과 테마 정의 (IC < 0.25)
theme = conditions.get('theme', '')
low_ic_themes = ['Telecom_Media', 'IT_Software']
theme_penalty = 5 if theme in low_ic_themes else 0

# 조정된 임계값
adjusted_buy_threshold = buy_threshold + theme_penalty

# 필터 조건 강화: 4개 중 3개 통과 필요
if final_score >= adjusted_buy_threshold and filters_passed >= 3:
    final_grade = '매수'
elif final_score >= adjusted_buy_threshold - 5 and filters_passed >= 3:
    final_grade = '매수 고려'
elif final_score < sell_threshold or momentum_score < 25:
    if final_score < sell_threshold - 10:
        final_grade = '매도'
    else:
        final_grade = '매도 고려'
else:
    final_grade = '중립'
```

---

## 영향 범위 (Impact Scope)

| 영역 | 영향 |
|------|------|
| **weight.py** | WeightCalculator의 기본 가중치 변경 → 모든 종목의 final_weights 영향 |
| **batch_weight.py** | batch 처리 시 동일한 가중치 적용 |
| **kr_main.py** | final_grade 결정 로직 변경 → 모든 종목의 등급 분류 영향 |
| **기존 데이터** | 새 로직으로 재분석 필요 |

---

## 부작용 (Side Effects)

| 항목 | 예상 효과 |
|------|-----------|
| **Growth 비중 증가** | 성장주 유리, 가치주 불리 |
| **BUY 임계값 하향** | 매수 신호 증가 (기존 70→43) |
| **RSI 필터** | 과매수 구간 진입 억제 → BUY 신호 감소 |
| **테마 페널티** | Telecom_Media, IT_Software 매수 등급 감소 |
| **하락장 전략** | PANIC/FEAR 시 매수 신호 대폭 감소 (threshold 48~50) |
| **필터 강화** | 4개 중 3개 통과 필요 → 매수 신호 감소 |

---

## 수정 순서

1. `kr/weight.py` - base_weights 수정
2. `kr/weight.py` - sentiment_weights 수정
3. `kr/batch_weight.py` - get_market_sentiment_weights 수정
4. `kr/kr_main.py` - regime_thresholds 수정
5. `kr/kr_main.py` - RSI 필터 추가
6. `kr/kr_main.py` - 테마 필터 추가

---

## 검증 계획

1. 단위 테스트: 각 함수별 변경 확인
2. 통합 테스트: 전체 분석 파이프라인 실행
3. 백테스트: phase3_7_ic_analysis.py로 IC/승률 비교
4. A/B 비교: 기존 vs 신규 결과 비교

---

*작성일: 2025-11-28*
*Phase 3.8 분석 결과 기반*
