# Phase 3.8 - 한국 주식 모멘텀 팩터 고도화 작업 계획

## 개요

Phase 3.7 Deep Dive 분석 결과, 모멘텀 전략 21개 중 9개(43%)가 음수 IC를 기록하며 역작동하는 심각한 문제가 발견됨. 본 Phase에서는 퀀트 모델링 관점에서 근본 원인을 분석하고, 한국 시장 특성에 맞는 고성능 모멘텀 전략으로 재설계함.

---

## 1. 분석 경과 요약

### 1.1 Phase 3.6 → 3.7 → 3.8 진행 흐름

| Phase | 핵심 발견 | 조치 | 결과 |
|-------|----------|------|------|
| **3.6** | Value IC -0.1105 (역상관) | V전략 14개 비활성화, V21-V26 신규 설계 | Value IC 개선 |
| **3.7** | Momentum IC -0.0470 (역작동) | Deep Dive 분석 실시 | 9개 전략 역작동 확인 |
| **3.8** | 9개 M전략 근본 개선 필요 | **본 Phase** | - |

### 1.2 문제 전략 상세

| 전략 | 명칭 | IC (30일) | 현재 로직 | 문제 유형 |
|------|------|-----------|----------|----------|
| **M9** | RSI Momentum | **-0.0742** | RSI 상승 추세 + 강세 구간 | 기술적 지표 |
| **M12** | Bollinger Bands Breakout | **-0.0567** | 상단 밴드 돌파 + 밴드 확장 | 기술적 지표 |
| **M10** | MACD Momentum | **-0.0551** | MACD 골든크로스 + 상승 추세 | 기술적 지표 |
| **M11** | Stochastic Momentum | **-0.0547** | 스토캐스틱 과매도 탈출 | 기술적 지표 |
| **M7** | Moving Average Alignment | **-0.0545** | 단/중/장기 이평선 정배열 | 기술적 지표 |
| **M20** | ATR Breakout | **-0.0528** | ATR 밴드 돌파 | 기술적 지표 |
| **M14** | Composite Technical | **-0.0496** | 복합 기술적 지표 정렬 | 기술적 지표 |
| **M15** | Foreign Ownership Increase | **-0.0451** | 외국인 지분율 증가 | 수급 지표 |
| **M13** | Block Trade Momentum | **-0.0361** | 대량 블록 거래 | 수급 지표 |

---

## 2. 근본 원인 분석

### 2.1 한국 시장 특성 vs 전략 설계 불일치

| 특성 | 미국 시장 | 한국 시장 | 현재 전략 문제 |
|------|----------|----------|---------------|
| **투자자 구성** | 기관 80%+ | 개인 60%+ | 기관 중심 로직 적용 |
| **모멘텀 지속성** | 3-12개월 | 1-4주 | 장기 모멘텀 가정 |
| **평균회귀 속도** | 느림 | 빠름 | 추세 추종만 고려 |
| **기술적 신호** | 높은 신뢰도 | 노이즈 많음 | 단일 지표 의존 |
| **외국인 매매** | 분산 | 집중 (삼전 등) | 종목 구분 없음 |

### 2.2 전략별 실패 원인 진단

#### A. 기술적 지표 전략 (M7, M9, M10, M11, M12, M14, M20)

**공통 문제:**
```
1. 미국 시장 기준 파라미터 사용 (RSI 70/30, BB 2σ 등)
2. 단방향 모멘텀만 추종 (상승만 포착, 하락 무시)
3. 시장 국면 무시 (강세장/약세장 동일 로직)
4. 노이즈 필터링 부재 (모든 신호 동일 취급)
```

**세부 진단:**

| 전략 | 실패 원인 | 한국 시장 현상 |
|------|----------|---------------|
| M7 (이평선 정배열) | 정배열 완성 시점 = 이미 과열 | 개인 매수 집중 → 기관 매도 |
| M9 (RSI 강세) | RSI 70+ = 과매수 신호 | 한국은 RSI 60부터 조정 시작 |
| M10 (MACD 크로스) | 후행 지표 특성 | 신호 발생 시 이미 늦음 |
| M11 (스토캐스틱) | 과매도 탈출 = 반등 기대 | 하락 추세 지속이 더 흔함 |
| M12 (BB 돌파) | 상단 돌파 = 강한 모멘텀 | 거짓 돌파 후 평균회귀 빈번 |
| M14 (복합 기술) | 여러 지표 동시 신호 | 동시 과열 = 조정 임박 |
| M20 (ATR 돌파) | 변동성 확대 = 추세 시작 | 변동성 확대 = 조정 신호 |

#### B. 수급 지표 전략 (M13, M15)

| 전략 | 실패 원인 | 한국 시장 현상 |
|------|----------|---------------|
| M13 (블록 매매) | 대량 거래 = 기관 관심 | 대량 매도도 블록, 방향성 미구분 |
| M15 (외국인 증가) | 외국인 = 스마트머니 | 특정 대형주 집중, 소형주 무의미 |

---

## 3. 개선 전략 설계

### 3.1 설계 원칙

```
1. 평균회귀 특성 반영: "과열 신호 = 매도", "침체 신호 = 매수"
2. 한국형 파라미터 적용: RSI 60/40, BB 1.5σ 등
3. 시장 국면 조건부 적용: 강세장/약세장 별도 로직
4. 복합 확인 조건 추가: 단일 신호 → 다중 검증
5. 시가총액/테마 차별화: 대형주 vs 소형주 별도 적용
```

### 3.2 전략별 개선안

---

#### **M7: Moving Average Alignment → Adaptive MA Regime**

**현재 문제:** 정배열 완성 = 이미 과열, 개인 매수 집중 시점

**개선 방향:**
- 정배열 "시작" 포착 (완성이 아닌 형성 초기)
- 이격도 기반 과열/침체 판단 추가
- 시장 국면별 가중치 차등 적용

**개선 로직:**
```python
def calculate_m7_enhanced(self):
    """
    M7 Enhanced: Adaptive MA Regime Strategy

    핵심 변경:
    1. 정배열 완성도 점수화 (0~100%)
    2. 이격도 과열/침체 구간 역발상 적용
    3. 국면별 가중치 동적 조정
    """
    # 1. MA 정배열 형성 초기 단계 감지
    #    - 완전 정배열(5 > 20 > 60 > 120) = 과열 위험
    #    - 부분 정배열(5 > 20, but 60 아직) = 매수 기회

    ma_scores = []
    if ma5 > ma20:  # 단기 정배열 시작
        if ma20 < ma60:  # 아직 중기 역배열 = 초기 단계
            ma_scores.append(('early_alignment', 80))
        elif ma60 < ma120:  # 중기 정배열, 장기 역배열
            ma_scores.append(('mid_alignment', 60))
        else:  # 완전 정배열 = 과열 위험
            ma_scores.append(('full_alignment', 30))  # 점수 하향

    # 2. 이격도 기반 조정
    gap_5_20 = (ma5 - ma20) / ma20 * 100
    if gap_5_20 > 5:  # 5% 이상 이격 = 과열
        score *= 0.7  # 30% 감점
    elif gap_5_20 < -3:  # -3% 이하 = 침체, 반등 기대
        score *= 1.3  # 30% 가점

    # 3. 거래량 확인 (정배열 + 거래량 증가 필수)
    if vol_ratio < 1.2:  # 거래량 미동반 정배열
        score *= 0.8

    return score
```

---

#### **M9: RSI Momentum → Korean RSI Mean Reversion**

**현재 문제:** RSI 70+ 강세 구간을 고점수로 평가 → 과매수 조정 맞음

**개선 방향:**
- 한국형 RSI 구간 재설정 (60/40 기준)
- 과열 구간 역발상 적용
- RSI 기울기(변화율) 추가 분석

**개선 로직:**
```python
def calculate_m9_korean_rsi(self):
    """
    M9 Korean RSI: Mean Reversion + Trend Hybrid

    핵심 변경:
    1. 한국형 RSI 구간: 과매수 65+, 과매도 35-
    2. 과열 구간에서 점수 역전 (평균회귀 반영)
    3. RSI 변화율로 추세 강도 측정
    """
    # 1. 한국형 RSI 구간별 점수
    if rsi >= 75:
        # 극단적 과매수 = 강한 매도 신호
        base_score = 15
        reason = "극단적 과매수, 조정 임박"
    elif rsi >= 65:
        # 과매수 = 약한 매도 신호
        base_score = 30
        reason = "과매수 구간"
    elif rsi >= 55:
        # 강세 진입 = 관망 (전통 모멘텀과 반대)
        base_score = 50
        reason = "강세 진입, 관망"
    elif rsi >= 45:
        # 중립 = 균형
        base_score = 55
        reason = "중립 구간"
    elif rsi >= 35:
        # 침체 = 매수 기회 (역발상)
        base_score = 70
        reason = "침체, 반등 기대"
    else:
        # 극단적 과매도 = 강한 매수 신호 (역발상)
        base_score = 85
        reason = "극단적 과매도, 반등 임박"

    # 2. RSI 변화율 (5일간 RSI 변화)
    rsi_change = rsi - rsi_5d_ago
    if rsi_change > 10:  # RSI 급등 = 과열 가속
        base_score -= 10
    elif rsi_change < -10:  # RSI 급락 = 침체 가속
        base_score += 10

    # 3. 시장 국면 조건
    if market_regime == 'PANIC':
        # 패닉장에서 과매도 신호 가중
        if rsi < 35:
            base_score += 15
    elif market_regime == 'GREED':
        # 탐욕장에서 과매수 신호 가중
        if rsi > 65:
            base_score -= 15

    return min(100, max(0, base_score))
```

---

#### **M10: MACD Momentum → MACD Divergence Focus**

**현재 문제:** MACD 골든크로스 = 후행 지표, 신호 발생 시 이미 늦음

**개선 방향:**
- 골든크로스 대신 **다이버전스** 중심으로 전환
- MACD 히스토그램 기울기 분석
- 선행 신호 포착에 집중

**개선 로직:**
```python
def calculate_m10_divergence(self):
    """
    M10 MACD Divergence: 선행 신호 포착

    핵심 변경:
    1. 골든크로스 → 다이버전스 탐지로 전환
    2. 히스토그램 기울기로 모멘텀 변화 선행 포착
    3. 제로라인 돌파 대기 신호 추가
    """
    # 1. 다이버전스 탐지 (선행 신호)
    #    - Bullish Divergence: 가격 저점↓, MACD 저점↑
    #    - Bearish Divergence: 가격 고점↑, MACD 고점↓

    if bullish_divergence:
        # 가격은 신저점이지만 MACD는 상승 = 반등 선행 신호
        divergence_score = 80
    elif bearish_divergence:
        # 가격은 신고점이지만 MACD는 하락 = 조정 선행 신호
        divergence_score = 25
    else:
        divergence_score = 50

    # 2. 히스토그램 기울기 (골든크로스 선행)
    hist_slope = (histogram - histogram_5d_ago) / 5
    if hist_slope > 0 and histogram < 0:
        # 음수 히스토그램이 상승 중 = 골든크로스 예고
        slope_score = 75
    elif hist_slope < 0 and histogram > 0:
        # 양수 히스토그램이 하락 중 = 데드크로스 예고
        slope_score = 30
    else:
        slope_score = 50

    # 3. 제로라인 위치
    if macd > 0 and macd > signal:
        # 이미 강세 = 추가 상승 여력 제한
        zero_score = 45
    elif macd < 0 and macd < signal:
        # 약세지만 반등 가능성
        zero_score = 55
    elif macd < 0 and macd > signal:
        # 약세에서 골든크로스 = 최적 매수 타이밍
        zero_score = 80

    final_score = divergence_score * 0.5 + slope_score * 0.3 + zero_score * 0.2
    return final_score
```

---

#### **M11: Stochastic Momentum → Stochastic Extreme Reversal**

**현재 문제:** 과매도 탈출을 매수 신호로 → 하락 추세 지속이 더 흔함

**개선 방향:**
- 극단 구간(10 이하, 90 이상)에서만 신호 발생
- %K/%D 크로스 대신 구간 체류 시간 분석
- 가격 확인 조건 필수 추가

**개선 로직:**
```python
def calculate_m11_extreme_reversal(self):
    """
    M11 Stochastic Extreme Reversal

    핵심 변경:
    1. 극단 구간(10/90)에서만 반전 신호
    2. 구간 체류 시간 분석 (오래 체류 = 강한 반전)
    3. 가격 반전 확인 필수
    """
    # 1. 극단 구간 탐지
    if stoch_k <= 10:
        # 극단적 과매도
        if stoch_k > stoch_k_prev:  # 상승 시작
            if price > price_low_5d:  # 가격도 반등 확인
                base_score = 85
                reason = "극단 과매도 반전 확인"
            else:
                base_score = 65
                reason = "극단 과매도, 가격 미확인"
        else:
            base_score = 50
            reason = "극단 과매도 지속, 대기"

    elif stoch_k >= 90:
        # 극단적 과매수
        if stoch_k < stoch_k_prev:  # 하락 시작
            base_score = 20
            reason = "극단 과매수 반전 시작"
        else:
            base_score = 30
            reason = "극단 과매수 지속"

    elif 20 <= stoch_k <= 80:
        # 중립 구간 = 신호 없음
        base_score = 50
        reason = "중립 구간, 신호 없음"

    # 2. 체류 시간 가중
    days_in_extreme = count_days_below_20 or count_days_above_80
    if days_in_extreme >= 5:
        # 오래 체류 = 강한 반전 기대
        base_score += 10 if stoch_k <= 20 else -10

    return base_score
```

---

#### **M12: Bollinger Bands Breakout → BB Squeeze & Mean Reversion**

**현재 문제:** 상단 돌파 = 강한 모멘텀 → 거짓 돌파 후 평균회귀 빈번

**개선 방향:**
- 돌파 대신 **Squeeze** (밴드 수축) 포착
- 밴드 내 위치에 따른 평균회귀 점수화
- 돌파 후 되돌림 패턴 필터링

**개선 로직:**
```python
def calculate_m12_squeeze_reversion(self):
    """
    M12 BB Squeeze & Mean Reversion

    핵심 변경:
    1. Squeeze 탐지 (변동성 수축 = 큰 움직임 예고)
    2. 밴드 내 위치로 평균회귀 점수화
    3. 돌파 후 되돌림 필터링
    """
    # 1. Bollinger Band Width (Squeeze 탐지)
    bb_width = (upper_band - lower_band) / middle_band * 100
    bb_width_percentile = percentile_rank(bb_width, bb_width_history_60d)

    if bb_width_percentile < 20:
        # Squeeze = 큰 움직임 예고
        squeeze_score = 75
        squeeze_flag = True
    else:
        squeeze_score = 50
        squeeze_flag = False

    # 2. 밴드 내 위치 (평균회귀)
    position = (close - lower_band) / (upper_band - lower_band)

    if position >= 1.0:
        # 상단 밴드 이상 = 과열, 평균회귀 기대
        position_score = 25
    elif position >= 0.8:
        # 상단 근접 = 경계
        position_score = 40
    elif position >= 0.5:
        # 중앙 = 중립
        position_score = 55
    elif position >= 0.2:
        # 하단 근접 = 반등 기대
        position_score = 70
    else:
        # 하단 밴드 이하 = 강한 반등 기대
        position_score = 85

    # 3. 돌파 후 되돌림 필터
    if yesterday_above_upper and today_below_upper:
        # 상단 돌파 실패 = 강한 매도 신호
        position_score -= 20
    elif yesterday_below_lower and today_above_lower:
        # 하단 돌파 실패 = 강한 매수 신호
        position_score += 20

    # 4. Squeeze + 위치 종합
    if squeeze_flag:
        # Squeeze 상태에서 밴드 위치가 더 중요
        final_score = position_score * 0.7 + squeeze_score * 0.3
    else:
        final_score = position_score

    return final_score
```

---

#### **M13: Block Trade Momentum → Net Block Flow**

**현재 문제:** 대량 거래 = 기관 관심 → 매도도 블록, 방향성 미구분

**개선 방향:**
- 블록 **순매수/순매도** 방향 구분
- 연속 블록 매매 패턴 분석
- 시가총액별 임계값 차등 적용

**개선 로직:**
```python
def calculate_m13_net_block_flow(self):
    """
    M13 Net Block Flow Strategy

    핵심 변경:
    1. 블록 순매수/순매도 방향 구분
    2. 연속성 분석 (3일 연속 순매수 등)
    3. 시가총액별 임계값 차등
    """
    # 1. 블록 순매수 계산 (최근 5일)
    net_block_5d = sum(block_buy - block_sell for each day in 5d)
    net_block_ratio = net_block_5d / market_cap

    # 2. 시가총액별 임계값
    if market_cap >= 10_000_000_000_000:  # 10조 이상 (대형주)
        threshold = 0.001  # 0.1%
    elif market_cap >= 1_000_000_000_000:  # 1조 이상 (중형주)
        threshold = 0.003  # 0.3%
    else:  # 소형주
        threshold = 0.01  # 1%

    # 3. 점수 계산
    if net_block_ratio > threshold * 2:
        base_score = 80
        reason = "강한 블록 순매수"
    elif net_block_ratio > threshold:
        base_score = 65
        reason = "블록 순매수"
    elif net_block_ratio > -threshold:
        base_score = 50
        reason = "블록 중립"
    elif net_block_ratio > -threshold * 2:
        base_score = 35
        reason = "블록 순매도"
    else:
        base_score = 20
        reason = "강한 블록 순매도"

    # 4. 연속성 가중
    consecutive_buy_days = count_consecutive_net_buy_days()
    if consecutive_buy_days >= 3:
        base_score += 10
    consecutive_sell_days = count_consecutive_net_sell_days()
    if consecutive_sell_days >= 3:
        base_score -= 10

    return base_score
```

---

#### **M14: Composite Technical → Regime-Adaptive Composite**

**현재 문제:** 여러 지표 동시 신호 = 동시 과열 = 조정 임박

**개선 방향:**
- 신호 동시성 = 과열 신호로 재해석
- 분산된 신호 = 건전한 상승으로 해석
- 시장 국면별 해석 차등화

**개선 로직:**
```python
def calculate_m14_regime_adaptive(self):
    """
    M14 Regime-Adaptive Composite Strategy

    핵심 변경:
    1. 동시 과열 = 부정적 신호
    2. 분산 신호 = 긍정적 신호
    3. 국면별 해석 차등화
    """
    # 1. 개별 지표 신호 수집
    signals = {
        'rsi_bullish': rsi > 50 and rsi < 70,  # 과열 아닌 강세
        'macd_bullish': macd > signal,
        'ma_bullish': price > ma20,
        'volume_confirm': volume > volume_ma20,
        'bb_position': bb_position > 0.3 and bb_position < 0.8
    }

    bullish_count = sum(signals.values())

    # 2. 동시성 분석
    if bullish_count >= 5:
        # 모든 지표 동시 강세 = 과열 위험
        sync_score = 35
        reason = "동시 과열, 조정 임박"
    elif bullish_count >= 4:
        # 대부분 강세 = 경계
        sync_score = 50
        reason = "강세 과열 경계"
    elif bullish_count == 3:
        # 적정 분산 = 건전한 상승
        sync_score = 70
        reason = "건전한 강세"
    elif bullish_count == 2:
        # 초기 강세 = 매수 기회
        sync_score = 75
        reason = "초기 강세 신호"
    elif bullish_count == 1:
        # 약한 신호 = 대기
        sync_score = 55
        reason = "단일 신호, 대기"
    else:
        # 전면 약세 = 역발상 매수
        sync_score = 65
        reason = "전면 약세, 반등 기대"

    # 3. 시장 국면 조정
    if market_regime in ['GREED', 'OVERHEATED']:
        if bullish_count >= 4:
            sync_score -= 15  # 탐욕장에서 동시 강세 = 더 위험
    elif market_regime in ['FEAR', 'PANIC']:
        if bullish_count <= 1:
            sync_score += 15  # 공포장에서 전면 약세 = 반등 기회

    return sync_score
```

---

#### **M15: Foreign Ownership Increase → Smart Money Contrarian**

**현재 문제:** 외국인 = 스마트머니 가정 → 특정 대형주 집중, 소형주 무의미

**개선 방향:**
- 대형주/소형주 별도 로직
- 외국인 + 기관 복합 수급 분석
- 과열 수급 역발상 적용

**개선 로직:**
```python
def calculate_m15_smart_money_contrarian(self):
    """
    M15 Smart Money Contrarian Strategy

    핵심 변경:
    1. 시가총액별 차등 적용
    2. 외국인 + 기관 복합 분석
    3. 과열 수급 역발상
    """
    # 1. 시가총액별 분기
    if market_cap >= 5_000_000_000_000:  # 5조 이상 (대형주)
        return self._large_cap_foreign_logic()
    else:  # 중소형주
        return self._small_cap_foreign_logic()

def _large_cap_foreign_logic(self):
    """대형주: 외국인 수급 중시, 과열 역발상"""
    foreign_net_20d = sum(foreign_net for 20 days)
    foreign_ratio_change = foreign_ratio - foreign_ratio_20d_ago

    if foreign_ratio_change > 1.0:  # 1%p 이상 증가
        # 대형주 외국인 급증 = 과열 위험
        base_score = 40
        reason = "외국인 과열, 조정 경계"
    elif foreign_ratio_change > 0.3:
        base_score = 55
        reason = "외국인 증가"
    elif foreign_ratio_change > -0.3:
        base_score = 50
        reason = "외국인 중립"
    elif foreign_ratio_change > -1.0:
        base_score = 60
        reason = "외국인 감소, 반등 기대"
    else:
        # 외국인 급감 = 역발상 매수 기회
        base_score = 70
        reason = "외국인 투매, 역발상 매수"

    return base_score

def _small_cap_foreign_logic(self):
    """중소형주: 외국인 의미 적음, 기관 중심"""
    inst_net_20d = sum(institutional_net for 20 days)
    inst_ratio = inst_net_20d / market_cap

    if inst_ratio > 0.02:  # 시총 대비 2% 순매수
        base_score = 70
        reason = "기관 강한 매수"
    elif inst_ratio > 0.005:
        base_score = 60
        reason = "기관 매수"
    elif inst_ratio > -0.005:
        base_score = 50
        reason = "기관 중립"
    else:
        base_score = 40
        reason = "기관 매도"

    return base_score
```

---

#### **M20: ATR Breakout → Volatility Regime Strategy**

**현재 문제:** 변동성 확대 = 추세 시작 가정 → 한국은 변동성 확대 = 조정 신호

**개선 방향:**
- 변동성 수축 → 확대 전환점 포착
- 변동성 과열 역발상 적용
- ATR 백분위 기반 상대 평가

**개선 로직:**
```python
def calculate_m20_volatility_regime(self):
    """
    M20 Volatility Regime Strategy

    핵심 변경:
    1. ATR 백분위 기반 상대 평가
    2. 변동성 수축 = 폭발 대기, 확대 = 조정 임박
    3. 가격 방향과 변동성 교차 분석
    """
    # 1. ATR 백분위 계산
    atr_percentile = percentile_rank(atr, atr_history_60d)

    # 2. 변동성 구간별 점수
    if atr_percentile >= 90:
        # 극단적 고변동성 = 조정 임박
        vol_score = 25
        reason = "극단 변동성, 조정 임박"
    elif atr_percentile >= 70:
        # 고변동성 = 경계
        vol_score = 40
        reason = "고변동성 경계"
    elif atr_percentile >= 30:
        # 정상 변동성
        vol_score = 50
        reason = "정상 변동성"
    elif atr_percentile >= 10:
        # 저변동성 = 폭발 대기
        vol_score = 70
        reason = "저변동성, 폭발 대기"
    else:
        # 극단적 저변동성 = 강한 움직임 임박
        vol_score = 80
        reason = "극저변동성, 큰 움직임 예상"

    # 3. 변동성 변화율 분석
    atr_change = (atr - atr_5d_ago) / atr_5d_ago
    if atr_change > 0.3:  # 30% 이상 급증
        # 변동성 급증 = 조정 가속
        vol_score -= 15
    elif atr_change < -0.2:  # 20% 이상 감소
        # 변동성 수축 = 안정화
        vol_score += 10

    # 4. 가격 방향과 교차 분석
    price_return_5d = (close - close_5d_ago) / close_5d_ago
    if atr_percentile >= 70 and price_return_5d > 0.05:
        # 고변동성 + 급등 = 과열 정점
        vol_score -= 10
    elif atr_percentile <= 30 and price_return_5d < -0.03:
        # 저변동성 + 하락 = 바닥 형성
        vol_score += 10

    return max(0, min(100, vol_score))
```

---

## 4. 구현 계획

### 4.1 파일 수정 목록

| 파일 | 수정 내용 |
|------|----------|
| `kr/kr_momentum_factor.py` | M7, M9, M10, M11, M12, M13, M14, M15, M20 메서드 교체 |
| `kr/weight.py` | 시장 국면별 가중치 재조정 |
| `kr/batch_weight.py` | 동기화 |

### 4.2 검증 계획

| 단계 | 내용 | 출력 |
|------|------|------|
| 1 | 개별 전략 IC 측정 | phase3_8_strategy_ic.csv |
| 2 | Decile 테스트 | phase3_8_decile_test.csv |
| 3 | 국면별 IC 분석 | phase3_8_regime_ic.csv |
| 4 | 테마별 IC 분석 | phase3_8_theme_ic.csv |
| 5 | 최종 Momentum Factor IC | phase3_8_final_ic.csv |

### 4.3 성공 기준

| 지표 | 현재 | 목표 |
|------|------|------|
| M7 IC | -0.0545 | **> 0.00** |
| M9 IC | -0.0742 | **> 0.00** |
| M10 IC | -0.0551 | **> 0.00** |
| M11 IC | -0.0547 | **> 0.00** |
| M12 IC | -0.0567 | **> 0.00** |
| M13 IC | -0.0361 | **> 0.00** |
| M14 IC | -0.0496 | **> 0.00** |
| M15 IC | -0.0451 | **> 0.00** |
| M20 IC | -0.0528 | **> 0.00** |
| **Momentum Factor IC** | -0.0470 | **> +0.03** |

---

## 5. 핵심 설계 원칙 요약

### 5.1 한국 시장 특화 원칙

```
1. 평균회귀 우선: "과열 = 매도", "침체 = 매수"
2. 한국형 파라미터: RSI 65/35, BB 1.5σ, 스토캐스틱 10/90
3. 선행 신호 중시: 완성된 신호보다 형성 중인 신호 포착
4. 시가총액 차등: 대형주/중소형주 별도 로직
5. 동시 과열 경계: 모든 지표 동시 강세 = 조정 임박
```

### 5.2 전략 제외 대신 재설계 이유

```
전략 제외의 문제점:
1. 정보 손실: 역작동해도 "어떤 정보"는 담고 있음
2. 모델 단순화: 복합 모델의 강건성 감소
3. 시장 변화 대응력 저하: 특정 국면에서 유효할 수 있음

재설계의 장점:
1. 한국 시장 특성 반영: 동일 지표를 다른 방식으로 해석
2. 정보 보존: 지표의 정보량 유지
3. 유연성 확보: 국면별 동적 조정 가능
4. 장기 성능: 시장 변화에 적응 가능
```

---

## 6. 다음 단계

1. Phase 3.8 전략 구현
2. 백테스트 및 IC 검증
3. Phase 3.9: Growth Factor 가중치 최적화
4. Phase 4.0: 전체 시스템 통합 테스트

---

*작성일: 2025-11-27*
*작성: Claude Code (Quant Expert Mode)*
