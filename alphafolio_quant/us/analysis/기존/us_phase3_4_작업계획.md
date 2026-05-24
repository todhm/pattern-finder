# US Quant Model Phase 3.4 - 3대 엔진 구현 작업계획

## 문서 정보
- 작성일: 2025-12-07
- 버전: 1.0
- 목적: 미국 시장 특화 퀀트 모델 고도화

---

## 1. 현재 상태 분석

### 1.1 구현 완료된 기능
| 구분 | 모듈 | 상태 |
|------|------|------|
| Factor | Value, Quality, Momentum, Growth v2.0 | 완료 |
| Sector | 섹터별 벤치마크, Healthcare 서브섹터 분류 | 완료 |
| Exchange | NYSE/NASDAQ 차별화 가중치 | 완료 |
| Regime | CRISIS/RECOVERY/BULL/NEUTRAL 감지 | 완료 |
| Interaction | Factor 조합 시너지 점수 | 완료 |
| Risk | Cash Runway Filter (Biotech), ORS | 완료 |

### 1.2 점수 체계 (유지)
```
final_score = (base_score × 0.70 + interaction_score × 0.30) - penalty
범위: 0-100점
```

### 1.3 핵심 부족점
1. **Volatility-aware Factor**: 변동성 노이즈 제거 없음
2. **Macro-aware Model**: 금리/인플레 감도 미반영
3. **Event-aware Model**: 실적/옵션/인사이더 시그널 미활용

---

## 2. 사용 가능한 DB 데이터

### 2.1 Volatility 관련
| 테이블 | 컬럼 | 용도 |
|--------|------|------|
| `us_atr` | atr_14, atr_20 | Average True Range |
| `us_bbands` | upper, middle, lower | Bollinger Bands |
| `us_stock_grade` | volatility_annual, beta | 연간 변동성, 베타 |
| `us_option_daily_summary` | avg_implied_volatility | 내재 변동성 |

### 2.2 Macro 관련
| 테이블 | 컬럼 | 용도 |
|--------|------|------|
| `us_fed_funds_rate` | value, date | Fed Funds Rate |
| `us_treasury_yield` | value, date | 국채 수익률 |
| `us_cpi` | value, date | 소비자물가지수 |
| `us_unemployment_rate` | value, date | 실업률 |
| `us_market_regime` | fed_rate, cpi_yoy, unemployment_rate | 통합 매크로 |

### 2.3 Event 관련
| 테이블 | 컬럼 | 용도 |
|--------|------|------|
| `us_earnings_calendar` | report_date, eps, eps_estimated | 실적 일정 |
| `us_earnings_estimates` | (확인 필요) | EPS 추정치 |
| `us_option_daily_summary` | total_call_volume, total_put_volume, avg_call_iv, avg_put_iv | 옵션 심리 |
| `us_insider_transactions` | acquisition_or_disposal, shares, share_price | 인사이더 매매 |

---

## 3. Engine 1: Volatility-adjusted Factor

### 3.1 목적
팩터 점수의 변동성 노이즈를 제거하여 **신호 대 잡음비(SNR)** 향상

### 3.2 구현 방법

#### 3.2.1 Factor Volatility Scaling
```python
# 각 팩터 점수를 변동성으로 정규화
def volatility_adjusted_score(raw_score: float, vol_20d: float, target_vol: float = 0.20):
    """
    변동성 조정 점수
    - raw_score: 원본 팩터 점수 (0-100)
    - vol_20d: 20일 실현 변동성
    - target_vol: 목표 변동성 (기본 20%)
    """
    if vol_20d <= 0:
        return raw_score

    # 변동성 배율 (고변동성 -> 점수 축소, 저변동성 -> 점수 확대)
    vol_ratio = target_vol / vol_20d
    vol_ratio = max(0.5, min(1.5, vol_ratio))  # 0.5~1.5 클램핑

    # 점수 조정 (중심점 50 기준)
    adjusted = 50 + (raw_score - 50) * vol_ratio
    return max(0, min(100, adjusted))
```

#### 3.2.2 IV Percentile 기반 Momentum 조정
```python
# us_option_daily_summary.avg_implied_volatility 활용
def momentum_iv_adjustment(momentum_score: float, iv_percentile: float):
    """
    IV가 높으면 Momentum 신뢰도 하락
    - iv_percentile: 0-100 (1년 기준)
    """
    if iv_percentile > 80:
        # 고IV 환경: Momentum 50% 할인
        return momentum_score * 0.5 + 25  # 50으로 수렴
    elif iv_percentile < 20:
        # 저IV 환경: Momentum 신뢰도 증가
        return min(100, momentum_score * 1.1)
    else:
        return momentum_score
```

### 3.3 적용 위치
- **파일**: `us/us_volatility_engine.py` (신규)
- **통합**: `us_main_v2.py`의 `_analyze_stock()` 내 팩터 점수 계산 후

### 3.4 구현 Task
| # | Task | 예상 복잡도 |
|---|------|------------|
| 1 | `us_volatility_engine.py` 모듈 생성 | 중 |
| 2 | 20일 실현 변동성 계산 함수 구현 | 하 |
| 3 | IV Percentile 계산 함수 구현 | 하 |
| 4 | Factor별 Volatility Scaling 적용 | 중 |
| 5 | `us_main_v2.py` 통합 | 중 |
| 6 | 단위 테스트 및 검증 | 중 |

---

## 4. Engine 2: Macro-sensitive Factor Weights

### 4.1 목적
금리/인플레 환경에 따라 **Factor 유효성을 동적 조정**

### 4.2 구현 방법

#### 4.2.1 Rate Sensitivity Matrix
```python
# 매크로 환경별 팩터 가중치 매트릭스
MACRO_WEIGHT_MATRIX = {
    # (Fed Rate, CPI) 조합 -> Factor Weights
    'HIGH_RATE_HIGH_INFLATION': {  # Fed > 4.5%, CPI > 3%
        'value': 0.30,
        'quality': 0.40,
        'momentum': 0.15,
        'growth': 0.15
    },
    'HIGH_RATE_LOW_INFLATION': {  # Fed > 4.5%, CPI <= 3%
        'value': 0.25,
        'quality': 0.35,
        'momentum': 0.20,
        'growth': 0.20
    },
    'LOW_RATE_HIGH_INFLATION': {  # Fed <= 4.5%, CPI > 3%
        'value': 0.35,
        'quality': 0.25,
        'momentum': 0.15,
        'growth': 0.25
    },
    'LOW_RATE_LOW_INFLATION': {  # Fed <= 4.5%, CPI <= 3%
        'value': 0.15,
        'quality': 0.20,
        'momentum': 0.25,
        'growth': 0.40
    }
}
```

#### 4.2.2 Treasury Curve Signal
```python
def get_yield_curve_signal(us_treasury_yield_data):
    """
    수익률 곡선 기울기 분석
    - Inverted (10Y - 2Y < 0): 방어적 -> Quality 상향
    - Steepening (10Y - 2Y > 1%): 공격적 -> Growth/Momentum 상향
    """
    # 10Y와 2Y 수익률 비교 (데이터 구조 확인 필요)
    spread = yield_10y - yield_2y

    if spread < 0:
        return 'INVERTED'  # 경기 침체 신호
    elif spread > 1.0:
        return 'STEEPENING'  # 경기 확장 신호
    else:
        return 'NORMAL'
```

#### 4.2.3 Real Rate Factor Adjustment
```python
def adjust_for_real_rate(fed_rate: float, cpi: float, factor_weights: dict):
    """
    실질금리 기반 Value/Growth 조정
    - 실질금리 높음 (> 2%): Value 유리
    - 실질금리 낮음 (< 0%): Growth 유리
    """
    real_rate = fed_rate - cpi

    if real_rate > 2.0:
        # 높은 실질금리: Value 가중치 +5%, Growth -5%
        factor_weights['value'] = min(0.40, factor_weights['value'] + 0.05)
        factor_weights['growth'] = max(0.10, factor_weights['growth'] - 0.05)
    elif real_rate < 0:
        # 음의 실질금리: Growth 가중치 +5%, Value -5%
        factor_weights['growth'] = min(0.50, factor_weights['growth'] + 0.05)
        factor_weights['value'] = max(0.10, factor_weights['value'] - 0.05)

    # 가중치 합 = 1.0 정규화
    total = sum(factor_weights.values())
    return {k: v/total for k, v in factor_weights.items()}
```

### 4.3 적용 위치
- **파일**: `us/us_macro_engine.py` (신규)
- **통합**: `us_main_v2.py`의 Regime 감지 후, Factor 가중치 결정 시

### 4.4 구현 Task
| # | Task | 예상 복잡도 |
|---|------|------------|
| 1 | `us_macro_engine.py` 모듈 생성 | 중 |
| 2 | Macro 환경 분류 함수 구현 | 하 |
| 3 | Rate Sensitivity Matrix 적용 | 중 |
| 4 | Yield Curve Signal 구현 | 중 |
| 5 | Real Rate Adjustment 구현 | 하 |
| 6 | 기존 `us_market_regime` 연동 | 중 |
| 7 | `us_main_v2.py` 통합 | 중 |

---

## 5. Engine 3: Event-driven Score Modifier

### 5.1 목적
단기 이벤트(실적, 옵션, 인사이더)를 반영하여 **수익률 예측 정확도 향상**

### 5.2 구현 방법

#### 5.2.1 Earnings Proximity Filter
```python
async def get_earnings_modifier(symbol: str, analysis_date: date, db):
    """
    실적 발표 전후 Momentum 점수 조정
    - 발표 -5일 ~ +1일: Momentum 점수 신뢰도 하락
    - 발표 후 Beat: +5점 보너스
    - 발표 후 Miss: -5점 패널티
    """
    # us_earnings_calendar에서 실적 일정 조회
    query = """
        SELECT report_date, eps, eps_estimated
        FROM us_earnings_calendar
        WHERE symbol = $1
          AND report_date BETWEEN $2 - INTERVAL '30 days' AND $2 + INTERVAL '30 days'
        ORDER BY report_date DESC
        LIMIT 1
    """
    result = await db.execute_query(query, symbol, analysis_date)

    if not result:
        return {'modifier': 0, 'reason': 'NO_EARNINGS_DATA'}

    report_date = result[0]['report_date']
    days_to_earnings = (report_date - analysis_date).days

    # 실적 발표 임박 (-5일 ~ 0일)
    if -5 <= days_to_earnings <= 0:
        return {
            'modifier': -5,
            'reason': f'EARNINGS_IMMINENT_{-days_to_earnings}D',
            'momentum_discount': 0.5  # Momentum 50% 할인
        }

    # 실적 발표 직후 (0일 ~ +3일)
    if 0 < days_to_earnings <= 3:
        eps = result[0].get('eps')
        eps_est = result[0].get('eps_estimated')
        if eps and eps_est:
            surprise = (eps - eps_est) / abs(eps_est) if eps_est != 0 else 0
            if surprise > 0.05:  # 5% 이상 Beat
                return {'modifier': 5, 'reason': f'EARNINGS_BEAT_{surprise:.1%}'}
            elif surprise < -0.05:  # 5% 이상 Miss
                return {'modifier': -5, 'reason': f'EARNINGS_MISS_{surprise:.1%}'}

    return {'modifier': 0, 'reason': 'NORMAL'}
```

#### 5.2.2 Options Sentiment Signal
```python
async def get_options_sentiment(symbol: str, analysis_date: date, db):
    """
    옵션 시장 심리 분석
    - Put/Call Ratio > 1.5: 약세 시그널
    - Unusual Call Volume: 강세 시그널
    - IV Skew (Put IV - Call IV) > 10%: 하방 리스크
    """
    query = """
        SELECT
            total_call_volume, total_put_volume,
            avg_call_iv, avg_put_iv, avg_implied_volatility
        FROM us_option_daily_summary
        WHERE symbol = $1 AND date = $2
    """
    result = await db.execute_query(query, symbol, analysis_date)

    if not result:
        return {'modifier': 0, 'reason': 'NO_OPTIONS_DATA'}

    row = result[0]
    call_vol = row['total_call_volume'] or 0
    put_vol = row['total_put_volume'] or 0

    modifier = 0
    reasons = []

    # Put/Call Ratio
    if call_vol > 0:
        pc_ratio = put_vol / call_vol
        if pc_ratio > 1.5:
            modifier -= 5
            reasons.append(f'HIGH_PC_RATIO_{pc_ratio:.2f}')
        elif pc_ratio < 0.5:
            modifier += 3
            reasons.append(f'LOW_PC_RATIO_{pc_ratio:.2f}')

    # IV Skew
    call_iv = row['avg_call_iv'] or 0
    put_iv = row['avg_put_iv'] or 0
    if put_iv > 0 and call_iv > 0:
        iv_skew = (put_iv - call_iv) / call_iv
        if iv_skew > 0.10:
            modifier -= 3
            reasons.append(f'NEGATIVE_SKEW_{iv_skew:.1%}')

    return {
        'modifier': modifier,
        'reason': ','.join(reasons) if reasons else 'NEUTRAL',
        'pc_ratio': pc_ratio if call_vol > 0 else None,
        'iv_skew': iv_skew if put_iv > 0 and call_iv > 0 else None
    }
```

#### 5.2.3 Insider Signal Integration
```python
async def get_insider_signal(symbol: str, analysis_date: date, db):
    """
    인사이더 매매 시그널
    - Cluster Buying (90일 내 3명 이상 매수): +10점
    - CEO/CFO 대량 매도: -5점
    """
    query = """
        SELECT
            executive, executive_title,
            acquisition_or_disposal, shares, share_price, date
        FROM us_insider_transactions
        WHERE symbol = $1
          AND date BETWEEN $2 - INTERVAL '90 days' AND $2
        ORDER BY date DESC
    """
    result = await db.execute_query(query, symbol, analysis_date)

    if not result:
        return {'modifier': 0, 'reason': 'NO_INSIDER_DATA'}

    buys = [r for r in result if r['acquisition_or_disposal'] == 'A']
    sells = [r for r in result if r['acquisition_or_disposal'] == 'D']

    modifier = 0
    reasons = []

    # Cluster Buying (3명 이상)
    unique_buyers = set(r['executive'] for r in buys)
    if len(unique_buyers) >= 3:
        modifier += 10
        reasons.append(f'CLUSTER_BUYING_{len(unique_buyers)}_EXECS')
    elif len(unique_buyers) >= 2:
        modifier += 5
        reasons.append(f'MULTIPLE_BUYING_{len(unique_buyers)}_EXECS')

    # CEO/CFO 매도 체크
    c_level_sells = [r for r in sells
                     if 'CEO' in (r['executive_title'] or '').upper()
                     or 'CFO' in (r['executive_title'] or '').upper()]
    if c_level_sells:
        total_sold = sum(r['shares'] or 0 for r in c_level_sells)
        if total_sold > 10000:  # 1만주 이상
            modifier -= 5
            reasons.append(f'C_LEVEL_SELLING_{total_sold:,}_SHARES')

    return {
        'modifier': modifier,
        'reason': ','.join(reasons) if reasons else 'NEUTRAL',
        'buy_count': len(buys),
        'sell_count': len(sells)
    }
```

### 5.3 적용 위치
- **파일**: `us/us_event_engine.py` (신규)
- **통합**: `us_main_v2.py`의 `total_score` 계산 후, 등급 결정 전

### 5.4 점수 적용 방식
```python
# total_score 계산 후
event_modifier = earnings_mod + options_mod + insider_mod
event_modifier = max(-15, min(15, event_modifier))  # -15 ~ +15 클램핑

final_score = max(0, min(100, total_score + event_modifier))
```

### 5.5 구현 Task
| # | Task | 예상 복잡도 |
|---|------|------------|
| 1 | `us_event_engine.py` 모듈 생성 | 중 |
| 2 | Earnings Proximity Filter 구현 | 중 |
| 3 | Options Sentiment Signal 구현 | 중 |
| 4 | Insider Signal Integration 구현 | 중 |
| 5 | Event Modifier 통합 로직 | 하 |
| 6 | `us_main_v2.py` 통합 | 중 |
| 7 | DB 쿼리 최적화 (병렬 처리) | 하 |

---

## 6. 통합 아키텍처

### 6.1 새로운 모듈 구조
```
us/
├── us_main_v2.py              # 메인 (수정)
├── us_volatility_engine.py    # Engine 1 (신규)
├── us_macro_engine.py         # Engine 2 (신규)
├── us_event_engine.py         # Engine 3 (신규)
├── us_market_regime.py        # 기존 (연동)
├── us_healthcare_optimizer.py # 기존 (유지)
└── ...
```

### 6.2 실행 흐름
```
1. Regime Detection (us_market_regime.py)
      ↓
2. Macro Environment Classification (us_macro_engine.py) [NEW]
      ↓
3. Dynamic Factor Weights Calculation
      ↓
4. Factor Score Calculation (Value, Quality, Momentum, Growth)
      ↓
5. Volatility Adjustment (us_volatility_engine.py) [NEW]
      ↓
6. Base Score Calculation (가중합)
      ↓
7. Interaction Score Calculation
      ↓
8. Total Score = Base × 0.70 + Interaction × 0.30
      ↓
9. Cash Runway Penalty (Healthcare only)
      ↓
10. Event Modifier (us_event_engine.py) [NEW]
      ↓
11. Final Score (0-100 클램핑)
      ↓
12. Grade Determination
```

### 6.3 성능 고려사항
- **병렬 처리**: Event Engine의 3개 쿼리를 `asyncio.gather()`로 동시 실행
- **캐싱**: Macro 데이터는 일간 단위로 캐싱 (모든 종목에 동일)
- **DB 인덱스**: 쿼리 성능을 위해 (symbol, date) 복합 인덱스 확인

---

## 7. 구현 우선순위

| 우선순위 | 엔진 | 이유 | 예상 소요 |
|---------|------|------|----------|
| **1** | Event-aware | 데이터 완비, NASDAQ에서 즉시 효과 | 2-3일 |
| **2** | Macro-aware | 기존 Regime 구조 확장, 중간 복잡도 | 2-3일 |
| **3** | Volatility-aware | 가장 근본적, Factor 재설계 필요 | 3-4일 |

---

## 8. 검증 계획

### 8.1 백테스트 기준
- **기간**: 2023-01-01 ~ 2024-12-31 (2년)
- **유니버스**: S&P 500 + NASDAQ 100
- **벤치마크**: 기존 모델 (Phase 3.2) vs 개선 모델 (Phase 3.4)

### 8.2 성과 지표
| 지표 | 목표 |
|------|------|
| IC (Information Coefficient) | +20% 개선 |
| Hit Rate (상위 20% → 수익) | 55% → 60% |
| Max Drawdown | 감소 |
| Sharpe Ratio | +0.1 개선 |

### 8.3 개별 엔진 검증
1. **Volatility Engine**: 고변동성 환경(VIX > 25) IC 개선 측정
2. **Macro Engine**: 금리 변동 시기(2022-2023) 성과 측정
3. **Event Engine**: 실적 시즌(1,4,7,10월) 수익률 분석

---

## 9. 리스크 및 한계

### 9.1 데이터 한계
- `us_earnings_calendar`: 컬럼 구조 재확인 필요 (date 컬럼 오류 발생)
- `us_treasury_yield`: 10Y/2Y 별도 저장 여부 확인 필요
- 옵션 데이터: 일부 소형주 커버리지 낮을 수 있음

### 9.2 과적합 리스크
- Macro Matrix의 임계값(4.5%, 3% 등)은 백테스트로 최적화 필요
- Event Modifier 가중치도 데이터 기반 튜닝 권장

### 9.3 실시간 적용 시 지연
- 매크로 데이터(Fed Rate, CPI)는 발표 주기 고려 필요
- 인사이더 데이터는 SEC 공시 지연(최대 2영업일) 존재

---

## 10. 결론

현재 모델은 **정적 팩터 모델**에 가깝습니다. 미국 시장 특성(높은 변동성, 금리 민감도, 이벤트 드리븐)을 반영하려면 3대 엔진이 필수입니다.

**핵심 메시지:**
1. DB에 필요한 데이터가 **이미 모두 존재**
2. `us_option_daily_summary`, `us_insider_transactions`, `us_market_regime`이 **미활용 상태**
3. Event Engine 우선 구현 시 **NASDAQ 예측력 즉시 개선** 기대

---

## 부록: 참고 자료

- [Inverse Factor Volatility Strategy (2024)](https://eujournal.org/index.php/esj/article/view/18274)
- [Wellington: Value Stocks and Macro Drivers](https://www.wellington.com/en/insights/time-for-value-stocks-to-bounce-back)
- [BlackRock 2025 Investment Directions](https://www.blackrock.com/us/financial-professionals/insights/investment-directions-spring-2025)
- [Morningstar: Why Value Works Better Outside US](https://www.morningstar.com/stocks/why-value-investing-has-worked-better-outside-us)
