# US Quant Model Phase 3.4.2 작업 계획

## 문서 정보
- **작성일**: 2025-12-07
- **목적**: 미국 퀀트 모델 Gap 분석 및 개선 계획
- **기반**: 외부 피드백 + DB 테이블 분석 + 학술 연구

---

## 1. 외부 피드백 요약

### 1.1 Regime Detection 문제점
> "Regime Detection은 너무 정적인 Rule 기반"
> "미국 시장은 한국보다 훨씬 더 올라갔다가 죽창이 반복된다"

| 현재 구현 | 문제점 |
|----------|--------|
| Bull/Bear/Sideways 단순 분류 | 미국 시장 복잡성 반영 못함 |
| VIX > 30 같은 고정 임계값 | 2020년 VIX 80까지 간 상황 대응 불가 |
| Rule 기반 분류 | 레짐 전환 "사후" 감지 (선행 신호 부재) |

### 1.2 피드백 제안: 4-Regime 모델
```
1. Liquidity Regime     - Fed RRP, SPX Volume, Credit Spread
2. Macro Regime         - Fed Rate, CPI, Yield Curve, MOVE Index
3. Volatility Regime    - VIX, IV Percentile, HV, VIX Term Structure
4. Risk Appetite Regime - Put/Call Ratio, DIX, Junk Bond Spread
```

### 1.3 피드백 제안: 선행 지표 추가
- MOVE Index (채권 변동성) - VIX보다 며칠 선행
- Dollar Index (DXY) - 글로벌 자금 흐름
- US10Y 종합 시그널
- JPY Funding Stress - 캐리 트레이드 해소 신호
- Liquidity Proxy (Fed RRP / SPX Volume)

### 1.4 피드백 제안: Jump Risk Score
> "Outlier Risk Score는 괜찮지만 Jump Risk가 빠짐"

미국 시장 Jump 원인:
- Earnings Gap
- Options Gamma Squeeze
- News-based Jump

필요 지표:
- **Earnings Gap Risk Score**
- **Gamma Exposure Score (GEX)**
- **Dark Pool Sentiment Score (DIX)**

### 1.5 피드백 제안: 3대 핵심 Engine
> "미국 퀀트 모델엔 반드시 다음 세 가지가 필요하다"

| Engine | 목적 | 피드백 설명 |
|--------|------|-------------|
| **Volatility-aware** | 변동성 노이즈 제거 | Momentum, Growth, Value 모두 적용 필요 |
| **Macro-aware** | 금리/유동성/인플레 반영 | Growth/Value는 금리 감도 다름 |
| **Event-aware** | Earnings/Options/Insider | NASDAQ·Healthcare·Tech에 필수 |

---

## 2. 현재 구현 상태 vs 피드백 비교

### 2.1 3대 Engine 구현 상태

| Engine | 피드백 요청 | 현재 구현 | 파일 | Gap |
|--------|------------|----------|------|-----|
| **Event-aware** | Earnings/Options/Insider | ✅ 구현됨 | `us_event_engine.py` | GEX 미포함 |
| **Macro-aware** | 금리/유동성/인플레 | ✅ 구현됨 | `us_market_regime.py` | MOVE/DXY 미포함 |
| **Volatility-aware** | 전체 Factor 노이즈 제거 | ⚠️ 부분 구현 | `us_momentum_factor_v2.py` | Momentum만 적용 |

### 2.2 현재 Regime 분류 로직 (`us_market_regime.py:381-422`)

```python
# 현재 5-Regime 분류 (정적 Rule 기반)
def _classify_regime(self) -> str:
    vix = self.indicators.get('vix_proxy', 20)
    spy_return_3m = self.indicators.get('spy_return_3m', 0)
    nasdaq_vs_spy = self.indicators.get('nasdaq_vs_spy_3m', 0)
    fed_change = self.indicators.get('fed_rate_change_6m', 0)
    cpi_yoy = self.indicators.get('cpi_yoy', 3)
    ma200_dist = self.indicators.get('spy_ma200_distance', 0)

    # 1. CRISIS (VIX > 30)
    if vix > 30:
        return 'CRISIS'

    # 2. AI_BULL
    if vix < 22 and nasdaq_vs_spy > 3 and spy_return_3m > 3 and ma200_dist > 0:
        return 'AI_BULL'

    # 3. TIGHTENING
    if fed_change > 0.25 and cpi_yoy > 3.5:
        return 'TIGHTENING'

    # 4. RECOVERY
    if spy_return_3m > 8 and vix < 25 and ma200_dist > 3:
        return 'RECOVERY'

    # 5. NEUTRAL (기본)
    return 'NEUTRAL'
```

**문제점**:
1. 고정 임계값 (VIX > 30이 항상 CRISIS인가?)
2. 단일 분류 방식 (다차원 스코어 합성 아님)
3. 선행 신호 부재 (이미 발생한 후 감지)

### 2.3 현재 Volatility Engine 적용 범위

| Factor | IV/HV 민감도 | 현재 적용 | 코드 위치 |
|--------|-------------|----------|----------|
| Momentum | 매우 높음 | ✅ 적용됨 | `us_momentum_factor_v2.py:1009-1091` |
| Growth | 높음 | ❌ 미적용 | - |
| Value | 중간 | ❌ 미적용 | - |
| Quality | 낮음 | ❌ 미적용 | - |

---

## 3. DB 테이블 분석 (us_SQL_table_info.csv)

### 3.1 이미 보유한 데이터 (즉시 활용 가능)

#### Macro 데이터
| 테이블명 | 주요 컬럼 | 현재 활용 |
|---------|----------|----------|
| `us_fed_funds_rate` | date, value | ✅ 사용 중 |
| `us_treasury_yield` | interval (10year, 2year), value | ✅ Yield Curve 구현됨 |
| `us_cpi` | date, value | ✅ 사용 중 |
| `us_unemployment_rate` | date, value | ✅ 사용 중 |

#### Options 데이터
| 테이블명 | 주요 컬럼 | 현재 활용 |
|---------|----------|----------|
| `us_option` | gamma, delta, open_interest, implied_volatility, strike, type | ⚠️ 미활용 (GEX 계산 가능!) |
| `us_option_daily_summary` | total_call_volume, total_put_volume, avg_implied_volatility, avg_call_iv, avg_put_iv | ✅ 사용 중 |

#### Event 데이터
| 테이블명 | 주요 컬럼 | 현재 활용 |
|---------|----------|----------|
| `us_earnings_calendar` | reportdate, estimate, fiscaldateending | ✅ Event Engine |
| `us_insider_transactions` | executive[], acquisition_or_disposal, shares | ✅ Event Engine |
| `us_news` | overall_sentiment_score, ticker_sentiment_score | ❌ 미활용! |

#### 기술적 지표
| 테이블명 | 주요 컬럼 | 현재 활용 |
|---------|----------|----------|
| `us_atr` | atr | ✅ Agent Metrics |
| `us_bbands` | real_upper_band, real_middle_band, real_lower_band | ⚠️ 부분 활용 |
| `us_indicators` | 종합 지표 (rsi, macd, vwap, atr 등) | ⚠️ 부분 활용 |

#### 시장 데이터
| 테이블명 | 주요 컬럼 | 현재 활용 |
|---------|----------|----------|
| `us_daily_etf` | SPY, QQQ close, volume | ✅ 사용 중 |
| `market_index` | NASDAQ, NYSE close, volume | ⚠️ 부분 활용 |
| `us_market_regime` | 레짐 결과 저장 | ✅ 사용 중 |

### 3.2 계산 가능한 데이터 (추가 로직 필요)

| 지표 | 계산 방법 | 소스 테이블 | 난이도 |
|-----|----------|------------|--------|
| **GEX (Gamma Exposure)** | `Σ(gamma × OI × 100 × spot)` | `us_option` | Medium |
| **Gamma Flip Level** | Net Gamma = 0인 strike | `us_option` | Medium |
| **OI Concentration** | 특정 strike OI 집중도 | `us_option` | Easy |
| **News Sentiment Score** | 이미 계산되어 저장됨 | `us_news` | Easy |
| **VIX Term Structure** | 근월/차월 IV 비교 | `us_option_daily_summary` | Easy |

### 3.3 미보유 데이터 (외부 수집 필요)

| 지표 | 설명 | 데이터 소스 | FRED Code | 우선순위 |
|-----|------|------------|-----------|---------|
| **MOVE Index** | 채권 변동성 (VIX for Bonds) | FRED | `MOVE` | 🔴 P0 |
| **DXY (Dollar Index)** | 달러 강세/약세 | FRED | `DTWEXBGS` | 🔴 P0 |
| **Fed RRP** | 유동성 프록시 | FRED | `RRPONTSYD` | 🟡 P1 |
| **Credit Spread** | BAA-AAA 스프레드 | FRED | `BAMLC0A0CM` | 🟡 P1 |
| **DIX (Dark Pool)** | Dark Pool 매수 비율 | SqueezeMetrics | (유료) | 🟡 P1 |

---

## 4. 개선안 상세 설계

### 4.1 Phase 4.0: 4-Dimension Regime Composite Model

#### 4.0.1 아키텍처
```
┌────────────────────────────────────────────────────────────┐
│  Composite Regime Score = f(L, M, V, R)                    │
│                                                            │
│  L: Liquidity Score (0-100)                                │
│     - Fed RRP (미수집 → 추가 필요)                          │
│     - SPX Volume (us_daily_etf)                            │
│     - Credit Spread (미수집 → 추가 필요)                    │
│                                                            │
│  M: Macro Score (0-100)                                    │
│     - Fed Rate (us_fed_funds_rate) ✅                       │
│     - CPI (us_cpi) ✅                                       │
│     - Yield Curve (us_treasury_yield) ✅                    │
│     - MOVE Index (미수집 → 추가 필요)                       │
│                                                            │
│  V: Volatility Score (0-100)                               │
│     - VIX Proxy (us_option_daily_summary) ✅                │
│     - IV Percentile (us_option_daily_summary) ✅            │
│     - HV 20d (us_daily 계산) ✅                             │
│     - VIX Term Structure (us_option 계산 가능)             │
│                                                            │
│  R: Risk Appetite Score (0-100)                            │
│     - Put/Call Ratio (us_option_daily_summary) ✅           │
│     - IV Skew (us_option_daily_summary) ✅                  │
│     - DIX (미수집 → 외부 필요)                              │
│     - News Sentiment (us_news) ⚠️ 미활용                   │
└────────────────────────────────────────────────────────────┘
```

#### 4.0.2 구현 파일
- **신규**: `us_regime_composite.py`
- **확장**: `us_market_regime.py` (기존 로직 유지하면서 Composite 추가)

#### 4.0.3 Score 계산 예시
```python
class USRegimeComposite:
    """4-Dimension Regime Composite Model"""

    async def calculate_composite_regime(self) -> Dict:
        # 각 Dimension 점수 계산 (0-100)
        liquidity_score = await self._calculate_liquidity_score()
        macro_score = await self._calculate_macro_score()
        volatility_score = await self._calculate_volatility_score()
        risk_appetite_score = await self._calculate_risk_appetite_score()

        # Composite Score (가중 평균)
        composite = (
            liquidity_score * 0.20 +
            macro_score * 0.30 +
            volatility_score * 0.25 +
            risk_appetite_score * 0.25
        )

        # 동적 Factor 가중치 결정
        weights = self._determine_weights(
            liquidity_score, macro_score,
            volatility_score, risk_appetite_score
        )

        return {
            'composite_score': composite,
            'liquidity_score': liquidity_score,
            'macro_score': macro_score,
            'volatility_score': volatility_score,
            'risk_appetite_score': risk_appetite_score,
            'weights': weights
        }
```

---

### 4.2 Phase 4.1: Jump Risk Engine

#### 4.1.1 목적
미국 시장의 급등/급락(Jump) 위험을 사전 감지

#### 4.1.2 구성 요소

| Component | 설명 | 데이터 소스 | 가중치 |
|-----------|------|------------|--------|
| **Earnings Gap Risk** | 실적 발표 기반 갭 위험 | `us_earnings_calendar`, 과거 갭 패턴 | 35% |
| **GEX Proxy** | Gamma Exposure 근사 | `us_option` (gamma, OI) | 35% |
| **News Sentiment Risk** | 뉴스 센티멘트 급변 | `us_news` | 20% |
| **Liquidity Risk** | 유동성 급감 | Volume, Bid-Ask | 10% |

#### 4.1.3 GEX Proxy 계산 로직
```python
async def calculate_gex_proxy(self, symbol: str, date: date) -> Dict:
    """
    GEX (Gamma Exposure) Proxy 계산

    실제 GEX = Σ(gamma × OI × 100 × spot_price)

    - Call gamma: positive (딜러가 숏 감마 → 가격 상승 시 매수 필요)
    - Put gamma: negative (딜러가 숏 감마 → 가격 하락 시 매도 필요)

    Positive Net GEX: 시장 안정화 (변동성 억제)
    Negative Net GEX: 변동성 폭발 (움직임 증폭)
    """
    query = """
    SELECT
        type,
        strike,
        gamma,
        open_interest,
        implied_volatility
    FROM us_option
    WHERE symbol = $1 AND date = $2
      AND open_interest > 100  -- 유의미한 OI만
    """

    result = await self.db.execute_query(query, symbol, date)

    # Spot price 조회
    spot = await self._get_spot_price(symbol, date)

    call_gex = 0
    put_gex = 0

    for row in result:
        gamma = float(row['gamma'] or 0)
        oi = int(row['open_interest'] or 0)
        contract_gex = gamma * oi * 100 * spot

        if row['type'] == 'call':
            call_gex += contract_gex
        else:
            put_gex -= contract_gex  # Put은 음수로 처리

    net_gex = call_gex + put_gex

    # GEX Score 변환 (0-100, 높을수록 불안정)
    # Negative GEX → 높은 점수 (위험)
    gex_score = self._normalize_gex_to_score(net_gex, spot)

    return {
        'net_gex': net_gex,
        'call_gex': call_gex,
        'put_gex': put_gex,
        'gex_score': gex_score,  # 0-100
        'signal': 'NEGATIVE_GEX' if net_gex < 0 else 'POSITIVE_GEX'
    }
```

#### 4.1.4 Gamma Flip Level 계산
```python
async def find_gamma_flip_level(self, symbol: str, date: date) -> Optional[float]:
    """
    Gamma Flip Level = Net Gamma가 0이 되는 가격

    이 레벨 위: Positive Gamma (안정)
    이 레벨 아래: Negative Gamma (불안정)
    """
    # Strike별 Net Gamma 계산 후 0 교차점 찾기
    ...
```

#### 4.1.5 신규 파일
- **신규**: `us_jump_risk_engine.py`

---

### 4.3 Phase 4.2: Volatility Engine 확장

#### 4.2.1 현재 상태
- Momentum Factor에만 IV/HV 기반 조정 적용
- `us_momentum_factor_v2.py:1009-1091`

#### 4.2.2 확장 계획
| Factor | 현재 | 목표 | 변동성 민감도 |
|--------|------|------|-------------|
| Momentum | ✅ 적용 | 유지 | 1.0 (기준) |
| Growth | ❌ 미적용 | 추가 | 0.7 |
| Value | ❌ 미적용 | 추가 | 0.3 |
| Quality | ❌ 미적용 | 추가 | 0.1 |

#### 4.2.3 구현 방안: 공통 모듈화
```python
# us_volatility_adjustment.py (신규)

class USVolatilityAdjustment:
    """변동성 기반 Factor 점수 조정 (공통 모듈)"""

    FACTOR_SENSITIVITY = {
        'momentum': 1.0,   # 가장 민감
        'growth': 0.7,     # 고변동성 시 성장주 할인율 급등
        'value': 0.3,      # 중간
        'quality': 0.1     # 방어적 (거의 영향 없음)
    }

    def adjust_factor_score(
        self,
        factor: str,
        raw_score: float,
        iv_percentile: float,
        hv_20d: float
    ) -> Dict:
        """
        Factor별 변동성 민감도에 따른 점수 조정

        고변동성 환경 (IV >= 80%):
        - Momentum: 큰 감점 (노이즈 많음)
        - Growth: 중간 감점 (할인율 상승)
        - Value: 작은 감점
        - Quality: 거의 영향 없음

        Returns:
            {
                'adjusted_score': float,
                'modifier': float,
                'reason': str
            }
        """
        sensitivity = self.FACTOR_SENSITIVITY.get(factor, 0)

        # 기존 Momentum의 조정 로직 재사용
        base_modifier = self._calculate_base_modifier(iv_percentile, hv_20d)

        # 민감도 적용
        final_modifier = base_modifier * sensitivity

        adjusted_score = max(0, min(100, raw_score + final_modifier))

        return {
            'adjusted_score': adjusted_score,
            'modifier': final_modifier,
            'reason': f'{factor.upper()}_VOL_ADJ({final_modifier:+.1f})'
        }
```

#### 4.2.4 적용 파일
- **수정**: `us_value_factor_v2.py` - Volatility Adjustment 추가
- **수정**: `us_quality_factor_v2.py` - Volatility Adjustment 추가
- **수정**: `us_growth_factor_v2.py` - Volatility Adjustment 추가
- **신규**: `us_volatility_adjustment.py` - 공통 모듈

---

### 4.4 Phase 4.3: 외부 데이터 수집

#### 4.3.1 FRED API 연동
```python
# us_fred_data_collector.py (신규)

import fredapi

class USFredDataCollector:
    """FRED API를 통한 매크로 데이터 수집"""

    SERIES_MAP = {
        'move_index': 'MOVE',           # ICE BofA MOVE Index
        'dxy': 'DTWEXBGS',              # Trade Weighted Dollar Index
        'fed_rrp': 'RRPONTSYD',         # Fed Reverse Repo
        'credit_spread': 'BAMLC0A0CM',  # BAA Corporate Bond Spread
        'vix': 'VIXCLS'                 # VIX (참고용)
    }

    async def collect_and_store(self, series_key: str):
        """FRED 데이터 수집 후 DB 저장"""
        ...
```

#### 4.3.2 신규 DB 테이블
```sql
-- us_macro_indicators (신규)
CREATE TABLE us_macro_indicators (
    date DATE NOT NULL,
    indicator VARCHAR(50) NOT NULL,
    value DECIMAL(20,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, indicator)
);

-- 저장할 지표
-- move_index, dxy, fed_rrp, credit_spread
```

---

### 4.5 Phase 4.4: HMM 기반 Regime Detection (선택적)

#### 4.5.1 배경
학술 연구: [Regime-Switching Factor Investing with Hidden Markov Models](https://www.mdpi.com/1911-8074/13/12/311)
- HMM 기반 레짐 감지가 Rule 기반 대비 Sharpe ratio 개선

#### 4.5.2 구현 (Optional)
```python
from hmmlearn import GaussianHMM

class USRegimeHMM:
    """HMM 기반 레짐 감지 (Rule 기반 보완)"""

    def __init__(self, n_regimes: int = 4):
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type='full',
            n_iter=100
        )

    def fit(self, features: np.ndarray):
        """
        Features:
        - SPY 일간 수익률
        - VIX 변화율
        - Yield Spread 변화
        - Credit Spread 변화
        """
        self.model.fit(features)

    def predict_regime(self, current_features: np.ndarray) -> int:
        """현재 레짐 예측 (0-3)"""
        return self.model.predict(current_features)[-1]

    def get_transition_prob(self) -> np.ndarray:
        """레짐 전환 확률 행렬"""
        return self.model.transmat_
```

#### 4.5.3 장단점
| 장점 | 단점 |
|------|------|
| 레짐 전환 선행 감지 가능 | 해석 어려움 |
| 데이터 기반 학습 | 과적합 위험 |
| 비정상성 패턴 포착 | 학습 데이터 의존 |

---

## 5. 우선순위 및 실행 계획

### 5.1 우선순위 매트릭스

| 순위 | 작업 | 난이도 | 예상 효과 | 데이터 상태 |
|-----|------|--------|----------|------------|
| 🔴 **P0** | GEX Proxy 계산 | Medium | Jump Risk 감지 2x | ✅ `us_option` 보유 |
| 🔴 **P0** | News Sentiment 활용 | Easy | Risk Appetite 신호 | ✅ `us_news` 보유 |
| 🔴 **P0** | MOVE/DXY 수집 | Easy | Macro 선행 신호 | ❌ FRED API 필요 |
| 🟠 **P1** | 4-Dimension Regime | Medium | 레짐 정확도 향상 | ⚠️ 부분 보유 |
| 🟠 **P1** | Volatility Engine 확장 | Easy | 전체 Factor 노이즈 감소 | ✅ 데이터 있음 |
| 🟡 **P2** | HMM Regime Detection | Hard | 레짐 전환 선행 | ✅ 데이터 있음 |
| 🟡 **P2** | DIX (Dark Pool) 통합 | Medium | Risk Appetite 신호 | ❌ 유료 데이터 |

### 5.2 Phase별 실행 계획

#### Phase 4.0 (즉시 실행 가능)
```
Week 1-2:
├─ GEX Proxy 계산 로직 구현 (us_option 테이블 활용)
├─ News Sentiment Score 활용 로직 추가 (us_news 테이블)
└─ VIX Term Structure 계산 추가
```

#### Phase 4.1 (데이터 수집 후)
```
Week 3-4:
├─ FRED API 연동 (MOVE, DXY, Credit Spread)
├─ us_macro_indicators 테이블 생성
└─ Macro Score 계산 로직 확장
```

#### Phase 4.2 (통합)
```
Week 5-6:
├─ 4-Dimension Regime Composite 구현
├─ Volatility Engine → 전체 Factor 확장
└─ Jump Risk Engine 통합
```

#### Phase 4.3 (선택적)
```
Week 7-8:
├─ HMM 기반 Regime Detection 실험
└─ 성능 비교 (Rule 기반 vs HMM)
```

---

## 6. 신규/수정 파일 목록

### 6.1 신규 파일
| 파일명 | 설명 | Phase |
|--------|------|-------|
| `us_jump_risk_engine.py` | Jump Risk 감지 (GEX, Earnings Gap, News) | 4.1 |
| `us_volatility_adjustment.py` | Volatility 조정 공통 모듈 | 4.2 |
| `us_regime_composite.py` | 4-Dimension Regime Score | 4.0 |
| `us_fred_data_collector.py` | FRED API 데이터 수집 | 4.1 |
| `us_regime_hmm.py` | HMM 기반 Regime Detection (선택적) | 4.3 |

### 6.2 수정 파일
| 파일명 | 수정 내용 | Phase |
|--------|----------|-------|
| `us_market_regime.py` | Composite Regime 통합 | 4.0 |
| `us_value_factor_v2.py` | Volatility Adjustment 추가 | 4.2 |
| `us_quality_factor_v2.py` | Volatility Adjustment 추가 | 4.2 |
| `us_growth_factor_v2.py` | Volatility Adjustment 추가 | 4.2 |
| `us_event_engine.py` | News Sentiment 추가 | 4.0 |
| `us_main_v2.py` | Jump Risk Engine 통합 | 4.1 |

### 6.3 신규 DB 테이블
| 테이블명 | 설명 | Phase |
|---------|------|-------|
| `us_macro_indicators` | FRED 매크로 데이터 | 4.1 |
| `us_jump_risk` | Jump Risk Score 저장 | 4.1 |
| `us_regime_composite` | 4-Dimension Regime Score | 4.0 |

---

## 7. 참고 자료

### 7.1 학술 연구
- [Regime-Switching Factor Investing with Hidden Markov Models](https://www.mdpi.com/1911-8074/13/12/311)
- [Machine Learning in Quant Investing | Acadian](https://www.acadian-asset.com/investment-insights/systematic-methods/machine-learning-in-quant-investing-revolution-or-evolution)
- [Machine Learning Enhanced Multi-Factor Strategies](https://www.arxiv.org/pdf/2507.07107)

### 7.2 지표 설명
- [MOVE Index | Charles Schwab](https://www.schwab.com/learn/story/whats-move-index-and-why-it-might-matter)
- [GEX (Gamma Exposure) | OptionsTrading IQ](https://optionstradingiq.com/what-is-gex/)
- [DIX (Dark Pool Index) | SqueezeMetrics](https://squeezemetrics.com/monitor/dix)

### 7.3 데이터 소스
- [FRED API](https://fred.stlouisfed.org/docs/api/fred/)
- [SqueezeMetrics](https://squeezemetrics.com/) (DIX/GEX - 유료)

---

## 8. 결론

### 8.1 핵심 Gap 요약
| 항목 | 현재 | 목표 |
|------|------|------|
| Regime Detection | 5-Regime 정적 Rule | 4-Dimension Composite Score |
| 선행 지표 | VIX Proxy만 | + MOVE, DXY, GEX |
| Jump Risk | Outlier Risk만 | + Earnings Gap, GEX, News |
| Volatility Engine | Momentum만 | 전체 Factor 적용 |

### 8.2 즉시 실행 가능 항목 (데이터 보유)
1. **GEX Proxy**: `us_option` 테이블로 계산 가능
2. **News Sentiment**: `us_news` 테이블 이미 보유
3. **VIX Term Structure**: `us_option_daily_summary`로 계산 가능

### 8.3 외부 수집 필요 항목
1. **MOVE Index**: FRED API (`MOVE`)
2. **DXY**: FRED API (`DTWEXBGS`)
3. **Credit Spread**: FRED API (`BAMLC0A0CM`)
4. **DIX**: SqueezeMetrics (유료)

---

*작성: Claude Code*
*최종 수정: 2025-12-07*
