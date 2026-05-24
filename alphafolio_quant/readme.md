# Quant Engine

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![asyncpg](https://img.shields.io/badge/asyncpg-async-blue)
![Railway](https://img.shields.io/badge/Railway-0B0D0E?logo=railway&logoColor=white)

시나리오 기반 주식 투자 전략 서비스 **멀티팩터 퀀트 분석 엔진**입니다.

---

## 목차

- [이 저장소의 역할](#이-저장소의-역할)
- [프로젝트 구조](#프로젝트-구조)
- [시스템 아키텍처](#시스템-아키텍처)
- [한국 퀀트 모델](#한국-퀀트-모델)
- [미국 퀀트 모델](#미국-퀀트-모델)
- [예측 적중률 추적 시스템](#예측-적중률-추적-시스템)
- [핵심 기술 특징](#핵심-기술-특징)
- [성능 특성](#성능-특성)
- [내부 디렉토리 구조](#내부-디렉토리-구조)
- [개발환경 및 사용기술](#개발환경-및-사용기술)
- [API 엔드포인트](#api-엔드포인트)
- [서비스 연동 (Chaining)](#서비스-연동-chaining)
- [License](#license)

---

## 이 저장소의 역할

전체 프로젝트 중 **Quant Engine** 컴포넌트를 담당합니다.

- 한국/미국 주식 약 7,000+ 종목 일괄 분석
- 4대 팩터 기반 투자 등급 산출 (A+ ~ F)
- IC(Information Coefficient) 기반 동적 가중치 시스템
- AI Agent용 트레이딩 지표 생성
- **Railway HTTP 서비스로 배포** (FastAPI + uvicorn)

## KR, US의 각 팩터별 전략 코드는 제외된 버전입니다.

## 프로젝트 구조

| 저장소 | 설명 | 기술 스택 |
|--------|------|-----------|
| [**api**](https://github.com/vinjung/alphafolio_api) | AI 채팅 백엔드 API | FastAPI, LangGraph, ChromaDB, Fine-tuned GPT |
| [**data**](https://github.com/vinjung/alphafolio_data) | 데이터 자동 수집 & 지표 계산 | FastAPI, asyncpg, Cloud Scheduler |
| [**chat**](https://github.com/vinjung/alphafolio_chat) | AI 비서 개발환경 | LangChain, LangGraph, ChromaDB |
| [**quant**](https://github.com/vinjung/alphafolio_quant) | **📍 멀티팩터 퀀트 분석 엔진 (현재 저장소)** | NumPy, SciPy, hmmlearn |
| [**stock_agent**](https://github.com/vinjung/alphafolio_stock_agent) | 종목 투자 전략 Multi-Agent AI | LangGraph, Task-driven Architecture |
| [**portfolio**](https://github.com/vinjung/alphafolio_portfolio) | 포트폴리오 생성 & 리밸런싱 엔진 | Risk Parity, VaR/CVaR, LangGraph |

---

## 시스템 아키텍처

### 전체 분석 파이프라인

```mermaid
flowchart TB
    subgraph Input["입력"]
        DATE[분석 날짜]
        SYMBOLS[종목 리스트]
    end

    subgraph Phase1["Phase 1: 공통 데이터"]
        ECON[경제 지표 분석]
        CYCLE[경제 사이클 매칭]
        REGIME[시장 레짐 감지]
    end

    subgraph Phase2["Phase 2: 조건 분석"]
        COND[9가지 조건 분석]
        CLASS[시장 상태 분류]
        WEIGHT[동적 가중치 계산]
    end

    subgraph Phase3["Phase 3: 팩터 계산"]
        VALUE[Value Factor]
        QUALITY[Quality Factor]
        MOMENTUM[Momentum Factor]
        GROWTH[Growth Factor]
    end

    subgraph Phase4["Phase 4: 추가 지표 계산"]
        RISK[리스크 지표]
        AGENT[Agent 지표]
        SCENARIO[시나리오 확률]
    end

    subgraph Output["출력 및 저장"]
        GRADE[투자 등급]
        DB[(stock_grade)]
    end

    DATE --> ECON
    SYMBOLS --> ECON
    ECON --> CYCLE
    CYCLE --> REGIME
    REGIME --> COND
    COND --> CLASS
    CLASS --> WEIGHT
    WEIGHT --> VALUE
    WEIGHT --> QUALITY
    WEIGHT --> MOMENTUM
    WEIGHT --> GROWTH
    VALUE --> RISK
    QUALITY --> RISK
    MOMENTUM --> RISK
    GROWTH --> RISK
    RISK --> AGENT
    AGENT --> SCENARIO
    SCENARIO --> GRADE
    GRADE --> DB
```

### 한국 vs 미국 모델 비교

```mermaid
flowchart LR
    subgraph KR["한국 모델"]
        KR_FACTOR[4 Factor<br/>73개 전략]
        KR_STATE[19개 시장 상태]
        KR_IC[IC 기반 가중치]
    end

    subgraph US["미국 모델"]
        US_FACTOR[4 Factor<br/>+ Interaction]
        US_ENGINE[3대 Engine]
        US_OPT[거래소/섹터 최적화]
    end

    KR_FACTOR --> KR_STATE
    KR_STATE --> KR_IC
    US_FACTOR --> US_ENGINE
    US_ENGINE --> US_OPT
```

---

## 한국 퀀트 모델

### 4대 팩터 (73개 전략)

| 팩터 | 전략 수 | 주요 전략 |
|------|---------|-----------|
| **Value** | 15개 | Magic Formula, EV/EBITDA, 순자산할인, 주주환원율 |
| **Quality** | 17개 | ROE 일관성, 영업마진, 이자보상배율, 현금흐름 |
| **Momentum** | 23개 | 가격 모멘텀, RSI, MACD, 외국인 순매수 |
| **Growth** | 18개 | 매출/EPS/영업이익 성장률, R&D 투자 |

**특징**: 섹터 멀티플라이어 (0.5~1.5) 적용

### 19개 시장 상태 분류

9가지 조건을 분석하여 19개 시장 상태 중 하나로 분류:

```mermaid
flowchart LR
    subgraph Conditions["9가지 조건"]
        C1[거래소]
        C2[시가총액]
        C3[유동성]
        C4[경제사이클]
        C5[시장심리]
        C6[섹터사이클]
        C7[테마]
        C8[변동성]
        C9[수급]
    end

    subgraph States["19개 시장 상태"]
        S1[KOSPI대형-확장과열-공격형]
        S2[KOSPI대형-침체패닉-초방어형]
        S3[KOSDAQ소형-핫섹터-초고위험형]
        S4[...]
    end

    Conditions --> States
```

| 그룹 | 상태 예시 | 특징 |
|------|----------|------|
| 대형주 (6개) | KOSPI대형-확장과열-공격형 | MEGA/LARGE, EXPANSION |
| 중형주 (6개) | KOSPI중형-확장과열-모멘텀형 | MEDIUM |
| 소형주 (4개) | KOSDAQ소형-핫섹터-초고위험형 | SMALL, HOT |
| 특수 (2개) | 전시장-극저유동성-고위험형 | PANIC, LOW liquidity |

### IC 기반 동적 가중치

```mermaid
flowchart LR
    BASE["기본 가중치<br/>V:15% Q:25%<br/>M:20% G:40%"]
    COND["조건별 조정계수"]
    NORM["정규화<br/>(합=100%)"]
    LIMIT["극단값 제한<br/>(5%~50%)"]

    BASE --> COND --> NORM --> LIMIT
```

| 팩터 | 기본 가중치 |
|------|------------|
| Growth | 40% |
| Quality | 25% |
| Momentum | 20% |
| Value | 15% |

### 최종 점수 계산 (IC 기반 가중치)

```
Final Score = Base Factor Score × 36%
            + Sector Rotation Score × 25%
            + Factor Combination Bonus × 39%
```

- **Base Factor Score**: 4대 팩터 × 동적 가중치 (조건부 중립화 적용 후)
- IC 양수 지표만 사용 (IC 음수 지표 제거: smart_money -0.0167, volatility_context -0.0194 등)
- IC 개선: +0.0038 → +0.0600 (+1,479%)

**조건부 중립화 (Phase 6)**: 특정 시장 상태에서 IC가 음수인 팩터를 50점으로 중립화
- 모멘텀 중립화: 모멘텀형, 역발상, 확장과열 시장
- 품질 중립화: 침체, 공포, 역발상, 탐욕, 모멘텀폭발 시장

### 등급 결정 (7단계 + 평가 불가)

| 등급 | 조건 |
|------|------|
| 강력 매수 | Final Score >= 임계값 AND 필터 3개+ 통과 AND 최적 모멘텀 (35~50) |
| 매수 | Final Score >= 임계값 AND 필터 3개+ 통과 |
| 매수 고려 | Final Score >= 임계값 AND 필터 2개+ 통과 |
| 중립 | 매수고려 ~ 매도고려 사이 |
| 매도 고려 | Final Score < 매도고려 임계값 OR 모멘텀 < 25 |
| 매도 | Final Score < 매도 임계값 AND 매도 필터 2개+ 통과 |
| 강력 매도 | Final Score < 매도 임계값 AND 강력매도 조건 3개+/6개 충족 |
| 평가 불가 | 데이터 부족 OR 신뢰도 < 30 |

---

## 미국 퀀트 모델

### 4대 팩터 + Factor Interaction

```mermaid
flowchart TB
    subgraph Factors["4대 팩터"]
        VL[Value<br/>VL1-VL6]
        QA[Quality<br/>QA1-QA6]
        MO[Momentum<br/>MO1-MO6]
        FG[Growth<br/>FG1-FG6 + NQ1-NQ4]
    end

    subgraph Interaction["Factor Interaction"]
        I1[I1: Growth x Quality]
        I2[I2: Growth x Momentum]
        I3[I3: Quality x Value]
        I4[I4: Momentum x Quality]
        I5[I5: All-Factor Agreement]
    end

    subgraph Score["점수 계산"]
        BASE[Base Score x 70%]
        INT[Interaction Score x 30%]
        TOTAL[Total Score]
    end

    VL --> BASE
    QA --> BASE
    MO --> BASE
    FG --> BASE
    VL --> I1
    QA --> I1
    MO --> I2
    FG --> I2
    I1 --> INT
    I2 --> INT
    I3 --> INT
    I4 --> INT
    I5 --> INT
    BASE --> TOTAL
    INT --> TOTAL
```

<details>
<summary><b>US 팩터별 전략 가중치</b></summary>

**Value v2.0** (성장 대비 합리적 가격):

| 전략 | 비중 | 설명 |
|------|------|------|
| RV1 | 25% | PEG Ratio |
| RV2 | 20% | Forward PE Relative |
| RV3 | 20% | EV/Revenue to Growth |
| RV4 | 15% | Rule of 40 |
| RV5 | 10% | FCF Yield Adjusted |
| RV6 | 10% | Price to Target |

**Quality v2.0** (Management Quality):

| 전략 | 비중 | 설명 |
|------|------|------|
| MQ1 | 20% | Gross Margin |
| MQ2 | 20% | ROIC |
| MQ3 | 15% | Operating Leverage |
| MQ4 | 15% | Earnings Quality |
| MQ5 | 15% | Balance Sheet |
| MQ6 | 15% | Margin Trend |

**Momentum v3.0** (Long-term Enhanced, 252일 기준):

| 전략 | 비중 | 설명 |
|------|------|------|
| EM1 | 15% | Risk-Adjusted Momentum (126일) |
| EM2 | 15% | Sector Relative Strength (126일) |
| EM3 | 20% | EPS Estimate Revision |
| EM4 | 20% | Revenue Estimate Revision |
| EM6 | 15% | Earnings Momentum |
| EM8 | 15% | Long-term Price Momentum (252일, IBD RS Style) |

> EM5 (단기 20일 Volume), EM7 (단기 Target Price) 제거. IC 분석: 3일 IC -0.047 (역작동), 252일 IC +0.064 (유효).

**Growth v2.0** (Future Growth):

| 전략 | 비중 | 설명 |
|------|------|------|
| FG1 | 20% | Revenue Growth YoY |
| FG2 | 20% | EPS Growth YoY |
| FG3 | 20% | Forward Growth |
| FG4 | 15% | Growth Consistency |
| FG5 | 15% | Revenue Acceleration |
| FG6 | 10% | Profit Growth |

</details>

### Factor Interaction (비선형 관계 해결)

| Term | 이름 | 가중치 | 계산 방식 |
|------|------|--------|-----------|
| I1 | Growth x Quality | 30% | 기하평균 + 시너지 보너스 |
| I2 | Growth x Momentum | 25% | 조화평균 |
| I3 | Quality x Value | 20% | 산술평균 + 보너스 |
| I4 | Momentum x Quality | 15% | 기하평균 - 패널티 |
| I5 | Conviction Score | 10% | 표준편차 기반 합의도 |

**배경**: Pearson IC (-0.097) vs Spearman IC (+0.174) 갭 해결

### 3대 Engine (Phase 3.4)

```mermaid
flowchart LR
    subgraph Engines["3대 Engine"]
        EVENT[Event Engine<br/>-20 ~ +20점]
        MACRO[Macro Engine<br/>가중치 조정]
        VOL[Volatility Engine<br/>민감도별 조정]
    end

    subgraph EventDetails["Event Engine 상세"]
        E1[실적 발표 임박]
        E2[옵션 센티먼트]
        E3[내부자 거래]
    end

    EVENT --> E1
    EVENT --> E2
    EVENT --> E3
```

<details>
<summary><b>Event Engine 상세</b></summary>

| 이벤트 | Modifier | 설명 |
|--------|----------|------|
| 실적 D-5~D-1 | -5점 | 실적 임박 불확실성 |
| Put/Call > 1.5 | -5점 | Bearish 센티먼트 |
| Put/Call < 0.5 | +3점 | Bullish 센티먼트 |
| 3+ 경영진 매수 | +10점 | Cluster Buying |
| CEO 대량 매도 | -5점 | C-level Selling |
| GEX 음수 (대규모) | -5점 | Gamma Exposure 변동성 위험 (Phase 3.4.2) |
| 뉴스 센티먼트 부정적 | -3~-5점 | 최근 뉴스 부정 감성 (Phase 3.4.2) |
| 뉴스 센티먼트 긍정적 | +3점 | 최근 뉴스 긍정 감성 |

</details>

<details>
<summary><b>Macro Engine 상세</b></summary>

**Rate Sensitivity Matrix**

| 환경 | Fed Rate | CPI | 가중치 조정 |
|------|----------|-----|-------------|
| HIGH_RATE_HIGH_INFLATION | > 4.5% | > 3.0% | Quality +5%, Value +5% |
| HIGH_RATE_LOW_INFLATION | > 4.5% | <= 3.0% | Value +3%, Quality +2% |
| LOW_RATE_HIGH_INFLATION | <= 4.5% | > 3.0% | Value +5% |
| LOW_RATE_LOW_INFLATION | <= 4.5% | <= 3.0% | Growth +4%, Momentum +3% |

**Yield Curve Signal**

| 신호 | 10Y-2Y 스프레드 | 조정 |
|------|-----------------|------|
| INVERTED | < 0% | Quality +3%, Momentum -3% |
| STEEPENING | > 1.0% | Growth +2%, Momentum +2% |

</details>

<details>
<summary><b>Volatility Engine 상세 (Phase 3.4.2)</b></summary>

**Factor별 민감도**

| Factor | Sensitivity | 설명 |
|--------|-------------|------|
| Momentum | 1.0 (100%) | 추세 신뢰도 직결 |
| Growth | 0.7 (70%) | 성장주 특성 |
| Value | 0.3 (30%) | 타이밍 의존도 낮음 |
| Quality | 0.1 (10%) | 펀더멘털 기반 |

**IV Percentile 기반 조정**

| IV Percentile | Modifier |
|---------------|----------|
| >= 90% | -10점 |
| >= 80% | -7점 |
| <= 10% | +5점 |
| <= 20% | +3점 |

</details>

### 시장 레짐 감지 (HMM 3-State)

Hidden Markov Model 기반 3-State 레짐 감지 (v4.0). 4차원 Composite Score (유동성, 매크로, 변동성, 리스크 선호도) 기반.

| 레짐 | 설명 | Growth | Momentum | Quality | Value |
|------|------|--------|----------|---------|-------|
| **BULL** | 강세장 (Low Vol + Positive Return) | 35% | 25% | 20% | 20% |
| **NEUTRAL** | 중립장 (Mixed Signals) | 28% | 20% | 27% | 25% |
| **BEAR** | 약세장 (High Vol + Negative Return) | 20% | 10% | 40% | 30% |

**4차원 Composite Score**:
- Liquidity Score: Fed RRP, SPX Volume, Credit Spread
- Macro Score: Fed Rate, CPI, Yield Curve, MOVE Index, DXY
- Volatility Score: VIX, IV Percentile, HV 20d
- Risk Appetite Score: Put/Call Ratio, IV Skew, News Sentiment

### US 등급 결정 (절대 점수 기반 7단계)

| 등급 | 점수 기준 |
|------|----------|
| 강력 매수 | >= 85 |
| 매수 | 75 ~ 85 |
| 매수 고려 | 65 ~ 75 |
| 중립 | 55 ~ 65 |
| 매도 고려 | 45 ~ 55 |
| 매도 | 35 ~ 45 |
| 강력 매도 | < 35 |

> KR은 시장 레짐별 동적 임계값 사용 (PANIC/FEAR/NEUTRAL/GREED/OVERHEATED별 다른 기준), US는 절대 점수 기반.

### 거래소/섹터별 최적화

<details>
<summary><b>NASDAQ 최적화</b></summary>

**NASDAQ 기본 가중치**
- Growth: 45% (IC 0.136 최고)
- Value: 20%
- Quality: 20%
- Momentum: 15% (IC 0.056 최저)

</details>

**Cash Runway 평가**

| Runway | 패널티 | 위험도 |
|--------|--------|--------|
| < 12개월 | -15점 | Critical |
| < 18개월 | -10점 | High Risk |
| < 24개월 | -5점 | Warning |

---

## 예측 적중률 추적 시스템

퀀트 분석 결과의 실제 성과를 추적하여 모델 신뢰도를 검증합니다.

### 테이블 구조

| 테이블 | 설명 |
|--------|------|
| kr_stock_prediction_history | 한국 종목 등급별 90일 후 적중 여부 |
| kr_stock_prediction_stats | 한국 종목별 적중률 통계 |
| us_stock_prediction_history | 미국 종목 등급별 90일 후 적중 여부 |
| us_stock_prediction_stats | 미국 종목별 적중률 통계 |

### 적중 로직

| 등급 방향 | 조건 | 적중 판정 |
|-----------|------|-----------|
| BUY (강력 매수/매수/매수 고려) | 90일 후 수익률 > 0 | 적중 |
| SELL (매도 고려/매도/강력 매도) | 90일 후 수익률 < 0 | 적중 |

---

## 핵심 기술 특징

<details>
<summary><b>배치 최적화 (99.96% 쿼리 감소)</b></summary>

**최적화 전**: 36,400 쿼리/배치
**최적화 후**: 15 쿼리/배치

**방법**:
1. 공통 데이터 1회 계산
2. 종목별 데이터 벌크 조회 (4-5 쿼리)
3. 메모리에서 가중치 계산 (DB 쿼리 없음)

</details>

<details>
<summary><b>Agent용 지표 생성</b></summary>

| 지표 | 설명 |
|------|------|
| ATR 기반 손절/익절 | 섹터별 ATR 배수 적용 |
| Entry Timing Score | 진입 타이밍 점수 |
| Position Size | 변동성 기반 포지션 사이즈 |
| Scenario Probability | 상승/횡보/하락 확률 |

**Multi-Tier Stop Loss**

| Tier | 유형 |
|------|------|
| 1 | ATR Stop |
| 2 | Trailing Stop |
| 3 | Time-Based Stop |
| 4 | Score Degradation |

</details>

<details>
<summary><b>리스크 지표</b></summary>

| 지표 | 설명 |
|------|------|
| VaR 95% | Value at Risk (일간) |
| CVaR 95% / CVaR 99% | Conditional VaR (극단 손실 평균) |
| VaR 60일 | 기간별 VaR (Hurst Exponent 기반 스케일링) |
| MDD 1년 | 최대 낙폭 |
| Beta | 시장 민감도 |
| Tail Beta | 극단 하락장 민감도 (하위 5% 수익률 기준) |
| Sharpe Ratio | 위험조정수익률 |
| Sortino Ratio | 하락 위험조정수익률 |
| Hurst Exponent | 추세 지속성 (0.5 미만: 평균회귀, 0.5 초과: 추세 지속) |
| Correlation (KOSPI/SPY) | 시장 상관계수 |
| Drawdown Duration | 평균 낙폭 지속 기간 (일) |
| Confidence Score | 데이터 충분성 기반 분석 신뢰도 (0~100) |
| Conviction Score | 팩터 점수 합의도 (표준편차 기반) |

</details>

<details>
<summary><b>등급 결정용 필터 지표</b></summary>

| 지표 | 설명 | 사용 |
|------|------|------|
| SuperTrend Signal | 90일 추세 판별 (매수/보유/매도) | 매수 필터 2 |
| Relative Strength | 90일 상대 강도 (벤치마크 대비 %) | 매수 필터 3 |
| Smart Money Score | 외국인/기관 수급 기반 점수 | 복합 보정 |
| Sector Rotation Signal | 업종 내 상대 강도 | Selection Score 반영 |
| Factor Combination Bonus | 다중 팩터 시너지 보너스 | 최종 점수 보정 |

</details>

<details>
<summary><b>ORS (Outlier Risk Score)</b></summary>

4가지 요소 × 25점 = 100점 만점. 이상치 종목을 포지션 사이징 및 리스크 플래그에 활용합니다.

**KR ORS 기준**:

| 요소 | 극위험 (25점) | 고위험 (20점) | 중위험 (15점) | 주의 (5점) |
|------|-------------|-------------|-------------|----------|
| 가격 | < 500원 | < 1,000원 | < 2,000원 | < 5,000원 |
| 유동성 | 거래대금 < 1억원 | < 5억원 | - | - |
| 변동성 | 연환산 > 80% | > 60% | > 45% | - |
| 시가총액 | < 500억원 | < 2,000억원 | < 5,000억원 | - |

**US ORS 기준**:

| 요소 | 극위험 (25점) | 고위험 (20점) |
|------|-------------|-------------|
| 가격 | < $1 | < $5 (Penny Stock) |
| 거래량 | < 50K | < 100K avg volume |
| 변동성 | 연환산 > 150% | > 100% |
| 시가총액 | < $100M | < $300M |

**Risk Flags**: `EXTREME_RISK` (ORS ≥ 60), `HIGH_RISK` (≥ 40), `MODERATE_RISK` (≥ 20), `NORMAL` (< 20)

</details>

<details>
<summary><b>대안 종목 매칭</b></summary>

매도/매도 고려 등급 종목에 대해 동일 업종 내 매수등급 대안 종목을 자동 추천합니다.

**매칭 우선순위**:
1. 동일 테마 (KR) / 동일 산업 (US)
2. 동일 업종 / 동일 섹터
3. 전체 종목

**저장 필드** (`kr_stock_grade` / `us_stock_grade`):
- `alt_symbol`, `alt_stock_name`, `alt_final_grade`, `alt_final_score`, `alt_match_type`, `alt_reasons`

</details>

<details>
<summary><b>IC 모니터링 (US)</b></summary>

60일 지연 IC와 팩터 스프레드를 모니터링하여 모델 유효성을 검증합니다.

**IC 건강도 임계값**:

| 상태 | IC 범위 | 설명 |
|------|---------|------|
| HEALTHY | >= 0.08 | 양호 |
| DEGRADED | 0.05 ~ 0.08 | 주의 |
| CRITICAL | 0.02 ~ 0.05 | 위험 |
| FAILED | < 0.02 | 무효 |

**팩터 스프레드**: Normal (30-40점), Compressed < 25점 (Crowding), Expanded > 45점

</details>

---

## 성능 특성

| 항목 | 한국 | 미국 |
|------|------|------|
| 분석 대상 | ~2,748 종목 | ~4,500 종목 |
| 배치 크기 | 20 종목/배치 | 25 종목/배치 |
| DB 연결 풀 | 10~45 connections | 10~30 connections |
| 평균 처리 시간 | ~10-15분 | ~15-20분 |

---

## 내부 디렉토리 구조

```
quant/
├── app.py                      # FastAPI 서버 (Railway 배포용)
├── requirements.txt            # Python 의존성
├── kr_prediction_collector.py  # 한국 예측 적중률 수집기
├── us_prediction_collector.py  # 미국 예측 적중률 수집기
│
├── kr/                         # 한국 주식 퀀트 모듈
│   ├── kr_main.py              # 메인 실행 & run_option1() API 진입점
│   ├── db_async.py             # 비동기 DB 커넥션 풀
│   ├── weight.py               # 조건 분석 & 동적 가중치
│   ├── batch_weight.py         # 배치 처리 최적화
│   ├── market_classifier.py    # 19개 시장 상태 분류
│   ├── kr_value_factor.py      # Value 팩터 (15개 전략)
│   ├── kr_quality_factor.py    # Quality 팩터 (17개 전략)
│   ├── kr_momentum_factor.py   # Momentum 팩터 (23개 전략)
│   ├── kr_growth_factor.py     # Growth 팩터 (18개 전략)
│   ├── kr_etc_factor.py        # 기타 팩터 (섹터별 특화 리스크 조정)
│   ├── kr_additional_metrics.py # 리스크/Agent 지표
│   ├── kr_interpretation.py    # 텍스트 해석 생성
│   ├── kr_alternative_matcher.py # 대안 종목 매칭
│   ├── kr_data_prefetcher.py   # 쿼리 최적화 (83% 감소)
│   ├── kr_outlier_risk.py      # 이상치 리스크 감지
│   ├── kr_probability_calibrator.py # 확률 캘리브레이션
│   └── kr_rolling_calibrator.py # 롤링 캘리브레이션
│
├── us/                         # 미국 주식 퀀트 모듈
│   ├── us_main.py              # 메인 실행 & run_option1() API 진입점
│   ├── us_db_async.py          # 비동기 DB 커넥션 풀
│   ├── weight_adjustments.py   # 동적 가중치 조정
│   ├── us_market_regime.py     # HMM 시장 레짐 감지 (3-State: BULL/NEUTRAL/BEAR)
│   ├── us_value_factor.py      # Value 팩터 (VL1-VL6)
│   ├── us_quality_factor.py    # Quality 팩터 (QA1-QA6)
│   ├── us_momentum_factor.py   # Momentum 팩터 (MO1-MO6)
│   ├── us_growth_factor.py     # Growth 팩터 (FG1-FG6, NQ1-NQ4)
│   ├── us_factor_interactions.py # Factor Interaction (I1-I5)
│   ├── us_agent_metrics.py     # Agent용 지표 생성
│   ├── us_event_engine.py      # Event Engine (실적, 옵션, 내부자)
│   ├── us_volatility_adjustment.py # Volatility Engine
│   ├── us_nasdaq_optimizer.py  # NASDAQ 전용 최적화
│   ├── us_healthcare_optimizer.py # 헬스케어 섹터 최적화
│   ├── us_exchange_optimizer.py # 거래소별 최적화
│   ├── us_sector_benchmarks.py # 섹터 벤치마크
│   ├── us_sector_dynamic_weights.py # 섹터별 동적 가중치
│   ├── us_alternative_matcher.py # 대안 종목 매칭
│   ├── us_data_prefetcher.py   # 쿼리 최적화
│   ├── us_outlier_risk.py      # 이상치 리스크 감지
│   └── us_ic_monitor.py        # IC 모니터링
│
└── cache/                      # 캐시 파일
    └── market_environment_cache.json
```

---

## 개발환경 및 사용기술

| 구분 | 기술 |
|------|------|
| Language | Python 3.11+ |
| Web Framework | FastAPI 0.128 |
| ASGI Server | uvicorn 0.40 |
| Database | PostgreSQL (asyncpg) |
| Deploy | Railway |
| Scheduler | Cron (data 서비스에서 트리거) |

### 주요 라이브러리

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| fastapi | 0.128.1 | HTTP API 서버 |
| uvicorn | 0.40.0 | ASGI 서버 |
| asyncpg | 0.30.0 | PostgreSQL 비동기 연결 |
| psycopg2-binary | 2.9.11 | PostgreSQL 동기 연결 |
| httpx | 0.28.1 | 비동기 HTTP 클라이언트 (서비스 체이닝) |
| requests | 2.32.5 | HTTP 클라이언트 (동기) |
| numpy | 2.3.4 | 수치 계산 |
| pandas | 2.3.3 | 데이터 처리 |
| scipy | 1.16.3 | 통계 계산 |
| statsmodels | 0.14.6 | 시계열 분석 |
| scikit-learn | 1.7.2 | 머신러닝 |
| hmmlearn | 0.3.3 | Hidden Markov Model (레짐 감지) |
| linearmodels | 7.0 | 패널 데이터 선형 모델 |
| openai | 2.6.0 | OpenAI API |
| fredapi | 0.5.2 | FRED 경제 데이터 API |
| beautifulsoup4 | 4.14.2 | HTML 파싱 |
| python-dotenv | 1.1.1 | 환경변수 로드 |
| matplotlib | 3.10.7 | 시각화 |
| seaborn | 0.13.2 | 통계 시각화 |
| tqdm | 4.67.1 | 진행률 표시 |
| psutil | 7.1.3 | 성능 모니터링 |

---

## API 엔드포인트

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/health` | 헬스 체크 | 불필요 |
| POST | `/kr/run` | 한국 주식 전체 종목 분석 (당일) | X-API-KEY |
| POST | `/us/run` | 미국 주식 전체 종목 분석 (자동 날짜 감지) | X-API-KEY |

---

## 서비스 연동 (Chaining)

### 전체 작업 흐름

```
data 서비스 (cron trigger)
    │
    ├─ KR: POST /collect/kr/daily-complete (월~금 16:05)
    │   └─→ quant: POST /kr/run
    │       └─→ portfolio: POST /recommend/daily {"country": "KR"}
    │
    └─ US: POST /collect/us/daily (화~토 06:05)
        └─→ quant: POST /us/run
            └─→ portfolio: POST /recommend/daily {"country": "US"}
```

---

## ⚠️ **사업 코드 - 제한적 공개**

🚫 **상업적 사용 / 수정 / 재배포 엄격 금지**
⏰ **임시 공개 후 Private 전환 예정**
👁️ **참고용으로만 사용하세요**

## License
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
