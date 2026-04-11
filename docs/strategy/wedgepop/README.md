# WedgepopStrategy

> TraderLion(Oliver Kell)이 "The Money Pattern"이라고 부르는 **Wedge Pop** 패턴을 코드로 옮긴 트레이딩 전략.
> 코드: [`backtester/strategy/adapters/wedgepop_strategy.py`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

---

## 목차

1. [전략 한 줄 정의](#1-전략-한-줄-정의)
2. [TraderLion Cycle of Price와 Wedge Pop](#2-traderlion-cycle-of-price와-wedge-pop)
3. [Wedge Pop 패턴 정의 (Detector가 잡는 것)](#3-wedge-pop-패턴-정의-detector가-잡는-것)
4. [전략 Lifecycle 한눈에](#4-전략-lifecycle-한눈에)
5. [진입 룰 (Entry)](#5-진입-룰-entry)
6. [포지션 사이징 (Position Sizing)](#6-포지션-사이징-position-sizing)
7. [초기 손절 (Initial Stop)](#7-초기-손절-initial-stop)
8. [익절/청산 룰 4가지 (Exit Conditions)](#8-익절청산-룰-4가지-exit-conditions)
9. [모든 파라미터 레퍼런스](#9-모든-파라미터-레퍼런스)
10. [사용 예시](#10-사용-예시)
11. [한계와 의도적으로 안 한 것](#11-한계와-의도적으로-안-한-것)
12. [참고](#12-참고)

---

## 1. 전략 한 줄 정의

> **EMA 아래에서 consolidation 중이던 종목이 처음으로 EMA를 돌파하는 날(=Wedge Pop)을 잡아 다음날 시초가에 매수하고, 4가지 청산 룰(hard stop / exhaustion / EMA trail / time stop) 중 가장 먼저 발동되는 걸로 빠져나오는 단일 포지션 전략.**

- **출처**: TraderLion (Oliver Kell)의 "Cycle of Price Action" 강의. 원문은 프로젝트 루트의 [`traderlion.docx`](../../../traderlion.docx) 참조 (".gitignore에 포함, git 추적 X").
- **별명**: *The Money Pattern* — 한 사이클에서 가장 의미 있는 진입점이라서 그렇게 부름.
- **위험관리 철학**: TraderLion / Van Tharp / Mark Minervini 공통 룰인 "거래당 자본의 1~2%만 위험에 노출". 본 전략은 fixed-fractional risk(`risk_per_trade`)로 강제.

---

## 2. TraderLion Cycle of Price와 Wedge Pop

TraderLion은 모든 종목이 다음 사이클을 반복한다고 본다:

```
                       (강세 사이클)
   Reversal Extension  →  Wedge Pop  →  EMA Crossback
        (바닥)            (★ 진입)        (재진입 기회)
                             ↓
                       Base n' Break
                       (추세 지속, 추가매수)
                             ↓
                     Exhaustion Extension
                          (꼭지)
                             ↓
                        Wedge Drop
                       (하락 사이클 시작)
```

| 단계 | 의미 | 본 전략에서 어떻게 쓰는가 |
|---|---|---|
| **Reversal Extension** | 하락 끝, capitulation. 거래량 폭발 + EMA 한참 아래에서 강한 반등 | 본 전략은 직접 진입 안 함. Wedge Pop 사전 단계로만 봄 |
| **Wedge Pop** ★ | EMA 아래 consolidation 후 **첫 EMA 돌파**. 매수자가 control 잡았다는 확정 신호 | **이 전략의 유일한 진입 트리거** |
| **EMA Crossback** | Wedge Pop 후 EMA로 되돌림. 더 타이트한 stop으로 추가 진입 가능 | 본 전략은 추가매수 안 함. trail stop이 EMA를 사용하는 이유는 이걸 inverse로 활용 — EMA 아래로 close하면 EMA Crossback 실패로 보고 청산 |
| **Base n' Break** | 상승 중 짧은 EMA 위 consolidation 후 신고가 돌파 | 다른 전략 (`BaseNBreakStrategy`, 미구현)의 영역 |
| **Exhaustion Extension** | 종가가 10 EMA 대비 비정상적으로 멀어짐. 과열 = mean reversion 임박 | **익절 조건 #2**. `extension_pct` / `extension_atr_mult`로 발동 |
| **Wedge Drop** | EMA 위로 올라갔던 종목이 다시 EMA 아래로 떨어짐 | **익절 조건 #3** (EMA Trail). 종가가 10 EMA 아래로 close하면 청산 |

> 본 전략은 사이클의 **Wedge Pop ~ Exhaustion Extension** 구간만 잡고 빠져나옴. 그 뒤 사이클(Wedge Drop, EMA Crossback Down 등)은 다른 전략의 영역.

---

## 3. Wedge Pop 패턴 정의 (Detector가 잡는 것)

본 전략의 진입 신호는 [`WedgePopDetector`](../../../backtester/pattern/adapters/wedge_pop.py)가 만든다. Detector가 한 봉을 Wedge Pop으로 인정하려면 다음 3가지를 모두 만족해야 함.

### (1) 사전 condition: Consolidation
- **조건**: 최근 `lookback`(기본 15)일 중 **`consolidation_pct`(기본 60%) 이상**의 봉이 fast EMA(10) **아래**에서 종가 마감.
- **수식**: `count(close[i-15:i] < ema_fast[i-15:i]) / 15 ≥ 0.6`
- **의도**: 한 번이라도 EMA 위로 올라간 종목은 의미 없음. "EMA 아래에 박혀 있던" 종목이 처음으로 올라오는 순간만 잡으려는 것.
- **왜 100%가 아니라 60%인가**: 실제 시장은 잡음이 있어서 1~2일 위로 삐져나갔다 다시 들어가는 경우가 흔함. 100%를 요구하면 거의 안 잡힘.

### (2) 돌파: Breakout
- **조건**: 종가가 fast EMA(10)와 slow EMA(20) **둘 다** 위.
- **수식**: `close[i] > ema_fast[i] AND close[i] > ema_slow[i]`
- **의도**: "처음 EMA 위로 올라오는 결정적인 봉"을 정의.

### (3) 강도 필터: Breakout Strength
- **조건** (둘 중 하나라도 만족): 
  - **EMA distance**: `(close − max(ema_fast, ema_slow)) / max(ema_fast, ema_slow) ≥ breakout_pct (기본 1.5%)`
  - **Daily momentum**: `(close − prev_close) / prev_close ≥ 1.5%`
- **의도**: EMA에 살짝 걸친 약한 돌파 제거. 단, EMA가 가격에 너무 가까운 tight consolidation의 경우 EMA distance만으로 판단하면 놓치므로, 일봉 자체의 강도(전일 종가 대비 % 변동)도 인정.

### (4) 손절 가격 (signal에 포함)
- **`stop_loss = min(Low[i-15:i])`** — 직전 15봉의 저점 중 최저값 = "consolidation low".
- 이게 다음 단계에서 strategy의 **initial stop**으로 그대로 사용됨.

### (5) Cooldown
- 한 신호 후 `lookback`(15)봉 동안 추가 신호 차단. Wedge Pop은 사이클당 1번만 일어난다는 가정 (Oliver Kell의 정의).

---

## 4. 전략 Lifecycle 한눈에

```
                     ┌─────────────────────────┐
                     │   Detector가 신호 발생    │
                     │  (T일 종가 기준 Wedge Pop) │
                     └───────────┬─────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │  T+1일 시초가에 매수             │
                │  shares = capital × risk%       │
                │           ÷ (entry − stop)      │
                └───────────┬────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────────┐
        │ 매일 open ~ close 구간에 4가지 exit 체크 (이 순서)  │
        │                                                  │
        │  1. Hard stop  : Low ≤ stop                       │
        │       → stop 가격에 청산 (손절)                    │
        │                                                  │
        │  2. Exhaustion : 종가가 10 EMA 대비 너무 위        │
        │       → 그날 종가에 청산 (익절, sell into strength)│
        │                                                  │
        │  3. EMA Trail  : 종가 < 10 EMA                     │
        │       → 그날 종가에 청산                            │
        │       (단, trail_after_profit=True일 땐 수익권일 때만)│
        │                                                  │
        │  4. Time stop  : 진입 후 max_holding_days 도달      │
        │       → 그날 종가에 청산                            │
        └─────────────────────────────────────────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  다음 신호 대기            │
                │  (단일 포지션, 겹침 금지)   │
                └────────────────────────┘
```

---

## 5. 진입 룰 (Entry)

### 룰
- Detector가 T일에 Wedge Pop 신호 발생 → **T+1일의 시초가(Open)에 매수**.
- 단일 포지션. 이미 보유 중이면 새 신호는 무시 (`next_open_idx` 가드).

### 왜 종가 추격이 아니라 다음날 시초가인가?
1. **현실적인 체결 가정**: 일봉을 쓰는 시스템에서 "종가에 산다"는 건 비현실적. 종가는 장 마감 시점인데 그때까지 봉이 끝나는지 확인하면 이미 늦음.
2. **좋은 fill**: TraderLion 도큐 — 돌파일 종가는 종종 그날의 고점 부근. 다음날 시초가는 평균적으로 약간 갭 다운 또는 보합으로 시작하는 경우가 많아 평균 진입가가 더 좋음.
3. **검증 가능**: T+1일 시초가는 backtest에서 명확히 정의된 가격 → look-ahead bias 없음.

### 왜 단일 포지션인가?
- 본 전략의 risk 관리(`risk_per_trade`)는 자본 전체 기준으로 한 번에 한 포지션만 사이즈를 계산. 동시에 여러 포지션을 쥐면 총 위험 노출이 누적됨.
- 추가매수(pyramiding)는 미구현 — 도큐의 "Add in Pieces" 룰을 옮기려면 EMA Crossback 검출이 필요한데 별도 작업.

### 옵션: Gap-up Confirmation (`require_gap_up`)
**디폴트 OFF**. ON으로 켜면 다음 조건 추가:

> **T+1일 시초가 > T일 종가** 일 때만 진입. 아니면 신호 스킵.

#### 왜?
TraderLion 도큐 (Wedge Pop / Gaps and Momentum 섹션) 직접 인용:
> *"**Most Wedge Pops that start new trends often include unfilled gaps with strong volume**, showing that buyers are firmly in control."*
>
> *"Unfilled gaps: These show strength, as the price doesn't pull back to close the gap, indicating strong buyer demand."*

즉 *"진짜 새 추세가 시작되는 Wedge Pop은 다음 봉에서 갭업 또는 강한 시가로 매수세가 follow-through 한다"*는 룰을 코드로 옮긴 것. 약한 돌파(다음날 매수세 사라짐)를 확인적으로(confirmatory) 거름.

#### 거르는 시나리오
| 시나리오 | T일 close | T+1일 open | OFF 동작 | ON 동작 |
|---|---|---|---|---|
| 강한 follow-through | $100 | $102 (+2%) | ✅ 진입 | ✅ 진입 |
| 약한 보합 | $100 | $99.50 | ✅ 진입 | ❌ 스킵 |
| 갭다운 (false breakout) | $100 | $97 | ✅ 진입 (손절 가능성↑) | ❌ 스킵 |

#### 실데이터 검증
| 종목 | T일 close | T+1일 open | 갭 | ON에서 통과? |
|---|---|---|---|---|
| ELF 2023-11-14 | $106.21 | $106.80 | +$0.59 (+0.56%) | ✅ |
| AMZN 2021-06-08 | $163.21 | $163.64 | +$0.43 (+0.26%) | ✅ |

→ 두 textbook Wedge Pop 모두 통과. 좋은 셋업은 거르지 않고, 약한 셋업만 거름.

#### 트레이드오프
- ✅ False breakout 거르기 → win rate ↑ 가능성
- ✅ 도큐 권장 룰과 정렬
- ❌ 거래 수 감소 → backtest 표본 줄어듦
- ❌ 가끔 갭다운 후 V자 반등 winner 놓침
- ❌ 단순 임계값 — `$0.01` 차이 갭업도 통과. 더 정교하게 하려면 `min_gap_pct`나 ATR 기반 갭 임계값 추가 필요 (현재 미구현)

#### 코드 위치
- [`wedgepop_strategy.py:_execute_trade`](../../../backtester/strategy/adapters/wedgepop_strategy.py) — entry_price 계산 직후 가드

### 코드 위치
- 진입 인덱스 계산: [`wedgepop_strategy.py:_next_open_index`](../../../backtester/strategy/adapters/wedgepop_strategy.py)
- 진입가 결정 + gap-up 가드: [`wedgepop_strategy.py:_execute_trade`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

---

## 6. 포지션 사이징 (Position Sizing)

### 룰: Fixed-Fractional Risk
**거래 1건에서 잃을 자본의 비율**(`risk_per_trade`)을 고정해 두고, 거기에서 역산해 매수할 주식 수를 결정.

```python
risk_amount   = capital × risk_per_trade        # 잃을 총 금액
risk_per_share = entry_price − stop_loss        # 1주당 손실
shares         = max(1, int(risk_amount / risk_per_share))
```

### 예시: capital = $100,000, risk_per_trade = 2%

| 종목 | entry | stop | risk/share | risk_amount | shares 계산 | shares | 매수금액 | 손절시 손실 |
|---|---|---|---|---|---|---|---|---|
| A (좁은 stop) | $100 | $98 | $2 | $2,000 | 2000/2 | **1,000주** | $100,000 | $2,000 (=2%) |
| B (넓은 stop) | $100 | $90 | $10 | $2,000 | 2000/10 | **200주** | $20,000 | $2,000 (=2%) |
| C (저가주) | $50 | $47 | $3 | $2,000 | 2000/3 | **666주** | $33,300 | $1,998 (≈2%) |

### 핵심 통찰

1. **모든 거래의 손실 상한이 동일** ($2,000)
   - 진입가, 손절폭, 변동성 다 달라도 손실 금액은 거의 동일.
   - 연속으로 손실 봐도 drawdown이 선형으로만 누적 → 회복 가능.

2. **손절폭이 좁을수록 더 많이 산다**
   - "좋은 setup일수록 stop이 entry에 가깝다" → 자동으로 더 큰 포지션.
   - "엉성한 setup일수록 stop이 멀다" → 자동으로 작은 포지션.
   - **수동 판단 없이 신호의 quality에 비례한 사이징**이 자동으로 됨.

3. **매수 금액 ≠ 위험**
   - A는 $100,000 다 쓰지만 진짜 위험에 노출된 건 $2,000뿐.
   - 나머지 $98,000은 stop 위에 있으므로 안전 영역.

4. **연속 손실 방어**
   - 2%씩 10번 연속 손실 → drawdown ≈ 18% (1.02^10이 아니라 10×2% 근사)
   - 만약 한 번에 20%씩 베팅했다면 5번 연속 손실 → -67% → 회복 거의 불가능.
   - "long-term 생존의 단일 가장 중요한 룰" — Van Tharp.

### 왜 자본이 변할 때마다 재계산하는가?
- 본 전략은 매 진입마다 **현재 자본**(`capital`) 기준으로 재계산. 즉 누적 수익이 나면 다음 거래에서 더 큰 포지션, 손실이 나면 더 작은 포지션.
- 이건 **compound growth**의 핵심: 이기는 동안 점점 큰 베팅, 지는 동안 점점 작은 베팅.

### 코드 위치
- [`wedgepop_strategy.py:188-191`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

---

## 7. 초기 손절 (Initial Stop)

### 룰
- 진입 동시에 stop = `signal.stop_loss` = **직전 15봉의 최저 Low** (consolidation low).
- 매일 intraday Low가 stop을 건드리면 즉시 청산 (Hard Stop).

### 왜 consolidation low인가?
TraderLion 도큐의 정확한 표현:
> *"Place a stop just below the pivot or the nearest low formed during the consolidation. This keeps your risk contained if the trade moves against you."*

**"Logical pivot stop"** — Wedge Pop의 정의가 "consolidation 후 첫 돌파"이기 때문에, consolidation의 최저점이 깨지면 패턴 자체가 무효가 됨. 즉:
- consolidation low = "이 가격 아래로 가면 내 가설이 틀렸다"
- 임의의 % stop이 아니라 **차트 구조에 근거한 stop**.

### 결과적으로 risk_per_share가 종목마다 다른 이유
- 변동성 큰 종목: consolidation 폭이 넓음 → stop이 멀음 → 작은 포지션
- 변동성 작은 종목: consolidation 폭이 좁음 → stop이 가까움 → 큰 포지션
- → **자동으로 변동성에 비례한 사이징**.

### 코드 위치
- Detector가 stop 계산: [`wedge_pop.py:103-105` `_build_signal`](../../../backtester/pattern/adapters/wedge_pop.py)
- Strategy가 hard stop 발동: [`wedgepop_strategy.py:213` `_find_exit`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

---

## 8. 익절/청산 룰 4가지 (Exit Conditions)

매 봉의 데이터(Low, Close, ema_trail, atr)를 보고 **이 순서대로** 체크. 가장 먼저 발동된 게 청산 가격.

### Exit 1: Hard Stop (손절)
- **조건**: `Low[i] ≤ stop_loss`
- **청산 가격**: `stop_loss` (intraday fill 가정 — 보수적으로 stop 가격에 정확히 체결됐다고 봄)
- **의미**: Wedge Pop 가설이 깨졌다 = consolidation low가 무너졌다 = 패턴 실패.
- **코드**: [`wedgepop_strategy.py:213-214`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

### Exit 2: Exhaustion Extension (과열 익절, Sell into Strength)
- **조건** (둘 중 하나라도 만족):
  - **% 트리거**: `(close − ema_trail) / ema_trail ≥ extension_pct` (기본 15%)
  - **ATR 트리거**: `close − ema_trail ≥ ATR(14) × extension_atr_mult` (기본 2.5)
- **청산 가격**: `close` (그날 종가)
- **의미**: 가격이 10 EMA 대비 너무 멀어졌다 = mean reversion 임박. 강세에 팔자.
- **TraderLion 도큐 원문**:
  > *"The price extends far above its moving averages, creating a gap or 'air' between the stock and its 10-day EMA. ... Sell a portion of your shares as the price extends far above the moving averages. This helps protect against a sharp reversal."*

#### 💡 잠깐, ATR이 뭐야?

**ATR (Average True Range) = 이 종목이 평균적으로 하루에 얼마씩 움직이는지를 $로 나타낸 값.** 한 마디로 **"종목별 일일 변동성"**.

| 종목 | ATR(14) ≈ | 한 줄 해석 |
|---|---|---|
| KO (코카콜라) | $0.80 | "하루에 $0.80 정도만 움직이는 얌전한 종목" |
| AAPL | $3 | "하루에 $3 정도 움직임" |
| TSLA | $15 | "하루에 $15 움직이는 변동성 큰 종목" |
| SMCI | $30+ | "하루에 $30 이상 출렁이는 폭주 종목" |

**왜 단순히 (High − Low)가 아니라 "True Range"인가?**
갭 업/다운이 있는 날에는 봉 자체의 폭(High − Low)이 진짜 변동을 못 잡음. 예를 들어:

```
어제 종가:  $100
오늘 시가:  $95   ← $5 갭다운
오늘 High:  $96
오늘 Low:   $94
```

- (High − Low) = $96 − $94 = **$2** ← 봉만 보면 거의 안 움직인 것처럼 보임
- 하지만 어제 종가 기준으론 **$6** 떨어진 거 ($100 → $94 최저)

**True Range는 이 갭까지 포함**해서 셋 중 가장 큰 값을 사용:

```
TR = max(
    오늘 High − 오늘 Low,           # = $2
    |오늘 High − 어제 Close|,       # = $4
    |오늘 Low − 어제 Close|         # = $6  ← 이게 채택
) = $6
```

**ATR(14)** = 최근 14봉의 True Range를 (지수)이동평균한 값.
→ **"갭까지 포함해서 본 진짜 일일 변동폭의 평균"**

**📌 본 strategy에서 ATR이 하는 역할**

> "이 종목이 보통 하루 X만큼 움직이는데, 지금은 10 EMA보다 `X × 2.5`만큼 위에 있다 = 평균 2.5일치 움직임을 단번에 갔다 = 너무 빠르다 → 익절."

수식: `임계값 = ATR(14) × extension_atr_mult`

| 종목 | ATR(14) | × 2.5 | "10 EMA보다 이만큼 위면 익절" |
|---|---|---|---|
| KO | $0.80 | $2.00 | 10 EMA + $2 |
| AAPL | $3 | $7.50 | 10 EMA + $7.5 |
| TSLA | $15 | $37.50 | 10 EMA + $37.5 |

→ % 기준은 한 숫자로 모든 종목을 묶어버리지만, ATR 기준은 **종목별 평균 변동성에 자동으로 비례한 익절선**을 만들어 줌. 얌전한 KO와 폭주 SMCI에 똑같은 임계값을 쓰지 않는다는 게 핵심.

#### 왜 두 가지 트리거?
종목별 변동성이 천차만별이라 단일 % 기준은 한쪽으로만 맞춰짐:

| 종목 타입 | % 트리거 (15%)만 | ATR 트리거 (×2.5)만 | 두 트리거 OR |
|---|---|---|---|
| 우량주 (KO, JNJ) | 잘 동작 | 거의 발동 안 됨 | OK |
| 고변동성 성장주 (SMCI, MSTR) | 너무 자주 발동 | 잘 동작 | OK |
| 저가주 (≤$10) | 잘 동작 | ATR 자체가 작아 둔감 | OK |

→ **OR 조합**으로 어떤 종목 타입이든 합리적으로 대응.

#### 예시
**케이스 1**: ema_trail = $100, ATR = $4
- close = $113 → distance = $13
- % 트리거: 13/100 = 13% < 15% ❌
- ATR 트리거: 13 ≥ 4 × 2.5 = 10 ✅ → 청산

**케이스 2**: ema_trail = $100, ATR = $1.5 (저변동성)
- close = $116 → distance = $16
- % 트리거: 16/100 = 16% ≥ 15% ✅ → 청산
- ATR 트리거: 16 ≥ 1.5 × 2.5 = 3.75 ✅ (둘 다 발동)

- **코드**: [`wedgepop_strategy.py:216-224`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

### Exit 3: EMA Trail (Wedge Drop / EMA Crossback 실패)
- **조건**: `close < ema_trail` (10 EMA)
- **단**: `trail_after_profit=True`(기본)일 때는 **현재 close가 entry보다 수익권**일 때만 발동.
- **청산 가격**: `close`
- **의미**: 추세 transition. 10 EMA 아래로 close 마감 = 매수자 control 상실 = "EMA Crossback이 실패했다 또는 Wedge Drop이 시작됐다" → 빠져나옴.

#### 왜 `trail_after_profit`이 필요한가?
TraderLion 도큐의 경고:
> *"The breakout candle often retests the EMAs. ... Tight stops can shake out at retest."*

진입 직후 1~3봉은 돌파 캔들의 retest로 EMA를 다시 잠깐 찍는 경우가 매우 흔함. 이때 trail stop이 활성화돼 있으면:
- 종가가 EMA 아래로 살짝 내려감 → 청산
- 그러나 다음날 다시 강하게 위로 → 놓침 (whipsaw)

해결: **수익권에 들어가기 전까지는 hard stop(consolidation low)만으로 보호**, 수익권에 들어가서야 trail 활성화.

```python
if close < ema:
    if not self.trail_after_profit or close > entry_price:
        return close, i  # 청산
```

- `trail_after_profit=False` 옵션도 제공 — 더 공격적인 trail을 원하는 경우.
- **코드**: [`wedgepop_strategy.py:227-229`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

### Exit 4: Time Stop
- **조건**: 진입 후 `max_holding_days`(기본 60)일 경과.
- **청산 가격**: 마지막 봉의 종가.
- **의미**: 위 3개 조건 중 어느 것도 발동 안 한 채로 너무 오래 횡보 → 자본을 묶어 두지 말고 빼기. 다른 setup에 자본을 재배치.
- **코드**: [`wedgepop_strategy.py:232`](../../../backtester/strategy/adapters/wedgepop_strategy.py)

### Exit 우선순위 다이어그램

```
매 봉마다:
  ┌─ Low ≤ stop?
  │     └─ YES → STOP 청산 (손실)
  │
  ├─ (close − ema)/ema ≥ pct?  OR  (close − ema) ≥ atr × mult?
  │     └─ YES → EXHAUSTION 청산 (수익)
  │
  ├─ close < ema_trail?  AND  (수익권 OR trail_after_profit=False)?
  │     └─ YES → TRAIL 청산 (대개 수익, 가끔 손실)
  │
  └─ 다음 봉으로

마지막 봉(entry+max_holding_days)에 도달:
  └─ TIME STOP 청산 (마지막 종가)
```

---

## 9. 모든 파라미터 레퍼런스

### `WedgepopStrategy` 생성자 파라미터

| 파라미터 | 기본값 | Streamlit Sidebar 노출 | 의미 |
|---|---|---|---|
| `market_data` | (필수 주입) | ❌ | OHLCV fetch port (`MarketDataPort`) |
| `detector` | (필수 주입) | ❌ | 패턴 detector port (`PatternDetector`). Wedge Pop 검출용 |
| `ema_trail` | `10` | ❌ (현재 하드코딩) | Trail / Exhaustion 계산에 쓰는 EMA 기간 |
| `atr_period` | `14` | ❌ | ATR 계산 기간 |
| `extension_pct` | `0.15` | ✅ ("Exhaustion % above 10 EMA") | Exhaustion 트리거 1: % 기준 |
| `extension_atr_mult` | `2.5` | ✅ ("Exhaustion ATR multiplier") | Exhaustion 트리거 2: ATR 배수 |
| `trail_after_profit` | `True` | ❌ | True면 수익권에서만 EMA trail 활성화 |
| `require_gap_up` | `False` | ✅ ("Require gap-up confirmation") | True면 T+1 open > T close 일 때만 진입 |

### `StrategyConfig` (실행 시 주입)

| 필드 | 기본값 | Sidebar | 의미 |
|---|---|---|---|
| `ticker` | (필수) | ✅ | 종목 코드 |
| `start_date` | (필수) | ✅ | 데이터 시작일 |
| `end_date` | (필수) | ✅ | 데이터 종료일 |
| `pattern_name` | (필수) | ❌ | 신호 이름. Wedgepop은 항상 `"wedge_pop"` |
| `initial_capital` | `100,000` | ✅ | 시작 자본 ($) |
| `risk_per_trade` | `0.02` | ✅ ("Risk per Trade %") | 거래당 위험 (자본 비율) |
| `max_holding_days` | `60` | ✅ ("Max Holding Days") | Time stop 발동까지 최대 보유 일수 |
| `pattern_params` | `{}` | ❌ | (예약) detector 파라미터 override용 |

### `WedgePopDetector` 파라미터 (참고용)
이 strategy는 detector를 외부에서 주입받아 쓰지만, detector 자체의 파라미터도 알아두면 튜닝에 유용.

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `lookback` | `15` | Consolidation 검사 / 손절 계산 lookback 봉 수 |
| `ema_fast` | `10` | Consolidation 기준 EMA |
| `ema_slow` | `20` | 돌파 확인용 보조 EMA |
| `consolidation_pct` | `0.6` | lookback 중 ema_fast 아래로 마감해야 할 봉의 비율 |
| `breakout_pct` | `0.015` | 돌파 강도 임계값 (1.5%) |

---

## 10. 사용 예시

### Streamlit 페이지로 (가장 쉬움)
```bash
docker compose up -d --build backtester
# 브라우저로 http://localhost:8501/strategy 접속
```
사이드바에서 `ticker`, 기간, risk 파라미터를 설정하고 **Run Strategy** 클릭. 차트에 진입(BUY)·손절(STOP)·익절(SELL) 마커가 $금액과 함께 표시됨.

### 코드로 직접 (스크립트, 노트북)
```python
from datetime import date

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig

market_data = YFinanceAdapter()
strategy = WedgepopStrategy(
    market_data=market_data,
    detector=WedgePopDetector(),
    extension_pct=0.15,
    extension_atr_mult=2.5,
    trail_after_profit=True,
)

config = StrategyConfig(
    ticker="ELF",
    start_date=date(2023, 9, 1),
    end_date=date(2024, 3, 1),
    pattern_name="wedge_pop",
    initial_capital=100_000,
    risk_per_trade=0.02,
    max_holding_days=60,
)

result = strategy.run(config)

print(f"Trades: {result.performance.total_trades}")
print(f"Win rate: {result.performance.win_rate:.0%}")
print(f"Total return: {result.performance.total_return_pct:.2%}")
for t in result.performance.trades:
    print(t)
```

### 미리 fetch한 df로 (페이지가 쓰는 방식)
```python
df = market_data.fetch_ohlcv("ELF", date(2023, 9, 1), date(2024, 3, 1))
result = strategy.execute(df, config)  # fetch 안 함
```

### 합성 데이터로 단위 테스트
[`backtester/tests/test_wedgepop_strategy.py`](../../../backtester/tests/test_wedgepop_strategy.py)에 4가지 exit 경로(hard stop / exhaustion / EMA trail / time stop)를 각각 발동시키는 fixture가 있음. 새 exit 룰 추가 시 같은 패턴으로 fixture 추가.

---

## 11. 한계와 의도적으로 안 한 것

### 단일 포지션
- 동시에 여러 종목 보유 X. 한 종목씩만.
- 멀티 종목 백테스트가 필요하면 strategy를 여러 번 돌리고 결과를 합치는 별도 orchestrator 필요.

### 분할 매도 (Scale-out) 미구현
- 도큐는 "exhaustion에서 일부만 팔고 나머지는 trail" 같은 partial exit를 권장.
- 현재 `Trade` 모델이 단일 entry/exit 1쌍이라 분할 매도를 표현 못 함.
- 추가하려면 `Trade` 모델 확장 + exit 로직 변경 필요.

### 추가매수 (Pyramiding) 미구현
- "EMA Crossback에서 추가" / "Base n' Break에서 추가" 룰 미적용.
- 단일 진입 → 단일 청산.

### Multi-timeframe 미사용
- Detector는 `weekly_df`, `monthly_df`를 받을 수 있게 설계돼 있지만 본 strategy는 daily만 사용.
- 도큐의 "주봉/월봉으로 컨펌" 룰을 적용하면 신호 수가 줄고 win rate가 올라갈 가능성.

### 슬리피지 / 수수료 미반영
- 진입가 = 다음날 정확한 Open
- Hard stop 청산가 = 정확한 stop_loss
- 실거래에선 슬리피지 + 수수료로 1~2% 추가 비용 발생 가능.

### 포지션 청산이 종가 가정
- Exhaustion / EMA Trail / Time stop 모두 그날 close에 청산 가정.
- 실제로는 신호 발생 후 익일 open 청산이 더 현실적일 수 있음 (look-ahead 회피).
- Detector가 종가 기준으로만 신호를 만들기 때문에 일관성 차원에서 종가 청산 사용 중.

### 시장 환경 필터 없음
- bull/bear regime, 지수 trend, sector strength 등 broader market context 미반영.
- 도큐는 "market이 Wedge Pop일 때 leading 종목들이 base n' break" 같은 top-down 흐름을 강조.

---

## 12. 참고

### 코드 위치
- 전략: [`backtester/strategy/adapters/wedgepop_strategy.py`](../../../backtester/strategy/adapters/wedgepop_strategy.py)
- 도메인 모델: [`backtester/strategy/domain/models.py`](../../../backtester/strategy/domain/models.py) (`Trade`, `StrategyPerformance`, `StrategyResult`)
- 포트: [`backtester/strategy/domain/ports.py`](../../../backtester/strategy/domain/ports.py) (`StrategyRunnerPort`)
- Detector: [`backtester/pattern/adapters/wedge_pop.py`](../../../backtester/pattern/adapters/wedge_pop.py)
- 차트: [`backtester/visualization/adapters/plotly_charts.py`](../../../backtester/visualization/adapters/plotly_charts.py) (`build_candlestick_with_trades`)
- Streamlit 페이지: [`backtester/pages/strategy.py`](../../../backtester/pages/strategy.py)
- 테스트: [`backtester/tests/test_wedgepop_strategy.py`](../../../backtester/tests/test_wedgepop_strategy.py)

### 외부 출처
- **TraderLion / Oliver Kell**: "Cycle of Price Action" 강의. [`traderlion.docx`](../../../traderlion.docx) (.gitignore)
- **Van Tharp**: *Trade Your Way to Financial Freedom* — fixed-fractional risk 룰의 고전 레퍼런스
- **Mark Minervini**: *Trade Like a Stock Market Wizard* — pivot stop, position sizing 비슷한 철학

### 헥사고날 아키텍처 위치
```
strategy/                    ← 본 전략이 사는 도메인
  domain/
    models.py               (Trade, StrategyPerformance, StrategyResult)
    ports.py                (StrategyRunnerPort)
  adapters/
    wedgepop_strategy.py    ★ 본 전략 (self-contained)
    runner.py               (legacy bridge: backtest → strategy)

의존성:
  WedgepopStrategy
    → MarketDataPort       (data/domain/ports.py)
    → PatternDetector       (pattern/domain/ports.py)
    → strategy/domain/*     (자기 도메인)
    ✗ backtest 의존 0       (도메인 격리)
```
`backtest` 도메인과 완전 분리. 실행 / 사이징 / exit / 성능 집계 모두 strategy 내부에서 해결.
