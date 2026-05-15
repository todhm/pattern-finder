# Bull Flag — 영상 리서치 노트 (2026-05)

두 YouTube 영상에서 Bull Flag 운용 룰을 추출. 기존 `ross_cameron_bull_flag_deep_dive.md` 는 Ross Cameron 의 *low-float small-cap* 셋업이고, 본 노트는 다음 두 영상을 정리한다:

| ID | 출처 (추정) | 길이/분량 | 성격 |
|---|---|---|---|
| `x8w_Spm7rmE` | Chart Mill — "5 Common Bull Flag Mistakes" | 짧음 (≈5 KB transcript) | 교육영상 — 흔한 실수 5종 |
| `SNjtH42aCuk` | 익명 scalper — Tesla 실전 라이브 코멘트 | 김 (≈19 KB transcript) | intraday 라이브 케이스 — 풀백 진입 매커닉 |

원문은 `docs/research/transcripts/{x8w_Spm7rmE,SNjtH42aCuk}.txt`. 캡션 dedupe 워크플로우는 [`docs/research/youtube-strategy-extraction.md`](../research/youtube-strategy-extraction.md).

---

## 1. Chart Mill — 흔한 실수 5종 (`x8w_Spm7rmE`)

스윙/swing-trader 관점의 daily-chart Bull Flag 가이드. 각 실수를 "감지기/전략에 어떻게 적용 가능한지" 와 짝지어 정리.

### 실수 1 — **돌파 전 진입 (early entry)**

> "Just because it looks like a bull flag doesn't mean it'll play out like one. Price could just keep drifting sideways or even roll over."

- **룰**: 깃대(flag pole) 위에서 *consolidation* 만 봐서는 안 됨. **strong breakout candle closing above the upper flag line + rising volume** 까지 기다린다.
- **코드 매핑**: `BullFlagDetector` 가 이미 break candle close > flag upper trendline 을 요구한다. `min_breakout_volume_ratio` 와 동일 의미.

### 실수 2 — **거래량 무시**

> "A proper bull flag should start with a strong volume surge on the flag pole, then decline during the pullback. If the breakout doesn't come with a new volume spike, that's a clear warning sign."

- **룰**: 3-stage 거래량 형태가 정상.
  1. Pole — **거래량 spike**
  2. Flag — **거래량 감소**
  3. Breakout — **거래량 재spike**
- **코드 매핑**: `pole_volume_ratio`, `flag_volume_decay`, `breakout_volume_ratio` 3개 게이트가 모두 통과해야 함. 현재 `BullFlagDetector` 는 pole + breakout 두 단계만 본다 — **flag 구간 거래량 감소** 게이트 추가 검토 (false-positive 줄이는 효과 클 듯).

### 실수 3 — **너무 깊은 풀백**

> "If the stock retraces more than 50% of the initial move or drops below key moving averages like the 20 EMA, you're not looking at a tight flag anymore."

- **룰**: 풀백 깊이 한도 = **pole 의 50%** 또는 **20 EMA 하방 이탈 금지**.
- **코드 매핑**: `max_pullback_pct = 0.5` 가 있으면 직접 매칭. 추가로 **20 EMA support** 게이트 — flag low 가 20 EMA 위에 있어야 함 (또는 violation 캔들 수 ≤ N).

### 실수 4 — **flag 가 너무 길다**

> "If a stock sits in a flag for too long, like more than 10 to 15 candles on a daily chart, the setup weakens."

- **룰**: 일봉 기준 **10~15 캔들 이내** 컨솔. 그 이상이면 모멘텀 fade.
- **코드 매핑**: `max_flag_bars` 파라미터. 현재 default 가 더 큰 값이면 10~15 로 좁히고 backtest 재실행 권장. 인트라데이(15m)에서는 비례 환산: 1d ≈ 26 × 15m → 약 *260~390 × 15m bars* — 너무 큼. 인트라데이는 영상 외 영역이므로 별도 튜닝.

### 실수 5 — **시장 역행**

> "Unless your stock is showing extreme relative strength, the odds are stacked against you."

- **룰**: 시장 약세 시에는 **relative strength 가 극단적으로 강한 종목** 만 거래.
- **코드 매핑**: 두 가지로 분해:
  - **market regime gate** — SPY / QQQ 가 N-day MA 위에 있을 때만 신호 통과
  - **RS gate** — `(stock 20d return) − (SPY 20d return)` 가 양수 또는 상위 percentile
- 현재 `BullFlagStrategy` 가 어느 정도 갖고 있는지 확인 필요. Wedge Pop 측의 `market regime` 필터는 이미 존재.

---

## 2. Tesla intraday scalper (`SNjtH42aCuk`)

다른 화자가 Tesla 실전을 라이브 해설. 인트라데이 *bull flag 풀백* 진입 매커닉이 매우 구체적이다.

### 2.1 셋업 전제 — "모든 게 정렬되어야 한다"

영상 핵심: 단독 차트의 bull flag 형태만으로는 부족. 다음 4개 layer 가 같은 방향이어야 한다.

| Layer | 조건 (Tesla 예시) |
|---|---|
| **시장 indices** | ES / NQ 둘 다 갭업 + 주요 resistance 돌파 — chop 환경 아님 |
| **종목 daily** | 2주 sideways → 금요일 igniting candle(거래량 spike) → 갭업 + short-term MA 재돌파 + resistance 돌파 |
| **종목 personality** | Tesla 같은 high-momentum 종목 — 한 방향으로 가면 멀리 간다. Apple/AMD/NVDA 와 다른 ATR 프로필 |
| **Pre-market** | pre-market high (≈ 260) 위 hold — *break* 가 아니라 *hold* |

> 인용: "If you just trade a random bull flag every single day, it's going to have a very low probability win rate. But when you get those unique instances where the market has a strong catalyst…"

### 2.2 진입 매커닉 — "green take red at the 10 EMA"

영상에서 두 번 반복된 entry 트리거. 단계:

1. **Opening drive** — 강한 첫 push (Tesla 256 → 263)
2. **첫 풀백 대기** — flag 형성. 풀백은 *controlled selling*, 거래량 감소
3. **10 EMA 지지 확인** — 풀백이 10 EMA 위에서 멈춰야 함. 10 EMA 아래로 *aggressive sell-off* 면 reversal 가능성, 스킵
4. **Downside wick + 첫 green-take-red 캔들** — 10 EMA 근처에서 (a) low 가 EMA 하방을 wick 으로 찍고 (b) close 가 직전 캔들 high 위로 → "green takes red"
5. **진입**: green-take-red 캔들 *high 돌파* 시 long
6. **손절**: green-take-red 캔들 *low* (또는 10 EMA 직하). Tesla 의 경우 ≈ $1 risk (높은 ATR 종목 기준)
7. **익절**: scalp — $1.5~$2 면 절반 이상 청산. all-day hold 아님

### 2.3 두 번째 bull flag — 동일 패턴 반복

영상은 같은 차트에서 두 번 같은 셋업을 잡음. 핵심:

- 첫 bull flag 후 강한 push → 다시 10 EMA 풀백 → 같은 green-take-red
- "buyers in control until the market turns" — 시장이 같이 가는 동안에는 setup 반복 가능
- 두 번째는 첫 번째보다 R/R 약간 나쁨 (이미 멀리 옴) → 보수적으로 더 작은 사이즈

### 2.4 earnings season 보너스

> "When you get stocks that are in earnings season and they are gapping up… they can have that gap and go continuation just because they are now trading on elevated volume because they have a catalyst."

- **catalyst (실적/뉴스) + gap-up** = bull flag 의 모멘텀 fuel
- 코드 매핑: 인트라데이 bull flag 스캐너에 *earnings calendar gate* 추가 가능 (D-1 ~ D+2 ?)

---

## 3. 두 영상을 합친 시사점 — 코드/필터에 반영할 후보

| 후보 | 출처 | 우선순위 | 메모 |
|---|---|---|---|
| Flag 구간 **거래량 감소** 게이트 추가 | Chart Mill #2 | 🟢 높음 | 현재 detector 미적용 추정. false-positive 가장 크게 줄일 듯 |
| **20 EMA 지지** 게이트 (flag low > 20 EMA) | Chart Mill #3 | 🟢 높음 | 깊은 풀백 reject. `max_pullback_pct=0.5` 와 OR/AND 조합 |
| `max_flag_bars` 일봉 **10~15** 로 좁히기 | Chart Mill #4 | 🟡 중간 | 현재 default 확인 후 sweep |
| **Market regime** + **Relative Strength** gate | Chart Mill #5 | 🟢 높음 | Wedge Pop 측 코드 재사용 |
| **10 EMA pullback** + **green-take-red** 트리거 (인트라데이) | Tesla scalper | 🟡 중간 | 인트라데이 entry candle 정의를 명확히. 이미 wick_play 와 유사 매커닉 |
| **Earnings/catalyst gate** | Tesla scalper | 🔵 낮음 | 데이터 소스 (yfinance earnings_dates 또는 EODHD) 확인 필요 |
| Pre-market high **hold** (인트라데이) | Tesla scalper | 🔵 낮음 | 데이터 — pre-market 바 fetch 가능한지 (yfinance 제한, EODHD 가능) |

### 합치는 식

기존 `bull_flag_filter_audit.md` 의 3분류(🟢확정/🟡애매/🔵추가)에 매핑하면:

- 🟢 **확정** 으로 격상 가능: flag 거래량 감소, 20 EMA 지지, max_flag_bars 10~15
- 🟡 **애매** 유지: green-take-red 정의, RS 임계값, earnings 윈도우 폭
- 🔵 **추가** 영역: pre-market hold, catalyst gate

---

## 4. 한계 / 영상이 답하지 않는 것

- **R/R 룰** — Chart Mill 영상은 익절/손절 비율 언급 없음. Tesla scalper 는 "1점 risk, $1.5~$2 익절" 한 사례만.
- **포지션 사이징** — 두 영상 모두 % risk per trade, max concurrent 미언급.
- **백테스트 메트릭** — Chart Mill 은 "success rate dramatically improve" 라는 말 외에 수치 없음. Tesla scalper 는 "random 매수는 low win rate" 만.
- **인트라데이 정확한 timeframe** — Tesla scalper 가 2분 캔들 언급 ("second two-minute candle"), 우리 구현은 15m 위주. 2m 까지 내려가야 같은 매커닉이 나오는지 별도 검증 필요.
- **공매도 / 베어 flag** — 영상 1 은 시장 약세에 *거래 자체를 안 함*, 영상 2 는 long-only scalper. 양방향은 다른 영상으로 보강.

---

## 5. Transcripts

- `docs/research/transcripts/x8w_Spm7rmE.txt` (5,322 chars)
- `docs/research/transcripts/SNjtH42aCuk.txt` (19,015 chars)

dedupe 워크플로우 / 키프레임 추출 방법은 `docs/research/youtube-strategy-extraction.md` 참조.
