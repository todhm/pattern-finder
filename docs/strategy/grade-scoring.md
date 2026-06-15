# Grade Scoring — 현재 방식 분석 + 개선안

`us_stock_grade.final_score` (0~100) 가 어떻게 산출되는지 전체 흐름 + 각 구성요소
의미 + 현 시스템의 검증된 결함 + 개선 방향.

소스: `alphafolio_quant/us/us_main.py` (orchestrator) + `us_value_factor.py` /
`us_quality_factor.py` / `us_momentum_factor.py` / `us_growth_factor.py` (4 base
factors) + `us_market_regime.py` (HMM regime) + `us_event_engine.py` (event
modifier) + `us_factor_interactions.py` (interaction + pattern adjustment) +
`weight_adjustments.py` / `us_sector_dynamic_weights.py` (regime/sector weight).

---

## 1. Final Score 산출 흐름 (한 종목 한 시점)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1 — HMM regime detection (전체 SPY 시계열 기반)                │
│   → REGIME ∈ {BULL, NEUTRAL, BEAR} → factor weights                │
│   BULL:    G 35% / M 25% / Q 20% / V 20%                            │
│   NEUTRAL: G 28% / Q 27% / V 25% / M 20%                            │
│   BEAR:    Q 40% / V 30% / G 20% / M 10%                            │
├─────────────────────────────────────────────────────────────────────┤
│  Step 2 — 4 factor 산출 (각 0~100, 종목 단위)                       │
│   value_score    (RV1~RV6 의 가중합)                                │
│   quality_score  (MQ1~MQ6, HC1~HC4 (Healthcare))                    │
│   momentum_score (EM1~EM8)                                          │
│   growth_score   (FG1~FG6, NQ1~NQ4 (NASDAQ))                        │
│   ※ 섹터별로 sub-indicator 활성/비활성. 예: RV5 는 Utilities/       │
│      Financial Services 한정, EM1 은 Energy/Basic Materials 한정.   │
├─────────────────────────────────────────────────────────────────────┤
│  Step 3 — Sector dynamic weight adjustment                          │
│   sector_weights.get_combined_weights(base_weights, exchange,       │
│       sector) → 기본 가중치에 섹터 보정 (Tech, Healthcare, Financial │
│       등). 예: Tech 종목은 G/M ↑, V ↓.                              │
├─────────────────────────────────────────────────────────────────────┤
│  Step 4 — base_score 계산                                            │
│   base_score = G·wG + M·wM + Q·wQ + V·wV                            │
├─────────────────────────────────────────────────────────────────────┤
│  Step 5 — interaction_score 산출 (USFactorInteractions)              │
│   I1 G×Q geom mean + synergy bonus (G≥70 & Q≥70)        (가중 30%)  │
│   I2 G×M harmonic mean                                  (가중 25%)  │
│   I3 Q×V geom mean (가치+퀄리티 콤보)                   (가중 20%)  │
│   I4 M×Q harmonic mean                                  (가중 15%)  │
│   I5 All-Factor Agreement (4 factor 동시 ≥70)           (가중 10%)  │
│   → interaction_score = Σ I·w                                       │
│   → conviction_score = I5.score (별도 출력)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Step 6 — total_score 합성                                           │
│   total_score = base_score × 0.70 + interaction_score × 0.30        │
├─────────────────────────────────────────────────────────────────────┤
│  Step 7 — Cash Runway Penalty (Healthcare 만)                       │
│   small biotech: cash / burn rate 가 12 개월 미만 → total_score -N  │
├─────────────────────────────────────────────────────────────────────┤
│  Step 8 — Event Modifier (Pass-B 만, with_event_modifier=True)      │
│   earnings_mod + options_mod + insider_mod + gex_mod + news_mod     │
│   합계 clamp [-20, +20] → total_score += event_modifier             │
│   → total_score = clamp(total_score + event_modifier, 0, 100)       │
├─────────────────────────────────────────────────────────────────────┤
│  Step 9 — Pattern Adjustment (apply_pattern_adjustment)              │
│   V-Q+M-G+ (Sell Pattern): cap, mid/micro 에서 -10~-15              │
│   V+Q+M+G+ (Buy Pattern): +3~+5                                     │
│   V-Q-M-G- (Strong Sell): mid/micro 에서 score=25 강제              │
│   ※ Mega/Large 의 strong sell 은 적용 안 함 (96% 가 오르더라)       │
├─────────────────────────────────────────────────────────────────────┤
│  Step 10 — factor_combination_bonus (us_main:_calculate_*)          │
│   Quality Compounder (Q≥70 & G≥70): +15                             │
│   Deep Value Turnaround (V≥75 & Q≥60 & M<40): +20                   │
│   GARP (G≥70 & V≥50): +10                                           │
│   Momentum Speculation Penalty (V<30 & M≥80): -10                   │
│   All-around Strong (모두 ≥65): +15                                 │
│   → clamp [-20, +30]                                                 │
│   ※ 이 보너스는 us_stock_grade.factor_combination_bonus 컬럼에      │
│      저장만 됨 — total_score 에 다시 더하지 않음. 진단용.           │
├─────────────────────────────────────────────────────────────────────┤
│  Step 11 — Grade 매핑 (절대 점수)                                    │
│   ≥85 강력 매수 / ≥75 매수 / ≥65 매수 고려 / ≥55 중립               │
│   ≥45 매도 고려 / ≥35 매도 / <35 강력 매도                          │
└─────────────────────────────────────────────────────────────────────┘
```

전체 식 (1 line):

```
final_score = clamp(
    [(G·wG + M·wM + Q·wQ + V·wV) × 0.7 + interaction_score × 0.3]
    − cash_runway_penalty
    + event_modifier
    +/− pattern_adjustment_delta,
  0, 100)
```

`factor_combination_bonus` 는 출력 컬럼이지만 `final_score` 에 반영되지 않음.

---

## 2. 각 Factor 의 Sub-Indicator (요약)

### Value (V) — `us_value_factor.py`

| ID | 이름 | 가중치 | 적용 섹터 | 의미 |
|----|------|------:|----------|------|
| RV1 | PEG Ratio | 25% | 전체 | PE / 성장률 |
| RV2 | Forward PE Relative | 20% | 전체 (역방향) | 섹터 대비 Forward PE — IC -0.139 이라 reversed |
| RV3 | EV/Revenue to Growth | 20% | 전체 | EV/Sales 를 성장률로 정규화 |
| RV4 | Rule of 40 | 15% | 전체 (Tech/SaaS) | 성장률 + 영업마진 ≥ 40% |
| RV5 | FCF Yield Adjusted | 10% | **Utilities / Financial Services 한정** | 무형자산 + R&D 보정 FCF / Market Cap |
| RV6 | Price to Target | 10% | 전체 | 애널리스트 목표가 대비 |

### Quality (Q) — `us_quality_factor.py`

| ID | 이름 | 가중치 | 의미 |
|----|------|------:|------|
| MQ1 | Gross Margin | 20% (Healthcare 12%) | 섹터 대비 총이익률 |
| MQ2 | ROIC | 20% (HC 12%) | 투하자본수익률 |
| MQ3 | Operating Leverage | 15% | 매출 증가에 따른 영업이익 탄력성 |
| MQ4 | Earnings Quality | 15% | Operating CF / Net Income (현금 vs 발생주의) |
| MQ5 | Balance Sheet | 15% | D/E + 유동성 |
| MQ6 | Margin Trend | 15% | 마진 시계열 개선/악화 |
| HC1~HC4 | Healthcare-specific | 40% (Healthcare 만) | Cash runway 등 |

### Momentum (M) — `us_momentum_factor.py`

| ID | 이름 | 가중치 | 적용 |
|----|------|------:|------|
| EM1 | Risk-Adjusted Momentum | 5% | **Energy/Basic Materials 한정**, IC -0.030 → reversed |
| EM2 | Sector Relative Strength | (disabled) | IC -0.078, 전체 disabled |
| EM3 | EPS Estimate Revision | 20% | 컨센서스 EPS 상향/하향 |
| EM4 | Revenue Estimate Revision | 20% | 컨센서스 매출 상향/하향 |
| EM5 | Volume Confirmation | 적용 안함 | **Basic Materials 한정** |
| EM6 | Earnings Momentum | 15% | 어닝 서프라이즈 누적 |
| EM7 | (disabled, 단기 target price) | — | 장기 예측 부적합 |
| EM8 | Long-term Price Momentum | 15% | **IBD RS Style 252일** (가격수익률 기반) |

### Growth (G) — `us_growth_factor.py`

| ID | 이름 | 가중치 | 적용 |
|----|------|------:|------|
| FG1 | Revenue Growth | 20% (NASDAQ 12%) | YoY 매출 성장 |
| FG2 | EPS Growth | 20% (NASDAQ 12%) | YoY EPS 성장 |
| FG3 | Forward Growth | 20% (NASDAQ 15%) | 애널리스트 예상 |
| FG4 | Growth Consistency | 15% (NASDAQ 8%) | 분기별 변동성 |
| FG5 | Revenue Acceleration | 15% (NASDAQ 10%) | 최근 vs 과거 평균 성장 |
| FG6 | Profit Growth | 10% (NASDAQ 8%) | 순익 성장 |
| NQ1~NQ4 | NASDAQ-specific | 35% (NASDAQ 한정) | 기관 보유 / 애널리스트 리비전 / 옵션 풋콜비 |

---

## 3. 핵심 문제점 — 실측 검증 결과

DB에 누적된 **319k grade rows × 919 dates × 4,410 symbols** (2019-01-02 ~
2026-05-29) 로 forward return / hit rate 검증.

### 3.1 시스템 alpha 가 *inverted* — 강력매수가 가장 안 맞음

| 등급 | n | avg score | median 1m | hit 1m | median 12m | **hit 12m** |
|------|---|-----------|----------|--------|-----------|------------|
| 강력 매수 | 11,605 | 89.0 | -0.98% | 47.3% | **-27.09%** | **30.9%** |
| 매수 | 48,474 | 79.0 | +0.32% | 50.7% | -1.93% | 48.7% |
| 매수 고려 | 96,474 | 69.8 | +0.40% | 51.0% | +4.22% | 52.6% |
| 중립 | 153,179 | 60.3 | +0.89% | 52.6% | +7.93% | 56.7% |
| 매도 고려 | 69,611 | 50.2 | -0.15% | 49.0% | **+11.36%** | 57.2% |
| 매도 | 42,466 | 40.7 | -1.82% | 44.4% | -5.39% | 45.4% |
| 강력 매도 | 17,477 | 30.0 | -2.97% | 42.6% | +3.23% | 52.0% |

**결정적 발견**:
- 강력 매수 종목의 **median 12 개월 수익률 = -27%** (절반 이상이 1년 후 27% 손실)
- 강력 매수 12 개월 hit rate = **30.9%** (coin flip 50% 보다도 낮음)
- 등급별 평균 수익률 순위가 **사실상 반전** (best=매도고려 +11.36%, worst=강력매수 -27.09%)
- ⇒ 점수 자체에 directional alpha 없음. 오히려 음(-)의 상관

### 3.2 backtest 의 +166% 가 점수 alpha 가 아닌 이유

- 룰 = top-3 / 20일 rebal. 짧은 holding period.
- 강력매수 pool 의 *단기 momentum-pop* 만 잡고 mean-reversion (1년 -27%) 오기 전 매도.
- 즉 **"점수가 미래를 예측한다"** 가 아니라 **"점수 + 빠른 rebal 이 variance 를
  harvest 한다"**. 점수 자체는 noise 또는 anti-signal.

### 3.3 데이터 결손이 점수를 오염 (별도 검증)

`MYRG / ASYS / TER` 케이스:
- value_score 의 sub-indicator (RV1~RV6) 가 PER/PEG/EV-Rev 데이터 없으면
  default 50점 부여 → 가치 평가 무력화
- growth_score 가 NQ4 (옵션 풋콜비) 한 지표만으로 94~99 점 만점에 가깝게 산출
- Step 10 의 "Momentum Speculation Penalty (V<30 & M≥80)" 가 default V=50 으로
  bypass 됨 → 모멘텀 chaser 종목에 페널티 안 들어감

### 3.4 entry_timing_score / price_position_52w 가 점수에 미반영

- `us_agent_metrics.py` 가 52주 고점 대비 위치 + score_trend_2w 로
  `entry_timing_score` 계산 → **별도 컬럼으로 저장만 됨**
- 52주 고점 87~95% 종목 = "Cautious (25점)" / "Risky (10점)" 신호가 있어도
  final_score 에는 들어가지 않음
- ⇒ MYRG (-2.9% from 52w high), ASYS (91.8% range), TER (87.1% range) 가 100점
  받는 이유

### 3.5 점수 캡 100 에서 saturation

- 5~10 개 종목이 final_score=100 으로 묶이면 top-3 선택이 사실상 임의
- backtest 의 결과 표본수가 매우 작음 (top-1 룰에서 4종목 / 1년 — NVTS 한
  종목이 +1,049% 로 평균 끌어올림)

---

## 4. 개선안

핵심 목표 — **directional alpha 회복** + **데이터 결손 종목 자동 제거** +
**점수 saturation 완화**.

### 4.1 (필수) Entry timing penalty 를 `final_score` 에 직접 반영

```python
# us_main.py:533 base_score 직후 추가
position_penalty = 0
if price_position_52w is not None:
    if price_position_52w >= 90:
        position_penalty = -10  # 52w 고점 근접 = 매수 위험
    elif price_position_52w >= 70:
        position_penalty = -3
    elif price_position_52w <= 30:
        position_penalty = +5   # 52w 저점 부근 = 진입 유리
total_score = max(0, total_score + position_penalty)
```

기대 효과: 강력매수 등급에서 high-position 종목 자동 강등. MYRG/ASYS/TER 같은
케이스가 매수~매수고려 등급으로 떨어짐.

### 4.2 (필수) 데이터 결손 종목의 점수 캡

```python
# 각 factor 의 sub-indicator 활성화 비율 (예: RV1~RV6 중 actually-computed 개수 / 총)
factor_coverage = {
    'value':    sum(d['has_data'] for d in value_v2_detail.values()) / len(...)
    'quality':  ...
    ...
}
min_coverage = min(factor_coverage.values())

if min_coverage < 0.30:
    # 4 factor 중 어떤 factor 가 30% 미만 sub-indicator 만 계산됐으면
    # final_score 를 max 70 으로 캡 (강력매수 불가)
    total_score = min(total_score, 70)
```

기대 효과: 펀더멘털 결손 (income statement 없음) 인 종목이 100점 받지 못함.

### 4.3 (필수) `factor_combination_bonus` 의 Pattern 4 임계 완화

```python
# 기존: V<30 & M>=80 → -10
# 변경: V<50 & M>=80 → -15  (default V=50 인 결손 종목도 잡힘)
if value < 50 and momentum >= 80:
    bonus -= 15
```

그리고 **`factor_combination_bonus` 를 `final_score` 에 실제 반영** (현재
저장만 하고 점수엔 안 더함):

```python
total_score = clamp(total_score + factor_combination_bonus, 0, 100)
```

### 4.4 (추천) Momentum factor 의 EM8 weight 축소

- EM8 (252일 가격 수익률) 이 momentum 의 사실상 단일 driver
  (sector-specific 인 EM1/EM5/EM6 등이 결손 시)
- EM8 단독으로 momentum=100 가능 → 강력매수의 변동성 폭발
- 개선: EM8 가중 15% → 10%, 나머지 5% 는 "Risk-Adjusted Momentum" (Sharpe-like)
  으로 분산

### 4.5 (추천) Hold-period-aware grade 별도 노출

현재는 단일 `final_grade`. 룰별 forward return 이 다르므로:

```
final_grade_1m   : 1개월 hold 권장 시 등급 (현재 final_score 기반)
final_grade_12m  : 12개월 hold 권장 시 등급 (mean-reversion 보정)
                   - 강력매수의 12m return median 이 음수면 등급 한 단계 강등
                   - 매도고려/중립 (12m hit 57%) 의 score boost
```

또는 `final_score` 자체를 hold-period 별로 다르게 산출.

### 4.6 (검증 필요) Cross-sectional rank-based scoring 으로 전환

- 현재 = 절대 점수 (0~100). 등급 임계가 hard threshold (85/75/65...) → 강세장
  / 약세장에서 등급 분포 변동 큼
- 대안: 매일 cross-sectional **rank** 로 점수 부여
  ```
  rank_score = PERCENT_RANK(...) OVER (PARTITION BY date ORDER BY raw_factor DESC) * 100
  ```
  → 매일 top-5% 가 강력매수, top-10% 가 매수 ... 분포 안정
- 단점: 시장 전반 약세장에서도 "강력매수" 가 5% 존재 → 백테스트 룰이 강세장
  편향 가정이면 손실

### 4.7 (실험적) IC-based factor reweight

현재 가중치는 학술 논문 기반 정적값. 실제 grades_pass_a 데이터로 검증된 IC:

```sql
-- 각 factor 의 12m forward return 과의 Spearman 상관계수 (IC)
WITH px AS (SELECT ..., LEAD(close, 252) OVER ... AS px_12m),
joined AS (
  SELECT g.value_score, g.quality_score, g.momentum_score, g.growth_score,
         (p.px_12m / p.close - 1) AS ret_12m
  FROM us_stock_grade g JOIN px p USING (symbol, date)
)
SELECT
  corr(value_score::float,    ret_12m) AS IC_value,
  corr(quality_score::float,  ret_12m) AS IC_quality,
  corr(momentum_score::float, ret_12m) AS IC_momentum,
  corr(growth_score::float,   ret_12m) AS IC_growth
FROM joined;
```

→ IC 가 음(-)인 factor 는 가중치 0 또는 -1 곱하기. (이미 RV2/EM1/EM2 는 IC 음수
관찰 후 reversed 되어 있음 — 같은 절차를 시스템 레벨로 확장.)

### 4.8 (구조적) 점수 saturation 회피 — log-scale 또는 soft-cap

```python
# 100 cap 대신 logistic transform
# total_score = 100 / (1 + exp(-(raw_score - 70) / 15))
# raw_score 가 매우 큰 (예: 130) 종목과 적당히 큰 (예: 90) 종목이 구분됨
```

기대 효과: 강력매수 후보가 5~10개 모두 100점에 묶이지 않고 95.3 / 92.1 / 88.7
... 식으로 차등 → top-3 선택이 임의가 아님.

---

## 5. 작업 우선순위

| 우선순위 | 작업 | 영향 |
|---------|------|------|
| 🔴 P0 | 4.1 entry timing penalty 반영 | 모멘텀 chaser 자동 강등 |
| 🔴 P0 | 4.2 데이터 결손 캡 | MYRG/ASYS 같은 100점 케이스 차단 |
| 🟡 P1 | 4.3 Pattern 4 임계 + bonus 실제 반영 | default V=50 우회 차단 |
| 🟡 P1 | 4.8 soft-cap (logistic) | top-3 saturation 해소 |
| 🟢 P2 | 4.7 IC 기반 reweight | 데이터 기반 검증 |
| 🟢 P2 | 4.6 cross-sectional rank | 시장 regime 영향 안정화 |
| ⚪ P3 | 4.5 hold-period grade | 룰별 등급 분기 |

P0 둘 만 적용해도 강력매수의 12m return median 이 음수 → 양수로 돌아설 가능성
큼 (cautious-position 종목이 빠지면서). 검증 절차:

1. 코드 변경 적용
2. 작은 윈도우 (예: 2024-01 ~ 2024-12) 에 grades_pass_a 재계산
3. forward return 검증 SQL 재실행 → median 12m 가 등급 순으로 monotonic 한지

---

## 6. 다음 단계

이 문서를 기반으로 P0 두 항목 (4.1, 4.2) 부터 우선 구현 → 진행 중인 `289aa1e2`
backtest 종료 후 동일 윈도우로 재산정 → A/B 비교.

---

## 7. 실증 백테스트 결과 — momentum 교체 + 레짐 adaptive (2026-06-08)

§3 의 "강력매수 12m -27%" 문제를 다루며 실제로 검증한 결과. **생존편향 제거 +
유동성 필터** 적용 (스크립트: `alphafolio_data/scripts/period_adaptive.py` 등).
상세 수식·전체 표는 `grade-scoring-formula.md §19` 참조.

### 7.1 momentum_score 를 52주 신고가 근접도로 교체 (적용 완료)

- 기존 momentum (EM1~EM8 가중합) 은 전 패널 IC ≈ **-0.04** (변별력 0) → 무신호.
- **near52h = close / 252일 최고가** (George-Hwang) 가 IC **+0.06** 으로 압도 →
  `us_momentum_factor.py` momentum_score 를 near52h 단독으로 교체함.

### 7.2 핵심 결론 (코드 수정 시 반영)

| 항목 | 결과 |
|---|---|
| **리밸 주기** | 월간(H≈21거래일)이 최적. 빈도↑ = 승률↓·MDD 급악화(-30%→-51%). `rebal_freq_days≈21` 유지, 늘리지 말 것. |
| **레짐 adaptive** | static(연 6.9%) 대비 약 2배(13%대). regime 별 부호/방어 전환이 핵심. |
| **팩터별 차별 적용** | **모멘텀만 하락장에 역전(flip), 밸류·퀄리티는 방어로 유지(`guru` 모드)** = CAGR 12.9% / 승률 61.5% / MDD -29.7% — 균형 최고. 전부 flip(allflip)은 CAGR 13.6%지만 승률·MDD 열위. |
| **승률 천장** | ~60% (분산 시 60.4%). 승률보다 레짐 방어로 MDD 축소가 실익. |

### 7.3 한계 / 다음 단계

- 위는 **proxy 팩터**(모멘텀=near52h, 밸류=1/per, 퀄리티=ROE) 기반 — **실제
  final_score 가 아님**. real final_score 는 us_stock_grade 에만 있고 현재 2019년치만.
- 거래비용 미반영, 단일 경로. 초기의 "저점매수 연 119%" 는 생존편향 산물(보정 후 0.9%).
- **TODO**: grades_pass_a 전 구간(2017~2026) 완성 후, real final_score 로
  §7.2 를 train/test 분리 재검증 → 4.6(cross-sectional rank)·4.7(IC reweight)·
  4.1(entry timing penalty) 와 함께 가중치·레짐 규칙 최종 확정.
