# Grade Scoring — 데이터 출처 + 수식 (구체)

`us_stock_grade.final_score` 가 어떻게 계산되는지 **DB 출처 → raw 값 → score
변환식 → 가중치 → 최종 합성** 까지 수식 위주로 정리.

소스: `alphafolio_quant/us/us_main.py` (orchestrator) + 4 factor 파일 +
`us_data_prefetcher.py` (DB fetch) + `us_event_engine.py` + `us_factor_interactions.py`.

---

## 0. 한 줄 공식 (전체 합성)

```
final_score = clamp(
    [(G · wG + M · wM + Q · wQ + V · wV) × 0.7  +  interaction_score × 0.3]
    − cash_runway_penalty
    + event_modifier
    ± pattern_adjustment_delta,
  0, 100
)
```

이 중 각 인자는 모두 0~100 normalized. 각 인자가 어떻게 산출되는지가 본문 내용.

---

## 1. 모든 입력 데이터의 DB 출처

### 1.1 `stock_data` (4 factor 의 공통 입력)

쿼리: `us_data_prefetcher.py:338`

```sql
SELECT symbol, stock_name, sector, industry, exchange, market_cap,
       per, forwardpe, peg, pricetosalesratiottm, evtorevenue, evtoebitda,
       pricetobookratio, trailingpe,
       quarterlyrevenuegrowthyoy, quarterlyearningsgrowthyoy,
       grossprofitttm, revenuettm, operatingmarginttm, profitmargin,
       returnonequityttm, returnonassetsttm, ebitda,
       beta, week52high, week52low, day50movingaverage, day200movingaverage,
       sharesoutstanding, sharesfloat, bookvalue, dilutedepsttm,
       analysttargetprice, analystratingstrongbuy, ...
FROM us_stock_basic
WHERE symbol = $1 AND date <= $2
ORDER BY date DESC LIMIT 1
```

→ `source='computed'` 행이 우선 선택됨 (시점별, look-ahead 안전).

| factor 가 사용하는 키 | us_stock_basic 컬럼 |
|---------------------|---------------------|
| `peg` | `peg` |
| `forward_pe` | `forwardpe` |
| `per` | `per` |
| `earnings_growth` | `quarterlyearningsgrowthyoy` |
| `revenue_growth` | `quarterlyrevenuegrowthyoy` |
| `operating_margin` | `operatingmarginttm` |
| `gross_margin` | `grossprofitttm / revenuettm` |
| `ev_revenue` | `evtorevenue` |
| `ev_ebitda` | `evtoebitda` |
| `market_cap` | `market_cap` |
| `current_price` | `us_daily.close` (직전 거래일) |
| `target_price` | `analysttargetprice` |
| `bookvalue` | `bookvalue` |
| `eps_ttm` | `dilutedepsttm` |
| `roe` | `returnonequityttm` |
| `roa` | `returnonassetsttm` |

### 1.2 `financials` (분기별 시계열)

쿼리: `us_data_prefetcher.py:443`

```sql
-- Income (최대 12 분기)
SELECT fiscal_date_ending, total_revenue, gross_profit, operating_income,
       net_income, research_and_development, operating_expenses,
       depreciation_and_amortization, ebitda
FROM us_income_statement
WHERE symbol = $1 AND available_at <= $2
ORDER BY fiscal_date_ending DESC LIMIT 12

-- CashFlow (최대 4 분기)
SELECT operating_cashflow, capital_expenditures, net_income, ...
FROM us_cash_flow WHERE symbol = $1 AND available_at <= $2
ORDER BY fiscal_date_ending DESC LIMIT 4

-- Balance (최신 1 행)
SELECT total_assets, total_liabilities, total_shareholder_equity,
       total_current_assets, total_current_liabilities,
       long_term_debt, short_term_debt, cash_and_cash_equivalents...
FROM us_balance_sheet WHERE symbol = $1 AND available_at <= $2
ORDER BY fiscal_date_ending DESC LIMIT 1
```

핵심: `available_at <= analysis_date` 필터로 look-ahead 차단. `available_at` 은
`us_earnings_history.reported_date` (실제 SEC 공시일) 또는 fallback `+45d`.

### 1.3 가격 / 거래량

```sql
SELECT date, open, high, low, close, volume FROM us_daily
WHERE symbol = $1 AND date <= $2 ORDER BY date DESC LIMIT 260
```

→ momentum factor 의 EM1/EM8 (가격 수익률), 52w high/low, 200MA, ATR, RSI 계산.

### 1.4 sector benchmarks (cross-sectional)

```sql
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY forwardpe) AS forward_pe_median,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY operatingmarginttm) AS op_margin_median,
       ...
FROM us_stock_basic
WHERE date = $analysis_date AND sector = $1 AND source = 'computed'
```

→ RV2 (섹터 대비 Forward PE) / MQ1 (섹터 대비 gross margin) 등이 사용.

### 1.5 event_modifier 입력

| modifier 컴포넌트 | DB 쿼리 |
|-----------------|---------|
| `earnings_modifier` | `us_earnings_calendar` WHERE symbol=? AND reportdate BETWEEN now AND now+7d |
| `options_modifier` | `us_option_daily_summary` WHERE symbol=? AND date<=$d (put/call, OI, GEX) |
| `insider_modifier` | `us_insider_transactions` WHERE symbol=? AND date BETWEEN $d-90d AND $d |
| `gex_modifier` | `us_option_daily_summary.gex_total` 직전 거래일 |
| `news_modifier` | `us_news` WHERE ticker=? AND time_published BETWEEN $d-7d AND $d (sentiment avg) |

### 1.6 HMM regime

```sql
SELECT close FROM us_daily_etf
WHERE symbol IN ('SPY','QQQ','TLT','GLD','IEF')
  AND date <= $analysis_date
ORDER BY date DESC LIMIT 504
```

→ 504일 returns → HMM (3-state) 학습 → 현재 시점 regime label
`{BULL, NEUTRAL, BEAR}` 반환.

---

## 2. Value Factor — `us_value_factor.py`

### 2.1 Sub-indicator → score 변환 (수식)

#### RV1: PEG Ratio (가중 25%)
- raw: `peg = stock_data['peg']`. 없으면 `pe / (growth × 100)` 으로 추정
- 섹터별 임계 `T = SECTOR_PEG_THRESHOLDS[sector]` (예: Default `{excellent: 1.3, good: 2.0, fair: 3.0}`)

```
        ┌ 20                            if peg < 0
        │ 85 + (T_ex − peg)/T_ex · 15   if 0 ≤ peg ≤ T_excellent
score = │ 65 + (T_g − peg)/(T_g − T_ex) · 20   if T_ex < peg ≤ T_good
        │ 45 + (T_f − peg)/(T_f − T_g) · 20    if T_g < peg ≤ T_fair
        └ max(15, 45 − (peg − T_fair) · 8)     if peg > T_fair
```

#### RV2: Forward PE Relative (가중 20%, **reversed** — IC=-0.139)
- raw: `ratio = stock_data['forwardpe'] / sector.forward_pe_median`

```
        ┌ 90                             if ratio ≤ 0.6
        │ 75 + (0.8 − ratio)/0.2 · 15    if 0.6 < ratio ≤ 0.8
        │ 55 + (1.0 − ratio)/0.2 · 20    if 0.8 < ratio ≤ 1.0
score = │ 40 + (1.2 − ratio)/0.2 · 15    if 1.0 < ratio ≤ 1.2
        │ 25 + (1.5 − ratio)/0.3 · 15    if 1.2 < ratio ≤ 1.5
        └ max(10, 25 − (ratio − 1.5) · 10)   if ratio > 1.5
```

→ `REVERSED_STRATEGIES = ['RV2']` 이라 `final = 100 − score` 로 뒤집힘.

#### RV3: EV/Revenue ÷ Growth (가중 20%)
- `ratio = stock_data['evtorevenue'] / (revenue_growth × 100)`

```
        ┌ 90                              if ratio ≤ 0.2
        │ 70 + (0.4 − ratio)/0.2 · 20     if 0.2 < ratio ≤ 0.4
score = │ 50 + (0.7 − ratio)/0.3 · 20     if 0.4 < ratio ≤ 0.7
        │ 35 + (1.0 − ratio)/0.3 · 15     if 0.7 < ratio ≤ 1.0
        └ max(10, 35 − (ratio − 1.0) · 15)  if ratio > 1.0
```

#### RV4: Rule of 40 (가중 15%, Tech/SaaS 만)
- `R = revenue_growth × 100 + operating_margin × 100`

```
        ┌ 95                             if R ≥ 60
        │ 80 + (R − 50)/10 · 15          if 50 ≤ R < 60
score = │ 65 + (R − 40)/10 · 15          if 40 ≤ R < 50
        │ 50 + (R − 30)/10 · 15          if 30 ≤ R < 40
        │ 35 + (R − 20)/10 · 15          if 20 ≤ R < 30
        └ max(10, 35 − (20 − R) · 1.5)   if R < 20
```

#### RV5: FCF Yield (가중 10%, Utilities/Financial 만)
- `fcf_yield = fcf_ttm / market_cap × 100`. piecewise linear `score` (10 ≥ excellent 90+, 6 ≥ good 70+, etc.)

#### RV6: Price to Target (가중 10%)
- `upside = (target_price − current_price) / current_price × 100`. piecewise linear

### 2.2 Value 합성

```
value_score = Σ (RVi.score × wi)   where Σwi = 1.0 (계산 가능한 i 만)
             = (RV1·0.25 + RV2_reversed·0.20 + RV3·0.20 +
                RV4·0.15 + RV5·0.10 + RV6·0.10) / Σwi_available
```

데이터 결손 sub-indicator 는 분모/분자에서 모두 제외. **모두 결손이면 default 50.**
(→ MYRG/ASYS 가 100점 받는 핵심 메커니즘.)

---

## 3. Quality Factor — `us_quality_factor.py`

| ID | Raw 수식 | DB 출처 | 가중 |
|----|---------|---------|-----|
| MQ1 | `gross_margin = grossprofitttm / revenuettm` → 섹터 median 대비 | `us_stock_basic` | 20% (HC 12%) |
| MQ2 | `ROIC ≈ EBIT × (1 − tax) / (Equity + Long_Term_Debt)` (`us_income_statement` + `balance_sheet` 12Q) | financials | 20% (HC 12%) |
| MQ3 | `op_leverage = ΔOperating_Income% / ΔRevenue%` (8Q 회귀) | `us_income_statement` 8Q | 15% |
| MQ4 | `quality = operating_cashflow / net_income` (TTM) | `us_cash_flow` + `us_income_statement` | 15% |
| MQ5 | `de_score = total_liabilities / equity` + `current_ratio = current_assets / current_liabilities` (둘 다 점수화 후 평균) | `us_balance_sheet` | 15% |
| MQ6 | `Δmargin = recent_4Q_op_margin − older_4Q_op_margin` (8Q 시계열) | `us_income_statement` 8Q | 15% |
| HC1~HC4 | Healthcare 전용 (analyst consensus / R&D intensity / cash runway) | mixed | 40% (HC 만) |

각 MQ 의 score 변환은 RV 와 유사한 piecewise linear. 예시 MQ4:
```
quality = op_cf / net_income
  ≥ 1.5 → 95  (cash >> earnings — 매우 좋음)
  ≥ 1.0 → 80 + (q − 1.0)/0.5 · 15
  ≥ 0.7 → 60 + (q − 0.7)/0.3 · 20
  ≥ 0.5 → 40 + (q − 0.5)/0.2 · 20
  <  0.5 → max(10, q · 80)
```

`quality_score = Σ MQi.score × wi / Σwi_available`.

---

## 4. Momentum Factor — `us_momentum_factor.py`

| ID | Raw 수식 | DB 출처 | 가중 |
|----|---------|---------|-----|
| EM1 | `sharpe-like = mean(ret_126d) / std(ret_126d)` | `us_daily` 126d | 5% (Energy/BM 만, **reversed**) |
| EM2 | sector 대비 ret_126d | `us_daily` + sector | **disabled** (IC -0.078) |
| EM3 | `EPS_revision = (curr_consensus − 30d_ago) / abs(30d_ago)` | `us_earnings_estimates` | 20% |
| EM4 | Revenue revision (동일 방식) | `us_earnings_estimates` | 20% |
| EM5 | `volume_ratio = vol_5d_avg / vol_30d_avg` × ret_20d | `us_daily` | (BM 만) |
| EM6 | `surprise_z = (reported − estimated) / std(historical_surprises)` 누적 4Q | `us_earnings_history` 4Q | 15% |
| EM7 | target price changes | (disabled) | — |
| EM8 | IBD RS 252d: `0.4·ret_3m + 0.2·ret_6m + 0.2·ret_9m + 0.2·ret_12m` | `us_daily` 252d | 15% |

EM8 score 변환:
```
       ┌ 100                      if RS_value ≥ 100
       │ 80 + (RS − 60)/40 · 20    if 60 ≤ RS < 100
       │ 60 + (RS − 30)/30 · 20    if 30 ≤ RS < 60
score =│ 40 + (RS − 10)/20 · 20    if 10 ≤ RS < 30
       │ 20 + (RS − 0)/10 · 20     if  0 ≤ RS < 10
       └ max(0, 20 + RS · 0.5)     if RS < 0
```

→ `momentum_score = Σ EMi.score × wi / Σwi_available`.

**핵심 결함**: EM3/EM4 가 `us_earnings_estimates` 결손 시 `no_data` → EM8 단독으로
momentum 결정. EM8 (가격수익률) 만으로 momentum=100 도달 가능 → 모멘텀 chaser
시스템.

---

## 5. Growth Factor — `us_growth_factor.py`

| ID | Raw 수식 | DB 출처 | 가중 |
|----|---------|---------|-----|
| FG1 | `rev_growth_yoy = (rev_TTM − rev_TTM_1y_ago) / rev_TTM_1y_ago` | `us_income_statement` 8Q | 20% (NASDAQ 12%) |
| FG2 | EPS yoy (동일) | `us_income_statement` 8Q | 20% (NASDAQ 12%) |
| FG3 | `forward_growth = (consensus_next_yr_rev − curr) / curr` | `us_earnings_estimates` | 20% (NASDAQ 15%) |
| FG4 | `consistency = mean(quarterly_growths) / std(quarterly_growths)` 8Q | `us_income_statement` | 15% (NASDAQ 8%) |
| FG5 | `acceleration = recent_2Q_avg_growth − older_2Q_avg_growth` | `us_income_statement` | 15% (NASDAQ 10%) |
| FG6 | `profit_growth = ni_TTM / ni_TTM_1y_ago − 1` | `us_income_statement` | 10% (NASDAQ 8%) |
| NQ1 | institutional ownership Δ | (외부 source) | 10% (NASDAQ) |
| NQ2 | analyst revisions count | `us_earnings_estimates` | 10% (NASDAQ) |
| NQ3 | growth sustainability index | derived | 8% (NASDAQ) |
| NQ4 | `put_call_ratio = total_puts / total_calls` (옵션 sentiment) | `us_option_daily_summary` | 7% (NASDAQ) |

FG1 score 변환 예시:
```
g = rev_growth_yoy (decimal, 0.30 = 30%)
       ┌ 100                       if g ≥ 0.30  (≥30% YoY)
       │ 80 + (g − 0.15)/0.15 · 20  if 0.15 ≤ g < 0.30
score =│ 60 + (g − 0.08)/0.07 · 20  if 0.08 ≤ g < 0.15  (`good` 임계)
       │ 40 + (g − 0.03)/0.05 · 20  if 0.03 ≤ g < 0.08
       │ 20 + g/0.03 · 20           if  0 ≤ g < 0.03
       └ max(0, 20 + g · 100)       if g < 0
```

→ `growth_score = Σ (FG/NQ).score × wi / Σwi_available`.

---

## 6. Sector Dynamic Weight 보정

`us_sector_dynamic_weights.py:get_combined_weights(base_weights, exchange, sector)`:

```python
# base_weights = REGIME_WEIGHTS[regime]  (예: BULL → G 0.35, M 0.25, Q 0.20, V 0.20)
# 섹터 적합도 조정:
if sector == 'TECHNOLOGY':
    growth += 0.05; value -= 0.05
elif sector == 'FINANCIAL SERVICES':
    value += 0.05; growth -= 0.05
elif sector == 'CONSUMER STAPLES':
    quality += 0.05; momentum -= 0.05
# ... NASDAQ exchange 가산 (growth +0.03)
# normalize 후 반환
```

→ 종목별 (sector, exchange) 에 따라 `wG, wM, wQ, wV` 가 미세 조정됨.

---

## 7. Base Score (4 factor 합성)

```
adjusted = sector_weights.get_combined_weights(REGIME_WEIGHTS[regime], exchange, sector)

base_score = growth_score   · adjusted['growth']    / 100
           + momentum_score · adjusted['momentum']  / 100
           + quality_score  · adjusted['quality']   / 100
           + value_score    · adjusted['value']     / 100
```

(`adjusted` 값은 0~100 normalized 라 `/100` 후 가중합)

---

## 8. Interaction Score — `us_factor_interactions.py`

5 가지 factor pair interaction. 각각 0~100 점수, 가중합.

| ID | 수식 | 가중 |
|----|------|-----|
| I1 | G×Q: `100 · √(g·q)` + synergy bonus (g≥70 & q≥70 시 max +10) | 30% |
| I2 | G×M: `100 · 2gm / (g+m)` (harmonic mean) | 25% |
| I3 | Q×V: `100 · √(q·v)` | 20% |
| I4 | M×Q: harmonic mean | 15% |
| I5 | All-factor agreement: 4 factor 모두 ≥70 이면 score = mean, 아니면 penalty | 10% |

(g, q, m, v 는 모두 0~1 로 normalize)

```
interaction_score = I1·0.30 + I2·0.25 + I3·0.20 + I4·0.15 + I5·0.10
conviction_score  = I5.score   (별도 컬럼)
```

---

## 9. Total Score 합성

```
total_score = base_score · 0.70 + interaction_score · 0.30
```

---

## 10. Cash Runway Penalty (Healthcare 만)

`us_healthcare_optimizer.py`:
```
burn_rate = abs(operating_cashflow_TTM) / 4   # 분기당 평균 burn
runway_months = (cash_and_equivalents / burn_rate) × 3

if sector == 'HEALTHCARE' and runway_months < 12:
    penalty = (12 − runway_months) · 2   # 12개월 미만이면 월당 -2점
    total_score = max(0, total_score − penalty)
```

---

## 11. Event Modifier — `us_event_engine.py`

각 컴포넌트 단독 −5 ~ +5 점, 합산 후 ±20 clamp.

| ID | 수식 | 입력 |
|----|------|-----|
| earnings_mod | upcoming earnings (≤7d) → -2, recent positive surprise → +3 | `us_earnings_calendar` + `us_earnings_history` |
| options_mod | put/call ratio: <0.5 → +3, >1.5 → -3 | `us_option_daily_summary` |
| insider_mod | cluster buying (3+ execs in 90d) → +10; CEO/CFO large selling → -5 | `us_insider_transactions` |
| gex_mod | positive GEX (gamma squeeze) → +2 | `us_option_daily_summary.gex_total` |
| news_mod | avg sentiment over 7d × 5 | `us_news.overall_sentiment_score` |

```
event_modifier = clamp(earnings + options + insider + gex + news, −20, +20)
total_score = clamp(total_score + event_modifier, 0, 100)
```

(Pass-A 는 `with_event_modifier=False` 라 0. Pass-B 에서만 적용.)

---

## 12. Pattern Adjustment — `us_factor_interactions.py:apply_pattern_adjustment`

```python
V_neg = (value < 50);    V_pos = (value ≥ 50)
Q_neg = (quality < 50);  Q_pos = (quality ≥ 50)
M_neg = (momentum < 50); M_pos = (momentum ≥ 50)
G_neg = (growth < 50);   G_pos = (growth ≥ 50)
```

| Pattern | 조건 | 시총 | 결과 |
|---------|------|-----|------|
| Sell (V−Q+M−G+) | V_neg & Q_pos & M_neg & G_pos | mega/large | **score = 40 강제** |
|  |  | mid | score −= 15 |
|  |  | small | score −= 10 |
|  |  | micro | score −= 15 |
| Buy (V+Q+M+G+) | 모두 ≥50 | mega~small | score += 5 |
|  |  | micro | score += 3 |
| Strong Sell (V−Q−M−G−) | 모두 <50 | mega/large | **무시** (96% up rate) |
|  |  | mid/micro | **score = 25 강제** |
|  |  | small | score −= 20 |

---

## 13. Factor Combination Bonus (저장만, score 미반영)

`us_main.py:_calculate_factor_combination_bonus(v, q, m, g)`:

```python
bonus = 0
if q ≥ 70 and g ≥ 70:           bonus += 15   # Quality Compounder
if v ≥ 75 and q ≥ 60 and m < 40: bonus += 20   # Deep Value Turnaround
if g ≥ 70 and v ≥ 50:           bonus += 10   # GARP
if v < 30 and m ≥ 80:           bonus −= 10   # Momentum Speculation Penalty
if all([v,q,m,g] ≥ 65):         bonus += 15   # All-around Strong
bonus = clamp(bonus, −20, +30)
```

→ `us_stock_grade.factor_combination_bonus` 컬럼에 **저장만**. `final_score` 에는
적용 안 됨.

---

## 14. 최종 grade 매핑

```python
if   total_score ≥ 85: grade = '강력 매수'
elif total_score ≥ 75: grade = '매수'
elif total_score ≥ 65: grade = '매수 고려'
elif total_score ≥ 55: grade = '중립'
elif total_score ≥ 45: grade = '매도 고려'
elif total_score ≥ 35: grade = '매도'
else:                   grade = '강력 매도'
```

---

## 15. End-to-end 예시 (TER, 2026-05-29)

DB 값 (`us_stock_basic source='computed'` + financials):
```
peg ≈ 0.23, forwardpe ≈ ?, evtorevenue=15.5, revenue_growth=0.87
operating_margin=0.27, gross_margin=0.59, op_cf=854M, net_income=854M
revenue_TTM=3.79B
sector='TECHNOLOGY' (Tech: RV4 활성)
```

Value 계산:
```
RV1 = piecewise(peg=0.23, T={ex:1.3, g:2.0, f:3.0}) ≈ 97.4
RV2 = (forwardpe missing → no_data)
RV3 = piecewise(ev/rev/growth = 15.5/87 = 0.18) ≈ 90
RV4 = piecewise(R = 0.87×100 + 0.27×100 = 113.95) ≈ 95
RV5 = no_data (sector not in [Utilities, FinSvc])
RV6 = no_data (no target)

valid_weights = 0.25 + 0.20 + 0.15 = 0.60
value_score = (97.4·0.25 + 90·0.20 + 95·0.15) / 0.60 ≈ 94.3
```

Quality 등도 마찬가지 → `quality_score=83.8`, `momentum=100`, `growth=91.5`.

Regime = BULL → 가중치 G 0.35 / M 0.25 / Q 0.20 / V 0.20. Tech 보정 후 가령
G 0.40 / M 0.25 / Q 0.20 / V 0.15.

```
base_score = 91.5·0.40 + 100·0.25 + 83.8·0.20 + 94.3·0.15
           = 36.6 + 25.0 + 16.76 + 14.15 = 92.51

interaction_score (g=91.5, q=83.8, m=100, v=94.3):
  I1 = 100·√(0.915·0.838) + synergy(0.1·min(0.717,0.46)) ≈ 87.9 + 4.6 = 92.5
  I2 = 100·2·0.915·1.0/(0.915+1.0) = 95.6
  I3 = 100·√(0.838·0.943) = 88.9
  I4 = 100·2·1.0·0.838/(1.0+0.838) = 91.2
  I5 = all ≥70 → mean(91.5,83.8,100,94.3) = 92.4
  interaction = 92.5·0.30 + 95.6·0.25 + 88.9·0.20 + 91.2·0.15 + 92.4·0.10
              ≈ 27.75 + 23.9 + 17.78 + 13.68 + 9.24 = 92.35

total_score = 92.51·0.70 + 92.35·0.30 = 64.76 + 27.71 = 92.47

cash_runway_penalty = 0 (Tech, not Healthcare)

event_modifier (Pass-B):
  earnings = +3 (recent positive surprise)
  options  = +2 (put/call ~0.67)
  insider  = +5 (15 buys / 9 execs)
  gex      = +1
  news     = +3 (bullish sentiment)
  total    = clamp(+14, -20, +20) = +14
→ total_score = clamp(92.47 + 14, 0, 100) = 100  ← cap

pattern_adjustment (all positive ≥50) = Buy Pattern → +5 → 100 (cap 유지)

final_score = 100
grade = '강력 매수'
```

→ TER 의 100점은 base 92 + event +14 + pattern +5 가 합쳐져 cap 됨. 정상 메커니즘.

**MYRG/ASYS 의 100점은** value/growth sub-indicator 가 데이터 결손으로 default
50 / NQ4 단독으로 부풀려져 score 가 비정상적으로 큼. 이게 7번 섹션의 `valid_weights`
divisor 가 작아서 weighted average 가 비정상.

---

## 16. 핵심 결함 요약 (수식 측면)

1. **`valid_weights` divisor**: sub-indicator 결손 시 `Σ wi_available` 로 나눠 평균
   → 한 지표만 계산돼도 그게 그대로 factor score. **결손 = noise 증폭**.

2. **`event_modifier` clamp +20**: base+interaction 이 80 이면 event +20 으로 100 cap
   → grade saturation.

3. **`factor_combination_bonus` 가 `total_score` 에 반영 안 됨**: Momentum
   Speculation Penalty (-10) 가 출력 컬럼에만 있고 점수 미적용 → MYRG/ASYS 같은
   케이스 의도된 페널티 우회.

4. **`pattern_adjustment` 의 mid/micro 캡**: V−Q−M−G− 의 mid 는 score=25 강제하지만
   mega/large 는 무시 → cap 가능성 다름.

5. **`entry_timing_score` 와 `outlier_risk_score`** 가 `us_stock_grade` 컬럼에
   저장되지만 `total_score` 공식에는 들어가지 않음. → 52w 고점 근처 종목 같은
   문제 케이스가 자동 강등 안 됨.

6. **REVERSED_STRATEGIES = ['RV2']**: IC 음수 (-0.139) 라 reversed 처리 — 같은
   검증을 모든 sub-indicator 에 시스템 레벨로 확장 필요 (현재 ad-hoc).

---

## 17. 개선 방향 (수식 변경 예시)

### 17.1 `final_score` 에 factor_combination_bonus 실제 반영

```python
# us_main.py
total_score = clamp(total_score + factor_combination_bonus, 0, 100)
# 현재는 이 라인이 없음 → 추가하기만 하면 default V=50 결손 종목의 -10 페널티 작동
```

### 17.2 결손 페널티 추가

```python
# 각 factor 의 valid_weights 가 너무 작으면 페널티
coverage = valid_weights_sum / total_weights_sum   # 0~1
if coverage < 0.5:
    score *= (0.5 + coverage)   # half-credit
```

### 17.3 entry_timing_score 반영

```python
# us_agent_metrics 에서 이미 계산하는 값
if price_position_52w ≥ 90:
    total_score = max(0, total_score − 10)
elif price_position_52w ≥ 70:
    total_score = max(0, total_score − 3)
```

### 17.4 soft cap

```python
# 100 hard cap 대신 logistic
total_score = 100 / (1 + exp(-(raw − 70) / 15))
# raw = 80 → 78.6, raw = 100 → 95.7, raw = 130 → 99.7
# 강력매수 cap 묶임 해소
```

이 4개만 적용하면 강력매수의 forward 12m median = -27% → 양수로 돌아설 가능성
큼 (실측 검증 필요).

---

## 18. 적용된 데이터 보강 (A + B + C, 2026-06-04 변경)

§3 의 sub-indicator 87~100% 결손 문제를 해결하기 위해 **데이터 source 자체를
보강**. quant 코드 가중치 변경 없음 — "있으면 쓰는" 정책 (데이터 없는 sub-indicator
는 score=None → weighted average 분모/분자에서 자동 제외).

### 18.1 A — `us_earnings_estimates` 적재 (DAG 7a-3)

DAG task `task_earnings_estimates` 신규 추가. Collector
`EarningsEstimatesCollectorOptimized` 활용 (`start_date` parameter 추가됨).

AV `EARNINGS_ESTIMATES` 응답 (종목당 1 call):
```json
{
  "estimate_date": "2027-09-30",
  "horizon": "fiscal year",
  "eps_estimate_average": "9.6552",
  "eps_estimate_average_7_days_ago":  "9.6552",
  "eps_estimate_average_30_days_ago": "9.5898",
  "eps_estimate_average_60_days_ago": "9.3551",
  "eps_estimate_average_90_days_ago": "9.3085",
  "eps_estimate_revision_up_trailing_7_days":   1,
  "eps_estimate_revision_down_trailing_7_days": 0,
  "eps_estimate_revision_up_trailing_30_days":  3,
  "eps_estimate_revision_down_trailing_30_days":1,
  "revenue_estimate_average": "517815320250"
}
```

활성된 sub-indicator:

| ID | factor | 수식 |
|----|--------|------|
| **EM3** EPS revision | Momentum | `(eps_estimate_average − eps_estimate_average_30_days_ago) / abs(eps_estimate_average_30_days_ago)` |
| **EM4** Revenue revision | Momentum | revenue 동일 |
| **NQ2** Analyst revision | Growth (NASDAQ) | revision_up_30days - revision_down_30days 비율 + 컨센서스 변화 합성 |

**한계**: AV 응답은 호출 시점 기준 historical (30/60/90d ago). 백테스트 2019
시점 grade 계산엔 그 시점의 30d_ago snapshot 이 없어 결손 그대로. 오늘부터 daily
누적 시작 → 30일 후 NQ2 시점-aware 가용.

### 18.2 B — `stock_basic_compute` 의 peg/forwardpe/target 컬럼 propagate

`_compute_stock_basic_window` 의 api_df fetch SQL 에 3 컬럼 추가:

```sql
SELECT DISTINCT ON (symbol)
       ..., peg AS api_peg,
            forwardpe AS api_forwardpe,
            analysttargetprice AS api_target_price
FROM us_stock_basic WHERE source='api'
ORDER BY symbol, date DESC
```

compute 단계에서 매 (symbol, date) row 에 propagate:
```python
merged['peg']                = merged['api_peg']
merged['forwardpe']          = merged['api_forwardpe']
merged['analysttargetprice'] = merged['api_target_price']
```

활성된 sub-indicator:

| ID | factor | 입력 | 수식 |
|----|--------|------|------|
| **RV1** PEG | Value | `stock_data['peg']` 또는 `forward_pe / (earnings_growth × 100)` | 섹터별 piecewise (§2.1) |
| **RV2** Forward PE | Value (reversed) | `stock_data['forwardpe'] / sector.forward_pe_median` | piecewise + reversed |
| **RV6** Price-to-Target | Value | `(target_price − current_price) / current_price × 100` | piecewise |

**한계** — 시점-aware 아님 (look-ahead 잠재):
- `us_stock_basic(api)` 가 매일 갱신되는 latest snapshot 1행만 (date=today)
- 모든 시점의 computed 행에 *latest* peg/forwardpe/target 동일 적용
- backtest 의 2019 시점 grade 가 2026 시점의 forwardpe 를 보는 셈
- 향후 보강: `us_earnings_estimates.created_at <= analysis_date` 필터로 시점별
  forward_eps fetch → `forward_pe = close / forward_eps` (별도 작업)

### 18.3 C — `us_institutional_holdings` 적재 (DAG 7a-4)

신규 alembic `0017_us_institutional_holdings.py` — `(symbol, report_date)` PK +
holdings JSONB 컬럼.
신규 `InstitutionalHoldingsCollector` (`finance_data.py`).
신규 `task_institutional_holdings` (`orchestrator/tasks.py`).

AV `INSTITUTIONAL_HOLDINGS` 응답 (종목당 1 call):
```json
{
  "total_institutional_holders": 6426,
  "total_institutional_shares": ...,
  "holders_with_increased_holdings": ...,
  "shares_with_increased_holdings": ...,
  "holders_with_decreased_holdings": ...,
  "total_institutional_ownership_percentage": ...,
  "holdings": [
    {"holder_name": "VANGUARD GROUP INC", "shares_held": ...,
     "shares_changed": ..., "change_type": "increased",
     "last_reported": "2025-12-31"}
  ]
}
```

collector 가 `holdings` array 를 `last_reported` 기준 분기별 그룹핑 → 분기별
aggregate 재계산 → 분기당 1 row INSERT (AV summary 는 latest 분기만이라).

활성된 sub-indicator:

| ID | factor | 수식 |
|----|--------|------|
| **NQ1** Institutional Quality | Growth (NASDAQ) | `(holders_increased − holders_decreased) / total_holders` + `shares_increased / total_shares` 합성 (weight 10%) |

**한계**: AV trailing 1 year (4 분기) 만. 2019~2025-Q2 grade 는 NULL 유지.
2025-Q3 이후 grade 부터 NQ1 활성. 더 깊은 historical 원할 시 SEC EDGAR 13F-HR
직접 파싱 (3-4주 작업, 무료) 별도.

### 18.4 결손 정책 — "있으면 쓰는"

모든 sub-indicator 의 가중치 (RV1 0.25, EM3 0.20, NQ1 0.10 등) **변경 없음**.
데이터 없는 sub-indicator 처리:

```
factor_score = Σ (subᵢ.score × wᵢ) / Σ wᵢ   for i in valid_subs (score is not None)
```

→ 결손 sub-indicator 의 가중치는 분모/분자 모두에서 제외. 모두 결손이면 default 50.

### 18.5 결함 16 의 일부 해소

| 이전 결함 | A+B+C 적용 후 |
|----------|--------------|
| Momentum 이 EM8 (가격) 단독 — EM3/EM4/EM6 100% 결손 | A 적용 시 EM3/EM4 활성 → momentum 가 가격 + 컨센서스 합성 |
| Value 가 RV4 (op_margin) 단독 — RV1/RV2/RV6 89-97% 결손 | B 적용 시 RV1/RV2/RV6 활성 → PEG/Forward-PE/Target 종합 |
| Growth 의 NQ1/NQ2 결손 95%+ | A + C 적용 시 NQ1 (최근 1년), NQ2 (forward) 부분 활성 |
| 100점 saturation (top 5-10 묶임) | sub-indicator 다양성 회복으로 분산 예상 (검증 필요) |

여전히 미해결 (별도 작업):
- 결함 1 (`valid_weights` divisor — 한 지표만 valid 시 noise 증폭)
- 결함 3 (`factor_combination_bonus` 가 `total_score` 미반영)
- 결함 5 (`entry_timing_score` / `outlier_risk_score` 점수 미반영)
- B 의 시점-aware 한계 (look-ahead)
- C 의 historical depth (4 분기 only — 2019~2024 NQ1 영구 결손)

### 18.6 검증 방법

진행 중 backtest run `5bddc0fd_us_2019-01-01` 완료 후, §3.1 의 등급별 forward
return / hit rate 표 재산정:

```sql
WITH px AS (
  SELECT symbol, date, close,
    LEAD(close, 21)  OVER (PARTITION BY symbol ORDER BY date) AS px_1m,
    LEAD(close, 252) OVER (PARTITION BY symbol ORDER BY date) AS px_12m
  FROM us_daily WHERE date BETWEEN '2018-06-01' AND '2026-06-04'
),
g AS (SELECT date, symbol, final_grade FROM us_stock_grade
      WHERE date BETWEEN '2019-01-02' AND '2026-05-29')
SELECT final_grade,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY (px_1m/close-1)*100)  AS median_1m,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY (px_12m/close-1)*100) AS median_12m,
  100.0 * COUNT(*) FILTER (WHERE px_12m > close) / COUNT(px_12m) AS hit_12m
FROM g JOIN px USING (symbol, date)
GROUP BY final_grade ORDER BY median_12m DESC;
```

기대 효과: 강력매수의 median 12m 가 -27% → 양수로 (또는 적어도 매수 등급의
median 보다 큰 값으로) 전환. 등급 순위와 forward return 순위가 monotonic 회복.

---

## 19. 실증 백테스트 — momentum 재정의 + 리밸주기 × 레짐 adaptive (2026-06-08)

검증 방법: `us_daily` 전체 패널(2017-11~2026-06) + `us_stock_basic source='computed'`
팩터로, **생존편향 제거**(상폐 종목은 마지막 체결가까지 손실 실현) + **유동성 필터**
(종가 ≥ $5 & 20일 평균 거래대금 ≥ $5M) 하에 측정. 스크립트: `alphafolio_data/scripts/`
(`momentum_guru_test.py`, `regime_adaptive_v2.py`, `factors_*adaptive.py`,
`period_adaptive.py`, `winrate*.py`).

### 19.1 momentum_score 재정의 (코드 적용 완료)

- 기존 EM1~EM8 가중합: 전체 패널 횡단면 rank-IC ≈ **-0.04** (변별력 std≈0, 전부 ~65점) → 무신호/약한 역전.
- guru/학계 방법 비교 (월말 리밸, fwd IC):
  - **52주 신고가 근접도 `near52h = close / 252일 최고가`** (George-Hwang 2004): IC **+0.057~+0.067** (IR ~0.35, 월 62~68% 양) ← **최고**
  - 6-1 모멘텀 +0.02, 12-1(JT) 약함/장기 음수, **IBD RS 음수(-)**
- → `us_momentum_factor.py` 의 momentum_score 를 **near52h 단독으로 교체**
  (`_calc_near52h_score`, `score = clamp((near52h − 0.4)/0.6 × 100, 0, 100)`).
  EM1~EM8 가중합 폐기.

### 19.2 리밸 주기(H 거래일) × 레짐 adaptive 방식 비교

레짐: SPY > 200MA → bull, else bear. mode 정의:
- `static` = 항상 고득점(top decile) 매수
- `allflip` = bear 에 4팩터 전부 역전(저득점 매수)
- `guru` = **모멘텀만 bear 에 역전, 밸류·퀄리티는 고득점 유지(방어)**

| mode | H | 리밸수 | 승률% | CAGR% | MDD% |
|---|---|---|---|---|---|
| static | 21 | 96 | 61.5 | 6.9 | -30.1 |
| static | 10/5/3 | 202/402/674 | 62→55 | ~6 | -37~-41 |
| allflip | **21** | 96 | 58.3 | **13.6** | -29.7 |
| allflip | 10/5/3 | — | 60→55 | 10/9/10 | -41~-47 |
| **guru** | **21** | 96 | **61.5** | **12.9** | **-29.7** |
| guru | 10/5/3 | — | 60→55 | 11/10.8/12.2 | -44~-51 |

### 19.3 결론 — 코드 수정 시 적용 지침

1. **리밸 주기는 월간(H≈21거래일)이 최적.** 빈도를 높일수록(H↓) 승률↓ + MDD 급악화
   (-30% → -51%), CAGR 개선 없음. **거래비용 0 가정이라 실제론 더 나쁨.**
   → `rebal_freq_days` 는 ~21 유지, 늘리지 말 것.
2. **레짐 적응형이 static 대비 약 2배 CAGR** (6.9% → 13%대). 즉 §1 의 단방향
   `base_score` 보다 **regime 별 부호/방어 전환**이 핵심 개선.
3. **`guru` (모멘텀만 bear 에 flip, 밸류/퀄리티는 방어 유지)** = CAGR 12.9% /
   승률 61.5% / MDD -29.7% 로 **균형 최고**. allflip 은 CAGR 13.6%(최고)지만
   승률 -3.2%p. → 퀄리티/밸류는 하락장에 뒤집지 말 것(방어), 모멘텀만 레짐 의존.
4. 승률은 ~60%가 사실상 천장 (top30% 분산 시 60.4% 최고). 승률 자체보다
   **레짐 방어로 MDD 를 줄이는 것**이 guru 들의 실익.

### 19.4 한계 (반드시 같이 기억)

- 위 팩터는 **proxy** (모멘텀=near52h, 밸류=1/per, 퀄리티=ROE) — **실제 final_score
  합성식이 아님**. real final_score 는 us_stock_grade 에만 있고 현재 2019년치만 존재.
- 단일 경로(2017~2026), **거래비용 미반영**, MDD ~30%.
- "저점매수(allflip)가 압도적"이라던 초기 결과(연 119%)는 **생존편향 산물**이었고,
  상폐 손실 반영 + 유동성 필터 후 0.9% 로 붕괴 — 위 표가 보정된 수치.
- **다음 단계**: grades_pass_a 가 2017~2026 전 구간 grade 를 생성 완료하면, proxy 가
  아닌 **실제 final_score 로 §19.2 를 재검증** (train/test 분리) 후 가중치 확정.
