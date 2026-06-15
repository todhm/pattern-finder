# Task `stock_basic` — US Stock Basic (Fundamentals)

DAG 위치: **step 3** (`partitions` → `stock_listing` → `finnhub_symbol` → **`stock_basic`** → `us_daily` / `us_weekly` / `earnings_history` / `financials` …)

코드:
- `alphafolio_data/orchestrator/tasks.py:437` `task_stock_basic`
- 헬퍼 `_symbols_due_for_report` (line 385)
- collector `alphafolio_data/us/alphavantage.py:78` `AlphaVantageCollector.collect_stock_data`

---

## 1. 무엇을 하는가

각 활성 종목에 대해 **AlphaVantage `OVERVIEW`** endpoint (`function=OVERVIEW&symbol=…`) 를 호출해서, 그 회사의 **현재 시점 펀더멘털 snapshot** (PER / PEG / market_cap / TTM revenue / 분기 성장률 / 애널리스트 레이팅 / 52주 high/low / 200일 MA / sector / industry / shares outstanding / …, 총 50+ 칼럼) 을 받아 `us_stock_basic` 테이블의 **`source='api'`** 행으로 적재한다.

API 응답의 의미는 "**호출 시점의 현재값**" 이라서 row 의 `date` 컬럼은 `date.today()` 로 박힌다 — 과거 시점의 값은 절대 들어오지 않는다. (과거 시점은 후속 task `stock_basic_compute` 가 별도로 재계산해 `source='computed'` 로 채운다 — §6 참조.)

## 2. 어떤 종목만 fetch 하는가 (Publication-aware skip)

전체 활성 종목 (~6,200) 을 매 run 마다 다 부르면 AV 일일 한도가 터지고, 이미 최근 분기보고를 한 종목은 그 다음 분기까지 OVERVIEW 값이 의미있게 바뀌지 않는다. 그래서 `_symbols_due_for_report(days=75)` 가 사전 필터:

```sql
WITH last_rep AS (
    SELECT symbol, MAX(reported_date) AS last_rep
    FROM us_earnings_history GROUP BY symbol
),
active AS (
    SELECT DISTINCT symbol FROM us_stock_basic WHERE is_active = true
)
SELECT a.symbol
FROM active a LEFT JOIN last_rep r USING (symbol)
WHERE r.last_rep IS NULL                              -- 신규/누락
   OR r.last_rep < CURRENT_DATE - make_interval(days => 75)
```

→ 마지막 SEC 공시일(`us_earnings_history.reported_date`) 이 **75일 이상 지났거나 한 번도 없는 종목만** 재수집. 분기 보고 주기가 90일 (≈ 13주) 이라 75일 이후면 새 10-Q/10-K 가 곧 나올 시점.

이전 collection-time blanket skip ("n일 전에 받아왔으면 안 받음") 과 달리 **발표일 기반** 이라 false skip 위험이 없다.

운영 예: 현재 run (`784becce_us_2019-01-01`) 에서 `due_count = 1,488` 종목만 호출, 약 **16.6분** 소요. 전체 활성 (≈ 6,200) 대비 24% 가 재수집 대상.

## 3. 적재 테이블 — `us_stock_basic`

| 항목 | 값 |
|---|---|
| PK | **`(symbol, date, source)`** |
| `source` CHECK | `'api'` 또는 `'computed'` |
| Default | `date = CURRENT_DATE`, `source = 'api'` |
| Upsert | `INSERT … ON CONFLICT (symbol, date, source) DO UPDATE SET <전체 컬럼>` |
| 인덱스 | `(date)`, `(date, symbol)`, `(symbol, date DESC)`, `(exchange)`, `(industry)`, `(sector)`, `(is_active)`, `(source)` |
| Alembic 마이그레이션 | `0001_initial` → `0011_stock_basic_time_series` → `0012_stock_basic_pk_with_source` |

같은 날 동일 종목을 두 번 호출하면 같은 `(symbol, date, source)` 행이 **UPDATE 로 덮어쓰임** — `date.today()` 가 키의 일부라 자정을 넘기면 새 row 가 생긴다.

### `source='api'` 가 채우는 50+ 컬럼 (요약)

- **회사 메타**: `symbol, stock_name, description, cik, exchange, currency, country, sector, industry, address, officialsite, fiscalyearend, latestquarter, is_active`
- **밸류에이션 (TTM 기준 latest)**: `market_cap, per, peg, bookvalue, pricetosalesratiottm, pricetobookratio, evtorevenue, evtoebitda, trailingpe, forwardpe`
- **수익성 / 성장 (TTM, YoY)**: `profitmargin, operatingmarginttm, returnonassetsttm, returnonequityttm, revenuettm, grossprofitttm, dilutedepsttm, revenuepersharettm, eps, quarterlyearningsgrowthyoy, quarterlyrevenuegrowthyoy, ebitda`
- **배당**: `dividendpershare, dividendyield, dividenddate, exdividenddate`
- **애널리스트**: `analysttargetprice, analystratingstrongbuy, analystratingbuy, analystratinghold, analystratingsell, analystratingstrongsell`
- **주가 메트릭**: `week52high, week52low, day50movingaverage, day200movingaverage, beta`
- **주식 수 / 보유 구조**: `sharesoutstanding, sharesfloat, percentinsiders, percentinstitutions`
- **메타**: `created_at, updated_at, date, source`

### 운영 카운트 (현재 시점 기준)

```
 source  |    min     |    max     |  rows   | syms
---------+------------+------------+---------+------
 api     | 2026-05-21 | 2026-06-05 |  14,797 | 6,202
 computed| 2017-11-27 | 2026-06-02 | 9,172,531 | 6,202
```

- `api` row 는 약 **최근 16일** 분만 살아있음 — 매일 75일 due 종목 ~1,500 × 16일 ≈ 14k. (오래된 `api` row 는 별도로 정리되지 않지만 PIT 쿼리가 `WHERE date <= analysis_date ORDER BY date DESC LIMIT 1` 라 always-latest 가 선택돼 무해)
- `computed` row 는 종목당 평균 **1,479개** = 2017년 말 ~ 2026년 거래일 수와 일치 (시점별 1행).

## 4. 데이터 범위 (Range)

| 차원 | 값 |
|---|---|
| `date` 범위 (`source='api'`) | 매일 새 `date.today()` 추가. 종목별 latest 1행이 의미있음 |
| `date` 범위 (`source='computed'`) | `[ctx.start_date - 800d, ctx.end_date]` (lookback 800d = sparse ticker + IPO buffer) |
| 종목 universe (active) | ~6,200 (NASDAQ + NYSE common stocks, ETF 제외) |
| API call/day | due 종목 수 (≈ 1,500 ± 분기 cycle) |
| 컬럼 수 | 56 (메타 5 포함) |

### 후속 task 가 의존하는 최소 범위

- `task_us_daily` 가 `SELECT DISTINCT symbol FROM us_stock_basic WHERE is_active=true` 로 universe 를 잡음 → **모든 활성 종목의 적어도 한 row** 필요 (`source` 무관).
- `task_us_weekly`, `task_earnings_history`, `task_financials`, `task_financials_verify`, `task_us_stock_basic_compute`, `task_em8_pre_filter` 모두 동일하게 universe 추출에 사용.
- Quant grade 계산은 `WHERE symbol=$1 AND date <= analysis_date ORDER BY date DESC LIMIT 1` → 시점별 PIT row 1개. `source='api'` snapshot 만 있어도 동작하지만 시점이 `analysis_date` 와 멀면 stale.

## 5. 후속 프로세스 — 어디서 어떻게 쓰이는가

### 5.1 Quant grade 계산 (가장 큰 소비자)

| 모듈 | 사용 컬럼 | 쿼리 패턴 |
|---|---|---|
| `us_value_factor.py` | `per, peg, forwardpe, pricetobookratio, evtorevenue, evtoebitda, dividendyield, analysttargetprice` | PIT 1행 `WHERE symbol=$1 AND date<=$2 ORDER BY date DESC LIMIT 1` |
| `us_quality_factor.py:202,870` | `grossprofitttm, revenuettm, operatingmarginttm, profitmargin, returnonequityttm, returnonassetsttm, ebitda, market_cap, beta, sharesoutstanding, bookvalue, dilutedepsttm` + HC `analysttargetprice, analystratingstrongbuy~strongsell` | 동일 |
| `us_growth_factor.py:198,713` | `quarterlyrevenuegrowthyoy, quarterlyearningsgrowthyoy, peg, forwardpe, per, revenuettm, grossprofitttm, operatingmarginttm, profitmargin, dilutedepsttm, ebitda` + NASDAQ `percentinstitutions, percentinsiders` | 동일 |
| `us_momentum_factor.py:314,~616,~712` | `revenue_growth, earnings_growth` (fallback when income_statement 누락 시 YoY 대체) | 동일 |
| `us_outlier_risk.py:122,215` | `market_cap, exchange, sector` | PIT — `DISTINCT ON (symbol)` batch CTE 와 single 모두 |
| `us_db_async.py:204,310` | `sector, industry` | `mv_us_sector_daily_performance` / `mv_us_industry_daily_performance` refresh INSERT (PIT `DISTINCT ON (symbol)`) |
| `us_sector_benchmarks.py` | `per, peg, pricetobookratio, evtorevenue, evtoebitda, pricetosalesratiottm` percentile by `(sector, date)` | 섹터별 valuation 분포 → 종목 점수 정규화 |
| `us_alternative_matcher.py:64,409,428` | `sector, industry` | sell-grade 종목의 동일 sector/industry 대체 매칭 |
| `us_data_prefetcher.py` | 위 컬럼 한방에 prefetch → factor 모듈에 inject | `(symbol, date<=analysis_date)` batch |

### 5.2 Orchestrator 후속 task 의 universe / skip 판단

| Task | 활용 패턴 |
|---|---|
| `us_daily`, `us_weekly` | `SELECT DISTINCT symbol FROM us_stock_basic WHERE is_active=true` 로 universe |
| `earnings_history`, `financials` | 동일 universe 추출. 추가로 `_symbols_due_for_report(75)` skip 로직 공유 |
| `financials_verify` | 활성 종목 중 12개월 내 fiscal data 없는 종목 = silent rate-limit 누락 후보 |
| `stock_basic_compute` | `api` row 의 `(symbol, exchange, currency, sector, industry, beta, is_active)` 메타 → `computed` row 에 carry-over |

### 5.3 Streamlit / API 응답

`reco/latest`, `signals/4_Multi_Wedgepop_Signals.py`, FastAPI 의 종목 디테일 응답 모두 `us_stock_basic` 의 latest row 를 직접 join 해서 `stock_name, sector, market_cap` 등을 채움.

## 6. `source='api'` vs `source='computed'` 의 분업

| 측면 | `source='api'` (이 task) | `source='computed'` (task 7b `stock_basic_compute`) |
|---|---|---|
| 채우는 시점 | `date = date.today()` 만 | `[start_date - 800d, end_date]` 의 매 **거래일** |
| 데이터 출처 | AV `OVERVIEW` (latest snapshot) | pandas as-of merge: `us_daily` + `us_income_statement` + `us_balance_sheet` + `us_cash_flow` |
| 채우는 컬럼 | 56 (메타 포함 전체) | 17 numeric (PE/PEG/EV-Rev/52w/200MA/TTM/YoY 등) + 메타 7 |
| 시점별 정확도 | ❌ 과거 시점은 latest 가 backfill 되는 듯 들어가 look-ahead 위험 | ✅ `fiscal_date_ending + 45d` 이후 financials 만 사용 → PIT 안전 |
| 백테스트 사용 | (단독으로는 위험 — kind='full' 에서는 항상 `computed` 와 함께) | ✅ 백테스트의 진짜 시점별 값 |
| 추천 (reco) 사용 | ✅ 오늘 한 시점만 보면 됨 — `reco` DAG 에서는 `stock_basic_compute` 를 빼고 이것만 씀 | (reco 에서 제외 — 청크 처리 1~2h 절약) |

> **kind='reco'** 에서는 `task_us_stock_basic_compute` 가 DAG 에서 빠진다 (`tasks.py:build_dag` 의 `dag = [t for t in dag if t["id"] != "stock_basic_compute"]`). quant 의 prefetch SQL 이 `WHERE date <= $d ORDER BY date DESC LIMIT 1` 로 source 를 구분하지 않으므로, reco 가 보는 "오늘 시점" 에는 매일 task_stock_basic 이 만든 `source='api'` row (그날 OVERVIEW snapshot) 가 자동 사용된다.
>
> **kind='full'** 에서는 둘 다 적재. 백테스트가 과거 시점에 PIT 정확도를 요구하면 `source='computed'` row 가 latest 가 되도록 (api row 는 오늘 1행만 있고 과거에는 없어서 ORDER BY date DESC 가 자연스럽게 computed 쪽으로 떨어짐).

## 7. PIT / Look-ahead bias 고려사항

OVERVIEW 응답은 "지금 이 순간" 의 값이라 다음 두 케이스에서 look-ahead 가 들어올 수 있다:

1. **`source='api'` 만 있고 backtest 가 과거 시점 분석**: 예) backtest 가 2024-03-15 시점을 분석하는데 `api` row 가 `date=2026-06-05` 만 있다면, `WHERE date <= '2024-03-15'` 조건에서 그 row 는 제외되어 누락. quant 가 시점별 값을 못 찾음.
   → 그래서 `kind='full'` 백테스트는 항상 `stock_basic_compute` 가 `source='computed'` 로 과거 시점들을 채워둔다.

2. **`source='api'` row 가 과거 date 로 저장**: AlphaVantage 가 가끔 latest snapshot 의 `LatestQuarter` 와 다른 시점 데이터를 섞어 보내면, 코드는 그 값을 그대로 today 의 row 로 저장한다. 이건 PIT 위반은 아니지만 (저장된 시점 = 호출 시점), 그 값 자체가 "현재" 가 아닐 수 있다는 한계.

**현재 quant 쿼리는 모두 PIT 가드**:
```sql
WHERE symbol = $1 AND date <= $2          -- analysis_date
ORDER BY date DESC LIMIT 1
```
→ `analysis_date` 이후의 row 는 절대 안 보이게 막아둠. 단, `analysis_date` 와 latest available row 사이 gap 이 크면 stale 값을 보게 된다 (특히 `kind='reco'` 모드에서 due 종목이 아닌 곳).

## 8. 실패 / Skip 시나리오

| 상황 | 처리 |
|---|---|
| AV `"Error Message"` 응답 | `logger.error` + 그 종목 skip, 다음 종목으로 |
| AV `"Information"` (rate limit) 메시지 | 60s sleep 후 retry (최대 `retry_count` 번) |
| 응답에 `Symbol` 키 없음 (delisted / 잘못된 ticker) | `logger.warning("No data found")` + `collection_state` 에 `'no_data'` 마킹 → 다음 run 에서 자동 skip |
| HTTP timeout | exponential backoff retry |
| `ALPHAVANTAGE_API_KEY` 미설정 | `RuntimeError` raise — task 실패 |
| `due_count == 0` | task 정상 종료 (`status='skipped_no_due'`), 후속 task 는 그대로 진행 |

## 9. 운영 노트

- AlphaVantage `OVERVIEW` 는 calls/minute 제한 (premium tier 600/min). `AlphaVantageCollector` 가 비동기로 호출하지만 rate limit 에 맞춰 sleep.
- 멱등성: 같은 날 두 번 돌려도 결과 동일 (UPSERT). 다만 `mark_collected` 가 `collection_state` 에 today date 를 박아둬서, 다음 run 에서는 같은 종목이 due 가 아니면 자동 skip.
- 컨테이너 재시작에도 영향 없음 — alembic 마이그레이션은 멱등.
- `is_active = false` 마킹은 별도 cleanup 흐름에서 (delisting_status 와 cross-check). 이 task 자체는 active 만 fetch.

---

## 한 줄 요약

> **AlphaVantage OVERVIEW 로 분기보고 75일 지난 종목 (~1,500 / day) 의 latest 펀더멘털 50+ 칼럼을 `us_stock_basic (source='api', date=today)` 에 upsert. 후속 task 의 universe 추출 + grade 4 factor 계산의 핵심 입력. 과거 시점 PIT 값은 task 7b `stock_basic_compute` 가 별도로 `source='computed'` 로 채움.**
