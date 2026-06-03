# Orchestrator DAG — Task ↔ Table Mapping

`alphafolio_data` 의 orchestrator (`orchestrator/tasks.py` + `build_dag()`) 가
실행하는 각 task 가 어떤 DB 테이블을 적재/갱신하는지 정리한 문서.
백테스트 (`kind='full'`) / 추천 (`kind='reco'`) DAG 모두 cover.

## DAG 위상 (US, `kind='full'`)

```
partitions ──┬─ stock_listing ─ finnhub_symbol ─ stock_basic ─┬─ us_daily ─┐
             │                                                 │            ├─ us_calculator ─┐
             ├─ listing_status                                  ├─ us_weekly ┤                 │
             ├─ us_etf                                          │            │                 │
             └─ macros                                          └─ earnings_history ─┐         │
                                                                                     │         │
                          ┌──── stock_basic_compute ── financials_verify ── financials
                          │                                                          
                          ├──── em8_pre_filter ─────────────────────────────────────┤
                          ├──── us_mv_sector_refresh ───────────────────────────────┤
                          │                                                          │
                          └──── grades_pass_a ── select_top_n ──┬── options_top_n ──┐
                                                                │                     │
                                                                └── news_history_backfill ── grades_pass_b ── backtest
```

`kind='reco'` 는 종단을 `backtest` → `top_signal_reco` 로 교체하고
`news_insider_top_n` (당일 top-3 한정) 을 `select_top_n` 다음에 추가.

## Task ↔ Table 매핑

| seq | Task ID | 적재 / 갱신 테이블 | 데이터 단위 | Skip 패턴 | 비고 |
|----:|---------|-------------------|------------|-----------|------|
| 0 | `partitions` | (DDL only) | 파티션 메타 | n/a | 월별 파티션 생성, 매 run 멱등 |
| 1 | `stock_listing` | Railway Volume CSV (`/app/log/us_common_stocks_*.csv`) | symbol list snapshot | n/a | NASDAQ+NYSE 풀 fetch |
| 1b | `listing_status` | `us_listing_status` | (symbol) PIT — `ipo_date`, `delisting_date`, `asset_type` | n/a (cheap, 2 calls) | active + delisted 합본. PIT universe 의 ground truth |
| 2 | `finnhub_symbol` | `us_symbol` | (symbol) snapshot | n/a | Finnhub symbol master |
| 3 | `stock_basic` | `us_stock_basic` (`source='api'`) | `(symbol, date=today)` snapshot | `collection_state` no_data + recent (7d) | AV OVERVIEW endpoint |
| 4 | `us_daily` | `us_daily` | `(symbol, date)` OHLCV | `collection_state.get_missing_dates()` per-date | outputsize=full (20+년 한 콜) |
| 5 | `us_etf` | `us_daily_etf` | `(symbol, date)` OHLCV (SPY/QQQ/sector ETFs) | per-date missing | HMM regime detector input |
| 5b | `us_weekly` | `us_weekly` | `(symbol, date)` weekly OHLCV | `check_existing_data` per-date | `start_date` parameter (default 2015-01-01) — backtest 윈도우에 맞춰 ctx 주입 |
| 6 | `us_calculator` | `us_indicators` | `(symbol, date)` — 14종 지표 통합 row | `pending_dates = trading_days - done_dates` | EM8 / ATR / EMA / MACD / BBands / RSI 등 |
| 7a | `earnings_history` | `us_earnings_history` | `(symbol, fiscal_date_ending)` + `reported_date` (실제 SEC 공시일) | publication-aware (75d due filter) | AV EARNINGS endpoint (121분기/30년) |
| 7 | `financials` | `us_income_statement`, `us_balance_sheet`, `us_cash_flow` | `(symbol, fiscal_date_ending)` + `available_at` (= earnings_history.reported_date or +45d fallback) | 6개월 내 fiscal data skip + `collection_state` no_data | AV 응답이 ~81분기 (20년) — `MAX_QUARTERS` 절단 제거 |
| 7a-2 | `financials_verify` | `us_income_statement` / `balance_sheet` / `cash_flow` + log only | 위와 동일 + fallback 비율 로그 | gap-fill (12개월 내 없는 종목만 좁은 universe 재시도) | silent rate-limit drop 보호 |
| 7b | `stock_basic_compute` | `us_stock_basic` (`source='computed'`) | `(symbol, date)` 시점별 17 컬럼 (PE/PEG/EV-Rev/52w high/200MA …) | `ON CONFLICT (symbol, date, source)` UPSERT | 청크 처리 (1년 × N 청크, lookback 800d). pandas as-of merge, look-ahead-bias 안전 |
| 8 | `macros` | `us_fed_funds_rate`, `us_treasury_yield`, `us_cpi`, `us_unemployment_rate` | `(date, [maturity])` series | 매번 fetch + `ON CONFLICT DO UPDATE` (4 calls cheap) | `start_date` parameter (default 2010-01-01) |
| 9 | `em8_pre_filter` | `daily_top_symbols` | `(date, symbol, rank, em8_score)` — 일자별 top-500 | 일자별 skip (이미 계산된 date 건너뜀) | 252-day IBD-RS lookback |
| 9b | `us_mv_sector_refresh` | `mv_us_sector_daily_performance` (materialized view) | `(date, sector_code, momentum, rank)` | per-date upsert | sector rotation score 입력 |
| 10 | `grades_pass_a` | `us_stock_grade` | `(symbol, date)` — 17 컬럼 등급 (Pass-A, event_modifier off) | `skip_existing=True` 옵션 (이미 grade 있는 day skip) | EM8 pre-filter top-N 만 grade |
| 10b | `select_top_n` | (in-memory list, no table) | — | n/a | grade DESC 상위 N (option backfill 대상 추출) |
| 11 | `options_top_n` | `us_option_daily_summary` (raw `us_option` 파티션은 summary 후 drop) | `(symbol, date)` 옵션 IV/Volume/GEX 요약 | 일자별 summary 존재 시 skip | top-N × date 만 — options 비용 절감 |
| 11b | `news_history_backfill` | `us_news` | `(url, ticker, time_published)` + sentiment | `(symbol, quarter_start)` `collection_state` 영구 success 마킹 | top-N union × 분기 chunks (AV NEWS_SENTIMENT 2018-01~) |
|  | `news_insider_top_n` (reco only) | `us_news`, `us_insider_transactions` | 당일 top-3 종목 한정 | n/a | reco DAG 에서만 등장 |
| 12 | `grades_pass_b` | `us_stock_grade` (`ON CONFLICT DO UPDATE`) | `(symbol, date)` 등급 재산정 | `skip_existing=False` | option / news / insider event_modifier 포함 |
| 13 | `backtest` | `backtest_runs`, `backtest_trades`, `backtest_nav_history` | run_id 별 trades + NAV path + metrics JSON | `ON CONFLICT` (run_id PK) | top-N grade 매수, rebal 주기마다 교체 |
|    | `top_signal_reco` (reco only) | (no table — orchestrator output JSON) | as_of_date의 top-N 추천 (buy_now + watchlist) | n/a | reco DAG 종단. `/reco/latest` endpoint 가 이 task output 조회 |

## DAG 외부 collector (cron 직접 호출, DAG 무관)

`/collect/us/daily` endpoint 안에서 cron 별도 실행 — backtest run 의 DAG 에 직접 들어있지 않지만 quant 가 사용하는 데이터.

| Collector | 적재 테이블 | Skip 패턴 |
|-----------|-------------|-----------|
| `InsiderTransactionsCollector` | `us_insider_transactions` | **(symbol, transaction_date) missing-date** (`get_symbols_with_missing_dates` helper). fetch 후 expected - received → `collection_state` no_data 영구 마킹 |
| `EarningsCalendarCollector` | `us_earnings_calendar` | bulk API (symbol parameter 없음) — 매번 풀 fetch + ON CONFLICT |
| `DividendsCollector` | `us_dividends` | (symbol, ex_dividend_date) missing-date 동일 패턴 |
| `SplitsCollector` | `us_splits` | (symbol, effective_date) missing-date 동일 패턴 |
| `USNewsCollector` (정기 호출, 별도) | `us_news` | url 단위 dedup |

## Skip 로직 표준 패턴 정리

| 패턴 | 적용 collector | 의미 |
|------|---------------|------|
| **Date-aware missing 직접** | `us_daily`, `us_weekly`, `us_etf`, `us_calculator`, `em8_pre_filter`, `options_top_n` | `(symbol or date)` 별 missing date 만 fetch |
| **collection_state no_data 영구 마킹** | `us_daily`, `us_weekly`, `us_stock_basic`, `us_income/balance/cashflow`, `us_earnings_history`, `us_insider_transactions`, `us_dividends`, `us_splits` | AV 응답에 실제 데이터 없는 (symbol, date) 영구 skip |
| **Helper `get_symbols_with_missing_dates`** | `InsiderTransactions`, `Dividends`, `Splits` | us_daily 거래일 expected − DB existing − collection_state no_data → unknown 있는 symbol set 반환 |
| **6개월 fiscal date skip** | `financials` 3종 | 분기 보고서 주기 (90일) 와 정합. 같은 분기 중복 fetch 회피 |
| **Publication-aware due filter** | `earnings_history` | reported_date 기준 75일 due 종목만 재수집 |
| **No skip (매번 fetch)** | `macros`, `EarningsCalendarCollector`, `partitions`, `stock_listing`, `finnhub_symbol`, `listing_status` | 한 콜에 풀 history 응답 + ON CONFLICT dedup. cheap 케이스 |

## `available_at` (look-ahead bias 방어)

`us_income_statement`, `us_balance_sheet`, `us_cash_flow` 의 `available_at` 컬럼은
quant scoring 의 시점별 조회 (`WHERE available_at <= analysis_date`) 기준점.

```sql
-- collector 가 INSERT 시 자동 채움 (finance_data.py:587-601 등)
UPDATE us_income_statement i
SET available_at = COALESCE(
  (SELECT eh.reported_date FROM us_earnings_history eh
   WHERE eh.symbol = i.symbol AND eh.fiscal_date_ending = i.fiscal_date_ending),
  (i.fiscal_date_ending + INTERVAL '45 days')::date
)
```

우선순위: `us_earnings_history.reported_date` (실제 SEC 공시일) → fallback `+45d` (10-Q 마감일).
`financials_verify` 가 fallback 비율을 분기-연도별로 로깅 → leak 위험 정량화.

## PIT universe (survivorship bias 방어)

`us_listing_status` 의 `(ipo_date, delisting_date, asset_type)` 가 quant
`_load_analysis_symbols(analysis_date)` 의 universe 필터:

```sql
SELECT symbol FROM us_listing_status
WHERE (ipo_date IS NULL OR ipo_date <= $analysis_date)
  AND (delisting_date IS NULL OR delisting_date > $analysis_date)
  AND (asset_type IS NULL OR asset_type = 'Stock')
```

테이블 비어있으면 (마이그레이션 직후) 자동으로 legacy `us_stock_basic` 쿼리로
fallback 하면서 warning 로그.
