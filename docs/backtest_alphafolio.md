# Alphafolio 백테스트 설계

세 라이브 서비스(`alphafolio_data` / `alphafolio_quant` / `alphafolio_portfolio`)를 그대로 사용해 과거 기간의 포트폴리오 운용 결과를 재현하는 방법론.

> 운영 코드(`live`)와 백테스트 코드를 *별도 코드*로 만들지 않는다. 같은 함수가 `analysis_date` 만 받아 굴러간다 — 라이브와 backtest의 결과가 일치한다는 보장은 같은 코드라는 점에서 출발.

---

## 1. 아키텍처

```
   ┌──────────────┐    backtest_id, date    ┌──────────────┐
   │ backtest     │───►  POST /us/run      ►│ alphafolio_  │
   │ runner       │                         │ quant        │
   │ (date loop)  │                         └──────┬───────┘
   │              │                                │ INSERT
   │              │                                ▼
   │              │                         ┌──────────────┐
   │              │───►  /recommend/daily  ►│ alphafolio_  │
   │              │                         │ portfolio    │
   └──────────────┘                         └──────┬───────┘
                                                   │ INSERT
                                                   ▼
                                            ┌──────────────┐
                                            │ Postgres     │
                                            │ alphafolio   │
                                            │  └ rows tagged│
                                            │    (mode,    │
                                            │    backtest_ │
                                            │    id)       │
                                            └──────────────┘
```

`alphafolio_data` 는 백테스트 루프에서 호출하지 **않는다** — 과거 데이터는 이미 DB에 적재되어 있다는 가정. 백테스트의 일이 *수집 재현*이 아니라 *분석/포트폴리오 의사결정 재현*이기 때문.

---

## 2. 백테스트 격리 — `(mode, backtest_id)` 패턴

`alphafolio_backtest/migrations/001_add_backtest_columns.sql` 가 다음 5개 테이블에 컬럼을 추가:

- `portfolio_master`
- `portfolio_holdings`
- `portfolio_transactions`
- `portfolio_rebalancing`
- `portfolio_daily_performance`

```sql
mode        VARCHAR(10) NOT NULL DEFAULT 'LIVE',
backtest_id VARCHAR(64) NULL,
INDEX (mode, backtest_id)
```

라이브 행은 default 값으로 자동 분리, 백테스트는 runner가 모든 INSERT에 태깅. 종료 후:

```sql
DELETE FROM portfolio_daily_performance WHERE backtest_id = 'bt_us_2024';
DELETE FROM portfolio_rebalancing       WHERE backtest_id = 'bt_us_2024';
DELETE FROM portfolio_transactions      WHERE backtest_id = 'bt_us_2024';
DELETE FROM portfolio_holdings          WHERE backtest_id = 'bt_us_2024';
DELETE FROM portfolio_master            WHERE backtest_id = 'bt_us_2024';
```

> 퀀트 결과(`kr_stock_grade` / `us_stock_grade`) 도 동일 컬럼을 추가해야 라이브 등급을 덮어쓰지 않는다. 백테스트 일자별 등급은 한 번 산출하면 재사용되므로(여러 portfolio 설정 비교 시) 중복 계산 비용 절감 효과도 있음.

---

## 3. 서비스 엔드포인트 확장 (TODO)

스켈레톤 runner의 contract는 다음과 같다. 실제로 받게 만들려면:

### 3.1 alphafolio_quant

```python
# app.py
@app.post("/us/run")
async def us_run(
    date: date | None = None,
    mode: Literal["LIVE", "BACKTEST"] = "LIVE",
    backtest_id: str | None = None,
    api_key: str = Depends(verify_api_key),
):
    from us.us_main import UsQuantSystem
    system = UsQuantSystem(mode=mode, backtest_id=backtest_id)
    result = await system.run(analysis_date=date)
    if mode == "LIVE":
        chain_result = await _call_portfolio_service("US")
    return {...}
```

`us_main.UsQuantSystem.run` 은 이미 `analysis_date: Optional[date]` 를 받으므로 호출부만 노출하면 된다. KR 측 `kr_main.run_option1` 도 `analysis_date` 를 받도록 시그니처 확장 (`analyze_single_stock` 은 이미 받음 — 메인 루프에 인자 전달).

### 3.2 alphafolio_portfolio

```python
# app.py
@app.post("/recommend/daily")
async def recommend_daily(request: Request, api_key: str = Depends(verify_api_key)):
    body = await request.json()
    country = body["country"]
    as_of   = body.get("date")
    mode    = body.get("mode", "LIVE")
    bt_id   = body.get("backtest_id")

    recommender = DailyRecommender(mode=mode, backtest_id=bt_id)
    recs, data_date = await recommender.get_recommendations(country, as_of=as_of)
    ...
    runner = RebalancingRunner(mode=mode, backtest_id=bt_id)
    await runner.run(live_only=(mode == "LIVE"), country=country, as_of=as_of,
                     skip_agent_report=(mode == "BACKTEST"))
```

`DailyRecommender` / `RebalancingRunner` / `portfolio_db.py` 의 모든 INSERT 가 `(mode, backtest_id)` 를 함께 쓰도록 패치. 가격/등급 조회 쿼리는 `as_of` 를 `<=` 필터로 변환.

---

## 4. PIT (Point-in-Time) 데이터 — 가장 큰 함정

같은 코드라도 *데이터* 가 미래를 알면 결과는 거짓이다. 현재 스키마의 위험 지점:

| 테이블 | 위험 | 권장 변경 |
|---|---|---|
| `kr_indicators`, `us_indicators` | 매일 덮어쓰기 → 과거 일자 재계산 시 *오늘* 의 RSI/MACD 사용 (look-ahead) | `as_of_date` 컬럼 추가, PK를 `(symbol, as_of_date)` 로. 과거 backfill 필요 |
| `kr_stock_basic`, `us_stock_basic` (섹터/시총) | 섹터 재분류·시총 변경이 덮어써짐 | SCD-2 (valid_from / valid_to) 또는 monthly snapshot |
| DART 재무제표 (`kr_financial_position` 등) | 회기 종료일 ≠ 공시일. naive 조회는 미공시 데이터 노출 | `report_filed_at` 컬럼 + `WHERE report_filed_at <= analysis_date` |
| Alpha Vantage 추정치 revision | 이력 보존 안 됨 | `*_history` 테이블 + 매일 추정치 스냅샷 |
| `kr_stock_prediction_stats` / `us_stock_prediction_stats` | 미래 90일 수익이 이미 입력된 상태 (적중률 계산 자체에 lookahead 내재) | runner가 `as_of` 이전 history 만으로 집계하도록 view 분리 또는 `prediction_made_at <= as_of` 필터 |

**검증 방법**: 동일 백테스트를 다른 시점에 두 번 실행 → 결과가 **달라지면** 어딘가 미래 데이터 누수. 같으면 PIT 보장됨.

---

## 5. 워크포워드 검증

한 윈도우 통째 백테스트의 hit rate는 **거의 항상 over-fitted**. 분할:

```
2020 ─── 2022  : in-sample   (파라미터 결정)
2023          : validation   (튜닝 멈춤)
2024 ─── 2025 : test         (한 번만 측정, 절대 다시 안 봄)
```

각 분할에 다른 `--bt-id` 부여 → 결과는 별도 row 집합 → 한 SQL 로 비교 가능.

IC 안정성 모니터링은 `alphafolio_quant` 가 이미 `us_ic_monitor.py` 로 갖고 있다 — out-of-sample IC 가 in-sample 의 50% 이상 유지되어야 신뢰 가능.

---

## 6. 측정 지표

`portfolio_daily_performance` 가 일별 NAV/Sharpe/MDD/벤치마크 alpha 를 이미 계산하므로 SQL aggregate:

```sql
WITH daily AS (
    SELECT trade_date,
           total_return_pct,
           benchmark_return_pct,
           cumulative_return_pct
    FROM portfolio_daily_performance
    WHERE backtest_id = 'bt_us_2024'
    ORDER BY trade_date
)
SELECT
    COUNT(*)                                                    AS days,
    AVG(total_return_pct - benchmark_return_pct) * 252          AS annual_alpha,
    AVG(total_return_pct) / STDDEV(total_return_pct) * SQRT(252) AS sharpe,
    MIN(cumulative_return_pct)                                  AS mdd_pct,
    SUM(CASE WHEN total_return_pct > benchmark_return_pct THEN 1 ELSE 0 END)::FLOAT
        / COUNT(*)                                              AS daily_hit_rate
FROM daily;
```

추가로 보면 좋은 지표:
- **턴오버**: `SELECT SUM(trade_amount)/AVG(nav) FROM portfolio_transactions WHERE backtest_id=...`
- **체결 비용 영향**: 슬리피지/수수료 차감 전 vs 후 alpha 차이
- **트리거별 P&L**: `portfolio_rebalancing_detail` × `portfolio_transactions` 조인으로 9개 트리거의 기여도

---

## 7. 실행 순서 (정리)

```bash
# 1. 라이브 스택 띄우기 (alphafolio_data 는 한 번 수집 후 stop OK)
docker compose up -d db alphafolio_quant alphafolio_portfolio

# 2. 백테스트 컬럼 마이그레이션 (최초 1회)
docker compose exec -T db psql -U alphafolio -d alphafolio \
    < alphafolio_backtest/migrations/001_add_backtest_columns.sql

# 3. (TODO) endpoint 확장 패치 적용 — 위 3절 참조

# 4. (TODO) PIT 데이터 backfill — 위 4절 참조

# 5. 백테스트 실행
docker compose --profile backtest run --rm alphafolio_backtest \
    python runner.py --start 2024-01-01 --end 2024-12-31 \
                     --country US --bt-id bt_us_2024_v1

# 6. 결과 분석
docker compose exec -T db psql -U alphafolio -d alphafolio \
    < /path/to/analysis.sql

# 7. 폐기
docker compose exec -T db psql -U alphafolio -d alphafolio -c \
    "DELETE FROM portfolio_daily_performance WHERE backtest_id='bt_us_2024_v1'; ..."
```

---

## 8. 코드 위치

| 항목 | 경로 |
|---|---|
| Runner | `alphafolio_backtest/runner.py` |
| 마이그레이션 | `alphafolio_backtest/migrations/001_add_backtest_columns.sql` |
| Dockerfile | `alphafolio_backtest/Dockerfile` (profile=backtest) |
| compose service | `docker-compose.yaml::alphafolio_backtest` |
| 라이브 서비스 docker | `alphafolio_{data,quant,portfolio}/Dockerfile` |

설계 변경 시 본 문서를 같이 갱신.
