# Pattern Finder - Claude Code 가이드

## 명령어 실행 규칙

모든 명령어는 반드시 `docker compose`를 통해 실행한다. 로컬 환경에서 직접 pip, pytest 등을 실행하지 않는다.

```bash
# 서비스 시작 (빌드 포함)
docker compose up -d --build backtester

# 명령어 실행 (실행 중인 컨테이너에서)
docker compose exec backtester pytest -v
docker compose exec backtester python -c "..."

# 서비스 중지
docker compose down
```

## 프로젝트 구조

- `backtester/` — 하나의 앱. 하나의 Dockerfile. 헥사고날 아키텍처(Ports & Adapters).
  - 각 도메인 패키지는 `domain/`(models, ports)과 `adapters/`(구현체)로 분리.
  - `data/` — 시장 데이터 수집/저장
    - `domain/ports.py`: `MarketDataPort` — OHLCV 데이터 조회 인터페이스
    - `adapters/yfinance_adapter.py`: yfinance 구현체. `end`는 inclusive하게 동작(yfinance 원래 exclusive)
    - `adapters/cached_market_data.py`: 파퀘 기반 디스크 캐시 데코레이터. `end >= today` 요청은 **자동 우회** — 장중 intraday 업데이트 반영
    - `adapters/wikipedia_universe.py`: S&P 500 / Nasdaq-100 종목 리스트
    - `domain/ports.py: IpoCalendarPort` — 기간별 IPO(첫 거래일·공모가). `adapters/finnhub_ipo_calendar.py`(Finnhub, 2010~, Mongo 월 캐시) → `adapters/alphavantage_listing_ipos.py`(LISTING_STATUS ipoDate 폴백). composition: `adapters/composed_ipo_calendar.py`
  - `pattern/` — 패턴 감지
    - `domain/ports.py`: `PatternDetector` — 패턴 감지 인터페이스 (EMA, 평균 거래량 헬퍼 포함)
    - `adapters/wedge_pop.py`, `adapters/exhaustion_extension_top.py` 등 — 개별 패턴 구현체
    - `helpers/pivots.py` — swing high/low, trendline fit (numpy 벡터화 완료)
  - `strategy/` — 전략 실행 (패턴 + 사이즈 + 엑싯 조합)
    - `adapters/wedgepop_strategy.py`: 단일 ticker Wedge Pop. 엑싯 규칙 5종(trendline_break, resistance_break, exhaustion_exit, breakeven_stop, smart_trail). No-leverage 포지션 캡
    - `adapters/multi_wedgepop_strategy.py`: 전체 universe에서 일자별 buy/sell ratio 최상위 종목 선택
  - `backtest/` — 백테스팅 엔진
  - `visualization/` — 시각화 (`adapters/plotly_charts.py`)
  - `signals/` — **매수 신호 watchlist + 영속화**
    - `domain/models.py`: `BuySignal`, `SignalStatus`(pending/taken/rejected/expired)
    - `domain/ports.py`: `SignalRepositoryPort`, `SignalScannerPort`
    - `adapters/in_memory_repo.py`: 프로세스 메모리 저장 (DB 불가 시 fallback)
    - `adapters/postgres_signal_repo.py`: **SQLAlchemy 기반** Postgres 어댑터
    - `adapters/orm.py`: `BuySignalRow` SQLAlchemy 모델 + 도메인 변환기
    - `adapters/universe_scanner.py`: 현재 필터로 최근 N일 신호 스캔. `refresh_targets()` / `build_signal_at(ticker, date)` 제공
  - `db/` — **DB 인프라** (공용)
    - `base.py`: `DeclarativeBase` (모든 ORM 모델이 상속)
    - `session.py`: engine singleton + `session_scope()`
  - `alembic/` — 마이그레이션 (컨테이너 기동 시 `alembic upgrade head` 자동 실행)
    - `env.py`: `DATABASE_URL` 환경변수 우선 사용, `target_metadata = Base.metadata`
    - `versions/`: 마이그레이션 파일
  - `streamlit_app.py` + `pages/` — Streamlit UI
    - `pages/1_Pattern_Detection.py`, `pages/2_Backtest_Results.py`, `pages/strategy.py`
    - `pages/3_Multi_Wedgepop.py`: universe 전체 백테스트
    - `pages/51_Lifecycle_Trade.py`: **«The Lifecycle Trade» IPO 생애주기** — 7장 스크린(IPO Alert/Rare Jewels/Liquidity Matters) + 6장 통계 재현 + 4매수지점×5매도규칙 $100K 시뮬 + ⑤ 계좌 시뮬(책 8장 자금관리로 그대로 돌리면 얼마). 엔진 `strategy/adapters/lifecycle_trade.py`, 계좌 엔진 `strategy/adapters/lifecycle_portfolio.py`, 배치 `lifecycle_batch_cli.py` → `lifecycle_portfolio_cli.py`, 문서 `docs/strategy_notes/라이프사이클_트레이드_*.md`
    - `pages/4_Multi_Wedgepop_Signals.py`: **Multi Wedgepop 실매수 후보 + watchlist 관리** — `3_Multi_Wedgepop` 백테스트와 동일 설정으로 현재 시점 signal 스캔. DB 영속 (`buy_signals` 테이블, 전략 간 공유), 최신 데이터 자동 리프레시, 수동 추가. 향후 전략마다 별도 `<N>_<strategy>_Signals.py` 페이지 추가 (scanner만 `SignalScannerPort` 구현체로 추가하면 됨).
  - `main.py` — FastAPI (uvicorn)
  - `minervini_trades_check_cli.py` — 거래 로그 CSV(Ticker, Entry Date…)의 각 진입을 진입 전날 기준 미너비니 STAGE 0~4 로 재판정 + 눈판독용 차트 PNG. 결과 예: `docs/strategy_notes/미너비니_WedgePop거래28건_*.md`
  - `minervini_watchlist_check_cli.py` — 44 스크리너 내보내기 CSV(티커, 산업군, RS…)의 후보 전부를 기준일 종가 기준 STAGE 0~4 재판정 + 차트 PNG + 종목별 도시에(주봉·일봉·실적·헤드라인). 0-3 은 `minervini_checklist_backtest_cli.py live` 결과(`live_json`)를 넘긴다. 결과 예: `docs/strategy_notes/미너비니_2026-10-13_스크리너_*.md`
  - `cache_fill_cli.py` — **캐시 앞단 채우기**. 날짜 단위 Mongo 캐시는 기준일이 거래일 하루만 앞으로 가도 전 종목이 미스가 되어 종목당 Alpha Vantage 1콜(분당 75콜 → 5,000종목 64분)을 다시 했다. `data/adapters/cache_frontier_fill.py` 가 Massive(Polygon) grouped-daily(`BulkDailyBarsPort`, 날짜당 1콜로 전 종목)로 최근 15거래일을 먼저 채우고, 분할 종목은 캐시를 비워 재조회시키며, grouped 에도 없고 최근 어느 소스가 'No data found' 를 기록한 폐지 심볼(SPAC 유닛·워런트 등 약 1,000개)에는 네 캐시 전부에 nodata 마커를 써서 거래일마다 네 소스를 두드리는 비용을 없앤다. 44 스크리너 페이지는 Run 시 자동 호출. 장 마감 후 크론으로 돌려 두면 첫 실행도 빠르다: `python cache_fill_cli.py '{"universe": "nasdaq_full@기준일", "as_of": "YYYY-MM-DD"}'`
  - `scan_cli.py` — 배치 스캔 CLI. JSON config(single or list) → JSON 결과. progress 로그 stderr + `/tmp/batch_partial.json` incremental dump
  - `tests/` — pytest 테스트
- `docker-compose.yaml` — backtester + PostgreSQL(db) 서비스. 컨테이너 기동 시 순서: `alembic upgrade head` → uvicorn + streamlit

## Docker 이미지

`.env` 파일로 Harbor/DockerHub 전환 가능. `.env` 없으면 DockerHub 기본값 사용.
backtester 컨테이너 안에서 uvicorn(8000)과 streamlit(8501)이 함께 실행된다.

## DB 마이그레이션 (Alembic)

스키마 변경 워크플로우:

```bash
# 1) ORM 모델 수정 (signals/adapters/orm.py 등)
# 2) 마이그레이션 자동 생성
docker compose exec backtester alembic revision --autogenerate -m "설명"
# 3) 생성된 alembic/versions/*.py 검토 — autogenerate가 놓치는 케이스 있음
# 4) 적용
docker compose exec backtester alembic upgrade head
# 되돌리기
docker compose exec backtester alembic downgrade -1
```

- 새 ORM 모델 추가 시 반드시 `db.base.Base` 상속 + `alembic/env.py`에서 import (autogenerate 스캔 대상)
- `DATABASE_URL`은 docker-compose가 주입(`postgresql://backtester:backtester@db:5432/backtester`)
- 컨테이너 재시작 시 자동으로 `upgrade head` 실행 — 프로덕션/개발 동일

## 헥사고날 아키텍처 규칙

- 도메인 레이어(`*/domain/`)는 **외부 프레임워크 의존 금지** — SQLAlchemy / yfinance / streamlit 등 import 불가
- 어댑터 레이어(`*/adapters/`)만이 외부 라이브러리에 의존
- 포트(`domain/ports.py`)는 ABC로 선언, 어댑터가 구현
- 새 저장소/데이터소스 추가 시: port 먼저 정의 → 테스트용 in-memory 어댑터 → 프로덕션 어댑터 (DB 등) 순서
- Composition root는 Streamlit page / FastAPI main / CLI — 이 레이어에서만 구현체 선택 후 주입

## Claude Code 설정

- `.claude/settings.json` — 팀 공유 설정 (git 추적)
- `.claude/settings.local.json` — 개인 permissions (git 제외)
