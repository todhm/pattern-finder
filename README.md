# pattern-finder

주식 차트 패턴 자동 감지 + 백테스팅 + 실시간 매수 신호 스캐너.

Wedge Pop (Oliver Kell, TraderLion) 패턴을 중심으로 S&P 500 / Nasdaq-100 universe를 스캔하고, 백테스트 + 포지션 사이징 + 엑싯 규칙을 관리하는 Streamlit 기반 워크벤치. Postgres에 영속화된 watchlist로 관심 종목을 근거 메모와 함께 추적.

## 빠른 시작

```bash
# 최초 기동 (이미지 빌드 + DB 마이그레이션 자동 실행)
docker compose up -d --build

# UI 접근
open http://localhost:8501        # Streamlit
open http://localhost:8000/docs   # FastAPI swagger

# 정지
docker compose down
```

컨테이너 기동 시퀀스: `alembic upgrade head` → FastAPI(uvicorn, 8000) + Streamlit(8501) 동시 실행.

## 주요 페이지

| 페이지 | 목적 |
|---|---|
| `1_Pattern_Detection` | 단일 종목 패턴 감지 + 캔들 차트 |
| `2_Backtest_Results` | 단일 종목 백테스트 결과 |
| `strategy` | 단일 종목 Wedge Pop 전략 튜닝 |
| `3_Multi_Wedgepop` | S&P 500 / Nasdaq-100 universe 백테스트 |
| `4_Multi_Wedgepop_Signals` | Multi Wedgepop 전략의 실시간 매수 후보 + **DB 영속 watchlist 관리** (향후 전략마다 개별 Signals 페이지 추가) |

## 아키텍처 (Hexagonal / Ports & Adapters)

```
backtester/
├── data/              OHLCV 수집 (yfinance + 파퀘 캐시 데코레이터)
├── pattern/           패턴 감지 (Wedge Pop, Exhaustion Top, ...)
├── strategy/          전략 실행 (엔트리 + 엑싯 + 포지션 사이징)
├── backtest/          백테스팅 엔진
├── visualization/     Plotly 차트
├── signals/           매수 신호 watchlist (도메인 + Postgres / in-memory 어댑터)
├── db/                공용 DB 인프라 (SQLAlchemy Base + session)
├── alembic/           마이그레이션
├── pages/             Streamlit UI
├── main.py            FastAPI
└── scan_cli.py        배치 스캔 CLI
```

각 도메인은 `domain/`(모델·포트) + `adapters/`(구현체) 분리. 도메인 레이어는 외부 프레임워크 의존 금지.

## Wedge Pop 전략 구성

**엔트리 필터 (signal 바 시점 평가)**
- EMA slow slope 범위
- Signal close-strength (close location in day's range)
- Signal bar euphoria cap ((close−open)/ATR 상한)
- Swing-breakout (signal bar high가 직전 swing high 돌파)
- Market regime (SPY > 200 SMA)

**엔트리 필터 (entry 바 시점 평가 — 다음 바 open 필요)**
- Gap-down rejection
- Gap-up confirmation
- EMA-extension cap

**엑싯 규칙 (5종 — 먼저 발동하는 쪽이 체결)**

| 규칙 | 트리거 |
|---|---|
| `trendline_break` | 최근 swing low들을 잇는 HL 추세선 아래로 low 침투 |
| `resistance_break` | window 내 swing high 레벨이 confirm된 후 low가 level 밑으로 피어싱 |
| `exhaustion_exit` | 바 high가 `ema_fast + extension_atr_mult × ATR` 이상 + slope/close-loc/sell-dom 조건 충족 |
| `breakeven_stop` | close가 `entry + arm_r × R` 이상 도달 후 low가 `entry + offset_r × R` 밑 |
| `smart_trail` | Chandelier exit — highest_high − N × ATR (N = 3/4/5 by R-tier) |

**포지션 사이징**

```
shares = min(capital × risk% / (entry − stop),  capital / entry)
```

두 번째 항은 **no-leverage cap**. 타이트 스탑이 shares를 부풀려 자본 초과 포지션이 되는 걸 방지.

## DB 마이그레이션 (Alembic + SQLAlchemy)

ORM 모델은 각 어댑터 레이어(예: `signals/adapters/orm.py`)에 위치. 모든 모델은 `db.base.Base` 상속.

```bash
# 모델 수정 후 마이그레이션 생성
docker compose exec backtester alembic revision --autogenerate -m "add foo"

# 생성된 alembic/versions/*.py 검토 (autogenerate는 일부 케이스 놓침 —
# 서버 기본값, 복잡한 인덱스 등)

# 적용
docker compose exec backtester alembic upgrade head

# 롤백
docker compose exec backtester alembic downgrade -1

# 현재 상태
docker compose exec backtester alembic current
docker compose exec backtester alembic history
```

`DATABASE_URL`은 docker-compose가 주입. 컨테이너 재시작 시 자동으로 `upgrade head` 실행 — 빠진 마이그레이션 없이 항상 최신 스키마.

## Multi Wedgepop Signals 페이지 기능

`pages/4_Multi_Wedgepop_Signals.py` — `3_Multi_Wedgepop` 백테스트와 동일한 설정으로 **지금 시점의 실매수 후보** 스캔 + 영속 watchlist 관리. 향후 전략마다 별도 Signals 페이지(`5_<strategy>_Signals.py`, `6_..._Signals.py` …) 추가 예정이며, watchlist 저장소(`buy_signals` 테이블)는 전략 간 공유.

- **스캔**: 현재 필터로 최근 N일간 발생한 wedge pop signal 추출. Multi Wedgepop과 동일한 선정 규칙 (일자별 buy/sell ratio 내림차순, `volume_ratio ≥ 1.0` 하드 필터)
- **손절·익절 라인**: 실제 `WedgepopStrategy._find_exit`가 발동하는 가격만 표시 (HL Trendline, Resistance supports/hurdles, Exhaustion Top 임계치, Next resistance)
- **포지션 사이즈**: 자본 × risk% 기반 + no-leverage cap. 자본 대비 % 실시간 계산
- **자동 새로고침**: 저장된 watchlist + 스캔 결과 모두 30분 이상 stale하면 페이지 로드 시 자동으로 최신 바 기준 재계산
- **수동 추가**: ticker + 날짜 입력 → 해당 날짜에 detector가 실제 신호 발화한 경우 BuySignal로 watchlist에 영속 저장
- **watchlist 관리**: 상태 전환 (pending / taken / rejected / expired), 근거 메모 편집, 삭제

## 배치 스캔 CLI

`scan_cli.py` — docker-exec 오버헤드 없이 여러 knob 조합을 일괄 실행.

```bash
# 단일 실행
docker compose exec backtester python scan_cli.py '{"max_tickers":0,"enable_trendline_exit":false}'

# 배치 (여러 variant를 한 번의 프로세스에서)
docker compose exec backtester python scan_cli.py '[
  {"label":"BASE", "knobs":{"max_tickers":0}},
  {"label":"NoTL", "knobs":{"max_tickers":0,"enable_trendline_exit":false}}
]'
```

stderr에 progress 로그, stdout에 최종 JSON 결과, 각 variant 끝날 때마다 `/tmp/batch_partial.json`에 incremental dump.

## 테스트

```bash
docker compose exec backtester pytest -v
```

## 명령어 실행 규칙

모든 명령어는 **반드시 `docker compose`를 통해** 실행. 로컬 환경에서 직접 pip / pytest 등을 실행하지 않음. 자세한 개발 가이드는 [`CLAUDE.md`](./CLAUDE.md) 참조.
