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
    - `adapters/yfinance_adapter.py`: yfinance 구현체
  - `pattern/` — 패턴 감지
    - `domain/ports.py`: `PatternDetector` — 패턴 감지 인터페이스 (EMA, 평균 거래량 헬퍼 포함)
    - `adapters/reversal_extension.py`, `adapters/wedge_pop.py`: 개별 패턴 구현체
  - `strategy/` — 전략 실행 (패턴 + 백테스트 조합)
    - `domain/ports.py`: `StrategyRunnerPort` — 전략 실행 인터페이스
    - `adapters/runner.py`: 구현체
  - `backtest/` — 백테스팅 엔진
    - `domain/ports.py`: `BacktestEnginePort` — 백테스트 실행 인터페이스
    - `adapters/engine.py`: 구현체
  - `visualization/` — 시각화
    - `domain/ports.py`: `ChartBuilderPort` — 캔들스틱, 에퀴티 커브 차트 인터페이스
    - `adapters/plotly_charts.py`: Plotly 구현체
  - `streamlit_app.py` + `pages/` — Streamlit UI
  - `main.py` — FastAPI (uvicorn)
  - `tests/` — pytest 테스트
- `docker-compose.yaml` — backtester + PostgreSQL(db) 서비스 정의

## Docker 이미지

`.env` 파일로 Harbor/DockerHub 전환 가능. `.env` 없으면 DockerHub 기본값 사용.
backtester 컨테이너 안에서 uvicorn(8000)과 streamlit(8501)이 함께 실행된다.

## Claude Code 설정

- `.claude/settings.json` — 팀 공유 설정 (git 추적)
- `.claude/settings.local.json` — 개인 permissions (git 제외)
