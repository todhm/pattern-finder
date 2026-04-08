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

- `backtester/` — 하나의 앱. 하나의 Dockerfile. 도메인 분리는 코드 레벨 패키지로만.
  - `data/` — 시장 데이터 수집/저장
  - `pattern/` — 패턴 감지 (Strategy 패턴으로 확장)
  - `backtest/` — 백테스팅 엔진
  - `visualization/` — 시각화
- `docker-compose.yaml` — 루트에서 앱 단위로 서비스 정의

## Docker 이미지

`.env` 파일로 Harbor/DockerHub 전환 가능. `.env` 없으면 DockerHub 기본값 사용.

## Claude Code 설정

- `.claude/settings.json` — 팀 공유 설정 (git 추적)
- `.claude/settings.local.json` — 개인 permissions (git 제외)
