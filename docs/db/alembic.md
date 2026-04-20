# DB — SQLAlchemy + Alembic

## 구성 요소

| 위치 | 역할 |
|---|---|
| `backtester/db/base.py` | `DeclarativeBase` — 모든 ORM 모델의 공통 부모 |
| `backtester/db/session.py` | `get_engine()`, `session_scope()` — engine/session 싱글톤 |
| `backtester/signals/adapters/orm.py` | `BuySignalRow` 등 ORM 모델 (각 도메인 어댑터 레이어에 분산 배치) |
| `backtester/alembic/env.py` | `DATABASE_URL` 환경변수 → `sqlalchemy.url`, `target_metadata = Base.metadata` |
| `backtester/alembic/versions/` | 마이그레이션 스크립트 |
| `backtester/alembic.ini` | Alembic 설정 |

## 환경변수

`DATABASE_URL` — docker-compose가 backtester 서비스에 주입. 기본값:

```
postgresql://backtester:backtester@db:5432/backtester
```

컨테이너 밖에서 실행하려면 `DATABASE_URL`을 직접 export.

## 일상 워크플로우

### 모델 수정 → 마이그레이션 생성
```bash
# 1. signals/adapters/orm.py (또는 다른 모듈의 ORM) 수정
# 2. autogenerate
docker compose exec backtester alembic revision --autogenerate -m "add column X to buy_signals"
# 3. 생성된 alembic/versions/*.py 검토 — autogenerate 놓치는 케이스
#    - JSONB 기본값, 서버 side default
#    - Enum 타입 변경
#    - 복합 인덱스 이름
# 4. 적용
docker compose exec backtester alembic upgrade head
```

### 새 ORM 모델 등록

1. 모듈에서 `from db.base import Base` 후 `class FooRow(Base): __tablename__ = ...`
2. `alembic/env.py`의 import 블록에 모듈 추가 (metadata 스캔 대상):
   ```python
   from signals.adapters import orm as _signals_orm  # noqa: F401
   from foo.adapters import orm as _foo_orm          # ← 추가
   ```
3. 마이그레이션 생성 + 적용

### 초기 배포 / 재시작

`docker-compose.yaml`의 command가 `alembic upgrade head`를 기동 전에 실행하므로 별도 조치 불필요. 이미 head인 경우 noop.

### 다운그레이드 / 히스토리

```bash
docker compose exec backtester alembic downgrade -1           # 1단계 롤백
docker compose exec backtester alembic downgrade <rev_id>     # 특정 리비전까지
docker compose exec backtester alembic current                # 현재 적용된 head
docker compose exec backtester alembic history --verbose      # 전체 히스토리
```

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `no such revision 'head'` | `alembic/versions/` 가 비어있음 → 초기 마이그레이션 autogenerate 필요 |
| `autogenerate` 가 변경사항 안 감지 | 새 모듈의 ORM을 `env.py`에서 import 안 했음. 또는 Base를 상속 안 했음 |
| `relation "..." already exists` | DB에 수동 생성된 테이블이 먼저 있음 → 수동 drop 후 `alembic upgrade head`, 또는 `alembic stamp head`로 "이미 적용됨" 마킹 |
| 빈 마이그레이션 (`pass`만 있음) | 스키마 diff 없음. 정상 |

## 패턴

- **ORM 모델은 어댑터 레이어에**. 도메인 dataclass와 분리해 프레임워크 결합 없게
- **도메인 변환기**는 ORM row에 `to_domain()` / `from_domain()` classmethod로 배치
- **session 수명은 메서드 단위 짧게**. `session_scope()` 문맥 안에서만 open → commit → close
- **마이그레이션 파일은 review 필수**. autogenerate를 맹신하지 말고 다운그레이드 경로 검증
