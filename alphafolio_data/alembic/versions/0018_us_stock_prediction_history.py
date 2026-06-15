"""us_stock_prediction_history + us_stock_prediction_stats — grade 적중률 추적.

배경:
  alphafolio_quant 의 run_option1 은 각 분석일마다 (1) us_stock_grade 저장 후
  (2) us_prediction_collector.batch_record_grades 로 그 등급을 예측 이력
  테이블에 적재한다. 이 두 테이블은 quant 코드가 "이미 존재한다"고 가정만
  할 뿐 어디에서도 CREATE 되지 않아, 공유 alphafolio DB 에 한 번도 만들어진
  적이 없었다. 그 결과 generate-grades 루프의 매 날짜가 step (2) 에서
  ``UndefinedTableError: relation "us_stock_prediction_history" does not exist``
  로 실패 처리되어 (grade 자체는 step (1) 에서 저장되지만) 오케스트레이터
  DAG 의 grades_pass_a/b 가 processed=0, failed=전체 로 보고됐다.

  alphafolio_data 와 alphafolio_quant 는 같은 alphafolio DB 를 공유하고,
  컨테이너 기동 시 alphafolio_data 가 alembic upgrade head 를 돌리므로 이
  두 테이블을 여기서 생성한다.

스키마 (us_prediction_collector.py 의 쿼리에서 역도출):
  history:
    - (symbol, grade_date) PK         — ON CONFLICT (symbol, grade_date)
    - predicted_grade / grade_direction(BUY|SELL)
    - actual_return_90d               — batch_update_returns 가 90일 후 채움
    - is_success                      — 원 설계는 트리거. 동일 의미의 GENERATED
                                        STORED 컬럼으로 대체(원자적·트리거 불필요):
                                        BUY → return>0, SELL → return<0, NULL→NULL
  stats: symbol PK + 적중률 집계 (update_stats UPSERT 대상)
"""
from alembic import op


revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_stock_prediction_history (
            symbol            TEXT NOT NULL,
            grade_date        DATE NOT NULL,
            predicted_grade   TEXT,
            grade_direction   TEXT,
            actual_return_90d DOUBLE PRECISION,
            is_success        BOOLEAN GENERATED ALWAYS AS (
                CASE
                    WHEN actual_return_90d IS NULL THEN NULL
                    WHEN grade_direction = 'BUY'  THEN actual_return_90d > 0
                    WHEN grade_direction = 'SELL' THEN actual_return_90d < 0
                    ELSE NULL
                END
            ) STORED,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (symbol, grade_date)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_stock_prediction_history_grade_date
            ON us_stock_prediction_history (grade_date);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_stock_prediction_history_return_null
            ON us_stock_prediction_history (grade_date)
            WHERE actual_return_90d IS NULL;
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_stock_prediction_stats (
            symbol             TEXT PRIMARY KEY,
            total_signals      INTEGER,
            total_successes    INTEGER,
            hit_rate           DOUBLE PRECISION,
            buy_signals        INTEGER,
            buy_successes      INTEGER,
            buy_hit_rate       DOUBLE PRECISION,
            buy_avg_return_90d DOUBLE PRECISION,
            sell_signals       INTEGER,
            sell_successes     INTEGER,
            sell_hit_rate      DOUBLE PRECISION,
            sell_avg_return_90d DOUBLE PRECISION,
            latest_grade       TEXT,
            latest_grade_date  DATE,
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS us_stock_prediction_stats;")
    op.execute("DROP INDEX IF EXISTS idx_us_stock_prediction_history_return_null;")
    op.execute("DROP INDEX IF EXISTS idx_us_stock_prediction_history_grade_date;")
    op.execute("DROP TABLE IF EXISTS us_stock_prediction_history;")
