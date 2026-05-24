"""mv_us_sector/industry_daily_performance — daily sector/industry aggregates.

이름에 ``mv_`` 가 붙어 있지만 PostgreSQL materialized view 가 아닌
**일반 테이블** 이다. alphafolio_quant 의 us_db_async.refresh_sector_performance_for_date /
refresh_industry_performance_for_date 가 매 분석일자마다 INSERT 하여 채운다.
이전엔 테이블 자체가 없어서 매 date 마다 INSERT 실패 → 경고만 남고 sector/industry
context 정보가 grade 계산에 전혀 반영되지 않았다. 본 migration 으로 테이블 생성.

별도 orchestrator task ``us_mv_sector_refresh`` 가 ctx.start_date ~ ctx.end_date
범위를 한 번에 bulk INSERT 하므로 grades_pass_a 안에서 per-date refresh 가
ON CONFLICT DO NOTHING 으로 빠르게 skip 된다.
"""
from alembic import op


revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS mv_us_sector_daily_performance (
            date           DATE    NOT NULL,
            sector_code    TEXT    NOT NULL,
            avg_return_30d NUMERIC,
            stock_count    INTEGER,
            sector_rank    INTEGER,
            PRIMARY KEY (date, sector_code)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_mv_sector_date
        ON mv_us_sector_daily_performance(date);
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS mv_us_industry_daily_performance (
            date           DATE    NOT NULL,
            industry_code  TEXT    NOT NULL,
            avg_score      NUMERIC,
            stock_count    INTEGER,
            industry_rank  INTEGER,
            PRIMARY KEY (date, industry_code)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_mv_industry_date
        ON mv_us_industry_daily_performance(date);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_mv_industry_date;")
    op.execute("DROP TABLE IF EXISTS mv_us_industry_daily_performance;")
    op.execute("DROP INDEX IF EXISTS idx_mv_sector_date;")
    op.execute("DROP TABLE IF EXISTS mv_us_sector_daily_performance;")
