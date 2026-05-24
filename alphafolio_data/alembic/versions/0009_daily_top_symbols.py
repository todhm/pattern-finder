"""daily_top_symbols — EM8 기반 일자별 종목 사전 필터 결과

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-21

EM8 (Long-term Price Momentum, IC +0.178) 을 SQL 일괄 계산으로 4592종목
전체에 적용한 후, 일자별 상위 N개 종목만 저장. quant의 grades_pass_a 가
이 테이블의 symbol 만 분석하면 약 10배 단축 (Python per-stock 루프와
수학적으로 동일함이 검증됨 — diff < 1e-15).

스키마:
  (date, symbol) → rank, em8_score
  PK = (date, symbol)
  Index = (date, rank) for fast top-N lookup per date
"""
from alembic import op


revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS daily_top_symbols (
            date         DATE NOT NULL,
            symbol       TEXT NOT NULL,
            rank         INT  NOT NULL,
            em8_score    DOUBLE PRECISION,
            computed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (date, symbol)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_top_symbols_date_rank
        ON daily_top_symbols (date, rank);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS daily_top_symbols;")
