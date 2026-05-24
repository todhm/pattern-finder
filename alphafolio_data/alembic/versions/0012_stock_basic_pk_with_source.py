"""us_stock_basic PK = (symbol, date, source) — api + computed 동시 저장 허용

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-21

목적: 같은 (symbol, date) 에 대해 api 응답값과 시점별 compute값을 두 row 로
나란히 저장. 0011 의 (symbol, date) PK 는 이걸 막아 PK 를 확장.

PK 변경:
  (symbol, date)  →  (symbol, date, source)
"""
from alembic import op


revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Drop 0011 PK (symbol, date)
    op.execute("ALTER TABLE us_stock_basic DROP CONSTRAINT us_stock_basic_pkey;")

    # 2. Add new PK (symbol, date, source) — api row + computed row 공존 가능
    op.execute("ALTER TABLE us_stock_basic ADD PRIMARY KEY (symbol, date, source);")

    # 3. Index on (date, symbol) for fast date-range queries (백테스트용)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_stock_basic_date_symbol
        ON us_stock_basic (date, symbol);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_us_stock_basic_date_symbol;")
    op.execute("ALTER TABLE us_stock_basic DROP CONSTRAINT us_stock_basic_pkey;")
    # 이전 PK (symbol, date) 로 복구 — source 컬럼 중복 row 가 있으면 첫 번째만 유지
    op.execute("""
        DELETE FROM us_stock_basic a USING us_stock_basic b
        WHERE a.symbol = b.symbol AND a.date = b.date
          AND a.ctid > b.ctid;
    """)
    op.execute("ALTER TABLE us_stock_basic ADD PRIMARY KEY (symbol, date);")
