"""us_stock_basic: add date + source columns, change PK to (symbol, date)

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-21

목적:
- us_stock_basic을 시계열 테이블로 전환 — 매일 daily snapshot 누적
- 'source' 컬럼으로 API 원본 vs compute 도출값 구분
  - source='api'      : AV OVERVIEW 응답 그대로 (오늘 호출분)
  - source='computed' : us_daily + us_income_statement 등에서 시점별 계산

PK 변경:
  (symbol)  →  (symbol, date)

기존 6,195개 row는 date=CURRENT_DATE, source='api' 로 마이그레이션됨.

향후 quant 쿼리는 시점별로:
  SELECT ... FROM us_stock_basic
  WHERE symbol = $1 AND date <= $analysis_date
  ORDER BY date DESC LIMIT 1
"""
from alembic import op


revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add new columns (nullable first to populate existing rows)
    op.execute("ALTER TABLE us_stock_basic ADD COLUMN IF NOT EXISTS date DATE;")
    op.execute("ALTER TABLE us_stock_basic ADD COLUMN IF NOT EXISTS source TEXT;")

    # 2. Populate existing rows: 기존 데이터는 모두 오늘 API 응답으로 간주
    op.execute("UPDATE us_stock_basic SET date = CURRENT_DATE WHERE date IS NULL;")
    op.execute("UPDATE us_stock_basic SET source = 'api' WHERE source IS NULL;")

    # 3. Add NOT NULL + DEFAULT (after population)
    op.execute("ALTER TABLE us_stock_basic ALTER COLUMN date   SET NOT NULL;")
    op.execute("ALTER TABLE us_stock_basic ALTER COLUMN source SET NOT NULL;")
    op.execute("ALTER TABLE us_stock_basic ALTER COLUMN date   SET DEFAULT CURRENT_DATE;")
    op.execute("ALTER TABLE us_stock_basic ALTER COLUMN source SET DEFAULT 'api';")

    # 4. CHECK constraint on source values
    op.execute("""
        ALTER TABLE us_stock_basic
        ADD CONSTRAINT us_stock_basic_source_check
        CHECK (source IN ('api', 'computed'));
    """)

    # 5. Drop old PK (symbol only) and create new PK (symbol, date)
    op.execute("ALTER TABLE us_stock_basic DROP CONSTRAINT us_stock_basic_pkey;")
    op.execute("ALTER TABLE us_stock_basic ADD PRIMARY KEY (symbol, date);")

    # 6. Helper indexes for common queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_stock_basic_symbol_date_desc
        ON us_stock_basic (symbol, date DESC);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_stock_basic_date
        ON us_stock_basic (date);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_stock_basic_source
        ON us_stock_basic (source);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_us_stock_basic_source;")
    op.execute("DROP INDEX IF EXISTS idx_us_stock_basic_date;")
    op.execute("DROP INDEX IF EXISTS idx_us_stock_basic_symbol_date_desc;")

    # Revert PK to (symbol) — only safe if each symbol has unique date
    op.execute("ALTER TABLE us_stock_basic DROP CONSTRAINT us_stock_basic_pkey;")
    # If multiple rows per symbol exist, keep only the latest date row
    op.execute("""
        DELETE FROM us_stock_basic a USING us_stock_basic b
        WHERE a.symbol = b.symbol AND a.date < b.date;
    """)
    op.execute("ALTER TABLE us_stock_basic ADD PRIMARY KEY (symbol);")

    op.execute("ALTER TABLE us_stock_basic DROP CONSTRAINT IF EXISTS us_stock_basic_source_check;")
    op.execute("ALTER TABLE us_stock_basic DROP COLUMN IF EXISTS source;")
    op.execute("ALTER TABLE us_stock_basic DROP COLUMN IF EXISTS date;")
