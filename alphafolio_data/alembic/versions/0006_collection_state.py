"""generic per-(collection, symbol, date) dedup state

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-20

Replaces the binary "has this symbol ever been seen" logic in legacy
collectors with date-granular tracking. Collectors should query this table
to figure out which DATES are missing for a symbol, then fetch only those.

Schema:
  collection_state(collection_name, symbol, date)
    PRIMARY KEY = (collection_name, symbol, date)

On first install, we also seed the table from existing data tables so
re-runs against a pre-populated DB skip what's already there:
  us_daily      → collection_state('us_daily', symbol, date)
  us_daily_etf  → collection_state('us_daily_etf', symbol, date)
  us_option     → collection_state('us_option', symbol, date) (distinct date per symbol)
  us_indicators → collection_state('us_indicators', symbol, date)
"""
from alembic import op


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS collection_state (
            collection_name TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            date            DATE NOT NULL,
            collected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (collection_name, symbol, date)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_collection_state_name_symbol
        ON collection_state (collection_name, symbol);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_collection_state_name_date
        ON collection_state (collection_name, date);
    """)

    # Seed from existing data tables (idempotent, no-op if data already there)
    op.execute("""
        INSERT INTO collection_state (collection_name, symbol, date)
        SELECT 'us_daily', symbol, date FROM us_daily
        ON CONFLICT DO NOTHING;
    """)
    op.execute("""
        INSERT INTO collection_state (collection_name, symbol, date)
        SELECT 'us_daily_etf', symbol, date FROM us_daily_etf
        ON CONFLICT DO NOTHING;
    """)
    op.execute("""
        INSERT INTO collection_state (collection_name, symbol, date)
        SELECT DISTINCT 'us_option', symbol, date FROM us_option
        ON CONFLICT DO NOTHING;
    """)
    op.execute("""
        INSERT INTO collection_state (collection_name, symbol, date)
        SELECT 'us_indicators', symbol, date FROM us_indicators
        ON CONFLICT DO NOTHING;
    """)
    op.execute("""
        INSERT INTO collection_state (collection_name, symbol, date)
        SELECT 'us_income_statement', symbol, fiscal_date_ending::date
        FROM us_income_statement
        ON CONFLICT DO NOTHING;
    """)
    op.execute("""
        INSERT INTO collection_state (collection_name, symbol, date)
        SELECT 'us_stock_basic', symbol, CURRENT_DATE
        FROM us_stock_basic
        ON CONFLICT DO NOTHING;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS collection_state;")
