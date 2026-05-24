"""widen us_symbol VARCHAR(20) columns to TEXT

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-19

stock_listing_downloader pulls from datahub.io which includes some 20+
character symbols (ETF option series, special tickers). The VARCHAR(20)
constraint inherited from finnhub_symbol's temp table schema caused all
8,340 symbol upserts to fail in a single transaction.

Widening every potentially-affected column to TEXT. Existing data preserved.
"""
from alembic import op


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE us_symbol ALTER COLUMN symbol         TYPE TEXT;")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN display_symbol TYPE TEXT;")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN figi           TYPE TEXT;")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN description    TYPE TEXT;")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN mic            TYPE TEXT;")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN currency       TYPE TEXT;")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN type           TYPE TEXT;")


def downgrade() -> None:
    # Best-effort rollback — may fail if data > 20/255 chars exists by then.
    op.execute("ALTER TABLE us_symbol ALTER COLUMN symbol         TYPE VARCHAR(20);")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN display_symbol TYPE VARCHAR(20);")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN figi           TYPE VARCHAR(20);")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN description    TYPE VARCHAR(255);")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN mic            TYPE VARCHAR(10);")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN currency       TYPE VARCHAR(10);")
    op.execute("ALTER TABLE us_symbol ALTER COLUMN type           TYPE VARCHAR(50);")
