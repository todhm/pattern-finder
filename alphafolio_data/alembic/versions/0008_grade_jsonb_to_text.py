"""relax JSONB columns to TEXT in {us,kr}_stock_grade

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-20

quant code sends these fields as JSON strings via json.dumps. Some
records contain non-JSON 'None', empty strings, or partial dumps that
fail strict JSONB validation. TEXT is permissive — downstream code
parses defensively with json.loads.

Uses DO block to only ALTER columns that exist (kr_stock_grade lacks
insider_signal, etc.).
"""
from alembic import op


revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


JSON_COLS = [
    "value_v2_detail", "quality_v2_detail",
    "momentum_v2_detail", "growth_v2_detail",
    "buy_triggers", "sell_triggers", "hold_triggers",
    "insider_signal",
]


def upgrade() -> None:
    for tbl in ("us_stock_grade", "kr_stock_grade"):
        for col in JSON_COLS:
            op.execute(f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{tbl}' AND column_name = '{col}'
                    ) THEN
                        ALTER TABLE {tbl}
                          ALTER COLUMN {col} TYPE TEXT USING {col}::TEXT;
                    END IF;
                END$$;
            """)


def downgrade() -> None:
    for tbl in ("us_stock_grade", "kr_stock_grade"):
        for col in JSON_COLS:
            op.execute(f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{tbl}' AND column_name = '{col}'
                    ) THEN
                        ALTER TABLE {tbl}
                          ALTER COLUMN {col} TYPE JSONB
                          USING NULLIF({col}, '')::JSONB;
                    END IF;
                END$$;
            """)
