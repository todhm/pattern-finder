"""scenario_*_return columns are TEXT, not DOUBLE PRECISION

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-20

quant code stores scenario_*_return as human-readable strings like
'+10~20%', '-5~+5%', etc. — not numeric. Migration 0005 inferred them
as DOUBLE PRECISION based on the '_return' suffix, but they are TEXT
labels for the bullish/sideways/bearish scenario range.

Fix: ALTER both us_stock_grade and kr_stock_grade to TEXT for these
3 columns. Existing rows preserved (table is empty in practice).
"""
from alembic import op


revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for col in ("scenario_bullish_return",
                "scenario_sideways_return",
                "scenario_bearish_return"):
        op.execute(
            f"ALTER TABLE us_stock_grade ALTER COLUMN {col} TYPE TEXT "
            f"USING {col}::TEXT;")
        op.execute(
            f"ALTER TABLE kr_stock_grade ALTER COLUMN {col} TYPE TEXT "
            f"USING {col}::TEXT;")


def downgrade() -> None:
    for col in ("scenario_bullish_return",
                "scenario_sideways_return",
                "scenario_bearish_return"):
        # Best-effort: TEXT → DOUBLE PRECISION fails if non-numeric data exists.
        # Only safe to downgrade on empty tables.
        op.execute(
            f"ALTER TABLE us_stock_grade ALTER COLUMN {col} TYPE DOUBLE PRECISION "
            f"USING NULLIF({col}, '')::DOUBLE PRECISION;")
        op.execute(
            f"ALTER TABLE kr_stock_grade ALTER COLUMN {col} TYPE DOUBLE PRECISION "
            f"USING NULLIF({col}, '')::DOUBLE PRECISION;")
