"""convert us_option to PARTITION BY RANGE (date)

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-19

USOptionCollector.ensure_daily_partition_exists() (us/us_option.py:408)
runs at collection time and tries:

    CREATE TABLE us_option_YYYY_MM_DD PARTITION OF us_option
    FOR VALUES FROM ('<date>') TO ('<next_date>')

That requires us_option to be declared `PARTITION BY RANGE (date)`.
Migration 0002 created it as a plain table; this migration drops and
recreates it as range-partitioned. Safe because us_option had no rows
at the time of this writing — verified via `SELECT COUNT(*) FROM us_option`.
If running this against a populated DB, dump+restore the table contents
manually first.
"""
from alembic import op


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the non-partitioned us_option (including indexes).
    op.execute("DROP TABLE IF EXISTS us_option CASCADE;")

    # Recreate as range-partitioned by date. Columns and types match the
    # CREATE TEMP TABLE temp_us_option block at us/us_option.py:305 and
    # the production schema observed via `\d us_option` post-0002.
    op.execute("""
        CREATE TABLE us_option (
            contract_id         VARCHAR(30) NOT NULL,
            symbol              VARCHAR(10),
            expiration          DATE,
            strike              NUMERIC(10, 2),
            type                VARCHAR(10),
            last                NUMERIC(10, 2),
            mark                NUMERIC(10, 2),
            bid                 NUMERIC(10, 2),
            bid_size            INTEGER,
            ask                 NUMERIC(10, 2),
            ask_size            INTEGER,
            volume              INTEGER,
            open_interest       INTEGER,
            date                DATE NOT NULL,
            implied_volatility  NUMERIC(10, 5),
            delta               NUMERIC(10, 5),
            gamma               NUMERIC(10, 5),
            theta               NUMERIC(10, 5),
            vega                NUMERIC(10, 5),
            rho                 NUMERIC(10, 5),
            PRIMARY KEY (contract_id, date)
        ) PARTITION BY RANGE (date);
    """)

    # Indexes propagate to partitions in PG12+. The collector also adds
    # per-partition unique + symbol indexes after creating each daily
    # partition, but having parent-level indexes helps planning.
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_date       ON us_option (date);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_symbol     ON us_option (symbol);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_expiration ON us_option (expiration);")


def downgrade() -> None:
    # Revert to non-partitioned form (matches 0002 schema).
    op.execute("DROP TABLE IF EXISTS us_option CASCADE;")
    op.execute("""
        CREATE TABLE us_option (
            contract_id         VARCHAR(30) NOT NULL,
            symbol              VARCHAR(10),
            expiration          DATE,
            strike              NUMERIC(10, 2),
            type                VARCHAR(10),
            last                NUMERIC(10, 2),
            mark                NUMERIC(10, 2),
            bid                 NUMERIC(10, 2),
            bid_size            INTEGER,
            ask                 NUMERIC(10, 2),
            ask_size            INTEGER,
            volume              INTEGER,
            open_interest       INTEGER,
            date                DATE NOT NULL,
            implied_volatility  NUMERIC(10, 5),
            delta               NUMERIC(10, 5),
            gamma               NUMERIC(10, 5),
            theta               NUMERIC(10, 5),
            vega                NUMERIC(10, 5),
            rho                 NUMERIC(10, 5),
            PRIMARY KEY (contract_id, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_date       ON us_option (date);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_symbol     ON us_option (symbol);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_expiration ON us_option (expiration);")
