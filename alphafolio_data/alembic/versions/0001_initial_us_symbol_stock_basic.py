"""initial schema: us_symbol + us_stock_basic

Revision ID: 0001
Revises:
Create Date: 2026-05-18

Columns for us_stock_basic mirror AlphaVantageCollector.transform_data()
field_mappings in alphafolio_data/us/alphavantage.py:140-200.
Postgres folds unquoted identifiers to lowercase, so all columns here are
lowercase to match the INSERT statement at line 226 of that file.

Uses raw CREATE TABLE IF NOT EXISTS so it's safe to apply on databases
where these tables were previously created by ad-hoc scripts (no Alembic
tracking). Downgrade drops them unconditionally.
"""
from alembic import op


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_symbol (
            symbol TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_stock_basic (
            symbol                       TEXT PRIMARY KEY,
            assettype                    TEXT,
            stock_name                   TEXT,
            description                  TEXT,
            cik                          TEXT,
            exchange                     TEXT,
            currency                     TEXT,
            country                      TEXT,
            sector                       TEXT,
            industry                     TEXT,
            address                      TEXT,
            officialsite                 TEXT,
            fiscalyearend                TEXT,
            latestquarter                DATE,
            market_cap                   BIGINT,
            ebitda                       BIGINT,
            per                          DOUBLE PRECISION,
            peg                          DOUBLE PRECISION,
            bookvalue                    DOUBLE PRECISION,
            dividendpershare             DOUBLE PRECISION,
            dividendyield                DOUBLE PRECISION,
            eps                          DOUBLE PRECISION,
            revenuepersharettm           DOUBLE PRECISION,
            profitmargin                 DOUBLE PRECISION,
            operatingmarginttm           DOUBLE PRECISION,
            returnonassetsttm            DOUBLE PRECISION,
            returnonequityttm            DOUBLE PRECISION,
            revenuettm                   BIGINT,
            grossprofitttm               BIGINT,
            dilutedepsttm                DOUBLE PRECISION,
            quarterlyearningsgrowthyoy   DOUBLE PRECISION,
            quarterlyrevenuegrowthyoy    DOUBLE PRECISION,
            analysttargetprice           DOUBLE PRECISION,
            analystratingstrongbuy       INTEGER,
            analystratingbuy             INTEGER,
            analystratinghold            INTEGER,
            analystratingsell            INTEGER,
            analystratingstrongsell      INTEGER,
            trailingpe                   DOUBLE PRECISION,
            forwardpe                    DOUBLE PRECISION,
            pricetosalesratiottm         DOUBLE PRECISION,
            pricetobookratio             DOUBLE PRECISION,
            evtorevenue                  DOUBLE PRECISION,
            evtoebitda                   DOUBLE PRECISION,
            beta                         DOUBLE PRECISION,
            week52high                   DOUBLE PRECISION,
            week52low                    DOUBLE PRECISION,
            day50movingaverage           DOUBLE PRECISION,
            day200movingaverage          DOUBLE PRECISION,
            sharesoutstanding            BIGINT,
            sharesfloat                  BIGINT,
            percentinsiders              DOUBLE PRECISION,
            percentinstitutions          DOUBLE PRECISION,
            dividenddate                 DATE,
            exdividenddate               DATE,
            is_active                    BOOLEAN NOT NULL DEFAULT TRUE,
            created_at                   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at                   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_basic_sector    ON us_stock_basic (sector);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_basic_industry  ON us_stock_basic (industry);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_basic_is_active ON us_stock_basic (is_active);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_basic_exchange  ON us_stock_basic (exchange);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS us_stock_basic;")
    op.execute("DROP TABLE IF EXISTS us_symbol;")
