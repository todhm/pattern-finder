"""us_earnings_history 테이블 — AV EARNINGS endpoint 의 quarterlyEarnings 적재.

목적:
  EARNINGS endpoint 가 제공하는 정확한 reportedDate (실제 공시일) 를 저장.
  income/balance/cashflow 는 모두 같은 10-Q/10-K 에 묶여 공시되므로
  available_at = us_earnings_history.reported_date 로 정밀 채울 수 있음.

스키마:
  PK: (symbol, fiscal_date_ending)
  reported_date  : 실제 공시일자 (AV.quarterlyEarnings[].reportedDate)
  reported_eps   : 발표 EPS
  estimated_eps  : 추정 EPS
  surprise       : 발표 - 추정
  surprise_percentage
  report_time    : 'pre-market' | 'post-market'

quant 쿼리 예시:
  SELECT i.* , COALESCE(e.reported_date,
                        i.fiscal_date_ending + INTERVAL '45 days') AS available_at
  FROM us_income_statement i
  LEFT JOIN us_earnings_history e USING (symbol, fiscal_date_ending)
  WHERE i.symbol = $1 AND COALESCE(...) <= $analysis_date
"""
from alembic import op


revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_earnings_history (
            symbol               TEXT        NOT NULL,
            fiscal_date_ending   DATE        NOT NULL,
            reported_date        DATE,
            reported_eps         DOUBLE PRECISION,
            estimated_eps        DOUBLE PRECISION,
            surprise             DOUBLE PRECISION,
            surprise_percentage  DOUBLE PRECISION,
            report_time          TEXT,
            created_at           TIMESTAMP   DEFAULT NOW(),
            updated_at           TIMESTAMP   DEFAULT NOW(),
            PRIMARY KEY (symbol, fiscal_date_ending)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_earnings_history_reported_date
        ON us_earnings_history (symbol, reported_date DESC);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_earnings_history_fiscal
        ON us_earnings_history (fiscal_date_ending);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_us_earnings_history_fiscal;")
    op.execute("DROP INDEX IF EXISTS idx_us_earnings_history_reported_date;")
    op.execute("DROP TABLE IF EXISTS us_earnings_history;")
