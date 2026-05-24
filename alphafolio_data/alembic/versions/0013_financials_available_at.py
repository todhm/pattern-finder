"""financials 3종 (Income/Balance/CashFlow) 에 available_at DATE 컬럼 추가.

목적:
  AV 의 분기 재무는 fiscal_date_ending 만 제공 — 실제 공시일자 모름.
  backtesting 시 look-ahead bias 방지를 위해 "데이터를 얻을 수 있는 시점"
  (= 시장에 reporting 되어 사용 가능해진 날짜) 을 별도 컬럼으로 저장.

추정 공식:
  available_at = fiscal_date_ending + 45 days
  (SEC 의 10-Q 제출 deadline 40-45일 기준 — Large Accelerated Filer 40일,
   Accelerated Filer 45일. 보수적으로 45일 채택. 향후 us_earnings_calendar
   의 actualReportDate 로 정밀화 가능.)

기존 row:
  모두 fiscal_date_ending + 45 days 로 자동 백필.

quant 쿼리 예시:
  SELECT ... FROM us_income_statement
  WHERE symbol = $1 AND available_at <= $analysis_date
  ORDER BY fiscal_date_ending DESC LIMIT 1
"""
from alembic import op


revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for tbl in ("us_income_statement", "us_balance_sheet", "us_cash_flow"):
        # 1. add column (nullable)
        op.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS available_at DATE;")

        # 2. backfill existing rows: fiscal_date_ending + 45 days
        op.execute(f"""
            UPDATE {tbl}
            SET available_at = fiscal_date_ending + INTERVAL '45 days'
            WHERE available_at IS NULL AND fiscal_date_ending IS NOT NULL;
        """)

        # 3. index for look-ahead-safe queries
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{tbl}_available_at
            ON {tbl} (symbol, available_at DESC);
        """)


def downgrade() -> None:
    for tbl in ("us_income_statement", "us_balance_sheet", "us_cash_flow"):
        op.execute(f"DROP INDEX IF EXISTS idx_{tbl}_available_at;")
        op.execute(f"ALTER TABLE {tbl} DROP COLUMN IF EXISTS available_at;")
