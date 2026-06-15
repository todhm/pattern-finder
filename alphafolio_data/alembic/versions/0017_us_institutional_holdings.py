"""us_institutional_holdings — AV INSTITUTIONAL_HOLDINGS endpoint 의 종목별 기관 보유.

목적:
  NQ1 (Institutional Quality Score) 활성용. 13F filings 기반 기관 보유주식,
  분기별 증감, 보유 holder 수 등을 종목별 시계열로 적재.

AV 응답 구조:
  symbol → {total_institutional_holders, total_institutional_shares,
           holders_with_increased_holdings/decreased_holdings/unchanged,
           total_institutional_ownership_percentage,
           holdings: [{holder_name, shares_held, shares_changed,
                      shares_changed_percentage, change_type, last_reported}]}

각 (symbol, report_date) 마다 1 row.
  - summary: 그 분기의 aggregate (holders 수 / shares 합 / increased/decreased
    counts / 보유 % )
  - holdings 의 각 holder 는 holdings JSONB 컬럼에 array 로 저장 (cheaper than
    별도 row × 4-6k 종목 × 분기 × top-20 holder = ~수백만 rows)

AV 가 trailing 1 year (4 분기) 만 응답 — historical 5년+ 는 불가.
사용자 의도 "있으면 쓰는 정도" 에 맞춰 결손 시 NQ1 quant 가 NULL 처리.
"""
from alembic import op


revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_institutional_holdings (
            symbol                                TEXT NOT NULL,
            report_date                           DATE NOT NULL,
            total_institutional_holders           INTEGER,
            total_institutional_shares            BIGINT,
            holders_with_increased_holdings       INTEGER,
            shares_with_increased_holdings        BIGINT,
            holders_with_decreased_holdings       INTEGER,
            shares_with_decreased_holdings        BIGINT,
            holders_with_unchanged_holdings       INTEGER,
            shares_with_unchanged_holdings        BIGINT,
            total_institutional_ownership_pct     DOUBLE PRECISION,
            holdings_json                          JSONB,
            created_at                            TIMESTAMP DEFAULT NOW(),
            updated_at                            TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (symbol, report_date)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_institutional_holdings_report_date
        ON us_institutional_holdings (report_date DESC);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_us_institutional_holdings_report_date;")
    op.execute("DROP TABLE IF EXISTS us_institutional_holdings;")
