"""us_listing_status — AV LISTING_STATUS endpoint 의 active+delisted 합본.

목적:
  Survivorship bias 제거. 백테스트 시점별 universe 를 정확히 구성하려면
  "그 시점에 listed 였던 종목" 이 필요 — us_stock_basic.is_active 는 현재
  상태만 보존하므로 (오늘 active=true 인 종목만) 약세장 결과를 인위적으로
  좋게 만든다 (망한 종목들이 자동 제외됨).

  AV LISTING_STATUS endpoint 는 한 번 호출에 다음을 줌:
    - state=active   : 13.8k 종목 (현재 상장)
    - state=delisted : 9.3k 종목 (1997-현재까지 상폐된 것)
  각 row 에 ipoDate + delistingDate 가 있어 시점별 PIT universe 가 가능.

PIT universe 조회 패턴:
  SELECT symbol FROM us_listing_status
  WHERE ipo_date IS NULL OR ipo_date <= $analysis_date
    AND (delisting_date IS NULL OR delisting_date > $analysis_date)
    AND asset_type = 'Stock'
"""
from alembic import op


revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_listing_status (
            symbol           TEXT NOT NULL,
            name             TEXT,
            exchange         TEXT,
            asset_type       TEXT,
            ipo_date         DATE,
            delisting_date   DATE,
            status           TEXT,
            created_at       TIMESTAMP DEFAULT NOW(),
            updated_at       TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (symbol)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_listing_status_dates
        ON us_listing_status (ipo_date, delisting_date);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_us_listing_status_status_type
        ON us_listing_status (status, asset_type);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_us_listing_status_status_type;")
    op.execute("DROP INDEX IF EXISTS idx_us_listing_status_dates;")
    op.execute("DROP TABLE IF EXISTS us_listing_status;")
