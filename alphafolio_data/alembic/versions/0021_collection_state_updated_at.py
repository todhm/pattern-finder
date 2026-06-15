"""collection_state — updated_at 컬럼 추가.

배경:
  EODHD 수집기(us/eodhd.py)가 collection_state 에 ON CONFLICT ... DO UPDATE SET
  updated_at=NOW() 로 upsert 하는데, collection_state 에는 collected_at 만 있고
  updated_at 컬럼이 없어 매 종목 "column updated_at of relation collection_state
  does not exist" 에러가 났다(reco stock_basic 수집 실패 원인 중 하나). 컬럼만
  추가하면 해소(additive, 무해).
"""
from alembic import op


revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE collection_state "
        "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE collection_state DROP COLUMN IF EXISTS updated_at;")
