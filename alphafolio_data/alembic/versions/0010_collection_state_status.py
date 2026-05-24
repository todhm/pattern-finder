"""add status column to collection_state — 'success' | 'no_data' | 'failed'

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-21

기존 collection_state는 "성공한 (symbol, date)" 만 기록했음.
status 추가로 "AV가 no_data 응답한 (symbol, date)" 도 기록 가능 → 다음 실행
때 재시도 방지. 4592 종목 중 약 100개가 매 실행마다 'No data found' 받던
것을 영구 제거.
"""
from alembic import op


revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE collection_state
        ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'success';
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_collection_state_status
        ON collection_state (collection_name, status, symbol);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_collection_state_status;")
    op.execute("ALTER TABLE collection_state DROP COLUMN IF EXISTS status;")
