"""us_stock_grade — alternative-stock(대체종목) 컬럼 추가.

배경:
  alphafolio_quant 후처리의 USAlternativeStockMatcher 가 매도등급 종목마다
  같은 industry/sector 의 매수등급 대체종목을 찾아 us_stock_grade 에 기록하는데,
  이 컬럼들이 공유 alphafolio DB 에 없어 "column alt_symbol ... does not exist"
  로 실패했다. (추가로 alt_reasons 는 List[Dict] 를 asyncpg 에 그대로 넘겨
  JSON 코덱 부재로 인코딩이 실패하던 코드 버그도 함께 수정 —
  us/us_alternative_matcher.py 에서 json.dumps 로 직렬화하도록 변경.)

타입은 matcher 의 UPDATE 문에서 역도출:
  alt_symbol/alt_stock_name/alt_final_grade/alt_match_type → TEXT
  alt_final_score → DOUBLE PRECISION (final_score 와 동일)
  alt_reasons → JSONB (json.dumps 직렬화한 매칭 사유 배열)
"""
from alembic import op


revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None

_TEXT_COLS = ["alt_symbol", "alt_stock_name", "alt_final_grade", "alt_match_type"]


def upgrade() -> None:
    for col in _TEXT_COLS:
        op.execute(
            f"ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS {col} TEXT;"
        )
    op.execute(
        "ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS alt_final_score DOUBLE PRECISION;"
    )
    op.execute(
        "ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS alt_reasons JSONB;"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE us_stock_grade DROP COLUMN IF EXISTS alt_reasons;")
    op.execute("ALTER TABLE us_stock_grade DROP COLUMN IF EXISTS alt_final_score;")
    for col in _TEXT_COLS:
        op.execute(f"ALTER TABLE us_stock_grade DROP COLUMN IF EXISTS {col};")
