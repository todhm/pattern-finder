"""us_stock_grade — factor rank/percentile 컬럼 추가.

배경:
  alphafolio_quant run_option1 의 분석-후처리(analyze_all_stocks_specific_dates
  말미)가 날짜별로 us_stock_grade 에 상대순위 컬럼을 UPDATE 하는데, 이 컬럼들이
  공유 alphafolio DB 에 없어 매 날짜마다 비치명적 경고로 실패했다:
    [경고] RS Rank 계산 실패: column "rs_rank" ... does not exist
    [경고] Industry Rank 계산 실패: column "industry_rank" ... does not exist
    [경고] Factor Rankings 계산 실패: column "value_rank" ... does not exist
  grade 본체(final_grade/final_score)는 정상 저장되므로 backtest 에는 영향이
  없지만, 추천 페이지용 순위 표시 데이터가 비어 있었다. 이 마이그레이션이
  그 컬럼들을 채워 후처리가 정상 동작하게 한다.

타입은 quant 의 UPDATE 문에서 역도출:
  - *_rank (rs/value/quality/momentum/growth): 사람이 읽는 라벨 문자열
    (예: '매우강함 (Top 10%)', '공동 3위') → TEXT
  - industry_rank: RANK() 정수 → INTEGER
  - *_percentile: ROUND(PERCENT_RANK()*100, 1) → DOUBLE PRECISION
    (기존 sector_percentile / vol_percentile 와 동일 타입)

NOTE: 같은 후처리의 alternative-matcher(alt_symbol/alt_reasons 등)는 단순
누락이 아니라 asyncpg JSON 코덱 부재로 List[Dict] 인코딩 자체가 실패하는
별도 코드 이슈라 여기서 다루지 않는다.
"""
from alembic import op


revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None

_RANK_TEXT_COLS = [
    "rs_rank", "value_rank", "quality_rank", "momentum_rank", "growth_rank",
]
_PERCENTILE_COLS = [
    "industry_percentile", "value_percentile", "quality_percentile",
    "momentum_percentile", "growth_percentile",
]


def upgrade() -> None:
    for col in _RANK_TEXT_COLS:
        op.execute(
            f"ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS {col} TEXT;"
        )
    op.execute(
        "ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS industry_rank INTEGER;"
    )
    for col in _PERCENTILE_COLS:
        op.execute(
            f"ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION;"
        )


def downgrade() -> None:
    for col in _PERCENTILE_COLS:
        op.execute(f"ALTER TABLE us_stock_grade DROP COLUMN IF EXISTS {col};")
    op.execute("ALTER TABLE us_stock_grade DROP COLUMN IF EXISTS industry_rank;")
    for col in _RANK_TEXT_COLS:
        op.execute(f"ALTER TABLE us_stock_grade DROP COLUMN IF EXISTS {col};")
