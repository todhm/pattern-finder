"""EODHDFundamentalsCollector pilot — 4 종목 1 콜씩 fetch + DB 적재 검증.

사용법:
    docker compose exec alphafolio_data python -m scripts.test_eodhd_pilot
    docker compose exec alphafolio_data python -m scripts.test_eodhd_pilot AAPL MYRG

기대 결과:
  - us_stock_basic source='api' date=today × 4 종목 적재
  - us_income_statement / us_balance_sheet / us_cash_flow 분기 시계열 적재
    (AAPL ~163 quarters since 1985 / MYRG ~139 / ASYS ~158 / TER ~163)
  - quant 가 NULL 처리하던 컬럼들 (forwardpe, peg, analysttargetprice,
    week52high/low, day50/200ma, beta, dividendyield, sharesfloat,
    percentinsiders/institutions) 모두 채워짐
"""
import asyncio
import logging
import os
import sys

import asyncpg

from us.eodhd import EODHDFundamentalsCollector


DEFAULT_SYMBOLS = ["AAPL", "MYRG", "ASYS", "TER"]


async def run(symbols):
    api_token = os.getenv("EODHD_API_TOKEN") or os.getenv("EODHD_API_KEY")
    database_url = os.environ["DATABASE_URL"]
    if not api_token:
        print("EODHD_API_TOKEN / EODHD_API_KEY not set")
        sys.exit(1)

    print(f"=== EODHD pilot: {symbols} ===")
    col = EODHDFundamentalsCollector(
        api_token, database_url,
        call_interval=0.1, max_concurrent=2,
        target_symbols=symbols,
    )
    await col.init_pool()
    try:
        result = await col.run_collection()
        print("collector result:", result)
    finally:
        await col.close_pool()

    # 검증 SQL
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            print("\n=== Q1. us_stock_basic source='api' today × 4 종목 ===")
            rows = await conn.fetch(
                """SELECT symbol, date, source, market_cap, per, peg,
                          forwardpe, percentinsiders, percentinstitutions,
                          analysttargetprice, week52high, day200movingaverage
                   FROM us_stock_basic
                   WHERE symbol = ANY($1::text[])
                     AND date = CURRENT_DATE AND source='api'
                   ORDER BY symbol""", symbols)
            for r in rows:
                print(f"  {dict(r)}")

            print("\n=== Q2. NULL 컬럼 카운트 (forwardpe/peg/atp/52wH/inst%) ===")
            row = await conn.fetchrow(
                """SELECT
                     COUNT(*) FILTER (WHERE forwardpe IS NULL)             AS null_forwardpe,
                     COUNT(*) FILTER (WHERE peg IS NULL)                   AS null_peg,
                     COUNT(*) FILTER (WHERE analysttargetprice IS NULL)    AS null_atp,
                     COUNT(*) FILTER (WHERE week52high IS NULL)            AS null_52wh,
                     COUNT(*) FILTER (WHERE percentinstitutions IS NULL)   AS null_inst,
                     COUNT(*) FILTER (WHERE day50movingaverage IS NULL)    AS null_ma50,
                     COUNT(*) FILTER (WHERE day200movingaverage IS NULL)   AS null_ma200,
                     COUNT(*) FILTER (WHERE beta IS NULL)                  AS null_beta
                   FROM us_stock_basic
                   WHERE symbol = ANY($1::text[]) AND date=CURRENT_DATE AND source='api'""",
                symbols)
            print(f"  {dict(row)}")

            print("\n=== Q3. 분기 시계열 row count ===")
            for tbl in ("us_income_statement", "us_balance_sheet", "us_cash_flow"):
                rows = await conn.fetch(
                    f"SELECT symbol, COUNT(*) AS quarters FROM {tbl} "
                    f"WHERE symbol = ANY($1::text[]) GROUP BY symbol ORDER BY symbol",
                    symbols)
                counts = {r["symbol"]: r["quarters"] for r in rows}
                print(f"  {tbl}: {counts}")

            print("\n=== Q4. available_at 정밀도 (AAPL 최신 6분기) ===")
            rows = await conn.fetch(
                """SELECT fiscal_date_ending, available_at,
                          available_at - fiscal_date_ending AS lag_days
                   FROM us_income_statement
                   WHERE symbol='AAPL'
                   ORDER BY fiscal_date_ending DESC LIMIT 6""")
            for r in rows:
                print(f"  {dict(r)}")
    finally:
        await pool.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    syms = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SYMBOLS
    asyncio.run(run(syms))


if __name__ == "__main__":
    main()
