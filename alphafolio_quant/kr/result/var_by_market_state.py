"""
VaR Exceedance Analysis by Market State
- Calculates VaR 95% exceedance rate by market state
- Excludes problematic groups: KOSDAQ small caps, theme stocks, hot sector high risk
"""

import os
import asyncio
import asyncpg
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Excluded market states
EXCLUDED_MARKET_STATES = [
    '테마특화-모멘텀폭발형',
    'KOSDAQ소형-핫섹터-초고위험형'
]

# KOSDAQ small cap patterns (to exclude)
KOSDAQ_SMALL_PATTERNS = ['KOSDAQ소형']


async def fetch_data():
    """Fetch data from kr_stock_grade table with price data"""
    database_url = os.getenv('DATABASE_URL')
    db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)

    try:
        # Step 1: Fetch grade data
        grade_query = """
        SELECT
            symbol,
            date,
            market_state,
            var_95,
            final_score,
            final_grade
        FROM kr_stock_grade
        WHERE date >= '2025-08-01' AND date <= '2026-01-13'
          AND var_95 IS NOT NULL
          AND market_state IS NOT NULL
        ORDER BY symbol, date
        """

        grade_rows = await conn.fetch(grade_query)
        print(f"Fetched {len(grade_rows)} grade records")

        # Step 2: Fetch all price data
        price_query = """
        SELECT symbol, date, close
        FROM kr_intraday_total
        WHERE date >= '2025-08-01' AND date <= '2026-05-01'
        ORDER BY symbol, date
        """

        price_rows = await conn.fetch(price_query)
        print(f"Fetched {len(price_rows)} price records")

        return grade_rows, price_rows

    finally:
        await conn.close()


def calculate_returns(grade_rows, price_rows):
    """Calculate actual returns from price data"""
    from datetime import timedelta

    # Build price lookup: {(symbol, date): close}
    print("Building price lookup...")
    price_lookup = {}
    for row in price_rows:
        key = (row['symbol'], row['date'])
        price_lookup[key] = float(row['close']) if row['close'] else None

    print(f"Price lookup built with {len(price_lookup)} entries")

    data = []
    periods = [3, 30, 60, 90]

    for i, row in enumerate(grade_rows):
        if i % 50000 == 0:
            print(f"Processing {i}/{len(grade_rows)} records...")

        symbol = row['symbol']
        analysis_date = row['date']

        # Get base price
        base_price = price_lookup.get((symbol, analysis_date))
        if base_price is None or base_price <= 0:
            continue

        record = {
            'symbol': symbol,
            'date': analysis_date,
            'market_state': row['market_state'],
            'var_95': float(row['var_95']) if row['var_95'] else None,
            'final_score': float(row['final_score']) if row['final_score'] else None,
            'final_grade': row['final_grade'],
        }

        # Calculate returns for each period
        for period in periods:
            target_date = analysis_date + timedelta(days=period)
            future_price = None

            # Look for price within +/- 5 days
            for offset in range(0, 6):
                check_date = target_date + timedelta(days=offset)
                future_price = price_lookup.get((symbol, check_date))
                if future_price is not None:
                    break

                if offset > 0:
                    check_date = target_date - timedelta(days=offset)
                    future_price = price_lookup.get((symbol, check_date))
                    if future_price is not None:
                        break

            if future_price and future_price > 0:
                record[f'return_{period}d'] = ((future_price - base_price) / base_price) * 100
            else:
                record[f'return_{period}d'] = None

        data.append(record)

    return pd.DataFrame(data)


def is_excluded(market_state):
    """Check if market state should be excluded"""
    if market_state is None:
        return True

    # Exact match exclusions
    if market_state in EXCLUDED_MARKET_STATES:
        return True

    # Pattern match exclusions (KOSDAQ small caps)
    for pattern in KOSDAQ_SMALL_PATTERNS:
        if market_state.startswith(pattern):
            return True

    return False


def calculate_var_exceedance(df, period_days):
    """Calculate VaR exceedance rate for a given period"""
    return_col = f'return_{period_days}d'

    # Filter valid data
    valid_df = df[['market_state', 'var_95', return_col]].dropna()

    if len(valid_df) == 0:
        return None

    # VaR exceedance: actual return < VaR (loss exceeded prediction)
    valid_df['exceeded'] = valid_df[return_col] < valid_df['var_95']

    exceedance_rate = valid_df['exceeded'].mean()
    sample_count = len(valid_df)

    return {
        'period_days': period_days,
        'exceedance_rate': exceedance_rate,
        'expected_rate': 0.05,
        'sample_count': sample_count,
        'exceeded_count': valid_df['exceeded'].sum()
    }


def analyze_by_market_state(df, period_days):
    """Analyze VaR exceedance by market state"""
    return_col = f'return_{period_days}d'

    results = []

    for market_state in df['market_state'].unique():
        if market_state is None:
            continue

        state_df = df[df['market_state'] == market_state]
        valid_df = state_df[['var_95', return_col]].dropna()

        if len(valid_df) < 10:  # Minimum sample size
            continue

        valid_df['exceeded'] = valid_df[return_col] < valid_df['var_95']

        results.append({
            'market_state': market_state,
            'period_days': period_days,
            'exceedance_rate': valid_df['exceeded'].mean(),
            'sample_count': len(valid_df),
            'exceeded_count': valid_df['exceeded'].sum(),
            'is_excluded': is_excluded(market_state)
        })

    return pd.DataFrame(results)


async def main():
    print("=" * 70)
    print("VaR Exceedance Analysis by Market State")
    print("=" * 70)
    print()

    # Fetch data
    print("Fetching data from database...")
    grade_rows, price_rows = await fetch_data()

    # Calculate returns
    print("Calculating returns...")
    df = calculate_returns(grade_rows, price_rows)
    print(f"Total records with valid data: {len(df)}")
    print()

    # Separate included and excluded data
    df['is_excluded'] = df['market_state'].apply(is_excluded)
    df_included = df[~df['is_excluded']]
    df_excluded = df[df['is_excluded']]

    print(f"Included records: {len(df_included)}")
    print(f"Excluded records: {len(df_excluded)}")
    print()

    # Results storage
    all_results = []

    print("=" * 70)
    print("COMPARISON: ALL vs EXCLUDED GROUPS REMOVED")
    print("=" * 70)
    print()

    for period in [3, 30, 60, 90]:
        print(f"--- {period}-Day VaR Exceedance ---")

        # All data
        all_result = calculate_var_exceedance(df, period)

        # Excluded groups removed
        included_result = calculate_var_exceedance(df_included, period)

        # Excluded groups only
        excluded_result = calculate_var_exceedance(df_excluded, period)

        if all_result and included_result:
            improvement = all_result['exceedance_rate'] - included_result['exceedance_rate']

            print(f"  ALL Data:       {all_result['exceedance_rate']*100:6.2f}% (n={all_result['sample_count']:,})")
            print(f"  EXCLUDED Only:  {excluded_result['exceedance_rate']*100:6.2f}% (n={excluded_result['sample_count']:,})" if excluded_result else "  EXCLUDED Only:  N/A")
            print(f"  FILTERED:       {included_result['exceedance_rate']*100:6.2f}% (n={included_result['sample_count']:,})")
            print(f"  Improvement:    {improvement*100:+6.2f}%p")
            print(f"  Target (5%):    {'PASS' if included_result['exceedance_rate'] <= 0.05 else 'FAIL'}")
            print()

            all_results.append({
                'period_days': period,
                'group': 'ALL',
                'exceedance_rate': all_result['exceedance_rate'],
                'sample_count': all_result['sample_count']
            })
            all_results.append({
                'period_days': period,
                'group': 'EXCLUDED_ONLY',
                'exceedance_rate': excluded_result['exceedance_rate'] if excluded_result else None,
                'sample_count': excluded_result['sample_count'] if excluded_result else 0
            })
            all_results.append({
                'period_days': period,
                'group': 'FILTERED',
                'exceedance_rate': included_result['exceedance_rate'],
                'sample_count': included_result['sample_count']
            })

    # Detailed market state analysis
    print("=" * 70)
    print("DETAILED MARKET STATE ANALYSIS (90-Day)")
    print("=" * 70)
    print()

    state_results = analyze_by_market_state(df, 90)
    state_results = state_results.sort_values('exceedance_rate', ascending=False)

    print("Market State                              | Exceedance |  Sample | Status")
    print("-" * 70)

    for _, row in state_results.iterrows():
        status = "EXCLUDED" if row['is_excluded'] else "INCLUDED"
        print(f"{row['market_state']:<41} | {row['exceedance_rate']*100:6.2f}%    | {row['sample_count']:6,} | {status}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(r'C:\project\alpha\quant\kr\result\var_exceedance_comparison.csv', index=False, encoding='utf-8-sig')

    state_results.to_csv(r'C:\project\alpha\quant\kr\result\var_by_market_state_detail.csv', index=False, encoding='utf-8-sig')

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Excluded groups:")
    for state in EXCLUDED_MARKET_STATES:
        print(f"  - {state}")
    print(f"  - {KOSDAQ_SMALL_PATTERNS[0]}* (all KOSDAQ small cap states)")
    print()
    print("Results saved to:")
    print("  - var_exceedance_comparison.csv")
    print("  - var_by_market_state_detail.csv")


if __name__ == '__main__':
    asyncio.run(main())
