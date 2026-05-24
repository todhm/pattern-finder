# -*- coding: utf-8 -*-
"""
US Quant Phase 2 - Deep Dive Analysis
=====================================

Run all 3 analyses in one execution:
- STEP 1: Momentum Factor Deep Dive (EM1~EM7)
- STEP 2: Healthcare Sector Deep Dive
- STEP 3: Exchange Deep Dive (NYSE vs NASDAQ)

Execute: python us/analysis/us_phase2_deep_dive.py
Output: C:/project/alpha/quant/us/result/ folder (CSV files)
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, date
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Database connection
DATABASE_URL = 'postgresql://postgres:KoHtrdVEltzXlLVcgYtnRbEBQIRfEhMv@switchback.proxy.rlwy.net:28289/railway'

# Output directory
OUTPUT_DIR = r'C:\project\alpha\quant\us\result'

# Return periods to analyze
RETURN_PERIODS = [3, 5, 10, 20, 30, 60, 90, 180, 252]

# Momentum strategies
MOMENTUM_STRATEGIES = ['EM1', 'EM2', 'EM3', 'EM4', 'EM5', 'EM6', 'EM7']

# Value strategies (from us_value_factor_v2.py)
VALUE_STRATEGIES = ['RV1', 'RV2', 'RV3', 'RV4', 'RV5', 'RV6']

# All factors
ALL_FACTORS = ['value_score', 'quality_score', 'momentum_score', 'growth_score', 'final_score']


async def connect_db():
    """Connect to PostgreSQL database"""
    return await asyncpg.connect(DATABASE_URL)


def calculate_ic(df, score_col, return_col):
    """Calculate Pearson and Spearman IC"""
    valid = df[[score_col, return_col]].dropna()

    if len(valid) < 30:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': len(valid)}

    try:
        pearson_ic, _ = stats.pearsonr(valid[score_col], valid[return_col])
        spearman_ic, _ = stats.spearmanr(valid[score_col], valid[return_col])
    except:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': len(valid)}

    return {
        'pearson': pearson_ic,
        'spearman': spearman_ic,
        'n': len(valid)
    }


def calculate_directional_accuracy(decile, returns):
    """
    Calculate directional accuracy based on decile

    Args:
        decile: 1-10 (D1-D3: sell signal, D8-D10: buy signal)
        returns: array/series of returns

    Returns:
        accuracy: correct prediction ratio (%)

    Logic:
        - D1-D3 (Sell signal): Negative return is CORRECT
        - D8-D10 (Buy signal): Positive return is CORRECT
        - D4-D7 (Neutral): 50% baseline
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 50.0

    if decile <= 3:  # D1-D3: Sell signal - negative return is correct
        return (returns < 0).mean() * 100
    elif decile >= 8:  # D8-D10: Buy signal - positive return is correct
        return (returns > 0).mean() * 100
    else:  # D4-D7: Neutral
        return 50.0


def get_signal_type(decile):
    """Get signal type based on decile"""
    if decile <= 3:
        return 'SELL'
    elif decile >= 8:
        return 'BUY'
    else:
        return 'HOLD'


async def load_base_data(conn):
    """
    Load base data from us_stock_grade with returns from us_daily
    """
    print("\n" + "="*100)
    print("Loading base data from database...")
    print("="*100)

    query = """
    WITH grade_data AS (
        SELECT
            g.symbol,
            g.date,
            g.stock_name,
            g.final_score,
            g.final_grade,
            g.value_score,
            g.quality_score,
            g.momentum_score,
            g.growth_score,
            g.momentum_v2_detail,
            g.value_v2_detail,
            g.market_state,
            b.sector,
            b.industry,
            b.exchange,
            b.market_cap
        FROM us_stock_grade g
        LEFT JOIN us_stock_basic b ON g.symbol = b.symbol
    ),
    future_returns AS (
        SELECT
            d1.symbol,
            d1.date as base_date,
            d2.date as future_date,
            (d2.close - d1.close) / NULLIF(d1.close, 0) * 100 as return_pct,
            (d2.date - d1.date) as days_diff
        FROM us_daily d1
        JOIN us_daily d2 ON d1.symbol = d2.symbol
        WHERE d2.date > d1.date
            AND (d2.date - d1.date) BETWEEN 1 AND 260
    )
    SELECT
        g.symbol,
        g.date,
        g.stock_name,
        g.final_score,
        g.final_grade,
        g.value_score,
        g.quality_score,
        g.momentum_score,
        g.growth_score,
        g.momentum_v2_detail,
        g.value_v2_detail,
        g.market_state,
        g.sector,
        g.industry,
        g.exchange,
        g.market_cap,
        MAX(CASE WHEN fr.days_diff BETWEEN 2 AND 4 THEN fr.return_pct END) as return_3d,
        MAX(CASE WHEN fr.days_diff BETWEEN 4 AND 6 THEN fr.return_pct END) as return_5d,
        MAX(CASE WHEN fr.days_diff BETWEEN 8 AND 12 THEN fr.return_pct END) as return_10d,
        MAX(CASE WHEN fr.days_diff BETWEEN 18 AND 22 THEN fr.return_pct END) as return_20d,
        MAX(CASE WHEN fr.days_diff BETWEEN 28 AND 32 THEN fr.return_pct END) as return_30d,
        MAX(CASE WHEN fr.days_diff BETWEEN 55 AND 65 THEN fr.return_pct END) as return_60d,
        MAX(CASE WHEN fr.days_diff BETWEEN 85 AND 95 THEN fr.return_pct END) as return_90d,
        MAX(CASE WHEN fr.days_diff BETWEEN 175 AND 185 THEN fr.return_pct END) as return_180d,
        MAX(CASE WHEN fr.days_diff BETWEEN 248 AND 260 THEN fr.return_pct END) as return_252d
    FROM grade_data g
    LEFT JOIN future_returns fr ON g.symbol = fr.symbol AND g.date = fr.base_date
    WHERE g.sector IS NOT NULL AND g.sector != ''
    GROUP BY g.symbol, g.date, g.stock_name, g.final_score, g.final_grade,
             g.value_score, g.quality_score, g.momentum_score, g.growth_score,
             g.momentum_v2_detail, g.value_v2_detail, g.market_state, g.sector, g.industry,
             g.exchange, g.market_cap
    """

    rows = await conn.fetch(query)
    df = pd.DataFrame([dict(row) for row in rows])

    # Convert Decimal columns to float
    numeric_cols = ['final_score', 'value_score', 'quality_score', 'momentum_score',
                    'growth_score', 'market_cap', 'return_3d', 'return_5d',
                    'return_10d', 'return_20d', 'return_30d', 'return_60d',
                    'return_90d', 'return_180d', 'return_252d']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")

    # Parse momentum_v2_detail JSON
    print("\nParsing momentum strategy details...")
    for strategy in MOMENTUM_STRATEGIES:
        df[strategy] = df['momentum_v2_detail'].apply(
            lambda x: extract_strategy_score(x, strategy) if x else np.nan
        )
    print(f"Parsed {len(MOMENTUM_STRATEGIES)} momentum strategies")

    # Parse value_v2_detail JSON
    print("Parsing value strategy details...")
    for strategy in VALUE_STRATEGIES:
        df[strategy] = df['value_v2_detail'].apply(
            lambda x: extract_strategy_score(x, strategy) if x else np.nan
        )
    print(f"Parsed {len(VALUE_STRATEGIES)} value strategies")

    return df


def extract_strategy_score(detail, strategy):
    """Extract individual strategy score from JSONB detail"""
    try:
        if isinstance(detail, str):
            detail = json.loads(detail)
        if detail and strategy in detail:
            return detail[strategy].get('score', np.nan)
    except:
        pass
    return np.nan


# ============================================================================
# STEP 1: Momentum Factor Deep Dive
# ============================================================================

async def step1_momentum_deep_dive(df):
    """
    STEP 1: Momentum Factor Deep Dive (EM1~EM7)
    """
    print("\n" + "="*100)
    print("STEP 1: Momentum Factor Deep Dive")
    print("="*100)

    # 1.1 Strategy-level IC Analysis
    print("\n[1.1] Strategy-level IC Analysis")
    print("-"*100)

    results = []
    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'
        if return_col not in df.columns:
            continue

        print(f"\n{period}d Returns:")
        print(f"{'Strategy':<15} {'Spearman IC':>12} {'Samples':>10} {'Status'}")
        print("-"*50)

        for strategy in MOMENTUM_STRATEGIES:
            if strategy not in df.columns:
                continue

            ic = calculate_ic(df, strategy, return_col)

            if np.isnan(ic['spearman']):
                status = 'N/A'
            elif ic['spearman'] < -0.05:
                status = 'NEGATIVE'
            elif ic['spearman'] < 0:
                status = 'Weak Neg'
            elif ic['spearman'] < 0.05:
                status = 'Neutral'
            else:
                status = 'Positive'

            print(f"{strategy:<15} {ic['spearman']:>12.4f} {ic['n']:>10,} {status}")

            results.append({
                'strategy': strategy,
                'period': f'{period}d',
                'pearson_ic': ic['pearson'],
                'spearman_ic': ic['spearman'],
                'n_samples': ic['n'],
                'status': status
            })

    # Also add total momentum_score
    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'
        if return_col not in df.columns:
            continue
        ic = calculate_ic(df, 'momentum_score', return_col)
        results.append({
            'strategy': 'momentum_score (total)',
            'period': f'{period}d',
            'pearson_ic': ic['pearson'],
            'spearman_ic': ic['spearman'],
            'n_samples': ic['n'],
            'status': 'Total'
        })

    df_ic = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_momentum_strategies_ic.csv')
    df_ic.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 1.2 Sector-level IC for each strategy
    print("\n[1.2] Momentum Strategies by Sector")
    print("-"*100)

    sector_results = []
    sectors = df['sector'].dropna().unique()

    for strategy in MOMENTUM_STRATEGIES:
        if strategy not in df.columns:
            continue

        print(f"\n{strategy}:")
        print(f"{'Sector':<25} {'IC (30d)':>12} {'Samples':>10}")
        print("-"*50)

        for sector in sorted(sectors):
            df_sector = df[df['sector'] == sector]
            ic = calculate_ic(df_sector, strategy, 'return_30d')

            if not np.isnan(ic['spearman']):
                print(f"{sector:<25} {ic['spearman']:>12.4f} {ic['n']:>10,}")

                sector_results.append({
                    'strategy': strategy,
                    'sector': sector,
                    'ic_30d': ic['spearman'],
                    'n_samples': ic['n']
                })

    df_sector_ic = pd.DataFrame(sector_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_momentum_by_sector.csv')
    df_sector_ic.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 1.3 Rolling IC (by date)
    print("\n[1.3] Rolling IC by Date")
    print("-"*100)

    rolling_results = []
    dates = sorted(df['date'].unique())

    for strategy in MOMENTUM_STRATEGIES:
        if strategy not in df.columns:
            continue

        for d in dates:
            df_date = df[df['date'] == d]
            ic = calculate_ic(df_date, strategy, 'return_30d')

            if not np.isnan(ic['spearman']):
                rolling_results.append({
                    'strategy': strategy,
                    'date': str(d),
                    'ic_30d': ic['spearman'],
                    'n_samples': ic['n']
                })

    df_rolling = pd.DataFrame(rolling_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_momentum_rolling.csv')
    df_rolling.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_path}")

    # Print summary by strategy
    if len(df_rolling) > 0:
        print(f"\n{'Strategy':<15} {'Mean IC':>10} {'Std':>10} {'Positive%':>10}")
        print("-"*50)
        for strategy in MOMENTUM_STRATEGIES:
            s_data = df_rolling[df_rolling['strategy'] == strategy]['ic_30d']
            if len(s_data) > 0:
                mean_ic = s_data.mean()
                std_ic = s_data.std()
                pos_pct = (s_data > 0).mean() * 100
                print(f"{strategy:<15} {mean_ic:>10.4f} {std_ic:>10.4f} {pos_pct:>9.1f}%")

    # 1.4 Decile Test for Momentum Strategies (with Directional Accuracy)
    print("\n[1.4] Decile Test for Momentum Strategies (with Directional Accuracy)")
    print("-"*100)

    decile_results = []

    for strategy in MOMENTUM_STRATEGIES:
        if strategy not in df.columns:
            continue

        df_valid = df[[strategy, 'return_30d']].dropna()

        if len(df_valid) < 100:
            continue

        try:
            df_valid['decile'] = pd.qcut(df_valid[strategy], q=10, labels=False, duplicates='drop') + 1
        except:
            continue

        decile_stats = df_valid.groupby('decile').agg({
            strategy: ['min', 'max', 'count'],
            'return_30d': ['mean', 'median']
        }).reset_index()

        decile_stats.columns = ['decile', 'score_min', 'score_max', 'count', 'avg_return', 'median_return']

        # Win rate (positive return ratio)
        win_rates = df_valid.groupby('decile').apply(
            lambda x: (x['return_30d'] > 0).mean() * 100,
            include_groups=False
        ).reset_index(name='win_rate')

        # Directional Accuracy (NEW)
        dir_accuracy = df_valid.groupby('decile').apply(
            lambda x: calculate_directional_accuracy(x.name, x['return_30d']),
            include_groups=False
        ).reset_index(name='directional_accuracy')

        # Signal type (NEW)
        signal_types = pd.DataFrame({
            'decile': range(1, 11),
            'signal': [get_signal_type(d) for d in range(1, 11)]
        })

        decile_stats = pd.merge(decile_stats, win_rates, on='decile')
        decile_stats = pd.merge(decile_stats, dir_accuracy, on='decile')
        decile_stats = pd.merge(decile_stats, signal_types, on='decile', how='left')
        decile_stats['strategy'] = strategy
        decile_results.append(decile_stats)

        # Print summary
        print(f"\n{strategy}:")
        print(f"{'Decile':<10} {'Avg Return':>12} {'Win Rate':>10} {'Dir Acc':>10} {'Signal':>8}")
        print("-"*55)
        for _, row in decile_stats.iterrows():
            print(f"D{int(row['decile']):<9} {row['avg_return']:>11.2f}% {row['win_rate']:>9.1f}% "
                  f"{row['directional_accuracy']:>9.1f}% {row['signal']:>8}")

        # Monotonicity check
        corr = stats.spearmanr(decile_stats['decile'], decile_stats['avg_return'])[0]
        print(f"Monotonicity (rho): {corr:.3f}")

        # Buy/Sell accuracy summary
        buy_data = df_valid[df_valid['decile'] >= 8]['return_30d']
        sell_data = df_valid[df_valid['decile'] <= 3]['return_30d']
        buy_acc = (buy_data > 0).mean() * 100 if len(buy_data) > 0 else 0
        sell_acc = (sell_data < 0).mean() * 100 if len(sell_data) > 0 else 0
        print(f"Buy Accuracy (D8-D10): {buy_acc:.1f}%  |  Sell Accuracy (D1-D3): {sell_acc:.1f}%")

    if decile_results:
        df_decile = pd.concat(decile_results, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'us_phase2_momentum_decile.csv')
        df_decile.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_path}")

    return df_ic


# ============================================================================
# STEP 2: Healthcare Sector Deep Dive
# ============================================================================

async def step2_healthcare_deep_dive(df):
    """
    STEP 2: Healthcare Sector Deep Dive
    """
    print("\n" + "="*100)
    print("STEP 2: Healthcare Sector Deep Dive")
    print("="*100)

    df_healthcare = df[df['sector'] == 'HEALTHCARE'].copy()
    print(f"\nHealthcare samples: {len(df_healthcare):,}")

    if len(df_healthcare) < 100:
        print("Insufficient Healthcare data. Skipping...")
        return None

    # 2.1 Industry-level IC
    print("\n[2.1] Healthcare Industry-level IC")
    print("-"*100)

    industry_results = []
    industries = df_healthcare['industry'].dropna().unique()

    print(f"{'Industry':<45} {'IC (30d)':>12} {'Samples':>10}")
    print("-"*70)

    for industry in sorted(industries):
        df_ind = df_healthcare[df_healthcare['industry'] == industry]

        if len(df_ind) < 30:
            continue

        ic = calculate_ic(df_ind, 'final_score', 'return_30d')

        if not np.isnan(ic['spearman']):
            print(f"{industry[:44]:<45} {ic['spearman']:>12.4f} {ic['n']:>10,}")

            industry_results.append({
                'industry': industry,
                'ic_30d': ic['spearman'],
                'n_samples': ic['n']
            })

    df_industry = pd.DataFrame(industry_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_healthcare_by_industry.csv')
    df_industry.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 2.2 Factor-level IC within Healthcare
    print("\n[2.2] Factor IC within Healthcare")
    print("-"*100)

    factor_results = []

    print(f"{'Factor':<20} {'IC (30d)':>12} {'IC (60d)':>12} {'Samples':>10}")
    print("-"*60)

    for factor in ALL_FACTORS:
        ic_30 = calculate_ic(df_healthcare, factor, 'return_30d')
        ic_60 = calculate_ic(df_healthcare, factor, 'return_60d')

        print(f"{factor:<20} {ic_30['spearman']:>12.4f} {ic_60['spearman']:>12.4f} {ic_30['n']:>10,}")

        factor_results.append({
            'factor': factor,
            'ic_30d': ic_30['spearman'],
            'ic_60d': ic_60['spearman'],
            'n_samples': ic_30['n']
        })

    df_factor = pd.DataFrame(factor_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_healthcare_by_factor.csv')
    df_factor.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 2.3 Market Cap breakdown within Healthcare
    print("\n[2.3] Healthcare by Market Cap")
    print("-"*100)

    df_healthcare['market_cap_cat'] = pd.cut(
        df_healthcare['market_cap'],
        bins=[0, 300e6, 2e9, 10e9, 200e9, float('inf')],
        labels=['Micro (<300M)', 'Small (300M-2B)', 'Mid (2B-10B)', 'Large (10B-200B)', 'Mega (>200B)']
    )

    cap_results = []

    print(f"{'Market Cap':<20} {'IC (30d)':>12} {'Samples':>10}")
    print("-"*45)

    for cat in ['Micro (<300M)', 'Small (300M-2B)', 'Mid (2B-10B)', 'Large (10B-200B)', 'Mega (>200B)']:
        df_cap = df_healthcare[df_healthcare['market_cap_cat'] == cat]

        if len(df_cap) < 30:
            continue

        ic = calculate_ic(df_cap, 'final_score', 'return_30d')

        if not np.isnan(ic['spearman']):
            print(f"{cat:<20} {ic['spearman']:>12.4f} {ic['n']:>10,}")

            cap_results.append({
                'market_cap_category': cat,
                'ic_30d': ic['spearman'],
                'n_samples': ic['n']
            })

    df_cap = pd.DataFrame(cap_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_healthcare_by_marketcap.csv')
    df_cap.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 2.4 Problem stocks in Healthcare
    print("\n[2.4] Healthcare Problem Stocks")
    print("-"*100)

    problem_stocks = []

    # High score but low return
    df_valid = df_healthcare[['symbol', 'stock_name', 'date', 'final_score', 'return_30d', 'industry']].dropna()

    high_score_low_return = df_valid[
        (df_valid['final_score'] > df_valid['final_score'].quantile(0.8)) &
        (df_valid['return_30d'] < -10)
    ].sort_values('final_score', ascending=False).head(15)

    print("\nHigh Score (>80%) but Return < -10%:")
    if len(high_score_low_return) > 0:
        print(high_score_low_return[['symbol', 'date', 'final_score', 'return_30d', 'industry']].to_string(index=False))
        high_score_low_return['problem_type'] = 'high_score_low_return'
        problem_stocks.append(high_score_low_return)

    # Low score but high return
    low_score_high_return = df_valid[
        (df_valid['final_score'] < df_valid['final_score'].quantile(0.2)) &
        (df_valid['return_30d'] > 20)
    ].sort_values('return_30d', ascending=False).head(15)

    print("\nLow Score (<20%) but Return > +20%:")
    if len(low_score_high_return) > 0:
        print(low_score_high_return[['symbol', 'date', 'final_score', 'return_30d', 'industry']].to_string(index=False))
        low_score_high_return['problem_type'] = 'low_score_high_return'
        problem_stocks.append(low_score_high_return)

    if problem_stocks:
        df_problems = pd.concat(problem_stocks, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'us_phase2_healthcare_problem_stocks.csv')
        df_problems.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_path}")

    return df_industry


# ============================================================================
# STEP 3: Exchange Deep Dive (NYSE vs NASDAQ)
# ============================================================================

async def step3_exchange_deep_dive(df):
    """
    STEP 3: Exchange Deep Dive (NYSE vs NASDAQ)
    """
    print("\n" + "="*100)
    print("STEP 3: Exchange Deep Dive (NYSE vs NASDAQ)")
    print("="*100)

    # Filter to main exchanges
    df_main = df[df['exchange'].isin(['NYSE', 'NASDAQ'])].copy()
    print(f"\nMain exchange samples: {len(df_main):,}")
    print(f"NYSE: {len(df_main[df_main['exchange'] == 'NYSE']):,}")
    print(f"NASDAQ: {len(df_main[df_main['exchange'] == 'NASDAQ']):,}")

    # 3.1 Sector distribution by exchange
    print("\n[3.1] Sector Distribution by Exchange")
    print("-"*100)

    sector_dist = df_main.groupby(['exchange', 'sector']).size().unstack(fill_value=0)

    dist_results = []
    for sector in sector_dist.columns:
        nyse_count = sector_dist.loc['NYSE', sector] if 'NYSE' in sector_dist.index else 0
        nasdaq_count = sector_dist.loc['NASDAQ', sector] if 'NASDAQ' in sector_dist.index else 0
        total = nyse_count + nasdaq_count

        dist_results.append({
            'sector': sector,
            'NYSE': nyse_count,
            'NASDAQ': nasdaq_count,
            'Total': total,
            'NYSE_pct': nyse_count / total * 100 if total > 0 else 0,
            'NASDAQ_pct': nasdaq_count / total * 100 if total > 0 else 0
        })

    df_dist = pd.DataFrame(dist_results).sort_values('Total', ascending=False)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_exchange_sector_dist.csv')
    df_dist.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"{'Sector':<25} {'NYSE':>10} {'NASDAQ':>10} {'NYSE%':>10}")
    print("-"*60)
    for _, row in df_dist.iterrows():
        print(f"{row['sector'][:24]:<25} {row['NYSE']:>10,} {row['NASDAQ']:>10,} {row['NYSE_pct']:>9.1f}%")
    print(f"\nSaved: {output_path}")

    # 3.2 Factor IC by exchange
    print("\n[3.2] Factor IC by Exchange")
    print("-"*100)

    factor_results = []

    print(f"{'Factor':<20} {'NYSE IC':>12} {'NASDAQ IC':>12} {'Difference':>12}")
    print("-"*60)

    for factor in ALL_FACTORS:
        ic_nyse = calculate_ic(df_main[df_main['exchange'] == 'NYSE'], factor, 'return_30d')
        ic_nasdaq = calculate_ic(df_main[df_main['exchange'] == 'NASDAQ'], factor, 'return_30d')

        diff = ic_nyse['spearman'] - ic_nasdaq['spearman']

        print(f"{factor:<20} {ic_nyse['spearman']:>12.4f} {ic_nasdaq['spearman']:>12.4f} {diff:>+12.4f}")

        factor_results.append({
            'factor': factor,
            'NYSE_ic': ic_nyse['spearman'],
            'NASDAQ_ic': ic_nasdaq['spearman'],
            'difference': diff,
            'NYSE_n': ic_nyse['n'],
            'NASDAQ_n': ic_nasdaq['n']
        })

    df_factor = pd.DataFrame(factor_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_exchange_factor_ic.csv')
    df_factor.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 3.3 Sector x Exchange cross analysis
    print("\n[3.3] Sector x Exchange Cross Analysis (IC Heatmap)")
    print("-"*100)

    cross_results = []

    print(f"{'Sector':<25} {'NYSE IC':>12} {'NASDAQ IC':>12} {'Diff':>10}")
    print("-"*65)

    for sector in sorted(df_main['sector'].dropna().unique()):
        df_nyse = df_main[(df_main['exchange'] == 'NYSE') & (df_main['sector'] == sector)]
        df_nasdaq = df_main[(df_main['exchange'] == 'NASDAQ') & (df_main['sector'] == sector)]

        ic_nyse = calculate_ic(df_nyse, 'final_score', 'return_30d')
        ic_nasdaq = calculate_ic(df_nasdaq, 'final_score', 'return_30d')

        if ic_nyse['n'] >= 30 or ic_nasdaq['n'] >= 30:
            diff = (ic_nyse['spearman'] if not np.isnan(ic_nyse['spearman']) else 0) - \
                   (ic_nasdaq['spearman'] if not np.isnan(ic_nasdaq['spearman']) else 0)

            nyse_str = f"{ic_nyse['spearman']:.4f}" if not np.isnan(ic_nyse['spearman']) else "N/A"
            nasdaq_str = f"{ic_nasdaq['spearman']:.4f}" if not np.isnan(ic_nasdaq['spearman']) else "N/A"

            print(f"{sector[:24]:<25} {nyse_str:>12} {nasdaq_str:>12} {diff:>+10.4f}")

            cross_results.append({
                'sector': sector,
                'NYSE_ic': ic_nyse['spearman'],
                'NASDAQ_ic': ic_nasdaq['spearman'],
                'difference': diff,
                'NYSE_n': ic_nyse['n'],
                'NASDAQ_n': ic_nasdaq['n']
            })

    df_cross = pd.DataFrame(cross_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_exchange_cross_analysis.csv')
    df_cross.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    return df_cross


# ============================================================================
# STEP 4: Value Factor Deep Dive
# ============================================================================

async def step4_value_deep_dive(df):
    """
    STEP 4: Value Factor Deep Dive (RV1~RV6)
    """
    print("\n" + "="*100)
    print("STEP 4: Value Factor Deep Dive")
    print("="*100)

    # 4.1 Strategy-level IC Analysis
    print("\n[4.1] Value Strategy-level IC Analysis")
    print("-"*100)

    results = []
    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'
        if return_col not in df.columns:
            continue

        print(f"\n{period}d Returns:")
        print(f"{'Strategy':<15} {'Spearman IC':>12} {'Samples':>10} {'Status'}")
        print("-"*50)

        for strategy in VALUE_STRATEGIES:
            if strategy not in df.columns:
                continue

            ic = calculate_ic(df, strategy, return_col)

            if np.isnan(ic['spearman']):
                status = 'N/A'
            elif ic['spearman'] < -0.05:
                status = 'NEGATIVE'
            elif ic['spearman'] < 0:
                status = 'Weak Neg'
            elif ic['spearman'] < 0.05:
                status = 'Neutral'
            else:
                status = 'Positive'

            print(f"{strategy:<15} {ic['spearman']:>12.4f} {ic['n']:>10,} {status}")

            results.append({
                'strategy': strategy,
                'period': f'{period}d',
                'pearson_ic': ic['pearson'],
                'spearman_ic': ic['spearman'],
                'n_samples': ic['n'],
                'status': status
            })

    # Also add total value_score
    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'
        if return_col not in df.columns:
            continue
        ic = calculate_ic(df, 'value_score', return_col)
        results.append({
            'strategy': 'value_score (total)',
            'period': f'{period}d',
            'pearson_ic': ic['pearson'],
            'spearman_ic': ic['spearman'],
            'n_samples': ic['n'],
            'status': 'Total'
        })

    df_ic = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_value_strategies_ic.csv')
    df_ic.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 4.2 Sector-level IC for value strategies
    print("\n[4.2] Value Strategies by Sector")
    print("-"*100)

    sector_results = []
    sectors = df['sector'].dropna().unique()

    for strategy in VALUE_STRATEGIES:
        if strategy not in df.columns:
            continue

        print(f"\n{strategy}:")
        print(f"{'Sector':<25} {'IC (30d)':>12} {'Samples':>10}")
        print("-"*50)

        for sector in sorted(sectors):
            df_sector = df[df['sector'] == sector]
            ic = calculate_ic(df_sector, strategy, 'return_30d')

            if not np.isnan(ic['spearman']):
                print(f"{sector:<25} {ic['spearman']:>12.4f} {ic['n']:>10,}")

                sector_results.append({
                    'strategy': strategy,
                    'sector': sector,
                    'ic_30d': ic['spearman'],
                    'n_samples': ic['n']
                })

    df_sector_ic = pd.DataFrame(sector_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_value_by_sector.csv')
    df_sector_ic.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 4.3 Decile Test for Value Strategies (with Directional Accuracy)
    print("\n[4.3] Decile Test for Value Strategies (with Directional Accuracy)")
    print("-"*100)

    decile_results = []

    for strategy in VALUE_STRATEGIES:
        if strategy not in df.columns:
            continue

        df_valid = df[[strategy, 'return_30d']].dropna()

        if len(df_valid) < 100:
            continue

        try:
            df_valid['decile'] = pd.qcut(df_valid[strategy], q=10, labels=False, duplicates='drop') + 1
        except:
            continue

        decile_stats = df_valid.groupby('decile').agg({
            strategy: ['min', 'max', 'count'],
            'return_30d': ['mean', 'median']
        }).reset_index()

        decile_stats.columns = ['decile', 'score_min', 'score_max', 'count', 'avg_return', 'median_return']

        # Win rate
        win_rates = df_valid.groupby('decile').apply(
            lambda x: (x['return_30d'] > 0).mean() * 100,
            include_groups=False
        ).reset_index(name='win_rate')

        # Directional Accuracy
        dir_accuracy = df_valid.groupby('decile').apply(
            lambda x: calculate_directional_accuracy(x.name, x['return_30d']),
            include_groups=False
        ).reset_index(name='directional_accuracy')

        # Signal type
        signal_types = pd.DataFrame({
            'decile': range(1, 11),
            'signal': [get_signal_type(d) for d in range(1, 11)]
        })

        decile_stats = pd.merge(decile_stats, win_rates, on='decile')
        decile_stats = pd.merge(decile_stats, dir_accuracy, on='decile')
        decile_stats = pd.merge(decile_stats, signal_types, on='decile', how='left')
        decile_stats['strategy'] = strategy
        decile_results.append(decile_stats)

        # Print summary
        print(f"\n{strategy}:")
        print(f"{'Decile':<10} {'Avg Return':>12} {'Win Rate':>10} {'Dir Acc':>10} {'Signal':>8}")
        print("-"*55)
        for _, row in decile_stats.iterrows():
            print(f"D{int(row['decile']):<9} {row['avg_return']:>11.2f}% {row['win_rate']:>9.1f}% "
                  f"{row['directional_accuracy']:>9.1f}% {row['signal']:>8}")

        # Monotonicity check
        corr = stats.spearmanr(decile_stats['decile'], decile_stats['avg_return'])[0]
        print(f"Monotonicity (rho): {corr:.3f}")

        # Buy/Sell accuracy summary
        buy_data = df_valid[df_valid['decile'] >= 8]['return_30d']
        sell_data = df_valid[df_valid['decile'] <= 3]['return_30d']
        buy_acc = (buy_data > 0).mean() * 100 if len(buy_data) > 0 else 0
        sell_acc = (sell_data < 0).mean() * 100 if len(sell_data) > 0 else 0
        print(f"Buy Accuracy (D8-D10): {buy_acc:.1f}%  |  Sell Accuracy (D1-D3): {sell_acc:.1f}%")

    if decile_results:
        df_decile = pd.concat(decile_results, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'us_phase2_value_decile.csv')
        df_decile.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_path}")

    return df_ic


# ============================================================================
# STEP 5: NASDAQ Detailed Analysis
# ============================================================================

async def step5_nasdaq_deep_dive(df):
    """
    STEP 5: NASDAQ Detailed Analysis
    """
    print("\n" + "="*100)
    print("STEP 5: NASDAQ Detailed Analysis")
    print("="*100)

    df_nasdaq = df[df['exchange'] == 'NASDAQ'].copy()
    print(f"\nNASDAQ samples: {len(df_nasdaq):,}")

    if len(df_nasdaq) < 100:
        print("Insufficient NASDAQ data. Skipping...")
        return None

    # 5.1 Sector-level IC within NASDAQ
    print("\n[5.1] NASDAQ Sector-level IC")
    print("-"*100)

    sector_results = []
    sectors = df_nasdaq['sector'].dropna().unique()

    print(f"{'Sector':<30} {'IC (30d)':>12} {'IC (90d)':>12} {'IC (252d)':>12} {'Samples':>10}")
    print("-"*80)

    for sector in sorted(sectors):
        df_sector = df_nasdaq[df_nasdaq['sector'] == sector]

        if len(df_sector) < 30:
            continue

        ic_30 = calculate_ic(df_sector, 'final_score', 'return_30d')
        ic_90 = calculate_ic(df_sector, 'final_score', 'return_90d')
        ic_252 = calculate_ic(df_sector, 'final_score', 'return_252d')

        ic_30_str = f"{ic_30['spearman']:.4f}" if not np.isnan(ic_30['spearman']) else "N/A"
        ic_90_str = f"{ic_90['spearman']:.4f}" if not np.isnan(ic_90['spearman']) else "N/A"
        ic_252_str = f"{ic_252['spearman']:.4f}" if not np.isnan(ic_252['spearman']) else "N/A"

        print(f"{sector[:29]:<30} {ic_30_str:>12} {ic_90_str:>12} {ic_252_str:>12} {ic_30['n']:>10,}")

        sector_results.append({
            'sector': sector,
            'ic_30d': ic_30['spearman'],
            'ic_90d': ic_90['spearman'],
            'ic_252d': ic_252['spearman'],
            'n_samples': ic_30['n']
        })

    df_sector = pd.DataFrame(sector_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_nasdaq_by_sector.csv')
    df_sector.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 5.2 Factor IC within NASDAQ
    print("\n[5.2] Factor IC within NASDAQ")
    print("-"*100)

    factor_results = []

    print(f"{'Factor':<20} {'IC (30d)':>12} {'IC (90d)':>12} {'IC (252d)':>12} {'Samples':>10}")
    print("-"*70)

    for factor in ALL_FACTORS:
        ic_30 = calculate_ic(df_nasdaq, factor, 'return_30d')
        ic_90 = calculate_ic(df_nasdaq, factor, 'return_90d')
        ic_252 = calculate_ic(df_nasdaq, factor, 'return_252d')

        ic_30_str = f"{ic_30['spearman']:.4f}" if not np.isnan(ic_30['spearman']) else "N/A"
        ic_90_str = f"{ic_90['spearman']:.4f}" if not np.isnan(ic_90['spearman']) else "N/A"
        ic_252_str = f"{ic_252['spearman']:.4f}" if not np.isnan(ic_252['spearman']) else "N/A"

        print(f"{factor:<20} {ic_30_str:>12} {ic_90_str:>12} {ic_252_str:>12} {ic_30['n']:>10,}")

        factor_results.append({
            'factor': factor,
            'ic_30d': ic_30['spearman'],
            'ic_90d': ic_90['spearman'],
            'ic_252d': ic_252['spearman'],
            'n_samples': ic_30['n']
        })

    df_factor = pd.DataFrame(factor_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_nasdaq_by_factor.csv')
    df_factor.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 5.3 NASDAQ Non-Tech Analysis
    print("\n[5.3] NASDAQ Non-Tech Analysis")
    print("-"*100)

    tech_sectors = ['TECHNOLOGY', 'COMMUNICATION SERVICES']
    df_nasdaq_nontech = df_nasdaq[~df_nasdaq['sector'].isin(tech_sectors)]

    print(f"NASDAQ Non-Tech samples: {len(df_nasdaq_nontech):,}")

    nontech_results = []

    print(f"\n{'Factor':<20} {'IC (30d)':>12} {'IC (90d)':>12} {'Samples':>10}")
    print("-"*55)

    for factor in ALL_FACTORS:
        ic_30 = calculate_ic(df_nasdaq_nontech, factor, 'return_30d')
        ic_90 = calculate_ic(df_nasdaq_nontech, factor, 'return_90d')

        print(f"{factor:<20} {ic_30['spearman']:>12.4f} {ic_90['spearman']:>12.4f} {ic_30['n']:>10,}")

        nontech_results.append({
            'factor': factor,
            'ic_30d': ic_30['spearman'],
            'ic_90d': ic_90['spearman'],
            'n_samples': ic_30['n']
        })

    df_nontech = pd.DataFrame(nontech_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_nasdaq_nontech.csv')
    df_nontech.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    return df_sector


# ============================================================================
# STEP 6: Quality Factor Deep Dive
# ============================================================================

async def step6_quality_deep_dive(df):
    """
    STEP 6: Quality Factor Deep Dive
    - quality_score 기반 분석
    - 섹터별, 기간별, Decile 테스트
    """
    print("\n" + "="*100)
    print("STEP 6: Quality Factor Deep Dive")
    print("="*100)

    # 6.1 Quality Factor IC by Return Period
    print("\n[6.1] Quality Factor IC by Return Period")
    print("-"*100)

    results = []
    print(f"{'Period':<15} {'Spearman IC':>12} {'Pearson IC':>12} {'Samples':>10} {'Status'}")
    print("-"*55)

    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'
        if return_col not in df.columns:
            continue

        ic = calculate_ic(df, 'quality_score', return_col)

        if np.isnan(ic['spearman']):
            status = 'N/A'
        elif ic['spearman'] < -0.05:
            status = 'NEGATIVE'
        elif ic['spearman'] < 0:
            status = 'Weak Neg'
        elif ic['spearman'] < 0.05:
            status = 'Neutral'
        else:
            status = 'Positive'

        print(f"{period}d{'':<10} {ic['spearman']:>12.4f} {ic['pearson']:>12.4f} {ic['n']:>10,} {status}")

        results.append({
            'period': f'{period}d',
            'pearson_ic': ic['pearson'],
            'spearman_ic': ic['spearman'],
            'n_samples': ic['n'],
            'status': status
        })

    df_ic = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_quality_ic.csv')
    df_ic.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 6.2 Quality Factor IC by Sector
    print("\n[6.2] Quality Factor IC by Sector")
    print("-"*100)

    sector_results = []
    sectors = df['sector'].dropna().unique()

    print(f"{'Sector':<30} {'IC (30d)':>12} {'IC (90d)':>12} {'IC (252d)':>12} {'Samples':>10}")
    print("-"*80)

    for sector in sorted(sectors):
        df_sector = df[df['sector'] == sector]

        if len(df_sector) < 30:
            continue

        ic_30 = calculate_ic(df_sector, 'quality_score', 'return_30d')
        ic_90 = calculate_ic(df_sector, 'quality_score', 'return_90d')
        ic_252 = calculate_ic(df_sector, 'quality_score', 'return_252d')

        ic_30_str = f"{ic_30['spearman']:.4f}" if not np.isnan(ic_30['spearman']) else "N/A"
        ic_90_str = f"{ic_90['spearman']:.4f}" if not np.isnan(ic_90['spearman']) else "N/A"
        ic_252_str = f"{ic_252['spearman']:.4f}" if not np.isnan(ic_252['spearman']) else "N/A"

        print(f"{sector[:29]:<30} {ic_30_str:>12} {ic_90_str:>12} {ic_252_str:>12} {ic_30['n']:>10,}")

        sector_results.append({
            'sector': sector,
            'ic_30d': ic_30['spearman'],
            'ic_90d': ic_90['spearman'],
            'ic_252d': ic_252['spearman'],
            'n_samples': ic_30['n']
        })

    df_sector = pd.DataFrame(sector_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_quality_by_sector.csv')
    df_sector.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 6.3 Quality Factor Rolling IC
    print("\n[6.3] Quality Factor Rolling IC by Date")
    print("-"*100)

    rolling_results = []
    dates = sorted(df['date'].unique())

    for d in dates:
        df_date = df[df['date'] == d]
        ic_30 = calculate_ic(df_date, 'quality_score', 'return_30d')
        ic_90 = calculate_ic(df_date, 'quality_score', 'return_90d')

        if not np.isnan(ic_30['spearman']):
            rolling_results.append({
                'date': str(d),
                'ic_30d': ic_30['spearman'],
                'ic_90d': ic_90['spearman'] if not np.isnan(ic_90['spearman']) else None,
                'n_samples': ic_30['n']
            })

    df_rolling = pd.DataFrame(rolling_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_quality_rolling.csv')
    df_rolling.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_path}")

    # Summary statistics
    if len(df_rolling) > 0:
        print(f"\nRolling IC Summary:")
        print(f"{'Metric':<20} {'IC 30d':>12} {'IC 90d':>12}")
        print("-"*45)
        mean_30 = df_rolling['ic_30d'].mean()
        std_30 = df_rolling['ic_30d'].std()
        pos_pct_30 = (df_rolling['ic_30d'] > 0).mean() * 100
        mean_90 = df_rolling['ic_90d'].dropna().mean() if 'ic_90d' in df_rolling.columns else np.nan
        std_90 = df_rolling['ic_90d'].dropna().std() if 'ic_90d' in df_rolling.columns else np.nan
        pos_pct_90 = (df_rolling['ic_90d'].dropna() > 0).mean() * 100 if 'ic_90d' in df_rolling.columns else np.nan

        print(f"{'Mean IC':<20} {mean_30:>12.4f} {mean_90:>12.4f}")
        print(f"{'Std IC':<20} {std_30:>12.4f} {std_90:>12.4f}")
        print(f"{'Positive Rate':<20} {pos_pct_30:>11.1f}% {pos_pct_90:>11.1f}%")

    # 6.4 Quality Factor Decile Test
    print("\n[6.4] Quality Factor Decile Test (with Directional Accuracy)")
    print("-"*100)

    decile_results = []

    for period in [30, 90, 252]:
        return_col = f'return_{period}d'
        if return_col not in df.columns:
            continue

        df_valid = df[['quality_score', return_col]].dropna()

        if len(df_valid) < 100:
            continue

        try:
            df_valid['decile'] = pd.qcut(df_valid['quality_score'], q=10, labels=False, duplicates='drop') + 1
        except:
            continue

        decile_stats = df_valid.groupby('decile').agg({
            'quality_score': ['min', 'max', 'count'],
            return_col: ['mean', 'median', 'std']
        }).reset_index()

        decile_stats.columns = ['decile', 'score_min', 'score_max', 'count', 'avg_return', 'median_return', 'std_return']

        # Win rate
        win_rates = df_valid.groupby('decile').apply(
            lambda x: (x[return_col] > 0).mean() * 100,
            include_groups=False
        ).reset_index(name='win_rate')

        # Directional Accuracy
        dir_accuracy = df_valid.groupby('decile').apply(
            lambda x: calculate_directional_accuracy(x.name, x[return_col]),
            include_groups=False
        ).reset_index(name='directional_accuracy')

        # Signal type
        signal_types = pd.DataFrame({
            'decile': range(1, 11),
            'signal': [get_signal_type(d) for d in range(1, 11)]
        })

        decile_stats = pd.merge(decile_stats, win_rates, on='decile')
        decile_stats = pd.merge(decile_stats, dir_accuracy, on='decile')
        decile_stats = pd.merge(decile_stats, signal_types, on='decile', how='left')
        decile_stats['period'] = f'{period}d'
        decile_results.append(decile_stats)

        # Print summary
        print(f"\nQuality Score - {period}d Returns:")
        print(f"{'Decile':<10} {'Avg Return':>12} {'Median':>12} {'Win Rate':>10} {'Dir Acc':>10} {'Signal':>8}")
        print("-"*70)
        for _, row in decile_stats.iterrows():
            print(f"D{int(row['decile']):<9} {row['avg_return']:>11.2f}% {row['median_return']:>11.2f}% "
                  f"{row['win_rate']:>9.1f}% {row['directional_accuracy']:>9.1f}% {row['signal']:>8}")

        # Monotonicity check
        corr = stats.spearmanr(decile_stats['decile'], decile_stats['avg_return'])[0]
        corr_med = stats.spearmanr(decile_stats['decile'], decile_stats['median_return'])[0]
        print(f"Monotonicity (Mean): {corr:.3f} | Monotonicity (Median): {corr_med:.3f}")

        # Buy/Sell accuracy summary
        buy_data = df_valid[df_valid['decile'] >= 8][return_col]
        sell_data = df_valid[df_valid['decile'] <= 3][return_col]
        buy_acc = (buy_data > 0).mean() * 100 if len(buy_data) > 0 else 0
        sell_acc = (sell_data < 0).mean() * 100 if len(sell_data) > 0 else 0
        print(f"Buy Accuracy (D8-D10): {buy_acc:.1f}%  |  Sell Accuracy (D1-D3): {sell_acc:.1f}%")

    if decile_results:
        df_decile = pd.concat(decile_results, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'us_phase2_quality_decile.csv')
        df_decile.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_path}")

    # 6.5 Quality vs Growth Factor Comparison
    print("\n[6.5] Quality vs Growth Factor Comparison")
    print("-"*100)

    comparison_results = []

    print(f"{'Period':<10} {'Quality IC':>12} {'Growth IC':>12} {'Diff (G-Q)':>12} {'Better':>10}")
    print("-"*60)

    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'
        if return_col not in df.columns:
            continue

        ic_quality = calculate_ic(df, 'quality_score', return_col)
        ic_growth = calculate_ic(df, 'growth_score', return_col)

        diff = ic_growth['spearman'] - ic_quality['spearman']
        better = 'Growth' if diff > 0 else 'Quality'

        print(f"{period}d{'':<5} {ic_quality['spearman']:>12.4f} {ic_growth['spearman']:>12.4f} "
              f"{diff:>+12.4f} {better:>10}")

        comparison_results.append({
            'period': f'{period}d',
            'quality_ic': ic_quality['spearman'],
            'growth_ic': ic_growth['spearman'],
            'difference': diff,
            'better': better
        })

    df_comparison = pd.DataFrame(comparison_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_phase2_quality_vs_growth.csv')
    df_comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    return df_ic


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution"""
    print("\n" + "="*100)
    print("US Quant Phase 2 - Deep Dive Analysis")
    print("="*100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Connect to database
    print("\nConnecting to database...")
    conn = await connect_db()
    print("Connected")

    try:
        # Load base data
        df = await load_base_data(conn)

        # STEP 1: Momentum Deep Dive
        await step1_momentum_deep_dive(df)

        # STEP 2: Healthcare Deep Dive
        await step2_healthcare_deep_dive(df)

        # STEP 3: Exchange Deep Dive
        await step3_exchange_deep_dive(df)

        # STEP 4: Value Factor Deep Dive (NEW)
        await step4_value_deep_dive(df)

        # STEP 5: NASDAQ Detailed Analysis (NEW)
        await step5_nasdaq_deep_dive(df)

        # STEP 6: Quality Factor Deep Dive (NEW)
        await step6_quality_deep_dive(df)

        # Summary
        print("\n" + "="*100)
        print("All analyses completed successfully!")
        print("="*100)

        print(f"\nGenerated files in {OUTPUT_DIR}:")
        print("  [STEP 1 - Momentum]")
        print("    - us_phase2_momentum_strategies_ic.csv")
        print("    - us_phase2_momentum_by_sector.csv")
        print("    - us_phase2_momentum_rolling.csv")
        print("    - us_phase2_momentum_decile.csv (with directional_accuracy)")
        print("  [STEP 2 - Healthcare]")
        print("    - us_phase2_healthcare_by_industry.csv")
        print("    - us_phase2_healthcare_by_factor.csv")
        print("    - us_phase2_healthcare_by_marketcap.csv")
        print("    - us_phase2_healthcare_problem_stocks.csv")
        print("  [STEP 3 - Exchange]")
        print("    - us_phase2_exchange_sector_dist.csv")
        print("    - us_phase2_exchange_factor_ic.csv")
        print("    - us_phase2_exchange_cross_analysis.csv")
        print("  [STEP 4 - Value] (NEW)")
        print("    - us_phase2_value_strategies_ic.csv")
        print("    - us_phase2_value_by_sector.csv")
        print("    - us_phase2_value_decile.csv (with directional_accuracy)")
        print("  [STEP 5 - NASDAQ] (NEW)")
        print("    - us_phase2_nasdaq_by_sector.csv")
        print("    - us_phase2_nasdaq_by_factor.csv")
        print("    - us_phase2_nasdaq_nontech.csv")
        print("  [STEP 6 - Quality] (NEW)")
        print("    - us_phase2_quality_ic.csv")
        print("    - us_phase2_quality_by_sector.csv")
        print("    - us_phase2_quality_rolling.csv")
        print("    - us_phase2_quality_decile.csv (with median_return, directional_accuracy)")
        print("    - us_phase2_quality_vs_growth.csv")

    finally:
        await conn.close()
        print("\nDatabase connection closed")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    asyncio.run(main())
