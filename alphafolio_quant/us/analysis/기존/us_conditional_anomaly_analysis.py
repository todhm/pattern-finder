# -*- coding: utf-8 -*-
"""
US Quant - Conditional Strategy & Anomaly Detection Analysis
=============================================================

1. Conditional Strategy Analysis: 섹터별 저성과 전략 양의 IC 탐색
2. Anomaly Detection: Score-Return 회귀선 이탈 종목 탐지

Execute: python us/analysis/us_conditional_anomaly_analysis.py
Output: C:/project/alpha/quant/us/result/ folder (CSV files)
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from scipy import stats
# Linear regression using numpy (no sklearn needed)
from datetime import datetime
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
RETURN_PERIODS = [30, 60, 90, 180, 252]

# Low-performing strategies to analyze
LOW_PERF_MOMENTUM = ['EM1', 'EM2', 'EM5']
LOW_PERF_VALUE = ['RV2', 'RV5']

# All strategies for comparison
ALL_MOMENTUM = ['EM1', 'EM2', 'EM3', 'EM4', 'EM5', 'EM6', 'EM7']
ALL_VALUE = ['RV1', 'RV2', 'RV3', 'RV4', 'RV5', 'RV6']


async def connect_db():
    """Connect to PostgreSQL database"""
    return await asyncpg.connect(DATABASE_URL)


def calculate_ic(df, score_col, return_col):
    """Calculate Pearson and Spearman IC"""
    valid = df[[score_col, return_col]].dropna()

    if len(valid) < 30:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': len(valid)}

    # 상수 체크: 표준편차가 0이면 상관계수 계산 불가
    if valid[score_col].std() == 0 or valid[return_col].std() == 0:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': len(valid)}

    try:
        pearson_ic, p_pval = stats.pearsonr(valid[score_col], valid[return_col])
        spearman_ic, s_pval = stats.spearmanr(valid[score_col], valid[return_col])
    except:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': len(valid)}

    return {
        'pearson': pearson_ic,
        'spearman': spearman_ic,
        'p_pvalue': p_pval,
        's_pvalue': s_pval,
        'n': len(valid)
    }


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


async def load_base_data(conn):
    """Load base data from database"""
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
                    'growth_score', 'market_cap', 'return_30d', 'return_60d',
                    'return_90d', 'return_180d', 'return_252d']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

    # Parse momentum_v2_detail JSON
    print("\nParsing momentum strategy details...")
    for strategy in ALL_MOMENTUM:
        df[strategy] = df['momentum_v2_detail'].apply(
            lambda x: extract_strategy_score(x, strategy) if x else np.nan
        )

    # Parse value_v2_detail JSON
    print("Parsing value strategy details...")
    for strategy in ALL_VALUE:
        df[strategy] = df['value_v2_detail'].apply(
            lambda x: extract_strategy_score(x, strategy) if x else np.nan
        )

    return df


# ============================================================================
# PART 1: Conditional Strategy Analysis
# ============================================================================

async def analyze_conditional_strategies(df):
    """
    PART 1: Conditional Strategy Analysis
    - 저성과 전략들이 특정 섹터에서 양의 IC를 보이는지 분석
    """
    print("\n" + "="*100)
    print("PART 1: Conditional Strategy Analysis")
    print("="*100)

    sectors = df['sector'].dropna().unique()

    # 1.1 Momentum 저성과 전략 섹터별 IC
    print("\n[1.1] Low-Performing Momentum Strategies by Sector")
    print("-"*100)

    momentum_results = []

    for strategy in LOW_PERF_MOMENTUM:
        if strategy not in df.columns:
            continue

        print(f"\n{strategy}:")
        print(f"{'Sector':<30} {'IC 30d':>10} {'IC 90d':>10} {'IC 252d':>10} {'Samples':>10} {'Usable?':>10}")
        print("-"*85)

        for sector in sorted(sectors):
            df_sector = df[df['sector'] == sector]

            ic_30 = calculate_ic(df_sector, strategy, 'return_30d')
            ic_90 = calculate_ic(df_sector, strategy, 'return_90d')
            ic_252 = calculate_ic(df_sector, strategy, 'return_252d')

            # Determine if usable (IC > 0.03)
            usable_30 = ic_30['spearman'] > 0.03 if not np.isnan(ic_30['spearman']) else False
            usable_90 = ic_90['spearman'] > 0.03 if not np.isnan(ic_90['spearman']) else False
            usable_252 = ic_252['spearman'] > 0.03 if not np.isnan(ic_252['spearman']) else False
            usable = 'YES' if (usable_30 or usable_90 or usable_252) else 'NO'

            ic_30_str = f"{ic_30['spearman']:.4f}" if not np.isnan(ic_30['spearman']) else "N/A"
            ic_90_str = f"{ic_90['spearman']:.4f}" if not np.isnan(ic_90['spearman']) else "N/A"
            ic_252_str = f"{ic_252['spearman']:.4f}" if not np.isnan(ic_252['spearman']) else "N/A"

            # Highlight positive IC
            if usable == 'YES':
                print(f"{sector[:29]:<30} {ic_30_str:>10} {ic_90_str:>10} {ic_252_str:>10} {ic_30['n']:>10,} {'** YES **':>10}")
            else:
                print(f"{sector[:29]:<30} {ic_30_str:>10} {ic_90_str:>10} {ic_252_str:>10} {ic_30['n']:>10,} {usable:>10}")

            momentum_results.append({
                'strategy': strategy,
                'sector': sector,
                'ic_30d': ic_30['spearman'],
                'ic_90d': ic_90['spearman'],
                'ic_252d': ic_252['spearman'],
                'n_samples': ic_30['n'],
                'usable_30d': usable_30,
                'usable_90d': usable_90,
                'usable_252d': usable_252
            })

    df_momentum = pd.DataFrame(momentum_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_conditional_momentum_by_sector.csv')
    df_momentum.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 1.2 Value 저성과 전략 섹터별 IC
    print("\n[1.2] Low-Performing Value Strategies by Sector")
    print("-"*100)

    value_results = []

    for strategy in LOW_PERF_VALUE:
        if strategy not in df.columns:
            continue

        print(f"\n{strategy}:")
        print(f"{'Sector':<30} {'IC 30d':>10} {'IC 90d':>10} {'IC 252d':>10} {'Samples':>10} {'Usable?':>10}")
        print("-"*85)

        for sector in sorted(sectors):
            df_sector = df[df['sector'] == sector]

            ic_30 = calculate_ic(df_sector, strategy, 'return_30d')
            ic_90 = calculate_ic(df_sector, strategy, 'return_90d')
            ic_252 = calculate_ic(df_sector, strategy, 'return_252d')

            # Determine if usable (IC > 0.03)
            usable_30 = ic_30['spearman'] > 0.03 if not np.isnan(ic_30['spearman']) else False
            usable_90 = ic_90['spearman'] > 0.03 if not np.isnan(ic_90['spearman']) else False
            usable_252 = ic_252['spearman'] > 0.03 if not np.isnan(ic_252['spearman']) else False
            usable = 'YES' if (usable_30 or usable_90 or usable_252) else 'NO'

            ic_30_str = f"{ic_30['spearman']:.4f}" if not np.isnan(ic_30['spearman']) else "N/A"
            ic_90_str = f"{ic_90['spearman']:.4f}" if not np.isnan(ic_90['spearman']) else "N/A"
            ic_252_str = f"{ic_252['spearman']:.4f}" if not np.isnan(ic_252['spearman']) else "N/A"

            if usable == 'YES':
                print(f"{sector[:29]:<30} {ic_30_str:>10} {ic_90_str:>10} {ic_252_str:>10} {ic_30['n']:>10,} {'** YES **':>10}")
            else:
                print(f"{sector[:29]:<30} {ic_30_str:>10} {ic_90_str:>10} {ic_252_str:>10} {ic_30['n']:>10,} {usable:>10}")

            value_results.append({
                'strategy': strategy,
                'sector': sector,
                'ic_30d': ic_30['spearman'],
                'ic_90d': ic_90['spearman'],
                'ic_252d': ic_252['spearman'],
                'n_samples': ic_30['n'],
                'usable_30d': usable_30,
                'usable_90d': usable_90,
                'usable_252d': usable_252
            })

    df_value = pd.DataFrame(value_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_conditional_value_by_sector.csv')
    df_value.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 1.3 Market State별 분석
    print("\n[1.3] Low-Performing Strategies by Market State")
    print("-"*100)

    market_state_results = []
    market_states = df['market_state'].dropna().unique()

    all_low_perf = LOW_PERF_MOMENTUM + LOW_PERF_VALUE

    for strategy in all_low_perf:
        if strategy not in df.columns:
            continue

        print(f"\n{strategy}:")
        print(f"{'Market State':<20} {'IC 30d':>10} {'IC 90d':>10} {'IC 252d':>10} {'Samples':>10}")
        print("-"*60)

        for state in sorted(market_states):
            df_state = df[df['market_state'] == state]

            if len(df_state) < 100:
                continue

            ic_30 = calculate_ic(df_state, strategy, 'return_30d')
            ic_90 = calculate_ic(df_state, strategy, 'return_90d')
            ic_252 = calculate_ic(df_state, strategy, 'return_252d')

            ic_30_str = f"{ic_30['spearman']:.4f}" if not np.isnan(ic_30['spearman']) else "N/A"
            ic_90_str = f"{ic_90['spearman']:.4f}" if not np.isnan(ic_90['spearman']) else "N/A"
            ic_252_str = f"{ic_252['spearman']:.4f}" if not np.isnan(ic_252['spearman']) else "N/A"

            print(f"{str(state)[:19]:<20} {ic_30_str:>10} {ic_90_str:>10} {ic_252_str:>10} {ic_30['n']:>10,}")

            market_state_results.append({
                'strategy': strategy,
                'market_state': state,
                'ic_30d': ic_30['spearman'],
                'ic_90d': ic_90['spearman'],
                'ic_252d': ic_252['spearman'],
                'n_samples': ic_30['n']
            })

    df_market = pd.DataFrame(market_state_results)
    output_path = os.path.join(OUTPUT_DIR, 'us_conditional_by_market_state.csv')
    df_market.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_path}")

    # 1.4 Summary: Usable Conditions
    print("\n[1.4] Summary: Usable Conditions for Low-Performing Strategies")
    print("-"*100)

    usable_conditions = []

    # Find usable momentum conditions
    for _, row in df_momentum.iterrows():
        if row['usable_30d'] or row['usable_90d'] or row['usable_252d']:
            periods = []
            if row['usable_30d']: periods.append(f"30d (IC={row['ic_30d']:.3f})")
            if row['usable_90d']: periods.append(f"90d (IC={row['ic_90d']:.3f})")
            if row['usable_252d']: periods.append(f"252d (IC={row['ic_252d']:.3f})")

            usable_conditions.append({
                'strategy': row['strategy'],
                'condition_type': 'Sector',
                'condition_value': row['sector'],
                'usable_periods': ', '.join(periods),
                'n_samples': row['n_samples']
            })

    # Find usable value conditions
    for _, row in df_value.iterrows():
        if row['usable_30d'] or row['usable_90d'] or row['usable_252d']:
            periods = []
            if row['usable_30d']: periods.append(f"30d (IC={row['ic_30d']:.3f})")
            if row['usable_90d']: periods.append(f"90d (IC={row['ic_90d']:.3f})")
            if row['usable_252d']: periods.append(f"252d (IC={row['ic_252d']:.3f})")

            usable_conditions.append({
                'strategy': row['strategy'],
                'condition_type': 'Sector',
                'condition_value': row['sector'],
                'usable_periods': ', '.join(periods),
                'n_samples': row['n_samples']
            })

    if usable_conditions:
        df_usable = pd.DataFrame(usable_conditions)
        print(df_usable.to_string(index=False))

        output_path = os.path.join(OUTPUT_DIR, 'us_conditional_usable_summary.csv')
        df_usable.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_path}")
    else:
        print("No usable conditions found.")

    return df_momentum, df_value


# ============================================================================
# PART 2: Anomaly Detection
# ============================================================================

async def analyze_anomalies(df):
    """
    PART 2: Anomaly Detection
    - Score-Return 회귀선 이탈 종목 탐지
    - Positive Anomaly: 저점수 + 고수익률
    - Negative Anomaly: 고점수 + 저수익률
    """
    print("\n" + "="*100)
    print("PART 2: Anomaly Detection")
    print("="*100)

    anomaly_results = []

    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'

        if return_col not in df.columns:
            continue

        print(f"\n[2.{RETURN_PERIODS.index(period)+1}] {period}d Return Anomaly Detection")
        print("-"*100)

        # Prepare data
        df_valid = df[['symbol', 'date', 'stock_name', 'final_score', return_col,
                       'sector', 'industry', 'exchange', 'market_cap']].dropna()

        if len(df_valid) < 100:
            print(f"Insufficient data for {period}d analysis")
            continue

        # Winsorize returns for regression (1st and 99th percentile)
        lower = df_valid[return_col].quantile(0.01)
        upper = df_valid[return_col].quantile(0.99)
        df_valid['return_winsorized'] = df_valid[return_col].clip(lower, upper)

        # Fit linear regression using numpy
        X = df_valid['final_score'].values
        y = df_valid['return_winsorized'].values

        # Calculate slope and intercept using least squares
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

        # Calculate predicted returns and residuals
        df_valid['predicted_return'] = slope * df_valid['final_score'] + intercept
        df_valid['residual'] = df_valid[return_col] - df_valid['predicted_return']

        # Calculate residual statistics
        residual_mean = df_valid['residual'].mean()
        residual_std = df_valid['residual'].std()

        print(f"Regression: Return = {slope:.4f} * Score + {intercept:.4f}")
        print(f"Residual Mean: {residual_mean:.4f}, Std: {residual_std:.4f}")

        # Define anomaly thresholds (2 standard deviations)
        positive_threshold = residual_mean + 2 * residual_std
        negative_threshold = residual_mean - 2 * residual_std

        # Identify anomalies
        df_valid['anomaly_type'] = 'Normal'
        df_valid.loc[df_valid['residual'] > positive_threshold, 'anomaly_type'] = 'Positive'
        df_valid.loc[df_valid['residual'] < negative_threshold, 'anomaly_type'] = 'Negative'

        # Focus on low score + high return (Positive Anomaly with low score)
        df_valid['is_special_positive'] = (
            (df_valid['anomaly_type'] == 'Positive') &
            (df_valid['final_score'] < df_valid['final_score'].quantile(0.3))
        )

        # Focus on high score + low return (Negative Anomaly with high score)
        df_valid['is_special_negative'] = (
            (df_valid['anomaly_type'] == 'Negative') &
            (df_valid['final_score'] > df_valid['final_score'].quantile(0.7))
        )

        # Count anomalies
        n_positive = (df_valid['anomaly_type'] == 'Positive').sum()
        n_negative = (df_valid['anomaly_type'] == 'Negative').sum()
        n_special_positive = df_valid['is_special_positive'].sum()
        n_special_negative = df_valid['is_special_negative'].sum()

        print(f"\nAnomaly Counts:")
        print(f"  Positive Anomalies (high return): {n_positive:,} ({n_positive/len(df_valid)*100:.2f}%)")
        print(f"  Negative Anomalies (low return): {n_negative:,} ({n_negative/len(df_valid)*100:.2f}%)")
        print(f"  Special Positive (low score + high return): {n_special_positive:,}")
        print(f"  Special Negative (high score + low return): {n_special_negative:,}")

        # Analyze Positive Anomalies (저점수 + 고수익률)
        print(f"\n--- Positive Anomalies (Low Score + High Return) ---")

        df_pos_anomaly = df_valid[df_valid['is_special_positive']].copy()

        if len(df_pos_anomaly) > 0:
            # Sector distribution
            print("\nSector Distribution:")
            sector_dist = df_pos_anomaly['sector'].value_counts()
            for sector, count in sector_dist.head(10).items():
                pct = count / len(df_pos_anomaly) * 100
                print(f"  {sector}: {count} ({pct:.1f}%)")

            # Top examples
            print("\nTop 10 Examples (Low Score + High Return):")
            top_pos = df_pos_anomaly.nsmallest(10, 'final_score')[
                ['symbol', 'date', 'final_score', return_col, 'residual', 'sector']
            ].copy()
            # Replace non-ASCII characters to avoid encoding issues
            for col in top_pos.select_dtypes(include=['object']).columns:
                top_pos[col] = top_pos[col].astype(str).str.replace('\xa0', ' ', regex=False)
            print(top_pos.to_string(index=False))

            # Save positive anomalies
            df_pos_anomaly['period'] = f'{period}d'
            anomaly_results.append(df_pos_anomaly)

        # Analyze Negative Anomalies (고점수 + 저수익률)
        print(f"\n--- Negative Anomalies (High Score + Low Return) ---")

        df_neg_anomaly = df_valid[df_valid['is_special_negative']].copy()

        if len(df_neg_anomaly) > 0:
            # Sector distribution
            print("\nSector Distribution:")
            sector_dist = df_neg_anomaly['sector'].value_counts()
            for sector, count in sector_dist.head(10).items():
                pct = count / len(df_neg_anomaly) * 100
                print(f"  {sector}: {count} ({pct:.1f}%)")

            # Top examples
            print("\nTop 10 Examples (High Score + Low Return):")
            top_neg = df_neg_anomaly.nlargest(10, 'final_score')[
                ['symbol', 'date', 'final_score', return_col, 'residual', 'sector']
            ].copy()
            # Replace non-ASCII characters to avoid encoding issues
            for col in top_neg.select_dtypes(include=['object']).columns:
                top_neg[col] = top_neg[col].astype(str).str.replace('\xa0', ' ', regex=False)
            print(top_neg.to_string(index=False))

            # Save negative anomalies
            df_neg_anomaly['period'] = f'{period}d'
            anomaly_results.append(df_neg_anomaly)

        # Analyze common characteristics of anomalies
        print(f"\n--- Anomaly Characteristics ---")

        if len(df_pos_anomaly) > 0:
            print("\nPositive Anomaly Characteristics:")
            print(f"  Avg Final Score: {df_pos_anomaly['final_score'].mean():.2f}")
            print(f"  Avg Return: {df_pos_anomaly[return_col].mean():.2f}%")
            print(f"  Median Market Cap: ${df_pos_anomaly['market_cap'].median()/1e9:.2f}B")

            # Exchange distribution
            exchange_dist = df_pos_anomaly['exchange'].value_counts(normalize=True) * 100
            print(f"  Exchange: " + ", ".join([f"{ex}: {pct:.1f}%" for ex, pct in exchange_dist.items()]))

        if len(df_neg_anomaly) > 0:
            print("\nNegative Anomaly Characteristics:")
            print(f"  Avg Final Score: {df_neg_anomaly['final_score'].mean():.2f}")
            print(f"  Avg Return: {df_neg_anomaly[return_col].mean():.2f}%")
            print(f"  Median Market Cap: ${df_neg_anomaly['market_cap'].median()/1e9:.2f}B")

            # Exchange distribution
            exchange_dist = df_neg_anomaly['exchange'].value_counts(normalize=True) * 100
            print(f"  Exchange: " + ", ".join([f"{ex}: {pct:.1f}%" for ex, pct in exchange_dist.items()]))

    # Save all anomalies
    if anomaly_results:
        df_all_anomalies = pd.concat(anomaly_results, ignore_index=True)

        # Save positive anomalies
        df_positive = df_all_anomalies[df_all_anomalies['is_special_positive']]
        if len(df_positive) > 0:
            output_path = os.path.join(OUTPUT_DIR, 'us_anomaly_positive.csv')
            df_positive.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\nSaved: {output_path} ({len(df_positive):,} records)")

        # Save negative anomalies
        df_negative = df_all_anomalies[df_all_anomalies['is_special_negative']]
        if len(df_negative) > 0:
            output_path = os.path.join(OUTPUT_DIR, 'us_anomaly_negative.csv')
            df_negative.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved: {output_path} ({len(df_negative):,} records)")

    # Anomaly Summary by Sector
    print("\n[2.6] Anomaly Summary by Sector")
    print("-"*100)

    if anomaly_results:
        df_all = pd.concat(anomaly_results, ignore_index=True)

        summary_results = []

        for sector in df_all['sector'].unique():
            df_sector = df_all[df_all['sector'] == sector]

            n_pos = df_sector['is_special_positive'].sum()
            n_neg = df_sector['is_special_negative'].sum()
            total = len(df_sector)

            if total > 0:
                summary_results.append({
                    'sector': sector,
                    'positive_anomalies': n_pos,
                    'negative_anomalies': n_neg,
                    'total_records': total,
                    'positive_rate': n_pos / total * 100,
                    'negative_rate': n_neg / total * 100
                })

        df_summary = pd.DataFrame(summary_results).sort_values('positive_rate', ascending=False)
        print(df_summary.to_string(index=False))

        output_path = os.path.join(OUTPUT_DIR, 'us_anomaly_summary_by_sector.csv')
        df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_path}")

    return anomaly_results


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution"""
    print("\n" + "="*100)
    print("US Quant - Conditional Strategy & Anomaly Detection Analysis")
    print("="*100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Connect to database
    print("\nConnecting to database...")
    conn = await connect_db()
    print("Connected")

    try:
        # Load base data
        df = await load_base_data(conn)

        # PART 1: Conditional Strategy Analysis
        await analyze_conditional_strategies(df)

        # PART 2: Anomaly Detection
        await analyze_anomalies(df)

        # Summary
        print("\n" + "="*100)
        print("All analyses completed successfully!")
        print("="*100)

        print(f"\nGenerated files in {OUTPUT_DIR}:")
        print("  [PART 1 - Conditional Strategy]")
        print("    - us_conditional_momentum_by_sector.csv")
        print("    - us_conditional_value_by_sector.csv")
        print("    - us_conditional_by_market_state.csv")
        print("    - us_conditional_usable_summary.csv")
        print("  [PART 2 - Anomaly Detection]")
        print("    - us_anomaly_positive.csv (Low Score + High Return)")
        print("    - us_anomaly_negative.csv (High Score + Low Return)")
        print("    - us_anomaly_summary_by_sector.csv")

    finally:
        await conn.close()
        print("\nDatabase connection closed")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    asyncio.run(main())
