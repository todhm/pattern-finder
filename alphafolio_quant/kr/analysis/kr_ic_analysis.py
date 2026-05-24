"""
Phase 3.7 IC Analysis - Comprehensive Analysis with Win Rate Optimization
=========================================================================

Analysis Features:
1. Factor Level IC Analysis (Value, Quality, Momentum, Growth)
2. Individual Value Strategy IC Analysis (V2, V3, V4, V13, V14, V21-V26, 11 strategies)
3. Rolling Window Validation (final_score + factors)
4. Decile Test (10-quantile analysis)
5. Theme/Sector Analysis
6. Factor-Level Advanced Analysis
7. Weight Comparison (Uniform vs Dynamic) - NEW
8. Parameter Surface Analysis (threshold optimization) - NEW
9. Accuracy and Win Rate Calculation - NEW
10. Market Timing Filter Effect Analysis - NEW

Feature Validation (Phase 3 구현 기능 검증):
11. Entry Timing Score Validation - 진입 타이밍 점수 검증
12. Stop Loss Validation - ATR 기반 손절 검증
13. Scenario Probability Calibration - 시나리오 확률 예측 검증
14. Buy Triggers Validation - 매수 트리거 검증
15. Position Size Validation - 변동성 기반 비중 검증

Phase 3.7 Changes:
- V21-V26 Korean-style value strategies added
- V1, V6-V12, V15-V20 disabled (negative IC)
- Market timing filters integrated
- Factor timing (aggressive cycle weights)

Execution: python phase3_7_ic_analysis.py
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import json
import os
import sys
import logging
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

# Logging level - WARNING and above only (hide INFO logs)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Add kr directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'kr'))

# Import modules
from kr.db_async import AsyncDatabaseManager
from kr.kr_value_factor import ValueFactorCalculator

load_dotenv()

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgresql+asyncpg://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')

# Output directory
OUTPUT_DIR = r'C:\project\alpha\quant\kr\result test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis dates (6 dates)
ANALYSIS_DATES = [
    '2025-08-04', '2025-08-11', '2025-08-19', '2025-08-22', '2025-08-07', '2025-08-27', '2025-09-22', '2025-09-25',
    '2025-09-01', '2025-09-05', '2025-09-09', '2025-09-12', '2025-09-16'
]

# Return periods to analyze
RETURN_PERIODS = [3, 15, 30, 45, 60]  # days

# Parallel processing settings
MAX_CONCURRENT = 40    # Maximum concurrent stock calculations
BATCH_SIZE = 100       # Batch size for progress tracking

# Parameter surface grid
BUY_THRESHOLD_RANGE = range(65, 86, 5)   # [65, 70, 75, 80, 85]
SELL_THRESHOLD_RANGE = range(35, 56, 5)  # [35, 40, 45, 50, 55]

# Phase 3.7 Value Strategies (11 strategies)
VALUE_STRATEGIES = {
    'V2_Magic_Formula': 'calculate_v2',
    'V3_Cash_Flow_Sustainability': 'calculate_v3',
    'V4_Sustainable_Dividend': 'calculate_v4',
    'V13_Magic_Formula_Enhanced': 'calculate_v13',
    'V14_Dividend_Growth': 'calculate_v14',
    'V21_Korea_Adjusted_PBR': 'calculate_v21_korea_adjusted_pbr',
    'V22_Quality_Dividend': 'calculate_v22_quality_dividend',
    'V23_Asset_Growth_Value': 'calculate_v23_asset_growth_value',
    'V24_Operating_Leverage': 'calculate_v24_operating_leverage',
    'V25_Cash_Rich_Undervalued': 'calculate_v25_cash_rich_undervalued',
    'V26_Smart_Money_Value': 'calculate_v26_smart_money_value',
}

# Market timing filter thresholds (Phase 3.7)
REGIME_THRESHOLDS = {
    'PANIC': (85, 50),
    'FEAR': (75, 45),
    'NEUTRAL': (70, 40),
    'GREED': (70, 45),
    'OVERHEATED': (80, 55)
}


# ============================================================================
# PART 1: DATA COLLECTION & PREPARATION
# ============================================================================

async def collect_data(pool):
    """Collect data from kr_stock_grade + kr_intraday_total"""

    print("=" * 100)
    print("STEP 1: Data Collection")
    print("=" * 100)

    # Convert dates to date objects
    date_objs = [datetime.strptime(d, '%Y-%m-%d').date() for d in ANALYSIS_DATES]

    async with pool.acquire() as conn:
        # Get all kr_stock_grade data for analysis dates
        print("\n[1/3] Querying kr_stock_grade...")
        grade_query = """
            SELECT
                symbol, stock_name, date,
                final_score, final_grade,
                value_score, quality_score, momentum_score, growth_score,
                sector_rotation_score, sector_momentum, sector_rank, sector_percentile,
                factor_combination_bonus,
                confidence_score, market_state,
                -- Feature Validation용 추가 컬럼
                entry_timing_score,
                stop_loss_pct, take_profit_pct, risk_reward_ratio,
                position_size_pct, atr_pct,
                scenario_bullish_prob, scenario_sideways_prob, scenario_bearish_prob,
                scenario_bullish_return, scenario_sideways_return, scenario_bearish_return,
                buy_triggers, sell_triggers, hold_triggers,
                score_trend_2w, price_position_52w,
                beta, volatility_annual
            FROM kr_stock_grade
            WHERE date = ANY($1::date[])
            ORDER BY date, symbol
        """
        grade_data = await conn.fetch(grade_query, date_objs)
        df_grade = pd.DataFrame(grade_data, columns=[
            'symbol', 'stock_name', 'date',
            'final_score', 'final_grade',
            'value_score', 'quality_score', 'momentum_score', 'growth_score',
            'sector_rotation_score', 'sector_momentum', 'sector_rank', 'sector_percentile',
            'factor_combination_bonus',
            'confidence_score', 'market_state',
            # Feature Validation용 추가 컬럼
            'entry_timing_score',
            'stop_loss_pct', 'take_profit_pct', 'risk_reward_ratio',
            'position_size_pct', 'atr_pct',
            'scenario_bullish_prob', 'scenario_sideways_prob', 'scenario_bearish_prob',
            'scenario_bullish_return', 'scenario_sideways_return', 'scenario_bearish_return',
            'buy_triggers', 'sell_triggers', 'hold_triggers',
            'score_trend_2w', 'price_position_52w',
            'beta', 'volatility_annual'
        ])

        # Convert Decimal columns to float
        numeric_cols = ['final_score', 'value_score', 'quality_score', 'momentum_score',
                        'growth_score', 'sector_rotation_score', 'sector_momentum',
                        'sector_rank', 'sector_percentile', 'factor_combination_bonus',
                        'confidence_score', 'entry_timing_score', 'stop_loss_pct',
                        'take_profit_pct', 'risk_reward_ratio', 'position_size_pct',
                        'atr_pct', 'scenario_bullish_prob', 'scenario_sideways_prob',
                        'scenario_bearish_prob', 'score_trend_2w', 'price_position_52w',
                        'beta', 'volatility_annual']
        for col in numeric_cols:
            if col in df_grade.columns:
                df_grade[col] = pd.to_numeric(df_grade[col], errors='coerce')

        print(f"   kr_stock_grade: {len(df_grade):,} records")

        # Get kr_stock_detail for theme/industry
        print("\n[2/3] Querying kr_stock_detail...")
        detail_query = """
            SELECT symbol, theme, industry, exchange
            FROM kr_stock_detail
        """
        detail_data = await conn.fetch(detail_query)
        df_detail = pd.DataFrame(detail_data, columns=['symbol', 'theme', 'industry', 'exchange'])
        print(f"   kr_stock_detail: {len(df_detail):,} records")

        # Merge theme/industry info
        df_grade = df_grade.merge(df_detail, on='symbol', how='left')

        # Get price data for return calculation
        print("\n[3/3] Querying kr_intraday_total price data...")

        # Calculate date range for price data (need +90 days buffer)
        min_date = min(date_objs)
        max_date = max(date_objs) + timedelta(days=120)

        price_query = """
            SELECT symbol, date, close
            FROM kr_intraday_total
            WHERE date >= $1 AND date <= $2
            ORDER BY symbol, date
        """
        price_data = await conn.fetch(price_query, min_date, max_date)
        df_price = pd.DataFrame(price_data, columns=['symbol', 'date', 'close'])
        print(f"   kr_intraday_total: {len(df_price):,} records")

    return df_grade, df_price


def calculate_forward_returns(df_grade, df_price):
    """Calculate forward returns (optimized version)"""

    print("\n" + "=" * 100)
    print("STEP 2: Forward Return Calculation")
    print("=" * 100)

    # Pivot price data for faster lookups
    df_price_pivot = df_price.pivot(index='date', columns='symbol', values='close')

    results_dict = {}

    print(f"\nProcessing {len(df_grade):,} symbol-date combinations...")

    for idx, row in df_grade.iterrows():
        symbol = row['symbol']
        date_t = row['date']
        key = (symbol, date_t)

        if symbol not in df_price_pivot.columns:
            continue

        try:
            price_t = df_price_pivot.loc[date_t, symbol]
        except KeyError:
            continue

        if pd.isna(price_t):
            continue

        result = {
            'symbol': symbol,
            'date': date_t,
            'close_t': price_t
        }

        future_dates = df_price_pivot.index[df_price_pivot.index > date_t]

        for period in RETURN_PERIODS:
            if len(future_dates) < period:
                result[f'return_{period}d'] = None
                result[f'close_t{period}'] = None
                continue

            date_future = future_dates[period - 1]
            price_future = df_price_pivot.loc[date_future, symbol]

            if pd.isna(price_future):
                result[f'return_{period}d'] = None
                result[f'close_t{period}'] = None
            else:
                ret = (price_future / price_t - 1) * 100
                result[f'return_{period}d'] = ret
                result[f'close_t{period}'] = price_future

        results_dict[key] = result

        if (idx + 1) % 5000 == 0:
            print(f"  Progress: {idx + 1:,}/{len(df_grade):,} ({(idx+1)/len(df_grade)*100:.1f}%)")

    df_returns = pd.DataFrame(list(results_dict.values()))
    df_analysis = df_grade.merge(df_returns, on=['symbol', 'date'], how='left')

    print(f"\nForward returns calculated: {len(df_analysis):,} records")
    for period in RETURN_PERIODS:
        col_name = f'return_{period}d'
        if col_name in df_analysis.columns:
            count = df_analysis[col_name].notna().sum()
            print(f"   {col_name}: {count:,} samples")

    return df_analysis


# ============================================================================
# PART 2: IC ANALYSIS
# ============================================================================

def calculate_ic(df, score_col, return_cols):
    """Calculate IC (Information Coefficient)"""

    results = {}

    for ret_col in return_cols:
        valid_data = df[[score_col, ret_col]].dropna()

        if len(valid_data) < 10:
            results[ret_col] = {
                'pearson_ic': None,
                'spearman_ic': None,
                'n_samples': len(valid_data)
            }
            continue

        score_values = valid_data[score_col].astype(float).values
        return_values = valid_data[ret_col].astype(float).values

        pearson_ic, pearson_p = pearsonr(score_values, return_values)
        spearman_ic, spearman_p = spearmanr(score_values, return_values)

        results[ret_col] = {
            'pearson_ic': pearson_ic,
            'pearson_p': pearson_p,
            'spearman_ic': spearman_ic,
            'spearman_p': spearman_p,
            'n_samples': len(valid_data)
        }

    return results


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
    elif decile >= 8:  # D8-D10: Buy signal - non-negative return is correct
        return (returns >= 0).mean() * 100
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


def ic_analysis_factor_level(df):
    """STEP 3A: Factor Level IC Analysis"""

    print("\n" + "=" * 100)
    print("STEP 3A: Factor Level IC Analysis")
    print("=" * 100)

    return_cols = [f'return_{period}d' for period in RETURN_PERIODS]
    ic_results = {}

    score_columns = {
        'final_score': 'Final Score',
        'value_score': 'Value Factor',
        'quality_score': 'Quality Factor',
        'momentum_score': 'Momentum Factor',
        'growth_score': 'Growth Factor',
        'sector_rotation_score': 'Sector Rotation',
        'factor_combination_bonus': 'Factor Combination',
    }

    all_results = []

    for score_col, score_name in score_columns.items():
        if score_col not in df.columns:
            print(f"\n[WARNING] {score_name} ({score_col}) not found. Skipping...")
            continue

        ic_results[score_col] = calculate_ic(df, score_col, return_cols)

        print(f"\n{score_name} ({score_col}):")
        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'
            if ret_col in ic_results[score_col]:
                result = ic_results[score_col][ret_col]
                if result['pearson_ic'] is not None and result['spearman_ic'] is not None:
                    print(f"  {period:2}d: Pearson={result['pearson_ic']:>7.4f}, "
                          f"Spearman={result['spearman_ic']:>7.4f}, n={result['n_samples']:>6,}")

                    all_results.append({
                        'factor': score_name,
                        'period': f'{period}d',
                        'pearson_ic': result['pearson_ic'],
                        'spearman_ic': result['spearman_ic'],
                        'n_samples': result['n_samples']
                    })

    # Save results
    df_results = pd.DataFrame(all_results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_factor_ic.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return ic_results


async def process_single_stock_strategies(row, db_manager, semaphore):
    """Calculate all strategies for a single stock (parallel processing)"""
    async with semaphore:
        symbol = row['symbol']
        analysis_date = row['date']

        try:
            calc = ValueFactorCalculator(
                symbol=symbol,
                db_manager=db_manager,
                market_state='etc',
                analysis_date=analysis_date
            )

            scores = {'symbol': symbol, 'date': analysis_date}

            for strategy_name, method_name in VALUE_STRATEGIES.items():
                try:
                    method = getattr(calc, method_name)
                    score = await method()
                    scores[strategy_name] = score if score is not None else 0
                except Exception:
                    scores[strategy_name] = 0

            return scores

        except Exception:
            scores = {'symbol': symbol, 'date': analysis_date}
            for strategy_name in VALUE_STRATEGIES.keys():
                scores[strategy_name] = 0
            return scores


async def ic_analysis_value_strategies(db_manager, df_returns, use_sample=False):
    """STEP 3B: Individual Value Strategy IC Analysis (Phase 3.7)"""

    print("\n" + "=" * 100)
    print("STEP 3B: Value Strategy IC Analysis (Phase 3.7 - 11 strategies)")
    print("=" * 100)
    print(f"Strategies: {', '.join(VALUE_STRATEGIES.keys())}")
    print(f"Max concurrent: {MAX_CONCURRENT}")

    return_cols = ['symbol', 'date'] + [f'return_{p}d' for p in RETURN_PERIODS] + ['theme', 'stock_name']
    return_cols = [col for col in return_cols if col in df_returns.columns]
    df_returns_work = df_returns[return_cols].copy()

    if use_sample:
        print(f"\n[WARNING] Sample mode: {min(500, len(df_returns_work))} records")
        df_returns_work = df_returns_work.sample(n=min(500, len(df_returns_work)), random_state=42)
    else:
        print(f"\nFull analysis: {len(df_returns_work):,} records")

    df_returns_work['date'] = pd.to_datetime(df_returns_work['date']).dt.date

    print(f"\nCalculating {len(VALUE_STRATEGIES)} strategies for {len(df_returns_work):,} samples...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = []
    for idx, row in df_returns_work.iterrows():
        task = process_single_stock_strategies(row, db_manager, semaphore)
        tasks.append(task)

    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    all_results = []

    print(f"\nBatch processing: {total_batches} batches (batch size: {BATCH_SIZE})")
    start_time = datetime.now()

    for i in range(0, len(tasks), BATCH_SIZE):
        batch_tasks = tasks[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        batch_results = await asyncio.gather(*batch_tasks)
        all_results.extend(batch_results)

        elapsed = (datetime.now() - start_time).total_seconds()
        processed = len(all_results)
        progress_pct = processed / len(tasks) * 100

        if processed > 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Batch {batch_num}/{total_batches} complete: {processed:,} processed")

    print(f"\nCalculation complete: {len(all_results):,} processed ({(datetime.now()-start_time).total_seconds()/60:.1f} min)")

    df_strategies = pd.DataFrame(all_results)
    df = pd.merge(df_strategies, df_returns_work, on=['symbol', 'date'], how='inner')
    print(f"Merged: {len(df):,} records")

    # Analyze each strategy
    results = []

    print("\n" + "-" * 100)
    print(f"{'Strategy':<35} {'Pearson IC':>12} {'Spearman IC':>12} {'Samples':>10} {'Status'}")
    print("-" * 100)

    for strategy in VALUE_STRATEGIES.keys():
        longest_period = max(RETURN_PERIODS)
        longest_ret_col = f'return_{longest_period}d'

        if longest_ret_col not in df.columns:
            continue

        ic = calculate_ic(df, strategy, [longest_ret_col])[longest_ret_col]

        if ic['spearman_ic'] is None:
            status = '[?] No Data'
            spearman_val = 0
        else:
            spearman_val = ic['spearman_ic']
            if spearman_val < -0.05:
                status = '[X] Strong Negative'
            elif spearman_val < 0:
                status = '[!] Weak Negative'
            elif spearman_val < 0.05:
                status = '[-] Neutral'
            elif spearman_val < 0.10:
                status = '[+] Positive'
            else:
                status = '[*] Strong Positive'

        pearson_val = ic['pearson_ic'] if ic['pearson_ic'] is not None else 0

        print(f"{strategy:<35} {pearson_val:>12.4f} {spearman_val:>12.4f} {ic['n_samples']:>10,} {status}")

        results.append({
            'strategy': strategy,
            'period': f'{longest_period}d',
            'pearson_ic': pearson_val,
            'spearman_ic': spearman_val,
            'n_samples': ic['n_samples'],
            'status': status.replace('[', '').replace(']', '').split()[0]
        })

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_value_strategies_ic.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    output_file2 = os.path.join(OUTPUT_DIR, 'phase3_7_value_strategies_calculated.csv')
    df.to_csv(output_file2, index=False)
    print(f"Saved: {output_file2}")

    return df_results, df


# ============================================================================
# PART 3: ROLLING WINDOW & DECILE TEST
# ============================================================================

def rolling_window_analysis(df):
    """STEP 4: Rolling Window Validation"""

    print("\n" + "=" * 100)
    print("STEP 4: Rolling Window Validation (Date-wise IC)")
    print("=" * 100)

    return_cols = [f'return_{period}d' for period in RETURN_PERIODS]
    dates = sorted(df['date'].unique())

    results = []

    print(f"\n{'Date':<15}", end='')
    for period in RETURN_PERIODS:
        print(f"{period}d IC".rjust(10), end='')
    print(f"{'Samples':>10}")
    print("-" * 100)

    for analysis_date in dates:
        date_data = df[df['date'] == analysis_date]
        ic_result = calculate_ic(date_data, 'final_score', return_cols)

        result_row = {'date': analysis_date, 'n_samples': len(date_data)}

        print(f"{str(analysis_date):<15}", end='')
        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'
            if ret_col in ic_result and ic_result[ret_col]['spearman_ic'] is not None:
                ic_val = ic_result[ret_col]['spearman_ic']
                result_row[f'ic_{period}d'] = ic_val
                print(f"{ic_val:>10.4f}", end='')
            else:
                result_row[f'ic_{period}d'] = None
                print(f"{'N/A':>10}", end='')

        print(f"{len(date_data):>10,}")
        results.append(result_row)

    df_results = pd.DataFrame(results)

    print("\n" + "-" * 100)
    print("Statistics:")
    for period in RETURN_PERIODS:
        ic_col = f'ic_{period}d'
        if ic_col in df_results.columns:
            valid_ic = df_results[ic_col].dropna()
            if len(valid_ic) > 0:
                mean_ic = valid_ic.mean()
                std_ic = valid_ic.std()
                print(f"  {period:2}d IC: Mean={mean_ic:>7.4f}, StdDev={std_ic:>7.4f}")

    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_rolling_ic.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def decile_test(df):
    """STEP 5: Decile Test (10-quantile analysis) with Directional Accuracy"""

    print("\n" + "=" * 100)
    print("STEP 5: Decile Test (with Directional Accuracy)")
    print("=" * 100)

    decile_results = []

    for period in RETURN_PERIODS:
        ret_col = f'return_{period}d'

        valid_data = df[['final_score', ret_col]].dropna().copy()
        valid_data['final_score'] = valid_data['final_score'].astype(float)
        valid_data[ret_col] = valid_data[ret_col].astype(float)

        if len(valid_data) < 100:
            continue

        valid_data['decile'] = pd.qcut(valid_data['final_score'], q=10, labels=False, duplicates='drop') + 1

        print(f"\n{period}-day Return:")
        print("-" * 120)
        print(f"{'Decile':>8} {'Score Range':>20} {'Count':>8} {'Avg Return':>12} {'Median':>10} "
              f"{'Win Rate':>10} {'Dir Acc':>10} {'Signal':>8}")
        print("-" * 120)

        for decile in sorted(valid_data['decile'].unique()):
            decile_data = valid_data[valid_data['decile'] == decile]

            score_min = decile_data['final_score'].min()
            score_max = decile_data['final_score'].max()
            count = len(decile_data)
            avg_return = decile_data[ret_col].mean()
            median_return = decile_data[ret_col].median()
            std_return = decile_data[ret_col].std()
            win_rate = (decile_data[ret_col] > 0).sum() / count * 100

            # Directional Accuracy (NEW)
            returns_array = decile_data[ret_col].values
            dir_accuracy = calculate_directional_accuracy(decile, returns_array)

            # Signal type
            signal = get_signal_type(decile)

            print(f"{decile:>8} {score_min:>8.1f}-{score_max:>8.1f} {count:>8,} "
                  f"{avg_return:>+11.2f}% {median_return:>+9.2f}% "
                  f"{win_rate:>9.1f}% {dir_accuracy:>9.1f}% {signal:>8}")

            decile_results.append({
                'period': period,
                'decile': decile,
                'score_min': score_min,
                'score_max': score_max,
                'count': count,
                'avg_return': avg_return,
                'median_return': median_return,
                'std_return': std_return,
                'win_rate': win_rate,
                'directional_accuracy': dir_accuracy,
                'signal': signal
            })

        # Long-Short spread
        if len(valid_data['decile'].unique()) >= 2:
            top_decile = valid_data[valid_data['decile'] == valid_data['decile'].max()][ret_col].mean()
            bottom_decile = valid_data[valid_data['decile'] == valid_data['decile'].min()][ret_col].mean()
            spread = top_decile - bottom_decile
            print("-" * 120)
            print(f"Long-Short Spread (Top - Bottom): {spread:+.2f}%")

            # Buy vs Sell accuracy summary
            buy_deciles = valid_data[valid_data['decile'] >= 8]
            sell_deciles = valid_data[valid_data['decile'] <= 3]
            buy_accuracy = (buy_deciles[ret_col] > 0).mean() * 100 if len(buy_deciles) > 0 else 0
            sell_accuracy = (sell_deciles[ret_col] < 0).mean() * 100 if len(sell_deciles) > 0 else 0
            print(f"Buy Signal Accuracy (D8-D10): {buy_accuracy:.1f}%  |  Sell Signal Accuracy (D1-D3): {sell_accuracy:.1f}%")

    df_results = pd.DataFrame(decile_results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_decile_test.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# PART 4: NEW ANALYSIS FUNCTIONS (Phase 3.7)
# ============================================================================

def weight_comparison_analysis(df):
    """
    STEP 6: Weight Comparison Analysis (Uniform vs Dynamic)
    Compare IC with uniform weights (25% each) vs dynamic weights
    """

    print("\n" + "=" * 100)
    print("STEP 6: Weight Comparison Analysis (Uniform vs Dynamic)")
    print("=" * 100)

    results = []

    # Calculate uniform weighted score
    factor_cols = ['value_score', 'quality_score', 'momentum_score', 'growth_score']
    if not all(col in df.columns for col in factor_cols):
        print("[WARNING] Factor columns missing. Skipping weight comparison...")
        return pd.DataFrame()

    # Convert decimal columns to float for arithmetic operations
    for col in factor_cols:
        df[col] = df[col].astype(float)

    df['uniform_score'] = (
        df['value_score'] * 0.25 +
        df['quality_score'] * 0.25 +
        df['momentum_score'] * 0.25 +
        df['growth_score'] * 0.25
    )

    # Compare IC
    return_cols = [f'return_{period}d' for period in RETURN_PERIODS]

    print("\nUniform Weights (25% each) vs Dynamic Weights (final_score):")
    print("-" * 100)
    print(f"{'Period':<10} {'Uniform IC':>15} {'Dynamic IC':>15} {'Difference':>15} {'Better'}")
    print("-" * 100)

    for period in RETURN_PERIODS:
        ret_col = f'return_{period}d'

        uniform_ic = calculate_ic(df, 'uniform_score', [ret_col])[ret_col]
        dynamic_ic = calculate_ic(df, 'final_score', [ret_col])[ret_col]

        uniform_val = uniform_ic['spearman_ic'] if uniform_ic['spearman_ic'] else 0
        dynamic_val = dynamic_ic['spearman_ic'] if dynamic_ic['spearman_ic'] else 0
        diff = dynamic_val - uniform_val
        better = 'Dynamic' if diff > 0 else 'Uniform'

        print(f"{period}d{'':<7} {uniform_val:>15.4f} {dynamic_val:>15.4f} {diff:>+15.4f} {better}")

        results.append({
            'period': f'{period}d',
            'uniform_ic': uniform_val,
            'dynamic_ic': dynamic_val,
            'difference': diff,
            'better': better,
            'n_samples': uniform_ic['n_samples']
        })

    # Market state analysis
    if 'market_state' in df.columns:
        print("\n" + "-" * 100)
        print("IC by Market State:")
        print("-" * 100)

        market_states = df['market_state'].dropna().unique()
        longest_ret = f'return_{max(RETURN_PERIODS)}d'

        for state in sorted(market_states)[:10]:  # Top 10 states
            state_data = df[df['market_state'] == state]
            if len(state_data) < 30:
                continue

            uniform_ic = calculate_ic(state_data, 'uniform_score', [longest_ret])[longest_ret]
            dynamic_ic = calculate_ic(state_data, 'final_score', [longest_ret])[longest_ret]

            uniform_val = uniform_ic['spearman_ic'] if uniform_ic['spearman_ic'] else 0
            dynamic_val = dynamic_ic['spearman_ic'] if dynamic_ic['spearman_ic'] else 0

            print(f"  {state:<35} Uniform: {uniform_val:>7.4f}, Dynamic: {dynamic_val:>7.4f}, "
                  f"Diff: {dynamic_val - uniform_val:>+7.4f}, n={len(state_data):,}")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_weight_comparison.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def parameter_surface_analysis(df):
    """
    STEP 7: Parameter Surface Analysis (Threshold Optimization)
    Grid search for optimal buy/sell thresholds
    """

    print("\n" + "=" * 100)
    print("STEP 7: Parameter Surface Analysis (Threshold Optimization)")
    print("=" * 100)

    results = []

    # Use 30-day return for optimization
    ret_col = 'return_30d'
    if ret_col not in df.columns:
        print(f"[WARNING] {ret_col} not found. Skipping parameter surface...")
        return pd.DataFrame()

    valid_data = df[['final_score', ret_col]].dropna()
    print(f"\nOptimizing on {len(valid_data):,} samples using {ret_col}")
    print(f"Buy threshold range: {list(BUY_THRESHOLD_RANGE)}")
    print(f"Sell threshold range: {list(SELL_THRESHOLD_RANGE)}")

    print("\n" + "-" * 100)
    print(f"{'Buy Th':<10} {'Sell Th':<10} {'Buy Count':>12} {'Sell Count':>12} "
          f"{'Buy WinRate':>12} {'Sell WinRate':>12} {'Total Acc':>12}")
    print("-" * 100)

    best_result = None
    best_total_acc = 0

    for buy_th in BUY_THRESHOLD_RANGE:
        for sell_th in SELL_THRESHOLD_RANGE:
            # Buy signals
            buy_data = valid_data[valid_data['final_score'] >= buy_th]
            buy_count = len(buy_data)
            buy_win_rate = (buy_data[ret_col] > 0).mean() * 100 if buy_count > 0 else 0

            # Sell signals
            sell_data = valid_data[valid_data['final_score'] < sell_th]
            sell_count = len(sell_data)
            sell_win_rate = (sell_data[ret_col] < 0).mean() * 100 if sell_count > 0 else 0

            # Total accuracy (buy correct + sell correct) / total signals
            total_signals = buy_count + sell_count
            if total_signals > 0:
                correct_signals = (buy_data[ret_col] > 0).sum() + (sell_data[ret_col] < 0).sum()
                total_acc = correct_signals / total_signals * 100
            else:
                total_acc = 0

            print(f"{buy_th:<10} {sell_th:<10} {buy_count:>12,} {sell_count:>12,} "
                  f"{buy_win_rate:>11.1f}% {sell_win_rate:>11.1f}% {total_acc:>11.1f}%")

            results.append({
                'buy_threshold': buy_th,
                'sell_threshold': sell_th,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'buy_win_rate': buy_win_rate,
                'sell_win_rate': sell_win_rate,
                'total_accuracy': total_acc
            })

            if total_acc > best_total_acc:
                best_total_acc = total_acc
                best_result = (buy_th, sell_th, total_acc)

    print("\n" + "-" * 100)
    if best_result:
        print(f"Optimal Thresholds: Buy >= {best_result[0]}, Sell < {best_result[1]}")
        print(f"Best Total Accuracy: {best_result[2]:.1f}%")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_parameter_surface.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def accuracy_winrate_analysis(df):
    """
    STEP 8: Accuracy and Win Rate Calculation
    Comprehensive performance metrics
    """

    print("\n" + "=" * 100)
    print("STEP 8: Accuracy and Win Rate Analysis")
    print("=" * 100)

    results = []

    # Grade mapping for comparison
    grade_to_expected = {
        'strong_buy': 'positive',      # Expect positive return
        'buy': 'positive',             # Expect positive return
        'neutral': 'neutral',           # No strong expectation
        'sell': 'negative',             # Expect negative return
        'strong_sell': 'negative'       # Expect negative return
    }

    for period in RETURN_PERIODS:
        ret_col = f'return_{period}d'

        if ret_col not in df.columns:
            continue

        valid_data = df[['final_score', 'final_grade', ret_col]].dropna()

        print(f"\n{period}-day Performance:")
        print("-" * 100)

        # Overall metrics
        total_count = len(valid_data)
        positive_returns = (valid_data[ret_col] > 0).sum()
        negative_returns = (valid_data[ret_col] < 0).sum()

        overall_win_rate = positive_returns / total_count * 100 if total_count > 0 else 0

        print(f"Total samples: {total_count:,}")
        print(f"Positive returns: {positive_returns:,} ({positive_returns/total_count*100:.1f}%)")
        print(f"Negative returns: {negative_returns:,} ({negative_returns/total_count*100:.1f}%)")
        print(f"Overall market win rate: {overall_win_rate:.1f}%")

        # Grade-wise analysis
        print(f"\n{'Grade':<15} {'Count':>10} {'Avg Return':>12} {'Win Rate':>12} {'Accuracy':>12}")
        print("-" * 70)

        grades = valid_data['final_grade'].unique()
        for grade in sorted(grades, key=lambda x: str(x)):
            grade_data = valid_data[valid_data['final_grade'] == grade]
            count = len(grade_data)
            avg_return = grade_data[ret_col].mean()
            win_rate = (grade_data[ret_col] > 0).mean() * 100

            # Accuracy: Did the prediction direction match?
            if grade in ['강력 매수', '매수', '매수 고려']:
                accuracy = (grade_data[ret_col] > 0).mean() * 100
            elif grade in ['매도', '매도 고려', '강력 매도']:
                accuracy = (grade_data[ret_col] < 0).mean() * 100
            else:
                accuracy = 50.0  # 중립 has no directional prediction

            print(f"{str(grade):<15} {count:>10,} {avg_return:>+11.2f}% {win_rate:>11.1f}% {accuracy:>11.1f}%")

            results.append({
                'period': f'{period}d',
                'grade': grade,
                'count': count,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'accuracy': accuracy
            })

        # Buy signal performance (high score)
        buy_threshold = 70
        buy_signals = valid_data[valid_data['final_score'] >= buy_threshold]
        if len(buy_signals) > 0:
            buy_win_rate = (buy_signals[ret_col] > 0).mean() * 100
            buy_avg_return = buy_signals[ret_col].mean()
            print(f"\nBuy Signals (Score >= {buy_threshold}):")
            print(f"  Count: {len(buy_signals):,}")
            print(f"  Win Rate: {buy_win_rate:.1f}%")
            print(f"  Avg Return: {buy_avg_return:+.2f}%")

        # Sell signal performance (low score)
        sell_threshold = 40
        sell_signals = valid_data[valid_data['final_score'] < sell_threshold]
        if len(sell_signals) > 0:
            sell_win_rate = (sell_signals[ret_col] < 0).mean() * 100
            sell_avg_return = sell_signals[ret_col].mean()
            print(f"\nSell Signals (Score < {sell_threshold}):")
            print(f"  Count: {len(sell_signals):,}")
            print(f"  Correct Rate (negative return): {sell_win_rate:.1f}%")
            print(f"  Avg Return: {sell_avg_return:+.2f}%")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_accuracy_winrate.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def timing_filter_analysis(df):
    """
    STEP 9: Market Timing Filter Effect Analysis
    Compare performance with/without Phase 3.7 filters
    """

    print("\n" + "=" * 100)
    print("STEP 9: Market Timing Filter Effect Analysis")
    print("=" * 100)

    results = []

    # Check required columns
    required_cols = ['final_score', 'momentum_score', 'return_30d']
    if not all(col in df.columns for col in required_cols):
        print("[WARNING] Required columns missing. Skipping timing filter analysis...")
        return pd.DataFrame()

    ret_col = 'return_30d'
    valid_data = df[['final_score', 'momentum_score', ret_col]].dropna()

    print(f"\nAnalyzing {len(valid_data):,} samples")

    # Scenario 1: No filter (original logic)
    print("\n" + "-" * 100)
    print("Scenario 1: No Filter (Score >= 70 = Buy)")
    print("-" * 100)

    buy_no_filter = valid_data[valid_data['final_score'] >= 70]
    if len(buy_no_filter) > 0:
        win_rate_no_filter = (buy_no_filter[ret_col] > 0).mean() * 100
        avg_return_no_filter = buy_no_filter[ret_col].mean()
        print(f"  Buy signals: {len(buy_no_filter):,}")
        print(f"  Win rate: {win_rate_no_filter:.1f}%")
        print(f"  Avg return: {avg_return_no_filter:+.2f}%")
    else:
        win_rate_no_filter = 0
        avg_return_no_filter = 0

    # Scenario 2: Momentum filter only
    print("\n" + "-" * 100)
    print("Scenario 2: Momentum Filter (Score >= 70 AND Momentum >= 35)")
    print("-" * 100)

    buy_momentum = valid_data[
        (valid_data['final_score'] >= 70) &
        (valid_data['momentum_score'] >= 35)
    ]
    if len(buy_momentum) > 0:
        win_rate_momentum = (buy_momentum[ret_col] > 0).mean() * 100
        avg_return_momentum = buy_momentum[ret_col].mean()
        print(f"  Buy signals: {len(buy_momentum):,}")
        print(f"  Win rate: {win_rate_momentum:.1f}%")
        print(f"  Avg return: {avg_return_momentum:+.2f}%")
        print(f"  Win rate improvement: {win_rate_momentum - win_rate_no_filter:+.1f}%p")
    else:
        win_rate_momentum = 0
        avg_return_momentum = 0

    # Scenario 3: Strict filter (Score >= 75 AND Momentum >= 40)
    print("\n" + "-" * 100)
    print("Scenario 3: Strict Filter (Score >= 75 AND Momentum >= 40)")
    print("-" * 100)

    buy_strict = valid_data[
        (valid_data['final_score'] >= 75) &
        (valid_data['momentum_score'] >= 40)
    ]
    if len(buy_strict) > 0:
        win_rate_strict = (buy_strict[ret_col] > 0).mean() * 100
        avg_return_strict = buy_strict[ret_col].mean()
        print(f"  Buy signals: {len(buy_strict):,}")
        print(f"  Win rate: {win_rate_strict:.1f}%")
        print(f"  Avg return: {avg_return_strict:+.2f}%")
        print(f"  Win rate improvement: {win_rate_strict - win_rate_no_filter:+.1f}%p")
    else:
        win_rate_strict = 0
        avg_return_strict = 0

    # Summary
    print("\n" + "=" * 100)
    print("Summary: Filter Effect")
    print("=" * 100)
    print(f"{'Scenario':<30} {'Signals':>12} {'Win Rate':>12} {'Avg Return':>12}")
    print("-" * 70)
    print(f"{'No Filter':<30} {len(buy_no_filter):>12,} {win_rate_no_filter:>11.1f}% {avg_return_no_filter:>+11.2f}%")
    print(f"{'Momentum Filter':<30} {len(buy_momentum):>12,} {win_rate_momentum:>11.1f}% {avg_return_momentum:>+11.2f}%")
    print(f"{'Strict Filter':<30} {len(buy_strict):>12,} {win_rate_strict:>11.1f}% {avg_return_strict:>+11.2f}%")

    results = [
        {'scenario': 'No Filter', 'signals': len(buy_no_filter),
         'win_rate': win_rate_no_filter, 'avg_return': avg_return_no_filter},
        {'scenario': 'Momentum Filter', 'signals': len(buy_momentum),
         'win_rate': win_rate_momentum, 'avg_return': avg_return_momentum},
        {'scenario': 'Strict Filter', 'signals': len(buy_strict),
         'win_rate': win_rate_strict, 'avg_return': avg_return_strict},
    ]

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_timing_filter.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# PART 4.5: FINAL GRADE OPTIMIZATION ANALYSIS (NEW)
# ============================================================================

def final_score_granular_analysis(df):
    """
    STEP 9A: Final Score Granular Analysis (5-point bins)

    Analyze actual performance by fine-grained final_score bins to determine
    optimal grade thresholds.
    """

    print("\n" + "=" * 100)
    print("STEP 9A: Final Score Granular Analysis (5-point bins)")
    print("=" * 100)

    results = []

    # Use 30d and 45d returns
    ret_cols = ['return_30d', 'return_45d']

    for ret_col in ret_cols:
        if ret_col not in df.columns:
            continue

        period = ret_col.replace('return_', '').replace('d', '')

        valid_data = df[['final_score', ret_col]].dropna().copy()
        valid_data['final_score'] = valid_data['final_score'].astype(float)
        valid_data[ret_col] = valid_data[ret_col].astype(float)

        if len(valid_data) < 100:
            continue

        print(f"\n[{period}d Return Analysis]")
        print("-" * 120)
        print(f"{'Score Range':>15} {'Count':>8} {'Avg Return':>12} {'Median':>10} {'StdDev':>10} "
              f"{'Win Rate':>10} {'Sharpe*':>10} {'Expected':>10} {'Grade Suggestion':>20}")
        print("-" * 120)

        # Create 5-point bins
        bins = list(range(0, 101, 5))
        valid_data['score_bin'] = pd.cut(valid_data['final_score'], bins=bins,
                                         labels=[f'{i}-{i+5}' for i in range(0, 100, 5)],
                                         include_lowest=True)

        for bin_label in valid_data['score_bin'].cat.categories:
            bin_data = valid_data[valid_data['score_bin'] == bin_label]

            if len(bin_data) < 5:
                continue

            count = len(bin_data)
            avg_ret = bin_data[ret_col].mean()
            median_ret = bin_data[ret_col].median()
            std_ret = bin_data[ret_col].std()
            win_rate = (bin_data[ret_col] > 0).mean() * 100

            # Approximate Sharpe Ratio (annualized)
            sharpe_approx = (avg_ret / std_ret * np.sqrt(252/int(period))) if std_ret > 0 else 0

            # Expected value (avg_return * win_rate / 100)
            expected_value = avg_ret * win_rate / 100

            # Grade suggestion based on performance
            if avg_ret >= 8 and win_rate >= 65:
                grade_suggest = '강력 매수'
            elif avg_ret >= 5 and win_rate >= 60:
                grade_suggest = '매수'
            elif avg_ret >= 3 and win_rate >= 55:
                grade_suggest = '매수 고려'
            elif avg_ret <= -5 or win_rate <= 40:
                grade_suggest = '매도'
            elif avg_ret <= -2 or win_rate <= 45:
                grade_suggest = '매도 고려'
            else:
                grade_suggest = '중립'

            print(f"{bin_label:>15} {count:>8,} {avg_ret:>+11.2f}% {median_ret:>+9.2f}% {std_ret:>9.2f}% "
                  f"{win_rate:>9.1f}% {sharpe_approx:>9.3f} {expected_value:>+9.2f}% {grade_suggest:>20}")

            results.append({
                'period': f'{period}d',
                'score_range': bin_label,
                'score_min': int(bin_label.split('-')[0]),
                'score_max': int(bin_label.split('-')[1]),
                'count': count,
                'avg_return': avg_ret,
                'median_return': median_ret,
                'std_return': std_ret,
                'win_rate': win_rate,
                'sharpe_approx': sharpe_approx,
                'expected_value': expected_value,
                'grade_suggestion': grade_suggest
            })

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_score_granular_analysis.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    # Summary recommendations
    if len(df_results) > 0:
        print("\n" + "=" * 100)
        print("Recommended Grade Thresholds (based on 45d returns):")
        print("=" * 100)

        df_45d = df_results[df_results['period'] == '45d'].copy()

        if len(df_45d) > 0:
            # Find optimal thresholds
            strong_buy = df_45d[(df_45d['avg_return'] >= 8) & (df_45d['win_rate'] >= 65)]
            buy = df_45d[(df_45d['avg_return'] >= 5) & (df_45d['win_rate'] >= 60)]
            sell = df_45d[(df_45d['avg_return'] <= -2) | (df_45d['win_rate'] <= 45)]

            if len(strong_buy) > 0:
                strong_buy_min = strong_buy['score_min'].min()
                print(f"강력 매수 (Strong Buy):  final_score >= {strong_buy_min}")

            if len(buy) > 0:
                buy_min = buy['score_min'].min()
                print(f"매수 (Buy):              final_score >= {buy_min}")

            if len(sell) > 0:
                sell_max = sell['score_max'].max()
                print(f"매도 (Sell):             final_score < {sell_max}")

    return df_results


def filter_score_matrix_analysis(df):
    """
    STEP 9B: Filter Count × Final Score Matrix Analysis

    2D matrix showing performance by (filter_count, score_range) combination
    to optimize the "filters_passed >= 3" logic.
    """

    print("\n" + "=" * 100)
    print("STEP 9B: Filter Count × Final Score Matrix Analysis")
    print("=" * 100)

    # Check if we have filter data
    # Note: The actual filter data might not be in kr_stock_grade
    # We'll use momentum_score as a proxy for filter count
    # In real implementation, you'd need to fetch actual filter data

    if 'momentum_score' not in df.columns:
        print("[WARNING] momentum_score not found. Using simplified analysis...")
        return pd.DataFrame()

    results = []
    ret_col = 'return_45d'

    if ret_col not in df.columns:
        print(f"[WARNING] {ret_col} not found. Skipping...")
        return pd.DataFrame()

    valid_data = df[['final_score', 'momentum_score', ret_col]].dropna().copy()
    valid_data['final_score'] = valid_data['final_score'].astype(float)
    valid_data['momentum_score'] = valid_data['momentum_score'].astype(float)
    valid_data[ret_col] = valid_data[ret_col].astype(float)

    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nAnalyzing {len(valid_data):,} samples")
    print("\nNote: Using momentum_score as proxy for filter strength")
    print("      (In production, use actual filter_passed count)")

    # Create score bins
    valid_data['score_bin'] = pd.cut(valid_data['final_score'],
                                     bins=[0, 50, 60, 70, 80, 100],
                                     labels=['<50', '50-60', '60-70', '70-80', '80+'],
                                     include_lowest=True)

    # Create momentum strength bins (proxy for filter count)
    valid_data['momentum_bin'] = pd.cut(valid_data['momentum_score'],
                                        bins=[0, 25, 35, 45, 100],
                                        labels=['Weak(0-25)', 'Moderate(25-35)', 'Strong(35-45)', 'VeryStrong(45+)'],
                                        include_lowest=True)

    print("\n" + "=" * 120)
    print("Matrix: Momentum Strength × Final Score Range (45d returns)")
    print("=" * 120)
    print(f"{'Momentum':>18} | {'<50':>15} | {'50-60':>15} | {'60-70':>15} | {'70-80':>15} | {'80+':>15}")
    print("-" * 120)

    for momentum_cat in ['Weak(0-25)', 'Moderate(25-35)', 'Strong(35-45)', 'VeryStrong(45+)']:
        row_str = f"{momentum_cat:>18} |"

        for score_cat in ['<50', '50-60', '60-70', '70-80', '80+']:
            cell_data = valid_data[
                (valid_data['momentum_bin'] == momentum_cat) &
                (valid_data['score_bin'] == score_cat)
            ]

            if len(cell_data) >= 10:
                avg_ret = cell_data[ret_col].mean()
                win_rate = (cell_data[ret_col] > 0).mean() * 100
                count = len(cell_data)

                cell_str = f"{avg_ret:>+5.1f}%/{win_rate:>4.0f}%"
                row_str += f" {cell_str:>15} |"

                results.append({
                    'momentum_strength': momentum_cat,
                    'score_range': score_cat,
                    'count': count,
                    'avg_return': avg_ret,
                    'win_rate': win_rate
                })
            else:
                row_str += f" {'N/A':>15} |"

        print(row_str)

    print("\nFormat: Avg Return / Win Rate")

    # Analysis: Find best combinations
    print("\n" + "=" * 100)
    print("Top 10 Best Combinations (by Expected Value):")
    print("=" * 100)

    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results['expected_value'] = df_results['avg_return'] * df_results['win_rate'] / 100
        df_results_sorted = df_results.sort_values('expected_value', ascending=False)

        print(f"{'Rank':>5} {'Momentum':>20} {'Score Range':>12} {'Avg Return':>12} {'Win Rate':>10} {'Expected':>10} {'Count':>8}")
        print("-" * 90)

        for idx, row in df_results_sorted.head(10).iterrows():
            print(f"{df_results_sorted.index.get_loc(idx)+1:>5} {row['momentum_strength']:>20} {row['score_range']:>12} "
                  f"{row['avg_return']:>+11.2f}% {row['win_rate']:>9.1f}% {row['expected_value']:>+9.2f}% {row['count']:>8,}")

    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_filter_score_matrix.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    return df_results


def multi_objective_threshold_optimization(df):
    """
    STEP 9C: Multi-Objective Threshold Optimization

    Find optimal buy/sell thresholds using multiple objective functions:
    1. Win Rate (accuracy)
    2. Expected Value (return * win_rate)
    3. Sharpe Ratio
    4. Profit Factor (total_profit / total_loss)
    """

    print("\n" + "=" * 100)
    print("STEP 9C: Multi-Objective Threshold Optimization")
    print("=" * 100)

    ret_col = 'return_45d'
    if ret_col not in df.columns:
        print(f"[WARNING] {ret_col} not found. Skipping...")
        return pd.DataFrame()

    valid_data = df[['final_score', ret_col]].dropna().copy()
    valid_data['final_score'] = valid_data['final_score'].astype(float)
    valid_data[ret_col] = valid_data[ret_col].astype(float)

    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nOptimizing on {len(valid_data):,} samples using {ret_col}")
    print(f"Threshold ranges: Buy [60-85], Sell [25-50]")

    results = []

    buy_range = range(60, 86, 5)
    sell_range = range(25, 51, 5)

    print("\n" + "=" * 100)
    print("Optimization Results:")
    print("=" * 100)

    for buy_th in buy_range:
        for sell_th in sell_range:
            # Buy signals
            buy_data = valid_data[valid_data['final_score'] >= buy_th]
            buy_count = len(buy_data)

            if buy_count > 0:
                buy_returns = buy_data[ret_col].values
                buy_win_rate = (buy_returns > 0).mean() * 100
                buy_avg_return = buy_returns.mean()
                buy_std = buy_returns.std()
                buy_positive_sum = buy_returns[buy_returns > 0].sum()
                buy_negative_sum = abs(buy_returns[buy_returns < 0].sum())
            else:
                buy_win_rate = 0
                buy_avg_return = 0
                buy_std = 0
                buy_positive_sum = 0
                buy_negative_sum = 0.01

            # Sell signals
            sell_data = valid_data[valid_data['final_score'] < sell_th]
            sell_count = len(sell_data)

            if sell_count > 0:
                sell_returns = sell_data[ret_col].values
                sell_win_rate = (sell_returns < 0).mean() * 100  # Correct prediction = negative return
                sell_avg_return = sell_returns.mean()
            else:
                sell_win_rate = 0
                sell_avg_return = 0

            # Objective 1: Win Rate (combined)
            total_signals = buy_count + sell_count
            if total_signals > 0:
                correct_buy = (buy_data[ret_col] > 0).sum()
                correct_sell = (sell_data[ret_col] < 0).sum()
                win_rate_objective = (correct_buy + correct_sell) / total_signals * 100
            else:
                win_rate_objective = 0

            # Objective 2: Expected Value (buy only)
            expected_value = buy_avg_return * buy_win_rate / 100

            # Objective 3: Sharpe Ratio (buy only, annualized)
            sharpe_ratio = (buy_avg_return / buy_std * np.sqrt(252/45)) if buy_std > 0 else 0

            # Objective 4: Profit Factor (buy only)
            profit_factor = buy_positive_sum / buy_negative_sum if buy_negative_sum > 0 else 0

            results.append({
                'buy_threshold': buy_th,
                'sell_threshold': sell_th,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'buy_win_rate': buy_win_rate,
                'buy_avg_return': buy_avg_return,
                'sell_win_rate': sell_win_rate,
                'win_rate_objective': win_rate_objective,
                'expected_value': expected_value,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor
            })

    df_results = pd.DataFrame(results)

    # Find optimal thresholds for each objective
    print(f"\n{'Objective':>30} | {'Buy Th':>8} | {'Sell Th':>8} | {'Value':>12} | {'Buy Signals':>12} | {'Sell Signals':>12}")
    print("-" * 100)

    objectives = [
        ('Win Rate (%)', 'win_rate_objective', True),
        ('Expected Value (%)', 'expected_value', True),
        ('Sharpe Ratio', 'sharpe_ratio', True),
        ('Profit Factor', 'profit_factor', True),
    ]

    optimal_thresholds = []

    for obj_name, obj_col, maximize in objectives:
        if maximize:
            best_row = df_results.loc[df_results[obj_col].idxmax()]
        else:
            best_row = df_results.loc[df_results[obj_col].idxmin()]

        print(f"{obj_name:>30} | {best_row['buy_threshold']:>8.0f} | {best_row['sell_threshold']:>8.0f} | "
              f"{best_row[obj_col]:>12.4f} | {best_row['buy_count']:>12,.0f} | {best_row['sell_count']:>12,.0f}")

        optimal_thresholds.append({
            'objective': obj_name,
            'buy_threshold': best_row['buy_threshold'],
            'sell_threshold': best_row['sell_threshold'],
            'objective_value': best_row[obj_col],
            'buy_count': best_row['buy_count'],
            'sell_count': best_row['sell_count'],
            'buy_win_rate': best_row['buy_win_rate'],
            'buy_avg_return': best_row['buy_avg_return']
        })

    # Consensus recommendation (mode or median)
    print("\n" + "=" * 100)
    print("Consensus Recommendation:")
    print("=" * 100)

    df_optimal = pd.DataFrame(optimal_thresholds)
    consensus_buy = int(df_optimal['buy_threshold'].median())
    consensus_sell = int(df_optimal['sell_threshold'].median())

    print(f"Buy Threshold:  {consensus_buy} (median of optimal values: {df_optimal['buy_threshold'].values})")
    print(f"Sell Threshold: {consensus_sell} (median of optimal values: {df_optimal['sell_threshold'].values})")

    # Performance at consensus thresholds
    consensus_row = df_results[
        (df_results['buy_threshold'] == consensus_buy) &
        (df_results['sell_threshold'] == consensus_sell)
    ]

    if len(consensus_row) > 0:
        row = consensus_row.iloc[0]
        print(f"\nPerformance at consensus thresholds:")
        print(f"  Buy Win Rate:     {row['buy_win_rate']:.1f}%")
        print(f"  Buy Avg Return:   {row['buy_avg_return']:+.2f}%")
        print(f"  Expected Value:   {row['expected_value']:+.2f}%")
        print(f"  Sharpe Ratio:     {row['sharpe_ratio']:.3f}")
        print(f"  Profit Factor:    {row['profit_factor']:.2f}")

    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_multi_objective_optimization.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    output_file2 = os.path.join(OUTPUT_DIR, 'phase3_7_optimal_thresholds.csv')
    df_optimal.to_csv(output_file2, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_file2}")

    return df_results


# ============================================================================
# PART 5: FEATURE VALIDATION (Phase 3 구현 기능 검증)
# ============================================================================

def validate_entry_timing_score(df):
    """STEP 11: Entry Timing Score 검증

    검증 질문: 높은 entry_timing_score로 진입하면 실제로 더 나은 수익률인가?
    """
    print("\n" + "=" * 100)
    print("STEP 11: Entry Timing Score Validation")
    print("=" * 100)

    results = []

    if 'entry_timing_score' not in df.columns:
        print("[WARNING] entry_timing_score column not found. Skipping...")
        return pd.DataFrame()

    valid_data = df[df['entry_timing_score'].notna()].copy()
    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nTotal samples with entry_timing_score: {len(valid_data):,}")

    # 1. IC 분석: entry_timing_score vs 수익률
    print("\n[1] Entry Timing Score IC Analysis:")
    print("-" * 80)
    print(f"{'Period':<10} {'Spearman IC':>12} {'Pearson IC':>12} {'N':>10}")
    print("-" * 80)

    for period in RETURN_PERIODS:
        ret_col = f'return_{period}d'
        if ret_col not in df.columns:
            continue

        subset = valid_data[['entry_timing_score', ret_col]].dropna()
        subset['entry_timing_score'] = pd.to_numeric(subset['entry_timing_score'], errors='coerce')
        subset[ret_col] = pd.to_numeric(subset[ret_col], errors='coerce')
        subset = subset.dropna()
        if len(subset) < 30:
            continue

        spearman_ic, _ = spearmanr(subset['entry_timing_score'], subset[ret_col])
        pearson_ic, _ = pearsonr(subset['entry_timing_score'], subset[ret_col])

        print(f"{period}d{'':<7} {spearman_ic:>12.4f} {pearson_ic:>12.4f} {len(subset):>10,}")

        results.append({
            'metric': 'entry_timing_score_ic',
            'period': f'{period}d',
            'spearman_ic': spearman_ic,
            'pearson_ic': pearson_ic,
            'n_samples': len(subset)
        })

    # 2. Quintile 분석: entry_timing_score 5분위별 수익률
    print("\n[2] Entry Timing Score Quintile Analysis (30d return):")
    print("-" * 80)

    ret_col = 'return_30d' if 'return_30d' in valid_data.columns else 'return_45d'
    if ret_col in valid_data.columns:
        quintile_data = valid_data[['entry_timing_score', ret_col]].dropna()
        try:
            quintile_data['quintile'] = pd.qcut(quintile_data['entry_timing_score'], q=5,
                                                labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'],
                                                duplicates='drop')

            print(f"{'Quintile':<12} {'Score Range':>20} {'Avg Return':>12} {'Median':>10} {'Win Rate':>10} {'Count':>8}")
            print("-" * 80)

            for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']:
                q_data = quintile_data[quintile_data['quintile'] == q]
                if len(q_data) > 0:
                    score_min = q_data['entry_timing_score'].min()
                    score_max = q_data['entry_timing_score'].max()
                    avg_ret = q_data[ret_col].mean()
                    median_ret = q_data[ret_col].median()
                    win_rate = (q_data[ret_col] > 0).mean() * 100

                    print(f"{q:<12} {score_min:>8.1f}-{score_max:>8.1f} {avg_ret:>+11.2f}% {median_ret:>+9.2f}% {win_rate:>9.1f}% {len(q_data):>8,}")

                    results.append({
                        'metric': 'entry_timing_quintile',
                        'period': ret_col,
                        'quintile': q,
                        'avg_return': avg_ret,
                        'median_return': median_ret,
                        'win_rate': win_rate,
                        'n_samples': len(q_data)
                    })
        except Exception as e:
            print(f"[WARNING] Quintile analysis failed: {e}")

    # 3. Entry Timing + Final Score 조합 분석
    print("\n[3] Entry Timing + Final Score Combined Analysis:")
    print("-" * 80)

    if 'return_45d' in valid_data.columns:
        combo_data = valid_data[['entry_timing_score', 'final_score', 'return_45d']].dropna()

        # High Entry Timing (>=70) + High Final Score (>=70)
        high_high = combo_data[(combo_data['entry_timing_score'] >= 70) & (combo_data['final_score'] >= 70)]
        high_low = combo_data[(combo_data['entry_timing_score'] >= 70) & (combo_data['final_score'] < 50)]
        low_high = combo_data[(combo_data['entry_timing_score'] < 50) & (combo_data['final_score'] >= 70)]
        low_low = combo_data[(combo_data['entry_timing_score'] < 50) & (combo_data['final_score'] < 50)]

        print(f"{'Combination':<30} {'Avg Return':>12} {'Median':>10} {'Win Rate':>10} {'Count':>8}")
        print("-" * 80)

        for name, subset in [('High Entry + High Score', high_high),
                             ('High Entry + Low Score', high_low),
                             ('Low Entry + High Score', low_high),
                             ('Low Entry + Low Score', low_low)]:
            if len(subset) > 10:
                avg_ret = subset['return_45d'].mean()
                median_ret = subset['return_45d'].median()
                win_rate = (subset['return_45d'] > 0).mean() * 100
                print(f"{name:<30} {avg_ret:>+11.2f}% {median_ret:>+9.2f}% {win_rate:>9.1f}% {len(subset):>8,}")

                results.append({
                    'metric': 'entry_timing_combo',
                    'period': '45d',
                    'combination': name,
                    'avg_return': avg_ret,
                    'median_return': median_ret,
                    'win_rate': win_rate,
                    'n_samples': len(subset)
                })

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_entry_timing_validation.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    return df_results


def validate_stop_loss(df):
    """STEP 12: Stop Loss 검증

    검증 질문: ATR 기반 손절이 최적인가? 손절 적중률 vs 조기 청산 비율
    """
    print("\n" + "=" * 100)
    print("STEP 12: Stop Loss Validation")
    print("=" * 100)

    results = []

    if 'stop_loss_pct' not in df.columns:
        print("[WARNING] stop_loss_pct column not found. Skipping...")
        return pd.DataFrame()

    valid_data = df[df['stop_loss_pct'].notna()].copy()
    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nTotal samples with stop_loss_pct: {len(valid_data):,}")

    # stop_loss_pct는 음수로 저장되어 있을 수 있음
    valid_data['stop_loss_abs'] = valid_data['stop_loss_pct'].abs()

    print(f"Stop Loss Range: {valid_data['stop_loss_abs'].min():.1f}% ~ {valid_data['stop_loss_abs'].max():.1f}%")
    print(f"Mean Stop Loss: {valid_data['stop_loss_abs'].mean():.2f}%")

    # 기간별 분석
    for period in [30, 45, 60]:
        ret_col = f'return_{period}d'
        if ret_col not in valid_data.columns:
            continue

        subset = valid_data[['stop_loss_abs', ret_col, 'final_score']].dropna()
        if len(subset) < 50:
            continue

        print(f"\n[{period}d Analysis]")
        print("-" * 80)

        # 손절 트리거 여부 (최저 수익률이 손절선 이하로 내려갔는지 - 근사치로 기간 수익률 사용)
        subset['would_stop'] = subset[ret_col] <= -subset['stop_loss_abs']
        subset['final_positive'] = subset[ret_col] > 0

        # Case 분류
        stop_triggered_positive = len(subset[(subset['would_stop']) & (subset['final_positive'])])
        stop_triggered_negative = len(subset[(subset['would_stop']) & (~subset['final_positive'])])
        no_stop_positive = len(subset[(~subset['would_stop']) & (subset['final_positive'])])
        no_stop_negative = len(subset[(~subset['would_stop']) & (~subset['final_positive'])])

        total = len(subset)

        print(f"{'Scenario':<40} {'Count':>10} {'Ratio':>10}")
        print("-" * 60)
        print(f"{'손절 발동 + 최종 양수 (조기 청산)':<40} {stop_triggered_positive:>10,} {stop_triggered_positive/total*100:>9.1f}%")
        print(f"{'손절 발동 + 최종 음수 (정확한 손절)':<40} {stop_triggered_negative:>10,} {stop_triggered_negative/total*100:>9.1f}%")
        print(f"{'손절 미발동 + 최종 양수 (정상 보유)':<40} {no_stop_positive:>10,} {no_stop_positive/total*100:>9.1f}%")
        print(f"{'손절 미발동 + 최종 음수 (손절 실패)':<40} {no_stop_negative:>10,} {no_stop_negative/total*100:>9.1f}%")

        # 손절 효과 분석
        if stop_triggered_positive + stop_triggered_negative > 0:
            stop_accuracy = stop_triggered_negative / (stop_triggered_positive + stop_triggered_negative) * 100
            early_exit_rate = stop_triggered_positive / (stop_triggered_positive + stop_triggered_negative) * 100
            print(f"\n손절 정확도: {stop_accuracy:.1f}% | 조기 청산율: {early_exit_rate:.1f}%")

        results.append({
            'period': f'{period}d',
            'stop_triggered_positive': stop_triggered_positive,
            'stop_triggered_negative': stop_triggered_negative,
            'no_stop_positive': no_stop_positive,
            'no_stop_negative': no_stop_negative,
            'total': total,
            'stop_accuracy': stop_triggered_negative / max(1, stop_triggered_positive + stop_triggered_negative) * 100,
            'early_exit_rate': stop_triggered_positive / max(1, stop_triggered_positive + stop_triggered_negative) * 100
        })

    # 손절 비율별 성과 비교
    print("\n[Stop Loss % Quintile Analysis - 45d]:")
    print("-" * 80)

    if 'return_45d' in valid_data.columns:
        sl_data = valid_data[['stop_loss_abs', 'return_45d']].dropna()
        try:
            sl_data['sl_quintile'] = pd.qcut(sl_data['stop_loss_abs'], q=5,
                                             labels=['Q1(Tight)', 'Q2', 'Q3', 'Q4', 'Q5(Wide)'],
                                             duplicates='drop')

            print(f"{'SL Quintile':<15} {'SL Range':>15} {'Avg Return':>12} {'Win Rate':>10}")
            print("-" * 60)

            for q in ['Q1(Tight)', 'Q2', 'Q3', 'Q4', 'Q5(Wide)']:
                q_data = sl_data[sl_data['sl_quintile'] == q]
                if len(q_data) > 0:
                    sl_min = q_data['stop_loss_abs'].min()
                    sl_max = q_data['stop_loss_abs'].max()
                    avg_ret = q_data['return_45d'].mean()
                    win_rate = (q_data['return_45d'] > 0).mean() * 100
                    print(f"{q:<15} {sl_min:>6.1f}%-{sl_max:>5.1f}% {avg_ret:>+11.2f}% {win_rate:>9.1f}%")
        except Exception as e:
            print(f"[WARNING] Quintile analysis failed: {e}")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_stop_loss_validation.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    return df_results


def validate_scenario_probability(df):
    """STEP 13: Scenario Probability Calibration 검증

    검증 질문: 예측 확률이 실제 실현 비율과 일치하는가?
    """
    print("\n" + "=" * 100)
    print("STEP 13: Scenario Probability Calibration")
    print("=" * 100)

    results = []

    required_cols = ['scenario_bullish_prob', 'scenario_sideways_prob', 'scenario_bearish_prob']
    if not all(col in df.columns for col in required_cols):
        print("[WARNING] Scenario probability columns not found. Skipping...")
        return pd.DataFrame()

    valid_data = df[df['scenario_bullish_prob'].notna()].copy()
    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nTotal samples with scenario probabilities: {len(valid_data):,}")

    # 실제 결과 분류 기준 (45일 수익률 기준)
    ret_col = 'return_45d' if 'return_45d' in valid_data.columns else 'return_60d'
    if ret_col not in valid_data.columns:
        print("[WARNING] No suitable return column found")
        return pd.DataFrame()

    scenario_data = valid_data[required_cols + [ret_col]].dropna()

    # 실제 결과 분류: Bullish (>5%), Sideways (-5%~5%), Bearish (<-5%)
    scenario_data['actual_scenario'] = pd.cut(
        scenario_data[ret_col],
        bins=[-float('inf'), -5, 5, float('inf')],
        labels=['Bearish', 'Sideways', 'Bullish']
    )

    print(f"\n[1] Actual Scenario Distribution ({ret_col}):")
    print("-" * 60)
    actual_dist = scenario_data['actual_scenario'].value_counts(normalize=True) * 100
    for scenario in ['Bullish', 'Sideways', 'Bearish']:
        if scenario in actual_dist.index:
            print(f"  {scenario}: {actual_dist[scenario]:.1f}%")

    # Calibration 분석: 예측 확률 구간별 실제 발생률
    print(f"\n[2] Bullish Probability Calibration:")
    print("-" * 80)
    print(f"{'Predicted Prob':<20} {'Actual Bullish':>15} {'Count':>10} {'Calibration':>15}")
    print("-" * 80)

    prob_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

    for low, high in prob_bins:
        bin_data = scenario_data[(scenario_data['scenario_bullish_prob'] >= low) &
                                  (scenario_data['scenario_bullish_prob'] < high)]
        if len(bin_data) > 10:
            actual_bullish = (bin_data['actual_scenario'] == 'Bullish').mean() * 100
            predicted_mid = (low + high) / 2
            calibration = actual_bullish - predicted_mid

            print(f"{low:>3}%-{high:<3}%{'':<12} {actual_bullish:>14.1f}% {len(bin_data):>10,} {calibration:>+14.1f}%")

            results.append({
                'scenario': 'Bullish',
                'prob_range': f'{low}-{high}%',
                'predicted_mid': predicted_mid,
                'actual_rate': actual_bullish,
                'calibration_error': calibration,
                'n_samples': len(bin_data)
            })

    print(f"\n[3] Bearish Probability Calibration:")
    print("-" * 80)
    print(f"{'Predicted Prob':<20} {'Actual Bearish':>15} {'Count':>10} {'Calibration':>15}")
    print("-" * 80)

    for low, high in prob_bins:
        bin_data = scenario_data[(scenario_data['scenario_bearish_prob'] >= low) &
                                  (scenario_data['scenario_bearish_prob'] < high)]
        if len(bin_data) > 10:
            actual_bearish = (bin_data['actual_scenario'] == 'Bearish').mean() * 100
            predicted_mid = (low + high) / 2
            calibration = actual_bearish - predicted_mid

            print(f"{low:>3}%-{high:<3}%{'':<12} {actual_bearish:>14.1f}% {len(bin_data):>10,} {calibration:>+14.1f}%")

            results.append({
                'scenario': 'Bearish',
                'prob_range': f'{low}-{high}%',
                'predicted_mid': predicted_mid,
                'actual_rate': actual_bearish,
                'calibration_error': calibration,
                'n_samples': len(bin_data)
            })

    # 최고 확률 시나리오 vs 실제 결과
    print(f"\n[4] Highest Probability Scenario Accuracy:")
    print("-" * 60)

    scenario_data['predicted_scenario'] = scenario_data[['scenario_bullish_prob', 'scenario_sideways_prob', 'scenario_bearish_prob']].idxmax(axis=1)
    scenario_data['predicted_scenario'] = scenario_data['predicted_scenario'].map({
        'scenario_bullish_prob': 'Bullish',
        'scenario_sideways_prob': 'Sideways',
        'scenario_bearish_prob': 'Bearish'
    })

    correct = (scenario_data['predicted_scenario'] == scenario_data['actual_scenario']).sum()
    total = len(scenario_data)
    accuracy = correct / total * 100

    print(f"Prediction Accuracy: {accuracy:.1f}% ({correct:,}/{total:,})")

    # 예측별 정확도
    for pred in ['Bullish', 'Sideways', 'Bearish']:
        pred_data = scenario_data[scenario_data['predicted_scenario'] == pred]
        if len(pred_data) > 0:
            pred_correct = (pred_data['predicted_scenario'] == pred_data['actual_scenario']).mean() * 100
            print(f"  {pred} predicted: {pred_correct:.1f}% accurate (n={len(pred_data):,})")

    results.append({
        'scenario': 'Overall',
        'prob_range': 'Max Prob',
        'predicted_mid': None,
        'actual_rate': accuracy,
        'calibration_error': None,
        'n_samples': total
    })

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_scenario_calibration.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    return df_results


def validate_buy_triggers(df):
    """STEP 14: Buy Triggers 검증

    검증 질문: 트리거 조건 충족 종목이 미충족 종목보다 성과가 좋은가?
    """
    print("\n" + "=" * 100)
    print("STEP 14: Buy Triggers Validation")
    print("=" * 100)

    results = []

    if 'buy_triggers' not in df.columns:
        print("[WARNING] buy_triggers column not found. Skipping...")
        return pd.DataFrame()

    valid_data = df[df['buy_triggers'].notna()].copy()
    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nTotal samples with buy_triggers: {len(valid_data):,}")

    # buy_triggers는 JSONB이므로 파싱 필요
    def count_triggers(triggers):
        """트리거 충족 개수 계산"""
        if triggers is None:
            return 0
        try:
            if isinstance(triggers, str):
                triggers = json.loads(triggers)
            if isinstance(triggers, dict):
                return sum(1 for v in triggers.values() if v is True or v == 'true' or v == 1)
            elif isinstance(triggers, list):
                return len(triggers)
            return 0
        except:
            return 0

    valid_data['trigger_count'] = valid_data['buy_triggers'].apply(count_triggers)

    print(f"Trigger count distribution:")
    print(valid_data['trigger_count'].value_counts().sort_index())

    # 트리거 개수별 성과 분석
    for period in [30, 45, 60]:
        ret_col = f'return_{period}d'
        if ret_col not in valid_data.columns:
            continue

        subset = valid_data[['trigger_count', ret_col, 'final_score']].dropna()

        print(f"\n[{period}d Return by Trigger Count]:")
        print("-" * 80)
        print(f"{'Triggers':<15} {'Avg Return':>12} {'Median':>10} {'Win Rate':>10} {'Count':>10}")
        print("-" * 80)

        for trigger_n in sorted(subset['trigger_count'].unique()):
            t_data = subset[subset['trigger_count'] == trigger_n]
            if len(t_data) > 20:
                avg_ret = t_data[ret_col].mean()
                median_ret = t_data[ret_col].median()
                win_rate = (t_data[ret_col] > 0).mean() * 100

                print(f"{trigger_n} triggers{'':<6} {avg_ret:>+11.2f}% {median_ret:>+9.2f}% {win_rate:>9.1f}% {len(t_data):>10,}")

                results.append({
                    'period': f'{period}d',
                    'trigger_count': trigger_n,
                    'avg_return': avg_ret,
                    'median_return': median_ret,
                    'win_rate': win_rate,
                    'n_samples': len(t_data)
                })

        # High Score (>=70) 내에서 트리거 효과
        high_score = subset[subset['final_score'] >= 70]
        if len(high_score) > 50:
            print(f"\n[{period}d Return - High Score (>=70) Only]:")
            print("-" * 60)

            many_triggers = high_score[high_score['trigger_count'] >= 3]
            few_triggers = high_score[high_score['trigger_count'] <= 1]

            if len(many_triggers) > 20 and len(few_triggers) > 20:
                many_avg = many_triggers[ret_col].mean()
                many_win = (many_triggers[ret_col] > 0).mean() * 100
                few_avg = few_triggers[ret_col].mean()
                few_win = (few_triggers[ret_col] > 0).mean() * 100

                print(f"  Many triggers (3+): Avg {many_avg:+.2f}%, Win Rate {many_win:.1f}% (n={len(many_triggers):,})")
                print(f"  Few triggers (0-1): Avg {few_avg:+.2f}%, Win Rate {few_win:.1f}% (n={len(few_triggers):,})")
                print(f"  Difference: {many_avg - few_avg:+.2f}%p return, {many_win - few_win:+.1f}%p win rate")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_buy_triggers_validation.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    return df_results


def validate_position_size(df):
    """STEP 15: Position Size 검증

    검증 질문: 변동성 기반 비중이 리스크 조정 수익률을 개선하는가?
    """
    print("\n" + "=" * 100)
    print("STEP 15: Position Size Validation")
    print("=" * 100)

    results = []

    if 'position_size_pct' not in df.columns:
        print("[WARNING] position_size_pct column not found. Skipping...")
        return pd.DataFrame()

    valid_data = df[df['position_size_pct'].notna()].copy()
    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nTotal samples with position_size_pct: {len(valid_data):,}")
    print(f"Position Size Range: {valid_data['position_size_pct'].min():.2f}% ~ {valid_data['position_size_pct'].max():.2f}%")
    print(f"Mean Position Size: {valid_data['position_size_pct'].mean():.2f}%")

    # 1. Position Size Quintile별 성과
    for period in [30, 45, 60]:
        ret_col = f'return_{period}d'
        if ret_col not in valid_data.columns:
            continue

        subset = valid_data[['position_size_pct', ret_col, 'volatility_annual', 'final_score']].dropna()
        if len(subset) < 100:
            continue

        print(f"\n[{period}d Analysis by Position Size Quintile]:")
        print("-" * 100)

        try:
            subset['ps_quintile'] = pd.qcut(subset['position_size_pct'], q=5,
                                            labels=['Q1(Small)', 'Q2', 'Q3', 'Q4', 'Q5(Large)'],
                                            duplicates='drop')

            print(f"{'Quintile':<12} {'PS Range':>15} {'Avg Return':>12} {'Volatility':>12} {'Sharpe*':>10} {'Win Rate':>10}")
            print("-" * 100)

            for q in ['Q1(Small)', 'Q2', 'Q3', 'Q4', 'Q5(Large)']:
                q_data = subset[subset['ps_quintile'] == q]
                if len(q_data) > 0:
                    ps_min = q_data['position_size_pct'].min()
                    ps_max = q_data['position_size_pct'].max()
                    avg_ret = q_data[ret_col].mean()
                    avg_vol = q_data['volatility_annual'].mean()
                    sharpe_approx = avg_ret / avg_vol if avg_vol > 0 else 0
                    win_rate = (q_data[ret_col] > 0).mean() * 100

                    print(f"{q:<12} {ps_min:>6.2f}%-{ps_max:>5.2f}% {avg_ret:>+11.2f}% {avg_vol:>11.1f}% {sharpe_approx:>9.3f} {win_rate:>9.1f}%")

                    results.append({
                        'period': f'{period}d',
                        'analysis': 'ps_quintile',
                        'group': q,
                        'avg_return': avg_ret,
                        'avg_volatility': avg_vol,
                        'sharpe_approx': sharpe_approx,
                        'win_rate': win_rate,
                        'n_samples': len(q_data)
                    })
        except Exception as e:
            print(f"[WARNING] Quintile analysis failed: {e}")

    # 2. 가중 수익률 시뮬레이션
    print(f"\n[Weighted Return Simulation - 45d]:")
    print("-" * 80)

    if 'return_45d' in valid_data.columns:
        sim_data = valid_data[['position_size_pct', 'return_45d', 'final_score']].dropna()
        sim_data['position_size_pct'] = sim_data['position_size_pct'].astype(float)
        sim_data['return_45d'] = sim_data['return_45d'].astype(float)
        sim_data['final_score'] = sim_data['final_score'].astype(float)

        # Buy candidates only (final_score >= 65)
        buy_candidates = sim_data[sim_data['final_score'] >= 65].copy()

        if len(buy_candidates) > 50:
            # 균등 비중 vs 변동성 기반 비중
            equal_weight_return = buy_candidates['return_45d'].mean()

            # 변동성 기반 가중 (position_size_pct 비례)
            total_position = buy_candidates['position_size_pct'].sum()
            buy_candidates['weight'] = buy_candidates['position_size_pct'] / total_position
            weighted_return = (buy_candidates['return_45d'] * buy_candidates['weight']).sum()

            print(f"Buy Candidates (Score >= 65): {len(buy_candidates):,} stocks")
            print(f"  Equal Weight Return:    {equal_weight_return:+.2f}%")
            print(f"  Position Size Weighted: {weighted_return:+.2f}%")
            print(f"  Difference:             {weighted_return - equal_weight_return:+.2f}%")

            results.append({
                'period': '45d',
                'analysis': 'weighted_simulation',
                'group': 'Equal Weight',
                'avg_return': equal_weight_return,
                'n_samples': len(buy_candidates)
            })
            results.append({
                'period': '45d',
                'analysis': 'weighted_simulation',
                'group': 'PS Weighted',
                'avg_return': weighted_return,
                'n_samples': len(buy_candidates)
            })

    # 3. High Score에서 Position Size 효과
    print(f"\n[Position Size Effect within High Score (>=70)]:")
    print("-" * 60)

    if 'return_45d' in valid_data.columns:
        high_score = valid_data[valid_data['final_score'] >= 70][['position_size_pct', 'return_45d']].dropna()

        if len(high_score) > 50:
            large_ps = high_score[high_score['position_size_pct'] >= high_score['position_size_pct'].median()]
            small_ps = high_score[high_score['position_size_pct'] < high_score['position_size_pct'].median()]

            print(f"  Large Position Size: Avg {large_ps['return_45d'].mean():+.2f}%, Win Rate {(large_ps['return_45d'] > 0).mean()*100:.1f}%")
            print(f"  Small Position Size: Avg {small_ps['return_45d'].mean():+.2f}%, Win Rate {(small_ps['return_45d'] > 0).mean()*100:.1f}%")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_position_size_validation.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# PART 6: THEME/SECTOR ANALYSIS
# ============================================================================

def theme_sector_analysis(df):
    """STEP 10: Theme/Sector Analysis"""

    print("\n" + "=" * 100)
    print("STEP 10: Theme/Sector Analysis")
    print("=" * 100)

    longest_period = max(RETURN_PERIODS)
    longest_ret = f'return_{longest_period}d'

    if longest_ret not in df.columns or 'theme' not in df.columns:
        print("[WARNING] Required columns missing. Skipping theme analysis...")
        return pd.DataFrame()

    themes = df['theme'].dropna().unique()
    theme_results = []

    for theme in sorted(themes):
        theme_data = df[df['theme'] == theme]

        if len(theme_data) < 30:
            continue

        ic_result = calculate_ic(theme_data, 'final_score', [longest_ret])

        if longest_ret in ic_result and ic_result[longest_ret]['spearman_ic'] is not None:
            ic = ic_result[longest_ret]['spearman_ic']
            n = ic_result[longest_ret]['n_samples']
            avg_return = theme_data[longest_ret].mean()
            win_rate = (theme_data[longest_ret] > 0).mean() * 100

            theme_results.append({
                'theme': theme,
                'ic': ic,
                'n_samples': n,
                'avg_return': avg_return,
                'win_rate': win_rate
            })

    df_results = pd.DataFrame(theme_results)
    df_results = df_results.sort_values('ic', ascending=False)

    print(f"\nTop 15 Themes (by IC, {longest_period}d return):")
    print("-" * 100)
    print(f"{'Theme':<40} {'IC':>10} {'Samples':>10} {'Avg Ret':>10} {'Win Rate':>10}")
    print("-" * 100)
    for _, row in df_results.head(15).iterrows():
        print(f"{row['theme']:<40} {row['ic']:>10.4f} {row['n_samples']:>10,} "
              f"{row['avg_return']:>+9.2f}% {row['win_rate']:>9.1f}%")

    print(f"\nBottom 15 Themes (by IC):")
    print("-" * 100)
    for _, row in df_results.tail(15).iterrows():
        print(f"{row['theme']:<40} {row['ic']:>10.4f} {row['n_samples']:>10,} "
              f"{row['avg_return']:>+9.2f}% {row['win_rate']:>9.1f}%")

    output_file = os.path.join(OUTPUT_DIR, 'phase3_7_theme_analysis.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution"""
    print("\n" + "=" * 100)
    print("Phase 3.7 - Comprehensive IC Analysis with Win Rate Optimization")
    print("=" * 100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Dates: {', '.join(ANALYSIS_DATES)}")
    print(f"Return Periods: {RETURN_PERIODS} days")
    print(f"Value Strategies: {len(VALUE_STRATEGIES)} (Phase 3.7)")
    print("=" * 100)

    # Create connection pool
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=10,
        max_size=45,
        command_timeout=60
    )

    db_manager = AsyncDatabaseManager()
    await db_manager.initialize()

    try:
        # ========================================
        # PHASE A: DATA COLLECTION
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE A: Data Collection & Preparation")
        print("=" * 100)

        df_grade, df_price = await collect_data(pool)
        df_analysis = calculate_forward_returns(df_grade, df_price)

        output_file = os.path.join(OUTPUT_DIR, 'phase3_7_analysis_data.csv')
        df_analysis.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_file}")

        # ========================================
        # PHASE B: IC ANALYSIS
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE B: IC Analysis")
        print("=" * 100)

        ic_results_factor = ic_analysis_factor_level(df_analysis)

        # Value strategy IC - DISABLED for faster execution
        # (Uses ValueFactorCalculator to recalculate V2~V26 strategies, very slow)
        # df_ic_value, df_strategies = await ic_analysis_value_strategies(
        #     db_manager, df_analysis, use_sample=False
        # )

        # ========================================
        # PHASE C: VALIDATION
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE C: Validation & Testing")
        print("=" * 100)

        df_rolling = rolling_window_analysis(df_analysis)
        df_decile = decile_test(df_analysis)

        # ========================================
        # PHASE D: NEW ANALYSES (Phase 3.7)
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE D: Phase 3.7 New Analyses")
        print("=" * 100)

        df_weight = weight_comparison_analysis(df_analysis)
        df_param = parameter_surface_analysis(df_analysis)
        df_accuracy = accuracy_winrate_analysis(df_analysis)
        df_timing = timing_filter_analysis(df_analysis)

        # ========================================
        # PHASE D.5: FINAL GRADE OPTIMIZATION (NEW)
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE D.5: Final Grade Optimization Analysis")
        print("=" * 100)

        df_granular = final_score_granular_analysis(df_analysis)
        df_filter_matrix = filter_score_matrix_analysis(df_analysis)
        df_multi_opt = multi_objective_threshold_optimization(df_analysis)

        # ========================================
        # PHASE E: THEME/SECTOR
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE E: Theme/Sector Analysis")
        print("=" * 100)

        df_theme = theme_sector_analysis(df_analysis)

        # ========================================
        # PHASE F: FEATURE VALIDATION (NEW)
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE F: Feature Validation (Phase 3 구현 기능 검증)")
        print("=" * 100)

        df_entry_timing = validate_entry_timing_score(df_analysis)
        df_stop_loss = validate_stop_loss(df_analysis)
        df_scenario = validate_scenario_probability(df_analysis)
        df_triggers = validate_buy_triggers(df_analysis)
        df_position = validate_position_size(df_analysis)

        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "=" * 100)
        print("All Analyses Complete!")
        print("=" * 100)

        print("\nGenerated Files:")
        print("  [Data]")
        print("  1. phase3_7_analysis_data.csv - Full analysis data")
        print("\n  [IC Analysis]")
        print("  2. phase3_7_factor_ic.csv - Factor level IC")
        print("\n  [Validation]")
        print("  3. phase3_7_rolling_ic.csv - Rolling window IC")
        print("  4. phase3_7_decile_test.csv - Decile test results")
        print("\n  [Phase 3.7 New]")
        print("  5. phase3_7_weight_comparison.csv - Uniform vs Dynamic weights")
        print("  6. phase3_7_parameter_surface.csv - Threshold optimization")
        print("  7. phase3_7_accuracy_winrate.csv - Accuracy and win rate")
        print("  8. phase3_7_timing_filter.csv - Timing filter effect")
        print("\n  [Final Grade Optimization - NEW]")
        print("  9. phase3_7_score_granular_analysis.csv - 5-point bin analysis with grade suggestions")
        print("  10. phase3_7_filter_score_matrix.csv - Filter count × Score range matrix")
        print("  11. phase3_7_multi_objective_optimization.csv - Multi-objective threshold optimization")
        print("  12. phase3_7_optimal_thresholds.csv - Recommended thresholds by objective")
        print("\n  [Theme/Sector]")
        print("  13. phase3_7_theme_analysis.csv - Theme analysis")
        print("\n  [Feature Validation - NEW]")
        print("  14. phase3_7_entry_timing_validation.csv - Entry timing score 검증")
        print("  15. phase3_7_stop_loss_validation.csv - Stop loss 검증")
        print("  16. phase3_7_scenario_calibration.csv - Scenario probability 검증")
        print("  17. phase3_7_buy_triggers_validation.csv - Buy triggers 검증")
        print("  18. phase3_7_position_size_validation.csv - Position size 검증")

    finally:
        await pool.close()
        await db_manager.close()
        print("\nDatabase connections closed")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    asyncio.run(main())
