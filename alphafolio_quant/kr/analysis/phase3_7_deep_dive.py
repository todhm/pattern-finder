"""
Phase 3.7 - Momentum Factor Deep Dive Analysis
===============================================

Purpose:
- Momentum Factor strategy-level performance evaluation (M1~M23)
- Value Factor selected strategies comparison (V2, V4, V13, V25)
- Identify problem strategies and improvement directions
- Telecom_Media theme detailed analysis

Analysis Dates: 2025-08-04, 2025-08-11, 2025-08-22, 2025-09-01, 2025-09-09, 2025-09-16

Execute: python phase3_7_deep_dive.py
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, date
import os
import sys
import logging
import signal
import json
from dotenv import load_dotenv

# Global checkpoint data for graceful shutdown
_checkpoint_data = {
    'results': [],
    'path': None,
    'enabled': False
}

def _save_checkpoint_on_exit(signum, frame):
    """Ctrl+C 시 체크포인트 저장 후 종료"""
    if _checkpoint_data['enabled'] and _checkpoint_data['results'] and _checkpoint_data['path']:
        print(f"\n\n[Interrupt] Saving checkpoint before exit...")
        try:
            df = pd.DataFrame(_checkpoint_data['results'])
            df.to_csv(_checkpoint_data['path'], index=False, encoding='utf-8-sig')
            print(f"[Checkpoint saved: {len(_checkpoint_data['results']):,} records to {_checkpoint_data['path']}]")
        except Exception as e:
            print(f"[Checkpoint save failed: {e}]")
    print("Exiting...")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, _save_checkpoint_on_exit)

# Add kr directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'kr'))

# Import Factor Calculators
from kr.db_async import AsyncDatabaseManager
from kr.kr_momentum_factor import MomentumFactorCalculator
from kr.kr_value_factor import ValueFactorCalculator

load_dotenv()

# Suppress detailed logs from kr modules - only show progress
logging.getLogger('kr.kr_momentum_factor').setLevel(logging.WARNING)
logging.getLogger('kr.kr_value_factor').setLevel(logging.WARNING)
logging.getLogger('kr.db_async').setLevel(logging.WARNING)


# ============================================================================
# Query Cache Wrapper - Reduces redundant DB queries
# ============================================================================
class CachedDatabaseManager:
    """
    Wrapper for AsyncDatabaseManager with query result caching.
    Same (query, params) combination returns cached result.
    """

    def __init__(self, db_manager: AsyncDatabaseManager):
        self._db_manager = db_manager
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _make_cache_key(self, query: str, params: tuple) -> str:
        """Create hashable cache key from query and params"""
        # Normalize query whitespace for better cache hits
        normalized_query = ' '.join(query.split())
        return f"{normalized_query}::{params}"

    async def execute_query(self, query: str, *params):
        """Execute query with caching"""
        cache_key = self._make_cache_key(query, params)

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1
        result = await self._db_manager.execute_query(query, *params)
        self._cache[cache_key] = result
        return result

    async def initialize(self, min_size=5, max_size=20):
        """Initialize with reduced pool size for stability"""
        await self._db_manager.initialize(min_size=min_size, max_size=max_size)

    async def close(self):
        """Close underlying db_manager"""
        await self._db_manager.close()

    def clear_cache(self):
        """Clear cache (call between batches to prevent memory bloat)"""
        self._cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self._cache)
        }

    def reset_stats(self):
        """Reset cache statistics"""
        self._cache_hits = 0
        self._cache_misses = 0


# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgresql+asyncpg://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')

# Output directory
OUTPUT_DIR = r'C:\project\alpha\quant\kr\result test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis dates
ANALYSIS_DATES = [
    '2025-08-04', '2025-08-11', '2025-08-19', '2025-08-22',
    '2025-09-01', '2025-09-05', '2025-09-09', '2025-09-12', '2025-09-16'
]

# Return periods to analyze
RETURN_PERIODS = [15, 30, 45, 60]


async def connect_db():
    """Connect to PostgreSQL database"""
    return await asyncpg.connect(DATABASE_URL)


def calculate_ic(df, score_col, return_col):
    """Calculate Pearson and Spearman IC"""
    valid = df[[score_col, return_col]].copy()
    valid[score_col] = pd.to_numeric(valid[score_col], errors='coerce')
    valid[return_col] = pd.to_numeric(valid[return_col], errors='coerce')
    valid = valid.dropna()

    if len(valid) < 30:
        return {'pearson': 0, 'spearman': 0, 'n': len(valid)}

    pearson_ic, _ = stats.pearsonr(valid[score_col], valid[return_col])
    spearman_ic, _ = stats.spearmanr(valid[score_col], valid[return_col])

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
    Load base data from kr_stock_grade, kr_intraday_total, and kr_stock_detail
    for the specified analysis dates
    """
    print("\n" + "="*100)
    print("Loading base data from database...")
    print("="*100)

    dates_str = "', '".join(ANALYSIS_DATES)

    query = f"""
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
            g.market_state
        FROM kr_stock_grade g
        WHERE g.date IN ('{dates_str}')
    ),
    price_data AS (
        SELECT
            symbol,
            date,
            close,
            market_cap
        FROM kr_intraday_total
        WHERE date IN ('{dates_str}')
    ),
    future_returns AS (
        SELECT
            t1.symbol,
            t1.date as base_date,
            t2.date as future_date,
            t2.close as future_close,
            t1.close as base_close,
            (t2.close - t1.close) / NULLIF(t1.close, 0) * 100 as return_pct,
            (t2.date - t1.date) as days_diff
        FROM kr_intraday_total t1
        JOIN kr_intraday_total t2 ON t1.symbol = t2.symbol
        WHERE t1.date IN ('{dates_str}')
            AND t2.date > t1.date
            AND (t2.date - t1.date) BETWEEN 10 AND 65
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
        g.market_state,
        d.theme,
        p.close,
        p.market_cap,
        MAX(CASE WHEN fr.days_diff BETWEEN 10 AND 20 THEN fr.return_pct END) as return_15d,
        MAX(CASE WHEN fr.days_diff BETWEEN 25 AND 35 THEN fr.return_pct END) as return_30d,
        MAX(CASE WHEN fr.days_diff BETWEEN 40 AND 50 THEN fr.return_pct END) as return_45d,
        MAX(CASE WHEN fr.days_diff BETWEEN 55 AND 65 THEN fr.return_pct END) as return_60d
    FROM grade_data g
    LEFT JOIN price_data p ON g.symbol = p.symbol AND g.date = p.date
    LEFT JOIN kr_stock_detail d ON g.symbol = d.symbol
    LEFT JOIN future_returns fr ON g.symbol = fr.symbol AND g.date = fr.base_date
    GROUP BY g.symbol, g.date, g.stock_name, g.final_score, g.final_grade,
             g.value_score, g.quality_score, g.momentum_score, g.growth_score,
             g.market_state, d.theme, p.close, p.market_cap
    """

    rows = await conn.fetch(query)
    df = pd.DataFrame([dict(row) for row in rows])

    print(f"Loaded {len(df):,} records")
    print(f"Date distribution:")
    for d in ANALYSIS_DATES:
        count = len(df[df['date'].astype(str) == d])
        print(f"  {d}: {count:,} records")

    return df


async def calculate_strategy_scores(db_manager, df_base):
    """
    STEP 1: Calculate individual strategy scores for Momentum (M1~M23) and Value (V2, V4, V13, V25)
    With checkpoint/resume feature for long-running processes

    Phase 3.8 Update: First try to read from v2_detail columns (fast path),
    fall back to calculation if not available (slow path)
    """
    print("\n" + "="*100)
    print("STEP 1: Calculating individual strategy scores")
    print("="*100)

    # Momentum strategies (M1~M23)
    momentum_strategies = {
        'M1': 'calculate_m1',
        'M2': 'calculate_m2',
        'M3': 'calculate_m3',
        'M4': 'calculate_m4',
        'M5': 'calculate_m5',
        'M6': 'calculate_m6',
        'M7': 'calculate_m7',
        'M8': 'calculate_m8',
        'M9': 'calculate_m9',
        'M10': 'calculate_m10',
        'M11': 'calculate_m11',
        'M12': 'calculate_m12',
        'M13': 'calculate_m13',
        'M14': 'calculate_m14',
        'M15': 'calculate_m15',
        'M16': 'calculate_m16',
        'M17': 'calculate_m17',
        'M18': 'calculate_m18',
        'M19': 'calculate_m19',
        'M20': 'calculate_m20',
        'M21': 'calculate_m21',
        'M22': 'calculate_m22',
        'M23': 'calculate_m23'
    }

    # Value strategies (selected)
    value_strategies = {
        'V2_Magic_Formula': 'calculate_v2',
        'V4_Sustainable_Dividend': 'calculate_v4',
        'V13_Magic_Formula_Enhanced': 'calculate_v13',
        'V25_Cash_Rich_Undervalued': 'calculate_v25'
    }

    # Sample data for analysis
    df_sample = df_base.copy()
    df_sample['date'] = pd.to_datetime(df_sample['date']).dt.date

    # =========================================================================
    # FAST PATH: Load pre-calculated v2_detail from kr_stock_grade
    # =========================================================================
    print("\n[FAST PATH] Checking for pre-calculated v2_detail data...")

    analysis_dates = df_sample['date'].unique().tolist()
    v2_detail_cache = {}  # key: "symbol_date" -> {'momentum': {}, 'value': {}}

    try:
        v2_query = """
        SELECT symbol, date, momentum_v2_detail, value_v2_detail
        FROM kr_stock_grade
        WHERE date = ANY($1::date[])
        AND (momentum_v2_detail IS NOT NULL OR value_v2_detail IS NOT NULL)
        """
        v2_results = await db_manager.execute_query(v2_query, analysis_dates)

        if v2_results:
            for row in v2_results:
                key = f"{row['symbol']}_{row['date']}"
                momentum_detail = {}
                value_detail = {}

                if row['momentum_v2_detail']:
                    try:
                        momentum_detail = json.loads(row['momentum_v2_detail']) if isinstance(row['momentum_v2_detail'], str) else row['momentum_v2_detail']
                    except:
                        pass

                if row['value_v2_detail']:
                    try:
                        value_detail = json.loads(row['value_v2_detail']) if isinstance(row['value_v2_detail'], str) else row['value_v2_detail']
                    except:
                        pass

                if momentum_detail or value_detail:
                    v2_detail_cache[key] = {
                        'momentum': momentum_detail,
                        'value': value_detail
                    }

            print(f"  Loaded {len(v2_detail_cache):,} records with v2_detail from DB")
        else:
            print("  No v2_detail data found - will calculate all scores")
    except Exception as e:
        print(f"  Error loading v2_detail: {e} - will calculate all scores")

    # Checkpoint settings
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'phase3_7_checkpoint.csv')
    CHECKPOINT_INTERVAL = 100  # Save every 100 records

    # Check for existing checkpoint
    processed_keys = set()
    all_results = []

    # Enable graceful shutdown checkpoint saving (Ctrl+C)
    global _checkpoint_data
    _checkpoint_data['path'] = CHECKPOINT_PATH
    _checkpoint_data['results'] = all_results
    _checkpoint_data['enabled'] = True

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nCheckpoint file found: {CHECKPOINT_PATH}")
        df_checkpoint = pd.read_csv(CHECKPOINT_PATH)
        all_results = df_checkpoint.to_dict('records')
        _checkpoint_data['results'] = all_results  # Update global reference
        processed_keys = set(
            f"{row['symbol']}_{row['date']}" for row in all_results
        )
        print(f"Resuming from checkpoint: {len(all_results):,} records already processed")

    print(f"\nCalculating strategies for {len(df_sample):,} samples...")
    print("Using BATCH PARALLEL processing (20 samples per batch)...")

    async def process_sample(row):
        """Process a single sample and return results

        Phase 3.8: First try to use cached v2_detail data (fast),
        fall back to calculation (slow) if not available
        """
        symbol = row['symbol']
        analysis_date = row['date']
        cache_key = f"{symbol}_{analysis_date}"

        try:
            scores = {
                'symbol': symbol,
                'date': analysis_date
            }

            # Check if we have cached v2_detail data (FAST PATH)
            cached_data = v2_detail_cache.get(cache_key, {})
            momentum_cache = cached_data.get('momentum', {})
            value_cache = cached_data.get('value', {})

            # Use cached Momentum scores or calculate
            if momentum_cache:
                # FAST PATH: Use pre-calculated scores from DB
                for strategy_name in momentum_strategies.keys():
                    scores[strategy_name] = momentum_cache.get(strategy_name, np.nan)
            else:
                # SLOW PATH: Calculate Momentum strategies
                mom_calc = MomentumFactorCalculator(
                    symbol=symbol,
                    db_manager=db_manager,
                    market_state=row.get('market_state', '기타'),
                    analysis_date=analysis_date
                )

                for strategy_name, method_name in momentum_strategies.items():
                    try:
                        method = getattr(mom_calc, method_name)
                        score = await method()
                        scores[strategy_name] = score if score is not None else np.nan
                    except Exception:
                        scores[strategy_name] = np.nan

            # Use cached Value scores or calculate
            if value_cache:
                # FAST PATH: Use pre-calculated scores from DB
                for strategy_name in value_strategies.keys():
                    # Handle different naming conventions (V2 vs V2_Magic_Formula)
                    short_name = strategy_name.split('_')[0]  # V2, V4, V13, V25
                    scores[strategy_name] = value_cache.get(strategy_name, value_cache.get(short_name, np.nan))
            else:
                # SLOW PATH: Calculate Value strategies
                val_calc = ValueFactorCalculator(
                    symbol=symbol,
                    db_manager=db_manager,
                    market_state=row.get('market_state', '기타'),
                    analysis_date=analysis_date
                )

                for strategy_name, method_name in value_strategies.items():
                    try:
                        method = getattr(val_calc, method_name)
                        score = await method()
                        scores[strategy_name] = score if score is not None else np.nan
                    except Exception:
                        scores[strategy_name] = np.nan

            return {'success': True, 'data': scores}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # Filter out already processed rows
    rows_list = []
    for _, row in df_sample.iterrows():
        key = f"{row['symbol']}_{row['date']}"
        if key not in processed_keys:
            rows_list.append(row)

    if len(rows_list) == 0:
        print("All samples already processed from checkpoint!")
    else:
        print(f"Remaining samples to process: {len(rows_list):,}")

    # Process in batches (reduced for DB stability)
    BATCH_SIZE = 20
    total_samples = len(rows_list)
    processed = len(all_results)  # Count checkpoint records
    errors = 0
    new_processed = 0

    for batch_start in range(0, total_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_samples)
        batch_rows = rows_list[batch_start:batch_end]

        batch_results = await asyncio.gather(*[process_sample(row) for row in batch_rows])

        for result in batch_results:
            if result['success']:
                all_results.append(result['data'])
                processed += 1
                new_processed += 1
            else:
                errors += 1

        progress_pct = (batch_end / total_samples) * 100

        # Get and display cache statistics
        cache_stats = db_manager.get_cache_stats()
        print(f"  Progress: {batch_end:,}/{total_samples:,} ({progress_pct:.1f}%) - Total: {processed}, New: {new_processed}, Errors: {errors} | Cache: {cache_stats['hit_rate']} hit ({cache_stats['hits']}/{cache_stats['total']})")

        # Checkpoint save every CHECKPOINT_INTERVAL records
        if new_processed > 0 and new_processed % CHECKPOINT_INTERVAL == 0:
            df_checkpoint = pd.DataFrame(all_results)
            df_checkpoint.to_csv(CHECKPOINT_PATH, index=False, encoding='utf-8-sig')
            print(f"  [Checkpoint saved: {len(all_results):,} records]")

        # Clear cache and reset stats between batches to prevent memory bloat
        db_manager.clear_cache()
        db_manager.reset_stats()

        # Sleep between batches to reduce DB load
        if batch_end < total_samples:
            await asyncio.sleep(0.3)

    # Final checkpoint save
    if new_processed > 0:
        df_checkpoint = pd.DataFrame(all_results)
        df_checkpoint.to_csv(CHECKPOINT_PATH, index=False, encoding='utf-8-sig')
        print(f"  [Final checkpoint saved: {len(all_results):,} records]")

    print(f"\nCalculation complete: {processed} total, {new_processed} new, {errors} errors")

    # Convert results to DataFrame
    df_strategies = pd.DataFrame(all_results)

    # Merge with base data
    df_sample['date'] = df_sample['date'].astype(str)
    df_strategies['date'] = df_strategies['date'].astype(str)

    df = pd.merge(df_strategies, df_sample, on=['symbol', 'date'], how='inner')
    print(f"Merged data: {len(df):,} records")

    # Save calculated strategies
    output_path = os.path.join(OUTPUT_DIR, 'phase3_7_strategies_calculated.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_path}")

    return df, list(momentum_strategies.keys()) + list(value_strategies.keys())


async def analyze_strategy_ic(df, all_strategies):
    """
    STEP 2: Analyze IC for each strategy
    """
    print("\n" + "="*100)
    print("STEP 2: Strategy-level IC Analysis")
    print("="*100)

    results = []

    for period in RETURN_PERIODS:
        return_col = f'return_{period}d'

        if return_col not in df.columns:
            continue

        print(f"\n{'='*100}")
        print(f"IC Analysis for {period}d Returns")
        print("="*100)
        print(f"{'Strategy':<35} {'Pearson IC':>12} {'Spearman IC':>12} {'Samples':>10} {'Status'}")
        print("-"*100)

        for strategy in all_strategies:
            if strategy not in df.columns:
                continue

            ic = calculate_ic(df, strategy, return_col)

            if ic['spearman'] < -0.05:
                status = 'Strong Negative'
            elif ic['spearman'] < 0:
                status = 'Weak Negative'
            elif ic['spearman'] < 0.05:
                status = 'Neutral'
            elif ic['spearman'] < 0.10:
                status = 'Positive'
            else:
                status = 'Strong Positive'

            print(f"{strategy:<35} {ic['pearson']:>12.4f} {ic['spearman']:>12.4f} {ic['n']:>10,} {status}")

            results.append({
                'strategy': strategy,
                'period': f'{period}d',
                'pearson_ic': ic['pearson'],
                'spearman_ic': ic['spearman'],
                'n_samples': ic['n'],
                'status': status
            })

    df_results = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, 'phase3_7_strategies_ic.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_path}")

    return df_results


async def analyze_rolling_window(df, problem_strategies):
    """
    STEP 3: Rolling window validation - IC stability by date
    """
    print("\n" + "="*100)
    print(f"STEP 3: Rolling Window Validation - {len(problem_strategies)} strategies")
    print("="*100)

    if len(problem_strategies) == 0:
        print("No problem strategies found. Skipping...")
        return None

    dates = sorted(df['date'].unique())
    all_results = []

    for strategy in problem_strategies:
        if strategy not in df.columns:
            continue

        print(f"\n{strategy}:")
        print("-" * 80)
        print(f"{'Date':<15} {'IC (30d)':>12} {'Samples':>10} {'Status'}")
        print("-" * 80)

        strategy_results = []

        for date in dates:
            df_date = df[df['date'] == date]
            ic = calculate_ic(df_date, strategy, 'return_30d')

            status = 'OK' if ic['spearman'] > 0 else 'NEG'
            print(f"{date:<15} {ic['spearman']:>12.4f} {ic['n']:>10,} {status}")

            strategy_results.append({
                'strategy': strategy,
                'date': date,
                'ic_30d': ic['spearman'],
                'n_samples': ic['n']
            })

        df_strategy = pd.DataFrame(strategy_results)
        print("\nStatistics:")
        print(f"  Mean IC:     {df_strategy['ic_30d'].mean():>8.4f}")
        print(f"  Std Dev:     {df_strategy['ic_30d'].std():>8.4f}")
        print(f"  Min IC:      {df_strategy['ic_30d'].min():>8.4f}")
        print(f"  Max IC:      {df_strategy['ic_30d'].max():>8.4f}")
        print(f"  Positive %:  {(df_strategy['ic_30d'] > 0).mean() * 100:>7.1f}%")

        all_results.extend(strategy_results)

    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, 'phase3_7_rolling_window.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_path}")

    return df_results


async def analyze_decile_test(df, problem_strategies):
    """
    STEP 4: Decile test - monotonicity check (with Directional Accuracy)
    """
    print("\n" + "="*100)
    print(f"STEP 4: Decile Test (with Directional Accuracy) - {len(problem_strategies)} strategies")
    print("="*100)

    if len(problem_strategies) == 0:
        print("No problem strategies found. Skipping...")
        return None

    all_results = []

    for strategy in problem_strategies:
        if strategy not in df.columns:
            continue

        print(f"\n{'='*100}")
        print(f"{strategy}")
        print('='*100)

        df_valid = df[[strategy, 'return_30d']].dropna()

        if len(df_valid) < 100:
            print(f"Insufficient data ({len(df_valid)} samples). Skipping...")
            continue

        try:
            df_valid['decile'] = pd.qcut(df_valid[strategy], q=10, labels=False, duplicates='drop') + 1
        except ValueError as e:
            print(f"Cannot create deciles: {e}")
            continue

        decile_stats = df_valid.groupby('decile').agg({
            strategy: ['min', 'max', 'count'],
            'return_30d': ['mean', 'median', 'std']
        }).reset_index()

        decile_stats.columns = ['decile', 'score_min', 'score_max', 'count',
                                'avg_return', 'median_return', 'std_return']

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

        print("\n" + "-"*120)
        print(f"{'Decile':<10} {'Score Range':<20} {'Count':>8} {'Avg Return':>12} {'Win Rate':>10} {'Dir Acc':>10} {'Signal':>8}")
        print("-"*120)

        for _, row in decile_stats.iterrows():
            decile_label = f"D{int(row['decile'])}"
            if row['decile'] == 1:
                decile_label += " (Low)"
            elif row['decile'] == decile_stats['decile'].max():
                decile_label += " (High)"

            score_range = f"{row['score_min']:.1f} - {row['score_max']:.1f}"
            print(f"{decile_label:<10} {score_range:<20} {row['count']:>8.0f} "
                  f"{row['avg_return']:>11.2f}% {row['win_rate']:>9.1f}% "
                  f"{row['directional_accuracy']:>9.1f}% {row['signal']:>8}")

        # Monotonicity check
        print("\n" + "-"*120)
        print("Monotonicity Check:")

        correlation = stats.spearmanr(decile_stats['decile'], decile_stats['avg_return'])[0]

        if correlation > 0.7:
            print(f"  Strong positive monotonicity (rho = {correlation:.3f})")
        elif correlation > 0.3:
            print(f"  Weak positive monotonicity (rho = {correlation:.3f})")
        elif correlation > -0.3:
            print(f"  No clear pattern (rho = {correlation:.3f})")
        else:
            print(f"  REVERSED! Higher score -> Lower return (rho = {correlation:.3f})")

        if len(decile_stats) >= 2:
            long_short_spread = decile_stats.iloc[-1]['avg_return'] - decile_stats.iloc[0]['avg_return']
            print(f"  Long-Short Spread: {long_short_spread:+.2f}%")

        # Buy/Sell accuracy summary (NEW)
        buy_data = df_valid[df_valid['decile'] >= 8]['return_30d']
        sell_data = df_valid[df_valid['decile'] <= 3]['return_30d']
        buy_acc = (buy_data > 0).mean() * 100 if len(buy_data) > 0 else 0
        sell_acc = (sell_data < 0).mean() * 100 if len(sell_data) > 0 else 0
        print(f"  Buy Accuracy (D8-D10): {buy_acc:.1f}%  |  Sell Accuracy (D1-D3): {sell_acc:.1f}%")

        decile_stats['strategy'] = strategy
        all_results.append(decile_stats)

    if all_results:
        df_all_results = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'phase3_7_decile_test.csv')
        df_all_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved to: {output_path}")
        return df_all_results

    return None


async def analyze_by_market_cap(df, problem_strategies):
    """
    STEP 5: Market cap breakdown analysis
    """
    print("\n" + "="*100)
    print(f"STEP 5: Market Cap Breakdown - {len(problem_strategies)} strategies")
    print("="*100)

    if len(problem_strategies) == 0:
        print("No problem strategies found. Skipping...")
        return None

    # Categorize by market cap
    if 'market_cap' in df.columns and df['market_cap'].notna().sum() > 0:
        thresholds = df['market_cap'].quantile([0.25, 0.50, 0.75])

        def categorize_market_cap(cap):
            if pd.isna(cap):
                return 'Unknown'
            elif cap >= thresholds.iloc[2]:
                return 'LARGE'
            elif cap >= thresholds.iloc[1]:
                return 'MEDIUM'
            elif cap >= thresholds.iloc[0]:
                return 'SMALL'
            else:
                return 'MICRO'

        df['market_cap_category'] = df['market_cap'].apply(categorize_market_cap)
    else:
        print("WARNING: market_cap not available, using close price as proxy")
        thresholds = df['close'].quantile([0.25, 0.50, 0.75])

        def categorize_by_price(price):
            if pd.isna(price):
                return 'Unknown'
            elif price >= thresholds.iloc[2]:
                return 'LARGE'
            elif price >= thresholds.iloc[1]:
                return 'MEDIUM'
            elif price >= thresholds.iloc[0]:
                return 'SMALL'
            else:
                return 'MICRO'

        df['market_cap_category'] = df['close'].apply(categorize_by_price)

    all_results = []

    for strategy in problem_strategies:
        if strategy not in df.columns:
            continue

        print(f"\n{strategy}:")
        print("-" * 80)
        print(f"{'Category':<15} {'IC (30d)':>12} {'Samples':>10} {'Status'}")
        print("-" * 80)

        for category in ['LARGE', 'MEDIUM', 'SMALL', 'MICRO']:
            df_cat = df[df['market_cap_category'] == category]

            if len(df_cat) < 30:
                continue

            ic = calculate_ic(df_cat, strategy, 'return_30d')
            status = 'OK' if ic['spearman'] > 0 else 'X'

            print(f"{category:<15} {ic['spearman']:>12.4f} {ic['n']:>10,} {status}")

            all_results.append({
                'strategy': strategy,
                'category': category,
                'ic_30d': ic['spearman'],
                'n_samples': ic['n']
            })

    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, 'phase3_7_by_market_cap.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_path}")

    return df_results


async def analyze_by_theme(df, problem_strategies):
    """
    STEP 6: Theme breakdown analysis
    """
    print("\n" + "="*100)
    print(f"STEP 6: Theme Breakdown - {len(problem_strategies)} strategies")
    print("="*100)

    if len(problem_strategies) == 0:
        print("No problem strategies found. Skipping...")
        return None

    theme_counts = df['theme'].value_counts()
    valid_themes = theme_counts[theme_counts >= 30].index.tolist()

    all_results = []

    for strategy in problem_strategies:
        if strategy not in df.columns:
            continue

        print(f"\n{'='*100}")
        print(f"{strategy}")
        print('='*100)

        strategy_results = []

        print(f"\n{'Theme':<30} {'IC (30d)':>12} {'Samples':>10} {'Status'}")
        print("-" * 80)

        for theme in sorted(valid_themes):
            df_theme = df[df['theme'] == theme]

            if len(df_theme) < 30:
                continue

            ic = calculate_ic(df_theme, strategy, 'return_30d')

            if ic['spearman'] < -0.10:
                status = 'Very Negative'
            elif ic['spearman'] < 0:
                status = 'Negative'
            elif ic['spearman'] < 0.05:
                status = 'Weak'
            else:
                status = 'Positive'

            print(f"{theme:<30} {ic['spearman']:>12.4f} {ic['n']:>10,} {status}")

            strategy_results.append({
                'strategy': strategy,
                'theme': theme,
                'ic_30d': ic['spearman'],
                'n_samples': ic['n']
            })

        df_strategy = pd.DataFrame(strategy_results)
        if len(df_strategy) > 0:
            df_strategy = df_strategy.sort_values('ic_30d', ascending=True)

            print("\n" + "-"*80)
            print(f"Bottom 5 Themes (Most Negative IC):")
            print(df_strategy.head(5)[['theme', 'ic_30d', 'n_samples']].to_string(index=False))

            print("\n" + "-"*80)
            print(f"Top 5 Themes (Most Positive IC):")
            print(df_strategy.tail(5)[['theme', 'ic_30d', 'n_samples']].to_string(index=False))

        all_results.extend(strategy_results)

    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, 'phase3_7_by_theme.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_path}")

    return df_results


async def sample_problem_stocks(df, problem_strategies):
    """
    STEP 7: Sample problem stocks - High Score + Low Return vs Low Score + High Return
    """
    print("\n" + "="*100)
    print(f"STEP 7: Problem Stock Sampling - {len(problem_strategies)} strategies")
    print("="*100)

    if len(problem_strategies) == 0:
        print("No problem strategies found. Skipping...")
        return None, None

    all_problem_stocks = []
    all_paradox_stocks = []

    for strategy in problem_strategies:
        if strategy not in df.columns:
            continue

        print(f"\n{'='*100}")
        print(f"{strategy}")
        print('='*100)

        cols = ['symbol', 'stock_name', 'date', strategy, 'return_30d', 'theme']
        available_cols = [c for c in cols if c in df.columns]
        df_valid = df[available_cols].dropna()

        # Case 1: High Score but Low Return
        print("\n" + "-"*80)
        print(f"Case 1: High {strategy} (>70) but Low Return (<-5%)")
        print("-"*80)

        high_threshold = df_valid[strategy].quantile(0.8)
        problem_stocks = df_valid[
            (df_valid[strategy] > high_threshold) &
            (df_valid['return_30d'] < -5)
        ].sort_values(strategy, ascending=False).head(10)

        if len(problem_stocks) > 0:
            print(problem_stocks.to_string(index=False))
            problem_stocks['strategy'] = strategy
            all_problem_stocks.append(problem_stocks)
        else:
            print("No stocks found in this category")

        # Case 2: Low Score but High Return
        print("\n" + "-"*80)
        print(f"Case 2: Low {strategy} (<30) but High Return (>5%)")
        print("-"*80)

        low_threshold = df_valid[strategy].quantile(0.2)
        paradox_stocks = df_valid[
            (df_valid[strategy] < low_threshold) &
            (df_valid['return_30d'] > 5)
        ].sort_values('return_30d', ascending=False).head(10)

        if len(paradox_stocks) > 0:
            print(paradox_stocks.to_string(index=False))
            paradox_stocks['strategy'] = strategy
            all_paradox_stocks.append(paradox_stocks)
        else:
            print("No stocks found in this category")

    # Save results
    if all_problem_stocks:
        df_problem = pd.concat(all_problem_stocks, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'phase3_7_problem_stocks.csv')
        df_problem.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nProblem stocks saved to: {output_path}")
    else:
        df_problem = None

    if all_paradox_stocks:
        df_paradox = pd.concat(all_paradox_stocks, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'phase3_7_paradox_stocks.csv')
        df_paradox.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Paradox stocks saved to: {output_path}")
    else:
        df_paradox = None

    return df_problem, df_paradox


async def analyze_telecom_media_theme(df, all_strategies):
    """
    STEP 8: Telecom_Media theme detailed analysis
    """
    print("\n" + "="*100)
    print("STEP 8: Telecom_Media Theme Detailed Analysis")
    print("="*100)

    df_telecom = df[df['theme'] == 'Telecom_Media'].copy()

    # Convert Decimal to float for numeric columns
    numeric_cols = ['momentum_score', 'return_30d', 'return_60d', 'final_score',
                    'value_score', 'quality_score', 'growth_score']
    for col in numeric_cols:
        if col in df_telecom.columns:
            df_telecom[col] = pd.to_numeric(df_telecom[col], errors='coerce')

    if len(df_telecom) < 30:
        print(f"Insufficient Telecom_Media data ({len(df_telecom)} samples). Skipping...")
        return None

    print(f"\nTelecom_Media samples: {len(df_telecom):,}")

    all_results = []

    # 1. Strategy IC for Telecom_Media
    print("\n" + "-"*80)
    print("1. Strategy IC within Telecom_Media theme")
    print("-"*80)
    print(f"{'Strategy':<35} {'IC (30d)':>12} {'IC (60d)':>12} {'Samples':>10} {'Status'}")
    print("-"*100)

    for strategy in all_strategies:
        if strategy not in df_telecom.columns:
            continue

        ic_30 = calculate_ic(df_telecom, strategy, 'return_30d')
        ic_60 = calculate_ic(df_telecom, strategy, 'return_60d')

        if ic_30['spearman'] < -0.05:
            status = 'Negative'
        elif ic_30['spearman'] < 0.05:
            status = 'Neutral'
        else:
            status = 'Positive'

        print(f"{strategy:<35} {ic_30['spearman']:>12.4f} {ic_60['spearman']:>12.4f} {ic_30['n']:>10,} {status}")

        all_results.append({
            'analysis_type': 'strategy_ic',
            'strategy': strategy,
            'ic_30d': ic_30['spearman'],
            'ic_60d': ic_60['spearman'],
            'n_samples': ic_30['n'],
            'status': status
        })

    # 2. Date-wise IC for Telecom_Media
    print("\n" + "-"*80)
    print("2. Date-wise IC stability for Telecom_Media")
    print("-"*80)

    dates = sorted(df_telecom['date'].unique())

    # Select top momentum strategies for date analysis
    top_strategies = ['M1', 'M4', 'M7', 'M9', 'M10']

    for strategy in top_strategies:
        if strategy not in df_telecom.columns:
            continue

        print(f"\n{strategy}:")
        print(f"{'Date':<15} {'IC (30d)':>12} {'Samples':>10}")
        print("-" * 50)

        for date in dates:
            df_date = df_telecom[df_telecom['date'] == date]
            ic = calculate_ic(df_date, strategy, 'return_30d')
            print(f"{date:<15} {ic['spearman']:>12.4f} {ic['n']:>10,}")

            all_results.append({
                'analysis_type': 'date_ic',
                'strategy': strategy,
                'date': date,
                'ic_30d': ic['spearman'],
                'n_samples': ic['n']
            })

    # 3. Decile test for Telecom_Media (using momentum_score)
    print("\n" + "-"*80)
    print("3. Decile test within Telecom_Media (momentum_score)")
    print("-"*80)

    if 'momentum_score' in df_telecom.columns:
        df_valid = df_telecom[['momentum_score', 'return_30d']].dropna()

        if len(df_valid) >= 50:
            try:
                df_valid['decile'] = pd.qcut(df_valid['momentum_score'], q=5, labels=False, duplicates='drop') + 1

                decile_stats = df_valid.groupby('decile').agg({
                    'momentum_score': ['min', 'max', 'count'],
                    'return_30d': ['mean', 'median']
                }).reset_index()

                decile_stats.columns = ['quintile', 'score_min', 'score_max', 'count', 'avg_return', 'median_return']

                print(f"{'Quintile':<10} {'Score Range':<20} {'Count':>8} {'Avg Return':>12}")
                print("-"*60)

                for _, row in decile_stats.iterrows():
                    score_range = f"{row['score_min']:.1f} - {row['score_max']:.1f}"
                    print(f"Q{int(row['quintile']):<9} {score_range:<20} {row['count']:>8.0f} {row['avg_return']:>11.2f}%")

                    all_results.append({
                        'analysis_type': 'quintile',
                        'quintile': int(row['quintile']),
                        'score_min': row['score_min'],
                        'score_max': row['score_max'],
                        'count': row['count'],
                        'avg_return': row['avg_return']
                    })

            except Exception as e:
                print(f"Cannot create quintiles: {e}")

    # 4. Problem stocks within Telecom_Media
    print("\n" + "-"*80)
    print("4. Problem stocks within Telecom_Media")
    print("-"*80)

    if 'momentum_score' in df_telecom.columns:
        df_valid = df_telecom[['symbol', 'stock_name', 'date', 'momentum_score', 'return_30d']].dropna()

        # High momentum but low return
        high_mom_low_ret = df_valid[
            (df_valid['momentum_score'] > df_valid['momentum_score'].quantile(0.7)) &
            (df_valid['return_30d'] < -5)
        ].sort_values('momentum_score', ascending=False).head(10)

        print("\nHigh Momentum Score but Low Return:")
        if len(high_mom_low_ret) > 0:
            print(high_mom_low_ret.to_string(index=False))

            for _, row in high_mom_low_ret.iterrows():
                all_results.append({
                    'analysis_type': 'problem_stock',
                    'symbol': row['symbol'],
                    'stock_name': row.get('stock_name', ''),
                    'date': row['date'],
                    'momentum_score': row['momentum_score'],
                    'return_30d': row['return_30d']
                })
        else:
            print("No problem stocks found")

    # Save results
    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, 'phase3_7_telecom_media_analysis.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_path}")

    return df_results


async def main():
    """Main execution"""
    print("\n" + "="*100)
    print("Phase 3.7 - Momentum Factor Deep Dive Analysis")
    print("="*100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Dates: {', '.join(ANALYSIS_DATES)}")

    # Initialize database with query caching
    print("\nInitializing database connection with query cache...")
    raw_db_manager = AsyncDatabaseManager()
    db_manager = CachedDatabaseManager(raw_db_manager)
    await db_manager.initialize(min_size=5, max_size=20)  # Reduced pool for stability
    conn = await connect_db()
    print("Connected (Query cache enabled)")

    try:
        # Load base data
        df_base = await load_base_data(conn)

        # STEP 1: Calculate strategy scores
        df, all_strategies = await calculate_strategy_scores(db_manager, df_base)

        # STEP 2: Strategy IC analysis
        df_ic_results = await analyze_strategy_ic(df, all_strategies)

        # Identify problem strategies (IC < -0.05 for 30d)
        df_ic_30d = df_ic_results[df_ic_results['period'] == '30d']
        problem_strategies = df_ic_30d[
            df_ic_30d['spearman_ic'] < -0.05
        ]['strategy'].unique().tolist()

        print("\n" + "="*100)
        print(f"Problem strategies found: {len(problem_strategies)} (30d IC < -0.05)")
        print("="*100)

        if problem_strategies:
            print("\nProblem strategy list:")
            for strategy in problem_strategies:
                ic_value = df_ic_30d[df_ic_30d['strategy'] == strategy]['spearman_ic'].values[0]
                print(f"  - {strategy}: IC = {ic_value:.4f}")

            # STEP 3: Rolling window
            await analyze_rolling_window(df, problem_strategies)

            # STEP 4: Decile test
            await analyze_decile_test(df, problem_strategies)

            # STEP 5: Market cap breakdown
            await analyze_by_market_cap(df, problem_strategies)

            # STEP 6: Theme breakdown
            await analyze_by_theme(df, problem_strategies)

            # STEP 7: Problem stock sampling
            await sample_problem_stocks(df, problem_strategies)
        else:
            print("All strategies are healthy! (IC >= -0.05)")

        # STEP 8: Telecom_Media detailed analysis (always run)
        await analyze_telecom_media_theme(df, all_strategies)

        print("\n" + "="*100)
        print("All analyses completed successfully!")
        print("="*100)

        print(f"\nGenerated files in {OUTPUT_DIR}:")
        print("  1. phase3_7_strategies_calculated.csv - Calculated strategy scores")
        print("  2. phase3_7_strategies_ic.csv - Strategy IC results")
        if problem_strategies:
            print("  3. phase3_7_rolling_window.csv - Date-wise IC (problem strategies)")
            print("  4. phase3_7_decile_test.csv - Decile test (problem strategies)")
            print("  5. phase3_7_by_market_cap.csv - Market cap IC (problem strategies)")
            print("  6. phase3_7_by_theme.csv - Theme IC (problem strategies)")
            print("  7. phase3_7_problem_stocks.csv - Problem stocks")
            print("  8. phase3_7_paradox_stocks.csv - Paradox stocks")
        print("  9. phase3_7_telecom_media_analysis.csv - Telecom_Media detailed analysis")

    finally:
        await conn.close()
        await db_manager.close()
        print("\nDatabase connection closed")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    asyncio.run(main())
