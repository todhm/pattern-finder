"""
US Stock IC Analysis - Comprehensive Factor Analysis
=====================================================

Analysis Features:
1. Factor Level IC Analysis (Value, Quality, Momentum, Growth)
2. Rolling Window Validation (final_score + factors)
3. Decile Test (10-quantile analysis)
4. Weight Comparison (Uniform vs Dynamic)
5. Parameter Surface Analysis (threshold optimization)
6. Accuracy and Win Rate Calculation
7. Sector Analysis

Execution: python us_ic_analysis.py
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import os
import logging
import warnings
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

# Statistical packages for advanced IC analysis
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging level - WARNING and above only
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgresql+asyncpg://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')

# Output directory
OUTPUT_DIR = r'C:\project\alpha\quant\us\result'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'ic_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Date suffix for output files
DATE_SUFFIX = datetime.now().strftime('%Y%m%d')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis dates (14 dates)
ANALYSIS_DATES = [
    '2024-01-16', '2024-03-06', '2024-05-08', '2024-07-10',
    '2024-09-19', '2024-11-21', '2025-01-09', '2025-03-13',
    '2025-05-29', '2025-07-16', '2025-07-29', '2025-08-13', '2025-08-28', '2025-09-09', '2025-09-17'
]

# Return periods to analyze
RETURN_PERIODS = [3, 5, 10, 20, 30, 60, 90, 180, 252]  # days

# Parameter surface grid
BUY_THRESHOLD_RANGE = range(65, 86, 5)   # [65, 70, 75, 80, 85]
SELL_THRESHOLD_RANGE = range(35, 56, 5)  # [35, 40, 45, 50, 55]


# ============================================================================
# PART 1: DATA COLLECTION & PREPARATION
# ============================================================================

async def collect_data(pool):
    """Collect data from us_stock_grade + us_daily"""

    print("=" * 100)
    print("STEP 1: Data Collection")
    print("=" * 100)

    # Convert dates to date objects
    date_objs = [datetime.strptime(d, '%Y-%m-%d').date() for d in ANALYSIS_DATES]

    async with pool.acquire() as conn:
        # Get all us_stock_grade data for analysis dates
        print("\n[1/3] Querying us_stock_grade...")
        grade_query = """
            SELECT
                symbol, stock_name, date,
                final_score, final_grade,
                value_score, quality_score, momentum_score, growth_score,
                confidence_score, market_state,
                beta, volatility_annual, var_95,
                -- 검증용 추가 컬럼
                entry_timing_score,
                stop_loss_pct, take_profit_pct, risk_reward_ratio,
                position_size_pct, atr_pct,
                scenario_bullish_prob, scenario_sideways_prob, scenario_bearish_prob,
                scenario_bullish_return, scenario_sideways_return, scenario_bearish_return,
                buy_triggers, sell_triggers,
                score_trend_2w, price_position_52w,
                conviction_score, interaction_score
            FROM us_stock_grade
            WHERE date = ANY($1::date[])
            ORDER BY date, symbol
        """
        grade_data = await conn.fetch(grade_query, date_objs)
        df_grade = pd.DataFrame(grade_data, columns=[
            'symbol', 'stock_name', 'date',
            'final_score', 'final_grade',
            'value_score', 'quality_score', 'momentum_score', 'growth_score',
            'confidence_score', 'market_state',
            'beta', 'volatility_annual', 'var_95',
            # 검증용 추가 컬럼
            'entry_timing_score',
            'stop_loss_pct', 'take_profit_pct', 'risk_reward_ratio',
            'position_size_pct', 'atr_pct',
            'scenario_bullish_prob', 'scenario_sideways_prob', 'scenario_bearish_prob',
            'scenario_bullish_return', 'scenario_sideways_return', 'scenario_bearish_return',
            'buy_triggers', 'sell_triggers',
            'score_trend_2w', 'price_position_52w',
            'conviction_score', 'interaction_score'
        ])

        # Convert Decimal to float for numeric columns
        numeric_cols = ['final_score', 'value_score', 'quality_score', 'momentum_score',
                        'growth_score', 'confidence_score', 'beta', 'volatility_annual', 'var_95',
                        'entry_timing_score', 'stop_loss_pct', 'take_profit_pct', 'risk_reward_ratio',
                        'position_size_pct', 'atr_pct', 'scenario_bullish_prob', 'scenario_sideways_prob',
                        'scenario_bearish_prob', 'score_trend_2w', 'price_position_52w',
                        'conviction_score', 'interaction_score']
        for col in numeric_cols:
            if col in df_grade.columns:
                df_grade[col] = pd.to_numeric(df_grade[col], errors='coerce')

        print(f"   us_stock_grade: {len(df_grade):,} records")

        # Get us_stock_basic for sector/industry
        print("\n[2/3] Querying us_stock_basic...")
        basic_query = """
            SELECT symbol, sector, industry, exchange
            FROM us_stock_basic
        """
        basic_data = await conn.fetch(basic_query)
        df_basic = pd.DataFrame(basic_data, columns=['symbol', 'sector', 'industry', 'exchange'])
        print(f"   us_stock_basic: {len(df_basic):,} records")

        # Merge sector/industry info
        df_grade = df_grade.merge(df_basic, on='symbol', how='left')

        # Get price data for return calculation
        print("\n[3/3] Querying us_daily price data...")

        # Calculate date range for price data (need +300 days buffer for 252-day analysis)
        min_date = min(date_objs)
        max_date = max(date_objs) + timedelta(days=300)

        # Get unique symbols from df_grade to reduce query size
        symbols_in_grade = df_grade['symbol'].unique().tolist()
        print(f"   Filtering for {len(symbols_in_grade):,} symbols from us_stock_grade...")

        price_query = """
            SELECT symbol, date, close
            FROM us_daily
            WHERE date >= $1 AND date <= $2
                AND symbol = ANY($3)
            ORDER BY date, symbol
        """
        price_data = await conn.fetch(price_query, min_date, max_date, symbols_in_grade)
        df_price = pd.DataFrame(price_data, columns=['symbol', 'date', 'close'])
        print(f"   us_daily: {len(df_price):,} records")

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


def winsorize_returns(returns, lower_percentile=5, upper_percentile=95):
    """
    Winsorize returns to reduce outlier impact

    Args:
        returns: numpy array of returns
        lower_percentile: lower bound percentile (default 5%)
        upper_percentile: upper bound percentile (default 95%)

    Returns:
        winsorized returns array
    """
    lower_bound = np.percentile(returns, lower_percentile)
    upper_bound = np.percentile(returns, upper_percentile)
    return np.clip(returns, lower_bound, upper_bound)


def calculate_winsorized_ic(df, score_col, return_cols, lower_pct=5, upper_pct=95):
    """Calculate Winsorized IC (outlier-adjusted)"""

    results = {}

    for ret_col in return_cols:
        valid_data = df[[score_col, ret_col]].dropna()

        if len(valid_data) < 10:
            results[ret_col] = {
                'winsorized_pearson_ic': None,
                'winsorized_spearman_ic': None,
                'n_samples': len(valid_data)
            }
            continue

        score_values = valid_data[score_col].astype(float).values
        return_values = valid_data[ret_col].astype(float).values

        # Winsorize returns
        winsorized_returns = winsorize_returns(return_values, lower_pct, upper_pct)

        pearson_ic, pearson_p = pearsonr(score_values, winsorized_returns)
        spearman_ic, spearman_p = spearmanr(score_values, winsorized_returns)

        results[ret_col] = {
            'winsorized_pearson_ic': pearson_ic,
            'winsorized_spearman_ic': spearman_ic,
            'n_samples': len(valid_data),
            'lower_bound': np.percentile(return_values, lower_pct),
            'upper_bound': np.percentile(return_values, upper_pct)
        }

    return results


# ============================================================================
# PART 2B: ADVANCED IC METHODS (Newey-West, Cluster-Robust, Panel Regression)
# ============================================================================

def calculate_ic_newey_west(scores: np.ndarray, returns: np.ndarray,
                            max_lags: int = None) -> Dict:
    """
    Calculate IC with Newey-West robust standard errors

    Accounts for:
    - Heteroskedasticity
    - Autocorrelation in residuals

    Args:
        scores: Factor scores
        returns: Forward returns
        max_lags: Max lags for HAC. If None, uses automatic selection

    Returns:
        dict with IC, t-stat (NW), p-value, SE
    """
    # Convert to float64 to handle Decimal/object types from DB
    scores = np.asarray(scores, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)

    valid_mask = ~(np.isnan(scores) | np.isnan(returns))
    scores = scores[valid_mask]
    returns = returns[valid_mask]

    if len(scores) < 10:
        return {
            'ic': None, 't_stat_nw': None, 'p_value_nw': None,
            'se_nw': None, 'n_samples': len(scores)
        }

    # Standardize for better numerical stability
    scores_std = (scores - scores.mean()) / scores.std()
    returns_std = (returns - returns.mean()) / returns.std()

    # OLS regression: return = alpha + beta * score
    X = sm.add_constant(scores_std)

    # Auto-select lags if not specified: Newey-West rule
    if max_lags is None:
        T = len(scores)
        max_lags = int(4 * (T / 100) ** (2/9))

    try:
        model = OLS(returns_std, X).fit(cov_type='HAC',
                                         cov_kwds={'maxlags': max_lags})

        # Beta coefficient is the IC (correlation in standardized form)
        ic = model.params[1]
        t_stat = model.tvalues[1]
        p_value = model.pvalues[1]
        se = model.bse[1]

        # Also calculate raw Spearman IC for reference
        spearman_ic, _ = spearmanr(scores, returns)

        return {
            'ic': ic,
            'spearman_ic': spearman_ic,
            't_stat_nw': t_stat,
            'p_value_nw': p_value,
            'se_nw': se,
            'n_samples': len(scores),
            'max_lags': max_lags
        }
    except Exception as e:
        logger.error(f"Newey-West calculation error: {e}")
        return {
            'ic': None, 't_stat_nw': None, 'p_value_nw': None,
            'se_nw': None, 'n_samples': len(scores)
        }


def calculate_ic_cluster_robust(df: pd.DataFrame, score_col: str, return_col: str,
                                cluster_col: str = 'sector') -> Dict:
    """
    Calculate IC with cluster-robust standard errors

    Clusters by sector to account for:
    - US market sector synchronization
    - Sector concentration effects

    Args:
        df: DataFrame with scores, returns, and cluster variable
        score_col: Column name for factor scores
        return_col: Column name for returns
        cluster_col: Column for clustering (default: sector)

    Returns:
        dict with IC, cluster-robust t-stat, p-value
    """
    valid_data = df[[score_col, return_col, cluster_col]].dropna()

    if len(valid_data) < 20:
        return {
            'ic': None, 't_stat_cluster': None, 'p_value_cluster': None,
            'n_samples': len(valid_data), 'n_clusters': 0
        }

    # Convert to float64 to handle Decimal/object types from DB
    scores = np.asarray(valid_data[score_col].values, dtype=np.float64)
    returns = np.asarray(valid_data[return_col].values, dtype=np.float64)
    clusters = valid_data[cluster_col].values

    # Standardize
    scores_std = (scores - scores.mean()) / scores.std()
    returns_std = (returns - returns.mean()) / returns.std()

    X = sm.add_constant(scores_std)

    try:
        # Fit with cluster-robust SE
        model = OLS(returns_std, X).fit(cov_type='cluster',
                                         cov_kwds={'groups': clusters})

        ic = model.params[1]
        t_stat = model.tvalues[1]
        p_value = model.pvalues[1]

        # Spearman IC
        spearman_ic, _ = spearmanr(scores, returns)

        n_clusters = len(np.unique(clusters))

        return {
            'ic': ic,
            'spearman_ic': spearman_ic,
            't_stat_cluster': t_stat,
            'p_value_cluster': p_value,
            'n_samples': len(valid_data),
            'n_clusters': n_clusters
        }
    except Exception as e:
        logger.error(f"Cluster-robust calculation error: {e}")
        return {
            'ic': None, 't_stat_cluster': None, 'p_value_cluster': None,
            'n_samples': len(valid_data), 'n_clusters': 0
        }


def enhanced_ic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run enhanced IC analysis with Newey-West and Cluster-Robust SE
    """
    print("\n" + "=" * 100)
    print("PHASE H: Enhanced IC Analysis (Newey-West + Cluster Robust)")
    print("=" * 100)

    return_cols = [f'return_{period}d' for period in RETURN_PERIODS]
    results = []

    score_columns = {
        'final_score': 'Final Score',
        'value_score': 'Value Factor',
        'quality_score': 'Quality Factor',
        'momentum_score': 'Momentum Factor',
        'growth_score': 'Growth Factor',
    }

    print(f"\n{'Factor':<20} {'Period':<8} {'Spearman IC':>12} {'NW t-stat':>12} {'NW p-val':>10} {'Cluster t':>12} {'Signif':>8}")
    print("-" * 100)

    for score_col, score_name in score_columns.items():
        if score_col not in df.columns:
            continue

        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'

            if ret_col not in df.columns:
                continue

            valid_data = df[[score_col, ret_col, 'sector']].dropna()

            if len(valid_data) < 20:
                continue

            scores = valid_data[score_col].values
            returns = valid_data[ret_col].values

            # Newey-West IC
            nw_result = calculate_ic_newey_west(scores, returns)

            # Cluster-Robust IC
            cluster_result = calculate_ic_cluster_robust(valid_data, score_col, ret_col, 'sector')

            # Determine significance
            signif = ""
            if nw_result['p_value_nw'] is not None:
                if nw_result['p_value_nw'] < 0.01:
                    signif = "***"
                elif nw_result['p_value_nw'] < 0.05:
                    signif = "**"
                elif nw_result['p_value_nw'] < 0.10:
                    signif = "*"

            result_row = {
                'factor': score_name,
                'period': period,
                'spearman_ic': nw_result.get('spearman_ic'),
                'ic_nw': nw_result.get('ic'),
                't_stat_nw': nw_result.get('t_stat_nw'),
                'p_value_nw': nw_result.get('p_value_nw'),
                'se_nw': nw_result.get('se_nw'),
                't_stat_cluster': cluster_result.get('t_stat_cluster'),
                'p_value_cluster': cluster_result.get('p_value_cluster'),
                'n_samples': nw_result.get('n_samples'),
                'n_clusters': cluster_result.get('n_clusters'),
                'significance': signif
            }
            results.append(result_row)

            # Print row
            sp_ic = nw_result.get('spearman_ic', 0) or 0
            t_nw = nw_result.get('t_stat_nw', 0) or 0
            p_nw = nw_result.get('p_value_nw', 1) or 1
            t_cl = cluster_result.get('t_stat_cluster', 0) or 0

            print(f"{score_name:<20} {period:>3}d     {sp_ic:>+12.4f} {t_nw:>+12.3f} {p_nw:>10.4f} {t_cl:>+12.3f} {signif:>8}")

    df_results = pd.DataFrame(results)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, f'us_enhanced_ic_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  -> Saved to {output_path}")

    return df_results


def ic_stability_analysis(df: pd.DataFrame) -> Dict:
    """
    Comprehensive IC stability analysis with visualizations
    """
    print("\n" + "=" * 100)
    print("PHASE I: IC Stability Analysis")
    print("=" * 100)

    try:
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False
        HAS_MATPLOTLIB = True
    except ImportError:
        print("  [WARNING] matplotlib not available. Skipping plots.")
        HAS_MATPLOTLIB = False

    score_columns = ['final_score', 'value_score', 'quality_score', 'momentum_score', 'growth_score']
    stability_results = {}

    # Analyze each factor
    for factor in score_columns:
        if factor not in df.columns:
            continue

        print(f"\n  Analyzing {factor}...")

        factor_stability = {}

        for period in [30, 60, 90, 180]:  # Focus on medium/long-term
            ret_col = f'return_{period}d'

            if ret_col not in df.columns:
                continue

            # Calculate date-wise IC
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'])

            date_ics = []
            for analysis_date in df_temp['date'].unique():
                date_data = df_temp[df_temp['date'] == analysis_date]
                valid = date_data[[factor, ret_col]].dropna()

                if len(valid) >= 10:
                    ic, _ = spearmanr(valid[factor], valid[ret_col])
                    date_ics.append({
                        'date': analysis_date,
                        'ic': ic,
                        'n': len(valid)
                    })

            if not date_ics:
                continue

            df_ic = pd.DataFrame(date_ics)
            df_ic = df_ic.sort_values('date')

            # IC Statistics
            ic_mean = df_ic['ic'].mean()
            ic_std = df_ic['ic'].std()
            ic_min = df_ic['ic'].min()
            ic_max = df_ic['ic'].max()

            # IC Autocorrelation (AR1)
            if len(df_ic) > 2:
                ic_ar1 = df_ic['ic'].autocorr(lag=1)
            else:
                ic_ar1 = None

            # Cumulative IC (for drift detection)
            df_ic['cumulative_ic'] = df_ic['ic'].cumsum()

            # Trend (simple linear regression)
            if len(df_ic) >= 3:
                x = np.arange(len(df_ic))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_ic['ic'])
                ic_trend = slope
                ic_trend_pval = p_value
            else:
                ic_trend = None
                ic_trend_pval = None

            # Store results
            factor_stability[f'{period}d'] = {
                'mean_ic': ic_mean,
                'std_ic': ic_std,
                'min_ic': ic_min,
                'max_ic': ic_max,
                'ar1': ic_ar1,
                'trend_slope': ic_trend,
                'trend_pval': ic_trend_pval,
                'n_periods': len(df_ic),
                'ic_series': df_ic
            }

            # Print summary
            ar1_str = f"{ic_ar1:.3f}" if ic_ar1 is not None else "N/A"
            trend_str = f"{ic_trend:.4f}" if ic_trend is not None else "N/A"

            print(f"    {period}d: Mean IC={ic_mean:+.4f}, Std={ic_std:.4f}, "
                  f"AR(1)={ar1_str}, Trend={trend_str}")

            # Detect potential issues
            if ic_mean < -0.02:
                print(f"    [WARNING] Negative mean IC!")
            if ic_trend is not None and ic_trend < -0.01:
                print(f"    [WARNING] Negative IC drift detected!")
            if ic_ar1 is not None and abs(ic_ar1) > 0.5:
                print(f"    [INFO] High IC autocorrelation")

        stability_results[factor] = factor_stability

    # Generate plots if matplotlib available
    if HAS_MATPLOTLIB and stability_results:
        print("\n  Generating IC stability plots...")

        fig, axes = plt.subplots(len(score_columns), 3, figsize=(18, 4*len(score_columns)))

        for i, factor in enumerate(score_columns):
            if factor not in stability_results:
                continue

            # Use 60d as primary period
            period_key = '60d' if '60d' in stability_results[factor] else '30d'

            if period_key not in stability_results[factor]:
                continue

            data = stability_results[factor][period_key]
            df_ic = data['ic_series']

            # Plot 1: IC Time Series
            ax1 = axes[i, 0] if len(score_columns) > 1 else axes[0]
            ax1.bar(range(len(df_ic)), df_ic['ic'], color=['green' if x > 0 else 'red' for x in df_ic['ic']])
            ax1.axhline(0, color='black', linewidth=0.5)
            ax1.axhline(data['mean_ic'], color='blue', linestyle='--', label=f"Mean: {data['mean_ic']:.4f}")
            ax1.fill_between(range(len(df_ic)),
                           data['mean_ic'] - 2*data['std_ic'],
                           data['mean_ic'] + 2*data['std_ic'],
                           alpha=0.2, color='blue', label='95% CI')
            ax1.set_title(f'{factor} - IC Time Series ({period_key})')
            ax1.set_xlabel('Date Index')
            ax1.set_ylabel('IC')
            ax1.legend(loc='upper right', fontsize=8)

            # Plot 2: Cumulative IC
            ax2 = axes[i, 1] if len(score_columns) > 1 else axes[1]
            ax2.plot(df_ic['cumulative_ic'], color='blue', linewidth=2)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_title(f'{factor} - Cumulative IC (Drift Detection)')
            ax2.set_xlabel('Date Index')
            ax2.set_ylabel('Cumulative IC')

            # Add trend line
            if data['trend_slope'] is not None:
                x_trend = np.arange(len(df_ic))
                y_trend = data['trend_slope'] * x_trend * len(df_ic)
                ax2.plot(y_trend, color='red', linestyle='--', alpha=0.7,
                        label=f"Trend: {data['trend_slope']:+.4f}/period")
                ax2.legend(loc='upper left', fontsize=8)

            # Plot 3: IC Distribution
            ax3 = axes[i, 2] if len(score_columns) > 1 else axes[2]
            ax3.hist(df_ic['ic'], bins=min(20, len(df_ic)), edgecolor='black', alpha=0.7)
            ax3.axvline(data['mean_ic'], color='red', linewidth=2, label=f"Mean: {data['mean_ic']:.4f}")
            ax3.axvline(0, color='black', linewidth=1)
            ax3.set_title(f'{factor} - IC Distribution')
            ax3.set_xlabel('IC')
            ax3.set_ylabel('Frequency')
            ax3.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f'ic_stability_plots_{DATE_SUFFIX}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved plots to {plot_path}")

    return stability_results


def panel_regression_ic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate IC using Panel Regression with Two-Way Fixed Effects

    Model: Return_it = alpha + beta * Score_it + gamma_i + delta_t + epsilon_it

    Where:
    - gamma_i: Entity (firm) fixed effect
    - delta_t: Time fixed effect
    - beta: True IC controlling for firm and time effects
    """
    print("\n" + "=" * 100)
    print("PHASE J: Panel Regression IC (Two-Way Fixed Effects)")
    print("=" * 100)

    # Check for linearmodels
    try:
        from linearmodels.panel import PanelOLS
        HAS_LINEARMODELS = True
    except ImportError:
        print("  [WARNING] linearmodels package not available.")
        print("  Install with: pip install linearmodels")
        print("  Falling back to simplified entity-demeaned regression.")
        HAS_LINEARMODELS = False

    results = []

    # Prepare panel data
    df_panel = df.copy()
    df_panel['date'] = pd.to_datetime(df_panel['date'])

    # Need symbol and date as index
    df_panel = df_panel.dropna(subset=['symbol', 'date'])

    score_columns = ['final_score', 'value_score', 'quality_score', 'momentum_score', 'growth_score']

    print(f"\n{'Factor':<20} {'Period':<8} {'Panel IC':>12} {'t-stat':>12} {'p-value':>10} {'R2':>10}")
    print("-" * 80)

    for factor in score_columns:
        if factor not in df_panel.columns:
            continue

        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'

            if ret_col not in df_panel.columns:
                continue

            # Prepare data
            work_df = df_panel[['symbol', 'date', factor, ret_col, 'sector']].dropna().copy()

            # Convert Decimal to float64 for numeric operations
            work_df[factor] = pd.to_numeric(work_df[factor], errors='coerce').astype(np.float64)
            work_df[ret_col] = pd.to_numeric(work_df[ret_col], errors='coerce').astype(np.float64)
            work_df = work_df.dropna(subset=[factor, ret_col])

            if len(work_df) < 50:
                continue

            # Number of unique entities and times
            n_entities = work_df['symbol'].nunique()
            n_times = work_df['date'].nunique()

            if n_entities < 5 or n_times < 2:
                continue

            if HAS_LINEARMODELS:
                try:
                    # Set multi-index for panel
                    panel_df = work_df.set_index(['symbol', 'date'])

                    # Standardize
                    panel_df[factor] = (panel_df[factor] - panel_df[factor].mean()) / panel_df[factor].std()
                    panel_df[ret_col] = (panel_df[ret_col] - panel_df[ret_col].mean()) / panel_df[ret_col].std()

                    # Two-way FE
                    model = PanelOLS(panel_df[ret_col],
                                    panel_df[[factor]],
                                    entity_effects=True,
                                    time_effects=True)

                    # Cluster by entity
                    result = model.fit(cov_type='clustered', cluster_entity=True)

                    panel_ic = result.params[factor]
                    t_stat = result.tstats[factor]
                    p_value = result.pvalues[factor]
                    r2 = result.rsquared_within

                except Exception as e:
                    logger.warning(f"Panel regression error for {factor}/{period}d: {e}")
                    continue
            else:
                # Simplified: Entity-demeaned regression
                work_df['score_demeaned'] = work_df.groupby('symbol')[factor].transform(
                    lambda x: x - x.mean())
                work_df['return_demeaned'] = work_df.groupby('symbol')[ret_col].transform(
                    lambda x: x - x.mean())

                # Also demean by time
                work_df['score_demeaned'] = work_df.groupby('date')['score_demeaned'].transform(
                    lambda x: x - x.mean())
                work_df['return_demeaned'] = work_df.groupby('date')['return_demeaned'].transform(
                    lambda x: x - x.mean())

                X = sm.add_constant(work_df['score_demeaned'])
                model = OLS(work_df['return_demeaned'], X).fit(cov_type='cluster',
                                                                cov_kwds={'groups': work_df['sector']})

                panel_ic = model.params['score_demeaned']
                t_stat = model.tvalues['score_demeaned']
                p_value = model.pvalues['score_demeaned']
                r2 = model.rsquared

            # Significance
            signif = ""
            if p_value < 0.01:
                signif = "***"
            elif p_value < 0.05:
                signif = "**"
            elif p_value < 0.10:
                signif = "*"

            results.append({
                'factor': factor,
                'period': period,
                'panel_ic': panel_ic,
                't_stat': t_stat,
                'p_value': p_value,
                'r2_within': r2,
                'n_samples': len(work_df),
                'n_entities': n_entities,
                'n_times': n_times,
                'significance': signif
            })

            print(f"{factor:<20} {period:>3}d     {panel_ic:>+12.4f} {t_stat:>+12.3f} {p_value:>10.4f} {r2:>10.4f} {signif}")

    df_results = pd.DataFrame(results)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, f'us_panel_ic_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  -> Saved to {output_path}")

    return df_results


def generate_summary_report(df_enhanced: pd.DataFrame, df_panel: pd.DataFrame,
                           stability_results: Dict) -> None:
    """Generate comprehensive summary report comparing IC methods"""

    print("\n" + "=" * 100)
    print("PHASE K: Summary Report - IC Method Comparison")
    print("=" * 100)

    print("\n### IC Method Comparison ###\n")

    score_columns = ['final_score', 'value_score', 'quality_score', 'momentum_score', 'growth_score']

    # Compare methods for key factors
    print(f"{'Factor':<20} {'Period':<8} {'Simple IC':>12} {'NW t-stat':>12} {'Panel IC':>12} {'Panel t':>12}")
    print("-" * 90)

    for factor in score_columns:
        for period in [30, 60, 90, 180]:
            # Get enhanced IC results
            enhanced_row = df_enhanced[(df_enhanced['factor'] == factor) &
                                       (df_enhanced['period'] == period)]

            # Get panel IC results
            panel_row = df_panel[(df_panel['factor'] == factor) &
                                (df_panel['period'] == period)]

            if len(enhanced_row) == 0:
                continue

            simple_ic = enhanced_row.iloc[0]['spearman_ic'] if len(enhanced_row) > 0 else None
            nw_t = enhanced_row.iloc[0]['t_stat_nw'] if len(enhanced_row) > 0 else None
            panel_ic = panel_row.iloc[0]['panel_ic'] if len(panel_row) > 0 else None
            panel_t = panel_row.iloc[0]['t_stat'] if len(panel_row) > 0 else None

            simple_str = f"{simple_ic:+.4f}" if simple_ic is not None else "N/A"
            nw_str = f"{nw_t:+.3f}" if nw_t is not None else "N/A"
            panel_ic_str = f"{panel_ic:+.4f}" if panel_ic is not None else "N/A"
            panel_t_str = f"{panel_t:+.3f}" if panel_t is not None else "N/A"

            print(f"{factor:<20} {period:>3}d     {simple_str:>12} {nw_str:>12} {panel_ic_str:>12} {panel_t_str:>12}")

    # IC Stability Summary
    print("\n\n### IC Stability Summary ###\n")
    print(f"{'Factor':<20} {'60d Mean IC':>12} {'IC Std':>10} {'AR(1)':>10} {'Trend':>12} {'Status':<15}")
    print("-" * 85)

    for factor in score_columns:
        if factor not in stability_results:
            continue

        if '60d' in stability_results[factor]:
            data = stability_results[factor]['60d']
        elif '30d' in stability_results[factor]:
            data = stability_results[factor]['30d']
        else:
            continue

        mean_ic = data['mean_ic']
        std_ic = data['std_ic']
        ar1 = data['ar1']
        trend = data['trend_slope']

        # Determine status
        status = "OK"
        if mean_ic < -0.01:
            status = "NEGATIVE IC"
        elif trend is not None and trend < -0.005:
            status = "DRIFTING DOWN"
        elif std_ic > 0.1:
            status = "UNSTABLE"

        ar1_str = f"{ar1:.3f}" if ar1 is not None else "N/A"
        trend_str = f"{trend:+.4f}" if trend is not None else "N/A"

        print(f"{factor:<20} {mean_ic:>+12.4f} {std_ic:>10.4f} {ar1_str:>10} {trend_str:>12} {status:<15}")

    print("\n" + "=" * 100)
    print("Advanced IC Analysis Complete")
    print("=" * 100)


def calculate_directional_accuracy(decile, returns):
    """
    Calculate directional accuracy based on decile

    Args:
        decile: 1-10 (D1-D3: sell signal, D8-D10: buy signal)
        returns: array of returns

    Returns:
        accuracy: correct prediction ratio (%)
    """
    if decile <= 3:  # D1-D3: Sell signal - negative return is correct
        return (returns < 0).mean() * 100
    elif decile >= 8:  # D8-D10: Buy signal - positive return is correct
        return (returns > 0).mean() * 100
    else:  # D4-D7: Neutral
        return 50.0


def ic_analysis_factor_level(df):
    """STEP 3: Factor Level IC Analysis"""

    print("\n" + "=" * 100)
    print("STEP 3: Factor Level IC Analysis")
    print("=" * 100)

    return_cols = [f'return_{period}d' for period in RETURN_PERIODS]
    ic_results = {}

    score_columns = {
        'final_score': 'Final Score',
        'value_score': 'Value Factor',
        'quality_score': 'Quality Factor',
        'momentum_score': 'Momentum Factor',
        'growth_score': 'Growth Factor',
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
    output_file = os.path.join(OUTPUT_DIR, f'us_factor_ic_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return ic_results


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

    output_file = os.path.join(OUTPUT_DIR, f'us_rolling_ic_{DATE_SUFFIX}.csv')
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
            if decile <= 3:
                signal = 'SELL'
            elif decile >= 8:
                signal = 'BUY'
            else:
                signal = 'HOLD'

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
    output_file = os.path.join(OUTPUT_DIR, f'us_decile_test_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# PART 4: WEIGHT COMPARISON & PARAMETER ANALYSIS
# ============================================================================

def weight_comparison_analysis(df):
    """STEP 6: Weight Comparison Analysis (Uniform vs Dynamic)"""

    print("\n" + "=" * 100)
    print("STEP 6: Weight Comparison Analysis (Uniform vs Dynamic)")
    print("=" * 100)

    results = []

    # Calculate uniform weighted score
    factor_cols = ['value_score', 'quality_score', 'momentum_score', 'growth_score']
    if not all(col in df.columns for col in factor_cols):
        print("[WARNING] Factor columns missing. Skipping weight comparison...")
        return pd.DataFrame()

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

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, f'us_weight_comparison_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def parameter_surface_analysis(df):
    """STEP 7: Parameter Surface Analysis (Threshold Optimization)"""

    print("\n" + "=" * 100)
    print("STEP 7: Parameter Surface Analysis (Threshold Optimization)")
    print("=" * 100)

    results = []

    # Use 20-day return for optimization
    ret_col = 'return_20d'
    if ret_col not in df.columns:
        ret_col = 'return_30d'  # Fallback
    if ret_col not in df.columns:
        print(f"[WARNING] Return column not found. Skipping parameter surface...")
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
    output_file = os.path.join(OUTPUT_DIR, f'us_parameter_surface_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def accuracy_winrate_analysis(df):
    """STEP 8: Accuracy and Win Rate Calculation"""

    print("\n" + "=" * 100)
    print("STEP 8: Accuracy and Win Rate Analysis")
    print("=" * 100)

    results = []

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
            if grade in ['강력 매수', '매수']:
                accuracy = (grade_data[ret_col] > 0).mean() * 100
            elif grade in ['매도', '강력 매도']:
                accuracy = (grade_data[ret_col] < 0).mean() * 100
            else:
                accuracy = 50.0  # Neutral has no directional prediction

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
    output_file = os.path.join(OUTPUT_DIR, f'us_accuracy_winrate_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# PART 5: SECTOR ANALYSIS
# ============================================================================

def sector_analysis(df):
    """STEP 9: Sector Analysis"""

    print("\n" + "=" * 100)
    print("STEP 9: Sector Analysis")
    print("=" * 100)

    longest_period = max(RETURN_PERIODS)
    longest_ret = f'return_{longest_period}d'

    if longest_ret not in df.columns or 'sector' not in df.columns:
        print("[WARNING] Required columns missing. Skipping sector analysis...")
        return pd.DataFrame()

    sectors = df['sector'].dropna().unique()
    sector_results = []

    for sector in sorted(sectors):
        sector_data = df[df['sector'] == sector]

        if len(sector_data) < 30:
            continue

        ic_result = calculate_ic(sector_data, 'final_score', [longest_ret])

        if longest_ret in ic_result and ic_result[longest_ret]['spearman_ic'] is not None:
            ic = ic_result[longest_ret]['spearman_ic']
            n = ic_result[longest_ret]['n_samples']
            avg_return = sector_data[longest_ret].mean()
            win_rate = (sector_data[longest_ret] > 0).mean() * 100

            sector_results.append({
                'sector': sector,
                'ic': ic,
                'n_samples': n,
                'avg_return': avg_return,
                'win_rate': win_rate
            })

    df_results = pd.DataFrame(sector_results)
    df_results = df_results.sort_values('ic', ascending=False)

    print(f"\nSector Performance (by IC, {longest_period}d return):")
    print("-" * 100)
    print(f"{'Sector':<30} {'IC':>10} {'Samples':>10} {'Avg Ret':>10} {'Win Rate':>10}")
    print("-" * 100)
    for _, row in df_results.iterrows():
        print(f"{str(row['sector']):<30} {row['ic']:>10.4f} {row['n_samples']:>10,} "
              f"{row['avg_return']:>+9.2f}% {row['win_rate']:>9.1f}%")

    output_file = os.path.join(OUTPUT_DIR, f'us_sector_analysis_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def exchange_analysis(df):
    """STEP 10: Exchange Analysis (NYSE vs NASDAQ)"""

    print("\n" + "=" * 100)
    print("STEP 10: Exchange Analysis (NYSE vs NASDAQ)")
    print("=" * 100)

    longest_period = max(RETURN_PERIODS)
    longest_ret = f'return_{longest_period}d'

    if longest_ret not in df.columns or 'exchange' not in df.columns:
        print("[WARNING] Required columns missing. Skipping exchange analysis...")
        return pd.DataFrame()

    exchanges = df['exchange'].dropna().unique()
    exchange_results = []

    print(f"\nExchange Performance (by IC, {longest_period}d return):")
    print("-" * 100)
    print(f"{'Exchange':<15} {'IC':>10} {'Samples':>10} {'Avg Ret':>10} {'Win Rate':>10}")
    print("-" * 100)

    for exchange in sorted(exchanges):
        exchange_data = df[df['exchange'] == exchange]

        if len(exchange_data) < 30:
            continue

        ic_result = calculate_ic(exchange_data, 'final_score', [longest_ret])

        if longest_ret in ic_result and ic_result[longest_ret]['spearman_ic'] is not None:
            ic = ic_result[longest_ret]['spearman_ic']
            n = ic_result[longest_ret]['n_samples']
            avg_return = exchange_data[longest_ret].mean()
            win_rate = (exchange_data[longest_ret] > 0).mean() * 100

            print(f"{str(exchange):<15} {ic:>10.4f} {n:>10,} {avg_return:>+9.2f}% {win_rate:>9.1f}%")

            exchange_results.append({
                'exchange': exchange,
                'ic': ic,
                'n_samples': n,
                'avg_return': avg_return,
                'win_rate': win_rate
            })

    df_results = pd.DataFrame(exchange_results)
    output_file = os.path.join(OUTPUT_DIR, f'us_exchange_analysis_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def exchange_factor_analysis(df):
    """STEP 10B: Exchange-Specific Factor IC Analysis"""

    print("\n" + "=" * 100)
    print("STEP 10B: Exchange-Specific Factor IC Analysis (NYSE vs NASDAQ)")
    print("=" * 100)

    if 'exchange' not in df.columns:
        print("[WARNING] Exchange column missing. Skipping exchange factor analysis...")
        return pd.DataFrame()

    score_columns = {
        'final_score': 'Final Score',
        'value_score': 'Value Factor',
        'quality_score': 'Quality Factor',
        'momentum_score': 'Momentum Factor',
        'growth_score': 'Growth Factor',
    }

    all_results = []

    # Main exchanges to analyze
    main_exchanges = ['NYSE', 'NASDAQ']

    for score_col, score_name in score_columns.items():
        if score_col not in df.columns:
            print(f"\n[WARNING] {score_name} ({score_col}) not found. Skipping...")
            continue

        print(f"\n{score_name} ({score_col}):")
        print("-" * 120)
        print(f"{'Exchange':<15} {'Period':>8}", end='')
        for ex in main_exchanges:
            print(f"{f' {ex} IC':>12}", end='')
        print(f"{' Diff':>12} {'Samples (NYSE/NASDAQ)'}")
        print("-" * 120)

        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'
            if ret_col not in df.columns:
                continue

            period_str = f"{period}d"
            ic_by_exchange = {}
            samples_by_exchange = {}

            # Calculate IC for each exchange
            for exchange in main_exchanges:
                exchange_data = df[df['exchange'] == exchange]

                if len(exchange_data) < 30:
                    ic_by_exchange[exchange] = None
                    samples_by_exchange[exchange] = 0
                    continue

                ic_result = calculate_ic(exchange_data, score_col, [ret_col])

                if ret_col in ic_result and ic_result[ret_col]['spearman_ic'] is not None:
                    ic_by_exchange[exchange] = ic_result[ret_col]['spearman_ic']
                    samples_by_exchange[exchange] = ic_result[ret_col]['n_samples']
                else:
                    ic_by_exchange[exchange] = None
                    samples_by_exchange[exchange] = 0

            # Print results
            print(f"{'':15} {period_str:>8}", end='')

            nyse_ic = ic_by_exchange.get('NYSE')
            nasdaq_ic = ic_by_exchange.get('NASDAQ')

            for ex in main_exchanges:
                ic_val = ic_by_exchange.get(ex)
                if ic_val is not None:
                    print(f" {ic_val:>11.4f}", end='')
                else:
                    print(f" {'N/A':>11}", end='')

            # Calculate difference (NYSE - NASDAQ)
            if nyse_ic is not None and nasdaq_ic is not None:
                diff = nyse_ic - nasdaq_ic
                print(f" {diff:>+11.4f}", end='')
            else:
                print(f" {'N/A':>11}", end='')

            nyse_samples = samples_by_exchange.get('NYSE', 0)
            nasdaq_samples = samples_by_exchange.get('NASDAQ', 0)
            print(f" {nyse_samples:>6,}/{nasdaq_samples:<6,}")

            # Store results
            all_results.append({
                'factor': score_name,
                'period': period_str,
                'NYSE_ic': nyse_ic,
                'NASDAQ_ic': nasdaq_ic,
                'difference': diff if nyse_ic is not None and nasdaq_ic is not None else None,
                'NYSE_samples': nyse_samples,
                'NASDAQ_samples': nasdaq_samples
            })

    # Summary statistics
    print("\n" + "-" * 120)
    print("Summary Statistics (Average IC across all periods):")
    print("-" * 120)
    print(f"{'Factor':<25} {'NYSE Avg IC':>15} {'NASDAQ Avg IC':>15} {'Avg Difference':>15}")
    print("-" * 120)

    df_results = pd.DataFrame(all_results)

    for score_col, score_name in score_columns.items():
        factor_data = df_results[df_results['factor'] == score_name]

        if len(factor_data) == 0:
            continue

        nyse_avg = factor_data['NYSE_ic'].dropna().mean()
        nasdaq_avg = factor_data['NASDAQ_ic'].dropna().mean()
        diff_avg = factor_data['difference'].dropna().mean()

        print(f"{score_name:<25} {nyse_avg:>15.4f} {nasdaq_avg:>15.4f} {diff_avg:>+15.4f}")

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f'us_exchange_factor_ic_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def winsorized_ic_analysis(df):
    """STEP 10B: Winsorized IC Analysis (Outlier-adjusted)"""

    print("\n" + "=" * 100)
    print("STEP 10B: Winsorized IC Analysis (5th-95th percentile)")
    print("=" * 100)

    return_cols = [f'return_{period}d' for period in RETURN_PERIODS]
    results = []

    score_columns = {
        'final_score': 'Final Score',
        'value_score': 'Value Factor',
        'quality_score': 'Quality Factor',
        'momentum_score': 'Momentum Factor',
        'growth_score': 'Growth Factor',
    }

    print("\nComparison: Original IC vs Winsorized IC (Pearson)")
    print("-" * 120)
    print(f"{'Factor':<20} {'Period':<8} {'Original':>12} {'Winsorized':>12} {'Diff':>10} {'Improvement':>12}")
    print("-" * 120)

    for score_col, score_name in score_columns.items():
        if score_col not in df.columns:
            continue

        original_ic = calculate_ic(df, score_col, return_cols)
        winsorized_ic = calculate_winsorized_ic(df, score_col, return_cols)

        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'

            orig_pearson = original_ic.get(ret_col, {}).get('pearson_ic')
            wins_pearson = winsorized_ic.get(ret_col, {}).get('winsorized_pearson_ic')

            if orig_pearson is not None and wins_pearson is not None:
                diff = wins_pearson - orig_pearson
                improvement = "Better" if wins_pearson > orig_pearson else "Worse"

                print(f"{score_name:<20} {period}d{'':<5} {orig_pearson:>12.4f} {wins_pearson:>12.4f} "
                      f"{diff:>+10.4f} {improvement:>12}")

                results.append({
                    'factor': score_name,
                    'period': f'{period}d',
                    'original_pearson_ic': orig_pearson,
                    'winsorized_pearson_ic': wins_pearson,
                    'original_spearman_ic': original_ic[ret_col].get('spearman_ic'),
                    'winsorized_spearman_ic': winsorized_ic[ret_col].get('winsorized_spearman_ic'),
                    'pearson_diff': diff,
                    'lower_bound': winsorized_ic[ret_col].get('lower_bound'),
                    'upper_bound': winsorized_ic[ret_col].get('upper_bound'),
                    'n_samples': winsorized_ic[ret_col].get('n_samples')
                })

    # Summary statistics
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        print("\n" + "-" * 120)
        print("Summary: Average IC Improvement by Factor")
        print("-" * 60)
        for factor in df_results['factor'].unique():
            factor_data = df_results[df_results['factor'] == factor]
            avg_improvement = factor_data['pearson_diff'].mean()
            print(f"  {factor}: {avg_improvement:+.4f}")

    output_file = os.path.join(OUTPUT_DIR, f'us_winsorized_ic_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# PART 6: FEATURE VALIDATION (Phase 3 구현 기능 검증)
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

    ret_col = 'return_30d' if 'return_30d' in valid_data.columns else 'return_60d'
    if ret_col in valid_data.columns:
        quintile_data = valid_data[['entry_timing_score', ret_col]].dropna()
        quintile_data['quintile'] = pd.qcut(quintile_data['entry_timing_score'], q=5, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'], duplicates='drop')

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

    # 3. Entry Timing + Final Score 조합 분석
    print("\n[3] Entry Timing + Final Score Combined Analysis:")
    print("-" * 80)

    if 'return_60d' in valid_data.columns:
        combo_data = valid_data[['entry_timing_score', 'final_score', 'return_60d']].dropna()

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
                avg_ret = subset['return_60d'].mean()
                median_ret = subset['return_60d'].median()
                win_rate = (subset['return_60d'] > 0).mean() * 100
                print(f"{name:<30} {avg_ret:>+11.2f}% {median_ret:>+9.2f}% {win_rate:>9.1f}% {len(subset):>8,}")

                results.append({
                    'metric': 'entry_timing_combo',
                    'period': '60d',
                    'combination': name,
                    'avg_return': avg_ret,
                    'median_return': median_ret,
                    'win_rate': win_rate,
                    'n_samples': len(subset)
                })

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, f'us_entry_timing_validation_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
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
    for period in [30, 60, 90]:
        ret_col = f'return_{period}d'
        if ret_col not in valid_data.columns:
            continue

        subset = valid_data[['stop_loss_abs', ret_col, 'final_score']].dropna()
        if len(subset) < 50:
            continue

        print(f"\n[{period}d Analysis]")
        print("-" * 80)

        # 손절 트리거 여부 (최저 수익률이 손절선 이하로 내려갔는지 - 근사치로 기간 수익률 사용)
        # 실제로는 기간 내 최저점을 봐야 하지만, 여기서는 기간 수익률로 근사
        subset['would_stop'] = subset[ret_col] <= -subset['stop_loss_abs']
        subset['final_positive'] = subset[ret_col] > 0

        # Case 1: 손절 후 회복 (조기 청산) - 기간 내 손절선 터치했지만 최종 양수
        # Case 2: 손절 정확 - 손절선 터치하고 최종도 음수
        # Case 3: 손절 미발동 후 손실 - 손절선 미터치했지만 최종 음수
        # Case 4: 손절 미발동 후 이익 - 손절선 미터치하고 최종 양수

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
    print("\n[Stop Loss % Quintile Analysis - 60d]:")
    print("-" * 80)

    if 'return_60d' in valid_data.columns:
        sl_data = valid_data[['stop_loss_abs', 'return_60d']].dropna()
        sl_data['sl_quintile'] = pd.qcut(sl_data['stop_loss_abs'], q=5, labels=['Q1(Tight)', 'Q2', 'Q3', 'Q4', 'Q5(Wide)'], duplicates='drop')

        print(f"{'SL Quintile':<15} {'SL Range':>15} {'Avg Return':>12} {'Win Rate':>10}")
        print("-" * 60)

        for q in ['Q1(Tight)', 'Q2', 'Q3', 'Q4', 'Q5(Wide)']:
            q_data = sl_data[sl_data['sl_quintile'] == q]
            if len(q_data) > 0:
                sl_min = q_data['stop_loss_abs'].min()
                sl_max = q_data['stop_loss_abs'].max()
                avg_ret = q_data['return_60d'].mean()
                win_rate = (q_data['return_60d'] > 0).mean() * 100
                print(f"{q:<15} {sl_min:>6.1f}%-{sl_max:>5.1f}% {avg_ret:>+11.2f}% {win_rate:>9.1f}%")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, f'us_stop_loss_validation_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
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

    # 실제 결과 분류 기준 (60일 수익률 기준)
    ret_col = 'return_60d' if 'return_60d' in valid_data.columns else 'return_90d'
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

    print(f"\n[3.5] Sideways Probability Calibration:")
    print("-" * 80)
    print(f"{'Predicted Prob':<20} {'Actual Sideways':>15} {'Count':>10} {'Calibration':>15}")
    print("-" * 80)

    for low, high in prob_bins:
        bin_data = scenario_data[(scenario_data['scenario_sideways_prob'] >= low) &
                                  (scenario_data['scenario_sideways_prob'] < high)]
        if len(bin_data) > 10:
            actual_sideways = (bin_data['actual_scenario'] == 'Sideways').mean() * 100
            predicted_mid = (low + high) / 2
            calibration = actual_sideways - predicted_mid

            print(f"{low:>3}%-{high:<3}%{'':<12} {actual_sideways:>14.1f}% {len(bin_data):>10,} {calibration:>+14.1f}%")

            results.append({
                'scenario': 'Sideways',
                'prob_range': f'{low}-{high}%',
                'predicted_mid': predicted_mid,
                'actual_rate': actual_sideways,
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
    output_file = os.path.join(OUTPUT_DIR, f'us_scenario_calibration_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
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
    import json

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
    for period in [30, 60, 90]:
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
    output_file = os.path.join(OUTPUT_DIR, f'us_buy_triggers_validation_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
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
    for period in [30, 60, 90]:
        ret_col = f'return_{period}d'
        if ret_col not in valid_data.columns:
            continue

        subset = valid_data[['position_size_pct', ret_col, 'volatility_annual', 'final_score']].dropna()
        if len(subset) < 100:
            continue

        print(f"\n[{period}d Analysis by Position Size Quintile]:")
        print("-" * 100)

        subset['ps_quintile'] = pd.qcut(subset['position_size_pct'], q=5,
                                         labels=['Q1(Small)', 'Q2', 'Q3', 'Q4', 'Q5(Large)'], duplicates='drop')

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

    # 2. 가중 수익률 시뮬레이션
    print(f"\n[Weighted Return Simulation - 60d]:")
    print("-" * 80)

    if 'return_60d' in valid_data.columns:
        sim_data = valid_data[['position_size_pct', 'return_60d', 'final_score']].dropna()
        sim_data['position_size_pct'] = sim_data['position_size_pct'].astype(float)
        sim_data['return_60d'] = sim_data['return_60d'].astype(float)
        sim_data['final_score'] = sim_data['final_score'].astype(float)

        # Buy candidates only (final_score >= 65)
        buy_candidates = sim_data[sim_data['final_score'] >= 65].copy()

        if len(buy_candidates) > 50:
            # 균등 비중 vs 변동성 기반 비중
            equal_weight_return = buy_candidates['return_60d'].mean()

            # 변동성 기반 가중 (position_size_pct 비례)
            total_position = buy_candidates['position_size_pct'].sum()
            buy_candidates['weight'] = buy_candidates['position_size_pct'] / total_position
            weighted_return = (buy_candidates['return_60d'] * buy_candidates['weight']).sum()

            print(f"Buy Candidates (Score >= 65): {len(buy_candidates):,} stocks")
            print(f"  Equal Weight Return:    {equal_weight_return:+.2f}%")
            print(f"  Position Size Weighted: {weighted_return:+.2f}%")
            print(f"  Difference:             {weighted_return - equal_weight_return:+.2f}%")

            results.append({
                'period': '60d',
                'analysis': 'weighted_simulation',
                'group': 'Equal Weight',
                'avg_return': equal_weight_return,
                'n_samples': len(buy_candidates)
            })
            results.append({
                'period': '60d',
                'analysis': 'weighted_simulation',
                'group': 'PS Weighted',
                'avg_return': weighted_return,
                'n_samples': len(buy_candidates)
            })

    # 3. High Score에서 Position Size 효과
    print(f"\n[Position Size Effect within High Score (>=70)]:")
    print("-" * 60)

    if 'return_60d' in valid_data.columns:
        high_score = valid_data[valid_data['final_score'] >= 70][['position_size_pct', 'return_60d']].dropna()

        if len(high_score) > 50:
            large_ps = high_score[high_score['position_size_pct'] >= high_score['position_size_pct'].median()]
            small_ps = high_score[high_score['position_size_pct'] < high_score['position_size_pct'].median()]

            print(f"  Large Position Size: Avg {large_ps['return_60d'].mean():+.2f}%, Win Rate {(large_ps['return_60d'] > 0).mean()*100:.1f}%")
            print(f"  Small Position Size: Avg {small_ps['return_60d'].mean():+.2f}%, Win Rate {(small_ps['return_60d'] > 0).mean()*100:.1f}%")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, f'us_position_size_validation_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


def validate_conviction_interaction(df):
    """STEP 16: Conviction Score & Interaction Score 검증

    검증 질문: Phase 3에서 추가된 conviction_score, interaction_score가 예측력 향상에 기여하는가?
    """
    print("\n" + "=" * 100)
    print("STEP 16: Conviction & Interaction Score Validation")
    print("=" * 100)

    results = []

    cols_needed = ['conviction_score', 'interaction_score']
    available_cols = [c for c in cols_needed if c in df.columns]

    if not available_cols:
        print("[WARNING] conviction_score/interaction_score columns not found. Skipping...")
        return pd.DataFrame()

    valid_data = df.dropna(subset=available_cols).copy()
    if len(valid_data) < 100:
        print(f"[WARNING] Insufficient data: {len(valid_data)} samples")
        return pd.DataFrame()

    print(f"\nTotal samples: {len(valid_data):,}")

    # IC 분석
    print("\n[1] IC Analysis:")
    print("-" * 80)
    print(f"{'Score':<20} {'Period':<10} {'Spearman IC':>12} {'N':>10}")
    print("-" * 80)

    for score_col in available_cols:
        for period in [30, 60, 90, 180, 252]:
            ret_col = f'return_{period}d'
            if ret_col not in valid_data.columns:
                continue

            subset = valid_data[[score_col, ret_col]].dropna()
            if len(subset) < 30:
                continue

            spearman_ic, _ = spearmanr(subset[score_col], subset[ret_col])
            print(f"{score_col:<20} {period}d{'':<7} {spearman_ic:>12.4f} {len(subset):>10,}")

            results.append({
                'score': score_col,
                'period': f'{period}d',
                'spearman_ic': spearman_ic,
                'n_samples': len(subset)
            })

    # High Conviction + High Score 분석
    if 'conviction_score' in valid_data.columns and 'return_60d' in valid_data.columns:
        print("\n[2] High Conviction Effect (60d):")
        print("-" * 80)

        combo_data = valid_data[['conviction_score', 'final_score', 'return_60d']].dropna()

        # High Score (>=70) 내에서 Conviction 효과
        high_score = combo_data[combo_data['final_score'] >= 70]
        if len(high_score) > 50:
            high_conv = high_score[high_score['conviction_score'] >= 70]
            low_conv = high_score[high_score['conviction_score'] < 50]

            if len(high_conv) > 20 and len(low_conv) > 20:
                print(f"Within High Score (>=70) stocks:")
                print(f"  High Conviction (>=70): Avg {high_conv['return_60d'].mean():+.2f}%, Win {(high_conv['return_60d'] > 0).mean()*100:.1f}% (n={len(high_conv)})")
                print(f"  Low Conviction (<50):   Avg {low_conv['return_60d'].mean():+.2f}%, Win {(low_conv['return_60d'] > 0).mean()*100:.1f}% (n={len(low_conv)})")

    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, f'us_conviction_interaction_validation_{DATE_SUFFIX}.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution"""
    print("\n" + "=" * 100)
    print("US Stock - Comprehensive IC Analysis")
    print("=" * 100)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Dates: {len(ANALYSIS_DATES)} dates")
    print(f"Return Periods: {RETURN_PERIODS} days")
    print("=" * 100)

    # Create connection pool
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=300
    )

    try:
        # ========================================
        # PHASE A: DATA COLLECTION
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE A: Data Collection & Preparation")
        print("=" * 100)

        df_grade, df_price = await collect_data(pool)
        df_analysis = calculate_forward_returns(df_grade, df_price)

        output_file = os.path.join(OUTPUT_DIR, f'us_analysis_data_{DATE_SUFFIX}.csv')
        df_analysis.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {output_file}")

        # ========================================
        # PHASE B: IC ANALYSIS
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE B: Factor Level IC Analysis")
        print("=" * 100)

        ic_results_factor = ic_analysis_factor_level(df_analysis)

        # ========================================
        # PHASE C: VALIDATION
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE C: Validation & Testing")
        print("=" * 100)

        df_rolling = rolling_window_analysis(df_analysis)
        df_decile = decile_test(df_analysis)

        # ========================================
        # PHASE D: WEIGHT & PARAMETER ANALYSIS
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE D: Weight & Parameter Analysis")
        print("=" * 100)

        df_weight = weight_comparison_analysis(df_analysis)
        df_param = parameter_surface_analysis(df_analysis)
        df_accuracy = accuracy_winrate_analysis(df_analysis)

        # ========================================
        # PHASE E: SECTOR & EXCHANGE ANALYSIS
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE E: Sector & Exchange Analysis")
        print("=" * 100)

        df_sector = sector_analysis(df_analysis)
        df_exchange = exchange_analysis(df_analysis)
        df_exchange_factor = exchange_factor_analysis(df_analysis)

        # ========================================
        # PHASE F: WINSORIZED IC ANALYSIS
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE F: Winsorized IC Analysis (Outlier Adjustment)")
        print("=" * 100)

        df_winsorized = winsorized_ic_analysis(df_analysis)

        # ========================================
        # PHASE G: FEATURE VALIDATION
        # ========================================
        print("\n" + "=" * 100)
        print("PHASE G: Feature Validation (Phase 3 구현 기능 검증)")
        print("=" * 100)

        df_entry_timing = validate_entry_timing_score(df_analysis)
        df_stop_loss = validate_stop_loss(df_analysis)
        df_scenario = validate_scenario_probability(df_analysis)
        df_triggers = validate_buy_triggers(df_analysis)
        df_position = validate_position_size(df_analysis)
        df_conviction = validate_conviction_interaction(df_analysis)

        # ========================================
        # PHASE H: ENHANCED IC ANALYSIS (Newey-West + Cluster-Robust)
        # ========================================
        df_enhanced_ic = enhanced_ic_analysis(df_analysis)

        # ========================================
        # PHASE I: IC STABILITY ANALYSIS
        # ========================================
        stability_results = ic_stability_analysis(df_analysis)

        # ========================================
        # PHASE J: PANEL REGRESSION IC
        # ========================================
        df_panel_ic = panel_regression_ic(df_analysis)

        # ========================================
        # PHASE K: SUMMARY REPORT
        # ========================================
        if len(df_enhanced_ic) > 0 and len(df_panel_ic) > 0:
            generate_summary_report(df_enhanced_ic, df_panel_ic, stability_results)

        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "=" * 100)
        print("All Analyses Complete!")
        print("=" * 100)

        print(f"\nGenerated Files (Date: {DATE_SUFFIX}):")
        print("  [Data]")
        print(f"  1. us_analysis_data_{DATE_SUFFIX}.csv - Full analysis data")
        print("\n  [IC Analysis]")
        print(f"  2. us_factor_ic_{DATE_SUFFIX}.csv - Factor level IC")
        print(f"  3. us_winsorized_ic_{DATE_SUFFIX}.csv - Winsorized IC (outlier-adjusted)")
        print("\n  [Validation]")
        print(f"  4. us_rolling_ic_{DATE_SUFFIX}.csv - Rolling window IC")
        print(f"  5. us_decile_test_{DATE_SUFFIX}.csv - Decile test (with directional accuracy)")
        print("\n  [Weight & Parameter]")
        print(f"  6. us_weight_comparison_{DATE_SUFFIX}.csv - Uniform vs Dynamic weights")
        print(f"  7. us_parameter_surface_{DATE_SUFFIX}.csv - Threshold optimization")
        print(f"  8. us_accuracy_winrate_{DATE_SUFFIX}.csv - Accuracy and win rate")
        print("\n  [Sector & Exchange]")
        print(f"  9. us_sector_analysis_{DATE_SUFFIX}.csv - Sector analysis")
        print(f"  10. us_exchange_analysis_{DATE_SUFFIX}.csv - Exchange analysis (NYSE vs NASDAQ)")
        print(f"  10B. us_exchange_factor_ic_{DATE_SUFFIX}.csv - Exchange-specific factor IC analysis")
        print("\n  [Feature Validation - NEW]")
        print(f"  11. us_entry_timing_validation_{DATE_SUFFIX}.csv - Entry timing score validation")
        print(f"  12. us_stop_loss_validation_{DATE_SUFFIX}.csv - Stop loss validation")
        print(f"  13. us_scenario_calibration_{DATE_SUFFIX}.csv - Scenario probability calibration")
        print(f"  14. us_buy_triggers_validation_{DATE_SUFFIX}.csv - Buy triggers validation")
        print(f"  15. us_position_size_validation_{DATE_SUFFIX}.csv - Position size validation")
        print(f"  16. us_conviction_interaction_validation_{DATE_SUFFIX}.csv - Conviction/Interaction validation")
        print("\n  [Advanced IC Analysis - NEW]")
        print(f"  17. us_enhanced_ic_{DATE_SUFFIX}.csv - Newey-West + Cluster-Robust IC")
        print(f"  18. us_panel_ic_{DATE_SUFFIX}.csv - Panel Regression IC (Two-way FE)")
        print(f"  19. ic_plots/ic_stability_plots_{DATE_SUFFIX}.png - IC Stability Visualizations")

    finally:
        await pool.close()
        print("\nDatabase connection closed")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    asyncio.run(main())
