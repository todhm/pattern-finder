"""
Phase 3.8 IC Analysis - Advanced Statistical Methods
=====================================================

Upgrades from Phase 3.7:
1. Newey-West Correction + Cluster-Robust SE
   - Heteroskedasticity and autocorrelation robust standard errors
   - Sector-based clustering for Korean market characteristics

2. IC Stability Plots + Rolling IC Visualization
   - Rolling IC time series with confidence bands
   - Cumulative IC for drift detection
   - IC distribution and autocorrelation analysis

3. Panel Regression IC (Two-way Fixed Effects)
   - Entity (firm) fixed effects
   - Time fixed effects
   - True factor IC controlling for market/firm effects

4. Extended Return Periods
   - Short-term: 1d, 5d, 10d (momentum validation)
   - Medium-term: 21d (1M), 42d (2M)
   - Long-term: 63d (3M) for Value/Quality factors

Dependencies:
    pip install statsmodels linearmodels matplotlib seaborn

Execution: python phase3_8_ic_analysis.py
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
import warnings
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

# Statistical packages
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging level
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Add kr directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgresql+asyncpg://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')

# Output directory
OUTPUT_DIR = r'C:\project\alpha\quant\kr\result test'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'ic_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION - Phase 3.8 Extended
# ============================================================================

# Extended Analysis dates
ANALYSIS_DATES = [
    '2025-08-04', '2025-08-11', '2025-08-19', '2025-08-22', '2025-08-07', '2025-08-27', '2025-09-22', '2025-09-25',
    '2025-09-01', '2025-09-05', '2025-09-09', '2025-09-12', '2025-09-16'
]

# Extended return periods (trading days)
RETURN_PERIODS = [1, 5, 10, 21, 42, 63]  # 1d, 1w, 2w, 1M, 2M, 3M

# Parallel processing settings
MAX_CONCURRENT = 40
BATCH_SIZE = 100

# Factor columns for analysis
FACTOR_COLUMNS = ['final_score', 'value_score', 'quality_score', 'momentum_score', 'growth_score']


# ============================================================================
# PART 1: DATA COLLECTION
# ============================================================================

async def collect_data(pool):
    """Collect data from kr_stock_grade + kr_intraday_total"""

    print("=" * 100)
    print("STEP 1: Data Collection (Phase 3.8)")
    print("=" * 100)

    date_objs = [datetime.strptime(d, '%Y-%m-%d').date() for d in ANALYSIS_DATES]

    async with pool.acquire() as conn:
        # Get grade data
        print("\n[1/3] Querying kr_stock_grade...")
        grade_query = """
            SELECT
                symbol, stock_name, date,
                final_score, final_grade,
                value_score, quality_score, momentum_score, growth_score,
                sector_rotation_score, market_state,
                confidence_score
            FROM kr_stock_grade
            WHERE date = ANY($1::date[])
        """
        grade_rows = await conn.fetch(grade_query, date_objs)
        df_grade = pd.DataFrame([dict(r) for r in grade_rows])
        print(f"  -> {len(df_grade):,} records loaded")

        # Get sector/theme data
        print("\n[2/3] Querying kr_stock_detail for sectors...")
        detail_query = """
            SELECT symbol, theme, exchange
            FROM kr_stock_detail
        """
        detail_rows = await conn.fetch(detail_query)
        df_detail = pd.DataFrame([dict(r) for r in detail_rows])

        # Merge sector info
        df_grade = df_grade.merge(df_detail[['symbol', 'theme', 'exchange']], on='symbol', how='left')
        df_grade['theme'] = df_grade['theme'].fillna('기타')
        df_grade['exchange'] = df_grade['exchange'].fillna('KOSPI')
        print(f"  -> Sector info merged")

        # Get price data for extended periods
        print("\n[3/3] Querying kr_intraday_total for returns...")
        max_period = max(RETURN_PERIODS)
        min_date = min(date_objs) - timedelta(days=10)
        max_date = max(date_objs) + timedelta(days=max_period + 30)

        price_query = """
            SELECT symbol, date, close
            FROM kr_intraday_total
            WHERE date >= $1 AND date <= $2
            ORDER BY symbol, date
        """
        price_rows = await conn.fetch(price_query, min_date, max_date)
        df_price = pd.DataFrame([dict(r) for r in price_rows])
        print(f"  -> {len(df_price):,} price records loaded")

    return df_grade, df_price


def calculate_forward_returns(df_grade, df_price):
    """Calculate forward returns for extended periods"""

    print("\n[4/4] Calculating forward returns...")
    print(f"  Return periods: {RETURN_PERIODS} days")

    df_price = df_price.copy()
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price = df_price.sort_values(['symbol', 'date'])

    # Create trading day index per symbol
    df_price['trading_day_idx'] = df_price.groupby('symbol').cumcount()

    results = []

    for _, row in df_grade.iterrows():
        symbol = row['symbol']
        analysis_date = pd.to_datetime(row['date'])

        result = row.to_dict()

        # Get price data for this symbol
        symbol_prices = df_price[df_price['symbol'] == symbol].copy()

        if len(symbol_prices) == 0:
            for period in RETURN_PERIODS:
                result[f'return_{period}d'] = None
            results.append(result)
            continue

        # Find analysis date index
        base_prices = symbol_prices[symbol_prices['date'] == analysis_date]

        if len(base_prices) == 0:
            for period in RETURN_PERIODS:
                result[f'return_{period}d'] = None
            results.append(result)
            continue

        base_close = base_prices.iloc[0]['close']
        if base_close is None or pd.isna(base_close):
            for period in RETURN_PERIODS:
                result[f'return_{period}d'] = None
            results.append(result)
            continue

        base_price = float(base_close)
        base_idx = int(base_prices.iloc[0]['trading_day_idx'])

        # Calculate returns for each period
        for period in RETURN_PERIODS:
            target_idx = base_idx + period
            future_prices = symbol_prices[symbol_prices['trading_day_idx'] == target_idx]

            if len(future_prices) == 0:
                result[f'return_{period}d'] = None
            else:
                future_close = future_prices.iloc[0]['close']
                if future_close is None or pd.isna(future_close):
                    result[f'return_{period}d'] = None
                else:
                    future_price = float(future_close)
                    ret = ((future_price - base_price) / base_price) * 100
                    result[f'return_{period}d'] = ret

        results.append(result)

    df_analysis = pd.DataFrame(results)

    # Summary
    for period in RETURN_PERIODS:
        col = f'return_{period}d'
        valid_count = df_analysis[col].notna().sum()
        print(f"  {period:2}d return: {valid_count:,} valid samples")

    return df_analysis


# ============================================================================
# PART 2: NEWEY-WEST IC CALCULATION
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
    # Convert to float to handle object dtype from DB
    scores = np.asarray(scores, dtype=float)
    returns = np.asarray(returns, dtype=float)

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
                                cluster_col: str = 'theme') -> Dict:
    """
    Calculate IC with cluster-robust standard errors

    Clusters by sector/theme to account for:
    - Korean market synchronization
    - Sector concentration effects

    Args:
        df: DataFrame with scores, returns, and cluster variable
        score_col: Column name for factor scores
        return_col: Column name for returns
        cluster_col: Column for clustering (default: theme/sector)

    Returns:
        dict with IC, cluster-robust t-stat, p-value
    """
    valid_data = df[[score_col, return_col, cluster_col]].dropna()

    if len(valid_data) < 20:
        return {
            'ic': None, 't_stat_cluster': None, 'p_value_cluster': None,
            'n_samples': len(valid_data), 'n_clusters': 0
        }

    # Convert to proper types to avoid numpy ufunc errors
    scores = valid_data[score_col].values.astype(float)
    returns = valid_data[return_col].values.astype(float)
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
    print("STEP 2: Enhanced IC Analysis (Newey-West + Cluster Robust)")
    print("=" * 100)

    return_cols = [f'return_{p}d' for p in RETURN_PERIODS]
    results = []

    print(f"\n{'Factor':<20} {'Period':<8} {'Spearman IC':>12} {'NW t-stat':>12} {'NW p-val':>10} {'Cluster t':>12} {'Signif':>8}")
    print("-" * 100)

    for factor in FACTOR_COLUMNS:
        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'

            if ret_col not in df.columns or factor not in df.columns:
                continue

            valid_data = df[[factor, ret_col, 'theme']].dropna()

            if len(valid_data) < 20:
                continue

            scores = valid_data[factor].values.astype(float)
            returns = valid_data[ret_col].values.astype(float)

            # Newey-West IC
            nw_result = calculate_ic_newey_west(scores, returns)

            # Cluster-Robust IC
            cluster_result = calculate_ic_cluster_robust(valid_data, factor, ret_col, 'theme')

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
                'factor': factor,
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

            print(f"{factor:<20} {period:>3}d     {sp_ic:>+12.4f} {t_nw:>+12.3f} {p_nw:>10.4f} {t_cl:>+12.3f} {signif:>8}")

    df_results = pd.DataFrame(results)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'phase3_8_enhanced_ic.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  -> Saved to {output_path}")

    return df_results


# ============================================================================
# PART 3: IC STABILITY ANALYSIS & VISUALIZATION
# ============================================================================

def calculate_rolling_ic(df: pd.DataFrame, factor: str, return_col: str,
                         window: int = 21) -> pd.DataFrame:
    """
    Calculate rolling IC time series

    Args:
        df: DataFrame with date, factor scores, returns
        factor: Factor column name
        return_col: Return column name
        window: Rolling window size in trading days

    Returns:
        DataFrame with date and rolling IC values
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    dates = sorted(df['date'].unique())

    rolling_ics = []

    for i, end_date in enumerate(dates):
        if i < window - 1:
            continue

        start_idx = max(0, i - window + 1)
        window_dates = dates[start_idx:i+1]

        window_data = df[df['date'].isin(window_dates)]
        valid_data = window_data[[factor, return_col]].dropna()

        if len(valid_data) < 10:
            rolling_ics.append({
                'date': end_date,
                'rolling_ic': None,
                'n_samples': len(valid_data)
            })
            continue

        ic, _ = spearmanr(valid_data[factor], valid_data[return_col])

        rolling_ics.append({
            'date': end_date,
            'rolling_ic': ic,
            'n_samples': len(valid_data)
        })

    return pd.DataFrame(rolling_ics)


def ic_stability_analysis(df: pd.DataFrame) -> Dict:
    """
    Comprehensive IC stability analysis with visualizations
    """
    print("\n" + "=" * 100)
    print("STEP 3: IC Stability Analysis")
    print("=" * 100)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        HAS_MATPLOTLIB = True
    except ImportError:
        print("  [WARNING] matplotlib not available. Skipping plots.")
        HAS_MATPLOTLIB = False

    stability_results = {}

    # Analyze each factor
    for factor in FACTOR_COLUMNS:
        print(f"\n  Analyzing {factor}...")

        factor_stability = {}

        for period in [21, 42, 63]:  # Focus on medium/long-term
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

        fig, axes = plt.subplots(len(FACTOR_COLUMNS), 3, figsize=(18, 4*len(FACTOR_COLUMNS)))

        for i, factor in enumerate(FACTOR_COLUMNS):
            if factor not in stability_results:
                continue

            # Use 42d (2M) as primary period
            period_key = '42d' if '42d' in stability_results[factor] else '21d'

            if period_key not in stability_results[factor]:
                continue

            data = stability_results[factor][period_key]
            df_ic = data['ic_series']

            # Plot 1: IC Time Series
            ax1 = axes[i, 0] if len(FACTOR_COLUMNS) > 1 else axes[0]
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
            ax2 = axes[i, 1] if len(FACTOR_COLUMNS) > 1 else axes[1]
            ax2.plot(df_ic['cumulative_ic'], color='blue', linewidth=2)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_title(f'{factor} - Cumulative IC (Drift Detection)')
            ax2.set_xlabel('Date Index')
            ax2.set_ylabel('Cumulative IC')

            # Add trend line
            if data['trend_slope'] is not None:
                x_trend = np.arange(len(df_ic))
                y_trend = data['trend_slope'] * x_trend * len(df_ic)  # Approximate cumulative trend
                ax2.plot(y_trend, color='red', linestyle='--', alpha=0.7,
                        label=f"Trend: {data['trend_slope']:+.4f}/period")
                ax2.legend(loc='upper left', fontsize=8)

            # Plot 3: IC Distribution
            ax3 = axes[i, 2] if len(FACTOR_COLUMNS) > 1 else axes[2]
            ax3.hist(df_ic['ic'], bins=min(20, len(df_ic)), edgecolor='black', alpha=0.7)
            ax3.axvline(data['mean_ic'], color='red', linewidth=2, label=f"Mean: {data['mean_ic']:.4f}")
            ax3.axvline(0, color='black', linewidth=1)
            ax3.set_title(f'{factor} - IC Distribution')
            ax3.set_xlabel('IC')
            ax3.set_ylabel('Frequency')
            ax3.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, 'ic_stability_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved plots to {plot_path}")

    return stability_results


# ============================================================================
# PART 4: PANEL REGRESSION IC
# ============================================================================

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
    print("STEP 4: Panel Regression IC (Two-Way Fixed Effects)")
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

    print(f"\n{'Factor':<20} {'Period':<8} {'Panel IC':>12} {'t-stat':>12} {'p-value':>10} {'R2':>10}")
    print("-" * 80)

    for factor in FACTOR_COLUMNS:
        for period in RETURN_PERIODS:
            ret_col = f'return_{period}d'

            if ret_col not in df_panel.columns or factor not in df_panel.columns:
                continue

            # Prepare data
            work_df = df_panel[['symbol', 'date', factor, ret_col, 'theme']].dropna()

            # Convert Decimal to float to avoid type errors
            work_df[factor] = work_df[factor].astype(float)
            work_df[ret_col] = work_df[ret_col].astype(float)

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
                                                                cov_kwds={'groups': work_df['theme']})

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
    output_path = os.path.join(OUTPUT_DIR, 'phase3_8_panel_ic.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  -> Saved to {output_path}")

    return df_results


# ============================================================================
# PART 5: SUMMARY COMPARISON
# ============================================================================

def generate_summary_report(df_enhanced: pd.DataFrame, df_panel: pd.DataFrame,
                           stability_results: Dict) -> None:
    """Generate comprehensive summary report"""

    print("\n" + "=" * 100)
    print("STEP 5: Summary Report")
    print("=" * 100)

    print("\n### IC Method Comparison ###\n")

    # Compare methods for key factors
    print(f"{'Factor':<20} {'Period':<8} {'Simple IC':>12} {'NW t-stat':>12} {'Panel IC':>12} {'Panel t':>12}")
    print("-" * 90)

    for factor in FACTOR_COLUMNS:
        for period in [21, 42, 63]:
            # Get enhanced IC results
            enhanced_row = df_enhanced[(df_enhanced['factor'] == factor) &
                                       (df_enhanced['period'] == period)]

            # Get panel IC results (with safety check for empty df)
            if len(df_panel) > 0 and 'factor' in df_panel.columns:
                panel_row = df_panel[(df_panel['factor'] == factor) &
                                    (df_panel['period'] == period)]
            else:
                panel_row = pd.DataFrame()

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
    print(f"{'Factor':<20} {'42d Mean IC':>12} {'IC Std':>10} {'AR(1)':>10} {'Trend':>12} {'Status':<15}")
    print("-" * 85)

    for factor in FACTOR_COLUMNS:
        if factor not in stability_results:
            continue

        if '42d' in stability_results[factor]:
            data = stability_results[factor]['42d']
        elif '21d' in stability_results[factor]:
            data = stability_results[factor]['21d']
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
    print("Phase 3.8 IC Analysis Complete")
    print("=" * 100)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function"""

    print("\n" + "=" * 100)
    print("Phase 3.8 - Advanced IC Analysis")
    print("Newey-West | Cluster-Robust | Panel Regression | IC Stability")
    print("=" * 100)

    start_time = datetime.now()

    # Connect to database
    print(f"\nConnecting to database...")
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)

    try:
        # Step 1: Collect data
        df_grade, df_price = await collect_data(pool)

        # Calculate extended returns
        df_analysis = calculate_forward_returns(df_grade, df_price)

        # Step 2: Enhanced IC with Newey-West and Cluster-Robust SE
        df_enhanced = enhanced_ic_analysis(df_analysis)

        # Step 3: IC Stability Analysis
        stability_results = ic_stability_analysis(df_analysis)

        # Step 4: Panel Regression IC
        df_panel = panel_regression_ic(df_analysis)

        # Step 5: Summary Report
        generate_summary_report(df_enhanced, df_panel, stability_results)

    finally:
        await pool.close()

    elapsed = datetime.now() - start_time
    print(f"\nTotal execution time: {elapsed}")


if __name__ == "__main__":
    asyncio.run(main())
