"""
Phase 3.6 - Final Score Out-of-Sample Validation

Purpose: Prevent overfitting by validating Final Score weight configurations
         on held-out test data with multiple performance metrics

Output Files:
1. phase3_6_final_score_validation_results.csv - All configurations with verdict
2. phase3_6_final_score_best_config.csv - Recommended configuration
3. phase3_6_final_score_metric_comparison.csv - Metric-by-metric breakdown
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calculate_ic(scores, returns):
    """
    Calculate Information Coefficient (IC)

    Returns:
        dict: {'pearson_ic': float, 'spearman_ic': float}
    """
    valid_mask = ~(np.isnan(scores) | np.isnan(returns))
    if valid_mask.sum() < 10:
        return {'pearson_ic': np.nan, 'spearman_ic': np.nan}

    scores_clean = scores[valid_mask]
    returns_clean = returns[valid_mask]

    try:
        pearson_ic, _ = pearsonr(scores_clean, returns_clean)
        spearman_ic, _ = spearmanr(scores_clean, returns_clean)
    except:
        pearson_ic, spearman_ic = np.nan, np.nan

    return {'pearson_ic': pearson_ic, 'spearman_ic': spearman_ic}


def calculate_ir(ic_list):
    """
    Calculate Information Ratio (IR)
    IR = mean(IC) / std(IC)

    Higher IR = more stable predictive power
    """
    ic_array = np.array(ic_list)
    ic_clean = ic_array[~np.isnan(ic_array)]

    if len(ic_clean) < 2:
        return np.nan

    mean_ic = np.mean(ic_clean)
    std_ic = np.std(ic_clean, ddof=1)

    if std_ic == 0:
        return np.nan

    return mean_ic / std_ic


def calculate_hit_rate(scores, returns):
    """
    Hit Rate = % of stocks where score direction matched return direction

    Example: If high score stock had positive return -> HIT
    """
    valid_mask = ~(np.isnan(scores) | np.isnan(returns))
    if valid_mask.sum() < 10:
        return np.nan

    scores_clean = scores[valid_mask]
    returns_clean = returns[valid_mask]

    # Direction match
    score_positive = scores_clean > 0
    return_positive = returns_clean > 0

    hits = (score_positive == return_positive).sum()
    total = len(scores_clean)

    return hits / total


def calculate_long_short_spread(df, score_col, return_col):
    """
    Long-Short Spread = Top decile return - Bottom decile return

    This is the actual portfolio return if we long top 10% and short bottom 10%
    """
    df_clean = df[[score_col, return_col]].dropna()

    if len(df_clean) < 20:
        return np.nan

    # Decile분류
    df_clean['decile'] = pd.qcut(df_clean[score_col], q=10, labels=False, duplicates='drop')

    top_return = df_clean[df_clean['decile'] == 9][return_col].mean()
    bottom_return = df_clean[df_clean['decile'] == 0][return_col].mean()

    return top_return - bottom_return


def calculate_sharpe_ratio(returns_list):
    """
    Sharpe Ratio = mean(return) / std(return)

    Assuming risk-free rate = 0 for simplicity
    """
    returns_array = np.array(returns_list)
    returns_clean = returns_array[~np.isnan(returns_array)]

    if len(returns_clean) < 2:
        return np.nan

    mean_return = np.mean(returns_clean)
    std_return = np.std(returns_clean, ddof=1)

    if std_return == 0:
        return np.nan

    return mean_return / std_return


def calculate_max_drawdown(cumulative_returns):
    """
    Maximum Drawdown = max percentage loss from peak to trough

    Lower is better (less risky)
    """
    cumulative_returns = np.array(cumulative_returns)

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown at each point
    drawdown = (cumulative_returns - running_max) / running_max

    max_dd = np.min(drawdown)

    return max_dd


def evaluate_configuration(df_subset, base_weight, sector_weight, combo_weight, return_col):
    """
    Evaluate a single Final Score weight configuration

    Returns:
        dict: All performance metrics for this configuration
    """
    # Final Score 계산
    df_subset['final_score_test'] = (
        df_subset['base_factor_final_score'] * base_weight +
        df_subset['sector_rotation_score'] * sector_weight +
        df_subset['factor_combo_score'] * combo_weight
    )

    # Group by date
    dates = sorted(df_subset['date'].unique())

    # Metric lists
    pearson_ic_list = []
    spearman_ic_list = []
    hit_rate_list = []
    long_short_spread_list = []

    for date in dates:
        df_date = df_subset[df_subset['date'] == date]

        # IC
        ic_result = calculate_ic(df_date['final_score_test'].values, df_date[return_col].values)
        pearson_ic_list.append(ic_result['pearson_ic'])
        spearman_ic_list.append(ic_result['spearman_ic'])

        # Hit Rate
        hit_rate = calculate_hit_rate(df_date['final_score_test'].values, df_date[return_col].values)
        hit_rate_list.append(hit_rate)

        # Long-Short Spread
        ls_spread = calculate_long_short_spread(df_date, 'final_score_test', return_col)
        long_short_spread_list.append(ls_spread)

    # Aggregate metrics
    pearson_ic_mean = np.nanmean(pearson_ic_list)
    spearman_ic_mean = np.nanmean(spearman_ic_list)
    pearson_ir = calculate_ir(pearson_ic_list)
    spearman_ir = calculate_ir(spearman_ic_list)
    hit_rate_mean = np.nanmean(hit_rate_list)
    ls_spread_mean = np.nanmean(long_short_spread_list)
    sharpe_ratio = calculate_sharpe_ratio(long_short_spread_list)

    # Max Drawdown (cumulative Long-Short returns)
    ls_clean = [x for x in long_short_spread_list if not np.isnan(x)]
    if len(ls_clean) > 0:
        cumulative_returns = np.cumsum(ls_clean)
        max_dd = calculate_max_drawdown(cumulative_returns)
    else:
        max_dd = np.nan

    return {
        'pearson_ic': pearson_ic_mean,
        'spearman_ic': spearman_ic_mean,
        'pearson_ir': pearson_ir,
        'spearman_ir': spearman_ir,
        'hit_rate': hit_rate_mean,
        'long_short_spread': ls_spread_mean,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'n_dates': len(dates)
    }


def check_overfitting(insample_metrics, outsample_metrics, threshold=0.3):
    """
    Check if configuration is overfitting

    Overfitting criteria:
    1. In-sample IC > Out-of-sample IC by more than threshold
    2. In-sample IC > 0 but Out-of-sample IC < 0
    3. In-sample Sharpe > Out-of-sample Sharpe by more than threshold

    Returns:
        dict: {'is_overfitting': bool, 'reason': str, 'severity': float}
    """
    # IC degradation
    ic_drop = insample_metrics['pearson_ic'] - outsample_metrics['pearson_ic']

    # Sign flip
    sign_flip = (insample_metrics['pearson_ic'] > 0) and (outsample_metrics['pearson_ic'] < 0)

    # Sharpe degradation
    sharpe_drop = insample_metrics['sharpe_ratio'] - outsample_metrics['sharpe_ratio']

    # Hit Rate degradation
    hit_drop = insample_metrics['hit_rate'] - outsample_metrics['hit_rate']

    reasons = []
    severity = 0.0

    if ic_drop > threshold:
        reasons.append(f"IC drop: {ic_drop:.4f} (>{threshold})")
        severity += ic_drop

    if sign_flip:
        reasons.append("Sign flip: positive in-sample but negative out-of-sample")
        severity += 0.5

    if sharpe_drop > threshold:
        reasons.append(f"Sharpe drop: {sharpe_drop:.4f} (>{threshold})")
        severity += sharpe_drop * 0.3

    if hit_drop > 0.1:  # 10% hit rate drop
        reasons.append(f"Hit rate drop: {hit_drop:.4f} (>0.1)")
        severity += hit_drop * 2

    is_overfitting = len(reasons) > 0
    reason_str = "; ".join(reasons) if reasons else "No overfitting detected"

    return {
        'is_overfitting': is_overfitting,
        'reason': reason_str,
        'severity': severity
    }


def generate_weight_configurations():
    """
    Generate test configurations

    Configurations to test:
    1. Current (baseline): 36%, 25%, 39%
    2. Equal weights: 33.33%, 33.33%, 33.33%
    3. Growth-focused: 20%, 20%, 60%
    4. Value-focused: 60%, 20%, 20%
    5. Sector-focused: 20%, 60%, 20%
    6. Balanced duo: 50%, 0%, 50%
    7. Sector-combo: 25%, 50%, 25%
    8. Base-only: 100%, 0%, 0%
    9. Combo-only: 0%, 0%, 100%
    10. Base-sector: 70%, 30%, 0%
    """
    configs = [
        {'name': 'Current_Baseline', 'base': 0.36, 'sector': 0.25, 'combo': 0.39},
        {'name': 'Equal_Weights', 'base': 0.3333, 'sector': 0.3333, 'combo': 0.3334},
        {'name': 'Growth_Focused', 'base': 0.20, 'sector': 0.20, 'combo': 0.60},
        {'name': 'Value_Focused', 'base': 0.60, 'sector': 0.20, 'combo': 0.20},
        {'name': 'Sector_Focused', 'base': 0.20, 'sector': 0.60, 'combo': 0.20},
        {'name': 'Balanced_Duo', 'base': 0.50, 'sector': 0.00, 'combo': 0.50},
        {'name': 'Sector_Combo', 'base': 0.25, 'sector': 0.50, 'combo': 0.25},
        {'name': 'Base_Only', 'base': 1.00, 'sector': 0.00, 'combo': 0.00},
        {'name': 'Combo_Only', 'base': 0.00, 'sector': 0.00, 'combo': 1.00},
        {'name': 'Base_Sector', 'base': 0.70, 'sector': 0.30, 'combo': 0.00},
    ]

    return configs


def main():
    print("=" * 80)
    print("Phase 3.6 - Final Score Out-of-Sample Validation")
    print("=" * 80)
    print()

    # STEP 1: Load data
    print("[STEP 1] Loading phase3_6_analysis_data.csv...")
    df = pd.read_csv(r'C:\project\alpha\quant\result test\phase3_6_analysis_data.csv')
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")
    print()

    # Check required columns
    required_cols = ['date', 'symbol', 'base_factor_final_score', 'sector_rotation_score',
                     'factor_combo_score', 'return_1m', 'return_20d']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return

    # STEP 2: Split In-Sample vs Out-of-Sample
    print("[STEP 2] Splitting In-Sample (7 dates) vs Out-of-Sample (2 dates)...")
    dates_sorted = sorted(df['date'].unique())
    print(f"  Total dates: {len(dates_sorted)}")
    print(f"  Dates: {dates_sorted}")
    print()

    if len(dates_sorted) < 9:
        print(f"WARNING: Expected 9 dates but found {len(dates_sorted)}. Adjusting split...")
        n_insample = int(len(dates_sorted) * 0.7)
    else:
        n_insample = 7

    insample_dates = dates_sorted[:n_insample]
    outsample_dates = dates_sorted[n_insample:]

    print(f"  In-Sample dates ({len(insample_dates)}): {insample_dates}")
    print(f"  Out-of-Sample dates ({len(outsample_dates)}): {outsample_dates}")
    print()

    df_insample = df[df['date'].isin(insample_dates)].copy()
    df_outsample = df[df['date'].isin(outsample_dates)].copy()

    print(f"  In-Sample records: {len(df_insample):,}")
    print(f"  Out-of-Sample records: {len(df_outsample):,}")
    print()

    # STEP 3: Test multiple configurations
    print("[STEP 3] Testing weight configurations...")
    print()

    configs = generate_weight_configurations()
    return_col = 'return_20d'  # Using 20-day return as target

    results = []

    for idx, config in enumerate(configs, 1):
        print(f"  [{idx}/{len(configs)}] Testing: {config['name']}")
        print(f"      Weights: Base={config['base']:.2f}, Sector={config['sector']:.2f}, Combo={config['combo']:.2f}")

        # In-Sample evaluation
        insample_metrics = evaluate_configuration(
            df_insample, config['base'], config['sector'], config['combo'], return_col
        )

        # Out-of-Sample evaluation
        outsample_metrics = evaluate_configuration(
            df_outsample, config['base'], config['sector'], config['combo'], return_col
        )

        # Overfitting check
        overfitting = check_overfitting(insample_metrics, outsample_metrics)

        # Determine verdict
        if overfitting['is_overfitting']:
            verdict = 'REJECTED_OVERFITTING'
        elif outsample_metrics['pearson_ic'] < 0:
            verdict = 'REJECTED_NEGATIVE_IC'
        elif outsample_metrics['hit_rate'] < 0.5:
            verdict = 'REJECTED_LOW_HITRATE'
        elif outsample_metrics['pearson_ic'] > 0.05 and outsample_metrics['hit_rate'] > 0.5:
            verdict = 'APPROVED'
        else:
            verdict = 'MARGINAL'

        print(f"      In-Sample IC: {insample_metrics['pearson_ic']:.4f}, Out-of-Sample IC: {outsample_metrics['pearson_ic']:.4f}")
        print(f"      Verdict: {verdict}")
        if overfitting['is_overfitting']:
            print(f"      Overfitting reason: {overfitting['reason']}")
        print()

        # Store results
        result_row = {
            'config_name': config['name'],
            'base_weight': config['base'],
            'sector_weight': config['sector'],
            'combo_weight': config['combo'],
            'insample_pearson_ic': insample_metrics['pearson_ic'],
            'insample_spearman_ic': insample_metrics['spearman_ic'],
            'insample_pearson_ir': insample_metrics['pearson_ir'],
            'insample_hit_rate': insample_metrics['hit_rate'],
            'insample_long_short_spread': insample_metrics['long_short_spread'],
            'insample_sharpe_ratio': insample_metrics['sharpe_ratio'],
            'insample_max_drawdown': insample_metrics['max_drawdown'],
            'outsample_pearson_ic': outsample_metrics['pearson_ic'],
            'outsample_spearman_ic': outsample_metrics['spearman_ic'],
            'outsample_pearson_ir': outsample_metrics['pearson_ir'],
            'outsample_hit_rate': outsample_metrics['hit_rate'],
            'outsample_long_short_spread': outsample_metrics['long_short_spread'],
            'outsample_sharpe_ratio': outsample_metrics['sharpe_ratio'],
            'outsample_max_drawdown': outsample_metrics['max_drawdown'],
            'overfitting_detected': overfitting['is_overfitting'],
            'overfitting_severity': overfitting['severity'],
            'overfitting_reason': overfitting['reason'],
            'verdict': verdict
        }

        results.append(result_row)

    # STEP 4: Save results
    print("[STEP 4] Saving results...")
    df_results = pd.DataFrame(results)

    # Sort by out-of-sample IC (descending)
    df_results = df_results.sort_values('outsample_pearson_ic', ascending=False)

    output_file = 'phase3_6_final_score_validation_results.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  [1/3] {output_file} saved ({len(df_results)} configurations)")

    # Best configuration
    df_approved = df_results[df_results['verdict'] == 'APPROVED']
    if len(df_approved) > 0:
        best_config = df_approved.iloc[0:1]
        best_file = 'phase3_6_final_score_best_config.csv'
        best_config.to_csv(best_file, index=False, encoding='utf-8-sig')
        print(f"  [2/3] {best_file} saved (recommended configuration)")
        print(f"        Best: {best_config['config_name'].values[0]}")
        print(f"        Out-of-Sample IC: {best_config['outsample_pearson_ic'].values[0]:.4f}")
    else:
        print(f"  [2/3] No APPROVED configuration found. All configurations rejected.")

    # Metric comparison
    metric_comparison = []
    for _, row in df_results.iterrows():
        for metric in ['pearson_ic', 'hit_rate', 'long_short_spread', 'sharpe_ratio']:
            metric_comparison.append({
                'config_name': row['config_name'],
                'metric': metric,
                'insample': row[f'insample_{metric}'],
                'outsample': row[f'outsample_{metric}'],
                'degradation': row[f'insample_{metric}'] - row[f'outsample_{metric}']
            })

    df_metric_comparison = pd.DataFrame(metric_comparison)
    metric_file = 'phase3_6_final_score_metric_comparison.csv'
    df_metric_comparison.to_csv(metric_file, index=False, encoding='utf-8-sig')
    print(f"  [3/3] {metric_file} saved (metric-by-metric breakdown)")
    print()

    # STEP 5: Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()

    approved_count = len(df_results[df_results['verdict'] == 'APPROVED'])
    rejected_count = len(df_results[df_results['verdict'].str.contains('REJECTED')])
    marginal_count = len(df_results[df_results['verdict'] == 'MARGINAL'])

    print(f"Total Configurations Tested: {len(df_results)}")
    print(f"  APPROVED: {approved_count}")
    print(f"  REJECTED: {rejected_count}")
    print(f"  MARGINAL: {marginal_count}")
    print()

    print("Top 3 Configurations by Out-of-Sample IC:")
    for idx, row in df_results.head(3).iterrows():
        print(f"  {row['config_name']:20s} | IC={row['outsample_pearson_ic']:+.4f} | Hit={row['outsample_hit_rate']:.3f} | {row['verdict']}")
    print()

    print("Overfitting Detected in:")
    df_overfit = df_results[df_results['overfitting_detected'] == True]
    for idx, row in df_overfit.iterrows():
        print(f"  {row['config_name']:20s} | Severity={row['overfitting_severity']:.4f}")
        print(f"      {row['overfitting_reason']}")
    print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
