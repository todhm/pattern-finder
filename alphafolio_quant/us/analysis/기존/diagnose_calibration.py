"""
Scenario Calibration Root Cause Diagnosis

Analyzes us_scenario_calibration_20251208.csv to find:
1. Calibration error patterns
2. Root causes of over/under-prediction
3. Improvement recommendations
"""

import pandas as pd
import numpy as np
import os

# Load calibration data
data_path = os.path.join(os.path.dirname(__file__), '..', 'result', 'us_scenario_calibration_20251208.csv')
df = pd.read_csv(data_path)

print("=" * 80)
print("SCENARIO CALIBRATION ROOT CAUSE DIAGNOSIS")
print("=" * 80)

# 1. Raw Data Summary
print("\n[1] RAW CALIBRATION DATA")
print("-" * 60)
for _, row in df.iterrows():
    if row['scenario'] != 'Overall':
        error_type = "UNDER-PREDICT" if row['calibration_error'] > 0 else "OVER-PREDICT"
        print(f"{row['scenario']:10} {row['prob_range']:8} | "
              f"Pred: {row['predicted_mid']:5.1f}% → Actual: {row['actual_rate']:5.1f}% | "
              f"Error: {row['calibration_error']:+6.1f}% ({error_type})")

# 2. Calculate ECE (Expected Calibration Error)
print("\n[2] CALIBRATION METRICS")
print("-" * 60)
valid_data = df[df['scenario'] != 'Overall']
total_samples = valid_data['n_samples'].sum()

weighted_error = 0
for _, row in valid_data.iterrows():
    weight = row['n_samples'] / total_samples
    weighted_error += weight * abs(row['calibration_error'])

max_error = valid_data['calibration_error'].abs().max()

print(f"Expected Calibration Error (ECE): {weighted_error:.2f}%")
print(f"Maximum Calibration Error (MCE):  {max_error:.2f}%")
print(f"Total Samples: {total_samples:,}")
print(f"Calibration Quality: {'POOR' if weighted_error > 15 else 'MODERATE' if weighted_error > 8 else 'GOOD'}")

# 3. Pattern Analysis
print("\n[3] ERROR PATTERN ANALYSIS")
print("-" * 60)

# Check for sigmoid distortion
low_end = valid_data[valid_data['predicted_mid'] <= 20]
mid_range = valid_data[(valid_data['predicted_mid'] > 20) & (valid_data['predicted_mid'] < 60)]
high_end = valid_data[valid_data['predicted_mid'] >= 60]

low_avg = low_end['calibration_error'].mean() if len(low_end) > 0 else 0
mid_avg = mid_range['calibration_error'].mean() if len(mid_range) > 0 else 0
high_avg = high_end['calibration_error'].mean() if len(high_end) > 0 else 0

print(f"Low End (0-20%):   Avg Error = {low_avg:+.1f}% ({'UNDER-PREDICT' if low_avg > 0 else 'OVER-PREDICT'})")
print(f"Mid Range (20-60%): Avg Error = {mid_avg:+.1f}%")
print(f"High End (60-80%): Avg Error = {high_avg:+.1f}% ({'UNDER-PREDICT' if high_avg > 0 else 'OVER-PREDICT'})")

sigmoid_detected = low_avg > 15 and high_avg < -15
print(f"\nSigmoid Distortion Detected: {'YES [!]' if sigmoid_detected else 'NO'}")

# 4. Root Cause Analysis
print("\n[4] ROOT CAUSE ANALYSIS")
print("-" * 60)

print("\n[*] PRIMARY ROOT CAUSES:")
print()

# Cause 1: Hardcoded probabilities
print("1. HARDCODED MACRO_ENVIRONMENT PROBABILITIES")
print("   " + "-" * 50)
print("   Problem: Static probabilities not learned from data")
print("   Evidence:")
print("   - STAGFLATION: 강세=5%, 약세=75% (too extreme)")
print("   - SOFT_LANDING: 강세=60%, 약세=10% (too optimistic)")
print("   - These never match actual market frequencies")
print()

# Cause 2: Score adjustment too narrow
print("2. SCORE ADJUSTMENT RANGE TOO NARROW (±10%)")
print("   " + "-" * 50)
print("   Problem: score_adj = (final_score - 50) / 500")
print("   Evidence:")
print("   - Maximum adjustment is only ±10%")
print("   - Cannot correct 20-26% calibration errors")
print("   - Extreme macro probabilities dominate output")
print()

# Cause 3: No calibration applied
print("3. EXISTING CALIBRATION COEFFICIENTS NOT APPLIED")
print("   " + "-" * 50)
print("   Problem: SCENARIO_CALIBRATION exists but unused")
print("   Evidence:")
print("   - us_agent_metrics.py line 79-82 defines coefficients")
print("   - calculate_scenario_probability() doesn't use them")
print("   - bull: slope=0.124, intercept=0.233 (never applied)")
print()

# Cause 4: Sample imbalance
print("4. SAMPLE SIZE IMBALANCE")
print("   " + "-" * 50)
for _, row in valid_data.iterrows():
    print(f"   {row['scenario']:10} {row['prob_range']:8}: {row['n_samples']:,} samples")

# 5. New Calibration Coefficients
print("\n[5] SUGGESTED NEW CALIBRATION COEFFICIENTS")
print("-" * 60)

for scenario in ['Bullish', 'Bearish', 'Sideways']:
    scenario_data = valid_data[valid_data['scenario'] == scenario]

    if len(scenario_data) >= 2:
        x = scenario_data['predicted_mid'].values / 100
        y = scenario_data['actual_rate'].values / 100

        # Linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2 + 1e-10)
        intercept = (np.sum(y) - slope * np.sum(x)) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
        r2 = 1 - ss_res / ss_tot

        print(f"\n{scenario}:")
        print(f"  calibrated = {slope:.3f} * predicted + {intercept:.3f}")
        print(f"  R² = {r2:.3f}")

        # Example
        test_vals = [0.1, 0.3, 0.5, 0.7]
        print(f"  Examples: ", end="")
        examples = [f"{v*100:.0f}%→{(slope*v+intercept)*100:.1f}%" for v in test_vals]
        print(", ".join(examples))

# 6. Improvement Recommendations
print("\n" + "=" * 80)
print("[6] IMPROVEMENT RECOMMENDATIONS")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY 1: Apply Calibration in calculate_scenario_probability()          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: us_agent_metrics.py line 1096-1098                                │
│                                                                             │
│ BEFORE (current):                                                           │
│   bullish_prob = round(adjusted_bull * 100)                                 │
│   bearish_prob = round(adjusted_bear * 100)                                 │
│                                                                             │
│ AFTER (recommended):                                                        │
│   # Apply calibration                                                       │
│   cal = SCENARIO_CALIBRATION                                                │
│   calibrated_bull = cal['bull']['slope'] * adjusted_bull                    │
│                     + cal['bull']['intercept']                              │
│   calibrated_bear = cal['bear']['slope'] * adjusted_bear                    │
│                     + cal['bear']['intercept']                              │
│   bullish_prob = round(calibrated_bull * 100)                               │
│   bearish_prob = round(calibrated_bear * 100)                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY 2: Update MACRO_ENVIRONMENT Probabilities                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: us_market_regime.py line 69-95                                    │
│                                                                             │
│ CURRENT (too extreme):                  RECOMMENDED (data-driven):          │
│ STAGFLATION: 강세=5%, 약세=75%    →    강세=15%, 약세=55%                   │
│ SOFT_LANDING: 강세=60%, 약세=10%  →    강세=45%, 약세=20%                   │
│ HARD_LANDING: 강세=10%, 약세=65%  →    강세=20%, 약세=50%                   │
│ REFLATION: 강세=50%, 약세=15%     →    강세=40%, 약세=25%                   │
│ DEFLATION: 강세=15%, 약세=55%     →    강세=20%, 약세=45%                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY 3: Widen Score Adjustment Range                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: us_agent_metrics.py line 1076                                     │
│                                                                             │
│ CURRENT:  score_adj = (final_score - 50) / 500  # ±10% max                  │
│ RECOMMENDED: score_adj = (final_score - 50) / 250  # ±20% max               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY 4: Update SCENARIO_CALIBRATION Coefficients                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Location: us_agent_metrics.py line 79-82                                    │
│                                                                             │
│ Use the newly calculated coefficients from section [5] above                │
│ These are derived from actual calibration data, not estimates               │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n[OK] Diagnosis Complete")
