"""
Macro Environment별 실제 시나리오 발생 빈도 측정

근본 원인 파악:
1. 역사적 Base Rate: Bullish/Sideways/Bearish 각각 몇 %?
2. Macro 조건부 확률: SOFT_LANDING일 때 실제 Bullish 비율은?
3. 직관 vs 실제 비교: 하드코딩된 60%와 실제 데이터 비교

시나리오 정의:
- Bullish:  60일 수익률 > +5%
- Sideways: 60일 수익률 -5% ~ +5%
- Bearish:  60일 수익률 < -5%
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from us_db_async import AsyncDatabaseManager


async def measure_actual_scenario_rates():
    """Macro Environment별 실제 시나리오 발생률 측정"""

    db = AsyncDatabaseManager()
    await db.initialize()

    print("=" * 80)
    print("MACRO ENVIRONMENT vs ACTUAL SCENARIO ANALYSIS")
    print("=" * 80)
    print()
    print("Purpose: Measure empirical scenario rates, not curve-fitted coefficients")
    print()

    # 1. SPY 데이터로 시장 전체 수익률 가져오기
    print("[STEP 1] Loading SPY data for market returns...")

    spy_query = """
    SELECT
        date,
        close,
        LEAD(close, 60) OVER (ORDER BY date) as close_60d_later
    FROM us_daily_etf
    WHERE symbol = 'SPY'
    ORDER BY date
    """

    spy_data = await db.execute_query(spy_query)

    if not spy_data:
        print("No SPY data found")
        return

    df_spy = pd.DataFrame(spy_data)
    df_spy['return_60d'] = (df_spy['close_60d_later'] - df_spy['close']) / df_spy['close'] * 100
    df_spy = df_spy.dropna(subset=['return_60d'])

    print(f"  Total trading days: {len(df_spy)}")
    print(f"  Date range: {df_spy['date'].min()} to {df_spy['date'].max()}")

    # 2. 시나리오 분류
    print("\n[STEP 2] Classifying actual scenarios...")

    def classify_scenario(ret):
        if ret > 5:
            return 'Bullish'
        elif ret < -5:
            return 'Bearish'
        else:
            return 'Sideways'

    df_spy['actual_scenario'] = df_spy['return_60d'].apply(classify_scenario)

    # 3. Base Rate 계산 (무조건적)
    print("\n" + "=" * 80)
    print("[RESULT 1] UNCONDITIONAL BASE RATES (Historical)")
    print("=" * 80)

    base_rates = df_spy['actual_scenario'].value_counts(normalize=True) * 100

    print(f"\nBased on SPY 60-day forward returns:")
    for scenario in ['Bullish', 'Sideways', 'Bearish']:
        rate = base_rates.get(scenario, 0)
        print(f"  {scenario}: {rate:.1f}%")

    print("\nComparison with MACRO_ENVIRONMENT hardcoded values:")
    print(f"  - If base rate is ~{base_rates.get('Bullish', 0):.0f}% Bullish, ")
    print(f"    then SOFT_LANDING 60% is {'too high' if 60 > base_rates.get('Bullish', 0) + 10 else 'reasonable'}")
    print(f"  - If base rate is ~{base_rates.get('Bearish', 0):.0f}% Bearish,")
    print(f"    then STAGFLATION 75% is {'too high' if 75 > base_rates.get('Bearish', 0) + 30 else 'reasonable'}")

    # 4. 매크로 지표 가져오기 (GDP, PMI, CPI)
    print("\n[STEP 3] Loading macro indicators (GDP, PMI, CPI)...")

    # GDP 데이터 가져오기
    gdp_query = "SELECT date, value as gdp FROM us_gdp ORDER BY date"
    gdp_data = await db.execute_query(gdp_query)
    df_gdp = pd.DataFrame(gdp_data) if gdp_data else pd.DataFrame()

    # PMI 데이터 가져오기
    pmi_query = "SELECT date, value as pmi FROM us_pmi ORDER BY date"
    pmi_data = await db.execute_query(pmi_query)
    df_pmi = pd.DataFrame(pmi_data) if pmi_data else pd.DataFrame()

    # CPI 데이터 가져오기
    cpi_query = "SELECT date, value as cpi FROM us_cpi ORDER BY date"
    cpi_data = await db.execute_query(cpi_query)
    df_cpi = pd.DataFrame(cpi_data) if cpi_data else pd.DataFrame()

    print(f"  GDP records: {len(df_gdp)}")
    print(f"  PMI records: {len(df_pmi)}")
    print(f"  CPI records: {len(df_cpi)}")

    # SPY에 매크로 데이터 병합 (월별 → 일별로 forward fill)
    df_spy['date'] = pd.to_datetime(df_spy['date'])

    if len(df_gdp) > 0:
        df_gdp['date'] = pd.to_datetime(df_gdp['date'])
        df_gdp['gdp'] = df_gdp['gdp'].astype(float)
        df_gdp['gdp_growth'] = df_gdp['gdp'].pct_change() * 100
        df_spy = pd.merge_asof(df_spy.sort_values('date'),
                               df_gdp[['date', 'gdp', 'gdp_growth']].sort_values('date'),
                               on='date', direction='backward')

    if len(df_pmi) > 0:
        df_pmi['date'] = pd.to_datetime(df_pmi['date'])
        df_pmi['pmi'] = df_pmi['pmi'].astype(float)
        df_spy = pd.merge_asof(df_spy.sort_values('date'),
                               df_pmi[['date', 'pmi']].sort_values('date'),
                               on='date', direction='backward')

    if len(df_cpi) > 0:
        df_cpi['date'] = pd.to_datetime(df_cpi['date'])
        df_cpi['cpi'] = df_cpi['cpi'].astype(float)
        df_cpi['cpi_yoy'] = df_cpi['cpi'].pct_change(12) * 100  # YoY change
        df_cpi['cpi_falling'] = df_cpi['cpi_yoy'].diff() < 0
        df_spy = pd.merge_asof(df_spy.sort_values('date'),
                               df_cpi[['date', 'cpi_yoy', 'cpi_falling']].sort_values('date'),
                               on='date', direction='backward')

    # 실제 MACRO_ENVIRONMENT 분류 (us_market_regime.py 로직과 동일)
    def classify_macro_real(row):
        """
        실제 GDP/PMI/CPI 기반 분류:
        - SOFT_LANDING: GDP+, PMI>50, CPI falling
        - HARD_LANDING: GDP-, PMI<50, CPI falling
        - REFLATION: GDP+, PMI>50, CPI rising
        - STAGFLATION: GDP-, PMI<50, CPI rising
        - DEFLATION: GDP-, PMI<50, CPI falling sharply
        """
        gdp_growth = row.get('gdp_growth', None)
        pmi = row.get('pmi', None)
        cpi_falling = row.get('cpi_falling', None)

        if pd.isna(gdp_growth) or pd.isna(pmi) or pd.isna(cpi_falling):
            return 'UNKNOWN'

        gdp_positive = gdp_growth > 0
        pmi_expansionary = pmi > 50

        if gdp_positive and pmi_expansionary:
            return 'SOFT_LANDING' if cpi_falling else 'REFLATION'
        else:
            return 'HARD_LANDING' if cpi_falling else 'STAGFLATION'

    df_spy['macro_env_real'] = df_spy.apply(classify_macro_real, axis=1)

    # 간이 분류도 유지 (비교용)
    df_spy['return_252d'] = df_spy['close'].pct_change(252) * 100
    df_spy['volatility_60d'] = df_spy['close'].pct_change().rolling(60).std() * np.sqrt(252) * 100

    def classify_macro_simple(row):
        ret_252 = row.get('return_252d', 0)
        vol = row.get('volatility_60d', 15)

        if pd.isna(ret_252) or pd.isna(vol):
            return 'UNKNOWN'

        if ret_252 > 10 and vol < 20:
            return 'SOFT_LANDING'
        elif ret_252 < -10:
            return 'HARD_LANDING'
        elif vol > 25:
            return 'STAGFLATION'
        elif ret_252 > 5:
            return 'REFLATION'
        else:
            return 'DEFLATION'

    df_spy['macro_env_simple'] = df_spy.apply(classify_macro_simple, axis=1)

    # 5. Macro Environment별 조건부 확률 계산
    print("\n" + "=" * 80)
    print("[RESULT 2] CONDITIONAL SCENARIO RATES BY MACRO ENVIRONMENT")
    print("=" * 80)

    # 실제 GDP/PMI/CPI 분류 결과 출력
    print("\n[2.1] Using REAL GDP/PMI/CPI Classification:")
    print("-" * 75)
    real_dist = df_spy['macro_env_real'].value_counts()
    for env, count in real_dist.items():
        print(f"  {env}: {count} days ({count/len(df_spy)*100:.1f}%)")

    macro_envs = ['SOFT_LANDING', 'HARD_LANDING', 'REFLATION', 'STAGFLATION', 'DEFLATION']

    # 실제 분류 기반 분석
    print("\n[2.2] Actual Scenario Rates by REAL Macro Classification:")
    print("-" * 75)

    # 현재 하드코딩된 값
    CURRENT_HARDCODED = {
        'SOFT_LANDING': {'Bullish': 60, 'Sideways': 30, 'Bearish': 10},
        'HARD_LANDING': {'Bullish': 10, 'Sideways': 25, 'Bearish': 65},
        'REFLATION': {'Bullish': 50, 'Sideways': 35, 'Bearish': 15},
        'STAGFLATION': {'Bullish': 5, 'Sideways': 20, 'Bearish': 75},
        'DEFLATION': {'Bullish': 15, 'Sideways': 30, 'Bearish': 55}
    }

    results = []

    print(f"{'Macro Env':<15} | {'Scenario':<10} | {'ACTUAL':>8} | {'HARDCODED':>10} | {'GAP':>8} | {'N':>6}")
    print("-" * 75)

    for env in macro_envs:
        env_data = df_spy[df_spy['macro_env_real'] == env]
        n_samples = len(env_data)

        if n_samples < 30:
            print(f"{env:<15} | {'N/A':<10} | {'N/A':>8} | {'N/A':>10} | {'N/A':>8} | {n_samples:>6}")
            continue

        actual_rates = env_data['actual_scenario'].value_counts(normalize=True) * 100

        for scenario in ['Bullish', 'Sideways', 'Bearish']:
            actual = actual_rates.get(scenario, 0)
            hardcoded = CURRENT_HARDCODED[env][scenario]
            gap = actual - hardcoded

            results.append({
                'macro_env': env,
                'scenario': scenario,
                'actual_rate': actual,
                'hardcoded_rate': hardcoded,
                'gap': gap,
                'n_samples': n_samples
            })

            gap_str = f"{gap:+.1f}%"
            if abs(gap) > 15:
                gap_str += " [!]"

            print(f"{env:<15} | {scenario:<10} | {actual:>7.1f}% | {hardcoded:>9}% | {gap_str:>8} | {n_samples:>6}")

        print("-" * 75)

    # 6. 근본 원인 분석
    print("\n" + "=" * 80)
    print("[RESULT 3] ROOT CAUSE ANALYSIS")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    if len(df_results) > 0:
        avg_abs_gap = df_results['gap'].abs().mean()
        max_gap = df_results.loc[df_results['gap'].abs().idxmax()]

        print(f"\nAverage absolute gap: {avg_abs_gap:.1f}%")
        print(f"Maximum gap: {max_gap['macro_env']} {max_gap['scenario']} = {max_gap['gap']:+.1f}%")

        # Bullish 과대예측 분석
        bullish_results = df_results[df_results['scenario'] == 'Bullish']
        if len(bullish_results) > 0:
            bullish_gap = bullish_results['gap'].mean()
            print(f"\nBullish prediction bias: {bullish_gap:+.1f}%")
            if bullish_gap < -10:
                print("  -> HARDCODED probabilities are TOO OPTIMISTIC for Bullish")
            elif bullish_gap > 10:
                print("  -> HARDCODED probabilities are TOO PESSIMISTIC for Bullish")

        # Bearish 과대예측 분석
        bearish_results = df_results[df_results['scenario'] == 'Bearish']
        if len(bearish_results) > 0:
            bearish_gap = bearish_results['gap'].mean()
            print(f"Bearish prediction bias: {bearish_gap:+.1f}%")
            if bearish_gap < -10:
                print("  -> HARDCODED probabilities are TOO PESSIMISTIC for Bearish")
            elif bearish_gap > 10:
                print("  -> HARDCODED probabilities are TOO OPTIMISTIC for Bearish")

    print("\n" + "=" * 80)
    print("[CONCLUSION] FUNDAMENTAL IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)

    print("""
INSTEAD OF CURVE-FITTING COEFFICIENTS, DO THIS:

1. MEASURE EMPIRICAL RATES:
   - Calculate actual Bullish/Bearish rates for each MACRO_ENVIRONMENT
   - Use at least 3-5 years of historical data
   - Update MACRO_ENVIRONMENT['scenario_prob'] with measured values

2. VALIDATE MACRO CLASSIFICATION:
   - Is the GDP/PMI/CPI classification accurately detecting market regimes?
   - Test: When classified as SOFT_LANDING, does market behave as expected?

3. CONSIDER TIME HORIZON MISMATCH:
   - Macro environment = current state
   - Scenario outcome = 60-day forward return
   - Are they logically connected?

4. ADJUST SCENARIO THRESHOLDS IF NEEDED:
   - Current: Bullish > +5%, Bearish < -5%
   - If market historically returns +10%/year, +5% in 60 days is normal, not bullish

5. RE-EXAMINE THE MODEL ARCHITECTURE:
   - Should probability come from macro environment alone?
   - Or should it incorporate momentum, sentiment, valuation?
""")

    await db.close()

    return df_results


if __name__ == '__main__':
    asyncio.run(measure_actual_scenario_rates())
