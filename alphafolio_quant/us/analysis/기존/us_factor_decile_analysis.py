"""
US Factor Decile/Quintile Analysis

Objectives:
- Verify factor monotonicity (Top Decile > Bottom Decile)
- Calculate Long-Short Spread (Top 10% - Bottom 10% return)
- Detect non-linearity (middle deciles)
- Healthcare sector Growth factor validation

Methodology:
- Decile Analysis: 10 groups (Fama-French style)
- Quintile Analysis: 5 groups (AQR Capital style)
- Monotonicity IC: Spearman correlation between group rank and return
- Long-Short Spread: Top Decile return - Bottom Decile return

Data Source:
- us_stock_grade table (historical data)
- forward_return_252d (1-year forward return)

File: us/analysis/us_factor_decile_analysis.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
import logging

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from us_db_async import AsyncDatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class USFactorDecileAnalysis:
    """
    US Factor Decile/Quintile Analysis

    Purpose: Verify factor effectiveness through decile/quintile sorting
    """

    def __init__(self, db: AsyncDatabaseManager):
        self.db = db

    async def load_analysis_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        min_samples_per_date: int = 100
    ) -> pd.DataFrame:
        """
        Load historical stock grade data for analysis

        Args:
            start_date: Analysis start date (if None, use MIN(date) from table)
            end_date: Analysis end date (if None, use MAX(date) - 365 days)
            min_samples_per_date: Minimum stocks per date

        Returns:
            DataFrame with columns:
            - date, symbol, sector
            - final_score, growth_score, momentum_score, quality_score, value_score
            - forward_return_252d (calculated)
        """

        # Auto-detect date range if not provided
        if start_date is None or end_date is None:
            date_query = """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM us_stock_grade
            WHERE final_score IS NOT NULL
            """
            date_result = await self.db.execute_query(date_query)

            if not date_result or not date_result[0]['min_date']:
                logger.error("No data found in us_stock_grade table")
                return pd.DataFrame()

            if start_date is None:
                start_date = date_result[0]['min_date']

            if end_date is None:
                # Need 252d forward return, so end at least 365 days before today
                max_date = date_result[0]['max_date']
                today = date.today()
                end_date = min(max_date, today - timedelta(days=365))

            logger.info(f"Auto-detected date range: {start_date} to {end_date}")

        query = """
        WITH stock_data AS (
            SELECT
                g.date,
                g.symbol,
                b.sector,
                b.industry,
                g.final_score,
                g.growth_score,
                g.momentum_score,
                g.quality_score,
                g.value_score,
                d_current.close as current_price,
                d_future.close as future_price_252d
            FROM us_stock_grade g
            INNER JOIN us_stock_basic b ON g.symbol = b.symbol
            INNER JOIN us_daily d_current ON g.symbol = d_current.symbol AND g.date = d_current.date
            LEFT JOIN us_daily d_future ON g.symbol = d_future.symbol
                AND d_future.date = (
                    SELECT date
                    FROM us_daily
                    WHERE symbol = g.symbol AND date > g.date
                    ORDER BY date ASC
                    LIMIT 1 OFFSET 251  -- 252 trading days forward
                )
            WHERE g.date BETWEEN $1 AND $2
              AND b.sector IS NOT NULL
              AND g.final_score IS NOT NULL
        )
        SELECT
            date,
            symbol,
            sector,
            industry,
            final_score,
            growth_score,
            momentum_score,
            quality_score,
            value_score,
            current_price,
            future_price_252d,
            CASE
                WHEN future_price_252d IS NOT NULL AND current_price > 0
                THEN (future_price_252d - current_price) / current_price * 100
                ELSE NULL
            END as forward_return_252d
        FROM stock_data
        WHERE future_price_252d IS NOT NULL
        ORDER BY date, symbol
        """

        logger.info(f"Loading data from {start_date} to {end_date}...")
        rows = await self.db.execute_query(query, start_date, end_date)

        if not rows:
            logger.warning("No data found")
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])

        # Convert to numeric
        numeric_cols = [
            'final_score', 'growth_score', 'momentum_score', 'quality_score', 'value_score',
            'current_price', 'future_price_252d', 'forward_return_252d'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with missing forward returns
        df = df.dropna(subset=['forward_return_252d'])

        logger.info(f"Loaded {len(df):,} samples ({df['symbol'].nunique()} unique symbols)")

        return df

    def perform_decile_analysis(
        self,
        df: pd.DataFrame,
        factor_name: str = 'growth_score',
        n_groups: int = 10
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform Decile/Quintile Analysis for a factor

        Args:
            df: Stock data with factor scores and forward returns
            factor_name: Factor column name
            n_groups: Number of groups (10=Decile, 5=Quintile)

        Returns:
            (group_stats_df, summary_dict)
        """

        if factor_name not in df.columns:
            raise ValueError(f"Factor '{factor_name}' not found in DataFrame")

        # Drop missing values
        analysis_df = df[[factor_name, 'forward_return_252d']].dropna().copy()

        if len(analysis_df) < n_groups * 10:
            logger.warning(f"Insufficient data for {n_groups}-group analysis: {len(analysis_df)} samples")
            return pd.DataFrame(), {}

        # Assign groups (1 to n_groups)
        analysis_df['factor_group'] = pd.qcut(
            analysis_df[factor_name],
            q=n_groups,
            labels=False,
            duplicates='drop'
        ) + 1

        # Group statistics
        group_stats = analysis_df.groupby('factor_group').agg({
            'forward_return_252d': ['mean', 'median', 'std', 'count'],
            factor_name: ['min', 'max']
        }).round(4)

        # Flatten column names
        group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]
        group_stats = group_stats.reset_index()

        # Monotonicity IC (Spearman correlation between group and return)
        monotonicity_ic, ic_pval = spearmanr(
            analysis_df['factor_group'],
            analysis_df['forward_return_252d']
        )

        # Long-Short Spread
        top_return = analysis_df[analysis_df['factor_group'] == n_groups]['forward_return_252d'].mean()
        bottom_return = analysis_df[analysis_df['factor_group'] == 1]['forward_return_252d'].mean()
        long_short_spread = top_return - bottom_return

        # Middle groups
        middle_groups = [n_groups // 2, n_groups // 2 + 1]
        middle_df = analysis_df[analysis_df['factor_group'].isin(middle_groups)]
        middle_return = middle_df['forward_return_252d'].mean()

        summary = {
            'factor': factor_name,
            'n_groups': n_groups,
            'n_samples': len(analysis_df),
            'monotonicity_ic': round(monotonicity_ic, 4),
            'ic_pval': round(ic_pval, 4),
            'top_decile_return': round(top_return, 2),
            'bottom_decile_return': round(bottom_return, 2),
            'long_short_spread': round(long_short_spread, 2),
            'middle_return': round(middle_return, 2)
        }

        return group_stats, summary

    def print_decile_results(
        self,
        group_stats: pd.DataFrame,
        summary: Dict,
        title: str = "Decile Analysis"
    ):
        """Print decile analysis results"""

        print("\n" + "="*80)
        print(f"{title}: {summary['factor']}")
        print("="*80)
        print(f"\nSamples: {summary['n_samples']:,}")
        print(f"Groups: {summary['n_groups']}")
        print(f"Monotonicity IC: {summary['monotonicity_ic']:.4f} (p-value: {summary['ic_pval']:.4f})")

        if summary['monotonicity_ic'] > 0.10:
            print("  -> Strong monotonicity [OK]")
        elif summary['monotonicity_ic'] > 0.05:
            print("  -> Moderate monotonicity [FAIR]")
        else:
            print("  -> Weak monotonicity [WARNING]")

        print("\nGroup-wise Statistics:")
        print(group_stats.to_string(index=False))

        print("\n" + "="*80)
        print("Long-Short Analysis:")
        print(f"  Top Decile ({summary['n_groups']}) Avg Return:    {summary['top_decile_return']:>7.2f}%")
        print(f"  Bottom Decile (1) Avg Return:  {summary['bottom_decile_return']:>7.2f}%")
        print(f"  Long-Short Spread:             {summary['long_short_spread']:>7.2f}%")

        if summary['long_short_spread'] > 30:
            print("  -> Very strong factor [EXCELLENT]")
        elif summary['long_short_spread'] > 15:
            print("  -> Strong factor [GOOD]")
        elif summary['long_short_spread'] > 5:
            print("  -> Moderate factor [FAIR]")
        else:
            print("  -> Weak factor [WARNING]")

        print("="*80)

        print(f"\nMiddle Groups ({summary['n_groups']//2}-{summary['n_groups']//2+1}) Avg Return: {summary['middle_return']:>7.2f}%")
        if abs(summary['middle_return']) > 5:
            print("  -> Middle groups show effect (good monotonicity)")
        else:
            print("  -> Middle groups show little effect (potential non-linearity)")

    async def analyze_all_factors(
        self,
        df: pd.DataFrame,
        n_groups: int = 10
    ) -> pd.DataFrame:
        """
        Analyze all factors (Growth, Momentum, Quality, Value, Final Score)

        Returns:
            Summary DataFrame with all factors
        """

        factors = [
            'growth_score',
            'momentum_score',
            'quality_score',
            'value_score',
            'final_score'
        ]

        summaries = []

        for factor in factors:
            if factor not in df.columns:
                logger.warning(f"Factor '{factor}' not found, skipping")
                continue

            print("\n\n" + "#"*80)
            print(f"# Analyzing: {factor}")
            print("#"*80)

            group_stats, summary = self.perform_decile_analysis(df, factor, n_groups)

            if not group_stats.empty:
                self.print_decile_results(group_stats, summary, f"Decile Analysis ({n_groups} groups)")
                summaries.append(summary)

        # Summary table
        summary_df = pd.DataFrame(summaries)

        print("\n\n" + "="*80)
        print("SUMMARY: All Factors Comparison")
        print("="*80)
        print(summary_df[['factor', 'monotonicity_ic', 'long_short_spread', 'n_samples']].to_string(index=False))
        print("="*80)

        return summary_df

    async def healthcare_growth_analysis(
        self,
        df: pd.DataFrame,
        n_groups: int = 10
    ) -> Dict:
        """
        Healthcare sector Growth factor special analysis

        Purpose: Validate Healthcare IC 0.119 -> 0.160 improvement strategy
        """

        healthcare_df = df[df['sector'] == 'Healthcare'].copy()

        if len(healthcare_df) < 100:
            logger.warning("Insufficient Healthcare samples")
            return {}

        print("\n\n" + "="*80)
        print("HEALTHCARE SECTOR: Growth Factor Decile Analysis")
        print("="*80)
        print(f"Total Healthcare stocks: {len(healthcare_df):,}")

        # Overall Healthcare Growth analysis
        group_stats, summary = self.perform_decile_analysis(
            healthcare_df,
            factor_name='growth_score',
            n_groups=n_groups
        )

        self.print_decile_results(group_stats, summary, "Healthcare Growth Factor")

        # Biotech vs Others comparison
        if 'industry' in healthcare_df.columns:
            biotech_df = healthcare_df[
                healthcare_df['industry'].str.contains('BIOTECH', case=False, na=False)
            ]
            others_df = healthcare_df[
                ~healthcare_df['industry'].str.contains('BIOTECH', case=False, na=False)
            ]

            print("\n" + "-"*80)
            print("Sub-industry Comparison:")
            print("-"*80)

            if len(biotech_df) >= 50:
                biotech_ic, biotech_pval = spearmanr(
                    biotech_df['growth_score'],
                    biotech_df['forward_return_252d'],
                    nan_policy='omit'
                )
                print(f"Biotech Growth IC: {biotech_ic:>7.4f} (p={biotech_pval:.4f}, n={len(biotech_df):,})")
            else:
                print(f"Biotech Growth IC: N/A (insufficient samples: {len(biotech_df)})")

            if len(others_df) >= 50:
                others_ic, others_pval = spearmanr(
                    others_df['growth_score'],
                    others_df['forward_return_252d'],
                    nan_policy='omit'
                )
                print(f"Other Healthcare Growth IC: {others_ic:>7.4f} (p={others_pval:.4f}, n={len(others_df):,})")
            else:
                print(f"Other Healthcare Growth IC: N/A (insufficient samples: {len(others_df)})")

            print("\n-> Conclusion:")
            if len(biotech_df) >= 50 and len(others_df) >= 50:
                if biotech_ic < 0.05 and others_ic > 0.10:
                    print("   Biotech shows weak Growth factor effectiveness")
                    print("   Other Healthcare shows strong Growth factor effectiveness")
                    print("   Strategy: Apply Growth-centric weights (Biotech 60%, Others 35%)")
                else:
                    print("   Both sub-industries show similar Growth factor effectiveness")

        return summary

    async def sector_factor_comparison(
        self,
        df: pd.DataFrame,
        factor_name: str = 'growth_score'
    ) -> pd.DataFrame:
        """
        Compare factor effectiveness across sectors

        Args:
            df: Full dataset
            factor_name: Factor to analyze

        Returns:
            Sector comparison DataFrame
        """

        sectors = df['sector'].unique()
        results = []

        print("\n\n" + "="*80)
        print(f"SECTOR COMPARISON: {factor_name}")
        print("="*80)

        for sector in sorted(sectors):
            sector_df = df[df['sector'] == sector].copy()

            if len(sector_df) < 100:
                continue

            # Calculate IC
            ic, pval = spearmanr(
                sector_df[factor_name],
                sector_df['forward_return_252d'],
                nan_policy='omit'
            )

            # Long-Short Spread (Top 20% vs Bottom 20%)
            top_20_threshold = sector_df[factor_name].quantile(0.80)
            bottom_20_threshold = sector_df[factor_name].quantile(0.20)

            top_20_return = sector_df[sector_df[factor_name] >= top_20_threshold]['forward_return_252d'].mean()
            bottom_20_return = sector_df[sector_df[factor_name] <= bottom_20_threshold]['forward_return_252d'].mean()
            spread = top_20_return - bottom_20_return

            results.append({
                'sector': sector,
                'n_samples': len(sector_df),
                f'{factor_name}_ic': round(ic, 4),
                'ic_pval': round(pval, 4),
                'top_20_return': round(top_20_return, 2),
                'bottom_20_return': round(bottom_20_return, 2),
                'long_short_spread': round(spread, 2)
            })

        sector_comparison_df = pd.DataFrame(results)
        sector_comparison_df = sector_comparison_df.sort_values(f'{factor_name}_ic', ascending=False)

        print(sector_comparison_df.to_string(index=False))
        print("="*80)

        return sector_comparison_df


# Standalone execution functions
async def run_full_analysis(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    output_dir: str = "us/result"
):
    """
    Run full decile analysis suite

    Args:
        start_date: Analysis start date (if None, use ALL data from table)
        end_date: Analysis end date (if None, use ALL data from table)
        output_dir: Output directory for CSV files
    """

    print("\n" + "="*80)
    print("US FACTOR DECILE ANALYSIS")
    print("="*80)
    if start_date and end_date:
        print(f"Period: {start_date} to {end_date}")
    else:
        print("Period: ALL available data in us_stock_grade table")
    print(f"Output: {output_dir}")
    print("="*80)

    # Initialize DB
    db = AsyncDatabaseManager()
    await db.initialize()

    try:
        analyzer = USFactorDecileAnalysis(db)

        # Step 1: Load data
        print("\nStep 1: Loading historical data...")
        df = await analyzer.load_analysis_data(start_date, end_date)

        if df.empty:
            print("No data available for analysis")
            return

        # Step 2: Analyze all factors
        print("\nStep 2: Analyzing all factors (Decile)...")
        summary_df = await analyzer.analyze_all_factors(df, n_groups=10)

        # Save summary
        import os
        os.makedirs(output_dir, exist_ok=True)
        summary_path = f"{output_dir}/factor_decile_summary_{start_date}_{end_date}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")

        # Step 3: Healthcare Growth special analysis
        print("\nStep 3: Healthcare Growth Factor Analysis...")
        healthcare_summary = await analyzer.healthcare_growth_analysis(df, n_groups=10)

        # Step 4: Sector comparison for all factors
        print("\nStep 4: Sector Comparison for Key Factors...")
        for factor in ['growth_score', 'momentum_score']:
            sector_comp_df = await analyzer.sector_factor_comparison(df, factor)

            sector_comp_path = f"{output_dir}/sector_{factor}_comparison_{start_date}_{end_date}.csv"
            sector_comp_df.to_csv(sector_comp_path, index=False)
            print(f"Saved: {sector_comp_path}")

        print("\n" + "="*80)
        print("DECILE ANALYSIS COMPLETE")
        print("="*80)

    finally:
        await db.close()


if __name__ == '__main__':
    # Analyze ALL data in us_stock_grade table
    # Dates will be auto-detected: MIN(date) to MAX(date) - 365 days
    asyncio.run(run_full_analysis())
