"""
US Double Sort Analysis (Sector × Factor Conditional IC)

Objectives:
- Calculate Conditional IC (IC within each sector)
- Build Sector × Factor IC Matrix
- Validate sector-specific factor weight differentiation
- NASDAQ exchange special analysis (Momentum effectiveness)

Methodology:
- Primary Sort: Sector classification (11 sectors)
- Secondary Sort: Factor quintile within each sector
- Conditional IC: Spearman correlation within sector
- Multiple horizons: 30d, 90d, 252d forward returns

Data Source:
- us_stock_grade table (historical data)
- us_stock_basic (sector, exchange)
- forward_return_252d, forward_return_90d, forward_return_30d

File: us/analysis/us_double_sort_analysis.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional
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


class USDoubleSortAnalysis:
    """
    US Double Sort Analysis (Sector × Factor)

    Purpose: Quantify factor effectiveness within each sector
    """

    def __init__(self, db: AsyncDatabaseManager):
        self.db = db

    async def load_analysis_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load historical stock grade data with multiple forward return horizons

        Args:
            start_date: Analysis start date (if None, use MIN(date) from table)
            end_date: Analysis end date (if None, use MAX(date) - 365 days)

        Returns:
            DataFrame with columns:
            - date, symbol, sector, exchange, industry
            - final_score, growth_score, momentum_score, quality_score, value_score
            - forward_return_252d, forward_return_90d, forward_return_30d
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
                b.exchange,
                b.industry,
                g.final_score,
                g.growth_score,
                g.momentum_score,
                g.quality_score,
                g.value_score,
                d_current.close as current_price,
                d_30d.close as price_30d,
                d_90d.close as price_90d,
                d_252d.close as price_252d
            FROM us_stock_grade g
            INNER JOIN us_stock_basic b ON g.symbol = b.symbol
            INNER JOIN us_daily d_current ON g.symbol = d_current.symbol AND g.date = d_current.date
            LEFT JOIN us_daily d_30d ON g.symbol = d_30d.symbol
                AND d_30d.date = (
                    SELECT date FROM us_daily
                    WHERE symbol = g.symbol AND date > g.date
                    ORDER BY date ASC LIMIT 1 OFFSET 29
                )
            LEFT JOIN us_daily d_90d ON g.symbol = d_90d.symbol
                AND d_90d.date = (
                    SELECT date FROM us_daily
                    WHERE symbol = g.symbol AND date > g.date
                    ORDER BY date ASC LIMIT 1 OFFSET 89
                )
            LEFT JOIN us_daily d_252d ON g.symbol = d_252d.symbol
                AND d_252d.date = (
                    SELECT date FROM us_daily
                    WHERE symbol = g.symbol AND date > g.date
                    ORDER BY date ASC LIMIT 1 OFFSET 251
                )
            WHERE g.date BETWEEN $1 AND $2
              AND b.sector IS NOT NULL
              AND g.final_score IS NOT NULL
        )
        SELECT
            date, symbol, sector, exchange, industry,
            final_score, growth_score, momentum_score, quality_score, value_score,
            current_price,
            CASE
                WHEN price_30d IS NOT NULL AND current_price > 0
                THEN (price_30d - current_price) / current_price * 100
                ELSE NULL
            END as forward_return_30d,
            CASE
                WHEN price_90d IS NOT NULL AND current_price > 0
                THEN (price_90d - current_price) / current_price * 100
                ELSE NULL
            END as forward_return_90d,
            CASE
                WHEN price_252d IS NOT NULL AND current_price > 0
                THEN (price_252d - current_price) / current_price * 100
                ELSE NULL
            END as forward_return_252d
        FROM stock_data
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
            'current_price', 'forward_return_30d', 'forward_return_90d', 'forward_return_252d'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Loaded {len(df):,} samples")
        logger.info(f"  - 252d returns: {df['forward_return_252d'].notna().sum():,}")
        logger.info(f"  - 90d returns: {df['forward_return_90d'].notna().sum():,}")
        logger.info(f"  - 30d returns: {df['forward_return_30d'].notna().sum():,}")

        return df

    def perform_double_sort_analysis(
        self,
        df: pd.DataFrame,
        return_horizon: str = 'forward_return_252d',
        min_samples: int = 100
    ) -> pd.DataFrame:
        """
        Perform Double Sort Analysis: Sector × Factor IC

        Args:
            df: Stock data
            return_horizon: 'forward_return_252d', 'forward_return_90d', or 'forward_return_30d'
            min_samples: Minimum samples per sector

        Returns:
            Sector × Factor IC matrix DataFrame
        """

        if return_horizon not in df.columns:
            raise ValueError(f"Return horizon '{return_horizon}' not found")

        sectors = df['sector'].unique()
        factors = ['growth_score', 'momentum_score', 'quality_score', 'value_score']

        results = []

        for sector in sectors:
            sector_df = df[df['sector'] == sector].copy()

            # Drop rows with missing returns
            sector_df = sector_df.dropna(subset=[return_horizon])

            if len(sector_df) < min_samples:
                logger.debug(f"Skipping {sector}: insufficient samples ({len(sector_df)})")
                continue

            sector_result = {
                'sector': sector,
                'n_samples': len(sector_df)
            }

            # Calculate IC for each factor
            for factor in factors:
                valid_data = sector_df[[factor, return_horizon]].dropna()

                if len(valid_data) < min_samples:
                    sector_result[f'{factor}_ic'] = np.nan
                    sector_result[f'{factor}_pval'] = np.nan
                    continue

                ic, pval = spearmanr(
                    valid_data[factor],
                    valid_data[return_horizon]
                )

                sector_result[f'{factor}_ic'] = round(ic, 4)
                sector_result[f'{factor}_pval'] = round(pval, 4)

            results.append(sector_result)

        # Create DataFrame
        sector_factor_ic_df = pd.DataFrame(results)

        # Sort by growth_score_ic (descending)
        if 'growth_score_ic' in sector_factor_ic_df.columns:
            sector_factor_ic_df = sector_factor_ic_df.sort_values(
                'growth_score_ic',
                ascending=False,
                na_position='last'
            )

        return sector_factor_ic_df

    def print_double_sort_results(
        self,
        sector_factor_ic_df: pd.DataFrame,
        return_horizon: str = '252d'
    ):
        """Print double sort analysis results"""

        print("\n" + "="*100)
        print(f"Double Sort Analysis: Sector × Factor IC Matrix ({return_horizon})")
        print("="*100)

        # Display table
        display_cols = ['sector', 'n_samples',
                        'growth_score_ic', 'momentum_score_ic',
                        'quality_score_ic', 'value_score_ic']

        display_df = sector_factor_ic_df[display_cols].copy()
        display_df.columns = ['Sector', 'Samples', 'Growth IC', 'Momentum IC', 'Quality IC', 'Value IC']

        print(display_df.to_string(index=False))
        print("="*100)

    def recommend_sector_weights(
        self,
        sector_factor_ic_df: pd.DataFrame
    ):
        """
        Recommend factor weights by sector based on IC

        Rules:
        - IC > 0.15: 30%+ weight
        - IC 0.10~0.15: 20-25% weight
        - IC 0.05~0.10: 15-20% weight
        - IC < 0.05: 10% weight
        """

        print("\n" + "="*100)
        print("Recommended Factor Weights by Sector (Based on IC)")
        print("="*100)

        for _, row in sector_factor_ic_df.iterrows():
            sector = row['sector']
            n_samples = row['n_samples']

            print(f"\n[{sector}] (n={n_samples:,})")

            # Extract factor ICs
            factor_ics = {
                'Growth': row.get('growth_score_ic', 0),
                'Momentum': row.get('momentum_score_ic', 0),
                'Quality': row.get('quality_score_ic', 0),
                'Value': row.get('value_score_ic', 0)
            }

            # Sort by IC
            sorted_factors = sorted(factor_ics.items(), key=lambda x: x[1], reverse=True)

            # Calculate weights (IC proportion for positive ICs)
            total_positive_ic = sum([ic for _, ic in sorted_factors if ic > 0])

            if total_positive_ic <= 0:
                print("  WARNING: All factors IC <= 0, use default weights")
                continue

            print("  Factor       IC      Recommended Weight  Status")
            print("  " + "-"*60)

            for factor, ic in sorted_factors:
                if ic > 0:
                    weight = (ic / total_positive_ic) * 100
                else:
                    weight = 5  # Minimum weight

                # Status
                if ic > 0.15:
                    status = "[EXCELLENT]"
                elif ic > 0.10:
                    status = "[GOOD]"
                elif ic > 0.05:
                    status = "[FAIR]"
                else:
                    status = "[WEAK]"

                print(f"  {factor:12} {ic:>6.3f}   {weight:>5.1f}%            {status}")

    async def nasdaq_sector_analysis(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        NASDAQ exchange sector × factor analysis

        Purpose: Validate NASDAQ Momentum reduction strategy
        """

        nasdaq_df = df[df['exchange'] == 'NASDAQ'].copy()

        if nasdaq_df.empty:
            logger.warning("No NASDAQ data found")
            return pd.DataFrame()

        print("\n\n" + "="*100)
        print("NASDAQ EXCHANGE: Sector × Factor Analysis")
        print("="*100)
        print(f"Total NASDAQ stocks: {len(nasdaq_df):,}")

        # Perform double sort on NASDAQ data
        nasdaq_ic_df = self.perform_double_sort_analysis(
            nasdaq_df,
            return_horizon='forward_return_252d',
            min_samples=50
        )

        self.print_double_sort_results(nasdaq_ic_df, '252d')

        # Focus on Technology & Healthcare (largest NASDAQ sectors)
        tech_df = nasdaq_df[nasdaq_df['sector'] == 'Technology'].copy()
        healthcare_df = nasdaq_df[nasdaq_df['sector'] == 'Healthcare'].copy()

        print("\n" + "="*100)
        print("FOCUS: Technology & Healthcare (Major NASDAQ Sectors)")
        print("="*100)

        factors = ['growth_score', 'momentum_score', 'quality_score', 'value_score']

        for sector_name, sector_df in [('Technology', tech_df), ('Healthcare', healthcare_df)]:
            if sector_df.empty:
                continue

            print(f"\n[{sector_name}] (n={len(sector_df):,})")
            print("  Factor       IC (252d)  IC (90d)  IC (30d)")
            print("  " + "-"*55)

            for factor in factors:
                # Calculate IC for multiple horizons
                ic_252d = ic_90d = ic_30d = np.nan

                data_252d = sector_df[[factor, 'forward_return_252d']].dropna()
                if len(data_252d) >= 50:
                    ic_252d, _ = spearmanr(data_252d[factor], data_252d['forward_return_252d'])

                data_90d = sector_df[[factor, 'forward_return_90d']].dropna()
                if len(data_90d) >= 50:
                    ic_90d, _ = spearmanr(data_90d[factor], data_90d['forward_return_90d'])

                data_30d = sector_df[[factor, 'forward_return_30d']].dropna()
                if len(data_30d) >= 50:
                    ic_30d, _ = spearmanr(data_30d[factor], data_30d['forward_return_30d'])

                factor_label = factor.replace('_score', '').capitalize()
                print(f"  {factor_label:12} {ic_252d:>6.3f}     {ic_90d:>6.3f}    {ic_30d:>6.3f}")

        print("\n" + "="*100)
        print("Validation Results:")
        print("  - Technology Momentum IC < 0.10: Momentum reduction strategy is valid")
        print("  - Healthcare Growth IC > 0.10: Growth-centric strategy is valid")
        print("="*100)

        return nasdaq_ic_df

    async def exchange_comparison(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare factor effectiveness across exchanges (NYSE vs NASDAQ)

        Purpose: Quantify exchange-level factor differences
        """

        print("\n\n" + "="*100)
        print("EXCHANGE COMPARISON: NYSE vs NASDAQ Factor IC")
        print("="*100)

        exchanges = ['NYSE', 'NASDAQ']
        factors = ['growth_score', 'momentum_score', 'quality_score', 'value_score']

        results = []

        for exchange in exchanges:
            exchange_df = df[df['exchange'] == exchange].copy()
            exchange_df = exchange_df.dropna(subset=['forward_return_252d'])

            if len(exchange_df) < 100:
                continue

            exchange_result = {
                'exchange': exchange,
                'n_samples': len(exchange_df)
            }

            for factor in factors:
                valid_data = exchange_df[[factor, 'forward_return_252d']].dropna()

                if len(valid_data) < 100:
                    exchange_result[f'{factor}_ic'] = np.nan
                    continue

                ic, pval = spearmanr(valid_data[factor], valid_data['forward_return_252d'])
                exchange_result[f'{factor}_ic'] = round(ic, 4)
                exchange_result[f'{factor}_pval'] = round(pval, 4)

            results.append(exchange_result)

        comparison_df = pd.DataFrame(results)

        print(comparison_df.to_string(index=False))
        print("="*100)

        # Calculate differences
        if len(comparison_df) == 2:
            print("\nKey Findings:")
            for factor in factors:
                ic_col = f'{factor}_ic'
                if ic_col in comparison_df.columns:
                    nyse_ic = comparison_df.loc[comparison_df['exchange'] == 'NYSE', ic_col].values
                    nasdaq_ic = comparison_df.loc[comparison_df['exchange'] == 'NASDAQ', ic_col].values

                    if len(nyse_ic) > 0 and len(nasdaq_ic) > 0:
                        diff = nasdaq_ic[0] - nyse_ic[0]
                        pct_diff = (diff / nyse_ic[0] * 100) if nyse_ic[0] != 0 else 0

                        factor_name = factor.replace('_score', '').capitalize()
                        print(f"  {factor_name:12} NASDAQ vs NYSE: {diff:>+6.3f} ({pct_diff:>+6.1f}%)")

        return comparison_df

    def visualize_heatmap(
        self,
        sector_factor_ic_df: pd.DataFrame,
        output_path: str = "us/analysis/output/sector_factor_ic_heatmap.png"
    ):
        """
        Visualize Sector × Factor IC as heatmap

        Args:
            sector_factor_ic_df: Sector × Factor IC matrix
            output_path: Output file path
        """

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping heatmap")
            return

        # Extract IC columns
        ic_columns = [col for col in sector_factor_ic_df.columns if col.endswith('_ic')]

        if not ic_columns:
            logger.warning("No IC columns found, skipping heatmap")
            return

        heatmap_data = sector_factor_ic_df[['sector'] + ic_columns].set_index('sector')

        # Clean column names
        heatmap_data.columns = [col.replace('_score_ic', '').upper() for col in heatmap_data.columns]

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            vmin=-0.10,
            vmax=0.30,
            cbar_kws={'label': 'Information Coefficient (IC)'}
        )
        plt.title('Sector × Factor IC Heatmap (252d Forward Return)', fontsize=14, fontweight='bold')
        plt.xlabel('Factor', fontsize=12)
        plt.ylabel('Sector', fontsize=12)
        plt.tight_layout()

        # Save
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"\nHeatmap saved: {output_path}")


# Standalone execution functions
async def run_full_analysis(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    output_dir: str = "us/result"
):
    """
    Run full double sort analysis suite

    Args:
        start_date: Analysis start date (if None, use ALL data from table)
        end_date: Analysis end date (if None, use ALL data from table)
        output_dir: Output directory for CSV files
    """

    print("\n" + "="*80)
    print("US DOUBLE SORT ANALYSIS (Sector × Factor)")
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
        analyzer = USDoubleSortAnalysis(db)

        # Step 1: Load data
        print("\nStep 1: Loading historical data...")
        df = await analyzer.load_analysis_data(start_date, end_date)

        if df.empty:
            print("No data available for analysis")
            return

        # Step 2: Overall sector × factor analysis (252d)
        print("\nStep 2: Sector × Factor IC Analysis (252d)...")
        sector_ic_252d = analyzer.perform_double_sort_analysis(
            df,
            return_horizon='forward_return_252d',
            min_samples=100
        )
        analyzer.print_double_sort_results(sector_ic_252d, '252d')

        # Save
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/sector_factor_ic_252d_{start_date}_{end_date}.csv"
        sector_ic_252d.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

        # Step 3: Recommend weights
        print("\nStep 3: Recommended Factor Weights by Sector...")
        analyzer.recommend_sector_weights(sector_ic_252d)

        # Step 4: NASDAQ special analysis
        print("\nStep 4: NASDAQ Exchange Analysis...")
        nasdaq_ic = await analyzer.nasdaq_sector_analysis(df)

        if not nasdaq_ic.empty:
            nasdaq_path = f"{output_dir}/nasdaq_sector_factor_ic_{start_date}_{end_date}.csv"
            nasdaq_ic.to_csv(nasdaq_path, index=False)
            print(f"\nSaved: {nasdaq_path}")

        # Step 5: Exchange comparison
        print("\nStep 5: Exchange Comparison (NYSE vs NASDAQ)...")
        exchange_comp = await analyzer.exchange_comparison(df)

        if not exchange_comp.empty:
            exchange_path = f"{output_dir}/exchange_factor_ic_comparison_{start_date}_{end_date}.csv"
            exchange_comp.to_csv(exchange_path, index=False)
            print(f"\nSaved: {exchange_path}")

        # Step 6: Heatmap visualization
        print("\nStep 6: Generating Heatmap...")
        heatmap_path = f"{output_dir}/sector_factor_ic_heatmap_{start_date}_{end_date}.png"
        analyzer.visualize_heatmap(sector_ic_252d, heatmap_path)

        print("\n" + "="*80)
        print("DOUBLE SORT ANALYSIS COMPLETE")
        print("="*80)

    finally:
        await db.close()


if __name__ == '__main__':
    # Analyze ALL data in us_stock_grade table
    # Dates will be auto-detected: MIN(date) to MAX(date) - 365 days
    asyncio.run(run_full_analysis())
