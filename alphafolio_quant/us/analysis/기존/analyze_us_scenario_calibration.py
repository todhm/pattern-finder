"""
US Scenario Calibration Analysis Script

Analyzes us_stock_grade data to generate insights for:
1. Sector-based Calibration Map
2. Backtest reliability adjustment
3. Error case detection
4. Return range accuracy improvement

US Model Characteristics:
- Bullish: return_60d > +5%
- Sideways: -5% <= return_60d <= +5%
- Bearish: return_60d < -5%
- Classification: Sector-based (not Theme)
- No Calibration applied (unlike KR model)

Output: CSV files in C:/project/alpha/quant/us/result/
"""

import asyncio
import asyncpg
import os
import pandas as pd
import numpy as np
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

# Output directory
OUTPUT_DIR = r"C:\project\alpha\quant\us\result"


class USScenarioCalibrationAnalyzer:
    """Analyzer for US scenario probability calibration"""

    def __init__(self):
        self.conn = None
        self.results = {}

    async def connect(self):
        """Connect to database"""
        database_url = os.getenv('DATABASE_URL')
        self.conn = await asyncpg.connect(database_url)
        print("Database connected")

    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            print("Database connection closed")

    async def run_all_analyses(self):
        """Run all analyses"""
        print("\n" + "=" * 80)
        print("US Scenario Calibration Analysis")
        print("=" * 80)
        print("Note: US model uses Bullish>+5%, Sideways -5%~+5%, Bearish<-5%")
        print("=" * 80)

        await self.connect()

        try:
            # Analysis 1: Sector-based actual distribution
            print("\n[1/8] Analyzing sector-based actual distribution...")
            await self.analyze_sector_actual_distribution()

            # Analysis 2: Calibration error by sector
            print("\n[2/8] Analyzing prediction error by sector...")
            await self.analyze_prediction_error()

            # Analysis 3: Error cases detection
            print("\n[3/8] Detecting error cases...")
            await self.detect_error_cases()

            # Analysis 4: Sample count accuracy
            print("\n[4/8] Analyzing sample count accuracy...")
            await self.analyze_sample_count_accuracy()

            # Analysis 5: Recommended calibration map
            print("\n[5/8] Generating recommended calibration map...")
            await self.generate_recommended_calibration()

            # Analysis 6: Return range accuracy
            print("\n[6/8] Analyzing return range accuracy...")
            await self.analyze_return_range_accuracy()

            # Analysis 7: Return percentiles by sector
            print("\n[7/8] Calculating return percentiles by sector...")
            await self.calculate_return_percentiles_by_sector()

            # Analysis 8: Return percentiles by score bucket
            print("\n[8/8] Calculating return percentiles by score bucket...")
            await self.calculate_return_percentiles_by_score()

            print("\n" + "=" * 80)
            print("All analyses completed!")
            print(f"Results saved to: {OUTPUT_DIR}")
            print("=" * 80)

        finally:
            await self.close()

    async def analyze_sector_actual_distribution(self):
        """
        Analysis 1: Sector-based actual return distribution
        Calculate actual bullish/sideways/bearish rates for each sector
        """
        query = """
        WITH actual_returns AS (
            SELECT
                b.sector,
                g.symbol,
                g.date as analysis_date,
                g.final_score,
                g.scenario_bullish_prob,
                g.scenario_sideways_prob,
                g.scenario_bearish_prob,
                g.scenario_sample_count,
                d_start.close as start_close,
                d_end.close as end_close,
                CASE
                    WHEN d_end.close IS NOT NULL AND d_start.close > 0
                    THEN (d_end.close - d_start.close) / d_start.close * 100
                    ELSE NULL
                END as actual_return_60d
            FROM us_stock_grade g
            JOIN us_stock_basic b ON g.symbol = b.symbol
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
                AND b.sector IS NOT NULL
        )
        SELECT
            sector,
            COUNT(*) as total_samples,
            COUNT(*) FILTER (WHERE actual_return_60d > 5) as actual_bullish,
            COUNT(*) FILTER (WHERE actual_return_60d BETWEEN -5 AND 5) as actual_sideways,
            COUNT(*) FILTER (WHERE actual_return_60d < -5) as actual_bearish,
            ROUND(AVG(actual_return_60d)::numeric, 2) as avg_return,
            ROUND(STDDEV(actual_return_60d)::numeric, 2) as std_return,
            ROUND(AVG(scenario_bullish_prob)::numeric, 1) as avg_predicted_bullish,
            ROUND(AVG(scenario_sideways_prob)::numeric, 1) as avg_predicted_sideways,
            ROUND(AVG(scenario_bearish_prob)::numeric, 1) as avg_predicted_bearish,
            ROUND(AVG(scenario_sample_count)::numeric, 0) as avg_sample_count
        FROM actual_returns
        GROUP BY sector
        ORDER BY total_samples DESC
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            total = r['total_samples']
            actual_bull_pct = r['actual_bullish'] / total * 100 if total > 0 else 0
            actual_side_pct = r['actual_sideways'] / total * 100 if total > 0 else 0
            actual_bear_pct = r['actual_bearish'] / total * 100 if total > 0 else 0

            data.append({
                'sector': r['sector'],
                'total_samples': total,
                'actual_bullish_count': r['actual_bullish'],
                'actual_sideways_count': r['actual_sideways'],
                'actual_bearish_count': r['actual_bearish'],
                'actual_bullish_pct': round(actual_bull_pct, 1),
                'actual_sideways_pct': round(actual_side_pct, 1),
                'actual_bearish_pct': round(actual_bear_pct, 1),
                'avg_return': r['avg_return'],
                'std_return': r['std_return'],
                'avg_predicted_bullish': r['avg_predicted_bullish'],
                'avg_predicted_sideways': r['avg_predicted_sideways'],
                'avg_predicted_bearish': r['avg_predicted_bearish'],
                'avg_sample_count': r['avg_sample_count']
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "1_sector_actual_distribution.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")
        print(f"  Total sectors: {len(data)}")

    async def analyze_prediction_error(self):
        """
        Analysis 2: Prediction error by sector
        Compare predicted vs actual probabilities
        """
        query = """
        WITH actual_returns AS (
            SELECT
                b.sector,
                g.scenario_bullish_prob,
                g.scenario_sideways_prob,
                g.scenario_bearish_prob,
                CASE
                    WHEN d_end.close IS NOT NULL AND d_start.close > 0
                    THEN (d_end.close - d_start.close) / d_start.close * 100
                    ELSE NULL
                END as actual_return
            FROM us_stock_grade g
            JOIN us_stock_basic b ON g.symbol = b.symbol
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
                AND b.sector IS NOT NULL
                AND g.scenario_bullish_prob IS NOT NULL
        )
        SELECT
            sector,
            COUNT(*) as total,
            ROUND(AVG(scenario_bullish_prob)::numeric, 1) as pred_bullish,
            ROUND(AVG(scenario_sideways_prob)::numeric, 1) as pred_sideways,
            ROUND(AVG(scenario_bearish_prob)::numeric, 1) as pred_bearish,
            ROUND(100.0 * COUNT(*) FILTER (WHERE actual_return > 5) / COUNT(*)::numeric, 1) as actual_bullish,
            ROUND(100.0 * COUNT(*) FILTER (WHERE actual_return BETWEEN -5 AND 5) / COUNT(*)::numeric, 1) as actual_sideways,
            ROUND(100.0 * COUNT(*) FILTER (WHERE actual_return < -5) / COUNT(*)::numeric, 1) as actual_bearish
        FROM actual_returns
        GROUP BY sector
        HAVING COUNT(*) >= 30
        ORDER BY total DESC
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            bull_error = float(r['pred_bullish'] or 0) - float(r['actual_bullish'] or 0)
            side_error = float(r['pred_sideways'] or 0) - float(r['actual_sideways'] or 0)
            bear_error = float(r['pred_bearish'] or 0) - float(r['actual_bearish'] or 0)

            data.append({
                'sector': r['sector'],
                'total_samples': r['total'],
                'pred_bullish': r['pred_bullish'],
                'actual_bullish': r['actual_bullish'],
                'bullish_error': round(bull_error, 1),
                'pred_sideways': r['pred_sideways'],
                'actual_sideways': r['actual_sideways'],
                'sideways_error': round(side_error, 1),
                'pred_bearish': r['pred_bearish'],
                'actual_bearish': r['actual_bearish'],
                'bearish_error': round(bear_error, 1),
                'total_abs_error': round(abs(bull_error) + abs(side_error) + abs(bear_error), 1)
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "2_sector_prediction_error.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")
        print(f"  Sectors with >= 30 samples: {len(data)}")

    async def detect_error_cases(self):
        """
        Analysis 3: Detect error cases
        Pattern: High predicted bearish but actually positive return
                 OR Low predicted bullish but high actual return
        """
        query = """
        WITH stock_analysis AS (
            SELECT
                b.sector,
                g.symbol,
                g.stock_name,
                g.date as analysis_date,
                g.final_score,
                g.final_grade,
                g.scenario_bullish_prob,
                g.scenario_sideways_prob,
                g.scenario_bearish_prob,
                g.scenario_sample_count,
                d_start.close as start_close,
                d_end.close as end_close,
                CASE
                    WHEN d_end.close IS NOT NULL AND d_start.close > 0
                    THEN (d_end.close - d_start.close) / d_start.close * 100
                    ELSE NULL
                END as actual_return
            FROM us_stock_grade g
            JOIN us_stock_basic b ON g.symbol = b.symbol
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
                AND g.scenario_bearish_prob IS NOT NULL
        )
        SELECT
            sector,
            symbol,
            stock_name,
            analysis_date,
            final_score,
            final_grade,
            scenario_bullish_prob,
            scenario_sideways_prob,
            scenario_bearish_prob,
            scenario_sample_count,
            ROUND(actual_return::numeric, 2) as actual_return,
            CASE
                WHEN scenario_bearish_prob > 35 AND actual_return > 10 THEN 'HIGH_BEAR_POSITIVE_RETURN'
                WHEN scenario_bullish_prob < 25 AND actual_return > 15 THEN 'LOW_BULL_HIGH_RETURN'
                WHEN scenario_bullish_prob > 50 AND actual_return < -10 THEN 'HIGH_BULL_NEGATIVE_RETURN'
                ELSE 'OTHER'
            END as error_type
        FROM stock_analysis
        WHERE (scenario_bearish_prob > 35 AND actual_return > 10)
           OR (scenario_bullish_prob < 25 AND actual_return > 15)
           OR (scenario_bullish_prob > 50 AND actual_return < -10)
        ORDER BY ABS(actual_return -
            CASE
                WHEN scenario_bullish_prob > scenario_bearish_prob THEN 10
                ELSE -10
            END) DESC
        LIMIT 500
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            data.append({
                'sector': r['sector'],
                'symbol': r['symbol'],
                'stock_name': r['stock_name'],
                'analysis_date': str(r['analysis_date']),
                'final_score': r['final_score'],
                'final_grade': r['final_grade'],
                'scenario_bullish_prob': r['scenario_bullish_prob'],
                'scenario_sideways_prob': r['scenario_sideways_prob'],
                'scenario_bearish_prob': r['scenario_bearish_prob'],
                'scenario_sample_count': r['scenario_sample_count'],
                'actual_return': r['actual_return'],
                'error_type': r['error_type']
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "3_error_cases.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")
        print(f"  Error cases found: {len(data)}")

    async def analyze_sample_count_accuracy(self):
        """
        Analysis 4: Prediction accuracy by sample count
        """
        query = """
        WITH predictions AS (
            SELECT
                CASE
                    WHEN g.scenario_sample_count < 20 THEN '0-19'
                    WHEN g.scenario_sample_count < 50 THEN '20-49'
                    WHEN g.scenario_sample_count < 100 THEN '50-99'
                    WHEN g.scenario_sample_count < 200 THEN '100-199'
                    ELSE '200+'
                END as sample_range,
                g.scenario_bullish_prob,
                g.scenario_sideways_prob,
                g.scenario_bearish_prob,
                CASE
                    WHEN (d_end.close - d_start.close) / d_start.close * 100 > 5 THEN 'BULLISH'
                    WHEN (d_end.close - d_start.close) / d_start.close * 100 < -5 THEN 'BEARISH'
                    ELSE 'SIDEWAYS'
                END as actual_outcome
            FROM us_stock_grade g
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
                AND g.scenario_sample_count IS NOT NULL
        )
        SELECT
            sample_range,
            COUNT(*) as total,
            ROUND(AVG(scenario_bullish_prob)::numeric, 1) as avg_pred_bullish,
            ROUND(AVG(scenario_sideways_prob)::numeric, 1) as avg_pred_sideways,
            ROUND(AVG(scenario_bearish_prob)::numeric, 1) as avg_pred_bearish,
            ROUND(100.0 * COUNT(*) FILTER (WHERE actual_outcome = 'BULLISH') / COUNT(*)::numeric, 1) as actual_bullish,
            ROUND(100.0 * COUNT(*) FILTER (WHERE actual_outcome = 'SIDEWAYS') / COUNT(*)::numeric, 1) as actual_sideways,
            ROUND(100.0 * COUNT(*) FILTER (WHERE actual_outcome = 'BEARISH') / COUNT(*)::numeric, 1) as actual_bearish
        FROM predictions
        GROUP BY sample_range
        ORDER BY
            CASE sample_range
                WHEN '0-19' THEN 1
                WHEN '20-49' THEN 2
                WHEN '50-99' THEN 3
                WHEN '100-199' THEN 4
                ELSE 5
            END
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            bull_error = abs(float(r['avg_pred_bullish'] or 0) - float(r['actual_bullish'] or 0))
            side_error = abs(float(r['avg_pred_sideways'] or 0) - float(r['actual_sideways'] or 0))
            bear_error = abs(float(r['avg_pred_bearish'] or 0) - float(r['actual_bearish'] or 0))

            data.append({
                'sample_range': r['sample_range'],
                'total_predictions': r['total'],
                'avg_pred_bullish': r['avg_pred_bullish'],
                'actual_bullish': r['actual_bullish'],
                'bullish_abs_error': round(bull_error, 1),
                'avg_pred_sideways': r['avg_pred_sideways'],
                'actual_sideways': r['actual_sideways'],
                'sideways_abs_error': round(side_error, 1),
                'avg_pred_bearish': r['avg_pred_bearish'],
                'actual_bearish': r['actual_bearish'],
                'bearish_abs_error': round(bear_error, 1),
                'total_abs_error': round(bull_error + side_error + bear_error, 1)
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "4_sample_count_accuracy.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")

    async def generate_recommended_calibration(self):
        """
        Analysis 5: Generate recommended calibration multipliers by sector
        """
        query = """
        WITH predictions AS (
            SELECT
                b.sector,
                g.scenario_bullish_prob as pred_bull,
                g.scenario_bearish_prob as pred_bear,
                CASE
                    WHEN (d_end.close - d_start.close) / d_start.close * 100 > 5 THEN 1
                    ELSE 0
                END as is_bullish,
                CASE
                    WHEN (d_end.close - d_start.close) / d_start.close * 100 < -5 THEN 1
                    ELSE 0
                END as is_bearish
            FROM us_stock_grade g
            JOIN us_stock_basic b ON g.symbol = b.symbol
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
                AND b.sector IS NOT NULL
                AND g.scenario_bullish_prob IS NOT NULL
        )
        SELECT
            sector,
            COUNT(*) as total,
            ROUND(AVG(pred_bull)::numeric, 1) as avg_pred_bullish,
            ROUND(100.0 * SUM(is_bullish) / COUNT(*)::numeric, 1) as actual_bullish_rate,
            ROUND(AVG(pred_bear)::numeric, 1) as avg_pred_bearish,
            ROUND(100.0 * SUM(is_bearish) / COUNT(*)::numeric, 1) as actual_bearish_rate
        FROM predictions
        GROUP BY sector
        HAVING COUNT(*) >= 50
        ORDER BY total DESC
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            pred_bull = float(r['avg_pred_bullish'] or 1)
            actual_bull = float(r['actual_bullish_rate'] or 1)
            pred_bear = float(r['avg_pred_bearish'] or 1)
            actual_bear = float(r['actual_bearish_rate'] or 1)

            # Calculate recommended multipliers
            bull_mult = actual_bull / pred_bull if pred_bull > 0 else 1.0
            bear_mult = actual_bear / pred_bear if pred_bear > 0 else 1.0

            data.append({
                'sector': r['sector'],
                'total_samples': r['total'],
                'avg_pred_bullish': r['avg_pred_bullish'],
                'actual_bullish_rate': r['actual_bullish_rate'],
                'recommended_bull_multiplier': round(bull_mult, 2),
                'avg_pred_bearish': r['avg_pred_bearish'],
                'actual_bearish_rate': r['actual_bearish_rate'],
                'recommended_bear_multiplier': round(bear_mult, 2),
                'bull_adjustment': 'REDUCE' if bull_mult < 0.9 else ('INCREASE' if bull_mult > 1.1 else 'OK'),
                'bear_adjustment': 'REDUCE' if bear_mult < 0.9 else ('INCREASE' if bear_mult > 1.1 else 'OK')
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "5_recommended_calibration_map.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")

    async def analyze_return_range_accuracy(self):
        """
        Analysis 6: Return range accuracy
        Compare predicted ranges vs actual returns
        """
        query = """
        WITH returns_data AS (
            SELECT
                b.sector,
                g.scenario_bullish_return,
                g.scenario_sideways_return,
                g.scenario_bearish_return,
                (d_end.close - d_start.close) / d_start.close * 100 as actual_return,
                CASE
                    WHEN (d_end.close - d_start.close) / d_start.close * 100 > 5 THEN 'BULLISH'
                    WHEN (d_end.close - d_start.close) / d_start.close * 100 < -5 THEN 'BEARISH'
                    ELSE 'SIDEWAYS'
                END as outcome_type
            FROM us_stock_grade g
            JOIN us_stock_basic b ON g.symbol = b.symbol
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
                AND b.sector IS NOT NULL
        )
        SELECT
            sector,
            outcome_type,
            COUNT(*) as count,
            ROUND(AVG(actual_return)::numeric, 2) as avg_actual_return,
            ROUND(STDDEV(actual_return)::numeric, 2) as std_actual_return,
            ROUND(MIN(actual_return)::numeric, 2) as min_return,
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as median,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p75,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p90,
            ROUND(MAX(actual_return)::numeric, 2) as max_return
        FROM returns_data
        GROUP BY sector, outcome_type
        HAVING COUNT(*) >= 10
        ORDER BY sector, outcome_type
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            data.append({
                'sector': r['sector'],
                'outcome_type': r['outcome_type'],
                'count': r['count'],
                'avg_actual_return': r['avg_actual_return'],
                'std_actual_return': r['std_actual_return'],
                'min_return': r['min_return'],
                'p10': r['p10'],
                'p25': r['p25'],
                'median': r['median'],
                'p75': r['p75'],
                'p90': r['p90'],
                'max_return': r['max_return'],
                'recommended_range': f"{r['p25']}~{r['p75']}"
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "6_return_range_accuracy.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")

    async def calculate_return_percentiles_by_sector(self):
        """
        Analysis 7: Return percentiles by sector (for narrowing return ranges)
        """
        query = """
        WITH returns_data AS (
            SELECT
                b.sector,
                (d_end.close - d_start.close) / d_start.close * 100 as actual_return
            FROM us_stock_grade g
            JOIN us_stock_basic b ON g.symbol = b.symbol
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
                AND b.sector IS NOT NULL
        )
        SELECT
            sector,
            COUNT(*) as total_samples,
            ROUND(AVG(actual_return)::numeric, 2) as mean_return,
            ROUND(STDDEV(actual_return)::numeric, 2) as std_return,
            ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p05,
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p50_median,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p75,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p90,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p95,
            -- Bullish cases only (>+5%)
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return > 5)::numeric, 2) as bullish_p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return > 5)::numeric, 2) as bullish_p75,
            -- Sideways cases only (-5% to +5%)
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return BETWEEN -5 AND 5)::numeric, 2) as sideways_p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return BETWEEN -5 AND 5)::numeric, 2) as sideways_p75,
            -- Bearish cases only (<-5%)
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return < -5)::numeric, 2) as bearish_p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return < -5)::numeric, 2) as bearish_p75
        FROM returns_data
        GROUP BY sector
        HAVING COUNT(*) >= 50
        ORDER BY total_samples DESC
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            data.append({
                'sector': r['sector'],
                'total_samples': r['total_samples'],
                'mean_return': r['mean_return'],
                'std_return': r['std_return'],
                'p05': r['p05'],
                'p10': r['p10'],
                'p25': r['p25'],
                'p50_median': r['p50_median'],
                'p75': r['p75'],
                'p90': r['p90'],
                'p95': r['p95'],
                'bullish_p25': r['bullish_p25'],
                'bullish_p75': r['bullish_p75'],
                'bullish_range': f"+{r['bullish_p25']}~+{r['bullish_p75']}" if r['bullish_p25'] else None,
                'sideways_p25': r['sideways_p25'],
                'sideways_p75': r['sideways_p75'],
                'sideways_range': f"{r['sideways_p25']}~{r['sideways_p75']}" if r['sideways_p25'] else None,
                'bearish_p25': r['bearish_p25'],
                'bearish_p75': r['bearish_p75'],
                'bearish_range': f"{r['bearish_p25']}~{r['bearish_p75']}" if r['bearish_p25'] else None
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "7_return_percentiles_by_sector.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")

    async def calculate_return_percentiles_by_score(self):
        """
        Analysis 8: Return percentiles by score bucket
        """
        query = """
        WITH returns_data AS (
            SELECT
                FLOOR(g.final_score / 10) * 10 as score_bucket,
                (d_end.close - d_start.close) / d_start.close * 100 as actual_return
            FROM us_stock_grade g
            JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM us_daily
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '55 days'
                    AND date <= g.date + INTERVAL '65 days'
                ORDER BY date
                LIMIT 1
            ) d_end ON true
            WHERE g.final_score IS NOT NULL
                AND g.date < CURRENT_DATE - INTERVAL '65 days'
                AND d_end.close IS NOT NULL
        )
        SELECT
            score_bucket,
            COUNT(*) as total_samples,
            ROUND(AVG(actual_return)::numeric, 2) as mean_return,
            ROUND(STDDEV(actual_return)::numeric, 2) as std_return,
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p50_median,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p75,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY actual_return)::numeric, 2) as p90,
            -- Bullish cases (>+5%)
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return > 5)::numeric, 2) as bullish_p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return > 5)::numeric, 2) as bullish_p75,
            -- Sideways cases (-5% to +5%)
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return BETWEEN -5 AND 5)::numeric, 2) as sideways_p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return BETWEEN -5 AND 5)::numeric, 2) as sideways_p75,
            -- Bearish cases (<-5%)
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return < -5)::numeric, 2) as bearish_p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY actual_return)
                FILTER (WHERE actual_return < -5)::numeric, 2) as bearish_p75,
            -- Counts by outcome
            COUNT(*) FILTER (WHERE actual_return > 5) as bullish_count,
            COUNT(*) FILTER (WHERE actual_return BETWEEN -5 AND 5) as sideways_count,
            COUNT(*) FILTER (WHERE actual_return < -5) as bearish_count
        FROM returns_data
        WHERE score_bucket IS NOT NULL
        GROUP BY score_bucket
        ORDER BY score_bucket
        """

        rows = await self.conn.fetch(query)

        data = []
        for r in rows:
            total = r['total_samples']
            bull_pct = r['bullish_count'] / total * 100 if total > 0 else 0
            side_pct = r['sideways_count'] / total * 100 if total > 0 else 0
            bear_pct = r['bearish_count'] / total * 100 if total > 0 else 0

            data.append({
                'score_bucket': int(r['score_bucket']) if r['score_bucket'] else 0,
                'total_samples': total,
                'mean_return': r['mean_return'],
                'std_return': r['std_return'],
                'p10': r['p10'],
                'p25': r['p25'],
                'p50_median': r['p50_median'],
                'p75': r['p75'],
                'p90': r['p90'],
                'bullish_count': r['bullish_count'],
                'bullish_pct': round(bull_pct, 1),
                'bullish_p25': r['bullish_p25'],
                'bullish_p75': r['bullish_p75'],
                'sideways_count': r['sideways_count'],
                'sideways_pct': round(side_pct, 1),
                'sideways_p25': r['sideways_p25'],
                'sideways_p75': r['sideways_p75'],
                'bearish_count': r['bearish_count'],
                'bearish_pct': round(bear_pct, 1),
                'bearish_p25': r['bearish_p25'],
                'bearish_p75': r['bearish_p75']
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, "8_return_percentiles_by_score.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved: {output_path}")


async def main():
    """Main entry point"""
    analyzer = USScenarioCalibrationAnalyzer()
    await analyzer.run_all_analyses()


if __name__ == "__main__":
    asyncio.run(main())
