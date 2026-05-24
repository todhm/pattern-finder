"""
US Sector Benchmarks Calculator v2.0

섹터별 기준값 계산 (Factor 점수 산출 시 섹터 특성 반영용)

주요 기능:
- 섹터별 밸류에이션 중앙값 (PE, Forward PE, PEG, PS, EV/EBITDA)
- 섹터별 수익성 기준 (Gross Margin, Operating Margin, ROIC, ROE)
- 섹터별 성장률 기준 (Revenue Growth, EPS Growth)
- 섹터별 모멘텀 기준 (1M, 3M, 6M Return)

Usage:
    calculator = USSectorBenchmarks(db_manager, analysis_date)
    benchmarks = await calculator.calculate_all_sectors()
    tech_benchmarks = await calculator.get_sector_benchmarks('TECHNOLOGY')

File: us/us_sector_benchmarks.py
"""

import logging
from typing import Dict, Optional, List
from datetime import date
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Sector Definitions
# ========================================================================

SECTORS = [
    'TECHNOLOGY',
    'HEALTHCARE',
    'FINANCIAL SERVICES',
    'CONSUMER CYCLICAL',
    'CONSUMER DEFENSIVE',
    'INDUSTRIALS',
    'ENERGY',
    'UTILITIES',
    'REAL ESTATE',
    'BASIC MATERIALS',
    'COMMUNICATION SERVICES'
]


class USSectorBenchmarks:
    """섹터별 기준값 계산"""

    def __init__(self, db_manager, analysis_date: Optional[date] = None):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
            analysis_date: 분석 날짜 (None이면 최신)
        """
        self.db = db_manager
        self.analysis_date = analysis_date
        self._cache = {}

    async def calculate_all_sectors(self) -> Dict[str, Dict]:
        """모든 섹터의 기준값 계산 및 DB 저장"""

        results = {}

        for sector in SECTORS:
            try:
                benchmarks = await self._calculate_sector_benchmarks(sector)
                results[sector] = benchmarks

                # DB 저장
                await self._save_to_db(sector, benchmarks)

                logger.info(f"Sector benchmarks calculated: {sector} ({benchmarks.get('stock_count', 0)} stocks)")

            except Exception as e:
                logger.error(f"Failed to calculate benchmarks for {sector}: {e}")
                results[sector] = {}

        return results

    async def get_sector_benchmarks(self, sector: str) -> Dict:
        """특정 섹터의 기준값 조회 (캐시 또는 DB)

        NONE/OTHER/empty 섹터는 전체 시장 평균 벤치마크 사용
        """

        # NONE/OTHER/empty 섹터는 전체 시장 평균 사용
        if sector in ['NONE', 'OTHER', '', None] or not sector:
            return await self._get_market_average_benchmarks()

        # 캐시 확인
        cache_key = f"{sector}_{self.analysis_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # DB에서 조회
        benchmarks = await self._load_from_db(sector)

        if benchmarks:
            self._cache[cache_key] = benchmarks
            return benchmarks

        # 없으면 계산
        benchmarks = await self._calculate_sector_benchmarks(sector)
        self._cache[cache_key] = benchmarks

        return benchmarks

    async def _get_market_average_benchmarks(self) -> Dict:
        """전체 시장 평균 벤치마크 (NONE/OTHER 섹터용)"""

        cache_key = f"MARKET_AVERAGE_{self.analysis_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        logger.info("Calculating market average benchmarks for NONE/OTHER sectors")

        # 전체 시장 벤치마크 계산 (모든 유효 섹터 종목 포함)
        benchmarks = await self._calculate_market_average_benchmarks()
        self._cache[cache_key] = benchmarks

        return benchmarks

    async def _calculate_market_average_benchmarks(self) -> Dict:
        """전체 시장 평균 벤치마크 계산"""

        # Valuation
        valuation_query = """
        SELECT
            COUNT(*) as stock_count,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY per) as pe_p25,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY per) as pe_median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY per) as pe_p75,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY forwardpe) as forward_pe_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY peg) as peg_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pricetosalesratiottm) as ps_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY evtoebitda) as ev_ebitda_median
        FROM us_stock_basic
        WHERE sector IS NOT NULL
            AND sector NOT IN ('NONE', 'OTHER', '')
            AND per IS NOT NULL
            AND per > 0
            AND per < 200
        """

        # Quality
        quality_query = """
        SELECT
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
                CASE WHEN grossprofitttm > 0 AND revenuettm > 0
                     THEN grossprofitttm::float / revenuettm
                     ELSE NULL END) as gross_margin_p25,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN grossprofitttm > 0 AND revenuettm > 0
                     THEN grossprofitttm::float / revenuettm
                     ELSE NULL END) as gross_margin_median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                CASE WHEN grossprofitttm > 0 AND revenuettm > 0
                     THEN grossprofitttm::float / revenuettm
                     ELSE NULL END) as gross_margin_p75,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY operatingmarginttm) as operating_margin_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY returnonequityttm) as roe_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY returnonassetsttm) as roa_median
        FROM us_stock_basic
        WHERE sector IS NOT NULL
            AND sector NOT IN ('NONE', 'OTHER', '')
            AND revenuettm > 0
        """

        # Growth
        growth_query = """
        SELECT
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quarterlyrevenuegrowthyoy) as revenue_growth_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quarterlyearningsgrowthyoy) as eps_growth_median
        FROM us_stock_basic
        WHERE sector IS NOT NULL
            AND sector NOT IN ('NONE', 'OTHER', '')
            AND quarterlyrevenuegrowthyoy IS NOT NULL
        """

        # Momentum
        momentum_query = """
        WITH all_stocks AS (
            SELECT symbol FROM us_stock_basic
            WHERE sector IS NOT NULL AND sector NOT IN ('NONE', 'OTHER', '')
        ),
        price_returns AS (
            SELECT
                d.symbol,
                d.close,
                LAG(d.close, 21) OVER (PARTITION BY d.symbol ORDER BY d.date) as close_1m,
                LAG(d.close, 63) OVER (PARTITION BY d.symbol ORDER BY d.date) as close_3m,
                LAG(d.close, 126) OVER (PARTITION BY d.symbol ORDER BY d.date) as close_6m,
                ROW_NUMBER() OVER (PARTITION BY d.symbol ORDER BY d.date DESC) as rn
            FROM us_daily d
            INNER JOIN all_stocks s ON d.symbol = s.symbol
            WHERE ($1::DATE IS NULL OR d.date <= $1::DATE)
        )
        SELECT
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN close_1m > 0 THEN (close - close_1m) / close_1m ELSE NULL END) as return_1m_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN close_3m > 0 THEN (close - close_3m) / close_3m ELSE NULL END) as return_3m_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN close_6m > 0 THEN (close - close_6m) / close_6m ELSE NULL END) as return_6m_median
        FROM price_returns
        WHERE rn = 1
        """

        result = {
            'stock_count': 0,
            'sector': 'MARKET_AVERAGE'
        }

        try:
            # Valuation
            val_result = await self.db.execute_query(valuation_query)
            if val_result and val_result[0]:
                row = val_result[0]
                result.update({
                    'stock_count': int(row['stock_count'] or 0),
                    'pe_p25': self._to_float(row['pe_p25']),
                    'pe_median': self._to_float(row['pe_median']),
                    'pe_p75': self._to_float(row['pe_p75']),
                    'forward_pe_median': self._to_float(row['forward_pe_median']),
                    'peg_median': self._to_float(row['peg_median']),
                    'ps_median': self._to_float(row['ps_median']),
                    'ev_ebitda_median': self._to_float(row['ev_ebitda_median'])
                })

            # Quality
            qual_result = await self.db.execute_query(quality_query)
            if qual_result and qual_result[0]:
                row = qual_result[0]
                result.update({
                    'gross_margin_p25': self._to_float(row['gross_margin_p25']),
                    'gross_margin_median': self._to_float(row['gross_margin_median']),
                    'gross_margin_p75': self._to_float(row['gross_margin_p75']),
                    'operating_margin_median': self._to_float(row['operating_margin_median']),
                    'roe_median': self._to_float(row['roe_median']),
                    'roic_median': self._to_float(row['roa_median']) * 1.2 if row['roa_median'] else None
                })

            # Growth
            growth_result = await self.db.execute_query(growth_query)
            if growth_result and growth_result[0]:
                row = growth_result[0]
                result.update({
                    'revenue_growth_median': self._to_float(row['revenue_growth_median']),
                    'eps_growth_median': self._to_float(row['eps_growth_median'])
                })

            # Momentum
            mom_result = await self.db.execute_query(momentum_query, self.analysis_date)
            if mom_result and mom_result[0]:
                row = mom_result[0]
                result.update({
                    'return_1m_median': self._to_float(row['return_1m_median']),
                    'return_3m_median': self._to_float(row['return_3m_median']),
                    'return_6m_median': self._to_float(row['return_6m_median'])
                })

            logger.info(f"Market average benchmarks calculated: {result.get('stock_count', 0)} stocks")

        except Exception as e:
            logger.error(f"Failed to calculate market average benchmarks: {e}")
            # Return default values
            result.update({
                'pe_p25': 15.0, 'pe_median': 20.0, 'pe_p75': 30.0,
                'forward_pe_median': 18.0, 'peg_median': 1.5,
                'ps_median': 3.0, 'ev_ebitda_median': 12.0,
                'gross_margin_p25': 0.25, 'gross_margin_median': 0.35, 'gross_margin_p75': 0.50,
                'operating_margin_median': 0.12, 'roe_median': 0.12, 'roic_median': 0.10,
                'revenue_growth_median': 0.08, 'eps_growth_median': 0.10,
                'return_1m_median': 0.02, 'return_3m_median': 0.05, 'return_6m_median': 0.10
            })

        return result

    async def _calculate_sector_benchmarks(self, sector: str) -> Dict:
        """섹터별 기준값 계산"""

        # 밸류에이션 지표
        valuation = await self._calc_valuation_benchmarks(sector)

        # 수익성 지표
        quality = await self._calc_quality_benchmarks(sector)

        # 성장률 지표
        growth = await self._calc_growth_benchmarks(sector)

        # 모멘텀 지표
        momentum = await self._calc_momentum_benchmarks(sector)

        return {
            **valuation,
            **quality,
            **growth,
            **momentum
        }

    async def _calc_valuation_benchmarks(self, sector: str) -> Dict:
        """밸류에이션 기준값 (PE, Forward PE, PEG, PS, EV/EBITDA)"""

        query = """
        SELECT
            COUNT(*) as stock_count,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY per) as pe_p25,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY per) as pe_median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY per) as pe_p75,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY forwardpe) as forward_pe_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY peg) as peg_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pricetosalesratiottm) as ps_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY evtoebitda) as ev_ebitda_median
        FROM us_stock_basic
        WHERE sector = $1
            AND per IS NOT NULL
            AND per > 0
            AND per < 200
        """

        try:
            result = await self.db.execute_query(query, sector)

            if result and result[0]:
                row = result[0]
                return {
                    'stock_count': int(row['stock_count'] or 0),
                    'pe_p25': self._to_float(row['pe_p25']),
                    'pe_median': self._to_float(row['pe_median']),
                    'pe_p75': self._to_float(row['pe_p75']),
                    'forward_pe_median': self._to_float(row['forward_pe_median']),
                    'peg_median': self._to_float(row['peg_median']),
                    'ps_median': self._to_float(row['ps_median']),
                    'ev_ebitda_median': self._to_float(row['ev_ebitda_median'])
                }
        except Exception as e:
            logger.warning(f"Valuation benchmarks query failed for {sector}: {e}")

        return {
            'stock_count': 0,
            'pe_p25': 15.0,
            'pe_median': 20.0,
            'pe_p75': 30.0,
            'forward_pe_median': 18.0,
            'peg_median': 1.5,
            'ps_median': 3.0,
            'ev_ebitda_median': 12.0
        }

    async def _calc_quality_benchmarks(self, sector: str) -> Dict:
        """수익성 기준값 (Gross Margin, Operating Margin, ROIC, ROE)"""

        query = """
        SELECT
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
                CASE WHEN grossprofitttm > 0 AND revenuettm > 0
                     THEN grossprofitttm::float / revenuettm
                     ELSE NULL END) as gross_margin_p25,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN grossprofitttm > 0 AND revenuettm > 0
                     THEN grossprofitttm::float / revenuettm
                     ELSE NULL END) as gross_margin_median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
                CASE WHEN grossprofitttm > 0 AND revenuettm > 0
                     THEN grossprofitttm::float / revenuettm
                     ELSE NULL END) as gross_margin_p75,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY operatingmarginttm) as operating_margin_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY returnonequityttm) as roe_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY returnonassetsttm) as roa_median
        FROM us_stock_basic
        WHERE sector = $1
            AND revenuettm > 0
        """

        try:
            result = await self.db.execute_query(query, sector)

            if result and result[0]:
                row = result[0]
                return {
                    'gross_margin_p25': self._to_float(row['gross_margin_p25']),
                    'gross_margin_median': self._to_float(row['gross_margin_median']),
                    'gross_margin_p75': self._to_float(row['gross_margin_p75']),
                    'operating_margin_median': self._to_float(row['operating_margin_median']),
                    'roe_median': self._to_float(row['roe_median']),
                    'roic_median': self._to_float(row['roa_median']) * 1.2  # ROA 기반 추정
                }
        except Exception as e:
            logger.warning(f"Quality benchmarks query failed for {sector}: {e}")

        return {
            'gross_margin_p25': 0.25,
            'gross_margin_median': 0.35,
            'gross_margin_p75': 0.50,
            'operating_margin_median': 0.12,
            'roe_median': 0.12,
            'roic_median': 0.10
        }

    async def _calc_growth_benchmarks(self, sector: str) -> Dict:
        """성장률 기준값 (Revenue Growth, EPS Growth)"""

        query = """
        SELECT
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quarterlyrevenuegrowthyoy) as revenue_growth_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quarterlyearningsgrowthyoy) as eps_growth_median
        FROM us_stock_basic
        WHERE sector = $1
            AND quarterlyrevenuegrowthyoy IS NOT NULL
        """

        try:
            result = await self.db.execute_query(query, sector)

            if result and result[0]:
                row = result[0]
                return {
                    'revenue_growth_median': self._to_float(row['revenue_growth_median']),
                    'eps_growth_median': self._to_float(row['eps_growth_median'])
                }
        except Exception as e:
            logger.warning(f"Growth benchmarks query failed for {sector}: {e}")

        return {
            'revenue_growth_median': 0.08,
            'eps_growth_median': 0.10
        }

    async def _calc_momentum_benchmarks(self, sector: str) -> Dict:
        """모멘텀 기준값 (1M, 3M, 6M Return)"""

        # 섹터 내 종목들의 최근 수익률 계산
        query = """
        WITH sector_stocks AS (
            SELECT symbol FROM us_stock_basic WHERE sector = $1
        ),
        price_returns AS (
            SELECT
                d.symbol,
                d.close,
                LAG(d.close, 21) OVER (PARTITION BY d.symbol ORDER BY d.date) as close_1m,
                LAG(d.close, 63) OVER (PARTITION BY d.symbol ORDER BY d.date) as close_3m,
                LAG(d.close, 126) OVER (PARTITION BY d.symbol ORDER BY d.date) as close_6m,
                ROW_NUMBER() OVER (PARTITION BY d.symbol ORDER BY d.date DESC) as rn
            FROM us_daily d
            INNER JOIN sector_stocks s ON d.symbol = s.symbol
            WHERE ($2::DATE IS NULL OR d.date <= $2::DATE)
        )
        SELECT
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN close_1m > 0 THEN (close - close_1m) / close_1m ELSE NULL END) as return_1m_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN close_3m > 0 THEN (close - close_3m) / close_3m ELSE NULL END) as return_3m_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                CASE WHEN close_6m > 0 THEN (close - close_6m) / close_6m ELSE NULL END) as return_6m_median
        FROM price_returns
        WHERE rn = 1
        """

        try:
            result = await self.db.execute_query(query, sector, self.analysis_date)

            if result and result[0]:
                row = result[0]
                return {
                    'return_1m_median': self._to_float(row['return_1m_median']),
                    'return_3m_median': self._to_float(row['return_3m_median']),
                    'return_6m_median': self._to_float(row['return_6m_median'])
                }
        except Exception as e:
            logger.warning(f"Momentum benchmarks query failed for {sector}: {e}")

        return {
            'return_1m_median': 0.02,
            'return_3m_median': 0.05,
            'return_6m_median': 0.10
        }

    async def _save_to_db(self, sector: str, benchmarks: Dict) -> bool:
        """섹터 기준값 DB 저장"""

        try:
            query = """
            INSERT INTO us_sector_benchmarks (
                date, sector, stock_count,
                pe_p25, pe_median, pe_p75, forward_pe_median, peg_median, ps_median, ev_ebitda_median,
                gross_margin_p25, gross_margin_median, gross_margin_p75, operating_margin_median, roic_median, roe_median,
                revenue_growth_median, eps_growth_median,
                return_1m_median, return_3m_median, return_6m_median
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
            )
            ON CONFLICT (date, sector) DO UPDATE SET
                stock_count = EXCLUDED.stock_count,
                pe_p25 = EXCLUDED.pe_p25,
                pe_median = EXCLUDED.pe_median,
                pe_p75 = EXCLUDED.pe_p75,
                forward_pe_median = EXCLUDED.forward_pe_median,
                peg_median = EXCLUDED.peg_median,
                ps_median = EXCLUDED.ps_median,
                ev_ebitda_median = EXCLUDED.ev_ebitda_median,
                gross_margin_p25 = EXCLUDED.gross_margin_p25,
                gross_margin_median = EXCLUDED.gross_margin_median,
                gross_margin_p75 = EXCLUDED.gross_margin_p75,
                operating_margin_median = EXCLUDED.operating_margin_median,
                roic_median = EXCLUDED.roic_median,
                roe_median = EXCLUDED.roe_median,
                revenue_growth_median = EXCLUDED.revenue_growth_median,
                eps_growth_median = EXCLUDED.eps_growth_median,
                return_1m_median = EXCLUDED.return_1m_median,
                return_3m_median = EXCLUDED.return_3m_median,
                return_6m_median = EXCLUDED.return_6m_median
            """

            target_date = self.analysis_date or date.today()

            await self.db.execute(
                query,
                target_date,
                sector,
                benchmarks.get('stock_count'),
                benchmarks.get('pe_p25'),
                benchmarks.get('pe_median'),
                benchmarks.get('pe_p75'),
                benchmarks.get('forward_pe_median'),
                benchmarks.get('peg_median'),
                benchmarks.get('ps_median'),
                benchmarks.get('ev_ebitda_median'),
                benchmarks.get('gross_margin_p25'),
                benchmarks.get('gross_margin_median'),
                benchmarks.get('gross_margin_p75'),
                benchmarks.get('operating_margin_median'),
                benchmarks.get('roic_median'),
                benchmarks.get('roe_median'),
                benchmarks.get('revenue_growth_median'),
                benchmarks.get('eps_growth_median'),
                benchmarks.get('return_1m_median'),
                benchmarks.get('return_3m_median'),
                benchmarks.get('return_6m_median')
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save sector benchmarks: {e}")
            return False

    async def _load_from_db(self, sector: str) -> Optional[Dict]:
        """DB에서 섹터 기준값 조회"""

        query = """
        SELECT * FROM us_sector_benchmarks
        WHERE sector = $1
            AND ($2::DATE IS NULL OR date = $2::DATE)
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, sector, self.analysis_date)

            if result and result[0]:
                row = result[0]
                return {col: self._to_float(row[col]) if col != 'sector' and col != 'date' else row[col]
                        for col in row.keys()}
        except Exception as e:
            logger.warning(f"Failed to load sector benchmarks from DB: {e}")

        return None

    def _to_float(self, value) -> Optional[float]:
        """Decimal/None을 float로 변환"""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except:
            return None


# ========================================================================
# Test
# ========================================================================

if __name__ == '__main__':
    import asyncio
    from us_db_async import AsyncDatabaseManager

    async def test():
        db = AsyncDatabaseManager()
        await db.initialize()

        calc = USSectorBenchmarks(db)

        print("\n" + "="*60)
        print("Calculating Sector Benchmarks")
        print("="*60)

        # 전체 섹터 계산
        results = await calc.calculate_all_sectors()

        for sector, benchmarks in results.items():
            print(f"\n{sector}:")
            print(f"  Stocks: {benchmarks.get('stock_count', 0)}")
            print(f"  PE Median: {benchmarks.get('pe_median', 'N/A')}")
            print(f"  Gross Margin: {benchmarks.get('gross_margin_median', 'N/A'):.1%}" if benchmarks.get('gross_margin_median') else "  Gross Margin: N/A")

        await db.close()

    asyncio.run(test())
