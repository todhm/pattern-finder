"""
US Stock Data Prefetcher Module

Centralized data prefetching for stock analysis.
Eliminates redundant DB queries by loading all needed data upfront.

Query Optimization:
- us_daily: 8 queries → 1 query (260 days OHLCV)
- us_stock_basic: 5 queries → 1 query (all columns)
- us_income_statement: 3 queries → 1 query (12 quarters)
- us_cash_flow: 2 queries → 1 query (4 quarters)
- us_earnings_estimates: 2 queries → 1 query
- us_option_daily_summary: 2 queries → 1 query

Total reduction: ~35 queries/stock → ~20 queries/stock (43% reduction)

File: us/us_data_prefetcher.py
Date: 2025-12-08
"""

import asyncio
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class USDataPrefetcher:
    """
    Centralized data prefetching for stock analysis.
    Eliminates redundant DB queries by loading all needed data upfront.
    """

    def __init__(self, db):
        """
        Initialize prefetcher with database manager.

        Args:
            db: AsyncDatabaseManager instance
        """
        self.db = db

    async def prefetch_all(self, symbol: str, analysis_date: date) -> Dict:
        """
        Prefetch all common data in parallel.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            analysis_date: Analysis date

        Returns:
            {
                'stock_basic': {...},      # us_stock_basic row
                'price_data': {...},       # 260 days OHLCV (structured)
                'financials': {...},       # income_statement + cash_flow + balance_sheet
                'estimates': {...},        # earnings estimates
                'options': {...}           # options summary
            }
        """
        tasks = [
            self._prefetch_stock_basic(symbol, analysis_date),
            self._prefetch_daily_prices(symbol, analysis_date),
            self._prefetch_financials(symbol, analysis_date),
            self._prefetch_estimates(symbol, analysis_date),
            self._prefetch_options(symbol, analysis_date)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            'stock_basic': results[0] if not isinstance(results[0], Exception) else None,
            'price_data': results[1] if not isinstance(results[1], Exception) else None,
            'financials': results[2] if not isinstance(results[2], Exception) else None,
            'estimates': results[3] if not isinstance(results[3], Exception) else None,
            'options': results[4] if not isinstance(results[4], Exception) else None
        }

    async def _prefetch_stock_basic(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch stock basic data **as of analysis_date** (point-in-time).

        ``us_stock_basic`` carries both ``source='computed'`` (one row per
        trading day, fundamentals derived from ``available_at``-anchored
        financials — see ``us_stock_basic_compute``) and ``source='api'``
        (latest snapshot). Filtering by ``date <= analysis_date`` naturally
        excludes API snapshots dated after the analysis date and picks the
        most recent computed row valid at that point in time — preventing
        look-ahead bias on PER/EPS/ROE/margins/52w/50ma/200ma etc.

        Columns include:
        - Core: symbol, stock_name, sector, industry, exchange, market_cap
        - Valuation: per, forwardpe, peg, pricetosalesratiottm, evtorevenue, evtoebitda, pricetobookratio
        - Quality: grossprofitttm, revenuettm, operatingmarginttm, profitmargin, returnonequityttm, returnonassetsttm
        - Technical: beta, week52high, week52low, day50movingaverage, day200movingaverage
        - Shares: sharesoutstanding, sharesfloat, bookvalue, dilutedepsttm
        - Analyst: analysttargetprice, analystratings
        """
        query = """
        SELECT
            symbol, stock_name, sector, industry, exchange, market_cap,
            -- Valuation
            per, forwardpe, peg, pricetosalesratiottm, evtorevenue, evtoebitda,
            pricetobookratio, trailingpe,
            -- Growth
            quarterlyrevenuegrowthyoy, quarterlyearningsgrowthyoy,
            -- Quality
            grossprofitttm, revenuettm, operatingmarginttm, profitmargin,
            returnonequityttm, returnonassetsttm, ebitda,
            -- Technical
            beta, week52high, week52low, day50movingaverage, day200movingaverage,
            -- Shares
            sharesoutstanding, sharesfloat, bookvalue, dilutedepsttm,
            -- Analyst
            analysttargetprice, analystratingstrongbuy, analystratingbuy,
            analystratinghold, analystratingsell, analystratingstrongsell
        FROM us_stock_basic
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if result:
                return dict(result[0])
            return None
        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch stock_basic - {e}")
            return None

    async def _prefetch_daily_prices(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch 260 days of OHLCV price data.

        Returns structured data for easy access:
        {
            'dates': [date1, date2, ...],  # newest first
            'opens': [open1, open2, ...],
            'highs': [high1, high2, ...],
            'lows': [low1, low2, ...],
            'closes': [close1, close2, ...],
            'volumes': [vol1, vol2, ...],
            'raw': [{'date': date, 'open': ..., ...}, ...]  # original rows
        }
        """
        query = """
        SELECT date, open, high, low, close, volume
        FROM us_daily
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 260
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if not result:
                return None

            # Structure the data for easy access
            price_data = {
                'dates': [],
                'opens': [],
                'highs': [],
                'lows': [],
                'closes': [],
                'volumes': [],
                'raw': []
            }

            for row in result:
                price_data['dates'].append(row['date'])
                price_data['opens'].append(float(row['open']) if row['open'] else None)
                price_data['highs'].append(float(row['high']) if row['high'] else None)
                price_data['lows'].append(float(row['low']) if row['low'] else None)
                price_data['closes'].append(float(row['close']) if row['close'] else None)
                price_data['volumes'].append(int(row['volume']) if row['volume'] else None)
                price_data['raw'].append(dict(row))

            return price_data

        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch daily prices - {e}")
            return None

    async def _prefetch_financials(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch financial statements **as of analysis_date** (point-in-time).

        Filtering by ``available_at <= analysis_date`` excludes quarters that
        had not yet been publicly reported on the analysis date —
        ``available_at`` is the actual SEC 8-K filing date (from
        ``us_earnings_history.reported_date``, fall-back +45d when missing).
        Without this filter, look-ahead is severe: a Q3 result reported on
        Jul-31 would otherwise be visible to analyses dated mid-July.

        Returns:
        {
            'income': [{fiscal_date_ending, total_revenue, ...}, ...],  # up to 12 quarters
            'cashflow': [{fiscal_date_ending, operating_cashflow, ...}, ...],  # up to 4 quarters
            'balance': [{fiscal_date_ending, total_assets, ...}, ...]  # latest
        }
        """
        # Income Statement (12 quarters)
        income_query = """
        SELECT fiscal_date_ending,
               total_revenue, gross_profit, operating_income, net_income,
               research_and_development, operating_expenses,
               depreciation_and_amortization, ebitda
        FROM us_income_statement
        WHERE symbol = $1 AND available_at <= $2
        ORDER BY fiscal_date_ending DESC
        LIMIT 12
        """

        # Cash Flow (4 quarters)
        cashflow_query = """
        SELECT fiscal_date_ending,
               operating_cashflow, capital_expenditures, net_income,
               depreciation_depletion_and_amortization
        FROM us_cash_flow
        WHERE symbol = $1 AND available_at <= $2
        ORDER BY fiscal_date_ending DESC
        LIMIT 4
        """

        # Balance Sheet (latest)
        balance_query = """
        SELECT fiscal_date_ending,
               total_assets, total_liabilities, total_shareholder_equity,
               total_current_assets, total_current_liabilities,
               long_term_debt, short_term_debt, short_long_term_debt_total,
               cash_and_cash_equivalents_at_carrying_value
        FROM us_balance_sheet
        WHERE symbol = $1 AND available_at <= $2
        ORDER BY fiscal_date_ending DESC
        LIMIT 1
        """

        try:
            # Execute all queries in parallel
            results = await asyncio.gather(
                self.db.execute_query(income_query, symbol, analysis_date),
                self.db.execute_query(cashflow_query, symbol, analysis_date),
                self.db.execute_query(balance_query, symbol, analysis_date),
                return_exceptions=True
            )

            financials = {
                'income': [dict(r) for r in results[0]] if results[0] and not isinstance(results[0], Exception) else [],
                'cashflow': [dict(r) for r in results[1]] if results[1] and not isinstance(results[1], Exception) else [],
                'balance': [dict(r) for r in results[2]] if results[2] and not isinstance(results[2], Exception) else []
            }

            return financials

        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch financials - {e}")
            return None

    async def _prefetch_estimates(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch earnings estimates **as of analysis_date** (point-in-time).

        Estimates revise over time as analysts update — filtering by
        ``estimate_date <= analysis_date`` ensures we only see the estimate
        that was current on the analysis date, not a later revision.

        Returns:
        {
            'next_quarter': {...},  # next fiscal quarter estimate
            'next_year': {...}      # next fiscal year estimate
        }
        """
        query = """
        SELECT estimate_date, horizon,
               eps_estimate_average, eps_estimate_high, eps_estimate_low,
               eps_estimate_average_7_days_ago, eps_estimate_average_30_days_ago,
               eps_estimate_revision_up_trailing_7_days,
               eps_estimate_revision_down_trailing_7_days,
               revenue_estimate_average
        FROM us_earnings_estimates
        WHERE symbol = $1
          AND estimate_date <= $2
          AND horizon IN ('next fiscal quarter', 'next fiscal year')
          AND eps_estimate_average IS NOT NULL
        ORDER BY estimate_date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if not result:
                return {'next_quarter': None, 'next_year': None}

            estimates = {
                'next_quarter': None,
                'next_year': None
            }

            for row in result:
                row_dict = dict(row)
                horizon = row_dict.get('horizon', '')
                if 'quarter' in horizon and estimates['next_quarter'] is None:
                    estimates['next_quarter'] = row_dict
                elif 'year' in horizon and estimates['next_year'] is None:
                    estimates['next_year'] = row_dict

            return estimates

        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch estimates - {e}")
            return None

    async def _prefetch_options(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch options summary data.

        Returns latest options summary including:
        - Volume: total_call_volume, total_put_volume
        - IV: avg_call_iv, avg_put_iv, avg_implied_volatility
        - GEX: net_gex, call_gex, put_gex, gex_ratio
        """
        query = """
        SELECT date, total_call_volume, total_put_volume,
               avg_call_iv, avg_put_iv, avg_implied_volatility,
               net_gex, call_gex, put_gex, gex_ratio
        FROM us_option_daily_summary
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if result:
                return dict(result[0])
            return None
        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch options - {e}")
            return None


class PrefetchedDataExtractor:
    """
    Helper class to extract specific data from prefetched results.
    Provides convenient methods for common data extraction patterns.
    """

    @staticmethod
    def get_closes(price_data: Dict, days: int = 30) -> List[float]:
        """
        Extract close prices for specified number of days.

        Args:
            price_data: Prefetched price data dict
            days: Number of days to extract (default 30)

        Returns:
            List of close prices (newest first)
        """
        if not price_data or 'closes' not in price_data:
            return []
        closes = price_data['closes'][:days]
        return [c for c in closes if c is not None]

    @staticmethod
    def get_returns(price_data: Dict, days: int = 252) -> List[float]:
        """
        Calculate daily returns from close prices.

        Args:
            price_data: Prefetched price data dict
            days: Number of days for return calculation (default 252)

        Returns:
            List of daily returns (percentage)
        """
        if not price_data or 'closes' not in price_data:
            return []

        closes = price_data['closes'][:days + 1]
        closes = [c for c in closes if c is not None]

        if len(closes) < 2:
            return []

        return [(closes[i] - closes[i + 1]) / closes[i + 1] * 100
                for i in range(len(closes) - 1)]

    @staticmethod
    def get_ttm_revenue(financials: Dict) -> Optional[float]:
        """
        Calculate trailing 12 months revenue from income statements.

        Args:
            financials: Prefetched financials dict

        Returns:
            TTM revenue or None
        """
        if not financials or 'income' not in financials:
            return None

        income = financials['income']
        if len(income) < 4:
            return None

        try:
            ttm_revenue = sum(
                float(q.get('total_revenue', 0) or 0)
                for q in income[:4]
            )
            return ttm_revenue if ttm_revenue > 0 else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def get_ttm_cashflow(financials: Dict) -> Optional[Dict]:
        """
        Get trailing 12 months cash flow data.

        Args:
            financials: Prefetched financials dict

        Returns:
            {
                'operating_cashflow': float,
                'capital_expenditures': float,
                'free_cash_flow': float
            }
        """
        if not financials or 'cashflow' not in financials:
            return None

        cf = financials['cashflow']
        if len(cf) < 4:
            return None

        try:
            ocf = sum(float(q.get('operating_cashflow', 0) or 0) for q in cf[:4])
            capex = sum(float(q.get('capital_expenditures', 0) or 0) for q in cf[:4])

            return {
                'operating_cashflow': ocf,
                'capital_expenditures': abs(capex),  # capex is typically negative
                'free_cash_flow': ocf - abs(capex)
            }
        except (ValueError, TypeError):
            return None

    @staticmethod
    def get_52w_range(price_data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Get 52-week high and low from price data.

        Args:
            price_data: Prefetched price data dict

        Returns:
            (52w_high, 52w_low) tuple
        """
        if not price_data:
            return None, None

        highs = price_data.get('highs', [])[:252]
        lows = price_data.get('lows', [])[:252]

        highs = [h for h in highs if h is not None]
        lows = [l for l in lows if l is not None]

        if not highs or not lows:
            return None, None

        return max(highs), min(lows)

    @staticmethod
    def get_avg_volume(price_data: Dict, days: int = 20) -> Optional[float]:
        """
        Calculate average volume for specified days.

        Args:
            price_data: Prefetched price data dict
            days: Number of days for average (default 20)

        Returns:
            Average volume or None
        """
        if not price_data or 'volumes' not in price_data:
            return None

        volumes = price_data['volumes'][:days]
        volumes = [v for v in volumes if v is not None]

        if not volumes:
            return None

        return sum(volumes) / len(volumes)

    @staticmethod
    def get_volatility_annual(price_data: Dict, days: int = 60) -> Optional[float]:
        """
        Calculate annualized volatility from daily returns.

        Args:
            price_data: Prefetched price data dict
            days: Number of days for volatility calculation (default 60)

        Returns:
            Annualized volatility (%) or None
        """
        returns = PrefetchedDataExtractor.get_returns(price_data, days)
        if len(returns) < 20:
            return None

        returns_decimal = [r / 100 for r in returns]  # Convert back to decimal
        std_daily = np.std(returns_decimal)
        return round(std_daily * np.sqrt(252) * 100, 2)

    @staticmethod
    def get_price_momentum(price_data: Dict) -> Optional[Dict]:
        """
        Calculate various price momentum metrics.

        Args:
            price_data: Prefetched price data dict

        Returns:
            {
                'return_1m': float,   # 21 trading days
                'return_3m': float,   # 63 trading days
                'return_6m': float,   # 126 trading days
                'return_1y': float    # 252 trading days
            }
        """
        if not price_data or 'closes' not in price_data:
            return None

        closes = price_data['closes']
        if not closes or closes[0] is None:
            return None

        current_price = closes[0]

        def calc_return(days: int) -> Optional[float]:
            if len(closes) > days and closes[days] is not None and closes[days] > 0:
                return round((current_price / closes[days] - 1) * 100, 2)
            return None

        return {
            'return_1m': calc_return(21),
            'return_3m': calc_return(63),
            'return_6m': calc_return(126),
            'return_1y': calc_return(252)
        }

    @staticmethod
    def get_latest_balance_sheet(financials: Dict) -> Optional[Dict]:
        """
        Get the latest balance sheet data.

        Args:
            financials: Prefetched financials dict

        Returns:
            Latest balance sheet dict or None
        """
        if not financials or 'balance' not in financials:
            return None

        balance = financials['balance']
        if not balance:
            return None

        return balance[0]

    @staticmethod
    def get_revenue_growth_rate(financials: Dict) -> Optional[float]:
        """
        Calculate YoY revenue growth rate from income statements.

        Args:
            financials: Prefetched financials dict

        Returns:
            YoY revenue growth rate (%) or None
        """
        if not financials or 'income' not in financials:
            return None

        income = financials['income']
        if len(income) < 5:  # Need current + 4 quarters ago
            return None

        try:
            current_revenue = float(income[0].get('total_revenue', 0) or 0)
            past_revenue = float(income[4].get('total_revenue', 0) or 0)

            if past_revenue > 0:
                return round((current_revenue / past_revenue - 1) * 100, 2)
            return None
        except (ValueError, TypeError, IndexError):
            return None
