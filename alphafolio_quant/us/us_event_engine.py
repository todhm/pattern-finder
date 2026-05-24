"""
US Event Engine - Phase 3.4.2 Event-aware Score Modifier

Purpose: Adjust stock scores based on event-driven signals
- Earnings Proximity Filter: Reduce momentum reliability near earnings
- Options Sentiment Signal: Put/Call ratio and IV skew analysis
- Insider Signal: Cluster buying/selling detection
- GEX Signal: Gamma Exposure based volatility risk (Phase 3.4.2)
- News Sentiment Signal: Recent news sentiment analysis (Phase 3.4.2)

DB Tables Used:
- us_earnings_calendar: reportdate, estimate, fiscaldateending
- us_option_daily_summary: total_call_volume, total_put_volume, avg_call_iv, avg_put_iv,
                           net_gex, call_gex, put_gex, gex_ratio (Phase 3.4.2)
- us_insider_transactions: executive (ARRAY), executive_title (ARRAY), acquisition_or_disposal, shares
- us_news: ticker, overall_sentiment_score, ticker_sentiment_score, time_published (Phase 3.4.2)

Score Impact:
- Event modifier range: -20 to +20 points (expanded from -15 to +15)
- Applied after total_score calculation, before grade determination

File: us/us_event_engine.py
"""

import logging
from typing import Dict, Optional
from datetime import date, timedelta
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class USEventEngine:
    """Event-aware Score Modifier Engine for US Stocks"""

    def __init__(self, db_manager):
        """
        Initialize Event Engine

        Args:
            db_manager: AsyncDatabaseManager instance
        """
        self.db = db_manager

    async def calculate_event_modifier(self, symbol: str, analysis_date: date) -> Dict:
        """
        Calculate combined event modifier for a stock

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date

        Returns:
            {
                'total_modifier': float (-20 to +20),
                'earnings_modifier': float,
                'options_modifier': float,
                'insider_modifier': float,
                'gex_modifier': float,
                'news_modifier': float,
                'reason': str,
                'details': Dict
            }
        """
        try:
            # Sequential processing (following existing factor pattern)
            earnings_result = await self._get_earnings_modifier(symbol, analysis_date)
            options_result = await self._get_options_sentiment(symbol, analysis_date)
            insider_result = await self._get_insider_signal(symbol, analysis_date)
            gex_result = await self._get_gex_modifier(symbol, analysis_date)
            news_result = await self._get_news_sentiment(symbol, analysis_date)

            # Combine modifiers
            earnings_mod = earnings_result.get('modifier', 0)
            options_mod = options_result.get('modifier', 0)
            insider_mod = insider_result.get('modifier', 0)
            gex_mod = gex_result.get('modifier', 0)
            news_mod = news_result.get('modifier', 0)

            total_modifier = earnings_mod + options_mod + insider_mod + gex_mod + news_mod
            total_modifier = max(-20, min(20, total_modifier))  # Clamp to -20 ~ +20

            # Build reason string
            reasons = []
            if earnings_result.get('reason') and earnings_result['reason'] != 'NO_DATA':
                reasons.append(f"Earnings:{earnings_result['reason']}")
            if options_result.get('reason') and options_result['reason'] != 'NO_DATA':
                reasons.append(f"Options:{options_result['reason']}")
            if insider_result.get('reason') and insider_result['reason'] != 'NO_DATA':
                reasons.append(f"Insider:{insider_result['reason']}")
            if gex_result.get('reason') and gex_result['reason'] != 'NO_DATA':
                reasons.append(f"GEX:{gex_result['reason']}")
            if news_result.get('reason') and news_result['reason'] != 'NO_DATA':
                reasons.append(f"News:{news_result['reason']}")

            reason_str = ', '.join(reasons) if reasons else 'NEUTRAL'

            logger.debug(f"{symbol}: Event modifier = {total_modifier:+.1f} ({reason_str})")

            return {
                'total_modifier': round(total_modifier, 1),
                'earnings_modifier': round(earnings_mod, 1),
                'options_modifier': round(options_mod, 1),
                'insider_modifier': round(insider_mod, 1),
                'gex_modifier': round(gex_mod, 1),
                'news_modifier': round(news_mod, 1),
                'reason': reason_str,
                'details': {
                    'earnings': earnings_result,
                    'options': options_result,
                    'insider': insider_result,
                    'gex': gex_result,
                    'news': news_result
                }
            }

        except Exception as e:
            logger.error(f"{symbol}: Event modifier calculation failed - {e}")
            return {
                'total_modifier': 0,
                'earnings_modifier': 0,
                'options_modifier': 0,
                'insider_modifier': 0,
                'gex_modifier': 0,
                'news_modifier': 0,
                'reason': f'ERROR:{str(e)[:50]}',
                'details': {}
            }

    async def _get_earnings_modifier(self, symbol: str, analysis_date: date) -> Dict:
        """
        Earnings Proximity Filter

        Logic:
        - D-5 to D-1: Momentum unreliable, modifier -5
        - D-day: Highest uncertainty, modifier -3
        - No upcoming earnings in 30 days: modifier 0 (neutral)

        DB Schema:
        - us_earnings_calendar: symbol, reportdate, estimate, fiscaldateending
        """
        query = """
        SELECT reportdate, estimate, fiscaldateending
        FROM us_earnings_calendar
        WHERE symbol = $1
          AND reportdate >= $2
        ORDER BY reportdate ASC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result:
                return {'modifier': 0, 'reason': 'NO_DATA', 'days_to_earnings': None}

            row = result[0]
            report_date = row['reportdate']
            days_to_earnings = (report_date - analysis_date).days

            modifier = 0
            reason = 'NORMAL'

            # Earnings imminent (D-5 to D-1)
            if 1 <= days_to_earnings <= 5:
                modifier = -5
                reason = f'IMMINENT_D-{days_to_earnings}'

            # Earnings today
            elif days_to_earnings == 0:
                modifier = -3
                reason = 'EARNINGS_TODAY'

            # Earnings within 2 weeks (mild caution)
            elif 6 <= days_to_earnings <= 14:
                modifier = -1
                reason = f'UPCOMING_D-{days_to_earnings}'

            return {
                'modifier': modifier,
                'reason': reason,
                'days_to_earnings': days_to_earnings,
                'report_date': str(report_date),
                'estimate': float(row['estimate']) if row['estimate'] else None
            }

        except Exception as e:
            logger.warning(f"{symbol}: Earnings query failed - {e}")
            return {'modifier': 0, 'reason': 'ERROR', 'days_to_earnings': None}

    async def _get_options_sentiment(self, symbol: str, analysis_date: date) -> Dict:
        """
        Options Sentiment Signal

        Logic:
        - Put/Call Ratio > 1.5: Bearish sentiment, modifier -5
        - Put/Call Ratio < 0.5: Bullish sentiment, modifier +3
        - IV Skew (Put IV - Call IV) > 10%: Downside fear, modifier -3

        DB Schema:
        - us_option_daily_summary: symbol, date, total_call_volume, total_put_volume,
          avg_call_iv, avg_put_iv, avg_implied_volatility
        """
        query = """
        SELECT
            total_call_volume,
            total_put_volume,
            avg_call_iv,
            avg_put_iv,
            avg_implied_volatility
        FROM us_option_daily_summary
        WHERE symbol = $1
          AND date <= $2
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result:
                return {'modifier': 0, 'reason': 'NO_DATA', 'pc_ratio': None, 'iv_skew': None}

            row = result[0]
            call_vol = self._to_float(row['total_call_volume']) or 0
            put_vol = self._to_float(row['total_put_volume']) or 0
            call_iv = self._to_float(row['avg_call_iv']) or 0
            put_iv = self._to_float(row['avg_put_iv']) or 0

            modifier = 0
            reasons = []

            # Put/Call Ratio analysis
            pc_ratio = None
            if call_vol > 0:
                pc_ratio = put_vol / call_vol

                if pc_ratio > 1.5:
                    modifier -= 5
                    reasons.append(f'HIGH_PC_{pc_ratio:.2f}')
                elif pc_ratio < 0.5:
                    modifier += 3
                    reasons.append(f'LOW_PC_{pc_ratio:.2f}')

            # IV Skew analysis (Put IV vs Call IV)
            iv_skew = None
            if call_iv > 0 and put_iv > 0:
                iv_skew = (put_iv - call_iv) / call_iv

                if iv_skew > 0.10:  # Put IV 10% higher than Call IV
                    modifier -= 3
                    reasons.append(f'NEG_SKEW_{iv_skew:.1%}')
                elif iv_skew < -0.10:  # Call IV higher (bullish)
                    modifier += 2
                    reasons.append(f'POS_SKEW_{iv_skew:.1%}')

            reason_str = ','.join(reasons) if reasons else 'NEUTRAL'

            return {
                'modifier': modifier,
                'reason': reason_str,
                'pc_ratio': round(pc_ratio, 3) if pc_ratio else None,
                'iv_skew': round(iv_skew, 4) if iv_skew else None,
                'call_volume': int(call_vol),
                'put_volume': int(put_vol)
            }

        except Exception as e:
            logger.warning(f"{symbol}: Options query failed - {e}")
            return {'modifier': 0, 'reason': 'ERROR', 'pc_ratio': None, 'iv_skew': None}

    async def _get_insider_signal(self, symbol: str, analysis_date: date) -> Dict:
        """
        Insider Signal Integration

        Logic:
        - Cluster Buying (3+ executives in 90 days): modifier +10
        - Multiple Buying (2 executives): modifier +5
        - CEO/CFO large selling (>10,000 shares): modifier -5

        DB Schema:
        - us_insider_transactions: date, symbol, executive (ARRAY), executive_title (ARRAY),
          acquisition_or_disposal ('A' or 'D'), shares, share_price
        """
        # Calculate 90-day lookback period
        start_date = analysis_date - timedelta(days=90)

        query = """
        SELECT
            date,
            executive,
            executive_title,
            acquisition_or_disposal,
            shares,
            share_price
        FROM us_insider_transactions
        WHERE symbol = $1
          AND date BETWEEN $2 AND $3
        ORDER BY date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, start_date, analysis_date)

            if not result:
                return {
                    'modifier': 0,
                    'reason': 'NO_DATA',
                    'buy_count': 0,
                    'sell_count': 0,
                    'unique_buyers': 0
                }

            # Separate buys and sells
            buys = []
            sells = []
            unique_buyers = set()
            c_level_sells = []

            for row in result:
                disposition = row['acquisition_or_disposal']
                shares = self._to_float(row['shares']) or 0
                executives = row['executive'] or []  # ARRAY type
                titles = row['executive_title'] or []  # ARRAY type

                if disposition == 'A':  # Acquisition (Buy)
                    buys.append(row)
                    for exec_name in executives:
                        if exec_name:
                            unique_buyers.add(exec_name.strip().upper())

                elif disposition == 'D':  # Disposal (Sell)
                    sells.append(row)
                    # Check for C-level selling
                    for title in titles:
                        if title:
                            title_upper = title.upper()
                            if 'CEO' in title_upper or 'CFO' in title_upper or 'CHIEF' in title_upper:
                                c_level_sells.append({'shares': shares, 'title': title})

            modifier = 0
            reasons = []

            # Cluster Buying analysis
            num_unique_buyers = len(unique_buyers)
            if num_unique_buyers >= 3:
                modifier += 10
                reasons.append(f'CLUSTER_BUY_{num_unique_buyers}')
            elif num_unique_buyers >= 2:
                modifier += 5
                reasons.append(f'MULTI_BUY_{num_unique_buyers}')
            elif num_unique_buyers == 1 and len(buys) >= 2:
                modifier += 2
                reasons.append('REPEAT_BUY')

            # C-level selling analysis
            if c_level_sells:
                total_c_level_sold = sum(s['shares'] for s in c_level_sells)
                if total_c_level_sold > 10000:
                    modifier -= 5
                    reasons.append(f'C_LEVEL_SELL_{int(total_c_level_sold):,}')
                elif total_c_level_sold > 5000:
                    modifier -= 2
                    reasons.append(f'C_LEVEL_SELL_{int(total_c_level_sold):,}')

            reason_str = ','.join(reasons) if reasons else 'NEUTRAL'

            return {
                'modifier': modifier,
                'reason': reason_str,
                'buy_count': len(buys),
                'sell_count': len(sells),
                'unique_buyers': num_unique_buyers,
                'c_level_sells': len(c_level_sells)
            }

        except Exception as e:
            logger.warning(f"{symbol}: Insider query failed - {e}")
            return {
                'modifier': 0,
                'reason': 'ERROR',
                'buy_count': 0,
                'sell_count': 0,
                'unique_buyers': 0
            }

    async def _get_gex_modifier(self, symbol: str, analysis_date: date) -> Dict:
        """
        GEX (Gamma Exposure) Signal - Phase 3.4.2

        Logic:
        - Negative Net GEX (dealers amplify volatility): Risk signal, modifier -5
        - Strong Positive Net GEX (dealers stabilize): Stability signal, modifier +3
        - GEX Ratio threshold based analysis

        DB Schema:
        - us_option_daily_summary: symbol, date, net_gex, call_gex, put_gex, gex_ratio
        """
        query = """
        SELECT
            net_gex,
            call_gex,
            put_gex,
            gex_ratio
        FROM us_option_daily_summary
        WHERE symbol = $1
          AND date <= $2
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result:
                return {
                    'modifier': 0,
                    'reason': 'NO_DATA',
                    'net_gex': None,
                    'gex_ratio': None
                }

            row = result[0]
            net_gex = self._to_float(row['net_gex'])
            gex_ratio = self._to_float(row['gex_ratio'])

            modifier = 0
            reasons = []

            if net_gex is not None:
                # Negative Net GEX: Dealers hedge same direction → Volatility amplifying
                if net_gex < 0:
                    modifier -= 5
                    reasons.append(f'NEG_GEX_{net_gex/1e9:.1f}B')
                # Strong Positive Net GEX: Dealers hedge opposite → Market stabilizing
                elif net_gex > 1e10:  # > $10B positive GEX
                    modifier += 3
                    reasons.append(f'POS_GEX_{net_gex/1e9:.1f}B')

            # Additional signal from GEX Ratio
            if gex_ratio is not None:
                if gex_ratio < -0.01:  # Strong negative ratio
                    modifier -= 2
                    reasons.append(f'NEG_RATIO_{gex_ratio:.4f}')
                elif gex_ratio > 0.02:  # Strong positive ratio
                    modifier += 2
                    reasons.append(f'POS_RATIO_{gex_ratio:.4f}')

            reason_str = ','.join(reasons) if reasons else 'NEUTRAL'

            return {
                'modifier': modifier,
                'reason': reason_str,
                'net_gex': net_gex,
                'gex_ratio': gex_ratio,
                'call_gex': self._to_float(row['call_gex']),
                'put_gex': self._to_float(row['put_gex'])
            }

        except Exception as e:
            logger.warning(f"{symbol}: GEX query failed - {e}")
            return {
                'modifier': 0,
                'reason': 'ERROR',
                'net_gex': None,
                'gex_ratio': None
            }

    async def _get_news_sentiment(self, symbol: str, analysis_date: date) -> Dict:
        """
        News Sentiment Signal - Phase 3.4.2

        Logic:
        - Strong positive sentiment (avg > 0.3): Bullish signal, modifier +3
        - Strong negative sentiment (avg < -0.3): Bearish signal, modifier -5
        - Recent news volume boost (many articles): amplify signal

        DB Schema:
        - us_news: ticker, date, overall_sentiment_score, ticker_sentiment_score, time_published
        """
        # 7-day lookback for news sentiment
        start_date = analysis_date - timedelta(days=7)

        query = """
        SELECT
            ticker_sentiment_score,
            overall_sentiment_score,
            date
        FROM us_news
        WHERE symbol = $1
          AND date BETWEEN $2 AND $3
        ORDER BY date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, start_date, analysis_date)

            if not result:
                return {
                    'modifier': 0,
                    'reason': 'NO_DATA',
                    'avg_sentiment': None,
                    'news_count': 0
                }

            # Calculate average sentiment
            sentiments = []
            for row in result:
                ticker_score = self._to_float(row['ticker_sentiment_score'])
                overall_score = self._to_float(row['overall_sentiment_score'])
                # Prefer ticker-specific sentiment, fallback to overall
                score = ticker_score if ticker_score is not None else overall_score
                if score is not None:
                    sentiments.append(score)

            if not sentiments:
                return {
                    'modifier': 0,
                    'reason': 'NO_VALID_SCORES',
                    'avg_sentiment': None,
                    'news_count': len(result)
                }

            avg_sentiment = sum(sentiments) / len(sentiments)
            news_count = len(sentiments)

            modifier = 0
            reasons = []

            # Sentiment-based modifier
            if avg_sentiment > 0.3:
                modifier += 3
                reasons.append(f'BULLISH_{avg_sentiment:.2f}')
            elif avg_sentiment > 0.15:
                modifier += 1
                reasons.append(f'POS_{avg_sentiment:.2f}')
            elif avg_sentiment < -0.3:
                modifier -= 5
                reasons.append(f'BEARISH_{avg_sentiment:.2f}')
            elif avg_sentiment < -0.15:
                modifier -= 2
                reasons.append(f'NEG_{avg_sentiment:.2f}')

            # Volume boost: many news articles amplify signal
            if news_count >= 5:
                if modifier > 0:
                    modifier += 1
                    reasons.append(f'HIGH_VOL_{news_count}')
                elif modifier < 0:
                    modifier -= 1
                    reasons.append(f'HIGH_VOL_{news_count}')

            reason_str = ','.join(reasons) if reasons else 'NEUTRAL'

            return {
                'modifier': modifier,
                'reason': reason_str,
                'avg_sentiment': round(avg_sentiment, 4),
                'news_count': news_count
            }

        except Exception as e:
            logger.warning(f"{symbol}: News sentiment query failed - {e}")
            return {
                'modifier': 0,
                'reason': 'ERROR',
                'avg_sentiment': None,
                'news_count': 0
            }

    def _to_float(self, value) -> Optional[float]:
        """Convert Decimal/int/str to float safely"""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# Convenience function for integration
async def calculate_event_modifier(symbol: str, db_manager, analysis_date: date) -> Dict:
    """
    Standalone function for event modifier calculation

    Args:
        symbol: Stock symbol
        db_manager: AsyncDatabaseManager instance
        analysis_date: Analysis date

    Returns:
        Event modifier result dict
    """
    engine = USEventEngine(db_manager)
    return await engine.calculate_event_modifier(symbol, analysis_date)
