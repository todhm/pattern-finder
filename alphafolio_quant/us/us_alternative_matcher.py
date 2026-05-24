"""
US Alternative Stock Matcher Module
매도 계열 등급 종목에 대해 매수 계열 등급 대체 종목 매칭

Sell grades: 매도 고려, 매도, 강력 매도
Buy grades: 매수, 강력 매수

Priority:
1. Same industry (us_stock_basic)
2. Same sector (us_stock_basic)
3. All stocks
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Grade definitions (Korean format matching us_main.py)
SELL_GRADES = ('매도 고려', '매도', '강력 매도')
BUY_GRADES = ('매수', '강력 매수')

# Batch size for UPDATE operations (matches quant analysis batch size)
BATCH_SIZE = 50


class USAlternativeStockMatcher:
    """
    Alternative stock recommendation matcher for US sell-grade stocks
    """

    def __init__(self, db_manager):
        """
        Initialize matcher

        Args:
            db_manager: AsyncDatabaseManager instance
        """
        self.db_manager = db_manager

    async def match_single_symbol(self, symbol: str, analysis_date) -> bool:
        """
        Match alternative stock for a single sell-grade symbol

        Args:
            symbol: Stock symbol to match
            analysis_date: Analysis date

        Returns:
            bool: True if matched successfully
        """
        try:
            # Get the sell stock info
            sell_query = """
                SELECT
                    g.symbol, g.stock_name, g.final_grade, g.final_score,
                    g.value_score, g.quality_score, g.momentum_score, g.growth_score,
                    g.interaction_score, g.conviction_score,
                    g.scenario_bullish_prob, g.scenario_bearish_prob,
                    g.sharpe_ratio, g.sortino_ratio,
                    b.industry, b.sector
                FROM us_stock_grade g
                LEFT JOIN us_stock_basic b ON g.symbol = b.symbol
                WHERE g.symbol = $1 AND g.date = $2
                AND g.final_grade IN ('매도 고려', '매도', '강력 매도')
            """
            sell_result = await self.db_manager.execute_query(sell_query, symbol, analysis_date)

            if not sell_result:
                logger.debug(f"{symbol} is not a sell-grade stock, skipping")
                return False

            sell_stock = dict(sell_result[0])

            # Get buy-grade stocks
            buy_stocks = await self._get_buy_stocks(analysis_date)

            if not buy_stocks:
                logger.warning(f"No buy-grade stocks found for {analysis_date}")
                return False

            # Build indices
            industry_index = self._build_industry_index(buy_stocks)
            sector_index = self._build_sector_index(buy_stocks)

            # Prepare match
            result = self._prepare_match(sell_stock, buy_stocks, industry_index, sector_index)

            if not result:
                return False

            # Single UPDATE for option 1
            success = await self._update_alternative(
                symbol=symbol,
                analysis_date=analysis_date,
                alt_symbol=result['alt_symbol'],
                alt_stock_name=result['alt_stock_name'],
                alt_final_grade=result['alt_final_grade'],
                alt_final_score=result['alt_final_score'],
                alt_match_type=result['alt_match_type'],
                alt_reasons=result['alt_reasons']
            )

            if success:
                logger.info(f"Matched alternative for {symbol}")

            return success

        except Exception as e:
            logger.error(f"match_single_symbol failed for {symbol}: {e}")
            return False

    async def match_for_date(self, analysis_date) -> Dict:
        """
        Match alternative stocks for all sell-grade stocks on a date

        Args:
            analysis_date: Analysis date

        Returns:
            dict: {matched: int, failed: int, skipped: int}
        """
        try:
            # Get all sell-grade stocks for the date
            sell_stocks = await self._get_sell_stocks(analysis_date)

            if not sell_stocks:
                logger.info(f"No sell-grade stocks found for {analysis_date}")
                return {'matched': 0, 'failed': 0, 'skipped': 0}

            logger.info(f"Found {len(sell_stocks)} sell-grade stocks for {analysis_date}")

            # Get all buy-grade stocks for the date (cache for performance)
            buy_stocks = await self._get_buy_stocks(analysis_date)

            if not buy_stocks:
                logger.warning(f"No buy-grade stocks found for {analysis_date}")
                return {'matched': 0, 'failed': 0, 'skipped': len(sell_stocks)}

            logger.info(f"Found {len(buy_stocks)} buy-grade stocks for matching")

            # Build lookup indices for fast matching
            industry_index = self._build_industry_index(buy_stocks)
            sector_index = self._build_sector_index(buy_stocks)

            # Collect match results for batch UPDATE
            update_params = []
            failed = 0

            for sell_stock in sell_stocks:
                try:
                    result = self._prepare_match(
                        sell_stock, buy_stocks, industry_index, sector_index
                    )
                    if result:
                        # (alt_symbol, alt_stock_name, alt_final_grade, alt_final_score,
                        #  alt_match_type, alt_reasons_json, symbol, date)
                        update_params.append((
                            result['alt_symbol'],
                            result['alt_stock_name'],
                            result['alt_final_grade'],
                            result['alt_final_score'],
                            result['alt_match_type'],
                            result['alt_reasons'],
                            sell_stock['symbol'],
                            analysis_date
                        ))
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Failed to prepare match {sell_stock['symbol']}: {e}")
                    failed += 1

            # Batch UPDATE
            matched = await self._batch_update_alternatives(update_params)

            logger.info(f"Matching complete: {matched} matched, {failed} failed")
            return {'matched': matched, 'failed': failed, 'skipped': 0}

        except Exception as e:
            logger.error(f"match_for_date failed: {e}")
            return {'matched': 0, 'failed': 0, 'skipped': 0}

    def _prepare_match(
        self,
        sell_stock: Dict,
        buy_stocks: List[Dict],
        industry_index: Dict,
        sector_index: Dict
    ) -> Optional[Dict]:
        """
        Prepare match result for a single sell stock (no DB operation)

        Priority: industry > sector > all

        Returns:
            Dict with match info or None if no match found
        """
        symbol = sell_stock['symbol']
        industry = sell_stock.get('industry')
        sector = sell_stock.get('sector')

        buy_stock = None
        match_type = None

        # Priority 1: Same industry
        if industry and industry in industry_index:
            candidates = industry_index[industry]
            if candidates:
                buy_stock = candidates[0]  # Already sorted by final_score DESC
                match_type = 'industry'

        # Priority 2: Same sector
        if not buy_stock and sector and sector in sector_index:
            candidates = sector_index[sector]
            if candidates:
                buy_stock = candidates[0]
                match_type = 'sector'

        # Priority 3: All stocks (best overall)
        if not buy_stock and buy_stocks:
            buy_stock = buy_stocks[0]  # Already sorted by final_score DESC
            match_type = 'all'

        if not buy_stock:
            logger.warning(f"No alternative found for {symbol}")
            return None

        # Generate reasons
        reasons = self._generate_reasons(sell_stock, buy_stock)

        return {
            'alt_symbol': buy_stock['symbol'],
            'alt_stock_name': buy_stock['stock_name'],
            'alt_final_grade': buy_stock['final_grade'],
            'alt_final_score': buy_stock['final_score'],
            'alt_match_type': match_type,
            'alt_reasons': reasons
        }

    async def _batch_update_alternatives(self, update_params: List[tuple]) -> int:
        """
        Batch UPDATE alternative stock info

        Args:
            update_params: List of tuples (alt_symbol, alt_stock_name, alt_final_grade,
                          alt_final_score, alt_match_type, alt_reasons_json, symbol, date)

        Returns:
            int: Number of successfully updated rows
        """
        if not update_params:
            return 0

        query = """
            UPDATE us_stock_grade
            SET
                alt_symbol = $1,
                alt_stock_name = $2,
                alt_final_grade = $3,
                alt_final_score = $4,
                alt_match_type = $5,
                alt_reasons = $6
            WHERE symbol = $7 AND date = $8
        """

        total_updated = 0
        try:
            # Process in batches
            for i in range(0, len(update_params), BATCH_SIZE):
                batch = update_params[i:i + BATCH_SIZE]
                await self.db_manager.executemany(query, batch)
                total_updated += len(batch)
                logger.debug(f"Batch UPDATE: {i + len(batch)}/{len(update_params)}")

            return total_updated

        except Exception as e:
            logger.error(f"Batch UPDATE failed: {e}")
            return total_updated

    def _generate_reasons(self, sell_stock: Dict, buy_stock: Dict) -> List[Dict]:
        """
        Generate top 3 reasons why buy stock is better

        Args:
            sell_stock: Sell stock data
            buy_stock: Buy stock data

        Returns:
            List of reason dicts: [{"category": str, "diff": float, "text": str}, ...]
        """
        reasons = []

        def safe_diff(buy_val, sell_val):
            if buy_val is None or sell_val is None:
                return None
            return float(buy_val) - float(sell_val)

        def safe_val(val):
            if val is None:
                return 0
            return float(val)

        # Calculate differences
        score_diff = safe_diff(buy_stock.get('final_score'), sell_stock.get('final_score'))
        quality_diff = safe_diff(buy_stock.get('quality_score'), sell_stock.get('quality_score'))
        momentum_diff = safe_diff(buy_stock.get('momentum_score'), sell_stock.get('momentum_score'))
        growth_diff = safe_diff(buy_stock.get('growth_score'), sell_stock.get('growth_score'))
        value_diff = safe_diff(buy_stock.get('value_score'), sell_stock.get('value_score'))
        interaction_diff = safe_diff(buy_stock.get('interaction_score'), sell_stock.get('interaction_score'))
        bullish_diff = safe_diff(buy_stock.get('scenario_bullish_prob'), sell_stock.get('scenario_bullish_prob'))
        bearish_diff = safe_diff(buy_stock.get('scenario_bearish_prob'), sell_stock.get('scenario_bearish_prob'))
        sharpe_diff = safe_diff(buy_stock.get('sharpe_ratio'), sell_stock.get('sharpe_ratio'))
        sortino_diff = safe_diff(buy_stock.get('sortino_ratio'), sell_stock.get('sortino_ratio'))
        conviction_diff = safe_diff(buy_stock.get('conviction_score'), sell_stock.get('conviction_score'))

        # Generate reason texts (Method B format without tags)
        # Final Score
        if score_diff is not None and score_diff > 0:
            buy_score = safe_val(buy_stock.get('final_score'))
            text = f"종합 점수가 {score_diff:.1f}점 높아 퀀트 모델 기준 투자 매력도가 크게 우수"
            reasons.append({'category': 'final_score', 'diff': score_diff, 'text': text})

        # Interaction Score (US specific)
        if interaction_diff is not None and interaction_diff > 0:
            buy_interaction = safe_val(buy_stock.get('interaction_score'))
            text = f"팩터 상호작용 점수가 {interaction_diff:.1f}점 높아 복합적 강점 보유 ({buy_interaction:.1f}점)"
            reasons.append({'category': 'interaction', 'diff': interaction_diff, 'text': text})

        # Conviction Score (US specific)
        if conviction_diff is not None and conviction_diff > 0:
            buy_conviction = safe_val(buy_stock.get('conviction_score'))
            text = f"확신도 점수가 {conviction_diff:.1f}점 높아 팩터 간 일치도 우수 ({buy_conviction:.1f}점)"
            reasons.append({'category': 'conviction', 'diff': conviction_diff, 'text': text})

        # Scenario (Bullish + Bearish combined)
        if bullish_diff is not None and bullish_diff > 0:
            buy_bullish = safe_val(buy_stock.get('scenario_bullish_prob'))
            bearish_change = abs(bearish_diff) if bearish_diff and bearish_diff < 0 else 0
            if bearish_change > 0:
                text = f"상승 시나리오 확률 {buy_bullish:.0f}%로 {bullish_diff:.0f}%p 높고, 하락 시나리오 확률은 {bearish_change:.0f}%p 낮음"
            else:
                text = f"상승 시나리오 확률이 {bullish_diff:.0f}%p 높음 ({buy_bullish:.0f}%)"
            reasons.append({'category': 'scenario', 'diff': bullish_diff, 'text': text})

        # Growth Factor
        if growth_diff is not None and growth_diff > 0:
            buy_growth = safe_val(buy_stock.get('growth_score'))
            text = f"성장 팩터 점수가 {growth_diff:.1f}점 높아 성장 잠재력이 우수 ({buy_growth:.1f}점)"
            reasons.append({'category': 'growth', 'diff': growth_diff, 'text': text})

        # Quality Factor
        if quality_diff is not None and quality_diff > 0:
            buy_quality = safe_val(buy_stock.get('quality_score'))
            text = f"품질 팩터 점수가 {quality_diff:.1f}점 높아 재무 안정성이 우수 ({buy_quality:.1f}점)"
            reasons.append({'category': 'quality', 'diff': quality_diff, 'text': text})

        # Momentum Factor
        if momentum_diff is not None and momentum_diff > 0:
            buy_momentum = safe_val(buy_stock.get('momentum_score'))
            text = f"모멘텀 팩터 점수가 {momentum_diff:.1f}점 높아 상승 추세가 강함 ({buy_momentum:.1f}점)"
            reasons.append({'category': 'momentum', 'diff': momentum_diff, 'text': text})

        # Value Factor
        if value_diff is not None and value_diff > 0:
            buy_value = safe_val(buy_stock.get('value_score'))
            text = f"가치 팩터 점수가 {value_diff:.1f}점 높아 저평가 매력이 있음 ({buy_value:.1f}점)"
            reasons.append({'category': 'value', 'diff': value_diff, 'text': text})

        # Sharpe Ratio
        if sharpe_diff is not None and sharpe_diff > 0:
            buy_sharpe = safe_val(buy_stock.get('sharpe_ratio'))
            text = f"샤프 비율이 {sharpe_diff:.2f} 높아 위험 대비 수익률이 우수 ({buy_sharpe:.2f})"
            reasons.append({'category': 'sharpe', 'diff': sharpe_diff, 'text': text})

        # Sortino Ratio
        if sortino_diff is not None and sortino_diff > 0:
            buy_sortino = safe_val(buy_stock.get('sortino_ratio'))
            text = f"소르티노 비율이 {sortino_diff:.2f} 높아 하방 위험 대비 수익률이 우수 ({buy_sortino:.2f})"
            reasons.append({'category': 'sortino', 'diff': sortino_diff, 'text': text})

        # Bearish only (if not already included in scenario)
        if bearish_diff is not None and bearish_diff < 0 and not any(r['category'] == 'scenario' for r in reasons):
            buy_bearish = safe_val(buy_stock.get('scenario_bearish_prob'))
            text = f"하락 시나리오 확률이 {abs(bearish_diff):.0f}%p 낮아 리스크가 적음 ({buy_bearish:.0f}%)"
            reasons.append({'category': 'bearish', 'diff': abs(bearish_diff), 'text': text})

        # Round diff values to avoid floating point precision issues
        for reason in reasons:
            reason['diff'] = round(reason['diff'], 2)

        # Sort by absolute diff value (descending) and take top 3
        reasons.sort(key=lambda x: abs(x['diff']), reverse=True)
        return reasons[:3]

    async def _get_sell_stocks(self, analysis_date) -> List[Dict]:
        """Get all sell-grade stocks with industry/sector info from us_stock_basic"""
        query = """
            SELECT
                g.symbol, g.stock_name, g.final_grade, g.final_score,
                g.value_score, g.quality_score, g.momentum_score, g.growth_score,
                g.interaction_score, g.conviction_score,
                g.scenario_bullish_prob, g.scenario_bearish_prob,
                g.sharpe_ratio, g.sortino_ratio,
                b.industry, b.sector
            FROM us_stock_grade g
            LEFT JOIN us_stock_basic b ON g.symbol = b.symbol
            WHERE g.date = $1
            AND g.final_grade IN ('매도 고려', '매도', '강력 매도')
            ORDER BY g.final_score ASC
        """
        result = await self.db_manager.execute_query(query, analysis_date)
        return [dict(row) for row in result] if result else []

    async def _get_buy_stocks(self, analysis_date) -> List[Dict]:
        """Get all buy-grade stocks with industry/sector info, sorted by score DESC"""
        query = """
            SELECT
                g.symbol, g.stock_name, g.final_grade, g.final_score,
                g.value_score, g.quality_score, g.momentum_score, g.growth_score,
                g.interaction_score, g.conviction_score,
                g.scenario_bullish_prob, g.scenario_bearish_prob,
                g.sharpe_ratio, g.sortino_ratio,
                b.industry, b.sector
            FROM us_stock_grade g
            LEFT JOIN us_stock_basic b ON g.symbol = b.symbol
            WHERE g.date = $1
            AND g.final_grade IN ('매수', '강력 매수')
            ORDER BY g.final_score DESC
        """
        result = await self.db_manager.execute_query(query, analysis_date)
        return [dict(row) for row in result] if result else []

    def _build_industry_index(self, buy_stocks: List[Dict]) -> Dict[str, List[Dict]]:
        """Build industry -> stocks index (already sorted by score)"""
        index = {}
        for stock in buy_stocks:
            industry = stock.get('industry')
            if industry:
                if industry not in index:
                    index[industry] = []
                index[industry].append(stock)
        return index

    def _build_sector_index(self, buy_stocks: List[Dict]) -> Dict[str, List[Dict]]:
        """Build sector -> stocks index (already sorted by score)"""
        index = {}
        for stock in buy_stocks:
            sector = stock.get('sector')
            if sector:
                if sector not in index:
                    index[sector] = []
                index[sector].append(stock)
        return index

    async def _update_alternative(
        self,
        symbol: str,
        analysis_date,
        alt_symbol: str,
        alt_stock_name: str,
        alt_final_grade: str,
        alt_final_score: float,
        alt_match_type: str,
        alt_reasons: List[Dict]
    ) -> bool:
        """Update alternative stock info in us_stock_grade"""
        try:
            query = """
                UPDATE us_stock_grade
                SET
                    alt_symbol = $1,
                    alt_stock_name = $2,
                    alt_final_grade = $3,
                    alt_final_score = $4,
                    alt_match_type = $5,
                    alt_reasons = $6
                WHERE symbol = $7 AND date = $8
            """
            await self.db_manager.execute_query(
                query,
                alt_symbol,
                alt_stock_name,
                alt_final_grade,
                alt_final_score,
                alt_match_type,
                alt_reasons,
                symbol,
                analysis_date
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update alternative for {symbol}: {e}")
            return False
