"""
Market State Classifier for Korean Stock Analysis
Classifies stocks into 19 market states (18 + Others) based on 8 conditions
Target: 95% coverage with 24-hour caching
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketClassifier:
    """
    Classify stocks into 19 market states based on 9 conditions (Phase 3.10)
    - Exchange (KOSPI/KOSDAQ)
    - Market Cap (MEGA/LARGE/MEDIUM/SMALL)
    - Liquidity (HIGH/MEDIUM/LOW)
    - Economic Cycle (EXPANSION/RECOVERY/SLOWDOWN/RECESSION/NEUTRAL)
    - Market Sentiment (OVERHEATED/GREED/NEUTRAL/FEAR/PANIC)
    - Sector Cycle (HOT/GROWING/STABLE/DECLINING/COLD)
    - Theme (17 themes)
    - Volatility (VERY_HIGH/HIGH/MEDIUM/LOW)
    - Supply/Demand (STRONG_BUY/INST_LED/FOREIGN_LED/STRONG_SELL/NEUTRAL) [Phase 3.10]
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize market classifier

        Args:
            cache_dir: Directory for cache file (default: kr/)
        """
        if cache_dir is None:
            cache_dir = os.path.dirname(__file__)

        self.cache_file = os.path.join(cache_dir, 'market_classification_cache.json')
        self.cache_duration = 86400  # 24 hours in seconds
        self.classifications = self._define_classifications()

    def _define_classifications(self) -> List[Dict]:
        """
        Define 18 classification rules
        Priority-based matching (higher priority = more specific)

        Returns:
            List of classification rule dictionaries
        """
        return [
            # Large Cap Group (6)
            {
                'id': 1,
                'name': 'KOSPI대형-확장과열-공격형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEGA', 'LARGE'],
                    'economic_cycle': ['EXPANSION'],
                    'market_sentiment': ['OVERHEATED', 'GREED'],
                    'sector_cycle': ['HOT', 'GROWING'],
                    'volatility': ['HIGH', 'VERY_HIGH'],
                    'supply_demand': ['STRONG_BUY', 'FOREIGN_LED']  # Phase 3.10
                },
                'priority': 10
            },
            {
                'id': 2,
                'name': 'KOSPI대형-확장중립-성장형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEGA', 'LARGE'],
                    'economic_cycle': ['EXPANSION', 'NEUTRAL'],
                    'market_sentiment': ['NEUTRAL', 'GREED'],
                    'sector_cycle': ['GROWING', 'STABLE'],
                    'volatility': ['MEDIUM', 'LOW']
                },
                'priority': 8
            },
            {
                'id': 3,
                'name': 'KOSPI대형-둔화공포-방어형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEGA', 'LARGE'],
                    'economic_cycle': ['SLOWDOWN'],
                    'market_sentiment': ['FEAR', 'NEUTRAL'],
                    'sector_cycle': ['STABLE', 'DECLINING'],
                    'volatility': ['LOW', 'MEDIUM']
                },
                'priority': 9
            },
            {
                'id': 4,
                'name': 'KOSPI대형-침체패닉-초방어형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEGA', 'LARGE'],
                    'economic_cycle': ['RECESSION'],
                    'market_sentiment': ['PANIC', 'FEAR'],
                    'sector_cycle': ['DECLINING', 'COLD'],
                    'volatility': ['LOW', 'MEDIUM'],
                    'supply_demand': ['STRONG_SELL', 'INST_LED', 'NEUTRAL']  # Phase 3.10
                },
                'priority': 10
            },
            {
                'id': 5,
                'name': 'KOSPI대형-회복탐욕-밸류형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEGA', 'LARGE'],
                    'economic_cycle': ['RECOVERY'],
                    'market_sentiment': ['GREED', 'NEUTRAL'],
                    'sector_cycle': ['STABLE', 'GROWING'],
                    'volatility': ['MEDIUM', 'LOW']
                },
                'priority': 9
            },
            {
                'id': 6,
                'name': 'KOSPI대형-중립안정-균형형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEGA', 'LARGE'],
                    'economic_cycle': ['NEUTRAL'],
                    'market_sentiment': ['NEUTRAL'],
                    'sector_cycle': ['STABLE'],
                    'volatility': ['MEDIUM', 'LOW']
                },
                'priority': 5
            },

            # Mid Cap Group (6)
            {
                'id': 7,
                'name': 'KOSPI중형-확장과열-모멘텀형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEDIUM'],
                    'economic_cycle': ['EXPANSION'],
                    'market_sentiment': ['OVERHEATED', 'GREED'],
                    'sector_cycle': ['HOT', 'GROWING'],
                    'volatility': ['HIGH', 'VERY_HIGH']
                },
                'priority': 9
            },
            {
                'id': 8,
                'name': 'KOSPI중형-회복중립-성장형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEDIUM'],
                    'economic_cycle': ['RECOVERY', 'NEUTRAL'],
                    'market_sentiment': ['NEUTRAL', 'GREED'],
                    'sector_cycle': ['GROWING', 'STABLE'],
                    'volatility': ['MEDIUM', 'HIGH']
                },
                'priority': 7
            },
            {
                'id': 9,
                'name': 'KOSPI중형-둔화공포-혼조형',
                'rules': {
                    'exchange': ['KOSPI'],
                    'market_cap_category': ['MEDIUM'],
                    'economic_cycle': ['SLOWDOWN', 'NEUTRAL'],
                    'market_sentiment': ['FEAR', 'NEUTRAL'],
                    'sector_cycle': ['STABLE', 'DECLINING'],
                    'volatility': ['MEDIUM']
                },
                'priority': 7
            },
            {
                'id': 10,
                'name': 'KOSDAQ중형-확장탐욕-공격성장형',
                'rules': {
                    'exchange': ['KOSDAQ'],
                    'market_cap_category': ['MEDIUM'],
                    'economic_cycle': ['EXPANSION'],
                    'market_sentiment': ['GREED', 'OVERHEATED'],
                    'sector_cycle': ['HOT', 'GROWING'],
                    'volatility': ['HIGH', 'VERY_HIGH'],
                    'supply_demand': ['STRONG_BUY', 'FOREIGN_LED']  # Phase 3.10
                },
                'priority': 10
            },
            {
                'id': 11,
                'name': 'KOSDAQ중형-회복중립-성장테마형',
                'rules': {
                    'exchange': ['KOSDAQ'],
                    'market_cap_category': ['MEDIUM'],
                    'economic_cycle': ['RECOVERY', 'NEUTRAL'],
                    'market_sentiment': ['NEUTRAL', 'GREED'],
                    'sector_cycle': ['GROWING', 'STABLE'],
                    'volatility': ['HIGH', 'MEDIUM']
                },
                'priority': 8
            },
            {
                'id': 12,
                'name': 'KOSDAQ중형-침체공포-역발상형',
                'rules': {
                    'exchange': ['KOSDAQ'],
                    'market_cap_category': ['MEDIUM'],
                    'economic_cycle': ['RECESSION', 'SLOWDOWN'],
                    'market_sentiment': ['FEAR', 'PANIC'],
                    'sector_cycle': ['COLD', 'DECLINING'],
                    'volatility': ['HIGH', 'VERY_HIGH']
                },
                'priority': 9
            },

            # Small Cap Group (4)
            {
                'id': 13,
                'name': 'KOSDAQ소형-핫섹터-초고위험형',
                'rules': {
                    'exchange': ['KOSDAQ'],
                    'market_cap_category': ['SMALL'],
                    'sector_cycle': ['HOT'],
                    'volatility': ['VERY_HIGH', 'HIGH'],
                    'liquidity_level': ['LOW', 'MEDIUM']
                },
                'priority': 10
            },
            {
                'id': 14,
                'name': 'KOSDAQ소형-성장테마-고위험형',
                'rules': {
                    'exchange': ['KOSDAQ'],
                    'market_cap_category': ['SMALL'],
                    'sector_cycle': ['GROWING', 'HOT'],
                    'volatility': ['HIGH', 'VERY_HIGH'],
                    'economic_cycle': ['EXPANSION', 'RECOVERY', 'NEUTRAL']
                },
                'priority': 8
            },
            {
                'id': 15,
                'name': 'KOSDAQ소형-침체-극단역발상형',
                'rules': {
                    'exchange': ['KOSDAQ'],
                    'market_cap_category': ['SMALL'],
                    'economic_cycle': ['RECESSION', 'SLOWDOWN'],
                    'market_sentiment': ['PANIC', 'FEAR'],
                    'sector_cycle': ['COLD', 'DECLINING'],
                    'volatility': ['VERY_HIGH', 'HIGH']
                },
                'priority': 9
            },
            {
                'id': 16,
                'name': 'KOSDAQ소형-회복-모멘텀형',
                'rules': {
                    'exchange': ['KOSDAQ'],
                    'market_cap_category': ['SMALL'],
                    'economic_cycle': ['RECOVERY'],
                    'sector_cycle': ['HOT', 'GROWING'],
                    'volatility': ['HIGH', 'VERY_HIGH']
                },
                'priority': 8
            },

            # Special Situation Group (2)
            {
                'id': 17,
                'name': '전시장-극저유동성-고위험형',
                'rules': {
                    'market_sentiment': ['PANIC'],
                    'volatility': ['VERY_HIGH'],
                    'liquidity_level': ['LOW'],
                    'economic_cycle': ['RECESSION']
                },
                'priority': 10
            },
            {
                'id': 18,
                'name': '테마특화-모멘텀폭발형',
                'rules': {
                    'sector_cycle': ['HOT'],
                    'volatility': ['VERY_HIGH'],
                    'theme': ['AI_BigData', 'Semiconductor', 'Bio_DrugRD']
                },
                'priority': 9
            }
        ]

    def _calculate_match_score(self, conditions: Dict, rules: Dict) -> float:
        """
        Calculate match score between conditions and classification rules

        Args:
            conditions: Stock conditions (8 values)
            rules: Classification rules

        Returns:
            Match score (0.0 ~ 1.0)
        """
        total_rules = len(rules)
        matched_rules = 0

        for rule_key, rule_values in rules.items():
            condition_value = conditions.get(rule_key)

            # Skip if condition not available
            if condition_value is None:
                continue

            # Check if condition matches any rule value
            if condition_value in rule_values:
                matched_rules += 1

        # Calculate score
        if total_rules == 0:
            return 0.0

        return matched_rules / total_rules

    def classify(self, conditions: Dict) -> str:
        """
        Classify stock into one of 19 market states

        Args:
            conditions: Dict with keys:
                - exchange: str (KOSPI/KOSDAQ)
                - market_cap_category: str (MEGA/LARGE/MEDIUM/SMALL)
                - liquidity_level: str (HIGH/MEDIUM/LOW)
                - economic_cycle: str (EXPANSION/RECOVERY/SLOWDOWN/RECESSION/NEUTRAL)
                - market_sentiment: str (OVERHEATED/GREED/NEUTRAL/FEAR/PANIC)
                - sector_cycle: str (HOT/GROWING/STABLE/DECLINING/COLD)
                - theme: str (17 themes)
                - volatility: str (VERY_HIGH/HIGH/MEDIUM/LOW)

        Returns:
            Classification name (18 types or '기타')
        """
        # Step 1: Try exact or high-confidence match (>= 80%)
        best_match = None
        best_score = 0.0
        best_priority = 0

        for classification in self.classifications:
            score = self._calculate_match_score(conditions, classification['rules'])

            # Prioritize by score first, then priority
            if score > best_score or (score == best_score and classification['priority'] > best_priority):
                best_score = score
                best_match = classification['name']
                best_priority = classification['priority']

        # High confidence match (>= 70%)
        if best_score >= 0.70:
            logger.debug(f"Classified with score {best_score:.2f}: {best_match}")
            return best_match

        # Step 2: Medium confidence match (>= 50%)
        if best_score >= 0.50:
            logger.debug(f"Classified with medium score {best_score:.2f}: {best_match}")
            return best_match

        # Step 3: Fallback classification based on primary factors
        fallback = self._fallback_classification(conditions)
        if fallback:
            logger.debug(f"Fallback classification: {fallback}")
            return fallback

        # Step 4: Default to '기타'
        logger.warning(f"Classification failed (score: {best_score:.2f}), using '기타'")
        return '기타'

    def _fallback_classification(self, conditions: Dict) -> Optional[str]:
        """
        Fallback classification based on primary factors
        Used when no good match found (< 50% score)

        Args:
            conditions: Stock conditions

        Returns:
            Fallback classification or None
        """
        exchange = conditions.get('exchange')
        market_cap = conditions.get('market_cap_category')
        economic_cycle = conditions.get('economic_cycle')
        market_sentiment = conditions.get('market_sentiment')

        # Large cap fallbacks
        if exchange == 'KOSPI' and market_cap in ['MEGA', 'LARGE']:
            if economic_cycle in ['EXPANSION', 'RECOVERY']:
                return 'KOSPI대형-확장중립-성장형'
            elif economic_cycle in ['SLOWDOWN', 'RECESSION']:
                return 'KOSPI대형-둔화공포-방어형'
            else:
                return 'KOSPI대형-중립안정-균형형'

        # Mid cap fallbacks
        if market_cap == 'MEDIUM':
            if exchange == 'KOSPI':
                return 'KOSPI중형-회복중립-성장형'
            else:
                return 'KOSDAQ중형-회복중립-성장테마형'

        # Small cap fallbacks
        if exchange == 'KOSDAQ' and market_cap == 'SMALL':
            if economic_cycle in ['RECESSION', 'SLOWDOWN']:
                return 'KOSDAQ소형-침체-극단역발상형'
            else:
                return 'KOSDAQ소형-성장테마-고위험형'

        # Crisis fallback
        if market_sentiment == 'PANIC':
            return '전시장-극저유동성-고위험형'

        return None

    def load_cache(self) -> Optional[Dict]:
        """
        Load classification cache if valid (within 24 hours)

        Returns:
            Cached classifications or None
        """
        if not os.path.exists(self.cache_file):
            logger.info("No cache file found")
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Check expiry
            expiry_time = datetime.fromisoformat(cache_data['expiry'])
            if datetime.now() > expiry_time:
                logger.info("Cache expired")
                return None

            logger.info(f"Cache loaded: {len(cache_data['classifications'])} classifications")
            return cache_data['classifications']

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def save_cache(self, classifications: Dict):
        """
        Save classifications to cache file

        Args:
            classifications: Dict of {symbol: classification_name}
        """
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'expiry': (datetime.now() + timedelta(seconds=self.cache_duration)).isoformat(),
                'classifications': classifications
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Cache saved: {len(classifications)} classifications")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def classify_batch(self, conditions_dict: Dict[str, Dict]) -> Dict[str, str]:
        """
        Classify multiple stocks at once

        Args:
            conditions_dict: Dict of {symbol: conditions}

        Returns:
            Dict of {symbol: classification_name}
        """
        # Try to load cache
        cached = self.load_cache()
        if cached:
            logger.info("Using cached classifications")
            return cached

        # Classify all stocks
        results = {}
        for symbol, conditions in conditions_dict.items():
            results[symbol] = self.classify(conditions)

        # Save to cache
        self.save_cache(results)

        return results

    def get_classification_stats(self, classifications: Dict[str, str]) -> Dict:
        """
        Get statistics about classifications

        Args:
            classifications: Dict of {symbol: classification_name}

        Returns:
            Statistics dict
        """
        from collections import Counter

        counts = Counter(classifications.values())
        total = len(classifications)

        stats = {
            'total_stocks': total,
            'classification_counts': dict(counts),
            'coverage': {
                'classified': total - counts.get('기타', 0),
                'others': counts.get('기타', 0),
                'coverage_rate': (total - counts.get('기타', 0)) / total * 100 if total > 0 else 0
            }
        }

        return stats


def test_classifier():
    """Test market classifier with sample conditions"""
    print("\n" + "="*80)
    print("Market Classifier Test")
    print("="*80 + "\n")

    classifier = MarketClassifier()

    # Test case 1: Large cap expansion
    test1 = {
        'exchange': 'KOSPI',
        'market_cap_category': 'MEGA',
        'liquidity_level': 'HIGH',
        'economic_cycle': 'EXPANSION',
        'market_sentiment': 'GREED',
        'sector_cycle': 'HOT',
        'theme': 'Electronics',
        'volatility': 'HIGH'
    }

    # Test case 2: Small cap recession
    test2 = {
        'exchange': 'KOSDAQ',
        'market_cap_category': 'SMALL',
        'liquidity_level': 'LOW',
        'economic_cycle': 'RECESSION',
        'market_sentiment': 'PANIC',
        'sector_cycle': 'COLD',
        'theme': 'Bio_DrugRD',
        'volatility': 'VERY_HIGH'
    }

    # Test case 3: Mid cap neutral
    test3 = {
        'exchange': 'KOSPI',
        'market_cap_category': 'MEDIUM',
        'liquidity_level': 'MEDIUM',
        'economic_cycle': 'NEUTRAL',
        'market_sentiment': 'NEUTRAL',
        'sector_cycle': 'STABLE',
        'theme': 'Traditional_Manufacturing',
        'volatility': 'MEDIUM'
    }

    tests = [
        ('Test 1: KOSPI Large Cap Expansion', test1),
        ('Test 2: KOSDAQ Small Cap Recession', test2),
        ('Test 3: KOSPI Mid Cap Neutral', test3)
    ]

    for name, conditions in tests:
        result = classifier.classify(conditions)
        print(f"{name}")
        print(f"Result: {result}")
        print("-" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_classifier()
