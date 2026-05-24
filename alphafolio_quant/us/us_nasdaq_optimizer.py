"""
US NASDAQ Exchange Optimizer

Data-driven optimization for NASDAQ exchange based on us_exchange_factor_ic_20251203.csv analysis.

Key findings:
- NASDAQ IC: 0.154 (NYSE 0.271 대비 43% 낮음)
- NASDAQ Momentum IC: 0.066 (NYSE 0.252의 26% 수준, 최악의 격차)
- NASDAQ Quality IC: 0.077 (NYSE 0.063 대비 22% 우위, 유일한 강점)
- NASDAQ Growth IC: 0.136 (가장 높음)
- Technology + Healthcare = 46% 차지, 둘 다 IC < 0.10

Strategy:
- Momentum 대폭 축소 (기존 30% -> 15%)
- Quality 강화 (NYSE 대비 우위 활용)
- Growth 중심 (IC 0.136 최고)
- Technology/Healthcare 추가 차별화

Target: NASDAQ IC 0.154 -> 0.195 (+27% improvement)

File: us/us_nasdaq_optimizer.py
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class USNASDAQOptimizer:
    """
    NASDAQ exchange optimization

    Approach:
    1. Momentum 축소 (IC 0.066, NYSE의 26%)
    2. Quality 강화 (IC 0.077, NYSE 대비 22% 우위)
    3. Growth 중심 (IC 0.136, 최고)
    4. Technology/Healthcare 섹터별 차별화
    """

    def __init__(self):
        pass

    def get_nasdaq_base_weights(self) -> Dict[str, float]:
        """
        Get NASDAQ base factor weights

        Data evidence (252d):
        - Growth IC: 0.136 (highest) -> 45% weight
        - Value IC: 0.072 (2nd) -> 20% weight
        - Quality IC: 0.057 (NYSE advantage) -> 20% weight
        - Momentum IC: 0.056 (lowest) -> 15% weight (reduced by 50%)

        CRITICAL: Weight sum MUST be 1.0 to maintain 100-point scale

        Returns:
            Dict[str, float]: Factor weights in 0-1 range, sum = 1.0
            Example: {'growth': 0.45, 'quality': 0.20, 'value': 0.20, 'momentum': 0.15}
        """

        weights = {
            'growth': 0.45,      # IC 0.136 (highest) -> 45%
            'value': 0.20,       # IC 0.072 (2nd) -> 20%
            'quality': 0.20,     # IC 0.057 (NYSE advantage) -> 20%
            'momentum': 0.15     # IC 0.056 (lowest) -> 15% (reduced from 30%)
        }  # Sum: 1.00

        # Validation: Weight sum must be 1.0
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 0.001, \
            f"NASDAQ base weights sum must be 1.0, got {weight_sum}"

        return weights

    def get_nasdaq_sector_weights(self, sector: str) -> Dict[str, float]:
        """
        Get NASDAQ sector-specific factor weights

        Data evidence (us_phase2_nasdaq_by_sector.csv 252d):
        - Technology IC: 0.061 (low) -> Growth 50%, Momentum 10%
        - Healthcare IC: 0.068 (low) -> Growth 50%, Momentum 10%
        - Consumer Defensive IC: 0.189 (high) -> Balanced
        - Financial Services IC: 0.174 (high) -> Balanced

        CRITICAL: Weight sum MUST be 1.0 to maintain 100-point scale

        Args:
            sector: Sector name

        Returns:
            Dict[str, float]: Factor weights in 0-1 range, sum = 1.0
        """

        weights = {
            # Technology: IC 0.061 (low) -> Momentum minimize, Growth maximize
            'Technology': {
                'growth': 0.50,      # Maximize Growth (IC 0.108 in Technology)
                'quality': 0.25,     # Quality important for stability
                'value': 0.15,       # Tech valuation difficult
                'momentum': 0.10     # Minimize (IC 0.062, worst)
            },  # Sum: 1.00

            # Healthcare: IC 0.068 (low) -> Growth centric
            # Note: NASDAQ Healthcare will use Healthcare optimizer (higher priority)
            # This is fallback only
            'Healthcare': {
                'growth': 0.50,      # Growth centric
                'quality': 0.30,     # Quality important
                'value': 0.10,       # Healthcare valuation difficult
                'momentum': 0.10     # Minimize
            },  # Sum: 1.00

            # Consumer Defensive: IC 0.189 (high) -> Balanced
            'Consumer Defensive': {
                'growth': 0.35,      # Moderate Growth
                'quality': 0.25,     # Balanced
                'value': 0.25,       # Value important
                'momentum': 0.15     # Standard NASDAQ weight
            },  # Sum: 1.00

            # Financial Services: IC 0.174 (high) -> Balanced
            'Financial Services': {
                'growth': 0.35,      # Moderate Growth
                'value': 0.25,       # Value important for financials
                'quality': 0.25,     # Quality important
                'momentum': 0.15     # Standard NASDAQ weight
            },  # Sum: 1.00

            # Communication Services: IC 0.137 (medium)
            'Communication Services': {
                'growth': 0.40,      # Growth-centric
                'quality': 0.25,     # Balanced
                'value': 0.20,       # Moderate
                'momentum': 0.15     # Standard NASDAQ weight
            },  # Sum: 1.00

            # Others: Use base weights
            'Others': self.get_nasdaq_base_weights()
        }

        # Select weights for sector
        if sector in weights:
            selected_weights = weights[sector]
        else:
            # Default to base weights
            selected_weights = weights['Others']

        # Validation: Weight sum must be 1.0
        weight_sum = sum(selected_weights.values())
        assert abs(weight_sum - 1.0) < 0.001, \
            f"NASDAQ sector weights sum must be 1.0, got {weight_sum} for sector '{sector}'"

        return selected_weights


# Validation test
if __name__ == '__main__':
    optimizer = USNASDAQOptimizer()

    print("NASDAQ Weight Validation:")
    print("=" * 80)

    # Test base weights
    print("\n[NASDAQ Base Weights]")
    base_weights = optimizer.get_nasdaq_base_weights()
    weight_sum = sum(base_weights.values())
    print(f"  Growth:   {base_weights['growth']:.2f}")
    print(f"  Quality:  {base_weights['quality']:.2f}")
    print(f"  Value:    {base_weights['value']:.2f}")
    print(f"  Momentum: {base_weights['momentum']:.2f}")
    print(f"  Sum:      {weight_sum:.2f} {'OK' if abs(weight_sum - 1.0) < 0.001 else 'FAIL'}")

    # Test sector weights
    sectors = ['Technology', 'Healthcare', 'Consumer Defensive', 'Financial Services', 'Others']

    for sector in sectors:
        weights = optimizer.get_nasdaq_sector_weights(sector)
        weight_sum = sum(weights.values())

        print(f"\n[{sector}]")
        print(f"  Growth:   {weights['growth']:.2f}")
        print(f"  Quality:  {weights['quality']:.2f}")
        print(f"  Value:    {weights['value']:.2f}")
        print(f"  Momentum: {weights['momentum']:.2f}")
        print(f"  Sum:      {weight_sum:.2f} {'OK' if abs(weight_sum - 1.0) < 0.001 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("All weights sum to 1.0 - 100 point scale maintained [OK]")
