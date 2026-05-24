"""
Scenario Probability (_prob) Improvement Analysis

Analyzes both KR and US _prob implementations to identify:
1. Current methodology comparison
2. Weaknesses and gaps
3. Improvement opportunities

Based on academic research:
- Ensemble methods for probability calibration
- Hybrid top-down/bottom-up approaches
- Bayesian probability updating
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class ProbMethod(Enum):
    """Probability estimation methods"""
    MACRO_BASED = "macro_based"           # US current: top-down from macro environment
    BACKTEST_BASED = "backtest_based"     # KR current: bottom-up from historical cases
    HYBRID = "hybrid"                      # Proposed: combine both
    BAYESIAN = "bayesian"                  # Proposed: prior + likelihood update
    ENSEMBLE = "ensemble"                  # Proposed: multiple model average


@dataclass
class MethodAnalysis:
    """Analysis result for each method"""
    method: ProbMethod
    strengths: List[str]
    weaknesses: List[str]
    data_requirements: List[str]
    computational_cost: str
    recommendation: str


def analyze_current_methods():
    """Analyze current KR and US _prob implementations"""

    print("=" * 80)
    print("SCENARIO PROBABILITY (_prob) IMPROVEMENT ANALYSIS")
    print("=" * 80)

    # US Current Implementation Analysis
    print("\n[1] US CURRENT IMPLEMENTATION")
    print("-" * 60)

    us_analysis = MethodAnalysis(
        method=ProbMethod.MACRO_BASED,
        strengths=[
            "Considers market-wide context (GDP, PMI, CPI)",
            "Fast computation (no DB query per stock)",
            "Captures macro regime shifts",
            "Recently improved with empirical data"
        ],
        weaknesses=[
            "Same probability for all stocks in same regime",
            "Only +-10% score adjustment (too narrow)",
            "No sector-specific adjustment",
            "No individual stock characteristics",
            "Limited historical validation period"
        ],
        data_requirements=[
            "GDP, PMI, CPI macro data",
            "Market regime classification"
        ],
        computational_cost="Low (O(1) per stock)",
        recommendation="Add sector/stock-specific adjustment layer"
    )

    print(f"Method: {us_analysis.method.value}")
    print(f"\nStrengths:")
    for s in us_analysis.strengths:
        print(f"  + {s}")
    print(f"\nWeaknesses:")
    for w in us_analysis.weaknesses:
        print(f"  - {w}")
    print(f"\nData Requirements: {', '.join(us_analysis.data_requirements)}")
    print(f"Computational Cost: {us_analysis.computational_cost}")
    print(f"Recommendation: {us_analysis.recommendation}")

    # KR Current Implementation Analysis
    print("\n[2] KR CURRENT IMPLEMENTATION")
    print("-" * 60)

    kr_analysis = MethodAnalysis(
        method=ProbMethod.BACKTEST_BASED,
        strengths=[
            "Data-driven (actual historical outcomes)",
            "Considers similar stocks (industry, score)",
            "Multi-layer calibration (map, regime, theme, floor)",
            "Provides sample count for confidence"
        ],
        weaknesses=[
            "No macro environment consideration",
            "Sample size issues for rare combinations",
            "Calibration multipliers may overfit",
            "Look-ahead bias risk if not careful",
            "Assumes past patterns repeat"
        ],
        data_requirements=[
            "Historical stock grades",
            "60-day forward returns",
            "Industry/theme classification",
            "Market regime for adjustment"
        ],
        computational_cost="Medium (O(n) DB query per stock)",
        recommendation="Add macro context as prior probability"
    )

    print(f"Method: {kr_analysis.method.value}")
    print(f"\nStrengths:")
    for s in kr_analysis.strengths:
        print(f"  + {s}")
    print(f"\nWeaknesses:")
    for w in kr_analysis.weaknesses:
        print(f"  - {w}")
    print(f"\nData Requirements: {', '.join(kr_analysis.data_requirements)}")
    print(f"Computational Cost: {kr_analysis.computational_cost}")
    print(f"Recommendation: {kr_analysis.recommendation}")

    return us_analysis, kr_analysis


def propose_improvements():
    """Propose improvement methods based on research"""

    print("\n" + "=" * 80)
    print("[3] PROPOSED IMPROVEMENT METHODS")
    print("=" * 80)

    improvements = []

    # Method 1: Hybrid Approach
    print("\n--- METHOD 1: HYBRID TOP-DOWN + BOTTOM-UP ---")
    hybrid = {
        'name': 'Hybrid Approach',
        'description': '''
        Combine macro (top-down) and backtest (bottom-up) approaches:

        P(scenario) = w1 * P_macro(scenario | regime)
                    + w2 * P_backtest(scenario | sector, score)

        Where:
        - w1 + w2 = 1
        - Weights can be dynamic based on data quality
        ''',
        'formula': '''
        # Example Implementation
        def hybrid_probability(macro_prob, backtest_prob, sample_count):
            # Weight backtest more if many samples
            w_backtest = min(0.7, sample_count / 100)
            w_macro = 1 - w_backtest

            return w_macro * macro_prob + w_backtest * backtest_prob
        ''',
        'pros': [
            'Best of both worlds',
            'Market context + stock specificity',
            'Graceful degradation (fallback to macro if no samples)'
        ],
        'cons': [
            'Need to tune weights',
            'More complex implementation'
        ],
        'implementation_effort': 'Medium'
    }
    print(f"Description: {hybrid['description']}")
    print(f"Pros: {hybrid['pros']}")
    print(f"Cons: {hybrid['cons']}")
    improvements.append(hybrid)

    # Method 2: Bayesian Updating
    print("\n--- METHOD 2: BAYESIAN UPDATING ---")
    bayesian = {
        'name': 'Bayesian Probability Updating',
        'description': '''
        Use macro probability as PRIOR, update with backtest LIKELIHOOD:

        P(scenario | data) = P(data | scenario) * P(scenario) / P(data)

        Prior: P(scenario) = Macro environment probability
        Likelihood: P(data | scenario) = Historical frequency given similar conditions
        ''',
        'formula': '''
        # Example Implementation
        def bayesian_probability(prior_macro, likelihood_backtest, evidence):
            """
            prior_macro: P(bullish) from MACRO_ENVIRONMENT
            likelihood_backtest: P(observed_features | bullish) from backtest
            evidence: Normalizing constant
            """
            posterior = (likelihood_backtest * prior_macro) / evidence
            return posterior
        ''',
        'pros': [
            'Theoretically sound (Bayes theorem)',
            'Naturally combines prior knowledge with evidence',
            'Handles uncertainty properly',
            'Can incorporate confidence levels'
        ],
        'cons': [
            'Requires likelihood estimation',
            'More complex probability modeling',
            'Need to estimate evidence (marginal likelihood)'
        ],
        'implementation_effort': 'High'
    }
    print(f"Description: {bayesian['description']}")
    print(f"Pros: {bayesian['pros']}")
    print(f"Cons: {bayesian['cons']}")
    improvements.append(bayesian)

    # Method 3: Ensemble Calibration
    print("\n--- METHOD 3: ENSEMBLE CALIBRATION ---")
    ensemble = {
        'name': 'Ensemble Model Calibration',
        'description': '''
        Train multiple probability models, calibrate with isotonic regression:

        Models:
        1. Macro-based model
        2. Backtest-based model
        3. Score-based model (logistic regression)

        Ensemble: Average or weighted vote
        Calibration: Isotonic regression on validation set
        ''',
        'formula': '''
        # Example Implementation
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression

        def ensemble_probability(models, X):
            # Get predictions from all models
            predictions = [model.predict_proba(X) for model in models]

            # Average
            ensemble_pred = np.mean(predictions, axis=0)

            # Calibrate with isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrated = calibrator.fit_transform(ensemble_pred, y_true)

            return calibrated
        ''',
        'pros': [
            'Reduces variance from any single model',
            'Isotonic regression corrects systematic bias',
            'Can add/remove models easily',
            'State-of-the-art calibration technique'
        ],
        'cons': [
            'Requires training data with labels',
            'Need validation set for calibration',
            'More computational overhead'
        ],
        'implementation_effort': 'High'
    }
    print(f"Description: {ensemble['description']}")
    print(f"Pros: {ensemble['pros']}")
    print(f"Cons: {ensemble['cons']}")
    improvements.append(ensemble)

    # Method 4: Dynamic Weight by Confidence
    print("\n--- METHOD 4: DYNAMIC WEIGHT BY CONFIDENCE ---")
    dynamic = {
        'name': 'Confidence-Weighted Combination',
        'description': '''
        Adjust weights dynamically based on each method's confidence:

        Confidence indicators:
        - Macro: VIX level (lower = more confident)
        - Backtest: Sample count (higher = more confident)
        - Score: Factor agreement (higher = more confident)

        Final = sum(weight_i * prob_i) where weight_i ~ confidence_i
        ''',
        'formula': '''
        # Example Implementation
        def confidence_weighted_prob(macro_prob, backtest_prob, score_prob,
                                     vix, sample_count, factor_agreement):
            # Calculate confidence for each source
            macro_conf = max(0.2, 1 - vix / 50)  # Lower VIX = more confident
            backtest_conf = min(0.8, sample_count / 50)  # More samples = more confident
            score_conf = factor_agreement  # 0-1 scale

            # Normalize weights
            total_conf = macro_conf + backtest_conf + score_conf
            w_macro = macro_conf / total_conf
            w_backtest = backtest_conf / total_conf
            w_score = score_conf / total_conf

            return w_macro * macro_prob + w_backtest * backtest_prob + w_score * score_prob
        ''',
        'pros': [
            'Adapts to current market conditions',
            'Uses confidence as natural weight',
            'Interpretable reasoning'
        ],
        'cons': [
            'Need to define confidence metrics',
            'Confidence estimation can be noisy'
        ],
        'implementation_effort': 'Medium'
    }
    print(f"Description: {dynamic['description']}")
    print(f"Pros: {dynamic['pros']}")
    print(f"Cons: {dynamic['cons']}")
    improvements.append(dynamic)

    return improvements


def generate_recommendation():
    """Generate final recommendation"""

    print("\n" + "=" * 80)
    print("[4] FINAL RECOMMENDATION")
    print("=" * 80)

    print("""
    RECOMMENDED APPROACH: HYBRID + CONFIDENCE WEIGHTING
    ====================================================

    Rationale:
    - Combines strengths of both current US and KR approaches
    - Practical to implement (medium effort)
    - Theoretically grounded
    - Adaptable to data availability

    Implementation Plan:

    PHASE 1: Unified Hybrid Model (Both KR and US)
    ----------------------------------------------
    1. Calculate macro_prob from MACRO_ENVIRONMENT (top-down)
    2. Calculate backtest_prob from historical similar cases (bottom-up)
    3. Combine with confidence-based weighting:

       final_prob = w_macro * macro_prob + w_backtest * backtest_prob

       where:
       - w_backtest = min(0.7, sample_count / 50)  # More samples = more trust
       - w_macro = 1 - w_backtest

    PHASE 2: Add Sector-Specific Adjustment
    ---------------------------------------
    - Train sector-specific calibration multipliers
    - Apply after hybrid calculation

    PHASE 3: Add Regime-Adaptive Weighting
    --------------------------------------
    - In high volatility: trust macro more (market-wide moves)
    - In low volatility: trust backtest more (stock-specific patterns)

    EXPECTED IMPROVEMENT:
    - ECE (Expected Calibration Error): -30% to -50%
    - Better alignment between predicted and actual probabilities
    - More consistent across different market conditions
    """)

    # Generate code template
    print("\n" + "-" * 60)
    print("CODE TEMPLATE FOR HYBRID APPROACH:")
    print("-" * 60)

    code_template = '''
async def calculate_hybrid_scenario_probability(
    self,
    symbol: str,
    final_score: float,
    sector: str,
    regime_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Hybrid scenario probability calculation.
    Combines macro (top-down) and backtest (bottom-up) approaches.
    """

    # Step 1: Get macro-based probability (top-down)
    macro_prob = self._get_macro_probability(regime_data)

    # Step 2: Get backtest-based probability (bottom-up)
    backtest_result = await self._get_backtest_probability(final_score, sector)
    backtest_prob = backtest_result['probability']
    sample_count = backtest_result['sample_count']

    # Step 3: Calculate confidence-based weights
    # More samples = more trust in backtest
    w_backtest = min(0.7, sample_count / 50)
    w_macro = 1 - w_backtest

    # Step 4: Combine probabilities
    hybrid_bullish = w_macro * macro_prob['bullish'] + w_backtest * backtest_prob['bullish']
    hybrid_bearish = w_macro * macro_prob['bearish'] + w_backtest * backtest_prob['bearish']
    hybrid_sideways = 100 - hybrid_bullish - hybrid_bearish

    # Step 5: Apply sector-specific calibration (optional)
    calibrated = self._apply_sector_calibration(
        hybrid_bullish, hybrid_sideways, hybrid_bearish, sector
    )

    return {
        'scenario_bullish_prob': round(calibrated['bullish']),
        'scenario_sideways_prob': round(calibrated['sideways']),
        'scenario_bearish_prob': round(calibrated['bearish']),
        'sample_count': sample_count,
        'macro_weight': w_macro,
        'backtest_weight': w_backtest
    }
'''
    print(code_template)


def main():
    """Main analysis function"""

    # Analyze current methods
    us_analysis, kr_analysis = analyze_current_methods()

    # Propose improvements
    improvements = propose_improvements()

    # Generate recommendation
    generate_recommendation()

    print("\n" + "=" * 80)
    print("[OK] Analysis Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
