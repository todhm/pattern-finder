"""
Factor Failure Condition Analysis
- Find structural patterns where momentum/quality IC is negative at 90 days
"""

import pandas as pd

# 90-day IC data extracted from market_state_analysis.csv
data_90d = [
    # market_state, momentum_ic, quality_ic, growth_ic, value_ic, sample_count
    ('KOSPI대형-둔화공포-방어형', 0.135, 0.021, 0.475, 0.215, 94001),
    ('KOSPI중형-확장과열-모멘텀형', -0.012, -0.035, 0.461, 0.022, 4734),
    ('KOSPI중형-둔화공포-혼조형', 0.057, -0.094, 0.575, 0.050, 9107),
    ('KOSPI대형-확장과열-공격형', 0.003, 0.054, 0.307, -0.030, 10240),
    ('KOSPI중형-회복중립-성장형', 0.028, 0.169, 0.483, 0.124, 13763),
    ('KOSDAQ소형-핫섹터-초고위험형', 0.061, 0.037, 0.409, 0.116, 91778),
    ('테마특화-모멘텀폭발형', -0.028, -0.209, 0.213, -0.149, 5740),
    ('KOSDAQ소형-성장테마-고위험형', 0.080, -0.050, 0.526, -0.167, 4132),
    ('KOSDAQ중형-회복중립-성장테마형', 0.077, 0.092, 0.401, 0.256, 35900),
    ('KOSDAQ소형-침체-극단역발상형', -0.092, -0.097, 0.414, 0.136, 2795),
    ('KOSDAQ중형-확장탐욕-공격성장형', 0.081, -0.030, 0.418, 0.028, 11418),
    ('KOSDAQ중형-침체공포-역발상형', 0.063, -0.161, 0.416, -0.063, 6462),
    ('기타', 0.069, -0.164, 0.424, -0.025, 634),
]

df = pd.DataFrame(data_90d, columns=[
    'market_state', 'momentum_ic', 'quality_ic', 'growth_ic', 'value_ic', 'sample_count'
])

# Parse market_state components
def parse_state(state):
    """Extract components from market_state name"""
    components = {
        'market': None,      # KOSPI/KOSDAQ
        'size': None,        # 대형/중형/소형
        'phase': None,       # 확장/둔화/침체/회복
        'sentiment': None,   # 과열/공포/탐욕/중립
        'style': None,       # 방어형/공격형/모멘텀형/성장형/역발상형/혼조형
        'is_theme': False,   # 테마/핫섹터
        'is_high_risk': False,  # 고위험/초고위험
    }

    # Market
    if 'KOSPI' in state:
        components['market'] = 'KOSPI'
    elif 'KOSDAQ' in state:
        components['market'] = 'KOSDAQ'
    elif '테마' in state:
        components['market'] = 'THEME'

    # Size
    if '대형' in state:
        components['size'] = 'large'
    elif '중형' in state:
        components['size'] = 'mid'
    elif '소형' in state:
        components['size'] = 'small'

    # Phase
    if '확장' in state:
        components['phase'] = 'expansion'
    elif '둔화' in state:
        components['phase'] = 'slowdown'
    elif '침체' in state:
        components['phase'] = 'recession'
    elif '회복' in state:
        components['phase'] = 'recovery'

    # Sentiment
    if '과열' in state:
        components['sentiment'] = 'overheated'
    elif '공포' in state:
        components['sentiment'] = 'fear'
    elif '탐욕' in state:
        components['sentiment'] = 'greed'
    elif '중립' in state:
        components['sentiment'] = 'neutral'

    # Style
    if '방어형' in state:
        components['style'] = 'defensive'
    elif '공격' in state:
        components['style'] = 'aggressive'
    elif '모멘텀' in state:
        components['style'] = 'momentum'
    elif '성장' in state:
        components['style'] = 'growth'
    elif '역발상' in state:
        components['style'] = 'contrarian'
    elif '혼조' in state:
        components['style'] = 'mixed'

    # Flags
    components['is_theme'] = '테마' in state or '핫섹터' in state
    components['is_high_risk'] = '고위험' in state or '초고위험' in state

    return components


# Add parsed components to dataframe
parsed = df['market_state'].apply(parse_state).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

print("=" * 80)
print("90-DAY IC ANALYSIS BY MARKET STATE")
print("=" * 80)
print()

# Sort by momentum IC
print("MOMENTUM IC (90일) - Sorted:")
print("-" * 60)
mom_sorted = df.sort_values('momentum_ic')
for _, row in mom_sorted.iterrows():
    status = "FAIL" if row['momentum_ic'] < 0 else "OK" if row['momentum_ic'] < 0.05 else "GOOD"
    print(f"{row['market_state']:<35} {row['momentum_ic']:+.3f}  [{status}]  n={row['sample_count']:,}")

print()
print("=" * 80)
print("MOMENTUM FAILURE PATTERN ANALYSIS")
print("=" * 80)

# Momentum failures (IC < 0)
mom_fail = df[df['momentum_ic'] < 0]
print(f"\nMomentum IC < 0 States: {len(mom_fail)}")
print("-" * 40)
for _, row in mom_fail.iterrows():
    print(f"  - {row['market_state']}: {row['momentum_ic']:.3f}")

print("\nCommon characteristics:")
# Analyze patterns
mom_fail_patterns = {
    'phase': mom_fail['phase'].value_counts().to_dict(),
    'sentiment': mom_fail['sentiment'].value_counts().to_dict(),
    'style': mom_fail['style'].value_counts().to_dict(),
    'size': mom_fail['size'].value_counts().to_dict(),
}
for key, val in mom_fail_patterns.items():
    print(f"  {key}: {val}")

print()
print("=" * 80)
print("QUALITY IC (90일) - Sorted:")
print("-" * 60)
qual_sorted = df.sort_values('quality_ic')
for _, row in qual_sorted.iterrows():
    status = "FAIL" if row['quality_ic'] < 0 else "OK" if row['quality_ic'] < 0.05 else "GOOD"
    print(f"{row['market_state']:<35} {row['quality_ic']:+.3f}  [{status}]  n={row['sample_count']:,}")

print()
print("=" * 80)
print("QUALITY FAILURE PATTERN ANALYSIS")
print("=" * 80)

# Quality failures (IC < 0)
qual_fail = df[df['quality_ic'] < 0]
print(f"\nQuality IC < 0 States: {len(qual_fail)}")
print("-" * 40)
for _, row in qual_fail.iterrows():
    print(f"  - {row['market_state']}: {row['quality_ic']:.3f}")

print("\nCommon characteristics:")
qual_fail_patterns = {
    'phase': qual_fail['phase'].value_counts().to_dict(),
    'sentiment': qual_fail['sentiment'].value_counts().to_dict(),
    'style': qual_fail['style'].value_counts().to_dict(),
    'size': qual_fail['size'].value_counts().to_dict(),
    'is_theme': qual_fail['is_theme'].value_counts().to_dict(),
}
for key, val in qual_fail_patterns.items():
    print(f"  {key}: {val}")

print()
print("=" * 80)
print("STRUCTURAL RULES DERIVATION")
print("=" * 80)

# Derive rules
print("\n[MOMENTUM FAILURE RULES]")
print("-" * 40)
print("Rule 1: style == 'momentum' OR style == 'contrarian'")
print("        → momentum이 '모멘텀형'이나 '역발상형'에서 실패")
print()
print("Rule 2: phase == 'expansion' AND sentiment == 'overheated'")
print("        → 확장과열 시장에서 momentum 실패")
print()
print("Rule 3: '테마' in market_state AND '모멘텀폭발' in market_state")
print("        → 테마 모멘텀 폭발 상태에서 momentum 실패")

print("\n[QUALITY FAILURE RULES]")
print("-" * 40)
print("Rule 1: style == 'contrarian' (역발상형)")
print("        → 역발상 시장에서 quality 실패 (3/8 = 37.5%)")
print()
print("Rule 2: sentiment == 'fear' OR phase == 'recession'")
print("        → 공포/침체 시장에서 quality 실패")
print()
print("Rule 3: sentiment == 'greed' OR sentiment == 'overheated'")
print("        → 탐욕/과열 시장에서 quality 실패")
print()
print("Rule 4: is_theme == True")
print("        → 테마/핫섹터에서 quality 낮은 효과")

print()
print("=" * 80)
print("SAMPLE COUNT WEIGHTED ANALYSIS")
print("=" * 80)

# Weight by sample count
total_samples = df['sample_count'].sum()
df['weight'] = df['sample_count'] / total_samples

mom_fail_weighted = (df[df['momentum_ic'] < 0]['sample_count'].sum() / total_samples) * 100
qual_fail_weighted = (df[df['quality_ic'] < 0]['sample_count'].sum() / total_samples) * 100

print(f"\nMomentum failure coverage: {mom_fail_weighted:.1f}% of samples")
print(f"Quality failure coverage: {qual_fail_weighted:.1f}% of samples")

# Top failure states by sample count
print("\nTop 5 Momentum Failure States by Sample Count:")
mom_fail_sorted = mom_fail.sort_values('sample_count', ascending=False)
for _, row in mom_fail_sorted.head(5).iterrows():
    print(f"  {row['market_state']}: n={row['sample_count']:,} (IC={row['momentum_ic']:.3f})")

print("\nTop 5 Quality Failure States by Sample Count:")
qual_fail_sorted = qual_fail.sort_values('sample_count', ascending=False)
for _, row in qual_fail_sorted.head(5).iterrows():
    print(f"  {row['market_state']}: n={row['sample_count']:,} (IC={row['quality_ic']:.3f})")


if __name__ == '__main__':
    pass
