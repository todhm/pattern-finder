"""
US Quant Deep Problem Analysis V2 - 개선 방안 도출을 위한 근거 분석
====================================================================
목표: 퀀트 전략 수정의 구체적 근거 제공
목적: 정확한 개선 방안 만들기

분석 방향:
1. Decile Anomaly → 어떤 팩터가 저평가를 유발하는지, 가중치 조정 근거
2. Healthcare → Healthcare 팩터 실패 원인, 대안 팩터 제안
3. Stop Loss → 실제 손익 백테스트, 구체적 ATR 배수 값 도출
4. Scenario → 과소추정 원인, calibration 공식 도출
5. EM2 → 구성요소별 문제 분석, 수정 방안
6. NASDAQ → 필요한 팩터와 가중치 조정 근거
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', '').replace('+asyncpg', '')
OUTPUT_DIR = r"C:\project\alpha\quant\us\result"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PROBLEM 1: Decile Anomaly - 팩터 저평가 원인 분석
# 목표: 어떤 팩터가 고수익 종목을 저평가하는지 파악 → 가중치 조정 근거
# =============================================================================

async def analyze_decile_anomaly_v2(pool: asyncpg.Pool) -> Dict:
    """
    D1~D5 저점수 + 고수익 종목 분석

    분석 내용:
    1. 각 팩터별 저평가 기여도 분석 (어떤 팩터가 점수를 깎았는지)
    2. 고수익 종목의 실제 재무/성장 특성
    3. 팩터별 수익률 예측력 비교 (어떤 팩터를 더 신뢰해야 하는지)
    4. 개선 방안: 팩터 가중치 조정 제안
    """
    print("\n" + "=" * 80)
    print("[PROBLEM 1] Decile Anomaly - 팩터 저평가 원인 분석")
    print("목표: 어떤 팩터가 고수익 종목을 저평가하는지 → 가중치 조정 근거")
    print("=" * 80)

    results = {
        'factor_undervaluation_analysis': {},
        'factor_ic_comparison': {},
        'weight_adjustment_recommendations': {},
        'detailed_findings': []
    }

    async with pool.acquire() as conn:
        # Query 1: 저점수 + 고수익 종목 상세 데이터
        query = """
        WITH scored_stocks AS (
            SELECT
                sg.symbol,
                sg.date,
                sg.final_score,
                sg.value_score,
                sg.quality_score,
                sg.momentum_score,
                sg.growth_score,
                sb.sector,
                sb.industry,
                sb.market_cap,
                NTILE(10) OVER (PARTITION BY sg.date ORDER BY sg.final_score) as decile
            FROM us_stock_grade sg
            JOIN us_stock_basic sb ON sg.symbol = sb.symbol
            WHERE sg.final_score IS NOT NULL
        ),
        with_returns AS (
            SELECT
                s.*,
                d_future.close / NULLIF(d_current.close, 0) - 1 as return_30d,
                d_future90.close / NULLIF(d_current.close, 0) - 1 as return_90d
            FROM scored_stocks s
            LEFT JOIN us_daily d_current ON s.symbol = d_current.symbol AND s.date = d_current.date
            LEFT JOIN us_daily d_future ON s.symbol = d_future.symbol AND d_future.date = s.date + INTERVAL '30 days'
            LEFT JOIN us_daily d_future90 ON s.symbol = d_future90.symbol AND d_future90.date = s.date + INTERVAL '90 days'
        ),
        -- 정상 종목: D8-D10 고점수 + 고수익
        normal_high_return AS (
            SELECT * FROM with_returns
            WHERE decile >= 8 AND (return_30d > 0.1 OR return_90d > 0.2)
        ),
        -- 이상 종목: D1-D5 저점수 + 고수익
        anomaly AS (
            SELECT * FROM with_returns
            WHERE decile <= 5 AND (return_30d > 0.1 OR return_90d > 0.2)
        )
        SELECT
            'anomaly' as group_type,
            a.symbol, a.date, a.decile, a.final_score,
            a.value_score, a.quality_score, a.momentum_score, a.growth_score,
            a.sector, a.industry, a.market_cap, a.return_30d, a.return_90d
        FROM anomaly a
        UNION ALL
        SELECT
            'normal' as group_type,
            n.symbol, n.date, n.decile, n.final_score,
            n.value_score, n.quality_score, n.momentum_score, n.growth_score,
            n.sector, n.industry, n.market_cap, n.return_30d, n.return_90d
        FROM normal_high_return n
        LIMIT 2000
        """

        rows = await conn.fetch(query)
        if not rows:
            print("  데이터 없음")
            return results

        df = pd.DataFrame([dict(r) for r in rows])

        # Decimal to float 변환
        numeric_cols = ['final_score', 'value_score', 'quality_score', 'momentum_score',
                        'growth_score', 'market_cap', 'return_30d', 'return_90d']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        anomaly_df = df[df['group_type'] == 'anomaly']
        normal_df = df[df['group_type'] == 'normal']

        print(f"\n  분석 대상: 이상 종목 {len(anomaly_df)}개, 정상 고수익 종목 {len(normal_df)}개")

        # =====================================================================
        # 분석 1: 팩터별 저평가 기여도 분석
        # =====================================================================
        print("\n  [분석 1] 팩터별 평균 점수 비교 (이상 종목 vs 정상 고수익 종목)")
        print("  " + "-" * 70)
        print(f"  {'팩터':<20} {'이상 종목':<12} {'정상 종목':<12} {'차이':<10} {'저평가 원인':<15}")
        print("  " + "-" * 70)

        factors = ['value_score', 'quality_score', 'momentum_score', 'growth_score']
        factor_gaps = {}

        for factor in factors:
            if factor not in df.columns:
                continue
            anomaly_mean = anomaly_df[factor].mean()
            normal_mean = normal_df[factor].mean()
            gap = anomaly_mean - normal_mean
            factor_gaps[factor] = gap

            # 저평가 원인 판단
            if gap < -0.15:
                cause = "★★★ 주요 원인"
            elif gap < -0.05:
                cause = "★★ 부분 원인"
            else:
                cause = "정상 범위"

            print(f"  {factor:<20} {anomaly_mean:.3f}        {normal_mean:.3f}        {gap:+.3f}     {cause}")

            results['factor_undervaluation_analysis'][factor] = {
                'anomaly_mean': float(anomaly_mean),
                'normal_mean': float(normal_mean),
                'gap': float(gap),
                'is_undervaluation_cause': gap < -0.05
            }

        # 가장 문제가 되는 팩터 식별
        worst_factor = min(factor_gaps.items(), key=lambda x: x[1])
        print(f"\n  → 가장 심한 저평가 팩터: {worst_factor[0]} (gap: {worst_factor[1]:+.3f})")

        # =====================================================================
        # 분석 2: 팩터별 실제 수익률 예측력 비교
        # =====================================================================
        print("\n  [분석 2] 팩터별 IC (수익률 예측력) - 어떤 팩터를 신뢰해야 하는지")
        print("  " + "-" * 70)
        print(f"  {'팩터':<20} {'전체 IC':<12} {'이상 종목 IC':<15} {'신뢰도 판정':<15}")
        print("  " + "-" * 70)

        for factor in factors:
            if factor not in df.columns:
                continue

            # 전체 IC
            valid_all = df[[factor, 'return_30d']].dropna()
            ic_all = stats.spearmanr(valid_all[factor], valid_all['return_30d'])[0] if len(valid_all) > 30 else 0

            # 이상 종목만의 IC
            valid_anomaly = anomaly_df[[factor, 'return_30d']].dropna()
            ic_anomaly = stats.spearmanr(valid_anomaly[factor], valid_anomaly['return_30d'])[0] if len(valid_anomaly) > 30 else 0

            # 신뢰도 판정
            if ic_all > 0.05:
                trust = "신뢰 가능"
            elif ic_all > 0:
                trust = "약한 예측력"
            else:
                trust = "★ 역방향 (문제)"

            print(f"  {factor:<20} {ic_all:.4f}      {ic_anomaly:.4f}          {trust}")

            results['factor_ic_comparison'][factor] = {
                'ic_all': float(ic_all),
                'ic_anomaly': float(ic_anomaly),
                'trustworthy': ic_all > 0.03
            }

        # =====================================================================
        # 분석 3: 이상 종목의 공통 특성 (개선 방안 힌트)
        # =====================================================================
        print("\n  [분석 3] 이상 종목의 공통 특성 분석")
        print("  " + "-" * 70)

        if 'sector' in anomaly_df.columns:
            sector_dist = anomaly_df['sector'].value_counts()
            total_sector = sector_dist.sum()
            print(f"  섹터 분포 (이상 종목):")
            for sector, count in sector_dist.head(5).items():
                pct = count / total_sector * 100
                print(f"    {sector}: {count}개 ({pct:.1f}%)")

            results['detailed_findings'].append({
                'type': 'sector_concentration',
                'description': f"이상 종목의 {sector_dist.index[0]} 집중: {sector_dist.iloc[0]/total_sector*100:.1f}%"
            })

        if 'market_cap' in anomaly_df.columns and anomaly_df['market_cap'].notna().any():
            anomaly_cap = anomaly_df['market_cap'].median()
            normal_cap = normal_df['market_cap'].median() if 'market_cap' in normal_df.columns else 0
            print(f"\n  시가총액 중앙값:")
            print(f"    이상 종목: ${anomaly_cap/1e9:.2f}B")
            print(f"    정상 종목: ${normal_cap/1e9:.2f}B")

            results['detailed_findings'].append({
                'type': 'market_cap',
                'anomaly_median': float(anomaly_cap),
                'normal_median': float(normal_cap)
            })

        # =====================================================================
        # 개선 방안 도출
        # =====================================================================
        print("\n  [개선 방안 도출]")
        print("  " + "=" * 70)

        recommendations = []

        # 저평가 원인 팩터에 대한 가중치 조정 제안
        for factor, data in results['factor_undervaluation_analysis'].items():
            if data['is_undervaluation_cause']:
                ic_data = results['factor_ic_comparison'].get(factor, {})
                ic = ic_data.get('ic_all', 0)

                if ic < 0.02:
                    rec = f"{factor}: 가중치 50% 감소 권장 (저평가 원인 + 낮은 예측력)"
                    weight_adj = 0.5
                else:
                    rec = f"{factor}: 가중치 20% 감소 권장 (저평가 원인이나 예측력은 있음)"
                    weight_adj = 0.8

                recommendations.append(rec)
                results['weight_adjustment_recommendations'][factor] = {
                    'current_issue': '저평가 유발',
                    'ic': ic,
                    'recommended_weight_multiplier': weight_adj,
                    'reasoning': rec
                }
                print(f"  → {rec}")

        # 예측력 높은 팩터 가중치 증가 제안
        for factor, data in results['factor_ic_comparison'].items():
            if data['ic_all'] > 0.05 and not results['factor_undervaluation_analysis'].get(factor, {}).get('is_undervaluation_cause', False):
                rec = f"{factor}: 가중치 30% 증가 권장 (높은 예측력, IC={data['ic_all']:.3f})"
                recommendations.append(rec)
                results['weight_adjustment_recommendations'][factor] = {
                    'current_issue': None,
                    'ic': data['ic_all'],
                    'recommended_weight_multiplier': 1.3,
                    'reasoning': rec
                }
                print(f"  → {rec}")

        results['recommendations'] = recommendations
        print(f"\n  총 {len(anomaly_df)}개 이상 종목 분석 완료")

    return results


# =============================================================================
# PROBLEM 2: Healthcare 섹터 - 팩터 실패 원인 분석
# 목표: Healthcare에서 왜 팩터가 실패하는지 → 대안 팩터 제안
# =============================================================================

async def analyze_healthcare_v2(pool: asyncpg.Pool) -> Dict:
    """
    Healthcare 섹터 팩터 실패 원인 분석

    분석 내용:
    1. Healthcare vs 비Healthcare 팩터 IC 비교
    2. Healthcare 하위 산업별 팩터 효과 분석
    3. Healthcare 특성 (임상시험, R&D 등) 반영 필요성
    4. 개선 방안: Healthcare 특화 팩터 가중치, 추가 팩터 제안
    """
    print("\n" + "=" * 80)
    print("[PROBLEM 2] Healthcare 섹터 - 팩터 실패 원인 분석")
    print("목표: Healthcare에서 왜 팩터가 실패하는지 → 대안 팩터 제안")
    print("=" * 80)

    results = {
        'sector_ic_comparison': {},
        'industry_analysis': {},
        'factor_failure_causes': {},
        'alternative_factors': {},
        'recommendations': []
    }

    async with pool.acquire() as conn:
        # Query: 섹터별 팩터 데이터
        query = """
        SELECT
            sg.symbol,
            sg.date,
            sg.final_score,
            sg.value_score,
            sg.quality_score,
            sg.momentum_score,
            sg.growth_score,
            sb.sector,
            sb.industry,
            sb.market_cap,
            d_future.close / NULLIF(d_current.close, 0) - 1 as return_30d,
            d_future90.close / NULLIF(d_current.close, 0) - 1 as return_90d
        FROM us_stock_grade sg
        JOIN us_stock_basic sb ON sg.symbol = sb.symbol
        LEFT JOIN us_daily d_current ON sg.symbol = d_current.symbol AND sg.date = d_current.date
        LEFT JOIN us_daily d_future ON sg.symbol = d_future.symbol AND d_future.date = sg.date + INTERVAL '30 days'
        LEFT JOIN us_daily d_future90 ON sg.symbol = d_future90.symbol AND d_future90.date = sg.date + INTERVAL '90 days'
        WHERE sg.final_score IS NOT NULL
        """

        rows = await conn.fetch(query)
        if not rows:
            print("  데이터 없음")
            return results

        df = pd.DataFrame([dict(r) for r in rows])

        numeric_cols = ['final_score', 'value_score', 'quality_score', 'momentum_score',
                        'growth_score', 'market_cap', 'return_30d', 'return_90d']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        healthcare_df = df[df['sector'] == 'HEALTHCARE']
        other_df = df[df['sector'] != 'HEALTHCARE']

        print(f"\n  Healthcare: {len(healthcare_df)}개, 비Healthcare: {len(other_df)}개")

        # =====================================================================
        # 분석 1: Healthcare vs 비Healthcare 팩터 IC 비교
        # =====================================================================
        print("\n  [분석 1] Healthcare vs 비Healthcare 팩터 IC 비교")
        print("  " + "-" * 70)
        print(f"  {'팩터':<20} {'Healthcare IC':<15} {'Other IC':<12} {'차이':<10} {'진단':<15}")
        print("  " + "-" * 70)

        factors = ['value_score', 'quality_score', 'momentum_score', 'growth_score', 'final_score']

        for factor in factors:
            if factor not in df.columns:
                continue

            # Healthcare IC
            valid_hc = healthcare_df[[factor, 'return_30d']].dropna()
            ic_hc = stats.spearmanr(valid_hc[factor], valid_hc['return_30d'])[0] if len(valid_hc) > 50 else np.nan

            # 비Healthcare IC
            valid_other = other_df[[factor, 'return_30d']].dropna()
            ic_other = stats.spearmanr(valid_other[factor], valid_other['return_30d'])[0] if len(valid_other) > 50 else np.nan

            gap = ic_hc - ic_other if pd.notna(ic_hc) and pd.notna(ic_other) else np.nan

            # 진단
            if pd.isna(ic_hc):
                diagnosis = "데이터 부족"
            elif ic_hc < 0:
                diagnosis = "★★★ 역방향 (심각)"
            elif ic_hc < ic_other * 0.5:
                diagnosis = "★★ 효과 미약"
            else:
                diagnosis = "정상"

            ic_hc_str = f"{ic_hc:.4f}" if pd.notna(ic_hc) else "N/A"
            ic_other_str = f"{ic_other:.4f}" if pd.notna(ic_other) else "N/A"
            gap_str = f"{gap:+.4f}" if pd.notna(gap) else "N/A"

            print(f"  {factor:<20} {ic_hc_str:<15} {ic_other_str:<12} {gap_str:<10} {diagnosis}")

            results['sector_ic_comparison'][factor] = {
                'healthcare_ic': float(ic_hc) if pd.notna(ic_hc) else None,
                'other_ic': float(ic_other) if pd.notna(ic_other) else None,
                'gap': float(gap) if pd.notna(gap) else None,
                'diagnosis': diagnosis
            }

        # =====================================================================
        # 분석 2: Healthcare 하위 산업별 분석
        # =====================================================================
        print("\n  [분석 2] Healthcare 하위 산업별 팩터 효과")
        print("  " + "-" * 70)

        if 'industry' in healthcare_df.columns:
            industries = healthcare_df['industry'].value_counts().head(8).index.tolist()

            print(f"  {'산업':<30} {'Final IC':<12} {'Value IC':<12} {'샘플수':<8}")
            print("  " + "-" * 70)

            for industry in industries:
                ind_df = healthcare_df[healthcare_df['industry'] == industry]
                if len(ind_df) < 30:
                    continue

                valid = ind_df[['final_score', 'return_30d']].dropna()
                valid_v = ind_df[['value_score', 'return_30d']].dropna()

                ic_final = stats.spearmanr(valid['final_score'], valid['return_30d'])[0] if len(valid) > 20 else np.nan
                ic_value = stats.spearmanr(valid_v['value_score'], valid_v['return_30d'])[0] if len(valid_v) > 20 else np.nan

                ic_final_str = f"{ic_final:.4f}" if pd.notna(ic_final) else "N/A"
                ic_value_str = f"{ic_value:.4f}" if pd.notna(ic_value) else "N/A"

                print(f"  {industry[:28]:<30} {ic_final_str:<12} {ic_value_str:<12} {len(ind_df):<8}")

                results['industry_analysis'][industry] = {
                    'final_ic': float(ic_final) if pd.notna(ic_final) else None,
                    'value_ic': float(ic_value) if pd.notna(ic_value) else None,
                    'n_samples': len(ind_df)
                }

        # =====================================================================
        # 분석 3: Healthcare 팩터 실패 원인 심층 분석
        # =====================================================================
        print("\n  [분석 3] Healthcare 팩터 실패 원인")
        print("  " + "-" * 70)

        # Value Score 실패 원인 분석
        print("\n  Value Score 실패 원인:")
        if 'value_score' in healthcare_df.columns:
            # 저 Value Score + 고수익 종목 특성
            low_value_high_return = healthcare_df[
                (healthcare_df['value_score'] < healthcare_df['value_score'].quantile(0.3)) &
                (healthcare_df['return_30d'] > healthcare_df['return_30d'].quantile(0.7))
            ]

            if len(low_value_high_return) > 0:
                if 'industry' in low_value_high_return.columns:
                    ind_dist = low_value_high_return['industry'].value_counts()
                    print(f"    - 저 Value + 고수익 종목 산업: {ind_dist.index[0] if len(ind_dist) > 0 else 'N/A'}")

                if 'market_cap' in low_value_high_return.columns:
                    med_cap = low_value_high_return['market_cap'].median()
                    print(f"    - 시가총액 중앙값: ${med_cap/1e9:.2f}B" if pd.notna(med_cap) else "    - 시가총액: N/A")

                results['factor_failure_causes']['value_score'] = {
                    'issue': 'Biotech/소형주에서 전통적 밸류에이션 지표가 무의미',
                    'affected_stocks': len(low_value_high_return),
                    'recommendation': 'Pipeline Value, R&D 진행률 팩터 추가 필요'
                }
                print(f"    → 원인: Biotech/소형주에서 전통적 PER, PBR 등이 무의미")
                print(f"    → 대안: Pipeline Value, R&D 효율성 팩터 필요")

        # Quality Score 실패 원인 분석
        print("\n  Quality Score 실패 원인:")
        if 'quality_score' in healthcare_df.columns:
            quality_ic = results['sector_ic_comparison'].get('quality_score', {}).get('healthcare_ic', 0)
            if quality_ic is not None and quality_ic < 0.02:
                results['factor_failure_causes']['quality_score'] = {
                    'issue': 'Biotech 기업은 수익성 지표가 음수인 경우가 많음',
                    'recommendation': '현금 소진율, 임상 단계별 성공률 팩터 추가'
                }
                print(f"    → 원인: Biotech 기업은 대부분 적자, ROE/ROA가 무의미")
                print(f"    → 대안: 현금 runway, 임상 성공률, 파트너십 보유 여부")

        # =====================================================================
        # 개선 방안 도출
        # =====================================================================
        print("\n  [개선 방안 도출]")
        print("  " + "=" * 70)

        recommendations = []

        # Healthcare 특화 가중치 조정
        for factor, data in results['sector_ic_comparison'].items():
            if data.get('healthcare_ic') is not None and data.get('other_ic') is not None:
                if data['healthcare_ic'] < 0:
                    rec = f"Healthcare {factor}: 가중치 0 또는 역방향 적용 (IC={data['healthcare_ic']:.3f})"
                    recommendations.append(rec)
                    print(f"  → {rec}")
                elif data['healthcare_ic'] < data['other_ic'] * 0.5:
                    rec = f"Healthcare {factor}: 가중치 50% 감소 (다른 섹터 대비 효과 미약)"
                    recommendations.append(rec)
                    print(f"  → {rec}")

        # 대안 팩터 제안
        print("\n  Healthcare 대안 팩터 제안:")
        alternative_factors = [
            "Pipeline Value: 임상 단계별 성공 확률 × 시장 규모",
            "Cash Runway: 현금 / 분기 소진율",
            "Clinical Momentum: 최근 임상 진행 이벤트",
            "Partnership Score: 대형 제약사 파트너십 보유 여부",
            "FDA Calendar: FDA 심사 일정 근접도"
        ]
        for alt in alternative_factors:
            print(f"    - {alt}")

        results['alternative_factors'] = alternative_factors
        results['recommendations'] = recommendations

        print(f"\n  Healthcare {len(healthcare_df)}개 종목 분석 완료")

    return results


# =============================================================================
# PROBLEM 3: Stop Loss ATR - 구체적 배수 도출
# 목표: 실제 손익 백테스트로 최적 ATR 배수 결정
# =============================================================================

async def analyze_stop_loss_v2(pool: asyncpg.Pool) -> Dict:
    """
    Stop Loss ATR 배수 최적화

    분석 내용:
    1. ATR 배수별 실제 손익 백테스트
    2. 손절 후 가격 회복 분석 (너무 빠른 손절인지)
    3. 섹터/변동성별 최적 ATR 배수
    4. 개선 방안: 구체적 ATR 배수 값 도출
    """
    print("\n" + "=" * 80)
    print("[PROBLEM 3] Stop Loss ATR - 구체적 배수 도출")
    print("목표: 실제 손익 백테스트로 최적 ATR 배수 결정")
    print("=" * 80)

    results = {
        'backtest_by_multiplier': {},
        'recovery_analysis': {},
        'sector_optimal_atr': {},
        'final_recommendation': {}
    }

    async with pool.acquire() as conn:
        # Query: ATR 데이터와 향후 가격 이동
        # Window function 중첩 방지를 위해 단계별로 계산
        query = """
        WITH price_with_prev AS (
            -- Step 1: 전일 종가 계산
            SELECT
                d.symbol,
                d.date,
                d.close,
                d.high,
                d.low,
                sb.sector,
                LAG(d.close) OVER (PARTITION BY d.symbol ORDER BY d.date) as prev_close
            FROM us_daily d
            JOIN us_stock_basic sb ON d.symbol = sb.symbol
            WHERE d.date >= CURRENT_DATE - INTERVAL '18 months'
        ),
        price_with_tr AS (
            -- Step 2: True Range 계산
            SELECT
                symbol,
                date,
                close,
                high,
                low,
                sector,
                GREATEST(
                    high - low,
                    ABS(high - COALESCE(prev_close, close)),
                    ABS(low - COALESCE(prev_close, close))
                ) as true_range
            FROM price_with_prev
        ),
        price_with_atr AS (
            -- Step 3: ATR 14일 평균 계산
            SELECT
                symbol,
                date,
                close,
                sector,
                AVG(true_range) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) as atr_14
            FROM price_with_tr
        ),
        with_future AS (
            SELECT
                p.symbol,
                p.date as entry_date,
                p.close as entry_price,
                p.atr_14,
                p.sector,
                -- 30일간 최저가
                (SELECT MIN(d2.low) FROM us_daily d2
                 WHERE d2.symbol = p.symbol
                 AND d2.date > p.date
                 AND d2.date <= p.date + INTERVAL '30 days') as min_low_30d,
                -- 30일 후 종가
                (SELECT d3.close FROM us_daily d3
                 WHERE d3.symbol = p.symbol
                 AND d3.date = p.date + INTERVAL '30 days') as close_30d,
                -- 60일 후 종가
                (SELECT d4.close FROM us_daily d4
                 WHERE d4.symbol = p.symbol
                 AND d4.date = p.date + INTERVAL '60 days') as close_60d
            FROM price_with_atr p
            WHERE p.atr_14 > 0
              AND p.date <= CURRENT_DATE - INTERVAL '60 days'
        )
        SELECT * FROM with_future
        WHERE atr_14 IS NOT NULL
          AND min_low_30d IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10000
        """

        rows = await conn.fetch(query)
        if not rows:
            print("  데이터 없음")
            return results

        df = pd.DataFrame([dict(r) for r in rows])

        numeric_cols = ['entry_price', 'atr_14', 'min_low_30d', 'close_30d', 'close_60d']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['entry_price', 'atr_14', 'min_low_30d'])
        print(f"\n  분석 대상: {len(df)}개 거래")

        # =====================================================================
        # 분석 1: ATR 배수별 실제 손익 백테스트
        # =====================================================================
        print("\n  [분석 1] ATR 배수별 실제 손익 백테스트")
        print("  " + "-" * 75)
        print(f"  {'ATR배수':<10} {'손절 발동률':<12} {'평균 손익':<12} {'승률':<10} {'손익비':<10} {'종합점수':<10}")
        print("  " + "-" * 75)

        multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        best_score = -float('inf')
        best_mult = None

        for mult in multipliers:
            # 손절가 계산
            df['stop_price'] = df['entry_price'] - df['atr_14'] * mult

            # 손절 발동 여부
            df['stop_hit'] = df['min_low_30d'] <= df['stop_price']

            # 손익 계산
            # 손절 발동 시: 손절가에서 청산
            # 손절 미발동 시: 30일 후 종가로 청산
            df['pnl'] = np.where(
                df['stop_hit'],
                (df['stop_price'] / df['entry_price'] - 1),  # 손절 시 손익
                (df['close_30d'] / df['entry_price'] - 1) if 'close_30d' in df.columns else 0  # 보유 시 손익
            )

            trigger_rate = df['stop_hit'].mean()
            avg_pnl = df['pnl'].mean()
            win_rate = (df['pnl'] > 0).mean()

            # 손익비 계산
            winners = df[df['pnl'] > 0]['pnl']
            losers = df[df['pnl'] < 0]['pnl']
            profit_ratio = abs(winners.mean() / losers.mean()) if len(losers) > 0 and losers.mean() != 0 else 0

            # 종합 점수: 평균 손익 × (1 + 승률) × 손익비
            score = avg_pnl * (1 + win_rate) * max(profit_ratio, 0.1)

            if score > best_score:
                best_score = score
                best_mult = mult

            print(f"  {mult}x        {trigger_rate*100:.1f}%         {avg_pnl*100:+.2f}%       {win_rate*100:.1f}%     {profit_ratio:.2f}      {score:.4f}")

            results['backtest_by_multiplier'][mult] = {
                'trigger_rate': float(trigger_rate),
                'avg_pnl': float(avg_pnl),
                'win_rate': float(win_rate),
                'profit_ratio': float(profit_ratio),
                'score': float(score)
            }

        print(f"\n  → 최적 ATR 배수: {best_mult}x (종합점수 기준)")

        # =====================================================================
        # 분석 2: 손절 후 가격 회복 분석
        # =====================================================================
        print("\n  [분석 2] 손절 후 가격 회복 분석 (너무 빠른 손절인지)")
        print("  " + "-" * 70)

        # 2.0x ATR 기준으로 손절 후 60일 후 가격 분석
        df['stop_price_2x'] = df['entry_price'] - df['atr_14'] * 2.0
        df['stop_hit_2x'] = df['min_low_30d'] <= df['stop_price_2x']

        stop_hit_df = df[df['stop_hit_2x'] & df['close_60d'].notna()]
        if len(stop_hit_df) > 0:
            # 손절 후 60일 뒤 가격이 진입가보다 높은 비율 (= 너무 빨리 손절함)
            recovery_rate = (stop_hit_df['close_60d'] > stop_hit_df['entry_price']).mean()
            avg_recovery = (stop_hit_df['close_60d'] / stop_hit_df['entry_price'] - 1).mean()

            print(f"  2.0x ATR 손절 발동 후:")
            print(f"    - 60일 후 진입가 회복 비율: {recovery_rate*100:.1f}%")
            print(f"    - 60일 후 평균 수익률: {avg_recovery*100:+.1f}%")

            if recovery_rate > 0.4:
                print(f"    → 경고: 회복률이 높음 - ATR 배수를 늘려야 함")

            results['recovery_analysis'] = {
                'multiplier_tested': 2.0,
                'recovery_rate': float(recovery_rate),
                'avg_recovery_return': float(avg_recovery),
                'recommendation': 'ATR 배수 증가 필요' if recovery_rate > 0.4 else '적정 수준'
            }

        # =====================================================================
        # 분석 3: 섹터별 최적 ATR 배수
        # =====================================================================
        print("\n  [분석 3] 섹터별 최적 ATR 배수")
        print("  " + "-" * 60)
        print(f"  {'섹터':<25} {'최적 ATR':<12} {'평균 손익':<12} {'샘플수':<8}")
        print("  " + "-" * 60)

        if 'sector' in df.columns:
            for sector in df['sector'].dropna().unique()[:10]:
                sector_df = df[df['sector'] == sector]
                if len(sector_df) < 100:
                    continue

                best_sector_mult = None
                best_sector_pnl = -float('inf')

                for mult in [1.5, 2.0, 2.5, 3.0, 3.5]:
                    sector_df['stop_price'] = sector_df['entry_price'] - sector_df['atr_14'] * mult
                    sector_df['stop_hit'] = sector_df['min_low_30d'] <= sector_df['stop_price']
                    sector_df['pnl'] = np.where(
                        sector_df['stop_hit'],
                        (sector_df['stop_price'] / sector_df['entry_price'] - 1),
                        (sector_df['close_30d'] / sector_df['entry_price'] - 1)
                    )
                    avg_pnl = sector_df['pnl'].mean()

                    if avg_pnl > best_sector_pnl:
                        best_sector_pnl = avg_pnl
                        best_sector_mult = mult

                print(f"  {sector[:23]:<25} {best_sector_mult}x         {best_sector_pnl*100:+.2f}%       {len(sector_df)}")

                results['sector_optimal_atr'][sector] = {
                    'optimal_multiplier': best_sector_mult,
                    'expected_pnl': float(best_sector_pnl),
                    'n_samples': len(sector_df)
                }

        # =====================================================================
        # 최종 권고사항
        # =====================================================================
        print("\n  [최종 권고사항]")
        print("  " + "=" * 70)

        results['final_recommendation'] = {
            'global_optimal_atr': best_mult,
            'reasoning': f"백테스트 결과 {best_mult}x ATR이 최고 종합점수",
            'implementation': f"stop_loss_pct = entry_price - ATR_14 * {best_mult}"
        }

        print(f"  → 전역 최적 ATR 배수: {best_mult}x")
        print(f"  → 구현: stop_loss = entry_price - ATR_14 × {best_mult}")

        # 섹터별 다른 값 적용 권고
        if results['sector_optimal_atr']:
            sector_atr_values = [v['optimal_multiplier'] for v in results['sector_optimal_atr'].values()]
            if max(sector_atr_values) - min(sector_atr_values) > 1.0:
                print(f"  → 섹터별 ATR 배수 차이가 큼 - 섹터별 동적 적용 권장")
                results['final_recommendation']['sector_specific'] = True

    return results


# =============================================================================
# PROBLEM 4: Scenario Calibration - 보정 공식 도출
# 목표: 과소/과대 추정 패턴 파악 → calibration 공식 도출
# =============================================================================

async def analyze_scenario_calibration_v2(pool: asyncpg.Pool) -> Dict:
    """
    Scenario 확률 Calibration 분석

    분석 내용:
    1. 예측 확률 구간별 실제 발생률 비교
    2. 과소/과대 추정 패턴 파악
    3. Calibration 보정 공식 도출
    4. 개선 방안: 구체적 보정 파라미터
    """
    print("\n" + "=" * 80)
    print("[PROBLEM 4] Scenario Calibration - 보정 공식 도출")
    print("목표: 과소/과대 추정 패턴 파악 → calibration 공식 도출")
    print("=" * 80)

    results = {
        'calibration_curve': {},
        'calibration_formula': {},
        'scenario_bias': {},
        'recommendations': []
    }

    async with pool.acquire() as conn:
        query = """
        SELECT
            sg.symbol,
            sg.date,
            sg.scenario_bullish_prob,
            sg.scenario_sideways_prob,
            sg.scenario_bearish_prob,
            sb.sector,
            CASE
                WHEN d_future.close / NULLIF(d_current.close, 0) - 1 > 0.15 THEN 'bull'
                WHEN d_future.close / NULLIF(d_current.close, 0) - 1 < -0.10 THEN 'bear'
                ELSE 'base'
            END as actual_scenario,
            d_future.close / NULLIF(d_current.close, 0) - 1 as return_90d
        FROM us_stock_grade sg
        JOIN us_stock_basic sb ON sg.symbol = sb.symbol
        LEFT JOIN us_daily d_current ON sg.symbol = d_current.symbol AND sg.date = d_current.date
        LEFT JOIN us_daily d_future ON sg.symbol = d_future.symbol AND d_future.date = sg.date + INTERVAL '90 days'
        WHERE sg.scenario_bullish_prob IS NOT NULL
          AND sg.scenario_bearish_prob IS NOT NULL
          AND d_future.close IS NOT NULL
        """

        rows = await conn.fetch(query)
        if not rows:
            print("  데이터 없음")
            return results

        df = pd.DataFrame([dict(r) for r in rows])

        numeric_cols = ['scenario_bullish_prob', 'scenario_sideways_prob',
                        'scenario_bearish_prob', 'return_90d']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 정수 확률을 소수로 변환 (만약 0-100 범위라면)
        for col in ['scenario_bullish_prob', 'scenario_sideways_prob', 'scenario_bearish_prob']:
            if df[col].max() > 1:
                df[col] = df[col] / 100.0

        print(f"\n  분석 대상: {len(df)}개 예측")

        # =====================================================================
        # 분석 1: Calibration Curve (예측 확률 vs 실제 발생률)
        # =====================================================================
        print("\n  [분석 1] Calibration Curve - 예측 확률 vs 실제 발생률")
        print("  " + "-" * 75)

        scenarios = [
            ('bull', 'scenario_bullish_prob'),
            ('base', 'scenario_sideways_prob'),
            ('bear', 'scenario_bearish_prob')
        ]

        for scenario_name, prob_col in scenarios:
            print(f"\n  {scenario_name.upper()} 시나리오:")
            print(f"  {'예측 구간':<15} {'예측 평균':<12} {'실제 발생률':<12} {'차이':<10} {'보정 계수':<12}")
            print("  " + "-" * 65)

            results['calibration_curve'][scenario_name] = {}

            if prob_col not in df.columns:
                continue

            # 확률 구간 생성
            bins = [0, 0.15, 0.25, 0.35, 0.50, 0.70, 1.0]
            labels = ['0-15%', '15-25%', '25-35%', '35-50%', '50-70%', '70-100%']
            df['prob_bin'] = pd.cut(df[prob_col], bins=bins, labels=labels)

            calibration_data = []
            for label in labels:
                bin_df = df[df['prob_bin'] == label]
                if len(bin_df) < 20:
                    continue

                predicted = bin_df[prob_col].mean()
                actual = (bin_df['actual_scenario'] == scenario_name).mean()
                diff = actual - predicted

                # 보정 계수: actual / predicted
                correction = actual / predicted if predicted > 0 else 1.0

                print(f"  {label:<15} {predicted:.1%}         {actual:.1%}         {diff:+.1%}      {correction:.2f}x")

                calibration_data.append({
                    'bin': label,
                    'predicted': predicted,
                    'actual': actual,
                    'correction_factor': correction
                })

                results['calibration_curve'][scenario_name][label] = {
                    'predicted': float(predicted),
                    'actual': float(actual),
                    'diff': float(diff),
                    'correction_factor': float(correction)
                }

            # 보정 공식 도출 (선형 회귀)
            if len(calibration_data) >= 3:
                predicted_vals = [d['predicted'] for d in calibration_data]
                actual_vals = [d['actual'] for d in calibration_data]

                # 선형 회귀: actual = a * predicted + b
                slope, intercept, r_value, p_value, std_err = stats.linregress(predicted_vals, actual_vals)

                results['calibration_formula'][scenario_name] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value**2),
                    'formula': f"calibrated_{scenario_name} = {slope:.3f} * predicted + {intercept:.3f}"
                }

                print(f"\n  보정 공식: calibrated = {slope:.3f} × predicted + {intercept:.3f} (R²={r_value**2:.3f})")

        # =====================================================================
        # 분석 2: 시나리오별 Bias 분석
        # =====================================================================
        print("\n  [분석 2] 시나리오별 전반적 Bias")
        print("  " + "-" * 60)
        print(f"  {'시나리오':<12} {'예측 평균':<12} {'실제 발생률':<12} {'Bias':<10}")
        print("  " + "-" * 60)

        for scenario_name, prob_col in scenarios:
            if prob_col not in df.columns:
                continue

            predicted_avg = df[prob_col].mean()
            actual_rate = (df['actual_scenario'] == scenario_name).mean()
            bias = predicted_avg - actual_rate

            # Bias 해석
            if bias > 0.05:
                bias_type = "과대추정"
            elif bias < -0.05:
                bias_type = "과소추정"
            else:
                bias_type = "적정"

            print(f"  {scenario_name:<12} {predicted_avg:.1%}         {actual_rate:.1%}         {bias:+.1%} ({bias_type})")

            results['scenario_bias'][scenario_name] = {
                'predicted_avg': float(predicted_avg),
                'actual_rate': float(actual_rate),
                'bias': float(bias),
                'bias_type': bias_type
            }

        # =====================================================================
        # 개선 방안
        # =====================================================================
        print("\n  [개선 방안]")
        print("  " + "=" * 70)

        for scenario_name, formula_data in results.get('calibration_formula', {}).items():
            rec = f"{scenario_name}: {formula_data['formula']}"
            results['recommendations'].append(rec)
            print(f"  → {rec}")

        # 저확률 구간 특별 처리
        print("\n  저확률 구간 (<20%) 특별 처리:")
        for scenario_name in ['bull', 'bear']:
            curve = results['calibration_curve'].get(scenario_name, {})
            low_prob_bins = ['0-15%', '15-25%']
            for bin_name in low_prob_bins:
                if bin_name in curve:
                    cf = curve[bin_name]['correction_factor']
                    if cf > 1.5 or cf < 0.5:
                        print(f"    {scenario_name} {bin_name}: 보정계수 {cf:.2f}x 적용 필요")

    return results


# =============================================================================
# PROBLEM 5: EM2 전략 - 구성요소 분석 및 수정 방안
# 목표: EM2의 어떤 구성요소가 문제인지 → 수정 방안 도출
# =============================================================================

async def analyze_em2_v2(pool: asyncpg.Pool) -> Dict:
    """
    EM2 전략 심층 분석

    분석 내용:
    1. EM2 구성요소별 IC 분해 (가능한 경우)
    2. EM2가 실패하는 시장 조건 분석
    3. 다른 EM 전략과의 상관관계 분석
    4. 개선 방안: 역전 신호, 구성요소 수정, 가중치 조정
    """
    print("\n" + "=" * 80)
    print("[PROBLEM 5] EM2 전략 - 구성요소 분석 및 수정 방안")
    print("목표: EM2의 어떤 구성요소가 문제인지 → 수정 방안 도출")
    print("=" * 80)

    results = {
        'em_strategy_comparison': {},
        'em2_failure_conditions': {},
        'em_correlations': {},
        'modification_recommendations': {}
    }

    async with pool.acquire() as conn:
        query = """
        SELECT
            sg.symbol,
            sg.date,
            sb.sector,
            sg.momentum_score,
            (sg.momentum_v2_detail->'EM1'->>'score')::float as em1_score,
            (sg.momentum_v2_detail->'EM2'->>'score')::float as em2_score,
            (sg.momentum_v2_detail->'EM3'->>'score')::float as em3_score,
            (sg.momentum_v2_detail->'EM4'->>'score')::float as em4_score,
            (sg.momentum_v2_detail->'EM5'->>'score')::float as em5_score,
            (sg.momentum_v2_detail->'EM6'->>'score')::float as em6_score,
            (sg.momentum_v2_detail->'EM7'->>'score')::float as em7_score,
            d_future.close / NULLIF(d_current.close, 0) - 1 as return_30d,
            d_future90.close / NULLIF(d_current.close, 0) - 1 as return_90d
        FROM us_stock_grade sg
        JOIN us_stock_basic sb ON sg.symbol = sb.symbol
        LEFT JOIN us_daily d_current ON sg.symbol = d_current.symbol AND sg.date = d_current.date
        LEFT JOIN us_daily d_future ON sg.symbol = d_future.symbol AND d_future.date = sg.date + INTERVAL '30 days'
        LEFT JOIN us_daily d_future90 ON sg.symbol = d_future90.symbol AND d_future90.date = sg.date + INTERVAL '90 days'
        WHERE sg.momentum_v2_detail IS NOT NULL
        """

        rows = await conn.fetch(query)
        if not rows:
            print("  데이터 없음")
            return results

        df = pd.DataFrame([dict(r) for r in rows])

        numeric_cols = ['momentum_score', 'em1_score', 'em2_score', 'em3_score',
                        'em4_score', 'em5_score', 'em6_score', 'em7_score',
                        'return_30d', 'return_90d']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"\n  분석 대상: {len(df)}개 종목-날짜")

        # =====================================================================
        # 분석 1: EM 전략별 IC 비교
        # =====================================================================
        print("\n  [분석 1] EM 전략별 IC 비교 (30일 수익률 기준)")
        print("  " + "-" * 70)
        print(f"  {'전략':<15} {'IC (30d)':<12} {'IC (90d)':<12} {'양/음':<10} {'판정':<15}")
        print("  " + "-" * 70)

        em_cols = ['em1_score', 'em2_score', 'em3_score', 'em4_score',
                   'em5_score', 'em6_score', 'em7_score']

        for col in em_cols:
            if col not in df.columns or df[col].isna().all():
                continue

            valid_30 = df[[col, 'return_30d']].dropna()
            valid_90 = df[[col, 'return_90d']].dropna()

            ic_30 = stats.spearmanr(valid_30[col], valid_30['return_30d'])[0] if len(valid_30) > 50 else np.nan
            ic_90 = stats.spearmanr(valid_90[col], valid_90['return_90d'])[0] if len(valid_90) > 50 else np.nan

            # 판정
            if pd.isna(ic_30):
                verdict = "데이터 부족"
            elif ic_30 < -0.05:
                verdict = "★★ 역방향 (수정필요)"
            elif ic_30 < 0:
                verdict = "★ 약한 역방향"
            elif ic_30 < 0.03:
                verdict = "효과 미약"
            else:
                verdict = "정상 작동"

            sign = "음수" if ic_30 < 0 else "양수"
            ic_30_str = f"{ic_30:.4f}" if pd.notna(ic_30) else "N/A"
            ic_90_str = f"{ic_90:.4f}" if pd.notna(ic_90) else "N/A"

            print(f"  {col:<15} {ic_30_str:<12} {ic_90_str:<12} {sign:<10} {verdict}")

            results['em_strategy_comparison'][col] = {
                'ic_30d': float(ic_30) if pd.notna(ic_30) else None,
                'ic_90d': float(ic_90) if pd.notna(ic_90) else None,
                'verdict': verdict
            }

        # =====================================================================
        # 분석 2: EM2 실패 조건 분석
        # =====================================================================
        print("\n  [분석 2] EM2가 실패하는 조건 분석")
        print("  " + "-" * 70)

        if 'em2_score' in df.columns and df['em2_score'].notna().sum() > 100:
            # 고 EM2 점수 + 저수익 종목 분석
            high_em2_low_return = df[
                (df['em2_score'] > df['em2_score'].quantile(0.7)) &
                (df['return_30d'] < df['return_30d'].quantile(0.3))
            ]

            # 저 EM2 점수 + 고수익 종목 분석
            low_em2_high_return = df[
                (df['em2_score'] < df['em2_score'].quantile(0.3)) &
                (df['return_30d'] > df['return_30d'].quantile(0.7))
            ]

            print(f"  고 EM2 + 저수익 종목: {len(high_em2_low_return)}개")
            print(f"  저 EM2 + 고수익 종목: {len(low_em2_high_return)}개")

            # 섹터별 분포
            if 'sector' in df.columns and len(high_em2_low_return) > 0:
                print("\n  고 EM2 + 저수익 종목의 섹터 분포:")
                sector_dist = high_em2_low_return['sector'].value_counts()
                for sector, count in sector_dist.head(5).items():
                    print(f"    {sector}: {count}개")

                results['em2_failure_conditions']['high_em2_low_return_sectors'] = sector_dist.head(5).to_dict()

            # EM2와 다른 EM 전략 비교
            print("\n  고 EM2 + 저수익 종목에서 다른 EM 전략의 점수:")
            for other_em in ['em1_score', 'em3_score', 'em4_score']:
                if other_em in high_em2_low_return.columns:
                    other_mean = high_em2_low_return[other_em].mean()
                    all_mean = df[other_em].mean()
                    diff = other_mean - all_mean
                    print(f"    {other_em}: {other_mean:.3f} (전체 평균 대비 {diff:+.3f})")

        # =====================================================================
        # 분석 3: EM 전략 간 상관관계
        # =====================================================================
        print("\n  [분석 3] EM 전략 간 상관관계")
        print("  " + "-" * 70)

        em_data = df[em_cols].dropna()
        if len(em_data) > 100:
            corr_matrix = em_data.corr()

            # EM2와 다른 전략 간 상관관계
            if 'em2_score' in corr_matrix.columns:
                print("  EM2와 다른 전략 간 상관관계:")
                for col in em_cols:
                    if col != 'em2_score' and col in corr_matrix.columns:
                        corr = corr_matrix.loc['em2_score', col]
                        print(f"    EM2 vs {col}: {corr:.3f}")
                        results['em_correlations'][f'em2_vs_{col}'] = float(corr)

        # =====================================================================
        # 개선 방안 도출
        # =====================================================================
        print("\n  [개선 방안 도출]")
        print("  " + "=" * 70)

        em2_data = results['em_strategy_comparison'].get('em2_score', {})
        em2_ic = em2_data.get('ic_30d', 0)

        if em2_ic is not None and em2_ic < -0.03:
            # 역전 신호 테스트
            print("\n  방안 1: EM2 신호 역전")
            if 'em2_score' in df.columns:
                df['em2_reversed'] = -df['em2_score']
                valid = df[['em2_reversed', 'return_30d']].dropna()
                reversed_ic = stats.spearmanr(valid['em2_reversed'], valid['return_30d'])[0]

                print(f"    원본 IC: {em2_ic:.4f}")
                print(f"    역전 IC: {reversed_ic:.4f}")

                if reversed_ic > abs(em2_ic):
                    rec = "EM2 신호 역전 적용 권장"
                    print(f"    → {rec}")
                    results['modification_recommendations']['signal_reversal'] = {
                        'original_ic': float(em2_ic),
                        'reversed_ic': float(reversed_ic),
                        'recommendation': rec
                    }

        # 가중치 조정 권장
        print("\n  방안 2: EM 전략 가중치 재조정")
        weight_recommendations = {}
        for col, data in results['em_strategy_comparison'].items():
            ic = data.get('ic_30d', 0)
            if ic is None:
                continue
            elif ic < -0.03:
                weight = -0.5  # 역전 적용
                action = "역전 × 0.5"
            elif ic < 0:
                weight = 0  # 제외
                action = "제외"
            elif ic < 0.02:
                weight = 0.3  # 감소
                action = "0.3x 가중치"
            else:
                weight = 1.0  # 유지
                action = "유지"

            weight_recommendations[col] = {'weight': weight, 'action': action}
            print(f"    {col}: {action} (IC={ic:.4f})")

        results['modification_recommendations']['weight_adjustments'] = weight_recommendations

    return results


# =============================================================================
# PROBLEM 6: NASDAQ 예측력 개선 - 필요한 팩터 분석
# 목표: NASDAQ에서 어떤 팩터가 필요한지 → 가중치/팩터 조정 근거
# =============================================================================

async def analyze_nasdaq_v2(pool: asyncpg.Pool) -> Dict:
    """
    NASDAQ 예측력 개선 분석

    분석 내용:
    1. NYSE vs NASDAQ 팩터별 효과 비교
    2. NASDAQ에서 부족한 팩터 식별
    3. NASDAQ 특화 팩터 필요성 분석
    4. 개선 방안: NASDAQ 전용 가중치, 추가 팩터 제안
    """
    print("\n" + "=" * 80)
    print("[PROBLEM 6] NASDAQ 예측력 개선 - 필요한 팩터 분석")
    print("목표: NASDAQ에서 어떤 팩터가 필요한지 → 가중치/팩터 조정 근거")
    print("=" * 80)

    results = {
        'exchange_factor_comparison': {},
        'nasdaq_weak_factors': {},
        'nasdaq_strong_factors': {},
        'weight_adjustment_formula': {},
        'additional_factor_recommendations': []
    }

    async with pool.acquire() as conn:
        query = """
        SELECT
            sg.symbol,
            sg.date,
            sb.exchange,
            sb.sector,
            sg.final_score,
            sg.value_score,
            sg.quality_score,
            sg.momentum_score,
            sg.growth_score,
            d_future.close / NULLIF(d_current.close, 0) - 1 as return_30d,
            d_future90.close / NULLIF(d_current.close, 0) - 1 as return_90d
        FROM us_stock_grade sg
        JOIN us_stock_basic sb ON sg.symbol = sb.symbol
        LEFT JOIN us_daily d_current ON sg.symbol = d_current.symbol AND sg.date = d_current.date
        LEFT JOIN us_daily d_future ON sg.symbol = d_future.symbol AND d_future.date = sg.date + INTERVAL '30 days'
        LEFT JOIN us_daily d_future90 ON sg.symbol = d_future90.symbol AND d_future90.date = sg.date + INTERVAL '90 days'
        WHERE sb.exchange IN ('NYSE', 'NASDAQ')
          AND sg.final_score IS NOT NULL
        """

        rows = await conn.fetch(query)
        if not rows:
            print("  데이터 없음")
            return results

        df = pd.DataFrame([dict(r) for r in rows])

        numeric_cols = ['final_score', 'value_score', 'quality_score', 'momentum_score',
                        'growth_score', 'return_30d', 'return_90d']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        nyse_df = df[df['exchange'] == 'NYSE']
        nasdaq_df = df[df['exchange'] == 'NASDAQ']

        print(f"\n  NYSE: {len(nyse_df)}개, NASDAQ: {len(nasdaq_df)}개")

        # =====================================================================
        # 분석 1: 팩터별 NYSE vs NASDAQ IC 비교
        # =====================================================================
        print("\n  [분석 1] 팩터별 NYSE vs NASDAQ IC 비교")
        print("  " + "-" * 75)
        print(f"  {'팩터':<18} {'NYSE IC':<12} {'NASDAQ IC':<12} {'Gap':<10} {'NASDAQ 진단':<15}")
        print("  " + "-" * 75)

        factors = ['final_score', 'value_score', 'quality_score', 'momentum_score', 'growth_score']

        for factor in factors:
            if factor not in df.columns:
                continue

            # NYSE IC
            valid_nyse = nyse_df[[factor, 'return_30d']].dropna()
            ic_nyse = stats.spearmanr(valid_nyse[factor], valid_nyse['return_30d'])[0] if len(valid_nyse) > 100 else np.nan

            # NASDAQ IC
            valid_nasdaq = nasdaq_df[[factor, 'return_30d']].dropna()
            ic_nasdaq = stats.spearmanr(valid_nasdaq[factor], valid_nasdaq['return_30d'])[0] if len(valid_nasdaq) > 100 else np.nan

            gap = ic_nyse - ic_nasdaq if pd.notna(ic_nyse) and pd.notna(ic_nasdaq) else np.nan

            # NASDAQ 진단
            if pd.isna(ic_nasdaq):
                diagnosis = "데이터 부족"
            elif ic_nasdaq < 0:
                diagnosis = "★★ 역방향 (심각)"
            elif gap > 0.05:
                diagnosis = "★ 효과 감소"
            else:
                diagnosis = "정상"

            ic_nyse_str = f"{ic_nyse:.4f}" if pd.notna(ic_nyse) else "N/A"
            ic_nasdaq_str = f"{ic_nasdaq:.4f}" if pd.notna(ic_nasdaq) else "N/A"
            gap_str = f"{gap:+.4f}" if pd.notna(gap) else "N/A"

            print(f"  {factor:<18} {ic_nyse_str:<12} {ic_nasdaq_str:<12} {gap_str:<10} {diagnosis}")

            results['exchange_factor_comparison'][factor] = {
                'nyse_ic': float(ic_nyse) if pd.notna(ic_nyse) else None,
                'nasdaq_ic': float(ic_nasdaq) if pd.notna(ic_nasdaq) else None,
                'gap': float(gap) if pd.notna(gap) else None,
                'diagnosis': diagnosis
            }

            # 약점/강점 분류
            if pd.notna(ic_nasdaq):
                if ic_nasdaq < 0 or (pd.notna(gap) and gap > 0.05):
                    results['nasdaq_weak_factors'][factor] = {
                        'ic': float(ic_nasdaq),
                        'gap_from_nyse': float(gap) if pd.notna(gap) else None
                    }
                elif ic_nasdaq > 0.03:
                    results['nasdaq_strong_factors'][factor] = {
                        'ic': float(ic_nasdaq)
                    }

        # =====================================================================
        # 분석 2: NASDAQ 섹터별 분석
        # =====================================================================
        print("\n  [분석 2] NASDAQ 섹터별 Final Score IC")
        print("  " + "-" * 60)
        print(f"  {'섹터':<25} {'IC':<12} {'샘플수':<10} {'진단':<15}")
        print("  " + "-" * 60)

        if 'sector' in nasdaq_df.columns:
            for sector in nasdaq_df['sector'].dropna().unique()[:10]:
                sector_df = nasdaq_df[nasdaq_df['sector'] == sector]
                if len(sector_df) < 50:
                    continue

                valid = sector_df[['final_score', 'return_30d']].dropna()
                ic = stats.spearmanr(valid['final_score'], valid['return_30d'])[0] if len(valid) > 30 else np.nan

                if pd.isna(ic):
                    diagnosis = "데이터 부족"
                elif ic < 0:
                    diagnosis = "★★ 문제 섹터"
                elif ic < 0.02:
                    diagnosis = "★ 효과 미약"
                else:
                    diagnosis = "정상"

                ic_str = f"{ic:.4f}" if pd.notna(ic) else "N/A"
                print(f"  {sector[:23]:<25} {ic_str:<12} {len(sector_df):<10} {diagnosis}")

        # =====================================================================
        # 분석 3: NASDAQ 특화 가중치 계산
        # =====================================================================
        print("\n  [분석 3] NASDAQ 특화 가중치 계산")
        print("  " + "-" * 60)

        # 각 팩터의 NASDAQ IC를 기반으로 가중치 계산
        print("  팩터별 NASDAQ 권장 가중치:")
        total_positive_ic = 0
        factor_weights = {}

        for factor in ['value_score', 'quality_score', 'momentum_score', 'growth_score']:
            data = results['exchange_factor_comparison'].get(factor, {})
            ic = data.get('nasdaq_ic', 0)
            if ic is not None and ic > 0:
                total_positive_ic += ic

        if total_positive_ic > 0:
            for factor in ['value_score', 'quality_score', 'momentum_score', 'growth_score']:
                data = results['exchange_factor_comparison'].get(factor, {})
                ic = data.get('nasdaq_ic', 0)
                if ic is not None and ic > 0:
                    weight = ic / total_positive_ic
                    factor_weights[factor] = weight
                    print(f"    {factor}: {weight:.1%}")
                else:
                    factor_weights[factor] = 0
                    print(f"    {factor}: 0% (IC<=0이므로 제외)")

            results['weight_adjustment_formula'] = {
                'method': 'IC 비례 가중치',
                'weights': factor_weights,
                'formula': 'NASDAQ_final = sum(factor_score × factor_weight)'
            }

        # =====================================================================
        # 개선 방안
        # =====================================================================
        print("\n  [개선 방안]")
        print("  " + "=" * 70)

        # 1. 약점 팩터 가중치 감소
        print("\n  1. 약점 팩터 가중치 감소:")
        for factor, data in results['nasdaq_weak_factors'].items():
            rec = f"    {factor}: NASDAQ에서 가중치 50% 감소 (IC={data['ic']:.3f})"
            print(rec)
            results['additional_factor_recommendations'].append(rec)

        # 2. 강점 팩터 가중치 증가
        print("\n  2. 강점 팩터 가중치 증가:")
        for factor, data in results['nasdaq_strong_factors'].items():
            rec = f"    {factor}: NASDAQ에서 가중치 50% 증가 (IC={data['ic']:.3f})"
            print(rec)
            results['additional_factor_recommendations'].append(rec)

        # 3. NASDAQ 특화 추가 팩터 제안
        print("\n  3. NASDAQ 특화 추가 팩터 제안:")
        nasdaq_specific_factors = [
            "Tech Moat Score: 기술적 해자 (특허, R&D 강도)",
            "Growth Sustainability: 성장 지속가능성 (매출 가속도)",
            "Institutional Flow: 기관 자금 흐름",
            "Analyst Revision: 애널리스트 추정치 변화",
            "Options Sentiment: 옵션 시장 심리 지표"
        ]
        for factor in nasdaq_specific_factors:
            print(f"    - {factor}")
            results['additional_factor_recommendations'].append(factor)

    return results


# =============================================================================
# RESULT SAVING - 개선 방안 중심으로 저장
# =============================================================================

async def save_results_v2(all_results: Dict, output_dir: str):
    """개선 방안 중심의 결과 저장"""
    print("\n" + "=" * 80)
    print("[결과 저장]")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    summary_data = []
    for problem, results in all_results.items():
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    summary_data.append({
                        'problem': problem,
                        'category': key,
                        'item': sub_key,
                        'value': str(sub_value)
                    })
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    summary_data.append({
                        'problem': problem,
                        'category': key,
                        'item': f'item_{i}',
                        'value': str(item)
                    })
            else:
                summary_data.append({
                    'problem': problem,
                    'category': key,
                    'item': 'value',
                    'value': str(value)
                })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, f'us_deep_analysis_v2_{timestamp}.csv')
        df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  CSV: {csv_path}")

    # Markdown 보고서 생성 - 개선 방안 중심
    md_content = generate_improvement_report(all_results, timestamp)
    md_path = os.path.join(output_dir, f'us_improvement_recommendations_{timestamp}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  MD: {md_path}")

    return csv_path, md_path


def generate_improvement_report(all_results: Dict, timestamp: str) -> str:
    """개선 방안 중심 마크다운 보고서 생성"""
    md = f"""# US Quant 전략 개선 방안 보고서
생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 개요

이 보고서는 US 퀀트 전략의 6가지 주요 문제에 대한 **구체적인 개선 방안**을 제시합니다.
각 문제에 대해 원인 분석과 함께 실행 가능한 수정 방안을 포함합니다.

---

"""

    # Problem 1: Decile Anomaly
    p1 = all_results.get('problem_1_decile_anomaly', {})
    md += """## 1. Decile Anomaly 개선 방안

### 문제 요약
D1~D5 저점수 종목 중 고수익 종목 발생 (팩터 저평가 문제)

### 원인 분석
"""
    if p1.get('factor_undervaluation_analysis'):
        for factor, data in p1['factor_undervaluation_analysis'].items():
            if data.get('is_undervaluation_cause'):
                md += f"- **{factor}**: 이상 종목에서 평균 {data['gap']:+.3f} 낮음 → 저평가 원인\n"

    md += "\n### 개선 방안\n"
    if p1.get('weight_adjustment_recommendations'):
        for factor, data in p1['weight_adjustment_recommendations'].items():
            md += f"- **{factor}**: {data.get('reasoning', 'N/A')}\n"
            md += f"  - 권장 가중치 배수: {data.get('recommended_weight_multiplier', 1.0)}x\n"

    # Problem 2: Healthcare
    p2 = all_results.get('problem_2_healthcare', {})
    md += """\n---

## 2. Healthcare 섹터 개선 방안

### 문제 요약
Healthcare 섹터에서 기존 팩터가 제대로 작동하지 않음

### 원인 분석
"""
    if p2.get('sector_ic_comparison'):
        for factor, data in p2['sector_ic_comparison'].items():
            if data.get('diagnosis') and '★' in data['diagnosis']:
                hc_ic = data.get('healthcare_ic', 'N/A')
                hc_ic_str = f"{hc_ic:.4f}" if isinstance(hc_ic, float) else hc_ic
                md += f"- **{factor}**: Healthcare IC={hc_ic_str} ({data['diagnosis']})\n"

    md += "\n### 개선 방안\n"
    md += "#### 기존 팩터 가중치 조정\n"
    for rec in p2.get('recommendations', []):
        md += f"- {rec}\n"

    md += "\n#### Healthcare 특화 대안 팩터\n"
    for factor in p2.get('alternative_factors', []):
        md += f"- {factor}\n"

    # Problem 3: Stop Loss
    p3 = all_results.get('problem_3_stop_loss', {})
    md += """\n---

## 3. Stop Loss ATR 개선 방안

### 문제 요약
현재 손절선이 적절하지 않아 최적화 필요

### 백테스트 결과
"""
    if p3.get('backtest_by_multiplier'):
        md += "| ATR 배수 | 발동률 | 평균 손익 | 승률 | 손익비 |\n"
        md += "|---------|-------|---------|-----|-------|\n"
        for mult, data in sorted(p3['backtest_by_multiplier'].items()):
            md += f"| {mult}x | {data['trigger_rate']*100:.1f}% | {data['avg_pnl']*100:+.2f}% | {data['win_rate']*100:.1f}% | {data['profit_ratio']:.2f} |\n"

    md += "\n### 최종 권고\n"
    if p3.get('final_recommendation'):
        rec = p3['final_recommendation']
        md += f"- **전역 최적 ATR 배수**: {rec.get('global_optimal_atr', 'N/A')}x\n"
        md += f"- **구현**: `{rec.get('implementation', 'N/A')}`\n"

    if p3.get('sector_optimal_atr'):
        md += "\n#### 섹터별 최적 ATR 배수\n"
        for sector, data in p3['sector_optimal_atr'].items():
            md += f"- {sector}: {data['optimal_multiplier']}x\n"

    # Problem 4: Scenario Calibration
    p4 = all_results.get('problem_4_scenario', {})
    md += """\n---

## 4. Scenario Calibration 개선 방안

### 문제 요약
시나리오 확률 예측이 실제와 불일치 (calibration 필요)

### Calibration 공식
"""
    if p4.get('calibration_formula'):
        for scenario, data in p4['calibration_formula'].items():
            md += f"- **{scenario}**: `{data.get('formula', 'N/A')}` (R²={data.get('r_squared', 0):.3f})\n"

    md += "\n### Bias 분석\n"
    if p4.get('scenario_bias'):
        for scenario, data in p4['scenario_bias'].items():
            md += f"- {scenario}: 예측 {data['predicted_avg']:.1%} vs 실제 {data['actual_rate']:.1%} → {data['bias_type']}\n"

    # Problem 5: EM2
    p5 = all_results.get('problem_5_em2', {})
    md += """\n---

## 5. EM2 전략 개선 방안

### 문제 요약
EM2 전략이 음의 IC를 보임 (역방향 작동)

### EM 전략별 IC 비교
"""
    if p5.get('em_strategy_comparison'):
        md += "| 전략 | IC (30d) | IC (90d) | 판정 |\n"
        md += "|-----|---------|---------|-----|\n"
        for strategy, data in p5['em_strategy_comparison'].items():
            ic_30 = data.get('ic_30d', 'N/A')
            ic_90 = data.get('ic_90d', 'N/A')
            ic_30_str = f"{ic_30:.4f}" if isinstance(ic_30, float) else ic_30
            ic_90_str = f"{ic_90:.4f}" if isinstance(ic_90, float) else ic_90
            md += f"| {strategy} | {ic_30_str} | {ic_90_str} | {data.get('verdict', 'N/A')} |\n"

    md += "\n### 개선 방안\n"
    if p5.get('modification_recommendations'):
        recs = p5['modification_recommendations']
        if 'signal_reversal' in recs:
            sr = recs['signal_reversal']
            md += f"#### 신호 역전\n"
            md += f"- 원본 IC: {sr.get('original_ic', 0):.4f}\n"
            md += f"- 역전 IC: {sr.get('reversed_ic', 0):.4f}\n"
            md += f"- **권고**: {sr.get('recommendation', 'N/A')}\n"

        if 'weight_adjustments' in recs:
            md += "\n#### 가중치 조정\n"
            for strategy, data in recs['weight_adjustments'].items():
                md += f"- {strategy}: {data.get('action', 'N/A')}\n"

    # Problem 6: NASDAQ
    p6 = all_results.get('problem_6_nasdaq', {})
    md += """\n---

## 6. NASDAQ 예측력 개선 방안

### 문제 요약
NASDAQ에서 팩터 효과가 NYSE 대비 낮음

### NYSE vs NASDAQ IC 비교
"""
    if p6.get('exchange_factor_comparison'):
        md += "| 팩터 | NYSE IC | NASDAQ IC | Gap | 진단 |\n"
        md += "|-----|---------|-----------|-----|-----|\n"
        for factor, data in p6['exchange_factor_comparison'].items():
            nyse_ic = data.get('nyse_ic', 'N/A')
            nasdaq_ic = data.get('nasdaq_ic', 'N/A')
            gap = data.get('gap', 'N/A')
            nyse_str = f"{nyse_ic:.4f}" if isinstance(nyse_ic, float) else nyse_ic
            nasdaq_str = f"{nasdaq_ic:.4f}" if isinstance(nasdaq_ic, float) else nasdaq_ic
            gap_str = f"{gap:+.4f}" if isinstance(gap, float) else gap
            md += f"| {factor} | {nyse_str} | {nasdaq_str} | {gap_str} | {data.get('diagnosis', 'N/A')} |\n"

    md += "\n### NASDAQ 특화 가중치\n"
    if p6.get('weight_adjustment_formula', {}).get('weights'):
        for factor, weight in p6['weight_adjustment_formula']['weights'].items():
            md += f"- {factor}: {weight:.1%}\n"

    md += "\n### 추가 권고사항\n"
    for rec in p6.get('additional_factor_recommendations', []):
        md += f"- {rec}\n"

    md += """\n---

## 종합 실행 계획

### 즉시 적용 가능
1. ATR 배수 조정 (Problem 3)
2. Scenario Calibration 공식 적용 (Problem 4)
3. EM2 신호 역전 (Problem 5)

### 코드 수정 필요
1. 팩터 가중치 조정 (Problem 1, 2, 6)
2. Healthcare 특화 팩터 추가 (Problem 2)
3. NASDAQ 특화 가중치 적용 (Problem 6)

### 추가 개발 필요
1. Healthcare Pipeline Value 팩터 (Problem 2)
2. Tech Moat Score 등 NASDAQ 특화 팩터 (Problem 6)

---

*이 보고서는 자동 생성되었습니다.*
"""

    return md


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("US QUANT DEEP PROBLEM ANALYSIS V2")
    print("목표: 퀀트 전략 수정의 구체적 근거 제공")
    print("=" * 80)
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"출력 폴더: {OUTPUT_DIR}")

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)

    all_results = {}

    try:
        all_results['problem_1_decile_anomaly'] = await analyze_decile_anomaly_v2(pool)
        all_results['problem_2_healthcare'] = await analyze_healthcare_v2(pool)
        all_results['problem_3_stop_loss'] = await analyze_stop_loss_v2(pool)
        all_results['problem_4_scenario'] = await analyze_scenario_calibration_v2(pool)
        all_results['problem_5_em2'] = await analyze_em2_v2(pool)
        all_results['problem_6_nasdaq'] = await analyze_nasdaq_v2(pool)

        csv_path, md_path = await save_results_v2(all_results, OUTPUT_DIR)

        print("\n" + "=" * 80)
        print("분석 완료")
        print("=" * 80)
        print(f"종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"결과 파일:")
        print(f"  - {csv_path}")
        print(f"  - {md_path}")

    finally:
        await pool.close()

    return all_results


if __name__ == '__main__':
    asyncio.run(main())
