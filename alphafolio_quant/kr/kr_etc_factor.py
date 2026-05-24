"""
기타 팩터 (Etc Factor) - 섹터별 특화 리스크 조정

Value, Quality, Momentum, Growth 4대 팩터로 포착할 수 없는
섹터 특화 리스크를 측정하고 최종 점수를 조정합니다.

현재 지원 섹터:
1. Telecom/Media (게임, 엔터, 미디어, OTT, 광고)

향후 확장 가능:
2. Financial (은행, 증권, 보험)
3. Healthcare (제약, 바이오, 의료기기)
4. REIT (부동산)
"""

import logging
from typing import Dict, Optional

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Telecom/Media 섹터 리스크 조정기
# ============================================================================

class TelecomMediaRiskAdjuster:
    """
    Telecom/Media 섹터 특화 리스크 조정

    적용 대상:
    - 게임 (소프트웨어 개발 및 공급업)
    - 엔터테인먼트 (오디오물 출판 및 원판 녹음업)
    - 미디어/방송 (영화, 비디오물, 방송프로그램 제작 및 배급업)
    - 광고 (광고업)
    - OTT/정보서비스 (기타 정보 서비스업)

    측정 팩터:
    - F1: Hit Dependency Penalty (히트작 의존도)
    - F2: Platform Risk (플랫폼 리스크)
    - F3: Content Pipeline Score (콘텐츠 파이프라인)
    - F4: Volatility Penalty (변동성 페널티)
    """

    # 업종별 플랫폼 의존도 매핑
    PLATFORM_RISK_MAP = {
        '소프트웨어 개발 및 공급업': 0.8,  # 앱스토어 의존
        '광고업': 0.7,  # 구글/페이스북 의존
        '기타 정보 서비스업': 0.7,  # 포털 의존 (웹툰, 전자책)
        '오디오물 출판 및 원판 녹음업': 0.5,  # 유통 플랫폼
        '영화, 비디오물, 방송프로그램 제작 및 배급업': 0.4,  # 극장/OTT
        '통신업': 0.2,  # 인프라 보유
    }

    # 시장 평균 변동성 (KOSPI 기준)
    MARKET_VOLATILITY = 30.0

    def __init__(self, symbol: str, db_manager, analysis_date=None):
        """
        초기화

        Args:
            symbol: 종목 코드
            db_manager: AsyncDatabaseManager 인스턴스
            analysis_date: 분석 날짜 (optional)
        """
        self.symbol = symbol
        self.db_manager = db_manager
        self.analysis_date = analysis_date

    async def execute_query(self, query: str, *params):
        """SQL 쿼리 실행"""
        try:
            return await self.db_manager.execute_query(query, *params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None

    # ========================================================================
    # F1. Hit Dependency Penalty (히트작 의존도 페널티)
    # ========================================================================

    async def calculate_f1_hit_dependency(self) -> Optional[float]:
        """
        F1. 히트작 의존도 페널티

        개념: 매출/이익이 급변하는 종목 = 히트작에 의존

        측정 방법:
        - 최근 4분기 매출 변동성 (CoV = Coefficient of Variation)
        - CoV = 표준편차 / 평균

        점수:
        - CoV > 0.5: 20점 (매우 위험)
        - CoV 0.3-0.5: 50점 (위험)
        - CoV < 0.3: 80점 (안정)

        Returns:
            float: 0-100 점수 (낮을수록 위험)
        """
        query = """
        WITH quarterly_revenue AS (
            SELECT
                bsns_year,
                report_code,
                thstrm_amount as revenue,
                rcept_dt,
                ROW_NUMBER() OVER (ORDER BY bsns_year DESC, rcept_dt DESC) as rn
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm = '매출액'
                AND thstrm_amount > 0
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 4
        )
        SELECT
            AVG(revenue) as mean_revenue,
            STDDEV(revenue) as std_revenue,
            COUNT(*) as quarters
        FROM quarterly_revenue
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or result[0]['quarters'] < 2:
            logger.info(f"F1 (Hit Dependency): Not enough data for {self.symbol}")
            return None

        mean_rev = float(result[0]['mean_revenue'])
        std_rev = float(result[0]['std_revenue']) if result[0]['std_revenue'] else 0

        if mean_rev == 0:
            return 50  # 기본값

        # CoV (Coefficient of Variation)
        cov = std_rev / mean_rev

        # 점수 계산
        if cov > 0.5:
            score = 20  # 매우 불안정 (히트작 의존)
            risk_level = "HIGH"
        elif cov > 0.3:
            score = 50  # 불안정
            risk_level = "MEDIUM"
        else:
            score = 80  # 안정
            risk_level = "LOW"

        logger.info(f"F1 (Hit Dependency): {self.symbol} CoV={cov:.2f}, Score={score}, Risk={risk_level}")

        return score

    # ========================================================================
    # F2. Platform Risk Factor (플랫폼 리스크)
    # ========================================================================

    async def calculate_f2_platform_risk(self) -> Optional[float]:
        """
        F2. 플랫폼 리스크

        개념: 외부 플랫폼(네이버, 구글, 앱스토어) 의존도

        측정 방법:
        - 업종별 플랫폼 의존도 매핑 (PLATFORM_RISK_MAP)

        점수:
        - 의존도 높음 (0.8): 20점
        - 의존도 중간 (0.5): 50점
        - 의존도 낮음 (0.2): 80점

        Returns:
            float: 0-100 점수 (낮을수록 위험)
        """
        query = """
        SELECT industry
        FROM kr_stock_detail
        WHERE symbol = $1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or not result[0]['industry']:
            logger.info(f"F2 (Platform Risk): No industry data for {self.symbol}")
            return 50  # 기본값

        industry = result[0]['industry']

        # 플랫폼 의존도 조회
        platform_dependency = self.PLATFORM_RISK_MAP.get(industry, 0.5)

        # 점수 계산 (의존도가 높을수록 낮은 점수)
        score = 100 * (1 - platform_dependency)

        logger.info(f"F2 (Platform Risk): {self.symbol} Industry={industry}, Dependency={platform_dependency:.1f}, Score={score:.1f}")

        return score

    # ========================================================================
    # F3. Content Pipeline Score (콘텐츠 파이프라인)
    # ========================================================================

    async def calculate_f3_content_pipeline(self) -> Optional[float]:
        """
        F3. 콘텐츠 파이프라인 점수

        개념: 향후 신작/신규 콘텐츠 출시 계획 유무

        측정 방법 (대용 지표):
        - R&D 비용 증가율 (kr_financial_position에서 조회)
        - R&D 데이터 없으면 CAPEX 증가율 사용

        점수:
        - 증가율 +30% 이상: 80점 (적극 투자)
        - 증가율 +10% ~ +30%: 60점 (보통 투자)
        - 증가율 -10% ~ +10%: 40점 (유지)
        - 증가율 -10% 이하: 20점 (투자 감소, 위험)

        Returns:
            float: 0-100 점수 (낮을수록 위험)
        """
        # R&D 비용 조회 시도
        rd_query = """
        SELECT
            thstrm_amount as current_rd,
            frmtrm_amount as prev_rd
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm LIKE '%연구%'
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC
        LIMIT 1
        """

        result = await self.execute_query(rd_query, self.symbol, self.analysis_date)

        current_val = None
        prev_val = None
        data_type = None

        if result and result[0]['current_rd'] and result[0]['prev_rd']:
            current_val = float(result[0]['current_rd'])
            prev_val = float(result[0]['prev_rd'])
            data_type = "R&D"

        # R&D 없으면 CAPEX 시도
        if not current_val:
            capex_query = """
            SELECT
                thstrm_amount as current_capex,
                frmtrm_amount as prev_capex
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'CF'
                AND account_nm LIKE '%투자%'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
            """

            result = await self.execute_query(capex_query, self.symbol, self.analysis_date)

            if result and result[0]['current_capex'] and result[0]['prev_capex']:
                current_val = float(result[0]['current_capex'])
                prev_val = float(result[0]['prev_capex'])
                data_type = "CAPEX"

        # 데이터 없으면 중립 점수
        if not current_val or not prev_val or prev_val == 0:
            logger.info(f"F3 (Pipeline): No R&D/CAPEX data for {self.symbol}, returning neutral 40")
            return 40

        # 증가율 계산
        growth_rate = ((current_val - prev_val) / abs(prev_val)) * 100

        # 점수 계산
        if growth_rate > 30:
            score = 80  # 적극 투자
            risk_level = "LOW"
        elif growth_rate > 10:
            score = 60  # 보통 투자
            risk_level = "MEDIUM"
        elif growth_rate > -10:
            score = 40  # 유지
            risk_level = "MEDIUM"
        else:
            score = 20  # 투자 감소
            risk_level = "HIGH"

        logger.info(f"F3 (Pipeline): {self.symbol} {data_type} Growth={growth_rate:+.1f}%, Score={score}, Risk={risk_level}")

        return score

    # ========================================================================
    # F4. Volatility Penalty (변동성 페널티)
    # ========================================================================

    async def calculate_f4_volatility_penalty(self, volatility_annual: float) -> float:
        """
        F4. 변동성 페널티

        개념: 높은 변동성 = 투기성 → 페널티

        측정 방법:
        - 시장 대비 상대 변동성 (stock_vol / market_vol)

        점수:
        - 상대 변동성 > 1.5: 70점 (매우 높음)
        - 상대 변동성 1.2-1.5: 80점 (높음)
        - 상대 변동성 < 1.2: 90점 (보통)

        Args:
            volatility_annual: 연간 변동성 (kr_stock_grade에서 전달)

        Returns:
            float: 0-100 점수 (낮을수록 위험)
        """
        if not volatility_annual or volatility_annual <= 0:
            logger.info(f"F4 (Volatility): No volatility data for {self.symbol}, returning neutral 80")
            return 80

        # 시장 대비 상대 변동성
        relative_volatility = volatility_annual / self.MARKET_VOLATILITY

        # 점수 계산
        if relative_volatility > 1.5:
            score = 70  # 매우 높은 변동성
            risk_level = "HIGH"
        elif relative_volatility > 1.2:
            score = 80  # 높은 변동성
            risk_level = "MEDIUM"
        else:
            score = 90  # 보통 변동성
            risk_level = "LOW"

        logger.info(f"F4 (Volatility): {self.symbol} Vol={volatility_annual:.1f}%, Relative={relative_volatility:.2f}, Score={score}, Risk={risk_level}")

        return score

    # ========================================================================
    # 최종 리스크 조정 점수 계산
    # ========================================================================

    async def calculate_risk_adjustment(self, base_score: float, volatility_annual: float = None) -> Dict:
        """
        Telecom/Media 섹터 리스크 조정 최종 점수

        가중치:
        - F1 (Hit Dependency): 30%
        - F2 (Platform Risk): 30%
        - F3 (Content Pipeline): 20%
        - F4 (Volatility): 20%

        리스크 조정 계수:
        - risk_multiplier = 0.5 + (risk_score / 200)
        - 범위: 0.5 ~ 1.0

        Args:
            base_score: 기존 팩터 점수 (Value, Quality, Momentum, Growth 평균)
            volatility_annual: 연간 변동성 (optional)

        Returns:
            Dict: {
                'adjusted_score': 조정된 최종 점수,
                'risk_score': 리스크 종합 점수 (0-100),
                'risk_multiplier': 리스크 조정 계수 (0.5-1.0),
                'f1_hit_dependency': F1 점수,
                'f2_platform_risk': F2 점수,
                'f3_content_pipeline': F3 점수,
                'f4_volatility': F4 점수
            }
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Telecom/Media Risk Adjustment: {self.symbol}")
        logger.info(f"Base Score: {base_score:.2f}")
        logger.info(f"{'='*80}")

        # F1-F4 계산
        f1 = await self.calculate_f1_hit_dependency()
        f2 = await self.calculate_f2_platform_risk()
        f3 = await self.calculate_f3_content_pipeline()
        f4 = await self.calculate_f4_volatility_penalty(volatility_annual) if volatility_annual else 80

        # None 값 처리 (중립 점수로 대체)
        f1 = f1 if f1 is not None else 50
        f2 = f2 if f2 is not None else 50
        f3 = f3 if f3 is not None else 40
        f4 = f4 if f4 is not None else 80

        # 리스크 종합 점수 (0-100)
        risk_score = (f1 * 0.3 + f2 * 0.3 + f3 * 0.2 + f4 * 0.2)

        # 리스크 조정 계수 (0.5 ~ 1.0)
        risk_multiplier = 0.5 + (risk_score / 200)

        # 최종 조정 점수
        adjusted_score = base_score * risk_multiplier

        logger.info(f"\n{'='*80}")
        logger.info(f"Risk Score: {risk_score:.2f} (F1:{f1:.1f}, F2:{f2:.1f}, F3:{f3:.1f}, F4:{f4:.1f})")
        logger.info(f"Risk Multiplier: {risk_multiplier:.3f}")
        logger.info(f"Adjusted Score: {base_score:.2f} → {adjusted_score:.2f} ({adjusted_score - base_score:+.2f})")
        logger.info(f"{'='*80}\n")

        return {
            'adjusted_score': round(adjusted_score, 1),
            'risk_score': round(risk_score, 1),
            'risk_multiplier': round(risk_multiplier, 3),
            'f1_hit_dependency': round(f1, 1),
            'f2_platform_risk': round(f2, 1),
            'f3_content_pipeline': round(f3, 1),
            'f4_volatility': round(f4, 1)
        }


# ============================================================================
# 향후 확장: 다른 섹터 리스크 조정기
# ============================================================================

class FinancialSectorRiskAdjuster:
    """
    금융 섹터 (은행, 증권, 보험) 리스크 조정

    TODO: 향후 구현
    - 금리 리스크
    - 부실채권 비율
    - 자본적정성 비율
    """
    pass


class HealthcareSectorRiskAdjuster:
    """
    헬스케어 섹터 (제약, 바이오, 의료기기) 리스크 조정

    TODO: 향후 구현
    - 임상 파이프라인
    - FDA 승인 확률
    - 특허 만료 리스크
    """
    pass
