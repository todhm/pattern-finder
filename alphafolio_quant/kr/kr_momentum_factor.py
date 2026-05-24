"""
Korean Stock Momentum Factor System
Implements 17 momentum factor strategies to score stocks on a 100-point scale
File: kr_momentum_factor.py
"""

import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Import market classifier
try:
    from market_classifier import MarketClassifier
except ImportError:
    from kr.market_classifier import MarketClassifier

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Sector-based Momentum Multipliers (134 sectors analyzed)
# Based on 30-day IC analysis from comprehensive IC report
# Multiplier range: 0.5 (worst sectors) to 1.5 (best sectors), default 1.0
# ========================================================================

SECTOR_MULTIPLIERS = {
    # Top performing sectors (IC > 0.4) - High multipliers
    '의복 액세서리 제조업': 1.50,
    '건물설비 설치 공사업': 1.50,
    '어로 어업': 1.45,
    '가전제품 및 정보통신장비 소매업': 1.40,
    '유원지 및 기타 오락관련 서비스업': 1.35,
    '가구 제조업': 1.35,
    '유리 및 유리제품 제조업': 1.35,
    '토목 건설업': 1.30,
    '동·식물성 유지 및 낙농제품 제조업': 1.30,
    '기타 사업지원 서비스업': 1.30,

    # Good performing sectors (IC 0.25-0.4) - Moderate-high multipliers
    '전문디자인업': 1.25,
    '기타 섬유제품 제조업': 1.25,
    '기타 비금속 광물제품 제조업': 1.25,
    '그외 기타 운송장비 제조업': 1.20,
    '건축기술, 엔지니어링 및 관련 기술 서비스업': 1.20,
    '특수 목적용 기계 제조업': 1.20,
    '구조용 금속제품, 탱크 및 증기발생기 제조업': 1.20,
    '그외 기타 전문, 과학 및 기술 서비스업': 1.20,
    '동물용 사료 및 조제식품 제조업': 1.20,
    '사진장비 및 광학기기 제조업': 1.20,

    # Average performing sectors (IC 0.1-0.25) - Slight positive multipliers
    '1차 금속 제조업': 1.10,
    '화학물질 및 화학제품 제조업': 1.10,
    '전자부품 제조업': 1.10,
    '의료용 물질 및 의약품 제조업': 1.10,
    '자동차 및 트레일러 제조업': 1.10,

    # Below average sectors (IC 0 to 0.1) - Neutral to slight negative
    '식료품 제조업': 1.00,
    '음료 제조업': 1.00,
    '섬유제품 제조업': 1.00,
    '화학섬유 제조업': 1.00,

    # Poor performing sectors (IC -0.1 to 0) - Low multipliers
    '서적, 잡지 및 기타 인쇄물 출판업': 0.85,
    '내화, 비내화 요업제품 제조업': 0.85,
    '기타 과학기술 서비스업': 0.80,
    '신발 및 신발 부분품 제조업': 0.80,
    '기타 상품 전문 소매업': 0.80,

    # Very poor performing sectors (IC < -0.3) - Very low multipliers
    '귀금속 및 장신용품 제조업': 0.75,
    '제재 및 목재 가공업': 0.75,
    '무기 및 총포탄 제조업': 0.70,
    '과실, 채소 가공 및 저장 처리업': 0.70,
    '기타 종이 및 판지 제품 제조업': 0.70,
    '금속 주조업': 0.70,
    '전기업': 0.70,
    '기타 전문 서비스업': 0.70,
    '해상 운송업': 0.65,
    '기반조성 및 시설물 축조관련 전문공사업': 0.65,
    '육상 여객 운송업': 0.60,
    '전구 및 조명장치 제조업': 0.60,
    '비알코올음료 및 얼음 제조업': 0.55,
    '스포츠 서비스업': 0.50,
    '기타 생활용품 소매업': 0.50,
}

# ========================================================================
# Momentum Disabled Themes (모멘텀 전략이 역작동하는 테마)
# Based on IC analysis: Telecom_Media IC = 0.159 (lowest), avg return = -9.5%
# ========================================================================

MOMENTUM_DISABLED_THEMES = ['Telecom_Media']

# ========================================================================
# Volatility Regime Thresholds (시장 변동성 기반 모멘텀 조정)
# Based on Barroso & Santa-Clara (2015) "Momentum has its Moments"
# ========================================================================

VOLATILITY_REGIME_CONFIG = {
    'HIGH_VOL': {
        'threshold': 1.5,  # Vol20d > 1.5%
        'momentum_weight': 0.5,  # 모멘텀 가중치 50%로 감소
        'description': '고변동성 - 모멘텀 크래시 위험'
    },
    'MED_VOL': {
        'threshold': 1.0,  # Vol20d > 1.0%
        'momentum_weight': 0.75,  # 모멘텀 가중치 75%
        'description': '중변동성'
    },
    'LOW_VOL': {
        'threshold': 0.0,  # Vol20d <= 1.0%
        'momentum_weight': 1.0,  # 모멘텀 가중치 정상
        'description': '저변동성 - 모멘텀 정상 작동'
    }
}

# Target volatility for volatility scaling (연환산 12%)
TARGET_VOLATILITY = 12.0

# ========================================================================
# Mean Reversion Risk Thresholds
# Based on empirical analysis: RSI > 75 + MA20 deviation > 15% = crash risk
# ========================================================================

MEAN_REVERSION_RISK_CONFIG = {
    'RSI_EXTREME': {'threshold': 80, 'penalty': 40},
    'RSI_OVERBOUGHT': {'threshold': 75, 'penalty': 25},
    'MA20_EXTREME_DEV': {'threshold': 20, 'penalty': 30},  # MA20 괴리율 20%
    'MA20_HIGH_DEV': {'threshold': 15, 'penalty': 20},  # MA20 괴리율 15%
    'BB_RSI_COMBO': {'rsi_threshold': 65, 'penalty': 15},  # BB 상단 + RSI > 65
}

# ========================================================================
# Market State-based Momentum Strategy Weights (19 market states × 20 strategies)
# Updated based on M1~M5 IC Analysis (2025-11-16)
# Key findings: M1 (IC 0.0362) >> M4 (0.0171) > M5 (0.0078) > M3 (-0.0006) > M2 (-0.0131)
# ========================================================================

MOMENTUM_STRATEGY_WEIGHTS = {
    # Large Cap Group (6)
    # Large cap: M1 IC 0.037, M4 0.013, M5 0.007, M3 -0.004, M2 -0.009
    'KOSPI대형-확장과열-공격형': {
        'M1': 2.5, 'M2': 0.5, 'M3': 0.3, 'M4': 1.7, 'M5': 1.5, 'M6': 1.3, 'M7': 1.6,
        'M8': 1.9, 'M9': 1.4, 'M10': 1.5, 'M11': 1.3, 'M12': 1.7, 'M13': 1.5, 'M14': 1.4,
        'M15': 1.2, 'M16': 1.8, 'M17': 1.6, 'M18': 1.7, 'M19': 1.9, 'M20': 1.8,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.8
    },
    'KOSPI대형-확장중립-성장형': {
        'M1': 2.0, 'M2': 0.4, 'M3': 0.2, 'M4': 1.3, 'M5': 1.2, 'M6': 1.5, 'M7': 1.3,
        'M8': 1.1, 'M9': 1.2, 'M10': 1.2, 'M11': 1.1, 'M12': 1.0, 'M13': 1.2, 'M14': 1.1,
        'M15': 1.4, 'M16': 1.1, 'M17': 1.0, 'M18': 0.9, 'M19': 1.0, 'M20': 1.0,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.2
    },
    'KOSPI대형-둔화공포-방어형': {
        'M1': 0.8, 'M2': 0.2, 'M3': 0.1, 'M4': 0.5, 'M5': 0.7, 'M6': 0.8, 'M7': 0.5,
        'M8': 0.3, 'M9': 0.6, 'M10': 0.5, 'M11': 0.5, 'M12': 0.4, 'M13': 0.6, 'M14': 0.5,
        'M15': 0.7, 'M16': 0.5, 'M17': 0.3, 'M18': 0.2, 'M19': 0.2, 'M20': 0.3,
        'M21': 0.0, 'M22': 0.0, 'M23': 0.6
    },
    'KOSPI대형-침체패닉-초방어형': {
        'M1': 0.4, 'M2': 0.1, 'M3': 0.0, 'M4': 0.2, 'M5': 0.4, 'M6': 0.5, 'M7': 0.3,
        'M8': 0.1, 'M9': 0.3, 'M10': 0.3, 'M11': 0.3, 'M12': 0.2, 'M13': 0.4, 'M14': 0.3,
        'M15': 0.5, 'M16': 0.3, 'M17': 0.1, 'M18': 0.1, 'M19': 0.1, 'M20': 0.2,
        'M21': 0.0, 'M22': 0.0, 'M23': 0.3
    },
    'KOSPI대형-회복탐욕-밸류형': {
        'M1': 1.6, 'M2': 0.3, 'M3': 0.2, 'M4': 1.2, 'M5': 1.1, 'M6': 1.4, 'M7': 1.2,
        'M8': 0.8, 'M9': 1.0, 'M10': 1.0, 'M11': 0.9, 'M12': 0.8, 'M13': 1.1, 'M14': 0.9,
        'M15': 1.3, 'M16': 1.0, 'M17': 0.7, 'M18': 0.6, 'M19': 0.7, 'M20': 0.8,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.0
    },
    'KOSPI대형-중립안정-균형형': {
        'M1': 1.3, 'M2': 0.3, 'M3': 0.1, 'M4': 0.9, 'M5': 0.8, 'M6': 1.0, 'M7': 0.9,
        'M8': 0.7, 'M9': 0.8, 'M10': 0.8, 'M11': 0.7, 'M12': 0.6, 'M13': 0.8, 'M14': 0.7,
        'M15': 0.9, 'M16': 0.7, 'M17': 0.6, 'M18': 0.5, 'M19': 0.6, 'M20': 0.6,
        'M21': 0.0, 'M22': 0.0, 'M23': 0.8
    },

    # Mid Cap Group (6)
    # Mid cap: M1 IC 0.084 (!), M4 0.030, M5 0.016, M3 0.006, M2 -0.018
    # Mid cap has BEST performance - increase weights significantly
    'KOSPI중형-확장과열-모멘텀형': {
        'M1': 3.0, 'M2': 0.5, 'M3': 0.8, 'M4': 2.2, 'M5': 2.0, 'M6': 1.5, 'M7': 1.8,
        'M8': 2.0, 'M9': 1.6, 'M10': 1.7, 'M11': 1.6, 'M12': 1.9, 'M13': 1.7, 'M14': 1.6,
        'M15': 1.4, 'M16': 2.0, 'M17': 1.8, 'M18': 1.9, 'M19': 2.0, 'M20': 2.0,
        'M21': 0.0, 'M22': 0.0, 'M23': 2.0
    },
    'KOSPI중형-회복중립-성장형': {
        'M1': 2.2, 'M2': 0.4, 'M3': 0.5, 'M4': 1.5, 'M5': 1.4, 'M6': 1.4, 'M7': 1.3,
        'M8': 1.0, 'M9': 1.1, 'M10': 1.1, 'M11': 1.0, 'M12': 1.0, 'M13': 1.2, 'M14': 1.0,
        'M15': 1.3, 'M16': 1.2, 'M17': 1.0, 'M18': 0.9, 'M19': 1.0, 'M20': 1.1,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.3
    },
    'KOSPI중형-둔화공포-혼조형': {
        'M1': 1.2, 'M2': 0.3, 'M3': 0.2, 'M4': 0.8, 'M5': 0.9, 'M6': 0.9, 'M7': 0.7,
        'M8': 0.5, 'M9': 0.7, 'M10': 0.7, 'M11': 0.6, 'M12': 0.5, 'M13': 0.7, 'M14': 0.6,
        'M15': 0.8, 'M16': 0.7, 'M17': 0.5, 'M18': 0.4, 'M19': 0.4, 'M20': 0.5,
        'M21': 0.0, 'M22': 0.0, 'M23': 0.7
    },
    'KOSDAQ중형-확장탐욕-공격성장형': {
        'M1': 3.0, 'M2': 0.6, 'M3': 0.8, 'M4': 2.3, 'M5': 2.0, 'M6': 1.6, 'M7': 1.9,
        'M8': 2.0, 'M9': 1.7, 'M10': 1.8, 'M11': 1.7, 'M12': 2.0, 'M13': 1.8, 'M14': 1.7,
        'M15': 1.5, 'M16': 2.0, 'M17': 1.9, 'M18': 2.0, 'M19': 2.0, 'M20': 2.0,
        'M21': 0.0, 'M22': 0.0, 'M23': 2.2
    },
    'KOSDAQ중형-회복중립-성장테마형': {
        'M1': 2.3, 'M2': 0.5, 'M3': 0.6, 'M4': 1.7, 'M5': 1.6, 'M6': 1.5, 'M7': 1.5,
        'M8': 1.3, 'M9': 1.3, 'M10': 1.4, 'M11': 1.3, 'M12': 1.3, 'M13': 1.4, 'M14': 1.3,
        'M15': 1.4, 'M16': 1.5, 'M17': 1.3, 'M18': 1.2, 'M19': 1.3, 'M20': 1.4,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.5
    },
    'KOSDAQ중형-침체공포-역발상형': {
        'M1': 1.0, 'M2': 0.2, 'M3': 0.1, 'M4': 0.6, 'M5': 0.9, 'M6': 0.9, 'M7': 0.6,
        'M8': 0.4, 'M9': 0.6, 'M10': 0.6, 'M11': 0.5, 'M12': 0.4, 'M13': 0.7, 'M14': 0.5,
        'M15': 0.8, 'M16': 0.6, 'M17': 0.4, 'M18': 0.3, 'M19': 0.3, 'M20': 0.4,
        'M21': 0.0, 'M22': 0.0, 'M23': 0.6
    },

    # Small Cap Group (4)
    # Small cap: M1 IC 0.012, M4 0.009, M5 -0.001, M3 -0.015, M2 -0.023 (!)
    # Small cap has POOR performance - reduce weights, M2/M3/M5 use normalized versions
    'KOSDAQ소형-핫섹터-초고위험형': {
        'M1': 2.5, 'M2': 0.8, 'M3': 0.5, 'M4': 2.0, 'M5': 1.2, 'M6': 1.3, 'M7': 1.8,
        'M8': 2.0, 'M9': 1.8, 'M10': 1.8, 'M11': 1.8, 'M12': 2.0, 'M13': 1.6, 'M14': 1.7,
        'M15': 1.2, 'M16': 2.0, 'M17': 2.0, 'M18': 2.0, 'M19': 2.0, 'M20': 2.0,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.8
    },
    'KOSDAQ소형-성장테마-고위험형': {
        'M1': 2.2, 'M2': 0.7, 'M3': 0.4, 'M4': 1.8, 'M5': 1.0, 'M6': 1.4, 'M7': 1.7,
        'M8': 1.8, 'M9': 1.6, 'M10': 1.6, 'M11': 1.6, 'M12': 1.8, 'M13': 1.5, 'M14': 1.5,
        'M15': 1.3, 'M16': 1.9, 'M17': 1.8, 'M18': 1.8, 'M19': 1.8, 'M20': 1.9,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.6
    },
    'KOSDAQ소형-침체-극단역발상형': {
        'M1': 0.6, 'M2': 0.2, 'M3': 0.1, 'M4': 0.4, 'M5': 0.5, 'M6': 0.7, 'M7': 0.4,
        'M8': 0.2, 'M9': 0.4, 'M10': 0.4, 'M11': 0.3, 'M12': 0.2, 'M13': 0.5, 'M14': 0.3,
        'M15': 0.6, 'M16': 0.4, 'M17': 0.2, 'M18': 0.1, 'M19': 0.1, 'M20': 0.2,
        'M21': 0.0, 'M22': 0.0, 'M23': 0.4
    },
    'KOSDAQ소형-회복-모멘텀형': {
        'M1': 2.0, 'M2': 0.6, 'M3': 0.3, 'M4': 1.6, 'M5': 0.9, 'M6': 1.3, 'M7': 1.6,
        'M8': 1.5, 'M9': 1.4, 'M10': 1.4, 'M11': 1.4, 'M12': 1.5, 'M13': 1.4, 'M14': 1.3,
        'M15': 1.2, 'M16': 1.7, 'M17': 1.5, 'M18': 1.5, 'M19': 1.5, 'M20': 1.6,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.4
    },

    # Special Situation Group (2)
    '전시장-극저유동성-고위험형': {
        'M1': 0.5, 'M2': 0.1, 'M3': 0.1, 'M4': 0.3, 'M5': 0.4, 'M6': 0.6, 'M7': 0.3,
        'M8': 0.2, 'M9': 0.4, 'M10': 0.3, 'M11': 0.3, 'M12': 0.2, 'M13': 0.5, 'M14': 0.3,
        'M15': 0.5, 'M16': 0.3, 'M17': 0.2, 'M18': 0.1, 'M19': 0.1, 'M20': 0.3,
        'M21': 0.0, 'M22': 0.0, 'M23': 0.3
    },
    '테마특화-모멘텀폭발형': {
        'M1': 3.0, 'M2': 0.5, 'M3': 0.5, 'M4': 2.5, 'M5': 2.0, 'M6': 1.7, 'M7': 1.9,
        'M8': 2.0, 'M9': 1.8, 'M10': 1.8, 'M11': 1.8, 'M12': 2.0, 'M13': 1.8, 'M14': 1.8,
        'M15': 1.6, 'M16': 2.0, 'M17': 2.0, 'M18': 2.0, 'M19': 2.0, 'M20': 2.0,
        'M21': 0.0, 'M22': 0.0, 'M23': 2.5
    },

    # Others (fallback)
    '기타': {
        'M1': 1.5, 'M2': 0.3, 'M3': 0.2, 'M4': 1.0, 'M5': 0.8, 'M6': 1.0, 'M7': 1.0,
        'M8': 1.0, 'M9': 1.0, 'M10': 1.0, 'M11': 1.0, 'M12': 1.0, 'M13': 1.0, 'M14': 1.0,
        'M15': 1.0, 'M16': 1.0, 'M17': 1.0, 'M18': 1.0, 'M19': 1.0, 'M20': 1.0,
        'M21': 0.0, 'M22': 0.0, 'M23': 1.0
    }
}


class MomentumFactorCalculator:
    """Calculate momentum factor scores using 20 different strategies"""

    def __init__(self, symbol, db_manager, market_state=None, analysis_date=None, market_cap_class=None):
        """
        Initialize Momentum Factor Calculator

        Args:
            symbol: Stock symbol
            db_manager: AsyncDatabaseManager instance
            market_state: Market state classification (optional)
            analysis_date: Specific date for analysis (optional, defaults to latest)
            market_cap_class: Market cap classification - MEGA/LARGE/MEDIUM/SMALL (optional)
        """
        self.symbol = symbol
        self.db_manager = db_manager
        self.market_state = market_state
        self.analysis_date = analysis_date
        self.market_cap_class = market_cap_class
        self.strategies_scores = {}

    async def execute_query(self, query, *params):
        """
        Execute SQL query using async database manager

        Args:
            query: SQL query string
            *params: Query parameters (unpacked)

        Returns:
            List of dict rows or None if error
        """
        try:
            return await self.db_manager.execute_query(query, *params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            return None

    async def _get_market_cap_class(self):
        """
        Get market cap classification for the symbol

        Returns:
            str: MEGA, LARGE, MEDIUM, or SMALL
        """
        if self.market_cap_class:
            return self.market_cap_class

        # Query market cap from kr_intraday_total
        if self.analysis_date:
            date_condition = f"AND date = '{self.analysis_date}'"
        else:
            date_condition = "ORDER BY date DESC LIMIT 1"

        query = f"""
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = '{self.symbol}'
        {date_condition}
        """

        result = await self.execute_query(query)

        if not result or len(result) == 0:
            return 'MEDIUM'  # Default

        market_cap = float(result[0]['market_cap']) if result[0]['market_cap'] else 0

        # Classification
        if market_cap >= 10_000_000_000_000:  # 10조
            return 'MEGA'
        elif market_cap >= 1_000_000_000_000:  # 1조
            return 'LARGE'
        elif market_cap >= 200_000_000_000:  # 2000억
            return 'MEDIUM'
        else:
            return 'SMALL'

    async def _get_sector_multiplier(self):
        """
        Get sector-based momentum multiplier for the symbol
        Uses existing kr_stock_detail.industry column

        Returns:
            float: Sector multiplier (0.5 to 1.5, default 1.0)
        """
        # Query sector/industry from kr_stock_detail
        query = """
        SELECT industry
        FROM kr_stock_detail
        WHERE symbol = $1
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or not result[0]['industry']:
            return 1.0  # Default neutral multiplier

        industry = result[0]['industry']

        # Return sector multiplier if exists, otherwise default
        return SECTOR_MULTIPLIERS.get(industry, 1.0)

    # ========================================================================
    # M1. Price Momentum Strategy
    # ========================================================================

    async def calculate_m1(self):
        """
        M1. Price Momentum Strategy
        Description: Comprehensive evaluation of short/mid/long-term return trends
        1-month, 3-month, 6-month, 12-month returns
        """
        query = """
        WITH price_data AS (
            SELECT
                date,
                close,
                LAG(close, 20) OVER (ORDER BY date) as close_20d,
                LAG(close, 60) OVER (ORDER BY date) as close_60d,
                LAG(close, 120) OVER (ORDER BY date) as close_120d,
                LAG(close, 240) OVER (ORDER BY date) as close_240d
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            close as current_price,
            close_20d,
            close_60d,
            close_120d,
            close_240d
        FROM price_data
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['current_price']:
            return None

        row = result[0]
        current = float(row['current_price'])

        # Calculate returns
        returns = {}
        if row['close_20d']:
            returns['1m'] = ((current - float(row['close_20d'])) / float(row['close_20d'])) * 100
        if row['close_60d']:
            returns['3m'] = ((current - float(row['close_60d'])) / float(row['close_60d'])) * 100
        if row['close_120d']:
            returns['6m'] = ((current - float(row['close_120d'])) / float(row['close_120d'])) * 100

        if not returns:
            return None

        # Calculate scores
        scores = []
        if '1m' in returns:
            short_score = min(100, max(0, returns['1m'] * 2 + 50))
            scores.append(('short', short_score, 0.3))

        if '3m' in returns:
            mid_score = min(100, max(0, returns['3m'] + 50))
            scores.append(('mid', mid_score, 0.4))

        if '6m' in returns:
            long_score = min(100, max(0, returns['6m'] * 0.5 + 50))
            scores.append(('long', long_score, 0.3))

        if not scores:
            return None

        # Weighted average
        total_weight = sum(s[2] for s in scores)
        weighted_score = sum(s[1] * s[2] for s in scores) / total_weight

        return weighted_score

    # ========================================================================
    # Small Cap Normalized Calculation Methods
    # ========================================================================

    async def _calculate_m2_small_cap(self):
        """
        M2 Small Cap Normalized: Earnings Momentum with volatility adjustment
        - Z-score normalization for earnings volatility
        - Outlier filtering (±3σ)
        - Persistence checking (trend over multiple quarters)
        """
        # Get EPS data
        eps_query = """
        SELECT eps
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        eps_result = await self.execute_query(eps_query, self.symbol)

        current_eps = None
        if eps_result and eps_result[0]['eps'] is not None:
            current_eps = float(eps_result[0]['eps'])

        # Get operating profit with quarterly data for persistence check
        op_query = """
        WITH quarterly_data AS (
            SELECT
                bsns_year,
                rcept_dt,
                thstrm_amount,
                frmtrm_amount,
                ROW_NUMBER() OVER (ORDER BY bsns_year DESC, rcept_dt DESC) as rn
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND thstrm_amount > 0
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        )
        SELECT
            thstrm_amount as current_op,
            frmtrm_amount as prev_op,
            rn
        FROM quarterly_data
        WHERE rn <= 4  -- Last 4 quarters for persistence check
        ORDER BY rn
        """

        op_result = await self.execute_query(op_query, self.symbol, self.analysis_date)

        if not op_result or len(op_result) == 0:
            # Fallback to EPS if no operating profit data
            if current_eps and current_eps > 0:
                eps_score = min(100, current_eps * 5)
                return eps_score
            return None

        # Calculate growth rates for multiple quarters
        growth_rates = []
        for row in op_result:
            current_op = float(row['current_op'])
            prev_op = float(row['prev_op']) if row['prev_op'] else None

            if prev_op and prev_op != 0:
                growth = ((current_op - prev_op) / abs(prev_op)) * 100
                growth_rates.append(growth)

        if not growth_rates:
            return None

        # Z-score normalization to handle volatility
        import statistics
        mean_growth = statistics.mean(growth_rates)

        if len(growth_rates) > 1:
            std_growth = statistics.stdev(growth_rates)

            # Outlier filtering: remove ±3σ outliers
            filtered_growth = [g for g in growth_rates if abs(g - mean_growth) <= 3 * std_growth]

            if filtered_growth:
                mean_growth = statistics.mean(filtered_growth)
                std_growth = statistics.stdev(filtered_growth) if len(filtered_growth) > 1 else std_growth
        else:
            std_growth = 50  # Default for single data point

        # Z-score calculation
        if std_growth > 0:
            z_score = mean_growth / std_growth
        else:
            z_score = 0

        # Convert Z-score to 0-100 scale
        # Z-score range: -3 to +3 -> 0 to 100
        normalized_score = min(100, max(0, (z_score + 3) / 6 * 100))

        # Persistence bonus: if trend is consistent
        positive_quarters = sum(1 for g in growth_rates if g > 0)
        persistence_ratio = positive_quarters / len(growth_rates)

        if persistence_ratio >= 0.75:  # 3/4 quarters positive
            persistence_bonus = 20
        elif persistence_ratio >= 0.5:  # 2/4 quarters positive
            persistence_bonus = 10
        else:
            persistence_bonus = 0

        final_score = min(100, normalized_score + persistence_bonus)

        return final_score

    async def _calculate_m3_small_cap(self):
        """
        M3 Small Cap Normalized: 52-Week High with shorter lookback
        - Shorter lookback period (26 weeks instead of 52)
        - Volume ratio filtering
        - Liquidity penalty for low-volume stocks
        """
        query = """
        WITH price_range AS (
            SELECT
                MAX(high) as week26_high,
                MIN(low) as week26_low,
                AVG(volume) as avg_volume
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= COALESCE($3::date, CURRENT_DATE) - INTERVAL '182 days'  -- ~26 weeks
                AND date <= COALESCE($3::date, CURRENT_DATE)
        ),
        current_data AS (
            SELECT close, volume
            FROM kr_intraday_total
            WHERE symbol = $2
                AND ($3::date IS NULL OR date = $3)
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            pr.week26_high,
            pr.week26_low,
            pr.avg_volume,
            cd.close as current_price,
            cd.volume as current_volume
        FROM price_range pr, current_data cd
        """

        result = await self.execute_query(query, self.symbol, self.symbol, self.analysis_date)

        if not result or not result[0]['week26_high'] or not result[0]['current_price']:
            return None

        high_26w = float(result[0]['week26_high'])
        low_26w = float(result[0]['week26_low'])
        current = float(result[0]['current_price'])
        avg_volume = float(result[0]['avg_volume']) if result[0]['avg_volume'] else 0
        current_volume = float(result[0]['current_volume']) if result[0]['current_volume'] else 0

        if high_26w == low_26w:
            return None

        # Calculate proximity to 26-week high
        high_ratio = (current / high_26w) * 100

        if current >= high_26w:
            high_score = 100
        elif high_ratio >= 95:
            high_score = 80
        elif high_ratio >= 90:
            high_score = 60
        else:
            high_score = max(0, high_ratio - 40)

        # Calculate range position
        range_position = min(100, max(0, ((current - low_26w) / (high_26w - low_26w)) * 100))

        # Volume ratio check
        if avg_volume > 0 and current_volume > 0:
            volume_ratio = current_volume / avg_volume

            if volume_ratio >= 1.5:  # Strong volume confirmation
                volume_mult = 1.2
            elif volume_ratio >= 1.0:  # Normal volume
                volume_mult = 1.0
            elif volume_ratio >= 0.5:  # Weak volume
                volume_mult = 0.8
            else:  # Very weak volume - penalty
                volume_mult = 0.6
        else:
            volume_mult = 0.8  # Default penalty for missing data

        # Liquidity penalty for very low-volume stocks
        if avg_volume < 10000:  # Less than 10k shares per day
            liquidity_penalty = 0.7
        elif avg_volume < 50000:  # Less than 50k shares per day
            liquidity_penalty = 0.85
        else:
            liquidity_penalty = 1.0

        # Calculate score with adjustments
        base_score = (high_score * 0.6) + (range_position * 0.4)
        adjusted_score = base_score * volume_mult * liquidity_penalty

        return min(100, adjusted_score)

    async def _calculate_m5_small_cap(self):
        """
        M5 Small Cap Normalized: Volume Momentum with shorter MA periods
        - Shorter MA periods (5/10/20/60 instead of 5/20/60/120)
        - Volatility adjustment
        - Volume filtering for noise reduction
        - Price filtering (low-priced stock penalty)
        """
        query = """
        WITH volume_data AS (
            SELECT
                volume,
                close,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as avg_vol_5d,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as avg_vol_10d,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_vol_20d,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as avg_vol_60d,
                STDDEV(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as vol_stddev,
                change_rate,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            volume as current_volume,
            close as current_price,
            avg_vol_5d,
            avg_vol_10d,
            avg_vol_20d,
            avg_vol_60d,
            vol_stddev,
            change_rate
        FROM volume_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['current_volume']:
            return None

        row = result[0]
        current_vol = float(row['current_volume'])
        current_price = float(row['current_price']) if row['current_price'] else 0
        avg_5d = float(row['avg_vol_5d']) if row['avg_vol_5d'] else current_vol
        avg_10d = float(row['avg_vol_10d']) if row['avg_vol_10d'] else current_vol
        avg_20d = float(row['avg_vol_20d']) if row['avg_vol_20d'] else current_vol
        avg_60d = float(row['avg_vol_60d']) if row['avg_vol_60d'] else current_vol
        vol_stddev = float(row['vol_stddev']) if row['vol_stddev'] else 0
        change_rate = float(row['change_rate']) if row['change_rate'] else 0

        # Volume alignment score (5 > 10 > 20 > 60 day averages)
        alignment_checks = 0
        total_checks = 0

        if avg_5d > avg_10d:
            alignment_checks += 1
        total_checks += 1

        if avg_10d > avg_20d:
            alignment_checks += 1
        total_checks += 1

        if avg_20d > avg_60d:
            alignment_checks += 1
        total_checks += 1

        alignment_score = (alignment_checks / total_checks) * 100 if total_checks > 0 else 50

        # Volatility adjustment: penalize extremely volatile volume
        if vol_stddev > 0 and avg_20d > 0:
            volatility_ratio = vol_stddev / avg_20d

            if volatility_ratio > 2.0:  # Very high volatility - noise
                volatility_mult = 0.6
            elif volatility_ratio > 1.0:  # High volatility
                volatility_mult = 0.8
            else:  # Normal volatility
                volatility_mult = 1.0
        else:
            volatility_mult = 1.0

        # Volume filtering: penalize very low volume (noise)
        if avg_20d < 5000:  # Less than 5k shares/day
            volume_filter = 0.5
        elif avg_20d < 20000:  # Less than 20k shares/day
            volume_filter = 0.7
        elif avg_20d < 50000:  # Less than 50k shares/day
            volume_filter = 0.85
        else:
            volume_filter = 1.0

        # Price filtering: low-priced stocks (<1000 won) get penalty
        if current_price < 1000:
            price_filter = 0.7
        elif current_price < 2000:
            price_filter = 0.85
        else:
            price_filter = 1.0

        # Short-term momentum (current vs 5-day average)
        if avg_5d > 0:
            short_momentum = (current_vol / avg_5d - 1) * 100
            short_score = min(100, max(0, short_momentum + 50))
        else:
            short_score = 50

        # Price direction multiplier (volume increase with price up is better)
        if change_rate > 0:
            price_mult = 1.3
        elif change_rate > -2:  # Small decline is OK
            price_mult = 1.0
        else:  # Large decline - reduce score
            price_mult = 0.7

        # Combine all factors
        base_score = (alignment_score * 0.6) + (short_score * 0.4)
        final_score = base_score * volatility_mult * volume_filter * price_filter * price_mult

        return min(100, final_score)

    # ========================================================================
    # M2. Earnings Momentum Strategy
    # ========================================================================

    async def calculate_m2(self):
        """
        M2. Earnings Momentum Strategy
        Description: Earnings improvement trend and earnings surprise
        QoQ and YoY EPS growth, operating profit growth

        Routes to normalized version for Small cap stocks
        """
        # Check market cap class and route accordingly
        market_cap_class = await self._get_market_cap_class()

        if market_cap_class == 'SMALL':
            return await self._calculate_m2_small_cap()

        # Standard calculation for MEGA/LARGE/MEDIUM
        # Get EPS data
        eps_query = """
        SELECT eps
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        eps_result = await self.execute_query(eps_query, self.symbol)

        current_eps = None
        if eps_result and eps_result[0]['eps'] is not None:
            current_eps = float(eps_result[0]['eps'])

        # Get operating profit growth from financial data
        op_query = """
        WITH latest_report AS (
            SELECT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND thstrm_amount > 0
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ),
        op_data AS (
            SELECT thstrm_amount, frmtrm_amount,
                   ROW_NUMBER() OVER (
                       ORDER BY
                           CASE
                               WHEN report_code = '11011' THEN 1
                               WHEN report_code = '11012' THEN 2
                               WHEN report_code = '11013' THEN 3
                               WHEN report_code = '11014' THEN 4
                               ELSE 5
                           END,
                           thstrm_amount DESC
                   ) as rn
            FROM kr_financial_position, latest_report
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND thstrm_amount > 0
                AND kr_financial_position.bsns_year = latest_report.bsns_year
                AND kr_financial_position.rcept_dt = latest_report.rcept_dt
        )
        SELECT
            thstrm_amount as current_op,
            frmtrm_amount as prev_op
        FROM op_data
        WHERE rn = 2
        """

        op_result = await self.execute_query(op_query, self.symbol, self.analysis_date)

        scores = []

        # Operating profit growth score
        if op_result and op_result[0]['current_op']:
            current_op = float(op_result[0]['current_op'])
            prev_op = float(op_result[0]['prev_op']) if op_result[0]['prev_op'] else None

            if prev_op and prev_op != 0:
                # Calculate growth rate if prev data available
                op_growth = ((current_op - prev_op) / abs(prev_op)) * 100
                op_score = min(100, max(0, op_growth + 50))
                scores.append(op_score)
            elif current_op > 0:
                # Fallback: Use absolute value of operating profit (scaled)
                # Assume good companies have OP > 50B KRW
                op_score = min(100, (current_op / 50_000_000_000) * 50)
                scores.append(op_score)

        # If no operating profit data, use EPS as proxy
        if not scores and current_eps:
            if current_eps > 0:
                eps_score = min(100, current_eps * 5)
                scores.append(eps_score)

        if not scores:
            return None

        return sum(scores) / len(scores)

    # ========================================================================
    # M3. 52-Week High Proximity Strategy
    # ========================================================================

    async def calculate_m3(self):
        """
        M3. 52-Week High Proximity Strategy
        Description: Stocks breaking or approaching 52-week high

        Routes to normalized version for Small cap stocks
        """
        # Check market cap class and route accordingly
        market_cap_class = await self._get_market_cap_class()

        if market_cap_class == 'SMALL':
            return await self._calculate_m3_small_cap()

        # Standard calculation for MEGA/LARGE/MEDIUM
        query = """
        WITH price_range AS (
            SELECT
                MAX(high) as week52_high,
                MIN(low) as week52_low
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= COALESCE($3::date, CURRENT_DATE) - INTERVAL '365 days'
                AND date <= COALESCE($3::date, CURRENT_DATE)
        ),
        current_price AS (
            SELECT close
            FROM kr_intraday_total
            WHERE symbol = $2
                AND ($3::date IS NULL OR date = $3)
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            pr.week52_high,
            pr.week52_low,
            cp.close as current_price
        FROM price_range pr, current_price cp
        """

        result = await self.execute_query(query, self.symbol, self.symbol, self.analysis_date)

        if not result or not result[0]['week52_high'] or not result[0]['current_price']:
            return None

        high_52w = float(result[0]['week52_high'])
        low_52w = float(result[0]['week52_low'])
        current = float(result[0]['current_price'])

        if high_52w == low_52w:
            return None

        # Calculate proximity to 52-week high
        high_ratio = (current / high_52w) * 100

        if current >= high_52w:
            high_score = 100
        elif high_ratio >= 95:
            high_score = 80
        elif high_ratio >= 90:
            high_score = 60
        else:
            high_score = max(0, high_ratio - 40)

        # Calculate range position (limit to 0-100)
        range_position = min(100, max(0, ((current - low_52w) / (high_52w - low_52w)) * 100))

        # Final score (max: 100*0.6 + 100*0.4 = 100)
        score = (high_score * 0.6) + (range_position * 0.4)

        return score

    # ========================================================================
    # M4. Relative Strength Strategy
    # ========================================================================

    async def calculate_m4(self):
        """
        M4. Relative Strength (RS) Strategy
        Description: Relative performance vs market (KOSPI)
        """
        # Get stock 20-day return
        stock_query = """
        WITH price_data AS (
            SELECT
                close,
                LAG(close, 20) OVER (ORDER BY date) as close_20d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND close IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            close,
            close_20d
        FROM price_data
        WHERE rn = 1
        """

        stock_result = await self.execute_query(stock_query, self.symbol, self.analysis_date)

        if not stock_result or not stock_result[0]['close'] or not stock_result[0]['close_20d']:
            return None

        current = float(stock_result[0]['close'])
        prev_20d = float(stock_result[0]['close_20d'])
        stock_return = ((current - prev_20d) / prev_20d) * 100

        # Get KOSPI 20-day return from market_index
        market_query = """
        WITH market_data AS (
            SELECT
                close,
                LAG(close, 20) OVER (ORDER BY date) as close_20d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM market_index
            WHERE exchange = 'KOSPI'
        )
        SELECT
            close,
            close_20d
        FROM market_data
        WHERE rn = 1
        """

        market_result = await self.execute_query(market_query)

        if not market_result or not market_result[0]['close'] or not market_result[0]['close_20d']:
            # Fallback: use 0% market return if no data
            market_return = 0
        else:
            market_current = float(market_result[0]['close'])
            market_prev_20d = float(market_result[0]['close_20d'])
            market_return = ((market_current - market_prev_20d) / market_prev_20d) * 100

        # Calculate relative strength
        market_rs = stock_return - market_return

        # Score: RS > 0 is good, scale -50 to +50 to 0-100
        market_score = min(100, max(0, market_rs * 2 + 50))

        return market_score

    # ========================================================================
    # M5. Volume Momentum Strategy
    # ========================================================================

    async def calculate_m5(self):
        """
        M5. Volume Momentum Strategy
        Description: Healthy momentum with increasing volume

        Routes to normalized version for Small cap stocks
        """
        # Check market cap class and route accordingly
        market_cap_class = await self._get_market_cap_class()

        if market_cap_class == 'SMALL':
            return await self._calculate_m5_small_cap()

        # Standard calculation for MEGA/LARGE/MEDIUM
        query = """
        WITH volume_data AS (
            SELECT
                volume,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_vol_20d,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as avg_vol_60d,
                change_rate,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            volume as current_volume,
            avg_vol_20d,
            avg_vol_60d,
            change_rate
        FROM volume_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['current_volume']:
            return None

        row = result[0]
        current_vol = float(row['current_volume'])
        avg_20d = float(row['avg_vol_20d']) if row['avg_vol_20d'] else current_vol
        avg_60d = float(row['avg_vol_60d']) if row['avg_vol_60d'] else current_vol
        change_rate = float(row['change_rate']) if row['change_rate'] else 0

        # Volume ratio
        vol_ratio = (current_vol / avg_20d) * 100 if avg_20d > 0 else 100
        vol_score = min(100, max(0, (vol_ratio - 100) + 50))

        # Volume trend
        vol_trend = (avg_20d / avg_60d) * 100 if avg_60d > 0 else 100
        trend_score = min(100, max(0, (vol_trend - 100) * 2 + 50))

        # Price direction multiplier
        price_mult = 1.5 if change_rate > 0 else 0.5

        score = ((vol_score * 0.5) + (trend_score * 0.5)) * price_mult

        return min(100, score)

    # ========================================================================
    # M6. Institutional/Foreign Buying Momentum Strategy
    # ========================================================================

    async def calculate_m6(self):
        """
        M6. Institutional/Foreign Buying Momentum Strategy
        Description: Continuous net buying by smart money
        """
        query = """
        SELECT
            SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '10 days' THEN inst_net_value ELSE 0 END) as inst_10d,
            SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '30 days' THEN inst_net_value ELSE 0 END) as inst_30d,
            SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '10 days' THEN foreign_net_value ELSE 0 END) as foreign_10d,
            SUM(CASE WHEN date >= CURRENT_DATE - INTERVAL '30 days' THEN foreign_net_value ELSE 0 END) as foreign_30d
        FROM kr_individual_investor_daily_trading
        WHERE symbol = $1
        """

        result = await self.execute_query(query, self.symbol)

        if not result:
            return None

        row = result[0]
        inst_10d = float(row['inst_10d']) if row['inst_10d'] else 0
        inst_30d = float(row['inst_30d']) if row['inst_30d'] else 0
        foreign_10d = float(row['foreign_10d']) if row['foreign_10d'] else 0
        foreign_30d = float(row['foreign_30d']) if row['foreign_30d'] else 0

        # Get market cap for ratio calculation
        mktcap_query = """
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        mktcap_result = await self.execute_query(mktcap_query, self.symbol, self.analysis_date)

        if not mktcap_result or not mktcap_result[0]['market_cap']:
            market_cap = 100000000000  # Default 100B won
        else:
            market_cap = float(mktcap_result[0]['market_cap'])

        # Calculate scores
        inst_score = min(50, max(0, (inst_10d / market_cap) * 1000 + 25))
        foreign_score = min(50, max(0, (foreign_10d / market_cap) * 1000 + 25))

        # Trend multipliers
        inst_mult = 1.2 if inst_10d > 0 and inst_30d > 0 else 0.8
        foreign_mult = 1.2 if foreign_10d > 0 and foreign_30d > 0 else 0.8

        score = (inst_score * inst_mult) + (foreign_score * foreign_mult)

        return min(100, score)

    # ========================================================================
    # M7. Moving Average Alignment Strategy
    # ========================================================================

    async def calculate_m7(self):
        """
        M7. Moving Average Alignment Strategy (Korean Market Optimized)
        Description: Golden alignment of short/mid/long-term MAs

        한국 시장 특성 반영:
        - 완전 정배열은 과열 신호로 해석 (평균 회귀 가능성)
        - 초기 정배열(2단계)이 최적 진입점
        - 이격도가 낮을수록 안전한 진입
        - 거래량 확인으로 신뢰도 보정
        """
        query = """
        WITH ma_data AS (
            SELECT
                close,
                volume,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ma5,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as ma60,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW) as ma120,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_vol_20d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            close as current_price,
            volume,
            ma5,
            ma20,
            ma60,
            ma120,
            avg_vol_20d
        FROM ma_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['current_price']:
            return None

        row = result[0]
        price = float(row['current_price'])
        volume = float(row['volume']) if row['volume'] else 0
        ma5 = float(row['ma5']) if row['ma5'] else price
        ma20 = float(row['ma20']) if row['ma20'] else price
        ma60 = float(row['ma60']) if row['ma60'] else price
        ma120 = float(row['ma120']) if row['ma120'] else price
        avg_vol_20d = float(row['avg_vol_20d']) if row['avg_vol_20d'] else volume

        # 정배열 점수 - 한국 시장 평균회귀 특성 반영
        # 완전 정배열 = 과열 위험, 초기 정배열 = 최적 진입점
        if price > ma5 and ma5 > ma20 and ma20 > ma60 and ma60 > ma120:
            # 완전 정배열: 이미 과열 상태, 평균 회귀 가능성 높음
            alignment_score = 30
        elif price > ma5 and ma5 > ma20 and ma20 > ma60:
            # 3단계 정배열: 추세 성숙기, 주의 필요
            alignment_score = 55
        elif price > ma5 and ma5 > ma20:
            # 2단계 정배열: 최적 진입점 (초기 추세)
            alignment_score = 75
        elif price > ma20:
            # MA20 위: 기본 상승 추세
            alignment_score = 60
        elif price > ma60:
            # MA60 위: 중기 지지, 반등 가능성
            alignment_score = 45
        else:
            # 역배열: 하락 추세
            alignment_score = 20

        # 이격도 점수 - 낮은 이격도가 안전한 진입
        if ma20 > 0:
            distance = ((price - ma20) / ma20) * 100
            if 0 <= distance <= 3:
                # 이격도 0-3%: 안전한 진입 구간
                distance_score = 80
            elif 3 < distance <= 7:
                # 이격도 3-7%: 양호
                distance_score = 65
            elif 7 < distance <= 10:
                # 이격도 7-10%: 과열 주의
                distance_score = 40
            elif distance > 10:
                # 이격도 10% 초과: 과열 위험
                distance_score = 20
            elif -5 <= distance < 0:
                # 이격도 -5~0%: MA20 근접, 지지선 테스트
                distance_score = 70
            else:
                # 이격도 -5% 미만: 하락 추세
                distance_score = 30
        else:
            distance_score = 50

        # 거래량 확인 배수 (Volume Confirmation Multiplier)
        if avg_vol_20d > 0:
            vol_ratio = volume / avg_vol_20d
            if 1.0 <= vol_ratio <= 1.5:
                # 적정 거래량 증가: 건강한 상승
                vol_mult = 1.1
            elif 0.7 <= vol_ratio < 1.0:
                # 평균 수준: 중립
                vol_mult = 1.0
            elif vol_ratio > 2.0:
                # 과도한 거래량: 세력 이탈 또는 급등 과열
                vol_mult = 0.85
            elif vol_ratio > 1.5:
                # 높은 거래량: 주의
                vol_mult = 0.95
            else:
                # 낮은 거래량: 신뢰도 낮음
                vol_mult = 0.9
        else:
            vol_mult = 1.0

        # 최종 점수 계산
        base_score = (alignment_score * 0.6) + (distance_score * 0.4)
        score = base_score * vol_mult

        return min(100, max(0, score))

    # ========================================================================
    # M8. Breakout Momentum Strategy
    # ========================================================================

    async def calculate_m8(self):
        """
        M8. Breakout Momentum Strategy
        Description: Breaking through major resistance and box range
        """
        query = """
        WITH price_range AS (
            SELECT
                MAX(high) as recent_high,
                MIN(low) as recent_low
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date BETWEEN COALESCE($3::date, CURRENT_DATE) - INTERVAL '60 days' AND COALESCE($3::date, CURRENT_DATE) - INTERVAL '5 days'
        ),
        current_data AS (
            SELECT
                close,
                volume,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_vol_20d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $2
                AND ($3::date IS NULL OR date <= $3)
        )
        SELECT
            pr.recent_high,
            pr.recent_low,
            cd.close as current_price,
            cd.volume,
            cd.avg_vol_20d
        FROM price_range pr, current_data cd
        WHERE cd.rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.symbol, self.analysis_date)

        if not result or not result[0]['recent_high'] or not result[0]['current_price']:
            return None

        row = result[0]
        recent_high = float(row['recent_high'])
        current = float(row['current_price'])
        volume = float(row['volume']) if row['volume'] else 0
        avg_vol = float(row['avg_vol_20d']) if row['avg_vol_20d'] else volume

        box_top = recent_high * 0.98

        # Breakout score
        if current > recent_high:
            breakout_score = 100
        elif current > box_top:
            breakout_score = 70
        else:
            breakout_score = 0

        # Breakout strength
        breakout_strength = ((current - recent_high) / recent_high) * 100 if recent_high > 0 else 0
        strength_score = min(100, max(0, breakout_strength * 10 + 50))

        # Volume confirmation
        vol_mult = 1.2 if volume > avg_vol * 1.5 else 0.8

        score = ((breakout_score * 0.6) + (strength_score * 0.4)) * vol_mult

        return min(100, score)

    # ========================================================================
    # M9. RSI Momentum Strategy
    # ========================================================================

    async def calculate_m9(self):
        """
        M9. RSI Momentum Strategy (Korean Market Optimized)
        Description: RSI reversal detection in oversold zones

        한국 시장 특성 반영:
        - RSI 30-42 과매도 탈출 구간이 최적 매수점
        - RSI 65 이상은 과열 위험 (평균회귀 예상)
        - 강세 다이버전스 (가격↓ + RSI↑) = 최적 매수 신호
        - 거래량 확인으로 반등 신뢰도 검증
        """
        # Get RSI data from kr_indicators
        rsi_query = """
        WITH rsi_data AS (
            SELECT
                rsi,
                LAG(rsi, 1) OVER (ORDER BY date) as rsi_1d_ago,
                LAG(rsi, 5) OVER (ORDER BY date) as rsi_5d_ago,
                LAG(rsi, 10) OVER (ORDER BY date) as rsi_10d_ago,
                MIN(rsi) OVER (ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as rsi_10d_min,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_indicators
            WHERE symbol = $1
                AND rsi IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            rsi as current_rsi,
            rsi_1d_ago,
            rsi_5d_ago,
            rsi_10d_ago,
            rsi_10d_min
        FROM rsi_data
        WHERE rn = 1
        """

        rsi_result = await self.execute_query(rsi_query, self.symbol, self.analysis_date)

        if not rsi_result or rsi_result[0]['current_rsi'] is None:
            return None

        rsi_row = rsi_result[0]
        rsi = float(rsi_row['current_rsi'])
        rsi_1d = float(rsi_row['rsi_1d_ago']) if rsi_row['rsi_1d_ago'] else rsi
        rsi_5d = float(rsi_row['rsi_5d_ago']) if rsi_row['rsi_5d_ago'] else rsi
        rsi_10d = float(rsi_row['rsi_10d_ago']) if rsi_row['rsi_10d_ago'] else rsi
        rsi_10d_min = float(rsi_row['rsi_10d_min']) if rsi_row['rsi_10d_min'] else rsi

        # Get price and volume data from kr_intraday_total
        price_query = """
        WITH price_data AS (
            SELECT
                close,
                volume,
                LAG(close, 5) OVER (ORDER BY date) as close_5d_ago,
                LAG(close, 10) OVER (ORDER BY date) as close_10d_ago,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_vol_20d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            close as current_price,
            volume,
            close_5d_ago,
            close_10d_ago,
            avg_vol_20d
        FROM price_data
        WHERE rn = 1
        """

        price_result = await self.execute_query(price_query, self.symbol, self.analysis_date)

        if price_result and price_result[0]['current_price']:
            price = float(price_result[0]['current_price'])
            volume = float(price_result[0]['volume']) if price_result[0]['volume'] else 0
            price_5d = float(price_result[0]['close_5d_ago']) if price_result[0]['close_5d_ago'] else price
            price_10d = float(price_result[0]['close_10d_ago']) if price_result[0]['close_10d_ago'] else price
            avg_vol_20d = float(price_result[0]['avg_vol_20d']) if price_result[0]['avg_vol_20d'] else volume
        else:
            price, volume, price_5d, price_10d, avg_vol_20d = 0, 0, 0, 0, 0

        # 1. RSI 레벨 점수 - 한국 시장 평균회귀 특성 반영
        if 30 <= rsi <= 42:
            # 과매도 탈출 구간 = 최적 매수점
            level_score = 80
        elif 42 < rsi <= 55:
            # 안정 구간
            level_score = 70
        elif 55 < rsi <= 65:
            # 주의 구간
            level_score = 50
        elif rsi > 65:
            # 과열 위험 (70 이상은 더 위험)
            level_score = 25 if rsi <= 70 else 15
        elif rsi < 30:
            # 과매도 지속 (반등 대기 또는 추가 하락)
            level_score = 55 if rsi >= 25 else 40

        # 2. RSI 반전 감지 (Reversal Detection)
        rsi_recovering = rsi > rsi_1d and rsi > rsi_5d
        rsi_from_oversold = rsi_10d_min <= 35 and rsi > rsi_10d_min + 5

        if rsi_from_oversold and rsi_recovering and rsi <= 50:
            # 황금 진입점: 과매도에서 반등 시작, 아직 중립 이하
            reversal_score = 90
        elif rsi_from_oversold and rsi <= 55:
            # 양호 진입점: 과매도 탈출 중
            reversal_score = 75
        elif rsi_recovering and rsi <= 45:
            # 초기 반등 (과매도 아니었지만 상승 전환)
            reversal_score = 65
        elif rsi > 65 and rsi > rsi_5d:
            # 과열 구간에서 계속 상승 = 위험
            reversal_score = 30
        elif rsi > 70:
            # 극단적 과열
            reversal_score = 20
        else:
            # 중립
            reversal_score = 50

        # 3. 다이버전스 점수 (Divergence Score)
        if price > 0 and price_5d > 0:
            price_change = (price - price_5d) / price_5d
            rsi_change = rsi - rsi_5d

            if price_change < -0.02 and rsi_change > 3:
                # 강세 다이버전스: 가격 하락 but RSI 상승 = 최적 매수 신호
                div_score = 95
            elif price_change < 0 and rsi_change > 0:
                # 약한 강세 다이버전스
                div_score = 75
            elif price_change > 0.02 and rsi_change < -3:
                # 약세 다이버전스: 가격 상승 but RSI 하락 = 매도 신호
                div_score = 25
            elif price_change > 0 and rsi_change < 0:
                # 약한 약세 다이버전스
                div_score = 40
            elif price_change > 0 and rsi_change > 0:
                # 동조화 (둘 다 상승) - 추가 상승 여력 제한적
                div_score = 55
            else:
                # 둘 다 하락 또는 중립
                div_score = 50
        else:
            div_score = 50

        # 4. 거래량 확인 (과매도 반등 시 중요)
        if avg_vol_20d > 0 and volume > 0:
            vol_ratio = volume / avg_vol_20d
            if rsi <= 45 and vol_ratio >= 1.3:
                # 과매도 구간에서 거래량 증가 = 반등 신뢰
                vol_mult = 1.15
            elif rsi <= 45 and vol_ratio >= 1.0:
                vol_mult = 1.05
            elif rsi > 65 and vol_ratio >= 1.5:
                # 과열 구간에서 거래량 폭증 = 세력 이탈 가능
                vol_mult = 0.85
            else:
                vol_mult = 1.0
        else:
            vol_mult = 1.0

        # 최종 점수 계산
        base_score = (level_score * 0.35) + (reversal_score * 0.35) + (div_score * 0.30)
        score = base_score * vol_mult

        return min(100, max(0, score))

    # ========================================================================
    # M10. MACD Momentum Strategy
    # ========================================================================

    async def calculate_m10(self):
        """
        M10. MACD Momentum Strategy (Korean Market Optimized)
        Description: MACD golden cross position and histogram reversal

        한국 시장 특성 반영:
        - 0선 아래 골든크로스가 최적 진입점 (반등 초기)
        - 0선 위 높은 위치 골든크로스는 후반 진입 (평균회귀 위험)
        - 히스토그램 음→양 전환이 최적 매수 시점
        - MACD 극단적 양수는 과열 신호
        """
        query = """
        WITH macd_data AS (
            SELECT
                macd,
                macd_signal,
                macd_hist,
                LAG(macd, 1) OVER (ORDER BY date) as macd_1d_ago,
                LAG(macd, 5) OVER (ORDER BY date) as macd_5d_ago,
                LAG(macd_signal, 1) OVER (ORDER BY date) as signal_1d_ago,
                LAG(macd_signal, 5) OVER (ORDER BY date) as signal_5d_ago,
                LAG(macd_hist, 1) OVER (ORDER BY date) as hist_1d_ago,
                LAG(macd_hist, 3) OVER (ORDER BY date) as hist_3d_ago,
                LAG(macd_hist, 5) OVER (ORDER BY date) as hist_5d_ago,
                MAX(macd) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as macd_20d_max,
                MIN(macd) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as macd_20d_min,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_indicators
            WHERE symbol = $1
                AND macd IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            macd,
            macd_signal as signal,
            macd_hist as histogram,
            macd_1d_ago,
            macd_5d_ago,
            signal_1d_ago,
            signal_5d_ago,
            hist_1d_ago,
            hist_3d_ago,
            hist_5d_ago,
            macd_20d_max,
            macd_20d_min
        FROM macd_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or result[0]['macd'] is None:
            return None

        row = result[0]
        macd = float(row['macd'])
        signal = float(row['signal']) if row['signal'] else 0
        hist = float(row['histogram']) if row['histogram'] else 0
        macd_1d = float(row['macd_1d_ago']) if row['macd_1d_ago'] else macd
        macd_5d = float(row['macd_5d_ago']) if row['macd_5d_ago'] else macd
        signal_1d = float(row['signal_1d_ago']) if row['signal_1d_ago'] else signal
        signal_5d = float(row['signal_5d_ago']) if row['signal_5d_ago'] else signal
        hist_1d = float(row['hist_1d_ago']) if row['hist_1d_ago'] else hist
        hist_3d = float(row['hist_3d_ago']) if row['hist_3d_ago'] else hist
        hist_5d = float(row['hist_5d_ago']) if row['hist_5d_ago'] else hist
        macd_20d_max = float(row['macd_20d_max']) if row['macd_20d_max'] else macd
        macd_20d_min = float(row['macd_20d_min']) if row['macd_20d_min'] else macd

        # MACD 범위 계산 (정규화용)
        macd_range = macd_20d_max - macd_20d_min if macd_20d_max != macd_20d_min else 1

        # 1. 골든크로스 위치 기반 점수
        golden_cross_recent = macd > signal and macd_1d <= signal_1d
        golden_cross_5d = macd > signal and macd_5d <= signal_5d
        dead_cross_approaching = macd > signal and (macd - signal) < (macd_1d - signal_1d) * 0.5

        if golden_cross_recent or golden_cross_5d:
            # 골든크로스 발생 - 위치에 따라 점수 차등
            if macd < 0:
                # 0선 아래 골든크로스 = 최적 반등 시점
                cross_score = 90
            elif macd < macd_range * 0.3:
                # 0선 근처 골든크로스 = 양호
                cross_score = 75
            else:
                # 0선 위 높은 위치 = 후반 진입, 주의
                cross_score = 45
        elif macd > signal:
            # 골든크로스 이후 상승 지속
            if dead_cross_approaching:
                # 데드크로스 접근 = 경고
                cross_score = 35
            elif macd > macd_range * 0.7:
                # 너무 높은 위치 = 과열
                cross_score = 40
            else:
                cross_score = 60
        else:
            # 데드크로스 상태
            if macd > macd_1d:
                # 반등 시도 중
                cross_score = 50
            else:
                cross_score = 25

        # 2. 히스토그램 전환점 감지
        hist_turning_positive = hist > 0 and hist_1d <= 0
        hist_turning_negative = hist < 0 and hist_1d >= 0
        hist_increasing = hist > hist_1d and hist_1d > hist_3d
        hist_decreasing = hist < hist_1d and hist_1d < hist_3d

        if hist_turning_positive:
            # 히스토그램 음→양 전환 = 최적 매수 시점
            hist_score = 85
        elif hist > 0 and hist_increasing:
            # 양수이고 증가 중 = 상승 지속
            hist_score = 65
        elif hist > 0 and hist_decreasing:
            # 양수이지만 감소 = 상승 둔화, 주의
            hist_score = 40
        elif hist_turning_negative:
            # 히스토그램 양→음 전환 = 매도 신호
            hist_score = 20
        elif hist < 0 and hist_increasing:
            # 음수이지만 증가 = 반등 준비
            hist_score = 70
        elif hist < 0 and hist_decreasing:
            # 음수이고 감소 = 하락 가속
            hist_score = 25
        else:
            hist_score = 50

        # 3. MACD 레벨 점수 (과열 판단)
        if macd < 0 and macd > macd_5d:
            # 0선 아래에서 상승 중 = 반등 초기, 최적
            level_score = 80
        elif macd < 0:
            # 0선 아래 = 하락 추세
            level_score = 45
        elif macd > macd_range * 0.8:
            # 극단적 양수 = 과열
            level_score = 25
        elif macd > macd_range * 0.5:
            # 높은 양수 = 주의
            level_score = 45
        elif macd > 0 and macd < macd_range * 0.3:
            # 낮은 양수 = 안정적
            level_score = 70
        else:
            level_score = 55

        # 최종 점수 계산
        score = (cross_score * 0.40) + (hist_score * 0.35) + (level_score * 0.25)

        return min(100, max(0, score))

    # ========================================================================
    # M11. Stochastic Momentum Strategy
    # ========================================================================

    async def calculate_m11(self):
        """
        M11. Stochastic Momentum Strategy (Korean Market Optimized)
        Description: Stochastic oversold reversal detection

        한국 시장 특성 반영:
        - 20-35 과매도 탈출 구간이 최적 매수점
        - 50 이상은 이미 과열 진입
        - 낮은 위치 골든크로스가 유효
        - 80 이상 극단적 과열은 평균회귀 예상
        """
        query = """
        WITH stoch_data AS (
            SELECT
                slowk,
                slowd,
                LAG(slowk, 1) OVER (ORDER BY date) as slowk_1d_ago,
                LAG(slowk, 3) OVER (ORDER BY date) as slowk_3d_ago,
                LAG(slowk, 5) OVER (ORDER BY date) as slowk_5d_ago,
                LAG(slowd, 1) OVER (ORDER BY date) as slowd_1d_ago,
                LAG(slowd, 5) OVER (ORDER BY date) as slowd_5d_ago,
                MIN(slowk) OVER (ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as slowk_10d_min,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_indicators
            WHERE symbol = $1
                AND slowk IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            slowk,
            slowd,
            slowk_1d_ago,
            slowk_3d_ago,
            slowk_5d_ago,
            slowd_1d_ago,
            slowd_5d_ago,
            slowk_10d_min
        FROM stoch_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or result[0]['slowk'] is None:
            return None

        row = result[0]
        slowk = float(row['slowk'])
        slowd = float(row['slowd']) if row['slowd'] else slowk
        slowk_1d = float(row['slowk_1d_ago']) if row['slowk_1d_ago'] else slowk
        slowk_3d = float(row['slowk_3d_ago']) if row['slowk_3d_ago'] else slowk
        slowk_5d = float(row['slowk_5d_ago']) if row['slowk_5d_ago'] else slowk
        slowd_1d = float(row['slowd_1d_ago']) if row['slowd_1d_ago'] else slowd
        slowd_5d = float(row['slowd_5d_ago']) if row['slowd_5d_ago'] else slowd
        slowk_10d_min = float(row['slowk_10d_min']) if row['slowk_10d_min'] else slowk

        # 1. 골든크로스 위치 기반 점수
        golden_cross_recent = slowk > slowd and slowk_1d <= slowd_1d
        golden_cross_5d = slowk > slowd and slowk_5d <= slowd_5d

        if golden_cross_recent or golden_cross_5d:
            # 골든크로스 발생 - 위치에 따라 점수 차등
            if slowk <= 30:
                # 과매도 구간 골든크로스 = 최적
                cross_score = 90
            elif slowk <= 50:
                # 중립 구간 골든크로스 = 양호
                cross_score = 70
            else:
                # 과매수 구간 골든크로스 = 후반 진입
                cross_score = 40
        elif slowk > slowd:
            # 골든크로스 이후 상승 지속
            if slowk > 70:
                # 과열 구간
                cross_score = 35
            else:
                cross_score = 55
        else:
            # 데드크로스 상태
            if slowk < slowk_1d:
                # 하락 지속
                cross_score = 25
            else:
                # 반등 시도
                cross_score = 45

        # 2. 스토캐스틱 존 점수 - 한국 시장 평균회귀 반영
        if 20 <= slowk <= 35:
            # 과매도 탈출 구간 = 최적 매수점
            zone_score = 85
        elif 35 < slowk <= 50:
            # 안정 구간
            zone_score = 70
        elif 50 < slowk <= 65:
            # 주의 구간
            zone_score = 50
        elif 65 < slowk <= 80:
            # 과열 구간
            zone_score = 35
        elif slowk > 80:
            # 극단적 과열
            zone_score = 20
        elif slowk < 20:
            # 극단적 과매도 (반등 대기 또는 추가 하락)
            if slowk > slowk_1d:
                zone_score = 75  # 반등 시작
            else:
                zone_score = 50  # 하락 지속

        # 3. 반전 감지 점수
        recovering_from_oversold = slowk_10d_min <= 25 and slowk > slowk_10d_min + 10
        falling_from_overbought = slowk_5d >= 75 and slowk < slowk_5d - 10

        if recovering_from_oversold and slowk <= 50:
            # 과매도에서 반등 중, 아직 중립 이하 = 황금 진입점
            reversal_score = 90
        elif recovering_from_oversold:
            # 과매도에서 반등했지만 이미 상승
            reversal_score = 60
        elif falling_from_overbought:
            # 과매수에서 하락 시작 = 매도 신호
            reversal_score = 25
        elif slowk > slowk_1d and slowk_1d > slowk_3d and slowk <= 45:
            # 연속 상승 중이고 낮은 구간 = 양호
            reversal_score = 75
        elif slowk < slowk_1d and slowk > 70:
            # 높은 구간에서 하락 시작 = 주의
            reversal_score = 35
        else:
            reversal_score = 50

        # 최종 점수 계산
        score = (cross_score * 0.30) + (zone_score * 0.40) + (reversal_score * 0.30)

        return min(100, max(0, score))

    # ========================================================================
    # M12. Bollinger Bands Breakout Strategy
    # ========================================================================

    async def calculate_m12(self):
        """
        M12. Bollinger Bands Breakout Strategy (Korean Market Optimized)
        Description: Mean reversion at band extremes with squeeze detection

        한국 시장 특성 반영:
        - 하단밴드 터치 후 반등이 최적 매수점
        - 상단밴드 돌파는 과열 신호 (평균회귀 예상)
        - 밴드 스퀴즈 후 상향 확장이 진정한 돌파 신호
        - %B 지표로 밴드 내 위치 정밀 측정
        """
        # Get Bollinger Bands data from kr_indicators
        bb_query = """
        WITH bb_data AS (
            SELECT
                real_upper_band as upper,
                real_middle_band as middle,
                real_lower_band as lower,
                LAG(real_upper_band, 1) OVER (ORDER BY date) as upper_1d_ago,
                LAG(real_lower_band, 1) OVER (ORDER BY date) as lower_1d_ago,
                LAG(real_middle_band, 1) OVER (ORDER BY date) as middle_1d_ago,
                LAG(real_upper_band, 5) OVER (ORDER BY date) as upper_5d_ago,
                LAG(real_lower_band, 5) OVER (ORDER BY date) as lower_5d_ago,
                LAG(real_middle_band, 5) OVER (ORDER BY date) as middle_5d_ago,
                LAG(real_upper_band, 20) OVER (ORDER BY date) as upper_20d_ago,
                LAG(real_lower_band, 20) OVER (ORDER BY date) as lower_20d_ago,
                LAG(real_middle_band, 20) OVER (ORDER BY date) as middle_20d_ago,
                MIN((real_upper_band - real_lower_band) / NULLIF(real_middle_band, 0))
                    OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as min_width_20d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_indicators
            WHERE symbol = $1
                AND real_upper_band IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            upper, middle, lower,
            upper_1d_ago, lower_1d_ago, middle_1d_ago,
            upper_5d_ago, lower_5d_ago, middle_5d_ago,
            upper_20d_ago, lower_20d_ago, middle_20d_ago,
            min_width_20d
        FROM bb_data
        WHERE rn = 1
        """

        bb_result = await self.execute_query(bb_query, self.symbol, self.analysis_date)

        if not bb_result or not bb_result[0]['upper'] or not bb_result[0]['middle']:
            return None

        bb_row = bb_result[0]
        upper = float(bb_row['upper'])
        middle = float(bb_row['middle'])
        lower = float(bb_row['lower']) if bb_row['lower'] else middle * 0.95
        upper_1d = float(bb_row['upper_1d_ago']) if bb_row['upper_1d_ago'] else upper
        lower_1d = float(bb_row['lower_1d_ago']) if bb_row['lower_1d_ago'] else lower
        upper_5d = float(bb_row['upper_5d_ago']) if bb_row['upper_5d_ago'] else upper
        lower_5d = float(bb_row['lower_5d_ago']) if bb_row['lower_5d_ago'] else lower
        middle_5d = float(bb_row['middle_5d_ago']) if bb_row['middle_5d_ago'] else middle
        upper_20d = float(bb_row['upper_20d_ago']) if bb_row['upper_20d_ago'] else upper
        lower_20d = float(bb_row['lower_20d_ago']) if bb_row['lower_20d_ago'] else lower
        middle_20d = float(bb_row['middle_20d_ago']) if bb_row['middle_20d_ago'] else middle
        min_width_20d = float(bb_row['min_width_20d']) if bb_row['min_width_20d'] else 0

        # Get current and previous price from kr_intraday_total
        price_query = """
        WITH price_data AS (
            SELECT
                close,
                LAG(close, 1) OVER (ORDER BY date) as close_1d_ago,
                LAG(close, 5) OVER (ORDER BY date) as close_5d_ago,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT close, close_1d_ago, close_5d_ago
        FROM price_data
        WHERE rn = 1
        """

        price_result = await self.execute_query(price_query, self.symbol, self.analysis_date)

        if price_result and price_result[0]['close']:
            price = float(price_result[0]['close'])
            price_1d = float(price_result[0]['close_1d_ago']) if price_result[0]['close_1d_ago'] else price
            price_5d = float(price_result[0]['close_5d_ago']) if price_result[0]['close_5d_ago'] else price
        else:
            price = middle
            price_1d = middle_1d if middle_1d else middle
            price_5d = middle_5d if middle_5d else middle

        # 밴드 폭 계산
        band_width = upper - lower
        if band_width <= 0:
            return 50  # 밴드 폭이 0이면 중립

        # 1. %B 지표 계산 (밴드 내 위치)
        percent_b = (price - lower) / band_width
        percent_b_1d = (price_1d - lower_1d) / (upper_1d - lower_1d) if (upper_1d - lower_1d) > 0 else 0.5

        # %B 기반 포지션 점수 - 한국 시장 평균회귀 반영
        if percent_b <= 0:
            # 하단밴드 이탈 (극단적 과매도)
            if percent_b > percent_b_1d:
                # 반등 시작
                position_score = 80
            else:
                # 하락 지속
                position_score = 55
        elif percent_b <= 0.2:
            # 하단밴드 근처 = 매수 기회
            if percent_b > percent_b_1d:
                position_score = 85  # 반등 중
            else:
                position_score = 70
        elif percent_b <= 0.4:
            # 하단~중앙 사이 = 양호
            position_score = 75
        elif percent_b <= 0.6:
            # 중앙밴드 부근 = 안정
            position_score = 65
        elif percent_b <= 0.8:
            # 중앙~상단 사이 = 주의
            position_score = 50
        elif percent_b <= 1.0:
            # 상단밴드 근처 = 과열 주의
            position_score = 35
        else:
            # 상단밴드 돌파 = 과열 (평균회귀 예상)
            position_score = 20

        # 2. 밴드 스퀴즈 감지 및 확장 점수
        current_width_ratio = band_width / middle if middle > 0 else 0
        prev_width_ratio = (upper_5d - lower_5d) / middle_5d if middle_5d > 0 else current_width_ratio

        # 스퀴즈 여부 판단 (현재 밴드폭이 20일 최소의 1.1배 이내)
        is_squeeze = min_width_20d > 0 and current_width_ratio <= min_width_20d * 1.1
        expanding = current_width_ratio > prev_width_ratio * 1.05
        contracting = current_width_ratio < prev_width_ratio * 0.95

        if is_squeeze:
            # 스퀴즈 상태 = 돌파 대기
            if price > middle and expanding:
                # 상향 돌파 준비
                squeeze_score = 80
            elif price < middle and expanding:
                # 하향 돌파 위험
                squeeze_score = 40
            else:
                squeeze_score = 70  # 대기
        elif expanding:
            if price > middle and price > price_1d:
                # 상향 확장 + 상승 = 초기 상승
                squeeze_score = 70
            elif price < middle:
                # 하향 확장 = 하락 가속
                squeeze_score = 35
            else:
                squeeze_score = 55
        elif contracting:
            # 수축 = 변동성 감소, 다음 움직임 대기
            squeeze_score = 60
        else:
            squeeze_score = 50

        # 3. 중앙밴드 트렌드 + 가격 위치
        middle_trend_up = middle > middle_5d
        price_above_middle = price > middle

        if middle_trend_up and price_above_middle:
            # 상승 추세 + 중앙밴드 위 = 건강한 상승
            trend_score = 70
        elif middle_trend_up and not price_above_middle:
            # 상승 추세이지만 중앙밴드 아래 = 조정 중
            if price > price_1d:
                trend_score = 75  # 반등 시도
            else:
                trend_score = 55
        elif not middle_trend_up and price_above_middle:
            # 하락 추세인데 중앙밴드 위 = 불안정
            trend_score = 45
        else:
            # 하락 추세 + 중앙밴드 아래
            if price > price_1d:
                trend_score = 60  # 반등 시도
            else:
                trend_score = 35

        # 최종 점수 계산
        score = (position_score * 0.45) + (squeeze_score * 0.30) + (trend_score * 0.25)

        return min(100, max(0, score))

    # ========================================================================
    # M13. Block Trade Momentum Strategy
    # ========================================================================

    async def calculate_m13(self):
        """
        M13. Block Trade Momentum Strategy (Korean Market Optimized)
        Description: Block trade direction and post-trade price stability

        한국 시장 특성 반영:
        - 블록딜 후 가격 방향이 핵심 (매집 vs 이탈)
        - 연속 블록딜 + 안정적 상승 = 기관 매집 신호
        - 블록딜 후 급락 = 세력 이탈 신호
        - 블록딜 직후보다 안정화 이후가 중요
        """
        # 블록딜 데이터 조회 (analysis_date 기준)
        query = """
        WITH block_data AS (
            SELECT
                block_volume_rate,
                change_rate,
                date,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_blocktrades
            WHERE symbol = $1
                AND date <= COALESCE($2::date, CURRENT_DATE)
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
        ),
        recent_blocks AS (
            SELECT
                block_volume_rate,
                change_rate,
                date,
                rn
            FROM block_data
            WHERE rn <= 10  -- 최근 10개 블록딜
        ),
        block_stats AS (
            SELECT
                COUNT(*) as total_count,
                COUNT(CASE WHEN rn <= 3 THEN 1 END) as recent_count,
                SUM(CASE WHEN rn <= 3 THEN block_volume_rate ELSE 0 END) as recent_volume_rate,
                AVG(change_rate) as avg_change,
                AVG(CASE WHEN rn <= 3 THEN change_rate END) as recent_avg_change,
                MAX(CASE WHEN rn = 1 THEN date END) as last_block_date,
                MAX(CASE WHEN rn = 1 THEN change_rate END) as last_change_rate
            FROM recent_blocks
        )
        SELECT * FROM block_stats
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        # 블록딜 데이터가 없으면 중립 점수
        if not result or result[0]['total_count'] is None or result[0]['total_count'] == 0:
            return 50

        row = result[0]
        total_count = int(row['total_count']) if row['total_count'] else 0
        recent_count = int(row['recent_count']) if row['recent_count'] else 0
        recent_volume_rate = float(row['recent_volume_rate']) if row['recent_volume_rate'] else 0
        avg_change = float(row['avg_change']) if row['avg_change'] else 0
        recent_avg_change = float(row['recent_avg_change']) if row['recent_avg_change'] else 0
        last_block_date = row['last_block_date']
        last_change_rate = float(row['last_change_rate']) if row['last_change_rate'] else 0

        # 블록딜 후 가격 변화 확인
        price_query = """
        WITH price_data AS (
            SELECT
                close,
                LAG(close, 5) OVER (ORDER BY date) as close_5d_ago,
                LAG(close, 10) OVER (ORDER BY date) as close_10d_ago,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date <= COALESCE($2::date, CURRENT_DATE)
        )
        SELECT close, close_5d_ago, close_10d_ago
        FROM price_data
        WHERE rn = 1
        """

        price_result = await self.execute_query(price_query, self.symbol, self.analysis_date)

        if price_result and price_result[0]['close']:
            price = float(price_result[0]['close'])
            price_5d = float(price_result[0]['close_5d_ago']) if price_result[0]['close_5d_ago'] else price
            price_10d = float(price_result[0]['close_10d_ago']) if price_result[0]['close_10d_ago'] else price
            price_change_5d = (price - price_5d) / price_5d * 100 if price_5d > 0 else 0
            price_change_10d = (price - price_10d) / price_10d * 100 if price_10d > 0 else 0
        else:
            price_change_5d = 0
            price_change_10d = 0

        # 1. 블록딜 후 가격 방향성 점수
        if recent_count > 0:
            if recent_avg_change > 0 and price_change_5d > 0:
                # 블록딜 상승 + 이후 가격 상승 = 매집 신호
                direction_score = 85
            elif recent_avg_change > 0 and price_change_5d < -2:
                # 블록딜 상승했지만 이후 하락 = 세력 이탈
                direction_score = 30
            elif recent_avg_change < -2 and price_change_5d < 0:
                # 블록딜 하락 + 이후 하락 지속 = 매도 압력
                direction_score = 25
            elif recent_avg_change < 0 and price_change_5d > 2:
                # 블록딜 하락 후 반등 = 저점 매집 가능
                direction_score = 70
            else:
                direction_score = 50
        else:
            direction_score = 50

        # 2. 블록딜 빈도 및 연속성 점수
        if recent_count >= 3:
            # 최근 연속 블록딜
            if price_change_5d > 0:
                # 연속 블록딜 + 상승 = 기관 매집
                frequency_score = 80
            else:
                # 연속 블록딜 + 하락 = 대량 매도
                frequency_score = 35
        elif recent_count >= 1:
            # 최근 블록딜 있음
            frequency_score = 60
        elif total_count >= 5:
            # 30일 내 블록딜 다수
            frequency_score = 55
        else:
            # 블록딜 적음
            frequency_score = 45

        # 3. 블록딜 안정성 점수 (급등락 vs 안정적 상승)
        if abs(last_change_rate) > 5:
            # 급등락 블록딜 = 불안정
            if price_change_5d > 0:
                stability_score = 55  # 급등 후 유지
            else:
                stability_score = 30  # 급등 후 하락
        elif 0 < last_change_rate <= 3:
            # 안정적 상승 블록딜
            if price_change_5d >= 0:
                stability_score = 75  # 안정적 상승 유지
            else:
                stability_score = 45
        elif -3 <= last_change_rate < 0:
            # 소폭 하락 블록딜
            if price_change_5d > 0:
                stability_score = 65  # 저점 매집 후 반등
            else:
                stability_score = 40
        else:
            stability_score = 50

        # 최종 점수 계산
        score = (direction_score * 0.40) + (frequency_score * 0.30) + (stability_score * 0.30)

        return min(100, max(0, score))

    # ========================================================================
    # M14. Composite Technical Indicator Momentum Strategy
    # ========================================================================

    async def calculate_m14(self):
        """
        M14. Composite Technical Indicator Momentum Strategy (Korean Market Optimized)
        Description: Oversold reversal detection across multiple indicators

        한국 시장 특성 반영:
        - 모든 지표 상승 = 과열 (평균회귀 예상)
        - 과매도 지표 다수 = 반등 기회
        - 개별 지표별 평균회귀 구간 적용
        - 지표 불일치 = 전환점 가능성
        """
        query = """
        WITH latest_indicators AS (
            SELECT
                rsi,
                macd,
                macd_signal,
                macd_hist,
                slowk,
                slowd,
                real_middle_band,
                real_upper_band,
                real_lower_band,
                mfi,
                roc,
                adx,
                cci
            FROM kr_indicators
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
            ORDER BY date DESC
            LIMIT 1
        ),
        latest_price AS (
            SELECT close
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            i.*,
            p.close
        FROM latest_indicators i, latest_price p
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result:
            return None

        row = result[0]

        # 과매도/과매수/중립 신호 카운트
        oversold_signals = 0  # 반등 기회
        overbought_signals = 0  # 과열 위험
        neutral_signals = 0
        total_indicators = 0

        # 1. RSI 분석 (한국 시장 평균회귀 반영)
        if row['rsi'] is not None:
            rsi = float(row['rsi'])
            total_indicators += 1
            if rsi <= 35:
                oversold_signals += 1  # 과매도 = 반등 기회
            elif rsi <= 45:
                oversold_signals += 0.5  # 반등 초기
            elif rsi >= 70:
                overbought_signals += 1  # 과열
            elif rsi >= 60:
                overbought_signals += 0.5  # 주의
            else:
                neutral_signals += 1

        # 2. Stochastic 분석
        if row['slowk'] is not None:
            slowk = float(row['slowk'])
            total_indicators += 1
            if slowk <= 25:
                oversold_signals += 1
            elif slowk <= 40:
                oversold_signals += 0.5
            elif slowk >= 80:
                overbought_signals += 1
            elif slowk >= 70:
                overbought_signals += 0.5
            else:
                neutral_signals += 1

        # 3. MFI 분석
        if row['mfi'] is not None:
            mfi = float(row['mfi'])
            total_indicators += 1
            if mfi <= 30:
                oversold_signals += 1
            elif mfi <= 45:
                oversold_signals += 0.5
            elif mfi >= 80:
                overbought_signals += 1
            elif mfi >= 70:
                overbought_signals += 0.5
            else:
                neutral_signals += 1

        # 4. CCI 분석
        if row['cci'] is not None:
            cci = float(row['cci'])
            total_indicators += 1
            if cci <= -100:
                oversold_signals += 1
            elif cci <= -50:
                oversold_signals += 0.5
            elif cci >= 150:
                overbought_signals += 1
            elif cci >= 100:
                overbought_signals += 0.5
            else:
                neutral_signals += 1

        # 5. MACD 분석 (히스토그램 방향 중요)
        if row['macd'] is not None and row['macd_signal'] is not None:
            macd = float(row['macd'])
            signal = float(row['macd_signal'])
            hist = float(row['macd_hist']) if row['macd_hist'] else macd - signal
            total_indicators += 1
            if macd < 0 and hist > 0:
                # 0선 아래에서 반등 시작
                oversold_signals += 1
            elif macd < 0 and hist < 0:
                # 하락 지속
                oversold_signals += 0.3
            elif macd > 0 and hist < 0:
                # 상승 둔화
                overbought_signals += 0.5
            elif macd > 0 and hist > 0 and macd > signal * 1.5:
                # 과열
                overbought_signals += 1
            else:
                neutral_signals += 1

        # 6. Bollinger Band 분석 (%B)
        if row['real_middle_band'] is not None and row['close'] is not None:
            middle = float(row['real_middle_band'])
            upper = float(row['real_upper_band']) if row['real_upper_band'] else middle * 1.02
            lower = float(row['real_lower_band']) if row['real_lower_band'] else middle * 0.98
            close = float(row['close'])
            total_indicators += 1

            if upper != lower:
                percent_b = (close - lower) / (upper - lower)
                if percent_b <= 0.1:
                    oversold_signals += 1
                elif percent_b <= 0.3:
                    oversold_signals += 0.5
                elif percent_b >= 0.95:
                    overbought_signals += 1
                elif percent_b >= 0.8:
                    overbought_signals += 0.5
                else:
                    neutral_signals += 1

        # 7. ROC 분석
        if row['roc'] is not None:
            roc = float(row['roc'])
            total_indicators += 1
            if roc <= -10:
                oversold_signals += 1
            elif roc <= -3:
                oversold_signals += 0.5
            elif roc >= 15:
                overbought_signals += 1
            elif roc >= 8:
                overbought_signals += 0.5
            else:
                neutral_signals += 1

        if total_indicators == 0:
            return None

        # 점수 계산 - 과매도 신호가 많을수록 반등 기회
        oversold_ratio = oversold_signals / total_indicators
        overbought_ratio = overbought_signals / total_indicators

        if oversold_ratio >= 0.5:
            # 과매도 지표 50% 이상 = 강한 반등 기회
            base_score = 80 + (oversold_ratio - 0.5) * 20
        elif oversold_ratio >= 0.3:
            # 과매도 지표 30% 이상 = 반등 기회
            base_score = 70
        elif overbought_ratio >= 0.5:
            # 과매수 지표 50% 이상 = 강한 과열 위험
            base_score = 25 - (overbought_ratio - 0.5) * 10
        elif overbought_ratio >= 0.3:
            # 과매수 지표 30% 이상 = 과열 주의
            base_score = 40
        else:
            # 중립
            base_score = 55

        # ADX로 추세 강도 보정
        adx_value = float(row['adx']) if row['adx'] is not None else 25
        if adx_value >= 30:
            # 강한 추세 = 반전 신호 약화
            if oversold_ratio >= 0.3:
                # 과매도이지만 강한 하락 추세 = 반등 불확실
                adx_adjust = -5
            elif overbought_ratio >= 0.3:
                # 과매수이고 강한 상승 추세 = 추가 상승 가능
                adx_adjust = 5
            else:
                adx_adjust = 0
        elif adx_value <= 20:
            # 약한 추세 = 반전 가능성 높음
            if oversold_ratio >= 0.3:
                adx_adjust = 5
            else:
                adx_adjust = 0
        else:
            adx_adjust = 0

        score = base_score + adx_adjust

        return min(100, max(0, score))

    # ========================================================================
    # M15. Foreign Ownership Increase Momentum Strategy
    # ========================================================================

    async def calculate_m15(self):
        """
        M15. Foreign Ownership Increase Momentum Strategy (Korean Market Optimized)
        Description: Foreign ownership change with price correlation analysis

        한국 시장 특성 반영:
        - 외국인 증가 + 가격 하락 = 저점 매집 (최적)
        - 외국인 증가 + 가격 상승 = 추격 매수 (주의)
        - 외국인 감소 + 가격 상승 = 세력 이탈 (위험)
        - 지분율 절대 수준 반영 (높은 지분율 = 추가 매수 여력 제한)
        """
        # 외국인 지분율 데이터 조회
        foreign_query = """
        WITH foreign_data AS (
            SELECT
                foreign_rate,
                date,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_foreign_ownership
            WHERE symbol = $1
                AND foreign_rate IS NOT NULL
                AND date <= COALESCE($2::date, CURRENT_DATE)
        )
        SELECT
            AVG(CASE WHEN rn <= 3 THEN foreign_rate END) as recent_3d_avg,
            AVG(CASE WHEN rn BETWEEN 4 AND 6 THEN foreign_rate END) as past_3d_avg,
            AVG(CASE WHEN rn BETWEEN 7 AND 10 THEN foreign_rate END) as prev_4d_avg,
            MAX(CASE WHEN rn = 1 THEN foreign_rate END) as current_rate,
            MAX(CASE WHEN rn = 5 THEN foreign_rate END) as rate_5d_ago,
            MAX(CASE WHEN rn = 10 THEN foreign_rate END) as rate_10d_ago,
            MAX(CASE WHEN rn = 20 THEN foreign_rate END) as rate_20d_ago
        FROM foreign_data
        WHERE rn <= 20
        """

        foreign_result = await self.execute_query(foreign_query, self.symbol, self.analysis_date)

        if not foreign_result or foreign_result[0]['recent_3d_avg'] is None:
            return 50  # 데이터 없으면 중립

        f_row = foreign_result[0]
        recent_avg = float(f_row['recent_3d_avg'])
        past_avg = float(f_row['past_3d_avg']) if f_row['past_3d_avg'] else recent_avg
        prev_avg = float(f_row['prev_4d_avg']) if f_row['prev_4d_avg'] else past_avg
        current_rate = float(f_row['current_rate']) if f_row['current_rate'] else recent_avg
        rate_5d_ago = float(f_row['rate_5d_ago']) if f_row['rate_5d_ago'] else current_rate
        rate_10d_ago = float(f_row['rate_10d_ago']) if f_row['rate_10d_ago'] else rate_5d_ago
        rate_20d_ago = float(f_row['rate_20d_ago']) if f_row['rate_20d_ago'] else rate_10d_ago

        # 가격 데이터 조회
        price_query = """
        WITH price_data AS (
            SELECT
                close,
                LAG(close, 5) OVER (ORDER BY date) as close_5d_ago,
                LAG(close, 10) OVER (ORDER BY date) as close_10d_ago,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date <= COALESCE($2::date, CURRENT_DATE)
        )
        SELECT close, close_5d_ago, close_10d_ago
        FROM price_data
        WHERE rn = 1
        """

        price_result = await self.execute_query(price_query, self.symbol, self.analysis_date)

        if price_result and price_result[0]['close']:
            price = float(price_result[0]['close'])
            price_5d = float(price_result[0]['close_5d_ago']) if price_result[0]['close_5d_ago'] else price
            price_10d = float(price_result[0]['close_10d_ago']) if price_result[0]['close_10d_ago'] else price
            price_change_5d = (price - price_5d) / price_5d * 100 if price_5d > 0 else 0
            price_change_10d = (price - price_10d) / price_10d * 100 if price_10d > 0 else 0
        else:
            price_change_5d = 0
            price_change_10d = 0

        # 외국인 지분율 변화 계산
        foreign_change_5d = current_rate - rate_5d_ago
        foreign_change_10d = current_rate - rate_10d_ago

        # 1. 외국인 + 가격 연동 점수 (핵심)
        foreign_increasing = foreign_change_5d > 0.1  # 0.1%p 이상 증가
        foreign_decreasing = foreign_change_5d < -0.1  # 0.1%p 이상 감소
        price_up = price_change_5d > 1  # 1% 이상 상승
        price_down = price_change_5d < -1  # 1% 이상 하락

        if foreign_increasing and price_down:
            # 외국인 증가 + 가격 하락 = 저점 매집 (최적)
            correlation_score = 85
        elif foreign_increasing and not price_up:
            # 외국인 증가 + 가격 보합 = 조용한 매집
            correlation_score = 75
        elif foreign_increasing and price_up:
            # 외국인 증가 + 가격 상승 = 추격 매수 (늦은 진입)
            correlation_score = 55
        elif foreign_decreasing and price_up:
            # 외국인 감소 + 가격 상승 = 세력 이탈 (위험)
            correlation_score = 25
        elif foreign_decreasing and price_down:
            # 외국인 감소 + 가격 하락 = 투매
            correlation_score = 35
        elif foreign_decreasing:
            # 외국인 감소
            correlation_score = 40
        else:
            # 변화 없음
            correlation_score = 50

        # 2. 외국인 지분율 절대 수준 점수
        if current_rate < 3:
            # 저지분율 = 추가 매수 여력 큼
            level_score = 70
        elif current_rate < 10:
            # 중저지분율 = 양호
            level_score = 65
        elif current_rate < 20:
            # 중간 지분율 = 적정
            level_score = 55
        elif current_rate < 30:
            # 고지분율 = 추가 매수 여력 제한
            level_score = 45
        else:
            # 매우 높은 지분율 = 매수 포화
            level_score = 35

        # 저지분율 + 증가 = 초기 매집 보너스
        if current_rate < 5 and foreign_increasing:
            level_score += 15

        # 3. 매집 패턴 점수
        steady_accumulation = (
            recent_avg > past_avg > prev_avg and
            abs(recent_avg - past_avg) < 1 and
            abs(past_avg - prev_avg) < 1
        )  # 소폭 연속 증가

        rapid_accumulation = foreign_change_10d > 2  # 10일간 2%p 이상 급증

        if steady_accumulation:
            # 점진적 매집 = 기관 패턴
            pattern_score = 75
        elif rapid_accumulation and price_change_10d <= 0:
            # 급격 매집 + 가격 안정 = 강한 매집
            pattern_score = 80
        elif rapid_accumulation and price_change_10d > 5:
            # 급격 매집 + 급등 = 추격 매수, 고점 가능
            pattern_score = 45
        elif recent_avg < past_avg < prev_avg:
            # 연속 감소 = 이탈 패턴
            pattern_score = 30
        else:
            pattern_score = 50

        # 최종 점수 계산
        score = (correlation_score * 0.45) + (level_score * 0.25) + (pattern_score * 0.30)

        return min(100, max(0, score))

    # ========================================================================
    # M16. Trading Value Surge Momentum Strategy
    # ========================================================================

    async def calculate_m16(self):
        """
        M16. Trading Value Surge Momentum Strategy
        Description: Sudden surge in trading value indicates strong interest
        """
        query = """
        WITH value_data AS (
            SELECT
                trading_value,
                AVG(trading_value) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_20d,
                AVG(trading_value) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as avg_5d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND trading_value IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            trading_value as current_value,
            avg_20d,
            avg_5d
        FROM value_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['current_value'] or not result[0]['avg_20d']:
            return None

        row = result[0]
        current = float(row['current_value'])
        avg_20d = float(row['avg_20d'])
        avg_5d = float(row['avg_5d']) if row['avg_5d'] else current

        if avg_20d == 0:
            return None

        # Current vs 20-day average ratio
        ratio_20d = current / avg_20d

        # 5-day average vs 20-day average (trend acceleration)
        if avg_20d > 0:
            trend_ratio = avg_5d / avg_20d
        else:
            trend_ratio = 1.0

        # Ratio score
        if ratio_20d >= 2.0:
            ratio_score = 100
        elif ratio_20d >= 1.5:
            ratio_score = 80
        elif ratio_20d >= 1.2:
            ratio_score = 60
        elif ratio_20d >= 1.0:
            ratio_score = 40
        else:
            ratio_score = 20

        # Trend acceleration score
        if trend_ratio >= 1.3:
            trend_score = 100
        elif trend_ratio >= 1.1:
            trend_score = 70
        elif trend_ratio >= 1.0:
            trend_score = 50
        else:
            trend_score = 30

        # Final score
        score = (ratio_score * 0.6) + (trend_score * 0.4)

        return score

    # ========================================================================
    # M17. Consecutive Up Days Momentum Strategy
    # ========================================================================

    async def calculate_m17(self):
        """
        M17. Consecutive Up Days Momentum Strategy
        Description: Consecutive positive days indicate strong uptrend
        """
        query = """
        SELECT change_rate, date
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
            AND change_rate IS NOT NULL
        ORDER BY date DESC
        LIMIT 20
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result:
            return None

        # Count consecutive up days from most recent
        consecutive_up = 0
        consecutive_down = 0
        total_up_days = 0
        total_down_days = 0

        for i, row in enumerate(result):
            change = float(row['change_rate'])

            # Count total up/down days in last 20 days
            if change > 0:
                total_up_days += 1
            elif change < 0:
                total_down_days += 1

            # Count consecutive days from today
            if i == 0:
                if change > 0:
                    consecutive_up += 1
                elif change < 0:
                    consecutive_down += 1
            else:
                # Continue counting consecutive days
                if consecutive_up > 0 and change > 0:
                    consecutive_up += 1
                elif consecutive_down > 0 and change < 0:
                    consecutive_down += 1
                else:
                    break

        # Consecutive up days score
        if consecutive_up >= 5:
            consec_score = 100
        elif consecutive_up >= 3:
            consec_score = 80
        elif consecutive_up >= 2:
            consec_score = 60
        elif consecutive_up == 1:
            consec_score = 40
        else:
            # Penalize consecutive down days
            if consecutive_down >= 3:
                consec_score = 0
            elif consecutive_down == 2:
                consec_score = 20
            else:
                consec_score = 30

        # Overall trend score (up days ratio in last 20 days)
        if len(result) > 0:
            up_ratio = total_up_days / len(result)
            trend_score = up_ratio * 100
        else:
            trend_score = 50

        # Final score
        score = (consec_score * 0.7) + (trend_score * 0.3)

        return score

    # ========================================================================
    # M18. Gap Up Momentum Strategy
    # ========================================================================

    async def calculate_m18(self):
        """
        M18. Gap Up Momentum Strategy
        Description: Gap up opening with sustained price action
        """
        query = """
        WITH price_data AS (
            SELECT
                date,
                open,
                close,
                high,
                low,
                LAG(close, 1) OVER (ORDER BY date) as prev_close,
                change_rate,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND open IS NOT NULL
                AND close IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            date,
            open,
            close,
            high,
            low,
            prev_close,
            change_rate,
            CASE
                WHEN prev_close IS NOT NULL AND prev_close > 0
                THEN ((open - prev_close) / prev_close * 100)
                ELSE 0
            END as gap_pct
        FROM price_data
        WHERE rn <= 10
        ORDER BY date DESC
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result:
            return None

        # Most recent day (today)
        today = result[0]
        gap_pct = float(today['gap_pct'])
        change_rate = float(today['change_rate']) if today['change_rate'] else 0

        open_price = float(today['open'])
        close_price = float(today['close'])
        high_price = float(today['high']) if today['high'] else close_price
        low_price = float(today['low']) if today['low'] else close_price

        # Count recent gap ups (last 10 days)
        gap_up_count = sum(1 for row in result if row['gap_pct'] and float(row['gap_pct']) > 1.0)

        # Gap magnitude score
        if gap_pct >= 5.0:
            gap_score = 100
        elif gap_pct >= 3.0:
            gap_score = 80
        elif gap_pct >= 1.5:
            gap_score = 60
        elif gap_pct >= 1.0:
            gap_score = 40
        else:
            gap_score = 20

        # Price sustaining score (close vs open after gap)
        if open_price > 0:
            sustain_ratio = (close_price - open_price) / open_price * 100
            if sustain_ratio >= 2.0:
                sustain_score = 100
            elif sustain_ratio >= 0:
                sustain_score = 70
            elif sustain_ratio >= -2.0:
                sustain_score = 40
            else:
                sustain_score = 10
        else:
            sustain_score = 50

        # Frequency score (gap ups in last 10 days)
        freq_score = min(100, gap_up_count * 25)

        # Final score
        score = (gap_score * 0.4) + (sustain_score * 0.4) + (freq_score * 0.2)

        return score

    # ========================================================================
    # M19. New High Streak Momentum Strategy
    # ========================================================================

    async def calculate_m19(self):
        """
        M19. New High Streak Momentum Strategy
        Description: Consecutive new high breakouts indicate strong uptrend
        """
        query = """
        WITH high_data AS (
            SELECT
                date,
                high,
                close,
                MAX(high) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as prev_20d_high,
                MAX(high) OVER (ORDER BY date ROWS BETWEEN 60 PRECEDING AND 21 PRECEDING) as prev_60d_high,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND high IS NOT NULL
                AND ($2::date IS NULL OR date <= $2)
        )
        SELECT
            date,
            high,
            close,
            prev_20d_high,
            prev_60d_high,
            CASE
                WHEN prev_20d_high IS NOT NULL AND high >= prev_20d_high
                THEN 1
                ELSE 0
            END as is_new_high
        FROM high_data
        WHERE rn <= 20
        ORDER BY date DESC
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or len(result) == 0:
            return None

        # Count new highs in different periods
        new_high_last_5 = 0
        new_high_last_10 = 0
        new_high_total = 0
        consecutive_new_high = 0

        for i, row in enumerate(result):
            is_new = row['is_new_high']

            if is_new:
                new_high_total += 1
                if i < 5:
                    new_high_last_5 += 1
                if i < 10:
                    new_high_last_10 += 1

            # Count consecutive new highs from today
            if i == 0 and is_new:
                consecutive_new_high = 1
            elif consecutive_new_high > 0 and is_new:
                consecutive_new_high += 1
            elif consecutive_new_high > 0:
                break

        # Consecutive new high score
        if consecutive_new_high >= 3:
            consec_score = 100
        elif consecutive_new_high == 2:
            consec_score = 70
        elif consecutive_new_high == 1:
            consec_score = 50
        else:
            consec_score = 20

        # Recent frequency score (last 5 days)
        recent_score = min(100, new_high_last_5 * 30)

        # Overall frequency score (last 20 days)
        overall_score = min(100, new_high_total * 10)

        # Final score
        score = (consec_score * 0.5) + (recent_score * 0.3) + (overall_score * 0.2)

        return score

    # ========================================================================
    # M20. ATR Breakout Momentum Strategy
    # ========================================================================

    async def calculate_m20(self):
        """
        M20. ATR Breakout Momentum Strategy (Korean Market Optimized)
        Description: Mean reversion at ATR extremes with volatility analysis

        한국 시장 특성 반영:
        - ATR 상단 돌파 = 과열 (평균회귀 예상)
        - ATR 하단 접근/터치 = 반등 기회
        - ATR 수축 후 확장 = 돌파 준비
        - 가격 방향 + ATR 연동 분석
        """
        query = """
        WITH atr_data AS (
            SELECT
                i.date,
                i.atr,
                i.sma as middle_price,
                t.close,
                t.high,
                t.low,
                t.volume,
                CASE
                    WHEN i.atr IS NOT NULL AND i.sma IS NOT NULL AND i.atr > 0
                    THEN (t.close - i.sma) / i.atr
                    ELSE NULL
                END as atr_distance,
                LAG(t.close, 1) OVER (ORDER BY i.date) as close_1d_ago,
                LAG(i.atr, 1) OVER (ORDER BY i.date) as atr_1d_ago,
                LAG(i.atr, 5) OVER (ORDER BY i.date) as atr_5d_ago,
                LAG(i.atr, 10) OVER (ORDER BY i.date) as atr_10d_ago,
                AVG(t.volume) OVER (ORDER BY i.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_vol_20d,
                MIN(i.atr) OVER (ORDER BY i.date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as atr_20d_min,
                ROW_NUMBER() OVER (ORDER BY i.date DESC) as rn
            FROM kr_indicators i
            JOIN kr_intraday_total t ON i.symbol = t.symbol AND i.date = t.date
            WHERE i.symbol = $1
                AND i.atr IS NOT NULL
                AND ($2::date IS NULL OR i.date <= $2)
        )
        SELECT *
        FROM atr_data
        WHERE rn <= 10
        ORDER BY rn
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['atr_distance']:
            return None

        # 현재 데이터
        today = result[0]
        atr_distance = float(today['atr_distance'])
        atr_value = float(today['atr'])
        close = float(today['close'])
        close_1d = float(today['close_1d_ago']) if today['close_1d_ago'] else close
        volume = float(today['volume']) if today['volume'] else 0
        avg_vol_20d = float(today['avg_vol_20d']) if today['avg_vol_20d'] else volume
        atr_1d = float(today['atr_1d_ago']) if today['atr_1d_ago'] else atr_value
        atr_5d = float(today['atr_5d_ago']) if today['atr_5d_ago'] else atr_value
        atr_10d = float(today['atr_10d_ago']) if today['atr_10d_ago'] else atr_value
        atr_20d_min = float(today['atr_20d_min']) if today['atr_20d_min'] else atr_value

        # 가격 변화
        price_change = (close - close_1d) / close_1d * 100 if close_1d > 0 else 0

        # 1. ATR Distance 점수 - 한국 시장 평균회귀 반영
        if atr_distance <= -1.5:
            # ATR 하단 이탈 = 극단적 과매도
            if price_change > 0:
                distance_score = 85  # 반등 시작
            else:
                distance_score = 65  # 하락 지속, 반등 대기
        elif atr_distance <= -1.0:
            # ATR 하단 근처 = 반등 기회
            if price_change > 0:
                distance_score = 80
            else:
                distance_score = 70
        elif atr_distance <= -0.5:
            # 중앙~하단 사이 = 양호
            distance_score = 75
        elif atr_distance <= 0.5:
            # 중앙 부근 = 안정
            distance_score = 65
        elif atr_distance <= 1.0:
            # 중앙~상단 사이 = 주의
            distance_score = 50
        elif atr_distance <= 1.5:
            # ATR 상단 근처 = 과열 주의
            distance_score = 35
        elif atr_distance <= 2.0:
            # ATR 상단 돌파 = 과열
            distance_score = 25
        else:
            # 극단적 상단 돌파 = 급등 후 급락 위험
            distance_score = 15

        # 2. ATR 변동성 분석
        atr_expanding = atr_value > atr_5d * 1.1  # ATR 10% 이상 확장
        atr_contracting = atr_value < atr_5d * 0.9  # ATR 10% 이상 수축
        atr_squeeze = atr_20d_min > 0 and atr_value <= atr_20d_min * 1.1  # 스퀴즈 상태

        if atr_squeeze:
            # ATR 스퀴즈 = 돌파 대기
            if price_change > 0:
                volatility_score = 75  # 상향 돌파 가능
            else:
                volatility_score = 60
        elif atr_expanding and atr_distance > 1.0:
            # 변동성 확장 + 상단 돌파 = 급등 과열
            volatility_score = 30
        elif atr_expanding and atr_distance < -1.0:
            # 변동성 확장 + 하단 터치 = 공포 매도 후 반등 가능
            if price_change > 0:
                volatility_score = 80
            else:
                volatility_score = 55
        elif atr_contracting:
            # 변동성 수축 = 안정화
            volatility_score = 65
        else:
            volatility_score = 50

        # 3. 반전 패턴 점수
        # 최근 10일 ATR distance 분석
        lower_touch_count = 0
        upper_touch_count = 0
        recovering_from_lower = False

        for i, row in enumerate(result):
            if row['atr_distance']:
                dist = float(row['atr_distance'])
                if dist < -1.0:
                    lower_touch_count += 1
                    if i > 0 and i <= 3:  # 최근 3일 내 하단 터치
                        recovering_from_lower = True
                elif dist > 1.5:
                    upper_touch_count += 1

        if recovering_from_lower and price_change > 0:
            # 최근 하단 터치 후 반등 중 = 최적 진입
            reversal_score = 85
        elif lower_touch_count >= 2 and atr_distance > -0.5:
            # 하단 반복 터치 후 회복 = 지지선 확인
            reversal_score = 75
        elif upper_touch_count >= 3:
            # 상단 반복 터치 = 과열 지속, 조정 임박
            reversal_score = 35
        elif upper_touch_count >= 2 and price_change < 0:
            # 상단 터치 후 하락 시작 = 조정 시작
            reversal_score = 30
        else:
            reversal_score = 50

        # 4. 거래량 확인
        if avg_vol_20d > 0:
            vol_ratio = volume / avg_vol_20d
            if atr_distance < -0.5 and vol_ratio > 1.3:
                # 하단 구간에서 거래량 증가 = 반등 신뢰
                vol_adjust = 5
            elif atr_distance > 1.0 and vol_ratio > 2.0:
                # 상단 구간에서 거래량 폭증 = 세력 이탈 가능
                vol_adjust = -5
            else:
                vol_adjust = 0
        else:
            vol_adjust = 0

        # 최종 점수 계산
        base_score = (distance_score * 0.40) + (volatility_score * 0.30) + (reversal_score * 0.30)
        score = base_score + vol_adjust

        return min(100, max(0, score))

    # ========================================================================
    # M21-M23. Reserved for Future Strategies
    # ========================================================================

    async def calculate_m21(self):
        """M21. Reserved for future use"""
        return None

    async def calculate_m22(self):
        """M22. Reserved for future use"""
        return None

    async def calculate_m23(self):
        """
        M23. Market Volatility Regime & Polarization Strategy
        Description: Detect winner-takes-all markets and volatility regimes
        - Market polarization detection (mean vs median divergence)
        - Volatility regime classification
        - Stock's relative performance in current regime

        Uses existing SQL data efficiently:
        - kr_intraday_total.change_rate (no manual calculation)
        - Single query for market statistics
        """
        query = """
        WITH market_stats AS (
            -- Get market-wide statistics for last 20 days
            SELECT
                date,
                AVG(change_rate) as market_mean,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY change_rate) as market_median,
                STDDEV(change_rate) as market_stddev,
                COUNT(*) as stock_count,
                COUNT(CASE WHEN change_rate > 0 THEN 1 END) as positive_count,
                COUNT(CASE WHEN change_rate < 0 THEN 1 END) as negative_count
            FROM kr_intraday_total
            WHERE date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '20 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
                AND change_rate IS NOT NULL
            GROUP BY date
            ORDER BY date DESC
        ),
        stock_recent AS (
            -- Get this stock's recent performance
            SELECT
                change_rate,
                close,
                volume,
                AVG(volume) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_volume,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
                AND change_rate IS NOT NULL
        ),
        recent_market AS (
            SELECT
                AVG(market_mean) as avg_mean,
                AVG(market_median) as avg_median,
                AVG(market_stddev) as avg_volatility,
                AVG(market_mean - market_median) as polarization,
                AVG(positive_count::DECIMAL / NULLIF(stock_count, 0)) as win_ratio
            FROM market_stats
            WHERE date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '10 days'
        )
        SELECT
            sr.change_rate as stock_change,
            sr.close,
            sr.volume,
            sr.avg_volume,
            rm.avg_mean as market_mean,
            rm.avg_median as market_median,
            rm.avg_volatility,
            rm.polarization,
            rm.win_ratio
        FROM stock_recent sr, recent_market rm
        WHERE sr.rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]:
            return None

        row = result[0]
        stock_change = float(row['stock_change']) if row['stock_change'] else 0
        market_mean = float(row['market_mean']) if row['market_mean'] else 0
        market_median = float(row['market_median']) if row['market_median'] else 0
        volatility = float(row['avg_volatility']) if row['avg_volatility'] else 1
        polarization = float(row['polarization']) if row['polarization'] else 0
        win_ratio = float(row['win_ratio']) if row['win_ratio'] else 0.5
        volume = float(row['volume']) if row['volume'] else 0
        avg_volume = float(row['avg_volume']) if row['avg_volume'] else 1

        # 1. Polarization Score (mean vs median divergence)
        # September example: mean 2.12%, median -1.12% → polarization = 3.24%
        # High polarization (>2%) = winner-takes-all market
        if abs(polarization) > 2.0:
            polarization_score = 100
        elif abs(polarization) > 1.0:
            polarization_score = 70
        elif abs(polarization) > 0.5:
            polarization_score = 50
        else:
            polarization_score = 30

        # 2. Volatility Regime Score
        # High volatility (>3%) = turbulent market, favor strong momentum stocks
        if volatility > 3.0:
            vol_regime = 'high'
            vol_score = 80
        elif volatility > 2.0:
            vol_regime = 'medium'
            vol_score = 60
        elif volatility > 1.0:
            vol_regime = 'low'
            vol_score = 40
        else:
            vol_regime = 'very_low'
            vol_score = 50

        # 3. Stock Performance in Current Regime
        # In polarized markets, being a "winner" (positive return) is critical
        if polarization > 1.0:  # Winner-takes-all market
            if stock_change > market_mean:
                # Stock is above average in polarized market
                performance_score = 100
            elif stock_change > market_median:
                # Stock is positive but below average
                performance_score = 70
            elif stock_change > 0:
                # Stock is slightly positive
                performance_score = 50
            else:
                # Stock is losing in winner-takes-all market
                performance_score = 10
        else:  # Normal market
            if stock_change > market_mean:
                performance_score = 80
            elif stock_change > 0:
                performance_score = 60
            else:
                performance_score = 40

        # 4. Volume Confirmation
        # High volume confirms the regime signal
        if avg_volume > 0:
            vol_ratio = volume / avg_volume
            if vol_ratio > 1.5:
                volume_mult = 1.2
            elif vol_ratio > 1.0:
                volume_mult = 1.0
            else:
                volume_mult = 0.85
        else:
            volume_mult = 1.0

        # 5. Winner Ratio Adjustment
        # In markets where <40% stocks are positive, winners are rare
        if win_ratio < 0.4:
            # Rare winners - boost score if stock is winning
            if stock_change > 0:
                winner_bonus = 20
            else:
                winner_bonus = -20
        elif win_ratio > 0.6:
            # Most stocks winning - normal market
            winner_bonus = 0
        else:
            winner_bonus = 0

        # Final Score Calculation
        base_score = (polarization_score * 0.3) + (performance_score * 0.5) + (vol_score * 0.2)
        final_score = min(100, max(0, base_score * volume_mult + winner_bonus))

        return final_score

    # ========================================================================
    # M24. Idiosyncratic Momentum (잔차 모멘텀)
    # Based on Hanauer & Windmüller research - outperforms all other strategies
    # ========================================================================

    async def calculate_m24(self):
        """
        M24. Idiosyncratic Momentum Strategy
        Description: Market/Theme effect를 제거한 종목 고유 모멘텀
        Formula: IdioMom = 종목수익률 - 시장수익률 - 테마평균수익률

        This strategy removes systematic risk factors to capture
        stock-specific momentum that is less prone to crashes.

        Based on research: Idiosyncratic momentum leads to
        - Higher Sharpe ratio (2x improvement over other methods)
        - Lower maximum drawdowns
        - Better January performance
        """
        query = """
        WITH stock_returns AS (
            -- 종목의 최근 20일 수익률
            SELECT
                date,
                change_rate as stock_return
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
            ORDER BY date DESC
            LIMIT 20
        ),
        stock_info AS (
            SELECT theme, exchange
            FROM kr_stock_detail
            WHERE symbol = $1
        ),
        market_returns AS (
            -- 시장(KOSPI/KOSDAQ) 수익률
            SELECT m.date, m.change_rate as market_return
            FROM market_index m, stock_info si
            WHERE m.exchange = si.exchange
                AND m.date IN (SELECT date FROM stock_returns)
        ),
        theme_returns AS (
            -- 같은 테마 종목들의 평균 수익률
            SELECT
                t.date,
                AVG(t.change_rate) as theme_return
            FROM kr_intraday_total t
            JOIN kr_stock_detail d ON t.symbol = d.symbol
            JOIN stock_info si ON d.theme = si.theme
            WHERE t.date IN (SELECT date FROM stock_returns)
                AND t.symbol != $1
            GROUP BY t.date
        )
        SELECT
            AVG(sr.stock_return) as avg_stock_return,
            AVG(mr.market_return) as avg_market_return,
            AVG(tr.theme_return) as avg_theme_return,
            AVG(sr.stock_return - COALESCE(mr.market_return, 0) - COALESCE(tr.theme_return, 0) + sr.stock_return) as idio_momentum,
            STDDEV(sr.stock_return) as stock_volatility,
            COUNT(*) as sample_count
        FROM stock_returns sr
        LEFT JOIN market_returns mr ON sr.date = mr.date
        LEFT JOIN theme_returns tr ON sr.date = tr.date
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['avg_stock_return']:
            return None

        row = result[0]
        idio_mom = float(row['idio_momentum']) if row['idio_momentum'] else 0
        stock_vol = float(row['stock_volatility']) if row['stock_volatility'] else 1
        sample_count = int(row['sample_count']) if row['sample_count'] else 0

        if sample_count < 10:
            return None  # 데이터 부족

        # Idiosyncratic momentum을 0-100 점수로 변환
        # 일평균 0.5% 이상이면 100점, -0.5% 이하면 0점
        if idio_mom >= 0.5:
            base_score = 100
        elif idio_mom <= -0.5:
            base_score = 0
        else:
            # -0.5% ~ 0.5% 범위를 0~100으로 선형 변환
            base_score = (idio_mom + 0.5) / 1.0 * 100

        # 변동성 조정: 고변동성 종목은 신호 약화
        if stock_vol > 5:
            vol_adjust = 0.7
        elif stock_vol > 3:
            vol_adjust = 0.85
        else:
            vol_adjust = 1.0

        final_score = min(100, max(0, base_score * vol_adjust))

        return final_score

    # ========================================================================
    # M25. Volatility Scaled Momentum (변동성 스케일링 모멘텀)
    # Based on Barroso & Santa-Clara (2015) - "Momentum has its Moments"
    # ========================================================================

    async def calculate_m25(self):
        """
        M25. Volatility Scaled Momentum Strategy
        Description: 목표 변동성(12%)으로 스케일링한 모멘텀
        Formula: VolScaledMom = RawMom × (TargetVol / RealizedVol)

        Research findings:
        - Risk of momentum is highly variable and predictable
        - Constant volatility targeting (12%) nearly doubled Sharpe ratio
        - Virtually eliminated momentum crashes
        """
        query = """
        WITH price_data AS (
            SELECT
                date,
                change_rate,
                close,
                LAG(close, 20) OVER (ORDER BY date) as close_20d_ago,
                STDDEV(change_rate) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as vol_20d
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
            ORDER BY date DESC
            LIMIT 30
        )
        SELECT
            change_rate as latest_return,
            close,
            close_20d_ago,
            vol_20d,
            CASE
                WHEN close_20d_ago > 0 THEN (close - close_20d_ago) / close_20d_ago * 100
                ELSE 0
            END as return_20d
        FROM price_data
        WHERE vol_20d IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['vol_20d']:
            return None

        row = result[0]
        return_20d = float(row['return_20d']) if row['return_20d'] else 0
        vol_20d = float(row['vol_20d']) if row['vol_20d'] else 1

        # 일별 변동성을 연환산 (√252)
        annual_vol = vol_20d * (252 ** 0.5)

        if annual_vol <= 0:
            return None

        # Volatility Scaling: 목표 변동성 12%
        vol_scale = min(2.0, max(0.3, TARGET_VOLATILITY / annual_vol))

        # Raw momentum (20일 수익률)을 vol-scaled로 변환
        scaled_return = return_20d * vol_scale

        # -20% ~ +20% 범위를 0~100으로 변환
        if scaled_return >= 20:
            score = 100
        elif scaled_return <= -20:
            score = 0
        else:
            score = (scaled_return + 20) / 40 * 100

        return min(100, max(0, score))

    # ========================================================================
    # M26. Mean Reversion Risk Score (평균회귀 위험 점수)
    # Based on empirical analysis: RSI > 75 + MA20 deviation > 15% = crash risk
    # ========================================================================

    async def calculate_m26(self):
        """
        M26. Mean Reversion Risk Detection Strategy
        Description: 과매수/급등 상태 탐지 및 평균회귀 위험 경고

        Risk Factors:
        - RSI > 80: 극단적 과매수 (penalty 40)
        - RSI > 75: 과매수 (penalty 25)
        - MA20 괴리 > 20%: 극단적 급등 (penalty 30)
        - MA20 괴리 > 15%: 급등 (penalty 20)
        - BB 상단 돌파 + RSI > 65: 복합 경고 (penalty 15)

        Returns 100 - risk_penalty (higher score = lower risk)
        """
        query = """
        WITH stock_data AS (
            SELECT
                t.close,
                t.change_rate,
                i.rsi,
                i.real_upper_band as bb_upper,
                i.real_middle_band as bb_mid,
                AVG(t.close) OVER (ORDER BY t.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20
            FROM kr_intraday_total t
            JOIN kr_indicators i ON t.symbol = i.symbol AND t.date = i.date
            WHERE t.symbol = $1
                AND ($2::date IS NULL OR t.date <= $2)
            ORDER BY t.date DESC
            LIMIT 1
        )
        SELECT
            close,
            rsi,
            bb_upper,
            ma20,
            CASE WHEN ma20 > 0 THEN (close / ma20 - 1) * 100 ELSE 0 END as ma20_deviation,
            CASE WHEN close > bb_upper THEN 1 ELSE 0 END as above_bb_upper
        FROM stock_data
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or result[0]['close'] is None:
            return None

        row = result[0]
        rsi = float(row['rsi']) if row['rsi'] else 50
        ma20_dev = float(row['ma20_deviation']) if row['ma20_deviation'] else 0
        above_bb = int(row['above_bb_upper']) if row['above_bb_upper'] else 0

        # Calculate risk penalty
        risk_penalty = 0

        # RSI-based risk
        if rsi >= MEAN_REVERSION_RISK_CONFIG['RSI_EXTREME']['threshold']:
            risk_penalty += MEAN_REVERSION_RISK_CONFIG['RSI_EXTREME']['penalty']
        elif rsi >= MEAN_REVERSION_RISK_CONFIG['RSI_OVERBOUGHT']['threshold']:
            risk_penalty += MEAN_REVERSION_RISK_CONFIG['RSI_OVERBOUGHT']['penalty']

        # MA20 deviation risk
        if ma20_dev >= MEAN_REVERSION_RISK_CONFIG['MA20_EXTREME_DEV']['threshold']:
            risk_penalty += MEAN_REVERSION_RISK_CONFIG['MA20_EXTREME_DEV']['penalty']
        elif ma20_dev >= MEAN_REVERSION_RISK_CONFIG['MA20_HIGH_DEV']['threshold']:
            risk_penalty += MEAN_REVERSION_RISK_CONFIG['MA20_HIGH_DEV']['penalty']

        # BB + RSI combo risk
        if above_bb and rsi >= MEAN_REVERSION_RISK_CONFIG['BB_RSI_COMBO']['rsi_threshold']:
            risk_penalty += MEAN_REVERSION_RISK_CONFIG['BB_RSI_COMBO']['penalty']

        # Final score: 100 - risk_penalty (higher = safer)
        # Max penalty = 85, so min score = 15
        final_score = max(0, 100 - risk_penalty)

        return final_score

    # ========================================================================
    # Helper: Get Market Volatility Regime
    # ========================================================================

    async def _get_market_volatility_regime(self):
        """
        Get current market volatility regime for regime-switching

        Returns:
            dict: {
                'regime': 'HIGH_VOL' | 'MED_VOL' | 'LOW_VOL',
                'vol_20d': float,
                'momentum_weight': float (0.5 ~ 1.0)
            }
        """
        query = """
        SELECT
            STDDEV(change_rate) as vol_20d
        FROM (
            SELECT change_rate
            FROM market_index
            WHERE exchange = 'KOSPI'
                AND ($1::date IS NULL OR date <= $1)
            ORDER BY date DESC
            LIMIT 20
        ) recent_data
        """

        result = await self.execute_query(query, self.analysis_date)

        if not result or result[0]['vol_20d'] is None:
            return {'regime': 'LOW_VOL', 'vol_20d': 1.0, 'momentum_weight': 1.0}

        vol_20d = float(result[0]['vol_20d'])

        if vol_20d > VOLATILITY_REGIME_CONFIG['HIGH_VOL']['threshold']:
            regime = 'HIGH_VOL'
            weight = VOLATILITY_REGIME_CONFIG['HIGH_VOL']['momentum_weight']
        elif vol_20d > VOLATILITY_REGIME_CONFIG['MED_VOL']['threshold']:
            regime = 'MED_VOL'
            weight = VOLATILITY_REGIME_CONFIG['MED_VOL']['momentum_weight']
        else:
            regime = 'LOW_VOL'
            weight = VOLATILITY_REGIME_CONFIG['LOW_VOL']['momentum_weight']

        return {
            'regime': regime,
            'vol_20d': vol_20d,
            'momentum_weight': weight
        }

    # ========================================================================
    # Helper: Get Stock Theme
    # ========================================================================

    async def _get_stock_theme(self):
        """
        Get stock's theme from kr_stock_detail

        Returns:
            str: Theme name or None
        """
        query = "SELECT theme FROM kr_stock_detail WHERE symbol = $1 LIMIT 1"
        result = await self.execute_query(query, self.symbol)

        if result and result[0]['theme']:
            return result[0]['theme']
        return None

    # ========================================================================
    # Calculate All Momentum Factor Scores
    # ========================================================================

    async def calculate_all_strategies(self):
        """
        Calculate all 26 momentum factor strategies (M1-M23 + M24-M26 new strategies)
        Returns: dict of {strategy_name: score}

        New strategies (Phase 3.11):
        - M24: Idiosyncratic Momentum (잔차 모멘텀) - Hanauer & Windmüller research
        - M25: Volatility Scaled Momentum - Barroso & Santa-Clara (2015)
        - M26: Mean Reversion Risk Score - 과매수/급등 위험 탐지
        """
        logger.info(f"Calculating all momentum factor strategies for {self.symbol}")

        strategies = {
            'M1_Price_Momentum': await self.calculate_m1(),
            'M2_Earnings_Momentum': await self.calculate_m2(),
            'M3_52W_High_Proximity': await self.calculate_m3(),
            'M4_Relative_Strength': await self.calculate_m4(),
            'M5_Volume_Momentum': await self.calculate_m5(),
            'M6_Institutional_Foreign_Buying': await self.calculate_m6(),
            'M7_Moving_Average_Alignment': await self.calculate_m7(),
            'M8_Breakout_Momentum': await self.calculate_m8(),
            'M9_RSI_Momentum': await self.calculate_m9(),
            'M10_MACD_Momentum': await self.calculate_m10(),
            'M11_Stochastic_Momentum': await self.calculate_m11(),
            'M12_Bollinger_Bands_Breakout': await self.calculate_m12(),
            'M13_Block_Trade_Momentum': await self.calculate_m13(),
            'M14_Composite_Technical_Indicators': await self.calculate_m14(),
            'M15_Foreign_Ownership_Increase': await self.calculate_m15(),
            'M16_Trading_Value_Surge': await self.calculate_m16(),
            'M17_Consecutive_Up_Days': await self.calculate_m17(),
            'M18_Gap_Up_Momentum': await self.calculate_m18(),
            'M19_New_High_Streak': await self.calculate_m19(),
            'M20_ATR_Breakout': await self.calculate_m20(),
            'M21_Reserved': await self.calculate_m21(),
            'M22_Reserved': await self.calculate_m22(),
            'M23_Market_Volatility_Regime': await self.calculate_m23(),
            # Phase 3.11 New Strategies (학술 연구 기반)
            'M24_Idiosyncratic_Momentum': await self.calculate_m24(),
            'M25_Volatility_Scaled_Momentum': await self.calculate_m25(),
            'M26_Mean_Reversion_Risk': await self.calculate_m26()
        }

        self.strategies_scores = strategies

        return strategies

    async def calculate_comprehensive_score(self):
        """
        Calculate comprehensive momentum factor score
        Formula: Comprehensive Score = Σ(Strategy Score) for all non-None strategies
        """
        if not self.strategies_scores:
            await self.calculate_all_strategies()

        valid_scores = [score for score in self.strategies_scores.values() if score is not None]

        if not valid_scores:
            return None

        comprehensive_score = sum(valid_scores)

        return comprehensive_score

    async def calculate_weighted_score(self, market_state=None):
        """
        Calculate weighted average score based on market state, sector, volatility regime, and theme

        Phase 3.11 Enhancements:
        - Regime-Switching: 시장 변동성에 따른 모멘텀 가중치 조정
        - Theme Override: Telecom_Media 등 역작동 테마 특별 처리
        - Mean Reversion Risk: M26 점수를 활용한 위험 감산

        Formula:
            Base Score = Σ(Strategy Score × Weight) / Σ(Weight)
            Regime Adjusted = Base Score × Regime Weight (0.5~1.0)
            Theme Adjusted = Regime Adjusted (or 0 if disabled theme)
            Final Score = Theme Adjusted × Sector Multiplier

        Args:
            market_state: Market state classification (one of 19 states)
                         If None, uses self.market_state

        Returns:
            dict: Enhanced result with regime/theme info
            Returns None if no valid scores or weights
        """
        # Use provided market_state or instance variable
        if market_state is None:
            market_state = self.market_state

        # Default to '기타' if no market state provided
        if market_state is None:
            market_state = '기타'
            logger.warning(f"No market state provided for {self.symbol}, using '기타'")

        # Get sector multiplier (uses existing kr_stock_detail.industry)
        sector_multiplier = await self._get_sector_multiplier()

        # Calculate strategy scores if not already done
        if not self.strategies_scores:
            await self.calculate_all_strategies()

        # ========================================================================
        # Phase 3.11: Get Volatility Regime
        # ========================================================================
        volatility_regime = await self._get_market_volatility_regime()
        regime_weight = volatility_regime['momentum_weight']

        # ========================================================================
        # Phase 3.11: Check Theme Override
        # ========================================================================
        theme = await self._get_stock_theme()
        theme_disabled = theme in MOMENTUM_DISABLED_THEMES if theme else False

        # Get weights for this market state
        weights = MOMENTUM_STRATEGY_WEIGHTS.get(market_state)

        if weights is None:
            logger.error(f"Invalid market state: {market_state}, using '기타'")
            weights = MOMENTUM_STRATEGY_WEIGHTS['기타']
            market_state = '기타'

        # ========================================================================
        # Phase 3.11: Add default weights for new strategies M24-M26
        # ========================================================================
        # M24 (Idiosyncratic): High weight - best performing strategy per research
        # M25 (Vol Scaled): Medium weight - volatility-adjusted momentum
        # M26 (Mean Reversion Risk): Used as penalty, not direct weight
        default_new_weights = {
            'M24': 2.0,  # Idiosyncratic Momentum - highest weight (research backed)
            'M25': 1.5,  # Volatility Scaled - medium-high weight
            'M26': 0.0   # Mean Reversion Risk - not weighted, used as penalty
        }

        # Calculate weighted average
        weighted_sum = 0.0
        weight_sum = 0.0
        valid_count = 0

        for strategy_name, score in self.strategies_scores.items():
            if score is not None:
                # Validate strategy score range
                if abs(score) > 100:
                    logger.error(f"[{self.symbol}] ABNORMAL STRATEGY: {strategy_name} = {score:.2f} (expected: 0-100)")
                elif score < 0:
                    logger.warning(f"[{self.symbol}] NEGATIVE STRATEGY: {strategy_name} = {score:.2f}")

                # Extract strategy key (e.g., 'M1_Price_Momentum_3M' -> 'M1')
                strategy_key = strategy_name.split('_')[0]

                # Get weight (check new weights first, then market state weights)
                if strategy_key in default_new_weights:
                    weight = default_new_weights[strategy_key]
                else:
                    weight = weights.get(strategy_key, 1.0)

                # Skip M26 from weighted sum (it's used as penalty)
                if strategy_key == 'M26':
                    continue

                weighted_sum += float(score) * weight
                weight_sum += weight
                valid_count += 1

        # Normalize by weight sum
        if weight_sum > 0 and valid_count > 0:
            base_weighted_score = weighted_sum / weight_sum

            # ========================================================================
            # Phase 3.11: Apply Regime-Switching
            # ========================================================================
            regime_adjusted_score = base_weighted_score * regime_weight

            # ========================================================================
            # Phase 3.11: Apply Theme Override
            # ========================================================================
            if theme_disabled:
                # Telecom_Media and similar themes: cap momentum score at 30
                theme_adjusted_score = min(30, regime_adjusted_score * 0.5)
                logger.warning(f"[{self.symbol}] Theme '{theme}' momentum disabled, capped to {theme_adjusted_score:.1f}")
            else:
                theme_adjusted_score = regime_adjusted_score

            # ========================================================================
            # Phase 3.11: Apply Mean Reversion Risk Penalty
            # ========================================================================
            m26_score = self.strategies_scores.get('M26_Mean_Reversion_Risk')
            if m26_score is not None and m26_score < 50:
                # M26 < 50 means high risk (RSI > 75 or MA20 deviation > 15%)
                # Apply penalty proportional to risk
                risk_penalty = (50 - m26_score) * 0.5  # Max penalty = 25 points
                theme_adjusted_score = max(0, theme_adjusted_score - risk_penalty)
                logger.info(f"[{self.symbol}] Mean Reversion Risk penalty: -{risk_penalty:.1f} (M26={m26_score:.1f})")

            # Apply sector multiplier to get final score
            sector_adjusted_score = theme_adjusted_score * sector_multiplier

            # Cap at 0-100 range
            sector_adjusted_score = min(100, max(0, sector_adjusted_score))

            # Calculate simple average (for reference)
            valid_scores = [score for score in self.strategies_scores.values() if score is not None]
            simple_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            # Get sector name for logging
            sector_query = "SELECT industry FROM kr_stock_detail WHERE symbol = $1 LIMIT 1"
            sector_result = await self.execute_query(sector_query, self.symbol)
            sector_name = sector_result[0]['industry'] if sector_result and sector_result[0]['industry'] else 'Unknown'

            result = {
                'weighted_score': round(sector_adjusted_score, 1),  # Final score with all adjustments
                'base_weighted_score': round(base_weighted_score, 1),  # Before any adjustments
                'regime_adjusted_score': round(regime_adjusted_score, 1),  # After regime adjustment
                'theme_adjusted_score': round(theme_adjusted_score, 1),  # After theme adjustment
                'simple_average': round(simple_avg, 1),  # Simple average for reference
                'sector_multiplier': round(sector_multiplier, 2),
                'sector': sector_name,
                'weight_sum': round(weight_sum, 2),
                'market_state': market_state,
                'valid_strategies': valid_count,
                # Phase 3.11 new fields
                'volatility_regime': volatility_regime['regime'],
                'volatility_20d': round(volatility_regime['vol_20d'], 3),
                'regime_weight': regime_weight,
                'theme': theme,
                'theme_disabled': theme_disabled,
                'm26_mean_reversion_risk': m26_score,
                'strategies': self.strategies_scores
            }

            logger.info(f"Weighted momentum score for {self.symbol}: {sector_adjusted_score:.2f} "
                       f"(base: {base_weighted_score:.2f}, regime: {volatility_regime['regime']} ×{regime_weight}, "
                       f"theme: {theme} {'[DISABLED]' if theme_disabled else ''}, "
                       f"sector: {sector_name} ×{sector_multiplier:.2f})")

            return result
        else:
            logger.warning(f"No valid scores for weighted calculation: {self.symbol}")
            return None

    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")



async def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("Korean Stock Momentum Factor System (Enhanced with Market State Weighting)")
    print("="*80 + "\n")

    # Get symbol input
    symbol = input("Enter stock symbol: ").strip()

    if not symbol:
        print("No symbol entered.")
        return

    try:
        # Step 1: Determine market state using weight.py
        print("\nStep 1: Analyzing market conditions...")
        try:
            from weight import ConditionAnalyzer
            analyzer = ConditionAnalyzer(symbol)
            conditions, _ = analyzer.analyze()
            market_state = conditions.get('market_state', '기타')
            print(f"Market State: {market_state}")
        except Exception as e:
            logger.warning(f"Could not determine market state: {e}")
            market_state = '기타'
            print(f"Market State: {market_state} (default)")

        # Step 2: Create calculator with market state
        print("\nStep 2: Calculating momentum factor strategies...")
        from db_async import AsyncDatabaseManager
        db_manager = AsyncDatabaseManager()
        await db_manager.initialize()

        calculator = MomentumFactorCalculator(symbol, db_manager, market_state=market_state)

        # Calculate all strategies
        strategies = await calculator.calculate_all_strategies()

        # Get weights for current market state
        weights = MOMENTUM_STRATEGY_WEIGHTS.get(market_state, MOMENTUM_STRATEGY_WEIGHTS['기타'])

        # Display results
        print("\n" + "="*80)
        print(f"Momentum Factor Strategies for {symbol}")
        print(f"Market State: {market_state}")
        print("="*80 + "\n")

        print(f"{'Strategy':<40s} {'Score':>8s} {'Weight':>8s} {'Weighted':>10s}")
        print("-"*80)

        for strategy_name, score in strategies.items():
            strategy_key = strategy_name.split('_')[0]
            weight = weights.get(strategy_key, 1.0)

            if score is not None:
                weighted = score * weight
                print(f"{strategy_name:<40s} {score:8.2f} {weight:8.2f} {weighted:10.2f}")
            else:
                print(f"{strategy_name:<40s} {'N/A':>8s} {weight:8.2f} {'N/A':>10s}")

        # Calculate scores
        comprehensive_score = await calculator.calculate_comprehensive_score()
        weighted_result = await calculator.calculate_weighted_score()

        # Display final scores
        print("\n" + "="*80)
        print("Final Scores")
        print("="*80)
        if comprehensive_score:
            print(f"Simple Sum (unweighted):        {comprehensive_score:.2f} points")
        else:
            print("Simple Sum (unweighted):        N/A")

        if weighted_result:
            print(f"\n[Market State-based Weighted Score]")
            print(f"Market State:                   {weighted_result['market_state']}")
            print(f"Weight Sum:                     {weighted_result['weight_sum']:.2f}")
            print(f"Valid Strategies:               {weighted_result['valid_strategies']}/20")
            print(f"Weighted Average Score:         {weighted_result['weighted_score']:.2f} / 100")
        else:
            print("\nWeighted score calculation failed.")

        print("="*80 + "\n")

        calculator.close_connection()

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
