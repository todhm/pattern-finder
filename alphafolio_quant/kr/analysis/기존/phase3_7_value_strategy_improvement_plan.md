# Phase 3.7 - Value Strategy Improvement Plan
## 한국 시장 맞춤형 밸류 전략 개선 작업계획서

**작성일**: 2025-11-22
**목표**: IC 최적화가 아닌, 한국 시장의 본질을 반영한 퀀트 전략 구축
**현재 문제**: Value IC -0.031 (역효과)
**개선 목표**: Value IC +0.02~0.05 (논리적 타당성 기반)

---

## 📋 목차

1. [작업 개요](#작업-개요)
2. [Phase 1: 즉시 적용 (1-2주)](#phase-1-즉시-적용)
3. [Phase 2: 중기 개선 (2-4주)](#phase-2-중기-개선)
4. [Phase 3: 가중치 업데이트](#phase-3-가중치-업데이트)
5. [Phase 4: 테스트 및 검증](#phase-4-테스트-및-검증)
6. [파일별 수정 체크리스트](#파일별-수정-체크리스트)
7. [코드 구현 상세](#코드-구현-상세)
8. [검증 기준](#검증-기준)

---

## 작업 개요

### 핵심 원칙

1. **IC 최적화가 목표가 아니다** - 전략의 논리적 타당성이 우선
2. **한국 시장 특성 반영** - Growth > Value, 적자 성장주 고수익
3. **섹터 맹신 금지** - 망한 섹터에서 1등해도 의미 없음
4. **과적합 방지** - 시장 원리 기반 설계

### 주요 개선 사항

| 항목 | 현재 문제 | 개선 방안 | 예상 효과 |
|------|----------|----------|----------|
| 적자 성장주 | 0점 → 기회 손실 | V17 추가 (PSR+Growth) | +141% 수익 포착 |
| 배당 함정 | 100점 → Value Trap | V4 재설계 (지속가능성) | -12% 손실 회피 |
| 망한 섹터 | 섹터 내 1등 90점 | Sector Health Check | 절대 평가로 전환 |
| EV/Sales | 성장률 미반영 | V18 추가 (Growth 조정) | 고성장주 인식 |
| Quality+Value | 단순 결합 | V20 (ROIC-Based) | 진짜 Quality 포착 |

---

## Phase 1: 즉시 적용 (1-2주)

### Task 1.1: V17 - 적자 성장 기업 전략 추가

**목표**: 적자 기업을 0점 처리하지 않고 성장성으로 평가

**수정 파일**: `kr\kr_value_factor.py`

#### Step 1: V17 함수 추가

**위치**: `kr\kr_value_factor.py` 하단 (기존 V16 다음)

```python
# ========================================================================
# V17. Growth-Adjusted Value for Loss-Making Companies (신규)
# ========================================================================

async def calculate_v17_growth_adjusted_value(self):
    """
    V17. Growth-Adjusted Value Strategy (적자 성장 기업용)

    Description: 적자 기업을 성장성과 PSR로 평가

    Logic:
    - 매출 성장률 (3년 CAGR, 1년 성장률)
    - 영업 마진 개선 추세
    - 현금 소진율 (생존 가능성)
    - PSR (섹터 대비)

    Scoring:
    - High Growth (CAGR >50%, 1Y >30%, 마진 개선, 생존 >12개월): 85점
    - Mid Growth (CAGR >20%, 1Y >10%, 마진 유지, 생존 >6개월): 65점
    - Low Growth / Troubled: 0-10점
    """

    # Step 1: 적자 여부 확인
    net_income_query = """
    SELECT thstrm_amount as net_income
    FROM kr_financial_position
    WHERE symbol = $1
        AND sj_div = 'IS'
        AND account_nm IN ('당기순이익(손실)', '당기순이익')
        AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
    ORDER BY bsns_year DESC, rcept_dt DESC
    LIMIT 1
    """

    ni_result = await self.execute_query(net_income_query, self.symbol, self.analysis_date)

    if not ni_result or ni_result[0]['net_income'] is None:
        return None

    net_income = float(ni_result[0]['net_income'])

    # 흑자면 이 전략 사용 안함 (V1~V16 사용)
    if net_income > 0:
        return None

    # Step 2: 매출 데이터 수집 (최근 3년)
    sales_query = """
    WITH sales_data AS (
        SELECT
            bsns_year,
            thstrm_amount as sales,
            ROW_NUMBER() OVER (PARTITION BY bsns_year ORDER BY rcept_dt DESC) as rn
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm IN ('매출액', '수익(매출액)')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            AND thstrm_amount > 0
        ORDER BY bsns_year DESC
    )
    SELECT bsns_year, sales
    FROM sales_data
    WHERE rn = 1
    ORDER BY bsns_year DESC
    LIMIT 3
    """

    sales_result = await self.execute_query(sales_query, self.symbol, self.analysis_date)

    if not sales_result or len(sales_result) < 2:
        return 0  # 매출 데이터 부족

    # Step 3: 성장률 계산
    sales_current = float(sales_result[0]['sales'])
    sales_1y_ago = float(sales_result[1]['sales']) if len(sales_result) > 1 else None
    sales_3y_ago = float(sales_result[2]['sales']) if len(sales_result) > 2 else None

    # 1년 성장률
    if sales_1y_ago and sales_1y_ago > 0:
        sales_growth_1y = (sales_current - sales_1y_ago) / sales_1y_ago
    else:
        sales_growth_1y = 0

    # 3년 CAGR
    if sales_3y_ago and sales_3y_ago > 0:
        sales_growth_3y = (sales_current / sales_3y_ago) ** (1/3) - 1
    else:
        sales_growth_3y = sales_growth_1y  # 3년 데이터 없으면 1년으로 대체

    # Step 4: 영업 마진 추세
    margin_query = """
    WITH margin_data AS (
        SELECT
            bsns_year,
            op.thstrm_amount as operating_profit,
            sales.thstrm_amount as sales,
            (op.thstrm_amount::DECIMAL / NULLIF(sales.thstrm_amount, 0)) * 100 as operating_margin
        FROM kr_financial_position op
        JOIN kr_financial_position sales
            ON op.symbol = sales.symbol
            AND op.bsns_year = sales.bsns_year
            AND op.rcept_dt = sales.rcept_dt
        WHERE op.symbol = $1
            AND op.sj_div = 'IS'
            AND sales.sj_div = 'IS'
            AND op.account_nm IN ('영업이익(손실)', '영업이익')
            AND sales.account_nm IN ('매출액', '수익(매출액)')
            AND op.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC
        LIMIT 2
    )
    SELECT operating_margin
    FROM margin_data
    ORDER BY bsns_year DESC
    """

    margin_result = await self.execute_query(margin_query, self.symbol, self.analysis_date)

    operating_margin_trend = 0
    if margin_result and len(margin_result) >= 2:
        margin_current = float(margin_result[0]['operating_margin']) if margin_result[0]['operating_margin'] else -999
        margin_prev = float(margin_result[1]['operating_margin']) if margin_result[1]['operating_margin'] else -999
        if margin_current > -999 and margin_prev > -999:
            operating_margin_trend = margin_current - margin_prev

    # Step 5: 현금 소진율 (생존 가능성)
    cash_query = """
    WITH latest_cf AS (
        SELECT
            cf.thstrm_amount as operating_cash_flow,
            cash.thstrm_amount as cash_and_equivalents
        FROM kr_financial_position cf
        JOIN kr_financial_position cash
            ON cf.symbol = cash.symbol
            AND cf.bsns_year = cash.bsns_year
            AND cf.rcept_dt = cash.rcept_dt
        WHERE cf.symbol = $1
            AND cf.sj_div = 'CF'
            AND cash.sj_div = 'BS'
            AND cf.account_nm LIKE '%영업활동%현금흐름%'
            AND cash.account_nm LIKE '%현금%'
            AND cf.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY cf.bsns_year DESC, cf.rcept_dt DESC
        LIMIT 1
    )
    SELECT operating_cash_flow, cash_and_equivalents
    FROM latest_cf
    """

    cash_result = await self.execute_query(cash_query, self.symbol, self.analysis_date)

    months_of_runway = 999  # 기본값: 충분
    if cash_result and cash_result[0]['operating_cash_flow'] and cash_result[0]['cash_and_equivalents']:
        ocf = float(cash_result[0]['operating_cash_flow'])
        cash = float(cash_result[0]['cash_and_equivalents'])
        if ocf < 0 and cash > 0:
            monthly_burn = abs(ocf) / 12
            if monthly_burn > 0:
                months_of_runway = cash / monthly_burn

    # Step 6: PSR (Price-to-Sales Ratio)
    psr_query = """
    SELECT
        kit.market_cap,
        fp.thstrm_amount as sales
    FROM kr_intraday_total kit
    LEFT JOIN (
        SELECT symbol, thstrm_amount
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm IN ('매출액', '수익(매출액)')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC
        LIMIT 1
    ) fp ON kit.symbol = fp.symbol
    WHERE kit.symbol = $1
        AND ($2::date IS NULL OR kit.date = $2)
    ORDER BY kit.date DESC
    LIMIT 1
    """

    psr_result = await self.execute_query(psr_query, self.symbol, self.analysis_date)

    psr = None
    if psr_result and psr_result[0]['market_cap'] and psr_result[0]['sales']:
        market_cap = float(psr_result[0]['market_cap'])
        sales = float(psr_result[0]['sales'])
        if sales > 0:
            psr = market_cap / sales

    # Step 7: 섹터 중앙값 PSR 계산
    sector_psr_query = """
    WITH sector_stocks AS (
        SELECT sd.symbol, sd.theme
        FROM kr_stock_detail sd
        WHERE sd.symbol = $1
    ),
    peer_psr AS (
        SELECT
            kit.symbol,
            kit.market_cap / NULLIF(fp.thstrm_amount, 0) as psr
        FROM kr_intraday_total kit
        JOIN kr_stock_detail sd ON kit.symbol = sd.symbol
        LEFT JOIN (
            SELECT DISTINCT ON (symbol)
                symbol, thstrm_amount
            FROM kr_financial_position
            WHERE sj_div = 'IS'
                AND account_nm IN ('매출액', '수익(매출액)')
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY symbol, bsns_year DESC, rcept_dt DESC
        ) fp ON kit.symbol = fp.symbol
        WHERE sd.theme = (SELECT theme FROM sector_stocks)
            AND ($2::date IS NULL OR kit.date = $2)
            AND kit.market_cap > 0
            AND fp.thstrm_amount > 0
    )
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY psr) as median_psr
    FROM peer_psr
    WHERE psr > 0 AND psr < 100
    """

    sector_psr_result = await self.execute_query(sector_psr_query, self.symbol, self.analysis_date)

    sector_median_psr = 2.0  # 기본값
    if sector_psr_result and sector_psr_result[0]['median_psr']:
        sector_median_psr = float(sector_psr_result[0]['median_psr'])

    # Step 8: 점수 계산
    score = 0

    # High Growth (초고성장)
    if (sales_growth_3y > 0.50 and          # 연 50% 이상 성장
        sales_growth_1y > 0.30 and          # 최근도 30% 이상
        operating_margin_trend > 0 and      # 마진 개선 중
        months_of_runway > 12):             # 1년 이상 생존 가능

        if psr and psr < sector_median_psr * 0.7:   # 섹터 대비 30% 할인
            score = 85  # 고성장 + 저평가 = 최고 점수
        elif psr and psr < sector_median_psr:
            score = 70
        else:
            score = 50  # 성장하지만 고평가

    # Mid Growth (중성장)
    elif (sales_growth_3y > 0.20 and
          sales_growth_1y > 0.10 and
          operating_margin_trend > -0.05 and
          months_of_runway > 6):

        if psr and psr < sector_median_psr * 0.8:
            score = 65
        elif psr and psr < sector_median_psr:
            score = 45
        else:
            score = 25

    # Low Growth / Troubled (저성장 / 위험)
    else:
        if months_of_runway < 6:
            score = 0   # 생존 위험
        elif sales_growth_1y < 0:
            score = 0   # 매출 역성장
        else:
            score = 10  # 위험하지만 일단 생존

    logger.info(f"V17 Score: {score:.1f} (Sales Growth 3Y: {sales_growth_3y*100:.1f}%, 1Y: {sales_growth_1y*100:.1f}%, Runway: {months_of_runway:.0f}M)")

    return score
```

#### Step 2: V1 함수 수정 (적자 폴백)

**위치**: `kr\kr_value_factor.py` 라인 190-259 (기존 calculate_v1 함수)

**수정 전**:
```python
if per is None or per <= 0:
    return None  # ❌ None = 0점 처리
```

**수정 후**:
```python
if per is None or per <= 0:
    # 적자 기업: V17로 폴백
    logger.info(f"V1: PER invalid ({per}), falling back to V17")
    return await self.calculate_v17_growth_adjusted_value()  # ✅ V17로 위임
```

**전체 수정 코드**:
```python
async def calculate_v1(self):
    """
    V1. Low PER Strategy (Enhanced with V17 fallback)
    Description: Find undervalued stocks with low price-to-earnings ratio
    Score = 100 - MIN(100, MAX(0, (PER / 15) × 50))
    Condition: PER > 0 (흑자), PER <= 0 (V17로 폴백)
    Interpretation: PER <= 15 gets 100 points, PER >= 30 gets 0 points
    """
    query = """
    SELECT per
    FROM kr_intraday_detail
    WHERE symbol = $1
    """

    result = await self.execute_query(query, self.symbol)
    per = None

    # Try to get PER from kr_intraday_detail first
    if result and result[0]['per'] is not None and result[0]['per'] > 0:
        per = float(result[0]['per'])

    # Fallback: Calculate PER from market cap and net income
    if per is None:
        fallback_query = """
        SELECT
            kit.market_cap,
            fp.thstrm_amount as net_income
        FROM kr_intraday_total kit
        LEFT JOIN (
            SELECT symbol, thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm IN ('당기순이익(손실)', '당기순이익')
                AND thstrm_amount > 0
                AND rcept_dt <= COALESCE($3::date, CURRENT_DATE)
            ORDER BY
                bsns_year DESC,
                rcept_dt DESC,
                CASE
                    WHEN report_code = '11011' THEN 1
                    WHEN report_code = '11012' THEN 2
                    WHEN report_code = '11013' THEN 3
                    WHEN report_code = '11014' THEN 4
                    ELSE 5
                END,
                thstrm_amount DESC
            LIMIT 1
        ) fp ON kit.symbol = fp.symbol
        WHERE kit.symbol = $2
            AND ($3::date IS NULL OR kit.date = $3)
        ORDER BY kit.date DESC
        LIMIT 1
        """

        fallback_result = await self.execute_query(fallback_query, self.symbol, self.symbol, self.analysis_date)

        if fallback_result and fallback_result[0]['market_cap'] and fallback_result[0]['net_income']:
            mktcap = float(fallback_result[0]['market_cap'])
            net_income = float(fallback_result[0]['net_income'])
            if net_income > 0 and mktcap > 0:
                per = mktcap / net_income

    # 적자 기업 처리: V17로 폴백
    if per is None or per <= 0:
        logger.info(f"V1: PER invalid ({per}), falling back to V17 (Growth-Adjusted Value)")
        return await self.calculate_v17_growth_adjusted_value()

    # 흑자 기업: 기존 로직
    score = 100 - min(100, max(0, (per / 15) * 50))

    return score
```

---

### Task 1.2: V4 - 지속 가능 배당 전략 재설계

**목표**: 단순 배당률이 아닌 배당 지속 가능성 평가

**수정 파일**: `kr\kr_value_factor.py`

#### Step 1: 기존 V4 백업

**위치**: `kr\kr_value_factor.py` 라인 700대 (기존 calculate_v4 함수 찾기)

**백업 코드**:
```python
async def calculate_v4_original(self):
    """
    V4. High Dividend Yield Strategy (ORIGINAL - DEPRECATED)

    Original IC: -0.057
    This version for comparison only
    """
    # 기존 코드 그대로 유지 (이름만 변경)
```

#### Step 2: V4 재설계 구현

**위치**: 기존 calculate_v4 함수 교체

```python
async def calculate_v4_sustainable_dividend(self):
    """
    V4. Sustainable Dividend Strategy (재설계)

    Description: 지속 가능한 배당 평가

    Components:
    1. Payout Ratio (배당성향): >80% = 위험
    2. FCF Coverage (현금흐름 커버리지): FCF < Dividend = 위험
    3. Dividend Growth (배당 성장): 3년 CAGR
    4. Dividend Yield: 최종 평가

    Scoring Logic:
    - 위험 신호 탐지: Payout >80%, FCF < Div, Div 감소 → 10-20점
    - 우수 배당: 성장 >5%, Payout <50%, FCF >1.5x → 75-90점
    - 안정 배당: 유지, Payout <60%, FCF >1.2x → 40-70점
    """

    # Step 1: 배당 데이터 수집
    dividend_query = """
    WITH latest_dividend AS (
        SELECT
            thstrm as current_dividend,
            frmtrm as prev_dividend,
            lwfr as prev2_dividend,
            stlm_dt
        FROM kr_dividends
        WHERE symbol = $1
            AND se LIKE '%배당%'
            AND stock_knd LIKE '%보통주%'
            AND stlm_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY stlm_dt DESC
        LIMIT 1
    )
    SELECT * FROM latest_dividend
    """

    div_result = await self.execute_query(dividend_query, self.symbol, self.analysis_date)

    if not div_result or div_result[0]['current_dividend'] is None:
        return None  # 배당 데이터 없음

    current_dividend = float(div_result[0]['current_dividend']) if div_result[0]['current_dividend'] else 0
    prev_dividend = float(div_result[0]['prev_dividend']) if div_result[0]['prev_dividend'] else 0
    prev2_dividend = float(div_result[0]['prev2_dividend']) if div_result[0]['prev2_dividend'] else 0

    if current_dividend <= 0:
        return None  # 무배당

    # Step 2: 주가 및 배당 수익률
    price_query = """
    SELECT close
    FROM kr_intraday_total
    WHERE symbol = $1
        AND ($2::date IS NULL OR date = $2)
    ORDER BY date DESC
    LIMIT 1
    """

    price_result = await self.execute_query(price_query, self.symbol, self.analysis_date)

    if not price_result or not price_result[0]['close']:
        return None

    price = float(price_result[0]['close'])
    dividend_yield = (current_dividend / price) * 100 if price > 0 else 0

    # Step 3: Payout Ratio (배당성향)
    payout_query = """
    WITH latest_financials AS (
        SELECT
            ni.thstrm_amount as net_income,
            eps.close as eps,
            eps.listed_shares
        FROM kr_financial_position ni
        LEFT JOIN kr_intraday_total eps ON ni.symbol = eps.symbol
        WHERE ni.symbol = $1
            AND ni.sj_div = 'IS'
            AND ni.account_nm IN ('당기순이익(손실)', '당기순이익')
            AND ni.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            AND ($2::date IS NULL OR eps.date = $2)
        ORDER BY ni.bsns_year DESC, ni.rcept_dt DESC, eps.date DESC
        LIMIT 1
    )
    SELECT net_income, listed_shares
    FROM latest_financials
    """

    payout_result = await self.execute_query(payout_query, self.symbol, self.analysis_date)

    payout_ratio = 999  # 기본값: 알 수 없음 (보수적)
    if payout_result and payout_result[0]['net_income'] and payout_result[0]['listed_shares']:
        net_income = float(payout_result[0]['net_income'])
        listed_shares = float(payout_result[0]['listed_shares'])
        if net_income > 0 and listed_shares > 0:
            total_dividend = current_dividend * listed_shares
            payout_ratio = (total_dividend / net_income) * 100

    # Step 4: FCF Coverage (잉여현금흐름 커버리지)
    fcf_query = """
    WITH latest_cf AS (
        SELECT
            ocf.thstrm_amount as operating_cash_flow,
            inv.thstrm_amount as capex
        FROM kr_financial_position ocf
        LEFT JOIN kr_financial_position inv
            ON ocf.symbol = inv.symbol
            AND ocf.bsns_year = inv.bsns_year
            AND ocf.rcept_dt = inv.rcept_dt
            AND inv.sj_div = 'CF'
            AND inv.account_nm LIKE '%투자활동%현금흐름%'
        WHERE ocf.symbol = $1
            AND ocf.sj_div = 'CF'
            AND ocf.account_nm LIKE '%영업활동%현금흐름%'
            AND ocf.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY ocf.bsns_year DESC, ocf.rcept_dt DESC
        LIMIT 1
    )
    SELECT operating_cash_flow, capex
    FROM latest_cf
    """

    fcf_result = await self.execute_query(fcf_query, self.symbol, self.analysis_date)

    fcf_coverage = 0  # 기본값: 커버 안됨 (보수적)
    if fcf_result and fcf_result[0]['operating_cash_flow']:
        ocf = float(fcf_result[0]['operating_cash_flow'])
        capex = float(fcf_result[0]['capex']) if fcf_result[0]['capex'] else 0
        fcf = ocf - abs(capex)  # FCF = OCF - CapEx

        # 총 배당금 계산
        if payout_result and payout_result[0]['listed_shares']:
            listed_shares = float(payout_result[0]['listed_shares'])
            total_dividend = current_dividend * listed_shares
            if total_dividend > 0:
                fcf_coverage = fcf / total_dividend

    # Step 5: 배당 성장률 (3년 CAGR)
    dividend_growth = 0
    if prev2_dividend > 0:
        dividend_growth = (current_dividend / prev2_dividend) ** (1/3) - 1
    elif prev_dividend > 0:
        dividend_growth = (current_dividend / prev_dividend) - 1

    # Step 6: 점수 계산
    score = 0

    # === 위험 신호 탐지 ===

    # 위험 1: 과도한 배당 (Payout Ratio > 80%)
    if payout_ratio > 80:
        logger.info(f"V4 WARNING: High payout ratio {payout_ratio:.1f}%")
        return 10  # 지속 불가능

    # 위험 2: FCF < 배당 (빚내서 배당)
    if fcf_coverage < 1.0:
        logger.info(f"V4 WARNING: FCF coverage {fcf_coverage:.2f}x < 1.0")
        return 15  # 위험

    # 위험 3: 배당 감소 추세
    if dividend_growth < -0.05:
        logger.info(f"V4 WARNING: Dividend declining {dividend_growth*100:.1f}%")
        return 20  # 배당 컷 위험

    # === 긍정 신호 평가 ===

    # 우수: 배당 증가 + 여유 있음
    if (dividend_growth > 0.05 and
        payout_ratio < 50 and
        fcf_coverage > 1.5):

        # 배당 수익률 기반 점수
        if dividend_yield > 4.0:      # 4% 이상
            score = 90
        elif dividend_yield > 3.0:    # 3% 이상
            score = 75
        else:
            score = 60

        logger.info(f"V4: Excellent dividend (Yield: {dividend_yield:.2f}%, Growth: {dividend_growth*100:.1f}%)")

    # 양호: 안정적
    elif (payout_ratio < 60 and
          fcf_coverage > 1.2 and
          dividend_growth > -0.02):

        if dividend_yield > 3.0:
            score = 70
        elif dividend_yield > 2.0:
            score = 55
        else:
            score = 40

        logger.info(f"V4: Stable dividend (Yield: {dividend_yield:.2f}%, Payout: {payout_ratio:.1f}%)")

    # 보통
    else:
        if dividend_yield > 3.0:
            score = 50
        else:
            score = 30

        logger.info(f"V4: Moderate dividend (Yield: {dividend_yield:.2f}%)")

    return score

# Alias for backward compatibility
async def calculate_v4(self):
    """V4 wrapper - calls sustainable dividend version"""
    return await self.calculate_v4_sustainable_dividend()
```

---

### Task 1.3: V18 - EV/Sales/Growth 전략 추가 (V9 대체)

**목표**: 단순 EV/Sales가 아닌 성장률로 조정된 밸류에이션

**수정 파일**: `kr\kr_value_factor.py`

#### Step 1: V9 백업

```python
async def calculate_v9_original(self):
    """
    V9. Low EV/Sales Strategy (ORIGINAL)

    Original IC: +0.055 (유일한 양수지만 약함)
    This version for comparison only
    """
    # 기존 코드 유지
```

#### Step 2: V18 추가

**위치**: `kr\kr_value_factor.py` 하단

```python
# ========================================================================
# V18. EV/Sales with Growth Adjustment (V9 개선 버전)
# ========================================================================

async def calculate_v18_ev_sales_growth(self):
    """
    V18. EV/Sales/Growth Strategy (V9 개선)

    Description: 성장률로 조정된 EV/Sales

    Logic:
    - EV/Sales/Growth = (EV/Sales) / (Sales Growth Rate)
    - PEG 비율과 유사한 개념
    - 성장 1%당 EV/Sales = "성장 보정 밸류에이션"

    Scoring:
    - 낮을수록 좋음 (고성장 저평가)
    - 섹터 내 상대 평가
    """

    # Step 1: Enterprise Value 계산
    ev_query = """
    WITH latest_data AS (
        SELECT
            kit.market_cap,
            fp_debt.thstrm_amount as total_debt,
            fp_cash.thstrm_amount as cash
        FROM kr_intraday_total kit
        LEFT JOIN (
            SELECT symbol, thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND account_nm LIKE '%부채%'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ) fp_debt ON kit.symbol = fp_debt.symbol
        LEFT JOIN (
            SELECT symbol, thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND account_nm LIKE '%현금%'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ) fp_cash ON kit.symbol = fp_cash.symbol
        WHERE kit.symbol = $1
            AND ($2::date IS NULL OR kit.date = $2)
        ORDER BY kit.date DESC
        LIMIT 1
    )
    SELECT market_cap, total_debt, cash
    FROM latest_data
    """

    ev_result = await self.execute_query(ev_query, self.symbol, self.analysis_date)

    if not ev_result or not ev_result[0]['market_cap']:
        return None

    market_cap = float(ev_result[0]['market_cap'])
    total_debt = float(ev_result[0]['total_debt']) if ev_result[0]['total_debt'] else 0
    cash = float(ev_result[0]['cash']) if ev_result[0]['cash'] else 0

    enterprise_value = market_cap + total_debt - cash

    if enterprise_value <= 0:
        return None

    # Step 2: Sales 및 성장률
    sales_query = """
    WITH sales_history AS (
        SELECT
            bsns_year,
            thstrm_amount as sales,
            ROW_NUMBER() OVER (PARTITION BY bsns_year ORDER BY rcept_dt DESC) as rn
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm IN ('매출액', '수익(매출액)')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC
    )
    SELECT bsns_year, sales
    FROM sales_history
    WHERE rn = 1
    ORDER BY bsns_year DESC
    LIMIT 3
    """

    sales_result = await self.execute_query(sales_query, self.symbol, self.analysis_date)

    if not sales_result or len(sales_result) < 2:
        return None

    sales_current = float(sales_result[0]['sales'])
    sales_3y_ago = float(sales_result[2]['sales']) if len(sales_result) > 2 else float(sales_result[1]['sales'])

    if sales_current <= 0 or sales_3y_ago <= 0:
        return None

    # 3년 CAGR
    years = 3 if len(sales_result) > 2 else 2
    sales_growth_rate = (sales_current / sales_3y_ago) ** (1/years) - 1

    # Step 3: EV/Sales 계산
    ev_sales = enterprise_value / sales_current

    # Step 4: EV/Sales/Growth 계산
    if sales_growth_rate > 0.05:  # 최소 5% 성장
        ev_sales_growth = ev_sales / (sales_growth_rate * 100)
    else:
        # 저성장 (<5%): 페널티
        ev_sales_growth = ev_sales * 10  # 높은 값 = 나쁨

    # Step 5: 섹터 내 백분위 계산
    sector_query = """
    WITH sector_stocks AS (
        SELECT sd.symbol, sd.theme
        FROM kr_stock_detail sd
        WHERE sd.symbol = $1
    ),
    peer_ev_sales_growth AS (
        SELECT
            kit.symbol,
            (kit.market_cap + COALESCE(debt.thstrm_amount, 0) - COALESCE(cash.thstrm_amount, 0)) /
            NULLIF(sales_current.thstrm_amount, 0) as ev_sales,
            CASE
                WHEN sales_growth.growth_rate > 0.05 THEN
                    ((kit.market_cap + COALESCE(debt.thstrm_amount, 0) - COALESCE(cash.thstrm_amount, 0)) /
                     NULLIF(sales_current.thstrm_amount, 0)) / (sales_growth.growth_rate * 100)
                ELSE
                    ((kit.market_cap + COALESCE(debt.thstrm_amount, 0) - COALESCE(cash.thstrm_amount, 0)) /
                     NULLIF(sales_current.thstrm_amount, 0)) * 10
            END as ev_sales_growth_metric
        FROM kr_intraday_total kit
        JOIN kr_stock_detail sd ON kit.symbol = sd.symbol
        LEFT JOIN (
            SELECT DISTINCT ON (symbol) symbol, thstrm_amount
            FROM kr_financial_position
            WHERE sj_div = 'BS' AND account_nm LIKE '%부채%'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY symbol, bsns_year DESC, rcept_dt DESC
        ) debt ON kit.symbol = debt.symbol
        LEFT JOIN (
            SELECT DISTINCT ON (symbol) symbol, thstrm_amount
            FROM kr_financial_position
            WHERE sj_div = 'BS' AND account_nm LIKE '%현금%'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY symbol, bsns_year DESC, rcept_dt DESC
        ) cash ON kit.symbol = cash.symbol
        LEFT JOIN (
            SELECT DISTINCT ON (symbol) symbol, thstrm_amount
            FROM kr_financial_position
            WHERE sj_div = 'IS' AND account_nm IN ('매출액', '수익(매출액)')
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY symbol, bsns_year DESC, rcept_dt DESC
        ) sales_current ON kit.symbol = sales_current.symbol
        LEFT JOIN (
            SELECT
                symbol,
                (MAX(thstrm_amount) / MIN(thstrm_amount)) ^ (1.0/2) - 1 as growth_rate
            FROM kr_financial_position
            WHERE sj_div = 'IS' AND account_nm IN ('매출액', '수익(매출액)')
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND bsns_year >= EXTRACT(YEAR FROM COALESCE($2::date, CURRENT_DATE)) - 3
            GROUP BY symbol
            HAVING COUNT(*) >= 2
        ) sales_growth ON kit.symbol = sales_growth.symbol
        WHERE sd.theme = (SELECT theme FROM sector_stocks)
            AND ($2::date IS NULL OR kit.date = $2)
            AND kit.market_cap > 0
            AND sales_current.thstrm_amount > 0
    )
    SELECT PERCENTILE_CONT(ARRAY[0.25, 0.5, 0.75]) WITHIN GROUP (ORDER BY ev_sales_growth_metric) as percentiles
    FROM peer_ev_sales_growth
    WHERE ev_sales_growth_metric > 0 AND ev_sales_growth_metric < 100
    """

    percentile_result = await self.execute_query(sector_query, self.symbol, self.analysis_date)

    # 백분위 기반 점수
    if percentile_result and percentile_result[0]['percentiles']:
        percentiles = percentile_result[0]['percentiles']
        p25, p50, p75 = percentiles[0], percentiles[1], percentiles[2]

        if ev_sales_growth < p25:
            score = 90  # 하위 25% = 최고 (낮을수록 좋음)
        elif ev_sales_growth < p50:
            score = 70  # 중앙값 이하
        elif ev_sales_growth < p75:
            score = 50  # 중앙값 이상
        else:
            score = 30  # 상위 25%
    else:
        # 절대 평가 폴백
        if ev_sales_growth < 2:
            score = 80
        elif ev_sales_growth < 5:
            score = 60
        elif ev_sales_growth < 10:
            score = 40
        else:
            score = 20

    logger.info(f"V18: EV/Sales/Growth = {ev_sales_growth:.2f} (EV/Sales: {ev_sales:.2f}, Growth: {sales_growth_rate*100:.1f}%), Score: {score:.1f}")

    return score
```

---

### Task 1.4: V20 - ROIC-Based Value 추가 (V6 대체)

**목표**: ROE+PER 단순 결합이 아닌 자본 효율성 기반 평가

**수정 파일**: `kr\kr_value_factor.py`

#### Step 1: V6 백업

```python
async def calculate_v6_original(self):
    """
    V6. High ROE + Low PER Strategy (ORIGINAL)

    Original IC: +0.030 (매우 약함)
    This version for comparison only
    """
    # 기존 코드 유지
```

#### Step 2: V20 추가

```python
# ========================================================================
# V20. ROIC-Based Value (V6 개선 버전)
# ========================================================================

async def calculate_v20_roic_value(self):
    """
    V20. ROIC-Based Value Strategy (V6 개선)

    Description: 자본 효율성 기반 밸류 평가

    Components:
    1. ROIC (Return on Invested Capital): NOPAT / Invested Capital
    2. EV/IC (Enterprise Value / Invested Capital)
    3. Magic Ratio: ROIC / (EV/IC)

    Logic:
    - ROIC: 투입 자본 대비 수익성 (ROE보다 정확)
    - EV/IC: 시장이 자본을 얼마로 평가하는가
    - Magic Ratio 높을수록 좋음 = 효율적인데 저평가

    Scoring:
    - ROIC >15% & EV/IC <1.5: 90점 (진짜 저평가)
    - ROIC >15% & EV/IC >2.5: 40점 (좋지만 비쌈)
    - ROIC <5% & EV/IC <1.5: 20점 (Value Trap)
    """

    # Step 1: NOPAT (Net Operating Profit After Tax)
    nopat_query = """
    WITH latest_is AS (
        SELECT
            op.thstrm_amount as operating_profit,
            tax.thstrm_amount as tax_expense,
            ni.thstrm_amount as net_income
        FROM kr_financial_position op
        LEFT JOIN kr_financial_position tax
            ON op.symbol = tax.symbol
            AND op.bsns_year = tax.bsns_year
            AND op.rcept_dt = tax.rcept_dt
            AND tax.sj_div = 'IS'
            AND tax.account_nm LIKE '%법인세%'
        LEFT JOIN kr_financial_position ni
            ON op.symbol = ni.symbol
            AND op.bsns_year = ni.bsns_year
            AND op.rcept_dt = ni.rcept_dt
            AND ni.sj_div = 'IS'
            AND ni.account_nm IN ('당기순이익(손실)', '당기순이익')
        WHERE op.symbol = $1
            AND op.sj_div = 'IS'
            AND op.account_nm IN ('영업이익(손실)', '영업이익')
            AND op.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY op.bsns_year DESC, op.rcept_dt DESC
        LIMIT 1
    )
    SELECT operating_profit, tax_expense, net_income
    FROM latest_is
    """

    nopat_result = await self.execute_query(nopat_query, self.symbol, self.analysis_date)

    if not nopat_result or not nopat_result[0]['operating_profit']:
        return None

    operating_profit = float(nopat_result[0]['operating_profit'])
    tax_expense = float(nopat_result[0]['tax_expense']) if nopat_result[0]['tax_expense'] else 0
    net_income = float(nopat_result[0]['net_income']) if nopat_result[0]['net_income'] else 0

    # Tax Rate 계산
    if net_income != 0:
        tax_rate = abs(tax_expense) / (net_income + abs(tax_expense))
    else:
        tax_rate = 0.25  # 기본 법인세율

    # NOPAT = Operating Profit × (1 - Tax Rate)
    nopat = operating_profit * (1 - tax_rate)

    # Step 2: Invested Capital = Total Assets - Current Liabilities
    ic_query = """
    WITH latest_bs AS (
        SELECT
            assets.thstrm_amount as total_assets,
            cl.thstrm_amount as current_liabilities
        FROM kr_financial_position assets
        LEFT JOIN kr_financial_position cl
            ON assets.symbol = cl.symbol
            AND assets.bsns_year = cl.bsns_year
            AND assets.rcept_dt = cl.rcept_dt
            AND cl.sj_div = 'BS'
            AND cl.account_nm LIKE '%유동부채%'
        WHERE assets.symbol = $1
            AND assets.sj_div = 'BS'
            AND assets.account_nm LIKE '%자산총계%'
            AND assets.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY assets.bsns_year DESC, assets.rcept_dt DESC
        LIMIT 1
    )
    SELECT total_assets, current_liabilities
    FROM latest_bs
    """

    ic_result = await self.execute_query(ic_query, self.symbol, self.analysis_date)

    if not ic_result or not ic_result[0]['total_assets']:
        return None

    total_assets = float(ic_result[0]['total_assets'])
    current_liabilities = float(ic_result[0]['current_liabilities']) if ic_result[0]['current_liabilities'] else 0

    invested_capital = total_assets - current_liabilities

    if invested_capital <= 0:
        return None

    # Step 3: ROIC 계산
    roic = (nopat / invested_capital) * 100

    # Step 4: Enterprise Value
    ev_query = """
    WITH latest_data AS (
        SELECT
            kit.market_cap,
            debt.thstrm_amount as total_debt,
            cash.thstrm_amount as cash
        FROM kr_intraday_total kit
        LEFT JOIN (
            SELECT symbol, thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND account_nm LIKE '%부채총계%'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ) debt ON kit.symbol = debt.symbol
        LEFT JOIN (
            SELECT symbol, thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND account_nm LIKE '%현금%'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ) cash ON kit.symbol = cash.symbol
        WHERE kit.symbol = $1
            AND ($2::date IS NULL OR kit.date = $2)
        ORDER BY kit.date DESC
        LIMIT 1
    )
    SELECT market_cap, total_debt, cash
    FROM latest_data
    """

    ev_result = await self.execute_query(ev_query, self.symbol, self.analysis_date)

    if not ev_result or not ev_result[0]['market_cap']:
        return None

    market_cap = float(ev_result[0]['market_cap'])
    total_debt = float(ev_result[0]['total_debt']) if ev_result[0]['total_debt'] else 0
    cash = float(ev_result[0]['cash']) if ev_result[0]['cash'] else 0

    enterprise_value = market_cap + total_debt - cash

    if enterprise_value <= 0:
        return None

    # Step 5: EV/IC 계산
    ev_ic = enterprise_value / invested_capital

    # Step 6: Magic Ratio
    if ev_ic > 0:
        magic_ratio = roic / ev_ic
    else:
        magic_ratio = 0

    # Step 7: 점수 계산
    score = 0

    # A. 높은 ROIC + 낮은 EV/IC = "진짜 저평가"
    if roic > 15 and ev_ic < 1.5:
        score = 90
        logger.info(f"V20: Excellent (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

    # B. 높은 ROIC + 높은 EV/IC = "좋지만 비쌈"
    elif roic > 15 and ev_ic > 2.5:
        score = 40
        logger.info(f"V20: Good but expensive (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

    # C. 낮은 ROIC + 낮은 EV/IC = "Value Trap"
    elif roic < 5 and ev_ic < 1.5:
        score = 20
        logger.info(f"V20: Value Trap (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

    # D. 중간 범위: Magic Ratio 기반
    else:
        # Magic Ratio 백분위 점수
        if magic_ratio > 10:
            score = 85
        elif magic_ratio > 5:
            score = 70
        elif magic_ratio > 2:
            score = 55
        elif magic_ratio > 1:
            score = 40
        else:
            score = 25

        logger.info(f"V20: Magic Ratio {magic_ratio:.2f} (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

    return score
```

---

### Task 1.5: calculate_all_strategies() 함수 업데이트

**위치**: `kr\kr_value_factor.py` (calculate_all_strategies 함수)

**수정 내용**:

```python
async def calculate_all_strategies(self):
    """
    Calculate all 20 value strategies (16 original + 4 new)

    Returns dict with strategy scores and weighted result
    """

    logger.info(f"Calculating value strategies for {self.symbol}...")

    # Original strategies (V1-V16)
    strategies_tasks = {
        'V1_Low_PER': self.calculate_v1(),  # ✅ V17 폴백 포함
        'V2_Magic_Formula': self.calculate_v2(),  # DEPRECATED
        'V3_Low_PBR': self.calculate_v3(),
        'V4_High_Dividend': self.calculate_v4(),  # ✅ 재설계됨
        'V5_Low_PSR': None,  # DEPRECATED
        'V6_High_ROE_Low_PER': None,  # V20로 대체
        'V7_Low_PCR': self.calculate_v7(),
        'V8_FCF_Yield': self.calculate_v8(),
        'V9_Low_EV_Sales': None,  # V18로 대체
        'V10_Low_Debt_Equity': self.calculate_v10(),
        'V11_High_Current_Ratio': self.calculate_v11(),
        'V12_ROE_PBR_Combined': self.calculate_v12(),
        'V13_Low_EV_EBITDA': self.calculate_v13(),
        'V14_Graham_Number': self.calculate_v14(),
        'V15_PEG_Ratio': self.calculate_v15(),
        'V16_Buffett_Indicator': self.calculate_v16(),
    }

    # New strategies (V17-V20)
    strategies_tasks.update({
        'V17_Growth_Adjusted': self.calculate_v17_growth_adjusted_value(),  # ✅ 신규
        'V18_EV_Sales_Growth': self.calculate_v18_ev_sales_growth(),  # ✅ 신규 (V9 대체)
        'V19_Value_Momentum': None,  # TODO: Phase 2
        'V20_ROIC_Value': self.calculate_v20_roic_value(),  # ✅ 신규 (V6 대체)
    })

    # Execute all strategies concurrently
    results = {}
    for name, task in strategies_tasks.items():
        if task is not None:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                results[name] = None
        else:
            results[name] = None

    self.strategies_scores = results

    # Get market state weights
    weights = self._get_strategy_weights()

    # Calculate weighted average
    weighted_result = self._calculate_weighted_average(results, weights)

    return {
        'strategies': results,
        'weighted_result': weighted_result,
        'market_state': self.market_state
    }
```

---

## Phase 2: 중기 개선 (2-4주)

### Task 2.1: V19 - Value Momentum 전략 추가

**목표**: 밸류에이션 추세 평가 (정적 밸류에이션 → 동적 밸류에이션)

**수정 파일**: `kr\kr_value_factor.py`

```python
# ========================================================================
# V19. Value Momentum (신규)
# ========================================================================

async def calculate_v19_value_momentum(self):
    """
    V19. Value Momentum Strategy

    Description: 밸류에이션 개선 추세 평가

    Logic:
    - PER/PBR이 하락하면서 (밸류 개선)
    - 주가는 상승하면 (시장이 인정)
    - = 진짜 밸류

    Scenarios:
    A. 밸류 개선 + 주가 상승 = 90점 (시장이 인정 시작)
    B. 밸류 개선 + 주가 정체 = 80점 (아직 발견 안됨, 기회!)
    C. 밸류 악화 + 주가 하락 = 10점 (Value Trap)
    D. 밸류 악화 + 주가 상승 = 30점 (버블 조짐)
    """

    # Step 1: 현재 PER
    per_current_query = """
    SELECT per
    FROM kr_intraday_detail
    WHERE symbol = $1
    LIMIT 1
    """

    per_current_result = await self.execute_query(per_current_query, self.symbol)

    if not per_current_result or not per_current_result[0]['per']:
        return None

    per_current = float(per_current_result[0]['per'])

    # Step 2: 60일 전 PER (역산)
    per_60d_query = """
    WITH price_60d AS (
        SELECT close as price_60d_ago
        FROM kr_intraday_total
        WHERE symbol = $1
            AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
        ORDER BY date DESC
        LIMIT 1
    ),
    latest_eps AS (
        SELECT eps
        FROM kr_intraday_detail
        WHERE symbol = $1
        LIMIT 1
    )
    SELECT
        p.price_60d_ago,
        e.eps,
        CASE
            WHEN e.eps > 0 THEN p.price_60d_ago / e.eps
            ELSE NULL
        END as per_60d_ago
    FROM price_60d p, latest_eps e
    """

    per_60d_result = await self.execute_query(per_60d_query, self.symbol, self.analysis_date)

    if not per_60d_result or not per_60d_result[0]['per_60d_ago']:
        return None

    per_60d_ago = float(per_60d_result[0]['per_60d_ago'])

    # Step 3: PER 개선율
    if per_60d_ago > 0 and per_current > 0:
        per_improvement = (per_60d_ago - per_current) / per_60d_ago
    else:
        return None

    # Step 4: 주가 수익률 (60일)
    price_return_query = """
    WITH prices AS (
        SELECT
            (SELECT close FROM kr_intraday_total
             WHERE symbol = $1 AND ($2::date IS NULL OR date = $2)
             ORDER BY date DESC LIMIT 1) as price_now,
            (SELECT close FROM kr_intraday_total
             WHERE symbol = $1 AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
             ORDER BY date DESC LIMIT 1) as price_60d_ago
    )
    SELECT
        price_now,
        price_60d_ago,
        (price_now - price_60d_ago) / NULLIF(price_60d_ago, 0) as price_return
    FROM prices
    """

    price_return_result = await self.execute_query(price_return_query, self.symbol, self.analysis_date)

    if not price_return_result or price_return_result[0]['price_return'] is None:
        return None

    price_return_60d = float(price_return_result[0]['price_return'])

    # Step 5: 시나리오 분석
    score = 50  # 기본값

    # A. 밸류 개선 + 주가 상승 = "시장이 인정하기 시작"
    if per_improvement > 0.10 and price_return_60d > 0.05:
        score = 90
        logger.info(f"V19: Market recognizing value (PER ↓{per_improvement*100:.1f}%, Price ↑{price_return_60d*100:.1f}%)")

    # B. 밸류 개선 + 주가 정체 = "아직 발견 안됨" (기회!)
    elif per_improvement > 0.10 and -0.05 < price_return_60d < 0.05:
        score = 80
        logger.info(f"V19: Undiscovered value opportunity (PER ↓{per_improvement*100:.1f}%, Price flat)")

    # C. 밸류 악화 + 주가 하락 = "실적 악화" (Value Trap)
    elif per_improvement < -0.10 and price_return_60d < -0.05:
        score = 10
        logger.info(f"V19: Value Trap (PER ↑{abs(per_improvement)*100:.1f}%, Price ↓{abs(price_return_60d)*100:.1f}%)")

    # D. 밸류 악화 + 주가 상승 = "버블 조짐"
    elif per_improvement < -0.10 and price_return_60d > 0.05:
        score = 30
        logger.info(f"V19: Potential bubble (PER ↑{abs(per_improvement)*100:.1f}%, Price ↑{price_return_60d*100:.1f}%)")

    # E. 소폭 변화
    else:
        score = 50
        logger.info(f"V19: Neutral momentum (PER {per_improvement*100:+.1f}%, Price {price_return_60d*100:+.1f}%)")

    return score
```

**Task 2.1 완료 후**: `calculate_all_strategies()` 함수에서 V19 활성화

```python
'V19_Value_Momentum': self.calculate_v19_value_momentum(),  # ✅ 활성화
```

---

### Task 2.2: Sector-Health-Weighted Peer-Relative 구현

**목표**: 망한 섹터 페널티 반영

**수정 파일**: `kr\kr_value_factor.py`

#### Step 1: Sector Health Check 함수 추가

```python
async def calculate_sector_health_score(self):
    """
    섹터 건강도 평가

    Returns:
        int: 0-100 (0=재앙, 100=최고)
    """

    # Step 1: 종목의 섹터 조회
    sector_query = """
    SELECT theme
    FROM kr_stock_detail
    WHERE symbol = $1
    """

    sector_result = await self.execute_query(sector_query, self.symbol)

    if not sector_result or not sector_result[0]['theme']:
        return 50  # 중립

    sector = sector_result[0]['theme']

    # Step 2: 섹터 30일 수익률
    sector_return_query = """
    WITH sector_stocks AS (
        SELECT symbol
        FROM kr_stock_detail
        WHERE theme = $1
    ),
    sector_returns AS (
        SELECT
            AVG(
                (current_price.close - past_price.close) / NULLIF(past_price.close, 0) * 100
            ) as avg_return_30d
        FROM sector_stocks ss
        JOIN kr_intraday_total current_price ON ss.symbol = current_price.symbol
        JOIN kr_intraday_total past_price
            ON ss.symbol = past_price.symbol
            AND past_price.date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
        WHERE ($2::date IS NULL OR current_price.date = $2)
            AND current_price.close IS NOT NULL
            AND past_price.close IS NOT NULL
    )
    SELECT avg_return_30d
    FROM sector_returns
    """

    sector_return_result = await self.execute_query(sector_return_query, sector, self.analysis_date)

    if not sector_return_result or sector_return_result[0]['avg_return_30d'] is None:
        return 50

    sector_return_30d = float(sector_return_result[0]['avg_return_30d'])

    # Step 3: 시장 전체 30일 수익률
    market_return_query = """
    WITH market_returns AS (
        SELECT
            AVG(
                (current_price.close - past_price.close) / NULLIF(past_price.close, 0) * 100
            ) as avg_return_30d
        FROM kr_intraday_total current_price
        JOIN kr_intraday_total past_price
            ON current_price.symbol = past_price.symbol
            AND past_price.date <= COALESCE($1::date, CURRENT_DATE) - INTERVAL '30 days'
        WHERE ($1::date IS NULL OR current_price.date = $1)
            AND current_price.close IS NOT NULL
            AND past_price.close IS NOT NULL
            AND current_price.market_cap > 100000000000  -- 1천억 이상만
    )
    SELECT avg_return_30d
    FROM market_returns
    """

    market_return_result = await self.execute_query(market_return_query, self.analysis_date)

    market_return_30d = 0
    if market_return_result and market_return_result[0]['avg_return_30d']:
        market_return_30d = float(market_return_result[0]['avg_return_30d'])

    # Step 4: 섹터 알파
    sector_alpha = sector_return_30d - market_return_30d

    # Step 5: 건강도 점수
    health_score = 50  # 기본값

    if sector_return_30d < -10:
        health_score = 20  # 섹터 폭락
        logger.info(f"Sector Health: {health_score} - Sector crash ({sector_return_30d:.1f}%)")
    elif sector_alpha < -5:
        health_score = 40  # 섹터 부진
        logger.info(f"Sector Health: {health_score} - Sector underperforming (α={sector_alpha:.1f}%)")
    elif sector_return_30d > 5 and sector_alpha > 0:
        health_score = 80  # 섹터 강세
        logger.info(f"Sector Health: {health_score} - Sector outperforming (α=+{sector_alpha:.1f}%)")
    else:
        health_score = 50  # 중립
        logger.info(f"Sector Health: {health_score} - Sector neutral ({sector_return_30d:.1f}%)")

    return health_score
```

#### Step 2: V1, V3, V7, V13 함수 수정 (Peer-Relative 적용)

**V1 수정 예시**:

```python
async def calculate_v1(self):
    """
    V1. Low PER Strategy (Sector-Health-Weighted Peer-Relative)
    """

    # ... (기존 PER 계산 로직) ...

    # 적자 기업 처리
    if per is None or per <= 0:
        logger.info(f"V1: PER invalid ({per}), falling back to V17")
        return await self.calculate_v17_growth_adjusted_value()

    # 흑자 기업: Sector-Health-Weighted 평가

    # Step 1: 섹터 건강도 체크
    sector_health = await self.calculate_sector_health_score()

    # Step 2: 섹터 중앙값 PER
    sector_per_query = """
    WITH sector_stocks AS (
        SELECT sd.symbol, sd.theme
        FROM kr_stock_detail sd
        WHERE sd.symbol = $1
    ),
    peer_pers AS (
        SELECT per
        FROM kr_intraday_detail kid
        JOIN kr_stock_detail sd ON kid.symbol = sd.symbol
        WHERE sd.theme = (SELECT theme FROM sector_stocks)
            AND kid.per > 0
            AND kid.per < 100
    )
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY per) as median_per
    FROM peer_pers
    """

    sector_per_result = await self.execute_query(sector_per_query, self.symbol)

    # Step 3: 전체 시장 중앙값 PER
    market_per_query = """
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY per) as median_per
    FROM kr_intraday_detail
    WHERE per > 0 AND per < 100
    """

    market_per_result = await self.execute_query(market_per_query)

    sector_median_per = 15  # 기본값
    market_median_per = 15  # 기본값

    if sector_per_result and sector_per_result[0]['median_per']:
        sector_median_per = float(sector_per_result[0]['median_per'])

    if market_per_result and market_per_result[0]['median_per']:
        market_median_per = float(market_per_result[0]['median_per'])

    # Step 4: Discount 계산
    peer_discount = (per - sector_median_per) / sector_median_per
    market_discount = (per - market_median_per) / market_median_per

    # Step 5: 섹터 건강도 기반 점수
    score = 50

    if sector_health >= 70:
        # 건강한 섹터: Peer-Relative 신뢰
        if peer_discount < -0.30:
            score = 90
        elif peer_discount < -0.15:
            score = 75
        elif peer_discount < 0:
            score = 60
        else:
            score = 40
        logger.info(f"V1: Healthy sector peer-relative (PER discount: {peer_discount*100:.1f}%)")

    elif sector_health >= 40:
        # 중립 섹터: Peer + Absolute 혼합
        peer_score = 60 if peer_discount < -0.30 else 50 if peer_discount < 0 else 35
        absolute_score = 100 - min(100, max(0, (per / 15) * 50))
        score = peer_score * 0.7 + absolute_score * 0.3
        logger.info(f"V1: Neutral sector mixed (Peer: {peer_score}, Absolute: {absolute_score:.0f})")

    else:
        # 망한 섹터: 절대 평가 + 페널티
        absolute_score = 100 - min(100, max(0, (per / 15) * 50))

        if market_discount < -0.50 and sector_health >= 20:
            # 망한 섹터지만 절대적으로 초저평가
            score = absolute_score * 0.8  # 페널티 20%
            logger.info(f"V1: Weak sector but ultra-cheap (Market discount: {market_discount*100:.1f}%)")
        else:
            score = absolute_score * 0.6  # 페널티 40%
            logger.info(f"V1: Weak sector penalty applied")

    return score
```

**동일한 로직을 V3, V7, V13에도 적용**

---

## Phase 3: 가중치 업데이트

### Task 3.1: VALUE_STRATEGY_WEIGHTS 딕셔너리 업데이트

**수정 파일**:
- `kr\kr_value_factor.py` (라인 40-145)
- `kr\weight.py`
- `kr\batch_weight.py`

**수정 내용**:

```python
VALUE_STRATEGY_WEIGHTS = {
    # Large Cap Group (6)
    'KOSPI대형-확장과열-공격형': {
        'V1': 0.3,
        'V2': 0.0,   # ✅ DEPRECATED
        'V3': 1.2,
        'V4': 0.8,   # ✅ 재설계됨
        'V5': 0.0,   # ✅ DEPRECATED
        'V6': 0.0,   # ✅ V20로 대체
        'V7': 0.7,
        'V8': 1.0,
        'V9': 0.0,   # ✅ V18로 대체
        'V10': 0.8,
        'V11': 0.5,
        'V12': 0.6,
        'V13': 0.8,
        'V14': 0.9,
        'V15': 0.4,
        'V16': 0.1,
        # 신규 전략
        'V17': 0.5,  # ✅ 적자 성장주
        'V18': 0.3,  # ✅ EV/Sales/Growth (V9 대체)
        'V19': 0.7,  # ✅ Value Momentum
        'V20': 0.2,  # ✅ ROIC Value (V6 대체)
    },

    # ... (19개 시장 상태 모두 동일하게 업데이트)

    # Special: 성장주 중심 시장 상태
    'KOSDAQ소형-성장테마-고위험형': {
        'V1': 0.2,
        'V2': 0.0,
        'V3': 0.5,
        'V4': 0.1,
        'V5': 0.0,
        'V6': 0.0,
        'V7': 0.4,
        'V8': 0.5,
        'V9': 0.0,
        'V10': 0.5,
        'V11': 0.3,
        'V12': 0.3,
        'V13': 0.6,
        'V14': 0.2,
        'V15': 0.8,
        'V16': 1.5,
        # 신규: 성장주에서는 V17, V18이 중요
        'V17': 1.8,  # ✅ 적자 성장주 최대 가중치
        'V18': 1.2,  # ✅ 성장률 조정 밸류
        'V19': 1.0,  # ✅ 밸류 모멘텀
        'V20': 0.5,  # ✅ ROIC (성장주에선 덜 중요)
    },
}
```

**주의**: 19개 시장 상태 모두 업데이트해야 함!

---

### Task 3.2: _get_strategy_weights() 함수 업데이트

**위치**: `kr\kr_value_factor.py`

```python
def _get_strategy_weights(self):
    """
    Get strategy weights based on market state

    Returns dict with 20 strategy weights (V1-V16 + V17-V20)
    """

    if self.market_state and self.market_state in VALUE_STRATEGY_WEIGHTS:
        weights = VALUE_STRATEGY_WEIGHTS[self.market_state]
    else:
        # 기본 가중치
        weights = VALUE_STRATEGY_WEIGHTS['기타']

    # 20개 전략 모두 포함 확인
    expected_strategies = [f'V{i}' for i in range(1, 21)]

    for strategy in expected_strategies:
        if strategy not in weights:
            weights[strategy] = 0.5  # 기본값
            logger.warning(f"Strategy {strategy} not in weights, using default 0.5")

    return weights
```

---

## Phase 4: 테스트 및 검증

### Task 4.1: 단위 테스트 작성

**파일 생성**: `phase3_7_value_test.py`

```python
"""
Phase 3.7 Value Strategy Test Suite
개선된 밸류 전략 검증
"""

import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import necessary modules
try:
    from database.async_db_manager import AsyncDatabaseManager
    from kr.kr_value_factor import ValueFactorCalculator
except ImportError:
    from async_db_manager import AsyncDatabaseManager
    from kr_value_factor import ValueFactorCalculator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValueStrategyTester:
    """밸류 전략 개선 테스트"""

    def __init__(self):
        self.db_manager = None
        self.test_results = []

    async def setup(self):
        """테스트 환경 설정"""
        self.db_manager = AsyncDatabaseManager()
        await self.db_manager.connect()

    async def teardown(self):
        """테스트 환경 정리"""
        if self.db_manager:
            await self.db_manager.close()

    async def test_v17_loss_making_stocks(self):
        """
        Test 1: 적자 성장주 테스트

        검증 대상:
        - 위니아에이드 (적자, 141% 수익)
        - 원익홀딩스 (적자, 115% 수익)
        - 뉴로핏 (적자, 52% 수익)

        기대 결과:
        - V1 (PER) → V17로 폴백
        - V17 점수 > 0 (기존: 0점)
        - 높은 성장률 인식
        """

        logger.info("\n" + "="*80)
        logger.info("TEST 1: V17 적자 성장주 테스트")
        logger.info("="*80)

        test_stocks = [
            ('377460', '위니아에이드', 141.28),
            ('030530', '원익홀딩스', 115.09),
            ('380550', '뉴로핏', 51.89),
        ]

        for symbol, name, expected_return in test_stocks:
            logger.info(f"\n테스트: {name} ({symbol})")

            calc = ValueFactorCalculator(
                symbol=symbol,
                db_manager=self.db_manager,
                analysis_date=datetime(2025, 8, 1)
            )

            # V1 (PER) 계산 - V17로 폴백되어야 함
            v1_score = await calc.calculate_v1()

            # V17 직접 계산
            v17_score = await calc.calculate_v17_growth_adjusted_value()

            logger.info(f"  V1 Score: {v1_score} (V17로 폴백)")
            logger.info(f"  V17 Score: {v17_score}")
            logger.info(f"  실제 수익률: +{expected_return:.1f}%")

            # 검증
            assert v1_score is not None, f"{name}: V1이 None (적자 처리 실패)"
            assert v1_score > 0, f"{name}: V1 점수가 0 (V17 폴백 실패)"
            assert v17_score is not None, f"{name}: V17이 None"
            assert v17_score >= 50, f"{name}: V17 점수 너무 낮음 ({v17_score})"

            self.test_results.append({
                'test': 'V17_Loss_Making',
                'stock': name,
                'v1_score': v1_score,
                'v17_score': v17_score,
                'return': expected_return,
                'pass': v1_score > 0 and v17_score >= 50
            })

        logger.info("\n✅ TEST 1 완료")

    async def test_v4_dividend_trap(self):
        """
        Test 2: 배당 함정 테스트

        검증 대상:
        - 유아이엘 (100점 받았지만 -12% 손실)
        - 서호전기 (100점 받았지만 -13% 손실)

        기대 결과:
        - 기존 V4: 100점
        - 신규 V4: <30점 (지속 불가능 탐지)
        """

        logger.info("\n" + "="*80)
        logger.info("TEST 2: V4 배당 함정 테스트")
        logger.info("="*80)

        test_stocks = [
            ('049520', '유아이엘', -12.09),
            ('065710', '서호전기', -12.85),
        ]

        for symbol, name, actual_return in test_stocks:
            logger.info(f"\n테스트: {name} ({symbol})")

            calc = ValueFactorCalculator(
                symbol=symbol,
                db_manager=self.db_manager,
                analysis_date=datetime(2025, 8, 1)
            )

            # 신규 V4 (지속 가능 배당)
            v4_new_score = await calc.calculate_v4_sustainable_dividend()

            # 기존 V4 (참고용)
            # v4_old_score = await calc.calculate_v4_original()

            logger.info(f"  V4 New Score: {v4_new_score}")
            logger.info(f"  실제 수익률: {actual_return:.1f}%")

            # 검증: 배당 함정을 낮은 점수로 탐지해야 함
            assert v4_new_score is not None, f"{name}: V4가 None"
            assert v4_new_score < 30, f"{name}: 배당 함정 미탐지 (점수: {v4_new_score})"

            self.test_results.append({
                'test': 'V4_Dividend_Trap',
                'stock': name,
                'v4_score': v4_new_score,
                'return': actual_return,
                'pass': v4_new_score < 30
            })

        logger.info("\n✅ TEST 2 완료")

    async def test_v18_growth_valuation(self):
        """
        Test 3: V18 성장률 조정 밸류에이션

        검증:
        - 고성장 기업의 높은 EV/Sales를 성장률로 조정
        - V9 대비 고성장주 점수 상승 확인
        """

        logger.info("\n" + "="*80)
        logger.info("TEST 3: V18 성장률 조정 밸류에이션")
        logger.info("="*80)

        # 고성장 종목 (적자 성장주 포함)
        test_stocks = [
            ('380550', '뉴로핏'),
            ('377480', '마음AI'),
        ]

        for symbol, name in test_stocks:
            logger.info(f"\n테스트: {name} ({symbol})")

            calc = ValueFactorCalculator(
                symbol=symbol,
                db_manager=self.db_manager,
                analysis_date=datetime(2025, 8, 1)
            )

            v18_score = await calc.calculate_v18_ev_sales_growth()

            logger.info(f"  V18 Score: {v18_score}")

            assert v18_score is not None, f"{name}: V18이 None"

            self.test_results.append({
                'test': 'V18_Growth_Valuation',
                'stock': name,
                'v18_score': v18_score,
                'pass': v18_score is not None
            })

        logger.info("\n✅ TEST 3 완료")

    async def test_v20_roic_quality(self):
        """
        Test 4: V20 ROIC 기반 Quality+Value

        검증:
        - 높은 ROIC + 낮은 EV/IC = 높은 점수
        - 낮은 ROIC + 낮은 EV/IC = 낮은 점수 (Value Trap)
        """

        logger.info("\n" + "="*80)
        logger.info("TEST 4: V20 ROIC 기반 Quality+Value")
        logger.info("="*80)

        # 대형 우량주 테스트
        test_stocks = [
            ('005930', '삼성전자'),
            ('000660', 'SK하이닉스'),
        ]

        for symbol, name in test_stocks:
            logger.info(f"\n테스트: {name} ({symbol})")

            calc = ValueFactorCalculator(
                symbol=symbol,
                db_manager=self.db_manager,
                analysis_date=datetime(2025, 8, 1)
            )

            v20_score = await calc.calculate_v20_roic_value()

            logger.info(f"  V20 Score: {v20_score}")

            self.test_results.append({
                'test': 'V20_ROIC_Quality',
                'stock': name,
                'v20_score': v20_score,
                'pass': v20_score is not None
            })

        logger.info("\n✅ TEST 4 완료")

    async def test_sector_health_check(self):
        """
        Test 5: 섹터 건강도 체크

        검증:
        - Finance 섹터: 건강도 낮음
        - IT/Semiconductor 섹터: 건강도 높음
        """

        logger.info("\n" + "="*80)
        logger.info("TEST 5: 섹터 건강도 체크")
        logger.info("="*80)

        test_sectors = [
            ('005930', '삼성전자', 'Semiconductor', '높음'),
            ('055550', '신한지주', 'Finance', '낮음'),
        ]

        for symbol, name, expected_sector, expected_health in test_sectors:
            logger.info(f"\n테스트: {name} ({expected_sector})")

            calc = ValueFactorCalculator(
                symbol=symbol,
                db_manager=self.db_manager,
                analysis_date=datetime(2025, 9, 1)
            )

            health_score = await calc.calculate_sector_health_score()

            logger.info(f"  Sector Health: {health_score}/100 (예상: {expected_health})")

            if expected_health == '높음':
                assert health_score >= 60, f"{name}: 섹터 건강도가 예상보다 낮음"
            elif expected_health == '낮음':
                assert health_score < 60, f"{name}: 섹터 건강도가 예상보다 높음"

            self.test_results.append({
                'test': 'Sector_Health',
                'stock': name,
                'sector': expected_sector,
                'health_score': health_score,
                'pass': True
            })

        logger.info("\n✅ TEST 5 완료")

    async def run_all_tests(self):
        """모든 테스트 실행"""

        await self.setup()

        try:
            await self.test_v17_loss_making_stocks()
            await self.test_v4_dividend_trap()
            await self.test_v18_growth_valuation()
            await self.test_v20_roic_quality()
            await self.test_sector_health_check()

            # 결과 요약
            logger.info("\n" + "="*80)
            logger.info("테스트 결과 요약")
            logger.info("="*80)

            total = len(self.test_results)
            passed = sum(1 for r in self.test_results if r.get('pass', False))

            logger.info(f"\n총 테스트: {total}")
            logger.info(f"통과: {passed}")
            logger.info(f"실패: {total - passed}")
            logger.info(f"통과율: {passed/total*100:.1f}%")

            if passed == total:
                logger.info("\n🎉 모든 테스트 통과!")
            else:
                logger.warning("\n⚠️ 일부 테스트 실패")
                for result in self.test_results:
                    if not result.get('pass', False):
                        logger.warning(f"  실패: {result['test']} - {result['stock']}")

        finally:
            await self.teardown()


async def main():
    """메인 테스트 실행"""
    tester = ValueStrategyTester()
    await tester.run_all_tests()


if __name__ == '__main__':
    asyncio.run(main())
```

---

### Task 4.2: IC 재계산 및 비교

**파일 생성**: `phase3_7_ic_comparison.py`

```python
"""
Phase 3.7 IC Comparison
개선 전후 IC 비교
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr

# ... (IC 계산 로직)

async def compare_ic_before_after():
    """
    개선 전후 IC 비교

    Before:
    - Value IC: -0.031
    - V1 (PER): -0.132
    - V4 (Dividend): -0.057
    - V9 (EV/Sales): +0.055

    After:
    - Value IC: 목표 +0.02~0.05
    - V1 (PER with V17 fallback): 목표 중립 이상
    - V4 (Sustainable Dividend): 목표 중립 이상
    - V18 (EV/Sales/Growth): 목표 +0.08 이상
    """

    # ... (구현)
```

---

## 파일별 수정 체크리스트

### ✅ kr\kr_value_factor.py

- [ ] V17 함수 추가 (적자 성장주)
- [ ] V1 함수 수정 (V17 폴백)
- [ ] V4 함수 재설계 (지속 가능 배당)
- [ ] V18 함수 추가 (EV/Sales/Growth)
- [ ] V20 함수 추가 (ROIC-Based Value)
- [ ] V19 함수 추가 (Value Momentum) - Phase 2
- [ ] calculate_sector_health_score() 추가
- [ ] V1, V3, V7, V13 Peer-Relative 수정 - Phase 2
- [ ] VALUE_STRATEGY_WEIGHTS 업데이트 (20개 전략)
- [ ] calculate_all_strategies() 업데이트
- [ ] _get_strategy_weights() 업데이트

### ✅ kr\weight.py

- [ ] VALUE_STRATEGY_WEIGHTS 동기화
- [ ] 신규 전략 가중치 추가 (V17-V20)

### ✅ kr\batch_weight.py

- [ ] VALUE_STRATEGY_WEIGHTS 동기화
- [ ] 신규 전략 가중치 추가 (V17-V20)

### ✅ 테스트 파일

- [ ] phase3_7_value_test.py 생성
- [ ] phase3_7_ic_comparison.py 생성

---

## 검증 기준

### 기능 검증

1. **V17 (적자 성장주)**
   - 위니아에이드, 원익홀딩스, 뉴로핏: 점수 > 0
   - 고성장 인식 (CAGR >50%)
   - PSR 기반 평가 작동

2. **V4 (지속 가능 배당)**
   - 유아이엘, 서호전기: 점수 < 30 (함정 탐지)
   - Payout Ratio >80% 탐지
   - FCF Coverage <1.0 탐지

3. **V18 (EV/Sales/Growth)**
   - 고성장주 점수 상승
   - 저성장주 페널티
   - 섹터 내 상대 평가

4. **V20 (ROIC-Based)**
   - 높은 ROIC + 낮은 EV/IC = 90점
   - 낮은 ROIC + 낮은 EV/IC = 20점

5. **Sector Health**
   - Finance 섹터: <50점
   - IT/Semi 섹터: >70점

### 성능 검증

1. **IC 개선**
   - Value IC: -0.031 → +0.02 이상
   - V1 IC: -0.132 → 중립 이상
   - V4 IC: -0.057 → 중립 이상
   - V18 IC: +0.08 이상 (V9 +0.055 개선)

2. **논리적 정합성**
   - 적자 성장주 0점 → 해결
   - 배당 함정 100점 → 해결
   - 망한 섹터 페널티 → 해결

---

## 작업 일정

### Week 1
- Task 1.1: V17 추가
- Task 1.2: V4 재설계
- Task 1.3: V18 추가
- Task 1.4: V20 추가

### Week 2
- Task 1.5: calculate_all_strategies 업데이트
- Task 3.1: VALUE_STRATEGY_WEIGHTS 업데이트
- Task 4.1: 테스트 작성 및 실행

### Week 3
- Task 2.1: V19 추가
- Task 2.2: Sector Health + Peer-Relative 구현

### Week 4
- Task 4.2: IC 재계산 및 검증
- 최종 리뷰 및 문서화

---

## 비상 계획

### 만약 IC가 개선되지 않는다면?

1. **로직 검증**
   - 계산 오류 확인
   - SQL 쿼리 검증
   - 데이터 품질 체크

2. **파라미터 조정**
   - 임계값 재조정
   - 가중치 미세 조정

3. **원칙 고수**
   - IC 낮아도 논리적으로 타당하면 유지
   - 과적합 방지 우선

---

**작업계획서 버전**: 1.0
**작성일**: 2025-11-22
**검토 주기**: 주간
**최종 목표**: 2025-12-20 완료
