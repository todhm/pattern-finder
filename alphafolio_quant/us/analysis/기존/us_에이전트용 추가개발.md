# US 퀀트 시스템 - 멀티 에이전트 AI용 추가 개발 계획

## 1. 배경 및 목적

### 사용자의 핵심 질문
| 질문 | 현재 해결 | 개선 필요 |
|------|----------|----------|
| 어떤 종목을 사야 하는가? | ✅ final_score, final_grade | - |
| 지금이 좋은 타이밍인가? | ⚠️ rs_value만 있음 | ✅ |
| 얼마나 사야 하는가? | ❌ 없음 | ✅ |
| 언제 팔아야 하는가? | ❌ 없음 | ✅ |
| 시나리오별 대응은? | ❌ 없음 | ✅ |

### 목표
- 멀티 에이전트 AI에게 **강세/횡보/약세 시나리오별 투자 전략** 제공
- **미국 시장 특화 데이터** (옵션, 내부자 거래, EPS 추정치) 활용
- 사용자가 **구체적으로 행동**할 수 있는 정보 제공

---

## 2. 한국 vs 미국 데이터 차이점

| 항목 | 한국 | 미국 | 활용 방안 |
|------|------|------|----------|
| **옵션 데이터** | ❌ 없음 | ✅ us_option_daily_summary | 원본 테이블 조회 (iv_percentile만 저장) |
| **내부자 거래** | ❌ 없음 | ✅ us_insider_transactions | 원본 테이블 조회 (insider_signal만 저장) |
| **기관/외국인 수급** | ✅ inst_net, foreign_net | ❌ 없음 (13F 분기만) | 미국은 옵션 데이터로 대체 |
| **애널리스트 추정치** | △ 제한적 | ✅ us_earnings_estimates | 원본 테이블 조회 (중복 저장 안함) |
| **뉴스 감성** | ❌ 없음 | ✅ us_news (sentiment) | 원본 테이블 조회 (중복 저장 안함) |
| **ATR** | 직접 계산 | ✅ us_atr 테이블 | 원본 테이블 조회 (중복 저장 안함) |

---

## 3. 기존 컬럼 제거/유지

### 제거 대상 (한국과 동일하게 정리)
```sql
-- 가짜 정밀도 컬럼 제거
ALTER TABLE us_stock_grade
DROP COLUMN IF EXISTS expected_range_3m_min,
DROP COLUMN IF EXISTS expected_range_3m_max,
DROP COLUMN IF EXISTS expected_range_1y_min,
DROP COLUMN IF EXISTS expected_range_1y_max,
DROP COLUMN IF EXISTS support_1,
DROP COLUMN IF EXISTS support_2,
DROP COLUMN IF EXISTS resistance_1,
DROP COLUMN IF EXISTS resistance_2,
DROP COLUMN IF EXISTS supertrend_value,
DROP COLUMN IF EXISTS trend,
DROP COLUMN IF EXISTS signal,
DROP COLUMN IF EXISTS score_change_1m,
DROP COLUMN IF EXISTS score_change_2m;
```

---

## 4. 신규 컬럼 추가

### 4.1 공통 컬럼 (한국과 동일 - 18개)

```sql
-- 진입 타이밍 (3개)
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS entry_timing_score DECIMAL(5,1);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS score_trend_2w DECIMAL(5,2);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS price_position_52w DECIMAL(5,2);

-- 리스크 관리 (5개)
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS atr_pct DECIMAL(5,2);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS stop_loss_pct DECIMAL(5,2);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS take_profit_pct DECIMAL(5,2);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS risk_reward_ratio DECIMAL(4,2);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS position_size_pct DECIMAL(4,2);

-- 시나리오 분석 (7개)
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS scenario_bullish_prob INTEGER;
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS scenario_sideways_prob INTEGER;
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS scenario_bearish_prob INTEGER;
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS scenario_bullish_return VARCHAR(20);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS scenario_sideways_return VARCHAR(20);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS scenario_bearish_return VARCHAR(20);
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS scenario_sample_count INTEGER;

-- 행동 트리거 (3개)
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS buy_triggers JSONB;
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS sell_triggers JSONB;
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS hold_triggers JSONB;
```

### 4.2 미국 시장 특화 컬럼 (2개)

> **참고**: 나머지 미국 특화 데이터는 이미 별도 테이블에 존재하므로 중복 저장하지 않음
> - `put_call_ratio`, `iv_skew` → `us_option_daily_summary`에서 조회
> - `insider_net_shares` → `us_insider_transactions`에서 조회
> - `eps_revision_pct` → `us_earnings_estimates`에서 조회
> - `analyst_consensus`, `target_upside` → `us_stock_basic`에서 조회
> - `news_sentiment` → `us_news`에서 조회

```sql
-- 계산/집계가 필요한 컬럼만 저장 (2개)
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS iv_percentile INTEGER;      -- 1년치 IV 백분위 계산 필요
ALTER TABLE us_stock_grade ADD COLUMN IF NOT EXISTS insider_signal VARCHAR(20); -- 30일치 내부자 거래 집계+판단 필요
```

---

## 5. 계산 로직 상세

### 5.1 ATR 기반 손절/익절 (us_atr 테이블 활용)

```python
async def calculate_atr_stop_take_profit(self, symbol: str, analysis_date: date) -> dict:
    """
    ATR 기반 손절/익절 계산 - us_atr 테이블 직접 활용

    미국은 us_atr 테이블에 이미 ATR이 계산되어 있음 (14일 기준)
    """
    query = """
    SELECT a.atr, d.close
    FROM us_atr a
    JOIN us_daily d ON a.symbol = d.symbol AND a.date = d.date
    WHERE a.symbol = $1 AND a.date <= $2
    ORDER BY a.date DESC
    LIMIT 1
    """
    result = await self.db.execute_query(query, symbol, analysis_date)

    if not result:
        return None

    atr = float(result[0]['atr'])
    close = float(result[0]['close'])
    atr_pct = (atr / close) * 100

    # 손절: 2 ATR, 익절: 3 ATR
    stop_loss_pct = -2 * atr_pct
    take_profit_pct = 3 * atr_pct
    risk_reward = abs(take_profit_pct / stop_loss_pct)

    return {
        'atr_pct': round(atr_pct, 2),
        'stop_loss_pct': round(stop_loss_pct, 2),
        'take_profit_pct': round(take_profit_pct, 2),
        'risk_reward_ratio': round(risk_reward, 2)
    }
```

### 5.2 IV Percentile 계산 (us_stock_grade 저장용)

```python
async def calculate_iv_percentile(self, symbol: str, analysis_date: date) -> Optional[int]:
    """
    IV Percentile 계산 - 현재 IV가 1년 중 몇 %에 해당하는지

    이 값은 1년치 데이터 기준 백분위 계산이 필요하므로 us_stock_grade에 저장
    (put_call_ratio, iv_skew는 us_option_daily_summary에서 직접 조회)
    """
    query = """
    SELECT avg_implied_volatility
    FROM us_option_daily_summary
    WHERE symbol = $1
      AND date >= $2 - INTERVAL '252 days'
      AND date <= $2
      AND avg_implied_volatility IS NOT NULL
    ORDER BY date DESC
    """
    result = await self.db.execute_query(query, symbol, analysis_date)

    if not result or len(result) < 20:  # 최소 20일 데이터 필요
        return None

    current_iv = float(result[0]['avg_implied_volatility'])
    iv_list = [float(x['avg_implied_volatility']) for x in result]

    iv_percentile = sum(1 for iv in iv_list if iv <= current_iv) / len(iv_list) * 100

    return round(iv_percentile)
```

### 5.3 내부자 거래 신호 (us_stock_grade 저장용)

```python
async def calculate_insider_signal(self, symbol: str, analysis_date: date) -> Optional[str]:
    """
    내부자 거래 신호 계산 (30일 기준)

    30일치 데이터 집계 + 판단이 필요하므로 us_stock_grade에 저장
    (상세 거래 내역은 us_insider_transactions에서 직접 조회)

    - acquisition_or_disposal: 'A' = 매수, 'D' = 매도
    - executive_title: CEO, CFO 등 고위직 매매 가중치 높음
    """
    query = """
    SELECT
        acquisition_or_disposal,
        shares,
        executive_title
    FROM us_insider_transactions
    WHERE symbol = $1
      AND date >= $2 - INTERVAL '30 days'
      AND date <= $2
    """
    result = await self.db.execute_query(query, symbol, analysis_date)

    if not result:
        return 'NEUTRAL'

    buy_shares = 0
    sell_shares = 0

    for r in result:
        shares = float(r['shares']) if r['shares'] else 0
        titles = r['executive_title'] or []

        # CEO, CFO 거래는 2배 가중치
        weight = 2 if any('CEO' in t or 'CFO' in t for t in titles) else 1

        if r['acquisition_or_disposal'] == 'A':
            buy_shares += shares * weight
        else:
            sell_shares += shares * weight

    net_shares = buy_shares - sell_shares

    if net_shares > 10000:
        return 'STRONG_BUY'
    elif net_shares > 0:
        return 'BUY'
    elif net_shares < -10000:
        return 'STRONG_SELL'
    elif net_shares < 0:
        return 'SELL'
    else:
        return 'NEUTRAL'
```

### 5.4 별도 테이블 조회 쿼리 (참고용)

> 아래 데이터들은 us_stock_grade에 저장하지 않고 필요시 원본 테이블에서 직접 조회

```sql
-- Put/Call 비율, IV Skew (us_option_daily_summary)
SELECT
    total_put_volume::float / NULLIF(total_call_volume, 0) as put_call_ratio,
    avg_put_iv - avg_call_iv as iv_skew
FROM us_option_daily_summary
WHERE symbol = $1 AND date = $2;

-- EPS Revision (us_earnings_estimates)
SELECT
    eps_estimate_average,
    eps_estimate_average_30_days_ago,
    CASE WHEN eps_estimate_average_30_days_ago != 0
         THEN (eps_estimate_average - eps_estimate_average_30_days_ago)
              / ABS(eps_estimate_average_30_days_ago) * 100
         ELSE NULL
    END as eps_revision_pct
FROM us_earnings_estimates
WHERE symbol = $1 AND estimate_date >= $2
ORDER BY estimate_date LIMIT 1;

-- 애널리스트 컨센서스 (us_stock_basic)
SELECT
    analystratingstrongbuy, analystratingbuy, analystratinghold,
    analystratingsell, analystratingstrongsell, analysttargetprice
FROM us_stock_basic
WHERE symbol = $1;

-- 뉴스 감성 7일 평균 (us_news)
SELECT AVG(ticker_sentiment_score) as news_sentiment
FROM us_news
WHERE ticker = $1
  AND time_published >= $2 - INTERVAL '7 days'
  AND ticker_sentiment_score IS NOT NULL;

-- 내부자 거래 상세 (us_insider_transactions)
SELECT date, executive, executive_title, acquisition_or_disposal, shares, share_price
FROM us_insider_transactions
WHERE symbol = $1 AND date >= $2 - INTERVAL '30 days'
ORDER BY date DESC;
```

### 5.5 미국 특화 행동 트리거 생성

```python
def generate_us_triggers(self, final_score: float, sector_percentile: float,
                          stop_loss_pct: float, take_profit_pct: float,
                          insider_signal: str) -> dict:
    """
    미국 시장 특화 행동 트리거 생성

    참고: put_call_ratio, eps_revision_pct 등은 별도 테이블에서 조회하여 사용
    """
    buy_triggers = [
        f"final_score {final_score + 5:.0f}점 이상 상승 시",
        f"섹터 순위 상위 {max(5, sector_percentile - 10):.0f}% 진입 시"
    ]

    # 미국 특화 매수 조건
    buy_triggers.append("Put/Call 0.7 이하로 하락 시 (us_option_daily_summary 조회)")
    if insider_signal == 'NEUTRAL':
        buy_triggers.append("내부자 순매수 전환 시 (CEO/CFO 매수)")
    buy_triggers.append("EPS 추정치 3% 이상 상향 시 (us_earnings_estimates 조회)")

    sell_triggers = [
        f"final_score {final_score - 15:.0f}점 이하 하락 시",
        f"손절선 {stop_loss_pct:.1f}% 도달 시",
        f"익절선 +{take_profit_pct:.1f}% 도달 시"
    ]

    # 미국 특화 매도 조건
    sell_triggers.append("Put/Call 1.2 이상 급등 시 (us_option_daily_summary 조회)")
    if insider_signal in ['NEUTRAL', 'BUY', 'STRONG_BUY']:
        sell_triggers.append("내부자 순매도 전환 시 (CEO/CFO 매도)")

    hold_triggers = [
        f"final_score {final_score - 5:.0f}~{final_score + 5:.0f}점 유지 시",
        "어닝 발표 1주일 이내 (변동성 대기)"
    ]

    return {
        'buy_triggers': buy_triggers,
        'sell_triggers': sell_triggers,
        'hold_triggers': hold_triggers
    }
```

---

## 6. 최종 데이터 구조 (멀티 에이전트 AI용)

```json
{
  "symbol": "AAPL",
  "stock_name": "Apple Inc.",
  "date": "2025-11-27",

  "core_judgment": {
    "final_score": 78.5,
    "final_grade": "A",
    "opinion": "매수",
    "value_score": 72.3,
    "quality_score": 85.2,
    "momentum_score": 76.5,
    "growth_score": 80.0
  },

  "entry_timing": {
    "entry_timing_score": 72,
    "score_trend_2w": 3.5,
    "price_position_52w": 78.5
  },

  "risk_management": {
    "atr_pct": 2.85,
    "stop_loss_pct": -5.7,
    "take_profit_pct": 8.55,
    "risk_reward_ratio": 1.5,
    "position_size_pct": 3.5,
    "var_95": 2.1,
    "cvar_95": 2.8
  },

  "us_market_specific_stored": {
    "_comment": "us_stock_grade에 저장 (계산 필요)",
    "iv_percentile": 35,
    "insider_signal": "BUY"
  },

  "us_market_specific_from_tables": {
    "_comment": "별도 테이블에서 조회 (중복 저장 안함)",
    "_source_us_option_daily_summary": {
      "put_call_ratio": 0.68,
      "iv_skew": 0.02
    },
    "_source_us_insider_transactions": {
      "insider_net_shares": 25000
    },
    "_source_us_earnings_estimates": {
      "eps_revision_pct": 2.5
    },
    "_source_us_stock_basic": {
      "analyst_consensus": "STRONG_BUY",
      "target_upside": 15.3
    },
    "_source_us_news": {
      "news_sentiment": 0.42
    }
  },

  "scenario_analysis": {
    "probabilities": {
      "bullish": 55,
      "sideways": 30,
      "bearish": 15
    },
    "expected_returns": {
      "bullish": "+12~20%",
      "sideways": "-3~+5%",
      "bearish": "-8~-15%"
    },
    "sample_count": 450,
    "confidence_note": "과거 유사 패턴 450건 기반"
  },

  "action_triggers": {
    "buy": [
      "final_score 84점 이상 상승",
      "Put/Call 0.5 이하로 하락 (us_option_daily_summary)",
      "내부자 추가 매수 발생"
    ],
    "sell": [
      "final_score 64점 이하 하락",
      "손절선 -5.7% 도달",
      "Put/Call 1.2 이상 급등 (us_option_daily_summary)"
    ],
    "hold": [
      "final_score 74~84점 유지",
      "어닝 발표 1주일 이내"
    ]
  },

  "relative_position": {
    "sector": "Technology",
    "rs_value": 25.3,
    "rs_rank": "A+ (Top 10%)"
  }
}
```

---

## 7. 구현 우선순위

| 순위 | 항목 | 파일 | 난이도 | 비고 |
|------|------|------|--------|------|
| 1 | ATR 기반 stop_loss/take_profit | us_agent_metrics.py (신규) | 낮음 | us_atr 활용 |
| 2 | entry_timing_score | us_agent_metrics.py | 중간 | - |
| 3 | position_size_pct | us_agent_metrics.py | 낮음 | - |
| 4 | iv_percentile | us_agent_metrics.py | 낮음 | 미국 특화 (저장) |
| 5 | insider_signal | us_agent_metrics.py | 중간 | 미국 특화 (저장) |
| 6 | triggers 자동 생성 | us_agent_metrics.py | 낮음 | 특화 조건 포함 |
| 7 | scenario_probability | us_agent_metrics.py | 높음 | - |
| 8 | us_main_v2.py 통합 | us_main_v2.py | 중간 | - |
| 9 | 기존 컬럼 제거 | SQL, us_main_v2.py | 낮음 | - |

> **참고**: put_call_ratio, iv_skew, eps_revision_pct, analyst_consensus, target_upside, news_sentiment는 별도 테이블에서 조회하므로 구현 불필요

---

## 8. 파일 변경 목록

| 파일 | 변경 내용 |
|------|----------|
| `us_agent_metrics.py` (신규) | 에이전트용 지표 계산 클래스 |
| `us_main_v2.py` | 에이전트 지표 계산 통합, grade_data 구조 변경 |
| `us_db_async.py` | INSERT 쿼리 컬럼 추가 |
| SQL | 테이블 컬럼 추가/삭제 |

---

## 9. 예상 효과

### Before (기존)
```
final_score: 78.5
final_grade: A
rs_value: 25.3
rs_rank: A+ (Top 10%)
```
→ 등급만 있고 **구체적 행동 지침 없음**

### After (개선)
```
entry_timing_score: 72/100 (양호)
stop_loss: -5.7% (ATR 기반)
take_profit: +8.55%
position_size: 3.5%

미국 특화 (us_stock_grade 저장):
- IV Percentile: 35 (낮은 변동성)
- Insider Signal: BUY

미국 특화 (별도 테이블 조회):
- Put/Call: 0.68 (us_option_daily_summary)
- EPS Revision: +2.5% (us_earnings_estimates)
- News Sentiment: 0.42 (us_news)
- Analyst Consensus: STRONG_BUY (us_stock_basic)

시나리오 (450건 분석 기반):
- 강세 55%: +12~20%
- 횡보 30%: -3~+5%
- 약세 15%: -8~-15%

매수 트리거: Put/Call 0.5 이하, 내부자 추가 매수
매도 트리거: 손절선 -5.7%, Put/Call 1.2 이상
```
→ **신뢰도 명시 + 미국 시장 특화 + 중복 저장 최소화 + 구체적 행동 지침 제공**

---

*작성일: 2025-11-28*
*버전: 1.0*
