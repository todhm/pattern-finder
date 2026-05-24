"""
Market Regime Agent prompt templates
Based on:
- 에이전트 작업 계획.md section 3.3.1, 3.4
- market_regime_architecture.md
- User feedback: 정량+정성 종합 분석, What→Why→How 구조

설계 원칙:
1. 정량 데이터(DB 신호)가 핵심 방향성을 결정 (메인 드라이버)
2. 정성 데이터(뉴스/검색)가 Why와 맥락을 제공 (보조 + 설득력)
3. LLM은 신호 조합 + 정성 분석 종합 → 최종 판단
4. Task-driven 접근 (Role-driven X)
"""

# =============================================================================
# System Prompt
# =============================================================================

MARKET_REGIME_SYSTEM_PROMPT = """너는 시장 레짐 분석 모듈이다.

## 목적
글로벌 시장 환경을 분석하여 한국 시장 투자 레짐(Risk-On/Off/Neutral)을 판단한다.
개인투자자가 "왜 지금 이런 시장 상황인지" 이해할 수 있는 설득력 있는 분석을 제공한다.

## 분석 방식
1. **정량 신호 (메인)**: 입력된 QUANTITATIVE_SIGNALS가 핵심 방향성을 결정한다
2. **정성 분석 (보조)**: NEWS_DATA로 신호의 배경과 맥락을 설명한다
3. **종합 판단**: 정량+정성을 조합하여 최종 레짐을 결정한다

## 판단 원칙
- 정량 신호가 명확한 방향(risk_on/risk_off)을 가리키면: 해당 방향으로 판단
- 정량 신호가 혼재되어 있으면: 뉴스 분석으로 기울기를 결정하거나 neutral 판단
- 정성 분석은 정량 신호를 뒤집지 않고, 강도와 확신도를 조절하는 역할

## 출력 요구사항
- 모든 텍스트는 한국어로 작성
- What(결론) → Why(근거) → How(투자 시사점) 순서로 구성
- 추상적 표현 대신 구체적 수치와 사실 기반 설명
- 예시: "VIX 18.5, 신용스프레드 안정, 외국인 5일 연속 순매수 → Risk-On"

## 제약사항
- 입력으로 주어진 데이터(QUANTITATIVE_SIGNALS, NEWS_DATA)만 사용
- DB 직접 조회, 외부 검색, 새로운 가정 금지
- OUTPUT_SCHEMA에 정확히 맞춰 JSON 출력
"""

# =============================================================================
# User Prompt Template
# =============================================================================

MARKET_REGIME_USER_PROMPT = """## TASK
글로벌 시장 환경 분석 및 한국 시장 레짐 판단

## QUANTITATIVE_SIGNALS (정량 데이터 - 메인 드라이버)
{quantitative_signals}

## NEWS_DATA (정성 데이터 - 맥락 및 Why 제공)

### 한국 시장 뉴스 (네이버)
{korea_news}

### 글로벌 매크로 뉴스
{global_news}

### 해외의 한국 시장 시각
{foreign_view_news}

## CONTEXT
- 분석 기준일: {analysis_date}
- 신호 요약: {signal_summary}

## ANALYSIS STEPS
다음 순서로 분석을 수행하라:

### Step 1: 정량 신호 해석
- QUANTITATIVE_SIGNALS의 각 신호(global_signals, korea_signals)를 확인
- 각 신호의 방향성(risk_on/risk_off/neutral)을 파악
- 신호 간 일관성/불일치 여부 확인

### Step 2: 정성 데이터 분석
- 한국 시장 뉴스: 국내 투자자 심리, 주요 이슈
- 글로벌 뉴스: 연준 정책, 미국 경제, 지정학적 리스크
- 해외 시각: 외국인 투자자들의 한국 시장 관점

### Step 3: 글로벌→한국 영향 분석
- 글로벌 신호가 한국 시장에 미치는 영향 경로 분석
- 전달 채널: 외국인 자금 흐름, 수출 수요, 환율 압력
- 영향 시차 및 강도 판단

### Step 4: 최종 레짐 결정
- 정량 신호 방향성 + 정성 분석 보조 → 최종 판단
- 확신도(confidence) 결정
- 판단 근거를 What→Why 형태로 정리

### Step 5: 투자 시사점 도출
- 현재 레짐에서의 투자 태도(공격적/보수적/방어적)
- 선호/회피 섹터
- 핵심 모니터링 포인트

## OUTPUT_SCHEMA
```json
{{
  "regime": "risk_on | risk_off | neutral",
  "confidence": 0.0-1.0,
  "regime_rationale": "레짐 판단 핵심 근거 (2-3문장, What→Why 형태)",

  "global_market": {{
    "us_market_regime": "risk_on | risk_off | neutral",
    "vix_level": VIX수치,
    "vix_signal": "low_fear | moderate | high_fear | extreme_fear",
    "credit_spread_signal": "tight | normal | wide | very_wide",
    "yield_curve_signal": "normal | flat | inverted",
    "dollar_trend": "strengthening | stable | weakening",
    "safe_haven_flow": "안전자산 흐름 설명",
    "key_indicators": [
      {{"name": "지표명", "value": 수치, "change": "변화", "signal": "신호", "interpretation": "해석"}}
    ],
    "summary": "글로벌 시장 요약 (2-3문장)"
  }},

  "global_to_korea_impact": {{
    "impact_level": "high | medium | low",
    "impact_direction": "positive | negative | mixed",
    "transmission_channels": ["외국인 자금 유입", "수출 수요 증가", ...],
    "key_factors": ["USD/KRW 안정", "미국 금리 동결", ...],
    "expected_lag": "immediate | 1-2 weeks | 1 month+",
    "narrative": "글로벌→한국 영향 설명 (2-3문장)"
  }},

  "korea_market": {{
    "kospi_trend": "uptrend | sideways | downtrend",
    "kosdaq_trend": "uptrend | sideways | downtrend",
    "foreign_investor_flow": "net_buying | neutral | net_selling",
    "foreign_flow_amount": "외국인 순매수 금액 설명",
    "exchange_rate_pressure": "appreciation | stable | depreciation",
    "usd_krw_level": 환율수치,
    "domestic_liquidity": "abundant | normal | tight",
    "market_sentiment": "bullish | neutral | bearish",
    "key_events": ["주요 이벤트 1", "주요 이벤트 2"],
    "summary": "한국 시장 요약 (2-3문장)"
  }},

  "news_sentiment": {{
    "korea_news_sentiment": "positive | neutral | negative",
    "korea_news_highlights": ["핵심 헤드라인 1", "핵심 헤드라인 2"],
    "global_news_sentiment": "positive | neutral | negative",
    "global_news_highlights": ["핵심 헤드라인 1", "핵심 헤드라인 2"],
    "foreign_view_on_korea": "positive | neutral | negative",
    "foreign_view_highlights": ["핵심 포인트 1", "핵심 포인트 2"]
  }},

  "investment_implications": {{
    "risk_appetite": "aggressive | moderate | conservative | defensive",
    "position_sizing": "포지션 사이징 가이드",
    "sector_preference": ["선호 섹터 1", "선호 섹터 2"],
    "sectors_to_avoid": ["회피 섹터 1", "회피 섹터 2"],
    "hedging_recommendation": "헤지 전략 권고",
    "time_horizon": "권장 투자 기간",
    "key_risks": ["핵심 리스크 1", "핵심 리스크 2"],
    "action_triggers": ["레짐 변화 트리거 1", "레짐 변화 트리거 2"]
  }},

  "evidence": [
    {{"type": "quantitative", "source": "데이터 출처", "fact": "구체적 사실", "interpretation": "해석"}},
    {{"type": "qualitative", "source": "뉴스 출처", "fact": "구체적 사실", "interpretation": "해석"}}
  ],
  "data_sources": ["us_vix", "us_credit_spread", "naver_news", ...],
  "analysis_timestamp": "ISO format timestamp"
}}
```

## OUTPUT
위 OUTPUT_SCHEMA에 정확히 맞는 JSON만 출력하라. 다른 텍스트는 출력하지 마라.
"""

# =============================================================================
# Validation Retry Prompt
# =============================================================================

MARKET_REGIME_RETRY_PROMPT = """## 검증 실패 - 수정 요청

이전 출력에서 다음 오류가 발견되었다:

### 오류 목록
{validation_errors}

### 이전 출력
{previous_output}

### 요청
위 오류들을 수정하여 OUTPUT_SCHEMA에 완전히 부합하는 JSON을 다시 출력하라.

특히 다음 사항을 확인하라:
1. regime은 반드시 "risk_on", "risk_off", "neutral" 중 하나
2. confidence는 0.0~1.0 범위의 숫자
3. 모든 필수 필드가 빠짐없이 채워져 있는지 확인
4. evidence에 정량/정성 근거가 각각 최소 2개 이상 포함

수정된 JSON만 출력하라.
"""

# =============================================================================
# Helper Functions
# =============================================================================

def format_market_regime_prompt(
    quantitative_signals: dict,
    korea_news: str,
    global_news: str,
    foreign_view_news: str,
    analysis_date: str,
    signal_summary: str
) -> str:
    """
    Format the market regime user prompt with data.

    Args:
        quantitative_signals: Output from regime_preprocessor.calculate_regime_signals()
        korea_news: Korean market news text (from Naver)
        global_news: Global macro news text (from Serper)
        foreign_view_news: Foreign view on Korea text (from Serper)
        analysis_date: Analysis reference date (YYYY-MM-DD)
        signal_summary: Brief summary of quantitative signals

    Returns:
        Formatted prompt string
    """
    import json

    return MARKET_REGIME_USER_PROMPT.format(
        quantitative_signals=json.dumps(quantitative_signals, ensure_ascii=False, indent=2),
        korea_news=korea_news if korea_news else "데이터 없음",
        global_news=global_news if global_news else "데이터 없음",
        foreign_view_news=foreign_view_news if foreign_view_news else "데이터 없음",
        analysis_date=analysis_date,
        signal_summary=signal_summary if signal_summary else "신호 요약 없음"
    )


def format_news_for_prompt(news_results: dict) -> tuple[str, str, str]:
    """
    Format search results into prompt-friendly text.

    Args:
        news_results: Output from search_market_news()

    Returns:
        Tuple of (korea_news, global_news, foreign_view_news) formatted strings
    """
    # Korean market news
    korea_lines = []
    for item in news_results.get("korea_market", [])[:5]:
        title = item.get("title", "")
        desc = item.get("description", "")
        if title:
            korea_lines.append(f"- {title}")
            if desc:
                korea_lines.append(f"  {desc[:200]}")
    korea_news = "\n".join(korea_lines) if korea_lines else "관련 뉴스 없음"

    # Global macro news
    global_lines = []
    for item in news_results.get("global_macro", [])[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if title:
            global_lines.append(f"- {title}")
            if snippet:
                global_lines.append(f"  {snippet[:200]}")
    global_news = "\n".join(global_lines) if global_lines else "관련 뉴스 없음"

    # Foreign view on Korea
    foreign_lines = []
    for item in news_results.get("korea_foreign_view", [])[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if title:
            foreign_lines.append(f"- {title}")
            if snippet:
                foreign_lines.append(f"  {snippet[:200]}")
    foreign_view_news = "\n".join(foreign_lines) if foreign_lines else "관련 뉴스 없음"

    return korea_news, global_news, foreign_view_news


def format_validation_retry_prompt(
    validation_errors: list[str],
    previous_output: dict
) -> str:
    """
    Format the validation retry prompt.

    Args:
        validation_errors: List of validation error messages
        previous_output: Previous LLM output that failed validation

    Returns:
        Formatted retry prompt string
    """
    import json

    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    return MARKET_REGIME_RETRY_PROMPT.format(
        validation_errors=errors_text,
        previous_output=json.dumps(previous_output, ensure_ascii=False, indent=2)
    )


def generate_signal_summary(quantitative_signals: dict) -> str:
    """
    Generate a brief summary of quantitative signals for prompt context.

    Args:
        quantitative_signals: Output from regime_preprocessor.calculate_regime_signals()

    Returns:
        Brief summary string
    """
    summary_parts = []

    if "signal_summary" in quantitative_signals:
        sig = quantitative_signals["signal_summary"]

        # Global signals summary
        global_ro = sig.get("global_risk_on_count", 0)
        global_rf = sig.get("global_risk_off_count", 0)
        if global_ro > global_rf:
            summary_parts.append(f"글로벌: Risk-On 우세({global_ro}:{global_rf})")
        elif global_rf > global_ro:
            summary_parts.append(f"글로벌: Risk-Off 우세({global_rf}:{global_ro})")
        else:
            summary_parts.append(f"글로벌: 혼재({global_ro}:{global_rf})")

        # Korea signals summary
        korea_ro = sig.get("korea_risk_on_count", 0)
        korea_rf = sig.get("korea_risk_off_count", 0)
        if korea_ro > korea_rf:
            summary_parts.append(f"한국: Risk-On 우세({korea_ro}:{korea_rf})")
        elif korea_rf > korea_ro:
            summary_parts.append(f"한국: Risk-Off 우세({korea_rf}:{korea_ro})")
        else:
            summary_parts.append(f"한국: 혼재({korea_ro}:{korea_rf})")

    return " | ".join(summary_parts) if summary_parts else "신호 분석 결과 없음"
