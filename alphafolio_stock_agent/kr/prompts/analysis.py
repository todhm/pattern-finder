"""
Analysis Agent prompt templates - Task-driven version (v2.0)
Based on improvement plan: docs/에이전트 결과물 개선 방안_v2.md
"""

# =============================================================================
# System Prompt - Task-driven
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """## TASK
제공된 PRE_FILLED_DATA를 기반으로 narrative를 작성하라.

## THINKING_PROCESS (사고 과정 - narrative 작성 전 반드시 수행)
1. 데이터 통합 분석: 모든 데이터를 종합하여 상충/일치 여부 파악
2. 논리적 연결: 거시환경 → 섹터 → 종목으로 이어지는 인과관계 구성
3. 데이터-해석 일관성: narrative가 data와 모순되지 않는지 자체 검증
4. 시나리오 차별화: 각 시나리오는 동일 데이터의 다른 전개 가능성

## DATA_INTEGRATION (데이터 통합 검증)
narrative 작성 전 아래 검증을 수행하라:

1. 수급 vs 기술적 지표:
   - 외국인/기관 순매수 상태와 RSI/MACD 신호가 일치하는가?
   - 불일치 시 그 이유를 narrative에 설명하라

2. 거시환경 vs 개별종목:
   - Risk-on/off가 이 종목의 섹터/시가총액 특성에 어떻게 적용되는가?
   - 일반론이 아닌 종목 특화 연결 논리를 제시하라

3. 확률 vs 현재 데이터:
   - 시나리오 확률이 현재 기술적/수급 상황과 정합성이 있는가?
   - 불일치 시 그 괴리를 narrative에서 설명하라

## CONSTRAINTS (필수 준수)
1. narrative 영역만 작성하라. data 영역은 시스템이 채웠으므로 수정하지 마라.
2. 숫자를 직접 생성하지 마라. PRE_FILLED_DATA의 숫자만 인용하라.
3. 모든 문장은 완전한 형태로 작성하라.
4. 기호 사용 금지: +, →, /, |
5. triggers는 system_triggers 내용을 스토리화하라. 새로운 트리거를 생성하지 마라.
6. 출력 언어는 한국어이다. 기술 용어는 한국어 표현 사용 (예: overbought -> 과매수).
7. 단순 나열 금지. 지표 값을 인과관계와 함께 해석하라.
8. 최소 2개 이상의 지표를 조합하여 복합 해석을 제시하라.
9. "따라서", "시사합니다", "의미합니다" 등 해석 연결어를 반드시 포함하라.
10. sector narrative에서 구체적인 섹터명(전자, 바이오, 통신 등)을 사용하지 마라. 반드시 "해당 섹터"로만 표현하라.

## TRANSFORMATION_RULES

### 숫자 인용 규칙
- data.rsi → "RSI는 {값}로"
- data.current_price → "현재 주가는 {값}원으로"
- data.ma20 → "{값}원의 20일 이동평균선"
- data.probability → "{값}%의 확률로"
- data.foreign_consecutive_buy → "외국인이 {값}일 연속 순매수"
- data.institutional_consecutive_buy → "기관이 {값}일 연속 순매수"
- data.volume_ratio → "거래량이 20일 평균 대비 {값*100}% 수준"
- data.support_level → "지지선 {값}원"
- data.resistance_level → "저항선 {값}원"
- data.take_profit → "익절 목표 범위는 {최소값}원에서 {최대값}원"
- data.stop_loss → "손절 범위는 {최소값}원에서 {최대값}원"

### 상태 해석 규칙
- rsi < 30 → "과매도 구간"
- rsi 30-70 → "중립 수준"
- rsi > 70 → "과매수 구간"
- adx < 20 → "추세 강도가 약해"
- adx 20-25 → "추세 강도가 보통이며"
- adx > 25 → "추세가 강하게 형성되어"
- macd_histogram > 0 → "단기 모멘텀이 긍정적"
- macd_histogram < 0 → "단기 모멘텀이 부정적"

### regime 해석 규칙
- risk_on → "위험자산 선호 환경으로 주식 투자에 유리한 시기입니다"
- risk_off → "안전자산 선호 환경으로 방어적 전략이 필요한 시기입니다"
- neutral → "중립적인 시장 환경으로 개별 종목의 펀더멘털이 중요한 시기입니다"

### triggers 변환 규칙
system_triggers의 각 트리거를 아래 형식으로 변환하라:
- "final_score {N}점 이상 상승 시" → "종합 점수가 {N}점 이상으로 상승할 경우 매수 신호가 강화됩니다"
- "외국인 {N}일 연속 순매수 전환 시" → "외국인이 {N}일 연속 순매수로 전환하면 상승 모멘텀이 형성됩니다"
- "섹터 순위 상위 {N}% 진입 시" → "섹터 순위가 상위 {N}%에 진입하면 업종 대비 강세가 확인됩니다"
- "final_score {N}점 이하 하락 시" → "종합 점수가 {N}점 이하로 하락하면 매도 신호가 강화됩니다"
- "손절선 {N}% 도달 시" → "손절선인 {N}% 하락에 도달하면 손절을 고려해야 합니다"
- "익절선 {N}% 도달 시" → "익절선인 {N}% 상승에 도달하면 이익 실현을 고려해야 합니다"
- "기관/외국인 {N}일 연속 순매도 시" → "기관과 외국인이 {N}일 연속 순매도하면 하락 압력이 강화됩니다"

### 시나리오별 전략 원칙
- 강세(bullish) 시나리오: 수익 극대화 목표. 추세 추종 전략
- 횡보(sideways) 시나리오: 물량 유지 목표. 박스권 매매 전략
- 약세(bearish) 시나리오: 손실 최소화 목표. 방어적 전략

### 시나리오 해석 원칙 (데이터 기반)
- 각 시나리오는 "해당 시나리오가 실현될 경우"의 대응 전략
- 현재 데이터가 긍정적이면 bearish 시나리오에서도 그 사실을 인정
- narrative는 data와 논리적으로 일관되어야 함
- 시나리오별 톤보다 데이터 정합성이 우선

### strategy 작성 규칙
- 포지션 비중을 백분율로 명시할 것
- 가격 조건은 PRE_FILLED_DATA의 support_level, resistance_level, take_profit, stop_loss 값을 인용할 것
- 조건부 액션 형식으로 작성할 것 (조건 충족 시 행동)
- 손절/익절 조건을 해당 시나리오 strategy에 통합

### risk_factors 작성 규칙
- bullish: 상승 시나리오가 무효화되는 조건 (수급 전환, 저항선 돌파 실패 등)
- sideways: 박스권 이탈 리스크 (상/하단 돌파 시 대응)
- bearish: 추가 하락 가속 요인 (지지선 붕괴, 수급 악화 심화 등)
- 3개 시나리오에 동일한 risk_factors 사용 금지
- 종목/섹터 특화 리스크를 포함할 것

### final_grade 결정 기준
- 강력매수: bullish >= 60% AND bearish <= 15%
- 매수: bullish >= 45% AND bearish <= 25%
- 중립: 위 조건 미충족 또는 sideways가 최고 확률
- 매도: bearish >= 45% AND bullish <= 25%
- 강력매도: bearish >= 60% AND bullish <= 15%

## INTERPRETATION_FRAMEWORK (전문가 해설 규칙)

단일 지표 나열이 아닌 복합 신호를 해석하여 전문가 수준의 인사이트를 제공하라.

### 1. 지표 간 상호작용 분석

#### RSI + 수급 조합
| RSI 상태 | 수급 상태 | 해석 |
|----------|----------|------|
| 중립(30-70) | 외국인/기관 순매수 | "수급 주도 상승 준비 단계" |
| 중립(30-70) | 외국인/기관 순매도 | "방향성 탐색 구간, 수급 전환 주시" |
| 과매수(>70) | 순매수 지속 | "추세 강화 중이나 단기 과열 주의" |
| 과매수(>70) | 순매도 전환 | "고점 신호, 차익실현 압력 증가" |
| 과매도(<30) | 순매수 전환 | "바닥 확인 신호, 반등 가능성" |
| 과매도(<30) | 순매도 지속 | "추가 하락 위험, 저점 매수 신중" |

#### MACD + ADX 조합
| MACD 상태 | ADX 상태 | 해석 |
|-----------|----------|------|
| 양수, 상승 | >25 (강한 추세) | "강한 상승 추세 진행 중" |
| 양수, 하락 | >25 | "상승 추세 둔화, 조정 가능성" |
| 음수, 하락 | >25 | "강한 하락 추세 진행 중" |
| 음수, 상승 | >25 | "하락 추세 둔화, 반등 시도" |
| 양수/음수 | <20 (약한 추세) | "추세 부재, 박스권 매매 고려" |

#### 가격 위치 + 볼린저 밴드 조합
| 가격 vs MA20 | 볼린저 위치 | 해석 |
|--------------|-------------|------|
| 위 (+1% 이상) | 상단 근접 | "단기 과열, 조정 후 재진입 고려" |
| 위 (+1% 이상) | 중단 부근 | "건강한 상승 추세" |
| 근접 (+-1%) | 중단 부근 | "방향성 결정 임박, 돌파 방향 주시" |
| 아래 (-1% 이상) | 하단 근접 | "단기 과매도, 기술적 반등 가능" |
| 아래 (-1% 이상) | 중단 부근 | "하락 추세 진행 중" |

### 2. 해설 품질 기준

#### BAD (단순 나열 - 금지):
"RSI는 42.72로 중립 수준입니다. MACD 히스토그램은 -280.38입니다. 외국인이 14일 연속 순매수하고 있습니다."

#### GOOD (복합 해석 - 필수):
"RSI 42.72(중립)와 외국인/기관 14일 연속 순매수의 조합은 수급 주도형 상승 준비 단계를 시사합니다. 다만 MACD 히스토그램 음수(-280)로 단기 모멘텀은 아직 회복되지 않아, 72,498원 저항선 돌파 여부가 방향성 결정의 핵심 변수입니다."

### 3. 시나리오 전략 해설 기준

각 시나리오의 strategy는 다음 3단계 구조로 작성:
1. 현재 상태 진단: 지금 어디에 있는가 (추세/모멘텀/수급 상태)
2. 핵심 변수 식별: 무엇이 방향을 결정하는가 (지지선/저항선/수급 전환점)
3. 대응 전략 도출: 어떻게 해야 하는가 (구체적 가격과 조건 명시)

## VALIDATION
출력 후 시스템이 다음을 검증한다:
1. narrative 내 모든 숫자가 INPUT_DATA와 일치하는지
2. triggers가 system_triggers를 기반으로 작성되었는지
3. 금지된 기호(+, →, /, |)가 사용되지 않았는지
4. 3개 시나리오 확률 합계가 100%인지

검증 실패 시 재시도 요청이 온다. 최대 2회.
"""

# =============================================================================
# User Prompt Template - Task-driven
# =============================================================================

ANALYSIS_USER_PROMPT = """## PRE_FILLED_DATA (시스템이 계산 완료 - 수정 금지)
아래 data 영역의 모든 숫자는 시스템이 계산한 값이다. 절대 수정하지 마라.

{pre_filled_template}

## REFERENCE_DATA (narrative 작성 시 참고)

### market_regime (Market Regime Agent 결과)
{market_regime}

### stock_research (Stock Research Agent 결과)
{stock_research}

### system_triggers (시스템 제공 트리거 - 반드시 사용)
{system_triggers}

## CONTEXT
- 분석 기준일: {analysis_date}
- 시장: {exchange}
- 종목명: {stock_name}
- 종목코드: {symbol}

## TASK
PRE_FILLED_DATA의 data 값을 기반으로 narrative 영역만 작성하라.

## OUTPUT_FORMAT
narrative 영역만 포함된 JSON을 출력하라. data 영역은 출력하지 마라.

```json
{{
  "final_grade": "투자등급 (강력매수/매수/중립/매도/매도 고려)",

  "market_environment_narrative": {{
    "global_env": "미국 VIX가 [PRE_FILLED_DATA.market_environment.data.vix]로 ... (2-3문장)",
    "domestic": "한국은행 기준금리가 ... 국내 유동성 환경이 ... (2-3문장)",
    "sector": "해당 섹터는 ... 모멘텀을 ... (2-3문장, 주의: 전자/바이오/통신 등 구체적 섹터명 사용 금지, 반드시 '해당 섹터'로만 표현)",
    "regime_interpretation": "전반적으로 [PRE_FILLED_DATA.market_environment.data.regime] 시장 환경으로 ... (1-2문장)"
  }},

  "technical_summary_narrative": {{
    "price_trend": "현재 주가는 [PRE_FILLED_DATA.technical_summary.data.current_price]원으로 ... 이동평균선 ... (2-3문장)",
    "indicators": "RSI는 [PRE_FILLED_DATA.technical_summary.data.rsi]로 ... MACD ... ADX ... (2-3문장)",
    "investor_flow": "외국인이 [PRE_FILLED_DATA.technical_summary.data.foreign_consecutive_buy]일 연속 ... (1-2문장)",
    "volume_analysis": "거래량이 20일 평균 대비 ... (1문장)"
  }},

  "scenarios_narrative": {{
    "bullish": {{
      "title": "상승 시나리오 제목 (1문장)",
      "probability_explanation": "과거 유사 패턴 [sample_count]건 분석 결과, [PRE_FILLED_DATA.scenarios.bullish.data.probability]%의 확률로 ... (1문장)",
      "confidence_rationale": "... 근거로 신뢰도가 ... (1-2문장)",
      "strategy": "현재 가격대에서 ... 익절 목표는 [PRE_FILLED_DATA.scenarios.bullish.data.take_profit] ... (2-3문장)",
      "triggers": ["system_triggers.buy 기반 문장1", "system_triggers.buy 기반 문장2"],
      "monitoring_points": ["[resistance_level]원 저항선 돌파 여부", "..."],
      "risk_factors": ["...", "..."]
    }},
    "sideways": {{
      "title": "횡보 시나리오 제목 (1문장)",
      "probability_explanation": "[PRE_FILLED_DATA.scenarios.sideways.data.probability]%의 확률로 ... (1문장)",
      "confidence_rationale": "... (1-2문장)",
      "strategy": "현재 가격대에서 보유를 유지하며 ... (2-3문장)",
      "triggers": ["system_triggers.hold 기반 문장1", "system_triggers.hold 기반 문장2"],
      "monitoring_points": ["...", "..."],
      "risk_factors": ["...", "..."]
    }},
    "bearish": {{
      "title": "하락 시나리오 제목 (1문장)",
      "probability_explanation": "[PRE_FILLED_DATA.scenarios.bearish.data.probability]%의 확률로 ... (1문장)",
      "confidence_rationale": "... (1-2문장)",
      "strategy": "손절을 고려하며 관망하는 전략이 필요합니다. 손절 범위는 [PRE_FILLED_DATA.scenarios.bearish.data.stop_loss] ... (2-3문장, 반드시 손절/관망/현금비중 포함)",
      "triggers": ["system_triggers.sell 기반 문장1", "system_triggers.sell 기반 문장2"],
      "monitoring_points": ["...", "..."],
      "risk_factors": ["...", "..."]
    }}
  }}
}}
```

## REQUIREMENTS
1. PRE_FILLED_DATA의 숫자를 그대로 인용하라. 임의 생성/수정 금지.
2. triggers는 반드시 system_triggers 내용을 TRANSFORMATION_RULES에 따라 변환하라.
3. 모든 텍스트는 완전한 문장으로 작성하라. 기호(+, /, |) 사용 금지.
4. triggers, monitoring_points, risk_factors는 각각 2개 이상 제시하라.
5. bearish 시나리오 strategy에 "매수" 용어 사용 금지. "손절/관망/현금비중" 필수 포함.

## OUTPUT
위 OUTPUT_FORMAT에 맞는 JSON만 출력하라. 다른 텍스트는 출력하지 마라.
"""

# =============================================================================
# Validation Retry Prompt - Task-driven
# =============================================================================

VALIDATION_RETRY_PROMPT = """## VALIDATION FAILED
다음 오류를 수정하라:

### ERRORS
{validation_errors}

### TRANSFORMATION_RULES 재확인
1. 숫자는 INPUT_DATA의 값을 정확히 인용하라. 반올림, 변환, 추정 금지.
2. triggers는 system_triggers 내용을 narrative로 변환하라. 새로운 트리거 생성 금지.
3. 기호(+, →, /, |) 대신 완전한 문장 사용.
4. 3개 시나리오 확률 합계는 100%.
5. bearish strategy에 "매수" 금지, "손절/관망/현금비중" 필수.

### PREVIOUS OUTPUT
{previous_output}

### CORRECTED OUTPUT
위 규칙을 준수하여 수정된 JSON만 출력하라.
"""

# =============================================================================
# Helper Functions
# =============================================================================

def extract_system_triggers(quant_summary: dict) -> dict:
    """
    Extract system triggers from quant_summary.

    Args:
        quant_summary: Quant analysis summary containing triggers

    Returns:
        System triggers dictionary with buy, sell, hold categories
    """
    import json

    triggers = quant_summary.get("triggers", {})

    # Parse triggers if they are JSON strings
    def parse_trigger(trigger_value):
        if isinstance(trigger_value, str):
            try:
                return json.loads(trigger_value)
            except json.JSONDecodeError:
                return [trigger_value]
        elif isinstance(trigger_value, list):
            return trigger_value
        return []

    return {
        "buy": parse_trigger(triggers.get("buy", [])),
        "sell": parse_trigger(triggers.get("sell", [])),
        "hold": parse_trigger(triggers.get("hold", []))
    }


def format_analysis_prompt(
    pre_filled_template: dict,
    market_regime: dict = None,
    stock_research: dict = None,
    system_triggers: dict = None,
    analysis_date: str = "",
    exchange: str = "KOSPI",
    stock_name: str = "",
    symbol: str = ""
) -> str:
    """
    Format the analysis user prompt with pre-filled template.

    Args:
        pre_filled_template: Pre-calculated output template from build_output_template()
        market_regime: Market Regime Agent output (optional)
        stock_research: Stock Research Agent output (optional)
        system_triggers: System-provided triggers
        analysis_date: Analysis reference date
        exchange: Stock exchange (KOSPI/KOSDAQ)
        stock_name: Stock name in Korean
        symbol: Stock code

    Returns:
        Formatted prompt string
    """
    import json

    # Format market regime - extract key information for analysis prompt
    if market_regime:
        regime_summary = {
            "regime": market_regime.get("regime", "neutral"),
            "confidence": market_regime.get("confidence", 0.5),
            "rationale": market_regime.get("regime_rationale", ""),
            "global_market_summary": market_regime.get("global_market", {}).get("summary", ""),
            "korea_market_summary": market_regime.get("korea_market", {}).get("summary", ""),
            "impact_direction": market_regime.get("global_to_korea_impact", {}).get("impact_direction", "mixed"),
            "investment_implications": {
                "risk_appetite": market_regime.get("investment_implications", {}).get("risk_appetite", "moderate"),
                "sector_preference": market_regime.get("investment_implications", {}).get("sector_preference", []),
                "key_risks": market_regime.get("investment_implications", {}).get("key_risks", [])
            }
        }
        market_regime_text = json.dumps(regime_summary, ensure_ascii=False, indent=2)
    else:
        market_regime_text = "시장 레짐 분석 결과 없음 (데이터 수집 실패 또는 미실행)"

    # Format stock research - extract key information for analysis prompt
    if stock_research:
        research_summary = {
            "overall_sentiment": stock_research.get("overall_sentiment", "neutral"),
            "confidence": stock_research.get("confidence", 0.5),
            "earnings_outlook": stock_research.get("earnings_outlook", {}),
            "risk_issues": stock_research.get("risk_issues", []),
            "sector_momentum": stock_research.get("sector_momentum", {}),
            "disclosure_summary": stock_research.get("disclosure_summary", {}),
            "metadata": {
                "data_collection": stock_research.get("metadata", {}).get("data_collection", {}),
                "stock_size": stock_research.get("metadata", {}).get("stock_size", "medium")
            }
        }
        stock_research_text = json.dumps(research_summary, ensure_ascii=False, indent=2)
    else:
        stock_research_text = "종목 정성 분석 결과 없음 (데이터 수집 실패 또는 미실행)"

    # Format system_triggers
    system_triggers_text = json.dumps(system_triggers or {}, ensure_ascii=False, indent=2)

    # Format pre-filled template (data sections only)
    pre_filled_text = json.dumps(pre_filled_template, ensure_ascii=False, indent=2)

    return ANALYSIS_USER_PROMPT.format(
        pre_filled_template=pre_filled_text,
        market_regime=market_regime_text,
        stock_research=stock_research_text,
        system_triggers=system_triggers_text,
        analysis_date=analysis_date,
        exchange=exchange or "KOSPI",
        stock_name=stock_name,
        symbol=symbol
    )


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

    return VALIDATION_RETRY_PROMPT.format(
        validation_errors=errors_text,
        previous_output=json.dumps(previous_output, ensure_ascii=False, indent=2)
    )
