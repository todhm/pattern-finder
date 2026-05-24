"""
Analysis Agent prompt templates - Task-driven version (v2.0) for US stocks
Based on V2 Data/Narrative separation architecture
Output language: Korean
"""

# =============================================================================
# System Prompt - Task-driven (Korean Output)
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

1. 옵션 플로우 vs 기술적 지표:
   - Put/Call 비율과 IV 백분위가 RSI/MACD 신호와 일치하는가?
   - 불일치 시 그 이유를 narrative에 설명하라

2. 거시환경 vs 개별종목:
   - Risk-on/off가 이 종목의 섹터/시가총액 특성에 어떻게 적용되는가?
   - 일반론이 아닌 종목 특화 연결 논리를 제시하라

3. 확률 vs 현재 데이터:
   - 시나리오 확률이 현재 기술적/옵션 상황과 정합성이 있는가?
   - 불일치 시 그 괴리를 narrative에서 설명하라

## CONSTRAINTS (필수 준수)
1. narrative 영역만 작성하라. data 영역은 시스템이 채웠으므로 수정하지 마라.
2. 숫자를 직접 생성하지 마라. PRE_FILLED_DATA의 숫자만 인용하라.
3. 모든 문장은 완전한 형태로 작성하라.
4. 기호 사용 금지: +, →, |
5. triggers는 system_triggers 내용을 스토리화하라. 새로운 트리거를 생성하지 마라.
6. 출력 언어는 한국어이다. 기술 용어는 한국어 표현 사용 (예: overbought -> 과매수).
7. 단순 나열 금지. 지표 값을 인과관계와 함께 해석하라.
8. 최소 2개 이상의 지표를 조합하여 복합 해석을 제시하라.
9. "따라서", "시사합니다", "의미합니다" 등 해석 연결어를 반드시 포함하라.
10. sector narrative에서 구체적인 섹터명(Technology, Healthcare, Finance 등)을 사용하지 마라. 반드시 "해당 섹터"로만 표현하라.

## TRANSFORMATION_RULES

### 숫자 인용 규칙
- data.rsi → "RSI는 {값}로"
- data.current_price → "현재 주가는 ${값}로"
- data.ma20 → "${값}의 20일 이동평균선"
- data.probability → "{값}%의 확률로"
- data.iv_percentile → "IV 백분위는 {값}"
- data.put_call_ratio → "Put/Call 비율이 {값}으로"
- data.volume_ratio → "거래량이 20일 평균 대비 {값*100}% 수준"
- data.support_level → "지지선 ${값}"
- data.resistance_level → "저항선 ${값}"
- data.take_profit → "익절 목표가 ${값}"
- data.stop_loss → "손절가 ${값}"

### 상태 해석 규칙
- rsi < 30 → "과매도 구간"
- rsi 30-70 → "중립 수준"
- rsi > 70 → "과매수 구간"
- adx < 20 → "추세 강도가 약해"
- adx 20-25 → "추세 강도가 보통이며"
- adx > 25 → "추세가 강하게 형성되어"
- macd_histogram > 0 → "단기 모멘텀이 긍정적"
- macd_histogram < 0 → "단기 모멘텀이 부정적"
- iv_percentile < 20 → "변동성이 낮은 환경"
- iv_percentile > 80 → "변동성이 높은 환경"
- put_call_ratio < 0.7 → "옵션 시장에서 강세 심리"
- put_call_ratio > 1.0 → "옵션 시장에서 약세 심리"

### regime 해석 규칙
- risk_on → "위험자산 선호 환경으로 주식 투자에 유리한 시기입니다"
- risk_off → "안전자산 선호 환경으로 방어적 전략이 필요한 시기입니다"
- neutral → "중립적인 시장 환경으로 개별 종목의 펀더멘털이 중요한 시기입니다"

### triggers 변환 규칙
system_triggers의 각 트리거를 아래 형식으로 변환하라:
- "final_score {N}점 이상 상승 시" → "종합 점수가 {N}점 이상으로 상승할 경우 매수 신호가 강화됩니다"
- "내부자 {N}일 연속 순매수 시" → "내부자가 {N}일 연속 순매수하면 매집이 확인됩니다"
- "섹터 순위 상위 {N}% 진입 시" → "섹터 순위가 상위 {N}%에 진입하면 업종 대비 강세가 확인됩니다"
- "final_score {N}점 이하 하락 시" → "종합 점수가 {N}점 이하로 하락하면 매도 신호가 강화됩니다"
- "손절선 {N}% 도달 시" → "손절선인 {N}% 하락에 도달하면 손절을 고려해야 합니다"
- "익절선 {N}% 도달 시" → "익절선인 {N}% 상승에 도달하면 이익 실현을 고려해야 합니다"
- "Put/Call 비율 {N} 초과 시" → "Put/Call 비율이 {N}을 초과하면 하락 압력이 강화됩니다"

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
- bullish: 상승 시나리오가 무효화될 조건 (옵션 플로우 역전, 저항선 돌파 실패 등)
- sideways: 박스권 이탈 위험 (상단/하단 이탈 시 대응)
- bearish: 추가 하락 가속 요인 (지지선 붕괴, 매도 압력 심화 등)
- 3개 시나리오에 동일한 risk_factors를 사용하지 마라
- 종목/섹터 특화 리스크를 포함하라

### final_grade 결정 기준 (7단계)
- 강력 매수: bullish >= 65% AND bearish <= 10%
- 매수: bullish >= 55% AND bearish <= 20%
- 매수 고려: bullish >= 45% AND bearish <= 30%
- 중립: 위 조건 미충족 OR sideways가 최고 확률
- 매도 고려: bearish >= 45% AND bullish <= 30%
- 매도: bearish >= 55% AND bullish <= 20%
- 강력 매도: bearish >= 65% AND bullish <= 10%

## INTERPRETATION_FRAMEWORK (전문가 분석 규칙)

단일 지표 나열이 아닌 복합 신호 해석으로 전문가 수준의 인사이트를 제공하라.

### 1. 지표 상호작용 분석

#### RSI + 옵션 플로우 조합
| RSI 상태 | 옵션 플로우 | 해석 |
|----------|-------------|------|
| 중립 (30-70) | 낮은 P/C 비율 | "옵션 심리가 상승 돌파 가능성을 지지합니다" |
| 중립 (30-70) | 높은 P/C 비율 | "기술적 중립에도 불구하고 옵션 헤지가 주의를 시사합니다" |
| 과매수 (>70) | 낮은 P/C 비율 | "상승 추세가 연장될 수 있으나 옵션 포지셔닝은 고점을 시사합니다" |
| 과매수 (>70) | 높은 P/C 비율 | "기술적, 옵션 모두 주의 신호를 나타냅니다" |
| 과매도 (<30) | 낮은 P/C 비율 | "바닥 형성과 함께 강세 옵션 포지셔닝이 나타납니다" |
| 과매도 (<30) | 높은 P/C 비율 | "극단적 공포 상태이나 옵션 항복은 아직 아닙니다" |

#### MACD + ADX 조합
| MACD 상태 | ADX 상태 | 해석 |
|-----------|----------|------|
| 양수, 상승 | >25 (강한 추세) | "강한 상승 추세가 진행 중입니다" |
| 양수, 하락 | >25 | "상승 추세가 둔화되고 있으며 조정 가능성이 있습니다" |
| 음수, 하락 | >25 | "강한 하락 추세가 진행 중입니다" |
| 음수, 상승 | >25 | "하락 추세가 둔화되며 반등을 시도하고 있습니다" |
| 양수/음수 | <20 (약한 추세) | "명확한 추세 없이 박스권 매매 상황입니다" |

### 2. 분석 품질 기준

#### BAD (단순 나열 - 금지):
"RSI는 42.72로 중립 수준입니다. MACD 히스토그램은 -0.45입니다. IV 백분위는 65입니다."

#### GOOD (복합 해석 - 권장):
"RSI가 42.72로 중립 수준이며 Put/Call 비율 0.85와 결합하면 최근 횡보에도 불구하고 옵션 트레이더들이 상승에 베팅하고 있음을 시사합니다. 그러나 IV 백분위가 65로 상승해 있어 시장은 단기 내 큰 움직임을 예상하고 있으며, $175 저항선이 핵심 분기점이 됩니다."

### 3. 시나리오 전략 기준

각 시나리오의 전략은 3단계 구조를 따라야 합니다:
1. 현재 상태 진단: 현재 위치 (추세/모멘텀/옵션 상태)
2. 핵심 변수 식별: 방향을 결정하는 요소 (지지/저항/옵션 플로우 변화)
3. 대응 전략: 구체적인 가격과 조건에 따른 행동

## VALIDATION
출력 후 시스템이 검증합니다:
1. narrative의 모든 숫자가 INPUT_DATA와 일치
2. triggers가 system_triggers 기반
3. 금지 기호 (+, →, |) 미사용
4. 3개 시나리오 확률 합계가 100%
5. bearish 전략에 방어적 키워드 포함 (손절, 청산, 축소, 헤지, 현금)

검증 실패 시 재시도 요청이 옵니다. 최대 2회 재시도.
"""

# =============================================================================
# User Prompt Template - Task-driven (Korean Output)
# =============================================================================

ANALYSIS_USER_PROMPT = """## PRE_FILLED_DATA (시스템 계산 값 - 수정 금지)
아래 data 섹션의 모든 숫자는 시스템이 계산한 값입니다. 수정하지 마세요.

{pre_filled_template}

## REFERENCE_DATA (narrative 작성 참조)

### market_regime (Market Regime Agent 결과)
{market_regime}

### stock_research (Stock Research Agent 결과)
{stock_research}

### system_triggers (시스템 제공 트리거 - 반드시 사용)
{system_triggers}

## CONTEXT
- 분석 기준일: {analysis_date}
- 거래소: {exchange}
- 종목명: {stock_name}
- 티커: {symbol}

## TASK
PRE_FILLED_DATA 값을 기반으로 narrative 섹션만 작성하라.

## OUTPUT_FORMAT
narrative 섹션만 포함하는 JSON을 출력하라. data 섹션은 출력하지 마라.

```json
{{
  "final_grade": "투자 등급 (강력 매수/매수/매수 고려/중립/매도 고려/매도/강력 매도)",

  "market_environment_narrative": {{
    "global_env": "VIX가 [PRE_FILLED_DATA.market_environment.data.vix]로... (2-3문장)",
    "fed_policy": "연준 금리가... 금리 환경은... (2-3문장)",
    "sector": "해당 섹터는... 모멘텀이... (2-3문장, 주의: Technology/Healthcare 등 구체적 섹터명 사용 금지, 반드시 '해당 섹터'로만 표현)",
    "regime_interpretation": "전반적으로 [PRE_FILLED_DATA.market_environment.data.regime] 시장 환경으로... (1-2문장)"
  }},

  "technical_summary_narrative": {{
    "price_trend": "현재 주가 $[PRE_FILLED_DATA.technical_summary.data.current_price]로... 이동평균선... (2-3문장)",
    "indicators": "RSI가 [PRE_FILLED_DATA.technical_summary.data.rsi]로... MACD... ADX... (2-3문장)",
    "options_flow": "Put/Call 비율이 [값]으로... IV 백분위... (1-2문장)",
    "insider_activity": "내부자 신호는 [값]으로... (1문장)"
  }},

  "scenarios_narrative": {{
    "bullish": {{
      "title": "강세 시나리오 제목 (1문장)",
      "probability_explanation": "[sample_count]개 과거 패턴 기반, [PRE_FILLED_DATA.scenarios.bullish.data.probability]% 확률로... (1문장)",
      "confidence_rationale": "... 신뢰도의 근거는... (1-2문장)",
      "strategy": "현재 가격대에서... 익절 목표가 $[PRE_FILLED_DATA.scenarios.bullish.data.take_profit]로... (2-3문장)",
      "triggers": ["system_triggers.buy 기반 문장 1", "system_triggers.buy 기반 문장 2"],
      "monitoring_points": ["$[resistance_level] 저항선 돌파", "..."],
      "risk_factors": ["...", "..."]
    }},
    "sideways": {{
      "title": "횡보 시나리오 제목 (1문장)",
      "probability_explanation": "[PRE_FILLED_DATA.scenarios.sideways.data.probability]% 확률로... (1문장)",
      "confidence_rationale": "... (1-2문장)",
      "strategy": "박스권 내에서 현재 포지션 유지... (2-3문장)",
      "triggers": ["system_triggers.hold 기반 문장 1", "system_triggers.hold 기반 문장 2"],
      "monitoring_points": ["...", "..."],
      "risk_factors": ["...", "..."]
    }},
    "bearish": {{
      "title": "약세 시나리오 제목 (1문장)",
      "probability_explanation": "[PRE_FILLED_DATA.scenarios.bearish.data.probability]% 확률로... (1문장)",
      "confidence_rationale": "... (1-2문장)",
      "strategy": "방어적 포지셔닝이 필요합니다. 손절가 [PRE_FILLED_DATA.scenarios.bearish.data.stop_loss]에서... (2-3문장, 반드시 손절/청산/축소/헤지/현금 포함)",
      "triggers": ["system_triggers.sell 기반 문장 1", "system_triggers.sell 기반 문장 2"],
      "monitoring_points": ["...", "..."],
      "risk_factors": ["...", "..."]
    }}
  }}
}}
```

## REQUIREMENTS
1. PRE_FILLED_DATA 숫자를 정확히 인용하라. 임의 생성/수정 금지.
2. triggers는 반드시 TRANSFORMATION_RULES에 따라 system_triggers 내용을 변환하라.
3. 모든 텍스트는 완전한 문장으로 작성하라. 기호 (+, |) 금지.
4. triggers, monitoring_points, risk_factors는 각각 최소 2개 이상.
5. bearish 시나리오 strategy에 "매수" 키워드 금지. 반드시 "손절/청산/축소/헤지/현금" 포함.

## OUTPUT
위 OUTPUT_FORMAT과 일치하는 JSON만 출력하라. 다른 텍스트 없음.
"""

# =============================================================================
# Validation Retry Prompt - Task-driven (Korean)
# =============================================================================

VALIDATION_RETRY_PROMPT = """## VALIDATION FAILED
아래 오류를 수정하라:

### ERRORS
{validation_errors}

### TRANSFORMATION_RULES 상기
1. INPUT_DATA 값을 정확히 인용하라. 반올림, 변환, 추정 금지.
2. system_triggers 내용을 narrative로 변환하라. 새로운 트리거 생성 금지.
3. 기호 (+, →, |) 대신 완전한 문장을 사용하라.
4. 3개 시나리오 확률 합계는 100%여야 한다.
5. bearish 전략: "매수" 금지, 반드시 "손절/청산/축소/헤지/현금" 포함.

### PREVIOUS OUTPUT
{previous_output}

### CORRECTED OUTPUT
위 규칙을 준수한 수정된 JSON만 출력하라.
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
    exchange: str = "NYSE",
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
        exchange: Stock exchange (NYSE/NASDAQ)
        stock_name: Company name
        symbol: Stock ticker

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
            "us_market_summary": market_regime.get("us_market", {}).get("summary", ""),
            "investment_implications": {
                "risk_appetite": market_regime.get("investment_implications", {}).get("risk_appetite", "moderate"),
                "sector_preference": market_regime.get("investment_implications", {}).get("sector_preference", []),
                "key_risks": market_regime.get("investment_implications", {}).get("key_risks", [])
            }
        }
        market_regime_text = json.dumps(regime_summary, ensure_ascii=False, indent=2)
    else:
        market_regime_text = "시장 레짐 분석 불가 (데이터 수집 실패 또는 미실행)"

    # Format stock research - extract key information for analysis prompt
    if stock_research:
        research_summary = {
            "overall_sentiment": stock_research.get("overall_sentiment", "neutral"),
            "confidence": stock_research.get("confidence", 0.5),
            "earnings_outlook": stock_research.get("earnings_outlook", {}),
            "risk_issues": stock_research.get("risk_issues", []),
            "sector_momentum": stock_research.get("sector_momentum", {}),
            "insider_activity": stock_research.get("insider_activity", {}),
            "metadata": {
                "data_collection": stock_research.get("metadata", {}).get("data_collection", {}),
                "market_cap_tier": stock_research.get("metadata", {}).get("market_cap_tier", "mid")
            }
        }
        stock_research_text = json.dumps(research_summary, ensure_ascii=False, indent=2)
    else:
        stock_research_text = "종목 리서치 불가 (데이터 수집 실패 또는 미실행)"

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
        exchange=exchange or "NYSE",
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
