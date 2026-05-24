"""
Stock Research Agent prompt templates
Based on:
- 에이전트 작업 계획.md section 3.3.2
- 에이전트 개발 계획.md section 10

설계 원칙:
1. LLM은 3가지 항목만 추출: 실적 전망(earnings_outlook), 리스크 이슈(risk_issues), sector 모멘텀(sector_momentum)
2. 입력 데이터에 없는 내용 생성 금지
3. 모든 summary에 source 매핑 필수
4. Task-driven 접근 (Role-driven X)
5. 퀀트 결과 변경 X, 방향성 확인 + 설명 풍부화 용도
"""
import json
from datetime import date


# =============================================================================
# Output Schema (for prompt reference)
# =============================================================================

OUTPUT_SCHEMA = """{
  "symbol": "종목코드",
  "stock_name": "종목명",
  "analysis_date": "YYYY-MM-DD",
  "sector": "섹터명",

  "overall_sentiment": "positive | neutral | negative",
  "confidence": 0.0-1.0,

  "earnings_outlook": {
    "direction": "positive | neutral | negative",
    "summary": "실적 전망 요약 (1-2문장)",
    "key_points": ["핵심 포인트 1", "핵심 포인트 2"],
    "sources": [
      {"type": "research_report | news | disclosure", "title": "제목", "publisher": "출처", "date": "날짜"}
    ]
  },

  "risk_issues": [
    {
      "category": "regulatory | market | operational | financial | other",
      "severity": "high | medium | low",
      "title": "리스크 제목",
      "summary": "리스크 설명 (1-2문장)",
      "sources": [{"type": "...", "title": "...", "publisher": "...", "date": "..."}]
    }
  ],

  "sector_momentum": {
    "sector": "섹터명",
    "direction": "positive | neutral | negative",
    "summary": "섹터 모멘텀 요약 (1-2문장)",
    "related_news_count": 뉴스건수,
    "sources": [{"type": "...", "title": "...", "publisher": "...", "date": "..."}]
  },

  "disclosure_summary": {
    "recent_count": 공시건수,
    "significant_items": [
      {"type": "earnings | shareholder | executive | dividend | investment | regulatory | other", "title": "공시 제목", "date": "날짜", "source": "DART | KIND"}
    ]
  },

  "metadata": {
    "data_collection": {
      "research_reports": 리포트건수,
      "news_articles": 뉴스건수,
      "disclosures": 공시건수
    },
    "excluded": {
      "sns": "Tier 3 excluded",
      "community": "Tier 3 excluded"
    },
    "stock_size": "large | medium | small",
    "collection_date": "YYYY-MM-DD"
  }
}"""


# =============================================================================
# System Prompt
# =============================================================================

STOCK_RESEARCH_SYSTEM_PROMPT = """너는 종목 리서치 데이터 추출 모듈이다.

## 목적
주어진 데이터에서 정보를 추출하여 구조화된 형태로 변환한다.
추출된 정보는 퀀트 분석 결과의 방향성을 확인하고 설명을 풍부하게 하는 용도로 사용된다.

## 핵심 원칙
1. **데이터 기반**: 입력된 INPUT_DATA에 있는 정보만 사용한다
2. **추출 전용**: 새로운 분석이나 예측을 생성하지 않는다
3. **출처 필수**: 모든 요약(summary)에는 반드시 source를 명시한다
4. **3가지 항목 집중**: earnings_outlook, risk_issues, sector_momentum만 추출한다

## 추출 가이드라인

### earnings_outlook (실적 전망)
- 증권사 리포트의 목표가, 투자의견, 실적 전망 추출
- 뉴스에서 실적 관련 내용 추출
- direction 판단 기준:
  - positive: 실적 개선, 목표가 상향, 매수 의견 우세
  - neutral: 혼재된 의견, 실적 유지 전망
  - negative: 실적 악화, 목표가 하향, 매도 의견 우세

### risk_issues (리스크 이슈)
- 규제, 시장, 운영, 재무 관련 리스크 추출
- category 분류:
  - regulatory: 정부 규제, 법적 이슈, 제재
  - market: 경쟁 심화, 수요 감소, 가격 하락
  - operational: 생산 차질, 품질 이슈, 인력 문제
  - financial: 부채, 유동성, 환율 리스크
  - other: 기타 리스크
- severity 판단:
  - high: 즉시 대응 필요, 실적/주가 직접 영향
  - medium: 주시 필요, 잠재적 영향
  - low: 참고 사항

### sector_momentum (섹터 모멘텀)
- sector 관련 뉴스에서 산업/섹터 흐름 추출
- 관련 종목, 정책, 시장 동향 포함
- direction 판단 기준:
  - positive: 섹터 성장, 정책 지원, 투자 확대
  - neutral: 특별한 변화 없음
  - negative: 섹터 침체, 규제 강화, 투자 축소

## 제약사항
- INPUT_DATA에 없는 정보 생성 금지
- 추측이나 가정 금지
- source가 없는 내용은 포함하지 않음
- OUTPUT_SCHEMA에 정확히 맞춰 JSON 출력
"""


# =============================================================================
# User Prompt Template
# =============================================================================

STOCK_RESEARCH_USER_PROMPT = """## TASK
종목 정성적 분석 데이터 추출

## TARGET_STOCK
- 종목명: {stock_name}
- 종목코드: {symbol}
- Sector: {sector}
- 종목규모: {stock_size}
- 분석일: {analysis_date}

## INPUT_DATA
{formatted_text}

## DATA_STATISTICS
- 증권사 리포트: {report_count}건
- 뉴스 기사: {news_count}건
- 공시: {disclosure_count}건

## EXTRACTION_INSTRUCTIONS

### Step 1: 증권사 리포트 분석
- 최신 리포트의 목표가와 투자의견 확인
- 실적 전망 관련 핵심 포인트 추출
- 리포트별 출처 정보 기록

### Step 2: 뉴스 분석
- 종목 관련 뉴스에서 실적/리스크 관련 내용 추출
- Sector 관련 뉴스에서 산업 동향 추출
- 긍정/부정 뉴스 비율 파악

### Step 3: 공시 분석
- 주요 공시 유형 파악 (실적, 지분변동, 임원변경 등)
- 투자 판단에 영향을 줄 수 있는 공시 식별

### Step 4: 종합 sentiment 판단
- earnings_outlook, risk_issues, sector_momentum 종합
- overall_sentiment 및 confidence 결정
- confidence 기준:
  - 0.8+: 데이터 충분, 방향성 명확
  - 0.5-0.8: 데이터 보통, 방향성 존재
  - 0.5 미만: 데이터 부족 또는 혼재

## OUTPUT_SCHEMA
```json
{output_schema}
```

## OUTPUT
위 OUTPUT_SCHEMA에 정확히 맞는 JSON만 출력하라. 다른 텍스트는 출력하지 마라.
INPUT_DATA에 정보가 부족한 경우에도 스키마의 모든 필드를 채워야 한다.
정보가 없으면 빈 리스트([])나 "정보 없음"으로 표시하라.
"""


# =============================================================================
# Validation Retry Prompt
# =============================================================================

STOCK_RESEARCH_RETRY_PROMPT = """## 검증 실패 - 수정 요청

이전 출력에서 다음 오류가 발견되었다:

### 오류 목록
{validation_errors}

### 이전 출력
{previous_output}

### 요청
위 오류들을 수정하여 OUTPUT_SCHEMA에 완전히 부합하는 JSON을 다시 출력하라.

특히 다음 사항을 확인하라:
1. overall_sentiment는 반드시 "positive", "neutral", "negative" 중 하나
2. confidence는 0.0~1.0 범위의 숫자
3. earnings_outlook.direction, sector_momentum.direction도 동일 규칙
4. risk_issues의 각 항목에 category, severity, title, summary 필수
5. 모든 summary에 sources 배열 포함 (비어있어도 됨)

수정된 JSON만 출력하라.
"""


# =============================================================================
# Helper Functions
# =============================================================================

def format_stock_research_prompt(
    symbol: str,
    stock_name: str,
    sector: str,
    stock_size: str,
    analysis_date: str,
    formatted_text: str,
    report_count: int,
    news_count: int,
    disclosure_count: int
) -> str:
    """
    Format the stock research user prompt with data.

    Args:
        symbol: Stock code (e.g., "005930")
        stock_name: Stock name in Korean
        sector: Sector from kr_stock_detail.theme
        stock_size: "large" | "medium" | "small"
        analysis_date: Analysis date (YYYY-MM-DD)
        formatted_text: Preprocessed text from research_preprocessor
        report_count: Number of research reports
        news_count: Number of news articles
        disclosure_count: Number of disclosures

    Returns:
        Formatted prompt string
    """
    return STOCK_RESEARCH_USER_PROMPT.format(
        symbol=symbol,
        stock_name=stock_name,
        sector=sector if sector else "Unknown",
        stock_size=stock_size,
        analysis_date=analysis_date,
        formatted_text=formatted_text if formatted_text else "데이터 없음",
        report_count=report_count,
        news_count=news_count,
        disclosure_count=disclosure_count,
        output_schema=OUTPUT_SCHEMA
    )


def format_research_retry_prompt(
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
    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    return STOCK_RESEARCH_RETRY_PROMPT.format(
        validation_errors=errors_text,
        previous_output=json.dumps(previous_output, ensure_ascii=False, indent=2)
    )


def get_output_schema() -> str:
    """
    Get the output schema string for reference.

    Returns:
        Output schema JSON string
    """
    return OUTPUT_SCHEMA


def validate_research_output(output: dict) -> list[str]:
    """
    Validate stock research output and return list of errors.

    Args:
        output: LLM output dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required top-level fields
    required_fields = ["symbol", "stock_name", "analysis_date", "sector",
                       "overall_sentiment", "confidence", "earnings_outlook",
                       "sector_momentum"]

    for field in required_fields:
        if field not in output:
            errors.append(f"Missing required field: {field}")

    # Check overall_sentiment value
    if output.get("overall_sentiment") not in ["positive", "neutral", "negative"]:
        errors.append("overall_sentiment must be 'positive', 'neutral', or 'negative'")

    # Check confidence range
    confidence = output.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            errors.append("confidence must be a number between 0 and 1")

    # Check earnings_outlook
    earnings = output.get("earnings_outlook", {})
    if earnings:
        if earnings.get("direction") not in ["positive", "neutral", "negative"]:
            errors.append("earnings_outlook.direction must be 'positive', 'neutral', or 'negative'")
        if not earnings.get("summary"):
            errors.append("earnings_outlook.summary is required")

    # Check sector_momentum
    sector_mom = output.get("sector_momentum", {})
    if sector_mom:
        if sector_mom.get("direction") not in ["positive", "neutral", "negative"]:
            errors.append("sector_momentum.direction must be 'positive', 'neutral', or 'negative'")
        if not sector_mom.get("summary"):
            errors.append("sector_momentum.summary is required")

    # Check risk_issues structure
    risk_issues = output.get("risk_issues", [])
    valid_categories = ["regulatory", "market", "operational", "financial", "other"]
    valid_severities = ["high", "medium", "low"]

    for i, risk in enumerate(risk_issues):
        if risk.get("category") not in valid_categories:
            errors.append(f"risk_issues[{i}].category must be one of {valid_categories}")
        if risk.get("severity") not in valid_severities:
            errors.append(f"risk_issues[{i}].severity must be one of {valid_severities}")
        if not risk.get("title"):
            errors.append(f"risk_issues[{i}].title is required")
        if not risk.get("summary"):
            errors.append(f"risk_issues[{i}].summary is required")

    return errors
