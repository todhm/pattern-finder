"""피셔 15 포인트 문서 판독 — 기준일 전에 제출된 공시(10-K·위임장·8-K 목록)와 수치표를 Claude 에게 주고 포인트별 점수를 받는다.

체크리스트 `docs/strategy_notes/피셔_위대한기업_체크리스트_2026-11.md` §3·§8·§10. 답은 내용 해시로 디스크에 캐시한다(같은 자료·같은 모델이면 재호출 없음).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field

MODELS = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-fable-5-2": (8.0, 40.0)}
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-fable-5-2", "claude-sonnet-5-5"}


class PointScore(BaseModel):
    point: int = Field(description="포인트 번호 1~15")
    score: Optional[int] = Field(default=None, description="0~3. 자료로 판단할 수 없으면 null(모름)")
    confidence: Literal["low", "medium", "high"] = Field(description="이 판정의 확신")
    evidence: str = Field(description="판정 근거 — 자료의 어느 부분·어느 숫자인지 (한국어 2~4문장). 자료에 없는 사실을 쓰지 않는다")


class FisherJudgement(BaseModel):
    points: list[PointScore] = Field(description="포인트 1~15 를 빠짐없이, 번호 순서대로")
    promotional: bool = Field(description="'선전 단계 회사'(검증할 영업 이력이 부족)인가")
    tone_only: bool = Field(description="좋아 보이는 근거가 대부분 회사의 홍보성 문구뿐이고 숫자·사실이 뒷받침하지 않는가")
    temporary_trouble: str = Field(description="책 5장의 '사기 좋은 때'에 해당하는 일시적 문제(신제품·신공장 초기 비용, 일회성 악재)가 자료에 보이면 적는다. 없으면 빈 글")
    red_flags: list[str] = Field(default_factory=list, description="책의 기준으로 걸리는 점 (최대 5개, 한국어)")
    strengths: list[str] = Field(default_factory=list, description="책의 기준으로 뛰어난 점 (최대 5개, 한국어)")
    verdict: Literal["buy", "watch", "pass"] = Field(description="피셔라면: buy = 살 만하다 / watch = 더 확인이 필요 / pass = 사지 않는다")
    summary: str = Field(description="왜 그런 결론인지 초보자도 알 수 있게 한 문단 (한국어)")


SYSTEM = """당신은 필립 피셔의 『위대한 기업에 투자하라』(Common Stocks and Uncommon Profits, 1958)의 방법으로 회사를 평가하는 분석가입니다.
사용자는 개인 학습용 검증 도구에서, 어떤 **기준일**에 구할 수 있었던 자료만 보여 주고 그 회사를 책의 15가지 포인트로 하나씩 판정해 달라고 합니다.
당신의 판정은 기록되어 기준일 뒤 1년의 실제 주가와 대조됩니다.

가장 중요한 규칙 — 시점:
- 판단 근거는 **주어진 자료뿐**입니다. 자료는 모두 기준일 전에 제출·발표된 것입니다.
- 당신은 이 회사가 기준일 뒤에 어떻게 됐는지 알고 있을 수 있습니다. 그 지식은 절대 쓰지 않습니다. 회사 이름이 가려져 있으면 어떤 회사인지 맞히려 하지 마세요.
- 자료에 없는 사실을 근거로 쓰지 않습니다. 자료로 알 수 없는 포인트는 추측하지 말고 score 를 null(모름)로 둡니다. 모름은 정직한 답입니다.

책의 태도:
- 피셔는 사람을 만나 듣는 "사실 수집(scuttlebutt)"으로 판단했습니다. 여기 있는 것은 회사가 스스로 쓴 문서뿐이라, 회사에 유리하게 쓰였다는 점을 감안해 읽습니다.
- 연차보고서의 "분위기"에 넘어가지 않습니다(하지 말 것 3번). 형용사·포부가 아니라 숫자와 구체적 사실만 근거로 칩니다.
- 이익 대비 주가가 높다는 이유로 깎지 않습니다(하지 말 것 4번). 주가·밸류에이션은 판정 대상이 아닙니다.
- 포인트 15(정직성)는 거부권입니다. 다른 것이 아무리 좋아도 의심할 근거가 있으면 0점입니다. 근거가 없으면 2점(문제 없음)이며 3점은 쓰지 않습니다.

점수: 0 = 반대 증거가 있다 / 1 = 약하거나 평범 / 2 = 충족 / 3 = 뚜렷이 탁월 / null = 자료로 판단 불가.

15가지 포인트와 판정 기준:
1. 적어도 몇 년 동안 매출을 크게 늘릴 시장 잠재력이 있는 제품·서비스가 있는가? — 여러 해 묶음으로 본다. 인수가 아니라 자기 제품의 성장인지, 시장이 아직 넓은지, 고객 한두 곳에 매달리지 않는지.
   0 매출 정체·감소 / 1 경제 성장 수준 / 2 3·5년 연 10% 안팎 이상이고 수요 근거가 구체적 / 3 연 15~20% 이상이 여러 해 이어졌고 시장이 아직 넓다.
2. 지금 제품의 성장 여력이 다했을 때 매출을 더 키울 신제품·신공정을 계속 개발하려는 경영진의 의지가 있는가? — 기존 사업과 연관된 분야의 연구가 좋다.
   0 단일 제품, 다음 계획 없음 / 1 말뿐 / 2 연관 분야 신제품이 실제 매출이 되고 있음 / 3 신제품이 반복해서 성장의 축이 된 이력.
3. 회사 규모에 비해 연구개발 노력이 얼마나 효과적인가? — 지출 비율이 아니라 쓴 돈이 매출·이익으로 돌아오는지. 불황에도 연구를 유지하는지. R&D 가 없는 업종이면 null.
4. 평균 이상의 영업 조직이 있는가? — 재구매·갱신·유지율, 유통망, 판매비 대비 매출. 문서로는 알기 어려우니 근거가 약하면 1 또는 null.
5. 영업이익률이 쓸 만한가? — 업계 안에서, 여러 해로. 성장을 위해 일부러 낮춘 젊은 회사는 예외. 한계 기업은 낮게.
6. 이익률을 지키거나 높이려고 무엇을 하고 있는가? — 가격 인상만이 아니라 효율·규모·제품 구성 개선. 최근 3년 추세.
7. 노사·인사 관계가 뛰어난가? — 이직·참여도·노조·감원·파업. 인적 자본 서술이 상투적이면 1 또는 null.
8. 임원들 사이의 관계가 뛰어난가? — 임원 교체 빈도(8-K 5.02 목록), 내부 승진, 보수 구조.
9. 경영진에 깊이가 있는가? — 1인 의존인지, 부문 책임자·승계가 있는지.
10. 원가 분석과 회계 관리가 얼마나 좋은가? — 내부통제의 중요한 취약점, 재작성, 감사인 교체. 문제 없으면 대개 2.
11. 그 업종 특유의, 경쟁사보다 뛰어난 면이 있는가? — 특허에만 기대면 약점. 노하우·규모·서비스·전환 비용.
12. 이익에 대해 단기적 시각인가 장기적 시각인가? — 재투자, 단기 실적을 희생한 장기 투자.
13. 가까운 장래에 증자가 필요해 기존 주주의 몫이 크게 깎이지 않겠는가? — 현금·차입 여력, 주식 수 추세, 주식보상.
14. 일이 잘될 때만 터놓고 말하고 문제가 생기면 입을 닫는가? — 나쁜 소식을 구체적 숫자로 설명하는지.
15. 경영진의 정직성에 의문의 여지가 없는가? — 특수관계자 거래, 과도한 보수, 증권 사기·SEC 조사.

'수치 요약'의 숫자는 도구가 기준일까지 발표된 분기 재무제표로 계산한 것입니다(섹터 내 순위 포함). 포인트 1·3·5·6·12·13 은 이 숫자와 문서를 함께 보고 판정합니다.
설명은 한국어로 씁니다."""


class FisherJudge:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", cache_dir: str | Path | None = None, client: anthropic.Anthropic | None = None):
        self.model, self.effort = model, effort
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=6)
        return self._client

    def judge(self, dossier_text: str) -> tuple[FisherJudgement, dict]:
        key = hashlib.sha256("\x00".join((self.model, self.effort, SYSTEM, dossier_text)).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return FisherJudgement(**saved["answer"]), {**saved["usage"], "cached": True}
        kwargs: dict = dict(model=self.model, max_tokens=16000, system=SYSTEM, output_format=FisherJudgement, output_config={"effort": self.effort},
                            messages=[{"role": "user", "content": dossier_text + "\n\n위 자료만으로 15가지 포인트를 하나씩 판정해 주세요."}])
        if self.model in _FALLBACK_MODELS:     # 안전 분류기가 거절하면 서버가 권장 대체 모델로 다시 돌린다
            resp = self.client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.parse(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError("모델이 요청을 거절했습니다")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise RuntimeError(f"답이 완성되지 않았습니다 (stop_reason={resp.stop_reason})")
        pin, pout = MODELS.get(self.model, (0.0, 0.0))
        usage = {"model": resp.model, "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                 "cost_usd": round((resp.usage.input_tokens * pin + resp.usage.output_tokens * pout) / 1e6, 4)}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"answer": resp.parsed_output.model_dump(), "usage": usage}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        return resp.parsed_output, {**usage, "cached": False}
