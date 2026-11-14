"""Claude 에게 ``ai_chart_judge.Packet`` 을 보여 주고 ``Judgement``(진입) / ``Manage``(보유 중)를 받는다 (Anthropic SDK, 구조화 출력).

키는 환경변수 ``ANTHROPIC_API_KEY`` (docker compose 의 env_file 로 주입). 모델 기본 ``claude-opus-5-5``.
``ClaudeJudge`` 는 답을 디스크에 캐시한다 — 키는 보낸 내용 전체(시스템 글 + 그림 + 표 + 모델 + effort)의 해시라서
같은 종목·같은 시점·같은 설정을 다시 돌리면 호출·비용 없이 같은 답이 나온다(기간을 늘려 다시 돌려도 앞부분은 재사용).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from strategy.adapters.ai_chart_judge import Judgement, Manage, Packet, ScanResult

DEFAULT_MODEL = "claude-opus-5-5"
MODELS = {  # 표시 이름, 100만 토큰당 (입력, 출력) 달러
    "claude-opus-5-5": ("Claude Opus 5.5", 4.0, 20.0),
    "claude-fable-5-2": ("Claude Fable 5.2", 8.0, 40.0),
    "claude-sonnet-5-5": ("Claude Sonnet 5.5", 2.0, 10.0),
    "claude-haiku-5-5": ("Claude Haiku 5.5", 0.10, 0.50),
}
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-fable-5-2", "claude-sonnet-5-5"}   # 서버 쪽 거절 대체(fallbacks) 지원 모델
EST_TOKENS = (9500, 1500)   # 호출 한 번의 대략적인 (입력, 출력) 토큰 — 비용 어림용

_FRAME = """당신은 차트를 읽는 숙련된 단기 트레이더입니다. 사용자는 개인 학습용 백테스트 도구에서 과거(또는 지금) 한 시점의 차트를 보여 주고, \
그 시점에 아래 패턴으로 롱(매수) 진입을 할지 판단해 달라고 합니다. 당신의 답은 기록되어 실제 뒤 봉으로 정산됩니다. \
같은 차트를 일정 간격으로 계속 다시 보여 주므로, 매번 그 시점에 보이는 것만으로 새로 판단합니다(앞에서 한 답은 기억하지 못합니다).

이 도구가 있는 이유는 같은 패턴을 딱딱한 숫자 규칙으로 옮긴 엔진이 놓치는 맥락을 사람처럼 전체로 읽기 위해서입니다. \
아래 원칙에 나오는 숫자는 감을 잡기 위한 것이지 문턱이 아니니, 조건을 하나씩 대조하지 말고 차트 전체의 이야기를 읽으세요.

[패턴: {title}]
{principles}

지켜야 할 것:
- 롱만 판단합니다. 레버리지는 없습니다. 한 번에 한 포지션, 전액 투입입니다.
- 답은 셋 중 하나입니다.
  (1) 신호 봉이 방금 마감됐다 → enter_long + market (다음 봉 시가에 매수).
  (2) 모양은 다 됐고 방아쇠만 남았다(저항 바로 아래에서 압축 중, 되돌림을 기다리는 중 등) → enter_long + 조건부 주문: \
breakout_close = 어떤 봉이 그 가격 위로 마감하면 그 종가에 매수, limit = 가격이 그 가격까지 내려오면 매수(눌림 매수). \
조건이 안 되면 주문은 그냥 사라지니, 사람이 가격 알림을 걸어 두는 것과 같습니다. 이럴 때 wait 라고만 답하면 다음 판단 전에 나온 신호를 놓칩니다.
  (3) 모양이 아직 덜 됐거나 자리가 아니다 → wait / no_trade.
- 자리가 아닌 시점이 더 많으니 근거가 약한데 억지로 진입을 만들지는 마세요. 하지만 패턴이 실제로 보이면 주저하지 말고 계획을 냅니다 — \
교과서처럼 완벽한 모양만 기다리면 아무 거래도 못 합니다. 손절로 위험이 정해져 있다는 점을 기억하세요.
- enter_long 이면 진입 방식과 손절을 반드시 가격으로 적습니다.
- 다음 판단은 {step}봉 뒤입니다. 조건부 진입(breakout_close·limit)은 그때까지만 유효하고, 그때 차트를 다시 보고 새로 정합니다. trigger_valid_bars 는 {step} 이하로 적으세요.
- 목표가는 차트에 근거가 있는 가격으로 적습니다. 보유 중에도 {step}봉마다 다시 물으니(계속 보유 / 청산 / 손절 올리기), 추세를 길게 탈 계획이면 목표2 를 비워 두어도 됩니다.
- 가격은 자료에 나온 눈금 그대로 씁니다. 기준 봉 뒤의 일은 알 수 없고, 추측해서 쓰지 않습니다.
- 자료가 '종목 X'·D0/D-1·T0/T-3 같은 가려진 이름이면 어떤 종목·날짜인지 맞히려 하지 말고 차트만 보세요.
- 설명은 한국어로, 차트에서 실제로 보이는 것(어느 봉 언저리, 어떤 가격대)을 짚어 구체적으로 씁니다."""

_MANAGE = """당신은 차트를 읽는 숙련된 단기 트레이더입니다. 사용자는 개인 학습용 백테스트 도구에서, 아래 패턴으로 이미 롱 포지션을 들고 있는 한 시점의 차트를 보여 주고 \
포지션을 어떻게 할지 묻습니다. 당신의 답은 기록되어 실제 뒤 봉으로 정산됩니다. 손절가와 목표가는 봉마다 기계적으로 적용되고 있으며, 당신은 {step}봉마다 한 번씩 \
'계속 보유(hold)' 또는 '다음 봉 시가에 전량 청산(exit)'을 정하고, 필요하면 손절을 올리거나 목표를 바꿉니다.

[패턴: {title}]
{principles}

지켜야 할 것:
- 차트에 entry(진입가)·stop(현재 손절)·target(현재 목표) 선이 그려져 있고 요약의 '보유 중'에 숫자가 있습니다.
- 패턴의 청산 원칙에 맞는 신호가 차트에 실제로 보일 때만 exit 합니다. 작은 흔들림마다 나가면 추세를 못 탑니다. 반대로 패턴이 깨졌으면 손절가까지 기다리지 않고 나갑니다.
- 손절은 올리기만 합니다(현재 손절보다 높고 현재가보다 낮게). 근거(새 지지·이동평균선·본전) 없이 바짝 붙이지 않습니다.
- 가격은 자료에 나온 눈금 그대로 씁니다. 기준 봉 뒤의 일은 알 수 없습니다.
- 설명은 한국어로 구체적으로 씁니다."""


_SCAN = """당신은 차트를 읽는 숙련된 단기 트레이더입니다. 사용자는 개인 학습용 도구에서 한 구간의 차트를 통째로 보여 주고, 그 안에서 아래 패턴이 나온 자리를 \
전부 찾아 표시해 달라고 합니다. 찾은 자리는 차트에 그려지고, 신호 봉 다음 봉 시가에 샀다고 보고 손익을 계산합니다.

[패턴: {title}]
{principles}

지켜야 할 것:
- 롱만 찾습니다. 원칙에 나오는 숫자는 감을 잡기 위한 것이지 문턱이 아닙니다 — 차트 전체의 이야기를 읽고, 사람이 "여기가 그 자리다"라고 할 곳을 고르세요.
- 신호 봉은 패턴의 진입 조건이 완성된 봉입니다(그 봉이 마감된 뒤 다음 봉 시가에 삽니다). 봉 이름은 봉 표의 bar 열 글자 그대로 적습니다.
- 당신은 뒤 봉까지 보고 있지만, 자리를 고르는 기준은 "신호 봉이 마감된 그 순간까지의 모양"이어야 합니다. 뒤에 올랐다는 이유로 고르거나 뒤에 내렸다는 이유로 빼지 마세요. \
모양이 패턴에 맞았는데 실패한 자리도 반드시 포함합니다. 손절·목표도 신호 봉까지의 차트로 정할 수 있는 가격만 씁니다.
- exit_bar 는 패턴의 청산 원칙에 해당하는 신호가 처음 나온 봉입니다. 가장 비싼 봉을 고르는 것이 아닙니다.
- 찾을 구간 밖(앞쪽 맥락 구간)의 봉은 신호 봉으로 쓰지 않습니다. 맥락 구간은 추세와 구조를 읽는 데만 씁니다.
- 자리가 없으면 빈 목록이 정답입니다. 억지로 만들지 마세요. 반대로 애매하지만 패턴으로 볼 만한 자리는 낮은 confidence 로 포함합니다.
- 가격은 자료에 나온 눈금 그대로 씁니다. 자료가 '종목 X'·D0/D-1·T0/T-3 같은 가려진 이름이면 종목·날짜를 맞히려 하지 마세요.
- 설명은 한국어로 구체적으로 씁니다."""


def entry_system(title: str, principles: str, step: int) -> str:
    return _FRAME.format(title=title, principles=principles.strip(), step=step)


def manage_system(title: str, principles: str, step: int) -> str:
    return _MANAGE.format(title=title, principles=principles.strip(), step=step)


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def estimate_cost(model: str, calls: int) -> float:
    _, pin, pout = MODELS.get(model, (model, 0.0, 0.0))
    return calls * (EST_TOKENS[0] * pin + EST_TOKENS[1] * pout) / 1e6


def _user_content(p: Packet, question: str, note: str = "") -> list[dict]:
    img = lambda b: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(b).decode("ascii")}}  # noqa: E731
    parts: list[dict] = []
    if p.context_png:
        parts += [{"type": "text", "text": "그림 1 — 큰 그림 (기준 봉 전에 완성된 상위 주기 봉. 가격 눈금은 아래 차트와 별개이니 모양만 보세요):"}, img(p.context_png)]
    parts += [{"type": "text", "text": f"그림 {2 if p.context_png else 1} — {p.interval} 봉 {p.n_bars}개 (맨 오른쪽 봉이 방금 마감된 기준 봉 {p.labels[-1]}):"}, img(p.chart_png)]
    scale_note = ("가격은 지수화되어 있습니다(차트 첫 봉 종가 = 100). 거래량은 중앙값 = 1 인 상대값입니다. 답의 가격도 이 눈금으로 쓰세요."
                  if p.blind else "가격은 실제 가격, 시각은 뉴욕 시각입니다.")
    text = (f"종목: {p.name}\n{scale_note}\n\n요약 (모두 기준 봉까지의 값):\n{json.dumps(p.facts, ensure_ascii=False, indent=1)}\n\n"
            f"최근 봉 표 (마지막 줄이 기준 봉):\n{p.bars_csv}\n\n"
            + (f"사용자 메모: {note.strip()}\n\n" if note and note.strip() else "") + question)
    parts.append({"type": "text", "text": text})
    return parts


class ClaudeJudge:
    """진입·관리 판단을 Claude 에게 묻는다. 디스크 캐시 + 여러 장 병렬."""

    def __init__(self, *, title: str, principles: str, step: int, model: str = DEFAULT_MODEL, effort: str = "medium", note: str = "",
                 cache_dir: str | Path | None = None, workers: int = 3, client: anthropic.Anthropic | None = None):
        self.model, self.effort, self.note, self.workers = model, effort, note, max(1, workers)
        self.sys_entry = entry_system(title, principles, step)
        self.sys_manage = manage_system(title, principles, step)
        self.sys_scan = _SCAN.format(title=title, principles=principles.strip())
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=6)      # 병렬 호출이 분당 한도(429)에 걸리면 SDK 가 물러났다 다시 시도
        return self._client

    # -- 캐시 ---------------------------------------------------------------
    def _key(self, system: str, content: list[dict], schema_name: str) -> str:
        h = hashlib.sha256()
        for part in (self.model, self.effort, schema_name, system, json.dumps(content, sort_keys=True, ensure_ascii=False)):
            h.update(part.encode("utf-8")); h.update(b"\x00")
        return h.hexdigest()

    def _ask(self, p: Packet, system: str, schema, question: str):
        content = _user_content(p, question, self.note)
        path = self.cache_dir / f"{self._key(system, content, schema.__name__)}.json" if self.cache_dir else None
        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return schema(**saved["answer"]), {**saved["usage"], "cost_usd": 0.0, "cached": True, "paid_usd": saved["usage"].get("cost_usd", 0.0)}
        kwargs: dict = dict(model=self.model, max_tokens=16000, system=system, messages=[{"role": "user", "content": content}],
                            output_format=schema, output_config={"effort": self.effort})
        if self.model in _FALLBACK_MODELS:     # 안전 분류기가 요청을 거절하면 서버가 권장 대체 모델로 같은 요청을 다시 돌린다
            resp = self.client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.parse(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError("모델이 요청을 거절했습니다" + (f" ({resp.stop_details.category})" if getattr(resp, "stop_details", None) else ""))
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise RuntimeError(f"답이 완성되지 않았습니다 (stop_reason={resp.stop_reason})")
        u = resp.usage
        _, pin, pout = MODELS.get(self.model, (self.model, 0.0, 0.0))
        usage = {"model": resp.model, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cost_usd": round((u.input_tokens * pin + u.output_tokens * pout) / 1e6, 5), "request_id": getattr(resp, "_request_id", None)}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"answer": resp.parsed_output.model_dump(), "usage": usage, "as_of": str(p.as_of)}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        return resp.parsed_output, {**usage, "cached": False}

    # -- 엔진이 부르는 두 함수 -------------------------------------------------
    def ask_entry(self, packets: list[Packet]) -> list[tuple[Judgement, dict]]:
        q = "기준 봉이 마감된 지금, 이 패턴으로 롱 진입을 어떻게 할지 판단해 주세요."
        if len(packets) == 1 or self.workers == 1:
            return [self._ask(p, self.sys_entry, Judgement, q) for p in packets]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(packets))) as ex:
            return list(ex.map(lambda p: self._ask(p, self.sys_entry, Judgement, q), packets))

    def ask_manage(self, packet: Packet) -> tuple[Manage, dict]:
        return self._ask(packet, self.sys_manage, Manage, "기준 봉이 마감된 지금, 들고 있는 롱 포지션을 어떻게 할지 정해 주세요.")

    def ask_scan(self, packet: Packet, first_label: str, last_label: str) -> tuple[ScanResult, dict]:
        q = (f"찾을 구간은 봉 '{first_label}' 부터 '{last_label}' 까지입니다(그 앞의 봉은 맥락용). 이 구간에서 패턴이 나온 자리를 전부 찾아 주세요. "
             "봉 표에는 차트의 모든 봉이 들어 있습니다.")
        return self._ask(packet, self.sys_scan, ScanResult, q)
