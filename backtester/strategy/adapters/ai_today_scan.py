"""오늘 진입할 종목 찾기 — 여러 종목의 '마지막 완성 봉'까지의 차트를 AI 에게 보여 주고 진입 판단을 받는다 (LLM 을 모르는 순수 부분).

두 단계: ① 1차(싼 모델)로 전 종목을 판정 → ② 진입(enter_long)이라고 한 종목 중 확신 높은 순으로 `max_confirm` 개를 2차(상위 모델)로 다시 판정.
둘 다 숫자 조건이 아니라 같은 자료 묶음(`ai_chart_judge.build_packet`: 일봉·주봉 그림 + 봉 표 + 요약)과 같은 패턴 원칙으로 판단한다.
호출은 넘겨받은 `ask(packets) -> [(Judgement, usage)]` 가 한다(`ClaudeJudge.ask_entry`). 페이지 `pages/69_Evidence_AI_Today.py`.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from strategy.adapters.ai_chart_judge import NY, ChartSpec, Judgement, Packet, build_packet


def completed_bars(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """장중이면 오늘의 덜 끝난 일봉을 뺀다(뉴욕 16:10 전). 판단은 항상 마감된 봉까지로 한다."""
    if df is None or df.empty:
        return df
    now = pd.Timestamp.now(tz=NY) if now is None else (pd.Timestamp(now).tz_convert(NY) if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz=NY))
    last = pd.Timestamp(df.index[-1])
    last_day = (last.tz_convert(NY) if last.tzinfo else last).date()
    if last_day == now.date() and (now.hour, now.minute) < (16, 10):
        return df.iloc[:-1]
    return df


def _row(sym: str, pk: Packet, j: Judgement, u: dict) -> dict:
    return {"symbol": sym, "as_of": str(pd.Timestamp(pk.as_of).date()), "close": pk.last / pk.scale, "judgement": j.model_dump(), "usage": u}


def scan_today(prices: dict[str, pd.DataFrame], spec: ChartSpec, ask_triage: Callable[[list[Packet]], list[tuple[Judgement, dict]]],
               ask_confirm: Optional[Callable[[list[Packet]], list[tuple[Judgement, dict]]]] = None, *, max_confirm: int = 15, min_bars: int = 400,
               chunk: int = 8, blind: bool = False, progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """{triage: [...], confirmed: [...], skipped: [...], cost} — confirmed 는 2차 판정까지 받은 종목(2차 결정 포함)."""
    syms = [s for s, d in prices.items() if d is not None and len(d) >= min_bars]
    skipped = sorted(set(prices) - set(syms))
    triage: list[dict] = []; packets: dict[str, Packet] = {}
    cost = 0.0; calls = 0
    for a in range(0, len(syms), chunk):
        batch = syms[a: a + chunk]
        if progress:
            progress(0.85 * a / max(1, len(syms)), f"1차 판정 {a}/{len(syms)} · 진입 후보 {sum(r['judgement']['decision'] == 'enter_long' for r in triage)}개")
        pks = []
        for s in batch:
            try:
                d = prices[s]
                pks.append((s, build_packet(d, d.index[-1], spec=spec, ticker=s, blind=blind)))
            except Exception:  # noqa: BLE001
                skipped.append(s)
        if not pks:
            continue
        for (s, pk), (j, u) in zip(pks, ask_triage([p for _, p in pks])):
            jr = j.to_real(pk.scale, text_ref=pk.last)
            triage.append(_row(s, pk, jr, u)); packets[s] = pk
            cost += u.get("cost_usd", 0.0); calls += 0 if u.get("cached") else 1
    cands = sorted((r for r in triage if r["judgement"]["decision"] == "enter_long"), key=lambda r: -r["judgement"]["confidence"])
    confirmed: list[dict] = []
    if ask_confirm is not None and cands:
        top = cands[:max_confirm]
        for a in range(0, len(top), chunk):
            batch = top[a: a + chunk]
            if progress:
                progress(0.85 + 0.15 * a / len(top), f"2차 판정 {a}/{len(top)}")
            pks = [packets[r["symbol"]] for r in batch]
            for r, pk, (j, u) in zip(batch, pks, ask_confirm(pks)):
                jr = j.to_real(pk.scale, text_ref=pk.last)
                confirmed.append({**_row(r["symbol"], pk, jr, u), "triage": r["judgement"]})
                cost += u.get("cost_usd", 0.0); calls += 0 if u.get("cached") else 1
    elif ask_confirm is None:
        confirmed = [{**r, "triage": r["judgement"]} for r in cands[:max_confirm]]
    if progress:
        progress(1.0, "정리 중")
    return {"triage": triage, "confirmed": confirmed, "skipped": skipped, "cost_usd": round(cost, 4), "calls": calls,
            "n_candidates": len(cands), "as_of": triage[0]["as_of"] if triage else None}


def plan(j: dict, close: float) -> dict:
    """판정 dict → 진입 계획 요약(진입 기준가·손절 %·목표 %·손익비)."""
    ref = j.get("entry_trigger") if j.get("entry_type") in ("breakout_close", "limit") and j.get("entry_trigger") else close
    stop, t1, t2 = j.get("stop"), j.get("target1"), j.get("target2")
    risk = (ref - stop) if stop else None
    return {"entry_ref": ref, "entry_how": {"market": "다음 봉 시가", "breakout_close": f"{ref:.2f} 위 마감 시", "limit": f"{ref:.2f} 까지 내려오면", "none": "—"}[j.get("entry_type", "none")],
            "stop_pct": (stop / ref - 1) if stop else None, "t1_pct": (t1 / ref - 1) if t1 else None, "t2_pct": (t2 / ref - 1) if t2 else None,
            "rr": ((t1 - ref) / risk) if (t1 and risk and risk > 0) else None}
