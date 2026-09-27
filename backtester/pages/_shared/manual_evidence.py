"""수동 체크리스트 항목의 데이터 보조 — 티커+기준일이 있으면 Alpha Vantage(분기
서프라이즈·재무상태표)와 EODHD 뉴스를 point-in-time 으로 불러와 2-7·2-8·2-9
옆에 근거와 자동 판정을 띄운다. 최종 체크는 여전히 사람이 한다.

반환 형식: ``{item_key: {"verdict": bool|None, "text": markdown}}``
(item_key 는 minervini_manual_checks 의 m2_7 / m2_8 / m2_9).
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from pages._shared.growth_sources import build_growth_adapters, fetch_growth_snapshot
from strategy.adapters.minervini_screener import evaluate_balance, evaluate_surprise

_GUIDANCE_KW = ("guidance", "outlook", "raises", "raised", "lowers", "lowered",
                "cuts", "forecast", "sees fy", "sees q")


def _surprise_text(r) -> str:
    rows = " · ".join(
        f"{x['분기말'][:7]}(발표 {x['발표일']}) ${x['실제']:.2f} vs ${x['추정']:.2f} "
        f"{'✅' if x['비트'] else '❌'}"
        + (f" {x['서프라이즈']:+.0%}" if x["서프라이즈"] is not None else "")
        for x in r.rows
    )
    head = ("최신 분기 비트" if r.latest_beat else "최신 분기 미스") + f" · 연속 비트 {r.streak}회"
    return f"**{head}** — {rows}"


def _balance_text(b) -> str:
    pct = lambda v: "—" if v is None else f"{v:+.0%}"
    parts = [f"최신 분기 {b.fiscal_date} (발표일 ≤ 기준일)",
             f"매출 YoY {pct(b.revenue_yoy)}"]
    parts.append("재고 없음(서비스업)" if b.inventory_absent
                 else f"재고 YoY {pct(b.inventory_yoy)}"
                 + (" ❌ 매출보다 25%p↑ 초과 → 적색경보" if b.inventory_flag else ""))
    parts.append(f"매출채권 YoY {pct(b.receivables_yoy)}"
                 + (" ⚠️ 매출 증가율 초과" if b.receivables_flag else ""))
    return " · ".join(parts)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_manual_evidence(ticker: str, as_of: date) -> dict[str, dict]:
    out: dict[str, dict] = {}
    adapters = build_growth_adapters()
    snap = fetch_growth_snapshot(ticker, adapters)
    if snap is None:
        return out

    # 2-7 어닝 서프라이즈 — 스냅샷의 estimatedEPS (추가 호출 없음)
    sr = evaluate_surprise(snap.quarters, as_of)
    out["m2_7"] = {
        "verdict": sr.latest_beat if sr.data_available else None,
        "text": _surprise_text(sr) if sr.data_available
        else "추정치 있는 분기 없음 (소스에 estimatedEPS 없음)",
    }

    # 2-9 재고·매출채권 — BALANCE_SHEET 1콜 (캐시)
    bal_adapter = next((a for a in adapters if hasattr(a, "fetch_balance_sheet")), None)
    if bal_adapter is not None:
        try:
            balances = bal_adapter.fetch_balance_sheet(ticker)
        except Exception:
            balances = []
        br = evaluate_balance(balances, snap.quarters, as_of)
        out["m2_9"] = {
            "verdict": br.passed,
            "text": _balance_text(br) if br.data_available
            else "재무상태표 분기 5개 미만 — 미확인",
        }

    # 2-8 가이던스 — 뉴스 헤드라인 키워드 필터 (EODHD, 기준일 −120일 ~ 기준일)
    try:
        from data.adapters.eodhd_news import EODHDNewsAdapter
        items = EODHDNewsAdapter().fetch_news(ticker, as_of - timedelta(days=120), as_of)
        hits = [it for it in items if any(k in it.title.lower() for k in _GUIDANCE_KW)]
        if hits:
            lines = " · ".join(f"{it.date} {it.title[:90]}" for it in hits[-5:])
            text = f"가이던스 관련 헤드라인 {len(hits)}건 (최근 5건) — {lines}"
        else:
            text = (f"뉴스 {len(items)}건 중 가이던스 키워드 헤드라인 없음 — "
                    "오래된 기준일은 커버리지가 얇다. 링크(EDGAR 8-K)로 확인")
        out["m2_8"] = {"verdict": None, "text": text}
    except Exception as e:  # 키 없음 등
        out["m2_8"] = {"verdict": None, "text": f"뉴스 소스 없음 ({type(e).__name__}) — 링크로 확인"}
    return out


def evidence_markdown(ev: dict | None) -> str:
    """항목 하나의 evidence dict → 캡션용 마크다운 한 줄."""
    if not ev:
        return ""
    mark = {True: "✅ 자동 판정 통과", False: "❌ 자동 판정 실패", None: "ℹ️ 데이터"}[ev.get("verdict")]
    return f"📊 {mark} — {ev.get('text', '')}"
