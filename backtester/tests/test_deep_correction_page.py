"""49 TQQQ 급락 회피 · 50 SOXL 월말 SMA 페이지 스모크 — 가짜 시장 데이터로 렌더링."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_leveraged_etf_page import FakeMarketData

PAGES = Path(__file__).resolve().parents[1] / "pages"


def _run(monkeypatch, page: str) -> AppTest:
    import data.adapters.composed_market_data as cmd

    monkeypatch.setattr(cmd, "build_default_market_data", lambda **kw: FakeMarketData())
    at = AppTest.from_file(str(PAGES / page), default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("오늘의 판단" in h.value for h in at.subheader)
    assert any("구간별 결과" in h.value for h in at.subheader)
    assert any("시작 시점별" in h.value for h in at.subheader)
    return at


def test_tqqq_deep_correction_page(monkeypatch):
    at = _run(monkeypatch, "49_TQQQ_Deep_Correction.py")
    banners = [b.value for b in list(at.success) + list(at.error) + list(at.warning)]
    assert any(any(a in b for a in ("보유", "매도", "재진입 매수", "현금 유지")) for b in banners)
    assert any("이탈선" in m.label for m in at.metric)
    assert at.sidebar.selectbox[0].value == "TQQQ"


def test_soxl_monthly_sma_page(monkeypatch):
    at = _run(monkeypatch, "50_SOXL_Monthly_SMA.py")
    assert any("오늘이 월말이라면" in m.label for m in at.metric)
    assert at.sidebar.selectbox[0].value == "SOXL" and at.sidebar.selectbox[1].value == "SOXX"
    assert any("월말 SMA" in b.value for b in list(at.info))
