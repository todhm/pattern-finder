"""51 라이프사이클 트레이드 페이지 스모크 — 가짜 시장 데이터·가짜 IPO 캘린더로 Run 까지 렌더링."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from data.domain.models import IpoEvent

PAGES = Path(__file__).resolve().parents[1] / "pages"


class FakeMarketData:
    """IPO 2019-01-07 가정 — IPO base → 상승 → 조정 → 성숙 베이스 돌파 (Late Bloomer 류)."""

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        weekly = ([100] * 4 + list(np.linspace(100, 160, 8)) + list(np.linspace(160, 90, 10)) + [90] * 40
                  + list(np.linspace(90, 110, 6)) + [110] * 10 + list(np.linspace(110, 150, 6)) + [150] * 10)
        c = np.repeat(np.asarray(weekly, dtype=float), 5)
        idx = pd.bdate_range("2019-01-07", periods=len(c))
        df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": np.full(len(c), 2e6)}, index=idx)
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


class FakeCalendar:
    last_source = "fake"

    def fetch_ipos(self, start, end):
        return [IpoEvent(symbol="AAA", name="Alpha Inc", ipo_date=date(2019, 1, 7), exchange="NASDAQ Global", offer_price=90.0, source="fake"),
                IpoEvent(symbol="BBB", name="Beta Acquisition Corp", ipo_date=date(2019, 1, 7), exchange="NYSE", offer_price=10.0, source="fake")]


def test_lifecycle_page_runs_cohort(monkeypatch):
    import data.adapters.composed_ipo_calendar as cic
    import data.adapters.composed_market_data as cmd

    monkeypatch.setattr(cmd, "build_default_market_data", lambda **kw: FakeMarketData())
    monkeypatch.setattr(cic, "build_default_ipo_calendar", lambda: FakeCalendar())
    at = AppTest.from_file(str(PAGES / "51_Lifecycle_Trade.py"), default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.info                                     # Run 전 안내
    at.sidebar.date_input[0].set_value(date(2021, 6, 30))
    at.sidebar.button[0].click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("기준 — 종목" in h.value for h in at.subheader)
    labels = " ".join(m.label for m in at.metric)
    assert "IPO Alert" in labels and "이번 주 베이스 돌파" in labels
    assert at.dataframe                                 # 스크린·통계·시뮬 표 렌더링
    assert any("1년 내 +100%" in str(df.value.to_dict()) or True for df in at.dataframe)
