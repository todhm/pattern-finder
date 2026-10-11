"""47 레버리지 ETF 랩 페이지 스모크 — 가짜 시장 데이터로 렌더링까지 (네트워크 없음)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from strategy.adapters.leveraged_etf_data import ETF_SPECS, load_lab_prices

PAGE = Path(__file__).resolve().parents[1] / "pages" / "47_Leveraged_ETF_Lab.py"


class FakeMarketData:
    """심볼별 결정적 랜덤 워크. ^IRX 는 2% 고정."""

    def fetch_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        idx = pd.bdate_range(start, end)
        if symbol == "^IRX":
            close = pd.Series(2.0, index=idx)
        else:
            rng = np.random.default_rng(abs(hash(symbol)) % 1000)
            close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(idx)))), index=idx)
        return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6})


def test_load_lab_prices_splices_synthetic_before_inception():
    md = FakeMarketData()
    p = load_lab_prices(md, "TQQQ", "QQQ", date(2005, 1, 3), date(2012, 12, 31), synthetic=True)
    assert p.synthetic_until == ETF_SPECS["TQQQ"].inception
    assert p.lev_close.index[0] < pd.Timestamp("2005-01-03")          # 워밍업 포함
    assert p.lev_close.index[-1] >= pd.Timestamp("2012-12-28")
    assert p.cash_rate is not None and abs(float(p.cash_rate.iloc[-1]) - 0.02) < 1e-9
    assert "합성" in p.source_note and p.lev_close.isna().sum() == 0
    real = load_lab_prices(md, "TQQQ", "QQQ", date(2015, 1, 2), date(2016, 1, 1), synthetic=True)
    assert real.synthetic_until is None and real.source_note == "TQQQ 실데이터"


def test_page_renders_with_fake_data(monkeypatch):
    import data.adapters.composed_market_data as cmd

    monkeypatch.setattr(cmd, "build_default_market_data", lambda **kw: FakeMarketData())
    at = AppTest.from_file(str(PAGE), default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("비교 요약" in h.value for h in at.subheader)
    assert len(at.tabs) >= 6                                               # ①~⑤ + 민감도
    assert any("장기보유" in m.label or "SMA" in m.label for m in at.metric)
