"""48 초단기 vs 장기보유 페이지 스모크 — 가짜 5분봉·일봉으로 렌더링 (네트워크 없음)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "pages" / "48_Intraday_vs_BuyHold.py"


class FakeMarketData:
    def fetch_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash(symbol)) % 997)
        if interval == "1d":
            idx = pd.bdate_range(start, end)
            close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(idx)))), index=idx)
            opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, len(idx)))
            return pd.DataFrame({"Open": opn, "High": np.maximum(opn, close) * 1.005, "Low": np.minimum(opn, close) * 0.995,
                                 "Close": close, "Volume": 1e6})
        days = pd.bdate_range(start, end)
        frames = []
        px = 100.0
        for d in days:
            idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", tz="America/New_York")
            r = rng.normal(0, 0.0015, 78)
            c = px * np.exp(np.cumsum(r)); o = np.concatenate([[px], c[:-1]])
            frames.append(pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.0005, "Low": np.minimum(o, c) * 0.9995,
                                        "Close": c, "Volume": 1e4}, index=idx))
            px = float(c[-1])
        return pd.concat(frames)


def test_page_renders_with_fake_data(monkeypatch):
    import data.adapters.composed_market_data as cmd

    monkeypatch.setattr(cmd, "build_default_market_data", lambda **kw: FakeMarketData())
    at = AppTest.from_file(str(PAGE), default_timeout=180)
    at.session_state  # noqa: B018
    # 기간을 짧게 (기본 2016~ 은 가짜 데이터라도 오래 걸림)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.tabs) == 3
    assert any("ORB 최종" in m.label for m in at.metric)
