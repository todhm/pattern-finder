"""AI 차트 판정: 자료 묶음에 뒤 봉이 새지 않는지, 블라인드 눈금 변환, 체결·정산 규칙, 기간 백테스트(가짜 판정기), 페이지 스모크."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.adapters.ai_chart_judge import ChartSpec, Judgement, Level, Manage, ScanResult, Signal, build_packet, settle
from strategy.adapters.ai_pattern_backtest import BtConfig, checkpoints, run_backtest, run_scan
from strategy.adapters.ai_patterns import PATTERNS

NY = "America/New_York"


def _bars(days: int = 4, seed: int = 0, freq: str = "15min") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([])
    for d in pd.bdate_range("2024-03-04", periods=days):
        idx = idx.append(pd.date_range(d + pd.Timedelta(hours=4), d + pd.Timedelta(hours=19, minutes=45), freq=freq, tz=NY))
    c = 400 + np.cumsum(rng.normal(0, 0.4, len(idx)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.2, "Low": np.minimum(o, c) - 0.2, "Close": c,
                         "Volume": rng.integers(1_000, 50_000, len(idx)).astype(float)}, index=idx)


def _daily(n: int = 120, end: str = "2024-03-07") -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n)
    c = np.linspace(350, 410, n) + np.sin(np.arange(n) / 5) * 4
    return pd.DataFrame({"Open": c - 1, "High": c + 2, "Low": c - 2, "Close": c, "Volume": 1e6}, index=idx)


def _j(**kw) -> Judgement:
    base = dict(decision="enter_long", setup="t", confidence=60, entry_type="market", stop=99.0, stop_basis="close", target1=102.0, target2=None,
                market_read="", structure_read="", indicator_read="", volume_read="", risks="", wait_for="", summary="")
    base.update(kw)
    return Judgement(**base)


SKIP = _j(decision="wait", entry_type="none", stop=None, target1=None)


# ---------------------------------------------------------------------------
# 자료 묶음
# ---------------------------------------------------------------------------
def test_packet_ignores_future_bars():
    bars = _bars()
    as_of = bars.index[150]
    a = build_packet(bars, as_of, context=_daily(), ticker="QQQ", blind=True)
    mutated = bars.copy()
    mutated.iloc[151:] *= 3.0                               # 뒤 봉을 아무리 바꿔도
    b = build_packet(mutated, as_of, context=_daily(), ticker="QQQ", blind=True)
    assert a.bars_csv == b.bars_csv and a.facts == b.facts and a.chart_png == b.chart_png and a.context_png == b.context_png
    assert a.as_of == as_of and a.labels[-1].startswith("D0 ")
    assert a.chart_png[:8] == b"\x89PNG\r\n\x1a\n" and a.context_png


def test_context_excludes_decision_day():
    bars = _bars()
    as_of = bars.index[150]                                 # 2024-03-06
    d = _daily()
    spiked = d.copy(); spiked.loc[spiked.index >= "2024-03-06", ["Open", "High", "Low", "Close"]] *= 5
    a = build_packet(bars, as_of, context=d, blind=False, ticker="QQQ")
    b = build_packet(bars, as_of, context=spiked, blind=False, ticker="QQQ")
    assert a.context_png == b.context_png


def test_blind_hides_identity_and_roundtrips_prices():
    bars = _bars()
    as_of = bars.index[150]
    p = build_packet(bars, as_of, ticker="QQQ", blind=True)
    text = p.bars_csv + str(p.facts) + p.name
    assert "QQQ" not in text and "2024" not in text and "03-0" not in text
    assert p.scale != 1.0 and abs(p.facts["기준 봉 종가"] / p.scale - bars.loc[as_of, "Close"]) < 0.01 / p.scale
    j = _j(stop=99.0, target1=102.0, entry_type="breakout_close", entry_trigger=100.5, trigger_valid_bars=4,
           levels=[Level(kind="trend_resistance", price=100.5, price_start=101.0, bars_ago_start=10, note="")]).to_real(p.scale)
    assert j.stop == pytest.approx(99.0 / p.scale) and j.levels[0].price_start == pytest.approx(101.0 / p.scale)
    t = _j(summary="저항 100.5~101.2 를 3번 두드림, 거래량 2.5배, 확신 60, +1.5% 위, 최근 90봉, EMA 100봉, 102 위 마감이면 매수").to_real(0.25, text_ref=100.0)
    assert t.summary == "저항 402.00~404.80 를 3번 두드림, 거래량 2.5배, 확신 60, +1.5% 위, 최근 90봉, EMA 100봉, 408.00 위 마감이면 매수"
    open_p = build_packet(bars, as_of, ticker="QQQ", blind=False)
    assert open_p.scale == 1.0 and open_p.labels[-1].startswith("03-06")


def test_daily_packet_weekly_context_and_position():
    d = _daily(400, end="2024-03-06")                       # 수요일에 끝남 → 그 주 주봉은 미완성
    spec = PATTERNS["wedge_pop"].chart
    p = build_packet(d, d.index[-1], spec=spec, ticker="AAPL", blind=True, position={"entry": 400.0, "stop": 390.0, "target": 430.0, "bars_held": 3})
    assert p.labels[-1] == "T0" and p.labels[-2] == "T-1" and "AAPL" not in p.bars_csv + str(p.facts)
    assert "ema10" in p.bars_csv.splitlines()[0] and "EMA200" in p.facts and p.context_png
    assert p.facts["보유 중"]["진입가"] == pytest.approx(400.0 * p.scale, abs=0.01)
    earlier = build_packet(d.iloc[:-1], d.index[-2], spec=spec, blind=True)
    again = build_packet(d, d.index[-2], spec=spec, blind=True)
    assert earlier.chart_png == again.chart_png and earlier.context_png == again.context_png


def test_opening_range_facts_for_1m():
    bars = _bars(days=3, freq="1min")
    reg = bars.between_time("09:30", "15:59")
    spec = PATTERNS["scraface"].chart
    as_of = [t for t in reg.index if t.strftime("%Y-%m-%d %H:%M") == "2024-03-06 09:50"][0]
    p = build_packet(reg, as_of, spec=spec, blind=False, ticker="TSLA")
    first5 = reg.loc["2024-03-06 09:30":"2024-03-06 09:34"]
    assert p.facts["오늘 첫 5분 고가/저가"] == [round(first5["High"].max(), 2), round(first5["Low"].min(), 2)]
    assert "오늘 첫 15분 고가/저가" in p.facts and "오늘 첫 30분 고가/저가" not in p.facts


# ---------------------------------------------------------------------------
# 체결·정산
# ---------------------------------------------------------------------------
def _fut(rows) -> pd.DataFrame:
    idx = pd.date_range("2024-03-06 10:45", periods=len(rows), freq="15min", tz=NY)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def test_settle_market_target_and_stop():
    win = settle(_j(), _fut([(100, 100.5, 99.8, 100.4), (100.4, 102.3, 100.2, 102.1), (102, 103, 101, 102)]), fee_bps=0)
    assert win["filled"] and win["entry"] == 100 and win["why"] == "목표1" and win["ret"] == pytest.approx(0.02) and win["R"] == pytest.approx(2.0)
    # 종가 이탈 손절: 저가가 닿기만 한 봉은 버티고, 아래로 마감한 봉에서 그 종가에 나간다
    lose = settle(_j(), _fut([(100, 100.2, 98.5, 99.5), (99.5, 99.6, 98.0, 98.4)]), fee_bps=0)
    assert lose["why"] == "손절" and lose["ret"] == pytest.approx(-0.016) and lose["bars"] == 1
    touch = settle(_j(stop_basis="touch"), _fut([(100, 100.2, 98.5, 99.5)]), fee_bps=0)
    assert touch["ret"] == pytest.approx(-0.01)


def test_settle_two_targets_breakout_limit():
    r = settle(_j(target2=104.0), _fut([(100, 102.5, 99.9, 102.2), (102.2, 103, 101.5, 103.0)]), fee_bps=5)
    assert r["why"] == "목표1 → 기간 끝" and r["ret"] == pytest.approx(0.5 * 0.02 + 0.5 * 0.03 - 0.001)
    fut = _fut([(100, 100.4, 99.9, 100.3), (100.3, 100.9, 100.2, 100.8), (100.8, 102.5, 100.7, 102.4)])
    r = settle(_j(entry_type="breakout_close", entry_trigger=100.5, trigger_valid_bars=3, stop=99.8, target1=102.0), fut, fee_bps=0)
    assert r["filled"] and r["entry"] == 100.8 and r["why"] == "목표1" and r["bars"] == 1
    none = settle(_j(entry_type="breakout_close", entry_trigger=100.5, trigger_valid_bars=1, stop=99.8), fut, fee_bps=0)
    assert not none["filled"] and "돌파 마감 없음" in none["why"]
    skip = settle(SKIP, fut, fee_bps=0)
    assert not skip["filled"] and skip["이후 최대 상승"] == pytest.approx(0.025)
    # 눌림 매수: 100.0 까지 내려온 봉에서 체결, 그 봉의 고가로는 목표를 치지 않는다
    lim = settle(_j(entry_type="limit", entry_trigger=100.0, trigger_valid_bars=3, stop=99.0, target1=101.0),
                 _fut([(100.6, 101.5, 100.3, 100.5), (100.5, 101.2, 99.9, 100.2), (100.2, 101.1, 100.1, 101.0)]), fee_bps=0)
    assert lim["filled"] and lim["entry"] == 100.0 and lim["bars"] == 1 and lim["why"] == "목표1"
    # 시가가 이미 목표 위면 그 목표는 버리고(즉시 청산 금지) 계속 든다
    gap = settle(_j(stop=99.0, target1=100.5), _fut([(101.0, 101.5, 100.8, 101.2), (101.2, 101.6, 101.0, 101.4)]), fee_bps=0)
    assert gap["bars"] == 1 and gap["why"] == "기간 끝"


# ---------------------------------------------------------------------------
# 기간 백테스트 (가짜 판정기)
# ---------------------------------------------------------------------------
def _trend_bars(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-03-04 09:30", periods=n, freq="15min", tz=NY)
    c = 100 + np.arange(n) * 0.1
    return pd.DataFrame({"Open": c - 0.05, "High": c + 0.1, "Low": c - 0.15, "Close": c, "Volume": 1000.0}, index=idx)


def test_backtest_walks_forward_and_never_shows_future():
    bars = _trend_bars()
    spec = ChartSpec("15m", vwap=True, session="crypto", view_bars=60, table_bars=20)
    cfg = BtConfig(step=5, blind=False, manage=True, chunk=3, fee_bps=0, max_hold=0)
    seen: list[pd.Timestamp] = []

    def ask_entry(pks):
        out = []
        for pk in pks:
            seen.append(pk.as_of)
            assert pk.bars_csv.splitlines()[-1].startswith(pk.labels[-1])
            out.append((_j(stop=pk.last - 1.0, target1=None, target2=None), {"cost_usd": 0.01}))
        return out

    n_manage = {"n": 0}

    def ask_manage(pk):
        n_manage["n"] += 1
        assert "보유 중" in pk.facts
        if n_manage["n"] % 2 == 0:
            return Manage(action="exit", confidence=70, reason="x"), {"cost_usd": 0.01}
        return Manage(action="hold", new_stop=pk.last - 0.3, confidence=50, reason="y"), {"cost_usd": 0.01}

    res = run_backtest(bars, spec, cfg, start=bars.index[0], end=bars.index[-1], ask_entry=ask_entry, ask_manage=ask_manage)
    tr = res["trades"]
    assert len(tr) >= 3 and all(t["why"] == "AI 청산" for t in tr[:-1])
    first = tr[0]
    assert first["entry_i"] == first["decision_i"] + 1 and first["bars"] == 11            # 5봉째 hold(손절 올림), 10봉째 exit → 11봉째 시가
    assert first["stop_last"] > first["stop0"] and len(first["manage_notes"]) == 2
    for a, b in zip(tr, tr[1:]):                                                            # 포지션은 겹치지 않는다
        assert b["decision_i"] >= a["exit_i"]
    assert res["stats"]["총 수익률"] > 0 and res["stats"]["거래 수"] == len(tr)
    assert res["equity"].iloc[-1] == pytest.approx(1 + res["stats"]["총 수익률"])
    assert res["stats"]["호출 수"] == len(seen) + n_manage["n"]
    assert all(t <= bars.index[-2] for t in seen)


def test_backtest_respects_windows_and_call_cap():
    bars = _bars(days=4)
    spec = PATTERNS["tradesharp_orb"].chart
    cfg = BtConfig(step=2, regular_only=True, ask_from="09:45", ask_until="11:30", flat_at_close=True, blind=True, manage=False, max_calls=5, chunk=2)
    cps = checkpoints(bars, spec, cfg, bars.index[0], bars.index[-1])
    hm = [bars.index[i].strftime("%H:%M") for i in cps]
    assert hm and all("09:45" <= t <= "11:30" for t in hm)
    res = run_backtest(bars, spec, cfg, start=bars.index[0], end=bars.index[-1], ask_entry=lambda pks: [(SKIP, {}) for _ in pks])
    assert res["stats"]["거래 수"] == 0 and res["stats"]["총 수익률"] == 0 and "호출 상한" in res["stats"]["중단"]
    # 장 마감 청산: 정규장 마지막 봉 종가에 나간다
    cfg2 = BtConfig(step=2, regular_only=True, ask_from="09:45", ask_until="10:00", flat_at_close=True, blind=False, manage=False, max_calls=50, chunk=2, fee_bps=0)
    res2 = run_backtest(bars, spec, cfg2, start=bars.index[0], end=bars.index[-1],
                        ask_entry=lambda pks: [(_j(stop=pk.last * 0.5, target1=None), {}) for pk in pks])
    assert res2["trades"] and all(t["why"] == "장 마감" and pd.Timestamp(t["exit_time"]).strftime("%H:%M") == "15:45" for t in res2["trades"])


def test_scan_maps_labels_and_settles():
    bars = _trend_bars(260)
    spec = ChartSpec("15m", vwap=True, session="crypto")
    cfg = BtConfig(blind=True, fee_bps=0, max_hold=0)
    asked = []

    def ask_scan(pk, first, last):
        asked.append((first, last, pk.n_bars))
        labs = pk.labels
        k0 = labs.index(first)
        assert pk.bars_csv.count("\n") == pk.n_bars                       # 차트의 모든 봉이 표에 있다
        sig = lambda k, **kw: Signal(bar=labs[k], setup="s", confidence=70, stop=pk.last * 0.5, note="n", **kw)  # noqa: E731
        return ScanResult(overview="o", signals=[
            sig(k0 + 5, exit_bar=labs[k0 + 15]),                           # AI 청산 봉 사용
            sig(k0 + 10),                                                  # 앞 거래 보유 중 → 건너뜀
            sig(k0 - 3),                                                   # 맥락 구간 → 버림
            Signal(bar="없는 봉", setup="s", confidence=1, stop=1.0, note="n"),
            sig(k0 + 30, target1=float(pk.last) * 10, exit_bar=labs[k0 + 20]),   # 청산 봉이 신호보다 앞 → 무시, 창 끝까지 보유
        ]), {"cost_usd": 0.02}

    res = run_scan(bars, spec, cfg, start=bars.index[100], end=bars.index[-1], ask_scan=ask_scan, window=100, lead=40)
    assert len(asked) == 2 and asked[0][2] == 140
    tr = res["trades"]
    assert [t["decision_i"] for t in tr[:2]] == [105, 130] and tr[0]["entry_i"] == 106 and tr[0]["exit_i"] == 115 and tr[0]["why"] == "AI 청산"
    assert tr[0]["ret"] == pytest.approx(bars["Close"].iloc[115] / bars["Open"].iloc[106] - 1)
    skipped = [d for d in res["decisions"] if not d["filled"]]
    assert any("보유 중" in d["why"] for d in skipped) and all(d["i"] >= 100 for d in res["decisions"])
    assert res["stats"]["호출 수"] == 2 and res["stats"]["비용($)"] == pytest.approx(0.04) and res["mode"] == "scan"
    no_exit = run_scan(bars, spec, cfg, start=bars.index[100], end=bars.index[-1], ask_scan=ask_scan, window=100, lead=40, use_ai_exit=False)
    assert no_exit["trades"][0]["exit_i"] == len(bars) - 1


def test_cache_roundtrip(tmp_path):
    from strategy.adapters.claude_chart_judge import ClaudeJudge

    class _Usage:
        input_tokens = 1000; output_tokens = 100

    class _Resp:
        stop_reason = "end_turn"; model = "m"; _request_id = "r"; parsed_output = SKIP; usage = _Usage()

    calls = {"n": 0}

    class _Msgs:
        def parse(self, **kw):
            calls["n"] += 1
            assert kw["output_format"] is Judgement and "fallbacks" in kw
            return _Resp()

    class _Beta:
        messages = _Msgs()

    class _Client:
        beta = _Beta()

    bars = _bars()
    pk = build_packet(bars, bars.index[150], blind=True)
    jd = ClaudeJudge(title="t", principles="p", step=4, cache_dir=tmp_path, client=_Client(), workers=2)
    (a, ua), = jd.ask_entry([pk])
    (b, ub), = jd.ask_entry([pk])
    assert calls["n"] == 1 and not ua["cached"] and ub["cached"] and ub["cost_usd"] == 0 and a == b
    jd2 = ClaudeJudge(title="t", principles="다른 원칙", step=4, cache_dir=tmp_path, client=_Client())
    jd2.ask_entry([pk])
    assert calls["n"] == 2                                   # 원칙 글이 바뀌면 다른 질문 → 새 호출


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page,key", [("56_VWAP_SR_Breakout_AI", "vwap_sr"), ("57_Wedge_Pop_AI", "wedge_pop"), ("63_Scraface_AI", "scraface"),
                                      ("62_Bitcoin_Fair_Value_Gap_AI", "fvg_btc")])
def test_page_smoke_without_key(monkeypatch, page, key):
    """키가 없어도 페이지가 뜨고(데이터는 가짜) 실행 버튼만 꺼진다."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False); monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    import data.adapters.alphavantage_adapter as av
    import data.adapters.composed_market_data as cm
    import data.adapters.yfinance_adapter as yf

    class _Fake:
        def fetch_ohlcv(self, sym, s, e, interval="1d"):
            if interval == "1d":
                return _daily(500, end=str(e))
            days = pd.bdate_range(end=str(e), periods=8)
            b = _bars(days=8, freq={"1m": "1min", "5m": "5min"}.get(interval, "15min"))
            b.index = b.index + (days[0] - b.index[0].tz_localize(None).normalize())
            return b

    for mod, name in ((av, "AlphaVantageAdapter"), (cm, "build_default_market_data"), (yf, "YFinanceAdapter")):
        monkeypatch.setattr(mod, name, lambda *a, **k: _Fake())
    assert key in PATTERNS
    at = AppTest.from_file(f"../pages/{page}.py", default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    assert any("ANTHROPIC_API_KEY" in w.value for w in at.warning)
    assert at.button[0].disabled and at.metric[1].value != "0"


def test_every_pattern_has_a_page():
    from pathlib import Path

    pages = "\n".join(p.read_text() for p in (Path(__file__).parent.parent / "pages").glob("*_AI.py"))
    for key in PATTERNS:
        assert f'render_page("{key}")' in pages
