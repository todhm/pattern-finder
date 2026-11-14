import numpy as np
import pandas as pd

from visualization.adapters.lightweight_charts import Pane, build_payload, to_html


def _df(idx):
    c = 100 + np.arange(len(idx), dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c + 0.5, "Volume": 1000.0}, index=idx)


def test_daily_payload_times_and_nan_gaps():
    idx = pd.date_range("2026-01-05", periods=30, freq="B")
    df = _df(idx)
    p = build_payload([Pane(candles=df, lines=[dict(name="ma", data=df["Close"].rolling(5).mean())],
                            markers=[dict(time=idx[9], text="b"), dict(time=idx[3], text="a")], hlines=[dict(price=110)])], idx, theme="dark")
    assert p["times"][0] == "2026-01-05" and len(p["times"]) == 30
    candle, line = p["panes"][0]["series"]
    assert candle["type"] == "candle" and len(candle["data"]) == 30
    assert "value" not in line["data"][0] and line["data"][4]["value"] == df["Close"].iloc[:5].mean()   # NaN → 빈 칸
    assert [m["text"] for m in p["panes"][0]["markers"]] == ["a", "b"]                                    # 시간순 정렬


def test_intraday_wallclock_segments_and_shades():
    idx = pd.date_range("2026-11-02 09:30", periods=20, freq="15min", tz="America/New_York")
    naive = idx.tz_localize(None); df = _df(naive)
    p = build_payload([Pane(candles=df, segments=[dict(points=[(naive[8], 105.0), (naive[2], 101.0)], color="#fff")],
                            shades=[(naive[5], naive[7], "rgba(1,1,1,0.1)")])], naive, intraday=True, theme="light")
    assert p["times"][0] == int(pd.Timestamp("2026-11-02 09:30").value // 10**9)                         # 벽시계 시각 그대로
    seg = [s for s in p["panes"][0]["series"] if s.get("segment")][0]
    assert [d["value"] for d in seg["data"]] == [101.0, 105.0]
    shade = [s for s in p["panes"][0]["series"] if s["type"] == "shade"][0]
    assert len(shade["data"]) == 3


def test_html_is_self_contained():
    idx = pd.date_range("2026-01-05", periods=10, freq="B")
    h = to_html([Pane(candles=_df(idx), title="</script>x")], idx, theme="dark")
    assert "LightweightCharts" in h and "http://" not in h.split("__never__")[0][:0] and "<\\/script>x" in h


def test_plotly_bridge_converts_candle_figure():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from visualization.adapters.plotly_bridge import Unsupported, figure_to_panes
    idx = pd.date_range("2026-01-05", periods=60, freq="B"); df = _df(idx)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], subplot_titles=("가격", "거래량"))
    fig.add_trace(go.Candlestick(x=idx, open=df.Open, high=df.High, low=df.Low, close=df.Close), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=df.Close.rolling(5).mean(), mode="lines", name="MA5", line=dict(color="orange", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[idx[3], idx[20], None, idx[30], idx[40]], y=[100, 120, None, 130, 131], mode="lines", name="추세선"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[idx[10]], y=[df.Low.iloc[10]], mode="markers+text", text=["매수"], marker=dict(symbol="triangle-up", color="green")), row=1, col=1)
    fig.add_trace(go.Bar(x=idx, y=df.Volume, marker_color=["red"] * 60), row=2, col=1)
    fig.add_hline(y=110, line_dash="dash", row=1, col=1)
    fig.add_vrect(x0=idx[5], x1=idx[8], fillcolor="red", opacity=0.1, row=1, col=1)
    fig.add_shape(type="rect", x0=idx[12], x1=idx[18], y0=105, y1=108, fillcolor="rgba(0,255,0,0.2)", row=1, col=1)
    # 주말(봉 없는 날)은 가장 가까운 봉으로
    fig.add_annotation(x=pd.Timestamp("2026-01-10"), y=200, text="<b>메모</b>", row=1, col=1)
    panes, base, intraday, _ = figure_to_panes(fig)
    assert len(panes) == 2 and not intraday and len(base) == 60
    main, vol = panes
    assert main.title == "가격" and vol.title == "거래량" and main.height > vol.height
    assert len(main.candles) == 60 and [ln["name"] for ln in main.lines] == ["MA5"] and len(main.segments) == 2
    assert main.segments[0]["name"] == "추세선" and main.segments[1]["name"] is None
    assert any(m["shape"] == "arrowUp" and m["text"] == "매수" and m["position"] == "below" for m in main.markers)
    assert any(m["text"] == "메모" and m["time"] in base for m in main.markers)
    assert len(main.hlines) == 1 and len(main.shades) == 1 and len(main.boxes) == 1 and len(vol.hist) == 1
    p = build_payload(panes, base, theme="dark")
    assert len(p["panes"][0]["boxes"]) == 1

    import pytest
    with pytest.raises(Unsupported):
        figure_to_panes(go.Figure(go.Scatter(x=idx, y=df.Close)))
