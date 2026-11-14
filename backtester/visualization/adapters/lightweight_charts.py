"""TradingView Lightweight Charts(오픈소스, Apache 2.0) 기반 차트 — Streamlit 안에 iframe 으로 넣는다.

plotly 캔들 차트의 대체. 확대·이동·십자선이 트레이딩뷰와 같고 봉이 많아도 빠르다.
라이브러리 파일은 `visualization/static/lightweight-charts.4.2.3.js` 를 html 에 그대로 넣으므로 인터넷이 없어도 된다.

    from visualization.adapters.lightweight_charts import Pane, render
    render([
        Pane(height=520, title="15분봉", candles=df, lines=[dict(name="VWAP", data=df["vwap"], color="#ff9100", width=2)],
             segments=[dict(points=[(t0, y0), (t1, y1)], color="#64b5f6", width=3)],
             markers=[dict(time=t, position="below", shape="arrowUp", color="#00e676", text="진입")]),
        Pane(height=140, title="거래량", hist=[dict(name="거래량", data=df["Volume"], colors=[...])]),
    ], index=df.index, intraday=True)

- 시간축은 봉 순서(빈 시간 없음) — plotly 의 rangebreaks 가 필요 없다.
- `segments` 의 두 점 사이는 봉 수 기준 직선(엔진의 "봉당 기울기"와 같다).
- 음영 박스는 기본 기능이 아니라 `shades`(세로 띠)와 선분으로 대신한다.
- 여러 Pane 은 같은 index 를 공유하고 확대·이동·십자선이 함께 움직인다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_JS = Path(__file__).resolve().parents[1] / "static" / "lightweight-charts.4.2.3.js"
_STYLE = {"solid": 0, "dot": 1, "dash": 2, "longdash": 3, "dashdot": 3, "sparse": 4}
_POS = {"above": "aboveBar", "below": "belowBar", "in": "inBar"}

THEMES = {
    "dark": {"bg": "#0e1117", "text": "#c9d1d9", "grid": "rgba(255,255,255,0.06)", "border": "rgba(255,255,255,0.18)", "legend": "rgba(14,17,23,0.72)"},
    "light": {"bg": "#ffffff", "text": "#1f2328", "grid": "rgba(0,0,0,0.06)", "border": "rgba(0,0,0,0.18)", "legend": "rgba(255,255,255,0.78)"},
}
UP, DOWN = "#26a69a", "#ef5350"


@dataclass
class Pane:
    height: int = 300
    title: str = ""
    candles: pd.DataFrame | None = None            # Open/High/Low/Close, index = 시간
    lines: list[dict] = field(default_factory=list)      # {name, data: Series, color, width=1.5, style="solid", legend=True}
    hist: list[dict] = field(default_factory=list)       # {name, data: Series, colors: str | list[str]}
    segments: list[dict] = field(default_factory=list)   # {points: [(time, value), ...], color, width=1.5, style="solid", name=None}
    markers: list[dict] = field(default_factory=list)    # {time, position: above|below|in, shape: arrowUp|arrowDown|circle|square, color, text}
    hlines: list[dict] = field(default_factory=list)     # {price, color, title="", style="dash", width=1}
    shades: list[tuple] = field(default_factory=list)    # (t0, t1, "rgba(...)") 세로 띠
    boxes: list[dict] = field(default_factory=list)      # {t0, t1, p0, p1, fill="rgba(...)", line=None} 가격·시간 사각형(캔들 뒤)
    legend_extra: list[dict] = field(default_factory=list)   # {name, color} 마커 등 범례에만 올릴 항목
    log: bool = False
    up_color: str = UP
    down_color: str = DOWN


@lru_cache(maxsize=1)
def _lib() -> str:
    return _JS.read_text()


def _auto_theme() -> str:
    try:
        import streamlit as st
        t = getattr(getattr(st, "context", None), "theme", None)
        typ = getattr(t, "type", None)
        if typ in ("dark", "light"):
            return typ
        return "light" if st.get_option("theme.base") == "light" else "dark"
    except Exception:
        return "dark"


def _naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _time_fn(intraday: bool):
    if intraday:
        return lambda ts: int(_naive(ts).value // 10**9)      # 벽시계 시각을 UTC 로 취급 → 축에 그 시각 그대로 표시
    return lambda ts: _naive(ts).strftime("%Y-%m-%d")


def _num(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def build_payload(panes: list[Pane], index: pd.Index, intraday: bool = False, visible_bars: int | None = None, theme: str | None = None) -> dict:
    tf = _time_fn(intraday)
    times = [tf(t) for t in index]
    pos = {t: i for i, t in enumerate(times)}
    out = []
    for p in panes:
        d: dict = {"height": int(p.height), "title": p.title, "log": bool(p.log), "series": [], "hlines": [], "markers": [], "up": p.up_color, "down": p.down_color,
                   "boxes": [], "extra": [{"name": e["name"], "color": e.get("color", "#90a4ae")} for e in p.legend_extra if e.get("name")]}
        for b in p.boxes:
            t0, t1 = tf(b["t0"]), tf(b["t1"])
            if t0 in pos and t1 in pos and _num(b["p0"]) is not None and _num(b["p1"]) is not None:
                if pos[t0] > pos[t1]:
                    t0, t1 = t1, t0
                d["boxes"].append({"t0": t0, "t1": t1, "p0": _num(b["p0"]), "p1": _num(b["p1"]), "fill": b.get("fill", "rgba(120,144,156,0.15)"), "line": b.get("line")})
        if p.candles is not None and len(p.candles):
            c = p.candles
            rows = [{"time": tf(t), "open": _num(o), "high": _num(h), "low": _num(lo), "close": _num(cl)}
                    for t, o, h, lo, cl in zip(c.index, c["Open"], c["High"], c["Low"], c["Close"]) if _num(cl) is not None]
            d["series"].append({"type": "candle", "name": "", "data": rows})
        for h in p.hist:
            s = h["data"]; cols = h.get("colors", "#78909c")
            cols = [cols] * len(s) if isinstance(cols, str) else list(cols)
            d["series"].append({"type": "hist", "name": h.get("name", ""), "color": cols[0] if cols else "#78909c", "scale": h.get("scale"),
                                "data": [{"time": tf(t), "value": _num(v), "color": col} for t, v, col in zip(s.index, s.to_numpy(), cols) if _num(v) is not None]})
        for ln in p.lines:
            s = ln["data"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            d["series"].append({"type": "line", "name": ln.get("name", ""), "color": ln.get("color", "#42a5f5"), "width": ln.get("width", 1.5),
                                "style": _STYLE.get(ln.get("style", "solid"), 0), "legend": ln.get("legend", True), "scale": ln.get("scale"),
                                "data": [({"time": tf(t), "value": v} if (v := _num(x)) is not None else {"time": tf(t)}) for t, x in zip(s.index, s.to_numpy())]})
        for sg in p.segments:
            pts = sorted({tf(t): _num(v) for t, v in sg["points"]}.items())
            pts = [{"time": t, "value": v} for t, v in pts if v is not None]
            if len(pts) >= 2:
                d["series"].append({"type": "line", "name": sg.get("name") or "", "color": sg.get("color", "#90a4ae"), "width": sg.get("width", 1.5),
                                    "style": _STYLE.get(sg.get("style", "solid"), 0), "legend": bool(sg.get("name")), "segment": True, "scale": sg.get("scale"), "data": pts})
        if p.shades:
            col = [None] * len(times)
            for t0, t1, c in p.shades:
                i0, i1 = pos.get(tf(t0)), pos.get(tf(t1))
                if i0 is None or i1 is None:
                    continue
                for i in range(min(i0, i1), max(i0, i1) + 1):
                    col[i] = c
            d["series"].append({"type": "shade", "name": "", "data": [{"time": t, "value": 1, "color": c} for t, c in zip(times, col) if c]})
        for h in p.hlines:
            if _num(h["price"]) is not None:
                d["hlines"].append({"price": _num(h["price"]), "color": h.get("color", "#90a4ae"), "title": h.get("title", ""),
                                    "style": _STYLE.get(h.get("style", "dash"), 2), "width": h.get("width", 1)})
        mk = [{"time": tf(m["time"]), "position": _POS.get(m.get("position", "above"), "aboveBar"), "shape": m.get("shape", "circle"),
               "color": m.get("color", "#ffeb3b"), "text": m.get("text", ""), "size": m.get("size", 1)} for m in p.markers]
        d["markers"] = sorted([m for m in mk if m["time"] in pos], key=lambda m: pos[m["time"]])
        out.append(d)
    return {"times": times, "panes": out, "intraday": bool(intraday), "visible": visible_bars, "theme": THEMES[theme or _auto_theme()]}


_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:__BG__;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
.pane{position:relative;width:100%}
.lg{position:absolute;left:8px;top:4px;z-index:5;font-size:11.5px;line-height:1.5;color:__TEXT__;background:__LEGEND__;padding:1px 6px;border-radius:4px;pointer-events:none;max-width:calc(100% - 90px)}
.hint{position:absolute;right:84px;bottom:32px;z-index:5;font-size:10.5px;color:__TEXT__;opacity:.38;pointer-events:none}
.lg b{font-weight:600;margin-right:8px}.lg span{margin-right:10px;white-space:nowrap}.lg i{display:inline-block;width:9px;height:3px;margin-right:4px;vertical-align:middle}
</style></head><body><div id="root"></div><script>__LIB__</script><script>
const P = __PAYLOAD__, T = P.theme, L = LightweightCharts, root = document.getElementById('root');
window.addEventListener('error', e => { const d = document.createElement('div'); d.style.cssText = 'color:#ef5350;font-size:12px;padding:8px'; d.textContent = '차트 오류: ' + e.message; document.body.prepend(d); });
const fmt = v => v == null ? '' : (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : Math.abs(v) >= 1e4 ? v.toFixed(0) : v.toFixed(2));
const charts = [], anchors = [], legends = [];
function boxPrimitive(boxes) {
  let chart = null, series = null;
  const view = {zOrder: () => 'bottom', renderer: () => ({draw: target => target.useBitmapCoordinateSpace(sc => {
    const ctx = sc.context, hr = sc.horizontalPixelRatio, vr = sc.verticalPixelRatio, ts = chart.timeScale();
    const half = (ts.options().barSpacing || 6) / 2;
    boxes.forEach(b => {
      const x0 = ts.timeToCoordinate(b.t0), x1 = ts.timeToCoordinate(b.t1), y0 = series.priceToCoordinate(b.p0), y1 = series.priceToCoordinate(b.p1);
      if (x0 == null || x1 == null || y0 == null || y1 == null) return;
      const X = (Math.min(x0, x1) - half) * hr, W = (Math.abs(x1 - x0) + 2 * half) * hr, Y = Math.min(y0, y1) * vr, H = Math.max(1, Math.abs(y1 - y0) * vr);
      if (b.fill) { ctx.fillStyle = b.fill; ctx.fillRect(X, Y, W, H); }
      if (b.line) { ctx.strokeStyle = b.line; ctx.lineWidth = Math.max(1, hr); ctx.strokeRect(X, Y, W, H); }
    });
  })})};
  return {attached: p => { chart = p.chart; series = p.series; }, detached: () => {}, updateAllViews: () => {}, paneViews: () => [view]};
}
P.panes.forEach((p, k) => {
  const el = document.createElement('div'); el.className = 'pane'; el.style.height = p.height + 'px'; root.appendChild(el);
  const last = k === P.panes.length - 1;
  const ch = L.createChart(el, {autoSize: true,
    layout: {background: {color: T.bg}, textColor: T.text, fontSize: 11, attributionLogo: k === 0},
    grid: {vertLines: {color: T.grid}, horzLines: {color: T.grid}},
    crosshair: {mode: 0},
    handleScroll: {mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false},
    handleScale: {mouseWheel: false, pinch: true, axisPressedMouseMove: true, axisDoubleClickReset: true},
    leftPriceScale: {visible: p.series.some(s => s.scale === 'left'), borderColor: T.border},
    rightPriceScale: {borderColor: T.border, mode: p.log ? 1 : 0, minimumWidth: 72},
    timeScale: {borderColor: T.border, visible: last, timeVisible: P.intraday, secondsVisible: false, rightOffset: 4, minBarSpacing: 0.3},
    localization: {locale: 'ko-KR'}});
  const anchor = ch.addLineSeries({color: 'rgba(0,0,0,0)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, priceScaleId: 'anchor'});
  ch.priceScale('anchor').applyOptions({visible: false});
  anchor.setData(P.times.map(t => ({time: t})));
  const items = []; let main = null;
  const hasCandle = p.series.some(s => s.type === 'candle');   // 캔들이 있는 Pane 은 캔들만으로 가격 눈금 자동 범위를 잡는다(채널·이평선이 눈금을 벌리지 않게)
  const hasMain = p.series.some(s => s.type === 'candle' || (s.type === 'line' && !s.segment));   // 선분(추세선·채널)은 가격 눈금 자동 범위에 넣지 않는다
  p.series.forEach(s => {
    let ser;
    if (s.type === 'candle') {
      ser = ch.addCandlestickSeries({upColor: p.up, downColor: p.down, borderUpColor: p.up, borderDownColor: p.down, wickUpColor: p.up, wickDownColor: p.down, priceLineVisible: false});
      main = ser; items.push({s, ser, candle: true});
    } else if (s.type === 'hist') {
      const ov = s.scale === 'overlay';
      ser = ch.addHistogramSeries(Object.assign({color: s.color, priceLineVisible: false, lastValueVisible: false, priceFormat: {type: 'volume'}}, ov ? {priceScaleId: 'vol'} : {}));
      if (ov) ch.priceScale('vol').applyOptions({visible: false, scaleMargins: {top: 0.8, bottom: 0}});
      else if (!main) main = ser;
      items.push({s, ser});
    } else if (s.type === 'shade') {
      ser = ch.addHistogramSeries({priceScaleId: 'shade', priceLineVisible: false, lastValueVisible: false});
      ch.priceScale('shade').applyOptions({visible: false, scaleMargins: {top: 0, bottom: 0}});
    } else {
      ser = ch.addLineSeries(Object.assign({color: s.color, lineWidth: Math.max(1, Math.round(s.width)), lineStyle: s.style, priceLineVisible: false,
        lastValueVisible: !s.segment && s.legend, crosshairMarkerVisible: !s.segment}, s.scale === 'left' ? {priceScaleId: 'left'} : {},
        (hasCandle || (s.segment && hasMain)) && s.scale !== 'left' ? {autoscaleInfoProvider: () => null} : {}));
      if (!main && s.scale !== 'left') main = ser; if (s.legend) items.push({s, ser});
    }
    ser.setData(s.data);
  });
  if (main) {
    p.hlines.forEach(h => main.createPriceLine({price: h.price, color: h.color, lineWidth: h.width, lineStyle: h.style, axisLabelVisible: true, title: h.title}));
    if (p.markers.length) main.setMarkers(p.markers);
    if (p.boxes.length) main.attachPrimitive(boxPrimitive(p.boxes));
  }
  if (last) { const hint = document.createElement('div'); hint.className = 'hint'; hint.textContent = 'Ctrl/⌘+휠 확대 · 끌기 이동 · 더블클릭 초기화'; el.appendChild(hint); }
  const lg = document.createElement('div'); lg.className = 'lg'; el.appendChild(lg);
  const draw = param => {
    let h = p.title ? '<b>' + p.title + '</b>' : '';
    items.forEach(it => {
      let d = param && param.seriesData ? param.seriesData.get(it.ser) : null;
      if (!d) { const arr = it.s.data; for (let i = arr.length - 1; i >= 0; i--) if (arr[i].value != null || arr[i].close != null) { d = arr[i]; break; } }
      if (it.candle) { if (d) { const c = d.close >= d.open ? p.up : p.down;
          h += '<span style="color:' + c + '">시 ' + fmt(d.open) + ' 고 ' + fmt(d.high) + ' 저 ' + fmt(d.low) + ' 종 ' + fmt(d.close) + ' (' + ((d.close / d.open - 1) * 100).toFixed(2) + '%)</span>'; } }
      else if (it.s.name) h += '<span><i style="background:' + it.s.color + '"></i>' + it.s.name + (d && d.value != null && !it.s.segment ? ' ' + fmt(d.value) : '') + '</span>';
    });
    p.extra.forEach(e => { h += '<span><i style="background:' + e.color + ';height:7px;width:7px;border-radius:4px"></i>' + e.name + '</span>'; });
    lg.innerHTML = h; lg.style.display = h ? '' : 'none';
  };
  draw(null); charts.push(ch); anchors.push(anchor); legends.push(draw);
});
let busy = false;
charts.forEach((ch, i) => {
  ch.timeScale().subscribeVisibleLogicalRangeChange(r => { if (busy || !r) return; busy = true; charts.forEach((o, j) => { if (j !== i) o.timeScale().setVisibleLogicalRange(r); }); busy = false; });
  ch.subscribeCrosshairMove(param => {
    legends[i](param);
    charts.forEach((o, j) => { if (j === i) return;
      if (param.time !== undefined) { try { o.setCrosshairPosition(NaN, param.time, anchors[j]); } catch (e) {} legends[j]({seriesData: null, time: param.time, _sync: true}); }
      else { o.clearCrosshairPosition(); legends[j](null); } });
  });
});
const n = P.times.length;
const reset = () => charts.forEach(ch => { if (P.visible && P.visible < n) ch.timeScale().setVisibleLogicalRange({from: n - P.visible - 0.5, to: n + 3}); else ch.timeScale().fitContent();
  ['right', 'left'].forEach(id => { try { ch.priceScale(id).applyOptions({autoScale: true}); } catch (e) {} }); });
reset();
// 휠: 그냥 굴리면 페이지 스크롤(차트가 가로채지 않음), Ctrl/⌘+휠·트랙패드 핀치 = 커서 기준 확대, 가로 휠 = 이동
root.querySelectorAll('.pane').forEach((el, i) => {
  el.addEventListener('wheel', e => {
    const ts = charts[i].timeScale(), r = ts.getVisibleLogicalRange(); if (!r) return;
    const horiz = Math.abs(e.deltaX) > Math.abs(e.deltaY);
    if (!(e.ctrlKey || e.metaKey) && !horiz) return;
    e.preventDefault();
    if (horiz && !(e.ctrlKey || e.metaKey)) { const d = e.deltaX / (ts.options().barSpacing || 6); ts.setVisibleLogicalRange({from: r.from + d, to: r.to + d}); return; }
    const rect = el.getBoundingClientRect(); let x = ts.coordinateToLogical(e.clientX - rect.left); if (x == null) x = (r.from + r.to) / 2;
    const k = Math.exp(Math.max(-60, Math.min(60, e.deltaY)) * 0.012);
    if ((r.to - r.from) * k < 5 && k < 1) return;
    ts.setVisibleLogicalRange({from: x - (x - r.from) * k, to: x + (r.to - x) * k});
  }, {passive: false});
  el.addEventListener('dblclick', reset);
});
</script></body></html>"""


def to_html(panes: list[Pane], index: pd.Index, intraday: bool = False, visible_bars: int | None = None, theme: str | None = None) -> str:
    payload = build_payload(panes, index, intraday, visible_bars, theme)
    th = payload["theme"]
    return (_HTML.replace("__BG__", th["bg"]).replace("__TEXT__", th["text"]).replace("__LEGEND__", th["legend"])
            .replace("__LIB__", _lib()).replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")))


def render(panes: list[Pane], index: pd.Index, intraday: bool = False, visible_bars: int | None = None, theme: str | None = None) -> None:
    """Streamlit 페이지에 그린다. 높이 = Pane 높이 합."""
    import streamlit.components.v1 as components
    components.html(to_html(panes, index, intraday, visible_bars, theme), height=sum(p.height for p in panes) + 4, scrolling=False)
