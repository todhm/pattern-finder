"""plotly 캔들 Figure → TradingView Lightweight Charts 변환.

기존 페이지는 plotly Figure 를 만들어 `st.plotly_chart(fig)` 로 그린다. 그 줄을 `st_candles(fig)` 로 바꾸면
  - 캔들이 들어 있는 그림은 Lightweight Charts(확대·이동·십자선이 트레이딩뷰식)로,
  - 캔들이 없거나 변환할 수 없는 요소(히트맵·가로 막대 등)가 있는 그림은 지금처럼 plotly 로
그린다. 페이지의 그림 만드는 코드는 그대로 둔다.

변환 규칙: Candlestick → 캔들 / Scatter lines → 선(점이 적으면 선분) / Scatter markers(+text) → 봉 위·아래 마커 /
Bar → 막대(가격 축에 겹친 거래량은 아래 20%) / shape line → 선분·수평선·세로 띠 / shape rect → 가격 박스·세로 띠 /
annotation → 글자 마커, subplot 제목 → Pane 제목 / subplot 행 → Pane(높이는 domain 비율).
x 값은 캔들 봉에 가장 가까운 봉으로 맞춘다(시간축이 봉 순서라 빈 날·빈 시간이 없다).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from visualization.adapters.lightweight_charts import DOWN, UP, Pane, render

_PALETTE = ["#42a5f5", "#ff9100", "#ab47bc", "#26a69a", "#ef5350", "#8d6e63", "#78909c", "#d4e157", "#ec407a", "#5c6bc0"]
_DASH = {"solid": "solid", "dot": "dot", "dash": "dash", "longdash": "longdash", "dashdot": "dashdot", "longdashdot": "dashdot"}
_NAMED = {"red": "239,83,80", "green": "38,166,154", "blue": "66,165,245", "orange": "255,145,0", "gray": "144,164,174", "grey": "144,164,174",
          "black": "0,0,0", "white": "255,255,255", "yellow": "253,216,53", "purple": "171,71,188", "cyan": "38,198,218", "magenta": "236,64,122",
          "lime": "0,230,118", "gold": "255,193,7", "pink": "244,143,177", "brown": "141,110,99", "navy": "40,53,147", "teal": "0,150,136",
          "lightgray": "207,216,220", "lightgrey": "207,216,220", "darkgray": "96,125,139", "darkgrey": "96,125,139", "royalblue": "65,105,225",
          "deepskyblue": "0,191,255", "dodgerblue": "30,144,255", "crimson": "220,20,60", "limegreen": "50,205,50", "darkorange": "255,140,0",
          "skyblue": "135,206,235", "silver": "192,192,192", "violet": "238,130,238", "salmon": "250,128,114", "khaki": "240,230,140"}


class Unsupported(Exception):
    pass


def _rgba(color, alpha: float | None = None) -> str:
    """plotly 색 문자열 → css 색. alpha 가 있으면 곱해 rgba 로."""
    if not isinstance(color, str) or not color:
        color = "#90a4ae"
    c = color.strip().lower()
    if alpha is None or alpha >= 0.999:
        return f"rgb({_NAMED[c]})" if c in _NAMED else color
    m = re.match(r"rgba?\(([^)]+)\)", c)
    if m:
        parts = [x.strip() for x in m.group(1).split(",")]
        a0 = float(parts[3]) if len(parts) > 3 else 1.0
        return f"rgba({parts[0]},{parts[1]},{parts[2]},{a0 * alpha:.3f})"
    if c.startswith("#") and len(c) in (4, 7):
        h = c[1:] if len(c) == 7 else "".join(ch * 2 for ch in c[1:])
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha:.3f})"
    if c in _NAMED:
        return f"rgba({_NAMED[c]},{alpha:.3f})"
    return color


def _times(x) -> pd.DatetimeIndex:
    vals = list(x) if x is not None else []
    idx = pd.to_datetime(pd.Index(vals), errors="coerce")
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(vals), errors="coerce", utc=True)).tz_convert(None)
    if idx.tz is not None:
        idx = idx.tz_localize(None)       # 벽시계 시각 그대로
    return idx


def _one_time(v):
    if v is None:
        return pd.NaT
    try:
        t = pd.Timestamp(v)
    except (ValueError, TypeError):
        return pd.NaT
    return t.tz_localize(None) if t.tzinfo is not None else t


def _clean(text) -> str:
    t = re.sub(r"<[^>]+>", " ", str(text or "")).strip()
    return t if len(t) <= 42 else t[:41] + "…"


def _axis_key(ref: str | None) -> str:
    ref = (ref or "y").split(" ")[0]
    return "yaxis" if ref == "y" else "yaxis" + ref[1:]


def figure_to_panes(fig) -> tuple[list[Pane], pd.DatetimeIndex, bool, str | None]:
    """(panes, index, intraday, theme). 변환 불가면 Unsupported."""
    traces = list(fig.data)
    candles = [t for t in traces if t.type in ("candlestick", "ohlc")]
    if not candles:
        raise Unsupported("캔들 없음")
    base = pd.DatetimeIndex([])
    for t in candles:
        base = base.union(_times(t.x).dropna())
    base = base.sort_values().unique()
    if len(base) < 2:
        raise Unsupported("봉이 너무 적음")
    intraday = bool((base != base.normalize()).any())
    lay = fig.layout

    def snap(ts: pd.DatetimeIndex) -> np.ndarray:
        return base.get_indexer(ts, method="nearest")

    # ---- 축 → Pane
    axes: dict[str, dict] = {}
    for t in traces:
        k = _axis_key(getattr(t, "yaxis", None))
        if k not in axes:
            ax = lay[k] if k in lay else None
            ov = getattr(ax, "overlaying", None) if ax is not None else None
            dom = list(ax.domain) if ax is not None and ax.domain else [0.0, 1.0]
            axes[k] = {"domain": dom, "overlay": _axis_key(ov) if ov else None, "log": (getattr(ax, "type", None) == "log") if ax is not None else False}
    roots = {k: v for k, v in axes.items() if not v["overlay"]}
    for k, v in axes.items():
        if v["overlay"] and v["overlay"] not in roots:
            roots[v["overlay"]] = {"domain": [0.0, 1.0], "overlay": None, "log": False}
    order = sorted(roots, key=lambda k: -roots[k]["domain"][1])
    total_h = int(lay.height or 600)
    span = sum(max(0.02, roots[k]["domain"][1] - roots[k]["domain"][0]) for k in order) or 1.0
    panes = {k: Pane(height=max(90, int(total_h * max(0.02, roots[k]["domain"][1] - roots[k]["domain"][0]) / span)), log=roots[k]["log"]) for k in order}
    hilo: dict[str, pd.DataFrame] = {}

    def pane_of(yref) -> tuple[Pane, str, bool]:
        k = _axis_key(yref)
        ov = axes.get(k, {}).get("overlay")
        root = ov or k
        if root not in panes:
            root = order[0]
        return panes[root], root, bool(ov)

    pal = 0
    for t in traces:
        pane, root, overlay = pane_of(getattr(t, "yaxis", None))
        if getattr(t, "visible", True) in (False, "legendonly"):
            continue
        op = t.opacity if t.opacity is not None else 1.0
        if t.type in ("candlestick", "ohlc"):
            ts = _times(t.x)
            df = pd.DataFrame({"Open": np.asarray(t.open, float), "High": np.asarray(t.high, float), "Low": np.asarray(t.low, float),
                               "Close": np.asarray(t.close, float)}, index=ts)
            df = df[~df.index.isna()]
            df = df[~df.index.duplicated(keep="last")].sort_index()
            pane.candles = df
            inc = getattr(getattr(t.increasing, "line", None), "color", None); dec = getattr(getattr(t.decreasing, "line", None), "color", None)
            pane.up_color = _rgba(inc) if inc else UP; pane.down_color = _rgba(dec) if dec else DOWN
            hilo[root] = df
        elif t.type == "bar":
            if getattr(t, "orientation", None) == "h" or t.base is not None:
                raise Unsupported("가로·누적 막대")
            ts = _times(t.x); y = np.asarray(t.y, float)
            ok = ~ts.isna() & np.isfinite(y)
            p_ = snap(ts[ok]); col = t.marker.color
            if isinstance(col, (list, tuple, np.ndarray)):
                cols = [_rgba(c if isinstance(c, str) else "#78909c", op if op < 1 else None) for c in np.asarray(col, dtype=object)[ok]]
            else:
                cols = [_rgba(col if isinstance(col, str) else "#78909c", op if op < 1 else 0.75)] * int(ok.sum())
            hd = pd.DataFrame({"v": y[ok], "c": cols}, index=base[p_])
            hd = hd[~hd.index.duplicated(keep="last")].sort_index()
            pane.hist.append(dict(name=_clean(t.name) if t.showlegend is not False else "", data=hd["v"], colors=hd["c"].tolist(),
                                  scale="overlay" if overlay else None))
        elif t.type in ("scatter", "scattergl"):
            mode = t.mode or "lines"
            xs = list(t.x) if t.x is not None else []; ys = list(t.y) if t.y is not None else []
            if not xs:
                continue
            color = getattr(t.line, "color", None) if "lines" in mode else None
            mcol = getattr(t.marker, "color", None)
            if color is None:
                color = mcol if isinstance(mcol, str) else None
            if color is None:
                color = _PALETTE[pal % len(_PALETTE)]; pal += 1
            name = _clean(t.name) if (t.name and t.showlegend is not False) else ""
            if "lines" in mode:
                # None 으로 끊긴 조각마다
                chunks, cur = [], []
                for xv, yv in zip(xs, ys):
                    tt = _one_time(xv)
                    try:
                        fv = float(yv) if yv is not None else np.nan
                    except (TypeError, ValueError):
                        fv = np.nan
                    if pd.isna(tt) or not np.isfinite(fv):
                        if cur:
                            chunks.append(cur); cur = []
                    else:
                        cur.append((tt, fv))
                if cur:
                    chunks.append(cur)
                width = float(getattr(t.line, "width", None) or 1.5); style = _DASH.get(getattr(t.line, "dash", None) or "solid", "dash")
                col = _rgba(color, op if op < 1 else None)
                fill = getattr(t, "fill", None)
                first = True
                for ch in chunks:
                    pidx = snap(pd.DatetimeIndex([c[0] for c in ch])); vals = [c[1] for c in ch]
                    if fill == "toself" and len(ch) >= 4:
                        fc = getattr(t, "fillcolor", None) or _rgba(color, 0.15)
                        pane.boxes.append(dict(t0=base[pidx.min()], t1=base[pidx.max()], p0=min(vals), p1=max(vals), fill=_rgba(fc, op if op < 1 else None),
                                               line=col if width > 0 else None))
                        continue
                    ded = dict(zip(pidx.tolist(), vals))
                    if len(ded) == 1:
                        continue
                    nm = name if first else ""
                    if len(chunks) == 1 and len(ded) >= max(10, int(0.3 * len(base))):
                        s = pd.Series(list(ded.values()), index=base[list(ded.keys())]).sort_index()
                        pane.lines.append(dict(name=nm, data=s, color=col, width=width, style=style, legend=bool(nm), scale="left" if overlay else None))
                    else:
                        pane.segments.append(dict(points=[(base[i], v) for i, v in ded.items()], color=col, width=width, style=style, name=nm or None,
                                                  scale="left" if overlay else None))
                    first = False
            elif "markers" in mode or "text" in mode:
                ts = _times(xs); sym = getattr(t.marker, "symbol", None)
                texts = t.text if "text" in mode and t.text is not None else None
                if isinstance(texts, str):
                    texts = [texts] * len(xs)
                ref = hilo.get(root)
                for i, (tt, yv) in enumerate(zip(ts, ys)):
                    if pd.isna(tt) or yv is None:
                        continue
                    try:
                        fv = float(yv)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(fv):
                        continue
                    bt = base[snap(pd.DatetimeIndex([tt]))[0]]
                    sm = str(sym[i] if isinstance(sym, (list, tuple, np.ndarray)) else (sym or "circle"))
                    c = mcol[i] if isinstance(mcol, (list, tuple, np.ndarray)) else (mcol if isinstance(mcol, str) else color)
                    if "up" in sm:
                        shape, posn = "arrowUp", "below"
                    elif "down" in sm:
                        shape, posn = "arrowDown", "above"
                    else:
                        shape = "square" if any(k in sm for k in ("x", "cross", "square", "diamond")) else "circle"
                        mid = float((ref.at[bt, "High"] + ref.at[bt, "Low"]) / 2) if ref is not None and bt in ref.index else fv
                        posn = "above" if fv >= mid else "below"
                    pane.markers.append(dict(time=bt, position=posn, shape=shape, color=_rgba(c if isinstance(c, str) else color),
                                             text=_clean(texts[i]) if texts is not None and i < len(texts) else "", size=1 if "markers" in mode else 0))
                if name:
                    pane.legend_extra.append(dict(name=name, color=_rgba(mcol if isinstance(mcol, str) else color)))
        else:
            raise Unsupported(f"trace {t.type}")

    # ---- shapes
    lo_t, hi_t = base[0], base[-1]
    for sh in (lay.shapes or ()):
        xref = sh.xref or "x"; yref = sh.yref or "y"
        x_dom = "domain" in xref or xref == "paper"; y_dom = "domain" in yref or yref == "paper"
        pane, root, _ = pane_of(yref if yref != "paper" else None)
        col = _rgba(getattr(sh.line, "color", None) or "#90a4ae", sh.opacity if sh.opacity is not None and sh.opacity < 1 else None)
        width = float(getattr(sh.line, "width", None) if getattr(sh.line, "width", None) is not None else 1.0)
        style = _DASH.get(getattr(sh.line, "dash", None) or "solid", "dash")
        t0 = lo_t if x_dom else _one_time(sh.x0); t1 = hi_t if x_dom else _one_time(sh.x1)
        if pd.isna(t0) or pd.isna(t1):
            continue
        b0, b1 = base[snap(pd.DatetimeIndex([t0]))[0]], base[snap(pd.DatetimeIndex([t1]))[0]]
        if sh.type == "line":
            if y_dom:                                   # 세로선
                pane.shades.append((b0, b0, _rgba(getattr(sh.line, "color", None) or "#90a4ae", 0.35)))
            elif x_dom and sh.y0 == sh.y1:              # 수평선
                pane.hlines.append(dict(price=sh.y0, color=col, style=style, width=max(1, int(round(width)))))
            elif b0 != b1:
                pane.segments.append(dict(points=[(b0, float(sh.y0)), (b1, float(sh.y1))], color=col, width=max(1.0, width), style=style))
        elif sh.type == "rect":
            fc = sh.fillcolor
            alpha = sh.opacity if sh.opacity is not None else None
            if y_dom:
                pane.shades.append((b0, b1, _rgba(fc or "#90a4ae", alpha if alpha is not None else 0.15)))
            else:
                pane.boxes.append(dict(t0=b0, t1=b1, p0=float(sh.y0), p1=float(sh.y1), fill=_rgba(fc, alpha) if fc else None, line=col if width > 0 else None))

    # ---- annotations: subplot 제목 → Pane 제목, 가격 위치 글자 → 마커
    for an in (lay.annotations or ()):
        txt = _clean(an.text)
        if not txt:
            continue
        xref = an.xref or "x"; yref = an.yref or "y"
        if "paper" in (xref, yref) or "domain" in xref or "domain" in yref:
            if yref == "paper" and an.y is not None and not an.showarrow:
                k = min(order, key=lambda k: abs(roots[k]["domain"][1] - float(an.y)))
                if abs(roots[k]["domain"][1] - float(an.y)) < 0.06 and not panes[k].title:
                    panes[k].title = txt
            elif "domain" in xref and "domain" not in yref and yref != "paper" and an.y is not None:      # add_hline 의 annotation_text
                pane, _, _ = pane_of(yref)
                for h in pane.hlines:
                    if abs(h["price"] - float(an.y)) < 1e-9 and not h.get("title"):
                        h["title"] = txt
            continue
        tt = _one_time(an.x)
        if pd.isna(tt) or an.y is None:
            continue
        pane, root, _ = pane_of(yref)
        bt = base[snap(pd.DatetimeIndex([tt]))[0]]
        ref = hilo.get(root)
        mid = float((ref.at[bt, "High"] + ref.at[bt, "Low"]) / 2) if ref is not None and bt in ref.index else float(an.y)
        fc = getattr(an.font, "color", None)
        pane.markers.append(dict(time=bt, position="above" if float(an.y) >= mid else "below", shape="circle", size=0, color=_rgba(fc or "#b0bec5"), text=txt))

    title = _clean(getattr(lay.title, "text", None))
    out = [panes[k] for k in order if panes[k].candles is not None or panes[k].lines or panes[k].hist or panes[k].segments]
    if not out:
        raise Unsupported("그릴 것이 없음")
    if title:
        out[0].title = title if not out[0].title else f"{title} · {out[0].title}"
    tmpl = lay.template
    bg = getattr(getattr(tmpl, "layout", None), "paper_bgcolor", None) if tmpl is not None else None
    theme = None
    if isinstance(lay.paper_bgcolor or bg, str):
        c = (lay.paper_bgcolor or bg).lower()
        theme = "dark" if c in ("#111111", "rgb(17,17,17)", "black", "#000", "#000000", "#0e1117") else None
    return out, base, intraday, theme


def st_candles(fig, theme: str | None = None, **plotly_kwargs):
    """`st.plotly_chart` 자리에 쓴다. 캔들 그림이면 Lightweight Charts, 아니면(또는 변환 실패) plotly 로 그린다."""
    import streamlit as st
    try:
        if plotly_kwargs.get("on_select") not in (None, "ignore"):
            raise Unsupported("선택 이벤트는 plotly 만")
        panes, index, intraday, auto = figure_to_panes(fig)
    except Unsupported:
        return st.plotly_chart(fig, **plotly_kwargs)
    except Exception as e:      # 변환기가 모르는 모양 — 화면은 기존 plotly 로 지킨다
        st.caption(f"(차트 변환 실패로 plotly 로 표시: {type(e).__name__}: {e})")
        return st.plotly_chart(fig, **plotly_kwargs)
    render(panes, index, intraday=intraday, theme=theme or auto)
    return None
