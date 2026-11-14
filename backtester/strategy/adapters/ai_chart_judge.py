"""AI 차트 판정 — 숫자 규칙 대신 모델이 차트 전체를 보고 진입·청산을 정한다 (LLM 을 모르는 순수 pandas + matplotlib 부분).

* ``ChartSpec`` — 패턴마다 다른 차트 구성(봉 주기, VWAP·EMA, 세션 음영).
* ``build_packet`` — 기준 봉(as_of, **마감된 봉**)까지만 잘라 모델에게 줄 자료 묶음을 만든다: 차트 PNG, 큰 그림 PNG(분봉이면 일봉, 일봉이면 주봉),
  최근 봉 표, 요약 사실. 기준 봉 뒤의 봉은 어떤 경로로도 들어가지 않는다(지표도 자른 뒤 계산).
  ``blind=True`` 면 티커를 지우고 날짜를 D-1/D0(일봉은 T-3/T0)로, 가격을 지수(첫 종가 = 100)로 바꿔 모델이 "그날을 기억"해서 맞히는 것을 막는다.
* ``Judgement`` / ``Manage`` — 모델이 돌려줄 구조화 답(진입 판단 / 보유 중 관리). 가격은 packet 눈금이고 ``to_real`` 로 실제 가격이 된다.
* ``settle`` — 판정 하나를 기준 봉 뒤의 봉으로 정산(단건용). 기간 백테스트는 ``ai_pattern_backtest.run_backtest``.

패턴별 원칙 글은 ``ai_patterns.py``, LLM 호출은 ``claude_chart_judge.py``, 페이지는 ``pages/_shared/ai_pattern_page.py``.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from strategy.adapters.vwap_sr_breakout import atr_series, session_vwap

NY = "America/New_York"
EMA_COLORS = ("#1e88e5", "#8e24aa", "#6d4c41", "#00897b")


@dataclass(frozen=True)
class ChartSpec:
    interval: str = "15m"                 # "1m" "5m" "15m" "1h" "1d"
    vwap: bool = True                     # 세션 VWAP (분봉만)
    emas: tuple[int, ...] = ()            # 차트에 그릴 EMA 기간
    session: str = "us"                   # "us" = 미국 주식(정규장 09:30~16:00, 시간외 음영) | "crypto" = 24시간
    view_bars: int = 160                  # 차트에 보일 봉 수
    table_bars: int = 64                  # 표로 줄 봉 수

    @property
    def intraday(self) -> bool:
        return not self.interval.endswith("d")


# ---------------------------------------------------------------------------
# 모델이 돌려주는 답
# ---------------------------------------------------------------------------
class Level(BaseModel):
    """모델이 차트에 그은 선 하나. 수평선이면 price 만, 기울어진 선이면 시작점(bars_ago_start 봉 전, price_start)도."""

    kind: Literal["resistance", "support", "trend_resistance", "trend_support"]
    price: float = Field(description="기준 봉 시점의 선 값 (자료에 나온 가격 눈금 그대로)")
    price_start: Optional[float] = Field(default=None, description="기울어진 선의 시작점 가격. 수평선이면 null")
    bars_ago_start: Optional[int] = Field(default=None, description="선이 시작하는 봉이 기준 봉에서 몇 봉 전인지. 수평선이면 그 수준이 처음 확인된 봉")
    note: str = Field(description="왜 이 선인지 한 줄 (한국어)")


_TEXT_FIELDS = ("setup", "market_read", "structure_read", "indicator_read", "volume_read", "risks", "wait_for", "summary")


def _txt_converter(k: float, text_ref: float | None):
    """설명 글 속의 지수 눈금 가격을 실제 가격으로: text_ref 의 0.8~1.25 배 안인 숫자만 (단위가 붙은 봉 수·퍼센트는 그대로)."""
    def txt(s: str) -> str:
        if not text_ref or not s:
            return s
        def rep(m):
            v = float(m.group(0))
            return f"{v * k:.2f}" if 0.8 * text_ref <= v <= 1.25 * text_ref else m.group(0)
        # 소수점 있는 수, 또는 뒤에 단위(봉·일·% 등)가 안 붙은 정수
        return re.sub(r"(?<![\d.])\d+\.\d+(?![\d%])|(?<![\d.\-+])\d{2,}(?![\d.%]|\s?(?:봉|일|개|배|회|번|분|시|주|년|월|R|EMA|SMA))", rep, s)
    return txt


class Judgement(BaseModel):
    decision: Literal["enter_long", "wait", "no_trade"] = Field(
        description="enter_long = 지금 조건이 갖춰졌다(바로 사거나, 적어 둔 조건이 되면 산다). wait = 모양은 만들어지는 중이라 더 지켜본다. no_trade = 이 자리는 건드리지 않는다")
    setup: str = Field(description="자리의 이름을 짧게 (예: '상승 삼각형 돌파', '추세 한가운데 — 자리 없음')")
    confidence: int = Field(description="이 결정에 대한 확신 0~100")
    entry_type: Literal["market", "breakout_close", "limit", "none"] = Field(
        description="market = 기준 봉 마감 직후(다음 봉 시가) 매수. breakout_close = 이후 어떤 봉이 entry_trigger 위로 마감하면 그 종가에 매수. "
                    "limit = 이후 가격이 entry_trigger 까지 내려오면 그 가격에 매수(눌림 매수). none = 진입 없음")
    entry_trigger: Optional[float] = Field(default=None, description="breakout_close 면 넘어야 하는 가격, limit 이면 기다리는 매수 가격")
    trigger_valid_bars: int = Field(default=0, description="조건부 진입이 유효한 봉 수 (그 안에 안 되면 취소). market/none 이면 0")
    stop: Optional[float] = Field(default=None, description="손절 가격. 진입이 없으면 null")
    stop_basis: Literal["close", "touch"] = Field(default="close", description="close = 봉이 손절가 아래로 마감해야 손절. touch = 닿기만 해도 손절")
    target1: Optional[float] = Field(default=None, description="첫 목표가 (절반 청산). 목표 없이 관리로 청산할 계획이면 null")
    target2: Optional[float] = Field(default=None, description="둘째 목표가 (나머지 청산). 없으면 null")
    levels: list[Level] = Field(default_factory=list, description="판단에 쓴 지지·저항·추세선 (가장 중요한 것부터, 최대 6개)")
    market_read: str = Field(description="큰 그림: 상위 주기 흐름과 지금 장의 성격 (한국어 2~3문장)")
    structure_read: str = Field(description="차트 구조: 이 패턴의 모양이 어디까지 만들어졌는지 (한국어 2~4문장)")
    indicator_read: str = Field(description="가격과 차트에 그려진 지표(VWAP·이동평균선)의 관계 (한국어 1~2문장)")
    volume_read: str = Field(description="거래량이 말해 주는 것 (한국어 1~2문장)")
    risks: str = Field(description="이 판단이 틀릴 수 있는 이유 (한국어 1~3문장)")
    wait_for: str = Field(description="wait/no_trade 일 때 무엇이 나오면 생각을 바꿀지. enter_long 이면 무엇이 나오면 계획을 버릴지 (한국어)")
    summary: str = Field(description="결정과 이유를 초보자도 알 수 있게 한 문단으로 (한국어)")

    def to_real(self, scale: float, text_ref: float | None = None) -> "Judgement":
        """packet 가격 눈금(지수) → 실제 가격. ``text_ref``(지수 눈금의 기준 봉 종가)를 주면 설명 글 속 가격도 바꾼다."""
        if scale == 1.0:
            return self
        k = 1.0 / scale
        mul = lambda v: None if v is None else float(v) * k  # noqa: E731
        txt = _txt_converter(k, text_ref)
        return self.model_copy(update={
            **{f: txt(getattr(self, f)) for f in _TEXT_FIELDS},
            "entry_trigger": mul(self.entry_trigger), "stop": mul(self.stop), "target1": mul(self.target1), "target2": mul(self.target2),
            "levels": [lv.model_copy(update={"price": lv.price * k, "price_start": mul(lv.price_start), "note": txt(lv.note)}) for lv in self.levels]})


class Manage(BaseModel):
    """보유 중 판단: 계속 들고 갈지, 다음 봉 시가에 나갈지, 손절·목표를 옮길지."""

    action: Literal["hold", "exit"] = Field(description="hold = 계속 보유. exit = 다음 봉 시가에 전량 청산")
    new_stop: Optional[float] = Field(default=None, description="손절을 올릴 가격 (현재 손절보다 높고 현재가보다 낮아야 함). 그대로면 null")
    new_target: Optional[float] = Field(default=None, description="남은 물량의 목표가를 바꿀 가격. 그대로면 null")
    confidence: int = Field(description="이 결정에 대한 확신 0~100")
    reason: str = Field(description="차트에서 무엇을 보고 그렇게 정했는지 (한국어 2~3문장)")

    def to_real(self, scale: float, text_ref: float | None = None) -> "Manage":
        if scale == 1.0:
            return self
        k = 1.0 / scale
        mul = lambda v: None if v is None else float(v) * k  # noqa: E731
        return self.model_copy(update={"new_stop": mul(self.new_stop), "new_target": mul(self.new_target), "reason": _txt_converter(k, text_ref)(self.reason)})


class Signal(BaseModel):
    """한 번에 훑기(scan)에서 모델이 찍은 패턴 자리 하나."""

    bar: str = Field(description="신호 봉의 이름 — 봉 표의 bar 열에 있는 글자 그대로 (예: 'D-1 10:30', 'T-12', '2026-10-02'). 이 봉이 마감된 뒤 다음 봉 시가에 매수한다")
    setup: str = Field(description="자리의 이름을 짧게 (한국어)")
    confidence: int = Field(description="이 자리가 패턴에 얼마나 잘 맞는지 0~100")
    stop: float = Field(description="손절 가격 — 신호 봉까지의 차트만으로 정할 수 있는 값")
    stop_basis: Literal["close", "touch"] = Field(default="close", description="close = 봉이 손절가 아래로 마감해야 손절. touch = 닿기만 해도 손절")
    target1: Optional[float] = Field(default=None, description="첫 목표가 (절반 청산) — 신호 봉까지의 차트만으로 정할 수 있는 값. 없으면 null")
    target2: Optional[float] = Field(default=None, description="둘째 목표가. 없으면 null")
    exit_bar: Optional[str] = Field(default=None, description="패턴의 청산 원칙에 맞는 신호가 처음 나온 봉의 이름(그 봉 종가에 청산). 그런 신호가 구간 안에 없으면 null")
    exit_reason: str = Field(default="", description="exit_bar 를 고른 이유 (한국어 한 줄). 없으면 빈 글")
    levels: list[Level] = Field(default_factory=list, description="이 자리에서 본 지지·저항·추세선 (최대 3개). bars_ago_start 는 신호 봉 기준")
    note: str = Field(description="차트에서 무엇을 보고 이 자리를 골랐는지 (한국어 2~3문장)")


class ScanResult(BaseModel):
    overview: str = Field(description="구간 전체의 흐름과 이 패턴이 얼마나 자주·깨끗하게 나왔는지 (한국어 3~5문장)")
    signals: list[Signal] = Field(default_factory=list, description="찾을 구간 안의 패턴 자리 전부, 시간 순서대로. 없으면 빈 목록")

    def to_real(self, scale: float, text_ref: float | None = None) -> "ScanResult":
        if scale == 1.0:
            return self
        k = 1.0 / scale
        mul = lambda v: None if v is None else float(v) * k  # noqa: E731
        txt = _txt_converter(k, text_ref)
        return self.model_copy(update={"overview": txt(self.overview), "signals": [
            sg.model_copy(update={"stop": sg.stop * k, "target1": mul(sg.target1), "target2": mul(sg.target2), "note": txt(sg.note), "setup": txt(sg.setup),
                                  "exit_reason": txt(sg.exit_reason),
                                  "levels": [lv.model_copy(update={"price": lv.price * k, "price_start": mul(lv.price_start), "note": txt(lv.note)}) for lv in sg.levels]})
            for sg in self.signals]})


# ---------------------------------------------------------------------------
# 모델에게 줄 자료 묶음
# ---------------------------------------------------------------------------
@dataclass
class Packet:
    as_of: pd.Timestamp            # 기준 봉 (마감된 마지막 봉)
    scale: float                   # packet 가격 = 실제 가격 × scale (blind 아니면 1)
    blind: bool
    name: str                      # 모델에게 보이는 종목 이름 (blind 면 "종목 X")
    chart_png: bytes
    context_png: Optional[bytes]   # 큰 그림 (분봉 → 일봉, 일봉 → 주봉)
    bars_csv: str                  # 최근 봉 표
    facts: dict = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)   # 차트·표의 봉 이름 (마지막이 기준 봉)
    n_bars: int = 0
    interval: str = "15m"
    last: float = 0.0              # packet 눈금의 기준 봉 종가


def _naive(idx: pd.DatetimeIndex, intraday: bool) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if not intraday:
        return idx.tz_localize(None).normalize() if idx.tz is None else idx.tz_convert(NY).tz_localize(None).normalize()
    return (idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)).tz_localize(None)


def _labels(x: pd.DatetimeIndex, blind: bool, intraday: bool) -> list[str]:
    n = len(x)
    if not intraday:
        return [("T0" if i == n - 1 else f"T-{n - 1 - i}") for i in range(n)] if blind else [f"{t:%Y-%m-%d}" for t in x]
    if not blind:
        return [f"{t:%m-%d %H:%M}" for t in x]
    days = x.normalize()
    uniq = list(pd.Index(days).unique())
    pos = {d: i - (len(uniq) - 1) for i, d in enumerate(uniq)}
    return [f"D{pos[d]:+d} {t:%H:%M}" if pos[d] else f"D0 {t:%H:%M}" for d, t in zip(days, x)]


def _prec(last: float) -> int:
    return 2 if last >= 20 else 3 if last >= 2 else 4


def _candles(ax, o, h, l, c, width=0.7):
    up = c >= o
    xs = np.arange(len(o))
    for mask, col in ((up, "#26a69a"), (~up, "#ef5350")):
        ax.vlines(xs[mask], l[mask], h[mask], color=col, linewidth=0.8, zorder=2)
        ax.bar(xs[mask], np.maximum(np.abs(c - o)[mask], (h.max() - l.min()) * 1e-4), bottom=np.minimum(o, c)[mask], width=width, color=col, zorder=3, linewidth=0)


def _figure(nrows_ratio, figsize):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=100)              # pyplot 을 안 써서 스레드에서도 안전
    FigureCanvasAgg(fig)
    axs = fig.subplots(len(nrows_ratio), 1, sharex=True, gridspec_kw={"height_ratios": nrows_ratio, "hspace": 0.03})
    return fig, axs


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _chart_png(v: pd.DataFrame, x: pd.DatetimeIndex, labels: list[str], title: str, spec: ChartSpec, marks: list[tuple[float, str, str]] | None = None) -> bytes:
    o, h, l, c = (v[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    n = len(v); xs = np.arange(n)
    fig, (ax, av) = _figure([4, 1], (15.6, 8.8))
    new_day = np.array([], dtype=int)
    if spec.intraday:
        day = x.normalize().to_numpy()
        new_day = np.flatnonzero(day[1:] != day[:-1]) + 1
        if spec.session == "us":
            mins = x.hour * 60 + x.minute
            ext = np.flatnonzero((mins < 9 * 60 + 30) | (mins >= 16 * 60))       # 시간외 봉은 배경을 회색으로
            for a in (ax, av):
                for i in ext:
                    a.axvspan(i - 0.5, i + 0.5, color="#eceff1", zorder=0, linewidth=0)
        for i in new_day:
            ax.axvline(i - 0.5, color="#90a4ae", linewidth=0.8, linestyle="--", zorder=1)
            av.axvline(i - 0.5, color="#90a4ae", linewidth=0.8, linestyle="--", zorder=1)
    _candles(ax, o, h, l, c)
    if spec.intraday and spec.vwap and "vwap" in v:
        vw = v["vwap"].to_numpy(dtype=float)
        for s in np.split(xs, new_day):                                          # VWAP 은 날마다 리셋 — 날 사이를 잇지 않는다
            ax.plot(s, vw[s], color="#ff9100", linewidth=1.8, zorder=4)
        ax.plot([], [], color="#ff9100", linewidth=1.8, label="session VWAP")
    for k, n_ema in enumerate(spec.emas):
        if f"ema{n_ema}" in v:
            ax.plot(xs, v[f"ema{n_ema}"].to_numpy(dtype=float), color=EMA_COLORS[k % len(EMA_COLORS)], linewidth=1.3, zorder=4, label=f"EMA{n_ema}")
    for price, lab, col in marks or []:
        ax.axhline(price, color=col, linewidth=1.1, linestyle="--", zorder=1)
        ax.annotate(f"{lab} {price:.{_prec(price)}f}", (0, price), xytext=(4, 3), textcoords="offset points", fontsize=9, color=col)
    ax.axhline(c[-1], color="#1565c0", linewidth=0.7, linestyle=":", zorder=1)
    ax.annotate(f"last close {c[-1]:.{_prec(c[-1])}f}", (n - 1, c[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9, color="#1565c0", va="center")
    ax.set_xlim(-1, n + max(4, n // 14))
    lo_y, hi_y = l.min(), h.max()
    for price, _, _ in marks or []:
        lo_y, hi_y = min(lo_y, price), max(hi_y, price)
    pad = (hi_y - lo_y) * 0.04
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.grid(True, color="#cfd8dc", linewidth=0.5, alpha=0.7); ax.yaxis.tick_right()
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_title(title, fontsize=11, loc="left")
    av.bar(xs, v["Volume"].to_numpy(dtype=float), width=0.7, color=np.where(c >= o, "#26a69a", "#ef5350"), linewidth=0)
    av.plot(xs, v["vol_avg"].to_numpy(dtype=float), color="#5c6bc0", linewidth=1.0, label="20-bar avg volume")
    av.legend(loc="upper left", fontsize=8, frameon=False); av.yaxis.tick_right(); av.grid(True, color="#cfd8dc", linewidth=0.5, alpha=0.7)
    step = max(1, n // 24)
    ticks = sorted(set(list(range(0, n, step)) + [n - 1]))
    av.set_xticks(ticks); av.set_xticklabels([labels[i] for i in ticks], rotation=60, ha="right", fontsize=8)
    return _png(fig)


def _context_png(d: pd.DataFrame, title: str, unit: str) -> bytes:
    o, h, l, c = (d[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    n = len(d); xs = np.arange(n)
    fig, (ax, av) = _figure([4, 1], (12.0, 6.4))
    _candles(ax, o, h, l, c)
    for w, col in ((20, "#ff9100"), (50, "#5c6bc0")):
        if "sma%d" % w in d:
            ax.plot(xs, d["sma%d" % w].to_numpy(dtype=float), color=col, linewidth=1.3, label=f"SMA{w}")
    ax.legend(loc="upper left", fontsize=9, frameon=False); ax.grid(True, color="#cfd8dc", linewidth=0.5, alpha=0.7); ax.yaxis.tick_right()
    ax.set_title(title, fontsize=11, loc="left")
    av.bar(xs, d["Volume"].to_numpy(dtype=float), width=0.7, color=np.where(c >= o, "#26a69a", "#ef5350"), linewidth=0); av.yaxis.tick_right()
    step = max(1, n // 12); ticks = list(range(n - 1, -1, -step))[::-1]
    av.set_xticks(ticks); av.set_xticklabels([f"{i - n}{unit}" for i in ticks], fontsize=8)
    return _png(fig)


def prepare(bars: pd.DataFrame, spec: ChartSpec | None = None) -> pd.DataFrame:
    """봉에 지표를 붙인다(세션 VWAP·EMA·ATR(20)·직전 20봉 평균 거래량). 모두 그 봉까지의 값만 써서 뒤 봉이 바뀌어도 앞 값은 그대로다."""
    spec = spec or ChartSpec()
    f = bars[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    if spec.intraday and spec.vwap:
        f["vwap"] = session_vwap(f)
    for n in spec.emas:
        f[f"ema{n}"] = f["Close"].ewm(span=n, adjust=False, min_periods=min(n, 5)).mean()
    f["atr"] = atr_series(f, 20)
    f["vol_avg"] = f["Volume"].rolling(20, min_periods=5).mean().shift(1)
    return f


def _weekly(d: pd.DataFrame) -> pd.DataFrame:
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    return w


def build_packet(bars: pd.DataFrame, as_of, *, spec: ChartSpec | None = None, context: pd.DataFrame | None = None, ticker: str = "", blind: bool = True,
                 position: dict | None = None, daily: pd.DataFrame | None = None) -> Packet:
    """기준 봉 ``as_of``(포함)까지의 봉으로 자료 묶음을 만든다.

    ``context`` = 큰 그림용 일봉(분봉 패턴일 때; 기준일 **전날까지만** 쓴다). 일봉 패턴이면 bars 자체에서 완성된 주봉을 만든다.
    ``position`` = 보유 중 관리 판단이면 {"entry", "stop", "target", "bars_held"} (실제 가격) — 차트에 선으로, 요약에 숫자로 들어간다.
    """
    spec = spec or ChartSpec()
    context = context if context is not None else daily
    as_of = pd.Timestamp(as_of)
    f = prepare(bars.loc[:as_of], spec)                 # 여기서 잘라 낸 뒤에야 지표를 계산 — 뒤 봉은 계산에도 안 들어간다
    if len(f) < 30:
        raise ValueError(f"기준 봉까지 봉이 {len(f)}개뿐입니다 (30개 이상 필요)")
    v = f.iloc[-spec.view_bars:]
    x = _naive(v.index, spec.intraday)
    scale = 100.0 / float(v["Close"].iloc[0]) if blind else 1.0
    px = [c for c in v.columns if c in ("Open", "High", "Low", "Close", "vwap", "atr") or c.startswith("ema")]
    vs = v.copy(); vs[px] = vs[px] * scale
    vol_unit = float(vs["Volume"].replace(0, np.nan).median()) if blind else 1.0   # blind 면 거래량도 중앙값 = 1 로
    vol_unit = vol_unit if np.isfinite(vol_unit) and vol_unit > 0 else 1.0
    vs[["Volume", "vol_avg"]] = vs[["Volume", "vol_avg"]] / vol_unit
    labels = _labels(x, blind, spec.intraday)
    name = "종목 X" if blind else (ticker or "종목")
    en_name = "Instrument X" if blind else (ticker or "Instrument")
    last = float(vs["Close"].iloc[-1]); p = _prec(last)
    tf = {"1d": "daily"}.get(spec.interval, spec.interval.replace("m", "-minute").replace("h", "-hour"))
    title = f"{en_name} - {tf} bars up to {labels[-1]}" + (" (grey background = pre/after-hours)" if spec.intraday and spec.session == "us" else "")
    marks = None
    if position:
        marks = [(position["entry"] * scale, "entry", "#2e7d32"), (position["stop"] * scale, "stop", "#c62828")]
        if position.get("target"):
            marks.append((position["target"] * scale, "target", "#00838f"))
    chart_png = _chart_png(vs, x, labels, title, spec, marks)

    extra = [c for c in vs.columns if c == "vwap" or c.startswith("ema")]
    tb = vs.iloc[-spec.table_bars:]
    rows = ["bar,open,high,low,close,volume" + "".join("," + c for c in extra)]
    vfmt = (lambda q: f"{q:.2f}") if blind else (lambda q: f"{q:.0f}")
    for lab, (_, r) in zip(labels[-spec.table_bars:], tb.iterrows()):
        rows.append(f"{lab},{r.Open:.{p}f},{r.High:.{p}f},{r.Low:.{p}f},{r.Close:.{p}f},{vfmt(r.Volume)}"
                    + "".join(f",{r[c]:.{p}f}" if np.isfinite(r[c]) else "," for c in extra))

    # 요약 사실 — 전부 기준 봉까지의 값
    atr = float(vs["atr"].iloc[-1]) if np.isfinite(vs["atr"].iloc[-1]) else float("nan")
    facts: dict = {"기준 봉": labels[-1], "기준 봉 종가": round(last, p), f"ATR({spec.interval} 20봉)": round(atr, p) if np.isfinite(atr) else None,
                   "기준 봉 거래량 ÷ 직전 20봉 평균": round(float(vs["Volume"].iloc[-1] / vs["vol_avg"].iloc[-1]), 2) if vs["vol_avg"].iloc[-1] > 0 else None}
    if "vwap" in vs:
        facts["세션 VWAP"] = round(float(vs["vwap"].iloc[-1]), p)
        if atr > 0:
            facts["종가−VWAP (ATR 배수)"] = round(float((last - vs["vwap"].iloc[-1]) / atr), 2)
    for n in spec.emas:
        e = vs[f"ema{n}"].iloc[-1]
        if np.isfinite(e):
            facts[f"EMA{n}"] = round(float(e), p)
    today = x.normalize()[-1]
    if spec.intraday:
        day = x.normalize(); mins = x.hour * 60 + x.minute
        t_today = vs[np.asarray(day == today)]
        facts["오늘 고가/저가 (지금까지)"] = [round(float(t_today["High"].max()), p), round(float(t_today["Low"].min()), p)]
        prev_days = [d for d in pd.Index(day).unique() if d < today]
        if spec.session == "us":
            reg = (mins >= 9 * 60 + 30) & (mins < 16 * 60)
            pre = vs[np.asarray((day == today) & (mins < 9 * 60 + 30))]
            reg_today = vs[np.asarray((day == today) & reg)]
            step_min = max(1, int(round((x[-1] - x[-2]).total_seconds() / 60))) if len(x) > 1 else 15
            facts["오늘 정규장이 시작된 뒤 지난 봉 수"] = int(len(reg_today))
            facts["정규장 마감(16:00)까지 남은 봉"] = int(max(0, (16 * 60 - (mins[-1] + step_min)) // step_min)) if mins[-1] < 16 * 60 else 0
            if len(pre):
                facts["오늘 장전 고가/저가"] = [round(float(pre["High"].max()), p), round(float(pre["Low"].min()), p)]
            if len(reg_today):
                facts["오늘 정규장 시가"] = round(float(reg_today["Open"].iloc[0]), p)
                m_today = mins[np.asarray(day == today) & reg]
                for width in (5, 15, 30):
                    if step_min <= width and m_today[-1] + step_min >= 9 * 60 + 30 + width:
                        orb = reg_today[np.asarray(m_today < 9 * 60 + 30 + width)]
                        facts[f"오늘 첫 {width}분 고가/저가"] = [round(float(orb["High"].max()), p), round(float(orb["Low"].min()), p)]
            if prev_days:
                pd_reg = vs[np.asarray((day == prev_days[-1]) & reg)]
                if len(pd_reg):
                    facts["전일 정규장 고가/저가/종가"] = [round(float(pd_reg["High"].max()), p), round(float(pd_reg["Low"].min()), p), round(float(pd_reg["Close"].iloc[-1]), p)]
        elif prev_days:
            pdv = vs[np.asarray(day == prev_days[-1])]
            facts["전일 고가/저가/종가"] = [round(float(pdv["High"].max()), p), round(float(pdv["Low"].min()), p), round(float(pdv["Close"].iloc[-1]), p)]
    else:
        hi = f["High"].iloc[-252:].max() * scale
        facts["최근 252봉 고점 대비"] = f"{(last / hi - 1):+.1%}"
        if len(vs) > 21:
            facts["최근 20봉 등락"] = f"{(last / vs['Close'].iloc[-21] - 1):+.1%}"
    if position:
        facts["보유 중"] = {"진입가": round(position["entry"] * scale, p), "현재 손절": round(position["stop"] * scale, p),
                          "현재 목표": round(position["target"] * scale, p) if position.get("target") else None,
                          "보유 봉 수": int(position.get("bars_held", 0)), "평가 손익": f"{(last / (position['entry'] * scale) - 1):+.2%}"}

    context_png = None
    if spec.intraday and context is not None and len(context):
        d = context[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
        d = d[np.asarray(_naive(d.index, False) < today)]       # 기준일 당일 일봉은 그날 종가를 담고 있으므로 뺀다
        unit, word = "d", "daily"
    elif not spec.intraday:
        d0 = f[["Open", "High", "Low", "Close", "Volume"]].copy(); d0.index = _naive(d0.index, False)
        d = _weekly(d0)
        if len(d) and d.index[-1] > today:                      # 아직 안 끝난 주는 뺀다
            d = d.iloc[:-1]
        unit, word = "w", "weekly"
    else:
        d = pd.DataFrame()
    if len(d) >= 20:
        d = d.copy()
        d["sma20"] = d["Close"].rolling(20).mean(); d["sma50"] = d["Close"].rolling(50).mean()
        d = d.iloc[-(90 if unit == "d" else 78):]
        ds = 100.0 / float(d["Close"].iloc[0]) if blind else 1.0   # 큰 그림은 조정가라 눈금이 다를 수 있다 — 모양만 본다
        for k in ("Open", "High", "Low", "Close", "sma20", "sma50"):
            d[k] = d[k] * ds
        context_png = _context_png(d, f"{en_name} - {word} bars, last {len(d)} COMPLETED before the decision bar (separate price scale, shape only)", unit)
        if np.isfinite(d["sma20"].iloc[-1]):
            facts[f"큰 그림({'일봉' if unit == 'd' else '주봉'}): 직전 종가가 20선 대비"] = f"{(d['Close'].iloc[-1] / d['sma20'].iloc[-1] - 1):+.1%}"
    return Packet(as_of=pd.Timestamp(v.index[-1]), scale=scale, blind=blind, name=name, chart_png=chart_png, context_png=context_png,
                  bars_csv="\n".join(rows), facts=facts, labels=labels, n_bars=len(v), interval=spec.interval, last=last)


# ---------------------------------------------------------------------------
# 판정 하나 정산 (단건)
# ---------------------------------------------------------------------------
def settle(j: Judgement, future: pd.DataFrame, *, fee_bps: float = 5.0, max_hold_bars: int = 0) -> dict:
    """실제 가격 눈금의 판정 ``j`` 를 기준 봉 **뒤** 봉 ``future`` 로 정산한다 (관리 호출 없음).

    체결: market = 첫 봉 시가, breakout_close = trigger_valid_bars 안에 종가가 trigger 를 넘는 첫 봉의 종가, limit = 저가가 trigger 에 닿는 첫 봉에서 min(시가, trigger).
    청산: 손절(close = 종가 이탈 시 그 종가 / touch = 저가가 닿으면 min(시가, 손절가)) → 목표1 에서 절반·목표2 에서 나머지(고가가 닿으면 그 가격)
    → 남은 물량은 max_hold_bars 또는 ``future`` 의 마지막 봉 종가. 같은 봉에서 touch 손절과 목표가 겹치면 손절 먼저(보수적).
    진입하지 않은 판정은 '그 뒤 움직임'(최대 상승·하락)만 잰다.
    """
    from strategy.adapters.ai_pattern_backtest import find_fill, walk_position

    out: dict = {"filled": False, "decision": j.decision}
    if future is None or future.empty:
        out["why"] = "뒤 봉 없음"
        return out
    F = future[["Open", "High", "Low", "Close"]].astype(float)
    ref = float(F["Open"].iloc[0])
    win = F.iloc[: max_hold_bars] if max_hold_bars else F
    out["이후 최대 상승"] = float(win["High"].max() / ref - 1); out["이후 최대 하락"] = float(win["Low"].min() / ref - 1)
    out["이후 종가 변화"] = float(win["Close"].iloc[-1] / ref - 1)
    fill = find_fill(j, F, 0, j.trigger_valid_bars if j.trigger_valid_bars > 0 else len(F))
    if not fill["filled"]:
        out["why"] = fill["why"]
        return out
    r = walk_position(F, fill, j, fee_bps=fee_bps, max_hold=max_hold_bars)
    out.update(r)
    return out
