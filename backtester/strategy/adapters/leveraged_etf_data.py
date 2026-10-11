"""레버리지 ETF 랩용 가격 로더 — 실데이터 + 상장 이전 합성 백캐스트 스플라이스.

페이지(47)와 CLI 가 같은 데이터를 쓰도록 한 곳에 둔다. 네트워크는 :class:`MarketDataPort` 를 통해서만.
합성 규칙은 24 페이지와 동일: 기초지수 ×N − 실제 T-bill(^IRX) 자금조달비용 − 실측 드래그(:data:`CALIBRATED_DRAG`),
실제 ETF 상장일부터는 실데이터 수익률로 이어붙인다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from data.domain.ports import MarketDataPort
from strategy.adapters.band_rebalance_strategy import CALIBRATED_DRAG, build_synthetic_leveraged, splice_series


@dataclass(frozen=True)
class EtfSpec:
    ticker: str
    underlying: str          # 합성 백캐스트 기초지수
    leverage: float
    inception: date
    note: str = ""


ETF_SPECS: dict[str, EtfSpec] = {
    "TQQQ": EtfSpec("TQQQ", "QQQ", 3.0, date(2010, 2, 11), "나스닥100 3배"),
    "QLD": EtfSpec("QLD", "QQQ", 2.0, date(2006, 6, 21), "나스닥100 2배"),
    "SOXL": EtfSpec("SOXL", "SOXX", 3.0, date(2010, 3, 11), "반도체(SOX) 3배 — SOXX 는 2001-07 부터"),
    "UPRO": EtfSpec("UPRO", "SPY", 3.0, date(2009, 6, 25), "S&P500 3배"),
    "SSO": EtfSpec("SSO", "SPY", 2.0, date(2006, 6, 21), "S&P500 2배"),
}

UNDERLYING_START = {"QQQ": date(1999, 3, 10), "SPY": date(1993, 1, 29), "SOXX": date(2001, 7, 13)}


@dataclass
class LabPrices:
    lev_close: pd.Series       # 거래 ETF 종가 (합성+실데이터 스플라이스 가능)
    signal_close: pd.Series    # 신호 지수 종가 (SMA 워밍업 포함, 평가 시작일보다 앞서 시작)
    cash_rate: pd.Series | None  # ^IRX/100 (연환산 소수) — 없으면 None
    synthetic_until: date | None  # 이 날짜 전까지는 합성
    source_note: str
    signal_df: pd.DataFrame | None = None   # 신호 지수 OHLC (단기 전략: ADX·저가 터치 등에 필요)


def _close(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].copy()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def load_lab_prices(md: MarketDataPort, etf: str, signal: str, start: date, end: date,
                    synthetic: bool = True, warmup_days: int = 400, annual_expense: float | None = None) -> LabPrices:
    """``etf`` 종가(필요 시 합성 백캐스트) + ``signal`` 지수 종가 + ^IRX.

    - ``synthetic=True`` 이고 ``start`` 가 상장일보다 앞서면 기초지수로 백캐스트해 실데이터와 스플라이스.
    - ``signal`` 은 ``start − warmup_days`` 부터 받아 SMA 워밍업을 확보한다.
    """
    spec = ETF_SPECS.get(etf)
    fetch_start = start - timedelta(days=warmup_days)
    sig_df = md.fetch_ohlcv(signal, fetch_start, end).copy()
    if sig_df.index.tz is not None:
        sig_df.index = sig_df.index.tz_localize(None)
    sig_df.index = sig_df.index.normalize()
    sig_df = sig_df[~sig_df.index.duplicated(keep="last")].sort_index()
    sig = sig_df["Close"].astype(float)

    cash = None
    try:
        cash = _close(md.fetch_ohlcv("^IRX", fetch_start, end)) / 100.0
    except Exception:
        cash = None

    synthetic_until = None
    if spec is None or not synthetic or start >= spec.inception:
        lev = _close(md.fetch_ohlcv(etf, fetch_start, end))
        note = f"{etf} 실데이터"
    else:
        under = _close(md.fetch_ohlcv(spec.underlying, fetch_start, end))
        drag = annual_expense if annual_expense is not None else CALIBRATED_DRAG.get(spec.leverage, 0.025)
        syn = build_synthetic_leveraged(under, spec.leverage, annual_expense=drag, financing_rate=cash)
        real = _close(md.fetch_ohlcv(etf, spec.inception, end)) if end > spec.inception else None
        lev = splice_series(syn, real)
        synthetic_until = spec.inception
        note = (f"{spec.underlying}×{spec.leverage:g} 합성(드래그 {drag:.2%}, ^IRX 조달비용) "
                f"→ {spec.inception} 부터 {etf} 실데이터")
    return LabPrices(lev_close=lev, signal_close=sig, cash_rate=cash, synthetic_until=synthetic_until, source_note=note,
                     signal_df=sig_df)
