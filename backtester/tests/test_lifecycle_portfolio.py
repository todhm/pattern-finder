"""계좌 시뮬 — 현금 회계·동시 보유 제한·유동성 필터·분할 매도."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategy.adapters.lifecycle_portfolio import PTrade, PortfolioConfig, align, hold_trades, run_portfolio

CAL = pd.bdate_range("2020-01-01", periods=60)


def _frames(prices: dict[str, float], vol: float = 1e6) -> dict[str, pd.DataFrame]:
    out = {}
    for s, p0 in prices.items():
        c = np.full(len(CAL), p0, dtype=float)
        out[s] = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(len(CAL), vol)}, index=CAL)
    return out


def _t(sym: str, entry_i: int, px: float, sells: list[tuple[int, float, float]], kind="first mature base") -> PTrade:
    return PTrade(sym, date(2019, 6, 1), kind, "40-Week", CAL[entry_i].date(), px, [(CAL[i].date(), p, f) for i, p, f in sells])


def test_cash_accounting_and_partial_sells():
    closes, dvol = align(_frames({"AAA": 10.0}), CAL)
    t = _t("AAA", 5, 10.0, [(10, 12.0, 0.5), (20, 14.0, 1.0)])
    res = run_portfolio([t], closes, dvol, None, PortfolioConfig(liquidity_usd=None, market_filter=False), "40-Week")
    assert len(res.trades) == 1
    size = 10_000.0                                     # 10% × $100K
    assert res.trades.iloc[0]["pnl"] == pytest.approx(size * (0.5 * 1.2 + 0.5 * 1.4) - size)
    assert res.final == pytest.approx(100_000 + 3_000)
    assert res.equity.iloc[5] == pytest.approx(100_000)  # 체결 직후 종가 = 진입가 → 변동 없음


def test_max_positions_and_one_recent_ipo():
    syms = {f"S{i}": 10.0 for i in range(12)}
    closes, dvol = align(_frames(syms), CAL)
    trades = [_t(s, 3, 10.0, [(30, 11.0, 1.0)]) for s in syms]       # 전부 상장 1년 이내
    res = run_portfolio(trades, closes, dvol, None, PortfolioConfig(liquidity_usd=None, market_filter=False, one_recent_ipo=True), "40-Week")
    assert len(res.trades) == 1 and res.skipped["IPO 1년 이내 동시 1종목"] == 11
    res2 = run_portfolio(trades, closes, dvol, None, PortfolioConfig(liquidity_usd=None, market_filter=False, one_recent_ipo=False), "40-Week")
    assert len(res2.trades) == 10 and res2.skipped["자리없음"] == 2


def test_liquidity_filter_uses_prior_dollar_volume():
    fr = _frames({"LOW": 10.0, "HI": 10.0}, vol=1e6)
    fr["HI"]["Volume"] = 5e6                                           # $50M/일
    closes, dvol = align(fr, CAL)
    trades = [_t("LOW", 25, 10.0, [(40, 11.0, 1.0)]), _t("HI", 25, 10.0, [(40, 11.0, 1.0)])]
    res = run_portfolio(trades, closes, dvol, None, PortfolioConfig(liquidity_usd=20e6, market_filter=False, one_recent_ipo=False), "40-Week")
    assert list(res.trades["symbol"]) == ["HI"] and res.skipped["유동성<$20M"] == 1


def test_market_filter_blocks_entries_below_sma():
    closes, dvol = align(_frames({"AAA": 10.0}), CAL)
    spy = pd.Series(np.linspace(300, 200, len(CAL)), index=CAL)       # 하락 중 → 전일 종가 < 평균
    res = run_portfolio([_t("AAA", 30, 10.0, [(40, 11.0, 1.0)])], closes, dvol, spy,
                        PortfolioConfig(liquidity_usd=None, market_filter=True), "40-Week")
    assert res.trades.empty and res.skipped["시장필터(SPY<200일선)"] == 1


def test_hold_control_and_open_position_marked_at_last_close():
    fr = _frames({"AAA": 10.0})
    fr["AAA"].loc[CAL[-1], "Close"] = 20.0
    closes, dvol = align(fr, CAL)
    held = hold_trades([_t("AAA", 5, 10.0, [])], closes, 252)        # 달력 끝까지
    assert held[0].sells[0][0] == CAL[-1].date() and held[0].pct_gain == pytest.approx(1.0)
    res = run_portfolio([_t("AAA", 5, 10.0, [])], closes, dvol, None, PortfolioConfig(liquidity_usd=None, market_filter=False), "40-Week")
    assert bool(res.trades.iloc[0]["open"]) and res.final == pytest.approx(110_000)
    m = res.metrics()
    assert m["거래 n"] == 1 and m["총수익"] == pytest.approx(0.10)
