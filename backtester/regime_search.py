"""하락장 회피 규칙 탐색 — "장기보유를 이기는 사고팔기가 하나라도 있나" 를 레짐 신호 전 계열에 대해 격자 탐색.

    docker compose exec backtester python regime_search.py '{"etf": "TQQQ", "signal": "QQQ", "out": "/tmp/regime.json"}'

신호 계열 (전부 신호 지수 일봉 기준, 체결은 레버리지 ETF):
  sma        : N일 SMA 위/아래 (밴드·주봉/월봉 판정 변형)                — 47 페이지 ②
  sma_asym   : 이탈은 느린 SMA(200), 재진입은 빠른 SMA(20/50) — V 자 반등을 덜 놓치게
  sma_dd     : SMA 아래 **그리고** 고점 대비 낙폭 > d% 일 때만 이탈 (작은 조정 무시), SMA 회복 시 재진입
  cross      : 50/200 골든·데드크로스
  vix        : VIX > v_off 이탈, < v_on 재진입 (히스테리시스)            — 공포 지표
  vix_size   : 비중 = clip(k / VIX, 0, 1)                               — 내재 변동성 타기팅
  ladder     : SMA200 이격 사다리 (+5% 이상 100% / 0~5% 66% / −5~0% 33% / 그 아래 0)
  faber      : 10개월 SMA, 월말에만 판정 (Meb Faber GTAA)
  off_frac   : 이탈 시 현금 대신 레버리지 ETF 를 f 만큼 남김 (0.33 ≈ 1배 지수 노출)
각 규칙을 2010~(실데이터) 와 1999~(합성 포함) 두 구간, 세금 있음/없음으로 평가한다.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.leveraged_etf_data import load_lab_prices
from strategy.adapters.leveraged_etf_lab import CostParams, simulate_exposure, sma_filter


# ----------------------------------------------------------------------------- 신호

def _hyst(on: pd.Series, off: pd.Series, start_on: bool = False) -> pd.Series:
    st = np.zeros(len(on)); cur = 1.0 if start_on else 0.0
    onv, offv = on.values, off.values
    for i in range(len(on)):
        if offv[i]:
            cur = 0.0
        elif onv[i]:
            cur = 1.0
        st[i] = cur
    return pd.Series(st, index=on.index)


def sma_asym(c: pd.Series, n_exit: int, n_entry: int) -> pd.Series:
    s_exit, s_entry = c.rolling(n_exit).mean(), c.rolling(n_entry).mean()
    return _hyst(c > s_entry, c < s_exit).where(s_exit.notna(), 0.0)


def sma_dd(c: pd.Series, n: int, dd: float, reenter_n: int | None = None) -> pd.Series:
    sma = c.rolling(n).mean()
    drawdown = c / c.rolling(252, min_periods=20).max() - 1.0
    off = (c < sma) & (drawdown < -dd)
    on = c > (c.rolling(reenter_n).mean() if reenter_n else sma)
    return _hyst(on, off, start_on=True).where(sma.notna(), 0.0)


def cross(c: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    return (f > s).astype(float).where(s.notna(), 0.0)


def vix_hyst(vix: pd.Series, v_off: float, v_on: float, idx: pd.DatetimeIndex) -> pd.Series:
    v = vix.reindex(idx).ffill()
    return _hyst(v < v_on, v > v_off, start_on=True)


def vix_size(vix: pd.Series, k: float, idx: pd.DatetimeIndex, cap: float = 1.0) -> pd.Series:
    v = vix.reindex(idx).ffill()
    return (k / v).clip(0.0, cap).fillna(0.0)


def ladder(c: pd.Series, n: int = 200, steps=((0.05, 1.0), (0.0, 0.66), (-0.05, 0.33))) -> pd.Series:
    sma = c.rolling(n).mean()
    d = c / sma - 1.0
    out = pd.Series(0.0, index=c.index)
    for thr, w in sorted(steps, key=lambda x: x[0]):
        out = out.where(d < thr, w)
    return out.where(sma.notna(), 0.0)


def faber(c: pd.Series, months: int = 10) -> pd.Series:
    m = c.resample("ME").last()
    sig = (m > m.rolling(months).mean()).astype(float)
    return sig.reindex(c.index, method="ffill").fillna(0.0)


def monthly(sig: pd.Series, c: pd.Series) -> pd.Series:
    """일별 신호를 월말(마지막 거래일) 값만 취해 다음 달 내내 유지 — 월말 라벨은 그 달 마지막 거래일이므로 선행 참조 없음."""
    last_days = sig.groupby(sig.index.to_period("M")).apply(lambda x: x.index[-1])
    m = sig.loc[last_days.values]
    return m.reindex(c.index, method="ffill").fillna(0.0)


# ----------------------------------------------------------------------------- 탐색

def build_grid(c: pd.Series, vix: pd.Series | None) -> dict[str, pd.Series]:
    g: dict[str, pd.Series] = {}
    for n in (100, 150, 200, 250):
        for w in (False, True):
            g[f"sma{n}{'w' if w else ''}"] = sma_filter(c, n, 0.0, 0.0, w)
        g[f"sma{n}_b3"] = sma_filter(c, n, 0.03, 0.03)
        g[f"sma{n}_m"] = monthly(sma_filter(c, n), c)
    for n_exit, n_entry in ((200, 20), (200, 50), (200, 100), (150, 20), (150, 50), (250, 50)):
        g[f"asym{n_exit}/{n_entry}"] = sma_asym(c, n_exit, n_entry)
    for n in (150, 200):
        for dd in (0.08, 0.10, 0.15, 0.20):
            g[f"smadd{n}_{int(dd*100)}"] = sma_dd(c, n, dd)
            g[f"smadd{n}_{int(dd*100)}_re50"] = sma_dd(c, n, dd, reenter_n=50)
    for n, dd in ((200, 0.12), (200, 0.15), (250, 0.15), (200, 0.15)):
        g[f"smadd{n}_{int(dd*100)}_re20"] = sma_dd(c, n, dd, reenter_n=20)
    g["smadd200_15_re50_m"] = monthly(sma_dd(c, 200, 0.15, 50), c)
    g["cross50/200"] = cross(c, 50, 200); g["cross20/100"] = cross(c, 20, 100); g["cross50/150"] = cross(c, 50, 150)
    g["ladder200"] = ladder(c, 200); g["ladder150"] = ladder(c, 150)
    g["ladder200_wide"] = ladder(c, 200, ((0.10, 1.0), (0.0, 0.66), (-0.10, 0.33)))
    g["faber10"] = faber(c, 10); g["faber6"] = faber(c, 6); g["faber12"] = faber(c, 12)
    if vix is not None:
        for v_off, v_on in ((30, 25), (30, 20), (35, 25), (25, 20), (40, 30)):
            g[f"vix{v_off}/{v_on}"] = vix_hyst(vix, v_off, v_on, c.index)
            g[f"sma200&vix{v_off}/{v_on}"] = g["sma200"] * g[f"vix{v_off}/{v_on}"]
        for k in (15, 20, 25, 30):
            g[f"vixsize{k}"] = vix_size(vix, k, c.index)
            g[f"sma200*vixsize{k}"] = g["sma200"] * g[f"vixsize{k}"]
            g[f"asym200/50*vixsize{k}"] = g["asym200/50"] * g[f"vixsize{k}"]
    # 이탈 시 1배 지수 노출(≈ 레버리지 ETF 1/3) 유지 변형
    for base in ("sma200", "sma200w", "asym200/50", "smadd200_15", "smadd200_15_re50", "smadd200_12_re20", "faber10", "cross50/200", "sma200_m"):
        if base in g:
            g[f"{base}_off33"] = g[base].clip(lower=0.33)
    return g


def evaluate(lev: pd.Series, sigs: dict[str, pd.Series], start: date, costs: CostParams, cash) -> pd.DataFrame:
    rows = []
    lev_s = lev[lev.index >= pd.Timestamp(start)]
    hold = simulate_exposure(lev_s, pd.Series(1.0, index=lev_s.index), costs, cash, "hold")
    base_mult = hold.equity.iloc[-1] / hold.equity.iloc[0]
    rows.append({"rule": "① 장기보유", "mult": base_mult, "cagr": hold.metrics["cagr"], "mdd": hold.metrics["mdd"],
                 "sharpe": hold.metrics["sharpe"], "trades": hold.trades, "exposure": hold.time_in_market, "vs_hold": 1.0})
    for k, s in sigs.items():
        tgt = s.reindex(lev_s.index).ffill().fillna(0.0)
        r = simulate_exposure(lev_s, tgt, costs, cash, k)
        m = r.equity.iloc[-1] / r.equity.iloc[0]
        rows.append({"rule": k, "mult": m, "cagr": r.metrics["cagr"], "mdd": r.metrics["mdd"], "sharpe": r.metrics["sharpe"],
                     "trades": r.trades, "exposure": r.time_in_market, "vs_hold": m / base_mult})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    etf, signal = cfg.get("etf", "TQQQ"), cfg.get("signal", "QQQ")
    md = build_default_market_data()
    early = date.fromisoformat(cfg.get("early_start", "1999-12-01"))
    late = date.fromisoformat(cfg.get("late_start", "2010-03-01"))
    end = date.today()
    prices = load_lab_prices(md, etf, signal, early, end, synthetic=True)
    c = prices.signal_close
    vix = None
    for src in (md, None):
        try:
            if src is None:
                from data.adapters.yfinance_adapter import YFinanceAdapter
                src = YFinanceAdapter()
            v = src.fetch_ohlcv("^VIX", early, end)["Close"]
            if v.index.tz is not None:
                v.index = v.index.tz_localize(None)
            v.index = v.index.normalize()
            vix = v[~v.index.duplicated()].astype(float)
            print(f"[regime] VIX {v.index[0].date()}~{v.index[-1].date()} n={len(v)}", file=sys.stderr)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[regime] VIX 실패 ({type(src).__name__}): {str(e)[:120]}", file=sys.stderr)
    sigs = build_grid(c, vix)
    print(f"[regime] rules={len(sigs)}", file=sys.stderr)
    base_costs = CostParams(rebalance_band=float(cfg.get("rebalance_band", 0.05)))
    out = {}
    for label, start in (("2010~", late), ("1999~", early)):
        for tax_label, costs in (("tax", base_costs), ("notax", replace(base_costs, tax_rate=0.0))):
            df = evaluate(prices.lev_close, sigs, start, costs, prices.cash_rate)
            out[f"{label}|{tax_label}"] = df.to_dict(orient="records")
            print(f"[regime] {label} {tax_label} done", file=sys.stderr)
    json.dump({"etf": etf, "signal": signal, "results": out}, open(cfg.get("out", "/tmp/regime.json"), "w"),
              ensure_ascii=False, default=str)
    # 요약 — 2010~ 세후 배수 상위 15 + 각 규칙의 1999~ 결과
    a = pd.DataFrame(out["2010~|tax"]).set_index("rule")
    b = pd.DataFrame(out["1999~|tax"]).set_index("rule")
    nt = pd.DataFrame(out["2010~|notax"]).set_index("rule")
    j = a.join(b, rsuffix="_99").join(nt[["mult", "mdd"]], rsuffix="_notax")
    j = j.sort_values("mult", ascending=False)
    print(f"\n{etf} — 2010~ 세후 배수 순 (장기보유 {a.loc['① 장기보유','mult']:.0f}x / 1999~ {b.loc['① 장기보유','mult']:.1f}x)")
    print(f"{'rule':26} {'2010 배수':>9} {'MDD':>6} {'매매':>5} {'노출':>5} | {'세전':>8} {'MDD':>6} | {'1999 배수':>9} {'MDD':>6}")
    for k, r in j.head(25).iterrows():
        print(f"{k:26} {r['mult']:8.1f}x {r['mdd']:+6.0%} {int(r['trades']):5d} {r['exposure']:5.0%} | "
              f"{r['mult_notax']:7.1f}x {r['mdd_notax']:+6.0%} | {r['mult_99']:8.1f}x {r['mdd_99']:+6.0%}")
    print("\n1999~ 세후 배수 순 상위 10")
    for k, r in j.sort_values("mult_99", ascending=False).head(10).iterrows():
        print(f"{k:26} 1999 {r['mult_99']:8.1f}x {r['mdd_99']:+6.0%} | 2010 {r['mult']:8.1f}x {r['mdd']:+6.0%}")


if __name__ == "__main__":
    main()
