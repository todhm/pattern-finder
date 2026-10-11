"""초단기(데이트레이딩) 전략이 레버리지 ETF 장기보유를 이기는가 — 논문 재현 랩.

⑥ 5분 오프닝 레인지 돌파 (ORB) — Zarattini & Aziz (2023) "Can Day Trading Really Be Profitable?" (SSRN 4416622)
   - 09:30 첫 5분봉이 양봉이면 두 번째 봉 시가에 매수, 음봉이면 매도(공매도). 도지(시가≈종가)면 쉼.
   - 스톱 = 첫 봉 저가(롱)/고가(숏). 변형: 14일 ATR 의 일정 비율(논문 최적 5%).
   - 목표 = 10R 또는 장 마감(EoD) 청산. 리스크 = 계좌의 1%. 주수 = int(min(A×1% / R, 4×A / P)).
   - 수수료 $0.0005/주. 슬리피지 0 가정. 2016-01-01 ~ 2023-02-17 QQQ: +675% (장기보유 +169%), 1,795회, 승률 24%,
     평균 0.13R, 샤프 1.12. TQQQ 로 거래하면 +1,485%, ATR 5% 스톱 + EoD 면 +9,350%.
⑦ 오버나이트 — 종가 매수 → 다음날 시가 매도 (밤) vs 시가 매수 → 종가 매도 (낮). 지수의 수익이 밤에 몰린다는
   "overnight return" 현상(Lou·Polk·Skouras 2019)을 TQQQ·QQQ 에서 수수료·세금 포함으로 확인.

순수 pandas. 데이터(5분봉·일봉)는 페이지/CLI 가 넣어 준다. 5분봉 인덱스는 tz-aware(America/New_York) 또는
NY 기준 naive 둘 다 받는다. 정규장(09:30~16:00) 밖 봉은 여기서 버린다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

from strategy.adapters.risk_metrics import compute_risk_metrics

NY = "America/New_York"


@dataclass(frozen=True)
class OrbParams:
    stop_mode: str = "range"        # "range" = 첫 봉 저/고가 | "atr" = ATR14 × atr_frac
    atr_frac: float = 0.05
    target_r: float | None = 10.0   # None = EoD 만
    risk_pct: float = 0.01
    max_leverage: float = 4.0
    doji_pct: float = 0.0           # |종가−시가| ≤ 시가×doji_pct 면 도지 (논문: 시가 ≈ 종가)
    allow_short: bool = True
    allow_long: bool = True
    initial_capital: float = 25_000.0
    commission_per_share: float = 0.0005   # 논문 — **실제(분할 전) 주식 수** 기준. 분할 조정 봉을 쓰면 unadjusted_factor 로 환산
    slippage_per_share: float = 0.0        # 편도 슬리피지 $/주 (실제 주식 수 기준)
    commission_pct: float = 0.0            # 국내 증권사형 (예: 토스 0.001 = 0.1%, 매수·매도 각각)
    tax_rate: float = 0.0                  # 양도세 — 연 단위로 이익·손실을 합산(순이익)해 과세, 손실 이월 없음
    tax_deduction: float = 0.0             # 연 기본공제 ($ 단위. 250만 원 ≈ $1,800)
    stop_first_if_both: bool = True        # 한 봉에 스톱·목표 둘 다 닿으면 스톱 우선 (보수적)


@dataclass
class OrbTrade:
    day: pd.Timestamp
    side: int                 # +1 롱 / −1 숏
    entry: float
    stop: float
    exit: float
    exit_reason: str          # "stop" | "target" | "eod"
    shares: int
    r_value: float            # $R (주당)
    pnl: float                # 수수료 차감 후 $
    r_multiple: float         # pnl / (shares × R)
    capped: bool              # 레버리지 상한에 걸렸나


@dataclass
class OrbResult:
    trades: list[OrbTrade]
    equity: pd.Series         # 일별 계좌 (거래 없는 날은 이월)
    params: OrbParams
    metrics: dict = field(default_factory=dict)
    tax_paid: float = 0.0

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


def _session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """정규장 09:30 ≤ t < 16:00 (NY) 봉만, naive NY 인덱스로."""
    b = bars.copy()
    if b.index.tz is not None:
        b.index = b.index.tz_convert(NY).tz_localize(None)
    b = b.between_time(time(9, 30), time(15, 59))
    return b.sort_index()


def atr14_from_bars(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """5분봉을 일봉으로 합쳐 14일 ATR (Wilder) — 전일까지 값을 당일에 쓴다 (shift 1)."""
    d = bars.groupby(bars.index.normalize()).agg(High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    prev = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev).abs(), (d["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().shift(1)


def split_factor_series(index: pd.DatetimeIndex, splits: list[tuple[str, float]]) -> pd.Series:
    """분할 조정가 → 실제가 환산 배수 (날짜별). ``splits`` = [(분할일, 배수)] — 분할일 이전 구간에 배수가 곱해진다.

    예: TQQQ [("2017-01-12", 2), ("2018-05-24", 3), ("2021-01-21", 2), ("2022-01-13", 2)] → 2016 년은 24배.
    """
    idx = pd.DatetimeIndex(index).normalize()
    f = pd.Series(1.0, index=idx)
    for d, m in splits:
        f[f.index < pd.Timestamp(d)] *= m
    return f


KNOWN_SPLITS: dict[str, list[tuple[str, float]]] = {
    "TQQQ": [("2017-01-12", 2), ("2018-05-24", 3), ("2021-01-21", 2), ("2022-01-13", 2)],
    "SOXL": [("2021-03-02", 15), ("2024-05-24", 1)],  # 2021-03 15:1 (2024 분할 없음 — 자리만)
    "QQQ": [], "SPY": [], "UPRO": [("2017-01-12", 2), ("2018-05-24", 2), ("2021-01-21", 2), ("2022-01-13", 2)],
}


def run_orb(signal_bars: pd.DataFrame, trade_bars: pd.DataFrame | None = None, p: OrbParams = OrbParams(),
            unadjusted_factor: pd.Series | None = None) -> OrbResult:
    """ORB 시뮬레이션. ``signal_bars`` 로 방향을 정하고 ``trade_bars`` 로 체결 (None 이면 같은 것).

    ``unadjusted_factor``: 날짜별 (실제가 ÷ 조정가) 배수. 봉이 분할 조정돼 있으면 주당 수수료·슬리피지를 이 배수로 나눠
    실제 주식 수 기준으로 맞춘다 (TQQQ 2016 년 조정가 $4 ≈ 실제 $96 — 안 맞추면 수수료가 24배 과대).

    논문의 "TQQQ 버전" = signal_bars 도 TQQQ (논문은 TQQQ 자체 5분봉으로 돌렸다). 방향만 QQQ 로 보고 TQQQ 를
    거래하는 변형은 signal=QQQ, trade=TQQQ 로.
    """
    sig = _session_bars(signal_bars)
    trd = _session_bars(trade_bars) if trade_bars is not None else sig
    atr = atr14_from_bars(trd) if p.stop_mode == "atr" else None
    equity = p.initial_capital
    realized_ytd = 0.0
    tax_paid = 0.0
    trades: list[OrbTrade] = []
    eq_rows: list[tuple[pd.Timestamp, float]] = []
    days = sorted(set(trd.index.normalize()) & set(sig.index.normalize()))
    for i, day in enumerate(days):
        s = sig[sig.index.normalize() == day]
        t = trd[trd.index.normalize() == day]
        if len(s) < 2 or len(t) < 2 or s.index[0].time() != time(9, 30) or t.index[0].time() != time(9, 30):
            eq_rows.append((day, equity)); continue
        o, c = float(s["Open"].iloc[0]), float(s["Close"].iloc[0])
        if abs(c - o) <= o * p.doji_pct:
            eq_rows.append((day, equity)); continue
        side = 1 if c > o else -1
        if (side > 0 and not p.allow_long) or (side < 0 and not p.allow_short):
            eq_rows.append((day, equity)); continue
        first = t.iloc[0]
        entry = float(t["Open"].iloc[1])
        if p.stop_mode == "atr":
            a = atr.get(day) if atr is not None else None
            if a is None or not np.isfinite(a) or a <= 0:
                eq_rows.append((day, equity)); continue
            r = float(a) * p.atr_frac
            stop = entry - side * r
        else:
            stop = float(first["Low"]) if side > 0 else float(first["High"])
            r = abs(entry - stop)
        if r <= 0:
            eq_rows.append((day, equity)); continue
        by_risk = equity * p.risk_pct / r
        by_lev = p.max_leverage * equity / entry
        shares = int(min(by_risk, by_lev))
        if shares <= 0:
            eq_rows.append((day, equity)); continue
        capped = by_lev < by_risk
        target = entry + side * p.target_r * r if p.target_r else None
        exit_px, reason = float(t["Close"].iloc[-1]), "eod"
        for _, bar in t.iloc[1:].iterrows():
            lo, hi, op = float(bar["Low"]), float(bar["High"]), float(bar["Open"])
            hit_stop = lo <= stop if side > 0 else hi >= stop
            hit_tgt = target is not None and ((hi >= target) if side > 0 else (lo <= target))
            if hit_stop and (p.stop_first_if_both or not hit_tgt):
                # 갭으로 스톱을 지나 열리면 시가 체결
                exit_px = min(op, stop) if side > 0 else max(op, stop)
                reason = "stop"; break
            if hit_tgt:
                exit_px = max(op, target) if side > 0 else min(op, target)
                reason = "target"; break
        gross = shares * (exit_px - entry) * side
        f = float(unadjusted_factor.get(day, 1.0)) if unadjusted_factor is not None else 1.0
        comm = 2 * shares * (p.commission_per_share + p.slippage_per_share) / f + p.commission_pct * shares * (entry + exit_px)
        pnl = gross - comm
        equity += pnl
        realized_ytd += pnl
        trades.append(OrbTrade(day, side, entry, stop, exit_px, reason, shares, r, pnl, pnl / (shares * r), capped))
        last_of_year = i == len(days) - 1 or days[i + 1].year != day.year
        if last_of_year and p.tax_rate > 0 and realized_ytd > p.tax_deduction:
            tax = (realized_ytd - p.tax_deduction) * p.tax_rate
            equity -= tax; tax_paid += tax
        if last_of_year:
            realized_ytd = 0.0
        eq_rows.append((day, equity))
    eq = pd.Series(dict(eq_rows), name="ORB").sort_index()
    res = OrbResult(trades, eq, p, tax_paid=tax_paid)
    res.metrics = orb_metrics(res)
    return res


def orb_metrics(res: OrbResult) -> dict:
    tr = res.frame
    m = compute_risk_metrics(res.equity) if len(res.equity) > 1 else {}
    if len(tr) == 0:
        return {**m, "trades": 0}
    wins = tr["pnl"] > 0
    return {
        **m,
        "trades": int(len(tr)), "win_rate": float(wins.mean()), "avg_r": float(tr["r_multiple"].mean()),
        "long_share": float((tr["side"] > 0).mean()), "capped_share": float(tr["capped"].mean()),
        "stop_share": float((tr["exit_reason"] == "stop").mean()), "target_share": float((tr["exit_reason"] == "target").mean()),
        "avg_win_r": float(tr.loc[wins, "r_multiple"].mean()) if wins.any() else 0.0,
        "avg_loss_r": float(tr.loc[~wins, "r_multiple"].mean()) if (~wins).any() else 0.0,
        "tax_paid": res.tax_paid,
    }


def alpha_beta(strategy_equity: pd.Series, bench_close: pd.Series) -> dict:
    """일별 수익률 회귀 Ret_strat = α + β·Ret_bench (논문 표 2 방식). α 는 연환산."""
    rs = strategy_equity.pct_change().dropna()
    rb = bench_close.pct_change().dropna()
    idx = rs.index.intersection(rb.index)
    if len(idx) < 30:
        return {"alpha_ann": np.nan, "beta": np.nan, "n": len(idx)}
    x, y = rb.loc[idx].values, rs.loc[idx].values
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)) if np.var(x, ddof=1) > 0 else np.nan
    alpha_d = float(y.mean() - beta * x.mean())
    return {"alpha_ann": (1 + alpha_d) ** 252 - 1, "beta": beta, "n": int(len(idx))}


# ----------------------------------------------------------------------------- ⑦ 오버나이트

def overnight_curves(daily: pd.DataFrame, commission_pct: float = 0.0, tax_rate: float = 0.0,
                     initial_capital: float = 25_000.0, tax_deduction: float = 0.0) -> pd.DataFrame:
    """일봉(Open/Close)으로 세 곡선: 장기보유 / 밤만(종가→익일 시가) / 낮만(시가→종가). 수수료는 매 거래 양방향 %.

    세금은 연말에 그 해 손익을 합산한 순이익이 ``tax_deduction`` 을 넘는 부분에 ``tax_rate`` (손실 이월 없음).
    장기보유는 세금 없음(미실현).
    """
    d = daily.dropna(subset=["Open", "Close"]).copy()
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    night = (d["Open"] / d["Close"].shift(1) - 1.0).fillna(0.0) - 2 * commission_pct
    day_ = (d["Close"] / d["Open"] - 1.0) - 2 * commission_pct
    hold = d["Close"].pct_change().fillna(0.0)

    def _compound(r: pd.Series, taxed: bool) -> pd.Series:
        eq = initial_capital; ytd_start = eq; out = []
        years = r.index.year
        for i, (ts, x) in enumerate(r.items()):
            eq *= 1.0 + float(x)
            if taxed and tax_rate > 0 and (i == len(r) - 1 or years[i + 1] != years[i]):
                gain = eq - ytd_start
                if gain > tax_deduction:
                    eq -= (gain - tax_deduction) * tax_rate
                ytd_start = eq
            out.append(eq)
        return pd.Series(out, index=r.index)

    return pd.DataFrame({"장기보유": _compound(hold, False), "밤만 (종가→시가)": _compound(night, True),
                         "낮만 (시가→종가)": _compound(day_, True)})


def curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in curves.columns:
        m = compute_risk_metrics(curves[k])
        rows.append({"전략": k, "최종 배수": curves[k].iloc[-1] / curves[k].iloc[0], "CAGR": m["cagr"], "MDD": m["mdd"],
                     "샤프": m["sharpe"], "최악 연도": m["worst_year"]})
    return pd.DataFrame(rows)


def yearly_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    out = {}
    for k, s in series.items():
        y = s.groupby(s.index.year).agg(["first", "last"])
        out[k] = y["last"] / y["first"] - 1.0
    return pd.DataFrame(out)
