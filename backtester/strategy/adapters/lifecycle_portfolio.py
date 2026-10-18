"""«The Lifecycle Trade» 계좌 단위 시뮬 — 거래 단위 통계(5장 방식)가 아니라 "자본 $100K 로 규칙을 그대로 돌리면 얼마가 되나".

입력은 `simulate_rule` 이 만든 거래(진입가·분할 매도 내역)와 종목별 종가. 책 8장 자금 관리를 그대로 적용한다:
  · 포지션 = 자본의 10% (거래당 손실 1% ÷ 손절 10%; 성숙주 20% 의 절반)
  · 동시 보유 ≤ max_positions, 상장 1년 이내 종목은 동시에 1개
  · 진입 전일 20일 평균 거래대금 ≥ $20M (7장 유동성 텔), 시장 필터(SPY > 200일선, 10장 "강세장에서만")
같은 날 신호가 여럿이면 거래대금 큰 순으로 채운다. 현금이 모자라면 건너뛴다(레버리지 없음).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

BUY_KINDS_ALL = ("IPO base", "initial base", "breakaway gap", "first mature base")


@dataclass(frozen=True)
class PortfolioConfig:
    capital: float = 100_000.0
    position_frac: float = 0.10
    max_positions: int = 10
    one_recent_ipo: bool = True          # 상장 52주 이내 종목 동시 1개 (8장)
    liquidity_usd: float | None = 20e6   # 진입 전일 20일 평균 거래대금. None = 필터 없음
    market_filter: bool = True           # SPY 전일 종가 > 200일 SMA 일 때만 진입
    kinds: tuple[str, ...] = BUY_KINDS_ALL
    recent_days: int = 365


@dataclass
class PTrade:
    symbol: str
    ipo_date: date | None
    kind: str
    rule: str
    entry_date: date
    entry_px: float
    sells: list[tuple[date, float, float]]        # (날짜, 가격, 원 수량 대비 비중)
    pct_gain: float = 0.0


@dataclass
class PortfolioResult:
    equity: pd.Series                    # 일별 계좌 가치
    invested: pd.Series                  # 일별 투자 비중 (주식 평가액 / 계좌)
    trades: pd.DataFrame                 # 실제 체결된 거래 (symbol, kind, entry_date, exit_date, size, pnl, pct)
    skipped: dict[str, int]              # 건너뛴 사유별 개수
    cfg: PortfolioConfig
    rule: str

    @property
    def final(self) -> float:
        return float(self.equity.iloc[-1])

    def metrics(self, bench: pd.Series | None = None) -> dict:
        eq = self.equity
        yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        peak = eq.cummax()
        dd = eq / peak - 1
        t = self.trades
        out = {"최종 자산": self.final, "총수익": self.final / self.cfg.capital - 1,
               "CAGR": (self.final / self.cfg.capital) ** (1 / yrs) - 1, "MDD": float(dd.min()),
               "거래 n": int(len(t)), "승률": float((t["pct"] > 0).mean()) if len(t) else float("nan"),
               "평균 거래수익": float(t["pct"].mean()) if len(t) else float("nan"),
               "최대 거래기여 $K": float(t["pnl"].max() / 1e3) if len(t) else float("nan"),
               "평균 투자비중": float(self.invested.mean()),
               "미체결(자리없음)": self.skipped.get("자리없음", 0), "미체결(필터)": sum(v for k, v in self.skipped.items() if k != "자리없음")}
        if bench is not None:
            b = bench.reindex(eq.index).ffill()
            out["SPY 총수익"] = float(b.iloc[-1] / b.iloc[0] - 1)
            out["SPY CAGR"] = float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1)
        return out

    def yearly(self) -> pd.DataFrame:
        eq = self.equity
        y = eq.groupby(eq.index.year).agg(["first", "last"])
        prev = y["last"].shift(1).fillna(self.cfg.capital)
        tr = self.trades.copy()
        tr["year"] = pd.to_datetime(tr["entry_date"]).dt.year
        g = tr.groupby("year")
        return pd.DataFrame({"연초→연말 수익": y["last"] / prev - 1, "연말 자산 $K": y["last"] / 1e3,
                             "진입 n": g.size().reindex(y.index).fillna(0).astype(int),
                             "진입 승률": g["pct"].apply(lambda s: float((s > 0).mean())).reindex(y.index)}).rename_axis("연도")


def trades_from_records(results: list[dict], rule: str) -> list[PTrade]:
    """배치 JSON(`lifecycle_batch_cli` results) → PTrade. 같은 (종목, 매수지점) 은 규칙별 1건."""
    out = []
    for r in results:
        ipo = r.get("ipo") or {}
        ipo_d = date.fromisoformat(ipo["ipo_date"]) if ipo.get("ipo_date") else None
        for t in r["trades"]:
            if t["rule"] != rule:
                continue
            sells = [(date.fromisoformat(str(s[0])[:10]), float(s[1]), float(s[2])) for s in t["sells"]]
            out.append(PTrade(r["symbol"], ipo_d, t["kind"], rule, date.fromisoformat(str(t["entry_date"])[:10]),
                              float(t["entry_px"]), sells, float(t["pct_gain"])))
    return out


def trades_from_results(results: list, rule: str) -> list[PTrade]:
    """`analyze_ticker` 결과(TickerResult) → PTrade (페이지용)."""
    out = []
    for r in results:
        ipo_d = r.ipo.ipo_date if r.ipo else None
        for t in r.trades:
            if t.rule != rule:
                continue
            out.append(PTrade(r.symbol, ipo_d, t.kind, rule, t.entry_date, float(t.entry_px),
                              [(d_, float(px), float(f)) for d_, px, f, _why in t.sells], float(t.pct_gain)))
    return out


def benchmark_series(market_data, symbol: str, start: date, end: date, log=lambda _m: None) -> pd.Series:
    """벤치마크 종가. 캐시가 일부 구간만 돌려주면(SPY 2024~ 만 캐시된 사례) yfinance 로 직접 받는다."""
    from datetime import timedelta

    def _norm(s: pd.Series) -> pd.Series:
        s = s.copy()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        return s[~s.index.duplicated()].sort_index()
    try:
        df = market_data.fetch_ohlcv(symbol, start - timedelta(days=90), end)
        s = _norm(df["Close"]) if df is not None and len(df) else pd.Series(dtype=float)
    except Exception as e:  # noqa: BLE001
        log(f"벤치마크 {symbol} 캐시 조회 실패 {type(e).__name__}: {e}")
        s = pd.Series(dtype=float)
    if s.empty or s.index[0] > pd.Timestamp(start) + pd.Timedelta(days=10):
        import yfinance as yf
        log(f"벤치마크 {symbol} 캐시 범위 부족({s.index[0].date() if len(s) else '없음'}~) → yfinance 직접 조회")
        df = yf.download(symbol, start=(start - timedelta(days=90)).isoformat(), end=(end + timedelta(days=1)).isoformat(),
                         auto_adjust=True, progress=False)
        close = df["Close"] if not isinstance(df.columns, pd.MultiIndex) else df["Close"].iloc[:, 0]
        s = _norm(close.astype(float))
    if s.empty or s.index[0] > pd.Timestamp(start) + pd.Timedelta(days=10):
        raise ValueError(f"벤치마크 {symbol} 가 {start} 부터 없음")
    return s


def hold_trades(trades: list[PTrade], closes: pd.DataFrame, hold_days: int = 252) -> list[PTrade]:
    """대조군: 같은 진입을 손절 없이 N 거래일 단순 보유 후 종가 매도."""
    out = []
    idx = closes.index
    for t in trades:
        if t.symbol not in closes.columns:
            continue
        i = idx.searchsorted(pd.Timestamp(t.entry_date))
        j = min(i + hold_days, len(idx) - 1)
        px = closes[t.symbol].iloc[j]
        if pd.isna(px):
            continue
        out.append(PTrade(t.symbol, t.ipo_date, t.kind, f"보유 {hold_days}일", t.entry_date, t.entry_px,
                          [(idx[j].date(), float(px), 1.0)], float(px) / t.entry_px - 1))
    return out


def dollar_volume_20(daily: pd.DataFrame) -> pd.Series:
    """진입 '전일' 기준 20일 평균 거래대금 — 신호일까지의 정보만 사용."""
    return (daily["Close"] * daily["Volume"]).rolling(20, min_periods=5).mean().shift(1)


def align(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """종목별 일봉 → (종가, 전일 20일 거래대금) 달력 정렬. 빈 날은 직전 값."""
    def _tz(s: pd.Series) -> pd.Series:
        s = s.copy()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        return s[~s.index.duplicated()]
    closes = pd.DataFrame({s: _tz(df["Close"]) for s, df in frames.items()}).reindex(calendar).ffill()
    dvol = pd.DataFrame({s: _tz(dollar_volume_20(df)) for s, df in frames.items()}).reindex(calendar).ffill()
    return closes, dvol


def run_portfolio(trades: list[PTrade], closes: pd.DataFrame, dvol: pd.DataFrame | None, spy: pd.Series | None,
                  cfg: PortfolioConfig = PortfolioConfig(), rule: str = "") -> PortfolioResult:
    cal = closes.index
    start_ts, end_ts = cal[0], cal[-1]
    by_entry: dict[pd.Timestamp, list[PTrade]] = {}
    for t in trades:
        if t.kind not in cfg.kinds or t.symbol not in closes.columns:
            continue
        ts = pd.Timestamp(t.entry_date)
        if ts < start_ts or ts > end_ts:
            continue
        by_entry.setdefault(ts, []).append(t)
    spy_ok = None
    if spy is not None and cfg.market_filter:
        s = spy.reindex(cal).ffill()
        spy_ok = (s.shift(1) > s.rolling(200, min_periods=100).mean().shift(1)).fillna(False)

    cash = cfg.capital
    positions: dict[int, dict] = {}      # id → {trade, shares, left, sells(iter), ipo_date, size}
    sells_by_day: dict[pd.Timestamp, list[tuple[int, float, float]]] = {}
    done: list[dict] = []
    skipped: dict[str, int] = {}
    eq_vals = np.empty(len(cal))
    inv_vals = np.empty(len(cal))
    pid = 0
    last_eq = cfg.capital

    def _skip(why: str) -> None:
        skipped[why] = skipped.get(why, 0) + 1

    for k, day in enumerate(cal):
        # 1) 매도 (전일 종가 신호 → 시뮬이 정한 가격)
        for pid_, frac, px in sells_by_day.pop(day, []):
            pos = positions.get(pid_)
            if pos is None:
                continue
            f = min(frac, pos["left"])
            if f <= 1e-9:
                continue
            cash += pos["shares"] * f * px
            pos["left"] -= f
            pos["proceeds"] += pos["shares"] * f * px
            if pos["left"] <= 1e-9:
                t = pos["trade"]
                done.append({"symbol": t.symbol, "kind": t.kind, "entry_date": t.entry_date, "exit_date": day.date(),
                             "size": pos["size"], "pnl": pos["proceeds"] - pos["size"], "pct": pos["proceeds"] / pos["size"] - 1, "open": False})
                del positions[pid_]
        # 2) 진입 (오늘 시가 체결 — 사이즈는 전일 계좌 가치 기준)
        cands = by_entry.get(day, [])
        if cands:
            if dvol is not None:
                cands = sorted(cands, key=lambda t: -float(np.nan_to_num(dvol.at[day, t.symbol], nan=0.0)))
            for t in cands:
                if spy_ok is not None and not bool(spy_ok.loc[day]):
                    _skip("시장필터(SPY<200일선)")
                    continue
                if cfg.liquidity_usd is not None:
                    dv = float(dvol.at[day, t.symbol]) if dvol is not None else float("nan")
                    if not (dv >= cfg.liquidity_usd):
                        _skip("유동성<$20M")
                        continue
                if any(p["trade"].symbol == t.symbol for p in positions.values()):
                    _skip("이미 보유")
                    continue
                if len(positions) >= cfg.max_positions:
                    _skip("자리없음")
                    continue
                recent = t.ipo_date is not None and (day.date() - t.ipo_date).days < cfg.recent_days
                if cfg.one_recent_ipo and recent and any(p["recent"] for p in positions.values()):
                    _skip("IPO 1년 이내 동시 1종목")
                    continue
                size = cfg.position_frac * last_eq
                if cash < size:
                    if cash < 0.5 * size:
                        _skip("자리없음")
                        continue
                    size = cash
                shares = size / t.entry_px
                cash -= size
                pid += 1
                positions[pid] = {"trade": t, "shares": shares, "left": 1.0, "size": size, "proceeds": 0.0, "recent": recent}
                for sd, px, frac in t.sells:
                    sells_by_day.setdefault(pd.Timestamp(sd), []).append((pid, frac, px))
        # 3) 종가 평가
        row = closes.iloc[k]
        mv = 0.0
        for p in positions.values():
            c = row[p["trade"].symbol]
            mv += p["shares"] * p["left"] * (float(c) if not pd.isna(c) else p["trade"].entry_px)
        last_eq = cash + mv
        eq_vals[k] = last_eq
        inv_vals[k] = mv / last_eq if last_eq > 0 else 0.0
    for p in positions.values():       # 기준일 미청산 → 마지막 종가로 평가
        t = p["trade"]
        c = closes[t.symbol].iloc[-1]
        val = p["proceeds"] + p["shares"] * p["left"] * (float(c) if not pd.isna(c) else t.entry_px)
        done.append({"symbol": t.symbol, "kind": t.kind, "entry_date": t.entry_date, "exit_date": None,
                     "size": p["size"], "pnl": val - p["size"], "pct": val / p["size"] - 1, "open": True})
    cols = ["symbol", "kind", "entry_date", "exit_date", "size", "pnl", "pct", "open"]
    return PortfolioResult(pd.Series(eq_vals, index=cal), pd.Series(inv_vals, index=cal), pd.DataFrame(done, columns=cols), skipped, cfg, rule)


def benchmark_metrics(spy: pd.Series, calendar: pd.DatetimeIndex, capital: float = 100_000.0) -> dict:
    s = spy.reindex(calendar).ffill()
    eq = s / s.iloc[0] * capital
    yrs = (calendar[-1] - calendar[0]).days / 365.25
    return {"최종 자산": float(eq.iloc[-1]), "총수익": float(eq.iloc[-1] / capital - 1), "CAGR": float((eq.iloc[-1] / capital) ** (1 / yrs) - 1),
            "MDD": float((eq / eq.cummax() - 1).min())}
