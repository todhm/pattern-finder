"""H20 — 15분봉 장중 규칙 → 일간 승수(전일 종가 대비) 시계열 (순수 pandas/numpy).

각 규칙은 (일별 승수 m_t, 일별 노출 fraction, 이탈일 플래그) 를 돌려준다. 승수는 '장중 보유' 승수로 나눠 수정 일봉 수익률에 곱해 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADE_COST = 0.0015   # 편도 수수료+슬리피지


@dataclass(frozen=True)
class IntradayParams:
    rule: str = "stop"          # stop | vwap | orb | overnight | intraday | stop_vwap | gap | firsthour
    x: float = 0.05
    reenter_lag: int = 0        # stop/orb: 이탈 뒤 재매수까지 추가로 현금인 거래일 수
    reenter_close: bool = False # gap: 당일 종가 재매수(기본 다음날 시가)

    @property
    def name(self) -> str:
        if self.rule in ("stop", "stop_vwap"):
            return f"intraday {self.rule} x{self.x:.0%}" + (f" lag{self.reenter_lag}" if self.reenter_lag else "")
        if self.rule in ("gap", "firsthour"):
            return f"intraday {self.rule} x{self.x:.0%}" + (" reclose" if self.reenter_close else "")
        return f"intraday {self.rule}"


def daily_bars(bars: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """정규장 15분봉 → {날짜: 그날 봉}."""
    b = bars.copy()
    if b.index.tz is not None:
        b.index = b.index.tz_convert("America/New_York").tz_localize(None)
    b = b.between_time("09:30", "15:59")
    return {d: g for d, g in b.groupby(b.index.normalize())}


def run_rule(bars: pd.DataFrame, p: IntradayParams, split_gap: float = 0.25) -> pd.DataFrame:
    """반환 DataFrame(index=날짜): mult(규칙 승수, 전일 15:45 종가 대비, 비용 포함), hold(보유 승수), exposure, exited."""
    days = daily_bars(bars)
    dates = sorted(days); rows = []
    prev_close = None; in_cash = False; cash_days_left = 0
    for d in dates:
        g = days[d]
        o, c = float(g["Open"].iloc[0]), float(g["Close"].iloc[-1])
        if prev_close is None or abs(o / prev_close - 1.0) > split_gap:
            rows.append({"date": d, "mult": c / prev_close if prev_close else 1.0, "hold": c / prev_close if prev_close else 1.0, "exposure": 1.0, "exited": False})
            prev_close = c; in_cash = False; cash_days_left = 0
            continue
        hold = c / prev_close
        closes = g["Close"].to_numpy(dtype=float)
        vwap = float((g["Close"] * g["Volume"]).sum() / max(g["Volume"].sum(), 1.0))
        mult, exposure, exited = hold, 1.0, False
        if p.rule == "overnight":
            mult = (o / prev_close) * (1 - TRADE_COST) ** 2; exposure = 0.3
        elif p.rule == "intraday":
            mult = (c / o) * (1 - TRADE_COST) ** 2; exposure = 0.7
        elif p.rule in ("stop", "stop_vwap"):
            if in_cash:
                if cash_days_left > 0:
                    mult, exposure = 1.0, 0.0; cash_days_left -= 1
                elif p.rule == "stop_vwap":
                    if c > vwap:
                        mult, exposure = 1.0, 0.0; in_cash = False          # 종가 재매수(그날은 현금)
                    else:
                        mult, exposure = 1.0, 0.0
                else:
                    mult = (c / o) * (1 - TRADE_COST); exposure = 0.7; in_cash = False   # 시가 재매수
                    hit = np.flatnonzero(closes <= prev_close * (1 - p.x))
                    if len(hit):
                        k = hit[0]; mult = (closes[k] / o) * (1 - TRADE_COST) ** 2; exposure = 0.3; exited = True; in_cash = True; cash_days_left = p.reenter_lag
            else:
                hit = np.flatnonzero(closes <= prev_close * (1 - p.x))
                if len(hit):
                    k = hit[0]; mult = (closes[k] / prev_close) * (1 - TRADE_COST); exposure = 0.5; exited = True; in_cash = True; cash_days_left = p.reenter_lag
        elif p.rule == "vwap":
            if in_cash:
                mult, exposure = 1.0, 0.0
                if c > vwap:
                    in_cash = False
            else:
                if c < vwap:
                    mult = hold * (1 - TRADE_COST); exposure = 1.0; exited = True; in_cash = True
        elif p.rule == "gap":
            if in_cash:
                mult = (c / o) * (1 - TRADE_COST); exposure = 0.7; in_cash = False
            if o <= prev_close * (1 - p.x):
                # 시가 매도(현금 상태였다면 사지 않음)
                mult = (o / prev_close) * (1 - TRADE_COST) if exposure == 1.0 else 1.0; exposure = 0.0; exited = True
                if p.reenter_close:
                    mult *= (1 - TRADE_COST); in_cash = False          # 종가 재매수 → 다음날은 보유
                else:
                    in_cash = True
        elif p.rule == "firsthour":
            if in_cash:
                mult = (c / o) * (1 - TRADE_COST); exposure = 0.7; in_cash = False; base = o
            else:
                base = prev_close
            if len(closes) >= 4 and closes[3] <= prev_close * (1 - p.x):
                mult = (closes[3] / base) * (1 - TRADE_COST) * ((1 - TRADE_COST) if base == o else 1.0); exposure = 0.3; exited = True; in_cash = True
        elif p.rule == "orb":
            if in_cash:
                mult = (c / o) * (1 - TRADE_COST); exposure = 0.7; in_cash = False
            else:
                mult = hold
            if len(closes) >= 3:
                rng_hi = float(g["High"].iloc[:2].max()); rng_lo = float(g["Low"].iloc[:2].min()); c3 = closes[2]
                base = o if exposure < 1.0 else prev_close
                if c3 < rng_lo:
                    mult = (c3 / base) * (1 - TRADE_COST) * ((1 - TRADE_COST) if exposure < 1.0 else 1.0); exposure = 0.3; exited = True; in_cash = True
        else:
            raise ValueError(p.rule)
        rows.append({"date": d, "mult": mult, "hold": hold, "exposure": exposure, "exited": exited})
        prev_close = c
    return pd.DataFrame(rows).set_index("date")


def daily_multiplier(rule_df: pd.DataFrame, adj_close: pd.Series) -> pd.Series:
    """수정 일봉 수익률 × (규칙 승수 ÷ 보유 승수)."""
    adj = adj_close.reindex(rule_df.index)
    adj_ret = adj / adj.shift(1)
    m = adj_ret * (rule_df["mult"] / rule_df["hold"])
    return m.fillna(1.0).rename("mult")


def after_tax_curve(mult: pd.Series, costs_tax_rate: float = 0.22, deduction_frac: float = 0.025) -> tuple[pd.Series, float]:
    """일간 승수 → 연말 과세(연간 순이익 × 세율, 공제는 자본 1 기준 비율, 이월 없음) 반영 자산 곡선(시작 1) 과 최종값."""
    eq = []; v = 1.0; year_start = 1.0; years = mult.index.year
    for i, (t, m) in enumerate(mult.items()):
        v *= float(m)
        if i == len(mult) - 1 or years[i + 1] != years[i]:
            gain = v - year_start
            tax = max(0.0, gain - deduction_frac) * costs_tax_rate
            v -= tax; year_start = v
        eq.append(v)
    s = pd.Series(eq, index=mult.index)
    return s, float(s.iloc[-1])


def placebo_multiplier(rule_df: pd.DataFrame, adj_close: pd.Series, seed: int, mode: str) -> pd.Series:
    """같은 수의 이탈일(또는 현금일) 을 무작위 날에 — 이탈일은 시가 매도(오버나이트만), 현금일은 승수 1."""
    rng = np.random.default_rng(seed)
    k = int(rule_df["exited"].sum()) if mode == "exit" else int((rule_df["exposure"] == 0).sum())
    adj = adj_close.reindex(rule_df.index); adj_ret = (adj / adj.shift(1)).fillna(1.0)
    pick = np.zeros(len(rule_df), dtype=bool); pick[rng.choice(len(rule_df), size=min(k, len(rule_df)), replace=False)] = True
    if mode == "exit":
        # 이탈일: 오버나이트(전일 종가→시가) 만 — 시가/전일종가 비율은 규칙 df 에서 복원 불가하므로 hold 의 절반 경로 근사: sqrt
        m = adj_ret.copy(); m[pick] = np.sqrt(adj_ret[pick]) * (1 - TRADE_COST) ** 2
    else:
        m = adj_ret.copy(); m[pick] = 1.0
    return m.rename("placebo")
