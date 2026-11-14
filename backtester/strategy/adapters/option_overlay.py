"""옵션 얹기 계좌 — 기초자산(TQQQ) 비중 w 위에 옵션 다리(leg)를 규칙대로 굴리는 일별 원장. 순수 numpy/pandas.

원칙(레버리지·공매도 금지 제약의 옵션 판):
 - 매수 옵션은 프리미엄 전액 현금 지불, 매도 콜은 보유 주식 수 이하(커버드), 매도 풋은 행사가 전액 현금 담보.
 - 현금은 음수가 되지 않는다(주식 목표 = w × stock_frac × (자산 − 매수 옵션 가치)).
체결: 그날 장 마감 호가 — 살 때 mid + fill×(ask−mid), 팔 때 mid − fill×(mid−bid). fill=1 이면 호가 전부 건넘.
만기: 만기일 이하 마지막 거래일 종가의 내재가치로 현금 정산. 분할: 행사가 ÷ r, 수량 × r.
규칙은 달력·전날까지의 비중만 쓴다(그날 종가로 신호를 만들지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Chain:
    """날짜별 체인을 한 덩어리 배열로. off[i]:off[i+1] 가 days[i] 의 줄."""
    days: pd.DatetimeIndex
    off: np.ndarray
    exp: np.ndarray        # 만기 (epoch 일수)
    call: np.ndarray       # bool
    strike: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    delta: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Chain":
        """df: date, expiration, type, strike, bid, ask, delta (date 오름차순 아니어도 됨)."""
        df = df.sort_values("date", kind="stable"); d = pd.to_datetime(df["date"]).to_numpy()
        days = pd.DatetimeIndex(np.unique(d)); off = np.append(np.searchsorted(d, days.to_numpy()), len(df))
        exp = pd.to_datetime(df["expiration"]).to_numpy().astype("datetime64[D]").astype(np.int64)
        return cls(days, off, exp, (df["type"].to_numpy() == "call"), df["strike"].to_numpy(float), df["bid"].to_numpy(float), df["ask"].to_numpy(float), df["delta"].to_numpy(float))

    def day(self, ts) -> slice | None:
        i = self.days.get_indexer([ts])[0]
        return None if i < 0 else slice(self.off[i], self.off[i + 1])


@dataclass
class Leg:
    """굴리는 옵션 다리 하나."""
    call: bool
    side: int                      # +1 매수, −1 매도
    delta: float                   # 목표 |델타|
    dte: int                       # 목표 잔존 일수(달력일)
    roll_dte: int = 0              # 잔존 ≤ 이 값이면 갈아탄다(0 = 만기까지)
    basis: str = "shares"          # shares(보유 주식 수) | notional(자산÷주가) | cash(남는 현금÷행사가) | budget(자산의 cover 비율만큼 매수)
    cover: float = 1.0
    active: pd.Series | None = None  # bool, 전날 기준으로 미리 shift 해서 넘긴다. False 면 포지션을 닫는다


@dataclass
class Spec:
    w: pd.Series                   # 기초자산 목표 비중(전날 기준으로 shift 된 것), 0~1
    legs: list[Leg] = field(default_factory=list)
    stock_frac: float = 1.0
    fill: float = 1.0
    stock_cost: float = 0.001
    fee: float = 0.0065            # 옵션 주당 수수료($0.65/계약)


def _quote(ch: Chain, sl: slice, exp: int, call: bool, k: float):
    m = np.flatnonzero((ch.exp[sl] == exp) & (ch.call[sl] == call))
    if not len(m):
        return None
    j = m[np.argmin(np.abs(ch.strike[sl][m] - k))] + sl.start
    if abs(ch.strike[j] - k) > 0.011 * max(k, 1.0):
        return None
    return j


def _pick(ch: Chain, sl: slice, today: int, leg: Leg):
    """목표 잔존에 가장 가까운 만기에서 목표 델타에 가장 가까운(호가 있는) 행사가."""
    exp, call = ch.exp[sl], ch.call[sl]; ok = (call == leg.call) & (ch.bid[sl] > 0) & (ch.ask[sl] > ch.bid[sl]) & (exp - today >= 5) & np.isfinite(ch.delta[sl])
    if not ok.any():
        return None
    exps = np.unique(exp[ok]); e = exps[np.argmin(np.abs(exps - today - leg.dte))]
    m = np.flatnonzero(ok & (exp == e)); j = m[np.argmin(np.abs(np.abs(ch.delta[sl][m]) - leg.delta))] + sl.start
    return j if abs(abs(ch.delta[j]) - leg.delta) <= 0.15 else None


def _later(ch: Chain, sl: slice | None, today: int, leg: Leg, exp: int) -> bool:
    """갈아탈 만기가 지금 것보다 뒤에 있을 때만 갈아탄다(긴 만기가 아직 상장되지 않았으면 그대로 든다)."""
    if sl is None:
        return False
    j = _pick(ch, sl, today, leg)
    return j is not None and ch.exp[j] > exp


def run_overlay(ch: Chain, raw_c: pd.Series, split: pd.Series, cash_rate: pd.Series, spec: Spec, opt_raw_c: pd.Series | None = None, opt_split: pd.Series | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """raw_c·split: 보유 주식의 원가격 종가와 그날 발효 분할 계수. cash_rate: 연율. 반환 = (자산 곡선, 옵션 거래 기록).
    opt_raw_c·opt_split 을 주면 옵션의 기초자산이 보유 주식과 다르다(예: 주식 TQQQ, 옵션 QQQ) — shares 기준 수량은 주식 평가액 ÷ 옵션 기초자산 가격."""
    days = raw_c.index; later = split[::-1].cumprod()[::-1].shift(-1).fillna(1.0); adj = (raw_c / later).to_numpy()
    raw = (raw_c if opt_raw_c is None else opt_raw_c.reindex(days).ffill()).to_numpy()
    sp = (split if opt_raw_c is None else (opt_split.reindex(days).fillna(1.0) if opt_split is not None else pd.Series(1.0, index=days))).to_numpy(); rate = cash_rate.reindex(days).ffill().fillna(0.0).to_numpy() / 252.0; w = spec.w.reindex(days).ffill().fillna(0.0).to_numpy()
    act = [np.ones(len(days), bool) if lg.active is None else lg.active.reindex(days).ffill().fillna(False).to_numpy(bool) for lg in spec.legs]
    dn = days.to_numpy().astype("datetime64[D]").astype(np.int64)
    shares, cash = 0.0, 1.0; pos: list[dict | None] = [None] * len(spec.legs); eq = np.empty(len(days)); log = []; prev_w = np.nan
    for t in range(len(days)):
        if t:
            cash *= 1.0 + rate[t]
        S = raw[t]; sl = ch.day(days[t]); event = t == 0 or w[t] != prev_w
        if sp[t] != 1.0:
            for p in pos:
                if p:
                    p["k"] /= sp[t]; p["q"] *= sp[t]; p["mark"] /= sp[t]
        for i, p in enumerate(pos):                                           # 평가 · 만기
            if not p:
                continue
            intr = max(S - p["k"], 0.0) if p["call"] else max(p["k"] - S, 0.0)
            j = _quote(ch, sl, p["exp"], p["call"], p["k"]) if sl is not None else None
            if j is not None and ch.ask[j] > 0:
                p["bid"], p["ask"] = ch.bid[j], ch.ask[j]; p["mark"] = 0.5 * (p["bid"] + p["ask"])
            else:
                p["bid"] = p["ask"] = p["mark"] = max(intr, min(p["mark"], intr + p.get("tv", 0.0)))
            p["tv"] = max(p["mark"] - intr, 0.0)
            if t + 1 == len(days) or dn[t + 1] > p["exp"]:                     # 다음 거래일이 만기 뒤 → 오늘 정산
                cash += p["q"] * intr; log.append((days[t], i, "만기", p["k"], p["q"], intr)); pos[i] = None; event = True
        for i, (lg, p) in enumerate(zip(spec.legs, pos)):                       # 닫기(갈아타기·비활성·비중 변경)
            if p and (not act[i][t] or (lg.roll_dte and p["exp"] - dn[t] <= lg.roll_dte and _later(ch, sl, dn[t], lg, p["exp"])) or (w[t] != prev_w and lg.basis in ("shares", "cash"))):
                mid = p["mark"]; px = mid + spec.fill * (p["ask"] - mid) if p["q"] < 0 else mid - spec.fill * (mid - p["bid"])
                cash += p["q"] * px - abs(p["q"]) * spec.fee; log.append((days[t], i, "닫기", p["k"], p["q"], px)); pos[i] = None; event = True
        if event and sl is not None or (sl is not None and any(p is None and act[i][t] for i, p in enumerate(pos))):
            total = shares * adj[t] + cash + sum(p["q"] * p["mark"] for p in pos if p)
            for i, lg in enumerate(spec.legs):                                 # 다리를 먼저 열고(목표 주식 수 기준) 주식을 맞춘다
                if pos[i] is not None or not act[i][t]:
                    continue
                j = _pick(ch, sl, dn[t], lg)
                if j is None:
                    continue
                k, bid, ask = ch.strike[j], ch.bid[j], ch.ask[j]; mid = 0.5 * (bid + ask); px = mid + spec.fill * (ask - mid) if lg.side > 0 else mid - spec.fill * (mid - bid)
                opt_long = sum(p["q"] * p["mark"] for p in pos if p and p["q"] > 0)
                stock_val = w[t] * spec.stock_frac * max(total - opt_long, 0.0)
                used = sum(-p["q"] * p["k"] for p in pos if p and p["q"] < 0 and not p["call"])
                q = {"shares": stock_val / S, "notional": total / S, "cash": max(total - opt_long - stock_val - used, 0.0) / k, "budget": total / px}[lg.basis] * lg.cover
                if lg.side > 0:
                    q = min(q, max(total - opt_long, 0.0) / (px + spec.fee))   # 프리미엄은 가진 자산까지만
                if q <= 0:
                    continue
                cash -= lg.side * q * px + q * spec.fee
                pos[i] = {"exp": int(ch.exp[j]), "call": lg.call, "k": float(k), "q": lg.side * q, "mark": mid, "bid": bid, "ask": ask, "tv": 0.0}
                log.append((days[t], i, "열기", float(k), lg.side * q, px))
            opt_long = sum(p["q"] * p["mark"] for p in pos if p and p["q"] > 0)
            total = shares * adj[t] + cash + sum(p["q"] * p["mark"] for p in pos if p)
            tgt = w[t] * spec.stock_frac * max(total - opt_long, 0.0) / adj[t]
            if tgt > shares:
                tgt = shares + (tgt - shares) / (1.0 + spec.stock_cost)        # 매수 비용만큼 덜 사서 현금이 음수가 되지 않게
            if abs(tgt - shares) * adj[t] > 1e-4 * total:
                cash -= (tgt - shares) * adj[t] + abs(tgt - shares) * adj[t] * spec.stock_cost; shares = tgt
        prev_w = w[t]
        eq[t] = shares * adj[t] + cash + sum(p["q"] * p["mark"] for p in pos if p)
    return pd.Series(eq, index=days), pd.DataFrame(log, columns=["date", "leg", "what", "strike", "qty", "price"])
