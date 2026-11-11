"""H17 — 신호 지수 교체/결합 스위치 (순수 pandas/numpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalIdxParams:
    signal: str = "spx"       # spx | both | either
    dd: float = 0.18
    n: int = 200
    reenter_n: int = 20

    @property
    def name(self) -> str:
        return f"sigidx {self.signal} dd{self.dd:.2f}"


def _conds(c: pd.Series, n: int, dd: float, re_n: int):
    sma = c.rolling(n).mean(); d = c / c.rolling(252, min_periods=20).max() - 1.0
    return ((c < sma) & (d < -dd)).to_numpy(), (c > c.rolling(re_n).mean()).to_numpy(), sma.isna().to_numpy()


def signal_index_switch(qqq: pd.Series, spx: pd.Series, p: SignalIdxParams) -> pd.Series:
    idx = qqq.dropna().index
    q = qqq.reindex(idx); s = spx.reindex(idx).ffill()
    off_q, on_q, nan_q = _conds(q, p.n, p.dd, p.reenter_n); off_s, on_s, nan_s = _conds(s, p.n, p.dd, p.reenter_n)
    if p.signal == "spx":
        off, on, nan = off_s, on_s, nan_s
    elif p.signal == "both":
        off, on, nan = off_q & off_s, on_q | on_s, nan_q | nan_s
    elif p.signal == "either":
        off, on, nan = off_q | off_s, on_q & on_s, nan_q | nan_s
    else:
        raise ValueError(p.signal)
    state = np.ones(len(idx)); cur = 1.0
    for i in range(len(idx)):
        if nan[i]:
            state[i] = 0.0; continue
        if cur > 0 and off[i]:
            cur = 0.0
        elif cur == 0 and on[i] and not off[i]:
            cur = 1.0
        state[i] = cur
    return pd.Series(state, index=idx, name=p.name)
