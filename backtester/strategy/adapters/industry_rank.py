"""산업군 순위 포트폴리오 — 월말에 알 수 있는 숫자로 산업을 줄 세워 상위 K 개를 동일 비중으로 들고 간다(순수 pandas).

입력은 전부 (월말 × 산업) 표. 점수 표의 t 행은 t 월 말까지의 자료만 쓰고, 수익은 t+1 월부터 받는다(run_topk 가 shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cum(ret: pd.DataFrame, a: int, b: int = 0) -> pd.DataFrame:
    """t-a+1 .. t-b 달의 누적 수익(전 구간에 값이 있어야 함)."""
    lg = np.log1p(ret); return np.expm1(lg.rolling(a - b, min_periods=a - b).sum().shift(b))


def signals(vw: pd.DataFrame, ew: pd.DataFrame, n: pd.DataFrame, size: pd.DataFrame, bm: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """높을수록 '사고 싶다' 인 점수 표들. 이름 → 표."""
    cap = n * size; share = cap.div(cap.sum(axis=1), axis=0); nshare = n.div(n.sum(axis=1), axis=0)
    s = {
        "mom12_1": _cum(vw, 12, 1), "mom6": _cum(vw, 6), "mom12": _cum(vw, 12), "mom1": _cum(vw, 1), "mom36": _cum(vw, 36), "mom60": _cum(vw, 60),
        "rev60": -_cum(vw, 60, 12),                                     # 5 년 진 산업(장기 반전)
        "nfirm_g12": n / n.shift(12) - 1.0, "nfirm_g36": n / n.shift(36) - 1.0,          # 새 회사가 몰려드는 산업
        "nshare_g36": nshare / nshare.shift(36) - 1.0,                   # 전체 상장사 중 비중이 느는 산업
        "share_g12": share / share.shift(12) - 1.0, "share_g36": share / share.shift(36) - 1.0, "share_g60": share / share.shift(60) - 1.0,  # 시총 비중이 느는 산업
        "share_lvl": share,                                              # 이미 큰 산업
        "small_lead12": _cum(ew, 12) - _cum(vw, 12),                     # 산업 안 작은 회사가 더 오름(저변 확산)
        "accel": _cum(vw, 6) - _cum(vw, 12, 6),                          # 최근 6 달이 그 전 6 달보다 강함
    }
    if bm is not None:
        s["bm_low"] = -bm; s["bm_high"] = bm                             # 시장이 성장을 비싸게 쳐 주는 산업 / 싼 산업
    r = lambda d: d.rank(axis=1, pct=True)  # noqa: E731
    s["newco+mom"] = (r(s["nfirm_g36"]) + r(s["mom12_1"])) / 2.0
    s["share+mom"] = (r(s["share_g36"]) + r(s["mom12_1"])) / 2.0
    s["newco+share+mom"] = (r(s["nfirm_g36"]) + r(s["share_g36"]) + r(s["mom12_1"])) / 3.0
    if bm is not None:
        s["growth_cheap"] = (r(s["nfirm_g36"]) + r(s["share_g36"]) + r(bm)) / 3.0   # 크고 있는데 아직 싼 산업
    return s


def topk_weights(score: pd.DataFrame, ret: pd.DataFrame, k: int, hold: int, rng: np.random.Generator | None = None, min_n: pd.DataFrame | None = None, n_min: int = 5) -> pd.DataFrame:
    """hold 달마다 상위 k 산업 동일 비중. rng 를 주면 같은 날 같은 수를 무작위로(플라세보). 다음 달 수익이 없는 산업·종목 수 n_min 미만 산업은 후보에서 뺀다."""
    sc = score.to_numpy(dtype=float); ok = ~np.isnan(sc) & ret.shift(-1).notna().to_numpy()
    if min_n is not None:
        ok &= min_n.reindex(index=score.index, columns=score.columns).fillna(0).to_numpy() >= n_min
    w = np.zeros(sc.shape); cur = None
    for i in range(len(sc)):
        if i % hold == 0:
            c = np.flatnonzero(ok[i])
            if len(c) >= k:
                cur = rng.choice(c, k, replace=False) if rng is not None else c[np.argsort(-sc[i, c], kind="stable")[:k]]
            else:
                cur = None
        if cur is not None:
            w[i, cur] = 1.0 / k
    return pd.DataFrame(w, index=score.index, columns=score.columns)


def port_returns(w: pd.DataFrame, ret: pd.DataFrame, rf: pd.Series, cost: float = 0.002) -> pd.Series:
    """t 월 말 비중 → t+1 월 수익. 비어 있으면 무위험. 교체한 몫에 cost(편도)."""
    wl = w.shift(1).fillna(0.0); inv = wl.sum(axis=1)
    r = (wl * ret.fillna(0.0)).sum(axis=1) + (1.0 - inv) * rf.reindex(w.index).fillna(0.0)
    return r - cost * wl.diff().abs().sum(axis=1).fillna(0.0)


def window_stats(r: pd.Series, windows: dict) -> dict:
    o = {}
    for wn, (a, z) in windows.items():
        x = r[(r.index >= a) & (r.index <= z)]; e = (1 + x).cumprod(); yrs = len(x) / 12.0
        o[wn] = {"mult": float(e.iloc[-1]), "cagr": float(e.iloc[-1] ** (1 / yrs) - 1), "mdd": float((e / e.cummax() - 1).min())}
    return o
