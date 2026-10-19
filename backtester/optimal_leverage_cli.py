"""Phase 0 — 나스닥100 연도별 μ·σ·L* (성장 최적 노출 배수) 표.

    docker compose exec backtester python optimal_leverage_cli.py '{"out": "/app/sweep_results/optimal_leverage.json", "md": "/app/sweep_results/optimal_leverage.md"}'

정의 (연도별, 일간 단순수익 r_t, 현금금리 c_t=^IRX/100/252):
- μ  = 252 · mean(r_t − c_t)      초과수익(연환산)
- σ  = sqrt(252) · std(r_t)        변동성(연환산)
- L* = μ / σ²                      장기 성장률 g(L) ≈ Lμ − L²σ²/2 를 최대로 하는 노출 배수
- g(L) 실현값 = 그 해 "L배 일일 리밸런싱" 의 실제 복리 수익 (L=1 지수, L=3 합성 3배: 3r_t − 2c_t − 드래그/252)
- TQQQ 실제 = 상장(2010-02) 이후 실제 TQQQ 연수익
로드맵 `TQQQ_초과수익_로드맵_2026-10.md` §2·§6 Phase 0.
"""

from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from data.adapters.composed_market_data import build_default_market_data

DRAG = 0.0251  # 실측 드래그 (leveraged_etf_data.CALIBRATED_DRAG 3x)


def _close(md, sym, start, end):
    df = md.fetch_ohlcv(sym, start, end)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()  # ^IRX 는 01:00 시각이 붙어 있어 날짜로 맞춘다
    s = df["Close"].astype(float).sort_index()
    return s[~s.index.duplicated(keep="last")]


def main(cfg: dict) -> dict:
    md = build_default_market_data()
    end = date.fromisoformat(cfg.get("as_of", "2026-10-16"))
    start = date(1985, 1, 1)
    ndx = None
    for sym in ("^NDX", "NDX"):
        try:
            ndx = _close(md, sym, start, end)
            if len(ndx) > 1000:
                break
        except Exception:  # noqa: BLE001
            ndx = None
    qqq = _close(md, "QQQ", date(1999, 3, 10), end)
    tqqq = _close(md, "TQQQ", date(2010, 2, 11), end)
    irx = (_close(md, "^IRX", start, end) / 100.0).dropna()
    if irx.empty:
        raise SystemExit("^IRX 없음 — 현금금리 데이터 필요")
    # 기초지수: ^NDX 가 있으면 1985~, 없으면 QQQ 1999~
    under = ndx if ndx is not None and len(ndx) > 1000 else qqq
    under_name = "^NDX" if under is ndx else "QQQ"
    r = under.pct_change().dropna()
    c = irx.reindex(r.index).ffill().bfill() / 252.0
    rows = []
    for y, g in r.groupby(r.index.year):
        cy = c.loc[g.index]
        if len(g) < 200 and y != end.year:
            continue
        mu = 252 * float((g - cy).mean())
        sig = float(np.sqrt(252) * g.std())
        lstar = mu / sig**2 if sig > 0 else float("nan")
        ret1 = float((1 + g).prod() - 1)
        r3 = 3 * g - 2 * cy - DRAG / 252
        ret3 = float((1 + r3).prod() - 1)
        row = {"year": int(y), "days": int(len(g)), "mu_excess": mu, "sigma": sig, "L_star": lstar,
               "ret_1x": ret1, "ret_3x_synth": ret3, "cash": float(252 * cy.mean())}
        if y >= 2010:
            tq = tqqq[tqqq.index.year == y]
            if len(tq) > 1:
                prev = tqqq[tqqq.index < tq.index[0]]
                base = prev.iloc[-1] if len(prev) else tq.iloc[0]
                row["ret_tqqq"] = float(tq.iloc[-1] / base - 1)
        rows.append(row)
    # 전체 구간 요약
    def summ(lo, hi):
        m = (r.index.year >= lo) & (r.index.year <= hi)
        g, cy = r[m], c[m]
        mu = 252 * float((g - cy).mean()); sig = float(np.sqrt(252) * g.std())
        yrs = len(g) / 252
        r3 = 3 * g - 2 * cy - DRAG / 252
        return {"period": f"{lo}~{hi}", "mu_excess": mu, "sigma": sig, "L_star": mu / sig**2,
                "cagr_1x": float((1 + g).prod() ** (1 / yrs) - 1), "cagr_3x_synth": float((1 + r3).prod() ** (1 / yrs) - 1),
                "g_formula_1x": mu - sig**2 / 2 + float(252 * cy.mean()), "g_formula_3x": 3 * mu - 9 * sig**2 / 2 + float(252 * cy.mean()) - DRAG}
    periods = [summ(a, b) for a, b in ((1986, 1999), (2000, 2009), (2010, 2026), (1986, 2026), (2000, 2026))
               if r.index.year.min() <= b and r.index.year.max() >= a]
    out = {"underlying": under_name, "as_of": end.isoformat(), "drag": DRAG, "years": rows, "periods": periods}
    if cfg.get("out"):
        with open(cfg["out"], "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    md_lines = [f"# 나스닥100 연도별 μ·σ·L* ({under_name}, ~{end}) — Phase 0", "",
                "| 연도 | μ 초과 | σ | **L\\*** | 1배 실제 | 3배 합성 | TQQQ 실제 | 3배가 최적? |", "|---|---|---|---|---|---|---|---|"]
    for w in rows:
        best = "✅ 3 이상" if w["L_star"] >= 3 else ("⚠️ 1~3" if w["L_star"] >= 1 else ("❌ 0~1" if w["L_star"] >= 0 else "❌ 음수(현금)"))
        tq = f"{w['ret_tqqq']:+.0%}" if "ret_tqqq" in w else "—"
        md_lines.append(f"| {w['year']} | {w['mu_excess']:+.0%} | {w['sigma']:.0%} | **{w['L_star']:+.1f}** | {w['ret_1x']:+.0%} | {w['ret_3x_synth']:+.0%} | {tq} | {best} |")
    md_lines += ["", "| 구간 | μ 초과 | σ | L\\* | 1배 CAGR | 3배 합성 CAGR | 공식 g(1) | 공식 g(3) |", "|---|---|---|---|---|---|---|---|"]
    for p in periods:
        md_lines.append(f"| {p['period']} | {p['mu_excess']:+.1%} | {p['sigma']:.1%} | **{p['L_star']:.2f}** | {p['cagr_1x']:+.1%} | {p['cagr_3x_synth']:+.1%} | {p['g_formula_1x']:+.1%} | {p['g_formula_3x']:+.1%} |")
    text = "\n".join(md_lines)
    if cfg.get("md"):
        with open(cfg["md"], "w") as f:
            f.write(text + "\n")
    print(text)
    return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
