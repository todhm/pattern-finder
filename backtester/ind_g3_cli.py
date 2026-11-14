"""산업군 성장 탐색 G3a — 산업 모멘텀을 실제로 살 수 있는 산업·테마 ETF 로 재현 (배당 포함 조정 종가, 월말 판정·월말 체결).

유니버스: sectors = SPDR 섹터(XL*) · industries = 받은 ETF 전부에서 뱅가드 섹터(중복)·SPY·QQQ 제외. ETF 는 상장 뒤 룩백만큼 지나야 후보.
룩백 L(6·9·12) × 건너뛰기(0·1) × K(1·3·5) × 교체(1·3 달) × 절대 모멘텀(끔 | 자기 수익이 단기 국채보다 낮으면 그 칸은 현금).
기준선 SPY·QQQ·유니버스 동일비중, 플라세보 = 같은 날 같은 수 무작위. 세전, 편도 0.1%(ETF). 봉인(2024-01~)은 cfg end 명시 때만.
실행: docker compose exec backtester python ind_g3_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.industry_rank import _cum, port_returns, topk_weights, window_stats

SR = Path("/app/sweep_results")
WINDOWS = {"2000~2009": ("2000-01-31", "2009-12-31"), "2010~2017": ("2010-01-31", "2017-12-31"), "2018~2023": ("2018-01-31", "2023-12-31")}
DUP = {"VGT", "VHT", "VFH", "VDE", "VIS", "VCR", "VDC", "VPU", "VAW", "VOX", "SPY", "QQQ", "KWEB"}


def load_etf(end: str) -> pd.DataFrame:
    px = {p.stem: pd.read_parquet(p)["adj"] for p in sorted((SR / "ind_etf_cache").glob("*.parquet"))}
    m = pd.DataFrame(px).sort_index(); m = m[m.index <= end].resample("ME").last(); return m.pct_change(fill_method=None)


def main(cfg):
    out = Path(cfg.get("out", SR / "ind_g3")); end = cfg.get("end", "2023-12-31"); seeds = int(cfg.get("seeds", 200)); cost = float(cfg.get("cost", 0.001))
    ret = load_etf(end); rf = ((1 + pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"] / 252.0).resample("ME").prod() - 1).reindex(ret.index).ffill().fillna(0.0)
    W = dict(WINDOWS) if end <= "2023-12-31" else {**WINDOWS, "봉인 2024~": ("2024-01-31", end)}
    U = {"섹터 SPDR": [c for c in ret if c.startswith("XL")], "산업·테마 전부": [c for c in ret if c not in DUP]}
    base = {"SPY": window_stats(ret["SPY"].fillna(0.0), W), "QQQ": window_stats(ret["QQQ"].fillna(0.0), W)}; spy = base["SPY"]; res = []; rng = np.random.default_rng(5)
    L = [f"# G3a — 산업 모멘텀을 실제 ETF 로 (개발 ~{end}, 세전, 편도 {cost:.1%})", "", "값 = 배수(MDD). 굵게 = 그 창에서 SPY 와 무작위 95 백분위를 둘 다 넘음.", ""]
    for un, cols in U.items():
        r = ret[cols]; cnt = r.notna().sum(axis=1); ones = r.notna().astype(float).where(r.notna()); base[f"{un} 동일비중"] = window_stats(r.mean(axis=1).fillna(0.0), W); plc = {}
        for k, h in itertools.product((1, 3, 5), (1, 3)):
            m = np.array([[window_stats(port_returns(topk_weights(_cum(r, 12, 0) * 0 + ones, r, k, h, rng=rng), r, rf, cost), W)[wn]["mult"] for wn in W] for _ in range(seeds)]); plc[(k, h)] = m
        L += [f"## {un} — 후보 수 {int(cnt[cnt.index >= '2000-01-31'].iloc[0])} (2000-01) → {int(cnt.iloc[-1])} ({end[:7]})", "", "| 기준선 | " + " | ".join(W) + " |", "|---|" + "---|" * len(W)]
        L += [f"| {k} | " + " | ".join(f"×{x['mult']:.2f} ({x['mdd']:+.0%})" for x in v.values()) + " |" for k, v in base.items() if k in ("SPY", "QQQ", f"{un} 동일비중")]
        L += [f"| 무작위 {k} 개·{h} 달 (중앙 / 95) | " + " | ".join(f"×{np.median(m[:, i]):.2f} / ×{np.percentile(m[:, i], 95):.2f}" for i in range(len(W))) + " |" for (k, h), m in plc.items()]
        L += ["", "| 룩백 | 건너뛰기 | K | 교체 | 절대 모멘텀 | " + " | ".join(W) + " | 통과 창 | 플라세보 백분위 | 연 교체율 |", "|---|---|---|---|---|" + "---|" * len(W) + "---|---|---|"]
        for L_, skip, k, h, ab in itertools.product((6, 9, 12), (0, 1), (1, 3, 5), (1, 3), (False, True)):
            sc = _cum(r, L_, skip); w = topk_weights(sc, r, k, h)
            if ab:
                own = (sc * (w > 0)).where(w > 0); hurdle = np.expm1(np.log1p(rf).rolling(L_ - skip).sum().shift(skip)); w = w.where(~(own.lt(hurdle, axis=0)), 0.0)   # 고른 것 중 국채보다 못한 칸은 현금(월별 재판정)
            st = window_stats(port_returns(w, r, rf, cost), W); m = plc[(k, h)]; ok = [st[wn]["mult"] > spy[wn]["mult"] and st[wn]["mult"] > np.percentile(m[:, i], 95) for i, wn in enumerate(WINDOWS)]
            pct = [float((m[:, i] < st[wn]["mult"]).mean()) for i, wn in enumerate(W)]; turn = float(w.diff().abs().sum(axis=1).mean() * 6)
            res.append({"univ": un, "L": L_, "skip": skip, "k": k, "hold": h, "abs": ab, **{wn: st[wn]["mult"] for wn in W}, **{f"{wn}_mdd": st[wn]["mdd"] for wn in W}, **{f"{wn}_pct": p for wn, p in zip(W, pct)}, "n_ok": sum(ok), "beat_spy": sum(st[wn]["mult"] > spy[wn]["mult"] for wn in WINDOWS), "beat_qqq": sum(st[wn]["mult"] > base["QQQ"][wn]["mult"] for wn in WINDOWS), "turn": turn})
            L.append(f"| {L_} | {skip} | {k} | {h} | {'켬' if ab else '끔'} | " + " | ".join((f"**×{st[wn]['mult']:.2f}**" if (i < len(ok) and ok[i]) else f"×{st[wn]['mult']:.2f}") + f" ({st[wn]['mdd']:+.0%})" for i, wn in enumerate(W)) + f" | {sum(ok)}/{len(WINDOWS)} | " + " · ".join(f"{x:.0%}" for x in pct) + f" | {turn:.0%} |")
        L.append("")
    df = pd.DataFrame(res); nw = len(WINDOWS); L += ["## 요약", "", "| 유니버스 | 절대 모멘텀 | 칸 | 전 창 통과 | 전 창 SPY 초과 | 전 창 QQQ 초과 | " + " | ".join(f"{wn} 중앙" for wn in W) + " | " + " | ".join(f"{wn} 백분위 평균" for wn in W) + " |", "|---|---|---|---|---|---|" + "---|" * (2 * len(W))]
    for (un, ab), g in df.groupby(["univ", "abs"]):
        L.append(f"| {un} | {'켬' if ab else '끔'} | {len(g)} | {int((g['n_ok'] == nw).sum())} | {int((g['beat_spy'] == nw).sum())} | {int((g['beat_qqq'] == nw).sum())} | " + " | ".join(f"×{g[wn].median():.2f}" for wn in W) + " | " + " | ".join(f"{g[f'{wn}_pct'].mean():.0%}" for wn in W) + " |")
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1)
    print("\n".join(x for x in L if x.startswith(("#", "| SPY", "| QQQ", "| 섹터", "| 산업", "| 무작위", "| 유니버스")))); print(df.sort_values(["n_ok", "beat_spy"], ascending=False).head(12).round(2).drop(columns=[c for c in df if c.endswith(("_mdd", "_pct"))]).to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
