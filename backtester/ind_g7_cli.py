"""산업군 성장 탐색 G7 — 긴 보유(1·3·5 년)로 보면 '나중에 크게 오른 산업' 을 미리 가려내는 숫자가 있는가.

매년 6 월 말(겹치지 않게 h 년마다가 아니라 해마다, t-통계는 겹침을 감안해 유효 표본 = 연수/h 로 보수적으로)에 49 산업을 지표로 줄 세우고
 ① 순위 상관(IC): 지표 순위 vs 앞 h 년 수익 순위  ② 상위 5 산업 − 전체 평균의 앞 h 년 수익 차(연율)  ③ 그 차가 + 였던 해의 비율, 전반/후반.
지표: G1 의 가격·종목 수·시총 비중·BE/ME 22 종(1931~), G4 의 SEC 매출 성장류(2012~).
실행: docker compose run --rm --no-deps backtester python ind_g7_cli.py '{}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ind_g1_cli import load
from ind_g4_cli import scores
from strategy.adapters.industry_rank import signals

SR = Path("/app/sweep_results")


def study(S: dict, vw: pd.DataFrame, n: pd.DataFrame, end: str, horizons=(1, 3, 5), split: int | None = None) -> pd.DataFrame:
    lg = np.log1p(vw); rows = []; junes = [t for t in vw.index if t.month == 6]
    for sn, sc in S.items():
        for h in horizons:
            fwd = np.expm1(lg.rolling(12 * h, min_periods=12 * h).sum().shift(-12 * h)); ic, sp, yrs = [], [], []
            for t in junes:
                s, f = sc.loc[t], fwd.loc[t]; ok = s.notna() & f.notna() & (n.loc[t].fillna(0) >= 5)
                if ok.sum() < 20:
                    continue
                s, f = s[ok], f[ok]; ic.append(s.rank().corr(f.rank())); top = f[s.nlargest(5).index]; sp.append((1 + top.mean()) ** (1 / h) - (1 + f.mean()) ** (1 / h)); yrs.append(t.year)
            if len(ic) < 6:
                continue
            ic, sp, yrs = np.array(ic), np.array(sp), np.array(yrs); neff = max(len(ic) / h, 2); mid = split or int(np.median(yrs)); a, b = yrs <= mid, yrs > mid
            rows.append({"score": sn, "h": h, "years": len(ic), "ic": ic.mean(), "ic_t": ic.mean() / (ic.std(ddof=1) / np.sqrt(neff)), "spread": sp.mean(), "spread_t": sp.mean() / (sp.std(ddof=1) / np.sqrt(neff)), "hit": (sp > 0).mean(),
                         "spread_1st": sp[a].mean() if a.any() else np.nan, "spread_2nd": sp[b].mean() if b.any() else np.nan, "split": mid})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> list[str]:
    L = ["| 지표 | 보유(년) | 표본 연수 | 순위 상관 | t | 상위 5 − 평균(연율) | t | + 였던 해 | 전반 | 후반 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["h", "spread_t"], ascending=[True, False]).iterrows():
        strong = abs(r["spread_t"]) >= 2 and np.sign(r["spread_1st"]) == np.sign(r["spread_2nd"])
        L.append(f"| {'**' + r['score'] + '**' if strong else r['score']} | {int(r['h'])} | {int(r['years'])} | {r['ic']:+.3f} | {r['ic_t']:+.1f} | {r['spread']:+.1%} | {r['spread_t']:+.1f} | {r['hit']:.0%} | {r['spread_1st']:+.1%} | {r['spread_2nd']:+.1%} |")
    return L


def main(cfg):
    out = Path(cfg.get("out", SR / "ind_g7")); end = cfg.get("end", "2023-12-31"); vw, ew, n, size, bm, mk = load(end)
    a = study(signals(vw, ew, n, size, bm), vw, n, end, split=1975); S4, vw4, n4, _, _ = scores(end); b = study({k: v for k, v in S4.items() if "대조" not in k}, vw4, n4, end, horizons=(1, 3))
    L = [f"# G7 — 긴 보유로 본 산업 선별 지표 (자료 ~{end}, 49 산업, 매년 6 월 말 판정)", "", "굵게 = 상위 5 산업의 초과 수익이 |t| ≥ 2 이고 전반·후반 부호가 같음. t 는 보유 기간 겹침을 감안해 유효 표본 = 연수 ÷ 보유 년수 로 계산.", "",
         "## 가격·종목 수·시총 비중·BE/ME 지표 (1931~, 전반 ≤ 1975 < 후반)", ""] + table(a) + ["", "## SEC 제출 매출 성장류 (2012~)", ""] + table(b)
    ns = lambda d: int(((d["spread_t"].abs() >= 2) & (np.sign(d["spread_1st"]) == np.sign(d["spread_2nd"]))).sum())  # noqa: E731
    L += ["", f"굵은 줄: 가격류 {ns(a)} / {len(a)} · SEC 성장류 {ns(b)} / {len(b)}"]
    Path(f"{out}.md").write_text("\n".join(L)); pd.concat([a.assign(set="price"), b.assign(set="sec")]).to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
