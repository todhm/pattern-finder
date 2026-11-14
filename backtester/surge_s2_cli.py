"""급등주 탐색 S2 — 급등 "뒤" 선별 보유 사건 연구 (개발 구간만).

사건일 t 종가까지의 정보로 고르고, t+1 시가에 사서 20·60·120 거래일째 종가에 판다.
같은 날 같은 방식으로 TQQQ 를 산 결과를 빼서 "TQQQ 대비 초과" 를 같이 낸다.
폐지로 끊기면 마지막 종가 청산, 한 건 +500% 상한. 비용 미반영.
실행: docker compose exec backtester python surge_s2_cli.py '{"out": "/app/sweep_results/surge_s2"}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import add_features, event_frame, load_symbol

SR = Path("/app/sweep_results")
TRAIN_END = "2023-12-31"
HS = (20, 60, 120)

BASE = {
    "하루 +15% 이상": lambda e: e["ret"] >= 0.15,
    "하루 +30% 이상": lambda e: e["ret"] >= 0.30,
    "하루 +50% 이상": lambda e: e["ret"] >= 0.50,
}
FILT = {
    "(조건 없음)": lambda e: e["ret"] > -9,
    "거래대금 20일 평균 $30M 이상": lambda e: e["p_dv20"] >= 3e7,
    "거래대금 20일 평균 $100M 이상": lambda e: e["p_dv20"] >= 1e8,
    "주가 $10 이상": lambda e: e["p_raw_c"] >= 10,
    "사건일 종가가 52주 신고가": lambda e: e["dd_now"] >= -0.001,
    "사건 전 52주 고점 -20% 이내": lambda e: e["p_dd252"] >= -0.2,
    "사건 전 52주 고점 -70% 이하": lambda e: e["p_dd252"] <= -0.7,
    "갭으로 대부분(갭 ≥ 수익의 70%)": lambda e: e["gap"] >= 0.7 * e["ret"],
    "장중에 오름(갭 < 수익의 30%)": lambda e: e["gap"] < 0.3 * e["ret"],
    "종가가 당일 범위 상단 20%": lambda e: e["close_pos"] >= 0.8,
    "종가가 당일 범위 하단 50%": lambda e: e["close_pos"] < 0.5,
    "사건일 거래대금 $100M 이상": lambda e: e["dv"] >= 1e8,
    "거래량 평소의 10배 이상": lambda e: e["vol_mult"] >= 10,
    "지난 1년 +30% 날 없음(첫 급등)": lambda e: e["p_big_days_252"] == 0,
    "200일선 위에서 발생": lambda e: e["p_vs_sma200"] > 0,
    "신고가 & $30M & 첫 급등": lambda e: (e["dd_now"] >= -0.001) & (e["p_dv20"] >= 3e7) & (e["p_big_days_252"] == 0),
    "신고가 & 갭 & 상단 마감": lambda e: (e["dd_now"] >= -0.001) & (e["gap"] >= 0.7 * e["ret"]) & (e["close_pos"] >= 0.8),
    "$10 이상 & $30M & 고점 -20% 이내": lambda e: (e["p_raw_c"] >= 10) & (e["p_dv20"] >= 3e7) & (e["p_dd252"] >= -0.2),
    "$10 이상 & 첫 급등 & 갭": lambda e: (e["p_raw_c"] >= 10) & (e["p_big_days_252"] == 0) & (e["gap"] >= 0.7 * e["ret"]),
}


def bench(hs=HS) -> pd.DataFrame:
    t = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet")
    out = pd.DataFrame(index=t.index)
    nxt = t["O"].shift(-1)
    for n in hs:
        out[f"b{n}"] = t["C"].shift(-n) / nxt - 1.0
    out.index.name = "date"
    return out.reset_index()


def table(e: pd.DataFrame, title: str) -> list[str]:
    p = lambda x: f"{x:+.1%}"  # noqa: E731
    L = [f"## {title}", "", f"{e['date'].min().date()}~{e['date'].max().date()}", ""]
    for bname, bfn in BASE.items():
        eb = e[bfn(e)]
        L += [f"### {bname} — {len(eb):,} 건", "", "| 거르개 | 건수 | 연 평균 | 20일 평균 | 중앙 | 이긴% | TQQQ 대비 | 60일 평균 | 중앙 | 이긴% | TQQQ 대비 | 120일 평균 | 중앙 | 이긴% | TQQQ 대비 | TQQQ 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        yrs = max((e["date"].max() - e["date"].min()).days / 365.25, 0.1)
        for fname, ffn in FILT.items():
            g = eb[ffn(eb).fillna(False)]
            if len(g) < 30:
                L.append(f"| {fname} | {len(g)} | " + " | ".join(["-"] * 14) + " |"); continue
            cells = []
            for n in HS:
                x = g[f"e{n}"]; cells += [p(x.mean()), p(x.median()), f"{(x > 0).mean():.0%}", p((x - g[f'b{n}']).mean())]
            L.append(f"| {fname} | {len(g):,} | {len(g) / yrs:.0f} | " + " | ".join(cells) + f" | {(g['e120'] > g['b120']).mean():.0%} |")
        L.append("")
    return L


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s2"))
    f = pd.read_parquet(SR / "surge_panel.parquet")
    e = event_frame(f, min_ret=0.15, horizons=HS)
    e = e.merge(bench(), on="date", how="left")
    e.to_parquet(f"{out}_events.parquet")
    tr = e[(e["date"] <= TRAIN_END)].dropna(subset=["e120", "b120"])
    L = ["# S2 — 급등 뒤 선별 보유 (개발 구간)", "", "사건일 종가까지의 정보로 고르고 다음날 시가 매수, n 거래일째 종가 매도. 체결 가능(전날 주가 ≥ $1·20일 거래대금 ≥ $1M·사건일 거래대금 ≥ $1M) 만.",
         "'TQQQ 대비' = 같은 날 같은 방식으로 TQQQ 를 산 수익을 뺀 값의 평균. 비용(왕복 1~2%) 미반영. 봉인 구간(2024-01~) 미열람.", ""]
    L += table(tr, "개발 전체 2017~2023")
    L += table(tr[tr["date"] <= "2020-06-30"], "개발 전반 2017~2020-06")
    L += table(tr[tr["date"] > "2020-06-30"], "개발 후반 2020-07~2023")
    md = "\n".join(L); Path(f"{out}.md").write_text(md); print(md)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
