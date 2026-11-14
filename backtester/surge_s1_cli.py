"""급등주 탐색 S1 — 전날 특징 구간별 "다음날 시가에 사면" 성과표 (개발 구간만).

특징은 전날 종가까지(p_*), 진입은 당일 시가, 청산은 당일 종가(1일)·5일째·20일째 종가.
폐지로 가격이 끊기면 마지막 종가로 청산한 것으로 본다. 한 건 수익은 +500% 에서 자른다.
실행: docker compose exec backtester python surge_s1_cli.py '{"out": "/app/sweep_results/surge_s1"}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import hold_returns, liquid

SR = Path("/app/sweep_results")
TRAIN_END = "2023-12-31"

CONDS = {
    "전체(체결 가능)": lambda f: f["p_raw_c"] > 0,
    "52주 고점 -90% 이하": lambda f: f["p_dd252"] <= -0.9,
    "52주 고점 -70~-90%": lambda f: (f["p_dd252"] > -0.9) & (f["p_dd252"] <= -0.7),
    "20일 수익 -30% 이하": lambda f: f["p_ret20"] <= -0.3,
    "20일 수익 +30~100%": lambda f: (f["p_ret20"] > 0.3) & (f["p_ret20"] <= 1.0),
    "20일 수익 +100% 초과": lambda f: f["p_ret20"] > 1.0,
    "5일 거래량 4배 초과": lambda f: f["p_vol_ratio5"] > 4,
    "5일 거래량 2~4배": lambda f: (f["p_vol_ratio5"] > 2) & (f["p_vol_ratio5"] <= 4),
    "5일 거래량 0.5배 이하": lambda f: f["p_vol_ratio5"] <= 0.5,
    "1년 +30% 날 4회 이상": lambda f: f["p_big_days_252"] > 3,
    "1년 +30% 날 2~3회": lambda f: (f["p_big_days_252"] > 1) & (f["p_big_days_252"] <= 3),
    "주가 $1~2": lambda f: f["p_raw_c"] <= 2,
    "주가 $2~5": lambda f: (f["p_raw_c"] > 2) & (f["p_raw_c"] <= 5),
    "전날 +30% 이상": lambda f: f["p_ret1"] >= 0.3,
    "전날 +50% 이상": lambda f: f["p_ret1"] >= 0.5,
    "전날 +100% 이상": lambda f: f["p_ret1"] >= 1.0,
    "전날 -30% 이하": lambda f: f["p_ret1"] <= -0.3,
    "거래량 4배 & 20일 +30% 초과": lambda f: (f["p_vol_ratio5"] > 4) & (f["p_ret20"] > 0.3),
    "거래량 4배 & 20일 -10~+30%": lambda f: (f["p_vol_ratio5"] > 4) & (f["p_ret20"] > -0.1) & (f["p_ret20"] <= 0.3),
    "-90% & 거래량 2배 초과": lambda f: (f["p_dd252"] <= -0.9) & (f["p_vol_ratio5"] > 2),
    "-70% 이하 & 1년 +30% 날 2회 이상": lambda f: (f["p_dd252"] <= -0.7) & (f["p_big_days_252"] > 1),
    "200일선 위 & 거래량 4배 & 급등 이력": lambda f: (f["p_vs_sma200"] > 0) & (f["p_vol_ratio5"] > 4) & (f["p_big_days_252"] > 0),
    "신고가 근처(-10% 이내) & 20일 +30% 초과": lambda f: (f["p_dd252"] > -0.1) & (f["p_ret20"] > 0.3),
    "신고가 근처 & 거래량 2배 초과": lambda f: (f["p_dd252"] > -0.1) & (f["p_vol_ratio5"] > 2),
}


def table(f: pd.DataFrame, title: str) -> list[str]:
    days = f["date"].nunique()
    L = [f"## {title}", "", f"{f['date'].min().date()}~{f['date'].max().date()} · {days:,} 거래일 · 체결 가능 {len(f):,} 종목-일", "",
         "| 조건 | 종목-일 | 하루 후보 수 | +50% 확률(1만당) | 시가→당일 종가 평균 | 중앙 | 이긴 비율 | →5일 평균 | 중앙 | →20일 평균 | 중앙 | 20일 이긴 비율 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    p = lambda x: f"{x:+.2%}"  # noqa: E731
    for name, fn in CONDS.items():
        g = f[fn(f).fillna(False)]
        if len(g) < 50:
            L.append(f"| {name} | {len(g):,} | - | - | - | - | - | - | - | - | - | - |"); continue
        L.append(f"| {name} | {len(g):,} | {len(g) / days:.1f} | {(g['surge50']).mean() * 1e4:.1f} | {p(g['h1'].mean())} | {p(g['h1'].median())} | {(g['h1'] > 0).mean():.0%} | {p(g['h5'].mean())} | {p(g['h5'].median())} | {p(g['h20'].mean())} | {p(g['h20'].median())} | {(g['h20'] > 0).mean():.0%} |")
    return L + [""]


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s1"))
    f = pd.read_parquet(SR / "surge_panel.parquet")
    f = hold_returns(f)
    f = f[liquid(f)]
    f["surge50"] = (f["ret"] >= 0.5) & (f["dv"] >= 1e6)
    tr = f[f["date"] <= TRAIN_END]
    L = ["# S1 — 전날 특징 구간별 다음날 시가 매수 성과 (개발 구간)", "", "비용 미반영(왕복 1~2% 를 빼고 읽을 것). 봉인 구간(2024-01~)은 여기서 보지 않는다.", ""]
    L += table(tr, "개발 전체 2017~2023")
    L += table(tr[tr["date"] <= "2020-06-30"], "개발 전반 2017~2020-06")
    L += table(tr[tr["date"] > "2020-06-30"], "개발 후반 2020-07~2023")
    md = "\n".join(L); Path(f"{out}.md").write_text(md); print(md)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
