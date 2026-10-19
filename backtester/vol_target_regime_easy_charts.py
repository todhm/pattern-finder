"""Phase 1 쉬운설명판 그림 3장 (변동성타기팅_레짐_쉬운설명_2026_10.md).

    docker compose exec backtester python vol_target_regime_easy_charts.py /app/sweep_results/easy_vt
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams, deep_correction_filter, simulate_exposure
from strategy.adapters.vol_target_regime_lab import VtParams, rolling, target_weights
from vol_target_regime_cli import load

fp = Path("/app/pattern-finder-cache/fonts/AppleGothic.ttf")
if fp.exists():
    fm.fontManager.addfont(str(fp)); plt.rcParams["font.family"] = fm.FontProperties(fname=str(fp)).get_name()
plt.rcParams["axes.unicode_minus"] = False

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/sweep_results/easy_vt"); out.mkdir(parents=True, exist_ok=True)
lev, ndx, cash = load(date(2026, 10, 16))
costs = CostParams(tax_rate=0.22, exec_lag=1, rebalance_band=0.05)
p = VtParams(target=0.75, predictor="rv20", regime=True)
w_vt = target_weights(lev, ndx, p)
w_dd = deep_correction_filter(ndx, 200, 0.18, 20).reindex(lev.index).ffill().fillna(0.0)

# 그림 1: 브레이크와 액셀 — 2000~2003, 2020, 2022 구간의 노출
fig, axes = plt.subplots(3, 1, figsize=(12, 11))
for ax, (a, b, title) in zip(axes, [("1999-10-01", "2003-06-30", "닷컴 붕괴 2000~2002"), ("2019-10-01", "2021-03-31", "코로나 2020"), ("2021-09-01", "2023-06-30", "긴축 2022")]):
    seg = ndx[a:b]
    ax.plot(seg.index, seg / seg.iloc[0] * 100, color="black", lw=1.4, label="나스닥100 (시작=100)")
    ax2 = ax.twinx()
    ax2.fill_between(w_vt[a:b].index, 0, w_vt[a:b] * 100, color="tab:blue", alpha=0.25, step="pre", label="다이얼+스위치 노출 (%)")
    ax2.step(w_dd[a:b].index, w_dd[a:b] * 100, color="tab:red", lw=1.2, where="pre", label="스위치(⑩)만: 100 또는 0")
    ax2.set_ylim(0, 105); ax2.set_ylabel("TQQQ 비중 %")
    ax.set_title(f"{title} — 검은선: 지수, 파란 면: 다이얼이 정한 비중, 빨간선: 스위치")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out / "easy_vt_brake.png", dpi=110); plt.close(fig)

# 그림 2: 교환비 — 2010~ 보유 대비 vs MDD (세후 22% / 9.9%)
pts = []
for fn, tax in (("vt_regime.json", 0.22), ("vt_regime_hi.json", 0.22), ("vt_regime_tax10.json", 0.099)):
    d = json.load(open(f"/app/sweep_results/{fn}"))
    for r in d["samples"]["A_2010"]:
        if r["name"] == "hold" or (("_rv20" not in r["name"]) and r["name"] != "dd18_only"):
            continue
        pts.append((tax, r["name"], r["ratio_to_hold"], r["mdd"]))
fig, ax = plt.subplots(figsize=(10, 6.5))
for tax, color, lab in ((0.22, "tab:red", "양도세 22%"), (0.099, "tab:green", "세율 9.9% (ISA 가정)")):
    xs = [m for t, n, r, m in pts if t == tax]; ys = [r for t, n, r, m in pts if t == tax]
    ax.scatter([x * 100 for x in xs], ys, color=color, s=60, label=lab, zorder=3)
    for t, n, r, m in pts:
        if t == tax:
            ax.annotate(n.replace("_rv20", "").replace("_dd", "+스위치").replace("dd18_only", "스위치만").replace("vt", "다이얼"), (m * 100, r), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axhline(0.9, color="gray", ls="--"); ax.text(-81, 0.91, "기준: 보유의 90%", fontsize=9, color="gray")
ax.axvline(-60, color="gray", ls="--"); ax.text(-59.5, 0.15, "기준: MDD -60%", fontsize=9, color="gray", rotation=90)
ax.scatter([-82], [1.0], color="black", s=90, marker="*", zorder=4); ax.annotate("TQQQ 보유", (-82, 1.0), xytext=(6, -12), textcoords="offset points")
ax.set_xlabel("최대 낙폭 MDD (%) — 오른쪽일수록 덜 떨어짐"); ax.set_ylabel("2010~2026 최종 자산 ÷ TQQQ 보유")
ax.set_title("교환비: 덜 떨어지게 만들수록(오른쪽) 강세장 수익을 내준다(아래). 세율이 곡선 전체를 올린다")
ax.grid(alpha=0.3); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(out / "easy_vt_tradeoff.png", dpi=110); plt.close(fig)

# 그림 3: 10년을 어느 달에 시작했든 — 창별 CAGR
df = rolling(lev, ndx, p, costs, cash, 10, date(1986, 1, 1))
dd_rows = []
for s in pd.date_range(date(1986, 1, 1), lev.index[-1] - pd.DateOffset(years=10), freq="3MS"):
    e = s + pd.DateOffset(years=10)
    l = lev[(lev.index >= s) & (lev.index <= e)]
    st = simulate_exposure(l, w_dd.reindex(l.index), costs, cash, "dd")
    dd_rows.append((st.final_after_liquidation / st.equity.iloc[0]) ** 0.1 - 1)
fig, ax = plt.subplots(figsize=(12, 6))
x = pd.to_datetime(df["start"])
ax.bar(x, df["hold_cagr"] * 100, width=70, color="lightgray", label="TQQQ 보유")
ax.plot(x, [v * 100 for v in dd_rows], color="tab:red", lw=1.5, label="스위치(⑩)만")
ax.plot(x, df["strat_cagr"] * 100, color="tab:blue", lw=2, label="다이얼 75% + 스위치")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("그 10년의 연평균 수익률 (%)"); ax.set_xlabel("10년 보유를 시작한 달")
ax.set_title("10년을 어느 달에 시작했든 — 회색 막대가 0 아래로 내려간 곳(보유가 원금을 잃은 10년)에서 파란선은 한 번도 0 아래가 없다")
ax.grid(alpha=0.3); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(out / "easy_vt_rolling.png", dpi=110); plt.close(fig)
print("saved", [p.name for p in out.iterdir()])
