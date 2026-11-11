"""H43 완성 패키지(R60) 쉬운 설명용 그림·이벤트 표.

실행: docker compose exec backtester python tqqq_h43_explain_cli.py '{"out_dir": "/app/sweep_results/easy_tqqq_h43"}'
설명 문서: docs/strategy_notes/TQQQ_탐색_최고결과_쉬운설명_2026-11.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.regime_slots import RegimeParams, regime_labels
from strategy.adapters.multi_asset_sim import simulate_weights
from strategy.adapters.tqqq_gates import GATE_COSTS
from tqqq_gates_cli import PRICES, load_cash_rate
from tqqq_h11_cli import load_qld
from tqqq_h43_cli import build

START, END = pd.Timestamp("2010-02-11"), pd.Timestamp("2026-10-24")
DD, DEPTH, R1 = (0.18, 0.24), 0.35, 0.50


def setup_font():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = "/app/pattern-finder-cache/fonts/AppleGothic.ttf"
    try:
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    except Exception:  # noqa: BLE001
        pass
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def sim(prices, w, cash):
    return simulate_weights(prices, w, GATE_COSTS, cash, "x")


def main(cfg):
    out_dir = Path(cfg.get("out_dir", "/app/sweep_results/easy_tqqq_h43")); out_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_font()
    px = pd.read_parquet(PRICES); cash = load_cash_rate()
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": load_qld(px.index, cash), "QQQ": px["QQQ"]})
    labels_cache = {DEPTH: regime_labels(px["QQQ"], RegimeParams(c_depth=DEPTH))}
    lab, slots, w = build(px, DD, DEPTH, R1, labels_cache)
    m = (px.index >= START) & (px.index <= END)
    P = prices[m]; W = w[m]; L = lab.reindex(px.index).ffill()[m]
    hold_t = pd.DataFrame(0.0, index=P.index, columns=P.columns); hold_t["TQQQ"] = 1.0
    hold_q = hold_t.copy(); hold_q["TQQQ"] = 0.0; hold_q["QQQ"] = 1.0
    rs, rt, rq = sim(P, W, cash), sim(P, hold_t, cash), sim(P, hold_q, cash)
    summary = {k: {"after_tax_mult": r.final_after_liquidation / GATE_COSTS.initial_capital, "mdd": r.metrics["mdd"], "trades": r.trades, "tax_over_capital": r.tax / GATE_COSTS.initial_capital}
               for k, r in (("package", rs), ("tqqq", rt), ("qqq", rq))}

    q = px["QQQ"][m]; t = px["TQQQ"][m]
    sma200 = px["QQQ"].rolling(200).mean()[m]; sma20 = px["QQQ"].rolling(20).mean()[m]
    ddq = (px["QQQ"] / px["QQQ"].rolling(252, min_periods=20).max() - 1.0)[m]
    ddt = (px["TQQQ"] / px["TQQQ"].rolling(252, min_periods=20).max() - 1.0)[m]
    tw = W["TQQQ"]

    # --- 이벤트 표(목표 비중이 바뀐 날)
    chg = tw[tw.diff().fillna(0.0).abs() > 1e-9]
    events = []
    for d, v in chg.items():
        prev = float(tw.shift(1).loc[d])
        events.append({"date": str(d.date()), "from": round(prev, 3), "to": round(float(v), 3), "regime": L.loc[d],
                       "qqq_dd": round(float(ddq.loc[d]), 3), "tqqq_dd": round(float(ddt.loc[d]), 3),
                       "qqq_vs_sma200": round(float(q.loc[d] / sma200.loc[d] - 1.0), 3), "qqq_vs_sma20": round(float(q.loc[d] / sma20.loc[d] - 1.0), 3)})

    # --- 그림 1: 전체 곡선 + 비중 + 낙폭
    fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.4, 1.4]})
    ax[0].plot(rs.equity.index, rs.equity / GATE_COSTS.initial_capital, color="#d62728", lw=1.8, label=f"패키지(H43) 세후 ×{rs.final_after_liquidation / GATE_COSTS.initial_capital:,.0f}")
    ax[0].plot(rt.equity.index, rt.equity / GATE_COSTS.initial_capital, color="#1f77b4", lw=1.2, label=f"TQQQ 그냥 보유 세후 ×{rt.final_after_liquidation / GATE_COSTS.initial_capital:,.0f}")
    ax[0].plot(rq.equity.index, rq.equity / GATE_COSTS.initial_capital, color="#7f7f7f", lw=1.0, label=f"QQQ 그냥 보유 세후 ×{rq.final_after_liquidation / GATE_COSTS.initial_capital:,.0f}")
    ax[0].set_yscale("log"); ax[0].set_title("2010-02-11 에 1 을 넣었을 때 몇 배가 됐나 (로그 축, 세금은 해마다 정산, 마지막에 전량 매도·세금 납부)"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=.3)
    c_mask = (L == "C")
    for a in ax:
        a.fill_between(L.index, 0, 1, where=c_mask, transform=a.get_xaxis_transform(), color="#9467bd", alpha=.15, lw=0)
    ax[1].fill_between(tw.index, 0, tw, step="post", color="#d62728", alpha=.5); ax[1].set_ylim(-.05, 1.05)
    ax[1].set_title("TQQQ 목표 비중 (1 = 전부 TQQQ, 0 = 전부 현금) — 보라색 띠 = '긴 약세장' 으로 판정한 기간"); ax[1].grid(alpha=.3)
    ax[2].plot(ddq.index, ddq, color="#1f77b4", lw=1); ax[2].axhline(-.18, color="#ff7f0e", ls="--", lw=1); ax[2].axhline(-.24, color="#d62728", ls="--", lw=1); ax[2].axhline(-.35, color="#9467bd", ls=":", lw=1)
    ax[2].set_title("QQQ 가 52주 고점에서 얼마나 빠졌나 — 주황 -18% 절반 매도 / 빨강 -24% 전량 매도 / 보라 -35% 긴 약세장 (모두 200일선 아래일 때만)"); ax[2].grid(alpha=.3)
    ax[3].plot(ddt.index, ddt, color="#2ca02c", lw=1)
    for lv in (-.5, -.65, -.8):
        ax[3].axhline(lv, color="#2ca02c", ls="--", lw=.8)
    ax[3].set_title("TQQQ 가 52주 고점에서 얼마나 빠졌나 — 현금일 때 -50/-65/-80% 선에 닿으면 ⅓ 씩 되산다(사다리)"); ax[3].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "easy_01_overview.png", dpi=110); plt.close(fig)

    # --- 그림 2: 에피소드 확대 (2018-12, 2020-03, 2022~23)
    zooms = [("2018-09-01", "2019-06-30", "2018년 말 조정: -18% 에 절반 → 20일선 회복에 되삼"),
             ("2020-01-15", "2020-07-31", "2020년 코로나: 절반→전량 매도 → TQQQ -50/-65% 사다리 ⅓·⅓ → 20일선에 나머지"),
             ("2021-11-01", "2023-06-30", "2022년 긴 약세장: 사다리 → -35% 넘자 '긴 약세장' 모드(월말에만 판단) → 2023-01 말 복귀")]
    fig, ax = plt.subplots(3, 2, figsize=(14, 12), gridspec_kw={"width_ratios": [3, 1]})
    for i, (s, e, title) in enumerate(zooms):
        mm = (q.index >= s) & (q.index <= e)
        a = ax[i, 0]; a.plot(q.index[mm], q[mm], color="k", lw=1.2, label="QQQ")
        a.plot(q.index[mm], sma200[mm], color="#1f77b4", lw=1, label="200일선"); a.plot(q.index[mm], sma20[mm], color="#2ca02c", lw=1, ls="--", label="20일선")
        a2 = a.twinx(); a2.fill_between(tw.index[mm], 0, tw[mm], step="post", color="#d62728", alpha=.25, label="TQQQ 비중"); a2.set_ylim(0, 1.05); a2.set_ylabel("TQQQ 비중")
        a.fill_between(L.index[mm], 0, 1, where=c_mask[mm], transform=a.get_xaxis_transform(), color="#9467bd", alpha=.15, lw=0)
        a.set_title(title, fontsize=11); a.legend(loc="lower left", fontsize=8); a.grid(alpha=.3)
        b = ax[i, 1]; b.plot(ddt.index[mm], ddt[mm], color="#2ca02c", lw=1)
        for lv in (-.5, -.65, -.8):
            b.axhline(lv, color="#2ca02c", ls="--", lw=.8)
        b.set_title("TQQQ 52주 고점 대비 낙폭", fontsize=10); b.grid(alpha=.3)
        for lab_ in b.get_xticklabels():
            lab_.set_rotation(30)
    fig.tight_layout(); fig.savefig(out_dir / "easy_02_episodes.png", dpi=110); plt.close(fig)

    # --- 그림 3: 창별 결과(봉인 해제 판 A-10 28창)
    sealed = json.load(open("/app/sweep_results/tqqq_h43_G7_sealed.json"))
    wdf = pd.DataFrame(sealed["windows"]); a10 = wdf[wdf.bundle == "A-10"].copy(); a10["start"] = pd.to_datetime(a10["start"])
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(a10)); lbl = [d.strftime("%Y-%m") for d in a10["start"]]
    ax[0].bar(x - .2, a10["strat_ret"] * 100, width=.4, color="#d62728", label="패키지 세후 수익률")
    ax[0].bar(x + .2, a10["tqqq_ret"] * 100, width=.4, color="#1f77b4", label="TQQQ 보유 세후 수익률")
    ax[0].set_yscale("log"); ax[0].set_ylabel("%(로그)"); ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
    ax[0].set_title("10년 창 28개(시작일만 분기마다 다름, 끝 = 시작+10년): 어느 해에 시작했든 비교")
    ax[1].bar(x - .2, a10["strat_mdd"] * 100, width=.4, color="#d62728", label="패키지 최대 낙폭")
    ax[1].bar(x + .2, a10["tqqq_mdd"] * 100, width=.4, color="#1f77b4", label="TQQQ 최대 낙폭")
    ax[1].set_ylabel("%"); ax[1].legend(loc="lower left"); ax[1].grid(alpha=.3, axis="y")
    ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, rotation=60, fontsize=8); ax[1].set_xlabel("창 시작 월")
    fig.tight_layout(); fig.savefig(out_dir / "easy_03_windows.png", dpi=110); plt.close(fig)

    # --- 그림 4: 규칙 순서도
    fig, a = plt.subplots(figsize=(13, 8.5)); a.axis("off")
    def box(xy, wh, text, fc):
        from matplotlib.patches import FancyBboxPatch
        a.add_patch(FancyBboxPatch(xy, *wh, boxstyle="round,pad=0.02", fc=fc, ec="#444", lw=1.2))
        a.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=10.5, linespacing=1.5)
    def arrow(p, q_, text=""):
        a.annotate("", xy=q_, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))
        if text:
            a.text((p[0] + q_[0]) / 2 + .01, (p[1] + q_[1]) / 2, text, fontsize=9.5, color="#333", va="center")
    box((.33, .86), (.34, .10), "매일 장 마감 후 QQQ 종가로 판단\n(체결은 다음 거래일)", "#f0f0f0")
    box((.30, .66), (.40, .12), "① 지금 '긴 약세장' 인가?\nQQQ 가 200일선 아래 상태가 이어지는 중이고\n(52주 고점 대비 -35% 이상 또는 아래 체류 120거래일 이상)", "#e8e0f5")
    arrow((.50, .86), (.50, .78))
    box((.04, .40), (.36, .18), "② 긴 약세장 모드\n그 달 마지막 거래일에만 스위치 18 판단\n· 200일선 아래 & 고점 -18% → 전량 매도\n· 20일선 위 → 전량 매수\n(사다리 없음, 월중엔 아무것도 안 함)", "#ddd0f0")
    box((.56, .38), (.40, .20), "③ 평소 모드 — 두 단계 이탈\n200일선 아래이면서\n· 고점 -18% → TQQQ 절반 매도\n· 고점 -24% → 전량 매도\nQQQ 가 20일선 위로 올라오면 전량 매수\n(단 아직 200일선 아래 & -18% 이면 보류)", "#fde0dc")
    arrow((.40, .66), (.22, .58)); arrow((.60, .66), (.76, .58)); a.text(.27, .635, "예", fontsize=11, fontweight="bold"); a.text(.70, .635, "아니오", fontsize=11, fontweight="bold")
    box((.56, .12), (.40, .20), "④ 사다리(현금·부분 상태일 때만)\nTQQQ 가 자기 52주 고점 대비\n-50% → ⅓ 매수 · -65% → ⅓ 더 · -80% → ⅓ 더\n하락 에피소드당 칸마다 한 번, TQQQ 가 새 고점을 찍으면 재장전\n20일선 재진입이 오면 어차피 100%", "#dff2df")
    arrow((.76, .38), (.76, .32))
    a.text(.5, .03, "비용 가정: 수수료+슬리피지 왕복 약 0.3%, 양도세 22%(연 통산, 250만 원 공제), 현금은 미국 3개월 국채 이자", ha="center", fontsize=9.5, color="#555")
    a.set_xlim(0, 1); a.set_ylim(0, 1); fig.savefig(out_dir / "easy_00_rule.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    out = {"summary": summary, "events": events, "regime_C_days": int(c_mask.sum()),
           "C_episodes": [(str(g.index[0].date()), str(g.index[-1].date())) for _, g in L[c_mask].groupby((~c_mask).cumsum()[c_mask])]}
    (out_dir / "easy_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
