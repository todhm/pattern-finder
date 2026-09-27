"""3-1 / 3-2 설명 차트 (PCTY 2020-01-02 검증 문서용).

컨테이너에서 실행:
  docker compose exec backtester python /app/_tmp_pcty_chart/make_charts.py
"""
from __future__ import annotations

import json

import matplotlib
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
fm.fontManager.addfont("/app/_tmp_pcty_chart/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "/app/_tmp_pcty_chart"
BASE = pd.Timestamp("2020-01-02")

def _load(sym):
    df = pd.read_csv(f"{OUT}/{sym}.csv", index_col=0)
    df.index = pd.DatetimeIndex(pd.to_datetime([s[:10] for s in df.index]))
    return df.astype(float)


spy = _load("SPY")
pcty = _load("PCTY")

# SPY 조정 구간 (문서 3-3 표와 동일): (고점일, 저점일)
CORR = [
    ("2018-09-20", "2018-12-24", "2018 Q4 조정"),
    ("2019-05-03", "2019-06-03", "2019-05 눌림"),
    ("2019-07-26", "2019-08-05", "2019-08 눌림"),
]
# PCTY 저점 탐색 창 끝 (조정 이후 재상승 전까지)
PCTY_WIN_END = ["2019-01-31", "2019-06-30", "2019-10-31"]

facts = {}
for (hi, lo, name), win_end in zip(CORR, PCTY_WIN_END):
    hi, lo, win_end = map(pd.Timestamp, (hi, lo, win_end))
    seg = pcty.loc[hi:win_end, "Close"]
    p_lo = seg.idxmin()
    # 거래일 차이 (SPY 인덱스 기준)
    idx = spy.index
    d = idx.get_loc(p_lo) - idx.get_loc(lo)
    facts[name] = {
        "spy_low": str(lo.date()),
        "pcty_low": str(p_lo.date()),
        "pcty_low_px": round(float(seg.min()), 2),
        "lead_days": int(d),  # 음수 = PCTY가 먼저
    }

# 52주(252거래일) 고가 대비 PCTY 위치
pcty["hi52"] = pcty["High"].rolling(252, min_periods=60).max()
pcty["dist52"] = pcty["Close"] / pcty["hi52"] - 1
spy["hi52"] = spy["High"].rolling(252, min_periods=60).max()
spy["dist52"] = spy["Close"] / spy["hi52"] - 1
for (_, lo, name) in CORR:
    lo = pd.Timestamp(lo)
    facts[name]["pcty_dist52_on_spy_low"] = round(float(pcty.loc[lo, "dist52"] * 100), 1)
    facts[name]["spy_dist52_on_spy_low"] = round(float(spy.loc[lo, "dist52"] * 100), 1)

# 2019-08 눌림 이후 신고가 복귀일
def first_new_high(df, after):
    seg = df.loc[after:]
    m = seg["Close"] >= seg["hi52"].shift(1) * 0.999
    m = m & (seg["Close"] > df.loc[:after, "Close"].max())
    return seg.index[m.values][0] if m.any() else None

spy_nh = first_new_high(spy, "2019-08-05")
pcty_nh = first_new_high(pcty, "2019-10-02")
facts["new_high_after_aug"] = {"spy": str(spy_nh.date()), "pcty": str(pcty_nh.date())}
print(json.dumps(facts, ensure_ascii=False, indent=1))

RED, BLUE, GREY = "#c0392b", "#1f5fd6", "#555"
BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
X0, X1 = pd.Timestamp("2018-08-15"), pd.Timestamp("2020-01-10")


# ─────────────────────────── 3-1 ────────────────────────────
fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.32, wspace=0.18)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
axz1 = fig.add_subplot(gs[2, 0])
axz2 = fig.add_subplot(gs[2, 1])

for ax, df, label, col in ((ax1, spy, "SPY 종가", "black"), (ax2, pcty, "PCTY 종가", BLUE)):
    ax.plot(df.index, df["Close"], color=col, lw=1.3, label=label)
    ax.set_xlim(X0, X1)
    ax.grid(alpha=0.3)
    ax.set_ylabel(f"{label.split()[0]} ($)")
    for (hi, lo, name) in CORR:
        ax.axvspan(pd.Timestamp(hi), pd.Timestamp(lo), color=RED, alpha=0.08)
    ax.axvline(BASE, color=BLUE, lw=1.6)

# SPY 저점 / PCTY 저점 표시
for (hi, lo, name) in CORR:
    lo = pd.Timestamp(lo)
    f = facts[name]
    p_lo = pd.Timestamp(f["pcty_low"])
    for ax in (ax1, ax2):
        ax.axvline(lo, color=RED, ls="--", lw=1.1)
    ax1.plot(lo, spy.loc[lo, "Close"], "o", color=RED, ms=7, zorder=5)
    ax2.plot(p_lo, f["pcty_low_px"], "o", color=BLUE, ms=7, zorder=5)
    ax2.axvline(p_lo, color=BLUE, ls=":", lw=1.1)
    ax1.annotate(f"SPY 저점 {lo.date()}", (lo, spy.loc[lo, "Close"]),
                 xytext=(8, -18), textcoords="offset points", fontsize=9.5, color=RED)
    lead = f["lead_days"]
    tag = (f"PCTY 저점 {p_lo.date()}\n지수보다 {abs(lead)}거래일 먼저 (O)" if lead < 0
           else f"PCTY 저점 {p_lo.date()}\n지수와 같은 날 (=)")
    ax2.annotate(tag, (p_lo, f["pcty_low_px"]),
                 xytext=(10, 14 if name == "2018 Q4 조정" else -34), textcoords="offset points",
                 fontsize=9.5, color=BLUE, ha="left")
    if name == "2019-08 눌림":
        p2 = pd.Timestamp("2019-10-02")
        px2 = float(pcty.loc[p2, "Close"])
        ax2.plot(p2, px2, "o", color=BLUE, ms=7, zorder=5)
        ax2.axvline(p2, color=BLUE, ls=":", lw=1.1)
        ax2.annotate(f"PCTY 2차 저점 {p2.date()}\n지수 반등 뒤 8주 더 빠짐 (X)", (p2, px2),
                     xytext=(10, -34), textcoords="offset points", fontsize=9.5, color=BLUE)
        f["pcty_low2"] = str(p2.date())

ax1.text(BASE, ax1.get_ylim()[1], " 기준일 2020-01-02", color=BLUE, va="top", fontsize=10)
ax1.legend(loc="upper left", fontsize=10)
ax2.legend(loc="upper left", fontsize=10)
ax1.set_title("3-1 지수보다 먼저 바닥을 만들었나?  — SPY 저점(빨간 점선) vs PCTY 저점(파란 점선)   판정: 통과(근소)",
              fontsize=13, pad=12)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax1.get_xticklabels(), visible=False)


def zoom(ax, name, a, b, title):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    s = spy.loc[a:b, "Close"]; p = pcty.loc[a:b, "Close"]
    s_n = s / s.iloc[0] * 100; p_n = p / p.iloc[0] * 100
    ax.plot(s_n.index, s_n, color="black", lw=1.4, label="SPY (구간 첫날=100)")
    ax.plot(p_n.index, p_n, color=BLUE, lw=1.4, label="PCTY (구간 첫날=100)")
    f = facts[name]
    lo, p_lo = pd.Timestamp(f["spy_low"]), pd.Timestamp(f["pcty_low"])
    ax.axvline(lo, color=RED, ls="--", lw=1.2)
    ax.axvline(p_lo, color=BLUE, ls=":", lw=1.4)
    ax.plot(lo, s_n.loc[lo], "o", color=RED, ms=7, zorder=5)
    ax.plot(p_lo, p_n.loc[p_lo], "o", color=BLUE, ms=7, zorder=5)
    ax.annotate(f"SPY 저점\n{lo.strftime('%m-%d')}", (lo, s_n.loc[lo]),
                xytext=(8, 14), textcoords="offset points", color=RED, fontsize=9, bbox=BOX)
    ax.annotate(f"PCTY 저점\n{p_lo.strftime('%m-%d')}", (p_lo, p_n.loc[p_lo]),
                xytext=(-66, 10), textcoords="offset points", color=BLUE, fontsize=9, bbox=BOX)
    lo_y = min(s_n.min(), p_n.min()); hi_y = max(s_n.max(), p_n.max())
    ax.set_ylim(lo_y - 3.5, hi_y + 1.5)
    if "pcty_low2" in f:
        p2 = pd.Timestamp(f["pcty_low2"])
        ax.axvline(p2, color=BLUE, ls=":", lw=1.4)
        ax.plot(p2, p_n.loc[p2], "o", color=BLUE, ms=7, zorder=5)
        ax.annotate(f"PCTY 2차 저점\n{p2.strftime('%m-%d')}", (p2, p_n.loc[p2]),
                    xytext=(10, -6), textcoords="offset points", color=BLUE, fontsize=9, bbox=BOX)
        s_hi = s_n.idxmax()
        ax.annotate(f"SPY는 이미 고점 부근", (p2, s_n.loc[p2]), xytext=(-110, 12),
                    textcoords="offset points", color="black", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_ylabel("정규화 (구간 시작=100)")


zoom(axz1, "2018 Q4 조정", "2018-12-03", "2019-01-18",
     "확대 ① 2018-12: PCTY 12-21 저점 → SPY 12-24 저점 (1거래일 선행)")
zoom(axz2, "2019-08 눌림", "2019-07-22", "2019-10-25",
     "확대 ② 2019-08~10: 08-05 같은 날 저점 → PCTY만 9월 소프트웨어 로테이션 때 재하락, 10-02 2차 저점 (후행)")

fig.text(0.5, 0.015,
         "판정 기준: 지수 조정 때 주도주는 지수보다 먼저 하락을 멈추고(저점이 앞서고) 지수가 돌아설 때 이미 오르고 있어야 한다. "
         "PCTY는 큰 조정(2018 Q4)에서는 1거래일 앞섰고, 2019년 두 눌림은 지수와 같은 날 바닥이었으나 2019-08 눌림 뒤에는 지수가 반등하는 동안 8주 더 빠져 10-02 2차 저점을 만들었다 → '근소 통과'.",
         ha="center", fontsize=9.5, color=GREY)
fig.savefig(f"{OUT}/pcty_3_1_bottom_lead.png", dpi=110, bbox_inches="tight")
plt.close(fig)


# ─────────────────────────── 3-2 ────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True,
                               gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.08})
ax1.plot(pcty.index, pcty["Close"], color=BLUE, lw=1.3, label="PCTY 종가")
ax1.plot(pcty.index, pcty["hi52"], color="#7f8c8d", lw=1.1, ls="--", label="PCTY 52주 고가(최근 252거래일 최고가)")
ax1.fill_between(pcty.index, pcty["Close"], pcty["hi52"], color=RED, alpha=0.10, label="52주 고가와의 거리")
for (hi, lo, name) in CORR:
    ax1.axvspan(pd.Timestamp(hi), pd.Timestamp(lo), color=RED, alpha=0.06)
    ax2.axvspan(pd.Timestamp(hi), pd.Timestamp(lo), color=RED, alpha=0.06)
ax1.set_xlim(X0, X1); ax1.grid(alpha=0.3); ax1.set_ylabel("PCTY ($)")

ax2.plot(pcty.index, pcty["dist52"] * 100, color=BLUE, lw=1.3, label="PCTY: 52주 고가 대비 %")
ax2.plot(spy.index, spy["dist52"] * 100, color="black", lw=1.0, alpha=0.7, label="SPY: 52주 고가 대비 %")
ax2.axhspan(-5, 0, color="green", alpha=0.10)
ax2.annotate("초록 띠 = 책의 이상형: 지수 저점일에도 신고가 -5% 이내 (PCYC 유형)",
             (pd.Timestamp("2019-11-25"), -5), xytext=(pd.Timestamp("2019-06-20"), -37),
             textcoords="data", color="green", fontsize=9.5, bbox=BOX,
             arrowprops=dict(arrowstyle="->", color="green"))
ax2.axhline(0, color="#7f8c8d", lw=0.8)
ax2.set_ylim(-45, 4); ax2.grid(alpha=0.3); ax2.set_ylabel("52주 고가 대비 (%)")

for (hi, lo, name) in CORR:
    lo = pd.Timestamp(lo)
    d = facts[name]["pcty_dist52_on_spy_low"]
    ds = facts[name]["spy_dist52_on_spy_low"]
    for ax in (ax1, ax2):
        ax.axvline(lo, color=RED, ls="--", lw=1.1)
    ax1.plot(lo, pcty.loc[lo, "Close"], "o", color=RED, ms=7, zorder=5)
    off = {"2018 Q4 조정": (10, 6), "2019-05 눌림": (-178, -34), "2019-08 눌림": (8, -34)}[name]
    ax1.annotate(f"SPY 저점일 {lo.date()}\nPCTY {pcty.loc[lo,'Close']:.2f} vs 52주고가 {pcty.loc[lo,'hi52']:.2f}",
                 (lo, pcty.loc[lo, "Close"]), xytext=off, textcoords="offset points",
                 fontsize=9, color=RED, bbox=BOX)
    ax2.plot(lo, d, "o", color=RED, ms=7, zorder=5)
    off2 = {"2018 Q4 조정": (8, -22), "2019-05 눌림": (8, -40), "2019-08 눌림": (8, -66)}[name]
    ax2.annotate(f"PCTY {d:+.0f}%  (SPY {ds:+.1f}%)\n→ 신고가 근처 아님 (X)", (lo, d),
                 xytext=off2, textcoords="offset points", fontsize=9.5, color=RED, bbox=BOX)

for ax in (ax1, ax2):
    ax.axvline(BASE, color=BLUE, lw=1.6)
ax1.text(BASE, ax1.get_ylim()[1], " 기준일 2020-01-02\n 52주 고가 -0.1%", color=BLUE, va="top", fontsize=10, ha="right")
# 신고가 복귀 표시
ax1.plot(pcty_nh, pcty.loc[pcty_nh, "Close"], "^", color="green", ms=9, zorder=5)
ax1.annotate(f"PCTY 신고가 복귀 {pcty_nh.date()}", (pcty_nh, pcty.loc[pcty_nh, "Close"]),
             xytext=(-150, 18), textcoords="offset points", fontsize=9.5, color="green",
             arrowprops=dict(arrowstyle="->", color="green"))
ax1.legend(loc="upper left", fontsize=10)
ax2.legend(loc="lower center", fontsize=9.5)
ax1.set_title("3-2 지수 조정 저점일에 PCTY는 신고가(근접)였나?  — 세 번의 SPY 저점일 모두 52주 고가에서 -13~-38%   판정: 미충족(보너스)",
              fontsize=13, pad=12)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.text(0.5, 0.02,
         "판정 기준: 시장이 무너지는 날 오히려 신고가(-5% 이내)를 쓰는 종목이 최상위 주도주(책의 PCYC 사례). "
         "PCTY는 지수 저점일마다 지수보다 2~3배 깊게 빠져 있었고(고베타 소프트웨어), 지수 반등 후 따라 올라 11월에 신고가로 복귀한 '추종형' → 보너스 미충족.",
         ha="center", fontsize=9.5, color=GREY)
fig.savefig(f"{OUT}/pcty_3_2_new_high_on_spy_low.png", dpi=110, bbox_inches="tight")
print("done")
