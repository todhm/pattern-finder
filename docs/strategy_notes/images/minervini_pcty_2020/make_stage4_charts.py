"""STAGE 4 설명 차트 3장 (PCTY 2020-01-02 검증 문서용).

컨테이너에서 실행:
  docker compose exec backtester python /app/_tmp_pcty_chart/make_stage4_charts.py
전제: /app/_tmp_pcty_chart/PCTY.csv (build_default_market_data().fetch_ohlcv 덤프),
      /app/_tmp_pcty_chart/AppleSDGothicNeo.ttc (호스트에서 복사)
"""
from __future__ import annotations

import matplotlib
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
fm.fontManager.addfont("/app/_tmp_pcty_chart/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["text.parse_math"] = False

OUT = "/app/_tmp_pcty_chart"
T = pd.Timestamp
BASE = T("2020-01-02")
RED, BLUE, GREEN, GREY, ORANGE = "#c0392b", "#1f5fd6", "#1e8449", "#555", "#e67e22"
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)

df = pd.read_csv(f"{OUT}/PCTY.csv", index_col=0)
df.index = pd.DatetimeIndex(pd.to_datetime([s[:10] for s in df.index]))
df = df.astype(float)
df["sma50"] = df["Close"].rolling(50).mean()
df["vol50"] = df["Volume"].rolling(50).mean()

# ── 핵심 레벨 (문서 §2·§3 STAGE 4와 동일) ──────────────────────────
LOW_2018 = (T("2018-12-24"), 53.46)       # 4-1 선행 상승 출발점 (저가)
LOW_JUN = (T("2019-06-03"), 87.39)        # 베이스 직전 6개월 저점
CUP_HI = (T("2019-08-30"), 112.00)        # 컵 왼쪽 고점 (베이스 시작)
CUP_LO = (T("2019-10-02"), 92.12)         # 컵 저점 (T1 끝)
PIVOT = (T("2019-11-29"), 122.65)         # 피봇 = T2 시작
T2_LO = (T("2019-12-03"), 112.72)         # T2 끝 (털어내기 저점)
T3_HI = (T("2019-12-09"), 122.00)         # T3 시작 (피봇 재시도 실패)
T3_LO = (T("2019-12-12"), 113.85)         # T3 끝 = 핸들 저점
SHAKE_PREV = (T("2019-12-02"), 114.31)    # 털어내기 직전 저점
STOP = 122.65 * 0.92                      # 112.84
BUY_MAX = 122.65 * 1.05                   # 128.78
HANDLE = (T("2019-12-12"), T("2019-12-31"))
TIGHT7 = (T("2019-12-23"), T("2019-12-31"))
DRY5 = (T("2019-12-24"), T("2019-12-31"))
UPPER_THIRD = CUP_LO[1] + (PIVOT[1] - CUP_LO[1]) * 2 / 3   # 112.47


def candles(ax, d, width=0.6):
    for ts, r in d.iterrows():
        up = r.Close >= r.Open
        c = GREEN if up else RED
        ax.plot([ts, ts], [r.Low, r.High], color=c, lw=0.9, zorder=2)
        ax.add_patch(Rectangle((mdates.date2num(ts) - width / 2, min(r.Open, r.Close)),
                               width, abs(r.Close - r.Open) or 0.05, color=c, zorder=3))


def volume(ax, d, highlight=None):
    ax.bar(d.index, d.Volume / 1e3, color="#95a5a6", width=0.7, label="거래량 (천 주)")
    ax.plot(d.index, d.vol50 / 1e3, color="black", lw=1.1, label="50일 평균 거래량")
    if highlight is not None:
        for ts, col in highlight:
            ax.bar(ts, d.loc[ts, "Volume"] / 1e3, color=col, width=0.7, zorder=4)
    ax.set_ylabel("거래량 (천 주)")
    ax.grid(alpha=0.3)


# ═══════════════════ A. 4-1 선행 상승 + 4-2 베이스 위치 ═══════════════════
d = df.loc["2018-11-15":"2020-01-03"]
fig, ax = plt.subplots(figsize=(16, 7.5))
ax.plot(d.index, d.Close, color=BLUE, lw=1.3, label="PCTY 종가")
ax.plot(d.index, d.sma50, color=ORANGE, lw=1.0, label="50일선")
ax.axvspan(CUP_HI[0], BASE, color=GREY, alpha=0.10)
ax.text(T("2019-10-15"), 66, "베이스(컵-위드-핸들)\n8/30 고점 → 1/2 돌파 = 18주\n(4-2: 3~60주 범위 안)",
        ha="center", fontsize=10, color=GREY, bbox=BOX)

for (ts, px), lab, col in ((LOW_2018, "2018-12-24 저가 $53.46\n(지수 조정 저점)", RED),
                           (LOW_JUN, "2019-06-03 저가 $87.39", RED),
                           (PIVOT, "2019-11-29 고점 $122.65\n= 베이스 고점 = 피봇", GREEN)):
    ax.plot(ts, px, "o", color=col, ms=8, zorder=5)
    ax.annotate(lab, (ts, px), xytext=(0, -34 if col == RED else 14), textcoords="offset points",
                ha="center", fontsize=9.5, color=col, bbox=BOX)

# 화살표: 선행 상승
ax.annotate("", xy=(PIVOT[0], PIVOT[1]), xytext=(LOW_2018[0], LOW_2018[1]),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.2, connectionstyle="arc3,rad=-0.15"))
ax.text(T("2019-04-20"), 108, "4-1 선행 상승 ①\n$53.46 → $122.65 = +129%\n(기준: +30% 이상)",
        color=GREEN, fontsize=11, ha="center", bbox=BOX)
ax.annotate("", xy=(PIVOT[0], PIVOT[1] - 1), xytext=(LOW_JUN[0], LOW_JUN[1]),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6, ls="--", connectionstyle="arc3,rad=-0.25"))
ax.text(T("2019-08-20"), 128, "4-1 선행 상승 ② (베이스 직전 6개월)\n$87.39 → $122.65 = +40%",
        color=GREEN, fontsize=10, ha="center", bbox=BOX)
ax.axvline(BASE, color=BLUE, lw=1.5)
ax.text(BASE, 56, " 기준일\n 2020-01-02", color=BLUE, fontsize=9.5, va="bottom")
ax.set_ylim(50, 135)
ax.set_ylabel("PCTY ($)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.set_title("4-1 베이스 이전 선행 상승 +30% 이상?  — 베이스(회색)는 +129% 상승 뒤에 나온 '쉼'이다   판정: 통과",
             fontsize=13, pad=12)
fig.text(0.5, 0.01,
         "판정 기준: VCP는 강한 상승 뒤에 나와야 의미가 있다(하락 추세 속 반등 박스는 대상 아님). "
         "저점에서 베이스 고점까지 +129%, 베이스 직전 6개월만 봐도 +40% → 기준 +30%를 충분히 넘는다.",
         ha="center", fontsize=9.5, color=GREY)
fig.savefig(f"{OUT}/pcty_4_1_prior_run.png", dpi=110, bbox_inches="tight")
plt.close(fig)


# ═══════════════════ B. VCP 해부 (4-2 ~ 4-7, 4-9, 4-11) ═══════════════════
d = df.loc["2019-08-12":"2020-01-03"]
fig, (ax, av) = plt.subplots(2, 1, figsize=(17, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
candles(ax, d)
ax.plot(d.index, d.sma50, color=ORANGE, lw=1.0, label="50일선")
ax.set_xlim(T("2019-08-09"), T("2020-01-06"))
ax.set_ylim(86, 133)

# 축소 3단 (고점→저점 화살표 + 라벨)
for (h, l), lab, col in (((CUP_HI, CUP_LO), "T1  컵\n$112.00 → $92.12\n-17.7%", RED),
                         ((PIVOT, T2_LO), "T2\n$122.65 → $112.72\n-8.1%", RED),
                         ((T3_HI, T3_LO), "T3\n$122.00 → $113.85\n-6.7%", RED)):
    ax.annotate("", xy=(l[0], l[1]), xytext=(h[0], h[1]),
                arrowprops=dict(arrowstyle="->", color=col, lw=2.0))
    ax.plot(h[0], h[1], "v", color=col, ms=8, zorder=6)
    ax.plot(l[0], l[1], "^", color=GREEN, ms=8, zorder=6)
mid = lambda a, b: a + (b - a) / 2
ax.text(T("2019-09-05"), 96.5, "T1  컵\n$112.00 → $92.12\n-17.7%", ha="center", color=RED, fontsize=10, bbox=BOX)
ax.text(T("2019-12-03"), 106.5, "T2\n$122.65 → $112.72\n-8.1%", ha="center", color=RED, fontsize=10, bbox=BOX)
ax.text(T("2019-12-17"), 103, "T3 (핸들)\n$122.00 → $113.85\n-6.7%", ha="center", color=RED, fontsize=10, bbox=BOX)
ax.text(T("2019-09-30"), 131.2, "4-4·4-5 축소 3회, 매번 절반 안팎으로 줄어듦:  -17.7%  →  -8.1%  →  -6.7%   (왼쪽 느슨 → 오른쪽 조밀)",
        ha="center", fontsize=11, color=RED, bbox=BOX)

# 컵 깊이 (4-3)
ax.annotate("", xy=(T("2019-11-08"), CUP_LO[1]), xytext=(T("2019-11-08"), CUP_HI[1]),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
ax.text(T("2019-10-22"), 90.3, "4-3 베이스 깊이 -17.7% (10~35% 범위 안)\n4-2 베이스 기간 8/30 → 1/2 = 18주", fontsize=9.5, color=GREY, bbox=BOX)

# 피봇 / 손절 / 상단 1/3
ax.axhline(PIVOT[1], color=BLUE, ls="--", lw=1.3)
ax.text(T("2019-08-12"), PIVOT[1] + 0.5, "4-11 피봇 $122.65 (11/29 고점; 12/09 122.00 재시도 실패)", color=BLUE, fontsize=9.5)
ax.axhline(STOP, color=RED, ls=":", lw=1.3)
ax.text(T("2019-09-19"), STOP - 1.9, f"5-1 손절 = 피봇 × 0.92 = ${STOP:.2f}  ≈ 12/03 베이스 저점 $112.72 (거의 일치)", color=RED, fontsize=9.5)
ax.axhline(UPPER_THIRD, color=GREEN, ls="-.", lw=1.0)
ax.axhspan(UPPER_THIRD, PIVOT[1], xmin=0.66, xmax=0.985, color=GREEN, alpha=0.07)
ax.text(T("2019-09-19"), UPPER_THIRD + 0.5, f"4-9 컵 상단 1/3 경계 = ${UPPER_THIRD:.2f}  → 핸들 저점 $113.85가 위(71% 지점)", color=GREEN, fontsize=9.5)

# 핸들 박스 (4-6)
hd = df.loc[HANDLE[0]:HANDLE[1]]
ax.add_patch(Rectangle((mdates.date2num(HANDLE[0]) - 0.5, hd.Low.min()),
                       mdates.date2num(HANDLE[1]) - mdates.date2num(HANDLE[0]) + 1,
                       hd.High.max() - hd.Low.min(), fill=False, ec=BLUE, lw=1.4, ls="-", zorder=5))
td = df.loc[TIGHT7[0]:TIGHT7[1]]
ax.add_patch(Rectangle((mdates.date2num(TIGHT7[0]) - 0.5, td.Low.min()),
                       mdates.date2num(TIGHT7[1]) - mdates.date2num(TIGHT7[0]) + 1,
                       td.High.max() - td.Low.min(), fill=True, fc=BLUE, alpha=0.12, ec=BLUE, lw=1.2, zorder=4))
ax.annotate(f"4-6 핸들 12/12~12/31 고저 {(hd.High.max()/hd.Low.min()-1)*100:.1f}%\n"
            f"마지막 7일(12/23~31) 고저 {(td.High.max()/td.Low.min()-1)*100:.1f}%  ← 책: 3~5%면 최상",
            (T("2019-12-27"), td.High.max()), xytext=(T("2019-10-14"), 126.3), textcoords="data",
            fontsize=10, color=BLUE, bbox=BOX, arrowprops=dict(arrowstyle="->", color=BLUE))

# 돌파일
ax.annotate("1/2 돌파일: 종가 $126.78\n(피봇 +3.4%)", (BASE, 126.78), xytext=(T("2019-12-21"), 109.5), textcoords="data", ha="center",
            fontsize=10, color=GREEN, bbox=BOX,
            arrowprops=dict(arrowstyle="->", color=GREEN))
ax.set_ylabel("PCTY ($)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9.5)
ax.set_title("STAGE 4 VCP 해부 — 18주 컵-위드-핸들, 3T 축소(-17.7 → -8.1 → -6.7%), 핸들 3.9% 압축 뒤 거래량 1.76배 돌파   판정: 4-2~4-7·4-9·4-11 전부 통과",
             fontsize=12.5, pad=12)

# 거래량 패널 (4-7, 4-12)
volume(av, d, highlight=[(ts, BLUE) for ts in df.loc[DRY5[0]:DRY5[1]].index] + [(BASE, GREEN)])
v50 = df.loc["2019-12-31", "vol50"]
av.annotate(f"4-7 드라이업: 돌파 직전 5일(12/24~31) 평균 {df.loc[DRY5[0]:DRY5[1],'Volume'].mean()/1e3:.0f}천\n"
            f"= 50일 평균({v50/1e3:.0f}천)의 {df.loc[DRY5[0]:DRY5[1],'Volume'].mean()/v50:.2f}배 → 팔 사람이 말랐다",
            (T("2019-12-27"), 220), xytext=(-330, 70), textcoords="offset points", fontsize=9.5,
            color=BLUE, bbox=BOX, arrowprops=dict(arrowstyle="->", color=BLUE))
av.annotate(f"4-12 돌파일 거래량 {df.loc[BASE,'Volume']/1e3:.0f}천\n= 50일 평균 × {df.loc[BASE,'Volume']/v50:.2f}  (기준 ×1.4 이상)",
            (BASE, df.loc[BASE, "Volume"] / 1e3), xytext=(-250, 45), textcoords="offset points", fontsize=9.5,
            color=GREEN, bbox=BOX, arrowprops=dict(arrowstyle="->", color=GREEN))
av.set_ylim(0, 1000)
av.text(T("2019-09-25"), 880, "왼쪽(8~10월): 큰 등락 + 큰 거래량 = 손바뀜", fontsize=9.5, color=GREY, bbox=BOX)
av.legend(loc="upper left", fontsize=9)
av.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig.text(0.5, 0.02,
         "읽는 순서: ① 왼쪽 큰 컵(T1) → ② 11월 상승으로 피봇 형성 → ③ 12월 두 번의 얕은 눌림(T2·T3)이 컵 상단 1/3 안에서 점점 좁아짐 → "
         "④ 마지막 7일 4% 박스 + 거래량 70%로 고요 → ⑤ 1/2 거래량 1.76배로 피봇 돌파. 손절선(빨간 점선)이 베이스 저점과 겹쳐 '틀리면 바로 알 수 있는' 자리.",
         ha="center", fontsize=9.5, color=GREY)
fig.savefig(f"{OUT}/pcty_4_vcp_anatomy.png", dpi=110, bbox_inches="tight")
plt.close(fig)


# ═══════════════════ C. 4-8 털어내기 · 4-10 추격 아님 · 4-11/4-12 돌파 ═══════════════════
d = df.loc["2019-11-18":"2020-01-03"]
fig, (ax, av) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
candles(ax, d, width=0.55)
ax.set_xlim(T("2019-11-15"), T("2020-01-06"))
ax.set_ylim(109, 131)

# 피봇 / 매수 허용 구간 / 손절
ax.axhline(PIVOT[1], color=BLUE, ls="--", lw=1.4)
ax.axhspan(PIVOT[1], BUY_MAX, color=GREEN, alpha=0.10)
ax.text(T("2019-11-15"), BUY_MAX + 0.3, f"매수 허용 구간: 피봇 $122.65 ~ +5% = ${BUY_MAX:.2f}  (초록 띠). 1/3 이후 이 위에서 추격 금지", color=GREEN, fontsize=9.5)
ax.text(T("2019-11-15"), PIVOT[1] - 1.0, "4-11 피봇 $122.65", color=BLUE, fontsize=9.5)
ax.axhline(STOP, color=RED, ls=":", lw=1.3)
ax.text(T("2019-11-24"), STOP - 1.0, f"5-1 손절 ${STOP:.2f}", color=RED, fontsize=9.5)

# 4-8 털어내기
ax.plot(SHAKE_PREV[0], SHAKE_PREV[1], "_", color=GREY, ms=18, mew=2.5, zorder=6)
ax.plot(T2_LO[0], T2_LO[1], "v", color=RED, ms=9, zorder=6)
ax.axhline(SHAKE_PREV[1], xmin=0.28, xmax=0.42, color=GREY, ls="--", lw=1.0)
ax.annotate(f"4-8 털어내기(약함): 12/03 저가 $112.72가\n직전 저점 12/02 $114.31을 -1.4% 하회 → 당일 종가 $119.94로 회복\n(undercut & rally. 책의 DECK·VIVO처럼 극적이진 않음)",
            (T2_LO[0], T2_LO[1]), xytext=(20, -55), textcoords="offset points", fontsize=9.5,
            color=RED, bbox=BOX, arrowprops=dict(arrowstyle="->", color=RED))

# 4-10 핸들 박스 (12월 3주 횡보)
hd = df.loc[HANDLE[0]:HANDLE[1]]
ax.add_patch(Rectangle((mdates.date2num(HANDLE[0]) - 0.5, hd.Low.min()),
                       mdates.date2num(HANDLE[1]) - mdates.date2num(HANDLE[0]) + 1,
                       hd.High.max() - hd.Low.min(), fill=True, fc=BLUE, alpha=0.06, ec=BLUE, lw=1.3, zorder=1))
ax.text(T("2019-12-21"), hd.Low.min() - 1.3,
        "4-10 돌파 전 3주(12/12~31) 113~122 박스 횡보 → '피봇 없이 우측만 급등'한 MGA 유형 아님\n"
        "돌파 전 10거래일 종가 변화 +5.8%, 그중 +4.9%가 1/2 하루치", ha="center", fontsize=9.5, color=BLUE, bbox=BOX)

# 돌파일
r = df.loc[BASE]
ax.annotate(f"1/2 돌파: 시가 $121.34 → 종가 $126.78 (+4.9%)\n피봇 대비 +3.4% = 허용 구간 안이지만 상단",
            (BASE, r.Close), xytext=(-250, 30), textcoords="offset points", fontsize=10,
            color=GREEN, bbox=BOX, arrowprops=dict(arrowstyle="->", color=GREEN))
ax.plot(T3_HI[0], T3_HI[1], "v", color=GREY, ms=8, zorder=6)
ax.annotate("12/09 $122.00 피봇 재시도 실패\n(피봇이 저항으로 확인됨)", (T3_HI[0], T3_HI[1]), xytext=(-40, 40),
            textcoords="offset points", fontsize=9, color=GREY, bbox=BOX, arrowprops=dict(arrowstyle="->", color=GREY))
ax.set_ylabel("PCTY ($)")
ax.grid(alpha=0.3)
ax.set_title("4-8 털어내기 · 4-10 우측 급등 추격 아님 · 4-11/4-12 피봇 돌파  — 핸들 확대 (2019-11-18 ~ 2020-01-02)",
             fontsize=12.5, pad=12)

volume(av, d, highlight=[(ts, BLUE) for ts in df.loc[DRY5[0]:DRY5[1]].index] + [(BASE, GREEN), (T2_LO[0], RED), (T3_LO[0], RED)])
av.text(T2_LO[0], df.loc[T2_LO[0], "Volume"] / 1e3 + 15, "털어내기일\n거래량 ↑", ha="center", fontsize=8.5, color=RED)
av.text(T3_LO[0], df.loc[T3_LO[0], "Volume"] / 1e3 + 15, "12/12 급락\n(10일 창 흐림)", ha="center", fontsize=8.5, color=RED)
av.set_ylim(0, 720)
av.text(T("2019-12-27"), 400, "5일 드라이업 0.70배", ha="center", fontsize=8.5, color=BLUE)
av.text(BASE, df.loc[BASE, "Volume"] / 1e3 + 15, "1.76배", ha="center", fontsize=9, color=GREEN, weight="bold")
av.legend(loc="upper left", fontsize=9)
av.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig.text(0.5, 0.02,
         "판정 기준: (4-8) 베이스 안에서 직전 저점을 살짝 깨고 바로 회복하면 약한 손이 털린 것 → 보너스. "
         "(4-10) 피봇 없이 우측만 수직 상승한 베이스는 추격 매수 위험 → PCTY는 3주 횡보 후 돌파라 아님. "
         "(4-12) 돌파일 거래량이 평균의 1.4배 이상이어야 기관 참여 확인 → 1.76배.",
         ha="center", fontsize=9.5, color=GREY)
fig.savefig(f"{OUT}/pcty_4_8_4_10_handle_breakout.png", dpi=110, bbox_inches="tight")
print("done")
