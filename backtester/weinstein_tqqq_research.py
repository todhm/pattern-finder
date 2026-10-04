"""와인스타인 «최적의 타이밍» 체크리스트를 TQQQ 에 적용하는 리서치 스크립트.

문서: docs/strategy_notes/와인스타인_최적의타이밍_체크리스트_2026_10.md
라이브러리: strategy/adapters/weinstein_stage.py

실행 (컨테이너 안):
    docker compose exec backtester python weinstein_tqqq_research.py [옵션]

옵션:
    --signal TQQQ|QQQ   신호(단계·베이스·돌파·RS)를 어느 차트로 계산할지. 기본 TQQQ.
                        QQQ 면 "지수에 30주 MA 를 적용"하는 8장 방식 — 신호는 QQQ, 매매는 TQQQ.
    --mode breakout|stage|pragmatic|avoid
                        avoid     = "피해야 할 구간만 피하고 나머지는 계속 보유". 기본 = TQQQ 보유, --avoid-rule 위험 신호가 켜지면
                                    --danger-asset(현금 또는 QQQ 1배)으로 피신, 꺼지면(QQQ 종가 > 30주 MA) 즉시 복귀. 베이스·돌파·거래량 전부 무시.
                        breakout  = 책의 종목 매수 절차(베이스 돌파 + buy-stop + 50/50 + 7·8장 규칙)
                        stage     = 단계 규칙만(2단계 진입 시 매수, 8-3 이탈 시 매도) — 8장 지수 타이밍 베이스라인
                        pragmatic = "눈으로 봤을 때 이 정도면 괜찮다" 베이스라인. 책의 취지만 남기고 수치는 느슨하게:
                                    매수 = QQQ 가 상승/평평한 30주 MA 위 + SPY 도 MA 위 + QQQ 26주 고점 3% 이내(돌파 중)
                                           + MA 이격 ≤ --ext-cap (추격 금지) + Nasdaq-100 2단계 비율 ≥ --breadth-min (종목들이 같이 가는가)
                                           + 위 조건이 --confirm 주 연속
                                    매도 = QQQ 주봉 종가 < --exit-ma 주 MA 가 --exit-confirm 주 연속 (기울기 조건 없음 — 3x 라 빨리 나간다)
                                           + (옵션) --overheat: QQQ MA 이격이 이 값 이상이면 절반 익절(6장 복합 방식)
    --leverage N        %-임계값(베이스 폭, MA 이격, 근접 저항, 풀백 허용폭)에 곱할 배수. 3x ETF 신호면 3.
    --vol-optional      4-1 거래량을 필수가 아닌 정보 항목으로 (ETF 는 거래량이 공포 때 터지므로 책 규칙과 안 맞음)
    --vol-insufficient  7-5 "빈약" 매각 임계(배). 0 이면 비활성.  --vol-add  2차 매수 자격 배수.

출력: 1) 연도별 수익률 2) 단계 분류 3) 전체 기간 시뮬 + 트레이드 목록 4) 창(좋았을 때/안 좋았을 때/평범)별 성과
      5) 창 안 트레이드의 체크리스트 상세 6) 현재(마지막 주봉) 체크리스트 전 항목.

시뮬레이션 규칙(투자가 모드, 문서 STAGE 7·8):
  - 직전 금요일 기준 사전 필터(시장 0-1, 업종 1-1~1-3, MA 방향 2-2, 후반 2-10, 저항 3-3, RS 기울기 5-1, 베이스 2-5/2-6)
    통과 시에만 저항선 +0.5% 에 buy-stop. 체결은 신호 차트 일봉으로 확인, 매매 차트(TQQQ) 가격으로 체결.
  - 돌파 주 금요일 종가 < 저항 → 실패, 다음 주 월요일 시가 청산(7-5). 거래량 < vol_insufficient 배 → 빈약 → 청산(7-5).
  - 2차 50%: 돌파 후 1~6주 안에 저가가 저항 +2%×lev 이내로 풀백 & 종가 > 저항·MA & 거래량 ≤ 돌파 주 60% (7-3).
  - 7-6: 돌파 3주 뒤 종가 < 저항 +3%×lev 이고 2차 미체결이면 절반 축소 + 스톱 = 저항 ×(1−5%×lev).
  - 청산: 돌파 후 6주 내 종가 < 저항(7-5) / 3단계 진입 시 절반(8-2) / 종가 < MA & 기울기 ≤ 0 전량(8-3) / 4단계(8-4) / 스톱.
  - 체결 슬리피지 0.5%.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.weinstein_stage import (
    Check,
    StageParams,
    add_ma,
    breadth_from_universe,
    checks_to_frame,
    classify_stages,
    evaluate_breakout,
    find_base,
    grade,
    mansfield_rs,
    to_weekly,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 500)

TODAY = date.today()
START = date(2009, 1, 1)
SLIP = 0.005
VOL_INSUFFICIENT = 1.5
VOL_ADD = 2.0
WINDOWS = [
    ("좋았을 때", "2023-01-01", "2024-12-31"),
    ("안 좋았을 때", "2022-01-01", "2022-12-31"),
    ("평범했을 때", "2015-01-01", "2016-12-31"),
]

md = CachedMarketDataAdapter(YFinanceAdapter())


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fetch_daily(sym: str, end: date = TODAY) -> pd.DataFrame:
    df = md.fetch_ohlcv(sym, START, end)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df.dropna(subset=["Close"])


def spy_pd_ratio(spy_daily: pd.DataFrame) -> pd.Series:
    """0-8 P/D = 1 / 배당수익률(후행 12개월)."""
    try:
        import yfinance as yf

        div = yf.Ticker("SPY").dividends
        div.index = pd.to_datetime(div.index).tz_localize(None).normalize()
        d12 = div.resample("D").sum().rolling(365, min_periods=1).sum()
        d12 = d12.reindex(spy_daily.index, method="ffill")
        return (1 / (d12 / spy_daily["Close"])).resample("W-FRI").last()
    except Exception as e:  # noqa: BLE001
        log("P/D 계산 실패:", e)
        return pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
# 시장 / 업종 체크 (STAGE 0 / 1)
# --------------------------------------------------------------------------- #
def market_checks(d: pd.Timestamp, W: dict, breadth: pd.DataFrame, pd_ratio: pd.Series) -> dict:
    spy, qqq, acwi = W["SPY"], W["QQQ"], W["ACWI"]
    s, q = spy.loc[:d].iloc[-1], qqq.loc[:d].iloc[-1]
    checks = []
    ok01 = bool(s.above and s.slope >= 0 and q.above and q.slope >= 0)
    checks.append(Check("0-1", "지수(SPY·QQQ) > 30주 MA & MA 상승",
                        f"SPY {s.Close:.0f}/{s.MA:.0f} slope {s.slope*100:+.2f}% · QQQ {q.Close:.0f}/{q.MA:.0f} slope {q.slope*100:+.2f}%",
                        ok01, True))
    st = int(W["SPY_stage"].loc[:d].iloc[-1])
    checks.append(Check("0-2", "지수 단계", f"SPY {st}단계", st in (1, 2), False))
    b = breadth.loc[:d]
    if len(b):
        br = b.iloc[-1]
        br8 = b.iloc[-9] if len(b) > 9 else br
        checks.append(Check("0-3", "Nasdaq-100 중 1·2단계 비율", f"{br.pct_stage12*100:.0f}% (8주 전 {br8.pct_stage12*100:.0f}%, 2단계만 {br.pct_stage2*100:.0f}%)",
                            bool(br.pct_stage12 >= 0.5), False, "상승 추세면 강세"))
        spy26 = spy.loc[:d]["Close"].iloc[-26:]
        at_high = bool(s.Close >= spy26.max() * 0.99)
        ad26 = b["ad_line"].iloc[-26:]
        ad_at_high = bool(br.ad_line >= ad26.max() - 0.01 * abs(ad26.max()) - 50)
        bear_div = at_high and not ad_at_high
        checks.append(Check("0-4", "A/D선 약세 다이버전스 없음", f"지수 26주 고점권 {at_high} / A/D 26주 고점권 {ad_at_high}",
                            not bear_div, False, "지수 신고점인데 A/D 미동반이면 경고"))
        mi = b["mi"]
        cross = ""
        if len(mi) > 13 and not np.isnan(mi.iloc[-1]):
            sgn = np.sign(mi.iloc[-13:])
            if (sgn.iloc[-1] > 0) and (sgn.min() < 0):
                cross = "최근 13주 내 −→+ 교차(장기 강세)"
            elif (sgn.iloc[-1] < 0) and (sgn.max() > 0):
                cross = "최근 13주 내 +→− 교차(하락 징후)"
        checks.append(Check("0-5", "탄력지수 MI(순상승 200일 평균) > 0", f"{br.mi:+.1f} {cross}", bool(br.mi > 0), False))
        nh8 = b["nh_nl"].iloc[-8:]
        slope = np.polyfit(range(len(nh8)), nh8.values, 1)[0] if len(nh8) >= 4 else np.nan
        checks.append(Check("0-6", "주간 신고−신저 > 0 & 방향", f"{br.nh_nl:+.0f} (8주 기울기 {slope:+.1f}/주)",
                            bool(br.nh_nl > 0), False, "지수 상승 중 지표 하락 = 표면 아래 부패"))
    a = acwi.loc[:d]
    if len(a):
        a = a.iloc[-1]
        checks.append(Check("0-7", "세계 증시(ACWI) > 30주 MA & 상승", f"{a.Close:.1f}/{_f(a.MA)} slope {a.slope*100:+.2f}%",
                            bool(a.above and a.slope >= 0), False))
    pdr = pd_ratio.loc[:d]
    if len(pdr) and not np.isnan(pdr.iloc[-1]):
        checks.append(Check("0-8", "P/D 비율 (책: 14~17 저평가, >26 위험, ≥30 과대)", f"{pdr.iloc[-1]:.1f}", None, False,
                            "현대 배당 환경에선 항상 30 초과 — 참고용"))
    m = d.month
    checks.append(Check("0-9", "계절·선거주기", f"{m}월{' (12월 최고)' if m == 12 else ' (9월 최악)' if m == 9 else ''}, 선거 다음 해 {(d.year % 4) == 1}", None))
    return {"checks": checks, "pass": ok01}


def sector_checks(d: pd.Timestamp, W: dict, breadth: pd.DataFrame) -> dict:
    q = W["QQQ"].loc[:d].iloc[-1]
    st = int(W["QQQ_stage"].loc[:d].iloc[-1])
    rs = W["QQQ_rs"].loc[:d]
    rs_now, rs_prev = rs.iloc[-1], rs.iloc[-14] if len(rs) > 14 else np.nan
    checks = [
        Check("1-1", "업종(Nasdaq-100=QQQ) 단계 ∈ {1,2}", f"{st}단계", st in (1, 2), True),
        Check("1-2", "업종 지수 > 30주 MA & 상승", f"{q.Close:.0f}/{q.MA:.0f} slope {q.slope*100:+.2f}%", bool(q.above and q.slope >= 0), True),
        Check("1-3", "업종 RS(QQQ/SPY 맨스필드) > 0 (플러스 영역)", f"{_f(rs_prev)} → {_f(rs_now)}",
              bool(rs_now > 0), True, "상승 중" if (not np.isnan(rs_prev) and rs_now >= rs_prev) else "하락 중(경고)"),
    ]
    b = breadth.loc[:d]
    if len(b):
        checks.append(Check("1-4", "업종 내 2단계 종목 비율 ≥ 50%", f"{b.iloc[-1].pct_stage2*100:.0f}%", bool(b.iloc[-1].pct_stage2 >= 0.5), False))
    return {"checks": checks, "pass": all(c.passed for c in checks if c.required and c.passed is not None)}


# --------------------------------------------------------------------------- #
# 시뮬레이션
# --------------------------------------------------------------------------- #
class Sim:
    """신호 차트(sig) 와 매매 차트(trd) 를 분리. 같은 심볼이면 책의 종목 매매 그대로."""

    def __init__(self, sig_w, sig_d, trd_w, trd_d, stages, rs, W, breadth, pd_ratio, p: StageParams, mode: str, alt: dict | None = None):
        self.sw, self.sd, self.tw, self.td = sig_w, sig_d, trd_w, trd_d
        self.inst = {"TQQQ": (trd_w, trd_d), **(alt or {})}   # 이름 → (weekly, daily)
        self.short_stage4 = False
        self.stages, self.rs, self.W, self.breadth, self.pdr, self.p, self.mode = stages, rs, W, breadth, pd_ratio, p, mode
        self.cash, self.shares, self.pos = 1.0, 0.0, None
        self.equity, self.trades, self.blocked = [], [], []

    # --- 가격 유틸 ---
    def _td(self, inst=None):
        name = inst or (self.pos["inst"] if self.pos else "TQQQ")
        return self.inst[name]

    def trd_px_after(self, d, inst=None):
        tw, td = self._td(inst)
        nd = td.loc[td.index > d]
        return (nd.index[0], float(nd["Open"].iloc[0])) if len(nd) else (d, float(tw["Close"].loc[:d].iloc[-1]))

    def trd_close_on(self, d, inst=None):
        _, td = self._td(inst)
        return float(td["Close"].loc[:d].iloc[-1])

    def mark(self, d):
        return self.cash + self.shares * self.trd_close_on(d)

    # --- 포지션 ---
    def open_pos(self, d_fill, px, frac, info):
        buy = self.cash * frac if self.pos is None else self.cash
        self.shares += buy / px
        self.cash -= buy
        if self.pos is None:
            self.pos = {"inst": "TQQQ", "entry_date": d_fill.date(), "entry_px": round(px, 2), "start_equity": self.cash + buy, "exits": [], "added": False}
            self.pos.update(info)

    def close_pos(self, i, reason, frac=1.0):
        d = self.sw.index[i]
        dt, px = self.trd_px_after(d)
        px *= 1 - SLIP
        sell = self.shares * frac
        self.cash += sell * px
        self.shares -= sell
        self.pos["exits"].append((dt.date(), round(px, 2), reason, round(frac, 2)))
        if frac >= 1.0 or self.shares < 1e-9:
            self.pos.update(exit_date=dt.date(), exit_px=round(px, 2),
                            ret=round(self.cash / self.pos["start_equity"] - 1, 4),
                            bh_ret=round(px / self.pos["entry_px"] - 1, 4),
                            weeks=int((pd.Timestamp(dt) - pd.Timestamp(self.pos["entry_date"])).days // 7))
            self.trades.append(self.pos)
            self.pos, self.shares = None, 0.0

    # --- 메인 루프 ---
    def run(self, start: pd.Timestamp):
        w = self.sw
        start_i = w.index.get_indexer([start], method="bfill")[0]
        trd_start = self.td.index[0] + pd.Timedelta(days=7)
        for i in range(len(w)):
            d = w.index[i]
            row = w.iloc[i]
            if d < trd_start:
                self.equity.append((d, self.cash))
                continue
            if i < max(start_i, 40):
                self.equity.append((d, self.mark(d)))
                continue
            if self.pos is not None:
                self.manage(i, row)
            elif self.mode == "stage":
                self.try_stage_entry(i, row)
            elif self.mode == "pragmatic":
                self.try_prag_entry(i, row)
            elif self.mode == "avoid":
                self.try_avoid_entry(i, row)
            else:
                self.try_breakout_entry(i, row)
            self.equity.append((d, self.mark(d)))
        if self.pos is not None:
            last = self.inst[self.pos["inst"]][0]["Close"].iloc[-1]
            self.pos.update(open=True, mark=round(float(last), 2),
                            ret=round((self.cash + self.shares * last) / self.pos["start_equity"] - 1, 4),
                            bh_ret=round(float(last) / self.pos["entry_px"] - 1, 4))
            self.trades.append(self.pos)
        eq = pd.Series({d: v for d, v in self.equity}).sort_index()
        return eq, self.trades, self.blocked

    def manage(self, i, row):
        pos, p = self.pos, self.p
        st = int(self.stages.iloc[i])
        if self.mode == "stage":
            if (not row.above) and row.slope <= 0:
                self.close_pos(i, "8-3 종가 < MA & MA 평평/하락")
            elif st == 4:
                self.close_pos(i, "8-4 4단계")
            return
        if self.mode == "avoid":
            d_i = self.sw.index[i]
            danger = self.is_danger(i, row)
            if pos["inst"] == "TQQQ" and danger:
                self.close_pos(i, f"위험 신호 ON ({self.avoid_rule}) → {self.danger_asset}")
                if self.danger_asset == "QQQ":
                    dt, px = self.trd_px_after(d_i, "QQQ")
                    self.open_pos(dt, px * (1 + SLIP), 1.0, {"inst": "QQQ", "i": i, "res": float(row.Close), "grade": "피신", "vol_ratio": np.nan,
                                                           "bo_volume": 0.0, "checks": None, "base_weeks": 0, "rs": round(float(self.rs.iloc[i]), 2),
                                                           "entry_type": "위험 구간 QQQ 1배"})
            elif pos["inst"] == "QQQ" and not danger and row.above:
                self.close_pos(i, "위험 신호 OFF → TQQQ 복귀")
                dt, px = self.trd_px_after(d_i, "TQQQ")
                self.open_pos(dt, px * (1 + SLIP), 1.0, {"inst": "TQQQ", "i": i, "res": float(row.Close), "grade": "복귀", "vol_ratio": np.nan,
                                                       "bo_volume": 0.0, "checks": None, "base_weeks": 0, "rs": round(float(self.rs.iloc[i]), 2),
                                                       "entry_type": "복귀 보유"})
            return
        if self.mode == "pragmatic" and pos.get("inst") == "SQQQ":
            pr = self.prag
            d_i = self.sw.index[i]
            tc = self.trd_close_on(d_i)
            pos["peak"] = max(pos.get("peak", pos["entry_px"]), tc)
            if pr["trail"] and tc <= pos["peak"] * (1 - pr["trail"]):
                self.close_pos(i, f"숏 트레일링 스톱 (SQQQ 고점 {pos['peak']:.2f} 대비 -{pr['trail']:.0%})")
            elif row.above or st != 4:
                self.close_pos(i, f"QQQ 종가 > 30주 MA 또는 4단계 종료(현재 {st}단계) → 숏 청산")
            return
        if self.mode == "pragmatic":
            pr = self.prag
            d_i = self.sw.index[i]
            tc = self.trd_close_on(d_i)
            pos["peak"] = max(pos.get("peak", pos["entry_px"]), tc)
            below = (self.sw["Close"].iloc[i - pr["exit_confirm"] + 1:i + 1] < self.sw["MA_exit"].iloc[i - pr["exit_confirm"] + 1:i + 1]).all()
            if pr["trail"] and tc <= pos["peak"] * (1 - pr["trail"]):
                self.close_pos(i, f"트레일링 스톱 (TQQQ 고점 {pos['peak']:.2f} 대비 -{pr['trail']:.0%})")
            elif below:
                self.close_pos(i, f"종가 < {pr['exit_ma']}주 MA ({pr['exit_confirm']}주 연속)")
            elif pr["overheat"] and row.ext >= pr["overheat"] and not pos.get("half_sold"):
                pos["half_sold"] = self.sw.index[i].date()
                self.close_pos(i, f"과열 익절 (MA 이격 {row.ext:.0%} ≥ {pr['overheat']:.0%}) 절반", 0.5)
            return
        k = i - pos["i"]
        res = pos["res"]
        d = self.sw.index[i]
        if (not pos["added"]) and 1 <= k <= 6 and pos["vol_ratio"] >= VOL_ADD:
            if row.Low <= res * (1 + p.scaled(0.02)) and row.Close > res and row.Close > row.MA and row.Volume <= 0.6 * pos["bo_volume"]:
                px = self.trd_close_on(d) * (1 + SLIP)
                self.open_pos(d, px, 1.0, {})
                pos["added"] = True
                pos["add"] = (d.date(), round(px, 2), round(row.Volume / pos["bo_volume"], 2))
        if k == 3 and not pos["added"] and row.Close < res * (1 + p.scaled(0.03)) and pos.get("reduced") is None:
            pos["reduced"] = d.date()
            pos["stop"] = res * (1 - p.scaled(0.05))
            self.close_pos(i, "7-6 돌파 후 3주 부진 → 절반 축소", 0.5)
            if self.pos is None:
                return
        if k <= 6 and row.Close < res:
            self.close_pos(i, "7-5 종가 < 돌파가 (실패 돌파)")
        elif pos.get("stop") and row.Close < pos["stop"]:
            self.close_pos(i, "스톱(돌파가 −5%×lev)")
        elif (not row.above) and row.slope <= 0:
            self.close_pos(i, "8-3 종가 < MA & MA 평평/하락")
        elif st == 3 and not pos.get("half_sold"):
            pos["half_sold"] = d.date()
            self.close_pos(i, "8-2 3단계 진입 → 투자가 절반 매도", 0.5)
        elif st == 4:
            self.close_pos(i, "8-4 4단계")

    def pre_filters(self, i, base, row):
        w = self.sw
        d_prev = w.index[i - 1]
        mk = market_checks(d_prev, self.W, self.breadth, self.pdr)
        sc = sector_checks(d_prev, self.W, self.breadth)
        checks = evaluate_breakout(w, self.stages, self.rs, i, base, self.p, mk, sc)
        cmap = {c.code: c for c in checks}
        pre_codes = ["0-1", "1-1", "1-2", "1-3", "2-2", "2-10", "3-3", "5-1", "2-6", "2-5"]
        pre_fail = [c for c in pre_codes if c in cmap and cmap[c].passed is False]
        return checks, cmap, pre_fail, mk, sc

    def try_stage_entry(self, i, row):
        """mode=stage: 단계가 2 로 올라오는 주(1→2 또는 4/3→2)에 매수. 시장 필터 0-1 만 사전 확인."""
        st, prev = int(self.stages.iloc[i]), int(self.stages.iloc[i - 1])
        if not (st == 2 and prev != 2):
            return
        d = self.sw.index[i]
        mk = market_checks(self.sw.index[i - 1], self.W, self.breadth, self.pdr)
        if not mk["pass"] and not (row.above and row.slope >= 0):
            self.blocked.append({"date": d.date(), "res": np.nan, "base_weeks": 0, "grade": "stage", "vol_ratio": np.nan,
                                 "rs": round(float(self.rs.iloc[i]), 2), "blocked_by": ["0-1"]})
            return
        dt, px = self.trd_px_after(d)
        px *= 1 + SLIP
        self.open_pos(dt, px, 1.0, {"i": i, "res": float(row.Close), "grade": f"{prev}→2", "vol_ratio": np.nan,
                                    "bo_volume": float(row.Volume), "checks": mk["checks"], "base_weeks": 0,
                                    "rs": round(float(self.rs.iloc[i]), 2), "entry_type": f"{prev}→2 단계 전환"})

    def is_danger(self, i, row):
        w, st = self.sw, int(self.stages.iloc[i])
        rule = self.avoid_rule
        if rule == "stage4":
            return st == 4
        if rule == "ma_falling":
            return (not row.above) and row.slope <= 0
        if rule == "ma_2wk":
            prev = w.iloc[i - 1]
            return (not row.above) and (not prev.above) and row.slope <= 0
        if rule == "ma_any":
            return not row.above
        raise ValueError(rule)

    def try_avoid_entry(self, i, row):
        """mode=avoid: 위험 신호가 꺼져 있고 QQQ 가 30주 MA 위면 조건 없이 TQQQ 보유."""
        d = self.sw.index[i]
        if np.isnan(row.MA):
            return
        if self.is_danger(i, row) or not row.above:
            if self.danger_asset == "QQQ" and self.pos is None:
                dt, px = self.trd_px_after(d, "QQQ")
                self.open_pos(dt, px * (1 + SLIP), 1.0, {"inst": "QQQ", "i": i, "res": float(row.Close), "grade": "피신", "vol_ratio": np.nan,
                                                       "bo_volume": 0.0, "checks": None, "base_weeks": 0, "rs": round(float(self.rs.iloc[i]), 2),
                                                       "entry_type": "위험 구간 QQQ 1배"})
            return
        dt, px = self.trd_px_after(d, "TQQQ")
        self.open_pos(dt, px * (1 + SLIP), 1.0, {"inst": "TQQQ", "i": i, "res": float(row.Close), "grade": "보유", "vol_ratio": np.nan,
                                               "bo_volume": 0.0, "checks": None, "base_weeks": 0, "rs": round(float(self.rs.iloc[i]), 2),
                                               "entry_type": "기본 보유"})

    def prag_ok(self, j):
        """pragmatic 매수 조건을 주 j 에서 평가. (통과여부, 실패사유 리스트)"""
        pr, w = self.prag, self.sw
        r = w.iloc[j]
        d = w.index[j]
        fails = []
        if not (r.above and r.slope >= 0):
            fails.append("QQQ<MA30 or MA 하락")
        sp = self.W["SPY"].loc[:d].iloc[-1]
        if not sp.above:
            fails.append("SPY<MA30")
        if not (r.Close >= r.hi26 * (1 - pr["near_high"])):
            fails.append(f"26주 고점 -{pr['near_high']:.0%} 밖")
        if r.ext > pr["ext_cap"]:
            fails.append(f"이격 {r.ext:.0%}>{pr['ext_cap']:.0%}")
        b = self.breadth.loc[:d]
        if len(b) and not np.isnan(b.iloc[-1].pct_stage2) and b.iloc[-1].pct_stage2 < pr["breadth_min"]:
            fails.append(f"2단계 비율 {b.iloc[-1].pct_stage2:.0%}<{pr['breadth_min']:.0%}")
        return (not fails), fails

    def try_prag_entry(self, i, row):
        pr = self.prag
        results = [self.prag_ok(j) for j in range(i - pr["confirm"] + 1, i + 1)]
        ok = all(r[0] for r in results)
        if not ok and self.short_stage4 and "SQQQ" in self.inst:
            # 7장: 4단계(주가 < 하락하는 30주 MA) + 시장도 약세 → 숏(SQQQ). 공매도 체크리스트: 2단계·강한 업종 금지 → QQQ 자체가 4단계일 때만
            d = self.sw.index[i]
            sp = self.W["SPY"].loc[:d].iloc[-1]
            st = int(self.stages.iloc[i])
            if st == 4 and (not row.above) and row.slope < 0 and (not sp.above) and self.sd.index[0] < d:
                dt, px = self.trd_px_after(d, "SQQQ")
                px *= 1 + SLIP
                b = self.breadth.loc[:d]
                br = f"{b.iloc[-1].pct_stage2:.0%}" if len(b) else "n/a"
                self.open_pos(dt, px, 1.0, {"inst": "SQQQ", "i": i, "res": float(row.Close), "grade": f"4단계 숏/2단계 {br}", "vol_ratio": np.nan,
                                            "bo_volume": float(row.Volume), "checks": None, "base_weeks": 0,
                                            "rs": round(float(self.rs.iloc[i]), 2), "entry_type": "4단계 → SQQQ"})
            return
        if not ok:
            # 직전 주는 통과였는데 이번 주 실패 → 기록(차단 사유 보고용). 연속 기록 폭주 방지: 고점 근처일 때만
            if row.Close >= row.hi26 * (1 - pr["near_high"]) and results[-1][1] and (i == 0 or not self.prag_ok(i - 1)[0] or pr["confirm"] > 1):
                self.blocked.append({"date": self.sw.index[i].date(), "res": round(float(row.hi26), 2), "base_weeks": 0, "grade": "prag",
                                     "vol_ratio": np.nan, "rs": round(float(self.rs.iloc[i]), 2), "blocked_by": results[-1][1]})
            return
        d = self.sw.index[i]
        dt, px = self.trd_px_after(d)
        px *= 1 + SLIP
        b = self.breadth.loc[:d]
        br = f"{b.iloc[-1].pct_stage2:.0%}" if len(b) else "n/a"
        self.open_pos(dt, px, 1.0, {"i": i, "res": float(row.hi26), "grade": f"ext {row.ext:.0%}/2단계 {br}", "vol_ratio": np.nan,
                                    "bo_volume": float(row.Volume), "checks": None, "base_weeks": 0,
                                    "rs": round(float(self.rs.iloc[i]), 2),
                                    "entry_type": f"{int(self.stages.iloc[i])}단계, 26주 고점 {row.Close/row.hi26-1:+.1%}"})

    def try_breakout_entry(self, i, row):
        w, p = self.sw, self.p
        base = find_base(w, i, p)
        prev_close = w["Close"].iloc[i - 1]
        if base is None or not (row.Close > base.high and prev_close <= base.high):
            return
        d = w.index[i]
        checks, cmap, pre_fail, mk, sc = self.pre_filters(i, base, row)
        g = grade(checks)
        vol_ratio = float(cmap["4-1"].value.rstrip("x"))
        rec = {"date": d.date(), "res": round(base.high, 2), "base_weeks": base.weeks, "base_width": round(base.width, 3),
               "grade": g, "vol_ratio": round(vol_ratio, 2), "rs": round(float(self.rs.iloc[i]), 2),
               "stage_prev": int(self.stages.iloc[i - 1]), "mkt": mk["pass"], "sector": sc["pass"]}
        if pre_fail:
            rec["blocked_by"] = pre_fail
            self.blocked.append(rec)
            return
        wk = self.sd.loc[(self.sd.index > w.index[i - 1]) & (self.sd.index <= d)]
        trig = base.high * 1.005
        hit = wk[wk["High"] >= trig]
        if not len(hit):
            return
        d_fill = hit.index[0]
        if self.sd is self.td:
            px = max(trig, float(hit["Open"].iloc[0])) * (1 + SLIP)
        else:  # 신호 차트 ≠ 매매 차트: 그날 매매 차트 종가로 근사
            px = self.trd_close_on(d_fill) * (1 + SLIP)
        self.open_pos(d_fill, px, 0.5, {"i": i, "res": base.high, "grade": g, "vol_ratio": vol_ratio, "bo_volume": float(row.Volume),
                                         "checks": checks, "base_weeks": base.weeks, "rs": round(float(self.rs.iloc[i]), 2),
                                         "entry_type": "1→2" if (self.stages.iloc[max(0, i - 8):i] == 1).any() else "2단계 지속"})
        if row.Close < base.high:
            self.close_pos(i, "7-5 돌파 주 종가 < 저항 (가짜 돌파)")
        elif VOL_INSUFFICIENT and vol_ratio < VOL_INSUFFICIENT:
            self.close_pos(i, f"7-5 거래량 빈약 {vol_ratio:.2f}x → 매각")


def max_dd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def max_dd_info(s: pd.Series) -> str:
    dd = s / s.cummax() - 1
    trough = dd.idxmin()
    peak = s.loc[:trough].idxmax()
    return f"{dd.min():+.1%} (고점 {peak.date()} → 저점 {trough.date()})"


def cagr(s: pd.Series) -> float:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else np.nan


def fmt_trade(t):
    return (f"{t['entry_date']} {t.get('inst','TQQQ')} @{t['entry_px']} [{t['entry_type']}, {t['grade']}, 거래량 {t['vol_ratio']}x, RS {t['rs']}, 베이스 {t['base_weeks']}주, res {t['res']:.2f}]"
            f" 2차 {t.get('add')} 축소 {t.get('reduced')} → {t.get('exit_date', '보유 중')} @{t.get('exit_px', t.get('mark'))} "
            f"{t['ret']:+.1%} (같은 기간 보유 {t['bh_ret']:+.1%}) {[x[2] for x in t['exits']]}")


def window_report(name, a, b, eq, tqqq_w, trades, blocked):
    e = eq.loc[a:b]
    bh = tqqq_w["Close"].loc[a:b]
    print(f"\n### {name}: {a} ~ {b}")
    print(f"- TQQQ 보유: {bh.iloc[-1]/bh.iloc[0]-1:+.1%}  (MDD {max_dd(bh):+.1%})")
    print(f"- 전략: {e.iloc[-1]/e.iloc[0]-1:+.1%}  (MDD {max_dd(e):+.1%})")
    inmkt = (e.diff().abs() > 1e-12).mean()
    print(f"- 시장 참여 비율(주 단위, 근사): {inmkt:.0%}")
    tw = [t for t in trades if pd.Timestamp(t['entry_date']) <= pd.Timestamp(b) and pd.Timestamp(t.get('exit_date', TODAY)) >= pd.Timestamp(a)]
    print(f"- 창 안 트레이드 {len(tw)}건:")
    for t in tw:
        print("    · " + fmt_trade(t))
    bw = [x for x in blocked if pd.Timestamp(a) <= pd.Timestamp(x['date']) <= pd.Timestamp(b)]
    print(f"- 창 안에서 돌파/전환했지만 필터로 매수 안 한 경우 {len(bw)}건:")
    for x in bw:
        print(f"    · {x['date']} res {x['res']} 베이스 {x['base_weeks']}주 등급 {x['grade']} 거래량 {x['vol_ratio']}x RS {x['rs']} 차단: {x['blocked_by']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="TQQQ", choices=["TQQQ", "QQQ"])
    ap.add_argument("--mode", default="breakout", choices=["breakout", "stage", "pragmatic", "avoid"])
    ap.add_argument("--avoid-rule", default="ma_falling", choices=["stage4", "ma_falling", "ma_2wk", "ma_any"],
                    help="avoid: 위험 신호. stage4=QQQ 4단계 / ma_falling=종가<30주MA & MA 기울기≤0 (책 8-3) / ma_2wk=2주 연속 종가<MA & 기울기≤0 / ma_any=종가<MA 1주")
    ap.add_argument("--danger-asset", default="cash", choices=["cash", "QQQ"], help="avoid: 위험 구간에 들고 있을 것")
    ap.add_argument("--ext-cap", type=float, default=0.12, help="pragmatic: 매수 시 QQQ MA 이격 상한")
    ap.add_argument("--breadth-min", type=float, default=0.45, help="pragmatic: Nasdaq-100 2단계 비율 하한 (breadth 없으면 무시)")
    ap.add_argument("--near-high", type=float, default=0.03, help="pragmatic: 26주 고점 대비 허용 거리")
    ap.add_argument("--confirm", type=int, default=1, help="pragmatic: 매수 조건 연속 충족 주 수")
    ap.add_argument("--exit-ma", type=int, default=30, help="pragmatic: 매도 기준 MA 주 수 (30 또는 10)")
    ap.add_argument("--exit-confirm", type=int, default=1, help="pragmatic: 매도 조건 연속 주 수")
    ap.add_argument("--overheat", type=float, default=0.0, help="pragmatic: QQQ MA 이격이 이 값 이상이면 절반 익절 (0=비활성)")
    ap.add_argument("--synthetic-3x", action="store_true", help="TQQQ 를 QQQ 일간수익률×3(연 0.95%% 비용) 합성으로 대체해 1999년부터 검증 (닷컴 붕괴 포함)")
    ap.add_argument("--short-stage4", action="store_true", help="pragmatic: QQQ 4단계 & SPY<MA 이면 SQQQ 보유 (7장 공매도)")
    ap.add_argument("--trail", type=float, default=0.0, help="pragmatic: TQQQ 포지션 고점(주봉 종가) 대비 이만큼 빠지면 추세 무관 전량 청산 (0=비활성)")
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--base-width", type=float, default=0.30)
    ap.add_argument("--sim-start", default="2011-01-01")
    ap.add_argument("--skip-breadth", action="store_true")
    ap.add_argument("--vol-insufficient", type=float, default=1.5)
    ap.add_argument("--vol-add", type=float, default=2.0)
    ap.add_argument("--vol-optional", action="store_true")
    args = ap.parse_args()
    global VOL_INSUFFICIENT, VOL_ADD
    VOL_INSUFFICIENT, VOL_ADD = args.vol_insufficient, args.vol_add
    p = StageParams(leverage=args.leverage, base_width=args.base_width, vol_required=not args.vol_optional)

    global START
    if args.synthetic_3x:
        START = date(1998, 1, 1)
    log("데이터 수집: TQQQ QQQ SPY ACWI")
    D = {s: fetch_daily(s) for s in ["TQQQ", "QQQ", "SPY", "ACWI"] + (["SQQQ"] if args.short_stage4 else [])}
    if args.synthetic_3x:
        q = D["QQQ"]
        r = q["Close"].pct_change().fillna(0.0) * 3 - 0.0095 / 252
        px = 100 * (1 + r).cumprod()
        ratio = px / q["Close"]
        syn = pd.DataFrame({"Open": q["Open"] * ratio.shift(1).fillna(ratio.iloc[0]), "High": q["High"] * ratio, "Low": q["Low"] * ratio,
                            "Close": px, "Volume": q["Volume"]}, index=q.index)
        D["TQQQ"] = syn
        log(f"합성 3x: {syn.index[0].date()} ~ {syn.index[-1].date()}, 실제 TQQQ 와 2011~ 구간 상관 확인용 종가비 {float(px.loc['2011-01-03':].iloc[0]):.1f}")
    W = {s: add_ma(to_weekly(D[s])) for s in D}
    W["SPY_stage"] = classify_stages(W["SPY"])
    W["QQQ_stage"] = classify_stages(W["QQQ"])
    W["QQQ_rs"] = mansfield_rs(W["QQQ"]["Close"], W["SPY"]["Close"])
    pd_ratio = spy_pd_ratio(D["SPY"])

    if args.skip_breadth:
        breadth = pd.DataFrame(columns=["pct_stage12", "pct_stage2", "ad_line", "mi", "nh", "nl", "nh_nl", "n_valid"])
    else:
        log("Nasdaq-100 유니버스 수집 (현재 구성종목 — 생존 편향 있음)")
        tickers = WikipediaUniverseAdapter().get_tickers("nasdaq100")
        dailies = {}
        for n, t in enumerate(tickers, 1):
            try:
                dailies[t] = fetch_daily(t, TODAY - timedelta(days=1))
            except Exception as e:  # noqa: BLE001
                log(f"  {t} 실패: {e}")
            if n % 25 == 0:
                log(f"  {n}/{len(tickers)}")
        breadth = breadth_from_universe(dailies)
        log(f"breadth 완료: {len(dailies)}종목")

    sig = args.signal
    sw, sd = W[sig], D[sig]
    stages = classify_stages(sw, p)
    rs = mansfield_rs(sw["Close"], W["SPY"]["Close"])
    tq = W["TQQQ"]

    print("# 와인스타인 체크리스트 × TQQQ 리서치 결과")
    print(f"기준일(마지막 주봉): {tq.index[-1].date()} · 신호 차트 {sig} · 모드 {args.mode} · leverage={p.leverage} "
          f"(베이스 폭 ≤{p.scaled(p.base_width):.0%}, MA 이격 ≤{p.scaled(p.late_stage_ma_ext):.0%}, 근접 저항 +{p.scaled(p.near_resistance_pct):.0%}) · "
          f"거래량 필수={p.vol_required}, 빈약매각<{VOL_INSUFFICIENT}x, 2차자격≥{VOL_ADD}x")
    if args.mode == "pragmatic":
        print(f"pragmatic 파라미터: ext_cap={args.ext_cap}, breadth_min={args.breadth_min}, near_high={args.near_high}, confirm={args.confirm}, exit_ma={args.exit_ma}, exit_confirm={args.exit_confirm}, overheat={args.overheat}, trail={args.trail}, short_stage4={args.short_stage4}")

    yr = D["TQQQ"]["Close"].resample("YE").last()
    qyr = D["QQQ"]["Close"].resample("YE").last()
    print("\n## 1. TQQQ / QQQ 연도별 수익률")
    tbl = pd.DataFrame({"TQQQ": (yr / yr.shift(1) - 1).dropna(), "QQQ": (qyr / qyr.shift(1) - 1).dropna()})
    tbl.index = tbl.index.year
    print(tbl.map(lambda v: f"{v:+.0%}").T.to_string())

    print(f"\n## 2. {sig} 단계 분류 (주 수) 및 전환 이력")
    print(stages[stages > 0].value_counts().sort_index().to_string())
    chg = stages[stages.diff() != 0]
    print(chg.tail(32).to_string())

    alt = {"QQQ": (W["QQQ"], D["QQQ"])}
    if args.short_stage4:
        alt["SQQQ"] = (W["SQQQ"], D["SQQQ"])
    sim = Sim(sw, sd, tq, D["TQQQ"], stages, rs, W, breadth, pd_ratio, p, args.mode, alt)
    sim.short_stage4 = args.short_stage4
    sim.avoid_rule, sim.danger_asset = args.avoid_rule, args.danger_asset
    if args.mode == "avoid":
        print(f"avoid 파라미터: rule={args.avoid_rule}, danger_asset={args.danger_asset}")
    sim.prag = dict(ext_cap=args.ext_cap, breadth_min=args.breadth_min, near_high=args.near_high, confirm=args.confirm,
                    exit_ma=args.exit_ma, exit_confirm=args.exit_confirm, overheat=args.overheat, trail=args.trail)
    sw["MA_exit"] = sw["Close"].rolling(args.exit_ma).mean()
    sw["hi26"] = sw["Close"].rolling(26).max()
    eq, trades, blocked = sim.run(pd.Timestamp(args.sim_start))
    print(f"\n## 3. 전체 기간 시뮬레이션 {eq.index[0].date()} ~ {eq.index[-1].date()}  (매매 대상 TQQQ)")
    bh = tq["Close"].loc[eq.index[0]:]
    qbh = W["QQQ"]["Close"].loc[eq.index[0]:]
    print(f"- 전략: 누적 {eq.iloc[-1]/eq.iloc[0]-1:+.1%}, CAGR {cagr(eq):+.1%}, MDD {max_dd_info(eq)}")
    print(f"- TQQQ 보유: 누적 {bh.iloc[-1]/bh.iloc[0]-1:+.1%}, CAGR {cagr(bh):+.1%}, MDD {max_dd_info(bh)}")
    dd = (eq / eq.cummax() - 1)
    worst = dd.groupby(dd.index.year).min()
    print("- 연도별 전략 최대 낙폭:", {int(y): f"{v:+.0%}" for y, v in worst.items() if v < -0.2})
    print(f"- QQQ 보유: 누적 {qbh.iloc[-1]/qbh.iloc[0]-1:+.1%}, CAGR {cagr(qbh):+.1%}, MDD {max_dd(qbh):+.1%}")
    closed = [t for t in trades if not t.get("open")]
    if closed:
        rets = pd.Series([t["ret"] for t in closed])
        print(f"- 트레이드 {len(closed)}건 (보유 중 {len(trades)-len(closed)}), 승률 {(rets>0).mean():.0%}, 평균 {rets.mean():+.1%}, 중앙값 {rets.median():+.1%}, 최대 {rets.max():+.1%}, 최소 {rets.min():+.1%}")
        print("- 등급별:\n" + pd.DataFrame(closed).groupby("grade")["ret"].agg(["count", "mean", "median"]).round(3).to_string())
        print("- 청산 사유:", dict(Counter(x[2].split(" ")[0] for t in closed for x in t["exits"])))
    print("\n### 트레이드 전체 목록")
    for t in trades:
        print("- " + fmt_trade(t))
    print(f"\n### 필터 차단 {len(blocked)}건 (사유별): {dict(Counter(c for b in blocked for c in b['blocked_by']))}")

    print("\n## 4. 좋았을 때 / 안 좋았을 때 / 평범했을 때")
    for name, a, b in WINDOWS:
        window_report(name, a, b, eq, tq, trades, blocked)

    print("\n## 5. 창 안 트레이드의 체크리스트 상세")
    for t in trades:
        if any(pd.Timestamp(a) <= pd.Timestamp(t["entry_date"]) <= pd.Timestamp(b) for _, a, b in WINDOWS) and t.get("checks"):
            print(f"\n### {t['entry_date']} 진입 (등급 {t['grade']})")
            print(checks_to_frame(t["checks"]).to_string(index=False))

    print("\n## 6. 현재 상태 — 마지막 주봉 기준 전 항목")
    d = sw.index[-1]
    i = len(sw) - 1
    mk = market_checks(d, W, breadth, pd_ratio)
    sc = sector_checks(d, W, breadth)
    print(checks_to_frame(mk["checks"] + sc["checks"]).to_string(index=False))
    for sym in ("TQQQ", "QQQ"):
        ww = W[sym]
        stg = classify_stages(ww, p if sym == sig else StageParams(leverage=3.0 if sym == "TQQQ" else 1.0))
        r = ww.iloc[-1]
        rr = mansfield_rs(ww["Close"], W["SPY"]["Close"])
        print(f"\n{sym}: 종가 {r.Close:.2f}, 30주 MA {r.MA:.2f}, 4주 기울기 {r.slope*100:+.2f}%, MA 이격 {r.ext*100:+.1f}%, 단계 {int(stg.iloc[-1])}, "
              f"맨스필드 RS(vs SPY) {rr.iloc[-1]:+.2f} (13주 전 {rr.iloc[-14]:+.2f}), 52주 고가 {ww['High'].iloc[-52:].max():.2f} (대비 {r.Close/ww['High'].iloc[-52:].max()-1:+.1%})")
    row = sw.iloc[-1]
    base_now = find_base(sw, i, p)
    if base_now:
        print(f"\n{sig} 현재 주 직전까지의 트레이딩 범위: {base_now.start.date()}~{base_now.end.date()} {base_now.weeks}주, 저항 {base_now.high:.2f}, 지지 {base_now.low:.2f}, "
              f"폭 {base_now.width:.1%}, 터치 {base_now.touches}, 종가/저항 {row.Close/base_now.high-1:+.1%} → "
              + ("저항 위(돌파 상태)" if row.Close > base_now.high else "아직 범위 안 (buy-stop 후보가 = 저항 +0.5%)"))
    base_prev = find_base(sw, i - 1, p)
    if base_prev and row.Close > base_prev.high >= sw["Close"].iloc[-2]:
        print("※ 이번 주가 돌파 주 — 체크리스트 평가:")
        print(checks_to_frame(evaluate_breakout(sw, stages, rs, i, base_prev, p, mk, sc)).to_string(index=False))
    print("\n최근 26주 안의 돌파 신호(필터 무관):")
    for j in range(i - 26, i + 1):
        bse = find_base(sw, j, p)
        if bse and sw["Close"].iloc[j] > bse.high >= sw["Close"].iloc[j - 1]:
            cks = evaluate_breakout(sw, stages, rs, j, bse, p, market_checks(sw.index[j - 1], W, breadth, pd_ratio), sector_checks(sw.index[j - 1], W, breadth))
            fails = [c.code for c in cks if c.required and c.passed is False]
            print(f"- {sw.index[j].date()} res {bse.high:.2f} 베이스 {bse.weeks}주 등급 {grade(cks)} 거래량 {[c for c in cks if c.code=='4-1'][0].value} 필수 실패 {fails}")
    open_pos = [t for t in trades if t.get("open")]
    print("\n시뮬레이션 상 현재 포지션:", f"{open_pos[0]['entry_date']} 진입 @{open_pos[0]['entry_px']} 평가 {open_pos[0]['ret']:+.1%}" if open_pos else "없음(현금)")
    if closed:
        t = closed[-1]
        print(f"마지막 청산: {t['exit_date']} @{t['exit_px']} 사유 {[x[2] for x in t['exits']]}")
    print(f"\n{sig} 주봉 최근 8주:")
    print(sw[["Open", "High", "Low", "Close", "Volume", "MA", "slope", "ext"]].tail(8).assign(stage=stages.tail(8), rs=rs.tail(8)).round(3).to_string())


if __name__ == "__main__":
    main()
