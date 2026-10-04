"""오닐 «최고의 주식 최적의 타이밍» 시장 타이밍(M) + 매도 규칙을 TQQQ 에 적용하는 리서치 스크립트.

문서: docs/strategy_notes/오닐_최고의주식_최적의타이밍_체크리스트_2026_10.md, 오닐_TQQQ_검증_2026_10.md
라이브러리: strategy/adapters/oneil_market_timing.py

실행: docker compose exec backtester python oneil_tqqq_research.py [--variant NAME ...] [--synthetic-3x]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.oneil_market_timing import ONeilParams, apply_sell_rules, run_state_machine, simulate_exposure

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
md = CachedMarketDataAdapter(YFinanceAdapter())
TODAY = date.today()
WINDOWS = [("좋았을 때", "2023-01-01", "2024-12-31"), ("안 좋았을 때", "2022-01-01", "2022-12-31"), ("평범했을 때", "2015-01-01", "2016-12-31")]
EVENTS = [  # 사건 카탈로그 (TQQQ 고점→저점)
    ("2011 유럽위기", "2011-07-07", "2011-10-03"), ("2015-08 차이나", "2015-07-20", "2015-08-25"), ("2016-01", "2015-12-01", "2016-02-11"),
    ("2018 Q4", "2018-08-29", "2018-12-24"), ("2020 코로나", "2020-02-19", "2020-03-23"), ("2022", "2021-11-19", "2022-12-28"),
    ("2025 관세", "2025-02-19", "2025-04-08"), ("2026-03", "2026-02-20", "2026-04-10"),
]

VARIANTS = {
    "A_book": dict(ftd_max_day=10),                                         # 분산일 4/6 + 낙폭 8% + FTD 4~10일(이후 리셋) + 손절 24%
    "B_no_dd": dict(ftd_max_day=10, drawdown_corr=0.0),                     # 낙폭 규칙 제거(분산일만)
    "C_strict56": dict(ftd_max_day=10, pressure_count=5, correction_count=6),
    "D_ftd_anyday": dict(ftd_max_day=0),                                    # 10일 제한 제거(IBD 현대 관행)
    "E_no_losscut": dict(ftd_max_day=0, loss_cut=0.0),
    "F_breadth": dict(ftd_max_day=0, breadth_divergence=True),              # 수동 항목 "주도주 꺾임" 프록시
    "G_defensive": dict(ftd_max_day=0, defensive_rotation=True),            # 수동 항목 "방어주 강세" 프록시
    "H_rule35": dict(ftd_max_day=0, rule35_ma200_down=True),
    "I_rule34": dict(ftd_max_day=0, rule34_extended_200=0.70),
    "J_all_manual": dict(ftd_max_day=0, breadth_divergence=True, defensive_rotation=True, rule35_ma200_down=True),
    "K_no_stall": dict(ftd_max_day=0, stall_enabled=False),
    # --- 거래량 정의 변형 (QQQ ETF 거래량은 거래소 총거래량의 나쁜 대리변수) ---
    "L_mktvol": dict(_mktvol=True, ftd_max_day=10),                         # 분산일·FTD 거래량 = Nasdaq-100 구성종목 거래량 합
    "M_mktvol_anyday": dict(_mktvol=True, ftd_max_day=0),
    "N_avg50vol": dict(ftd_volume_mode="avg50", ftd_max_day=10),            # FTD 거래량 > 50일 평균
    "O_avg50vol_anyday": dict(ftd_volume_mode="avg50", ftd_max_day=0),
    "P_mktvol_anyday_manual": dict(_mktvol=True, ftd_max_day=0, breadth_divergence=True, defensive_rotation=True),
    "Q_mktvol_anyday_nodd": dict(_mktvol=True, ftd_max_day=0, drawdown_corr=0.0),
    "R_mktvol_anyday_56": dict(_mktvol=True, ftd_max_day=0, pressure_count=5, correction_count=6),
    "S_mktvol_anyday_57_dd12": dict(_mktvol=True, ftd_max_day=0, pressure_count=5, correction_count=7, drawdown_corr=0.12),
    "T_mktvol_anyday_nopressure": dict(_mktvol=True, ftd_max_day=0, pressure_count=99, correction_count=6),  # 50% 단계 없이 100/0 만
    "U_mktvol_ftd_above50": dict(_mktvol=True, ftd_max_day=0, reentry_above_ma50=True),                       # (판단) FTD + 50일선 위에서만 복귀
    "V_mktvol_above50_nopressure": dict(_mktvol=True, ftd_max_day=0, reentry_above_ma50=True, pressure_count=99),
    "W_mktvol_above50_57_dd12": dict(_mktvol=True, ftd_max_day=0, reentry_above_ma50=True, pressure_count=5, correction_count=7, drawdown_corr=0.12),
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def fetch(sym, start=date(1998, 1, 1), end=TODAY):
    df = md.fetch_ohlcv(sym, start, end)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def synth3x(qqq):
    r = qqq["Close"].pct_change().fillna(0) * 3 - 0.0095 / 252
    px = 100 * (1 + r).cumprod()
    ratio = px / qqq["Close"]
    return pd.DataFrame({"Open": qqq["Open"] * ratio.shift(1).bfill(), "High": qqq["High"] * ratio, "Low": qqq["Low"] * ratio,
                         "Close": px, "Volume": qqq["Volume"]}, index=qqq.index)


def stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, dd.min(), dd.idxmin().date()


def breadth_pct_above_50(dailies: dict[str, pd.DataFrame]) -> pd.Series:
    closes = pd.DataFrame({t: d["Close"] for t, d in dailies.items()}).sort_index()
    ma50 = closes.rolling(50).mean()
    return ((closes > ma50).sum(axis=1) / closes.notna().sum(axis=1)).rename("pct50")


def event_table(sm: pd.DataFrame, trd: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    rows = []
    for name, top, bottom in EVENTS:
        top, bottom = pd.Timestamp(top), pd.Timestamp(bottom)
        if top < sm.index[0] or bottom > sm.index[-1]:
            continue
        seg = sm.loc[top:bottom]
        tq = trd["Close"].reindex(sm.index).ffill()
        first_out = seg.index[seg["exposure"] < 1.0]
        first_cash = seg.index[seg["exposure"] == 0.0]
        peak_px = tq.loc[top]
        def at(dts):
            if len(dts) == 0:
                return "못 잡음", np.nan
            d0 = dts[0]
            return f"{d0.date()} (+{(d0 - top).days}일, TQQQ {tq.loc[d0]/peak_px-1:+.0%})", d0
        out_s, _ = at(first_out)
        cash_s, _ = at(first_cash)
        after = sm.loc[bottom:]
        back = after.index[after["exposure"] >= 1.0]
        back_s = f"{back[0].date()} (+{(back[0]-bottom).days}일, 저점 대비 {tq.loc[back[0]]/tq.loc[bottom]-1:+.0%})" if len(back) else "복귀 없음"
        bh = tq.loc[bottom] / peak_px - 1
        rows.append({"사건": name, "TQQQ 낙폭": f"{bh:+.0%}", "첫 축소(50%)": out_s, "현금(0%)": cash_s, "100% 복귀": back_s})
    return pd.DataFrame(rows)


def false_alarms(sm: pd.DataFrame, trd: pd.DataFrame) -> pd.DataFrame:
    """노출 < 100% 였던 구간 중 사건 카탈로그에 없는 구간 (오경보) — 그 사이 TQQQ 변동."""
    tq = trd["Close"].reindex(sm.index).ffill()
    below = sm["exposure"] < 1.0
    rows = []
    start = None
    for d, b in below.items():
        if b and start is None:
            start = d
        elif (not b) and start is not None:
            end = d
            in_event = any(pd.Timestamp(t) - pd.Timedelta(days=45) <= start <= pd.Timestamp(bt) + pd.Timedelta(days=120) for _, t, bt in EVENTS)
            rows.append({"시작": start.date(), "끝": end.date(), "일수": (end - start).days, "TQQQ 변동(그 사이)": f"{tq.loc[end]/tq.loc[start]-1:+.1%}",
                         "사건 구간?": "사건" if in_event else "오경보"})
            start = None
    return pd.DataFrame(rows)


def run_variant(name, over, qqq, trd, start, breadth50=None, xlu_ratio=None, mkt_vol=None):
    over = dict(over)
    use_mkt = over.pop("_mktvol", False)
    p = ONeilParams(**over)
    sm = run_state_machine(qqq, p, breadth50, xlu_ratio, mkt_vol if (use_mkt and mkt_vol is not None) else None)
    exp, sell_log = apply_sell_rules(sm["exposure"], trd, p)
    sm["exposure"] = exp
    sm = sm.loc[start:]
    eq = simulate_exposure(sm["exposure"], trd)
    eq = eq / eq.iloc[0]
    return p, sm, eq, sell_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--synthetic-3x", action="store_true")
    ap.add_argument("--start", default=None)
    ap.add_argument("--skip-breadth", action="store_true")
    ap.add_argument("--detail", nargs="*", default=["A_book", "M_mktvol_anyday"], help="상세 출력할 변형")
    args = ap.parse_args()

    log("데이터: QQQ TQQQ SPY XLU")
    qqq, tqqq, xlu = fetch("QQQ"), fetch("TQQQ"), fetch("XLU")
    trd = synth3x(qqq) if args.synthetic_3x else tqqq
    start = pd.Timestamp(args.start or ("2000-01-03" if args.synthetic_3x else "2011-01-03"))
    qqq = qqq.loc[: trd.index[-1]]
    xlu_ratio = ((xlu["Close"] / qqq["Close"]).pct_change(20)).reindex(qqq.index)

    breadth50, mkt_vol = None, None
    if not args.skip_breadth:
        tickers = WikipediaUniverseAdapter().get_tickers("nasdaq100")
        dailies = {}
        for t in tickers:
            try:
                dailies[t] = fetch(t, date(2009, 1, 1), TODAY - timedelta(days=1))
            except Exception as e:  # noqa: BLE001
                log(f"  {t} 실패: {e}")
        breadth50 = breadth_pct_above_50(dailies).reindex(qqq.index)
        mkt_vol = pd.DataFrame({t: d["Volume"] for t, d in dailies.items()}).sort_index().sum(axis=1).reindex(qqq.index)
        mkt_vol = mkt_vol.where(mkt_vol > 0)
        log(f"breadth: {len(dailies)}종목, 시장 거래량 프록시 {mkt_vol.dropna().index[0].date()}~")

    bh = trd["Close"].loc[start:]
    bh = bh / bh.iloc[0]
    qbh = qqq["Close"].loc[start:]
    qbh = qbh / qbh.iloc[0]
    print(f"# 오닐 시장 타이밍 × TQQQ  ({'합성 3x' if args.synthetic_3x else '실제 TQQQ'} {start.date()} ~ {trd.index[-1].date()})")
    c, m, md_ = stats(bh)
    print(f"- TQQQ 보유: CAGR {c:+.1%}, MDD {m:+.1%} ({md_})")
    c, m, md_ = stats(qbh)
    print(f"- QQQ 보유: CAGR {c:+.1%}, MDD {m:+.1%} ({md_})")

    rows, results = [], {}
    for name in args.variants:
        p, sm, eq, sell_log = run_variant(name, VARIANTS[name], qqq, trd, start, breadth50, xlu_ratio, mkt_vol)
        results[name] = (p, sm, eq, sell_log)
        c, m, md_ = stats(eq)
        row = {"변형": name, "CAGR": f"{c:+.1%}", "MDD": f"{m:+.1%}", "MDD저점": md_,
               "노출 100% 비율": f"{(sm['exposure']>=1).mean():.0%}", "현금 비율": f"{(sm['exposure']==0).mean():.0%}",
               "상태변경": int((sm['state'] != sm['state'].shift(1)).sum()), "손절": sum('손절' in s['rule'] for s in sell_log)}
        for wname, a, b in WINDOWS:
            e = eq.loc[a:b]; h = bh.loc[a:b]
            if len(e) > 10:
                row[wname] = f"{e.iloc[-1]/e.iloc[0]-1:+.0%} (보유 {h.iloc[-1]/h.iloc[0]-1:+.0%})"
        rows.append(row)
    print("\n## 변형 비교")
    print(pd.DataFrame(rows).to_string(index=False))

    for name in args.detail:
        if name not in results:
            continue
        detail(name, results[name], qqq, trd, bh, breadth50, xlu_ratio)


def detail(name, res, qqq, trd, bh, breadth50, xlu_ratio):
    p, sm, eq, sell_log = res
    print(f"\n## 상세: {name}  {VARIANTS[name]}")
    print("\n### 사건 카탈로그 적중")
    print(event_table(sm, trd, sm["exposure"]).to_string(index=False))
    fa = false_alarms(sm, trd)
    print(f"\n### 노출 축소 구간 전체 ({len(fa)}회, 오경보 {int((fa['사건 구간?']=='오경보').sum()) if len(fa) else 0}회)")
    print(fa.to_string(index=False))
    print("\n### 상태 변경 이력 (최근 40개)")
    ch = sm[sm["event"] != ""][["state", "dist_count", "rally_day", "exposure", "event"]]
    print(ch.tail(40).to_string())
    if sell_log:
        print("\n### 매도 규칙 발동")
        print(pd.DataFrame(sell_log).to_string(index=False))
    yr = pd.DataFrame({"전략": eq.resample("YE").last().pct_change(), "TQQQ": bh.resample("YE").last().pct_change()}).dropna()
    yr.index = yr.index.year
    print("\n### 연도별")
    print(yr.map(lambda v: f"{v:+.0%}").T.to_string())

    # 현재 상태
    last = sm.iloc[-1]
    print(f"\n## 현재 상태 ({sm.index[-1].date()}) — {name}")
    print(f"- 시장 상태: {last.state}, 분산일 {int(last.dist_count)}개 (압박 ≥{p.pressure_count}, 조정 ≥{p.correction_count}), 랠리 {int(last.rally_day)}일차, 노출 {last.exposure:.0%}")
    recent = sm.tail(p.dist_window)
    dd = recent[recent["dist_day"] | recent["stall_day"]]
    print("- 최근 25거래일 분산일/정체일:")
    for d, r in dd.iterrows():
        print(f"    {d.date()} {'분산' if r.dist_day else '정체'}  QQQ {qqq.loc[d,'Close']:.2f} ({qqq['Close'].pct_change().loc[d]:+.2%}) 거래량 {qqq.loc[d,'Volume']/1e6:.0f}M")
    qc = qqq["Close"]
    print(f"- QQQ {qc.iloc[-1]:.2f}: 52주 고점 대비 {qc.iloc[-1]/qc.iloc[-252:].max()-1:+.1%}, 50일선 {qc.rolling(50).mean().iloc[-1]:.0f}, 200일선 {qc.rolling(200).mean().iloc[-1]:.0f} (20일 기울기 {'상승' if qc.rolling(200).mean().diff(20).iloc[-1]>0 else '하락'})")
    tc = trd["Close"]
    print(f"- TQQQ {tc.iloc[-1]:.2f}: 200일선 위 {tc.iloc[-1]/tc.rolling(200).mean().iloc[-1]-1:+.0%} (규칙 34 기준 70%), 10주선 {tc.rolling(50).mean().iloc[-1]:.2f}")
    if breadth50 is not None:
        print(f"- Nasdaq-100 중 50일선 위 비율: {breadth50.dropna().iloc[-1]:.0%}")
    print(f"- XLU/QQQ 20일 비율 변화: {xlu_ratio.dropna().iloc[-1]:+.1%} (방어주 로테이션 기준 +5%)")
    print("- 최근 15거래일 상태:")
    print(sm.tail(15)[["state", "dist_count", "rally_day", "exposure", "dist_day", "stall_day", "event"]].assign(QQQ=qc.tail(15).round(2)).to_string())


if __name__ == "__main__":
    main()
