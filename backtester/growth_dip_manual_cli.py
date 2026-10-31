"""성장주 자기 PER 눌림(설계서 §2.7) — **수동 체크리스트(§3-1) 레짐 표본 점검** 보조 도구. 사람이 블라인드로 판정할 자료만 만든다.

    python growth_dip_manual_cli.py data                                  # 가격 캐시(parquet) + 레짐별 표본 달
    python growth_dip_manual_cli.py candidates 2022-06 [x_pct=30] [trigger=per_own] [out_dir]
        # 그 달 리스트 종목의 자기 PER 트리거(엔진과 같은 적격 조건)를 **트리거일까지만** 그린 차트 + 도시에(md)
    python growth_dip_manual_cli.py settle decisions.json                 # 사람 결정 → 다음날 시가 체결, PER 회복 종가 청산, 자동(전부 매수)·QQQ 대조

결정 파일: {"month": "2022-06", "x_pct": 30, "trigger": "per_own", "per_exit_pct": 0, "max_days": 250,
            "picks": {"AAPL@2022-06-14": {"buy": true, "why": "..."}, ...}}   — picks 에 없는 후보는 건너뜀으로 본다.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.evidence_backtest import classify_regimes, pick_dates
from strategy.adapters.growth_dip_scalp import (
    ScalpRules, _common_masks, build_arrays, daily_trigger_mask, earnings_mask, list_mask, next_earnings_exit, per_columns, precompute, weekly_lists,
)
from strategy.adapters.minervini_screener import evaluate_growth

CACHE = Path("/app/sweep_results/growth_dip_manual_prices.parquet")
OUT = Path("/app/sweep_results/growth_dip_manual")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd_data() -> None:
    from data.adapters.wikipedia_universe import default_universe_provider
    from data.adapters.yfinance_bulk import download_many
    prov = default_universe_provider()
    tk = list(dict.fromkeys(prov.get_tickers("sp500") + prov.get_tickers("nasdaq100"))) + ["QQQ", "SPY"]
    pr = download_many(tk, date(2014, 1, 1), date.today())
    df = pd.concat([g.assign(ticker=t) for t, g in pr.items()])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    reg = classify_regimes(pr["QQQ"])
    print("cached", len(pr), "tickers", df.index.min().date(), "~", df.index.max().date())
    for n, s in ((2, 7), (2, 11), (2, 23)):
        print(f"seed {s} n_each {n}:", [(r, d.strftime("%Y-%m")) for r, d in pick_dates(reg, n, s, "2017-01-01", "2024-12-31")])


def _load() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(CACHE)
    return {t: g.drop(columns="ticker").sort_index() for t, g in df.groupby("ticker")}


class Prep:
    """엔진과 같은 전처리: 가격·펀더멘털·자기 PER 열·리스트·배열·적격 마스크."""

    def __init__(self, rules: ScalpRules, start: str = "2016-01-04"):
        from growth_dip_scalp_cli import load_snapshots
        prices = _load()
        self.qqq = prices.pop("QQQ"); prices.pop("SPY", None)
        self.cal = pd.to_datetime(self.qqq.index)
        self.data = precompute(prices, self.cal)
        self.index_df = precompute({"QQQ": self.qqq}, self.cal)["QQQ"]
        self.snaps, self.industry = load_snapshots(list(self.data))
        self.regimes = {(d.year, d.month): r for d, r in classify_regimes(self.qqq).regime.items()}
        self.eps_tbl: dict[str, tuple[np.ndarray, np.ndarray, list]] = {}
        for t, qs in self.snaps.items():
            rows = sorted(((pd.Timestamp(q.effective_report_date), q.eps_actual, q) for q in qs if q.eps_actual is not None), key=lambda r: r[0])
            if len(rows) >= 4:
                self.eps_tbl[t] = (np.array([r[0].value for r in rows]), np.array([r[1] for r in rows], dtype=float), [r[2] for r in rows])
                per_columns(self.data[t], self.eps_tbl[t][0], self.eps_tbl[t][1], rules.per_lookback)
        self.rules = rules
        self.start, self.end = pd.Timestamp(start), self.cal[-1]
        cache: dict = {}

        def growth_fn(t: str, as_of: date) -> bool:
            k = (t, as_of)
            if k not in cache:
                g = evaluate_growth(self.snaps.get(t, []), as_of, eps_min_yoy=rules.eps_min, rev_min_yoy=rules.rev_min, n_required=rules.growth_required)
                cache[k] = bool(g.data_available and g.passed)
            return cache[k]
        self.lists = weekly_lists(list(self.data), self.cal, growth_fn, self.data, rules, self.start, self.end)
        self.arr = build_arrays(self.data, self.cal, self.index_df, self.start, self.end)
        self.lmask = list_mask(self.arr, self.lists)
        # 발표일 제외 + 실적까지 거래일 < min 제외 (CLI 와 같은 식)
        earn_days: dict[str, set[pd.Timestamp]] = {}
        for t, qs in self.snaps.items():
            s = set()
            for q in qs:
                if q.report_date is not None:
                    pos = self.cal.searchsorted(pd.Timestamp(q.report_date))
                    for k in range(0, rules.earnings_exclude_days + 1):
                        if pos + k < len(self.cal):
                            s.add(self.cal[pos + k])
            earn_days[t] = s
        em = earnings_mask(self.arr, earn_days)
        self.earn_dates = {t: [pd.Timestamp(q.report_date) for q in qs if q.report_date] for t, qs in self.snaps.items()}
        if rules.min_days_to_earnings > 0:
            nxt = next_earnings_exit(self.arr, self.earn_dates, 0)
            dte = nxt - np.arange(len(self.arr.days))[:, None]
            em = em | ((nxt < len(self.arr.days)) & (dte < rules.min_days_to_earnings))
        self.emask = em
        valid, idx_ok, liquid = _common_masks(self.arr, rules)
        trig = daily_trigger_mask(self.arr, rules)
        self.elig = trig & self.lmask & ~self.emask & idx_ok[:, None] & liquid

    def all_events(self, lo: str, hi: str) -> list[tuple[int, int]]:
        """lo~hi 전체의 적격 트리거 (i, j) — 종목·달마다 첫 날만 (수동 표본과 같은 정의)."""
        days = self.arr.days
        out = []
        for ym in sorted({(d.year, d.month) for d in days if pd.Timestamp(lo) <= d <= pd.Timestamp(hi)}):
            out += self.events(f"{ym[0]:04d}-{ym[1]:02d}")
        return out

    def events(self, month: str) -> list[tuple[int, int]]:
        """그 달의 적격 트리거 (i, j) — 종목당 첫 날만."""
        y, m = (int(x) for x in month.split("-"))
        days = self.arr.days
        seen = set(); out = []
        for i in np.flatnonzero((days.year == y) & (days.month == m)):
            for j in np.flatnonzero(self.elig[i]):
                if j not in seen:
                    seen.add(j); out.append((int(i), int(j)))
        return out


def _dossier(p: Prep, i: int, j: int) -> dict:
    """트리거일까지의 정보만으로 §3-1 의 숫자."""
    t = p.arr.tickers[j]; d = p.data[t]; day = p.arr.days[i]
    hist = d.loc[:day]
    c = float(hist["Close"].iloc[-1]); eps = float(hist["ttm_eps"].iloc[-1]); pref = float(hist["per_ref"].iloc[-1])
    win = hist["per"].iloc[-(p.rules.per_lookback + 1):-1].dropna()
    px_win = hist["Close"].iloc[-(p.rules.per_lookback + 1):-1]
    eps_then = float(hist["ttm_eps"].iloc[-(p.rules.per_lookback + 1)]) if len(hist) > p.rules.per_lookback else np.nan
    rep_ns, eps_q, qs = p.eps_tbl[t]
    k = int(np.searchsorted(rep_ns, day.value, side="right"))
    quarters = []
    for q in qs[max(0, k - 8):k]:
        yoy = next((q.eps_actual / abs(pq.eps_actual) - 1 for pq in qs[:k] if pq.fiscal_date == _year_ago(q.fiscal_date) and pq.eps_actual), None)
        quarters.append({"분기": str(q.fiscal_date), "발표": str(q.effective_report_date), "EPS": round(q.eps_actual, 2), "YoY": None if yoy is None else round(yoy, 2)})
    last_rep = [r for r in p.earn_dates.get(t, []) if r <= day]
    gap = None; since = None
    if last_rep:
        lr = max(last_rep); pos = p.cal.searchsorted(lr)
        if pos < len(p.cal) and p.cal[pos] in d.index:
            r = d.loc[p.cal[pos]]
            gap = float(r["Open"] / r["prev_close"] - 1) if pd.notna(r["prev_close"]) else None
        since = int(((p.cal > lr) & (p.cal <= day)).sum())
    q60 = p.qqq["Close"].reindex(p.cal).loc[:day]
    return {
        "티커": t, "트리거일": str(day.date()), "종가": round(c, 2), "레벨": round(float(pref * (1 - p.rules.x_pct / 100) * eps), 2),
        "PER 지금": round(c / eps, 1), "원래 PER(250일 중앙)": round(pref, 1),
        "PER 창 최소/최대": [round(float(win.min()), 1), round(float(win.max()), 1)] if len(win) else None,
        "P-1 가격 변화(250일 중앙 대비)": round(c / float(px_win.median()) - 1, 3) if len(px_win) else None,
        "P-1 EPS 변화(250일 전 대비)": round(eps / eps_then - 1, 3) if eps_then and eps_then > 0 else None,
        "P-2 마지막 발표": str(max(last_rep).date()) if last_rep else None, "P-2 발표일 갭": None if gap is None else round(gap, 3), "P-2 발표 뒤 거래일": since,
        "P-3 분기 EPS": quarters,
        "P-5 60일 종목/QQQ": [round(c / float(hist["Close"].iloc[-61]) - 1, 3) if len(hist) > 61 else None,
                            round(float(q60.iloc[-1] / q60.iloc[-61]) - 1, 3) if len(q60) > 61 else None],
        "P-6 50일선/200일선 대비": [round(c / float(hist["sma50"].iloc[-1]) - 1, 3) if pd.notna(hist["sma50"].iloc[-1]) else None,
                               round(c / float(hist["sma200"].iloc[-1]) - 1, 3) if pd.notna(hist["sma200"].iloc[-1]) else None],
        "P-6 200일선 기울기(21일)": round(float(hist["sma200"].iloc[-1] / hist["sma200"].iloc[-22] - 1), 3) if len(hist) > 22 and pd.notna(hist["sma200"].iloc[-22]) else None,
        "P-6 52주 고점 대비": round(c / float(hist["High"].iloc[-252:].max()) - 1, 3),
        "P-7 거래량 5일/20일": round(float(hist["Volume"].iloc[-5:].mean() / hist["vol20"].iloc[-1]), 2) if pd.notna(hist["vol20"].iloc[-1]) else None,
    }


def _year_ago(fd) -> object:
    try:
        return fd.replace(year=fd.year - 1)
    except Exception:  # noqa: BLE001
        return None


def _chart(p: Prep, i: int, j: int, out_png: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = "/app/pattern-finder-cache/fonts/AppleGothic.ttf"
    try:
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    except Exception:  # noqa: BLE001
        pass
    plt.rcParams["axes.unicode_minus"] = False
    t = p.arr.tickers[j]; day = p.arr.days[i]
    d = p.data[t].loc[:day].dropna(subset=["Close"])
    q = p.qqq.reindex(p.cal).loc[:day]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1.2]})

    def candles(ax, df, title):
        x = np.arange(len(df)); up = df["Close"] >= df["Open"]
        ax.bar(x, (df["Close"] - df["Open"]).abs().where(lambda s: s > 0, 1e-9), bottom=df[["Open", "Close"]].min(axis=1), width=0.6,
               color=np.where(up, "#d62728", "#1f77b4"))
        ax.vlines(x, df["Low"], df["High"], color=np.where(up, "#d62728", "#1f77b4"), linewidth=0.6)
        for col, cc in (("sma50", "orange"), ("sma200", "purple")):
            if col in df:
                ax.plot(x, df[col].to_numpy(), color=cc, linewidth=0.9)
        ax.set_yscale("log"); ax.set_title(title, fontsize=9); ax.tick_params(labelsize=6)
        step = max(1, len(df) // 8); ax.set_xticks(x[::step]); ax.set_xticklabels([ix.strftime("%y-%m-%d") for ix in df.index[::step]], fontsize=6)
        return x

    wk = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna().tail(104)
    wk["sma50"] = d["Close"].rolling(250).mean().resample("W-FRI").last().reindex(wk.index)      # 주봉 위 1년선
    candles(axes[0, 0], wk, f"{t} 주봉 2년 (~{day.date()}) · 보라=250일선")
    dd = d.tail(130)
    x = candles(axes[0, 1], dd, f"{t} 일봉 6개월 · 주황=50일선 보라=200일선 · 레벨 {float(p.arr.PREF[i, j] * (1 - p.rules.x_pct / 100) * p.arr.EPS[i, j]):.2f}")
    axes[0, 1].axhline(float(p.arr.PREF[i, j] * (1 - p.rules.x_pct / 100) * p.arr.EPS[i, j]), color="green", linestyle="--", linewidth=0.8)
    # PER 선 (1년) + 원래 PER
    pp = d["per"].tail(260); xx = np.arange(len(pp))
    axes[1, 0].plot(xx, pp.to_numpy(), color="black", linewidth=0.9, label="PER(종가/TTM EPS)")
    axes[1, 0].plot(xx, d["per_ref"].tail(260).to_numpy(), color="gray", linewidth=0.9, linestyle="--", label="원래 PER(250일 중앙, 전날까지)")
    axes[1, 0].plot(xx, (d["per_ref"].tail(260) * (1 - p.rules.x_pct / 100)).to_numpy(), color="green", linewidth=0.8, linestyle=":", label=f"−{p.rules.x_pct:.0f}%")
    axes[1, 0].legend(fontsize=7); axes[1, 0].set_title("PER 1년", fontsize=9); axes[1, 0].tick_params(labelsize=6)
    step = max(1, len(pp) // 8); axes[1, 0].set_xticks(xx[::step]); axes[1, 0].set_xticklabels([ix.strftime("%y-%m-%d") for ix in pp.index[::step]], fontsize=6)
    # 거래량 + QQQ 6개월
    axes[1, 1].bar(x, dd["Volume"].to_numpy(), color="#999999", width=0.6)
    axes[1, 1].plot(x, dd["vol20"].to_numpy(), color="red", linewidth=0.8)
    ax2 = axes[1, 1].twinx(); qq = q["Close"].tail(len(dd)); ax2.plot(np.arange(len(qq)), qq.to_numpy(), color="blue", linewidth=0.9); ax2.tick_params(labelsize=6)
    axes[1, 1].set_title("거래량(회색)·20일 평균(빨강) · QQQ(파랑, 오른축)", fontsize=9); axes[1, 1].tick_params(labelsize=6)
    axes[1, 1].set_xticks(x[::max(1, len(dd) // 8)]); axes[1, 1].set_xticklabels([ix.strftime("%y-%m-%d") for ix in dd.index[::max(1, len(dd) // 8)]], fontsize=6)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=72); plt.close(fig)


def _rules(x_pct: float, trigger: str, trend_filter: str | None = None) -> ScalpRules:
    return ScalpRules(trigger=trigger, x_pct=x_pct, hold_mode="per", max_hold_days=0, index_filter=False, trend_filter=trend_filter, fill="pessimistic")


def cmd_candidates(month: str, x_pct: float = 30.0, trigger: str = "per_own", out_dir: str | None = None) -> None:
    out = Path(out_dir) if out_dir else OUT / month
    p = Prep(_rules(x_pct, trigger))
    ev = p.events(month)
    log(f"{month}: 적격 트리거 {len(ev)} 종목 (x {x_pct}, {trigger}, 리스트 평균 {np.mean([len(v) for v in p.lists.values()]):.0f} 종목/주)")
    L = [f"# {month} 자기 PER 트리거 후보 (x {x_pct}, {trigger}) — 트리거일까지의 정보만", ""]
    rows = []
    for i, j in ev:
        ds = _dossier(p, i, j); rows.append(ds)
        png = out / f"cand_{ds['티커']}_{ds['트리거일']}.png"
        _chart(p, i, j, png)
        L.append(f"## {ds['티커']} @ {ds['트리거일']}  ![]({png.name})")
        L.append("")
        for k, v in ds.items():
            if k in ("티커", "트리거일"):
                continue
            if k == "P-3 분기 EPS":
                L.append(f"- {k}: " + " · ".join(f"{q['분기'][:7]} {q['EPS']}" + (f"({q['YoY']:+.0%})" if q['YoY'] is not None else "") for q in v))
            else:
                L.append(f"- {k}: {v}")
        L.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dossier.md").write_text("\n".join(L))
    (out / "candidates.json").write_text(json.dumps({"month": month, "x_pct": x_pct, "trigger": trigger, "rows": rows}, ensure_ascii=False, indent=1, default=str))
    print(f"{out}/dossier.md · 후보 {len(rows)}")


def _forward(p: Prep, i: int, j: int, per_exit_pct: float, max_days: int) -> dict:
    """다음날 시가 진입 → 종가 PER ≥ 진입 시 원래 PER × (1 − exit) 청산 / EPS 소멸 / 기한 / 구간 종료."""
    arr = p.arr; T = len(arr.days); fee = p.rules.fee_side
    if i + 1 >= T or np.isnan(arr.O[i + 1, j]):
        return {"ret": None, "why": "데이터 없음"}
    e = float(arr.O[i + 1, j]); target = float(arr.PREF[i, j]) * (1 - per_exit_pct / 100)
    q = p.qqq["Close"].reindex(arr.days)
    for k in range(i + 1, min(T, i + 1 + max_days)):
        c = float(arr.C[k, j]); eps = float(arr.EPS[k, j])
        if np.isnan(c):
            continue
        why = None
        if np.isnan(eps) or eps <= 0:
            why = "PER 소멸"
        elif c / eps >= target:
            why = "PER 회복"
        elif k == T - 1:
            why = "구간 종료"
        elif k == i + max_days:
            why = "기한"
        if why:
            return {"ret": c / e - 1 - 2 * fee, "why": why, "hold": k - i - 1, "exit": str(arr.days[k].date()), "entry": round(e, 2),
                    "QQQ": float(q.iloc[k] / q.iloc[i + 1] - 1), "최저": float(np.nanmin(arr.L[i + 1:k + 1, j]) / e - 1)}
    k = min(T - 1, i + max_days); c = float(arr.C[k, j])
    return {"ret": c / e - 1 - 2 * fee, "why": "기한", "hold": k - i - 1, "exit": str(arr.days[k].date()), "entry": round(e, 2),
            "QQQ": float(q.iloc[k] / q.iloc[i + 1] - 1), "최저": float(np.nanmin(arr.L[i + 1:k + 1, j]) / e - 1)}


def cmd_settle(path: str) -> None:
    dec = json.load(open(path))
    month = dec["month"]; x_pct = float(dec.get("x_pct", 30)); trigger = dec.get("trigger", "per_own")
    exit_pct = float(dec.get("per_exit_pct", 0)); max_days = int(dec.get("max_days", 250))
    p = Prep(_rules(x_pct, trigger))
    ev = p.events(month)
    picks = dec.get("picks", {})
    rows = []
    for i, j in ev:
        t = p.arr.tickers[j]; key = f"{t}@{p.arr.days[i].date()}"
        fw = _forward(p, i, j, exit_pct, max_days)
        pk = picks.get(key, {})
        rows.append({"후보": key, "사람": "매수" if pk.get("buy") else "건너뜀", "이유": pk.get("why", ""), **fw})
    df = pd.DataFrame(rows)
    ok = df[df["ret"].notna()]

    def agg(sub: pd.DataFrame) -> dict:
        if not len(sub):
            return {"n": 0}
        return {"n": int(len(sub)), "평균": float(sub["ret"].mean()), "중앙": float(sub["ret"].median()), "승률": float((sub["ret"] > 0).mean()),
                "QQQ 같은 기간 평균": float(sub["QQQ"].mean()), "평균 보유일": float(sub["hold"].mean()), "최악": float(sub["ret"].min()),
                "청산": sub["why"].value_counts().to_dict()}
    summary = {"자동(전부)": agg(ok), "사람 매수": agg(ok[ok["사람"] == "매수"]), "사람 건너뜀": agg(ok[ok["사람"] == "건너뜀"])}
    out = {"month": month, "x_pct": x_pct, "trigger": trigger, "per_exit_pct": exit_pct, "max_days": max_days, "rows": rows, "summary": summary}
    op = Path(path).with_name(Path(path).stem + "_settle.json")
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"
    L = [f"# {month} 수동 판정 결과 (x {x_pct} {trigger}, 청산 PER 회복 −{exit_pct}% / {max_days}일)", "",
         "| 후보 | 사람 | 이유 | 진입 | 청산 | 보유일 | 사유 | 손익 | 최저 | QQQ 같은 기간 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['후보']} | {r['사람']} | {r['이유']} | {r.get('entry', '—')} | {r.get('exit', '—')} | {r.get('hold', '—')} | {r['why']} | {f(r['ret'])} | {f(r.get('최저'))} | {f(r.get('QQQ'))} |")
    L += ["", "| 묶음 | n | 평균 | 중앙 | 승률 | 평균 보유일 | 최악 | QQQ 같은 기간 | 청산 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary.items():
        if v["n"]:
            L.append(f"| {k} | {v['n']} | {f(v['평균'])} | {f(v['중앙'])} | {v['승률']:.0%} | {v['평균 보유일']:.0f} | {f(v['최악'])} | {f(v['QQQ 같은 기간 평균'])} | {v['청산']} |")
        else:
            L.append(f"| {k} | 0 | | | | | | | |")
    mp = op.with_suffix(".md"); mp.write_text("\n".join(L))
    print(f"{op} · {mp}")
    print("\n".join(L[-5:]))


def cmd_features(x_pct: float = 30.0, trigger: str = "per_own", lo: str = "2017-01-01", hi: str = "2025-10-31",
                 per_exit_pct: float = 0.0, max_days: int = 250, out_dir: str | None = None) -> None:
    """전체 적격 이벤트(포지션 제한 없음)를 같은 규칙으로 정산하고 진입일 특징을 붙인다 → CSV + 좋은/나쁜 거래 비교 md (설계서 §3-2)."""
    out = Path(out_dir) if out_dir else OUT / "features"
    out.mkdir(parents=True, exist_ok=True)
    p = Prep(_rules(x_pct, trigger))
    ev = p.all_events(lo, hi)
    log(f"적격 이벤트 {len(ev)} ({lo}~{hi}, x {x_pct}, {trigger})")
    rows = []
    for n, (i, j) in enumerate(ev):
        ds = _dossier(p, i, j); fw = _forward(p, i, j, per_exit_pct, max_days)
        if fw.get("ret") is None:
            continue
        q = ds["P-3 분기 EPS"]; yoys = [x["YoY"] for x in q if x["YoY"] is not None]
        day = pd.Timestamp(ds["트리거일"])
        a, b = ds["P-6 50일선/200일선 대비"]; s60, q60 = ds["P-5 60일 종목/QQQ"]
        eps_chg = ds["P-1 EPS 변화(250일 전 대비)"]; px_chg = ds["P-1 가격 변화(250일 중앙 대비)"]
        win = ds["PER 창 최소/최대"]
        rows.append({
            "ticker": ds["티커"], "day": ds["트리거일"], "year": day.year, "regime": p.regimes.get((day.year, day.month), "?"),
            "industry": p.industry.get(ds["티커"]) or "?",
            "kind": "EPS상승형" if (eps_chg or 0) >= 0.2 and (px_chg or 0) >= -0.05 else ("가격하락형" if (px_chg or 0) < -0.05 else "혼합"),
            "px_vs_med": px_chg, "eps_chg": eps_chg, "per_now": ds["PER 지금"], "per_ref": ds["원래 PER(250일 중앙)"],
            "discount": ds["PER 지금"] / ds["원래 PER(250일 중앙)"] - 1,
            "win_max_over_ref": (win[1] / ds["원래 PER(250일 중앙)"]) if win else None,
            "days_since_report": ds["P-2 발표 뒤 거래일"], "report_gap": ds["P-2 발표일 갭"],
            "last_yoy": yoys[-1] if yoys else None, "n_pos_yoy4": sum(y > 0 for y in yoys[-4:]) if yoys else None,
            "px_vs_sma50": a, "px_vs_sma200": b, "sma50_below_200": (a is not None and b is not None and a > b),
            "sma200_slope": ds["P-6 200일선 기울기(21일)"], "dd_52w": ds["P-6 52주 고점 대비"],
            "ret60": s60, "qqq60": q60, "rel60": (s60 - q60) if (s60 is not None and q60 is not None) else None,
            "vol_ratio": ds["P-7 거래량 5일/20일"], "price": ds["종가"],
            "ret": fw["ret"], "qqq": fw["QQQ"], "excess": fw["ret"] - fw["QQQ"], "hold": fw["hold"], "why": fw["why"], "worst": fw["최저"],
        })
        if (n + 1) % 100 == 0:
            log(f"  {n + 1}/{len(ev)}")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"per_features_{trigger}_x{x_pct:.0f}.csv", index=False)
    (out / f"per_features_{trigger}_x{x_pct:.0f}.md").write_text(_feature_report(df, x_pct, trigger, max_days))
    print(out)


def _feature_report(df: pd.DataFrame, x_pct: float, trigger: str, max_days: int) -> str:
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"
    split_year = int(df["year"].median())
    L = [f"# 자기 PER 트리거({trigger}, X {x_pct:.0f}) 전체 이벤트 — 좋은/나쁜 거래의 진입일 특징", "",
         f"- 이벤트 {len(df)} 건 ({df['day'].min()} ~ {df['day'].max()}), 종목·달마다 첫 트리거, 포지션 제한 없음. 다음날 시가 매수 → 종가 PER ≥ 진입 시 원래 PER 청산 / EPS 소멸 / {max_days} 거래일.",
         f"- 전체: 평균 {f(df['ret'].mean())}, 중앙 {f(df['ret'].median())}, 승률 {(df['ret'] > 0).mean():.0%}, QQQ 같은 기간 평균 {f(df['qqq'].mean())}, QQQ 이김 {(df['excess'] > 0).mean():.0%}. 청산 {df['why'].value_counts().to_dict()}",
         f"- '나쁜 거래' = 손실(ret < 0) {int((df['ret'] < 0).sum())} 건, '좋은 거래' = QQQ 보다 +10%p 이상 {int((df['excess'] >= 0.10).sum())} 건.", ""]
    num = [("discount", "PER 할인 폭(지금/원래 −1)"), ("px_vs_med", "가격 vs 250일 중앙"), ("eps_chg", "TTM EPS 변화(250일)"), ("per_now", "PER 지금"),
           ("per_ref", "원래 PER"), ("win_max_over_ref", "PER 창 최대/원래(적자탈출 오염)"), ("days_since_report", "발표 뒤 거래일"), ("report_gap", "발표일 갭"),
           ("last_yoy", "최근 분기 EPS YoY"), ("n_pos_yoy4", "최근 4분기 YoY>0 개수"), ("px_vs_sma50", "가격 vs 50일선"), ("px_vs_sma200", "가격 vs 200일선"),
           ("sma200_slope", "200일선 21일 기울기"), ("dd_52w", "52주 고점 대비"), ("rel60", "60일 종목−QQQ"), ("qqq60", "QQQ 60일"), ("vol_ratio", "거래량 5일/20일"), ("price", "주가")]
    L += ["## 1. 숫자 특징 — 3분위별 (낮음 / 중간 / 높음)", "", "| 특징 | 구간 | n | 평균 수익 | 손실 비율 | QQQ 초과 평균 | 좋은 거래 비율 |", "|---|---|---|---|---|---|---|"]
    for col, name in num:
        s = df[col].astype(float)
        if s.notna().sum() < 30 or s.nunique() < 4:
            continue
        try:
            cat = pd.qcut(s, 3, labels=["낮음", "중간", "높음"], duplicates="drop")
        except ValueError:
            continue
        for lab in cat.cat.categories:
            g = df[cat == lab]
            lo_, hi_ = float(s[cat == lab].min()), float(s[cat == lab].max())
            rng_ = f"{f(lo_)}~{f(hi_)}" if abs(float(s.max())) < 5 else f"{lo_:.0f}~{hi_:.0f}"
            L.append(f"| {name} | {lab} ({rng_}) | {len(g)} | {f(g['ret'].mean())} | {(g['ret'] < 0).mean():.0%} | {f(g['excess'].mean())} | {(g['excess'] >= 0.10).mean():.0%} |")
    L += ["", "## 2. 범주 특징", "", "| 특징 | 값 | n | 평균 수익 | 손실 비율 | QQQ 초과 평균 |", "|---|---|---|---|---|---|"]
    for col in ("kind", "regime", "sma50_below_200", "why", "year"):
        for v, g in df.groupby(col):
            L.append(f"| {col} | {v} | {len(g)} | {f(g['ret'].mean())} | {(g['ret'] < 0).mean():.0%} | {f(g['excess'].mean())} |")
    ind = df.groupby("industry").agg(n=("ret", "size"), ret=("ret", "mean"), loss=("ret", lambda x: (x < 0).mean()), ex=("excess", "mean"))
    ind = ind[ind.n >= 8].sort_values("ret")
    L += ["", "## 3. 산업군 (8 건 이상)", "", "| 산업군 | n | 평균 수익 | 손실 비율 | QQQ 초과 |", "|---|---|---|---|---|"]
    L += [f"| {k} | {r.n} | {f(r.ret)} | {r.loss:.0%} | {f(r.ex)} |" for k, r in ind.iterrows()]
    # 손실 vs 이익 평균 특징
    L += ["", "## 4. 나쁜 거래(손실) vs 좋은 거래(QQQ +10%p) 의 진입일 평균", "", "| 특징 | 손실 거래 | 좋은 거래 | 나머지 |", "|---|---|---|---|"]
    bad = df[df["ret"] < 0]; good = df[df["excess"] >= 0.10]; rest = df[(df["ret"] >= 0) & (df["excess"] < 0.10)]
    for col, name in num:
        if df[col].notna().sum() < 30:
            continue
        fmt = (lambda v: f"{v:.2f}") if col in ("per_now", "per_ref", "win_max_over_ref", "days_since_report", "n_pos_yoy4", "vol_ratio", "price") else f
        L.append(f"| {name} | {fmt(bad[col].astype(float).median())} | {fmt(good[col].astype(float).median())} | {fmt(rest[col].astype(float).median())} |")
    L += [f"| kind EPS상승형 비율 | {(bad['kind'] == 'EPS상승형').mean():.0%} | {(good['kind'] == 'EPS상승형').mean():.0%} | {(rest['kind'] == 'EPS상승형').mean():.0%} |",
          f"| 하락장 진입 비율 | {(bad['regime'] == '하락장').mean():.0%} | {(good['regime'] == '하락장').mean():.0%} | {(rest['regime'] == '하락장').mean():.0%} |"]
    L += ["", f"## 5. 앞 기간(~{split_year}) / 뒤 기간({split_year + 1}~) 나눠 보기 — 특징이 기간을 넘어 유지되는가", "",
          "| 특징 | 구간 | 앞 n | 앞 평균 | 앞 손실 | 뒤 n | 뒤 평균 | 뒤 손실 |", "|---|---|---|---|---|---|---|---|"]
    early = df[df.year <= split_year]; late = df[df.year > split_year]
    for col, name in num:
        s = df[col].astype(float)
        if s.notna().sum() < 30 or s.nunique() < 4:
            continue
        try:
            edges = s.quantile([1 / 3, 2 / 3]).to_numpy()
        except ValueError:
            continue
        for lab, m in (("낮음", lambda v: v <= edges[0]), ("높음", lambda v: v > edges[1])):
            ge, gl = early[m(early[col].astype(float))], late[m(late[col].astype(float))]
            L.append(f"| {name} | {lab} | {len(ge)} | {f(ge['ret'].mean())} | {(ge['ret'] < 0).mean() if len(ge) else float('nan'):.0%} | {len(gl)} | {f(gl['ret'].mean())} | {(gl['ret'] < 0).mean() if len(gl) else float('nan'):.0%} |")
    for col in ("kind", "regime"):
        for v in df[col].unique():
            ge, gl = early[early[col] == v], late[late[col] == v]
            L.append(f"| {col} | {v} | {len(ge)} | {f(ge['ret'].mean())} | {(ge['ret'] < 0).mean() if len(ge) else float('nan'):.0%} | {len(gl)} | {f(gl['ret'].mean())} | {(gl['ret'] < 0).mean() if len(gl) else float('nan'):.0%} |")
    return "\n".join(L)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "data":
        cmd_data()
    elif cmd == "candidates":
        cmd_candidates(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 30.0, sys.argv[4] if len(sys.argv) > 4 else "per_own",
                       sys.argv[5] if len(sys.argv) > 5 else None)
    elif cmd == "settle":
        cmd_settle(sys.argv[2])
    elif cmd == "features_md":   # 저장된 CSV 로 md 만 다시
        _df = pd.read_csv(sys.argv[2]); Path(sys.argv[2]).with_suffix(".md").write_text(_feature_report(_df, float(sys.argv[3]), sys.argv[4], 250)); print("md ok")
    elif cmd == "features":   # python growth_dip_manual_cli.py features [x_pct] [trigger] [lo] [hi]
        cmd_features(float(sys.argv[2]) if len(sys.argv) > 2 else 30.0, sys.argv[3] if len(sys.argv) > 3 else "per_own",
                     sys.argv[4] if len(sys.argv) > 4 else "2017-01-01", sys.argv[5] if len(sys.argv) > 5 else "2025-10-31")
    else:
        raise SystemExit(__doc__)
