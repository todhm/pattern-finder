"""근거중심매매 **블라인드 수동 판정** 검증 (2026-11-12, 체크리스트 M-I 절차).

첫 수동 백테스트(2026-10-28)는 판정자가 티커·날짜를 알고 3 창을 순서대로 봤다. 이 도구는
  1) 새 창(레짐별 n_each 개, 이전에 본 창과 겹치지 않게)에서 금요일마다 엔진의 주봉 후보(예 ≥ 4)를 k 개씩 무작위로 뽑아 **풀**을 만들고,
  2) 티커·날짜·가격을 가린 차트(가격 = 기준일 종가 100, x축 = 기준일까지 남은 봉 수)를 **섞은 순서**로 그려 판정자에게 주고,
  3) 판정(buy/skip + 확신 1~5)을 받아 같은 풀·같은 엔진 규칙에서 [사람 승인] vs [기계: 풀 전부] vs [풀 안 무작위 같은 개수] 를 비교한다.
풀·규칙·체결이 같으므로 차이는 '사람의 차트 판독'뿐이다.

    python evidence_blind_cli.py pool   '{"seed": 101, "n_each": 2, "horizon_days": 75, "k": 5, "out_dir": "/app/sweep_results/evidence_blind_2026-11-12"}'
    python evidence_blind_cli.py settle '{"out_dir": "...", "decisions": ["...json", ...], "placebo_seeds": 100, "report_md": "...md", "out": "...json"}'

결정 파일: {"c001": {"v": "buy"|"skip", "conv": 1~5, "why": "..."} , ...} (여러 파일이면 뒤 파일이 앞을 덮지 않고 합쳐짐 — 중복 id 는 첫 파일 우선, 재판정 일치율은 `agree` 로).
    python evidence_blind_cli.py agree a.json b.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.evidence_backtest import (
    SimRules, buy_hold, classify_regimes, entry_levels, pick_dates, precompute, reports_at, run_sim,
)
from strategy.adapters.evidence_checklist import EvidenceParams

CACHE = Path("/app/sweep_results/evidence_manual_prices.parquet")
# 이전 라운드에서 결과를 이미 본 창(자동 12 + 수동 3) — 새 창은 이 날짜들과 ±150일 안 겹치게
SEEN = ["2007-11-30", "2008-03-31", "2009-07-31", "2010-01-29", "2010-05-28", "2011-12-30", "2018-02-28", "2019-07-31",
        "2022-05-31", "2022-06-30", "2024-06-28", "2024-12-31"]
_G = None


def _load() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(CACHE)
    return {t: g.drop(columns="ticker").sort_index() for t, g in df.groupby("ticker")}


def choose_windows(qqq: pd.DataFrame, n_each: int, seed: int, hi: str) -> list[tuple[str, pd.Timestamp]]:
    """레짐별 n_each 개. 이미 본 창과 ±150일, 서로 ±150일 안 겹치는 것만 seed 순서대로 채운다."""
    reg = classify_regimes(qqq)
    seen = [pd.Timestamp(s) for s in SEEN]
    out: list[tuple[str, pd.Timestamp]] = []
    for s in range(seed, seed + 200):
        for r, d in pick_dates(reg, 6, s, "2006-01-01", hi):
            if sum(1 for rr, _ in out if rr == r) >= n_each:
                continue
            if any(abs((d - x).days) < 150 for x in seen + [x for _, x in out]):
                continue
            out.append((r, d))
        if len(out) >= 3 * n_each:
            break
    return sorted(out, key=lambda x: x[1])


def _screen(args):
    day, rules_d = args
    data, _ = _G
    rules = SimRules(**rules_d); p = EvidenceParams()
    out = []
    for t in data:
        rep = reports_at(data[t][0], data[t][1], day, p, rules)
        if rep is None:
            continue
        wr, _ = rep
        if wr.yes_count >= rules.weekly_min_yes:
            out.append((t, "최악" in wr.channel.quadrant, wr.div.bearish_count >= 2))
    return day, out


def window_fridays(cal: pd.DatetimeIndex, start: pd.Timestamp, h: int) -> tuple[pd.DatetimeIndex, list[pd.Timestamp]]:
    days = cal[cal >= start][: h + 1]
    return days, [d for i, d in enumerate(days) if d.weekday() == 4 and i < len(days) - 10]


# ---------------------------------------------------------------------------
# 가린 차트
# ---------------------------------------------------------------------------

def _panel(ax, rep, bars: int, base: float, title: str, unit: str, volume: bool = False, levels: dict | None = None) -> None:
    d = rep.df.tail(bars); off = len(rep.df) - len(d); x = np.arange(len(d))
    up = (d["Close"] >= d["Open"]).to_numpy()
    col = np.where(up, "#d62728", "#1f77b4")
    ax.bar(x, (d["Close"] - d["Open"]).abs().clip(lower=1e-9), bottom=d[["Open", "Close"]].min(axis=1), width=0.65, color=col)
    ax.vlines(x, d["Low"], d["High"], color=col, linewidth=0.7)
    for n, c in ((20, "red"), (50, "orange"), (100, "deepskyblue"), (200, "purple")):
        ax.plot(x, d[f"EMA{n}"].to_numpy(), color=c, linewidth=0.9)
    if "SMA365" in d:
        ax.plot(x, d["SMA365"].to_numpy(), color="black", linewidth=0.9)
    for ch, c in zip(rep.channel.channels, ("#b8860b", "#2e8b57")):
        ser = ch.series(len(rep.df)).iloc[off:]
        for nm, ls in (("upper", "-"), ("mid", ":"), ("lower", "-")):
            ax.plot(x, ser[nm].to_numpy(), color=c, linewidth=0.8, linestyle=ls)
    lo, hi = float(d["Low"].min()), float(d["High"].max())
    if rep.fib:
        for r, v in rep.fib.levels.items():
            if isinstance(r, float) and r in (0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618) and lo * 0.9 < v < hi * 1.1:
                ax.axhline(v, color="gray", linewidth=0.5, linestyle="--"); ax.text(len(d) - 1, v, f"{r}", fontsize=6, color="gray", va="bottom")
    for b in rep.boxes:
        if b.end >= off:
            ax.axhspan(b.bottom, b.top, xmin=max(0, (b.start - off)) / len(d), xmax=1, color="green", alpha=0.08)
    for f in rep.fvgs:
        if f.fill < 1 and f.pos >= off:
            ax.axhspan(f.bottom, f.top, xmin=(f.pos - off) / len(d), xmax=1, color="orange" if f.bullish else "gray", alpha=0.15)
    sw = rep.swings[rep.swings.pos >= off]
    ax.scatter(sw[sw.kind == "H"].pos - off, sw[sw.kind == "H"].price, marker="v", color="#ff7f0e", s=14)
    ax.scatter(sw[sw.kind == "L"].pos - off, sw[sw.kind == "L"].price, marker="^", color="#2ca02c", s=14)
    if levels:
        for nm, c in (("stop", "red"), ("second", "blue"), ("target1", "green")):
            v = levels.get(nm)
            if v and lo * 0.85 < v < hi * 1.25:
                ax.axhline(v, color=c, linewidth=1.0, linestyle="-.")
    ax.set_yscale("log")
    y0, y1 = ax.get_ylim()
    ticks = np.geomspace(y0, y1, 9)[1:-1]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{v / base * 100:.0f}" for v in ticks], fontsize=6); ax.minorticks_off()
    step = max(1, len(d) // 8)
    ax.set_xticks(x[::step]); ax.set_xticklabels([f"-{len(d) - 1 - i}{unit}" for i in x[::step]], fontsize=6)
    ax.set_title(title, fontsize=8)
    ax2 = ax.twinx(); ax2.plot(x, d["RSI"].to_numpy(), color="#9467bd", linewidth=0.7, alpha=0.7); ax2.set_ylim(0, 300); ax2.set_yticks([])
    for v in (30, 70):
        ax2.axhline(v, color="#9467bd", linewidth=0.3, alpha=0.5)
    if volume:
        ax3 = ax.twinx(); ax3.bar(x, d["Volume"].to_numpy(), width=0.65, color=col, alpha=0.25); ax3.set_ylim(0, float(d["Volume"].max()) * 5); ax3.set_yticks([])


def render(cid: str, wr, dr, idx_txt: str, rules: SimRules, out_png: Path) -> dict:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = float(dr.df["Close"].iloc[-1])
    pb = float(wr.df["Low"].iloc[int(wr.fib.hi_pos):].min()) if wr.fib is not None else None
    lv = entry_levels(wr, base, rules, pb)
    n = lambda v: None if v is None else round(v / base * 100, 1)
    levels = {} if lv is None else {"stop": lv[0], "second": lv[1], "target1": lv[2], "final": lv[3]}
    plan = ("금요일 종가(=100) 진입 가정 레벨 없음(근거선 −20% 초과 또는 목표 없음)" if lv is None else
            f"금요일 종가(=100) 진입 가정: 2차(파랑) {n(lv[1]) or '-'} · 손절(빨강) {n(lv[0])} · 1:1(초록) {n(lv[2])} · 최종 {n(lv[3])}")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    _panel(axes[0], wr, 160, base, f"주봉 160 — 예 {wr.yes_count}/5 · {wr.ma.alignment} · 채널 접촉 {wr.channel.touch or '-'} · {wr.channel.quadrant} · 피보 {wr.fib.kind if wr.fib else '-'} {wr.fib.zone if wr.fib else '-'}", "w")
    _panel(axes[1], dr, 200, base, f"일봉 200 — 예 {dr.yes_count}/5 · {dr.ma.alignment} · 캔들 {','.join(dr.candle.patterns) or '-'} · RSI {dr.div.rsi:.0f}", "d", levels=levels)
    _panel(axes[2], dr, 45, base, "일봉 최근 45 (거래량 포함)", "d", volume=True, levels=levels)
    div = ("상승 다이버전스 " if wr.div.bullish else "") + (f"하락 다이버전스 {wr.div.bearish_count}회 " if wr.div.bearish_count else "")
    fig.suptitle(f"[{cid}]  주봉 캔들 {','.join(wr.candle.patterns) or '-'} · 주봉 RSI {wr.div.rsi:.0f} {div}· 주봉 MA 접촉 {','.join(wr.ma.touching) or '-'} · 매물대 {wr.supply.why[:36]}\n"
                 f"{plan}\n{idx_txt}", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out_png, dpi=76); plt.close(fig)
    return {k: n(v) for k, v in levels.items()}


def cmd_pool(cfg: dict) -> None:
    global _G
    from evidence_manual_bt import setup_font
    setup_font()
    out = Path(cfg["out_dir"]); (out / "charts").mkdir(parents=True, exist_ok=True); (out / "_key").mkdir(exist_ok=True)
    h = int(cfg.get("horizon_days", 75)); k = int(cfg.get("k", 5)); seed = int(cfg.get("seed", 101))
    rules = SimRules(horizon_days=h, index_gate="none")
    prices = _load(); qqq = prices.pop("QQQ"); prices.pop("SPY")
    cal = pd.to_datetime(qqq.index)
    p = EvidenceParams()
    data = precompute(prices, p); idx = precompute({"QQQ": qqq}, p)["QQQ"]
    hi = (cal[-1] - pd.Timedelta(days=int((h + 140) * 1.5))).date().isoformat()   # 사건 연구용 뒤 26주가 남게
    wins = choose_windows(qqq, int(cfg.get("n_each", 2)), seed, hi)
    print("windows", [(r, d.date().isoformat()) for r, d in wins], file=sys.stderr)
    jobs = []
    for _, start in wins:
        _, frs = window_fridays(cal, start, h)
        jobs += [(f, asdict(rules)) for f in frs]
    _G = (data, idx)
    screen = {}
    with ProcessPoolExecutor(max_workers=int(cfg.get("workers", 4)), mp_context=mp.get_context("fork")) as ex:
        for day, lst in ex.map(_screen, jobs):
            screen[day] = lst
            print("screen", day.date(), len(lst), file=sys.stderr)
    rng = np.random.default_rng(seed)
    items = []
    for reg, start in wins:
        _, frs = window_fridays(cal, start, h)
        for f in frs:
            el = sorted(t for t, worst, bear2 in screen[f] if not worst and not bear2)
            for t in (rng.choice(el, size=min(k, len(el)), replace=False) if el else []):
                items.append({"window": start.date().isoformat(), "regime": reg, "friday": f.date().isoformat(), "ticker": str(t), "n_eligible": len(el)})
    order = rng.permutation(len(items))
    key = {}
    for n, j in enumerate(order, 1):
        it = items[j]; cid = f"c{n:03d}"
        f = pd.Timestamp(it["friday"])
        wr, dr = reports_at(data[it["ticker"]][0], data[it["ticker"]][1], f, p, rules)
        qw, _ = reports_at(idx[0], idx[1], f, p, rules)
        qc = idx[0]["Close"].loc[:f]
        idx_txt = (f"지수(QQQ) 주봉: {qw.ma.alignment} · {qw.channel.quadrant} · 20주선 {'위' if qw.ma.above_ema20 else '아래'} · RSI {qw.div.rsi:.0f} · "
                   f"52주 고점 대비 {float(qc.iloc[-1] / qc.tail(252).max() - 1):+.0%} · 4주 {float(qc.iloc[-1] / qc.iloc[-21] - 1):+.0%}")
        lv = render(cid, wr, dr, idx_txt, rules, out / "charts" / f"{cid}.png")
        key[cid] = {**it, "levels_norm": lv, "weekly_yes": wr.yes_count, "daily_yes": dr.yes_count}
        if n % 25 == 0:
            print("render", n, len(items), file=sys.stderr)
    json.dump({"cfg": cfg, "rules": asdict(rules), "windows": [(r, d.date().isoformat()) for r, d in wins],
               "screen": {d.date().isoformat(): v for d, v in screen.items()}, "key": key}, open(out / "_key" / "key.json", "w"), ensure_ascii=False, indent=1)
    print(f"pool {len(items)} charts → {out}/charts", file=sys.stderr)


# ---------------------------------------------------------------------------
# 정산
# ---------------------------------------------------------------------------

def _load_dec(paths: list[str]) -> dict:
    dec = {}
    for pth in paths:
        for cid, v in json.load(open(pth)).items():
            dec.setdefault(cid, v)
    return dec


def cmd_agree(a: str, b: str) -> None:
    da, db = json.load(open(a)), json.load(open(b))
    common = sorted(set(da) & set(db))
    same = sum(1 for c in common if da[c]["v"] == db[c]["v"])
    ca = np.array([da[c].get("conv", 3) for c in common], float); cb = np.array([db[c].get("conv", 3) for c in common], float)
    ba = sum(1 for c in common if da[c]["v"] == "buy"); bb = sum(1 for c in common if db[c]["v"] == "buy"); both = sum(1 for c in common if da[c]["v"] == db[c]["v"] == "buy")
    pe = (ba / len(common)) * (bb / len(common)) + (1 - ba / len(common)) * (1 - bb / len(common))
    kappa = (same / len(common) - pe) / (1 - pe) if pe < 1 else float("nan")
    print(json.dumps({"공통": len(common), "같은 판정": same, "일치율": round(same / len(common), 3), "kappa": round(kappa, 3),
                      "A buy": ba, "B buy": bb, "둘 다 buy": both, "확신 상관": round(float(np.corrcoef(ca, cb)[0, 1]), 3)}, ensure_ascii=False))


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.1%}" if pct else f"{v:.2f}"


def cmd_settle(cfg: dict) -> None:
    out_dir = Path(cfg["out_dir"])
    K = json.load(open(out_dir / "_key" / "key.json"))
    key = K["key"]; h = int(K["rules"]["horizon_days"])
    dec = _load_dec(cfg["decisions"])
    missing = [c for c in key if c not in dec]
    if missing:
        print(f"# 판정 없는 차트 {len(missing)}개 → skip 처리: {missing[:10]}", file=sys.stderr)
    prices = _load(); qqq = prices.pop("QQQ"); spy = prices.pop("SPY")
    cal = pd.to_datetime(qqq.index); p = EvidenceParams()
    need = sorted({v["ticker"] for v in key.values()})
    full_uni = bool(cfg.get("full_auto", True))
    data = precompute(prices if full_uni else {t: prices[t] for t in need}, p)
    idx = precompute({"QQQ": qqq}, p)["QQQ"]
    screen = {pd.Timestamp(d): [tuple(x) for x in v] for d, v in K["screen"].items()}
    n_pl = int(cfg.get("placebo_seeds", 100))
    conv_min = int(cfg.get("conv_min", 0))

    def ok(cid):
        d = dec.get(cid)
        return bool(d) and d["v"] == "buy" and int(d.get("conv", 3)) >= conv_min

    sizings = {"책(10%+10%, ≤5종목)": dict(position_frac=0.2, max_positions=5), "집중(17%+17%, ≤3종목)": dict(position_frac=0.34, max_positions=3)}
    rows = []; ev_rows = []
    for reg, start_iso in K["windows"]:
        start = pd.Timestamp(start_iso)
        its = {c: v for c, v in key.items() if v["window"] == start_iso}
        pool: dict[pd.Timestamp, list[str]] = {}; appr: dict[pd.Timestamp, list[str]] = {}
        for c, v in its.items():
            f = pd.Timestamp(v["friday"])
            pool.setdefault(f, []).append(v["ticker"])
            if ok(c):
                appr.setdefault(f, []).append(v["ticker"])
        row = {"레짐": reg, "시작": start_iso, "풀": len(its), "승인": sum(len(v) for v in appr.values())}
        for sname, sz in sizings.items():
            rules = SimRules(horizon_days=h, index_gate="none", **sz)
            cache = dict(screen)
            pick_h = lambda day, cands, free: [t for t in cands if t in appr.get(day, ())]
            pick_a = lambda day, cands, free: [t for t in cands if t in pool.get(day, ())]
            rh = run_sim(data, cal, start, rules, picker=pick_h, screen_cache=cache, index_data=idx)
            ra = run_sim(data, cal, start, rules, picker=pick_a, screen_cache=cache, index_data=idx)
            pls = []
            for s in range(n_pl):
                rng = np.random.default_rng(5000 + s)
                def pick_p(day, cands, free, rng=rng):
                    pl = [t for t in cands if t in pool.get(day, ())]
                    kk = min(len(pl), len(appr.get(day, ())))
                    return list(rng.choice(pl, size=kk, replace=False)) if kk else []
                pls.append(run_sim(data, cal, start, rules, picker=pick_p, screen_cache=cache, index_data=idx).stats)
            pr = np.array([x["수익률"] for x in pls])
            end = pd.Timestamp(rh.end)
            row[sname] = {"사람": rh.stats, "기계(풀 전부)": ra.stats, "무작위 평균": float(pr.mean()), "무작위 중앙": float(np.median(pr)),
                         "무작위 MDD": float(np.mean([x["MDD"] for x in pls])), "사람 백분위": float((pr < rh.stats["수익률"]).mean()),
                         "무작위 거래": float(np.mean([x["거래 수"] for x in pls]))}
            if sname.startswith("책"):
                row["종료"] = rh.end; row["QQQ"] = buy_hold(qqq, start, end); row["SPY"] = buy_hold(spy, start, end)
                row["사람 거래"] = [{"티커": tr.ticker, "신호": tr.signal, "진입": tr.entry, "진입가": round(tr.entry_px, 2), "2차": bool(tr.second_done),
                                   "1:1": bool(tr.tp1_done), "손익": round(tr.pnl(), 4), "수익률": round(tr.pnl() / tr.cost, 4),
                                   "청산": [(d, round(px, 2), rs) for d, px, _, rs in tr.exits]} for tr in rh.trades]
                if full_uni:
                    for g in ("none", "worst"):
                        row[f"기계 전체(게이트 {g})"] = run_sim(data, cal, start, replace(rules, index_gate=g), screen_cache=dict(screen), index_data=idx).stats["수익률"]
        # ---- 사건 연구: 풀 전 종목을 슬롯 제한 없이 같은 규칙으로 (뒤 26주 더 보유)
        rules_e = SimRules(horizon_days=h + 130, index_gate="none", position_frac=0.002, max_positions=100000)
        re_ = run_sim(data, cal, start, rules_e, picker=lambda day, cands, free: [t for t in cands if t in pool.get(day, ())],
                      screen_cache=dict(screen), index_data=idx)
        tmap = {(tr.ticker, tr.signal): tr for tr in re_.trades}
        for c, v in its.items():
            f = pd.Timestamp(v["friday"]); d = data[v["ticker"]][0]; qd = idx[0]
            fw = {}
            for wk in (4, 13, 26):
                pos = d.index.searchsorted(f) + wk * 5; qpos = qd.index.searchsorted(f) + wk * 5
                if pos < len(d) and qpos < len(qd):
                    fw[f"fw{wk}"] = float(d["Close"].iloc[pos] / d["Close"].loc[f] - 1) - float(qd["Close"].iloc[qpos] / qd["Close"].loc[f] - 1)
            tr = tmap.get((v["ticker"], v["friday"]))
            dd = dec.get(c, {"v": "skip", "conv": 1})
            ev_rows.append({"id": c, "window": start_iso, "regime": reg, "friday": v["friday"], "ticker": v["ticker"], "v": dd["v"], "conv": int(dd.get("conv", 3)),
                            "why": dd.get("why", ""), "entered": tr is not None, "trade_ret": (tr.pnl() / tr.cost if tr else np.nan),
                            "tp1": (tr.tp1_done if tr else None), **fw})
        rows.append(row)
        b = row["책(10%+10%, ≤5종목)"]
        print(f"done {reg} {start_iso} 사람 {b['사람']['수익률']:+.1%} 기계 {b['기계(풀 전부)']['수익률']:+.1%} 무작위 {b['무작위 평균']:+.1%} QQQ {row['QQQ']:+.1%}", file=sys.stderr)

    ev = pd.DataFrame(ev_rows)
    ev["buy"] = ev.v == "buy"
    # 금요일 층화 순열 검정: 승인 라벨을 같은 금요일 안에서 섞어 (승인 − 거절) 평균 차의 분포
    rng = np.random.default_rng(7)
    def perm(col: str, sub: pd.DataFrame) -> tuple[float, float, int, int]:
        s = sub.dropna(subset=[col])
        if s.buy.sum() == 0 or (~s.buy).sum() == 0:
            return float("nan"), float("nan"), int(s.buy.sum()), int((~s.buy).sum())
        obs = s.loc[s.buy, col].mean() - s.loc[~s.buy, col].mean()
        grp = [g.index.to_numpy() for _, g in s.groupby("friday")]
        b = s.buy.to_numpy().copy(); x = s[col].to_numpy(); pos = {ix: i for i, ix in enumerate(s.index)}
        cnt = 0
        for _ in range(2000):
            bb = b.copy()
            for g in grp:
                ii = [pos[i] for i in g]; bb[ii] = rng.permutation(b[ii])
            if x[bb].mean() - x[~bb].mean() >= obs:
                cnt += 1
        return float(obs), cnt / 2000, int(s.buy.sum()), int((~s.buy).sum())

    cols = [("trade_ret", "거래 수익률(책 규칙, 진입한 것만)"), ("fw4", "4주 뒤 QQQ 대비"), ("fw13", "13주 뒤 QQQ 대비"), ("fw26", "26주 뒤 QQQ 대비")]
    study = {}
    for col, nm in cols:
        obs, pv, nb, ns = perm(col, ev)
        s = ev.dropna(subset=[col])
        study[nm] = {"승인 평균": float(s.loc[s.buy, col].mean()) if nb else None, "거절 평균": float(s.loc[~s.buy, col].mean()) if ns else None,
                     "승인 중앙": float(s.loc[s.buy, col].median()) if nb else None, "거절 중앙": float(s.loc[~s.buy, col].median()) if ns else None,
                     "승인 승률": float((s.loc[s.buy, col] > 0).mean()) if nb else None, "거절 승률": float((s.loc[~s.buy, col] > 0).mean()) if ns else None,
                     "차이": obs, "p(순열, 한쪽)": pv, "n 승인": nb, "n 거절": ns,
                     "확신 순위상관": float(s[[col, "conv"]].corr(method="spearman").iloc[0, 1]) if len(s) > 5 else None,
                     "레짐별 차이": {r: perm(col, ev[ev.regime == r])[0] for r in ev.regime.unique()}}
    by_conv = {int(c): {nm: float(g[col].mean()) for col, nm in cols} | {"n": int(len(g))} for c, g in ev.groupby("conv")}
    res = {"cfg": cfg, "windows": K["windows"], "rows": rows, "study": study, "by_conv": by_conv,
           "entered": {"승인": float(ev.loc[ev.buy, "entered"].mean()) if ev.buy.any() else None, "거절": float(ev.loc[~ev.buy, "entered"].mean())},
           "tp1": {"승인": float(ev.loc[ev.buy & ev.entered, "tp1"].astype(float).mean()) if (ev.buy & ev.entered).any() else None,
                   "거절": float(ev.loc[~ev.buy & ev.entered, "tp1"].astype(float).mean())},
           "events": ev.drop(columns="buy").to_dict("records")}
    json.dump(res, open(cfg["out"], "w"), ensure_ascii=False, indent=1, default=str)
    write_md(res, cfg["report_md"])


def write_md(res: dict, path: str) -> None:
    rows = res["rows"]
    L = ["# 근거중심매매 블라인드 수동 판정 — 정산", "", f"설정 {res['cfg']}", ""]
    for sname in [k for k in rows[0] if k.startswith(("책", "집중"))]:
        L += [f"## 계좌 — {sname}", "", "| 레짐 | 창 | 풀/승인 | 사람 | MDD | 거래·승률 | 기계(풀 전부) | 무작위(같은 수) | 사람 백분위 | QQQ | SPY |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            b = r[sname]; hs = b["사람"]; a = b["기계(풀 전부)"]
            L.append(f"| {r['레짐']} | {r['시작']} → {r['종료']} | {r['풀']}/{r['승인']} | {_fmt(hs['수익률'])} | {_fmt(hs['MDD'])} | {hs['거래 수']}·{'—' if hs['승률'] is None else format(hs['승률'], '.0%')} | "
                     f"{_fmt(a['수익률'])} ({a['거래 수']}건, MDD {_fmt(a['MDD'])}) | {_fmt(b['무작위 평균'])} ({b['무작위 거래']:.1f}건) | {b['사람 백분위']:.0%} | {_fmt(r['QQQ'])} | {_fmt(r['SPY'])} |")
        m = lambda f: float(np.mean([f(r) for r in rows]))
        L.append(f"| **평균** | | | **{_fmt(m(lambda r: r[sname]['사람']['수익률']))}** | {_fmt(m(lambda r: r[sname]['사람']['MDD']))} | | {_fmt(m(lambda r: r[sname]['기계(풀 전부)']['수익률']))} | "
                 f"{_fmt(m(lambda r: r[sname]['무작위 평균']))} | {m(lambda r: r[sname]['사람 백분위']):.0%} | {_fmt(m(lambda r: r['QQQ']))} | {_fmt(m(lambda r: r['SPY']))} |")
        L.append("")
    if "기계 전체(게이트 none)" in rows[0]:
        L += ["기계 전체(풀 제한 없이 엔진 그대로, 책 사이징): " + " · ".join(f"{r['시작']} 게이트 없음 {_fmt(r['기계 전체(게이트 none)'])} / 최악 4분면 게이트 {_fmt(r['기계 전체(게이트 worst)'])}" for r in rows), ""]
    L += ["## 사건 연구 — 풀 전체(슬롯 제한 없음), 승인 vs 거절", "",
          f"진입(일봉 확인 캔들)까지 간 비율: 승인 {_fmt(res['entered']['승인'])} · 거절 {_fmt(res['entered']['거절'])} / 1:1 도달: 승인 {_fmt(res['tp1']['승인'])} · 거절 {_fmt(res['tp1']['거절'])}", "",
          "| 지표 | n 승인/거절 | 승인 평균 | 거절 평균 | 차이 | p(금요일 층화 순열) | 승인 중앙 | 거절 중앙 | 승인 승률 | 거절 승률 | 확신 순위상관 | 레짐별 차이 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, s in res["study"].items():
        L.append(f"| {nm} | {s['n 승인']}/{s['n 거절']} | {_fmt(s['승인 평균'])} | {_fmt(s['거절 평균'])} | {_fmt(s['차이'])} | {_fmt(s['p(순열, 한쪽)'], False)} | {_fmt(s['승인 중앙'])} | {_fmt(s['거절 중앙'])} | "
                 f"{_fmt(s['승인 승률'])} | {_fmt(s['거절 승률'])} | {_fmt(s['확신 순위상관'], False)} | {' · '.join(f'{k} {_fmt(v)}' for k, v in s['레짐별 차이'].items())} |")
    L += ["", "확신 점수별 평균: " + " / ".join(f"{c}점(n={v['n']}): " + ", ".join(f"{k.split('(')[0].strip()} {_fmt(x)}" for k, x in v.items() if k != 'n') for c, v in sorted(res["by_conv"].items())), ""]
    L += ["## 사람 승인 거래 (책 사이징)", ""]
    for r in rows:
        L += [f"### {r['레짐']} {r['시작']} → {r['종료']}", "| 티커 | 신호(금) | 진입 | 진입가 | 2차 | 1:1 | 거래 수익률 | 청산 |", "|---|---|---|---|---|---|---|---|"]
        for t in r["사람 거래"]:
            L.append(f"| {t['티커']} | {t['신호']} | {t['진입']} | {t['진입가']} | {'○' if t['2차'] else '-'} | {'○' if t['1:1'] else '-'} | {t['수익률']:+.1%} | {t['청산']} |")
        L.append("")
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pool":
        cmd_pool(json.loads(sys.argv[2]))
    elif cmd == "settle":
        cmd_settle(json.loads(sys.argv[2]))
    elif cmd == "agree":
        cmd_agree(sys.argv[2], sys.argv[3])
