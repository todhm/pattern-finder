"""근거중심매매 — **종목 선별은 책 그대로, 베팅(비중·분할·익절·손절)만 최적화** (2026-11-13).

책의 주봉 5대 근거 선별(예 ≥ 4, 최악 4분면·하락 다이버전스 2회 제외, 지수 게이트)은 고정하고
  사이징(종목 수×비중) × 2차 분할 × 1:1 익절 비율·본전 스탑 × 최종 목표 × 손절 방식(일봉/주봉 종가/20주선) × 진입(일봉 확인/주봉 신호 다음날)
격자를 **개발 창**(seed 7, 이전 12 창)에서 돌려 고르고, **검증 창**(다른 seed, 개발 창과 ±120일 안 겹침)에서 다시 잰다.
같은 베팅 규칙에 무작위 종목을 넣은 플라세보로 "베팅 효과"와 "선별 효과"를 가른다. 남는 현금을 QQQ 에 두는 겹침 수익도 같이 낸다.

    python evidence_betting_cli.py grid     '{"out_dir": "/app/sweep_results/evidence_betting_2026-11-13", "workers": 7}'
    python evidence_betting_cli.py validate '{"out_dir": "...", "top": 5, "placebo_seeds": 20, "workers": 7}'
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_blind_cli import _load
from strategy.adapters.evidence_backtest import (
    SimRules, buy_hold, classify_regimes, pick_dates, placebo_sel, precompute, reports_at, run_sim,
)
from strategy.adapters.evidence_checklist import EvidenceParams

_G = None

AXES = {
    "sizing": [(5, 0.2), (3, 0.33), (10, 0.1), (20, 0.05)],          # (최대 종목, 종목당 비중) — 합 ≤ 100%, 레버리지 없음
    "second": [True, False],                                           # 2차 분할 / 1차에 전부
    "tp1": [(0.75, True), (0.5, True), (0.0, True), (0.0, False)],     # (1:1 익절 비율, 본전 스탑)
    "final": [True, False],                                            # 최종 목표 청산 / 없음
    "stop": [("daily", False), ("weekly", False), ("none", True)],     # 일봉 종가 근거선 / 주봉 종가 근거선 / 가격 손절 없이 20주선 이탈
    "entry": [True, False],                                            # 일봉 확인 캔들 / 주봉 신호 다음 거래일
}
BOOK = {"sizing": (5, 0.2), "second": True, "tp1": (0.75, True), "final": True, "stop": ("daily", False), "entry": True}


def to_rules(v: dict, h: int) -> SimRules:
    return SimRules(horizon_days=h, max_positions=v["sizing"][0], position_frac=v["sizing"][1], second_buy=v["second"],
                    tp1_frac=v["tp1"][0], breakeven_on_tp1=v["tp1"][1], use_final=v["final"],
                    stop_mode=v["stop"][0], trend_exit_any=v["stop"][1], require_daily_candle=v["entry"])


def label(v: dict) -> str:
    st = {"daily": "일봉손절", "weekly": "주봉손절", "none": "20주선"}[v["stop"][0]]
    tp = f"1:1에 {v['tp1'][0]:.0%}" + ("" if v["tp1"][1] else "·본전X")
    return (f"{v['sizing'][0]}종목×{v['sizing'][1]:.0%} · {'분할' if v['second'] else '한번에'} · {tp} · "
            f"{'최종O' if v['final'] else '최종X'} · {st} · {'일봉확인' if v['entry'] else '주봉진입'}")


def windows(qqq: pd.DataFrame, cfg: dict) -> dict[str, list[tuple[str, str]]]:
    reg = classify_regimes(qqq)
    n = int(cfg.get("n_each", 4)); hi = cfg.get("date_hi", "2026-03-31")
    dev = pick_dates(reg, n, int(cfg.get("dev_seed", 7)), "2006-01-01", hi)
    val: list[tuple[str, pd.Timestamp]] = []
    for s in range(int(cfg.get("val_seed", 23)), int(cfg.get("val_seed", 23)) + 50):
        for r, d in pick_dates(reg, 8, s, "2006-01-01", hi):
            if sum(1 for rr, _ in val if rr == r) >= n:
                continue
            if any(abs((d - x).days) < 120 for _, x in dev + val):
                continue
            val.append((r, d))
        if len(val) >= 3 * n:
            break
    f = lambda L: [(r, d.date().isoformat()) for r, d in sorted(L, key=lambda x: x[1])]
    return {"dev": f(dev), "val": f(val)}


def _screen(day):
    data = _G["data"]; rules = SimRules(); p = EvidenceParams()
    out = []
    for t in data:
        rep = reports_at(data[t][0], data[t][1], day, p, rules)
        if rep is None:
            continue
        wr, _ = rep
        if wr.yes_count >= rules.weekly_min_yes:
            out.append((t, "최악" in wr.channel.quadrant, wr.div.bearish_count >= 2))
    return day, out


def _sim(job):
    vi, v, start_iso, h, pl_seed = job
    g = _G; rules = to_rules(v, h); start = pd.Timestamp(start_iso)
    picker = None if pl_seed is None else placebo_sel(np.random.default_rng(9000 + pl_seed), list(g["data"]))
    r = run_sim(g["data"], g["cal"], start, rules, picker=picker, screen_cache=dict(g["screen"]), index_data=g["idx"])
    eq = r.equity; q = g["qqq"]["Close"].reindex(eq.index).ffill()
    sr = eq.pct_change().fillna(0.0); qr = q.pct_change().fillna(0.0)
    ov = (1 + sr + r.cash.shift(1).fillna(1.0) * qr).cumprod()      # 남는 현금을 QQQ 에 둔 겹침 곡선
    return {"vi": vi, "start": start_iso, "pl": pl_seed, "ret": r.stats["수익률"], "mdd": r.stats["MDD"], "n": r.stats["거래 수"],
            "win": r.stats["승률"], "cash": float(r.cash.mean()), "ret_q": float(ov.iloc[-1] - 1), "mdd_q": float((ov / ov.cummax() - 1).min()),
            "qqq": buy_hold(g["qqq"], start, pd.Timestamp(r.end)), "spy": buy_hold(g["spy"], start, pd.Timestamp(r.end))}


def _setup(cfg: dict) -> dict:
    global _G
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    h = int(cfg.get("horizon_days", 130)); workers = int(cfg.get("workers", 6))
    prices = _load(); qqq = prices.pop("QQQ"); spy = prices.pop("SPY")
    cal = pd.to_datetime(qqq.index); p = EvidenceParams()
    t0 = time.time()
    data = precompute(prices, p); idx = precompute({"QQQ": qqq}, p)["QQQ"]
    print(f"precompute {len(data)} {time.time() - t0:.0f}s", file=sys.stderr)
    W = windows(qqq, cfg)
    print("windows", W, file=sys.stderr)
    _G = {"data": data, "cal": cal, "idx": idx, "qqq": qqq, "spy": spy, "screen": {}}
    sp = out / "screen.pkl"
    screen = pickle.load(open(sp, "rb")) if sp.exists() else {}
    need = []
    for _, s in W["dev"] + W["val"]:
        days = cal[cal >= pd.Timestamp(s)][: h + 1]
        need += [d for i, d in enumerate(days) if d.weekday() == 4 and i < len(days) - 10 and d not in screen]
    need = sorted(set(need))
    if need:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            for k, (day, lst) in enumerate(ex.map(_screen, need, chunksize=4)):
                screen[day] = lst
                if k % 50 == 0:
                    print(f"screen {k}/{len(need)} {time.time() - t0:.0f}s", file=sys.stderr)
        pickle.dump(screen, open(sp, "wb"))
    _G["screen"] = screen
    return {"out": out, "h": h, "workers": workers, "W": W}


def _run(jobs: list, workers: int, tag: str) -> list[dict]:
    res = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        for k, r in enumerate(ex.map(_sim, jobs, chunksize=8)):
            res.append(r)
            if k % 200 == 0:
                print(f"{tag} {k}/{len(jobs)} {time.time() - t0:.0f}s", file=sys.stderr)
    return res


def _agg(rows: list[dict], W: list[tuple[str, str]]) -> dict:
    df = pd.DataFrame(rows); reg = dict((s, r) for r, s in W)
    df["reg"] = df.start.map(reg)
    o = {"ret": float(df.ret.mean()), "mdd": float(df.mdd.mean()), "worst": float(df.ret.min()), "n": float(df.n.mean()), "cash": float(df.cash.mean()),
         "ret_q": float(df.ret_q.mean()), "mdd_q": float(df.mdd_q.mean()), "qqq": float(df.qqq.mean()), "spy": float(df.spy.mean()),
         "beat_qqq": int((df.ret > df.qqq).sum()), "beat_qqq_q": int((df.ret_q > df.qqq).sum()), "windows": int(len(df))}
    for r, gdf in df.groupby("reg"):
        o[r] = float(gdf.ret.mean()); o[r + "_q"] = float(gdf.ret_q.mean()); o[r + "_qqq"] = float(gdf.qqq.mean())
    return o


def cmd_grid(cfg: dict) -> None:
    S = _setup(cfg)
    keys = list(AXES)
    variants = [dict(zip(keys, c)) for c in itertools.product(*AXES.values())]
    if cfg.get("max_variants"):
        variants = variants[: int(cfg["max_variants"])]
    jobs = [(vi, v, s, S["h"], None) for vi, v in enumerate(variants) for _, s in S["W"]["dev"]]
    res = _run(jobs, S["workers"], "grid")
    by = {}
    for r in res:
        by.setdefault(r["vi"], []).append(r)
    table = [{"vi": vi, "v": variants[vi], "label": label(variants[vi]), "book": variants[vi] == BOOK, **_agg(rows, S["W"]["dev"])} for vi, rows in by.items()]
    json.dump({"cfg": cfg, "windows": S["W"], "table": table, "raw": res}, open(S["out"] / "grid.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"grid {len(table)} variants → {S['out']}/grid.json", file=sys.stderr)


def cmd_validate(cfg: dict) -> None:
    S = _setup(cfg)
    G = json.load(open(S["out"] / "grid.json"))
    tab = G["table"]
    for t in tab:
        t["v"] = {k: (tuple(x) if isinstance(x, list) else x) for k, x in t["v"].items()}
    top_n = int(cfg.get("top", 5)); n_pl = int(cfg.get("placebo_seeds", 20))
    picks: list[tuple[str, dict]] = [("책 그대로", BOOK)]
    for k, t in enumerate(sorted(tab, key=lambda t: -t["ret"])[:top_n], 1):
        picks.append((f"수익 {k}위", t["v"]))
    for k, t in enumerate(sorted(tab, key=lambda t: -t["ret_q"])[:2], 1):
        picks.append((f"현금→QQQ {k}위", t["v"]))
    # 축별 평균이 가장 좋은 값만 모은 조합(격자 한 점에 덜 기대는 선택)
    marg = {}
    for ax in AXES:
        vals = {}
        for t in tab:
            vals.setdefault(str(t["v"][ax]), []).append(t["ret"])
        best = max(vals, key=lambda k: np.mean(vals[k]))
        marg[ax] = next(x for x in AXES[ax] if str(x) == best)
    picks.append(("축별 최선 조합", marg))
    seen = []; uniq = []
    for nm, v in picks:
        if v in seen:
            uniq[seen.index(v)] = (uniq[seen.index(v)][0] + " = " + nm, v)
        else:
            seen.append(v); uniq.append((nm, v))
    jobs = []
    for pi, (_, v) in enumerate(uniq):
        for part in ("dev", "val"):
            for _, s in S["W"][part]:
                jobs.append((pi, v, s, S["h"], None))
                jobs += [(pi, v, s, S["h"], k) for k in range(n_pl)]
    res = _run(jobs, S["workers"], "validate")
    dev_s = {s for _, s in S["W"]["dev"]}
    rows = []
    for pi, (nm, v) in enumerate(uniq):
        row = {"name": nm, "label": label(v), "v": v}
        for part in ("dev", "val"):
            ws = S["W"][part]; inpart = lambda r: (r["start"] in dev_s) == (part == "dev")
            real = [r for r in res if r["vi"] == pi and r["pl"] is None and inpart(r)]
            row[part] = _agg(real, ws)
            pl_means = []; pl_q = []; beat = 0
            for k in range(n_pl):
                pr = [r for r in res if r["vi"] == pi and r["pl"] == k and inpart(r)]
                pl_means.append(np.mean([r["ret"] for r in pr])); pl_q.append(np.mean([r["ret_q"] for r in pr]))
            for r in real:
                pw = [x["ret"] for x in res if x["vi"] == pi and x["pl"] is not None and x["start"] == r["start"]]
                beat += int(r["ret"] > np.mean(pw))
            row[part] |= {"placebo": float(np.mean(pl_means)), "placebo_q": float(np.mean(pl_q)),
                          "pct": float((np.array(pl_means) < row[part]["ret"]).mean()), "beat_placebo": beat,
                          "per_window": [{"start": r["start"], "ret": r["ret"], "ret_q": r["ret_q"], "qqq": r["qqq"], "mdd": r["mdd"], "n": r["n"]} for r in sorted(real, key=lambda r: r["start"])]}
        rows.append(row)
    out = {"cfg": cfg, "windows": S["W"], "marginal_best": marg, "rows": rows}
    json.dump(out, open(S["out"] / "validate.json", "w"), ensure_ascii=False, indent=1, default=str)
    write_md(G, out, S["out"] / "report.md")


def write_md(G: dict, V: dict, path: Path) -> None:
    f = lambda x: "—" if x is None else f"{x:+.1%}"
    tab = G["table"]
    L = ["# 근거중심매매 — 베팅만 최적화 (선별은 책 그대로)", "",
         f"개발 창 {[s for _, s in G['windows']['dev']]}", f"검증 창 {[s for _, s in G['windows']['val']]}", "",
         f"격자 {len(tab)} 조합 × 개발 {len(G['windows']['dev'])} 창. QQQ 개발 평균 {f(tab[0]['qqq'])}.", "",
         "## 1. 개발 창 — 축별 평균 (다른 축 전부 평균)", "", "| 축 | 값 | 평균 수익 | 평균 MDD | 현금→QQQ 수익 | 평균 현금 비중 |", "|---|---|---|---|---|---|"]
    names = {"sizing": "사이징", "second": "2차 분할", "tp1": "1:1 익절(비율, 본전)", "final": "최종 목표", "stop": "손절", "entry": "진입"}
    for ax in AXES:
        for val in AXES[ax]:
            sub = [t for t in tab if str(t["v"][ax]) in (str(val), str(list(val)) if isinstance(val, tuple) else str(val))]
            if sub:
                L.append(f"| {names[ax]} | {val}{' ← 책' if BOOK[ax] == val else ''} | {f(np.mean([t['ret'] for t in sub]))} | {f(np.mean([t['mdd'] for t in sub]))} | "
                         f"{f(np.mean([t['ret_q'] for t in sub]))} | {np.mean([t['cash'] for t in sub]):.0%} |")
    L += ["", f"QQQ 를 이긴 조합(개발 평균): {sum(1 for t in tab if t['ret'] > t['qqq'])}/{len(tab)} · 현금→QQQ 겹침으로 이긴 조합: {sum(1 for t in tab if t['ret_q'] > t['qqq'])}/{len(tab)}", "",
          "## 2. 개발 창 상위 10 + 책", "", "| 순위 | 조합 | 수익 | MDD | 최악 창 | 거래/창 | 현금 | 현금→QQQ | QQQ 이긴 창 | 상승장 | 평범 | 하락장 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    srt = sorted(tab, key=lambda t: -t["ret"])
    for k, t in enumerate(srt[:10] + [t for t in tab if t["book"]], 1):
        rk = "책" if t["book"] and k > 10 else str(srt.index(t) + 1)
        L.append(f"| {rk} | {t['label']} | {f(t['ret'])} | {f(t['mdd'])} | {f(t['worst'])} | {t['n']:.1f} | {t['cash']:.0%} | {f(t['ret_q'])} | {t['beat_qqq']}/{t['windows']} | "
                 f"{f(t.get('상승장'))} | {f(t.get('평범'))} | {f(t.get('하락장'))} |")
    L += ["", "## 3. 검증 — 고른 조합을 새 창에서 (플라세보 = 같은 베팅·무작위 종목)", "",
          "| 조합 | 구간 | 수익 | MDD | 최악 창 | QQQ | QQQ 이긴 창 | 플라세보 | 플라세보 백분위 | 현금→QQQ | (플라세보) | 겹침으로 QQQ 이긴 창 | 상승장/평범/하락장 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in V["rows"]:
        for part, nm in (("dev", "개발"), ("val", "검증")):
            a = r[part]
            L.append(f"| {r['name']}<br>{r['label']} | {nm} | {f(a['ret'])} | {f(a['mdd'])} | {f(a['worst'])} | {f(a['qqq'])} | {a['beat_qqq']}/{a['windows']} | {f(a['placebo'])} | {a['pct']:.0%} | "
                     f"{f(a['ret_q'])} | {f(a['placebo_q'])} | {a['beat_qqq_q']}/{a['windows']} | {f(a.get('상승장'))}/{f(a.get('평범'))}/{f(a.get('하락장'))} |")
    L += ["", "## 4. 검증 창별", ""]
    for r in V["rows"]:
        L.append(f"- **{r['name']}**: " + " · ".join(f"{w['start']} {f(w['ret'])} (QQQ {f(w['qqq'])})" for w in r["val"]["per_window"]))
    path.write_text("\n".join(L))


if __name__ == "__main__":
    {"grid": cmd_grid, "validate": cmd_validate}[sys.argv[1]](json.loads(sys.argv[2]))
