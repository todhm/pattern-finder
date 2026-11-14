"""급등주 탐색 S7 — 수동 블라인드 판정 도구(사람 눈이 소거 뒤 후보에서 더 나은 종목을 고르는가).

  candidates  표본 날짜 12 개(반기마다 무작위 하루)에서 소거 4 개를 통과한 12-1 모멘텀 상위 16 종목을
              기준일까지만 그린 익명 차트(티커·날짜·가격 수준 숨김)로 저장. 정답표 key.json 은 정산 전 열지 않는다.
  settle      판정 json {"D01": {"buy": ["A", ...], "avoid": ["B", ...]}, ...} → 다음날 시가 매수·63 거래일 보유 정산.

실행: docker compose exec backtester python surge_manual_cli.py candidates '{}'
      docker compose exec backtester python surge_manual_cli.py settle '{"decisions": "/app/sweep_results/surge_manual/decisions.json"}'
"""

from __future__ import annotations

import itertools
import json
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import scores, wide

SR = Path("/app/sweep_results")
FONT = "/app/pattern-finder-cache/fonts/AppleGothic.ttf"


def _load():
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"])
    w = wide(panel); del panel
    return w


def sample_dates(idx: pd.DatetimeIndex, start: str, end: str, hold: int, rng: np.random.Generator) -> list[pd.Timestamp]:
    """반기마다 무작위 거래일 하루. 보유가 end 안에서 끝나는 날만."""
    last = idx[idx <= end][-1]; out = []
    for y, h in itertools.product(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1), (1, 2)):
        lo, hi = pd.Timestamp(y, 1 if h == 1 else 7, 1), pd.Timestamp(y, 6, 30) if h == 1 else pd.Timestamp(y, 12, 31)
        pos = [i for i in np.flatnonzero((idx >= max(lo, pd.Timestamp(start))) & (idx <= hi)) if i + 1 + hold < len(idx) and idx[i + 1 + hold] <= last]
        if pos:
            out.append(idx[int(rng.choice(pos))])
    return out


def pool(w, up, sc, s: int, n: int, thr: float) -> list[str]:
    c = w["C"]; win = w["dv"].iloc[max(0, s - 19): s + 1]; dv20 = win.mean().where(win.count() >= 10)
    ok = w["alive"].iloc[s] & (w["raw_c"].iloc[s] >= 10.0) & (dv20 >= 5e7) & ((c.iloc[s] / c.iloc[s - 20] - 1.0) <= 0.5) & (up.iloc[s] >= thr).fillna(False)
    return list(sc.iloc[s].where(ok).dropna().nlargest(n).index)


def _panel(ax, axv, c: pd.Series, dv: pd.Series | None, title: str, look: int = 252):
    s50, s200 = c.rolling(50, min_periods=30).mean(), c.rolling(200, min_periods=120).mean()
    c, s50, s200 = c.iloc[-look:], s50.iloc[-look:], s200.iloc[-look:]
    base = c.dropna().iloc[0]; x = np.arange(-len(c) + 1, 1)
    ax.plot(x, c / base * 100, color="black", lw=1.1); ax.plot(x, s50 / base * 100, color="tab:blue", lw=0.8); ax.plot(x, s200 / base * 100, color="tab:red", lw=0.8)
    ax.set_yscale("log"); ax.set_title(title, fontsize=11, loc="left"); ax.grid(alpha=0.3, which="both"); ax.tick_params(labelsize=7)
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(ScalarFormatter())
    if dv is not None and axv is not None:
        v = (dv / dv.rolling(50, min_periods=20).mean()).iloc[-look:]
        up = (c.diff() >= 0).to_numpy()
        axv.bar(x, v, color=np.where(up, "tab:green", "tab:red"), width=1.0); axv.axhline(1.0, color="gray", lw=0.5); axv.set_ylim(0, 5); axv.tick_params(labelsize=6)


def cmd_candidates(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT); plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
    out = Path(cfg.get("out", SR / "surge_manual")); out.mkdir(parents=True, exist_ok=True)
    start, end = cfg.get("start", "2018-02-01"), cfg.get("end", "2023-12-31"); hold = int(cfg.get("hold", 63)); n = int(cfg.get("n", 16)); thr = float(cfg.get("thr", 0.51))
    rng = np.random.default_rng(int(cfg.get("seed", 20261112)))
    w = _load(); c = w["C"]; idx = c.index
    r = c.pct_change(fill_method=None); up = (r > 0).where(r.notna()).rolling(252, min_periods=200).mean(); sc = scores(w, "mom12_1")
    qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")["C"]
    dates = sample_dates(idx, start, end, hold, rng); order = rng.permutation(len(dates)); key = {}
    for code_i, di in enumerate(order, 1):
        d = dates[int(di)]; s = idx.get_loc(d); code = f"D{code_i:02d}"
        syms = pool(w, up, sc, s, n, thr); syms = [syms[i] for i in rng.permutation(len(syms))]
        labels = list(string.ascii_uppercase[: len(syms)]); key[code] = {"date": str(d.date()), "symbols": dict(zip(labels, syms))}
        for part, chunk in enumerate([list(zip(labels, syms))[i: i + 8] for i in range(0, len(syms), 8)], 1):
            fig = plt.figure(figsize=(18, 13)); gs = fig.add_gridspec(6, 3, height_ratios=[3, 1] * 3, hspace=0.35, wspace=0.12)
            ax = fig.add_subplot(gs[0, 0]); _panel(ax, None, qq[qq.index <= d].iloc[-460:], None, f"{code} 시장(나스닥100 ETF) · 기준일까지 1년")
            for k, (lab, sym) in enumerate(chunk, 1):
                rr, cc = divmod(k, 3); a = fig.add_subplot(gs[2 * rr, cc]); av = fig.add_subplot(gs[2 * rr + 1, cc], sharex=a)
                _panel(a, av, c[sym].iloc[: s + 1].iloc[-460:], w["dv"][sym].iloc[: s + 1].iloc[-460:], f"{code}-{lab}")
            fig.suptitle(f"{code} 후보 {part}/2 — 검정 종가(시작=100, 로그) · 파랑 50일선 · 빨강 200일선 · 아래 거래대금(50일 평균=1)", fontsize=12, y=0.995)
            fig.savefig(out / f"{code}_{part}.png", dpi=70, bbox_inches="tight"); plt.close(fig)
        print(code, len(syms), file=sys.stderr, flush=True)
    (out / "key.json").write_text(json.dumps({"hold": hold, "n": n, "thr": thr, "codes": key}, ensure_ascii=False, indent=1))
    print(f"{len(key)} 날짜 · 차트 {out}")


def settle_table(w, key: dict, dec: dict, tq: pd.DataFrame, qq: pd.DataFrame, cost: float = 0.005) -> pd.DataFrame:
    idx = w["C"].index; hold = int(key["hold"]); rows = []
    for code, k in sorted(key["codes"].items(), key=lambda kv: kv[1]["date"]):
        s = idx.get_loc(pd.Timestamp(k["date"])); e, x = s + 1, s + 1 + hold
        ret = {lab: float(w["O_ff"][sym].iloc[x] / w["O_ff"][sym].iloc[e] - 1.0) for lab, sym in k["symbols"].items()}
        net = lambda labs: float(np.mean([ret[a] for a in labs]) * 1.0 + (1 + np.mean([ret[a] for a in labs])) * (-cost)) if labs else np.nan  # noqa: E731
        d = dec.get(code, {}); buy, avoid = d.get("buy", []), d.get("avoid", [])
        b = lambda f: float(f["O"].reindex(idx).iloc[x] / f["O"].reindex(idx).iloc[e] - 1.0)  # noqa: E731
        combos = np.array([np.mean([ret[a] for a in cmb]) for cmb in itertools.combinations(ret, len(buy))]) if buy else np.array([np.nan])
        rows.append({"code": code, "date": k["date"], "exit": str(idx[x].date()), "buy": net(buy), "all": net(list(ret)), "avoid": net(avoid), "rest": net([a for a in ret if a not in buy]),
                     "tqqq": b(tq), "qqq": b(qq), "pct": float((np.mean([ret[a] for a in buy]) > combos).mean()) if buy else np.nan,
                     "buy_names": ",".join(f"{k['symbols'][a]}{ret[a]:+.0%}" for a in buy), "avoid_names": ",".join(f"{k['symbols'][a]}{ret[a]:+.0%}" for a in avoid)})
    return pd.DataFrame(rows)


def cmd_settle(cfg):
    out = Path(cfg.get("out", SR / "surge_manual")); key = json.loads((out / "key.json").read_text()); dec = json.loads(Path(cfg["decisions"]).read_text())
    w = _load(); tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet"); qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")
    t = settle_table(w, key, dec, tq, qq); p = lambda v: "-" if pd.isna(v) else f"{v:+.1%}"  # noqa: E731
    L = ["# S7 — 수동 블라인드 판정 정산", "", f"소거 4 개 통과·12-1 모멘텀 상위 {key['n']} 종목 중 사람이 고른 묶음 · 다음날 시가 매수 · {key['hold']} 거래일 보유 · 비용 0.5% · 세전 · 폐지 종목은 마지막 종가", "",
         "| 코드 | 기준일 | 고른 묶음 | 후보 전부 | 안 고른 것 | 피한 묶음 | TQQQ | QQQ | 무작위 백분위 | 고른 종목 | 피한 종목 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.code} | {r.date} | {p(r.buy)} | {p(r.all)} | {p(r.rest)} | {p(r.avoid)} | {p(r.tqqq)} | {p(r.qqq)} | {r.pct:.0%} | {r.buy_names} | {r.avoid_names} |")
    cols = ["buy", "all", "rest", "avoid", "tqqq", "qqq"]; m = t[cols].mean(); md = t[cols].median(); pr = (1 + t[cols]).prod()
    L += [f"| **평균** | | {p(m.buy)} | {p(m['all'])} | {p(m.rest)} | {p(m.avoid)} | {p(m.tqqq)} | {p(m.qqq)} | {t['pct'].mean():.0%} | | |",
          f"| **중앙** | | {p(md.buy)} | {p(md['all'])} | {p(md.rest)} | {p(md.avoid)} | {p(md.tqqq)} | {p(md.qqq)} | {t['pct'].median():.0%} | | |",
          f"| **이어 곱** | | ×{pr.buy:.2f} | ×{pr['all']:.2f} | ×{pr.rest:.2f} | ×{pr.avoid:.2f} | ×{pr.tqqq:.2f} | ×{pr.qqq:.2f} | | | |", "",
          f"고른 묶음이 후보 전부를 이긴 날짜 {int((t.buy > t['all']).sum())}/{len(t)} · TQQQ 를 이긴 날짜 {int((t.buy > t.tqqq).sum())}/{len(t)} · 피한 묶음보다 나은 날짜 {int((t.buy > t.avoid).sum())}/{len(t)}"]
    (out / "settle.md").write_text("\n".join(L)); t.to_json(out / "settle.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    {"candidates": cmd_candidates, "settle": cmd_settle}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
