"""«The Lifecycle Trade» 계좌 시뮬 CLI — 배치 JSON 의 거래를 $100K 계좌에 책 8장 자금 관리로 넣으면 얼마가 되나.

    docker compose exec backtester python lifecycle_portfolio_cli.py \\
        '{"cohort": "/app/sweep_results/lifecycle_cohort_2016_2023.json", "start": "2016-01-04", "end": "2026-10-10",
          "out": "/app/sweep_results/lifecycle_portfolio_2016_2023.json", "report_md": "/app/sweep_results/lifecycle_portfolio_2016_2023.md",
          "png": "/app/sweep_results/lifecycle_imgs/portfolio_equity.png"}'

시나리오(열) × 매도 규칙(행) 매트릭스 + SPY 보유 대조 + 연도별 + 상위 기여 거래를 마크다운으로 쓴다.
"""

from __future__ import annotations

import json
import sys
from datetime import date

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.lifecycle_portfolio import (
    PortfolioConfig,
    align,
    benchmark_metrics,
    benchmark_series,
    hold_trades,
    run_portfolio,
    trades_from_records,
)
from strategy.adapters.lifecycle_trade import RULES, fetch_many

SCENARIOS: dict[str, PortfolioConfig] = {
    "A 필터 없음": PortfolioConfig(liquidity_usd=None, market_filter=False, one_recent_ipo=False),
    "B 유동성 $20M": PortfolioConfig(liquidity_usd=20e6, market_filter=False, one_recent_ipo=False),
    "C 책 그대로": PortfolioConfig(),
    "D 책 그대로·첫 성숙 베이스만": PortfolioConfig(kinds=("first mature base",)),
    "E 책 그대로·풀 사이즈 20%": PortfolioConfig(position_frac=0.20, max_positions=5),
}
SCENARIO_DESC = {
    "A 필터 없음": "4매수지점 전부, 유동성·시장 필터 없음, 10%×10종목",
    "B 유동성 $20M": "A + 진입 전일 20일 거래대금 ≥ $20M",
    "C 책 그대로": "B + SPY>200일선 + 상장 1년 이내 동시 1종목 (8장·10장)",
    "D 책 그대로·첫 성숙 베이스만": "C 에서 첫 성숙 베이스 돌파만 매수",
    "E 책 그대로·풀 사이즈 20%": "C 에서 포지션 20%×5종목 (Liquidity Matters 급 종목 가정)",
}


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _table(d: pd.DataFrame) -> str:
    cols = [str(c) for c in d.columns]
    rows = [["" if v is None else str(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def _fmt(m: dict) -> dict:
    o = {}
    for k, v in m.items():
        if isinstance(v, float):
            if k in ("최종 자산",):
                o[k] = f"${v / 1e3:,.0f}K"
            elif k.endswith("$K"):
                o[k] = f"{v:,.0f}"
            elif k in ("거래 n",):
                o[k] = f"{v:.0f}"
            else:
                o[k] = "—" if pd.isna(v) else f"{v:+.1%}"
        else:
            o[k] = str(v)
    return o


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    with open(cfg["cohort"]) as f:
        data = json.load(f)
    results = data["results"]
    start = date.fromisoformat(cfg.get("start", "2016-01-04"))
    end = date.fromisoformat(cfg.get("end", data["config"].get("as_of", date.today().isoformat())))
    capital = float(cfg.get("capital", 100_000))
    md = build_default_market_data()
    spy = benchmark_series(md, cfg.get("benchmark", "SPY"), start, end, _log)
    cal = spy.index[(spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))]
    syms = sorted({r["symbol"] for r in results if r["trades"]})
    _log(f"거래 종목 {len(syms)} 일봉 로드")
    frames, errors = fetch_many(md, syms, start, end, int(cfg.get("workers", 4)), _log)
    _log(f"일봉 {len(frames)}/{len(syms)} (실패 {len(errors)})")
    closes, dvol = align(frames, cal)
    bench = benchmark_metrics(spy, cal, capital)

    rules = list(cfg.get("rules", RULES)) + ["보유 252일"]
    base_trades = {r: trades_from_records(results, r) for r in RULES}
    base_trades["보유 252일"] = hold_trades(base_trades["40-Week"], closes, 252)
    grid: list[dict] = []
    runs: dict[tuple[str, str], object] = {}
    for sc_name, sc in SCENARIOS.items():
        sc = PortfolioConfig(**{**sc.__dict__, "capital": capital})
        for rule in rules:
            res = run_portfolio(base_trades[rule], closes, dvol, spy, sc, rule)
            runs[(sc_name, rule)] = res
            m = res.metrics(spy)
            grid.append({"시나리오": sc_name, "규칙": rule, **m})
            _log(f"  {sc_name} / {rule}: 최종 ${m['최종 자산'] / 1e3:,.0f}K CAGR {m['CAGR']:+.1%} MDD {m['MDD']:+.0%} 거래 {m['거래 n']}")
    g = pd.DataFrame(grid)

    lines = [f"# 계좌 시뮬 — $100K 로 규칙을 그대로 돌리면 ({start} ~ {end})", "",
             f"코호트: `{cfg['cohort']}` · 종목 {len(results)} · 거래 종목 {len(frames)} · 포지션 사이즈 = 계좌의 10%(책: 손실 1% ÷ 손절 10%) · 레버리지 없음 · 체결가는 거래 시뮬과 동일(시가 ±0.5%)", "",
             "## SPY 그냥 보유 (대조)", "", _table(pd.DataFrame([_fmt(bench)])), "", "## 시나리오", "",
             _table(pd.DataFrame([{"시나리오": k, "내용": v} for k, v in SCENARIO_DESC.items()])), ""]
    for sc_name in SCENARIOS:
        sub = g[g["시나리오"] == sc_name].drop(columns=["시나리오", "SPY 총수익", "SPY CAGR"])
        lines += [f"## {sc_name}", "", _table(pd.DataFrame([_fmt(r) for r in sub.to_dict("records")])), ""]
    piv = g.pivot(index="규칙", columns="시나리오", values="CAGR").reindex(rules)
    lines += ["## CAGR 한눈에 (행: 매도 규칙, 열: 시나리오)", "", _table(piv.map(lambda x: f"{x:+.1%}").reset_index()), "",
              f"SPY CAGR {bench['CAGR']:+.1%} · MDD {bench['MDD']:+.0%}", ""]
    pivf = g.pivot(index="규칙", columns="시나리오", values="최종 자산").reindex(rules)
    lines += ["## 최종 자산 $K 한눈에", "", _table(pivf.map(lambda x: f"{x / 1e3:,.0f}").reset_index()), ""]
    for key in (("C 책 그대로", "40-Week"), ("C 책 그대로", "Midterm"), ("D 책 그대로·첫 성숙 베이스만", "40-Week"), ("A 필터 없음", "40-Week")):
        res = runs[key]
        y = res.yearly().reset_index()
        y["연초→연말 수익"] = y["연초→연말 수익"].map(lambda x: f"{x:+.1%}")
        y["연말 자산 $K"] = y["연말 자산 $K"].map(lambda x: f"{x:,.0f}")
        y["진입 승률"] = y["진입 승률"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}")
        lines += [f"## 연도별 — {key[0]} / {key[1]}", "", _table(y), "",
                  "건너뛴 진입: " + ", ".join(f"{k} {v}" for k, v in sorted(res.skipped.items(), key=lambda kv: -kv[1])), ""]
        top = res.trades.nlargest(8, "pnl")[["symbol", "kind", "entry_date", "exit_date", "pct", "pnl"]].copy()
        top["pct"] = top["pct"].map(lambda x: f"{x:+.0%}")
        top["pnl"] = top["pnl"].map(lambda x: f"${x / 1e3:,.0f}K")
        bot = res.trades.nsmallest(5, "pnl")[["symbol", "kind", "entry_date", "exit_date", "pct", "pnl"]].copy()
        bot["pct"] = bot["pct"].map(lambda x: f"{x:+.0%}")
        bot["pnl"] = bot["pnl"].map(lambda x: f"${x / 1e3:,.0f}K")
        lines += ["상위 기여 8:", "", _table(top), "", "하위 5:", "", _table(bot), ""]
    report = "\n".join(lines)
    print(report)
    if cfg.get("report_md"):
        with open(cfg["report_md"], "w") as f:
            f.write(report)
    if cfg.get("out"):
        curves = {f"{k[0]} / {k[1]}": {d_.date().isoformat(): round(float(v), 2) for d_, v in r.equity.iloc[::5].items()}
                  for k, r in runs.items()}
        with open(cfg["out"], "w") as f:
            json.dump({"config": cfg, "benchmark": bench, "grid": grid, "curves": curves,
                       "spy": {d_.date().isoformat(): round(float(v), 2) for d_, v in (spy.reindex(cal).ffill() / spy.reindex(cal).ffill().iloc[0] * capital).iloc[::5].items()}},
                      f, ensure_ascii=False, indent=1, default=str)
    if cfg.get("png"):
        _plot(runs, spy.reindex(cal).ffill() / spy.reindex(cal).ffill().iloc[0] * capital, cfg["png"])


def _plot(runs: dict, spy_eq: pd.Series, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from pathlib import Path
    fp = Path("/app/pattern-finder-cache/fonts/AppleGothic.ttf")
    if fp.exists():
        fm.fontManager.addfont(str(fp))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(fp)).get_name()
    plt.rcParams["text.parse_math"] = False
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(spy_eq.index, spy_eq.values / 1e3, color="gray", lw=1.5, label="SPY 그냥 보유")
    styles = {("A 필터 없음", "40-Week"): ("tab:red", "A 필터 없음 / 40-Week"),
              ("B 유동성 $20M", "40-Week"): ("tab:orange", "B 유동성만 / 40-Week"),
              ("C 책 그대로", "40-Week"): ("tab:blue", "C 책 그대로 / 40-Week"),
              ("C 책 그대로", "Midterm"): ("tab:green", "C 책 그대로 / Midterm"),
              ("D 책 그대로·첫 성숙 베이스만", "40-Week"): ("tab:purple", "D 첫 성숙 베이스만 / 40-Week")}
    for k, (c, lab) in styles.items():
        if k in runs:
            e = runs[k].equity
            ax.plot(e.index, e.values / 1e3, color=c, lw=1.6, label=f"{lab} (끝 ${e.iloc[-1] / 1e3:,.0f}K)")
    ax.axhline(100, color="k", lw=0.6, ls=":")
    ax.set_ylabel("계좌 가치 ($K)")
    ax.set_title("$100K 로 라이프사이클 규칙을 그대로 돌렸을 때 — 포지션 10%, 손절 -10%")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
