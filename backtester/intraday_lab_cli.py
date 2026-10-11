"""초단기 전략 vs 장기보유 배치 — ORB(논문 재현·변형) + 오버나이트를 QQQ·TQQQ 에 돌려 JSON/MD.

    docker compose exec backtester python intraday_lab_cli.py '{"start": "2016-01-01", "end": "2023-02-17"}'
    docker compose exec backtester python intraday_lab_cli.py '{"runs": [...], "out": "/tmp/orb.json", "report_md": "/tmp/orb.md"}'

run 키: signal(QQQ), trade(QQQ|TQQQ), start, end, params{stop_mode, atr_frac, target_r, risk_pct, max_leverage,
commission_per_share, commission_pct, tax_rate, allow_short, allow_long}, label
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, timedelta

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.intraday_lab import (
    KNOWN_SPLITS,
    OrbParams,
    alpha_beta,
    curve_summary,
    overnight_curves,
    run_orb,
    split_factor_series,
    yearly_table,
)
from strategy.adapters.risk_metrics import compute_risk_metrics

_BARS: dict[tuple, pd.DataFrame] = {}


def _bars(md, sym: str, start: date, end: date, interval: str) -> pd.DataFrame:
    key = (sym, start, end, interval)
    if key not in _BARS:
        print(f"[orb] fetch {sym} {interval} {start}~{end}", file=sys.stderr)
        df = md.fetch_ohlcv(sym, start, end, interval)
        _BARS[key] = df
    return _BARS[key]


def _daily_close(md, sym, start, end) -> pd.Series:
    d = _bars(md, sym, start, end, "1d")
    s = d["Close"].copy()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def run_one(md, cfg: dict) -> dict:
    sig, trd = cfg.get("signal", "QQQ"), cfg.get("trade", "QQQ")
    start = date.fromisoformat(cfg.get("start", "2016-01-01"))
    end = date.fromisoformat(cfg["end"]) if cfg.get("end") else date.today()
    p = OrbParams(**cfg.get("params", {}))
    sb = _bars(md, sig, start - timedelta(days=30), end, "5m")
    tb = sb if trd == sig else _bars(md, trd, start - timedelta(days=30), end, "5m")
    tb_cut = tb[tb.index >= pd.Timestamp(start, tz="America/New_York")]
    factor = split_factor_series(tb_cut.index.tz_convert("America/New_York").tz_localize(None).normalize().unique(),
                                 KNOWN_SPLITS.get(trd, []))
    res = run_orb(sb[sb.index >= pd.Timestamp(start, tz="America/New_York")], tb_cut if trd != sig else None, p,
                  unadjusted_factor=factor)
    qqq = _daily_close(md, "QQQ", start, end)
    tq = _daily_close(md, "TQQQ", start, end)
    bh_q = qqq / qqq.iloc[0] * p.initial_capital
    bh_t = tq / tq.iloc[0] * p.initial_capital
    ab = alpha_beta(res.equity, qqq)
    tr = res.frame
    out = {
        "label": cfg.get("label", f"ORB {sig}→{trd}"), "signal": sig, "trade": trd, "start": start.isoformat(),
        "end": end.isoformat(), "params": asdict(p),
        "final": float(res.equity.iloc[-1]), "multiple": float(res.equity.iloc[-1] / p.initial_capital),
        "metrics": {k: (v if not hasattr(v, "isoformat") else v.isoformat()) for k, v in res.metrics.items()},
        "alpha_beta": ab,
        "bench": {"QQQ": float(bh_q.iloc[-1] / p.initial_capital), "TQQQ": float(bh_t.iloc[-1] / p.initial_capital),
                  "QQQ_mdd": compute_risk_metrics(bh_q)["mdd"], "TQQQ_mdd": compute_risk_metrics(bh_t)["mdd"]},
        "yearly": yearly_table({"ORB": res.equity, "QQQ": bh_q, "TQQQ": bh_t}).rename_axis("year").reset_index()
                  .to_dict(orient="records"),
        "equity_monthly": {ts.strftime("%Y-%m"): round(float(v)) for ts, v in res.equity.resample("ME").last().items()},
        "trades_by_reason": tr["exit_reason"].value_counts().to_dict() if len(tr) else {},
    }
    return out


def run_overnight(md, cfg: dict) -> dict:
    start = date.fromisoformat(cfg.get("start", "2016-01-01"))
    end = date.fromisoformat(cfg["end"]) if cfg.get("end") else date.today()
    out = {"start": start.isoformat(), "end": end.isoformat(), "symbols": {}}
    for sym in cfg.get("symbols", ["QQQ", "TQQQ"]):
        d = _bars(md, sym, start, end, "1d")
        res = {}
        for label, kw in (("논문식 (수수료 0)", {}), ("토스 0.1% + 양도세 22%", {"commission_pct": 0.001, "tax_rate": 0.22, "tax_deduction": 1_800.0})):
            c = overnight_curves(d, **kw)
            res[label] = {"summary": curve_summary(c).to_dict(orient="records"),
                          "yearly": yearly_table({k: c[k] for k in c.columns}).rename_axis("year").reset_index().to_dict(orient="records")}
        out["symbols"][sym] = res
    return out


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:+.1f}%"


def report_md(result: dict) -> str:
    L = ["# 초단기 전략 vs 장기보유 — ORB(Zarattini & Aziz 2023 재현) + 오버나이트 (자동 생성)", ""]
    runs = result.get("orb", [])
    if runs:
        L += ["## ⑥ 5분 ORB", "", "| 설정 | 기간 | 최종 배수 | CAGR | MDD | 샤프 | 거래 | 승률 | 평균 R | 롱 비중 | 레버리지 캡 | α(연) | β | QQQ 보유 | TQQQ 보유 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            m, ab, b = r["metrics"], r["alpha_beta"], r["bench"]
            L.append(f"| {r['label']} | {r['start']}~{r['end']} | {r['multiple']:.2f}x | {_pct(m.get('cagr'))} | {_pct(m.get('mdd'))} | "
                     f"{m.get('sharpe', 0):.2f} | {m.get('trades', 0)} | {_pct(m.get('win_rate'))} | {m.get('avg_r', 0):+.2f}R | "
                     f"{_pct(m.get('long_share'))} | {_pct(m.get('capped_share'))} | {_pct(ab.get('alpha_ann'))} | {ab.get('beta', float('nan')):.2f} | "
                     f"{b['QQQ']:.2f}x | {b['TQQQ']:.2f}x |")
        L.append("")
        for r in runs:
            L += [f"### {r['label']} — 연도별", "", "| 연도 | ORB | QQQ | TQQQ |", "|---|---|---|---|"]
            for y in r["yearly"]:
                L.append(f"| {y['year']} | {_pct(y.get('ORB'))} | {_pct(y.get('QQQ'))} | {_pct(y.get('TQQQ'))} |")
            L.append("")
    ov = result.get("overnight")
    if ov:
        L += [f"## ⑦ 오버나이트 ({ov['start']}~{ov['end']})", ""]
        for sym, res in ov["symbols"].items():
            for label, blk in res.items():
                L += [f"### {sym} — {label}", "", "| 전략 | 최종 배수 | CAGR | MDD | 샤프 | 최악 연도 |", "|---|---|---|---|---|---|"]
                for s in blk["summary"]:
                    L.append(f"| {s['전략']} | {s['최종 배수']:.2f}x | {_pct(s['CAGR'])} | {_pct(s['MDD'])} | {s['샤프']:.2f} | {_pct(s['최악 연도'])} |")
                L.append("")
    return "\n".join(L)


def main() -> None:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if "analyze" in cfg:
        print(report_md(json.load(open(cfg["analyze"]))))
        return
    md = build_default_market_data()
    runs = cfg.get("runs") or [cfg]
    result = {"orb": [run_one(md, c) for c in runs]}
    if cfg.get("overnight", True):
        result["overnight"] = run_overnight(md, cfg.get("overnight_cfg", {"start": runs[0].get("start", "2016-01-01"),
                                                                            "end": runs[0].get("end")}))
    if cfg.get("out"):
        json.dump(result, open(cfg["out"], "w"), ensure_ascii=False, default=str)
    if cfg.get("report_md"):
        open(cfg["report_md"], "w").write(report_md(result))
        print(f"[orb] report → {cfg['report_md']}", file=sys.stderr)
    if not cfg.get("out") and not cfg.get("report_md"):
        print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
