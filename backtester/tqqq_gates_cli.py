"""TQQQ 넘기 관문 판정 CLI — 목표 비중 전략 하나를 §1 창 묶음으로 돌려 G1~G6 판정 (프로토콜 §2-3).

    docker compose exec backtester python tqqq_gates_cli.py '{"strategy": "hold", "out": "/app/sweep_results/tqqq_gates_hold.json"}'
    docker compose exec backtester python tqqq_gates_cli.py '{"strategy": "switch", "dd": 0.18, "n": 200, "reenter": 20, "placebo_seeds": 30, "n_trials": 0}'

strategy: hold | switch(`deep_correction_filter(QQQ)` — dd·n·reenter·off_fraction) | target_parquet(열 하나짜리 parquet 경로 `target_path`).
edition 기본 dev. **sealed 는 명시해야 돌고, 호출 쪽이 진행 장부에 열람을 적는다.**
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.tqqq_gates import (
    DIVIDEND_YIELDS, GATE_COSTS, GATE_COSTS_KR, gate_verdicts, load_baseline, placebo_medians, sealed_verdicts, verdict_markdown, window_markdown, window_table,
)

PRICES = "/app/sweep_results/tqqq_qqq_spliced_1985_2026-10-24.parquet"
BASELINE = "/app/sweep_results/tqqq_baseline_windows_2026-11-10.csv"
IRX = "/app/sweep_results/irx_1985_2026-10-24.parquet"


def load_cash_rate() -> pd.Series | None:
    p = Path(IRX)
    if p.exists():
        return pd.read_parquet(p).iloc[:, 0]
    try:
        from data.adapters.composed_market_data import build_default_market_data
        from strategy.adapters.leveraged_etf_data import _close
        md = build_default_market_data()
        s = _close(md.fetch_ohlcv("^IRX", date(1985, 1, 1), date(2026, 10, 24))) / 100.0
        s = s.dropna()
        s.to_frame("irx").to_parquet(p)
        print(f"^IRX {len(s)} 일 저장 → {p}", file=sys.stderr)
        return s
    except Exception as e:   # noqa: BLE001
        print(f"^IRX 없음 — 현금 이자 0 으로: {e}", file=sys.stderr)
        return None


def build_target(cfg: dict, px: pd.DataFrame) -> tuple[pd.Series, str]:
    kind = cfg.get("strategy", "hold")
    if kind == "hold":
        return pd.Series(1.0, index=px.index), "TQQQ 단순 보유"
    if kind == "switch":
        dd, n, re, off = float(cfg.get("dd", 0.18)), int(cfg.get("n", 200)), int(cfg.get("reenter", 20)), float(cfg.get("off_fraction", 0.0))
        return deep_correction_filter(px["QQQ"], n, dd, re, off), f"스위치 dd{dd:.2f} n{n} re{re} off{off:g}"
    if kind == "target_parquet":
        t = pd.read_parquet(cfg["target_path"]).iloc[:, 0]
        t.index = pd.to_datetime(t.index)
        return t.astype(float), cfg.get("name", Path(cfg["target_path"]).stem)
    raise ValueError(kind)


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES)
    base = load_baseline(BASELINE)
    cash = load_cash_rate() if cfg.get("cash_rate", "irx") == "irx" else None
    edition = cfg.get("edition", "dev")
    target, name = build_target(cfg, px)
    name = cfg.get("name", name)
    costs = GATE_COSTS_KR if cfg.get("tax_model") == "kr" else GATE_COSTS        # H13: 배당세 15.4% 포함 정밀 모델
    tbl = window_table(px["TQQQ"], target, base, edition, costs=costs, cash_rate=cash)
    print(f"{name}: {edition} 창 {len(tbl)} 개 {time.time() - t0:.1f}s", file=sys.stderr)
    out = {"name": name, "edition": edition, "cfg": cfg, "windows": json.loads(tbl.to_json(orient="records", date_format="iso"))}
    if edition == "sealed":
        out["verdict"] = sealed_verdicts(tbl)
        md = verdict_markdown(name, tbl, {"G1": out["verdict"]["G1"], "G2": out["verdict"]["G2"], "G3": out["verdict"]["G3"],
                                          "first_fail": None if out["verdict"]["passed"] else next(k for k in ("G1", "G2", "G3") if not out["verdict"][k]["passed"]),
                                          "passed_through": next((i for i, k in enumerate(("G1", "G2", "G3")) if not out["verdict"][k]["passed"]), 3)}, edition,
                              note="봉인 해제 판 (G7) — G1~G3 만")
    else:
        meds = None
        n_seeds = int(cfg.get("placebo_seeds", 0))
        if n_seeds > 0:
            meds = placebo_medians(px["TQQQ"], target, base, n_seeds, int(cfg.get("placebo_seed0", 20261110)), cfg.get("placebo_mode", "runs"),
                                   edition, costs, cash,
                                   progress=lambda i, m: print(f"  placebo {i + 1}/{n_seeds} median ratio {m:.3f} ({time.time() - t0:.0f}s)", file=sys.stderr))
            out["placebo_medians"] = meds
        b_from = pd.Timestamp(cfg["b_start_from"]) if cfg.get("b_start_from") else None
        out["verdict"] = gate_verdicts(tbl, int(cfg.get("n_trials", 0)), meds, cfg.get("neighbor_verdicts"), b_from)
        md = verdict_markdown(name, tbl, out["verdict"], edition, note=cfg.get("note", ""))
    md += "\n" + window_markdown(tbl) + "\n"
    out["elapsed_s"] = time.time() - t0
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md)
    return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
