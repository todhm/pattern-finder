"""급등주 탐색 — 생존 편향 없는 전 종목 일봉 패널 수집 (Massive grouped-daily, 날짜당 1콜, 플랜상 최근 2년).

실행: docker compose exec backtester python surge_panel_cli.py '{"start": "2024-11-12", "end": "2026-11-10"}'
날짜별 parquet 캐시 `sweep_results/grouped_daily_cache/YYYY-MM-DD.parquet` (휴장일은 빈 파일) → `build` 로 long parquet 하나.
계획 docs/strategy_notes/surge_search_runs/급등주_탐색_계획.md
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

CACHE = Path("/app/sweep_results/grouped_daily_cache")
PANEL = Path("/app/sweep_results/surge_panel_long.parquet")


def fetch(cfg: dict) -> None:
    from data.adapters.massive_grouped_daily import MassiveGroupedDailyAdapter
    CACHE.mkdir(parents=True, exist_ok=True)
    a = MassiveGroupedDailyAdapter(rate_retry_sleep=15.0, max_rate_retries=8)
    d = date.fromisoformat(cfg["start"]); end = date.fromisoformat(cfg["end"]); pace = float(cfg.get("pace", 12.5))
    days = []
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    todo = [x for x in days if not (CACHE / f"{x}.parquet").exists()]
    print(f"거래일 후보 {len(days)} · 받을 날 {len(todo)}", file=sys.stderr, flush=True)
    for i, x in enumerate(todo):
        t0 = time.time()
        try:
            df = a.fetch_day(x)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(todo)}] {x} 실패: {str(e)[:120]}", file=sys.stderr, flush=True); time.sleep(pace); continue
        df.to_parquet(CACHE / f"{x}.parquet")
        if (i + 1) % 20 == 0 or i == len(todo) - 1:
            print(f"[{i + 1}/{len(todo)}] {x} {len(df)} 종목", file=sys.stderr, flush=True)
        time.sleep(max(0.0, pace - (time.time() - t0)))


def build() -> pd.DataFrame:
    frames = []
    for f in sorted(CACHE.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"index": "symbol"}); df["date"] = pd.Timestamp(f.stem)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(PANEL)
    print(f"패널 {len(out):,} 행 · {out['date'].nunique()} 일 · {out['symbol'].nunique():,} 종목 → {PANEL}", file=sys.stderr)
    return out


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if cfg.get("mode") == "build":
        build()
    else:
        fetch(cfg); build()
