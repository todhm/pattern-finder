"""Ross 불플래그 슬리피지 스트레스 테스트 — 재개 가능한 배치 스크립트.

alphafolio ``us_daily``로 사전 선별한 갭 이벤트 종목(전체 유니버스
불필요 — 관문 조건인 갭·가격대는 일봉으로 판별되므로)에 대해:

  Phase 1  prefetch  — 종목별 daily/1m/5m/float를 Mongo 캐시에 적재.
                       **종목 단위로 진행상황을 저장**하므로 중간에
                       끊어도 재실행하면 이어서 받는다.
  Phase 2  run       — 캐시 위에서 MultiBullFlagStrategy 실행(빠름),
                       트레이드 로그를 JSON으로 저장.
  Phase 3  sweep     — 트레이드 로그에 슬리피지 0/10/25/50/100bp를
                       사후 적용해 엣지가 어디서 죽는지 표로 출력.

사용법 (컨테이너 안에서)::

    # 후보 파일은 sweep_results/gap_candidates.json (alphafolio에서 생성)
    python bullflag_slippage_sweep.py --phase prefetch   # 끊겨도 재실행하면 이어감
    python bullflag_slippage_sweep.py --phase run
    python bullflag_slippage_sweep.py --phase sweep
    python bullflag_slippage_sweep.py                    # all: 위 3단계 연속

산출물: sweep_results/bullflag_prefetch_progress.json (진행상황),
        sweep_results/bullflag_trades.json (트레이드 로그).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta

RESULTS_DIR = "sweep_results"
CANDIDATES = os.path.join(RESULTS_DIR, "gap_candidates.json")
PROGRESS = os.path.join(RESULTS_DIR, "bullflag_prefetch_progress.json")
TRADES = os.path.join(RESULTS_DIR, "bullflag_trades.json")


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _atomic_write(path: str, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _load_candidates(max_tickers: int) -> tuple[list[str], date, date]:
    with open(CANDIDATES) as f:
        d = json.load(f)
    start = date.fromisoformat(d["window"][0])
    end = date.fromisoformat(d["window"][1])
    tickers = [c["symbol"] for c in d["candidates"][:max_tickers]]
    return tickers, start, end


class StaticUniverseProvider:
    """고정 티커 리스트 — 사전 선별 후보를 그대로 쓰는 universe."""

    def __init__(self, tickers: list[str]):
        self._tickers = tickers

    def get_tickers(self, universe: str) -> list[str]:
        return list(self._tickers)


def _build_sources():
    from data.adapters.composed_fundamentals import build_default_fundamentals
    from data.adapters.composed_market_data import build_default_market_data
    from data.adapters.regular_session_filter import RegularSessionFilterAdapter
    from data.domain.market_calendar import NY

    md = RegularSessionFilterAdapter(build_default_market_data(), market=NY)
    return md, build_default_fundamentals()


def phase_prefetch(tickers, start, end, workers: int) -> None:
    """종목별 데이터 적재 — 진행상황을 종목 단위로 저장 (재개 가능)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    md, fundamentals = _build_sources()
    progress: dict[str, str] = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            progress = json.load(f)
    todo = [t for t in tickers if progress.get(t) not in ("ok", "no_data")]
    _log(f"prefetch: 총 {len(tickers)}개 중 {len(todo)}개 남음 "
         f"(완료 {len(tickers) - len(todo)}개)")

    def fetch_one(t: str) -> tuple[str, str]:
        try:
            daily = md.fetch_ohlcv(t, start - timedelta(days=90), end, interval="1d")
            if daily is None or len(daily) < 30:
                return t, "no_data"
            md.fetch_ohlcv(t, start, end, interval="1m")
            md.fetch_ohlcv(t, start, end, interval="5m")
            fundamentals.fetch(t)
            return t, "ok"
        except Exception as e:  # noqa: BLE001 — 상태로 기록하고 계속
            return t, f"error: {str(e)[:120]}"

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, t): t for t in todo}
        for fut in as_completed(futures):
            t, status = fut.result()
            progress[t] = status
            done += 1
            # 매 종목마다 저장 — 어느 시점에 끊겨도 재개 가능.
            _atomic_write(PROGRESS, progress)
            _log(f"[{done}/{len(todo)}] {t}: {status}")
    ok = sum(1 for v in progress.values() if v == "ok")
    _log(f"prefetch 완료: ok {ok} / no_data "
         f"{sum(1 for v in progress.values() if v == 'no_data')} / "
         f"error {sum(1 for v in progress.values() if str(v).startswith('error'))}")


def phase_run(tickers, start, end, workers: int,
              require_float: bool = False,
              max_float_mil: float = 20.0) -> None:
    """캐시 위에서 멀티 불플래그 실행 → 트레이드 저장."""
    from pattern.adapters.bull_flag import BullFlagDetector
    from strategy.adapters.bull_flag_strategy import BullFlagStrategy
    from strategy.adapters.multi_bull_flag_strategy import MultiBullFlagStrategy
    from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule
    from data.domain.market_calendar import NY

    md, fundamentals = _build_sources()
    fee = TossFeeSchedule()

    def detector_factory(*, float_shares, splits, pm_high_by_date=None):
        # float 데이터가 없는 소형주가 많아(yfinance 커버리지 한계)
        # 기본은 float-미상 통과 + 상한 완화. --require-float로 정통 모드.
        return BullFlagDetector(
            float_shares=float_shares, splits=splits,
            require_float_filter=require_float,
            max_float_shares=max_float_mil * 1e6,
        )

    def strategy_factory(*, detector):
        return BullFlagStrategy(detector=detector, fee_schedule=fee)

    multi = MultiBullFlagStrategy(
        market_data=md, market_data_5m=md, daily_market_data=md,
        fundamentals=fundamentals,
        universe_provider=StaticUniverseProvider(tickers),
        detector_factory=detector_factory,
        strategy_factory=strategy_factory,
        market=NY, max_workers=workers, require_float_filter=require_float,
        chunk_months=1,
    )
    cfg = MultiStrategyConfig(
        universe="static", start_date=start, end_date=end,
        initial_capital=100_000.0, fee_schedule=fee,
    )
    _log(f"run: {len(tickers)}개 종목 × {start}~{end}")
    r = multi.run(cfg)
    out = {
        "window": [str(start), str(end)],
        "tickers_scanned": r.tickers_scanned,
        "total_signals": r.total_signals,
        "win_rate": r.win_rate,
        "final_capital": r.final_capital,
        "trades": [
            {
                "ticker": t.ticker,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "entry": t.entry_price,
                "exit": t.exit_price,
                "shares": t.shares,
                "commission": t.commission,
                "gross_pnl": t.gross_pnl,
            }
            for t in r.trades
        ],
    }
    _atomic_write(TRADES, out)
    _log(
        f"run 완료: 스캔 {r.tickers_scanned}, 시그널 {r.total_signals}, "
        f"트레이드 {len(r.trades)}, 승률 {r.win_rate:.1%}, "
        f"최종 {r.final_capital:,.0f} → {TRADES}"
    )


def phase_sweep(slippages_bp=(0, 10, 25, 50, 100)) -> None:
    """트레이드 로그에 슬리피지를 사후 적용해 엣지 사망 지점 탐색."""
    with open(TRADES) as f:
        d = json.load(f)
    trades = d["trades"]
    if not trades:
        _log("트레이드가 없어 스윕 불가 — run 결과 확인 필요.")
        return
    capital0 = 100_000.0
    print(f"\n트레이드 {len(trades)}건 ({d['window'][0]} ~ {d['window'][1]})")
    print(f"{'슬리피지(편도)':>12}{'최종자본':>12}{'수익률':>9}{'승률':>7}{'평균손익/건':>11}")
    for bp in slippages_bp:
        cap = capital0
        wins = 0
        pnls = []
        for t in trades:
            slip = (bp / 10_000.0) * (
                t["entry"] * t["shares"] + t["exit"] * t["shares"]
            )
            pnl = t["gross_pnl"] - t["commission"] - slip
            cap += pnl
            pnls.append(pnl)
            if pnl > 0:
                wins += 1
        print(
            f"{bp:>10}bp{cap:>12,.0f}{cap / capital0 - 1:>9.1%}"
            f"{wins / len(trades):>7.1%}{sum(pnls) / len(pnls):>11,.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["prefetch", "run", "sweep", "all"],
                    default="all")
    ap.add_argument("--max-tickers", type=int, default=300,
                    help="갭 이벤트 수 상위 N개만 스캔 (기본 300)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--require-float", action="store_true",
                    help="영상 정통: float 미상 종목 제외 (기본은 통과)")
    ap.add_argument("--max-float-mil", type=float, default=20.0,
                    help="float 상한 (백만 주, 기본 20)")
    args = ap.parse_args()

    tickers, start, end = _load_candidates(args.max_tickers)
    _log(f"후보 {len(tickers)}개 · 윈도우 {start} ~ {end}")
    if args.phase in ("prefetch", "all"):
        phase_prefetch(tickers, start, end, args.workers)
    if args.phase in ("run", "all"):
        phase_run(tickers, start, end, args.workers,
                  require_float=args.require_float,
                  max_float_mil=args.max_float_mil)
    if args.phase in ("sweep", "all"):
        phase_sweep()


if __name__ == "__main__":
    sys.exit(main())
