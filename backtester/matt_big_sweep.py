"""Re-run winners on expanded 15-ticker universe."""
import sys, os, pickle
sys.path.insert(0, '/app')
from datetime import date
from matt_research import run_one  # uses TICKERS but we'll override store

# load big cache
with open("/tmp/matt_cache_big.pkl", "rb") as f:
    store = pickle.load(f)

# monkey-patch: run_one iterates over matt_research.TICKERS module list.
# easier: copy run_one inline with TICKERS from store keys.
from pattern.adapters.matt_diamond_bull_flag import MattDiamondBullFlagDetector
from strategy.adapters.matt_diamond_bull_flag_strategy import MattDiamondBullFlagStrategy
from strategy.domain.models import StrategyConfig, TossFeeSchedule

START = date(2025, 1, 1)
END = date.today()
TICKERS = [k for k in store.keys() if not k.startswith("_")]

def run_one_big(detector_kwargs=None, strategy_kwargs=None):
    detector_kwargs = detector_kwargs or {}
    strategy_kwargs = strategy_kwargs or {}
    agg = []
    for t in TICKERS:
        td = store[t]
        det_kw = dict(require_premarket_high=True, premarket_high_by_date=td["pm"], market_regime_ok_by_date={})
        det_kw.update(detector_kwargs)
        det = MattDiamondBullFlagDetector(**det_kw)
        strat = MattDiamondBullFlagStrategy(det, fee_schedule=TossFeeSchedule(), **strategy_kwargs)
        cfg = StrategyConfig(ticker=t, start_date=START, end_date=END, pattern_name="matt_diamond_bull_flag", initial_capital=100_000.0, risk_per_trade=0.01)
        result = strat.run(td["intra"], td["daily"], cfg)
        agg.extend(result.performance.trades)
    wins = [t for t in agg if t.pnl > 0]; losses = [t for t in agg if t.pnl <= 0]
    pnl_w = sum(t.pnl for t in wins); pnl_l = abs(sum(t.pnl for t in losses)) or 1e-9
    return {"n": len(agg), "wins": len(wins), "losses": len(losses),
            "win_rate": len(wins)/max(1,len(agg)), "profit_factor": pnl_w/pnl_l,
            "total_pnl": sum(t.pnl for t in agg), "trades": agg}

def line(label, r):
    s = f"{label:70s}  n={r['n']:>4d}  W={r['wins']:>3d}/L={r['losses']:>3d}  win%={r['win_rate']*100:>5.1f}  PF={r['profit_factor']:>4.2f}  PnL=${r['total_pnl']:>+9,.0f}"
    print(s, flush=True)

print(f"=== Universe: {len(TICKERS)} tickers, 2025/01/01~today ===")
line("BASELINE (current defaults)", run_one_big())

print("\n--- Single high-impact gates ---")
line("resistance_break", run_one_big(detector_kwargs={"require_daily_resistance_break": True}))
line("regime ON", run_one_big(detector_kwargs={"market_regime_ok_by_date": store["_regime"]}))
line("pole>=1.5%", run_one_big(detector_kwargs={"pole_min_pct": 0.015}))
line("min_rvol=1.3", run_one_big(detector_kwargs={"min_rvol": 1.3}))
line("close>EMA", run_one_big(detector_kwargs={"require_pullback_close_above_ema": True}))
line("pullback_max=3", run_one_big(detector_kwargs={"pullback_max_bars": 3}))
line("max_nth=1", run_one_big(detector_kwargs={"max_nth_pullback": 1}))

print("\n--- WINNERS (target=2.5R + resistance) ---")
line("resistance + target=2.5R", run_one_big(detector_kwargs={"require_daily_resistance_break": True}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + pullback=3 + target=2.5R", run_one_big(detector_kwargs={"require_daily_resistance_break": True, "pullback_max_bars": 3}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + pullback=3 + target=3.0R", run_one_big(detector_kwargs={"require_daily_resistance_break": True, "pullback_max_bars": 3}, strategy_kwargs={"target_at_r_multiple": 3.0}))
line("regime + resistance + target=2.5R", run_one_big(detector_kwargs={"require_daily_resistance_break": True, "market_regime_ok_by_date": store["_regime"]}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("regime + resistance + pole>=1.5% + target=2.5R", run_one_big(detector_kwargs={"require_daily_resistance_break": True, "market_regime_ok_by_date": store["_regime"], "pole_min_pct": 0.015}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("regime + resistance + pullback=3 + target=2.5R", run_one_big(detector_kwargs={"require_daily_resistance_break": True, "market_regime_ok_by_date": store["_regime"], "pullback_max_bars": 3}, strategy_kwargs={"target_at_r_multiple": 2.5}))

print("\n--- Alternatives without resistance gate ---")
line("regime + pole>=1.5% + close>EMA + target=2.5R", run_one_big(detector_kwargs={"market_regime_ok_by_date": store["_regime"], "pole_min_pct": 0.015, "require_pullback_close_above_ema": True}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("regime + pole>=1.5% + pullback=3 + target=2.5R", run_one_big(detector_kwargs={"market_regime_ok_by_date": store["_regime"], "pole_min_pct": 0.015, "pullback_max_bars": 3}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("rvol=1.3 + pole>=1.5% + pullback=3 + target=2.5R", run_one_big(detector_kwargs={"min_rvol": 1.3, "pole_min_pct": 0.015, "pullback_max_bars": 3}, strategy_kwargs={"target_at_r_multiple": 2.5}))
print("DONE", flush=True)
