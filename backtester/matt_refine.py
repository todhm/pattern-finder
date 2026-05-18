"""Refine around the 77.8% winner. Cross-check stability + edge cases."""
import sys, pickle
sys.path.insert(0, '/app')
from datetime import date
from pattern.adapters.matt_diamond_bull_flag import MattDiamondBullFlagDetector
from strategy.adapters.matt_diamond_bull_flag_strategy import MattDiamondBullFlagStrategy
from strategy.domain.models import StrategyConfig, TossFeeSchedule

with open("/tmp/matt_cache_big.pkl", "rb") as f:
    store = pickle.load(f)
TICKERS = [k for k in store if not k.startswith("_")]
START = date(2025, 1, 1)
END = date.today()

def run(det_kw=None, strat_kw=None):
    det_kw = det_kw or {}; strat_kw = strat_kw or {}
    agg = []
    by_ticker = {}
    for t in TICKERS:
        td = store[t]
        base = dict(require_premarket_high=True, premarket_high_by_date=td["pm"])
        base.update(det_kw)
        det = MattDiamondBullFlagDetector(**base)
        strat = MattDiamondBullFlagStrategy(det, fee_schedule=TossFeeSchedule(), **strat_kw)
        cfg = StrategyConfig(ticker=t, start_date=START, end_date=END, pattern_name="matt_diamond_bull_flag", initial_capital=100_000.0, risk_per_trade=0.01)
        trades = strat.run(td["intra"], td["daily"], cfg).performance.trades
        agg.extend(trades)
        by_ticker[t] = trades
    wins = [t for t in agg if t.pnl > 0]; losses = [t for t in agg if t.pnl <= 0]
    pw = sum(t.pnl for t in wins); pl = abs(sum(t.pnl for t in losses)) or 1e-9
    return {"n": len(agg), "wins": len(wins), "losses": len(losses), "win": len(wins)/max(1,len(agg)), "pf": pw/pl, "pnl": sum(t.pnl for t in agg), "by_ticker": by_ticker}

def line(label, r):
    print(f"{label:65s}  n={r['n']:>3d}  W={r['wins']:>3d}/L={r['losses']:>3d}  win%={r['win']*100:>5.1f}  PF={r['pf']:>4.2f}  PnL=${r['pnl']:>+9,.0f}", flush=True)

# Verified winner
WINNER = {
    "require_daily_resistance_break": True,
    "market_regime_ok_by_date": store["_regime"],
    "pole_min_pct": 0.015,
    "min_resistance_break_pct": 0.005,
    "require_gtr_volume_expansion": True,
    "gtr_volume_expansion_mult": 1.0,
}
STRAT = {"target_at_r_multiple": 2.5}
print("=== WINNER baseline ===")
line("regime + rb + pole + rb_pct=0.5% + GTR_vol", run(WINNER, STRAT))

# Sensitivity around each knob
print("\n--- Sensitivity tests ---")
for label, mod in [
    ("regime OFF", {"market_regime_ok_by_date": {}}),
    ("rb_pct=0.0", {"min_resistance_break_pct": 0.0}),
    ("rb_pct=0.3%", {"min_resistance_break_pct": 0.003}),
    ("rb_pct=1.0%", {"min_resistance_break_pct": 0.010}),
    ("GTR vol mult=1.2×", {"gtr_volume_expansion_mult": 1.2}),
    ("GTR vol mult=0.8×", {"gtr_volume_expansion_mult": 0.8}),
    ("pole>=1.0% (lower)", {"pole_min_pct": 0.010}),
    ("pole>=2.0% (higher)", {"pole_min_pct": 0.020}),
    ("close>EMA on", {"require_pullback_close_above_ema": True}),
    ("pullback_max=3", {"pullback_max_bars": 3}),
    ("pullback_max=7", {"pullback_max_bars": 7}),
    ("VWAP hold also", {"require_pullback_above_vwap": True}),
    ("max_nth=1", {"max_nth_pullback": 1}),
    ("target=2.0R", {}),
    ("target=3.0R", {}),
    ("target=4.0R", {}),
]:
    kw = dict(WINNER); kw.update(mod)
    skw = dict(STRAT)
    if "target=" in label:
        skw["target_at_r_multiple"] = float(label.split("=")[1].split("R")[0])
    line(label, run(kw, skw))

# Per-ticker drill
print("\n--- WINNER per-ticker breakdown ---")
r = run(WINNER, STRAT)
for t in TICKERS:
    tr = r["by_ticker"][t]
    if not tr:
        continue
    w = sum(1 for x in tr if x.pnl > 0)
    pnl = sum(x.pnl for x in tr)
    print(f"  {t:6s} n={len(tr):>2d}  W={w}/{len(tr)-w}  PnL=${pnl:>+7,.0f}")
print("DONE", flush=True)
