"""Ablate the new high-impact gates on top of the resistance+regime+target winner."""
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
    for t in TICKERS:
        td = store[t]
        base = dict(require_premarket_high=True, premarket_high_by_date=td["pm"])
        base.update(det_kw)
        det = MattDiamondBullFlagDetector(**base)
        strat = MattDiamondBullFlagStrategy(det, fee_schedule=TossFeeSchedule(), **strat_kw)
        cfg = StrategyConfig(ticker=t, start_date=START, end_date=END, pattern_name="matt_diamond_bull_flag", initial_capital=100_000.0, risk_per_trade=0.01)
        agg.extend(strat.run(td["intra"], td["daily"], cfg).performance.trades)
    wins = [t for t in agg if t.pnl > 0]; losses = [t for t in agg if t.pnl <= 0]
    pw = sum(t.pnl for t in wins); pl = abs(sum(t.pnl for t in losses)) or 1e-9
    return {"n": len(agg), "wins": len(wins), "losses": len(losses), "win": len(wins)/max(1,len(agg)), "pf": pw/pl, "pnl": sum(t.pnl for t in agg)}

def line(label, r):
    print(f"{label:65s}  n={r['n']:>3d}  W={r['wins']:>3d}/L={r['losses']:>3d}  win%={r['win']*100:>5.1f}  PF={r['pf']:>4.2f}  PnL=${r['pnl']:>+9,.0f}", flush=True)

# Starting from the prior winner: regime + resistance + pole>=1.5% + target=2.5R
BASE = {
    "require_daily_resistance_break": True,
    "market_regime_ok_by_date": store["_regime"],
    "pole_min_pct": 0.015,
}
STRAT = {"target_at_r_multiple": 2.5}
line("STACK = regime + resistance + pole>=1.5% + target=2.5R", run(BASE, STRAT))

print("\n--- New gates added on top of STACK ---")
for label, extra in [
    ("+ skip first 3 min", {"skip_first_n_minutes": 3}),
    ("+ skip first 5 min", {"skip_first_n_minutes": 5}),
    ("+ skip first 7 min", {"skip_first_n_minutes": 7}),
    ("+ GTR vol >= 1.0× prev red", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0}),
    ("+ GTR vol >= 1.5× prev red", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.5}),
    ("+ GTR vol >= 2.0× prev red", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 2.0}),
    ("+ VWAP hold", {"require_pullback_above_vwap": True}),
    ("+ max_consecutive_reds=3", {"max_consecutive_red_bars": 3}),
    ("+ max_consecutive_reds=2", {"max_consecutive_red_bars": 2}),
    ("+ resistance_break_pct=0.2%", {"min_resistance_break_pct": 0.002}),
    ("+ resistance_break_pct=0.5%", {"min_resistance_break_pct": 0.005}),
    ("+ resistance_break_pct=1.0%", {"min_resistance_break_pct": 0.010}),
]:
    kw = dict(BASE); kw.update(extra)
    line(label, run(kw, STRAT))

print("\n--- Two-gate combos ---")
for label, extra in [
    ("+ skip5 + VWAP", {"skip_first_n_minutes": 5, "require_pullback_above_vwap": True}),
    ("+ skip5 + GTR vol 1.5×", {"skip_first_n_minutes": 5, "require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.5}),
    ("+ skip3 + GTR vol 1.0×", {"skip_first_n_minutes": 3, "require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0}),
    ("+ VWAP + GTR vol 1.0×", {"require_pullback_above_vwap": True, "require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0}),
    ("+ rb=0.5% + VWAP", {"min_resistance_break_pct": 0.005, "require_pullback_above_vwap": True}),
    ("+ rb=0.5% + GTR vol 1.0×", {"min_resistance_break_pct": 0.005, "require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0}),
]:
    kw = dict(BASE); kw.update(extra)
    line(label, run(kw, STRAT))

print("DONE", flush=True)
