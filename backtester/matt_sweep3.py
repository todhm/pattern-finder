import sys; sys.path.insert(0, '/app')
from matt_research import fetch_universe, run_one, print_summary
store = fetch_universe()
out = []
def line(label, r):
    s = f"{label:70s}  n={r['n']:>4d}  W={r['wins']:>3d}/L={r['losses']:>3d}  win%={r['win_rate']*100:>5.1f}  PF={r['profit_factor']:>4.2f}  PnL=${r['total_pnl']:>+9,.0f}"
    out.append(s); print(s, flush=True)

# Best stacks so far + target variations
line("resistance + pole>=1.5% + max_nth=1", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.015, "max_nth_pullback": 1}))
line("resistance + pole>=1.5% + max_nth=1 + target=2.5R", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.015, "max_nth_pullback": 1}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + pole>=1.5% + max_nth=1 + target=3.0R", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.015, "max_nth_pullback": 1}, strategy_kwargs={"target_at_r_multiple": 3.0}))
line("resistance + rvol=1.3 + pole>=1.5%", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.015, "min_rvol": 1.3}))
line("resistance + rvol=1.3 + pole>=1.5% + target=2.5R", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.015, "min_rvol": 1.3}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + rvol=1.3 + pole>=1.5% + target=3.0R", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.015, "min_rvol": 1.3}, strategy_kwargs={"target_at_r_multiple": 3.0}))

# regime ON + resistance
line("regime + resistance", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "market_regime_ok_by_date": store["_regime"]}))
line("regime + resistance + target=2.5R", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "market_regime_ok_by_date": store["_regime"]}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("regime + resistance + pole>=1.5% + target=2.5R", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "market_regime_ok_by_date": store["_regime"], "pole_min_pct": 0.015}, strategy_kwargs={"target_at_r_multiple": 2.5}))

# More aggressive — ATR stop variations
line("resistance + atr=2.0 + target=2.5", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "atr_stop_multiplier": 2.0}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + atr=0.5 + target=2.5", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "atr_stop_multiplier": 0.5}, strategy_kwargs={"target_at_r_multiple": 2.5}))

# BE off + bigger target
line("resistance + target=2.5R + BE off + time_stop off", run_one(store, detector_kwargs={"require_daily_resistance_break": True}, strategy_kwargs={"target_at_r_multiple": 2.5, "enable_breakeven_after_r": None, "enable_time_stop": False}))
line("resistance + target=3.0R + BE off + time_stop off", run_one(store, detector_kwargs={"require_daily_resistance_break": True}, strategy_kwargs={"target_at_r_multiple": 3.0, "enable_breakeven_after_r": None, "enable_time_stop": False}))
line("resistance + target=4.0R", run_one(store, detector_kwargs={"require_daily_resistance_break": True}, strategy_kwargs={"target_at_r_multiple": 4.0}))

# pullback_max_bars tighter
line("resistance + pullback=3 + target=2.5", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pullback_max_bars": 3}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + pullback=6 + target=2.5", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pullback_max_bars": 6}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + close>EMA + target=2.5", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "require_pullback_close_above_ema": True}, strategy_kwargs={"target_at_r_multiple": 2.5}))

open("/tmp/sweep3_out.txt", "w").write("\n".join(out))
print("DONE", flush=True)
