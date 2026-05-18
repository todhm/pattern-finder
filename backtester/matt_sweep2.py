import sys; sys.path.insert(0, '/app')
from matt_research import fetch_universe, run_one, print_summary
store = fetch_universe()
out = []
def line(label, r):
    s = f"{label:65s}  n={r['n']:>4d}  W={r['wins']:>3d}/L={r['losses']:>3d}  win%={r['win_rate']*100:>5.1f}  PF={r['profit_factor']:>4.2f}  PnL=${r['total_pnl']:>+9,.0f}"
    out.append(s); print(s, flush=True)

line("resistance_break alone", run_one(store, detector_kwargs={"require_daily_resistance_break": True}))
line("+ lookback=10", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "resistance_lookback_days": 10}))
line("+ lookback=30", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "resistance_lookback_days": 30}))
line("+ lookback=60", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "resistance_lookback_days": 60}))
line("resistance + close>EMA", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "require_pullback_close_above_ema": True}))
line("resistance + pole>=1%", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.010}))
line("resistance + pole>=1.5%", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.015}))
line("resistance + min_rvol=1.3", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "min_rvol": 1.3}))
line("resistance + max_nth=1", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "max_nth_pullback": 1}))
line("resistance + target=2.0R", run_one(store, detector_kwargs={"require_daily_resistance_break": True}, strategy_kwargs={"target_at_r_multiple": 2.0}))
line("resistance + target=2.5R", run_one(store, detector_kwargs={"require_daily_resistance_break": True}, strategy_kwargs={"target_at_r_multiple": 2.5}))
line("resistance + BE off + target=2.0R", run_one(store, detector_kwargs={"require_daily_resistance_break": True}, strategy_kwargs={"target_at_r_multiple": 2.0, "enable_breakeven_after_r": None}))
line("resistance + close>EMA + target=2.0R + BE off", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "require_pullback_close_above_ema": True}, strategy_kwargs={"target_at_r_multiple": 2.0, "enable_breakeven_after_r": None}))
line("resistance + pole>=1% + target=2.0R + BE off", run_one(store, detector_kwargs={"require_daily_resistance_break": True, "pole_min_pct": 0.010}, strategy_kwargs={"target_at_r_multiple": 2.0, "enable_breakeven_after_r": None}))

open("/tmp/sweep2_out.txt", "w").write("\n".join(out))
print("DONE", flush=True)
