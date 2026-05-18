"""Validate new defaults match the WINNER config."""
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

agg = []
by_ticker = {}
for t in TICKERS:
    td = store[t]
    # Use bare defaults — no extra kwargs. Confirms factory defaults are correct.
    det = MattDiamondBullFlagDetector(premarket_high_by_date=td["pm"])
    strat = MattDiamondBullFlagStrategy(det, fee_schedule=TossFeeSchedule())
    cfg = StrategyConfig(ticker=t, start_date=START, end_date=END, pattern_name="matt_diamond_bull_flag", initial_capital=100_000.0, risk_per_trade=0.01)
    trades = strat.run(td["intra"], td["daily"], cfg).performance.trades
    agg.extend(trades)
    by_ticker[t] = trades

wins = [t for t in agg if t.pnl > 0]; losses = [t for t in agg if t.pnl <= 0]
pw = sum(t.pnl for t in wins); pl = abs(sum(t.pnl for t in losses)) or 1e-9
print(f"=== FINAL DEFAULTS — bare construction, 15 tickers × 17 months ===")
print(f"n={len(agg)}  W={len(wins)}/L={len(losses)}  win%={len(wins)/max(1,len(agg))*100:.1f}  PF={pw/pl:.2f}  PnL=${sum(t.pnl for t in agg):+,.0f}")
print(f"\nPer-ticker:")
for t in TICKERS:
    tr = by_ticker[t]
    if not tr:
        continue
    w = sum(1 for x in tr if x.pnl > 0)
    pnl = sum(x.pnl for x in tr)
    print(f"  {t:6s} n={len(tr):>2d}  W={w}/{len(tr)-w}  PnL=${pnl:>+7,.0f}")
print(f"\nAll trades:")
for tr in agg:
    print(f"  {tr.entry_ts}  entry=${tr.entry_price:.2f}  exit=${tr.exit_price:.2f}  R/L={'W' if tr.pnl>0 else 'L'}  PnL=${tr.pnl:+,.0f}  reason={tr.exit_reason}")
