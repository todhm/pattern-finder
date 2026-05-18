"""Try to find high-WR config that also fires on Matt's source case."""
import sys, pickle
sys.path.insert(0, '/app')
from datetime import date, timedelta
from pattern.adapters.matt_diamond_bull_flag import MattDiamondBullFlagDetector
from strategy.adapters.matt_diamond_bull_flag_strategy import MattDiamondBullFlagStrategy
from strategy.domain.models import StrategyConfig, TossFeeSchedule
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import NY
from pattern.adapters.matt_diamond_bull_flag import compute_premarket_high_by_date

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

print("=== Compromise search: shorter resistance lookbacks + GTR vol ===")
STRAT = {"target_at_r_multiple": 2.5}
for lb in [3, 5, 10, 15, 20]:
    for rb_pct in [0.0, 0.005]:
        kw = {
            "require_daily_resistance_break": True,
            "resistance_lookback_days": lb,
            "min_resistance_break_pct": rb_pct,
            "pole_min_pct": 0.015,
            "require_gtr_volume_expansion": True,
            "gtr_volume_expansion_mult": 1.0,
        }
        line(f"lookback={lb} rb_pct={rb_pct*100:.1f}% + GTR vol", run(kw, STRAT))

print("\n=== Drop resistance entirely; rely on volume/momentum only ===")
print("(GTR vol + pole + close>EMA + various)")
for label, kw in [
    ("GTR vol 1.0× + pole>=1.5%", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0, "pole_min_pct": 0.015}),
    ("GTR vol 1.5× + pole>=1.5%", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.5, "pole_min_pct": 0.015}),
    ("GTR vol 1.0× + pole>=2.0%", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0, "pole_min_pct": 0.020}),
    ("GTR vol 1.0× + pole + close>EMA", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0, "pole_min_pct": 0.015, "require_pullback_close_above_ema": True}),
    ("GTR vol 1.0× + VWAP", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0, "require_pullback_above_vwap": True}),
    ("GTR vol 1.0× + pole + VWAP + close>EMA", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0, "pole_min_pct": 0.015, "require_pullback_above_vwap": True, "require_pullback_close_above_ema": True}),
    ("GTR vol 1.0× + pole + VWAP + close>EMA + min_rvol=1.3", {"require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0, "pole_min_pct": 0.015, "require_pullback_above_vwap": True, "require_pullback_close_above_ema": True, "min_rvol": 1.3}),
]:
    line(label, run(kw, STRAT))

# Test on Matt source case
print("\n=== Matt source TSLA 2025-03-24 with each config ===")
md = RegularSessionFilterAdapter(build_default_market_data(), market=NY)
md_raw = build_default_market_data()
target = date(2025, 3, 24)
df_intra = md.fetch_ohlcv("TSLA", target - timedelta(days=2), target + timedelta(days=1), interval="1m")
df_daily = md.fetch_ohlcv("TSLA", target - timedelta(days=120), target + timedelta(days=1), interval="1d")
df_raw = md_raw.fetch_ohlcv("TSLA", target - timedelta(days=2), target + timedelta(days=1), interval="1m")
if df_intra.index.tz is not None and str(df_intra.index.tz) != NY.tz:
    df_intra.index = df_intra.index.tz_convert(NY.tz)
if df_raw.index.tz is None:
    df_raw.index = df_raw.index.tz_localize(NY.tz)
elif str(df_raw.index.tz) != NY.tz:
    df_raw.index = df_raw.index.tz_convert(NY.tz)
pm_t = compute_premarket_high_by_date(df_raw)

for label, kw in [
    ("WINNER (lookback=20 rb=0.5%)", {"require_daily_resistance_break": True, "min_resistance_break_pct": 0.005, "pole_min_pct": 0.015, "require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0}),
    ("WINNER (lookback=5 rb=0)", {"require_daily_resistance_break": True, "resistance_lookback_days": 5, "pole_min_pct": 0.015, "require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0}),
    ("No resistance + GTR vol + pole 1.5%", {"pole_min_pct": 0.015, "require_gtr_volume_expansion": True, "gtr_volume_expansion_mult": 1.0}),
]:
    det = MattDiamondBullFlagDetector(premarket_high_by_date=pm_t, **kw)
    sigs = [s for s in det.detect(df_intra, df_daily) if s.session_date == target]
    print(f"  {label:55s}  → {len(sigs)} signals on Matt source")
print("DONE", flush=True)
