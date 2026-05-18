"""Verify Matt source case still fires with new defaults."""
import sys, pickle
sys.path.insert(0, '/app')
from datetime import date, timedelta
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import NY
from pattern.adapters.matt_diamond_bull_flag import (
    MattDiamondBullFlagDetector, compute_market_regime_ok, compute_premarket_high_by_date,
)

md_raw = build_default_market_data()
md = RegularSessionFilterAdapter(md_raw, market=NY)

target = date(2025, 3, 24)
df_intra = md.fetch_ohlcv("TSLA", target - timedelta(days=2), target + timedelta(days=1), interval="1m")
df_daily = md.fetch_ohlcv("TSLA", target - timedelta(days=120), target + timedelta(days=1), interval="1d")
df_raw = md_raw.fetch_ohlcv("TSLA", target - timedelta(days=2), target + timedelta(days=1), interval="1m")
df_spy = md.fetch_ohlcv("SPY", target - timedelta(days=120), target + timedelta(days=1), interval="1d")
if df_intra.index.tz is not None and str(df_intra.index.tz) != NY.tz:
    df_intra.index = df_intra.index.tz_convert(NY.tz)
if df_raw.index.tz is None:
    df_raw.index = df_raw.index.tz_localize(NY.tz)
elif str(df_raw.index.tz) != NY.tz:
    df_raw.index = df_raw.index.tz_convert(NY.tz)
pm = compute_premarket_high_by_date(df_raw)
regime = compute_market_regime_ok(df_spy, sma_period=50)

# New proposed defaults
det = MattDiamondBullFlagDetector(
    premarket_high_by_date=pm,
    require_daily_resistance_break=True,
    min_resistance_break_pct=0.005,
    pole_min_pct=0.015,
    require_gtr_volume_expansion=True,
    gtr_volume_expansion_mult=1.0,
)
sigs = det.detect(df_intra, df_daily)
on_target = [s for s in sigs if s.session_date == target]
print(f"Matt 영상 정통 (TSLA 2025-03-24) with NEW DEFAULTS: {len(on_target)} signals")
for s in on_target:
    print(f"  nth={s.nth_pullback}  entry={s.entry_ts.strftime('%H:%M')}  @${s.entry_price:.2f}  stop=${s.stop_loss:.2f}  pm_high=${s.pm_high:.2f}")
