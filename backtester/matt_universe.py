"""Expand universe to ~15 liquid large-caps and retest winners."""
import sys, os, pickle, pandas as pd
sys.path.insert(0, '/app')
from datetime import date, timedelta
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import NY
from pattern.adapters.matt_diamond_bull_flag import compute_market_regime_ok, compute_premarket_high_by_date

TICKERS = ["NVDA", "TSLA", "QQQ", "AAPL", "AMD", "META", "MSFT", "GOOG", "AMZN", "NFLX", "SPY", "IWM", "COIN", "PLTR", "MSTR"]
START = date(2025, 1, 1)
END = date.today()
CACHE = "/tmp/matt_cache_big.pkl"

if os.path.exists(CACHE):
    print("Cache exists, using")
    sys.exit()

md_raw = build_default_market_data()
md = RegularSessionFilterAdapter(md_raw, market=NY)
df_spy = md.fetch_ohlcv("SPY", START - timedelta(days=120), END, interval="1d")
regime_ok = compute_market_regime_ok(df_spy, sma_period=50)
store = {"_regime": regime_ok}

for t in TICKERS:
    try:
        print(f"  fetching {t}...", flush=True)
        df_intra = md.fetch_ohlcv(t, START, END, interval="1m")
        df_daily = md.fetch_ohlcv(t, START - timedelta(days=120), END, interval="1d")
        df_raw = md_raw.fetch_ohlcv(t, START, END, interval="1m")
        if df_intra.index.tz is not None and str(df_intra.index.tz) != NY.tz:
            df_intra.index = df_intra.index.tz_convert(NY.tz)
        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize(NY.tz)
        elif str(df_raw.index.tz) != NY.tz:
            df_raw.index = df_raw.index.tz_convert(NY.tz)
        pm = compute_premarket_high_by_date(df_raw)
        store[t] = {"intra": df_intra, "daily": df_daily, "pm": pm}
    except Exception as e:
        print(f"  {t} fetch fail: {e}")
        continue

with open(CACHE, "wb") as f:
    pickle.dump(store, f)
print(f"DONE: cached {len(store)-1} tickers")
