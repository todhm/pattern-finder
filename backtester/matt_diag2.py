import sys; sys.path.insert(0, '/app')
from datetime import date, timedelta
import pandas as pd
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import NY

md = RegularSessionFilterAdapter(build_default_market_data(), market=NY)
target = date(2025, 3, 24)
df_daily = md.fetch_ohlcv("TSLA", target - timedelta(days=60), target + timedelta(days=1), interval="1d")

# 20-day high prior to target
hist = df_daily[df_daily.index.date < target]
nbar_high = hist["High"].tail(20).max()
today_open = df_daily[df_daily.index.date == target].iloc[0]["Open"]
print(f"TSLA 2025-03-24 Open: ${today_open:.2f}")
print(f"20-day high before target: ${nbar_high:.2f}")
print(f"Open / 20d high = {today_open/nbar_high:.4f}")
print(f"Open above 20d high by: {(today_open-nbar_high)/nbar_high*100:+.2f}%")
