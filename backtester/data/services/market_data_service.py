from datetime import date

import pandas as pd
import yfinance as yf


class MarketDataService:
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        indexed by DatetimeIndex.
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.isoformat(), end=end.isoformat())
        if df.empty:
            raise ValueError(f"No data found for {symbol} between {start} and {end}")
        return df[["Open", "High", "Low", "Close", "Volume"]]
