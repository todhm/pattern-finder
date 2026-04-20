from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from data.domain.ports import MarketDataPort


class YFinanceAdapter(MarketDataPort):
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        yfinance's ``end`` parameter is **exclusive**, so passing
        ``end=today`` returns bars strictly before today and the
        live/just-closed session is missed. Scanner callers need the
        current-day bar (intraday snapshot during the session, final
        close after), so we forward ``end + 1 day`` to make the
        caller-provided ``end`` effectively inclusive.
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
        )
        if df.empty:
            raise ValueError(f"No data found for {symbol} between {start} and {end}")
        return df[["Open", "High", "Low", "Close", "Volume"]]
