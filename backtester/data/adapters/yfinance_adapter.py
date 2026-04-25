from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from data.domain.ports import MarketDataPort

NY_TZ = "America/New_York"


class YFinanceAdapter(MarketDataPort):
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        yfinance's ``end`` parameter is **exclusive**, so passing
        ``end=today`` returns bars strictly before today and the
        live/just-closed session is missed. Scanner callers need the
        current-day bar (intraday snapshot during the session, final
        close after), so we forward ``end + 1 day`` to make the
        caller-provided ``end`` effectively inclusive.

        For sub-daily intervals (``"1h"``, ``"30m"``, ``"15m"``, ...)
        yfinance returns tz-aware timestamps in the exchange timezone.
        We normalize those to ``America/New_York`` so the index has a
        stable, non-ambiguous tz contract.

        The fetch deliberately requests **pre + post market data**
        (``prepost=True``) and does *not* drop non-regular-session
        bars here. Filtering to 09:30–16:00 ET is the responsibility
        of :class:`RegularSessionFilterAdapter` so the cache layer in
        between (parquet on disk) keeps the raw data — a future ETH-
        aware strategy can then compose without the filter wrapper
        and read the same cached files. yfinance enforces a 60-day
        cap on sub-hour intervals; wider windows raise from the
        upstream call.
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=interval,
            prepost=True,
        )
        if df.empty:
            raise ValueError(
                f"No data found for {symbol} between {start} and {end} ({interval})"
            )
        if interval != "1d":
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize(NY_TZ)
            else:
                idx = idx.tz_convert(NY_TZ)
            df.index = idx
        return df[["Open", "High", "Low", "Close", "Volume"]]
