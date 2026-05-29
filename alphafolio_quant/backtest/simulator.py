"""Portfolio simulator — walks trading days, rebalances on schedule,
builds a virtual portfolio from top-N grades in {us,kr}_stock_grade.
"""
import logging
from datetime import date
from typing import List, Dict

logger = logging.getLogger(__name__)


# 시장별 사용 테이블 — alphafolio_portfolio에서 동일하게 사용 중인 컬럼 가정
COUNTRY_TABLES = {
    "US": {"grade": "us_stock_grade", "price": "us_daily",
           "benchmark_table": "us_daily_etf", "benchmark_symbol": "SPY"},
    "KR": {"grade": "kr_stock_grade", "price": "kr_intraday_total",
           "benchmark_table": None, "benchmark_symbol": None},
}

# A real trading day has thousands of price rows; market holidays occasionally
# leave a single stray row in the price table. Treating that as a trading day
# valued every holding with no price at 0, cratering NAV for one day and
# destroying MDD/Sortino. Require a minimum row count to count as a trading day.
MIN_ROWS_PER_TRADING_DAY = 100


class PortfolioSimulator:
    def __init__(
        self,
        pool,
        country: str,
        start_date: date,
        end_date: date,
        initial_cash: float,
        top_n: int,
        rebal_freq_days: int,
        grades_filter: List[str],
        commission_rate: float,
        slippage_rate: float,
    ):
        self.pool = pool
        self.country = country.upper()
        if self.country not in COUNTRY_TABLES:
            raise ValueError(f"country must be KR or US, got {country}")
        self.start_date = start_date
        self.end_date = end_date
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.top_n = top_n
        self.rebal_freq_days = rebal_freq_days
        self.grades_filter = grades_filter
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.holdings: Dict[str, int] = {}
        self.nav_history: List[Dict] = []
        self.trades: List[Dict] = []
        # Most recent close seen per symbol — carry-forward valuation so a
        # holding is never marked to 0 on a day its price row is missing.
        self.last_prices: Dict[str, float] = {}

        tables = COUNTRY_TABLES[self.country]
        self.grade_table = tables["grade"]
        self.price_table = tables["price"]
        self.benchmark_table = tables["benchmark_table"]
        self.benchmark_symbol = tables["benchmark_symbol"]

    async def _get_top_n_grades(self, conn, d) -> List[str]:
        q = f"""
            SELECT symbol, final_score
            FROM {self.grade_table}
            WHERE date = $1 AND final_grade = ANY($2::text[])
            ORDER BY final_score DESC NULLS LAST
            LIMIT $3
        """
        rows = await conn.fetch(q, d, self.grades_filter, self.top_n)
        return [r["symbol"] for r in rows]

    async def _get_close_prices(self, conn, symbols: List[str], d) -> Dict[str, float]:
        if not symbols:
            return {}
        q = f"""
            SELECT symbol, close
            FROM {self.price_table}
            WHERE date = $1 AND symbol = ANY($2::text[])
        """
        rows = await conn.fetch(q, d, symbols)
        out = {r["symbol"]: float(r["close"]) for r in rows if r["close"] is not None}
        # Remember the latest close per symbol for carry-forward valuation.
        self.last_prices.update(out)
        return out

    async def _trading_days(self, conn) -> List[date]:
        q = f"""
            SELECT date
            FROM {self.price_table}
            WHERE date BETWEEN $1 AND $2
            GROUP BY date
            HAVING COUNT(*) >= $3
            ORDER BY date
        """
        rows = await conn.fetch(q, self.start_date, self.end_date,
                                MIN_ROWS_PER_TRADING_DAY)
        return [r["date"] for r in rows]

    async def _benchmark_prices(self, conn, days: List[date]) -> Dict[date, float]:
        """Buy-and-hold benchmark closes keyed by date (empty if unconfigured)."""
        if not self.benchmark_symbol or not self.benchmark_table:
            return {}
        q = f"""
            SELECT date, close FROM {self.benchmark_table}
            WHERE symbol = $1 AND date BETWEEN $2 AND $3
        """
        rows = await conn.fetch(q, self.benchmark_symbol, days[0], days[-1])
        return {r["date"]: float(r["close"]) for r in rows if r["close"] is not None}

    async def _rebalance(self, conn, d):
        target = await self._get_top_n_grades(conn, d)
        if not target:
            logger.debug(f"[{d}] no top-N candidates, skip rebalance")
            return

        all_syms = list(set(list(self.holdings.keys()) + target))
        prices = await self._get_close_prices(conn, all_syms, d)

        # Sell those no longer in target
        for sym in list(self.holdings.keys()):
            if sym in target:
                continue
            shares = self.holdings.pop(sym)
            price = prices.get(sym)
            if price is None:
                # No price available — leave shares but log
                logger.warning(f"[{d}] no price for {sym}, skip sell")
                self.holdings[sym] = shares
                continue
            sell_price = price * (1 - self.slippage_rate)
            gross = shares * sell_price
            commission = gross * self.commission_rate
            self.cash += gross - commission
            self.trades.append({
                "date": d, "symbol": sym, "action": "SELL",
                "shares": shares, "price": sell_price,
                "amount": gross, "commission": commission,
            })

        # Buy: equal-weight new entrants only (don't churn existing holdings)
        new_entries = [s for s in target if s not in self.holdings]
        if not new_entries:
            return

        per_stock_budget = self.cash / len(new_entries)
        for sym in new_entries:
            price = prices.get(sym)
            if price is None or price <= 0:
                logger.warning(f"[{d}] no price for {sym}, skip buy")
                continue
            buy_price = price * (1 + self.slippage_rate)
            cost_per_share = buy_price * (1 + self.commission_rate)
            shares = int(per_stock_budget / cost_per_share) if cost_per_share > 0 else 0
            if shares <= 0:
                continue
            gross = shares * buy_price
            commission = gross * self.commission_rate
            total = gross + commission
            if total > self.cash:
                continue
            self.cash -= total
            self.holdings[sym] = shares
            self.trades.append({
                "date": d, "symbol": sym, "action": "BUY",
                "shares": shares, "price": buy_price,
                "amount": gross, "commission": commission,
            })

    async def _compute_nav(self, conn, d) -> tuple:
        if not self.holdings:
            return self.cash, 0.0
        prices = await self._get_close_prices(conn, list(self.holdings.keys()), d)
        holdings_value = 0.0
        for sym, shares in self.holdings.items():
            # today's close → last known close → 0 (only if never priced)
            px = prices.get(sym)
            if px is None:
                px = self.last_prices.get(sym, 0.0)
            holdings_value += shares * px
        return self.cash + holdings_value, holdings_value

    async def run(self):
        async with self.pool.acquire() as conn:
            days = await self._trading_days(conn)
            if not days:
                raise ValueError(
                    f"No price data in {self.price_table} for "
                    f"{self.country} between {self.start_date}~{self.end_date}. "
                    f"Backfill data first via alphafolio_data."
                )

            logger.info(
                f"[backtest] simulating {len(days)} trading days "
                f"({days[0]} ~ {days[-1]}), top_n={self.top_n}, "
                f"rebal_every={self.rebal_freq_days}d"
            )

            bench_prices = await self._benchmark_prices(conn, days)
            bench_shares = None        # set on first priced day
            last_bench = None          # carry-forward benchmark close

            days_since_rebal = self.rebal_freq_days  # rebal on day 0

            for d in days:
                if days_since_rebal >= self.rebal_freq_days:
                    await self._rebalance(conn, d)
                    days_since_rebal = 0

                nav, holdings_value = await self._compute_nav(conn, d)

                # Buy-and-hold benchmark NAV (carry-forward if a close is missing)
                benchmark_value = None
                bp = bench_prices.get(d, last_bench)
                if bp is not None:
                    last_bench = bp
                    if bench_shares is None:
                        bench_shares = self.initial_cash / bp
                    benchmark_value = bench_shares * bp

                self.nav_history.append({
                    "date": d,
                    "nav": nav,
                    "cash": self.cash,
                    "holdings_value": holdings_value,
                    "holdings_count": len(self.holdings),
                    "benchmark_value": benchmark_value,
                })
                days_since_rebal += 1

        return self.nav_history, self.trades
