# US Stock Analysis Query Optimization Design

## Overview
- **Date**: 2025-12-08
- **Target**: Reduce DB queries from 35/stock to ~20/stock (43% reduction)
- **Scope**: us_main_v2.py option 2 full stock analysis flow

---

## Current State Analysis

### Query Count by Table (per stock)

| Table | Current Queries | Optimized | Reduction |
|-------|----------------|-----------|-----------|
| us_daily | 8 | 1 | -7 |
| us_stock_basic | 5 | 1 | -4 |
| us_income_statement | 3 | 1 | -2 |
| us_cash_flow | 2 | 1 | -1 |
| us_earnings_estimates | 2 | 1 | -1 |
| us_option_daily_summary | 2 | 1 | -1 |
| us_balance_sheet | 1 | 1 | 0 |
| us_atr | 1 | 1 | 0 |
| us_stock_grade | 1 | 1 | 0 |
| us_earnings_calendar | 1 | 1 | 0 |
| us_insider_transactions | 1 | 1 | 0 |
| us_news | 1 | 1 | 0 |
| **Total** | **35** | **~20** | **-15** |

---

## Optimization Architecture

### New Data Flow

```
_analyze_stock(symbol, analysis_date)
│
├── Phase 1: Prefetch Common Data (parallel)
│   ├── _prefetch_stock_basic()      → stock_basic_data
│   ├── _prefetch_daily_prices()     → price_data (260 days)
│   ├── _prefetch_financials()       → financial_data
│   └── _prefetch_estimates()        → estimate_data
│
├── Phase 2: Calculate Factors (parallel, using prefetched data)
│   ├── calculate_value_score(prefetched_data)
│   ├── calculate_quality_score(prefetched_data)
│   ├── calculate_momentum_score(prefetched_data)
│   ├── calculate_growth_score(prefetched_data)
│   ├── _calculate_var_cvar_from_data(price_data)
│   ├── _calculate_rs_value_from_data(price_data)
│   ├── _calculate_ors_from_data(price_data, stock_basic_data)
│   └── _calculate_risk_metrics_from_data(price_data)  # NEW
│
├── Phase 3: Event/Agent Metrics (individual queries)
│   ├── event_engine.calculate_event_modifier()
│   └── agent_metrics.calculate_all()
│
└── Phase 4: Save Results
    └── _save_stock_grade()
```

---

## Implementation Plan

### Phase 1: Prefetch Infrastructure

#### 1.1 New Class: USDataPrefetcher

```python
class USDataPrefetcher:
    """
    Centralized data prefetching for stock analysis.
    Eliminates redundant DB queries by loading all needed data upfront.
    """

    def __init__(self, db: AsyncDatabaseManager):
        self.db = db
        self._cache = {}

    async def prefetch_all(self, symbol: str, analysis_date: date) -> Dict:
        """
        Prefetch all common data in parallel.

        Returns:
            {
                'stock_basic': {...},      # us_stock_basic row
                'price_data': {...},       # 260 days OHLCV
                'financials': {...},       # income_statement + cash_flow + balance_sheet
                'estimates': {...},        # earnings estimates
                'options': {...}           # options summary
            }
        """
        tasks = [
            self._prefetch_stock_basic(symbol),
            self._prefetch_daily_prices(symbol, analysis_date),
            self._prefetch_financials(symbol),
            self._prefetch_estimates(symbol),
            self._prefetch_options(symbol, analysis_date)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            'stock_basic': results[0] if not isinstance(results[0], Exception) else None,
            'price_data': results[1] if not isinstance(results[1], Exception) else None,
            'financials': results[2] if not isinstance(results[2], Exception) else None,
            'estimates': results[3] if not isinstance(results[3], Exception) else None,
            'options': results[4] if not isinstance(results[4], Exception) else None
        }
```

#### 1.2 Prefetch Queries

**_prefetch_stock_basic()**
```sql
SELECT
    symbol, stock_name, sector, industry, exchange, market_cap,
    -- Valuation
    per, forwardpe, peg, pricetosalesratiottm, evtorevenue, evtoebitda,
    pricetobookratio, trailingpe,
    -- Growth
    quarterlyrevenuegrowthyoy, quarterlyearningsgrowthyoy,
    -- Quality
    grossprofitttm, revenuettm, operatingmarginttm, profitmargin,
    returnonequityttm, returnonassetsttm, ebitda,
    -- Technical
    beta, week52high, week52low, day50movingaverage, day200movingaverage,
    -- Shares
    sharesoutstanding, sharesfloat, bookvalue, dilutedepsttm,
    -- Analyst
    analysttargetprice, analystratingstrongbuy, analystratingbuy,
    analystratinghold, analystratingsell, analystratingstrongsell
FROM us_stock_basic
WHERE symbol = $1
```

**_prefetch_daily_prices()**
```sql
SELECT date, open, high, low, close, volume
FROM us_daily
WHERE symbol = $1 AND date <= $2
ORDER BY date DESC
LIMIT 260
```

**_prefetch_financials()**
```sql
-- Income Statement (12 quarters)
SELECT 'income' as source, fiscal_date_ending,
       total_revenue, gross_profit, operating_income, net_income,
       research_and_development, operating_expenses,
       depreciation_and_amortization, ebitda
FROM us_income_statement
WHERE symbol = $1
ORDER BY fiscal_date_ending DESC
LIMIT 12;

-- Cash Flow (4 quarters)
SELECT 'cashflow' as source, fiscal_date_ending,
       operating_cashflow, capital_expenditures, net_income,
       depreciation_depletion_and_amortization
FROM us_cash_flow
WHERE symbol = $1
ORDER BY fiscal_date_ending DESC
LIMIT 4;

-- Balance Sheet (latest)
SELECT 'balance' as source, fiscal_date_ending,
       total_assets, total_liabilities, total_shareholder_equity,
       total_current_assets, total_current_liabilities,
       long_term_debt, short_term_debt, short_long_term_debt_total,
       cash_and_cash_equivalents_at_carrying_value
FROM us_balance_sheet
WHERE symbol = $1
ORDER BY fiscal_date_ending DESC
LIMIT 1
```

**_prefetch_estimates()**
```sql
SELECT estimate_date, horizon,
       eps_estimate_average, eps_estimate_high, eps_estimate_low,
       eps_estimate_average_7_days_ago, eps_estimate_average_30_days_ago,
       eps_estimate_revision_up_trailing_7_days,
       eps_estimate_revision_down_trailing_7_days,
       revenue_estimate_average
FROM us_earnings_estimates
WHERE symbol = $1
  AND horizon IN ('next fiscal quarter', 'next fiscal year')
  AND eps_estimate_average IS NOT NULL
ORDER BY estimate_date DESC
LIMIT 2
```

**_prefetch_options()**
```sql
SELECT date, total_call_volume, total_put_volume,
       avg_call_iv, avg_put_iv, avg_implied_volatility,
       net_gex, call_gex, put_gex, gex_ratio
FROM us_option_daily_summary
WHERE symbol = $1 AND date <= $2
ORDER BY date DESC
LIMIT 1
```

---

### Phase 2: Factor Module Updates

#### 2.1 Interface Changes

Each factor module needs a new entry point that accepts prefetched data:

```python
# us_value_factor_v2.py
async def calculate_value_score(
    symbol: str,
    db: AsyncDatabaseManager,
    benchmarks: Dict,
    analysis_date: date,
    volatility_engine: Optional[USVolatilityAdjustment] = None,
    prefetched_data: Optional[Dict] = None  # NEW PARAMETER
) -> Dict:
    """
    If prefetched_data is provided, use it instead of querying DB.
    Falls back to DB query if prefetched_data is None (backward compatibility).
    """
    if prefetched_data:
        stock_data = prefetched_data['stock_basic']
        price_data = prefetched_data['price_data']
        financial_data = prefetched_data['financials']
    else:
        # Legacy: query DB directly
        stock_data = await _load_stock_data(symbol, db, analysis_date)
        ...
```

#### 2.2 Data Extraction Helpers

```python
class PrefetchedDataExtractor:
    """Helper class to extract specific data from prefetched results"""

    @staticmethod
    def get_closes(price_data: Dict, days: int = 30) -> List[float]:
        """Extract close prices for specified days"""
        return price_data['closes'][:days]

    @staticmethod
    def get_returns(price_data: Dict, days: int = 252) -> List[float]:
        """Calculate daily returns from close prices"""
        closes = price_data['closes'][:days+1]
        return [(closes[i] - closes[i+1]) / closes[i+1]
                for i in range(len(closes)-1)]

    @staticmethod
    def get_ttm_cashflow(financials: Dict) -> Dict:
        """Get trailing 12 months cash flow data"""
        cf = financials.get('cashflow', [])
        if len(cf) < 4:
            return None
        return {
            'operating_cashflow': sum(q['operating_cashflow'] for q in cf[:4]),
            'capital_expenditures': sum(q['capital_expenditures'] for q in cf[:4])
        }

    @staticmethod
    def get_52w_range(price_data: Dict) -> Tuple[float, float]:
        """Get 52-week high and low"""
        highs = price_data['highs'][:252]
        lows = price_data['lows'][:252]
        return max(highs), min(lows)
```

---

### Phase 3: Risk Metrics Implementation

#### 3.1 New Methods in us_main_v2.py

```python
def _calculate_risk_metrics_from_data(self, price_data: Dict) -> Dict:
    """
    Calculate all risk metrics from prefetched price data.

    Metrics:
        - volatility_annual: Annualized volatility (60-day)
        - max_drawdown_1y: Maximum drawdown (252-day)
        - sharpe_ratio: Risk-adjusted return
        - sortino_ratio: Downside risk-adjusted return
        - calmar_ratio: Return / Max Drawdown
    """
    closes = price_data['closes']

    if len(closes) < 60:
        return {
            'volatility_annual': None,
            'max_drawdown_1y': None,
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'calmar_ratio': None
        }

    # Calculate returns (60 days for volatility, 252 for MDD)
    returns_60d = [(closes[i] - closes[i+1]) / closes[i+1]
                   for i in range(min(60, len(closes)-1))]

    # 1. Volatility (annualized)
    volatility_annual = np.std(returns_60d) * np.sqrt(252) * 100

    # 2. Max Drawdown (1 year)
    closes_1y = closes[:min(252, len(closes))][::-1]  # oldest first
    running_max = closes_1y[0]
    max_drawdown = 0
    for price in closes_1y:
        running_max = max(running_max, price)
        drawdown = (price - running_max) / running_max * 100
        max_drawdown = min(max_drawdown, drawdown)

    # 3. Risk-Free Rate (approximate 4.5% annual = 0.0178% daily)
    rf_daily = 0.045 / 252

    # 4. Sharpe Ratio
    avg_return = np.mean(returns_60d)
    std_return = np.std(returns_60d)
    sharpe_ratio = (avg_return - rf_daily) / std_return * np.sqrt(252) if std_return > 0 else None

    # 5. Sortino Ratio (downside deviation)
    downside_returns = [r for r in returns_60d if r < 0]
    downside_std = np.std(downside_returns) if downside_returns else std_return
    sortino_ratio = (avg_return - rf_daily) / downside_std * np.sqrt(252) if downside_std > 0 else None

    # 6. Calmar Ratio
    annual_return = avg_return * 252 * 100
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else None

    return {
        'volatility_annual': round(volatility_annual, 2),
        'max_drawdown_1y': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe_ratio, 4) if sharpe_ratio else None,
        'sortino_ratio': round(sortino_ratio, 4) if sortino_ratio else None,
        'calmar_ratio': round(calmar_ratio, 4) if calmar_ratio else None
    }
```

---

### Phase 4: Updated _analyze_stock Flow

```python
async def _analyze_stock(self, symbol: str, analysis_date: date) -> Optional[Dict]:
    """
    Optimized stock analysis with prefetched data.
    Reduces DB queries from 35 to ~20 per stock.
    """

    # ============================================================
    # Phase 1: Prefetch all common data (5 parallel queries)
    # ============================================================
    prefetcher = USDataPrefetcher(self.db)
    prefetched = await prefetcher.prefetch_all(symbol, analysis_date)

    # Validate required data
    if not prefetched['stock_basic'] or not prefetched['price_data']:
        return None

    stock_info = prefetched['stock_basic']
    price_data = prefetched['price_data']

    sector = stock_info.get('sector')
    exchange = stock_info.get('exchange', 'NYSE')
    if not sector:
        return None

    # ============================================================
    # Phase 2: Create shared volatility engine from prefetched data
    # ============================================================
    volatility_engine = self._create_volatility_engine_from_data(price_data)

    # ============================================================
    # Phase 3: Calculate all factors in parallel (using prefetched data)
    # ============================================================
    benchmarks = await self._get_sector_benchmarks(sector)

    # Factor calculations with prefetched data
    value_task = calculate_value_score(
        symbol, self.db, benchmarks, analysis_date,
        volatility_engine, prefetched_data=prefetched
    )
    quality_task = calculate_quality_score(
        symbol, self.db, benchmarks, analysis_date,
        volatility_engine, prefetched_data=prefetched
    )
    momentum_task = calculate_momentum_score(
        symbol, self.db, benchmarks, analysis_date, exchange,
        volatility_engine, prefetched_data=prefetched
    )
    growth_task = calculate_growth_score(
        symbol, self.db, benchmarks, analysis_date, exchange,
        volatility_engine, prefetched_data=prefetched
    )

    # These use prefetched price_data directly (no DB query)
    var_cvar_result = self._calculate_var_cvar_from_data(price_data)
    rs_value = self._calculate_rs_value_from_data(price_data)
    ors_result = self._calculate_ors_from_data(
        symbol, price_data, stock_info.get('market_cap')
    )
    risk_metrics = self._calculate_risk_metrics_from_data(price_data)

    # Event engine still needs individual queries
    event_task = self.event_engine.calculate_event_modifier(symbol, analysis_date)

    # Gather factor results
    value_result, quality_result, momentum_result, growth_result, event_result = \
        await asyncio.gather(
            value_task, quality_task, momentum_task, growth_task, event_task
        )

    # ... rest of scoring logic ...

    # ============================================================
    # Phase 4: Agent Metrics (still needs some individual queries)
    # ============================================================
    agent_metrics = USAgentMetrics(self.db)
    agent_result = await agent_metrics.calculate_all(
        symbol=symbol,
        analysis_date=analysis_date,
        current_score=total_score,
        sector=sector,
        var_95=var_cvar_result.get('var_95'),
        prefetched_data=prefetched  # Pass for optimization
    )

    # ============================================================
    # Phase 5: Build result with new risk metrics
    # ============================================================
    return {
        'symbol': symbol,
        'stock_name': stock_info.get('stock_name'),
        'beta': stock_info.get('beta'),
        # ... existing fields ...

        # NEW: Risk Metrics from Phase 3.4.3
        'volatility_annual': risk_metrics.get('volatility_annual'),
        'max_drawdown_1y': risk_metrics.get('max_drawdown_1y'),
        'sharpe_ratio': risk_metrics.get('sharpe_ratio'),
        'sortino_ratio': risk_metrics.get('sortino_ratio'),
        'calmar_ratio': risk_metrics.get('calmar_ratio'),
    }
```

---

## File Changes Summary

### New Files
| File | Description |
|------|-------------|
| `us_data_prefetcher.py` | Centralized data prefetching class |

### Modified Files
| File | Changes |
|------|---------|
| `us_main_v2.py` | Add prefetch calls, new risk metrics, updated _analyze_stock flow |
| `us_value_factor_v2.py` | Add prefetched_data parameter, conditional DB query |
| `us_quality_factor_v2.py` | Add prefetched_data parameter, conditional DB query |
| `us_momentum_factor_v2.py` | Add prefetched_data parameter, conditional DB query |
| `us_growth_factor_v2.py` | Add prefetched_data parameter, conditional DB query |
| `us_agent_metrics.py` | Add prefetched_data parameter for optimization |

---

## Implementation Phases

### Phase 1: Core Infrastructure (Day 1)
1. Create `us_data_prefetcher.py` with all prefetch methods
2. Add `_calculate_risk_metrics_from_data()` to us_main_v2.py
3. Add `_calculate_var_cvar_from_data()` to us_main_v2.py
4. Add `_calculate_rs_value_from_data()` to us_main_v2.py
5. Add `_calculate_ors_from_data()` to us_main_v2.py

### Phase 2: Factor Integration (Day 2)
1. Update `us_value_factor_v2.py` with prefetched_data support
2. Update `us_quality_factor_v2.py` with prefetched_data support
3. Update `us_momentum_factor_v2.py` with prefetched_data support
4. Update `us_growth_factor_v2.py` with prefetched_data support

### Phase 3: Main Flow Update (Day 3)
1. Update `_analyze_stock()` with new flow
2. Update `us_agent_metrics.py` with prefetched_data support
3. Update INSERT statement (add 5 new columns)
4. Testing and validation

---

## Validation Plan

### Query Count Verification
```python
# Add query counter decorator for testing
import functools

query_count = 0

def count_queries(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        global query_count
        query_count += 1
        return await func(*args, **kwargs)
    return wrapper
```

### Expected Results
```
Before optimization:
  - Queries per stock: ~35
  - Time per stock: ~800ms
  - Total for 5000 stocks: ~67 minutes

After optimization:
  - Queries per stock: ~20
  - Time per stock: ~500ms (estimated)
  - Total for 5000 stocks: ~42 minutes
  - Improvement: 37% faster
```

---

## Rollback Plan

1. All changes are additive (new parameters with defaults)
2. Original methods preserved with `_from_data` suffix variants
3. `prefetched_data=None` falls back to legacy behavior
4. Can disable prefetch by not passing prefetched_data parameter
