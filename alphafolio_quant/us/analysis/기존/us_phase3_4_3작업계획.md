# US Phase 3.4.3: NULL Column Resolution Work Plan

## Overview
- **Target**: us_stock_grade table NULL columns resolution (21 columns)
- **Date**: 2025-12-08
- **Reference**: KR system (kr_additional_metrics.py) as implementation guide

---

## Current Status Analysis

### NULL Column Summary (22 total)
| Category | Count | Columns |
|----------|-------|---------|
| INSERT missing | 21 | See detailed list below |
| INSERT exists but NULL | 1 | scenario_sample_count |

### Comparison: KR vs US Implementation
| Metric | KR (kr_additional_metrics.py) | US (us_agent_metrics.py) |
|--------|-------------------------------|--------------------------|
| Total methods | 25+ | 11 |
| Risk metrics | 10+ | 6 |
| Factor metrics | 5+ | 0 |
| Text generation | 4+ | 0 |

---

## Work Plan: 3-Phase Approach

### Phase 1: INSERT Addition (Simple Query-Based)
**Priority**: HIGH | **Effort**: LOW | **Files**: us_main_v2.py

#### 1.1 stock_name (From us_stock_basic)
- **Source**: `us_stock_basic.stock_name`
- **Method**: JOIN query in analysis flow
- **Implementation**:
  ```python
  # In analyze_single_stock(), add:
  stock_info_query = """
      SELECT stock_name, beta FROM us_stock_basic WHERE symbol = $1
  """
  stock_info = await self.db.fetch_one(stock_info_query, symbol)
  stock_name = stock_info['stock_name'] if stock_info else None
  beta = stock_info['beta'] if stock_info else None
  ```
- **INSERT Change**: Add `stock_name` column and value
- **Estimated Time**: 30 min

#### 1.2 beta (From us_stock_basic)
- **Source**: `us_stock_basic.beta`
- **Method**: Same query as stock_name
- **INSERT Change**: Add `beta` column and value
- **Estimated Time**: Included in 1.1

---

### Phase 2: UPDATE Query Enhancement
**Priority**: MEDIUM | **Effort**: LOW | **Files**: us_main_v2.py

#### 2.1 rs_rank (Already Implemented - Verify)
- **Current**: UPDATE query exists at us_main_v2.py:1333-1355
- **Issue**: Verify execution timing and data population
- **Action**: Check if rs_rank is being populated correctly
- **Estimated Time**: 15 min (verification only)

---

### Phase 3: New Implementation
**Priority**: HIGH | **Effort**: HIGH | **Files**: us_agent_metrics.py, us_main_v2.py

#### Category A: Risk Metrics (5 columns)
##### 3.A.1 volatility_annual
- **Definition**: Annualized volatility (%)
- **KR Reference**: `kr_additional_metrics.py:124-170`
- **Formula**: `std(daily_returns) * sqrt(252) * 100`
- **Data Source**: us_daily (60-day returns)
- **Implementation**:
  ```python
  async def calculate_volatility_annual(self, symbol: str, analysis_date: date) -> Optional[float]:
      query = """
          SELECT close FROM us_daily
          WHERE symbol = $1 AND date <= $2
          ORDER BY date DESC LIMIT 61
      """
      rows = await self.db.fetch(query, symbol, analysis_date)
      if len(rows) < 20:
          return None

      prices = [float(r['close']) for r in rows]
      returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]

      import numpy as np
      volatility = np.std(returns) * np.sqrt(252) * 100
      return round(volatility, 2)
  ```
- **Estimated Time**: 1 hour

##### 3.A.2 max_drawdown_1y
- **Definition**: Maximum drawdown in past 1 year (%)
- **KR Reference**: `kr_additional_metrics.py:990-1027`
- **Formula**: `min((price - running_max) / running_max) * 100`
- **Data Source**: us_daily (252 trading days)
- **Implementation**:
  ```python
  async def calculate_max_drawdown_1y(self, symbol: str, analysis_date: date) -> Optional[float]:
      query = """
          SELECT date, close FROM us_daily
          WHERE symbol = $1 AND date <= $2
          ORDER BY date DESC LIMIT 252
      """
      rows = await self.db.fetch(query, symbol, analysis_date)
      if len(rows) < 20:
          return None

      prices = [float(r['close']) for r in reversed(rows)]
      running_max = prices[0]
      max_drawdown = 0

      for price in prices:
          running_max = max(running_max, price)
          drawdown = (price - running_max) / running_max * 100
          max_drawdown = min(max_drawdown, drawdown)

      return round(max_drawdown, 2)
  ```
- **Estimated Time**: 1 hour

##### 3.A.3 sharpe_ratio
- **Definition**: Risk-adjusted return (Sharpe = (R - Rf) / std)
- **KR Reference**: `kr_additional_metrics.py:2123-2180`
- **Formula**: `(avg_return - risk_free_rate) / std(returns) * sqrt(252)`
- **Data Source**: us_daily + us_fed_funds_rate
- **Implementation**:
  ```python
  async def calculate_risk_adjusted_returns(self, symbol: str, analysis_date: date) -> Dict:
      # Get 60-day returns
      query = """
          SELECT close FROM us_daily
          WHERE symbol = $1 AND date <= $2
          ORDER BY date DESC LIMIT 61
      """
      rows = await self.db.fetch(query, symbol, analysis_date)

      # Get risk-free rate
      rf_query = """
          SELECT value FROM us_fed_funds_rate
          WHERE date <= $1 ORDER BY date DESC LIMIT 1
      """
      rf_row = await self.db.fetch_one(rf_query, analysis_date)
      rf_daily = (float(rf_row['value']) / 100 / 252) if rf_row else 0.0001

      prices = [float(r['close']) for r in rows]
      returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]

      import numpy as np
      avg_return = np.mean(returns)
      std_return = np.std(returns)

      # Sharpe Ratio
      sharpe = (avg_return - rf_daily) / std_return * np.sqrt(252) if std_return > 0 else None

      # Sortino Ratio (downside deviation only)
      downside_returns = [r for r in returns if r < 0]
      downside_std = np.std(downside_returns) if downside_returns else std_return
      sortino = (avg_return - rf_daily) / downside_std * np.sqrt(252) if downside_std > 0 else None

      # Calmar Ratio
      max_dd = await self.calculate_max_drawdown_1y(symbol, analysis_date)
      calmar = (avg_return * 252 * 100) / abs(max_dd) if max_dd and max_dd < 0 else None

      return {
          'sharpe_ratio': round(sharpe, 4) if sharpe else None,
          'sortino_ratio': round(sortino, 4) if sortino else None,
          'calmar_ratio': round(calmar, 4) if calmar else None
      }
  ```
- **Estimated Time**: 2 hours

##### 3.A.4 sortino_ratio
- **Included in**: 3.A.3 sharpe_ratio implementation

##### 3.A.5 calmar_ratio
- **Included in**: 3.A.3 sharpe_ratio implementation

---

#### Category B: Factor Momentum (4 columns)
##### 3.B.1 ~ 3.B.4: value_momentum, quality_momentum, momentum_momentum, growth_momentum
- **Definition**: Factor score change vs previous month
- **KR Reference**: `kr_additional_metrics.py:891-928`
- **Formula**: `current_score - prev_month_score`
- **Data Source**: us_stock_grade (historical)
- **Implementation**:
  ```python
  async def calculate_factor_momentum(self, symbol: str, analysis_date: date,
                                       current_scores: Dict) -> Dict[str, Optional[float]]:
      # Get 1 month ago scores
      prev_date = analysis_date - timedelta(days=30)
      query = """
          SELECT value_score, quality_score, momentum_score, growth_score
          FROM us_stock_grade
          WHERE symbol = $1 AND date <= $2
          ORDER BY date DESC LIMIT 1
      """
      prev_row = await self.db.fetch_one(query, symbol, prev_date)

      if not prev_row:
          return {
              'value_momentum': None,
              'quality_momentum': None,
              'momentum_momentum': None,
              'growth_momentum': None
          }

      return {
          'value_momentum': round(current_scores['value'] - float(prev_row['value_score']), 1),
          'quality_momentum': round(current_scores['quality'] - float(prev_row['quality_score']), 1),
          'momentum_momentum': round(current_scores['momentum'] - float(prev_row['momentum_score']), 1),
          'growth_momentum': round(current_scores['growth'] - float(prev_row['growth_score']), 1)
      }
  ```
- **Dependency**: Requires historical us_stock_grade data
- **Estimated Time**: 1.5 hours

---

#### Category C: Industry Metrics (2 columns)
##### 3.C.1 ~ 3.C.2: industry_rank, industry_percentile
- **Definition**: Rank within same industry by final_score
- **KR Reference**: `kr_additional_metrics.py:930-988`
- **Formula**: RANK() OVER (PARTITION BY industry)
- **Data Source**: us_stock_grade + us_stock_basic
- **Implementation Option A**: Per-stock query
  ```python
  async def calculate_industry_ranking(self, symbol: str, analysis_date: date,
                                        final_score: float) -> Tuple[Optional[int], Optional[float]]:
      query = """
          WITH industry_stocks AS (
              SELECT g.symbol, g.final_score,
                     RANK() OVER (ORDER BY g.final_score DESC) as rank,
                     COUNT(*) OVER () as total
              FROM us_stock_grade g
              JOIN us_stock_basic b ON g.symbol = b.symbol
              WHERE g.date = $1
                AND b.industry = (SELECT industry FROM us_stock_basic WHERE symbol = $2)
          )
          SELECT rank, total,
                 ROUND(100.0 * rank / NULLIF(total, 0), 1) as percentile
          FROM industry_stocks
          WHERE symbol = $2
      """
      row = await self.db.fetch_one(query, analysis_date, symbol)

      if not row:
          return None, None

      return int(row['rank']), float(row['percentile'])
  ```
- **Implementation Option B**: Batch UPDATE after all inserts (recommended)
  ```python
  async def update_industry_rankings(self, analysis_date: date):
      query = """
          WITH rankings AS (
              SELECT g.symbol,
                     RANK() OVER (PARTITION BY b.industry ORDER BY g.final_score DESC) as industry_rank,
                     PERCENT_RANK() OVER (PARTITION BY b.industry ORDER BY g.final_score DESC) * 100 as industry_percentile
              FROM us_stock_grade g
              JOIN us_stock_basic b ON g.symbol = b.symbol
              WHERE g.date = $1
          )
          UPDATE us_stock_grade g
          SET industry_rank = r.industry_rank,
              industry_percentile = r.industry_percentile
          FROM rankings r
          WHERE g.symbol = r.symbol AND g.date = $1
      """
      await self.db.execute_query(query, analysis_date)
  ```
- **Estimated Time**: 2 hours

---

#### Category D: Sector Metrics (4 columns)
##### 3.D.1 ~ 3.D.4: sector_rotation_score, sector_momentum, sector_rank, sector_percentile
- **Definition**: Sector-level performance metrics
- **KR Reference**: `kr_additional_metrics.py:1988-2120`
- **Data Source**: mv_us_sector_daily_performance + us_stock_basic
- **Implementation**:
  ```python
  async def calculate_sector_rotation_signal(self, symbol: str, analysis_date: date) -> Dict:
      # Get stock's sector
      sector_query = "SELECT sector FROM us_stock_basic WHERE symbol = $1"
      sector_row = await self.db.fetch_one(sector_query, symbol)
      if not sector_row:
          return {'sector_rotation_score': None, 'sector_momentum': None,
                  'sector_rank': None, 'sector_percentile': None}

      sector = sector_row['sector']

      # Get sector performance
      perf_query = """
          SELECT sector_code, avg_return_30d as momentum, sector_rank,
                 PERCENT_RANK() OVER (ORDER BY avg_return_30d DESC) * 100 as percentile
          FROM mv_us_sector_daily_performance
          WHERE date = $1 AND sector_code = $2
      """
      perf = await self.db.fetch_one(perf_query, analysis_date, sector)

      if not perf:
          return {'sector_rotation_score': None, 'sector_momentum': None,
                  'sector_rank': None, 'sector_percentile': None}

      # Calculate rotation score (0-100)
      momentum = float(perf['momentum']) if perf['momentum'] else 0
      percentile = float(perf['percentile']) if perf['percentile'] else 50

      rotation_score = int(min(100, max(0, 50 + momentum * 10 + (50 - percentile) * 0.5)))

      return {
          'sector_rotation_score': rotation_score,
          'sector_momentum': round(momentum, 2),
          'sector_rank': int(perf['sector_rank']) if perf['sector_rank'] else None,
          'sector_percentile': round(percentile, 2)
      }
  ```
- **Estimated Time**: 2 hours

---

#### Category E: Score Metrics (2 columns)
##### 3.E.1 confidence_score
- **Definition**: How aligned are the 4 factor scores (0-100)
- **KR Reference**: `kr_additional_metrics.py:79-122`
- **Formula**: `100 - (std(4 scores) / 25 * 100)`
- **Implementation**:
  ```python
  def calculate_confidence_score(self, value: float, quality: float,
                                  momentum: float, growth: float) -> float:
      import numpy as np
      scores = [value, quality, momentum, growth]
      std_dev = np.std(scores)

      # Lower std = higher confidence (scores agree)
      # Max std is ~35 when scores are at extremes
      confidence = max(0, 100 - (std_dev / 35 * 100))
      return round(confidence, 1)
  ```
- **Estimated Time**: 30 min

##### 3.E.2 factor_combination_bonus
- **Definition**: Bonus/penalty for proven factor combinations
- **KR Reference**: `kr_additional_metrics.py:1649-1739`
- **Patterns**:
  - High Value + High Quality + Low Momentum = +20 (Turnaround)
  - High Quality + High Growth = +15 (Compounders)
  - Low Value + High Momentum = -10 (Speculation)
- **Implementation**:
  ```python
  def calculate_factor_combination_bonus(self, value: float, quality: float,
                                          momentum: float, growth: float) -> int:
      bonus = 0

      # Pattern 1: Quality Compounder (Q>70, G>70)
      if quality >= 70 and growth >= 70:
          bonus += 15

      # Pattern 2: Deep Value Turnaround (V>75, Q>60, M<40)
      if value >= 75 and quality >= 60 and momentum < 40:
          bonus += 20

      # Pattern 3: Growth at Reasonable Price (G>70, V>50)
      if growth >= 70 and value >= 50:
          bonus += 10

      # Pattern 4: Momentum Speculation Penalty (V<30, M>80)
      if value < 30 and momentum > 80:
          bonus -= 10

      # Pattern 5: All-around Strong (all >65)
      if all(s >= 65 for s in [value, quality, momentum, growth]):
          bonus += 15

      return max(-20, min(30, bonus))  # Clamp to [-20, +30]
  ```
- **Estimated Time**: 1 hour

---

#### Category F: Text Generation (5 columns)
##### 3.F.1 ~ 3.F.5: risk_profile_text, risk_recommendation, time_series_text, signal_overall, market_state
- **Definition**: Human-readable analysis text
- **KR Reference**: Various text generation patterns
- **Implementation**:
  ```python
  def generate_analysis_texts(self, metrics: Dict) -> Dict[str, str]:
      # risk_profile_text
      vol = metrics.get('volatility_annual', 50)
      mdd = metrics.get('max_drawdown_1y', -20)
      beta = metrics.get('beta', 1.0)

      if vol > 50 or mdd < -30 or beta > 1.5:
          risk_profile = "High Risk - High volatility and significant drawdown potential"
      elif vol > 30 or mdd < -20 or beta > 1.2:
          risk_profile = "Moderate Risk - Average market volatility exposure"
      else:
          risk_profile = "Low Risk - Below average volatility and drawdown"

      # risk_recommendation
      if vol > 50:
          risk_rec = "Suitable for aggressive investors with high risk tolerance"
      elif vol > 30:
          risk_rec = "Suitable for balanced investors seeking moderate growth"
      else:
          risk_rec = "Suitable for conservative investors prioritizing capital preservation"

      # time_series_text
      score_trend = metrics.get('score_trend_2w', 0)
      if score_trend > 5:
          time_series = "Improving - Score trending up over past 2 weeks"
      elif score_trend < -5:
          time_series = "Declining - Score trending down over past 2 weeks"
      else:
          time_series = "Stable - Score relatively unchanged"

      # signal_overall
      final_score = metrics.get('final_score', 50)
      if final_score >= 80:
          signal = "Strong Buy - Exceptional multi-factor alignment"
      elif final_score >= 65:
          signal = "Buy - Favorable risk-reward profile"
      elif final_score >= 50:
          signal = "Hold - Neutral outlook"
      elif final_score >= 35:
          signal = "Reduce - Below average fundamentals"
      else:
          signal = "Sell - Significant downside risk"

      # market_state (from regime data)
      regime = metrics.get('regime', 'NORMAL')
      vix = metrics.get('vix', 20)
      market_state = f"{regime}-VIX{int(vix)}"

      return {
          'risk_profile_text': risk_profile,
          'risk_recommendation': risk_rec,
          'time_series_text': time_series,
          'signal_overall': signal,
          'market_state': market_state
      }
  ```
- **Estimated Time**: 2 hours

---

## Implementation Order (Recommended)

### Stage 1: Quick Wins (Day 1)
1. **Phase 1.1-1.2**: stock_name, beta (INSERT addition) - 30 min
2. **Phase 2.1**: rs_rank verification - 15 min
3. **Phase 3.E.1**: confidence_score - 30 min
4. **Phase 3.E.2**: factor_combination_bonus - 1 hour

**Total Stage 1**: ~2.5 hours

### Stage 2: Risk Metrics (Day 2)
5. **Phase 3.A.1**: volatility_annual - 1 hour
6. **Phase 3.A.2**: max_drawdown_1y - 1 hour
7. **Phase 3.A.3-5**: sharpe/sortino/calmar_ratio - 2 hours

**Total Stage 2**: ~4 hours

### Stage 3: Factor & Ranking (Day 3)
8. **Phase 3.B.1-4**: factor_momentum (4 columns) - 1.5 hours
9. **Phase 3.C.1-2**: industry_rank, industry_percentile - 2 hours

**Total Stage 3**: ~3.5 hours

### Stage 4: Sector & Text (Day 4)
10. **Phase 3.D.1-4**: sector metrics (4 columns) - 2 hours
11. **Phase 3.F.1-5**: text generation (5 columns) - 2 hours

**Total Stage 4**: ~4 hours

---

## File Changes Summary

### Files to Modify
| File | Changes |
|------|---------|
| `us_agent_metrics.py` | Add 10+ new calculation methods |
| `us_main_v2.py` | Update INSERT (add 21 columns), call new methods |

### New Methods in us_agent_metrics.py
```
+ calculate_volatility_annual()
+ calculate_max_drawdown_1y()
+ calculate_risk_adjusted_returns()  # returns sharpe, sortino, calmar
+ calculate_factor_momentum()        # returns 4 momentum values
+ calculate_industry_ranking()       # returns rank, percentile
+ calculate_sector_rotation_signal() # returns 4 sector values
+ calculate_confidence_score()
+ calculate_factor_combination_bonus()
+ generate_analysis_texts()          # returns 5 text values
```

### INSERT Statement Changes (us_main_v2.py)
Current: 43 columns
After: 64 columns (+21)

---

## Testing Strategy

### Unit Tests
1. Each new method with mock data
2. Edge cases: missing data, insufficient history, NULL values

### Integration Tests
1. Single stock analysis with all new columns
2. Verify no NULL values in output
3. Compare with KR output format

### Validation Queries
```sql
-- Check NULL columns after implementation
SELECT
    'stock_name' as column_name, COUNT(*) - COUNT(stock_name) as null_count,
    COUNT(*) as total_count
FROM us_stock_grade WHERE date = '2025-12-08'
UNION ALL
SELECT 'beta', COUNT(*) - COUNT(beta), COUNT(*) FROM us_stock_grade WHERE date = '2025-12-08'
-- ... repeat for all 21 columns
```

---

## Risk Assessment

### Low Risk
- stock_name, beta: Simple query addition
- confidence_score, factor_combination_bonus: Pure calculation

### Medium Risk
- volatility_annual, max_drawdown_1y: Requires sufficient price history
- factor_momentum: Requires historical grade data (may be NULL initially)

### High Risk
- industry_rank: Performance concern with 5000+ stocks
- sector_rotation_score: Depends on MV data availability

---

## Dependencies

### Data Prerequisites
1. `us_stock_basic`: stock_name, beta, industry, sector
2. `us_daily`: price history (60+ days)
3. `us_stock_grade`: historical scores (30+ days)
4. `mv_us_sector_daily_performance`: sector metrics
5. `us_fed_funds_rate`: risk-free rate

### External Dependencies
- numpy: For statistical calculations

---

## Rollback Plan

1. Keep backup of current us_main_v2.py and us_agent_metrics.py
2. New columns are nullable - can revert INSERT changes
3. UPDATE queries can be re-run after fixes

---

## Success Criteria

1. All 21 NULL columns populated for new analysis runs
2. No performance degradation (< 10% slower)
3. NULL rate < 5% for columns with sufficient data
4. KR-US output format consistency

---

## Notes

- Prioritize data-driven metrics over text generation
- Text generation can use simpler rules initially
- factor_momentum will have NULLs until historical data accumulates
- Consider adding Phase 3.4.4 for any remaining gaps

---

## Implementation Status (2025-12-09)

### ✅ COMPLETED - All 4 Stages

| Stage | Description | Status | Columns |
|-------|-------------|--------|---------|
| Stage 1 | Quick Wins | ✅ Complete | stock_name, beta, confidence_score, factor_combination_bonus |
| Stage 2 | Risk Metrics | ✅ Complete | volatility_annual, max_drawdown_1y, sharpe_ratio, sortino_ratio, calmar_ratio |
| Stage 3 | Factor & Ranking | ✅ Complete | value_momentum, quality_momentum, momentum_momentum, growth_momentum, industry_rank, industry_percentile |
| Stage 4 | Sector & Text | ✅ Complete | sector_rotation_score, sector_momentum, sector_rank, sector_percentile, risk_profile_text, risk_recommendation, time_series_text, signal_overall, market_state |

### Implementation Summary

**Files Modified:**
- `us/us_main_v2.py`:
  - Added `_calculate_factor_momentum()` method
  - Added `_calculate_sector_metrics()` method
  - Added `_generate_analysis_texts()` method
  - Updated `_analyze_stock()` to call new methods
  - Updated INSERT statement: 52 → 65 columns (+13)
  - Added industry_rank batch UPDATE after RS Rank calculation

**Total New Columns: 15**
- Factor Momentum: 4 columns (value_momentum, quality_momentum, momentum_momentum, growth_momentum)
- Sector Metrics: 4 columns (sector_rotation_score, sector_momentum, sector_rank, sector_percentile)
- Text Generation: 5 columns (risk_profile_text, risk_recommendation, time_series_text, signal_overall, market_state)
- Industry Ranking: 2 columns (industry_rank, industry_percentile - batch UPDATE)

**Key Design Decisions:**
1. Factor momentum uses historical us_stock_grade data (30 days back)
2. Sector metrics query mv_us_sector_daily_performance with fallback to previous days
3. Industry ranking calculated via batch UPDATE after all INSERTs (follows RS Rank pattern)
4. Text generation uses rule-based approach based on existing metrics
