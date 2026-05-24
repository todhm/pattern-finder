"""Backtest module hosted inside alphafolio_quant FastAPI app.

Phases:
- generate_grades_for_range(): loop run_option1(target_date) to populate
  us_stock_grade / kr_stock_grade for past dates (slow, hours).
- PortfolioSimulator: replay top-N grades into a virtual portfolio,
  produce daily NAV history.
- metrics.calculate_metrics(): Sharpe / Sortino / MDD / Calmar / CAGR.
- storage: persist results to alphafolio DB (backtest_runs /
  backtest_nav_history / backtest_trades tables, auto-created).
"""
