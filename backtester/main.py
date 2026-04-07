from datetime import date

from fastapi import FastAPI, Query

from backtest.services.engine import BacktestEngine
from data.services.market_data_service import MarketDataService
from pattern.services.reversal_extension import ReversalExtensionDetector
from pattern.services.wedge_pop import WedgePopDetector

app = FastAPI(title="Backtester", version="0.1.0")

DETECTORS = {
    "reversal_extension": ReversalExtensionDetector(),
    "wedge_pop": WedgePopDetector(),
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patterns")
def list_patterns():
    return {"patterns": list(DETECTORS.keys())}


@app.get("/detect/{pattern_name}")
def detect_pattern(
    pattern_name: str,
    symbol: str = Query(...),
    start: date = Query(...),
    end: date = Query(...),
):
    detector = DETECTORS.get(pattern_name)
    if detector is None:
        return {"error": f"Unknown pattern: {pattern_name}"}

    df = MarketDataService().fetch_ohlcv(symbol, start, end)
    signals = detector.detect(df)
    return {"symbol": symbol, "pattern": pattern_name, "signals": signals}


@app.get("/backtest/{pattern_name}")
def run_backtest(
    pattern_name: str,
    symbol: str = Query(...),
    start: date = Query(...),
    end: date = Query(...),
    initial_capital: float = Query(100_000),
):
    detector = DETECTORS.get(pattern_name)
    if detector is None:
        return {"error": f"Unknown pattern: {pattern_name}"}

    data_svc = MarketDataService()
    df = data_svc.fetch_ohlcv(symbol, start, end)
    signals = detector.detect(df)

    engine = BacktestEngine(initial_capital=initial_capital)
    result = engine.run(df, signals)
    return {"symbol": symbol, "pattern": pattern_name, "result": result}
