"""
Quant Analysis API Server
Railway deployment FastAPI server for quant analysis system.
Provides HTTP endpoints for KR/US stock analysis triggered by upstream data service.
"""
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import httpx

load_dotenv()

# Configure sys.path for kr/, us/, backtest/ submodule imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KR_DIR = os.path.join(BASE_DIR, "kr")
US_DIR = os.path.join(BASE_DIR, "us")
BACKTEST_DIR = os.path.join(BASE_DIR, "backtest")

for path in [BASE_DIR, KR_DIR, US_DIR, BACKTEST_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========================================
# API Key Authentication
# ========================================
def verify_api_key(x_api_key: str = Header(None, alias="X-API-KEY")) -> str:
    """Verify X-API-KEY header (same pattern as data service)"""
    expected_key = os.getenv("API_SECRET_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API_SECRET_KEY not configured on server")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-KEY header is required")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ========================================
# FastAPI App
# ========================================
app = FastAPI(
    title="Quant Analysis API",
    description="Quant analysis system API for Railway deployment",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Quant Analysis API Server Starting")
    logger.info(f"Environment: {'Railway' if os.getenv('RAILWAY_ENVIRONMENT') else 'Local'}")
    logger.info(f"API Key configured: {bool(os.getenv('API_SECRET_KEY'))}")
    logger.info(f"Portfolio URL configured: {bool(os.getenv('PORTFOLIO_SERVICE_URL'))}")
    logger.info("=" * 60)


# ========================================
# Dashboard (static) + Backfill proxy
# ========================================
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard not built")
    return FileResponse(index_path)


@app.post("/backfill/{endpoint:path}")
async def backfill_proxy(endpoint: str, request: Request):
    """Proxy backfill requests to alphafolio_data service.

    Dashboard endpoint — no browser-side auth. Server uses API_SECRET_KEY
    from secrets/alphafolio.env (mounted via env_file in docker-compose)
    when forwarding to alphafolio_data.
    """
    api_key = os.getenv("API_SECRET_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="API_SECRET_KEY not configured in secrets/alphafolio.env",
        )

    data_url = os.getenv("DATA_SERVICE_URL", "http://alphafolio_data:8000")
    target = f"{data_url}/{endpoint}"
    query = str(request.url.query)
    if query:
        target = f"{target}?{query}"

    logger.info(f"[backfill proxy] -> {target}")
    try:
        async with httpx.AsyncClient(timeout=7200) as client:
            resp = await client.post(target, headers={"X-API-KEY": api_key})
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}
        return JSONResponse(status_code=resp.status_code, content=payload)
    except httpx.RequestError as e:
        logger.exception("backfill proxy failed")
        raise HTTPException(status_code=502, detail=f"alphafolio_data unreachable: {e}")


# ========================================
# Health Check
# ========================================
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "quant",
        "timestamp": datetime.now().isoformat()
    }


# ========================================
# KR Analysis Endpoint
# ========================================
@app.post("/kr/run")
async def kr_run(api_key: str = Depends(verify_api_key)):
    """
    Korean stock full analysis (kr_main.py option 1).
    - Analyzes all stocks for today's date
    - Runs KR Prediction Collector
    - Chains to portfolio service on success
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("POST /kr/run - Started")
    logger.info("=" * 60)

    try:
        from kr.kr_main import run_option1
        result = await run_option1()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"KR analysis completed in {duration:.2f}s")

        # Chain to portfolio service
        chain_result = await _call_portfolio_service("KR")

        return {
            "status": "success",
            "message": "KR analysis completed",
            "duration_seconds": duration,
            "analysis_result": result,
            "chain_portfolio": chain_result,
            "timestamp": end_time.isoformat()
        }

    except Exception as e:
        logger.error(f"KR analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"KR analysis failed: {str(e)}")


# ========================================
# US Analysis Endpoint
# ========================================
@app.post("/us/run")
async def us_run(api_key: str = Depends(verify_api_key)):
    """
    US stock full analysis (us_main.py option 1).
    - Auto-detects latest data date from us_daily table
    - Analyzes all stocks
    - Runs US Prediction Collector
    - Chains to portfolio service on success
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("POST /us/run - Started")
    logger.info("=" * 60)

    try:
        from us.us_main import run_option1
        result = await run_option1()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"US analysis completed in {duration:.2f}s")

        # Chain to portfolio service
        chain_result = await _call_portfolio_service("US")

        return {
            "status": "success",
            "message": "US analysis completed",
            "duration_seconds": duration,
            "analysis_result": result,
            "chain_portfolio": chain_result,
            "timestamp": end_time.isoformat()
        }

    except Exception as e:
        logger.error(f"US analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"US analysis failed: {str(e)}")


# ========================================
# Chaining Helper
# ========================================
async def _call_portfolio_service(country: str) -> dict:
    """Call portfolio service /recommend/daily endpoint"""
    portfolio_url = os.getenv("PORTFOLIO_SERVICE_URL")
    api_key = os.getenv("API_SECRET_KEY")

    if not portfolio_url:
        logger.warning("PORTFOLIO_SERVICE_URL not configured, skipping chain")
        return {"status": "skipped", "reason": "PORTFOLIO_SERVICE_URL not configured"}

    try:
        import httpx
        endpoint = f"{portfolio_url}/recommend/daily"
        logger.info(f"Chaining to portfolio service: {endpoint} (country={country})")

        async with httpx.AsyncClient(timeout=3600) as client:
            response = await client.post(
                endpoint,
                json={"country": country},
                headers={"X-API-KEY": api_key} if api_key else {}
            )
            response.raise_for_status()
            logger.info(f"Portfolio service ({country}) called successfully")
            return {"status": "success", "response": response.json()}

    except Exception as e:
        logger.error(f"Portfolio service call failed: {e}")
        return {"status": "failed", "error": str(e)}


# ========================================
# Backtest Endpoints
# ========================================
from backtest import runner as bt_runner
from backtest import storage as bt_storage
from backtest.models import BacktestRequest, GradeGenerationRequest


# NOTE: backtest endpoints are dashboard-facing (no browser auth needed).
# Service-to-service auth (verify_api_key) stays on /kr/run, /us/run only.


@app.post("/backtest/run")
async def backtest_run_endpoint(req: BacktestRequest):
    params = req.model_dump()
    result = await bt_runner.run_backtest(params)
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result)
    return result


@app.post("/backtest/generate-grades")
async def backtest_generate_grades_endpoint(req: GradeGenerationRequest):
    """Populate historical {us,kr}_stock_grade by looping run_option1(target_date).

    SLOW — 5~15 min per trading day. Run in background or pick a small range.
    Underlying data (us_daily, fundamentals, etc.) must already exist in DB.
    """
    try:
        result = await bt_runner.generate_grades_for_range(
            req.country, req.start_date, req.end_date, req.skip_existing,
            use_prefilter=req.use_prefilter,
            prefilter_top_n=req.prefilter_top_n,
            with_event_modifier=req.with_event_modifier,
        )
        return {"status": "completed", **result}
    except Exception as e:
        logger.exception("grade generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/results")
async def backtest_list_results(limit: int = 50):
    runs = await bt_storage.list_runs(limit)
    return {"runs": runs, "count": len(runs)}


@app.get("/backtest/results/{run_id}")
async def backtest_get_result(
    run_id: str,
    include_nav: bool = False,
    include_trades: bool = False,
):
    result = await bt_storage.get_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if include_nav:
        result["nav_history"] = await bt_storage.get_nav_history(run_id)
    if include_trades:
        result["trades"] = await bt_storage.get_trades(run_id)
    return result


@app.delete("/backtest/results/{run_id}")
async def backtest_delete_result(run_id: str):
    deleted = await bt_storage.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"status": "deleted", "run_id": run_id}


# ========================================
# Entry Point
# ========================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
