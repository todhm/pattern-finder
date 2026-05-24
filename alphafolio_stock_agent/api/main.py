"""
Stock Agents API Server
FastAPI application for AI agent strategy generation
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.dependencies import RedisManager
from api.routers import analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    print("[INFO] Stock Agents API starting...")
    yield
    # Shutdown
    print("[INFO] Closing Redis connections...")
    await RedisManager.close_all()
    print("[INFO] Stock Agents API shutdown complete")


app = FastAPI(
    title="Stock Agents API",
    description="AI Multi-Agent Investment Strategy Generator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (로컬 개발 시에만 필요, Railway에서는 Private Network 사용)
import os
if os.getenv("RAILWAY_ENVIRONMENT") is None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(analysis.router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "stock-agents-api",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check for Railway/load balancer"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
