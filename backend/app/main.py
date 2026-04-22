"""
FastAPI application entry point — Phase 4.

Initialises the app, mounts routers, configures CORS, and sets up
structured logging.  At startup the best ML model and processed Parquet
datasets are loaded into memory so every request is served in <200 ms.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import check_db_connection
from app.routers import analysis, dashboard, forecast, health, history, inventory, models_router, pipeline, products
from app.services import data_store, model_loader

settings = get_settings()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Pre-load model + data at startup; release on shutdown."""
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # DB connectivity (non-fatal if DB is down — parquet-backed endpoints still work)
    if check_db_connection():
        logger.info("Database connection: OK")
    else:
        logger.warning(
            "Database connection: FAILED — check DATABASE_URL. "
            "Parquet-backed endpoints (forecast, inventory, dashboard) will still work."
        )

    # Load ML model artifact
    try:
        model_loader.load_model()
        logger.info("ML model loaded: %s", model_loader.get_model_name())
    except FileNotFoundError as exc:
        logger.error("Model load failed: %s", exc)

    # Load processed Parquet datasets
    try:
        data_store.load_data()
        logger.info("Parquet data loaded successfully.")
    except FileNotFoundError as exc:
        logger.error("Data store load failed: %s", exc)

    yield

    logger.info("Shutting down %s", settings.APP_NAME)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Predictive analytics SaaS for e-commerce growth and inventory management. "
        "Provides demand forecasting, inventory intelligence, and model metrics via REST API."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow all origins in dev; lock down in production via env
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request latency logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_request_latency(request: Request, call_next) -> Response:
    """Log method, path, status code, and elapsed time for every request."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    logger.info(
        "%s %s → %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router,         prefix=settings.API_PREFIX)
app.include_router(forecast.router,       prefix=settings.API_PREFIX)
app.include_router(inventory.router,      prefix=settings.API_PREFIX)
app.include_router(history.router,        prefix=settings.API_PREFIX)
app.include_router(dashboard.router,      prefix=settings.API_PREFIX)
app.include_router(models_router.router,  prefix=settings.API_PREFIX)
app.include_router(products.router,       prefix=settings.API_PREFIX)
app.include_router(pipeline.router,       prefix=settings.API_PREFIX)
app.include_router(analysis.router,       prefix=settings.API_PREFIX)


# ---------------------------------------------------------------------------
# Root redirect info
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health",
        "endpoints": {
            "forecast": f"{settings.API_PREFIX}/forecast/{{stock_code}}",
            "inventory": f"{settings.API_PREFIX}/inventory/{{stock_code}}",
            "history": f"{settings.API_PREFIX}/history",
            "dashboard": f"{settings.API_PREFIX}/dashboard/summary",
            "metrics": f"{settings.API_PREFIX}/models/metrics",
            "products": f"{settings.API_PREFIX}/products?q={{query}}",
        },
    }
