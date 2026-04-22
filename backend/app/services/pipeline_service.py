"""
Pipeline orchestration service.

Runs ETL + ML training as a background thread so the FastAPI server
stays responsive.  Only one pipeline job runs at a time.
"""

import logging
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_ARTIFACTS_DIR = _PROJECT_ROOT / "ml" / "artifacts"

# ── Job state (module-level singleton) ────────────────────────────────────────

_lock = threading.Lock()

_job: dict[str, Any] = {
    "job_id": "none",
    "status": "idle",
    "progress": 0,
    "message": "No pipeline has been run yet.",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "etl_rows": None,
    "etl_products": None,
    "best_model": None,
    "best_rmse": None,
}


def get_status() -> dict[str, Any]:
    with _lock:
        return dict(_job)


def _update(**kwargs: Any) -> None:
    with _lock:
        _job.update(kwargs)


def _set_running(job_id: str) -> bool:
    """Transition to running state.  Returns False if already running."""
    with _lock:
        if _job["status"] in ("etl_running", "ml_running", "reloading"):
            return False
        _job.update(
            job_id=job_id,
            status="etl_running",
            progress=5,
            message="Starting ETL pipeline…",
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            error=None,
            etl_rows=None,
            etl_products=None,
            best_model=None,
            best_rmse=None,
        )
        return True


def _run_pipeline(raw_path: Path, job_id: str, column_mapping: dict | None = None) -> None:
    """Background thread: ETL → ML training → cache reload."""
    # Ensure ml.* and etl.* are importable from this thread
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    backend_root = str(_PROJECT_ROOT / "backend")
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    try:
        settings = get_settings()

        # ── Step 1: ETL ──────────────────────────────────────────────────────
        logger.info("[pipeline:%s] Starting ETL on %s", job_id, raw_path)
        _update(
            status="etl_running",
            progress=10,
            message="Running ETL pipeline (extract → clean → transform → features → save to Parquet + DB)…",
        )

        from etl.pipeline import run_etl_pipeline  # noqa: PLC0415

        etl_result = run_etl_pipeline(
            raw_path=raw_path,
            processed_dir=_PROCESSED_DIR,
            db_url=settings.DATABASE_URL,
            skip_db=False,
            column_mapping=column_mapping or None,
        )
        n_features = len(etl_result["features"])
        n_products = len(etl_result["products"])
        logger.info("[pipeline:%s] ETL done — %d feature rows, %d products", job_id, n_features, n_products)
        _update(
            status="ml_running",
            progress=40,
            message=f"ETL complete ({n_products} products, {n_features:,} rows). Starting ML training…",
            etl_rows=n_features,
            etl_products=n_products,
        )

        # ── Step 2: ML training ──────────────────────────────────────────────
        logger.info("[pipeline:%s] Starting ML training", job_id)
        _update(
            progress=45,
            message="Tuning LightGBM and XGBoost, then evaluating their averaged ensemble with time-series CV…",
        )

        from ml.trainer import train_all_models  # noqa: PLC0415

        ml_result = train_all_models(
            features_path=_PROCESSED_DIR / "features.parquet",
            artifacts_dir=_ARTIFACTS_DIR,
        )
        best_name = ml_result.best_model_name
        best_rmse = float(ml_result.comparison.iloc[0]["rmse"])
        logger.info("[pipeline:%s] ML done — best=%s RMSE=%.4f", job_id, best_name, best_rmse)
        _update(
            status="reloading",
            progress=85,
            message=f"Training complete. Best model: {best_name} (RMSE={best_rmse:.4f}). Reloading caches…",
            best_model=best_name,
            best_rmse=best_rmse,
        )

        # ── Step 3: Reload in-process caches ────────────────────────────────
        logger.info("[pipeline:%s] Reloading data_store and model_loader caches", job_id)
        from app.services import data_store, model_loader  # noqa: PLC0415

        model_loader.load_model()
        data_store.load_data()
        logger.info("[pipeline:%s] Caches reloaded", job_id)

        _update(
            status="completed",
            progress=100,
            message=f"Pipeline complete! Best model: {best_name} (RMSE={best_rmse:.4f}). All endpoints updated.",
            completed_at=datetime.now(timezone.utc),
        )

    except Exception as exc:
        logger.exception("[pipeline:%s] Pipeline failed: %s", job_id, exc)
        _update(
            status="failed",
            progress=0,
            message="Pipeline failed — see error details.",
            error=str(exc),
            completed_at=datetime.now(timezone.utc),
        )
    finally:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass


def start_pipeline(raw_path: Path, column_mapping: dict | None = None) -> str | None:
    """Kick off the pipeline in a background thread.

    Returns the job_id string, or None if a job is already running.
    """
    job_id = uuid.uuid4().hex
    if not _set_running(job_id):
        return None
    thread = threading.Thread(
        target=_run_pipeline,
        args=(raw_path, job_id, column_mapping),
        daemon=True,
        name=f"pipeline-{job_id[:8]}",
    )
    thread.start()
    return job_id
