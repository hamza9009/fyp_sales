"""
Pipeline router — upload dataset and trigger ETL + ML training.

POST /pipeline/inspect   multipart/form-data  file=<csv|xlsx>
     Reads column headers and returns an auto-suggested column mapping.
     No data is stored; only the header row is parsed.

POST /pipeline/upload    multipart/form-data  file=<csv|xlsx>  mapping=<json>
     Saves the file and starts the background pipeline with the given mapping.

GET  /pipeline/status    returns current job state
"""

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile, File

from app.schemas.pipeline import ColumnInspectResponse, PipelineJobResponse
from app.services import pipeline_service

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_MAX_BYTES = 100 * 1024 * 1024  # 100 MB


async def _read_and_validate(file: UploadFile) -> tuple[bytes, str]:
    """Read file content, enforce size + extension limits."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Upload a .csv, .xlsx, or .xls file.",
        )
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1_048_576:.1f} MB). Maximum is 100 MB.",
        )
    return content, suffix


@router.post("/inspect", response_model=ColumnInspectResponse, status_code=200)
async def inspect_columns(file: UploadFile = File(...)) -> ColumnInspectResponse:
    """Read column headers from an uploaded file and return a suggested mapping.

    Only the header row is parsed — no pipeline is started, no file is saved.
    Use the returned ``suggested_mapping`` as the default selection in the
    column-mapping UI; allow the user to override any field before submitting
    ``POST /pipeline/upload``.
    """
    content, suffix = await _read_and_validate(file)

    from etl.extract import inspect_columns as _inspect  # noqa: PLC0415

    try:
        result = _inspect(content, suffix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ColumnInspectResponse(**result)


@router.post("/upload", response_model=PipelineJobResponse, status_code=202)
async def upload_dataset(
    file: UploadFile = File(...),
    mapping: str = Form("{}"),
) -> PipelineJobResponse:
    """Upload a dataset and run the full ETL + ML pipeline.

    Args:
        file:    CSV or Excel file (max 100 MB).
        mapping: JSON string ``{internal_snake_case: user_column_name}`` —
                 the column mapping chosen in the UI.  Pass ``{}`` to fall
                 back to automatic case-insensitive matching.

    The pipeline runs in the background.  Poll ``GET /pipeline/status``
    to track progress.  Only one job may run at a time.
    """
    content, suffix = await _read_and_validate(file)

    # Parse and validate the column mapping
    try:
        column_mapping: dict[str, str] = json.loads(mapping) if mapping.strip() else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid mapping JSON: {exc}") from exc

    # Write to a temp file (pipeline_service deletes it when done)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_")
    tmp.write(content)
    tmp.flush()
    tmp.close()

    job_id = pipeline_service.start_pipeline(
        Path(tmp.name),
        column_mapping=column_mapping or None,
    )
    if job_id is None:
        raise HTTPException(
            status_code=409,
            detail="A pipeline job is already running. Wait for it to complete before uploading again.",
        )

    return PipelineJobResponse(**pipeline_service.get_status())


@router.get("/status", response_model=PipelineJobResponse)
def get_pipeline_status() -> PipelineJobResponse:
    """Return the current pipeline job status."""
    return PipelineJobResponse(**pipeline_service.get_status())
