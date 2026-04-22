"""
Model artifact registry — Phase 3.

Handles serialisation and deserialisation of trained model objects
(joblib) plus metrics and feature-importance JSON reports.

Artifacts are written to :data:`ml.config.ARTIFACTS_DIR` by default.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib

from ml.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filename: str, artifacts_dir: str | Path | None = None) -> Path:
    """Serialise a model object to disk with joblib.

    Args:
        model: Any fitted forecaster object (must be pickle-compatible).
        filename: Artifact filename, e.g. ``"xgboost_model.joblib"``.
        artifacts_dir: Override artifact directory.  Defaults to
                       :data:`ml.config.ARTIFACTS_DIR`.

    Returns:
        Absolute :class:`pathlib.Path` of the saved artifact.
    """
    out_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    _ensure_dir(out_dir)
    out_path = out_dir / filename
    joblib.dump(model, out_path, compress=3)
    logger.info("Saved model artifact → %s", out_path)
    return out_path


def load_model(filename: str, artifacts_dir: str | Path | None = None) -> Any:
    """Deserialise a model artifact from disk.

    Args:
        filename: Artifact filename, e.g. ``"best_model.joblib"``.
        artifacts_dir: Override artifact directory.

    Returns:
        The deserialised model object.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
    """
    art_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    path = art_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    model = joblib.load(path)
    logger.info("Loaded model artifact ← %s", path)
    return model


def save_json(data: dict | list, filename: str, artifacts_dir: str | Path | None = None) -> Path:
    """Write a dictionary or list to a JSON file.

    Args:
        data: Serialisable Python object.
        filename: Output filename, e.g. ``"metrics_report.json"``.
        artifacts_dir: Override artifact directory.

    Returns:
        Absolute :class:`pathlib.Path` of the saved file.
    """
    out_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    _ensure_dir(out_dir)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved JSON artifact → %s", out_path)
    return out_path


def load_json(filename: str, artifacts_dir: str | Path | None = None) -> Any:
    """Read a JSON artifact from the registry.

    Args:
        filename: Artifact filename.
        artifacts_dir: Override artifact directory.

    Returns:
        Deserialised Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    art_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    path = art_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
