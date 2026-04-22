"""
Phase 3 ML training pipeline — CLI entry point.

Run from the project root::

    python -m ml.pipeline

Or with overrides::

    python -m ml.pipeline \\
        --features data/processed/features.parquet \\
        --artifacts ml/artifacts

The script will:
  1. Load the Phase 2 feature Parquet
  2. Keep the latest dates as an untouched final test split
  3. Tune LightGBM and XGBoost with randomized time-series cross-validation
  4. Refit each tuned model on the training window and evaluate an averaged ensemble
  5. Print MAE / RMSE for each candidate on the untouched test split
  6. Save all model artifacts and JSON reports to ml/artifacts/
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover - graceful fallback when rich is unavailable
    RichHandler = None


def _configure_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = []

    if RichHandler is not None:
        console_handler = RichHandler(show_path=False, rich_tracebacks=True, markup=False)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    handlers.append(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ml.pipeline",
        description="Phase 3: Train and evaluate demand forecasting models.",
    )
    parser.add_argument(
        "--features",
        default=None,
        metavar="PATH",
        help="Path to features.parquet (default: data/processed/features.parquet)",
    )
    parser.add_argument(
        "--artifacts",
        default=None,
        metavar="DIR",
        help="Output directory for model artifacts (default: ml/artifacts)",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=None,
        metavar="N",
        help="Randomized-search trials per tuned model (default: config value)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        metavar="N",
        help="Time-series CV folds on the training window (default: config value)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Optional log file path (default: <artifacts>/training_pipeline.log)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point — returns exit code (0 = success, 1 = failure)."""
    args = _parse_args()
    artifacts_dir = Path(args.artifacts) if args.artifacts else Path("ml/artifacts")
    log_file = Path(args.log_file) if args.log_file else artifacts_dir / "training_pipeline.log"
    _configure_logging(log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info("Training logs will also be written to %s", log_file)

    # Import here so that logging is configured first
    from ml.trainer import train_all_models

    try:
        result = train_all_models(
            features_path=args.features,
            artifacts_dir=args.artifacts,
            search_iterations=args.search_iterations,
            cv_splits=args.cv_splits,
            show_progress=True,
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Training pipeline failed: %s", exc)
        return 1

    # ── Print summary to stdout ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 — TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest model : {result.best_model_name}")
    print(f"Test split : {result.split.test_start.date()} → {result.split.test_end.date()}")
    print(f"Test rows  : {len(result.split.test):,}")
    print()

    comp = result.comparison[["rank", "model", "mae", "rmse", "best"]]
    print(comp.to_string(index=False))
    print()

    if result.feature_importance:
        print("Top-5 features (best model — gain importance):")
        fi = result.feature_importance.get(result.best_model_name, {})
        for feat, score in sorted(fi.items(), key=lambda x: -x[1])[:5]:
            print(f"  {feat:<25} {score:.4f}")
    print()

    print("Artifacts:")
    for label, path in result.artifact_paths.items():
        print(f"  {label:<20} {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
