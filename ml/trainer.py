"""
Model training orchestrator — Phase 3.

Coordinates the full ML pipeline:
  Load data → Enrich features → Time-split → Tune boosted models
  → Evaluate → Compare → Save artifacts

Usage::

    from ml.trainer import train_all_models

    result = train_all_models()
    print(result.best_model_name)
    print(result.comparison)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from ml.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_ARTIFACT,
    CV_SCORING,
    CV_SEARCH_ITERATIONS,
    CV_SPLITS,
    ENCODER_ARTIFACT,
    ENSEMBLE_ARTIFACT,
    FEATURE_COLS,
    FEATURE_IMPORTANCE_ARTIFACT,
    FEATURES_PARQUET,
    LGB_ARTIFACT,
    LGB_PARAM_DISTRIBUTIONS,
    LGB_PARAMS,
    METRICS_ARTIFACT,
    TARGET_COL,
    XGB_PARAM_DISTRIBUTIONS,
    XGB_PARAMS,
    XGB_ARTIFACT,
)
from ml.data_loader import load_features
from ml.evaluate import compare_models, compute_metrics
from ml.models.averaging_ensemble import AveragingEnsembleForecaster
from ml.models.base import BaseForecaster
from ml.models.lightgbm_model import LightGBMForecaster
from ml.models.xgboost_model import XGBoostForecaster
from ml.progress import TrainingProgressTracker
from ml.registry import save_json, save_model
from ml.splitter import SplitResult, time_split
from ml.target_encoding import TargetEncodingRegressor
from ml.tuning import ModelSearchSpec, resolve_cv_splits, run_randomized_search

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    comparison: pd.DataFrame
    best_model_name: str
    best_model: BaseForecaster
    metrics: dict[str, dict]
    feature_importance: dict[str, dict]
    split: SplitResult
    encoder: LabelEncoder
    artifact_paths: dict[str, str] = field(default_factory=dict)


def _order_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return a globally chronological frame for time-based CV and refits."""
    order_cols = ["sale_date"]
    if "stock_code" in df.columns:
        order_cols.append("stock_code")
    return df.sort_values(order_cols).reset_index(drop=True)


def _train_and_evaluate(
    model: BaseForecaster,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], float]:
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(y_test, y_pred, model_name=model.name)
    return metrics, elapsed


def _evaluate_fitted_model(
    model: BaseForecaster,
    *,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate an already-fitted model on the untouched test partition."""
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred, model_name=model.name)


def _build_search_specs() -> list[tuple[ModelSearchSpec, Any]]:
    """Return randomized-search definitions plus final wrapper factories."""
    xgb_search_params = {key: value for key, value in XGB_PARAMS.items() if key != "early_stopping_rounds"}
    xgb_search_params["n_jobs"] = 1
    lgb_search_params = {key: value for key, value in LGB_PARAMS.items() if key != "early_stopping_rounds"}
    lgb_search_params["n_jobs"] = 1

    return [
        (
            ModelSearchSpec(
                name="XGBoost",
                estimator=TargetEncodingRegressor(
                    XGBRegressor(**xgb_search_params),
                ),
                param_distributions=XGB_PARAM_DISTRIBUTIONS,
                param_prefix="base_estimator__",
            ),
            lambda params: XGBoostForecaster(params={**XGB_PARAMS, **params}),
        ),
        (
            ModelSearchSpec(
                name="LightGBM",
                estimator=TargetEncodingRegressor(
                    lgb.LGBMRegressor(**lgb_search_params),
                ),
                param_distributions=LGB_PARAM_DISTRIBUTIONS,
                param_prefix="base_estimator__",
            ),
            lambda params: LightGBMForecaster(params={**LGB_PARAMS, **params}),
        ),
    ]


def train_all_models(
    features_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    search_iterations: int | None = CV_SEARCH_ITERATIONS,
    cv_splits: int | None = CV_SPLITS,
    show_progress: bool = False,
) -> TrainingResult:
    """Run the complete Phase 3 ML training pipeline.

    Steps
    -----
    1. Load & enrich features (unit_price join + stock_code encoding)
    2. Time-based 80/20 train/test split with the latest dates kept untouched
    3. Tune LightGBM and XGBoost using randomized time-series CV on training only
    4. Refit each tuned model on the full training window
    5. Evaluate LightGBM, XGBoost, and their averaged ensemble once on the untouched test set
    6. Save encoder, all model artifacts, metrics JSON, feature importance JSON
    """
    logger.info("=" * 60)
    logger.info("PHASE 3 — ML TRAINING  (time-series CV tuning)")
    logger.info("=" * 60)

    if search_iterations is None:
        search_iterations = CV_SEARCH_ITERATIONS
    if cv_splits is None:
        cv_splits = CV_SPLITS
    if search_iterations < 1:
        raise ValueError(f"search_iterations must be at least 1; got {search_iterations}")

    art_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    art_dir.mkdir(parents=True, exist_ok=True)
    search_specs = _build_search_specs()
    total_progress_steps = 5 + len(search_specs)

    with TrainingProgressTracker(
        total_steps=total_progress_steps,
        enabled=show_progress,
    ) as progress:
        # ── 1. Load + enrich ────────────────────────────────────────────────────
        df, encoder = load_features(features_path or FEATURES_PARQUET)
        progress.advance(f"Loaded features ({len(df):,} rows)")

        # ── 2. Save encoder immediately (needed by backend at startup) ─────────
        encoder_path = art_dir / ENCODER_ARTIFACT
        joblib.dump(encoder, encoder_path, compress=3)
        logger.info("Encoder saved → %s  (%d classes)", encoder_path, len(encoder.classes_))
        progress.advance("Saved stock code encoder")

        # ── 3. Time-based split ─────────────────────────────────────────────────
        split = time_split(df)
        train_df = _order_by_time(split.train)
        test_df = _order_by_time(split.test)
        effective_cv_splits = resolve_cv_splits(train_df["sale_date"].nunique(), cv_splits)
        progress.advance(
            f"Created time split ({len(train_df):,} train rows / {len(test_df):,} test rows)"
        )

        # ── 4. Train + evaluate ─────────────────────────────────────────────────
        artifact_map = {
            "XGBoost":       XGB_ARTIFACT,
            "LightGBM":      LGB_ARTIFACT,
            "LightGBM + XGBoost Ensemble": ENSEMBLE_ARTIFACT,
        }

        all_metrics: dict[str, dict] = {}
        all_importance: dict[str, dict] = {}
        artifact_paths: dict[str, str] = {"encoder": str(encoder_path)}
        trained_models: dict[str, BaseForecaster] = {}

        for search_spec, model_factory in search_specs:
            logger.info("-" * 50)
            logger.info("Tuning: %s", search_spec.name)
            search_result = run_randomized_search(
                X_train=train_df[FEATURE_COLS],
                y_train=train_df[TARGET_COL],
                sale_dates=train_df["sale_date"],
                spec=search_spec,
                n_splits=effective_cv_splits,
                n_iter=search_iterations,
                scoring=CV_SCORING,
            )

            model = model_factory(search_result.best_params)
            metrics, elapsed = _train_and_evaluate(
                model,
                train_df=train_df,
                test_df=test_df,
            )
            training_metadata = model.get_training_metadata() or {}
            all_metrics[model.name] = {
                **metrics,
                "train_time_sec": round(elapsed, 3),
                "cv_rmse": search_result.best_cv_rmse,
                "cv_splits": search_result.n_splits,
                "cv_search_iterations": search_result.n_iter,
                "selection_metric": CV_SCORING,
                "tuned_params": search_result.best_params,
                "refit_details": training_metadata,
            }
            trained_models[model.name] = model

            fi = model.get_feature_importance()
            if fi:
                all_importance[model.name] = fi

            fname = artifact_map.get(model.name, f"{model.name.lower().replace(' ', '_')}.joblib")
            saved = save_model(model, fname, art_dir)
            artifact_paths[model.name] = str(saved)
            logger.info("[%s] %.2f s | MAE=%.4f RMSE=%.4f", model.name, elapsed, metrics["mae"], metrics["rmse"])
            progress.advance(f"Completed {model.name}")

        logger.info("-" * 50)
        ensemble = AveragingEnsembleForecaster(
            [trained_models["LightGBM"], trained_models["XGBoost"]],
            already_fitted=True,
        )
        ensemble_metrics = _evaluate_fitted_model(
            ensemble,
            test_df=test_df,
        )
        ensemble_train_time = round(
            float(all_metrics["LightGBM"]["train_time_sec"]) + float(all_metrics["XGBoost"]["train_time_sec"]),
            3,
        )
        all_metrics[ensemble.name] = {
            **ensemble_metrics,
            "train_time_sec": ensemble_train_time,
            "cv_rmse": None,
            "cv_splits": effective_cv_splits,
            "cv_search_iterations": search_iterations,
            "selection_metric": CV_SCORING,
            "tuned_params": {
                "strategy": "mean_average",
                "components": ["LightGBM", "XGBoost"],
            },
            "refit_details": ensemble.get_training_metadata() or {},
        }
        trained_models[ensemble.name] = ensemble
        ensemble_saved = save_model(ensemble, artifact_map[ensemble.name], art_dir)
        artifact_paths[ensemble.name] = str(ensemble_saved)

        ensemble_fi = ensemble.get_feature_importance()
        if ensemble_fi:
            all_importance[ensemble.name] = ensemble_fi
        logger.info(
            "[%s] %.2f s | MAE=%.4f RMSE=%.4f",
            ensemble.name,
            ensemble_train_time,
            ensemble_metrics["mae"],
            ensemble_metrics["rmse"],
        )
        progress.advance(f"Completed {ensemble.name}")

        # ── 5. Rank ─────────────────────────────────────────────────────────────
        comparison = compare_models(all_metrics)
        best_name: str = comparison.iloc[0]["model"]
        best_model = trained_models[best_name]

        logger.info("=" * 60)
        logger.info("BEST MODEL: %s  (RMSE=%.4f  MAE=%.4f)", best_name, comparison.iloc[0]["rmse"], comparison.iloc[0]["mae"])
        logger.info("=" * 60)

        best_path = save_model(best_model, BEST_MODEL_ARTIFACT, art_dir)
        artifact_paths["best"] = str(best_path)

        # ── 6. Persist reports ───────────────────────────────────────────────────
        report: dict[str, Any] = {
            "split": {
                "cutoff_date":    str(split.cutoff_date.date()),
                "train_rows":     len(split.train),
                "test_rows":      len(split.test),
                "train_products": split.n_train_products,
                "test_products":  split.n_test_products,
                "cv_splits":      effective_cv_splits,
            },
            "models":     all_metrics,
            "best_model": best_name,
            "optimisations": [
                "Untouched final test set held out by latest dates only",
                f"RandomizedSearchCV on training window only ({search_iterations} trials, {effective_cv_splits} time-based folds)",
                f"Primary model-selection metric: {CV_SCORING}",
                "Smoothed stock-code target encoding added on top of label encoding inside each boosted model",
                "LightGBM objective switched to regression_l1 for better robustness to outliers",
                "Chronological internal validation used only during final boosted-model refits",
                "Final boosted models retrained on the full training window using early-stopped round counts",
                "LightGBM + XGBoost ensemble evaluated by averaging component predictions on the untouched test set",
            ],
        }
        metrics_path = save_json(report, METRICS_ARTIFACT, art_dir)
        artifact_paths["metrics"] = str(metrics_path)

        if all_importance:
            fi_path = save_json(all_importance, FEATURE_IMPORTANCE_ARTIFACT, art_dir)
            artifact_paths["feature_importance"] = str(fi_path)

        logger.info("PHASE 3 COMPLETE — artifacts in %s", art_dir)
        progress.advance("Saved model artifacts and reports")

    return TrainingResult(
        comparison=comparison,
        best_model_name=best_name,
        best_model=best_model,
        metrics=all_metrics,
        feature_importance=all_importance,
        split=split,
        encoder=encoder,
        artifact_paths=artifact_paths,
    )
