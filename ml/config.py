"""
ML Phase 3 configuration constants.

All tunable ML parameters are centralised here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
SALES_PARQUET    = PROCESSED_DIR / "sales_daily.parquet"
PRODUCTS_PARQUET = PROCESSED_DIR / "products.parquet"

# ── Columns ────────────────────────────────────────────────────────────────────
TARGET_COL: str = "total_quantity"

# Full feature set used for training.
# stock_code_encoded — ordinal LabelEncoding of the product SKU;
#   lets the model learn per-product demand patterns.
# stock_code_target_encoded — added internally by model wrappers / CV estimators
#   using smoothed target encoding fit only on the relevant training fold.
# unit_price — static price per unit; correlated with product tier / demand scale.
FEATURE_COLS: list[str] = [
    # Product identity
    "stock_code_encoded",
    "unit_price",
    # Lag features
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    # Rolling statistics
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
    # Calendar features
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "is_month_end",
    "day_of_year",
]

# Columns kept alongside features + target in split DataFrames
ID_COLS: list[str] = ["stock_code", "sale_date"]

# ── Target transform ───────────────────────────────────────────────────────────
# The target (total_quantity) has skewness ≈ 19.4 and max 4848.
# Applying log1p before training and expm1 after prediction dramatically reduces
# the influence of a small number of high-volume rows on RMSE.
USE_LOG_TRANSFORM: bool = True

# ── Train / test split ─────────────────────────────────────────────────────────
TRAIN_RATIO: float = 0.8
CV_SPLITS: int = 5
CV_SEARCH_ITERATIONS: int = 30
CV_SCORING: str = "neg_root_mean_squared_error"
VALIDATION_RATIO: float = 0.1
SEARCH_RANDOM_STATE: int = 42
TARGET_ENCODING_KEY_COL: str = "stock_code_encoded"
TARGET_ENCODING_FEATURE_COL: str = "stock_code_target_encoded"
TARGET_ENCODING_SMOOTHING: float = 20.0

# ── Random Forest hyperparameters ──────────────────────────────────────────────
# Tuned for log-scale targets: unlimited depth is safe since residuals are small;
# min_samples_leaf=3 provides light regularisation.
RF_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}
RF_PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [None, 8, 12, 16, 24],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.7, 1.0],
}

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
# Tuned for log-scale targets: more rounds (1 000 max), deeper trees,
# smaller min_child_weight, early stopping = 50 rounds.
XGB_PARAMS: dict = {
    "n_estimators": 1000,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.05,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}
XGB_PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [200, 500, 1000, 1500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.05, 0.1, 0.2],
    "reg_alpha": [0.0, 0.05, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

# ── LightGBM hyperparameters ───────────────────────────────────────────────────
LGB_PARAMS: dict = {
    "n_estimators": 1000,
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "regression_l1",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "early_stopping_rounds": 50,  # extracted and passed via callback in model
}
LGB_PARAM_DISTRIBUTIONS: dict = {
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [-1, 5, 8, 12, 16],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 500, 1000, 2000],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0],
}

# ── CatBoost hyperparameters ───────────────────────────────────────────────────
# Ordered boosting (default) avoids target leakage in categorical encoding.
CATBOOST_PARAMS: dict = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
    "allow_writing_files": False,
}
CATBOOST_PARAM_DISTRIBUTIONS: dict = {
    "iterations": [200, 500, 1000, 1500],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0, 9.0],
    "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    "random_strength": [0.0, 0.5, 1.0, 2.0],
}

# Columns treated as categorical by CatBoost (ordered target encoding applied)
CATBOOST_CAT_FEATURES: list[str] = [
    "stock_code_encoded",
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "is_month_end",
]

# ── Artifact filenames ─────────────────────────────────────────────────────────
NAIVE_ARTIFACT              = "naive_model.joblib"
RF_ARTIFACT                 = "random_forest_model.joblib"
XGB_ARTIFACT                = "xgboost_model.joblib"
LGB_ARTIFACT                = "lightgbm_model.joblib"
CATBOOST_ARTIFACT           = "catboost_model.joblib"
ENSEMBLE_ARTIFACT           = "lightgbm_xgboost_ensemble.joblib"
BEST_MODEL_ARTIFACT         = "best_model.joblib"
ENCODER_ARTIFACT            = "stock_code_encoder.joblib"
METRICS_ARTIFACT            = "metrics_report.json"
FEATURE_IMPORTANCE_ARTIFACT = "feature_importance.json"
