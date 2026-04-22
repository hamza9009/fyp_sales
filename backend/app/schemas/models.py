"""Pydantic response schemas for the model metrics endpoints."""

from pydantic import BaseModel, Field


class SingleModelMetrics(BaseModel):
    """Evaluation metrics for one trained model."""

    model_name: str
    mae: float = Field(..., description="Mean Absolute Error on test set.")
    rmse: float = Field(..., description="Root Mean Squared Error on test set.")
    train_time_sec: float = Field(..., description="Wall-clock training time in seconds.")
    rank: int = Field(..., description="Rank by RMSE (1 = best).")
    is_best: bool = Field(..., description="True if this is the selected best model.")


class SplitInfo(BaseModel):
    """Train/test split details."""

    cutoff_date: str
    train_rows: int
    test_rows: int
    train_products: int
    test_products: int


class ModelMetricsResponse(BaseModel):
    """Full model metrics comparison response."""

    best_model: str = Field(..., description="Name of the best model (lowest RMSE).")
    split: SplitInfo
    models: list[SingleModelMetrics] = Field(..., description="All models ranked by RMSE.")

    model_config = {"json_schema_extra": {"example": {
        "best_model": "LightGBM",
        "split": {
            "cutoff_date": "2011-10-06",
            "train_rows": 129005,
            "test_rows": 62238,
            "train_products": 2092,
            "test_products": 2300,
        },
        "models": [
            {"model_name": "LightGBM", "mae": 24.31, "rmse": 61.11,
             "train_time_sec": 8.54, "rank": 1, "is_best": True},
        ],
    }}}
