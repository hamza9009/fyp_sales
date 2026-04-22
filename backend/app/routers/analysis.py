"""Analysis router — univariate & bivariate data analytics."""

from fastapi import APIRouter, HTTPException

from app.schemas.analysis import BivariateAnalysisResponse, UnivariateAnalysisResponse
from app.services import analysis_service

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/univariate", response_model=UnivariateAnalysisResponse)
def get_univariate() -> UnivariateAnalysisResponse:
    """Return univariate distribution statistics for the loaded dataset."""
    try:
        return UnivariateAnalysisResponse(**analysis_service.get_univariate_analysis())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/bivariate", response_model=BivariateAnalysisResponse)
def get_bivariate() -> BivariateAnalysisResponse:
    """Return bivariate analysis — relationships between key variables."""
    try:
        return BivariateAnalysisResponse(**analysis_service.get_bivariate_analysis())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
