"""Products router — Phase 4 (optimised).

GET /api/v1/products?q=<query>&limit=<N>
    Search products by stock_code or description substring.
"""

from fastapi import APIRouter, Query

from app.schemas.products import ProductResult, ProductSearchResponse
from app.services import data_store

router = APIRouter(prefix="/products", tags=["products"])


@router.get("", response_model=ProductSearchResponse)
def search_products(
    q: str = Query(..., min_length=1, description="Search term (stock_code or description)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
) -> ProductSearchResponse:
    """Return products whose stock_code or description contains ``q``."""
    hits = data_store.search_products(q, limit=limit)
    return ProductSearchResponse(
        query=q,
        results=[ProductResult(**h) for h in hits],
        total=len(hits),
    )
