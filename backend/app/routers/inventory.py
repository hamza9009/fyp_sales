"""
Inventory router — Phase 4.

GET /inventory/{stock_code}
  Returns inventory intelligence signals (stockout risk, reorder suggestion)
  for the given product.
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.inventory import InventoryResponse
from app.services import data_store, inventory_service, model_loader, persistence_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.get(
    "/{product_query}",
    response_model=InventoryResponse,
    summary="Inventory intelligence for a product",
    description=(
        "Returns simulated inventory signals including stockout risk, "
        "reorder point, and days of stock remaining for the given StockCode "
        "or product description. "
        "NOTE: Stock levels are simulated from demand data (UCI dataset has no stock column)."
    ),
)
def get_inventory(
    product_query: str,
    client_id: str | None = Header(default=None, alias="X-Client-Id"),
    db: Session = Depends(get_db),
) -> InventoryResponse:
    """Compute inventory signals for a stock code or product description.

    Args:
        product_query: Product StockCode or description (path parameter).

    Returns:
        :class:`~app.schemas.inventory.InventoryResponse`.

    Raises:
        404 if the product is not found.
        503 if the model or data is not loaded.
    """
    product_info = data_store.resolve_product_identifier(product_query)
    if product_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_query}' not found in the dataset.",
        )
    stock_code = product_info["stock_code"]

    try:
        signal = inventory_service.compute_inventory_signal(stock_code)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = InventoryResponse(**signal)
    model_name = model_loader.get_model_name()

    try:
        persistence_service.persist_inventory_signal(db, signal)
    except Exception as exc:  # pragma: no cover - non-critical persistence path
        db.rollback()
        logger.warning("Inventory persistence failed for %s: %s", stock_code, exc)

    try:
        persistence_service.persist_prediction_history(
            db,
            client_id=client_id,
            endpoint="inventory",
            query_text=product_query,
            resolved_stock_code=stock_code,
            model_name=model_name,
            horizon_days=None,
            request_payload=None,
            response_payload=response.model_dump(mode="json"),
        )
    except Exception as exc:  # pragma: no cover - non-critical persistence path
        db.rollback()
        logger.warning("Inventory history persistence failed for %s: %s", stock_code, exc)

    return response
