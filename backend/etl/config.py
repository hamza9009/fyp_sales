"""
ETL configuration constants.

All tunable ETL parameters are centralised here to avoid magic numbers
scattered across the pipeline modules.
"""

# ── Column name standardisation map (raw → snake_case) ─────────────────────────
RAW_COLUMN_MAP: dict[str, str] = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}

# All columns that must be present after standardisation
REQUIRED_COLUMNS: list[str] = list(RAW_COLUMN_MAP.values())

# Columns that MUST be present for the pipeline to succeed
MANDATORY_COLUMNS: list[str] = [
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
]

# Optional columns — pipeline runs without them; these defaults are injected
OPTIONAL_COLUMN_DEFAULTS: dict[str, str] = {
    "customer_id": "UNKNOWN",
    "country": "Unknown",
}

# ── Cleaning thresholds ─────────────────────────────────────────────────────────
# Rows with unit_price at or below this value are considered invalid
MIN_UNIT_PRICE: float = 0.0
# Returns and zero-quantity rows are excluded (quantity must be >= this)
MIN_QUANTITY: int = 1

# ── Feature engineering windows ────────────────────────────────────────────────
# Lag sizes in days for the target variable (total_quantity shifted back N days)
LAG_DAYS: list[int] = [1, 7, 14, 28]
# Rolling window sizes in days for mean and std features
ROLLING_WINDOWS: list[int] = [7, 14, 28]

# ── Quality filter ──────────────────────────────────────────────────────────────
# Products with fewer than this many daily observations are dropped from the
# ML feature dataset — they lack enough history for reliable lag/rolling features.
MIN_PRODUCT_OBSERVATIONS: int = 30

# ── Processed data filenames ────────────────────────────────────────────────────
PRODUCTS_FILENAME: str = "products.parquet"
PROCESSED_SALES_FILENAME: str = "sales_daily.parquet"
PROCESSED_FEATURES_FILENAME: str = "features.parquet"
