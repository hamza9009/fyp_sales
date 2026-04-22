# RetailSense Project Documentation

## 1. Project Summary

RetailSense is a full-stack predictive analytics SaaS prototype for e-commerce data. It ingests retail transactions, cleans and aggregates them, engineers forecasting features, trains demand models, exposes insights through a FastAPI backend, and renders those insights in a Next.js frontend.

This repository is not just an ML experiment. It is organized as an end-to-end system with these major layers:

- `backend/etl`: batch ETL pipeline
- `ml/`: model training and artifact generation
- `backend/app`: API layer and application services
- `frontend/`: dashboard UI
- `data/processed`: generated Parquet datasets used by both ML and API services
- `ml/artifacts`: trained model artifacts and metrics reports
- `tests/`: ETL, ML, and backend API tests

## 2. Repository Layout

```text
.
├── backend/
│   ├── app/                 # FastAPI app, routers, services, ORM models, config
│   ├── etl/                 # Extract, clean, transform, feature, load, stream modules
│   ├── alembic/             # DB migrations
│   ├── Dockerfile
│   ├── alembic.ini
│   └── requirements.txt
├── ml/                      # Training pipeline, model wrappers, evaluation, registry
├── frontend/                # Next.js 16 + React 19 dashboard
├── data/
│   ├── raw/
│   └── processed/           # products.parquet, sales_daily.parquet, features.parquet
├── tests/
│   ├── backend/
│   ├── etl/
│   └── ml/
├── docker-compose.yml       # PostgreSQL + backend
├── Online Retail.xlsx       # raw source dataset
├── instructions.md
└── project_detail.md
```

## 3. High-Level Architecture

```text
Raw retail file
    ->
ETL pipeline
    ->
Processed Parquet datasets + optional PostgreSQL upserts
    ->
ML training pipeline
    ->
Model artifacts + metrics JSON
    ->
FastAPI startup loads Parquet + model into memory
    ->
API endpoints serve forecasts, inventory signals, dashboard data, and model metrics
    ->
Next.js frontend fetches API data and renders charts/pages
```

Important implementation detail: the current API serves most business logic from in-memory Parquet data and saved model artifacts, not from live SQL queries. PostgreSQL exists and is migrated, but only part of the runtime path depends on it today.

## 4. End-to-End Data and Request Flow

### 4.1 ETL Pipeline

The ETL flow lives in `backend/etl/pipeline.py` and orchestrates:

1. Extract
2. Clean
3. Transform
4. Feature engineering
5. Load

#### Extract

`backend/etl/extract.py` handles:

- reading `.csv`, `.xls`, and `.xlsx`
- standardizing raw columns to internal snake_case names
- optional user-provided column mapping for uploaded datasets
- automatic alias matching when explicit mapping is not supplied
- parsing `invoice_date`
- numeric conversion of `quantity` and `unit_price`
- injecting defaults for optional fields like `customer_id` and `country`

This module also powers the upload UI's "inspect headers" step. The upload flow first reads only the header row and suggests a mapping before the file is accepted for full processing.

#### Clean

`backend/etl/clean.py` applies explicit business rules:

- remove cancellation invoices where `invoice_no` starts with `C`
- impute recoverable missing categorical values using group mode or global mode
- impute recoverable missing numeric values using product-level mean with global fallback
- use a readable fallback placeholder for unresolved non-key descriptions
- preserve missing invoice numbers with synthetic placeholders rather than dropping the row
- drop only rows whose essential identity fields still cannot be recovered
- remove returns and zero-quantity rows
- remove prices that are still invalid after imputation
- strip whitespace from key string fields

The cleaning logic is auditable: every removal category is counted and logged.

#### Transform

`backend/etl/transform.py` performs two main transformations:

- adds `revenue = quantity * unit_price`
- aggregates transaction-level rows into one row per `(stock_code, sale_date)`

It also builds a product catalog DataFrame with:

- the modal description per SKU
- median unit price per SKU
- modal country per SKU

The daily aggregate output matches the conceptual `sales_daily` table schema:

- `stock_code`
- `sale_date`
- `total_quantity`
- `total_revenue`
- `num_transactions`

#### Feature Engineering

`backend/etl/features.py` creates the ML-ready dataset used by training and inference.

Generated features:

- calendar: `day_of_week`, `month`, `quarter`, `is_weekend`, `is_month_end`, `day_of_year`
- lags: `lag_1`, `lag_7`, `lag_14`, `lag_28`
- rolling stats: `rolling_mean_*`, `rolling_std_*` for 7, 14, and 28 day windows

Leakage prevention is explicit:

- lag features are computed per product
- rolling windows use `shift(1)` before aggregation
- sparse products with fewer than 30 observations are removed
- rows missing required lag history are dropped

#### Load

`backend/etl/load.py` does two separate things:

- saves Parquet files to `data/processed/`
- optionally upserts `products` and `sales_daily` into PostgreSQL

Database writes use PostgreSQL `INSERT ... ON CONFLICT DO UPDATE`, so rerunning ETL is idempotent for those two tables.

#### Streaming Simulation

`backend/etl/stream.py` is not a real event-streaming system. It is a deterministic replay utility that yields chronological batches from processed daily sales data. It exists to satisfy the "stream simulation" requirement without Kafka or another broker.

### 4.2 ML Training Pipeline

The ML training entrypoint is `ml/pipeline.py`, which calls `ml/trainer.py`.

#### Data Loading for ML

`ml/data_loader.py` enriches the ETL feature dataset by:

- loading `features.parquet`
- joining `unit_price` from `products.parquet`
- label-encoding `stock_code` into `stock_code_encoded`

That encoder is saved because the backend needs the exact same mapping during inference.

#### Train/Test Split

`ml/splitter.py` uses a time-based split by unique dates instead of random row sampling. This is the correct choice for forecasting because it prevents temporal leakage.

Implementation details:

- default train ratio is `0.8`
- cutoff is based on unique calendar dates
- all rows before the cutoff go to training
- all rows on or after the cutoff go to test

#### Models

`ml/trainer.py` currently instantiates these model classes:

- `NaiveForecaster`
- `RandomForestForecaster`
- `XGBoostForecaster`
- `LightGBMForecaster`
- `CatBoostForecaster`

Model wrappers live under `ml/models/`.

Shared design points:

- models implement the same `BaseForecaster` interface
- tree models use the common feature list from `ml/config.py`
- non-negative clipping is enforced on predictions
- log-transform support is used for skewed demand targets

#### Evaluation and Ranking

`ml/evaluate.py` computes:

- MAE
- RMSE

RMSE is the primary ranking metric. `compare_models()` sorts ascending by RMSE and marks the top-ranked model as best.

#### Artifacts

Training writes artifacts to `ml/artifacts/`, including:

- individual model files
- `best_model.joblib`
- `stock_code_encoder.joblib`
- `metrics_report.json`
- `feature_importance.json`

### 4.3 Backend Startup

The FastAPI entrypoint is `backend/app/main.py`.

At startup, the lifespan hook does three things:

1. checks database connectivity
2. loads the best ML model and label encoder from `ml/artifacts/`
3. loads processed Parquet datasets from `data/processed/`

This means the backend is optimized for serving from memory after startup. If the database is unavailable, the app still starts and Parquet-backed endpoints continue to work.

### 4.4 Forecast Request Flow

Endpoint: `GET /api/v1/forecast/{product_query}?horizon=...`

Request handling path:

- router resolves `product_query` from either stock code or product description
- `forecast_service.generate_forecast()` gets the latest 28 days of sales history
- the saved model predicts one day at a time
- each prediction is fed back into the rolling window for the next step
- date features are recomputed for each forecast date
- predicted revenue is derived from unit price if available
- the response is also written to the `forecasts` table and linked to a `model_runs` row

This is true multi-step recursive forecasting, not a one-shot direct horizon model.

### 4.5 Inventory Request Flow

Endpoint: `GET /api/v1/inventory/{product_query}`

Request handling path:

- router resolves `product_query` from either stock code or product description
- `inventory_service.compute_inventory_signal()` builds a dense daily demand history
- a stock replay is run day by day using:
  - initial stock
  - reorder point
  - order-up-to target
  - lead time
  - replenishment arrivals
  - lost-sales / stockout events
- average daily demand is computed from the recent window
- next-day demand is taken from the forecast service
- forward stockout risk is estimated over forecast demand using repeated simulation
- the computed signal is also written to the `inventory_signals` table

Current inventory assumptions:

- starting stock = average daily demand * stock multiplier
- reorder point = average daily demand * (lead time + safety stock days)
- target stock = average daily demand * target cover days
- stockout event occurs whenever simulated demand exceeds available stock

The backend is explicit that these are demand-derived estimates, not real inventory measurements.

### 4.6 Dashboard, Analysis, Product Search, and Metrics Flow

These endpoints are also Parquet/artifact-backed:

- `GET /api/v1/dashboard/summary`
  - aggregates total revenue, quantity, product counts, top products, 30-day trend, and best model snapshot
- `GET /api/v1/models/metrics`
  - reads `metrics_report.json` and returns ranked model metrics
- `GET /api/v1/products?q=...`
  - performs case-insensitive substring search over in-memory product catalog data
- `GET /api/v1/analysis/univariate`
  - builds histograms and grouped summaries from cached DataFrames
- `GET /api/v1/analysis/bivariate`
  - builds revenue-by-month, quarter summaries, price-vs-demand buckets, and monthly trends

### 4.7 Upload-and-Retrain Flow

The upload flow is the most dynamic part of the system.

Frontend page: `frontend/app/upload/page.tsx`

Backend router: `backend/app/routers/pipeline.py`

Service: `backend/app/services/pipeline_service.py`

Actual execution flow:

1. user uploads a CSV/Excel file
2. `/pipeline/inspect` reads headers only and suggests a column mapping
3. user confirms or corrects the mapping
4. `/pipeline/upload` stores the upload in a temp file
5. `pipeline_service.start_pipeline()` starts a background thread
6. background thread runs:
   - ETL pipeline with Parquet output plus PostgreSQL upserts
   - ML training
   - cache reload for `data_store` and `model_loader`
7. frontend polls `/pipeline/status`

Only one job may run at a time. Job state is stored as a module-level singleton dictionary protected by a threading lock.

## 5. Backend Module Breakdown

### Routers

Routers under `backend/app/routers/` are thin HTTP layers. They mostly:

- validate request parameters
- convert service exceptions into HTTP errors
- shape responses through Pydantic schemas

### Services

Services under `backend/app/services/` hold almost all business logic:

- `data_store.py`: loads and serves cached Parquet DataFrames
- `forecast_service.py`: iterative multi-step forecasting
- `inventory_service.py`: simulated inventory intelligence
- `dashboard_service.py`: summary aggregations
- `metrics_service.py`: model metrics report loading
- `model_loader.py`: best-model and encoder loading
- `pipeline_service.py`: background ETL + ML orchestration
- `analysis_service.py`: analytical chart datasets

### Schemas

Pydantic response models live under `backend/app/schemas/`. They give the API stable typed contracts for the frontend.

### Database

`backend/app/database.py` defines:

- lazy SQLAlchemy engine creation
- lazy session factory creation
- `Base` declarative model
- DB health check helper

The engine is created lazily so imports do not fail when the DB is unreachable during tests or local development.

## 6. Database Schema and What Is Actually Used

Alembic migration `backend/alembic/versions/20260414_0001_initial_schema.py` creates six tables:

| Table | Purpose | Current runtime usage |
| --- | --- | --- |
| `products` | product catalog | actively upserted by ETL when DB loading is enabled |
| `sales_daily` | aggregated daily sales | actively upserted by ETL when DB loading is enabled |
| `model_runs` | training metadata | populated lazily when runtime forecast persistence links stored predictions to the current best model |
| `forecasts` | persisted predictions | forecast endpoint computes in memory, then stores the returned forecast rows in PostgreSQL |
| `inventory_signals` | persisted inventory alerts | inventory endpoint computes in memory, then stores the returned signal in PostgreSQL |
| `api_logs` | request logging | schema exists, but middleware logs to stdout only and does not persist DB rows |

This distinction matters: the database schema is broader than the currently wired runtime behavior.

## 7. Frontend Implementation

The frontend is a Next.js 16 app using React 19 and Recharts.

### Routing and Layout

- `frontend/app/page.tsx` redirects `/` to `/dashboard`
- `frontend/app/layout.tsx` wraps all pages with a persistent sidebar
- `frontend/components/Sidebar.tsx` defines the main navigation

### API Client

`frontend/lib/api.ts` contains typed fetch wrappers matching backend schemas.

Pages use this client rather than calling `fetch()` directly in arbitrary places.

### Main Pages

- `dashboard/page.tsx`
  - overview metrics
  - revenue trend
  - top products
  - best model summary
  - embedded univariate and bivariate analysis tabs
- `forecast/page.tsx`
  - product search/autocomplete
  - horizon selection
  - forecast chart and daily table
- `inventory/page.tsx`
  - stock-code lookup
  - risk banner
  - reorder and simulated stock signals
- `models/page.tsx`
  - comparative model charts
  - radar chart
  - full metrics table
- `upload/page.tsx`
  - upload wizard
  - column-mapping UI
  - pipeline progress polling
- `analysis/page.tsx`
  - standalone analytics page for univariate/bivariate charts

### Shared UI Components

- `ProductSearch.tsx`: debounced product autocomplete using `/products`
- `StatCard.tsx`, `LoadingSpinner.tsx`, `ErrorMessage.tsx`: shared presentation helpers

## 8. How to Run the Project

### Backend Only

From the project root:

```bash
cd backend
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose

From the project root:

```bash
docker compose up --build
```

Current `docker-compose.yml` starts:

- PostgreSQL on `5432`
- backend on `8000`

It does not start the frontend.

### Frontend

From the project root:

```bash
cd frontend
npm install
npm run dev
```

## 9. Tests

The test suite is organized by layer.

### ETL Tests

Under `tests/etl/`, the suite checks:

- extraction and column standardization
- cleaning rules
- aggregation logic
- feature engineering

These are mostly pure-function tests using synthetic data.

### ML Tests

Under `tests/ml/`, the suite checks:

- model interfaces and prediction shape
- metric computation
- ranking logic
- time-based splitting

The tests currently cover the naive, random forest, and XGBoost wrappers directly.

### Backend Tests

Under `tests/backend/`, the suite checks:

- health endpoint behavior
- endpoint schemas
- forecast and inventory validation
- dashboard and model metrics responses

Most backend route tests mock the service layer so they do not require real Parquet files, DB connections, or model artifacts.

## 10. Current Artifacts in This Repository Snapshot

The committed/generated files under `ml/artifacts/` currently include:

- `best_model.joblib`
- `naive_model.joblib`
- `random_forest_model.joblib`
- `xgboost_model.joblib`
- `stock_code_encoder.joblib`
- `metrics_report.json`
- `feature_importance.json`

The current `metrics_report.json` says:

- best model: `XGBoost`
- split cutoff: `2011-10-06`
- train rows: `129005`
- test rows: `62238`

This artifact snapshot currently lists metrics for Naive, Random Forest, and XGBoost only.

## 11. Important Implementation Gaps and Inconsistencies

These are the main issues a maintainer should know immediately:

1. Frontend default API port mismatch

The frontend client defaults to:

```text
http://localhost:8001/api/v1
```

But the backend and Docker config expose:

```text
http://localhost:8000
```

So the frontend needs `NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1` in local development unless the backend is intentionally run on `8001`.

2. Docker Compose does not include the frontend

`docker-compose.yml` only starts PostgreSQL and the backend. The UI is a separate local process.

3. Runtime behavior is hybrid file-backed plus DB-backed

The API still serves from in-memory Parquet data and model/joblib artifacts for speed, but it now also persists ETL outputs and runtime forecast/inventory results to PostgreSQL. The main live read path is still:

- in-memory Parquet data
- model/joblib artifacts
- JSON metrics report

4. ML code and saved artifacts are not fully aligned

The training code now instantiates five model classes, but the current artifact directory and committed metrics report only show three model outputs. That means the repo snapshot reflects a partial or older training run compared to the current trainer implementation.

5. Analysis page exists, but primary navigation does not expose it

`frontend/app/analysis/page.tsx` exists, but the sidebar currently links only to:

- Dashboard
- Forecast
- Inventory
- Models
- Upload Data

The dashboard page already embeds analysis tabs, which is likely why the standalone page is not linked.

## 12. Practical Mental Model for This Codebase

The simplest accurate way to understand the project is:

- ETL produces clean daily product-level data
- ML trains on that processed data and saves the best model plus reports
- FastAPI loads those files into memory at startup
- frontend pages call typed endpoints that mostly read from those in-memory caches
- upload/retrain refreshes the same caches without restarting the backend

That is the core implementation pattern of this repository.
