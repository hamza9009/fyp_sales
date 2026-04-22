/**
 * API client — Phase 5
 * Typed fetch wrappers for all Phase 4 backend endpoints.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001/api/v1";

const CLIENT_ID_STORAGE_KEY = "retailsense_client_id";

// ── Types matching backend Pydantic schemas ────────────────────────────────────

export interface ForecastPoint {
  forecast_date: string;
  predicted_quantity: number;
  predicted_revenue: number | null;
}

export interface ForecastResponse {
  stock_code: string;
  description: string | null;
  model_name: string;
  horizon_days: number;
  forecast: ForecastPoint[];
}

export interface InventoryResponse {
  stock_code: string;
  description: string | null;
  as_of_date: string;
  avg_daily_demand: number;
  predicted_next_demand: number;
  initial_stock_level: number;
  target_stock_level: number;
  simulated_stock_level: number;
  reorder_point: number;
  days_of_stock_remaining: number;
  pending_restock_quantity: number;
  next_restock_date: string | null;
  last_restock_date: string | null;
  stockout_days_last_30: number;
  projected_stockout_days: number;
  service_level_last_30: number;
  stockout_risk: number;
  alert_level: "low" | "medium" | "high" | "critical";
  reorder_suggested: boolean;
}

export interface DailySalesTrend {
  sale_date: string;
  total_quantity: number;
  total_revenue: number;
}

export interface TopProduct {
  stock_code: string;
  description: string | null;
  total_revenue: number;
  total_quantity: number;
  num_days_active: number;
}

export interface ModelSummary {
  model_name: string;
  mae: number;
  rmse: number;
  train_rows: number;
  test_rows: number;
  cutoff_date: string;
}

export interface DashboardSummaryResponse {
  total_products: number;
  total_revenue: number;
  total_quantity: number;
  date_range_start: string;
  date_range_end: string;
  total_days: number;
  last_30_days_trend: DailySalesTrend[];
  top_products: TopProduct[];
  best_model: ModelSummary;
}

export interface SingleModelMetrics {
  model_name: string;
  mae: number;
  rmse: number;
  train_time_sec: number;
  rank: number;
  is_best: boolean;
}

export interface SplitInfo {
  cutoff_date: string;
  train_rows: number;
  test_rows: number;
  train_products: number;
  test_products: number;
}

export interface ModelMetricsResponse {
  best_model: string;
  split: SplitInfo;
  models: SingleModelMetrics[];
}

export interface ProductResult {
  stock_code: string;
  description: string | null;
  unit_price: number | null;
}

export interface ProductSearchResponse {
  query: string;
  results: ProductResult[];
  total: number;
}

export interface PredictionHistoryItem {
  id: number;
  endpoint: "forecast" | "inventory" | string;
  query_text: string;
  resolved_stock_code: string;
  model_name: string | null;
  horizon_days: number | null;
  request_payload: Record<string, unknown> | null;
  response_payload: Record<string, unknown>;
  created_at: string;
}

export interface PredictionHistoryResponse {
  client_id: string | null;
  total: number;
  items: PredictionHistoryItem[];
}

// ── Pipeline types ─────────────────────────────────────────────────────────────

export type PipelineStatus =
  | "idle" | "etl_running" | "ml_running" | "reloading" | "completed" | "failed";

export interface PipelineJobResponse {
  job_id: string;
  status: PipelineStatus;
  progress: number;
  message: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  etl_rows: number | null;
  etl_products: number | null;
  best_model: string | null;
  best_rmse: number | null;
}

export interface RequiredField {
  internal_name: string;
  label: string;
  desc: string;
  required: boolean;
}

export interface ColumnInspectResponse {
  columns: string[];
  suggested_mapping: Record<string, string>; // internal_name → user_column
  required_fields: RequiredField[];
}

// ── Analysis types ─────────────────────────────────────────────────────────────

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface LabelValue {
  label: string;
  value: number;
}

export interface ScatterPoint {
  x: number;
  y: number;
  label: string | null;
}

export interface UnivariateAnalysisResponse {
  quantity_histogram: HistogramBin[];
  revenue_histogram: HistogramBin[];
  price_histogram: HistogramBin[];
  sales_by_day_of_week: LabelValue[];
  sales_by_month: LabelValue[];
  products_by_activity: LabelValue[];
}

export interface BivariateAnalysisResponse {
  revenue_by_month: LabelValue[];
  quantity_by_day_of_week: LabelValue[];
  revenue_by_quarter: LabelValue[];
  price_vs_quantity_buckets: ScatterPoint[];
  monthly_quantity_trend: LabelValue[];
}

// ── Fetch helpers ──────────────────────────────────────────────────────────────

function createClientId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getClientId(): string | null {
  if (typeof window === "undefined") return null;

  try {
    const existing = window.localStorage.getItem(CLIENT_ID_STORAGE_KEY);
    if (existing) return existing;

    const nextId = createClientId();
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, nextId);
    return nextId;
  } catch {
    return null;
  }
}

function buildHeaders(headers?: HeadersInit): Headers {
  const merged = new Headers(headers);
  const clientId = getClientId();
  if (clientId) {
    merged.set("X-Client-Id", clientId);
  }
  return merged;
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: init.cache ?? "no-store",
    headers: buildHeaders(init.headers),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getDashboardSummary: () =>
    apiFetch<DashboardSummaryResponse>("/dashboard/summary"),

  getForecast: (stockCode: string, horizon = 7) =>
    apiFetch<ForecastResponse>(`/forecast/${encodeURIComponent(stockCode)}?horizon=${horizon}`),

  getInventory: (stockCode: string) =>
    apiFetch<InventoryResponse>(`/inventory/${encodeURIComponent(stockCode)}`),

  getPredictionHistory: (limit = 20) =>
    apiFetch<PredictionHistoryResponse>(`/history?limit=${limit}`),

  getModelMetrics: () =>
    apiFetch<ModelMetricsResponse>("/models/metrics"),

  searchProducts: (q: string, limit = 20) =>
    apiFetch<ProductSearchResponse>(`/products?q=${encodeURIComponent(q)}&limit=${limit}`),

  getPipelineStatus: () =>
    apiFetch<PipelineJobResponse>("/pipeline/status"),

  inspectColumns: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${API_BASE}/pipeline/inspect`, {
      method: "POST",
      body: form,
      headers: buildHeaders(),
    })
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail ?? `HTTP ${res.status}`);
        }
        return res.json() as Promise<ColumnInspectResponse>;
      });
  },

  uploadDataset: (file: File, mapping: Record<string, string>) => {
    const form = new FormData();
    form.append("file", file);
    form.append("mapping", JSON.stringify(mapping));
    return fetch(`${API_BASE}/pipeline/upload`, {
      method: "POST",
      body: form,
      headers: buildHeaders(),
    })
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail ?? `HTTP ${res.status}`);
        }
        return res.json() as Promise<PipelineJobResponse>;
      });
  },

  getUnivariateAnalysis: () =>
    apiFetch<UnivariateAnalysisResponse>("/analysis/univariate"),

  getBivariateAnalysis: () =>
    apiFetch<BivariateAnalysisResponse>("/analysis/bivariate"),
};
