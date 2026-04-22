"use client";

import { useCallback, useEffect, useState } from "react";
import { Clock3, Package, RefreshCw, TrendingUp } from "lucide-react";
import {
  api,
  ForecastResponse,
  InventoryResponse,
  PredictionHistoryItem,
  PredictionHistoryResponse,
} from "@/lib/api";
import ErrorMessage from "@/components/ErrorMessage";
import LoadingSpinner from "@/components/LoadingSpinner";

function isForecastPayload(payload: Record<string, unknown>): payload is ForecastResponse {
  return Array.isArray(payload.forecast);
}

function isInventoryPayload(payload: Record<string, unknown>): payload is InventoryResponse {
  return typeof payload.alert_level === "string" && typeof payload.stockout_risk === "number";
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function HistoryCard({ item }: { item: PredictionHistoryItem }) {
  const payload = item.response_payload;
  const isForecast = isForecastPayload(payload);
  const isInventory = isInventoryPayload(payload);

  return (
    <article className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center gap-3 justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center ${
              item.endpoint === "forecast" ? "bg-indigo-50 text-indigo-600" : "bg-amber-50 text-amber-600"
            }`}
          >
            {item.endpoint === "forecast" ? <TrendingUp size={18} /> : <Package size={18} />}
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-900">
              {item.endpoint === "forecast" ? "Forecast Query" : "Inventory Query"}
            </div>
            <div className="text-xs text-slate-500">
              Asked for <span className="font-mono text-slate-700">{item.query_text}</span>
              {" "}and resolved to{" "}
              <span className="font-mono text-slate-700">{item.resolved_stock_code}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Clock3 size={13} />
          <span>{formatTimestamp(item.created_at)}</span>
        </div>
      </div>

      <div className="px-6 py-5 space-y-4">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="rounded-lg bg-slate-50 px-4 py-3">
            <div className="text-[11px] uppercase tracking-wide text-slate-400">Model</div>
            <div className="text-sm font-semibold text-slate-700">{item.model_name ?? "—"}</div>
          </div>
          <div className="rounded-lg bg-slate-50 px-4 py-3">
            <div className="text-[11px] uppercase tracking-wide text-slate-400">Stock Code</div>
            <div className="text-sm font-mono font-semibold text-slate-700">{item.resolved_stock_code}</div>
          </div>
          <div className="rounded-lg bg-slate-50 px-4 py-3">
            <div className="text-[11px] uppercase tracking-wide text-slate-400">Horizon</div>
            <div className="text-sm font-semibold text-slate-700">
              {item.horizon_days != null ? `${item.horizon_days} days` : "Not applicable"}
            </div>
          </div>
          <div className="rounded-lg bg-slate-50 px-4 py-3">
            <div className="text-[11px] uppercase tracking-wide text-slate-400">Description</div>
            <div className="text-sm font-semibold text-slate-700 truncate">
              {typeof payload.description === "string" ? payload.description : "—"}
            </div>
          </div>
        </div>

        {isForecast && (
          <div className="space-y-3">
            <div className="text-sm font-semibold text-slate-700">
              Saved forecast response
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50">
                    <th className="text-left px-4 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">Date</th>
                    <th className="text-right px-4 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">Qty</th>
                    <th className="text-right px-4 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">Revenue</th>
                  </tr>
                </thead>
                <tbody>
                  {payload.forecast.slice(0, 7).map((point) => (
                    <tr key={point.forecast_date} className="border-t border-slate-100">
                      <td className="px-4 py-2 font-mono text-slate-600">{point.forecast_date}</td>
                      <td className="px-4 py-2 text-right font-semibold text-indigo-600">
                        {point.predicted_quantity.toFixed(1)}
                      </td>
                      <td className="px-4 py-2 text-right text-slate-600">
                        {point.predicted_revenue != null ? `£${point.predicted_revenue.toFixed(2)}` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {isInventory && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="rounded-lg border border-slate-200 px-4 py-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-400">Alert Level</div>
              <div className="text-sm font-semibold text-slate-700 capitalize">{payload.alert_level}</div>
            </div>
            <div className="rounded-lg border border-slate-200 px-4 py-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-400">Stockout Risk</div>
              <div className="text-sm font-semibold text-slate-700">{(payload.stockout_risk * 100).toFixed(1)}%</div>
            </div>
            <div className="rounded-lg border border-slate-200 px-4 py-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-400">Days Left</div>
              <div className="text-sm font-semibold text-slate-700">{payload.days_of_stock_remaining} days</div>
            </div>
            <div className="rounded-lg border border-slate-200 px-4 py-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-400">Reorder Point</div>
              <div className="text-sm font-semibold text-slate-700">{payload.reorder_point.toFixed(1)}</div>
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

export default function HistoryPage() {
  const [history, setHistory] = useState<PredictionHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async (showSpinner = true) => {
    if (showSpinner) {
      setLoading(true);
    }
    setError(null);
    try {
      const data = await api.getPredictionHistory(20);
      setHistory(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadHistory(false);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadHistory]);

  return (
    <div className="p-8 space-y-8">
      <div className="flex flex-wrap gap-4 items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Prediction History</h1>
          <p className="text-sm text-slate-500 mt-1">
            Recent forecast and inventory queries saved for this browser session
          </p>
        </div>
        <button
          onClick={() => void loadHistory()}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {error && <ErrorMessage message={error} />}
      {loading && <LoadingSpinner label="Loading saved prediction history…" />}

      {!loading && history && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Saved Queries</div>
              <div className="text-2xl font-bold text-slate-900">{history.total}</div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Forecast Queries</div>
              <div className="text-2xl font-bold text-indigo-600">
                {history.items.filter((item) => item.endpoint === "forecast").length}
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Inventory Queries</div>
              <div className="text-2xl font-bold text-amber-600">
                {history.items.filter((item) => item.endpoint === "inventory").length}
              </div>
            </div>
          </div>

          {history.items.length === 0 ? (
            <div className="bg-white rounded-xl border border-dashed border-slate-300 p-10 text-center text-sm text-slate-500">
              No saved prediction history yet. Run a forecast or inventory query and it will appear here.
            </div>
          ) : (
            <div className="space-y-5">
              {history.items.map((item) => (
                <HistoryCard key={item.id} item={item} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
