"use client";

import { useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { Search } from "lucide-react";
import { api, ForecastResponse } from "@/lib/api";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorMessage from "@/components/ErrorMessage";
import ProductSearch from "@/components/ProductSearch";

const SAMPLE_CODES = ["85123A", "85099B", "22423", "47566", "20725"];

export default function ForecastPage() {
  const [stockCode, setStockCode] = useState("");
  const [horizon, setHorizon] = useState(7);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(code?: string) {
    const query = (code ?? stockCode).trim().toUpperCase();
    if (!query) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.getForecast(query, horizon);
      setResult(data);
      setStockCode(data.stock_code);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const chartData = result?.forecast.map((p) => ({
    date: p.forecast_date.slice(5),
    quantity: p.predicted_quantity,
    revenue: p.predicted_revenue ?? undefined,
  })) ?? [];

  const maxQty = result ? Math.max(...result.forecast.map((p) => p.predicted_quantity)) : 0;
  const avgQty = result
    ? result.forecast.reduce((s, p) => s + p.predicted_quantity, 0) / result.forecast.length
    : 0;

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Product Forecast</h1>
        <p className="text-sm text-slate-500 mt-1">
          Multi-step daily demand prediction using the best trained model
        </p>
      </div>

      {/* Search panel */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4">
        <div className="flex gap-3 items-end flex-wrap">
          <div className="flex-1 min-w-48">
            <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
              Stock Code or Description
            </label>
            <ProductSearch
              value={stockCode}
              onChange={setStockCode}
              onSelect={(code) => handleSearch(code)}
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
              Horizon (days)
            </label>
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 bg-white"
            >
              {[3, 5, 7, 10, 14].map((d) => (
                <option key={d} value={d}>{d} days</option>
              ))}
            </select>
          </div>
          <button
            onClick={() => handleSearch()}
            disabled={loading || !stockCode.trim()}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
          >
            <Search size={15} />
            {loading ? "Searching…" : "Forecast"}
          </button>
        </div>

        {/* Sample codes */}
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-xs text-slate-400">Try:</span>
          {SAMPLE_CODES.map((code) => (
            <button
              key={code}
              onClick={() => { setStockCode(code); handleSearch(code); }}
              className="text-xs px-2.5 py-1 rounded-full bg-slate-100 hover:bg-indigo-50 hover:text-indigo-600 text-slate-600 transition-colors font-mono"
            >
              {code}
            </button>
          ))}
        </div>
      </div>

      {error && <ErrorMessage message={error} />}
      {loading && <LoadingSpinner label="Generating forecast…" />}

      {result && (
        <>
          {/* Product info */}
          <div className="flex flex-wrap gap-4 items-center bg-white rounded-xl border border-slate-200 shadow-sm px-6 py-4">
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wide">Product</span>
              <div className="font-mono font-bold text-indigo-600">{result.stock_code}</div>
            </div>
            <div className="h-8 w-px bg-slate-200" />
            <div className="flex-1">
              <span className="text-xs text-slate-400 uppercase tracking-wide">Description</span>
              <div className="font-medium text-slate-700 text-sm">{result.description ?? "—"}</div>
            </div>
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wide">Model</span>
              <div className="text-sm font-semibold text-slate-700">{result.model_name}</div>
            </div>
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wide">Peak Demand</span>
              <div className="text-sm font-bold text-green-600">{maxQty.toFixed(0)} units</div>
            </div>
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wide">Avg Demand</span>
              <div className="text-sm font-bold text-slate-700">{avgQty.toFixed(1)} units/day</div>
            </div>
          </div>

          {/* Forecast chart */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <h2 className="text-sm font-semibold text-slate-700 mb-4">
              {result.horizon_days}-Day Demand Forecast
            </h2>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={48}
                  label={{ value: "Units", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 11, fill: "#94a3b8" } }}
                />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
                  formatter={(v, name) => {
                    const n = Number(v);
                    return name === "quantity"
                      ? [`${n.toFixed(1)} units`, "Predicted Qty"]
                      : [`£${n.toFixed(2)}`, "Predicted Revenue"];
                  }}
                />
                <ReferenceLine y={avgQty} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "avg", fontSize: 10, fill: "#94a3b8" }} />
                <Area
                  type="monotone"
                  dataKey="quantity"
                  stroke="#6366f1"
                  strokeWidth={2.5}
                  fill="url(#forecastGrad)"
                  dot={{ fill: "#6366f1", r: 4, strokeWidth: 0 }}
                  activeDot={{ r: 6 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Detailed table */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100">
              <h2 className="text-sm font-semibold text-slate-700">Daily Breakdown</h2>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50">
                  <th className="text-left px-6 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Date</th>
                  <th className="text-right px-6 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Predicted Qty</th>
                  <th className="text-right px-6 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Predicted Revenue</th>
                </tr>
              </thead>
              <tbody>
                {result.forecast.map((p, i) => (
                  <tr key={p.forecast_date} className={i % 2 === 0 ? "" : "bg-slate-50"}>
                    <td className="px-6 py-3 font-mono text-slate-600">{p.forecast_date}</td>
                    <td className="px-6 py-3 text-right font-semibold text-indigo-600">
                      {p.predicted_quantity.toFixed(1)}
                    </td>
                    <td className="px-6 py-3 text-right text-slate-600">
                      {p.predicted_revenue != null ? `£${p.predicted_revenue.toFixed(2)}` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
