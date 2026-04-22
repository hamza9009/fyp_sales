"use client";

import { useState } from "react";
import { Search, AlertTriangle, CheckCircle, XCircle, Info } from "lucide-react";
import { api, InventoryResponse } from "@/lib/api";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorMessage from "@/components/ErrorMessage";

const SAMPLE_CODES = ["85123A", "85099B", "22423", "47566", "20725"];

const ALERT_CONFIG = {
  low:      { color: "text-green-600",  bg: "bg-green-50  border-green-200",  icon: CheckCircle,   label: "Low Risk" },
  medium:   { color: "text-amber-600",  bg: "bg-amber-50  border-amber-200",  icon: AlertTriangle, label: "Medium Risk" },
  high:     { color: "text-orange-600", bg: "bg-orange-50 border-orange-200", icon: AlertTriangle, label: "High Risk" },
  critical: { color: "text-rose-600",   bg: "bg-rose-50   border-rose-200",   icon: XCircle,       label: "Critical" },
};

function RiskBar({ risk }: { risk: number }) {
  const pct = Math.round(risk * 100);
  const color = risk < 0.25 ? "bg-green-500" : risk < 0.5 ? "bg-amber-500" : risk < 0.75 ? "bg-orange-500" : "bg-rose-500";
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-slate-500 mb-1.5">
        <span>Stockout Risk</span>
        <span className="font-semibold">{pct}%</span>
      </div>
      <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function InventoryPage() {
  const [stockCode, setStockCode] = useState("");
  const [result, setResult] = useState<InventoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(code?: string) {
    const query = (code ?? stockCode).trim().toUpperCase();
    if (!query) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.getInventory(query);
      setResult(data);
      setStockCode(data.stock_code);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const cfg = result ? ALERT_CONFIG[result.alert_level] : null;

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Inventory Alerts</h1>
        <p className="text-sm text-slate-500 mt-1">
          Stockout risk, reorder intelligence, and demand signals per product
        </p>
      </div>

      {/* Search panel */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4">
        <div className="flex gap-3 items-end flex-wrap">
          <div className="flex-1 min-w-48">
            <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
              Stock Code or Description
            </label>
            <input
              type="text"
              value={stockCode}
              onChange={(e) => setStockCode(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="e.g. 85123A or WHITE HANGING HEART"
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>
          <button
            onClick={() => handleSearch()}
            disabled={loading || !stockCode.trim()}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
          >
            <Search size={15} />
            {loading ? "Checking…" : "Check Stock"}
          </button>
        </div>
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
      {loading && <LoadingSpinner label="Computing inventory signals…" />}

      {result && cfg && (
        <>
          {/* Alert banner */}
          <div className={`flex items-start gap-4 rounded-xl border p-5 ${cfg.bg}`}>
            <cfg.icon size={22} className={`${cfg.color} shrink-0 mt-0.5`} />
            <div>
              <div className={`font-bold text-base ${cfg.color}`}>{cfg.label}</div>
              <div className="text-sm text-slate-600 mt-0.5">
                {result.stock_code} · {result.description ?? "—"}
              </div>
              {result.reorder_suggested && (
                <div className="mt-2 text-sm font-semibold text-rose-600">
                  ⚠ Reorder recommended — stock at or below reorder point
                </div>
              )}
            </div>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: "Avg Daily Demand",    value: `${result.avg_daily_demand.toFixed(1)} units`,   sub: "last 28 days" },
              { label: "Next-Day Forecast",   value: `${result.predicted_next_demand.toFixed(1)} units`, sub: "model prediction" },
              { label: "Days of Stock Left",  value: `${result.days_of_stock_remaining} days`,        sub: "at current demand" },
              { label: "Simulated Stock",     value: result.simulated_stock_level.toFixed(0),          sub: "estimated units" },
              { label: "Reorder Point",       value: result.reorder_point.toFixed(0),                 sub: "trigger threshold" },
              { label: "Initial Stock",       value: result.initial_stock_level.toFixed(0),            sub: "simulation starting stock" },
              { label: "Target Stock",        value: result.target_stock_level.toFixed(0),             sub: "order-up-to level" },
              { label: "Pending Restock",     value: result.pending_restock_quantity.toFixed(0),       sub: result.next_restock_date ? `arrives ${result.next_restock_date}` : "no open replenishment" },
              { label: "Stockout Days",       value: `${result.stockout_days_last_30} / 30`,           sub: "historical replay" },
              { label: "Service Level",       value: `${(result.service_level_last_30 * 100).toFixed(1)}%`, sub: "last 30 simulated days" },
              { label: "Projected Stockouts", value: `${result.projected_stockout_days}`,              sub: "forward simulation horizon" },
              { label: "Last Restock",        value: result.last_restock_date ?? "—",                  sub: "most recent replenishment" },
              { label: "As Of",               value: result.as_of_date,                                sub: "last observed date" },
            ].map(({ label, value, sub }) => (
              <div key={label} className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">{label}</div>
                <div className="text-lg font-bold text-slate-800">{value}</div>
                <div className="text-xs text-slate-400 mt-0.5">{sub}</div>
              </div>
            ))}
          </div>

          {/* Risk bar */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <RiskBar risk={result.stockout_risk} />
            <div className="mt-5 flex items-start gap-2 text-xs text-slate-400">
              <Info size={13} className="shrink-0 mt-0.5" />
              <span>
                Inventory is simulated day by day using the replay equation
                {" "}
                <span className="font-mono text-slate-500">stock[t] = stock[t-1] - demand[t] + restock[t]</span>.
                The UCI dataset does not include on-hand stock, so reorder events and stockout days are generated from this replenishment policy.
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
