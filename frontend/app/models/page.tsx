"use client";

import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Cell,
} from "recharts";
import { Trophy } from "lucide-react";
import { api, ModelMetricsResponse, SingleModelMetrics } from "@/lib/api";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorMessage from "@/components/ErrorMessage";

const COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"];

function cleanName(name: string) {
  return name.replace(" (lag-1)", "");
}

function buildBarData(models: SingleModelMetrics[]) {
  return models.map((m) => ({
    name: cleanName(m.model_name),
    MAE: +m.mae.toFixed(4),
    RMSE: +m.rmse.toFixed(4),
    "Train Time (s)": +m.train_time_sec.toFixed(2),
  }));
}

function buildRadarData(models: SingleModelMetrics[]) {
  const keys = ["mae", "rmse", "train_time_sec"] as const;
  const maxVals = Object.fromEntries(
    keys.map((k) => [k, Math.max(...models.map((m) => m[k]))])
  ) as Record<(typeof keys)[number], number>;
  const minVals = Object.fromEntries(
    keys.map((k) => [k, Math.min(...models.map((m) => m[k]))])
  ) as Record<(typeof keys)[number], number>;

  const dims: { key: (typeof keys)[number]; label: string }[] = [
    { key: "mae",           label: "MAE" },
    { key: "rmse",          label: "RMSE" },
    { key: "train_time_sec", label: "Speed" },
  ];

  return dims.map((d) => {
    const row: Record<string, string | number> = { metric: d.label };
    for (const model of models) {
      const range = maxVals[d.key] - minVals[d.key];
      row[cleanName(model.model_name)] =
        range === 0 ? 100 : Math.round(((maxVals[d.key] - model[d.key]) / range) * 100);
    }
    return row;
  });
}

const ttStyle = { fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" };

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
      <h2 className="text-sm font-semibold text-slate-700">{title}</h2>
      {subtitle && <p className="text-xs text-slate-400 mt-0.5 mb-4">{subtitle}</p>}
      {!subtitle && <div className="mb-4" />}
      {children}
    </div>
  );
}

export default function ModelsPage() {
  const [data, setData] = useState<ModelMetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getModelMetrics()
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error)
    return (
      <div className="p-8">
        <ErrorMessage message={`Failed to load metrics: ${error}`} />
      </div>
    );
  if (!data) return <LoadingSpinner label="Loading model metrics…" />;

  const barData   = buildBarData(data.models);
  const radarData = buildRadarData(data.models);
  const modelNames = data.models.map((m) => cleanName(m.model_name));

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Model Performance</h1>
        <p className="text-sm text-slate-500 mt-1">
          Comparative evaluation — time-based train/test split, no data leakage
        </p>
      </div>

      {/* Best model banner */}
      <div className="flex items-center gap-4 bg-indigo-50 border border-indigo-200 rounded-xl px-6 py-4">
        <div className="w-10 h-10 bg-indigo-600 rounded-full flex items-center justify-center shrink-0">
          <Trophy size={18} className="text-white" />
        </div>
        <div>
          <div className="text-xs text-indigo-400 uppercase tracking-wide font-semibold">
            Best Model (lowest RMSE)
          </div>
          <div className="text-lg font-bold text-indigo-700">{data.best_model}</div>
        </div>
      </div>

      {/* Split KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {[
          { label: "Cutoff Date",    value: data.split.cutoff_date },
          { label: "Train Rows",     value: data.split.train_rows.toLocaleString() },
          { label: "Test Rows",      value: data.split.test_rows.toLocaleString() },
          { label: "Train Products", value: data.split.train_products.toLocaleString() },
          { label: "Test Products",  value: data.split.test_products.toLocaleString() },
        ].map(({ label, value }) => (
          <div key={label} className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
            <p className="text-xs text-slate-400 uppercase tracking-wide">{label}</p>
            <p className="font-semibold text-slate-800 mt-1">{value}</p>
          </div>
        ))}
      </div>

      {/* MAE & RMSE — grouped bar */}
      <ChartCard
        title="MAE & RMSE Comparison"
        subtitle="Lower is better — RMSE is the primary model-selection criterion"
      >
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={50} />
            <Tooltip
              contentStyle={ttStyle}
              formatter={(v, name) => [`${Number(v).toFixed(4)}`, String(name)]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="MAE"  fill="#6366f1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="RMSE" fill="#22d3ee" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Training Time */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ChartCard
          title="Training Time"
          subtitle="Wall-clock seconds to fit the model — reflects computational cost"
        >
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={45}
                unit="s"
              />
              <Tooltip
                contentStyle={ttStyle}
                formatter={(v) => [`${Number(v).toFixed(2)}s`, "Train Time"]}
              />
              <Bar dataKey="Train Time (s)" radius={[4, 4, 0, 0]}>
                {barData.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 flex items-center">
          <p className="text-sm text-slate-500">
            Evaluation focuses on `MAE` and `RMSE`. `RMSE` remains the selection metric for the best model.
          </p>
        </div>
      </div>

      {/* Performance Radar */}
      <ChartCard
        title="Performance Radar"
        subtitle="Normalized score per axis — higher = better. Best performer on each metric scores 100."
      >
        <ResponsiveContainer width="100%" height={340}>
          <RadarChart data={radarData} margin={{ top: 10, right: 40, left: 40, bottom: 10 }}>
            <PolarGrid stroke="#e2e8f0" />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: "#64748b" }} />
            <PolarRadiusAxis
              angle={90}
              domain={[0, 100]}
              tick={{ fontSize: 10 }}
              tickCount={5}
            />
            {modelNames.map((name, idx) => (
              <Radar
                key={name}
                name={name}
                dataKey={name}
                stroke={COLORS[idx % COLORS.length]}
                fill={COLORS[idx % COLORS.length]}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            ))}
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={ttStyle}
              formatter={(v, name) => [`${Number(v)}/100`, String(name)]}
            />
          </RadarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Full metrics table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100">
          <h2 className="text-sm font-semibold text-slate-700">Full Metrics Report</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50">
                {["Rank", "Model", "MAE", "RMSE", "Train Time (s)", ""].map((h) => (
                  <th
                    key={h}
                    className="text-left px-6 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.models.map((m, i) => (
                <tr key={m.model_name} className={i % 2 === 0 ? "" : "bg-slate-50"}>
                  <td className="px-6 py-3 font-bold text-slate-500">#{m.rank}</td>
                  <td className="px-6 py-3 font-semibold text-slate-800">{m.model_name}</td>
                  <td className="px-6 py-3 text-slate-600">{m.mae.toFixed(4)}</td>
                  <td className="px-6 py-3 text-slate-600">{m.rmse.toFixed(4)}</td>
                  <td className="px-6 py-3 text-slate-600">{m.train_time_sec.toFixed(2)}s</td>
                  <td className="px-6 py-3">
                    {m.is_best && (
                      <span className="inline-flex items-center gap-1 text-xs font-semibold bg-indigo-100 text-indigo-700 px-2.5 py-0.5 rounded-full">
                        <Trophy size={11} /> Best
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Notes */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-sm text-amber-800 space-y-2">
        <div className="font-semibold">About these metrics</div>
        <ul className="list-disc list-inside space-y-1 text-amber-700">
          <li>MAE: average absolute units off per prediction — easy to interpret</li>
          <li>RMSE: penalises large errors more heavily — primary selection criterion</li>
          <li>Radar scores are normalized per axis: best model = 100, worst = 0</li>
        </ul>
      </div>
    </div>
  );
}
