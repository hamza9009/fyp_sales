"use client";

import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ScatterChart, Scatter,
  LineChart, Line,
} from "recharts";
import {
  api,
  UnivariateAnalysisResponse,
  BivariateAnalysisResponse,
  HistogramBin,
  LabelValue,
} from "@/lib/api";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorMessage from "@/components/ErrorMessage";

type Tab = "univariate" | "bivariate";

const PALETTE = [
  "#6366f1", "#06b6d4", "#10b981", "#f59e0b",
  "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6",
];

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h3 className="text-sm font-semibold text-slate-700 mb-3">{children}</h3>;
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
      <SectionTitle>{title}</SectionTitle>
      {children}
    </div>
  );
}

function HistogramChart({ data, color = "#6366f1", yLabel = "Count" }: {
  data: HistogramBin[];
  color?: string;
  yLabel?: string;
}) {
  const chartData = data.map((b) => ({
    bin: `${b.bin_start.toFixed(1)}`,
    count: b.count,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="bin" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={44}
          label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 10, fill: "#94a3b8" } }} />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v: unknown) => [Number(v).toLocaleString(), "Count"]}
        />
        <Bar dataKey="count" fill={color} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function CategoryBar({ data, color = "#6366f1", valueFormatter }: {
  data: LabelValue[];
  color?: string;
  valueFormatter?: (v: number) => string;
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="label" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={52}
          tickFormatter={valueFormatter ?? ((v) => v.toFixed(0))} />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v: unknown) => [valueFormatter ? valueFormatter(Number(v)) : Number(v).toFixed(2), ""]}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function MonthlyTrend({ data }: { data: LabelValue[] }) {
  // Show only every 3rd label for readability
  const thinned = data.map((d, i) => ({ ...d, displayLabel: i % 3 === 0 ? d.label : "" }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={thinned} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="displayLabel" tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={56}
          tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v: unknown) => [Number(v).toLocaleString(), "Units"]}
          labelFormatter={(_: unknown, payload) => payload?.[0]?.payload?.label ?? ""}
        />
        <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function PriceScatter({ data }: { data: { x: number; y: number; label: string | null }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ScatterChart margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="x" type="number" tick={{ fontSize: 10 }} tickLine={false} axisLine={false}
          label={{ value: "Unit Price (£)", position: "insideBottom", offset: -2, style: { fontSize: 10, fill: "#94a3b8" } }} />
        <YAxis dataKey="y" type="number" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={44}
          label={{ value: "Avg Daily Qty", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 10, fill: "#94a3b8" } }} />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v: unknown, name: string) => [Number(v).toFixed(2), name === "x" ? "Price (£)" : "Avg Daily Qty"]}
        />
        <Scatter data={data} fill="#6366f1" opacity={0.75} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export default function AnalysisPage() {
  const [tab, setTab] = useState<Tab>("univariate");
  const [uni, setUni] = useState<UnivariateAnalysisResponse | null>(null);
  const [bi, setBi] = useState<BivariateAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([api.getUnivariateAnalysis(), api.getBivariateAnalysis()])
      .then(([u, b]) => { setUni(u); setBi(b); })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Computing analysis…" />;
  if (error) return <div className="p-8"><ErrorMessage message={`Analysis failed: ${error}`} /></div>;
  if (!uni || !bi) return null;

  const fmtRevenue = (v: number) =>
    v >= 1_000_000 ? `£${(v / 1_000_000).toFixed(1)}M` : `£${(v / 1_000).toFixed(0)}K`;

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Data Analysis</h1>
        <p className="text-sm text-slate-500 mt-1">
          Univariate distributions and bivariate relationships in the loaded dataset
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-slate-100 rounded-lg p-1 w-fit">
        {(["univariate", "bivariate"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-5 py-1.5 rounded-md text-sm font-semibold capitalize transition-colors ${
              tab === t ? "bg-white text-indigo-700 shadow-sm" : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* ── Univariate ─────────────────────────────────────────────────────────── */}
      {tab === "univariate" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="Daily Quantity Distribution (per product-day)">
              <HistogramChart data={uni.quantity_histogram} color="#6366f1" yLabel="Days" />
              <p className="text-xs text-slate-400 mt-2">
                Top 1% outliers excluded for readability.
              </p>
            </ChartCard>

            <ChartCard title="Daily Revenue Distribution (per product-day)">
              <HistogramChart data={uni.revenue_histogram} color="#10b981" yLabel="Days" />
              <p className="text-xs text-slate-400 mt-2">
                Top 1% outliers excluded for readability.
              </p>
            </ChartCard>

            <ChartCard title="Product Unit Price Distribution">
              <HistogramChart data={uni.price_histogram} color="#f59e0b" yLabel="Products" />
            </ChartCard>

            <ChartCard title="Avg Daily Quantity by Day of Week">
              <CategoryBar data={uni.sales_by_day_of_week} color="#6366f1" />
            </ChartCard>

            <ChartCard title="Avg Daily Quantity by Month">
              <CategoryBar data={uni.sales_by_month} color="#06b6d4" />
            </ChartCard>

            <ChartCard title="Products by Number of Active Days">
              <CategoryBar data={uni.products_by_activity} color="#8b5cf6"
                valueFormatter={(v) => v.toFixed(0)} />
              <p className="text-xs text-slate-400 mt-2">
                Number of unique dates each product appears in sales history.
              </p>
            </ChartCard>
          </div>
        </div>
      )}

      {/* ── Bivariate ──────────────────────────────────────────────────────────── */}
      {tab === "bivariate" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="Total Revenue by Month (all years aggregated)">
              <CategoryBar data={bi.revenue_by_month} color="#6366f1"
                valueFormatter={fmtRevenue} />
            </ChartCard>

            <ChartCard title="Avg Daily Quantity by Day of Week">
              <CategoryBar data={bi.quantity_by_day_of_week} color="#10b981" />
            </ChartCard>

            <ChartCard title="Total Revenue by Quarter">
              <CategoryBar data={bi.revenue_by_quarter} color="#f59e0b"
                valueFormatter={fmtRevenue} />
            </ChartCard>

            <ChartCard title="Unit Price vs Avg Daily Demand (price buckets)">
              <PriceScatter data={bi.price_vs_quantity_buckets} />
              <p className="text-xs text-slate-400 mt-2">
                Each point represents a unit-price bucket; y-axis is mean daily demand within that bucket.
              </p>
            </ChartCard>
          </div>

          <ChartCard title="Monthly Total Quantity Trend (full dataset)">
            <MonthlyTrend data={bi.monthly_quantity_trend} />
            <p className="text-xs text-slate-400 mt-2">
              Total units sold per calendar month across all products.
            </p>
          </ChartCard>
        </div>
      )}
    </div>
  );
}
