"use client";

import { useEffect, useState } from "react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import {
  api,
  DashboardSummaryResponse,
  UnivariateAnalysisResponse,
  BivariateAnalysisResponse,
  HistogramBin,
  LabelValue,
} from "@/lib/api";
import StatCard from "@/components/StatCard";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorMessage from "@/components/ErrorMessage";

type Tab = "overview" | "univariate" | "bivariate";

// ── Formatters ─────────────────────────────────────────────────────────────────

function fmt(n: number) {
  if (n >= 1_000_000) return `£${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `£${(n / 1_000).toFixed(1)}K`;
  return `£${n.toFixed(0)}`;
}

function fmtQty(n: number) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

// ── Palette ────────────────────────────────────────────────────────────────────

const PALETTE = [
  "#6366f1", "#06b6d4", "#10b981", "#f59e0b",
  "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6",
];

// ── Small reusable chart wrappers ──────────────────────────────────────────────

function ChartCard({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
      <p className="text-sm font-semibold text-slate-700 mb-4">{title}</p>
      {children}
      {note && <p className="text-xs text-slate-400 mt-2">{note}</p>}
    </div>
  );
}

function HistBar({ data, color, yLabel }: { data: HistogramBin[]; color: string; yLabel: string }) {
  const d = data.map((b) => ({ bin: b.bin_start.toFixed(1), count: b.count }));
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={d} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="bin" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={44}
          label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 10, fill: "#94a3b8" } }} />
        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v) => [Number(v).toLocaleString(), "Count"]} />
        <Bar dataKey="count" fill={color} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function CatBar({ data, valueFormatter }: { data: LabelValue[]; valueFormatter?: (v: number) => string }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="label" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={52}
          tickFormatter={valueFormatter ?? ((v) => Number(v).toFixed(0))} />
        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v) => [valueFormatter ? valueFormatter(Number(v)) : Number(v).toFixed(2), ""]} />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((_e, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function TrendLine({ data }: { data: LabelValue[] }) {
  const thinned = data.map((d, i) => ({ ...d, displayLabel: i % 3 === 0 ? d.label : "" }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={thinned} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="displayLabel" tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={56}
          tickFormatter={(v) => `${(Number(v) / 1000).toFixed(0)}K`} />
        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v) => [Number(v).toLocaleString(), "Units"]}
          labelFormatter={(_l, p) => (p && p[0] ? String(p[0].payload.label) : "")} />
        <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function PriceDemand({ data }: { data: { x: number; y: number; label: string | null }[] }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <ScatterChart margin={{ top: 4, right: 8, left: 0, bottom: 16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="x" type="number" tick={{ fontSize: 10 }} tickLine={false} axisLine={false}
          label={{ value: "Unit Price (£)", position: "insideBottom", offset: -8, style: { fontSize: 10, fill: "#94a3b8" } }} />
        <YAxis dataKey="y" type="number" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={44}
          label={{ value: "Avg Daily Qty", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 10, fill: "#94a3b8" } }} />
        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }}
          formatter={(v, name) => [Number(v).toFixed(2), name === "x" ? "Price (£)" : "Avg Qty"]} />
        <Scatter data={data} fill="#6366f1" opacity={0.8} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ── Tab button ─────────────────────────────────────────────────────────────────

function TabBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick}
      className={`px-5 py-1.5 rounded-md text-sm font-semibold transition-colors ${
        active ? "bg-white text-indigo-700 shadow-sm" : "text-slate-500 hover:text-slate-700"
      }`}>
      {children}
    </button>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [tab, setTab] = useState<Tab>("overview");
  const [summary, setSummary] = useState<DashboardSummaryResponse | null>(null);
  const [uni, setUni] = useState<UnivariateAnalysisResponse | null>(null);
  const [bi, setBi] = useState<BivariateAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getDashboardSummary(),
      api.getUnivariateAnalysis(),
      api.getBivariateAnalysis(),
    ])
      .then(([s, u, b]) => { setSummary(s); setUni(u); setBi(b); })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading dashboard…" />;
  if (error) return <div className="p-8"><ErrorMessage message={`Dashboard failed: ${error}`} /></div>;
  if (!summary || !uni || !bi) return null;

  const trendData = summary.last_30_days_trend.map((d) => ({
    date: d.sale_date.slice(5),
    revenue: Math.round(d.total_revenue),
    quantity: d.total_quantity,
  }));

  const topData = summary.top_products.map((p) => ({
    name: p.stock_code,
    revenue: Math.round(p.total_revenue),
    desc: p.description ?? p.stock_code,
  }));

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-sm text-slate-500 mt-1">
          {summary.date_range_start} → {summary.date_range_end} · {summary.total_days} days of data
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-slate-100 rounded-lg p-1 w-fit">
        <TabBtn active={tab === "overview"}    onClick={() => setTab("overview")}>Overview</TabBtn>
        <TabBtn active={tab === "univariate"}  onClick={() => setTab("univariate")}>Univariate</TabBtn>
        <TabBtn active={tab === "bivariate"}   onClick={() => setTab("bivariate")}>Bivariate</TabBtn>
      </div>

      {/* ── OVERVIEW ──────────────────────────────────────────────────────────── */}
      {tab === "overview" && (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard title="Total Products"  value={summary.total_products.toLocaleString()} sub="unique SKUs" accent="indigo" />
            <StatCard title="Total Revenue"   value={fmt(summary.total_revenue)}              sub="all-time"    accent="green" />
            <StatCard title="Units Sold"      value={fmtQty(summary.total_quantity)}          sub="all-time"    accent="amber" />
            <StatCard title="Best Model RMSE" value={summary.best_model.rmse.toFixed(2)}      sub={summary.best_model.model_name} accent="rose" />
          </div>

          {/* Revenue trend */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <p className="text-sm font-semibold text-slate-700 mb-4">Revenue Trend — Last 30 Days</p>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={trendData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false}
                  tickFormatter={(v) => `£${(Number(v) / 1000).toFixed(0)}K`} width={56} />
                <Tooltip formatter={(v) => [`£${Number(v).toLocaleString()}`, "Revenue"]}
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                <Area type="monotone" dataKey="revenue" stroke="#6366f1" strokeWidth={2} fill="url(#revGrad)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Top products */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <p className="text-sm font-semibold text-slate-700 mb-4">Top 10 Products by Revenue</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={topData} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} tickLine={false} axisLine={false}
                  tickFormatter={(v) => `£${(Number(v) / 1000).toFixed(0)}K`} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={64} />
                <Tooltip
                  formatter={(v, _n, item) => [`£${Number(v).toLocaleString()}`, (item as { payload?: { desc?: string } }).payload?.desc ?? "Revenue"]}
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                <Bar dataKey="revenue" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Best model */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <p className="text-sm font-semibold text-slate-700 mb-4">Best Model Performance</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {[
                { label: "Model",        value: summary.best_model.model_name },
                { label: "MAE",          value: summary.best_model.mae.toFixed(4) },
                { label: "RMSE",         value: summary.best_model.rmse.toFixed(4) },
                { label: "Train / Test", value: `${summary.best_model.train_rows.toLocaleString()} / ${summary.best_model.test_rows.toLocaleString()}` },
              ].map(({ label, value }) => (
                <div key={label} className="flex flex-col gap-1">
                  <span className="text-xs text-slate-400 uppercase tracking-wide">{label}</span>
                  <span className="font-semibold text-slate-800">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ── UNIVARIATE ────────────────────────────────────────────────────────── */}
      {tab === "univariate" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartCard title="Daily Quantity Distribution" note="Top 1% outliers excluded.">
            <HistBar data={uni.quantity_histogram} color="#6366f1" yLabel="Days" />
          </ChartCard>

          <ChartCard title="Daily Revenue Distribution" note="Top 1% outliers excluded.">
            <HistBar data={uni.revenue_histogram} color="#10b981" yLabel="Days" />
          </ChartCard>

          <ChartCard title="Product Unit Price Distribution">
            <HistBar data={uni.price_histogram} color="#f59e0b" yLabel="Products" />
          </ChartCard>

          <ChartCard title="Avg Daily Quantity by Day of Week">
            <CatBar data={uni.sales_by_day_of_week} />
          </ChartCard>

          <ChartCard title="Avg Daily Quantity by Month">
            <CatBar data={uni.sales_by_month} />
          </ChartCard>

          <ChartCard title="Products by Number of Active Days"
            note="Number of unique sale dates per product.">
            <CatBar data={uni.products_by_activity} valueFormatter={(v) => v.toFixed(0)} />
          </ChartCard>
        </div>
      )}

      {/* ── BIVARIATE ─────────────────────────────────────────────────────────── */}
      {tab === "bivariate" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="Total Revenue by Month (all years)">
              <CatBar data={bi.revenue_by_month} valueFormatter={fmt} />
            </ChartCard>

            <ChartCard title="Avg Daily Demand by Day of Week">
              <CatBar data={bi.quantity_by_day_of_week} />
            </ChartCard>

            <ChartCard title="Total Revenue by Quarter">
              <CatBar data={bi.revenue_by_quarter} valueFormatter={fmt} />
            </ChartCard>

            <ChartCard title="Unit Price vs Avg Daily Demand"
              note="Each point = a price bucket. Shows how price level relates to average daily demand.">
              <PriceDemand data={bi.price_vs_quantity_buckets} />
            </ChartCard>
          </div>

          <ChartCard title="Monthly Total Quantity Trend (full dataset)"
            note="Total units sold per calendar month across all products.">
            <TrendLine data={bi.monthly_quantity_trend} />
          </ChartCard>
        </div>
      )}
    </div>
  );
}
