"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload, CheckCircle, XCircle, Loader2, FileText, RefreshCw,
  ArrowRight, ArrowLeft, AlertCircle, Check,
} from "lucide-react";
import {
  api, ColumnInspectResponse, PipelineJobResponse, PipelineStatus, RequiredField,
} from "@/lib/api";

// ── Pipeline step config ───────────────────────────────────────────────────────

const PIPELINE_STEPS: { statuses: PipelineStatus[]; label: string; desc: string }[] = [
  { statuses: ["etl_running"], label: "Extract, Clean & Transform",
    desc: "Parsing file, removing cancellations, building daily aggregates" },
  { statuses: ["ml_running"],  label: "ML Training",
    desc: "Training LightGBM and XGBoost, then averaging them into an ensemble" },
  { statuses: ["reloading"],   label: "Reload Caches",
    desc: "Updating in-memory data store and ML model for live predictions" },
  { statuses: ["completed"],   label: "Ready",
    desc: "All endpoints now serve predictions from the new dataset" },
];

const RUNNING: PipelineStatus[] = ["etl_running", "ml_running", "reloading"];
const ALL_STATUSES: PipelineStatus[] = ["etl_running", "ml_running", "reloading", "completed"];

function stepState(step: typeof PIPELINE_STEPS[number], current: PipelineStatus) {
  if (step.statuses.includes(current)) return "active";
  return ALL_STATUSES.indexOf(current) > ALL_STATUSES.indexOf(step.statuses[0]) ? "done" : "pending";
}

// ── Page ───────────────────────────────────────────────────────────────────────

type PageStep = "upload" | "mapping" | "pipeline";

export default function UploadPage() {
  const [pageStep, setPageStep]       = useState<PageStep>("upload");
  const [dragOver, setDragOver]       = useState(false);
  const [inspecting, setInspecting]   = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [file, setFile]               = useState<File | null>(null);
  const [inspect, setInspect]         = useState<ColumnInspectResponse | null>(null);
  const [mapping, setMapping]         = useState<Record<string, string>>({});
  const [job, setJob]                 = useState<PipelineJobResponse | null>(null);
  const fileRef                       = useRef<HTMLInputElement>(null);
  const pollRef                       = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  const startPolling = useCallback(() => {
    stopPolling();
    pollRef.current = window.setInterval(() => {
      api.getPipelineStatus().then((s) => {
        setJob(s);
        if (!RUNNING.includes(s.status)) stopPolling();
      }).catch(() => {});
    }, 2000);
  }, [stopPolling]);

  useEffect(() => {
    api.getPipelineStatus().then(setJob).catch(() => {});
    return stopPolling;
  }, [stopPolling]);

  async function handleFileSelected(f: File) {
    setFile(f);
    setInspecting(true);
    setInspectError(null);
    try {
      const result = await api.inspectColumns(f);
      setInspect(result);
      setMapping({ ...result.suggested_mapping });
      setPageStep("mapping");
    } catch (e: unknown) {
      setInspectError(e instanceof Error ? e.message : "Failed to read file headers.");
    } finally {
      setInspecting(false);
    }
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) handleFileSelected(f);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFileSelected(f);
  }

  async function runPipeline() {
    if (!file) return;
    setPageStep("pipeline");
    try {
      const j = await api.uploadDataset(file, mapping);
      setJob(j);
      if (RUNNING.includes(j.status)) startPolling();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setJob({
        job_id: "error", status: "failed", progress: 0,
        message: "Upload failed.",
        error: msg.includes("Not Found")
          ? "Backend pipeline endpoint not found. Make sure the server is running."
          : msg,
        started_at: null, completed_at: null,
        etl_rows: null, etl_products: null, best_model: null, best_rmse: null,
      });
    }
  }

  function reset() {
    stopPolling();
    setPageStep("upload");
    setFile(null);
    setInspect(null);
    setMapping({});
    setJob(null);
    setInspectError(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="p-8 space-y-8 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Upload Dataset</h1>
        <p className="text-sm text-slate-500 mt-1">
          Upload a CSV or Excel file, map your columns, and the system retrains all models automatically.
        </p>
      </div>

      {/* Step indicators */}
      <div className="flex items-center gap-2 text-sm">
        {(["upload", "mapping", "pipeline"] as PageStep[]).map((s, i) => {
          const labels = ["Upload File", "Map Columns", "Run Pipeline"];
          const active = pageStep === s;
          const done   = ["upload", "mapping", "pipeline"].indexOf(pageStep) > i;
          return (
            <span key={s} className="flex items-center gap-2">
              <span className={`flex items-center gap-1.5 font-medium ${
                active ? "text-indigo-600" : done ? "text-green-600" : "text-slate-400"
              }`}>
                <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                  active ? "bg-indigo-100 text-indigo-600"
                  : done  ? "bg-green-100 text-green-600"
                  : "bg-slate-100 text-slate-400"
                }`}>
                  {done ? <Check size={12} /> : i + 1}
                </span>
                {labels[i]}
              </span>
              {i < 2 && <ArrowRight size={14} className="text-slate-300 shrink-0" />}
            </span>
          );
        })}
      </div>

      {/* ── Step 1: Upload ── */}
      {pageStep === "upload" && (
        <div className="space-y-6">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !inspecting && fileRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-12 flex flex-col items-center gap-4 transition-colors ${
              inspecting
                ? "opacity-60 cursor-wait border-slate-200 bg-slate-50"
                : dragOver
                ? "cursor-pointer border-indigo-400 bg-indigo-50"
                : "cursor-pointer border-slate-200 bg-white hover:border-indigo-300 hover:bg-indigo-50/40"
            }`}
          >
            {inspecting
              ? <Loader2 size={36} className="text-indigo-500 animate-spin" />
              : <Upload size={36} className={dragOver ? "text-indigo-500" : "text-slate-400"} />}
            <div className="text-center">
              <p className="font-semibold text-slate-700">
                {inspecting ? "Reading column headers…" : "Drop your file here or click to browse"}
              </p>
              <p className="text-sm text-slate-400 mt-1">Accepts .csv, .xlsx, .xls — max 100 MB</p>
            </div>
          </div>
          <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={onFileChange} />

          {inspectError && (
            <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-xl p-4">
              <XCircle size={16} className="text-rose-500 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-700 text-sm">Could not read file</p>
                <p className="text-rose-600 text-xs mt-1 font-mono">{inspectError}</p>
              </div>
            </div>
          )}

          {/* Format hint table */}
          <div className="bg-slate-50 rounded-xl border border-slate-200 overflow-hidden text-sm">
            <p className="px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide border-b border-slate-200">
              Example — UCI Online Retail format (column names are case-insensitive)
            </p>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-100">
                    <th className="text-left px-5 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wide">Column</th>
                    <th className="text-left px-5 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wide">Example</th>
                    <th className="text-left px-5 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wide">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["InvoiceNo",   "536365",             "Unique transaction ID"],
                    ["StockCode",   "85123A",             "Product code"],
                    ["Description", "WHITE METAL LANTERN","Product name"],
                    ["Quantity",    "6",                  "Units sold (negative = return)"],
                    ["InvoiceDate", "01/12/2010 08:26",   "Date and time of transaction"],
                    ["UnitPrice",   "2.55",               "Price per unit in GBP"],
                    ["CustomerID",  "17850",              "Unique customer identifier"],
                    ["Country",     "United Kingdom",     "Customer country"],
                  ].map(([col, ex, desc], i) => (
                    <tr key={col} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                      <td className="px-5 py-2 font-mono font-semibold text-indigo-600">{col}</td>
                      <td className="px-5 py-2 font-mono text-slate-500 text-xs">{ex}</td>
                      <td className="px-5 py-2 text-slate-600">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Step 2: Column mapping ── */}
      {pageStep === "mapping" && inspect && (
        <ColumnMapper
          file={file!}
          inspect={inspect}
          mapping={mapping}
          setMapping={setMapping}
          onBack={reset}
          onConfirm={runPipeline}
        />
      )}

      {/* ── Step 3: Pipeline progress ── */}
      {pageStep === "pipeline" && (
        <PipelineProgress job={job} onReset={reset} />
      )}
    </div>
  );
}

// ── Column mapper component ────────────────────────────────────────────────────

function ColumnMapper({
  file, inspect, mapping, setMapping, onBack, onConfirm,
}: {
  file: File;
  inspect: ColumnInspectResponse;
  mapping: Record<string, string>;
  setMapping: (m: Record<string, string>) => void;
  onBack: () => void;
  onConfirm: () => void;
}) {
  const { columns, required_fields } = inspect;

  const mandatoryFields = required_fields.filter((f) => f.required);
  const optionalFields  = required_fields.filter((f) => !f.required);
  const autoCount       = Object.keys(inspect.suggested_mapping).length;

  // Confirm is blocked only when a mandatory field has no selection
  const mandatoryOk = mandatoryFields.every((f) => !!mapping[f.internal_name]);

  function setField(internal: string, userCol: string) {
    const next = { ...mapping };
    if (userCol === "") {
      delete next[internal]; // empty = "not in my data" (optional fields only)
    } else {
      next[internal] = userCol;
    }
    setMapping(next);
  }

  function renderRow(field: RequiredField) {
    const selected = mapping[field.internal_name] ?? "";
    const isAuto   = selected !== "" && selected === inspect.suggested_mapping[field.internal_name];
    const isMapped = selected !== "";
    const isError  = field.required && !isMapped;

    return (
      <div key={field.internal_name} className="grid grid-cols-[1fr_auto_1fr] items-center gap-4 px-5 py-4">
        {/* Field description */}
        <div className="flex items-start gap-2">
          <div>
            <div className="flex items-center gap-2">
              <p className="text-sm font-semibold text-slate-800">{field.label}</p>
              <span className={`text-[10px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded ${
                field.required
                  ? "bg-rose-100 text-rose-600"
                  : "bg-slate-100 text-slate-400"
              }`}>
                {field.required ? "Required" : "Optional"}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">{field.desc}</p>
          </div>
        </div>

        {/* Arrow */}
        <ArrowRight size={14} className="text-slate-300 shrink-0" />

        {/* User column selector */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <select
              value={selected}
              onChange={(e) => setField(field.internal_name, e.target.value)}
              className={`w-full text-sm rounded-lg border px-3 py-2 pr-8 appearance-none bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300 ${
                isError
                  ? "border-rose-300 text-slate-400"
                  : isMapped
                  ? "border-slate-200 text-slate-700"
                  : "border-slate-200 text-slate-400"
              }`}
            >
              {field.required
                ? <option value="">— select a column —</option>
                : <option value="">— not in my data —</option>
              }
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center">
              <svg className="w-3 h-3 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>

          {/* Status indicator */}
          <div className="shrink-0 w-5 h-5 flex items-center justify-center">
            {isError
              ? <AlertCircle size={16} className="text-rose-400" />
              : isMapped
              ? <CheckCircle size={16} className={isAuto ? "text-green-500" : "text-indigo-500"} />
              : <div className="w-4 h-4 rounded-full border-2 border-slate-200" />}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* File info bar */}
      <div className="flex items-center gap-3 bg-indigo-50 border border-indigo-100 rounded-xl px-5 py-3">
        <FileText size={16} className="text-indigo-500 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-indigo-700 truncate">{file.name}</p>
          <p className="text-xs text-indigo-400 mt-0.5">
            {columns.length} column{columns.length !== 1 ? "s" : ""} detected
            {autoCount > 0 && ` · ${autoCount} of ${required_fields.length} auto-matched`}
          </p>
        </div>
      </div>

      {!mandatoryOk && (
        <div className="flex items-start gap-3 bg-amber-50 border border-amber-200 rounded-xl p-4">
          <AlertCircle size={15} className="text-amber-500 shrink-0 mt-0.5" />
          <p className="text-sm text-amber-700">
            Select a column for every <strong>Required</strong> field before running the pipeline.
            Optional fields can be left as "— not in my data —".
          </p>
        </div>
      )}

      {/* Mapping table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-5 py-3 bg-slate-50 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-700">Map your columns to the required fields</p>
          <p className="text-xs text-slate-400 mt-0.5">
            All your file's columns are available in every dropdown. Optional fields can be skipped.
          </p>
        </div>

        {/* Mandatory fields */}
        <div className="divide-y divide-slate-100">
          {mandatoryFields.map(renderRow)}
        </div>

        {/* Optional fields section */}
        {optionalFields.length > 0 && (
          <>
            <div className="px-5 py-2 bg-slate-50 border-y border-slate-100">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Optional fields</p>
            </div>
            <div className="divide-y divide-slate-100">
              {optionalFields.map(renderRow)}
            </div>
          </>
        )}

        {/* Legend */}
        <div className="px-5 py-3 bg-slate-50 border-t border-slate-100 flex flex-wrap items-center gap-5 text-xs text-slate-500">
          <span className="flex items-center gap-1.5"><CheckCircle size={12} className="text-green-500" /> Auto-matched</span>
          <span className="flex items-center gap-1.5"><CheckCircle size={12} className="text-indigo-500" /> Manually selected</span>
          <span className="flex items-center gap-1.5"><AlertCircle size={12} className="text-rose-400" /> Required — must select</span>
          <span className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full border-2 border-slate-200" /> Optional — skipped</span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-700 transition-colors"
        >
          <ArrowLeft size={14} />
          Choose a different file
        </button>

        <button
          onClick={onConfirm}
          disabled={!mandatoryOk}
          className={`flex items-center gap-2 text-sm font-semibold px-5 py-2.5 rounded-xl transition-colors ${
            mandatoryOk
              ? "bg-indigo-600 text-white hover:bg-indigo-700"
              : "bg-slate-100 text-slate-400 cursor-not-allowed"
          }`}
        >
          Confirm & Run Pipeline
          <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}

// ── Pipeline progress component ────────────────────────────────────────────────

function PipelineProgress({
  job, onReset,
}: {
  job: PipelineJobResponse | null;
  onReset: () => void;
}) {
  if (!job || job.status === "idle") {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 size={24} className="text-indigo-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-sm mb-2">
          <span className="font-semibold text-slate-700">
            {job.status === "completed" ? "Pipeline Complete"
             : job.status === "failed"   ? "Pipeline Failed"
             : "Running Pipeline…"}
          </span>
          <span className="text-slate-400">{job.progress}%</span>
        </div>
        <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              job.status === "failed"    ? "bg-rose-500"
              : job.status === "completed" ? "bg-green-500"
              : "bg-indigo-500"
            }`}
            style={{ width: `${job.progress}%` }}
          />
        </div>
      </div>

      <p className="text-sm text-slate-600">{job.message}</p>

      {/* Step indicators */}
      <div className="space-y-3">
        {PIPELINE_STEPS.map((step) => {
          const state = stepState(step, job.status);
          return (
            <div key={step.label} className="flex items-start gap-3">
              <div className="mt-0.5 shrink-0">
                {state === "done"
                  ? <CheckCircle size={18} className="text-green-500" />
                  : state === "active"
                  ? <Loader2 size={18} className="text-indigo-500 animate-spin" />
                  : <div className="w-[18px] h-[18px] rounded-full border-2 border-slate-200" />}
              </div>
              <div>
                <p className={`text-sm font-semibold ${
                  state === "done" ? "text-green-700"
                  : state === "active" ? "text-indigo-700"
                  : "text-slate-400"
                }`}>{step.label}</p>
                <p className="text-xs text-slate-400">{step.desc}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Completion results */}
      {job.status === "completed" && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-slate-100">
          {[
            { label: "Products",     value: job.etl_products?.toLocaleString() ?? "—" },
            { label: "Feature Rows", value: job.etl_rows?.toLocaleString() ?? "—" },
            { label: "Best Model",   value: job.best_model ?? "—" },
            { label: "Best RMSE",    value: job.best_rmse != null ? job.best_rmse.toFixed(4) : "—" },
          ].map(({ label, value }) => (
            <div key={label}>
              <p className="text-xs text-slate-400 uppercase tracking-wide">{label}</p>
              <p className="font-bold text-slate-800 text-sm mt-0.5">{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {job.status === "failed" && job.error && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-xl p-4">
          <XCircle size={16} className="text-rose-500 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-rose-700 text-sm">Error</p>
            <p className="text-rose-600 mt-1 text-xs font-mono break-all">{job.error}</p>
          </div>
        </div>
      )}

      {(job.status === "completed" || job.status === "failed") && (
        <button
          onClick={onReset}
          className="flex items-center gap-2 text-sm text-slate-500 hover:text-indigo-600 transition-colors"
        >
          <RefreshCw size={14} />
          Upload another dataset
        </button>
      )}
    </div>
  );
}
