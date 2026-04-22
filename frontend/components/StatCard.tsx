interface StatCardProps {
  title: string;
  value: string;
  sub?: string;
  accent?: "indigo" | "green" | "amber" | "rose";
}

const ACCENT_MAP = {
  indigo: "bg-indigo-50 text-indigo-600",
  green:  "bg-green-50  text-green-600",
  amber:  "bg-amber-50  text-amber-600",
  rose:   "bg-rose-50   text-rose-600",
};

export default function StatCard({ title, value, sub, accent = "indigo" }: StatCardProps) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 flex flex-col gap-1 shadow-sm">
      <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
        {title}
      </span>
      <span className={`text-2xl font-bold ${ACCENT_MAP[accent].split(" ")[1]}`}>
        {value}
      </span>
      {sub && <span className="text-xs text-slate-400">{sub}</span>}
    </div>
  );
}
