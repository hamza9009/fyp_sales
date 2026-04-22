import { AlertCircle } from "lucide-react";

export default function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-3 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl px-5 py-4 text-sm">
      <AlertCircle size={18} className="shrink-0" />
      <span>{message}</span>
    </div>
  );
}
