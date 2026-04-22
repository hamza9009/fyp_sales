"use client";

/**
 * ProductSearch — autocomplete input for stock code lookup.
 *
 * Debounces the user's keystrokes, calls GET /api/v1/products?q=...,
 * and renders a dropdown of matching products.  Selecting a result
 * calls onSelect with the chosen stock_code.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { Search } from "lucide-react";
import { api, ProductResult } from "@/lib/api";

interface ProductSearchProps {
  value: string;
  onChange: (value: string) => void;
  onSelect: (stockCode: string) => void;
  placeholder?: string;
}

const DEBOUNCE_MS = 300;
const MIN_CHARS = 2;

export default function ProductSearch({
  value,
  onChange,
  onSelect,
  placeholder = "e.g. 85123A or WHITE HANGING",
}: ProductSearchProps) {
  const [suggestions, setSuggestions] = useState<ProductResult[]>([]);
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.trim().length < MIN_CHARS) {
      setSuggestions([]);
      setOpen(false);
      return;
    }
    try {
      const res = await api.searchProducts(q.trim(), 10);
      setSuggestions(res.results);
      setOpen(res.results.length > 0);
      setActiveIdx(-1);
    } catch (_e) {
      setSuggestions([]);
      setOpen(false);
    }
  }, []);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => fetchSuggestions(value), DEBOUNCE_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [value, fetchSuggestions]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && activeIdx >= 0) {
      e.preventDefault();
      const chosen = suggestions[activeIdx];
      onChange(chosen.stock_code);
      onSelect(chosen.stock_code);
      setOpen(false);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  function pickSuggestion(s: ProductResult) {
    onChange(s.stock_code);
    onSelect(s.stock_code);
    setOpen(false);
  }

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div className="relative">
        <Search
          size={15}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
        />
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value.toUpperCase())}
          onKeyDown={handleKeyDown}
          onFocus={() => suggestions.length > 0 && setOpen(true)}
          placeholder={placeholder}
          className="w-full border border-slate-200 rounded-lg pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
          autoComplete="off"
        />
      </div>

      {open && (
        <ul className="absolute z-50 mt-1 w-full bg-white border border-slate-200 rounded-xl shadow-lg max-h-64 overflow-y-auto">
          {suggestions.map((s, i) => (
            <li
              key={s.stock_code}
              onMouseDown={() => pickSuggestion(s)}
              className={`flex items-start gap-3 px-4 py-2.5 cursor-pointer text-sm transition-colors ${
                i === activeIdx ? "bg-indigo-50" : "hover:bg-slate-50"
              }`}
            >
              <span className="font-mono font-semibold text-indigo-600 shrink-0 w-20">
                {s.stock_code}
              </span>
              <span className="text-slate-600 truncate">{s.description ?? "—"}</span>
              {s.unit_price != null && (
                <span className="ml-auto shrink-0 text-slate-400 text-xs">
                  £{s.unit_price.toFixed(2)}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
