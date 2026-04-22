"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  TrendingUp,
  Package,
  BarChart3,
  Zap,
  Upload,
  History,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/forecast",  label: "Forecast",  icon: TrendingUp },
  { href: "/inventory", label: "Inventory",  icon: Package },
  { href: "/history",   label: "History",    icon: History },
  { href: "/models",    label: "Models",     icon: BarChart3 },
  { href: "/upload",    label: "Upload Data", icon: Upload },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      style={{ width: "var(--sidebar-width)", minWidth: "var(--sidebar-width)" }}
      className="h-screen sticky top-0 flex flex-col bg-slate-900 text-slate-100 shrink-0"
    >
      {/* Logo */}
      <div className="flex items-center gap-2 px-5 py-5 border-b border-slate-700">
        <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center">
          <Zap size={16} className="text-white" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-bold text-white">RetailSense</div>
          <div className="text-xs text-slate-400">Analytics SaaS</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + "/");
          const isDivider = href === "/upload";
          return (
            <div key={href}>
              {isDivider && <div className="my-3 border-t border-slate-700" />}
              <Link
                href={href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  active
                    ? "bg-indigo-600 text-white"
                    : "text-slate-400 hover:bg-slate-800 hover:text-white"
                }`}
              >
                <Icon size={18} />
                {label}
              </Link>
            </div>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-slate-700 text-xs text-slate-500">
        MSc FYP · Phase 5
      </div>
    </aside>
  );
}
