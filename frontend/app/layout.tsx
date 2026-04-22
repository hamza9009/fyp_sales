import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "RetailSense — Predictive Analytics",
  description: "E-commerce demand forecasting & inventory intelligence SaaS",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full flex">
        <Sidebar />
        <main className="flex-1 overflow-y-auto bg-slate-50">
          {children}
        </main>
      </body>
    </html>
  );
}
