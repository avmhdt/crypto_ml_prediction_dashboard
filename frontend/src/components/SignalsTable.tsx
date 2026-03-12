"use client";

import { useState } from "react";
import type { Signal } from "@/lib/types";

interface SignalsTableProps {
  signals: Signal[];
}

type SortKey = keyof Signal;

function SideBadge({ side }: { side: -1 | 1 }) {
  const isLong = side === 1;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
        isLong
          ? "bg-green-500/10 text-green-400"
          : "bg-red-500/10 text-red-400"
      }`}
    >
      <span className="text-[9px]">{isLong ? "\u25b2" : "\u25bc"}</span>
      {isLong ? "Long" : "Short"}
    </span>
  );
}

function MetaProbBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 70
      ? "bg-green-500"
      : pct >= 50
        ? "bg-yellow-500"
        : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="num text-xs text-zinc-400">{pct}%</span>
    </div>
  );
}

export function SignalsTable({ signals }: SignalsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("timestamp");
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = [...signals].sort((a, b) => {
    const aVal = a[sortKey] ?? 0;
    const bVal = b[sortKey] ?? 0;
    if (aVal < bVal) return sortAsc ? -1 : 1;
    if (aVal > bVal) return sortAsc ? 1 : -1;
    return 0;
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const formatPrice = (v: number | null) =>
    v != null
      ? v.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })
      : "\u2014";

  const columns: {
    key: SortKey;
    label: string;
    align?: string;
  }[] = [
    { key: "timestamp", label: "Time" },
    { key: "side", label: "Side" },
    { key: "size", label: "Size", align: "text-right" },
    { key: "entry_price", label: "Entry", align: "text-right" },
    { key: "sl_price", label: "Stop Loss", align: "text-right" },
    { key: "pt_price", label: "Take Profit", align: "text-right" },
    { key: "meta_probability", label: "Meta P" },
  ];

  return (
    <div className="glow-card">
      <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-2.5">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-medium text-zinc-400">
            Signal History
          </h3>
          <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-500">
            {signals.length}
          </span>
        </div>
        {signals.length > 0 && (
          <span className="text-[10px] text-zinc-600">
            Click headers to sort
          </span>
        )}
      </div>
      <div className="max-h-[360px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-[#111118]">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={`cursor-pointer px-3 py-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500 transition-colors hover:text-zinc-300 ${col.align ?? "text-left"}`}
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span className="ml-1 text-blue-400">
                      {sortAsc ? "\u25b2" : "\u25bc"}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-12 text-center"
                >
                  <div className="flex flex-col items-center gap-2">
                    <span className="text-2xl opacity-20">&lt;/&gt;</span>
                    <span className="text-xs text-zinc-500">
                      No signals yet &mdash; awaiting model predictions
                    </span>
                  </div>
                </td>
              </tr>
            ) : (
              sorted.map((signal, idx) => (
                <tr
                  key={signal.id}
                  className={`border-t border-zinc-800/30 transition-colors hover:bg-zinc-800/20 ${idx === 0 && sortKey === "timestamp" && !sortAsc ? "animate-fade-in" : ""}`}
                >
                  <td className="num px-3 py-2 text-xs text-zinc-400">
                    {formatTime(signal.timestamp)}
                  </td>
                  <td className="px-3 py-2">
                    <SideBadge side={signal.side} />
                  </td>
                  <td className="num px-3 py-2 text-right text-xs text-zinc-300">
                    {signal.size.toFixed(3)}
                  </td>
                  <td className="num px-3 py-2 text-right text-xs text-zinc-300">
                    {formatPrice(signal.entry_price)}
                  </td>
                  <td className="num px-3 py-2 text-right text-xs text-red-400/70">
                    {formatPrice(signal.sl_price)}
                  </td>
                  <td className="num px-3 py-2 text-right text-xs text-green-400/70">
                    {formatPrice(signal.pt_price)}
                  </td>
                  <td className="px-3 py-2">
                    <MetaProbBar value={signal.meta_probability} />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
