"use client";

import { useState } from "react";
import type { Signal } from "@/lib/types";

interface SignalsTableProps {
  signals: Signal[];
}

type SortKey = keyof Signal;

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
    return new Date(ts).toLocaleTimeString();
  };

  const columns: { key: SortKey; label: string; format?: (v: unknown) => string }[] = [
    { key: "timestamp", label: "Time", format: (v) => formatTime(v as number) },
    {
      key: "side",
      label: "Side",
      format: (v) => ((v as number) === 1 ? "LONG" : "SHORT"),
    },
    { key: "size", label: "Size", format: (v) => (v as number).toFixed(2) },
    {
      key: "entry_price",
      label: "Entry",
      format: (v) => (v as number).toFixed(2),
    },
    {
      key: "sl_price",
      label: "SL",
      format: (v) => (v != null ? (v as number).toFixed(2) : "—"),
    },
    {
      key: "pt_price",
      label: "PT",
      format: (v) => (v != null ? (v as number).toFixed(2) : "—"),
    },
    {
      key: "meta_probability",
      label: "Meta P",
      format: (v) => ((v as number) * 100).toFixed(1) + "%",
    },
  ];

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950">
      <div className="border-b border-zinc-800 px-4 py-2">
        <h3 className="text-sm font-medium text-zinc-300">
          Signals ({signals.length})
        </h3>
      </div>
      <div className="max-h-80 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-zinc-900">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="cursor-pointer px-3 py-2 text-left text-xs font-medium text-zinc-400 hover:text-white"
                >
                  {col.label}
                  {sortKey === col.key && (sortAsc ? " ▲" : " ▼")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-8 text-center text-zinc-500"
                >
                  No signals yet — awaiting data
                </td>
              </tr>
            ) : (
              sorted.map((signal) => (
                <tr
                  key={signal.id}
                  className="border-t border-zinc-800/50 hover:bg-zinc-900/50"
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={`px-3 py-1.5 ${
                        col.key === "side"
                          ? signal.side === 1
                            ? "text-green-400"
                            : "text-red-400"
                          : "text-zinc-300"
                      }`}
                    >
                      {col.format
                        ? col.format(signal[col.key])
                        : String(signal[col.key] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
