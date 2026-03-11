"use client";

import type { Metrics } from "@/lib/types";

interface MetricsPanelProps {
  metrics: Metrics | null;
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  if (!metrics) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
        <h3 className="text-sm font-medium text-zinc-300">Model Metrics</h3>
        <p className="mt-2 text-xs text-zinc-500">Loading...</p>
      </div>
    );
  }

  const stats = [
    {
      label: "Total Signals",
      value: metrics.total_signals.toString(),
      color: "text-white",
    },
    {
      label: "Long",
      value: metrics.long_signals.toString(),
      color: "text-green-400",
    },
    {
      label: "Short",
      value: metrics.short_signals.toString(),
      color: "text-red-400",
    },
    {
      label: "Avg Meta Prob",
      value: (metrics.avg_meta_prob * 100).toFixed(1) + "%",
      color: "text-blue-400",
    },
    {
      label: "Avg Bet Size",
      value: metrics.avg_bet_size.toFixed(2),
      color: "text-yellow-400",
    },
  ];

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
      <h3 className="mb-3 text-sm font-medium text-zinc-300">Model Metrics</h3>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {stats.map((stat) => (
          <div key={stat.label}>
            <p className="text-xs text-zinc-500">{stat.label}</p>
            <p className={`text-lg font-semibold ${stat.color}`}>
              {stat.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
