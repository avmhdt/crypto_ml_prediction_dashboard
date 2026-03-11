"use client";

import type { Metrics } from "@/lib/types";

interface MetricsPanelProps {
  metrics: Metrics | null;
}

function MetricCard({
  label,
  value,
  color,
  icon,
}: {
  label: string;
  value: string;
  color: string;
  icon: string;
}) {
  return (
    <div className="glow-card px-4 py-3">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
            {label}
          </p>
          <p className={`num mt-1 text-xl font-bold ${color}`}>{value}</p>
        </div>
        <span className="text-lg opacity-50">{icon}</span>
      </div>
    </div>
  );
}

function LoadingCard() {
  return (
    <div className="glow-card px-4 py-3">
      <div className="shimmer mb-2 h-3 w-16 rounded" />
      <div className="shimmer h-6 w-12 rounded" />
    </div>
  );
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  if (!metrics) {
    return (
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <LoadingCard key={i} />
        ))}
      </div>
    );
  }

  const cards = [
    {
      label: "Total Signals",
      value: metrics.total_signals.toLocaleString(),
      color: "text-white",
      icon: "#",
    },
    {
      label: "Long Signals",
      value: metrics.long_signals.toLocaleString(),
      color: "text-green-400",
      icon: "\u2191",
    },
    {
      label: "Short Signals",
      value: metrics.short_signals.toLocaleString(),
      color: "text-red-400",
      icon: "\u2193",
    },
    {
      label: "Meta Probability",
      value: (metrics.avg_meta_prob * 100).toFixed(1) + "%",
      color: "text-blue-400",
      icon: "\u25ce",
    },
    {
      label: "Avg Bet Size",
      value: metrics.avg_bet_size.toFixed(3),
      color: "text-yellow-400",
      icon: "\u25c8",
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {cards.map((card) => (
        <MetricCard key={card.label} {...card} />
      ))}
    </div>
  );
}
