"use client";

import { useState, useEffect, useCallback } from "react";
import { Header } from "@/components/Header";
import { Controls } from "@/components/Controls";
import { Chart } from "@/components/Chart";
import { SignalsTable } from "@/components/SignalsTable";
import { MetricsPanel } from "@/components/MetricsPanel";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { BarData, Signal, Metrics, DashboardConfig, WSMessage } from "@/lib/types";

const DEFAULT_CONFIG: DashboardConfig = {
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"],
  bar_types: [
    "time", "tick", "volume", "dollar",
    "tick_imbalance", "volume_imbalance", "dollar_imbalance",
    "tick_run", "volume_run", "dollar_run",
  ],
  labeling_methods: ["triple_barrier", "trend_scanning", "directional_change"],
};

export default function DashboardPage() {
  const [config, setConfig] = useState<DashboardConfig>(DEFAULT_CONFIG);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [barType, setBarType] = useState("time");
  const [labeling, setLabeling] = useState("triple_barrier");
  const [bars, setBars] = useState<BarData[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  // Fetch config on mount
  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then(setConfig)
      .catch(() => {});
  }, []);

  // Fetch bars when selection changes
  useEffect(() => {
    fetch(`/api/bars/${symbol}/${barType}?limit=500`)
      .then((r) => r.json())
      .then((data) => setBars(Array.isArray(data) ? data : []))
      .catch(() => setBars([]));
  }, [symbol, barType]);

  // Fetch signals when selection changes
  useEffect(() => {
    fetch(`/api/signals/${symbol}?bar_type=${barType}&labeling=${labeling}&limit=100`)
      .then((r) => r.json())
      .then((data) => setSignals(Array.isArray(data) ? data : []))
      .catch(() => setSignals([]));
  }, [symbol, barType, labeling]);

  // Fetch metrics when selection changes
  useEffect(() => {
    fetch(`/api/metrics/${symbol}?bar_type=${barType}&labeling=${labeling}`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, [symbol, barType, labeling]);

  // Handle WebSocket messages
  const onMessage = useCallback(
    (msg: WSMessage) => {
      if (msg.type === "bar" && msg.data.bar_type === barType) {
        setBars((prev) => {
          const updated = [...prev, msg.data as BarData];
          return updated.slice(-500);
        });
      }
      if (msg.type === "signal") {
        const signal = msg.data as Signal;
        if (signal.bar_type === barType && signal.labeling_method === labeling) {
          setSignals((prev) => [signal, ...prev].slice(0, 100));
        }
      }
    },
    [barType, labeling],
  );

  const { connected } = useWebSocket({
    symbol,
    onMessage,
  });

  return (
    <div className="flex min-h-screen flex-col bg-[var(--background)] text-[var(--foreground)]">
      <Header connected={connected} symbol={symbol} />
      <Controls
        symbols={config.symbols}
        barTypes={config.bar_types}
        labelingMethods={config.labeling_methods}
        selectedSymbol={symbol}
        selectedBarType={barType}
        selectedLabeling={labeling}
        onSymbolChange={setSymbol}
        onBarTypeChange={setBarType}
        onLabelingChange={setLabeling}
      />
      <main className="flex-1 space-y-4 p-4 lg:p-6">
        {/* Metrics row */}
        <MetricsPanel metrics={metrics} />

        {/* Chart */}
        <Chart bars={bars} signals={signals} />

        {/* Signals table */}
        <SignalsTable signals={signals} />

        {/* Footer bar */}
        <div className="flex items-center justify-between rounded-lg border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-[10px] text-zinc-600">
          <span>
            AFML Pipeline &middot; LightGBM + Purged CV + Meta-Labeling + Bet Sizing
          </span>
          <span className="num">
            {barType.replace(/_/g, " ")} &middot; {labeling.replace(/_/g, " ")}
          </span>
        </div>
      </main>
    </div>
  );
}
