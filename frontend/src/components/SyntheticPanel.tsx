"use client";

import { useState, useEffect, useRef } from "react";
import { createChart, LineSeries, type IChartApi } from "lightweight-charts";
import type { SynthRunData, SynthPoint } from "@/lib/types";

interface SyntheticPanelProps {
  barType: string;
  labeling: string;
}

// ─── Signal Recovery Curve (hero chart) ──────────────────────────
function RecoveryCurve({ data }: { data: SynthRunData }) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartApi = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartRef.current || !data.points.length) return;

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 350,
      layout: { background: { color: "transparent" }, textColor: "#71717a" },
      grid: { vertLines: { color: "#27272a40" }, horzLines: { color: "#27272a40" } },
      rightPriceScale: {
        borderColor: "#27272a",
        scaleMargins: { top: 0.05, bottom: 0.05 },
      },
      crosshair: { mode: 0 },
      // Use a simple numeric scale for x-axis (Sharpe ratio)
      timeScale: { borderColor: "#27272a", visible: false },
    });
    chartApi.current = chart;

    // Accuracy line
    const accSeries = chart.addSeries(LineSeries, {
      color: "#22c55e",
      lineWidth: 3,
      pointMarkersVisible: true,
      pointMarkersRadius: 5,
    });

    // Sort points by sharpe
    const sorted = [...data.points].sort((a, b) => a.sharpe - b.sharpe);

    // Map sharpe to a pseudo-time for lightweight-charts (it requires time-based x-axis)
    // Use sharpe * 86400 as "days" offset from a base date
    const baseTime = 1704067200; // 2024-01-01 UTC
    const accData = sorted.map((pt) => ({
      time: (baseTime + pt.sharpe * 86400) as any,
      value: pt.oos_accuracy * 100,
    }));
    accSeries.setData(accData);

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data]);

  // Render chart + custom x-axis labels (Sharpe values)
  const sorted = [...data.points].sort((a, b) => a.sharpe - b.sharpe);

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Signal Recovery Curve — OOS Accuracy vs. Injected Sharpe Ratio
        </h3>
        <span className="text-[10px] text-zinc-600">
          Higher Sharpe = stronger planted signal
        </span>
      </div>
      <div ref={chartRef} />
      {/* Custom x-axis labels */}
      <div className="flex justify-between px-2 mt-1">
        {sorted.map((pt) => (
          <div key={pt.sharpe} className="text-center">
            <div className="num text-[10px] text-zinc-500">{pt.sharpe.toFixed(1)}</div>
          </div>
        ))}
      </div>
      <div className="text-center mt-0.5 text-[9px] text-zinc-600">
        Injected Sharpe Ratio (annualized)
      </div>
      {/* 50% baseline annotation */}
      <div className="mt-2 flex items-center gap-2 text-[10px] text-zinc-600">
        <span className="inline-block h-px w-4 border-t border-dashed border-zinc-500" />
        50% = coin flip (no signal)
      </div>
    </div>
  );
}

// ─── Detection Threshold Card ────────────────────────────────────
function DetectionThreshold({ threshold }: { threshold: number }) {
  const color = threshold <= 1.0
    ? "text-green-400"
    : threshold <= 2.0
    ? "text-yellow-400"
    : "text-red-400";

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 text-center">
      <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500 mb-2">
        Detection Threshold
      </div>
      <div className={`num text-4xl font-bold ${color}`}>
        {threshold === Infinity ? "∞" : threshold.toFixed(1)}
      </div>
      <div className="text-[11px] text-zinc-500 mt-1">
        Sharpe Ratio
      </div>
      <div className="text-[10px] text-zinc-600 mt-2 max-w-xs mx-auto">
        Minimum signal strength for the pipeline to achieve &gt;55% OOS accuracy.
        {threshold <= 2.0 && " Crypto markets typically operate below this threshold."}
      </div>
    </div>
  );
}

// ─── Metrics at Key SNR Levels ───────────────────────────────────
function SNRMetricsGrid({ points }: { points: SynthPoint[] }) {
  const keyLevels = [0.0, 1.0, 2.0, 5.0];
  const labels: Record<number, string> = {
    0.0: "Pure Noise",
    1.0: "Weak Signal",
    2.0: "Moderate Signal",
    5.0: "Strong Signal",
  };
  const colors: Record<number, string> = {
    0.0: "text-zinc-400",
    1.0: "text-yellow-400",
    2.0: "text-blue-400",
    5.0: "text-green-400",
  };

  const pointMap = new Map(points.map((p) => [p.sharpe, p]));

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {keyLevels.map((level) => {
        const pt = pointMap.get(level);
        if (!pt) return null;
        return (
          <div key={level} className="glow-card rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
            <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
              {labels[level]}
            </div>
            <div className="text-[9px] text-zinc-600 mb-1">Sharpe {level.toFixed(1)}</div>
            <div className={`num text-xl font-bold ${colors[level]}`}>
              {(pt.oos_accuracy * 100).toFixed(1)}%
            </div>
            <div className="mt-1 space-y-0.5 text-[10px] text-zinc-500">
              <div>Sharpe: {pt.equity_sharpe.toFixed(2)}</div>
              <div>Trades: {pt.num_trades}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Methodology Explanation ─────────────────────────────────────
function Methodology({ barType, labeling }: { barType: string; labeling: string }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        Methodology
      </h3>
      <p className="text-xs text-zinc-500 leading-relaxed">
        Synthetic tick data is generated using Geometric Brownian Motion with regime-switching
        drift. The Sharpe ratio controls the signal-to-noise ratio — at Sharpe 0 the price is
        a pure random walk; at higher values a planted directional signal emerges. Order flow
        (<code className="text-blue-400">is_buyer_maker</code>) correlates with the regime
        direction, activating microstructural features. At each SNR level, the full AFML
        pipeline runs: ticks → <code className="text-blue-400">{barType.replace(/_/g, " ")}</code> bars
        → 76 features → LightGBM → meta-labeling → equity simulation. The recovery curve
        above shows the pipeline&apos;s detection power as a function of signal strength.
      </p>
      <p className="mt-2 text-xs text-zinc-600 leading-relaxed">
        This validates the pipeline itself — proving it extracts signal when present and
        correctly reports no signal when absent. The crypto market result (log_loss ≈ 0.693,
        ~50% accuracy) is consistent with a market whose SNR falls below this pipeline&apos;s
        detection threshold.
      </p>
    </div>
  );
}

// ─── Main Panel ──────────────────────────────────────────────────
export function SyntheticPanel({ barType, labeling }: SyntheticPanelProps) {
  const [data, setData] = useState<SynthRunData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setData(null);

    const controller = new AbortController();
    fetch(
      `/api/synth-latest?bar_type=${barType}&labeling=${labeling}`,
      { signal: controller.signal },
    )
      .then((r) => {
        if (r.status === 404) throw new Error("no-data");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") {
          setError(e.message === "no-data" ? "no-data" : e.message);
        }
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [barType, labeling]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="text-sm text-zinc-500">Loading synthetic validation results...</div>
      </div>
    );
  }

  if (error === "no-data" || !data) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] py-16 px-8">
        <div className="text-sm font-medium text-zinc-400 mb-2">No Synthetic Validation Results</div>
        <div className="text-xs text-zinc-600 text-center max-w-md">
          Run the synthetic validation script to generate results:
        </div>
        <code className="mt-3 rounded bg-zinc-900 px-3 py-2 text-[11px] text-blue-400">
          python scripts/synthetic_validation.py --bar-type {barType} --labeling {labeling}
        </code>
        <div className="mt-2 text-[10px] text-zinc-600">
          Or seed demo data: <code className="text-blue-400">python scripts/seed_synthetic.py</code>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-900/50 bg-red-950/20 p-6 text-center">
        <div className="text-sm text-red-400">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">
            Signal Recovery Analysis
          </h2>
          <p className="text-[10px] text-zinc-500">
            {data.points.length} SNR levels &middot; {barType.replace(/_/g, " ")} bars
            &middot; {labeling.replace(/_/g, " ")}
            &middot; {new Date(data.created_at).toLocaleDateString()}
          </p>
        </div>
        <div className="rounded-md border border-zinc-700/50 bg-zinc-800/50 px-2.5 py-1">
          <span className="text-[10px] text-zinc-500">Run </span>
          <span className="num text-xs font-semibold text-white">#{data.id}</span>
        </div>
      </div>

      {/* Signal Recovery Curve (hero) */}
      <RecoveryCurve data={data} />

      {/* Detection threshold */}
      <DetectionThreshold threshold={data.detection_threshold} />

      {/* Metrics at key SNR levels */}
      <SNRMetricsGrid points={data.points} />

      {/* Methodology */}
      <Methodology barType={barType} labeling={labeling} />
    </div>
  );
}
