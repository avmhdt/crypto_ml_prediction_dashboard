"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { createChart, AreaSeries, type IChartApi, type ISeriesApi } from "lightweight-charts";
import type { WFRunData, WFWindow, WFAggregateMetric } from "@/lib/types";

interface WalkForwardPanelProps {
  symbol: string;
  barType: string;
  labeling: string;
}

// ─── Helper: format ms timestamp to short date ───────────────────
function fmtDate(ms: number): string {
  return new Date(ms).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "2-digit",
  });
}

function fmtPct(v: number): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

// ─── Overfitting Gap Card ────────────────────────────────────────
function WFOverfittingGap({ data }: { data: WFRunData }) {
  const gap = data.overfitting_gap;
  const gapColor = gap > 0.1 ? "text-red-400" : gap > 0.05 ? "text-yellow-400" : "text-green-400";

  return (
    <div className="grid grid-cols-3 gap-3">
      {/* In-Sample */}
      <div className="glow-card rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
        <div className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
          In-Sample (Train)
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <span className="text-xs text-zinc-400">Avg Recall</span>
            <span className="num text-sm font-semibold text-white">
              {(data.avg_insample_recall * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Gap */}
      <div className="flex flex-col items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
          Overfitting Gap
        </div>
        <span className={`num text-2xl font-bold ${gapColor}`}>
          {(gap * 100).toFixed(1)}%
        </span>
        <span className="text-[10px] text-zinc-500 mt-1">in-sample − OOS</span>
      </div>

      {/* Out-of-Sample */}
      <div className="glow-card rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
        <div className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
          Out-of-Sample (Test)
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <span className="text-xs text-zinc-400">Avg Accuracy</span>
            <span className="num text-sm font-semibold text-white">
              {(data.avg_oos_accuracy * 100).toFixed(1)}%
            </span>
          </div>
          {data.aggregate?.["sharpe"] && (
            <div className="flex justify-between">
              <span className="text-xs text-zinc-400">Avg Sharpe</span>
              <span className="num text-sm font-semibold text-white">
                {data.aggregate["sharpe"].mean.toFixed(2)}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Stitched OOS Equity Curve ───────────────────────────────────
function WFEquityCurve({ data }: { data: WFRunData }) {
  const chartRef = useRef<HTMLDivElement>(null);
  const ddRef = useRef<HTMLDivElement>(null);
  const chartApi = useRef<IChartApi | null>(null);
  const ddChartApi = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartRef.current || !data.stitched_timestamps.length) return;

    // Equity chart
    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 300,
      layout: { background: { color: "transparent" }, textColor: "#71717a" },
      grid: { vertLines: { color: "#27272a40" }, horzLines: { color: "#27272a40" } },
      timeScale: { timeVisible: true, borderColor: "#27272a" },
      rightPriceScale: { borderColor: "#27272a" },
      crosshair: { mode: 0 },
    });
    chartApi.current = chart;

    const eqSeries = chart.addSeries(AreaSeries, {
      lineColor: "#22c55e",
      topColor: "rgba(34,197,94,0.15)",
      bottomColor: "rgba(34,197,94,0.01)",
      lineWidth: 2,
    });

    const eqData = data.stitched_timestamps.map((ts, i) => ({
      time: Math.floor(ts / 1000) as any,
      value: data.stitched_equity[i],
    }));
    eqSeries.setData(eqData);
    chart.timeScale().fitContent();

    // Drawdown chart
    let ddChart: IChartApi | null = null;
    if (ddRef.current && data.stitched_drawdown.length) {
      ddChart = createChart(ddRef.current, {
        width: ddRef.current.clientWidth,
        height: 120,
        layout: { background: { color: "transparent" }, textColor: "#71717a" },
        grid: { vertLines: { color: "#27272a40" }, horzLines: { color: "#27272a40" } },
        timeScale: { timeVisible: true, borderColor: "#27272a" },
        rightPriceScale: { borderColor: "#27272a" },
        crosshair: { mode: 0 },
      });
      ddChartApi.current = ddChart;

      const ddSeries = ddChart.addSeries(AreaSeries, {
        lineColor: "#ef4444",
        topColor: "rgba(239,68,68,0.01)",
        bottomColor: "rgba(239,68,68,0.15)",
        lineWidth: 1,
      });

      const ddData = data.stitched_timestamps.map((ts, i) => ({
        time: Math.floor(ts / 1000) as any,
        value: data.stitched_drawdown[i] * 100,
      }));
      ddSeries.setData(ddData);
      ddChart.timeScale().fitContent();

      // Sync time scales
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) ddChart?.timeScale().setVisibleLogicalRange(range);
      });
    }

    const handleResize = () => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth });
      if (ddRef.current && ddChart) ddChart.applyOptions({ width: ddRef.current.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      ddChart?.remove();
    };
  }, [data]);

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Walk-Forward Equity (OOS Only)
        </h3>
        {data.stitched_equity.length > 0 && (
          <span className="num text-sm font-semibold text-white">
            ${data.stitched_equity[data.stitched_equity.length - 1]?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        )}
      </div>
      <div ref={chartRef} />
      <div className="mt-1 text-[10px] text-zinc-600">Underwater Drawdown (%)</div>
      <div ref={ddRef} />
    </div>
  );
}

// ─── Window Timeline ─────────────────────────────────────────────
function WFWindowTimeline({ data }: { data: WFRunData }) {
  if (!data.windows.length) return null;

  const minTs = data.windows[0].train_start;
  const maxTs = data.windows[data.windows.length - 1].test_end;
  const range = maxTs - minTs || 1;

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        Window Timeline ({data.num_windows} windows: {data.train_days}d train / {data.test_days}d test)
      </h3>
      <div className="relative h-8 rounded bg-zinc-900">
        {data.windows.map((w) => {
          const trainLeft = ((w.train_start - minTs) / range) * 100;
          const trainWidth = ((w.train_end - w.train_start) / range) * 100;
          const testLeft = ((w.test_start - minTs) / range) * 100;
          const testWidth = ((w.test_end - w.test_start) / range) * 100;
          return (
            <div key={w.window_index}>
              <div
                className="absolute top-0 h-full bg-blue-900/40 border-r border-blue-700/30"
                style={{ left: `${trainLeft}%`, width: `${trainWidth}%` }}
                title={`Train: ${fmtDate(w.train_start)} – ${fmtDate(w.train_end)}`}
              />
              <div
                className="absolute top-0 h-full bg-green-600/50 border-r border-green-400/30"
                style={{ left: `${testLeft}%`, width: `${testWidth}%` }}
                title={`Test: ${fmtDate(w.test_start)} – ${fmtDate(w.test_end)} | Return: ${fmtPct(w.total_return)}`}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-1.5 flex justify-between text-[9px] text-zinc-600">
        <span>{fmtDate(minTs)}</span>
        <div className="flex gap-4">
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-blue-900/60" /> Train</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-green-600/60" /> Test (OOS)</span>
        </div>
        <span>{fmtDate(maxTs)}</span>
      </div>
    </div>
  );
}

// ─── Per-Window Metrics Table ────────────────────────────────────
function WFWindowTable({ windows }: { windows: WFWindow[] }) {
  const [sortKey, setSortKey] = useState<keyof WFWindow>("window_index");
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    const s = [...windows].sort((a, b) => {
      const av = a[sortKey] as number;
      const bv = b[sortKey] as number;
      return sortAsc ? av - bv : bv - av;
    });
    return s;
  }, [windows, sortKey, sortAsc]);

  const toggleSort = (key: keyof WFWindow) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  };

  const th = (label: string, key: keyof WFWindow) => (
    <th
      className="cursor-pointer px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
      onClick={() => toggleSort(key)}
    >
      {label} {sortKey === key ? (sortAsc ? "▲" : "▼") : ""}
    </th>
  );

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      <div className="max-h-[320px] overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[var(--surface)] border-b border-[var(--border)]">
            <tr>
              {th("#", "window_index")}
              <th className="px-3 py-2 text-left text-[10px] font-medium uppercase tracking-wider text-zinc-500">Test Period</th>
              {th("Bars", "num_test_bars")}
              {th("OOS Acc", "oos_accuracy")}
              {th("Sharpe", "sharpe")}
              {th("Return", "total_return")}
              {th("Max DD", "max_dd")}
              {th("Trades", "num_trades")}
            </tr>
          </thead>
          <tbody>
            {sorted.map((w) => (
              <tr key={w.window_index} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                <td className="num px-3 py-1.5 text-zinc-400">{w.window_index}</td>
                <td className="px-3 py-1.5 text-zinc-400 whitespace-nowrap">
                  {fmtDate(w.test_start)} – {fmtDate(w.test_end)}
                </td>
                <td className="num px-3 py-1.5 text-zinc-300">{w.num_test_bars}</td>
                <td className="num px-3 py-1.5 text-zinc-300">{(w.oos_accuracy * 100).toFixed(1)}%</td>
                <td className={`num px-3 py-1.5 ${w.sharpe >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {w.sharpe.toFixed(2)}
                </td>
                <td className={`num px-3 py-1.5 ${w.total_return >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {fmtPct(w.total_return)}
                </td>
                <td className="num px-3 py-1.5 text-red-400">{w.max_dd.toFixed(2)}%</td>
                <td className="num px-3 py-1.5 text-zinc-300">{w.num_trades}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Aggregate Stats with CIs ────────────────────────────────────
function WFAggregateStats({ aggregate }: { aggregate: Record<string, WFAggregateMetric> }) {
  const metrics = [
    { key: "oos_accuracy", label: "OOS Accuracy", fmt: (v: number) => `${(v * 100).toFixed(1)}%`, fmtCI: (v: number) => `${(v * 100).toFixed(1)}%` },
    { key: "sharpe", label: "Sharpe Ratio", fmt: (v: number) => v.toFixed(2), fmtCI: (v: number) => v.toFixed(2) },
    { key: "total_return", label: "Return / Window", fmt: (v: number) => `${v.toFixed(2)}%`, fmtCI: (v: number) => `${v.toFixed(2)}%` },
    { key: "win_rate", label: "Win Rate", fmt: (v: number) => `${v.toFixed(1)}%`, fmtCI: (v: number) => `${v.toFixed(1)}%` },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {metrics.map(({ key, label, fmt, fmtCI }) => {
        const stat = aggregate[key];
        if (!stat) return null;
        return (
          <div key={key} className="glow-card rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
            <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
              {label}
            </div>
            <div className="mt-1 num text-xl font-bold text-white">
              {fmt(stat.mean)}
            </div>
            <div className="mt-0.5 text-[10px] text-zinc-500">
              ± {fmt(stat.std)} &middot; 95% CI [{fmtCI(stat.ci_lower)}, {fmtCI(stat.ci_upper)}]
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Main Panel ──────────────────────────────────────────────────
export function WalkForwardPanel({ symbol, barType, labeling }: WalkForwardPanelProps) {
  const [data, setData] = useState<WFRunData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setData(null);

    const controller = new AbortController();
    fetch(
      `/api/wf-latest/${symbol}?bar_type=${barType}&labeling=${labeling}`,
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
  }, [symbol, barType, labeling]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="text-sm text-zinc-500">Loading walk-forward results...</div>
      </div>
    );
  }

  if (error === "no-data" || !data) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] py-16 px-8">
        <div className="text-sm font-medium text-zinc-400 mb-2">No Walk-Forward Results</div>
        <div className="text-xs text-zinc-600 text-center max-w-md">
          Run the walk-forward validation script to generate results:
        </div>
        <code className="mt-3 rounded bg-zinc-900 px-3 py-2 text-[11px] text-blue-400">
          python scripts/walk_forward.py --symbol {symbol} --bar-type {barType} --labeling {labeling}
        </code>
        <div className="mt-2 text-[10px] text-zinc-600">
          Or seed demo data: <code className="text-blue-400">python scripts/seed_walk_forward.py</code>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-900/50 bg-red-950/20 p-6 text-center">
        <div className="text-sm text-red-400">Error loading walk-forward data: {error}</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">
            Walk-Forward Validation
          </h2>
          <p className="text-[10px] text-zinc-500">
            {data.num_windows} windows &middot; {data.train_days}d train / {data.test_days}d test
            &middot; {new Date(data.created_at).toLocaleDateString()}
          </p>
        </div>
        <div className="rounded-md border border-zinc-700/50 bg-zinc-800/50 px-2.5 py-1">
          <span className="text-[10px] text-zinc-500">Run </span>
          <span className="num text-xs font-semibold text-white">#{data.id}</span>
        </div>
      </div>

      {/* Overfitting gap comparison */}
      <WFOverfittingGap data={data} />

      {/* Stitched OOS equity curve */}
      <WFEquityCurve data={data} />

      {/* Window timeline */}
      <WFWindowTimeline data={data} />

      {/* Per-window metrics table */}
      <WFWindowTable windows={data.windows} />

      {/* Aggregate stats with CIs */}
      <WFAggregateStats aggregate={data.aggregate} />
    </div>
  );
}
