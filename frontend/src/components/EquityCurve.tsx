"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import {
  createChart,
  AreaSeries,
  type IChartApi,
  type ISeriesApi,
  type Time,
  ColorType,
} from "lightweight-charts";
import { SimulationToggle } from "@/components/SimulationToggle";
import type {
  EquityData,
  SimulationConfig,
  RealisticMetrics,
  EquityComparisonData,
  CostBreakdown,
} from "@/lib/types";

interface EquityCurveProps {
  symbol: string;
  barType: string;
  labeling: string;
  simulationConfig: SimulationConfig;
  onSimulationConfigChange: (config: SimulationConfig) => void;
}

const CHART_OPTS = {
  layout: {
    background: { type: ColorType.Solid as const, color: "#0a0a0f" },
    textColor: "#71717a",
    fontFamily: "'Geist Mono', 'SF Mono', monospace",
    fontSize: 11,
  },
  grid: {
    vertLines: { color: "#18181f" },
    horzLines: { color: "#18181f" },
  },
  crosshair: {
    mode: 0 as const,
    vertLine: { color: "#3b82f6", width: 1 as const, style: 3 as const, labelBackgroundColor: "#1d4ed8" },
    horzLine: { color: "#3b82f6", width: 1 as const, style: 3 as const, labelBackgroundColor: "#1d4ed8" },
  },
  rightPriceScale: { borderColor: "#1e1e2a" },
  timeScale: { borderColor: "#1e1e2a", timeVisible: true, secondsVisible: false },
};

function MetricCard({
  label,
  value,
  suffix,
  color,
}: {
  label: string;
  value: string | number;
  suffix?: string;
  color: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-4 py-3">
      <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
        {label}
      </span>
      <span className={`num text-lg font-bold ${color}`}>
        {value}
        {suffix && <span className="text-xs font-normal text-zinc-500">{suffix}</span>}
      </span>
    </div>
  );
}

/** Mini horizontal bar chart for cost breakdown */
function CostBreakdownBar({ breakdown }: { breakdown: CostBreakdown }) {
  const items: { label: string; value: number; color: string }[] = [
    { label: "Exchange Fee", value: breakdown.exchange_fee, color: "bg-blue-500" },
    { label: "Funding", value: breakdown.funding_cost, color: "bg-purple-500" },
    { label: "Spread", value: breakdown.spread_cost, color: "bg-amber-500" },
    { label: "Slippage", value: breakdown.slippage, color: "bg-orange-500" },
    { label: "Impact", value: breakdown.market_impact, color: "bg-red-500" },
  ];

  const total = breakdown.total || 1; // avoid division by zero

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-4 py-3">
      <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
        Cost Breakdown (${breakdown.total.toFixed(2)})
      </span>
      {/* Stacked bar */}
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-[var(--background)]">
        {items.map((item) => {
          const pct = (item.value / total) * 100;
          if (pct < 0.5) return null;
          return (
            <div
              key={item.label}
              className={`${item.color} transition-all`}
              style={{ width: `${pct}%` }}
              title={`${item.label}: $${item.value.toFixed(2)} (${pct.toFixed(1)}%)`}
            />
          );
        })}
      </div>
      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {items.map((item) => (
          <span key={item.label} className="flex items-center gap-1 text-[9px] text-zinc-500">
            <span className={`inline-block h-1.5 w-1.5 rounded-full ${item.color}`} />
            {item.label}: ${item.value.toFixed(2)}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Deduplicate and sort timestamp data, returning sorted indices */
function deduplicateTimestamps(timestamps: number[]): number[] {
  const seen = new Map<number, number>();
  for (let i = 0; i < timestamps.length; i++) {
    seen.set(timestamps[i], i);
  }
  return [...seen.values()].sort((a, b) => timestamps[a] - timestamps[b]);
}

/** Convert equity data arrays to lightweight-charts format */
function toChartData(data: EquityData, indices: number[]) {
  return {
    equity: indices.map((i) => ({
      time: (data.timestamps[i] / 1000) as Time,
      value: data.equity[i],
    })),
    invested: indices.map((i) => ({
      time: (data.timestamps[i] / 1000) as Time,
      value: data.total_invested[i],
    })),
    drawdown: indices.map((i) => ({
      time: (data.timestamps[i] / 1000) as Time,
      value: data.drawdown[i],
    })),
  };
}

export function EquityCurve({
  symbol,
  barType,
  labeling,
  simulationConfig,
  onSimulationConfigChange,
}: EquityCurveProps) {
  const equityContainerRef = useRef<HTMLDivElement>(null);
  const ddContainerRef = useRef<HTMLDivElement>(null);
  const equityChartRef = useRef<IChartApi | null>(null);
  const ddChartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const equitySeriesRef = useRef<ISeriesApi<any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const investedSeriesRef = useRef<ISeriesApi<any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ddSeriesRef = useRef<ISeriesApi<any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const realisticSeriesRef = useRef<ISeriesApi<any> | null>(null);

  const [simpleData, setSimpleData] = useState<EquityData | null>(null);
  const [realisticData, setRealisticData] = useState<(EquityData & { metrics: RealisticMetrics }) | null>(null);
  const [loading, setLoading] = useState(false);

  const mode = simulationConfig.mode;

  // Build the fetch URL based on mode
  const fetchUrl = useMemo(() => {
    const base = `/api/equity/${symbol}?bar_type=${barType}&labeling=${labeling}&starting_capital=${simulationConfig.starting_capital}&fees_bps=${simulationConfig.fees_bps}`;
    if (mode === "simple") return base + "&simulation_mode=simple";
    const realisticParams = `&simulation_mode=${mode}&vip_tier=${simulationConfig.vip_tier}&bnb_discount=${simulationConfig.bnb_discount}&urgency=${simulationConfig.urgency}`;
    return base + realisticParams;
  }, [symbol, barType, labeling, simulationConfig, mode]);

  // Fetch equity data (debounced)
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);

    const timer = setTimeout(() => {
      fetch(fetchUrl, { signal: controller.signal })
        .then((r) => r.json())
        .then((d) => {
          if (mode === "both") {
            // Response shape: EquityComparisonData
            const comparison = d as EquityComparisonData;
            setSimpleData(comparison.simple);
            setRealisticData(comparison.realistic);
          } else if (mode === "realistic") {
            // Response shape: EquityData with RealisticMetrics
            setSimpleData(null);
            setRealisticData(d as EquityData & { metrics: RealisticMetrics });
          } else {
            // Response shape: EquityData
            setSimpleData(d as EquityData);
            setRealisticData(null);
          }
          setLoading(false);
        })
        .catch((e) => {
          if (e.name !== "AbortError") {
            setSimpleData(null);
            setRealisticData(null);
            setLoading(false);
          }
        });
    }, 400);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [fetchUrl, mode]);

  // Initialize charts
  useEffect(() => {
    if (!equityContainerRef.current || !ddContainerRef.current) return;

    const eqChart = createChart(equityContainerRef.current, {
      ...CHART_OPTS,
      width: equityContainerRef.current.clientWidth,
      height: 320,
      rightPriceScale: {
        ...CHART_OPTS.rightPriceScale,
        scaleMargins: { top: 0.05, bottom: 0.05 },
      },
    });

    const ddChart = createChart(ddContainerRef.current, {
      ...CHART_OPTS,
      width: ddContainerRef.current.clientWidth,
      height: 120,
      rightPriceScale: {
        ...CHART_OPTS.rightPriceScale,
        scaleMargins: { top: 0.05, bottom: 0.05 },
      },
    });

    // Simple equity line (green)
    const eqSeries = eqChart.addSeries(AreaSeries, {
      lineColor: "#22c55e",
      topColor: "rgba(34,197,94,0.28)",
      bottomColor: "rgba(34,197,94,0.02)",
      lineWidth: 2,
      priceFormat: { type: "custom", formatter: (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }) },
    });

    // Total invested (separate scale)
    const invSeries = eqChart.addSeries(AreaSeries, {
      lineColor: "rgba(59,130,246,0.6)",
      topColor: "rgba(59,130,246,0.15)",
      bottomColor: "rgba(59,130,246,0.01)",
      lineWidth: 1,
      priceScaleId: "invested",
      priceFormat: { type: "custom", formatter: (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }) },
    });
    eqChart.priceScale("invested").applyOptions({
      scaleMargins: { top: 0.3, bottom: 0.0 },
      visible: false,
    });

    // Realistic equity line (blue) - used in "both" and "realistic" modes
    const realisticSeries = eqChart.addSeries(AreaSeries, {
      lineColor: "#3b82f6",
      topColor: "rgba(59,130,246,0.20)",
      bottomColor: "rgba(59,130,246,0.02)",
      lineWidth: 2,
      priceFormat: { type: "custom", formatter: (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }) },
    });

    // Drawdown area
    const ddSeries = ddChart.addSeries(AreaSeries, {
      lineColor: "#ef4444",
      topColor: "rgba(239,68,68,0.01)",
      bottomColor: "rgba(239,68,68,0.35)",
      lineWidth: 1,
      priceFormat: { type: "custom", formatter: (v: number) => `${(v * 100).toFixed(1)}%` },
      invertFilledArea: true,
    });

    equityChartRef.current = eqChart;
    ddChartRef.current = ddChart;
    equitySeriesRef.current = eqSeries;
    investedSeriesRef.current = invSeries;
    ddSeriesRef.current = ddSeries;
    realisticSeriesRef.current = realisticSeries;

    // Sync time scales (guard prevents infinite ping-pong)
    let syncing = false;
    eqChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range && !syncing) { syncing = true; ddChart.timeScale().setVisibleLogicalRange(range); syncing = false; }
    });
    ddChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range && !syncing) { syncing = true; eqChart.timeScale().setVisibleLogicalRange(range); syncing = false; }
    });

    // Resize
    const eqObs = new ResizeObserver((entries) => {
      eqChart.applyOptions({ width: entries[0].contentRect.width });
    });
    const ddObs = new ResizeObserver((entries) => {
      ddChart.applyOptions({ width: entries[0].contentRect.width });
    });
    eqObs.observe(equityContainerRef.current);
    ddObs.observe(ddContainerRef.current);

    return () => {
      eqObs.disconnect();
      ddObs.disconnect();
      eqChart.remove();
      ddChart.remove();
      equityChartRef.current = null;
      ddChartRef.current = null;
      equitySeriesRef.current = null;
      investedSeriesRef.current = null;
      ddSeriesRef.current = null;
      realisticSeriesRef.current = null;
    };
  }, []);

  // Update chart data
  useEffect(() => {
    if (!equitySeriesRef.current || !investedSeriesRef.current || !ddSeriesRef.current || !realisticSeriesRef.current) return;

    // Determine which data to show on the green (primary) series
    const primaryData = mode === "realistic" ? null : simpleData;
    // Determine which data to show on the blue (realistic) series
    const secondaryData = realisticData;

    // Clear all series first
    equitySeriesRef.current.setData([]);
    investedSeriesRef.current.setData([]);
    ddSeriesRef.current.setData([]);
    realisticSeriesRef.current.setData([]);

    // In "simple" mode: green = simple, blue hidden
    // In "realistic" mode: green hidden, blue = realistic
    // In "both" mode: green = simple, blue = realistic

    if (primaryData && primaryData.timestamps.length > 0) {
      const indices = deduplicateTimestamps(primaryData.timestamps);
      const chartData = toChartData(primaryData, indices);
      equitySeriesRef.current.setData(chartData.equity);
      investedSeriesRef.current.setData(chartData.invested);
      ddSeriesRef.current.setData(chartData.drawdown);
    }

    if (secondaryData && secondaryData.timestamps.length > 0) {
      const indices = deduplicateTimestamps(secondaryData.timestamps);
      const chartData = toChartData(secondaryData, indices);
      realisticSeriesRef.current.setData(chartData.equity);

      // If mode is "realistic" (no simple data), use realistic data for drawdown too
      if (mode === "realistic") {
        ddSeriesRef.current.setData(chartData.drawdown);
        investedSeriesRef.current.setData(chartData.invested);
      }
    }
  }, [simpleData, realisticData, mode]);

  // Determine which metrics to display
  const primaryMetrics = mode === "realistic" ? realisticData?.metrics : simpleData?.metrics;
  const realisticMetrics = realisticData?.metrics as RealisticMetrics | undefined;
  const hasData = (simpleData && simpleData.timestamps.length > 0) || (realisticData && realisticData.timestamps.length > 0);
  const hasNoData = !loading &&
    ((mode === "simple" && simpleData?.timestamps.length === 0) ||
     (mode === "realistic" && realisticData?.timestamps.length === 0) ||
     (mode === "both" && simpleData?.timestamps.length === 0 && realisticData?.timestamps.length === 0));

  const m = primaryMetrics;
  const returnColor = m && m.total_return >= 0 ? "text-green-400" : "text-red-400";
  const ddColor = "text-red-400";

  return (
    <div className="glow-card">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-2">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-zinc-400">Equity Curve</span>
          {loading && (
            <span className="text-[10px] text-zinc-600 animate-pulse">simulating...</span>
          )}
          {mode === "both" && !loading && hasData && (
            <span className="flex items-center gap-2 text-[9px]">
              <span className="flex items-center gap-1">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500" />
                <span className="text-zinc-500">Simple</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-blue-500" />
                <span className="text-zinc-500">Realistic</span>
              </span>
            </span>
          )}
        </div>
        <SimulationToggle
          config={simulationConfig}
          onChange={onSimulationConfigChange}
        />
      </div>

      {/* Primary metrics cards */}
      {m && m.num_trades > 0 && (
        <div className="grid grid-cols-5 gap-2 px-4 py-3 border-b border-[var(--border)]">
          <MetricCard
            label="Total Return"
            value={`${m.total_return >= 0 ? "+" : ""}${m.total_return.toFixed(2)}`}
            suffix="%"
            color={returnColor}
          />
          <MetricCard
            label="Sharpe Ratio"
            value={m.sharpe.toFixed(2)}
            color={m.sharpe >= 1 ? "text-green-400" : m.sharpe >= 0 ? "text-amber-400" : "text-red-400"}
          />
          <MetricCard
            label="Max Drawdown"
            value={m.max_dd.toFixed(2)}
            suffix="%"
            color={ddColor}
          />
          <MetricCard
            label="Win Rate"
            value={m.win_rate.toFixed(1)}
            suffix="%"
            color={m.win_rate >= 50 ? "text-green-400" : "text-amber-400"}
          />
          <MetricCard
            label="Trades"
            value={m.num_trades}
            color="text-blue-400"
          />
        </div>
      )}

      {/* Realistic-specific metrics (shown in "both" or "realistic" mode when available) */}
      {realisticMetrics && (mode === "both" || mode === "realistic") && (
        <div className="border-b border-[var(--border)]">
          <div className="grid grid-cols-6 gap-2 px-4 py-3">
            <MetricCard
              label="Fill Rate"
              value={(realisticMetrics.fill_rate * 100).toFixed(1)}
              suffix="%"
              color={realisticMetrics.fill_rate >= 0.9 ? "text-green-400" : realisticMetrics.fill_rate >= 0.7 ? "text-amber-400" : "text-red-400"}
            />
            <MetricCard
              label="Avg Slippage"
              value={realisticMetrics.avg_slippage_bps.toFixed(2)}
              suffix=" bps"
              color={realisticMetrics.avg_slippage_bps <= 2 ? "text-green-400" : realisticMetrics.avg_slippage_bps <= 5 ? "text-amber-400" : "text-red-400"}
            />
            <MetricCard
              label="Maker Ratio"
              value={(realisticMetrics.maker_ratio * 100).toFixed(1)}
              suffix="%"
              color={realisticMetrics.maker_ratio >= 0.5 ? "text-green-400" : "text-amber-400"}
            />
            <MetricCard
              label="Avg Wait"
              value={realisticMetrics.avg_queue_wait_ms.toFixed(0)}
              suffix=" ms"
              color="text-zinc-300"
            />
            <MetricCard
              label="Funding Cost"
              value={`$${realisticMetrics.funding_total.toFixed(2)}`}
              color={realisticMetrics.funding_total <= 0 ? "text-green-400" : "text-red-400"}
            />
            <MetricCard
              label="Unfilled"
              value={realisticMetrics.num_unfilled}
              color={realisticMetrics.num_unfilled === 0 ? "text-green-400" : "text-amber-400"}
            />
          </div>
          {/* Cost breakdown bar */}
          <div className="px-4 pb-3">
            <CostBreakdownBar breakdown={realisticMetrics.cost_breakdown} />
          </div>
        </div>
      )}

      {/* Equity chart */}
      <div ref={equityContainerRef} className="w-full" />

      {/* Drawdown label */}
      <div className="flex items-center border-t border-b border-[var(--border)] px-4 py-1">
        <span className="text-[10px] font-medium text-zinc-500">Underwater Drawdown</span>
      </div>

      {/* Drawdown chart */}
      <div ref={ddContainerRef} className="w-full" />

      {/* No data message */}
      {hasNoData && (
        <div className="flex items-center justify-center py-8 text-sm text-zinc-600">
          No signals available for simulation. Wait for signals to accumulate.
        </div>
      )}
    </div>
  );
}
