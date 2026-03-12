"use client";

import { useEffect, useRef, useState } from "react";
import {
  createChart,
  AreaSeries,
  type IChartApi,
  type ISeriesApi,
  type Time,
  ColorType,
} from "lightweight-charts";
import type { EquityData } from "@/lib/types";

interface EquityCurveProps {
  symbol: string;
  barType: string;
  labeling: string;
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

export function EquityCurve({ symbol, barType, labeling }: EquityCurveProps) {
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

  const [capitalInput, setCapitalInput] = useState("10000");
  const [feesInput, setFeesInput] = useState("10");
  const [data, setData] = useState<EquityData | null>(null);
  const [loading, setLoading] = useState(false);

  const startingCapital = parseFloat(capitalInput) || 10000;
  const feesBps = parseFloat(feesInput) || 10;

  // Fetch equity data (debounced)
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);

    const timer = setTimeout(() => {
      fetch(
        `/api/equity/${symbol}?bar_type=${barType}&labeling=${labeling}&starting_capital=${startingCapital}&fees_bps=${feesBps}`,
        { signal: controller.signal },
      )
        .then((r) => r.json())
        .then((d) => {
          setData(d);
          setLoading(false);
        })
        .catch((e) => {
          if (e.name !== "AbortError") {
            setData(null);
            setLoading(false);
          }
        });
    }, 400);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [symbol, barType, labeling, startingCapital, feesBps]);

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

    // Equity line
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
    };
  }, []);

  // Update chart data
  useEffect(() => {
    if (!data || !equitySeriesRef.current || !investedSeriesRef.current || !ddSeriesRef.current) return;

    if (data.timestamps.length === 0) {
      equitySeriesRef.current.setData([]);
      investedSeriesRef.current.setData([]);
      ddSeriesRef.current.setData([]);
      return;
    }

    // Deduplicate timestamps (ascending, keep last)
    const seen = new Map<number, number>();
    for (let i = 0; i < data.timestamps.length; i++) {
      seen.set(data.timestamps[i], i);
    }
    const indices = [...seen.values()].sort((a, b) => data.timestamps[a] - data.timestamps[b]);

    const eqData = indices.map((i) => ({
      time: (data.timestamps[i] / 1000) as Time,
      value: data.equity[i],
    }));
    const invData = indices.map((i) => ({
      time: (data.timestamps[i] / 1000) as Time,
      value: data.total_invested[i],
    }));
    const ddData = indices.map((i) => ({
      time: (data.timestamps[i] / 1000) as Time,
      value: data.drawdown[i],
    }));

    equitySeriesRef.current.setData(eqData);
    investedSeriesRef.current.setData(invData);
    ddSeriesRef.current.setData(ddData);
  }, [data]);

  const m = data?.metrics;
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
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-[10px] text-zinc-500">
            Capital
            <input
              type="number"
              value={capitalInput}
              onChange={(e) => setCapitalInput(e.target.value)}
              className="w-20 rounded border border-[var(--border)] bg-[var(--background)] px-2 py-0.5 text-[11px] text-zinc-300 num outline-none focus:border-blue-500/50"
              min={100}
              step={1000}
            />
          </label>
          <label className="flex items-center gap-1.5 text-[10px] text-zinc-500">
            Fees (bps)
            <input
              type="number"
              value={feesInput}
              onChange={(e) => setFeesInput(e.target.value)}
              className="w-16 rounded border border-[var(--border)] bg-[var(--background)] px-2 py-0.5 text-[11px] text-zinc-300 num outline-none focus:border-blue-500/50"
              min={0}
              step={1}
            />
          </label>
        </div>
      </div>

      {/* Metrics cards */}
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

      {/* Equity chart */}
      <div ref={equityContainerRef} className="w-full" />

      {/* Drawdown label */}
      <div className="flex items-center border-t border-b border-[var(--border)] px-4 py-1">
        <span className="text-[10px] font-medium text-zinc-500">Underwater Drawdown</span>
      </div>

      {/* Drawdown chart */}
      <div ref={ddContainerRef} className="w-full" />

      {/* No data message */}
      {data && data.timestamps.length === 0 && !loading && (
        <div className="flex items-center justify-center py-8 text-sm text-zinc-600">
          No signals available for simulation. Wait for signals to accumulate.
        </div>
      )}
    </div>
  );
}
