"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
  ColorType,
} from "lightweight-charts";
import type { BarData, Signal } from "@/lib/types";

interface ChartProps {
  bars: BarData[];
  signals: Signal[];
}

export function Chart({ bars, signals }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candlestickRef = useRef<ISeriesApi<any> | null>(null);

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#09090b" },
        textColor: "#a1a1aa",
      },
      grid: {
        vertLines: { color: "#27272a" },
        horzLines: { color: "#27272a" },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: "#3f3f46",
      },
      timeScale: {
        borderColor: "#3f3f46",
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height: 500,
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      wickUpColor: "#22c55e",
    });

    chartRef.current = chart;
    candlestickRef.current = candlestickSeries;

    // Handle resize
    const resizeObserver = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      chart.applyOptions({ width });
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candlestickRef.current = null;
    };
  }, []);

  // Update candlestick data
  useEffect(() => {
    if (!candlestickRef.current || bars.length === 0) return;

    const data: CandlestickData<Time>[] = bars
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((bar) => ({
        time: (bar.timestamp / 1000) as Time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));

    candlestickRef.current.setData(data);

    // Add signal markers
    const markers = signals
      .filter((s) => s.timestamp >= bars[0]?.timestamp)
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((signal) => ({
        time: (signal.timestamp / 1000) as Time,
        position: signal.side === 1 ? ("belowBar" as const) : ("aboveBar" as const),
        color: signal.side === 1 ? "#22c55e" : "#ef4444",
        shape: signal.side === 1 ? ("arrowUp" as const) : ("arrowDown" as const),
        text: `${signal.side === 1 ? "L" : "S"} ${signal.size.toFixed(2)}`,
      }));

    // v5 uses attachPrimitive or chart-level markers — skip markers for now
    // if markers API is available on the series
    if (typeof (candlestickRef.current as any).setMarkers === "function") {
      (candlestickRef.current as any).setMarkers(markers);
    }
  }, [bars, signals]);

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-2">
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
