"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volumeRef = useRef<ISeriesApi<any> | null>(null);

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0a0a0f" },
        textColor: "#71717a",
        fontFamily: "'Geist Mono', 'SF Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#18181f" },
        horzLines: { color: "#18181f" },
      },
      crosshair: {
        mode: 0,
        vertLine: {
          color: "#3b82f6",
          width: 1,
          style: 3,
          labelBackgroundColor: "#1d4ed8",
        },
        horzLine: {
          color: "#3b82f6",
          width: 1,
          style: 3,
          labelBackgroundColor: "#1d4ed8",
        },
      },
      rightPriceScale: {
        borderColor: "#1e1e2a",
        scaleMargins: { top: 0.1, bottom: 0.25 },
      },
      timeScale: {
        borderColor: "#1e1e2a",
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height: 520,
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#22c55e",
      wickDownColor: "#71717a",
      wickUpColor: "#71717a",
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    candlestickRef.current = candlestickSeries;
    volumeRef.current = volumeSeries;

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
      volumeRef.current = null;
    };
  }, []);

  // Update candlestick and volume data
  useEffect(() => {
    if (!candlestickRef.current || !volumeRef.current || bars.length === 0)
      return;

    const sortedBars = [...bars].sort((a, b) => a.timestamp - b.timestamp);

    const candleData: CandlestickData<Time>[] = sortedBars.map((bar) => ({
      time: (bar.timestamp / 1000) as Time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));

    const volumeData = sortedBars.map((bar) => ({
      time: (bar.timestamp / 1000) as Time,
      value: bar.volume,
      color: bar.close >= bar.open ? "#22c55e20" : "#ef444420",
    }));

    candlestickRef.current.setData(candleData);
    volumeRef.current.setData(volumeData);

    // Add signal markers
    const markers = signals
      .filter((s) => s.timestamp >= sortedBars[0]?.timestamp)
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((signal) => ({
        time: (signal.timestamp / 1000) as Time,
        position:
          signal.side === 1
            ? ("belowBar" as const)
            : ("aboveBar" as const),
        color: signal.side === 1 ? "#22c55e" : "#ef4444",
        shape:
          signal.side === 1
            ? ("arrowUp" as const)
            : ("arrowDown" as const),
        text: `${signal.side === 1 ? "L" : "S"} ${signal.size.toFixed(2)}`,
      }));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (typeof (candlestickRef.current as any).setMarkers === "function") {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (candlestickRef.current as any).setMarkers(markers);
    }
  }, [bars, signals]);

  const barCount = bars.length;
  const lastPrice = bars.length > 0 ? bars[bars.length - 1]?.close : null;
  const prevPrice =
    bars.length > 1 ? bars[bars.length - 2]?.close : lastPrice;
  const priceChange =
    lastPrice && prevPrice ? ((lastPrice - prevPrice) / prevPrice) * 100 : 0;

  return (
    <div className="glow-card">
      {/* Chart header */}
      <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-2">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-zinc-400">
            Price Chart
          </span>
          {lastPrice && (
            <span className="num text-sm font-bold text-white">
              {lastPrice.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </span>
          )}
          {priceChange !== 0 && (
            <span
              className={`num text-xs font-medium ${
                priceChange >= 0 ? "text-green-400" : "text-red-400"
              }`}
            >
              {priceChange >= 0 ? "+" : ""}
              {priceChange.toFixed(2)}%
            </span>
          )}
        </div>
        <span className="text-[10px] text-zinc-600">
          {barCount > 0 ? `${barCount} bars` : "No data"}
        </span>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
