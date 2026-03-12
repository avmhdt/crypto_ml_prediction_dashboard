"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  createSeriesMarkers,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type IPriceLine,
  type CandlestickData,
  type SeriesMarker,
  type Time,
  ColorType,
} from "lightweight-charts";
import type { BarData, Signal } from "@/lib/types";

interface ChartProps {
  bars: BarData[];
  signals: Signal[];
  labeling: string;
}

export function Chart({ bars, signals, labeling }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candlestickRef = useRef<ISeriesApi<any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volumeRef = useRef<ISeriesApi<any> | null>(null);
  const markersPluginRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const barrierLinesRef = useRef<IPriceLine[]>([]);
  const verticalBarrierRef = useRef<HTMLDivElement | null>(null);
  const rangeHandlerRef = useRef<(() => void) | null>(null);
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);

  // Clear barrier price lines and vertical time barrier
  const clearBarriers = useCallback(() => {
    if (candlestickRef.current) {
      for (const line of barrierLinesRef.current) {
        candlestickRef.current.removePriceLine(line);
      }
    }
    barrierLinesRef.current = [];
    if (verticalBarrierRef.current) {
      verticalBarrierRef.current.style.display = "none";
    }
    if (rangeHandlerRef.current && chartRef.current) {
      chartRef.current
        .timeScale()
        .unsubscribeVisibleLogicalRangeChange(rangeHandlerRef.current);
      rangeHandlerRef.current = null;
    }
    setSelectedSignal(null);
  }, []);

  // Draw barrier price lines for a triple_barrier signal
  const drawBarriers = useCallback((signal: Signal) => {
    if (!candlestickRef.current) return;

    clearBarriers();
    setSelectedSignal(signal);

    const lines: IPriceLine[] = [];

    if (signal.sl_price != null) {
      lines.push(
        candlestickRef.current.createPriceLine({
          price: signal.sl_price,
          color: "#ef4444",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `SL ${signal.sl_price.toFixed(2)}`,
        })
      );
    }

    if (signal.pt_price != null) {
      lines.push(
        candlestickRef.current.createPriceLine({
          price: signal.pt_price,
          color: "#22c55e",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `TP ${signal.pt_price.toFixed(2)}`,
        })
      );
    }

    if (signal.entry_price != null) {
      lines.push(
        candlestickRef.current.createPriceLine({
          price: signal.entry_price,
          color: "#3b82f6",
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title: `Entry ${signal.entry_price.toFixed(2)}`,
        })
      );
    }

    barrierLinesRef.current = lines;

    // Draw vertical time barrier
    if (signal.time_barrier != null && chartRef.current) {
      const tbTime = (signal.time_barrier / 1000) as Time;
      const chart = chartRef.current;

      const updateVerticalPos = () => {
        const coord = chart.timeScale().timeToCoordinate(tbTime);
        if (
          coord !== null &&
          coord !== undefined &&
          verticalBarrierRef.current
        ) {
          verticalBarrierRef.current.style.left = `${coord}px`;
          verticalBarrierRef.current.style.display = "block";
        } else if (verticalBarrierRef.current) {
          verticalBarrierRef.current.style.display = "none";
        }
      };

      // Defer to ensure chart layout is complete
      requestAnimationFrame(() => {
        requestAnimationFrame(updateVerticalPos);
      });
      chart
        .timeScale()
        .subscribeVisibleLogicalRangeChange(updateVerticalPos);
      rangeHandlerRef.current = updateVerticalPos;
    }
  }, [clearBarriers]);

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
        rightOffset: 60,
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

    // Create markers plugin (v5 API)
    const markersPlugin = createSeriesMarkers(candlestickSeries, []);
    markersPluginRef.current = markersPlugin;

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
      markersPlugin.detach();
      chart.remove();
      chartRef.current = null;
      candlestickRef.current = null;
      volumeRef.current = null;
      markersPluginRef.current = null;
      barrierLinesRef.current = [];
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
  }, [bars]);

  // Update signal markers
  useEffect(() => {
    if (!markersPluginRef.current || bars.length === 0) return;

    const sortedBars = [...bars].sort((a, b) => a.timestamp - b.timestamp);
    const firstTimestamp = sortedBars[0]?.timestamp ?? 0;

    const markers: SeriesMarker<Time>[] = signals
      .filter((s) => s.timestamp >= firstTimestamp)
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
        text: `${signal.side === 1 ? "LONG" : "SHORT"} ${signal.size.toFixed(2)}`,
      }));

    markersPluginRef.current.setMarkers(markers);
  }, [bars, signals]);

  // Handle click for barrier visualization (triple_barrier only)
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const handleClick = (param: { time?: Time }) => {
      if (!param.time) {
        clearBarriers();
        return;
      }

      // Only show barriers for triple_barrier labeling
      if (labeling !== "triple_barrier") {
        clearBarriers();
        return;
      }

      const clickedTime = (param.time as number) * 1000;

      // Find the closest signal within a tolerance window
      const tolerance = 60 * 60 * 1000; // 1 hour tolerance
      let closest: Signal | null = null;
      let closestDist = Infinity;

      for (const signal of signals) {
        const dist = Math.abs(signal.timestamp - clickedTime);
        if (dist < tolerance && dist < closestDist) {
          closest = signal;
          closestDist = dist;
        }
      }

      if (closest && (closest.sl_price != null || closest.pt_price != null)) {
        drawBarriers(closest);
      } else {
        clearBarriers();
      }
    };

    chart.subscribeClick(handleClick);
    return () => {
      chart.unsubscribeClick(handleClick);
    };
  }, [signals, labeling, clearBarriers, drawBarriers]);

  // Clear barriers when labeling method changes
  useEffect(() => {
    clearBarriers();
  }, [labeling, clearBarriers]);

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
        <div className="flex items-center gap-3">
          {selectedSignal && (
            <button
              onClick={clearBarriers}
              className="flex items-center gap-1.5 rounded border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 text-[10px] font-medium text-blue-400 transition-colors hover:bg-blue-500/20"
            >
              <span>{selectedSignal.side === 1 ? "\u25b2" : "\u25bc"}</span>
              {selectedSignal.side === 1 ? "LONG" : "SHORT"} barriers
              <span className="ml-1 text-zinc-500">\u2715</span>
            </button>
          )}
          {labeling === "triple_barrier" && !selectedSignal && signals.length > 0 && (
            <span className="text-[10px] text-zinc-600">
              Click a signal to show barriers
            </span>
          )}
          <span className="text-[10px] text-zinc-600">
            {barCount > 0 ? `${barCount} bars` : "No data"}
          </span>
        </div>
      </div>
      <div style={{ position: "relative" }}>
        <div ref={containerRef} className="w-full" />
        <div
          ref={verticalBarrierRef}
          style={{
            position: "absolute",
            top: 0,
            height: "100%",
            width: 0,
            borderLeft: "1px dashed #f59e0b",
            display: "none",
            pointerEvents: "none",
            zIndex: 10,
          }}
        >
          <span
            style={{
              position: "absolute",
              top: 4,
              left: 4,
              fontSize: "10px",
              color: "#f59e0b",
              whiteSpace: "nowrap",
              background: "#0a0a0f",
              padding: "1px 4px",
              borderRadius: "2px",
            }}
          >
            Time Barrier
          </span>
        </div>
      </div>
    </div>
  );
}
