"use client";

import { useState, useEffect } from "react";

import type { ViewMode } from "@/lib/types";

interface HeaderProps {
  connected: boolean;
  symbol: string;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}

export function Header({ connected, symbol, viewMode, onViewModeChange }: HeaderProps) {
  const [time, setTime] = useState("");

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setTime(
        now.toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }) +
          "." +
          String(now.getMilliseconds()).padStart(3, "0").slice(0, 2)
      );
    };
    tick();
    const id = setInterval(tick, 100);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="relative flex items-center justify-between border-b border-[var(--border)] bg-[var(--surface)] px-6 py-3">
      {/* Gradient top border accent */}
      <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-blue-500/60 to-transparent" />

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2.5">
          {/* Logo mark */}
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 text-xs font-bold text-white shadow-lg shadow-blue-500/20">
            ML
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-white">
              Alatar
            </h1>
            <p className="text-[10px] leading-none text-zinc-500">
              Crypto ML Predictions
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-5">
        {/* Active symbol badge */}
        <div className="flex items-center gap-1.5 rounded-md border border-zinc-700/50 bg-zinc-800/50 px-2.5 py-1">
          <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
            Active
          </span>
          <span className="num text-sm font-semibold text-white">{symbol}</span>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center rounded-lg border border-zinc-700/50 bg-zinc-800/50 p-0.5">
          {(["live", "walk-forward"] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => onViewModeChange(mode)}
              className={`rounded-md px-3 py-1 text-[11px] font-medium transition-all ${
                viewMode === mode
                  ? "bg-blue-600 text-white shadow-sm shadow-blue-500/20"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {mode === "live" ? "Live" : "Walk-Forward"}
            </button>
          ))}
        </div>

        {/* Live clock */}
        <span className="num text-xs text-zinc-500">{time}</span>

        {/* Connection status */}
        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${
              connected
                ? "bg-green-500 pulse-live"
                : "bg-red-500"
            }`}
          />
          <span
            className={`text-[11px] font-medium ${
              connected ? "text-green-400" : "text-red-400"
            }`}
          >
            {connected ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </div>
    </header>
  );
}
