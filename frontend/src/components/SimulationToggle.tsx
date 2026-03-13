"use client";

import type { SimulationConfig } from "@/lib/types";

interface SimulationToggleProps {
  config: SimulationConfig;
  onChange: (config: SimulationConfig) => void;
}

const MODES = ["simple", "realistic", "both"] as const;

const INPUT_CLS =
  "rounded border border-[var(--border)] bg-[var(--background)] px-2 py-0.5 text-[11px] text-zinc-300 num outline-none focus:border-blue-500/50";

export function SimulationToggle({ config, onChange }: SimulationToggleProps) {
  const update = (patch: Partial<SimulationConfig>) =>
    onChange({ ...config, ...patch });

  return (
    <div className="flex flex-wrap items-center gap-3">
      {/* Mode segmented control */}
      <div className="flex rounded-md border border-[var(--border)] overflow-hidden">
        {MODES.map((mode) => (
          <button
            key={mode}
            onClick={() => update({ mode })}
            className={`px-3 py-0.5 text-[10px] font-medium capitalize transition-colors ${
              config.mode === mode
                ? "bg-blue-600/80 text-white"
                : "bg-[var(--surface)] text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {mode}
          </button>
        ))}
      </div>

      {/* Capital - always visible */}
      <label className="flex items-center gap-1.5 text-[10px] text-zinc-500">
        Capital
        <input
          type="number"
          value={config.starting_capital}
          onChange={(e) =>
            update({ starting_capital: parseFloat(e.target.value) || 10000 })
          }
          className={`w-20 ${INPUT_CLS}`}
          min={100}
          step={1000}
        />
      </label>

      {/* Fees - always visible */}
      <label className="flex items-center gap-1.5 text-[10px] text-zinc-500">
        Fees (bps)
        <input
          type="number"
          value={config.fees_bps}
          onChange={(e) =>
            update({ fees_bps: parseFloat(e.target.value) || 10 })
          }
          className={`w-16 ${INPUT_CLS}`}
          min={0}
          step={1}
        />
      </label>

      {/* Realistic-only controls */}
      {(config.mode === "realistic" || config.mode === "both") && (
        <>
          {/* VIP Tier */}
          <label className="flex items-center gap-1.5 text-[10px] text-zinc-500">
            VIP
            <select
              value={config.vip_tier}
              onChange={(e) => update({ vip_tier: parseInt(e.target.value) })}
              className={`w-14 ${INPUT_CLS} cursor-pointer`}
            >
              {Array.from({ length: 10 }, (_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </label>

          {/* BNB Discount */}
          <label className="flex items-center gap-1.5 text-[10px] text-zinc-500 cursor-pointer">
            <input
              type="checkbox"
              checked={config.bnb_discount}
              onChange={(e) => update({ bnb_discount: e.target.checked })}
              className="h-3 w-3 rounded border-[var(--border)] bg-[var(--background)] accent-blue-500"
            />
            BNB
          </label>

          {/* Urgency slider */}
          <label className="flex items-center gap-1.5 text-[10px] text-zinc-500">
            Urgency
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={config.urgency}
              onChange={(e) =>
                update({ urgency: parseFloat(e.target.value) })
              }
              className="h-1 w-16 cursor-pointer accent-blue-500"
            />
            <span className="num w-6 text-[10px] text-zinc-400">
              {config.urgency.toFixed(1)}
            </span>
          </label>
        </>
      )}
    </div>
  );
}
