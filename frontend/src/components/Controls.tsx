"use client";

interface ControlsProps {
  symbols: string[];
  barTypes: string[];
  labelingMethods: string[];
  selectedSymbol: string;
  selectedBarType: string;
  selectedLabeling: string;
  onSymbolChange: (symbol: string) => void;
  onBarTypeChange: (barType: string) => void;
  onLabelingChange: (labeling: string) => void;
}

export function Controls({
  symbols,
  barTypes,
  labelingMethods,
  selectedSymbol,
  selectedBarType,
  selectedLabeling,
  onSymbolChange,
  onBarTypeChange,
  onLabelingChange,
}: ControlsProps) {
  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-zinc-800 bg-zinc-950 px-6 py-3">
      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-zinc-400">Symbol</label>
        <select
          value={selectedSymbol}
          onChange={(e) => onSymbolChange(e.target.value)}
          className="rounded border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-white focus:border-blue-500 focus:outline-none"
        >
          {symbols.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-zinc-400">Bar Type</label>
        <select
          value={selectedBarType}
          onChange={(e) => onBarTypeChange(e.target.value)}
          className="rounded border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-white focus:border-blue-500 focus:outline-none"
        >
          {barTypes.map((bt) => (
            <option key={bt} value={bt}>
              {bt.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-zinc-400">Labeling</label>
        <select
          value={selectedLabeling}
          onChange={(e) => onLabelingChange(e.target.value)}
          className="rounded border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-white focus:border-blue-500 focus:outline-none"
        >
          {labelingMethods.map((lm) => (
            <option key={lm} value={lm}>
              {lm.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
