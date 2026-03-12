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

const DISPLAY_NAMES: Record<string, string> = {
  time: "Time",
  tick: "Tick",
  volume: "Volume",
  dollar: "Dollar",
  tick_imbalance: "Tick Imb",
  volume_imbalance: "Vol Imb",
  dollar_imbalance: "$ Imb",
  tick_run: "Tick Run",
  volume_run: "Vol Run",
  dollar_run: "$ Run",
  triple_barrier: "Triple Barrier",
  trend_scanning: "Trend Scan",
  directional_change: "Dir Change",
};

function PillGroup<T extends string>({
  label,
  items,
  selected,
  onChange,
  colorFn,
}: {
  label: string;
  items: T[];
  selected: T;
  onChange: (v: T) => void;
  colorFn?: (item: T, isSelected: boolean) => string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[10px] font-medium uppercase tracking-widest text-zinc-500">
        {label}
      </span>
      <div className="flex flex-wrap gap-1">
        {items.map((item) => {
          const isSelected = item === selected;
          const colorClass = colorFn
            ? colorFn(item, isSelected)
            : isSelected
              ? "border-blue-500/50 bg-blue-500/10 text-blue-400"
              : "border-transparent bg-zinc-800/40 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300";
          return (
            <button
              key={item}
              onClick={() => onChange(item)}
              className={`rounded-md border px-2.5 py-1 text-xs font-medium transition-all duration-150 ${colorClass}`}
            >
              {DISPLAY_NAMES[item] ?? item.replace(/USDT$/, "")}
            </button>
          );
        })}
      </div>
    </div>
  );
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
    <div className="flex flex-wrap items-start gap-6 border-b border-[var(--border)] bg-[var(--surface)] px-6 py-3">
      <PillGroup
        label="Symbol"
        items={symbols}
        selected={selectedSymbol}
        onChange={onSymbolChange}
        colorFn={(_, isSelected) =>
          isSelected
            ? "border-yellow-500/40 bg-yellow-500/10 text-yellow-400"
            : "border-transparent bg-zinc-800/40 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300"
        }
      />
      <div className="hidden h-10 w-px self-center bg-zinc-800 sm:block" />
      <PillGroup
        label="Bar Type"
        items={barTypes}
        selected={selectedBarType}
        onChange={onBarTypeChange}
      />
      <div className="hidden h-10 w-px self-center bg-zinc-800 sm:block" />
      <PillGroup
        label="Labeling"
        items={labelingMethods}
        selected={selectedLabeling}
        onChange={onLabelingChange}
      />
    </div>
  );
}
