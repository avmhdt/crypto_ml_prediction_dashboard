"use client";

interface HeaderProps {
  connected: boolean;
  symbol: string;
}

export function Header({ connected, symbol }: HeaderProps) {
  return (
    <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-950 px-6 py-3">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold text-white">
          Crypto ML Prediction Dashboard
        </h1>
        <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
          AFML
        </span>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-sm text-zinc-400">{symbol}</span>
        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-zinc-500">
            {connected ? "Live" : "Disconnected"}
          </span>
        </div>
      </div>
    </header>
  );
}
