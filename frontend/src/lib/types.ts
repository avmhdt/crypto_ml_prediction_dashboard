export interface BarData {
  symbol: string;
  bar_type: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  dollar_volume: number;
  tick_count: number;
  duration_us: number;
}

export interface Signal {
  id: number;
  symbol: string;
  bar_type: string;
  labeling_method: string;
  timestamp: number;
  side: -1 | 1;
  size: number;
  entry_price: number;
  sl_price: number | null;
  pt_price: number | null;
  time_barrier: number | null;
  meta_probability: number;
}

export interface Tick {
  price: number;
  qty: number;
  time: number;
  is_buyer_maker: boolean;
}

export interface DashboardConfig {
  bar_types: string[];
  labeling_methods: string[];
  symbols: string[];
}

export interface Metrics {
  total_signals: number;
  long_signals: number;
  short_signals: number;
  avg_meta_prob: number;
  avg_bet_size: number;
}

export type WSMessage =
  | { type: "bar"; data: BarData }
  | { type: "signal"; data: Signal }
  | { type: "tick"; data: Tick };
