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

export interface BBO {
  symbol: string;
  bid: number;
  bid_qty: number;
  ask: number;
  ask_qty: number;
  spread: number;
  mid: number;
  time: number;
}

export type WSMessage =
  | { type: "bar"; data: BarData }
  | { type: "signal"; data: Signal }
  | { type: "tick"; data: Tick }
  | { type: "bbo"; data: BBO };

export interface EquityMetrics {
  sharpe: number;
  max_dd: number;
  total_return: number;
  win_rate: number;
  num_trades: number;
}

export interface EquityData {
  timestamps: number[];
  equity: number[];
  drawdown: number[];
  total_invested: number[];
  metrics: EquityMetrics;
}

export interface SimulationConfig {
  mode: "simple" | "realistic" | "both";
  starting_capital: number;
  fees_bps: number;
  vip_tier: number;
  bnb_discount: boolean;
  urgency: number;
  order_timeout_ms: number;
}

export interface CostBreakdown {
  exchange_fee: number;
  funding_cost: number;
  spread_cost: number;
  slippage: number;
  market_impact: number;
  total: number;
}

export interface RealisticMetrics extends EquityMetrics {
  fill_rate: number;
  avg_slippage_bps: number;
  maker_ratio: number;
  avg_queue_wait_ms: number;
  funding_total: number;
  num_unfilled: number;
  cost_breakdown: CostBreakdown;
}

export interface EquityComparisonData {
  simple: EquityData;
  realistic: EquityData & { metrics: RealisticMetrics };
}
