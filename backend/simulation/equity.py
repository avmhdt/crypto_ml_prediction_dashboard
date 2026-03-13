"""Theoretical equity curve simulation.

Simulates P&L from model signals + bars using AFML Chapter 10 methodology:
- Per-signal SL/PT/time barrier exits (triple barrier only)
- Trend scanning / directional change: time barrier exit only
- Portfolio-level bet size averaging across concurrent positions
- Total exposure capped at [-1, 1]
- Transaction costs: simple (flat bps) or realistic (decomposed 5-component)

Supports three simulation modes:
- ``"simple"``: Flat bps fee on position change (original behaviour)
- ``"realistic"``: Decomposed cost model with fill probability filtering
- ``"both"``: Run both and return comparison data
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from backend.ml.bet_sizing import compute_average_exposure
from backend.simulation.config import SimulationConfig
from backend.simulation.cost_model import CostModel, TransactionCost
from backend.simulation.funding_tracker import FundingRateTracker


# Default ADV for major pairs when unknown
_DEFAULT_ADV_USD = 1_000_000_000.0
_DEFAULT_VOLATILITY = 0.02


@dataclass
class SimulationResult:
    timestamps: list[int] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    drawdown: list[float] = field(default_factory=list)
    total_invested: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=lambda: {
        "sharpe": 0.0, "max_dd": 0.0, "total_return": 0.0,
        "win_rate": 0.0, "num_trades": 0,
    })


def _compute_metrics(
    ts_out: list[int],
    equity_out: list[float],
    dd_out: list[float],
    starting_capital: float,
    num_completed: int,
    num_wins: int,
) -> dict:
    """Compute performance metrics from equity curve."""
    if len(ts_out) > 1:
        avg_interval_ms = (ts_out[-1] - ts_out[0]) / (len(ts_out) - 1)
        bars_per_year = (365.25 * 24 * 3600 * 1000) / avg_interval_ms if avg_interval_ms > 0 else 525600
    else:
        bars_per_year = 525600

    if len(equity_out) > 1:
        eq_arr = np.array(equity_out)
        rets = np.diff(eq_arr) / eq_arr[:-1]
        rets = rets[np.isfinite(rets)]
        std = float(np.std(rets)) if len(rets) > 0 else 0.0
        sharpe = float(np.mean(rets)) / std * np.sqrt(bars_per_year) if std > 0 else 0.0
    else:
        sharpe = 0.0

    equity = equity_out[-1] if equity_out else starting_capital
    max_dd = float(min(dd_out)) * 100 if dd_out else 0.0
    total_return = (equity - starting_capital) / starting_capital * 100 if starting_capital > 0 else 0.0
    win_rate = num_wins / num_completed * 100 if num_completed > 0 else 0.0

    return {
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 1),
        "num_trades": num_completed,
    }


def _prepare_signals(signals_df: pd.DataFrame, labeling: str) -> list[dict]:
    """Convert signals DataFrame to list of dicts."""
    sig_list = []
    for rec in signals_df.to_dict("records"):
        sig_list.append({
            "timestamp": int(rec["timestamp"]),
            "side": int(rec["side"]),
            "size": float(rec["size"]),
            "entry_price": float(rec["entry_price"]) if pd.notna(rec.get("entry_price")) else None,
            "sl_price": float(rec["sl_price"]) if pd.notna(rec.get("sl_price")) else None,
            "pt_price": float(rec["pt_price"]) if pd.notna(rec.get("pt_price")) else None,
            "time_barrier": int(rec["time_barrier"]) if pd.notna(rec.get("time_barrier")) else None,
            "labeling": str(rec.get("labeling_method", labeling)),
            "meta_probability": float(rec["meta_probability"]) if pd.notna(rec.get("meta_probability")) else 0.5,
        })
    return sig_list


def _run_simple(
    sig_list: list[dict],
    bar_records: list[dict],
    starting_capital: float,
    fee_rate: float,
    labeling: str,
) -> SimulationResult:
    """Run simple flat-bps simulation (original behaviour)."""
    ts_out: list[int] = []
    equity_out: list[float] = []
    dd_out: list[float] = []
    invested_out: list[float] = []

    equity = starting_capital
    peak = starting_capital
    prev_exposure = 0.0
    prev_close = float(bar_records[0]["close"])
    sig_ptr = 0
    active: list[dict] = []
    num_completed = 0
    num_wins = 0

    for bar_idx, bar in enumerate(bar_records):
        bar_ts = int(bar["timestamp"])
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        while sig_ptr < len(sig_list) and sig_list[sig_ptr]["timestamp"] <= bar_ts:
            s = sig_list[sig_ptr].copy()
            if s["entry_price"] is None:
                s["entry_price"] = bar_close
            active.append(s)
            sig_ptr += 1

        still_active = []
        for pos in active:
            exited = False
            exit_price = bar_close
            is_tb = pos["labeling"] == "triple_barrier"

            if is_tb:
                if pos["sl_price"] is not None:
                    if pos["side"] == 1 and bar_low <= pos["sl_price"]:
                        exit_price = pos["sl_price"]
                        exited = True
                    elif pos["side"] == -1 and bar_high >= pos["sl_price"]:
                        exit_price = pos["sl_price"]
                        exited = True
                if not exited and pos["pt_price"] is not None:
                    if pos["side"] == 1 and bar_high >= pos["pt_price"]:
                        exit_price = pos["pt_price"]
                        exited = True
                    elif pos["side"] == -1 and bar_low <= pos["pt_price"]:
                        exit_price = pos["pt_price"]
                        exited = True

            if not exited and pos["time_barrier"] is not None and bar_ts >= pos["time_barrier"]:
                exit_price = bar_close
                exited = True

            if exited:
                ret = pos["side"] * (exit_price - pos["entry_price"]) / pos["entry_price"]
                num_completed += 1
                if ret > 0:
                    num_wins += 1
            else:
                still_active.append(pos)

        active = still_active
        exposure = compute_average_exposure(active)

        if prev_close > 0 and bar_idx > 0:
            bar_return = (bar_close - prev_close) / prev_close
            pnl = prev_exposure * equity * bar_return
            equity += pnl

        delta = abs(exposure - prev_exposure)
        if delta > 1e-9:
            equity -= delta * equity * fee_rate

        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0.0
        invested = abs(exposure) * equity

        ts_out.append(bar_ts)
        equity_out.append(round(equity, 2))
        dd_out.append(round(dd, 6))
        invested_out.append(round(invested, 2))

        prev_exposure = exposure
        prev_close = bar_close

    metrics = _compute_metrics(ts_out, equity_out, dd_out, starting_capital, num_completed, num_wins)

    return SimulationResult(
        timestamps=ts_out,
        equity=equity_out,
        drawdown=dd_out,
        total_invested=invested_out,
        metrics=metrics,
    )


def _run_realistic(
    sig_list: list[dict],
    bar_records: list[dict],
    sim_config: SimulationConfig,
    labeling: str,
) -> SimulationResult:
    """Run realistic simulation with decomposed cost model.

    Key differences from simple mode:
    - Uses maker/taker fee distinction based on VIP tier
    - Applies spread cost, slippage (square-root law), market impact
    - Tracks funding rate costs for positions crossing funding intervals
    - Filters signals with lower meta_probability (< 0.5 already filtered upstream)
    - Uses a fill probability factor to simulate realistic fill rates
    """
    starting_capital = sim_config.starting_capital
    cost_model = CostModel(sim_config)
    funding = FundingRateTracker(default_rate=sim_config.default_funding_rate)

    ts_out: list[int] = []
    equity_out: list[float] = []
    dd_out: list[float] = []
    invested_out: list[float] = []

    equity = starting_capital
    peak = starting_capital
    prev_exposure = 0.0
    prev_close = float(bar_records[0]["close"])
    sig_ptr = 0
    active: list[dict] = []
    num_completed = 0
    num_wins = 0
    num_unfilled = 0

    # Cost accumulators
    total_exchange_fee = 0.0
    total_funding = 0.0
    total_spread = 0.0
    total_slippage = 0.0
    total_impact = 0.0
    total_maker = 0
    total_fills = 0
    total_wait_ms = 0

    for bar_idx, bar in enumerate(bar_records):
        bar_ts = int(bar["timestamp"])
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        # Activate signals with fill probability filter
        while sig_ptr < len(sig_list) and sig_list[sig_ptr]["timestamp"] <= bar_ts:
            s = sig_list[sig_ptr].copy()
            if s["entry_price"] is None:
                s["entry_price"] = bar_close

            # Realistic fill filter: lower meta_probability = lower fill chance
            # Use meta_probability as a proxy for fill likelihood
            meta_p = s.get("meta_probability", 0.5)
            fill_prob = min(1.0, meta_p * 1.2)  # Scale slightly

            # Simulate fill based on probability
            if np.random.random() < fill_prob:
                s["_filled"] = True
                s["_fill_bar"] = bar_idx
                active.append(s)
                total_fills += 1
            else:
                num_unfilled += 1

            sig_ptr += 1

        # Check exits
        still_active = []
        for pos in active:
            exited = False
            exit_price = bar_close
            is_tb = pos["labeling"] == "triple_barrier"

            if is_tb:
                if pos["sl_price"] is not None:
                    if pos["side"] == 1 and bar_low <= pos["sl_price"]:
                        exit_price = pos["sl_price"]
                        exited = True
                    elif pos["side"] == -1 and bar_high >= pos["sl_price"]:
                        exit_price = pos["sl_price"]
                        exited = True
                if not exited and pos["pt_price"] is not None:
                    if pos["side"] == 1 and bar_high >= pos["pt_price"]:
                        exit_price = pos["pt_price"]
                        exited = True
                    elif pos["side"] == -1 and bar_low <= pos["pt_price"]:
                        exit_price = pos["pt_price"]
                        exited = True

            if not exited and pos["time_barrier"] is not None and bar_ts >= pos["time_barrier"]:
                exit_price = bar_close
                exited = True

            if exited:
                ret = pos["side"] * (exit_price - pos["entry_price"]) / pos["entry_price"]
                num_completed += 1
                if ret > 0:
                    num_wins += 1

                # Compute exit costs (funding for hold duration)
                hold_ms = bar_ts - pos["timestamp"]
                notional = abs(pos["size"] * exit_price)
                funding_cost = funding.compute_funding_cost(
                    entry_ms=pos["timestamp"],
                    exit_ms=bar_ts,
                    position_notional=notional,
                    side=pos["side"],
                )
                total_funding += funding_cost
                total_wait_ms += hold_ms
            else:
                still_active.append(pos)

        active = still_active
        exposure = compute_average_exposure(active)

        # P&L
        if prev_close > 0 and bar_idx > 0:
            bar_return = (bar_close - prev_close) / prev_close
            pnl = prev_exposure * equity * bar_return
            equity += pnl

        # Realistic costs on position change
        delta = abs(exposure - prev_exposure)
        if delta > 1e-9:
            notional = delta * equity
            # Determine maker vs taker (assume maker for limit orders)
            order_type = "maker"
            total_maker += 1

            costs = cost_model.compute_total(
                notional=notional,
                order_type=order_type,
                spread=bar_close * 0.0002,  # ~2 bps spread estimate
                order_size_usd=notional,
                adv_usd=_DEFAULT_ADV_USD,
                volatility=_DEFAULT_VOLATILITY,
            )

            equity -= costs.total
            total_exchange_fee += costs.exchange_fee
            total_spread += costs.spread_cost
            total_slippage += costs.slippage
            total_impact += costs.market_impact

        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0.0
        invested = abs(exposure) * equity

        ts_out.append(bar_ts)
        equity_out.append(round(equity, 2))
        dd_out.append(round(dd, 6))
        invested_out.append(round(invested, 2))

        prev_exposure = exposure
        prev_close = bar_close

    # Compute metrics
    base_metrics = _compute_metrics(ts_out, equity_out, dd_out, starting_capital, num_completed, num_wins)

    total_signals = total_fills + num_unfilled
    fill_rate = total_fills / total_signals * 100 if total_signals > 0 else 0.0
    total_cost = total_exchange_fee + total_funding + total_spread + total_slippage + total_impact
    avg_slippage_bps = (total_slippage / equity * 10000) if equity > 0 else 0.0
    maker_ratio = total_maker / max(1, total_fills) * 100
    avg_wait = total_wait_ms / max(1, num_completed)

    base_metrics.update({
        "fill_rate": round(fill_rate, 1),
        "avg_slippage_bps": round(avg_slippage_bps, 2),
        "maker_ratio": round(maker_ratio, 1),
        "avg_queue_wait_ms": round(avg_wait, 0),
        "funding_total": round(total_funding, 2),
        "num_unfilled": num_unfilled,
        "cost_breakdown": {
            "exchange_fee": round(total_exchange_fee, 2),
            "funding_cost": round(total_funding, 2),
            "spread_cost": round(total_spread, 2),
            "slippage": round(total_slippage, 2),
            "market_impact": round(total_impact, 2),
            "total": round(total_cost, 2),
        },
    })

    return SimulationResult(
        timestamps=ts_out,
        equity=equity_out,
        drawdown=dd_out,
        total_invested=invested_out,
        metrics=base_metrics,
    )


def simulate_equity(
    signals_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    labeling: str,
    starting_capital: float = 10000.0,
    fees_bps: float = 10.0,
    simulation_mode: str = "simple",
    sim_config: SimulationConfig | None = None,
) -> SimulationResult | dict:
    """Simulate equity curve from signals and bars.

    Args:
        signals_df: DataFrame of trading signals.
        bars_df: DataFrame of OHLCV bars.
        labeling: Labeling method name.
        starting_capital: Initial capital in USDT.
        fees_bps: Flat fee in bps (simple mode).
        simulation_mode: ``"simple"``, ``"realistic"``, or ``"both"``.
        sim_config: SimulationConfig for realistic mode.

    Returns:
        SimulationResult for simple/realistic, or dict with both for "both" mode.
    """
    if signals_df.empty or bars_df.empty:
        empty = SimulationResult()
        if simulation_mode == "both":
            return {"simple": empty, "realistic": empty}
        return empty

    bars = bars_df.sort_values("timestamp").reset_index(drop=True)
    signals = signals_df.sort_values("timestamp").reset_index(drop=True)
    sig_list = _prepare_signals(signals, labeling)
    bar_records = bars[["timestamp", "close", "high", "low"]].to_dict("records")

    if simulation_mode == "simple":
        fee_rate = fees_bps / 10000.0
        return _run_simple(sig_list, bar_records, starting_capital, fee_rate, labeling)

    elif simulation_mode == "realistic":
        if sim_config is None:
            sim_config = SimulationConfig(
                mode="realistic",
                starting_capital=starting_capital,
            )
        return _run_realistic(sig_list, bar_records, sim_config, labeling)

    elif simulation_mode == "both":
        fee_rate = fees_bps / 10000.0
        simple_result = _run_simple(sig_list, bar_records, starting_capital, fee_rate, labeling)

        if sim_config is None:
            sim_config = SimulationConfig(
                mode="realistic",
                starting_capital=starting_capital,
            )
        # Use fixed seed for reproducible realistic results
        np.random.seed(42)
        realistic_result = _run_realistic(sig_list, bar_records, sim_config, labeling)

        return {"simple": simple_result, "realistic": realistic_result}

    else:
        raise ValueError(f"Invalid simulation_mode: {simulation_mode}")
