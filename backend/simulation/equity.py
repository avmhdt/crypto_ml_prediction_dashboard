"""Theoretical equity curve simulation.

Simulates P&L from model signals + bars using AFML Chapter 10 methodology:
- Per-signal SL/PT/time barrier exits (triple barrier only)
- Trend scanning / directional change: time barrier exit only
- Portfolio-level bet size averaging across concurrent positions
- Total exposure capped at [-1, 1]
- Transaction costs (fees + slippage) in bps on entry and exit
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


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


def simulate_equity(
    signals_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    labeling: str,
    starting_capital: float = 10000.0,
    fees_bps: float = 10.0,
) -> SimulationResult:
    """Simulate equity curve from signals and bars.

    At each bar:
    1. Activate new signals whose timestamp <= bar timestamp
    2. Check exits: SL/PT/time for triple_barrier, time-only for others
    3. Compute avg exposure = (1/n) * sum(side_i * size_i), cap [-1, 1]
    4. P&L = prev_exposure * equity * bar_return
    5. Fees charged on |delta_exposure| * equity * fee_rate
    """
    if signals_df.empty or bars_df.empty:
        return SimulationResult()

    bars = bars_df.sort_values("timestamp").reset_index(drop=True)
    signals = signals_df.sort_values("timestamp").reset_index(drop=True)
    fee_rate = fees_bps / 10000.0

    # Pre-process signals into plain dicts (avoid iterrows overhead)
    sig_list = []
    for rec in signals.to_dict("records"):
        sig_list.append({
            "timestamp": int(rec["timestamp"]),
            "side": int(rec["side"]),
            "size": float(rec["size"]),
            "entry_price": float(rec["entry_price"]) if pd.notna(rec.get("entry_price")) else None,
            "sl_price": float(rec["sl_price"]) if pd.notna(rec.get("sl_price")) else None,
            "pt_price": float(rec["pt_price"]) if pd.notna(rec.get("pt_price")) else None,
            "time_barrier": int(rec["time_barrier"]) if pd.notna(rec.get("time_barrier")) else None,
            "labeling": str(rec.get("labeling_method", labeling)),
        })

    # Pre-convert bars to list of dicts (avoid iloc overhead in hot loop)
    bar_records = bars[["timestamp", "close", "high", "low"]].to_dict("records")

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

        # 1. Activate new signals
        while sig_ptr < len(sig_list) and sig_list[sig_ptr]["timestamp"] <= bar_ts:
            s = sig_list[sig_ptr]
            if s["entry_price"] is None:
                s["entry_price"] = bar_close
            active.append(s)
            sig_ptr += 1

        # 2. Check exits
        still_active = []
        for pos in active:
            exited = False
            exit_price = bar_close
            is_tb = pos["labeling"] == "triple_barrier"

            if is_tb:
                # Stop-loss
                if pos["sl_price"] is not None:
                    if pos["side"] == 1 and bar_low <= pos["sl_price"]:
                        exit_price = pos["sl_price"]
                        exited = True
                    elif pos["side"] == -1 and bar_high >= pos["sl_price"]:
                        exit_price = pos["sl_price"]
                        exited = True
                # Profit-take
                if not exited and pos["pt_price"] is not None:
                    if pos["side"] == 1 and bar_high >= pos["pt_price"]:
                        exit_price = pos["pt_price"]
                        exited = True
                    elif pos["side"] == -1 and bar_low <= pos["pt_price"]:
                        exit_price = pos["pt_price"]
                        exited = True

            # Time barrier (all methods)
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

        # 3. Compute exposure (AFML Ch.10 averaging)
        if active:
            raw = sum(p["side"] * p["size"] for p in active) / len(active)
            exposure = max(-1.0, min(1.0, raw))
        else:
            exposure = 0.0

        # 4. P&L from price movement on previous exposure
        if prev_close > 0 and bar_idx > 0:
            bar_return = (bar_close - prev_close) / prev_close
            pnl = prev_exposure * equity * bar_return
            equity += pnl

        # 5. Fees on position change
        delta = abs(exposure - prev_exposure)
        if delta > 1e-9:
            equity -= delta * equity * fee_rate

        # 6. Record
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0.0
        invested = abs(exposure) * equity

        ts_out.append(bar_ts)
        equity_out.append(round(equity, 2))
        dd_out.append(round(dd, 6))
        invested_out.append(round(invested, 2))

        prev_exposure = exposure
        prev_close = bar_close

    # Performance metrics — estimate annualization from actual bar frequency
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

    max_dd = float(min(dd_out)) * 100 if dd_out else 0.0
    total_return = (equity - starting_capital) / starting_capital * 100 if starting_capital > 0 else 0.0
    win_rate = num_wins / num_completed * 100 if num_completed > 0 else 0.0

    return SimulationResult(
        timestamps=ts_out,
        equity=equity_out,
        drawdown=dd_out,
        total_invested=invested_out,
        metrics={
            "sharpe": round(sharpe, 2),
            "max_dd": round(max_dd, 2),
            "total_return": round(total_return, 2),
            "win_rate": round(win_rate, 1),
            "num_trades": num_completed,
        },
    )
