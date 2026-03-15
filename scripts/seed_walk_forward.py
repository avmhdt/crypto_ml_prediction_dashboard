"""Seed synthetic walk-forward results for the portfolio demo.

Generates plausible WF results so the dashboard looks populated without
needing to run actual training on the D: drive tick data.

Usage:
    python scripts/seed_walk_forward.py
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.database import get_connection, init_schema, save_wf_result
from backend.ml.walk_forward import WindowResult, WalkForwardResult, stitch_equity_curves, bootstrap_aggregate


def _generate_synthetic_equity(
    n_points: int, starting_capital: float, daily_return_mean: float,
    daily_volatility: float, rng: np.random.RandomState,
) -> tuple[list[float], list[float]]:
    """Generate a realistic-looking equity curve with noise."""
    returns = rng.normal(daily_return_mean, daily_volatility, n_points)
    equity = [starting_capital]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = equity[1:]  # remove initial capital duplicate

    # Compute drawdown
    peak = starting_capital
    drawdown = []
    for e in equity:
        peak = max(peak, e)
        dd = (e - peak) / peak if peak > 0 else 0.0
        drawdown.append(round(dd, 6))

    return [round(e, 2) for e in equity], drawdown


def _generate_synthetic_windows(
    num_windows: int = 24,
    train_days: int = 90,
    test_days: int = 30,
    starting_capital: float = 10000.0,
    seed: int = 42,
) -> list[WindowResult]:
    """Generate plausible synthetic window results."""
    rng = np.random.RandomState(seed)

    # Start from 2024-01-01
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ms_per_day = 86_400_000
    points_per_window = 200  # ~200 equity points per 30-day test window

    windows = []
    current_capital = starting_capital

    for i in range(num_windows):
        step_offset = i * test_days
        train_start = base_date + timedelta(days=step_offset)
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)

        train_start_ms = int(train_start.timestamp() * 1000)
        train_end_ms = int(train_end.timestamp() * 1000)
        test_start_ms = int(test_start.timestamp() * 1000)
        test_end_ms = int(test_end.timestamp() * 1000)

        # Generate timestamps spread across test period
        timestamps = np.linspace(test_start_ms, test_end_ms, points_per_window, dtype=int).tolist()

        # Slightly negative drift (honest — crypto is hard to predict)
        daily_return = rng.uniform(-0.002, 0.001)
        daily_vol = rng.uniform(0.008, 0.015)

        equity, drawdown = _generate_synthetic_equity(
            points_per_window, current_capital, daily_return, daily_vol, rng,
        )

        # OOS metrics: realistic range for an AFML pipeline on crypto
        oos_accuracy = rng.uniform(0.48, 0.54)
        oos_precision = rng.uniform(0.46, 0.55)
        oos_recall = rng.uniform(0.45, 0.58)
        primary_recall = rng.uniform(0.55, 0.75)  # in-sample is higher (overfitting)
        meta_precision = rng.uniform(0.50, 0.65)
        sharpe = rng.uniform(-0.5, 1.0)
        total_return = ((equity[-1] - current_capital) / current_capital) * 100
        max_dd = min(drawdown) * 100 if drawdown else 0.0
        win_rate = rng.uniform(0.42, 0.55)
        num_trades = int(rng.uniform(15, 80))
        num_train_bars = int(rng.uniform(800, 2000))
        num_test_bars = int(rng.uniform(200, 600))

        windows.append(WindowResult(
            window_index=i,
            train_start=train_start_ms,
            train_end=train_end_ms,
            test_start=test_start_ms,
            test_end=test_end_ms,
            num_train_bars=num_train_bars,
            num_test_bars=num_test_bars,
            num_train_samples=int(num_train_bars * 0.9),
            num_test_signals=num_trades,
            primary_recall=round(primary_recall, 4),
            meta_precision=round(meta_precision, 4),
            oos_accuracy=round(oos_accuracy, 4),
            oos_precision=round(oos_precision, 4),
            oos_recall=round(oos_recall, 4),
            sharpe=round(sharpe, 2),
            max_dd=round(max_dd, 2),
            total_return=round(total_return, 2),
            win_rate=round(win_rate * 100, 1),
            num_trades=num_trades,
            timestamps=timestamps,
            equity=equity,
            drawdown=drawdown,
        ))

        # Next window starts at this window's ending capital
        current_capital = equity[-1]

    return windows


def main():
    print("Seeding synthetic walk-forward results...")

    combos = [
        ("BTCUSDT", "time", "triple_barrier"),
        ("BTCUSDT", "tick", "trend_scanning"),
        ("ETHUSDT", "time", "triple_barrier"),
    ]

    conn = get_connection()
    init_schema(conn)

    for symbol, bar_type, labeling in combos:
        print(f"  Generating: {symbol} / {bar_type} / {labeling}")

        windows = _generate_synthetic_windows(
            num_windows=24,
            train_days=90,
            test_days=30,
            starting_capital=10000.0,
            seed=hash(f"{symbol}_{bar_type}_{labeling}") % (2**31),
        )

        stitched_ts, stitched_eq, stitched_dd = stitch_equity_curves(windows, 10000.0)
        aggregate = bootstrap_aggregate(windows)

        avg_insample = float(np.mean([w.primary_recall for w in windows]))
        avg_oos = float(np.mean([w.oos_accuracy for w in windows]))

        result = WalkForwardResult(
            symbol=symbol,
            bar_type=bar_type,
            labeling_method=labeling,
            train_days=90,
            test_days=30,
            step_days=30,
            num_windows=len(windows),
            windows=windows,
            stitched_timestamps=stitched_ts,
            stitched_equity=stitched_eq,
            stitched_drawdown=stitched_dd,
            aggregate=aggregate,
            avg_insample_recall=round(avg_insample, 4),
            avg_oos_accuracy=round(avg_oos, 4),
            overfitting_gap=round(avg_insample - avg_oos, 4),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        run_id = save_wf_result(conn, result)
        print(f"    Saved as run_id={run_id} ({len(windows)} windows)")

    conn.close()
    print(f"\nDone! Seeded {len(combos)} walk-forward runs.")


if __name__ == "__main__":
    main()
