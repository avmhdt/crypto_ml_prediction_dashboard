"""Walk-forward validation script: rolling train/test evaluation.

Usage:
    python scripts/walk_forward.py
    python scripts/walk_forward.py --symbol BTCUSDT --bar-type time --labeling triple_barrier
    python scripts/walk_forward.py --train-days 90 --test-days 30 --step-days 30
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import TICK_DATA_DIR, BarConfig, TripleBarrierConfig, TrainingConfig
from backend.data.database import get_connection, init_schema, save_wf_result
from backend.ml.walk_forward import run_walk_forward


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol")
    parser.add_argument("--bar-type", default="time", help="Bar type")
    parser.add_argument("--labeling", default="triple_barrier", help="Labeling method")
    parser.add_argument("--train-days", type=int, default=90, help="Training window (days)")
    parser.add_argument("--test-days", type=int, default=30, help="Test window (days)")
    parser.add_argument("--step-days", type=int, default=30, help="Step size (days)")
    parser.add_argument("--starting-capital", type=float, default=10000.0)
    parser.add_argument("--fees-bps", type=float, default=10.0)
    parser.add_argument("--data-dir", type=str, default=str(TICK_DATA_DIR))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Symbol: {args.symbol}")
    print(f"  Bar type: {args.bar_type}")
    print(f"  Labeling: {args.labeling}")
    print(f"  Windows: {args.train_days}d train / {args.test_days}d test / {args.step_days}d step")
    print(f"  Capital: ${args.starting_capital:,.0f}  Fees: {args.fees_bps} bps")
    print(f"{'='*60}\n")

    # No Optuna per window — use default hyperparams for speed
    training_config = TrainingConfig(optuna_n_trials=0, optuna_timeout=0)
    bar_config = BarConfig()
    barrier_config = TripleBarrierConfig()

    t0 = time.time()
    result = run_walk_forward(
        symbol=args.symbol,
        bar_type=args.bar_type,
        labeling_method=args.labeling,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        data_dir=Path(args.data_dir),
        bar_config=bar_config,
        barrier_config=barrier_config,
        training_config=training_config,
        starting_capital=args.starting_capital,
        fees_bps=args.fees_bps,
    )
    elapsed = time.time() - t0

    # Save to DB
    conn = get_connection()
    init_schema(conn)
    run_id = save_wf_result(conn, result)
    conn.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD COMPLETE")
    print(f"  Run ID: {run_id}")
    print(f"  Windows: {result.num_windows}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(result.num_windows,1):.1f}s/window)")
    print(f"{'='*60}\n")

    print(f"{'Window':>6} {'OOS Acc':>8} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Trades':>7}")
    print(f"{'-'*6:>6} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*7:>7}")
    for w in result.windows:
        print(f"{w.window_index:>6} {w.oos_accuracy:>8.4f} {w.sharpe:>8.2f} "
              f"{w.total_return:>7.2f}% {w.max_dd:>7.2f}% {w.num_trades:>7}")

    print(f"\n  Aggregate (mean ± std, 95% CI):")
    for metric, stats in result.aggregate.items():
        print(f"    {metric:<20} {stats['mean']:>8.4f} ± {stats['std']:.4f}  "
              f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")

    print(f"\n  In-sample recall (avg): {result.avg_insample_recall:.4f}")
    print(f"  OOS accuracy (avg):    {result.avg_oos_accuracy:.4f}")
    print(f"  Overfitting gap:       {result.overfitting_gap:.4f}")
    print(f"\n  Saved to DB as run_id={run_id}")


if __name__ == "__main__":
    main()
