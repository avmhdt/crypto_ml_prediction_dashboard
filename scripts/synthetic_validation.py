"""Run synthetic signal validation — SNR sweep across the AFML pipeline.

Usage:
    python scripts/synthetic_validation.py
    python scripts/synthetic_validation.py --bar-type tick_imbalance --n-ticks 200000
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import BarConfig, TripleBarrierConfig, TrainingConfig
from backend.data.database import get_connection, init_schema, save_synth_result
from backend.synthetic.validation import run_synthetic_validation


def main():
    parser = argparse.ArgumentParser(description="Run synthetic signal validation")
    parser.add_argument("--bar-type", default="time", help="Bar type")
    parser.add_argument("--labeling", default="triple_barrier", help="Labeling method")
    parser.add_argument("--n-ticks", type=int, default=500_000)
    parser.add_argument("--sharpe-levels", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"  SYNTHETIC SIGNAL VALIDATION")
    print(f"  Bar type: {args.bar_type}")
    print(f"  Labeling: {args.labeling}")
    print(f"  Ticks: {args.n_ticks:,}")
    print(f"  Sharpe levels: {args.sharpe_levels}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = run_synthetic_validation(
        bar_type=args.bar_type,
        labeling_method=args.labeling,
        sharpe_levels=args.sharpe_levels,
        n_ticks=args.n_ticks,
    )
    elapsed = time.time() - t0

    conn = get_connection()
    init_schema(conn)
    run_id = save_synth_result(conn, result)
    conn.close()

    print(f"\n{'='*60}")
    print(f"  SYNTHETIC VALIDATION COMPLETE")
    print(f"  Run ID: {run_id}")
    print(f"  Detection threshold: Sharpe {result.detection_threshold}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    print(f"{'Sharpe':>8} {'OOS Acc':>8} {'Prec':>8} {'Recall':>8} {'Eq SR':>8} {'Trades':>7}")
    print(f"{'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*7:>7}")
    for pt in result.points:
        print(f"{pt.sharpe:>8.1f} {pt.oos_accuracy:>8.4f} {pt.oos_precision:>8.4f} "
              f"{pt.oos_recall:>8.4f} {pt.equity_sharpe:>8.2f} {pt.num_trades:>7}")

    print(f"\n  Detection threshold: Sharpe >= {result.detection_threshold} for >55% accuracy")


if __name__ == "__main__":
    main()
