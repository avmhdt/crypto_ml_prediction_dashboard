"""Batch training script: trains all bar_type × labeling_method combinations.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --days 7 --trials 10 --timeout 120
"""
import argparse
import logging
import re
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import BAR_TYPES, LABELING_METHODS, TICK_DATA_DIR, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train all model combinations")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to train")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of recent days of data to use")
    parser.add_argument("--trials", type=int, default=10,
                        help="Optuna trials per combination")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Optuna timeout per combination (seconds)")
    parser.add_argument("--bar-types", nargs="+", default=None,
                        help="Specific bar types to train (default: all)")
    parser.add_argument("--labeling-methods", nargs="+", default=None,
                        help="Specific labeling methods (default: all)")
    parser.add_argument("--data-dir", type=str, default=str(TICK_DATA_DIR))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

    bar_types = args.bar_types or BAR_TYPES
    labeling_methods = args.labeling_methods or LABELING_METHODS
    total = len(bar_types) * len(labeling_methods)

    print(f"\n{'='*60}")
    print(f"  BATCH TRAINING: {args.symbol}")
    print(f"  Data: last {args.days} days from {args.data_dir}")
    print(f"  Bar types: {len(bar_types)}")
    print(f"  Labeling methods: {len(labeling_methods)}")
    print(f"  Combinations: {total}")
    print(f"  Optuna: {args.trials} trials, {args.timeout}s timeout each")
    print(f"{'='*60}\n")

    # Compute time range — scan the data directory for the actual date range
    # Check both aggTrades (tick_data/) and trades (trades_data/) directories
    all_csvs = []
    for sub in ["tick_data", "trades_data"]:
        data_path = Path(args.data_dir) / sub / "futures_um" / args.symbol
        if data_path.exists():
            all_csvs.extend(sorted(data_path.glob("*.csv")))
    # Extract dates from filenames and find the full range
    all_dates = []
    for f in all_csvs:
        m = _DATE_PATTERN.search(f.name)
        if m:
            all_dates.append(datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc))
    if all_dates:
        data_start = min(all_dates)
        data_end = max(all_dates)
        end_ms = int((data_end + timedelta(days=1)).timestamp() * 1000)
        if args.days >= (data_end - data_start).days:
            start_ms = int(data_start.timestamp() * 1000)
        else:
            start_ms = int((data_end - timedelta(days=args.days - 1)).timestamp() * 1000)
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    print(f"Time range: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()}")

    from backend.data.csv_reader import iter_trade_files
    n_files = len(iter_trade_files(args.symbol, Path(args.data_dir), start_ms, end_ms))
    print(f"Data files: {n_files} CSVs (streaming, ~40 MB peak per file)\n")

    if n_files == 0:
        print("ERROR: No data files found. Check data path and time range.")
        sys.exit(1)

    # Train each combination — trades are streamed per-combo, not pre-loaded
    from backend.ml.training import train_pipeline
    from backend.config import BarConfig, TripleBarrierConfig, MODELS_DIR

    training_config = TrainingConfig(
        optuna_n_trials=args.trials,
        optuna_timeout=args.timeout,
    )
    bar_config = BarConfig()
    barrier_config = TripleBarrierConfig()

    results_all = []
    completed = 0
    failed = 0

    for bar_type in bar_types:
        for labeling in labeling_methods:
            completed += 1
            combo = f"{bar_type}/{labeling}"
            print(f"\n[{completed}/{total}] Training: {combo}")
            print(f"{'-'*50}")

            t1 = time.time()
            try:
                result = train_pipeline(
                    symbol=args.symbol,
                    bar_type=bar_type,
                    labeling_method=labeling,
                    data_dir=Path(args.data_dir),
                    bar_config=bar_config,
                    barrier_config=barrier_config,
                    training_config=training_config,
                    trades=None,  # stream from disk per-combo
                )
                elapsed = time.time() - t1
                result["elapsed_seconds"] = elapsed
                results_all.append(result)

                print(f"  Bars: {result['num_bars']:,} | "
                      f"Samples: {result['num_samples']:,} | "
                      f"Features: {result['num_features']}")
                print(f"  Recall: {result['primary_recall']:.4f} | "
                      f"Precision: {result['meta_precision']:.4f}")
                print(f"  Time: {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - t1
                failed += 1
                print(f"  FAILED ({elapsed:.1f}s): {e}")
                results_all.append({
                    "symbol": args.symbol,
                    "bar_type": bar_type,
                    "labeling_method": labeling,
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                })

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Succeeded: {total - failed}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"{'='*60}\n")

    successes = [r for r in results_all if "error" not in r]
    if successes:
        print(f"{'Bar Type':<20} {'Labeling':<22} {'Recall':>8} {'Precision':>10} {'Time':>8}")
        print(f"{'-'*20} {'-'*22} {'-'*8} {'-'*10} {'-'*8}")
        for r in successes:
            print(f"{r['bar_type']:<20} {r['labeling_method']:<22} "
                  f"{r['primary_recall']:>8.4f} {r['meta_precision']:>10.4f} "
                  f"{r['elapsed_seconds']:>7.1f}s")

    # Save results
    results_path = MODELS_DIR / "training_results.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
