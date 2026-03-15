"""Run synthetic validation for ALL bar type × labeling method combinations.

Populates the dashboard's Synthetic panel with actual pipeline results.

Usage:
    python scripts/run_all_synthetic.py
    python scripts/run_all_synthetic.py --n-ticks 200000  # faster
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import BAR_TYPES, LABELING_METHODS
from backend.data.database import get_connection, init_schema, save_synth_result
from backend.synthetic.validation import run_synthetic_validation

SHARPE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def main():
    parser = argparse.ArgumentParser(description="Run synthetic validation for all combos")
    parser.add_argument("--n-ticks", type=int, default=500_000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

    combos = [(bt, lm) for bt in BAR_TYPES for lm in LABELING_METHODS]
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"  SYNTHETIC VALIDATION — ALL COMBOS")
    print(f"  {total} combinations ({len(BAR_TYPES)} bar types × {len(LABELING_METHODS)} labeling methods)")
    print(f"  Ticks per run: {args.n_ticks:,}")
    print(f"  Sharpe levels: {SHARPE_LEVELS}")
    print(f"{'='*60}\n")

    conn = get_connection()
    init_schema(conn)

    ok_count = 0
    fail_count = 0
    t_total = time.time()

    for idx, (bar_type, labeling) in enumerate(combos, 1):
        print(f"[{idx:2d}/{total}] {bar_type} / {labeling} ... ", end="", flush=True)
        t0 = time.time()
        try:
            result = run_synthetic_validation(
                bar_type=bar_type,
                labeling_method=labeling,
                sharpe_levels=SHARPE_LEVELS,
                n_ticks=args.n_ticks,
            )
            run_id = save_synth_result(conn, result)
            elapsed = time.time() - t0
            print(
                f"OK  run_id={run_id:3d}  "
                f"points={len(result.points)}/7  "
                f"threshold={result.detection_threshold}  "
                f"({elapsed:.1f}s)"
            )
            ok_count += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAIL  ({elapsed:.1f}s) — {e}")
            fail_count += 1

    conn.close()
    total_elapsed = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"  COMPLETE: {ok_count}/{total} succeeded, {fail_count}/{total} failed")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
