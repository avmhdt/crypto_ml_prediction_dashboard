"""Seed synthetic signal validation results for the portfolio demo.

Generates plausible synthetic validation results so the dashboard's
Synthetic tab looks populated without running the full pipeline.

Usage:
    python scripts/seed_synthetic.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.database import get_connection, init_schema, save_synth_result
from backend.synthetic.validation import (
    SyntheticPointResult,
    SyntheticValidationResult,
    compute_detection_threshold,
)


def _generate_plausible_points(
    sharpe_levels: list[float],
    seed: int = 42,
) -> list[SyntheticPointResult]:
    """Generate plausible synthetic validation points.

    The recovery curve should show:
    - Sharpe 0: ~50% accuracy (coin flip)
    - Sharpe 0.5: ~51-52% (barely above chance)
    - Sharpe 1.0: ~53-55% (weak detection)
    - Sharpe 1.5: ~56-60% (moderate)
    - Sharpe 2.0: ~60-65% (clear detection)
    - Sharpe 3.0: ~68-75% (strong)
    - Sharpe 5.0: ~78-85% (very strong)
    """
    rng = np.random.RandomState(seed)

    # Sigmoid-like accuracy curve
    def accuracy_at_sharpe(s):
        # Logistic curve centered at sharpe=1.5, steepness=1.2
        base = 0.50
        max_gain = 0.38  # max accuracy = 88%
        return base + max_gain / (1 + np.exp(-1.2 * (s - 1.5)))

    points = []
    for s in sharpe_levels:
        acc = accuracy_at_sharpe(s) + rng.uniform(-0.015, 0.015)
        acc = max(0.45, min(0.95, acc))

        # Precision and recall track accuracy roughly
        prec = acc + rng.uniform(-0.03, 0.03)
        rec = acc + rng.uniform(-0.03, 0.03)

        # Equity Sharpe scales with accuracy
        eq_sharpe = (acc - 0.5) * 8 + rng.uniform(-0.3, 0.3)

        # Return and drawdown
        total_return = eq_sharpe * 3 + rng.uniform(-2, 2)
        max_dd = -abs(rng.uniform(2, 15) - eq_sharpe * 2)
        win_rate = acc * 100 + rng.uniform(-3, 3)
        num_trades = int(rng.uniform(30, 120))

        points.append(SyntheticPointResult(
            sharpe=s,
            oos_accuracy=round(acc, 4),
            oos_precision=round(max(0.3, min(0.95, prec)), 4),
            oos_recall=round(max(0.3, min(0.95, rec)), 4),
            equity_sharpe=round(eq_sharpe, 2),
            total_return=round(total_return, 2),
            max_dd=round(max_dd, 2),
            win_rate=round(max(30, min(95, win_rate)), 1),
            num_trades=num_trades,
            num_bars=int(rng.uniform(400, 800)),
            num_samples=int(rng.uniform(350, 750)),
        ))

    return points


def main():
    print("Seeding synthetic signal validation results...")

    combos = [
        ("time", "triple_barrier"),
        ("tick_imbalance", "triple_barrier"),
        ("dollar", "triple_barrier"),
    ]

    sharpe_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    conn = get_connection()
    init_schema(conn)

    for bar_type, labeling in combos:
        print(f"  Generating: {bar_type} / {labeling}")

        seed = hash(f"synth_{bar_type}_{labeling}") % (2**31)
        points = _generate_plausible_points(sharpe_levels, seed=seed)
        threshold = compute_detection_threshold(points)

        result = SyntheticValidationResult(
            bar_type=bar_type,
            labeling_method=labeling,
            n_ticks=500_000,
            sharpe_levels=sharpe_levels,
            points=points,
            detection_threshold=threshold,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        run_id = save_synth_result(conn, result)
        print(f"    Saved as run_id={run_id} ({len(points)} points, threshold={threshold})")

    conn.close()
    print(f"\nDone! Seeded {len(combos)} synthetic validation runs.")


if __name__ == "__main__":
    main()
