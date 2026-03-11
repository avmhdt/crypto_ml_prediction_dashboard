"""CLI script for offline model training.

Usage:
    python scripts/train.py --symbol BTCUSDT --bar-type tick_imbalance --labeling triple_barrier
    python scripts/train.py --symbol BTCUSDT --bar-type tick_imbalance --labeling triple_barrier --data-dir "D:\Position.One\tick data"
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import BAR_TYPES, LABELING_METHODS, SYMBOLS, TICK_DATA_DIR
from backend.ml.training import train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train ML models for crypto prediction")
    parser.add_argument("--symbol", required=True, choices=SYMBOLS,
                        help="Trading pair symbol")
    parser.add_argument("--bar-type", required=True, choices=BAR_TYPES,
                        help="Bar type to generate")
    parser.add_argument("--labeling", required=True, choices=LABELING_METHODS,
                        help="Labeling method")
    parser.add_argument("--data-dir", type=str, default=str(TICK_DATA_DIR),
                        help="Path to tick data directory")
    parser.add_argument("--optuna-trials", type=int, default=100,
                        help="Number of Optuna optimization trials")
    parser.add_argument("--optuna-timeout", type=int, default=600,
                        help="Optuna timeout in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from backend.config import TrainingConfig
    training_config = TrainingConfig(
        optuna_n_trials=args.optuna_trials,
        optuna_timeout=args.optuna_timeout,
    )

    results = train_pipeline(
        symbol=args.symbol,
        bar_type=args.bar_type,
        labeling_method=args.labeling,
        data_dir=Path(args.data_dir),
        training_config=training_config,
    )

    print("\n=== Training Results ===")
    print(f"Symbol: {results['symbol']}")
    print(f"Bar Type: {results['bar_type']}")
    print(f"Labeling: {results['labeling_method']}")
    print(f"Trades: {results['num_trades']:,}")
    print(f"Bars: {results['num_bars']:,}")
    print(f"Samples: {results['num_samples']:,}")
    print(f"Features: {results['num_features']}")
    print(f"Primary Recall: {results['primary_recall']:.4f}")
    print(f"Meta Precision: {results['meta_precision']:.4f}")
    print(f"Bet Size Distribution: {results['bet_size_distribution']}")
    print(f"\nArtifacts saved to: {results['artifacts']['primary']}")


if __name__ == "__main__":
    main()
