"""Seed the dashboard DuckDB with historical bars and synthetic signals.

Generates bars from CSV trade data and runs ML inference to produce
real signals, giving the dashboard chart and table data to display.

Usage:
    python scripts/seed_dashboard.py
    python scripts/seed_dashboard.py --days 3 --bar-types time tick volume
"""
import argparse
import logging
import sys
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import (
    BAR_TYPES, LABELING_METHODS, SYMBOLS, TICK_DATA_DIR,
    BarConfig, TripleBarrierConfig, DB_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Seed dashboard with historical data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=3,
                        help="Days of recent data to seed")
    parser.add_argument("--bar-types", nargs="+",
                        default=["time", "tick", "volume", "dollar"],
                        help="Bar types to generate")
    parser.add_argument("--data-dir", type=str, default=str(TICK_DATA_DIR))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

    # Time range
    end_ms = int(datetime(2025, 7, 3, tzinfo=timezone.utc).timestamp() * 1000)
    start_ms = int((datetime(2025, 7, 3, tzinfo=timezone.utc) -
                     timedelta(days=args.days)).timestamp() * 1000)

    print(f"Seeding dashboard DB: {DB_PATH}")
    print(f"Symbol: {args.symbol}")
    print(f"Bar types: {args.bar_types}")
    print(f"Time range: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()}")

    # Load trades
    from backend.data.csv_reader import read_trades_for_symbol
    print(f"\nLoading trades...")
    trades = read_trades_for_symbol(
        args.symbol, Path(args.data_dir),
        start_time=start_ms, end_time=end_ms,
    )
    if trades.empty:
        print("ERROR: No trades found. Check data path.")
        sys.exit(1)
    print(f"Loaded {len(trades):,} trades")

    # Connect to dashboard DuckDB
    import duckdb
    from backend.data.database import init_schema
    conn = duckdb.connect(str(DB_PATH))
    init_schema(conn)

    # Generate bars for each type and insert
    from backend.ml.training import generate_bars, generate_labels
    from backend.ml.primary_model import PrimaryModel
    from backend.ml.meta_labeling import MetaLabelingModel
    from backend.ml.bet_sizing import compute_bet_sizes
    from backend.features import compute_all_features

    bar_config = BarConfig()
    barrier_config = TripleBarrierConfig()

    for bar_type in args.bar_types:
        print(f"\n--- Generating {bar_type} bars ---")
        try:
            bars_df = generate_bars(trades, args.symbol, bar_type, bar_config)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if bars_df.empty or len(bars_df) < 10:
            print(f"  Skipped: only {len(bars_df)} bars")
            continue
        print(f"  Generated {len(bars_df):,} bars")

        # Insert bars into DuckDB
        inserted = 0
        for _, row in bars_df.iterrows():
            try:
                conn.execute(
                    """INSERT INTO bars (symbol, bar_type, timestamp, open, high, low,
                       close, volume, dollar_volume, tick_count, duration_us)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        args.symbol, bar_type,
                        int(row["timestamp"]),
                        float(row["open"]), float(row["high"]),
                        float(row["low"]), float(row["close"]),
                        float(row["volume"]),
                        float(row.get("dollar_volume", row["volume"] * row["close"])),
                        int(row.get("tick_count", 0)),
                        int(row.get("duration_us", 0)),
                    ],
                )
                inserted += 1
            except Exception:
                pass
        print(f"  Inserted {inserted:,} bars into DB")

        # Try to generate signals using trained models if available
        _try_seed_signals(
            conn, bars_df, args.symbol, bar_type,
            barrier_config, bar_config,
        )

    conn.close()
    print(f"\nDone! Dashboard DB seeded at {DB_PATH}")
    print("Restart the backend server to pick up the new data.")


def _try_seed_signals(conn, bars_df, symbol, bar_type, barrier_config, bar_config):
    """Try to generate signals from trained models and insert into DB."""
    from backend.config import MODELS_DIR, LABELING_METHODS

    for labeling in LABELING_METHODS:
        prefix = f"{symbol}_{bar_type}_{labeling}"
        primary_path = MODELS_DIR / f"{prefix}_primary.joblib"
        secondary_path = MODELS_DIR / f"{prefix}_secondary.joblib"
        scaler_path = MODELS_DIR / f"{prefix}_scaler.joblib"
        features_path = MODELS_DIR / f"{prefix}_features.json"

        if not primary_path.exists():
            continue

        print(f"  Generating {labeling} signals from trained model...")
        try:
            from backend.ml.primary_model import PrimaryModel
            from backend.ml.meta_labeling import MetaLabelingModel
            from backend.ml.bet_sizing import compute_bet_sizes
            from backend.features import compute_all_features

            # Compute features
            features = compute_all_features(bars_df, window=20)
            features = features.ffill()
            valid = features.notna().all(axis=1)
            features = features[valid]
            valid_bars = bars_df.loc[features.index]

            if len(features) < 10:
                continue

            # Load and run primary model
            primary = PrimaryModel()
            primary.load(primary_path, scaler_path, features_path)
            preds = primary.predict(features)

            # Load and run meta model
            meta = MetaLabelingModel()
            meta.load(secondary_path)
            meta_probs = meta.predict_proba(features, preds)
            bet_sizes = compute_bet_sizes(meta_probs)

            # Generate signals for predictions with sufficient confidence
            signal_count = 0
            for i in range(len(preds)):
                if bet_sizes[i] < 0.25:
                    continue  # Skip low-confidence predictions

                row = valid_bars.iloc[i]
                price = float(row["close"])
                vol = float(row["high"] - row["low"])  # Simple volatility proxy
                side = int(preds[i])

                sl_price = price - side * vol * barrier_config.sl_multiplier
                pt_price = price + side * vol * barrier_config.pt_multiplier

                try:
                    conn.execute(
                        """INSERT INTO signals (symbol, bar_type, labeling_method,
                           timestamp, side, size, entry_price, sl_price, pt_price,
                           time_barrier, meta_probability)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            symbol, bar_type, labeling,
                            int(row["timestamp"]), side,
                            float(bet_sizes[i]), price,
                            sl_price, pt_price,
                            int(row["timestamp"]) + barrier_config.max_holding_period * 60000,
                            float(meta_probs[i]),
                        ],
                    )
                    signal_count += 1
                except Exception:
                    pass

            print(f"    {labeling}: {signal_count} signals")

        except Exception as e:
            print(f"    {labeling}: failed - {e}")


if __name__ == "__main__":
    main()
