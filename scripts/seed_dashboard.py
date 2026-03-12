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

        # Shift timestamps so data ends at "now" (prevents prune_old_data from deleting)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        max_ts = int(bars_df["timestamp"].max())
        ts_offset = now_ms - max_ts
        bars_df["timestamp"] = bars_df["timestamp"] + ts_offset
        print(f"  Shifted timestamps to end at now (offset: {ts_offset / 3600000:.1f}h)")

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

        # Try ML model signals first, fall back to synthetic
        ml_ok = _try_seed_signals(
            conn, bars_df, args.symbol, bar_type,
            barrier_config, bar_config,
        )
        if not ml_ok:
            _seed_synthetic_signals(conn, bars_df, args.symbol, bar_type)

    # Force WAL flush so data survives force-kills
    conn.execute("CHECKPOINT")
    conn.close()
    print(f"\nDone! Dashboard DB seeded at {DB_PATH}")
    print("Restart the backend server to pick up the new data.")


def _try_seed_signals(conn, bars_df, symbol, bar_type, barrier_config, bar_config):
    """Try to generate signals from trained models. Returns True if any succeeded."""
    from backend.config import MODELS_DIR, LABELING_METHODS
    from backend.labeling.triple_barrier import _daily_volatility
    import pandas as pd

    # Compute volatility once for all labeling methods (invariant across loop)
    rolling_vol = _daily_volatility(
        pd.Series(bars_df["close"].values, dtype=np.float64),
        span=barrier_config.volatility_window,
    )
    rolling_vol.index = bars_df.index  # align with bars_df index

    # Horizon scaling: estimate vol over the holding period
    ts_vals = bars_df["timestamp"].values
    avg_bar_ms = float(np.median(np.diff(ts_vals))) if len(ts_vals) > 1 else 60000.0
    horizon_ms = barrier_config.max_holding_period * 60000
    horizon_bars = max(horizon_ms / avg_bar_ms, 1.0)

    any_succeeded = False
    for labeling in LABELING_METHODS:
        prefix = f"{symbol}_{bar_type}_{labeling}"
        primary_path = MODELS_DIR / f"{prefix}_primary.joblib"
        scaler_path = MODELS_DIR / f"{prefix}_scaler.joblib"
        features_path = MODELS_DIR / f"{prefix}_features.json"
        secondary_path = MODELS_DIR / f"{prefix}_secondary.joblib"

        if not primary_path.exists():
            continue

        try:
            from backend.ml.primary_model import PrimaryModel
            from backend.ml.meta_labeling import MetaLabelingModel
            from backend.ml.bet_sizing import compute_bet_sizes
            from backend.features import compute_all_features

            features = compute_all_features(bars_df, window=20)
            features = features.ffill()
            valid = features.notna().all(axis=1)
            features = features[valid]
            valid_bars = bars_df.loc[features.index]
            valid_vol = rolling_vol.loc[features.index].values

            if len(features) < 10:
                continue

            primary = PrimaryModel()
            primary.load(primary_path, scaler_path, features_path)
            preds = primary.predict(features)

            meta = MetaLabelingModel()
            meta_features_path = MODELS_DIR / f"{prefix}_meta_features.json"
            meta.load(secondary_path, meta_features_path)
            # Backward compat: if no meta_features.json, use primary's feature names
            if not meta.feature_names:
                meta.feature_names = primary.feature_names
            meta_probs = meta.predict_proba(features, preds)
            bet_sizes = compute_bet_sizes(meta_probs)

            signal_count = 0
            for i in range(len(preds)):
                if meta_probs[i] < 0.5:
                    continue
                if bet_sizes[i] < 0.25:
                    continue
                row = valid_bars.iloc[i]
                price = float(row["close"])
                side = int(preds[i])
                # Horizon-scaled volatility: per-bar vol * sqrt(horizon_bars)
                per_bar_vol = float(valid_vol[i])
                horizon_vol = per_bar_vol * np.sqrt(horizon_bars)
                sl_price = price * (1 - side * barrier_config.sl_multiplier * horizon_vol)
                pt_price = price * (1 + side * barrier_config.pt_multiplier * horizon_vol)
                try:
                    conn.execute(
                        """INSERT INTO signals (symbol, bar_type, labeling_method,
                           timestamp, side, size, entry_price, sl_price, pt_price,
                           time_barrier, meta_probability)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [symbol, bar_type, labeling, int(row["timestamp"]), side,
                         float(bet_sizes[i]), price, sl_price, pt_price,
                         int(row["timestamp"]) + barrier_config.max_holding_period * 60000,
                         float(meta_probs[i])],
                    )
                    signal_count += 1
                except Exception:
                    pass
            if signal_count > 0:
                any_succeeded = True
            print(f"    {labeling}: {signal_count} signals (ML)")

        except Exception as e:
            print(f"    {labeling}: ML failed - {e}")

    return any_succeeded


def _seed_synthetic_signals(conn, bars_df, symbol, bar_type):
    """Generate synthetic signals for dashboard display when ML models fail."""
    from backend.config import LABELING_METHODS
    from backend.labeling.triple_barrier import _daily_volatility
    import pandas as pd

    barrier_config = TripleBarrierConfig()

    np.random.seed(42 + hash(bar_type) % 1000)
    total = 0

    # Reuse the same EWMA volatility as triple_barrier.py
    rolling_vol = _daily_volatility(
        pd.Series(bars_df["close"].values, dtype=np.float64),
        span=barrier_config.volatility_window,
    )
    rolling_vol.index = bars_df.index

    # Horizon scaling: estimate vol over the holding period
    ts_vals = bars_df["timestamp"].values
    avg_bar_ms = float(np.median(np.diff(ts_vals))) if len(ts_vals) > 1 else 60000.0
    horizon_ms = barrier_config.max_holding_period * 60000
    horizon_bars = max(horizon_ms / avg_bar_ms, 1.0)

    for idx, row in bars_df.iterrows():
        if np.random.random() > 0.12:
            continue
        side = int(np.random.choice([-1, 1]))
        meta_prob = round(float(np.random.uniform(0.45, 0.95)), 4)
        size = float(np.random.choice([0.25, 0.5, 0.75, 1.0],
                                       p=[0.15, 0.3, 0.35, 0.2]))
        price = float(row["close"])
        # Horizon-scaled volatility: per-bar vol * sqrt(horizon_bars)
        per_bar_vol = float(rolling_vol.at[idx])
        horizon_vol = per_bar_vol * np.sqrt(horizon_bars)
        sl = price * (1 - side * barrier_config.sl_multiplier * horizon_vol)
        pt = price * (1 + side * barrier_config.pt_multiplier * horizon_vol)
        tb = int(row["timestamp"]) + horizon_ms

        for labeling in LABELING_METHODS:
            try:
                conn.execute(
                    """INSERT INTO signals (symbol, bar_type, labeling_method,
                       timestamp, side, size, entry_price, sl_price, pt_price,
                       time_barrier, meta_probability)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [symbol, bar_type, labeling, int(row["timestamp"]),
                     side, size, price, sl, pt, tb, meta_prob],
                )
                total += 1
            except Exception:
                pass

    per_labeling = total // len(LABELING_METHODS) if LABELING_METHODS else 0
    print(f"  Synthetic signals: {per_labeling} per labeling method ({total} total)")


if __name__ == "__main__":
    main()
