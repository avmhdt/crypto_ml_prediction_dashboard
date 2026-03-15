"""Training orchestrator: CSV → bars → labels → weights → features → train → meta → save.

Reads historical data directly from D: drive CSVs, runs the full AFML pipeline,
and saves model artifacts to backend/models/.
"""
import logging
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.metrics import log_loss, recall_score, precision_score

from backend.config import (
    BarConfig, TripleBarrierConfig, TrainingConfig,
    MODELS_DIR, TICK_DATA_DIR,
)
from backend.data.csv_reader import read_trades_for_symbol, iter_trade_files, read_single_file
from backend.bars import BAR_CLASSES
from backend.labeling.triple_barrier import triple_barrier_labels
from backend.labeling.trend_scanning import trend_scanning_labels
from backend.labeling.directional_change import dc_labels_from_volatility
from backend.weights.sample_weights import compute_sample_weights
from backend.features import compute_all_features
from backend.ml.purged_cv import PurgedKFoldCV
from backend.ml.primary_model import PrimaryModel
from backend.ml.meta_labeling import MetaLabelingModel
from backend.ml.bet_sizing import bet_size_from_probability

logger = logging.getLogger(__name__)

LABELING_FUNCTIONS = {
    "triple_barrier": lambda bars, cfg: triple_barrier_labels(bars, cfg),
    "trend_scanning": lambda bars, _: trend_scanning_labels(bars),
    "directional_change": lambda bars, _: dc_labels_from_volatility(bars),
}


def _make_generator(symbol: str, bar_type: str, bar_config: BarConfig):
    """Create a bar generator for the given type."""
    bar_class = BAR_CLASSES[bar_type]
    if bar_type == "time":
        return bar_class(symbol, bar_config.time_interval)
    elif bar_type in ("tick", "volume", "dollar"):
        thresholds = {
            "tick": bar_config.tick_count,
            "volume": bar_config.volume_threshold,
            "dollar": bar_config.dollar_threshold,
        }
        return bar_class(symbol, thresholds[bar_type])
    else:
        return bar_class(
            symbol,
            expected_num_ticks_init=bar_config.tick_count,
            num_prev_bars=bar_config.ewma_span,
        )


def generate_bars(trades: pd.DataFrame, symbol: str, bar_type: str,
                  bar_config: BarConfig) -> pd.DataFrame:
    """Generate bars from a pre-loaded trades DataFrame."""
    generator = _make_generator(symbol, bar_type, bar_config)
    bars = generator.process_ticks(
        trades["price"].values,
        trades["qty"].values,
        trades["time"].values,
        trades["is_buyer_maker"].values,
    )
    if not bars:
        return pd.DataFrame()
    return pd.DataFrame([b.to_dict() for b in bars])


def generate_bars_streaming(
    symbol: str,
    bar_type: str,
    bar_config: BarConfig,
    data_dir: Path,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Generate bars by streaming CSVs one file at a time.

    Instead of loading all trades into memory (~78 GB for 2011 files),
    reads each CSV, feeds its ticks to the bar generator, then frees
    the CSV data. Peak memory is ~one CSV (~40 MB) + accumulated bars.
    """
    generator = _make_generator(symbol, bar_type, bar_config)
    files = iter_trade_files(symbol, data_dir, start_time, end_time)

    all_bars = []
    total_ticks = 0

    for i, (csv_path, fmt) in enumerate(files):
        chunk = read_single_file(csv_path, fmt, symbol, start_time, end_time)
        if chunk is None or chunk.empty:
            continue

        n = len(chunk)
        total_ticks += n

        new_bars = generator.process_ticks(
            chunk["price"].values,
            chunk["qty"].values,
            chunk["time"].values,
            chunk["is_buyer_maker"].values,
        )
        all_bars.extend(new_bars)

        # Log progress every 100 files
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            logger.info(
                f"  Streamed {i+1}/{len(files)} files, "
                f"{total_ticks:,} ticks → {len(all_bars):,} bars"
            )

        # Free the chunk immediately
        del chunk

    logger.info(f"Streaming complete: {total_ticks:,} ticks → {len(all_bars):,} bars")

    if not all_bars:
        return pd.DataFrame()
    return pd.DataFrame([b.to_dict() for b in all_bars])


def generate_labels(bars: pd.DataFrame, labeling_method: str,
                    barrier_config: TripleBarrierConfig) -> pd.DataFrame:
    """Generate labels using the specified method."""
    label_fn = LABELING_FUNCTIONS[labeling_method]
    return label_fn(bars, barrier_config)


def train_pipeline(
    symbol: str,
    bar_type: str,
    labeling_method: str,
    data_dir: Path = TICK_DATA_DIR,
    bar_config: BarConfig | None = None,
    barrier_config: TripleBarrierConfig | None = None,
    training_config: TrainingConfig | None = None,
    trades: pd.DataFrame | None = None,
) -> dict:
    """Full training pipeline: data → model artifacts.

    Parameters
    ----------
    trades : pd.DataFrame | None
        Pre-loaded trade data. When provided, skips CSV loading (useful
        for batch training where the same data is reused across multiple
        bar_type/labeling combinations).

    Returns dict with paths to saved artifacts and training metrics.
    """
    bar_config = bar_config or BarConfig()
    barrier_config = barrier_config or TripleBarrierConfig()
    training_config = training_config or TrainingConfig()

    # Step 1: Generate bars — use streaming to avoid loading all trades into memory
    logger.info(f"Generating {bar_type} bars...")
    if trades is not None:
        # Pre-loaded trades (small dataset or testing)
        logger.info(f"Using {len(trades):,} pre-loaded trades")
        bars = generate_bars(trades, symbol, bar_type, bar_config)
    else:
        # Stream through CSV files one at a time (~40 MB peak vs ~78 GB all-at-once)
        logger.info(f"Streaming from {data_dir}")
        bars = generate_bars_streaming(
            symbol, bar_type, bar_config, data_dir,
            start_time=None, end_time=None,
        )
    if bars.empty or len(bars) < 100:
        raise ValueError(f"Insufficient bars generated: {len(bars)}")
    logger.info(f"Generated {len(bars):,} bars")

    # Step 2: Generate labels
    logger.info(f"Generating {labeling_method} labels...")
    labels_df = generate_labels(bars, labeling_method, barrier_config)
    if labels_df.empty:
        raise ValueError("No labels generated")

    # Merge labels with bars
    bars = bars.merge(labels_df[["timestamp", "label"]], on="timestamp", how="inner")
    bars = bars.dropna(subset=["label"])
    logger.info(f"Bars with labels: {len(bars):,}")

    # Step 3: Compute features
    logger.info("Computing features...")
    features = compute_all_features(bars, window=training_config.feature_window)

    # Align features with labeled bars — drop warm-up NaN rows
    # Forward-fill then drop any remaining leading NaN rows
    features = features.ffill()
    valid_mask = features.notna().all(axis=1)
    features = features[valid_mask]
    bars = bars.loc[features.index]
    if len(features) == 0:
        raise ValueError("No valid samples after feature warm-up (all rows have NaN)")
    labels = bars["label"].values.astype(int)
    unique_labels = set(np.unique(labels))
    if not unique_labels.issubset({-1, 1}):
        raise ValueError(f"Unexpected labels after dropna: {unique_labels}. "
                         "NaN labels may have leaked through.")
    n_long = int((labels == 1).sum())
    n_short = int((labels == -1).sum())
    logger.info(f"Samples after feature warm-up: {len(features):,}, "
                f"Features: {features.shape[1]}")
    logger.info(f"Label distribution: LONG={n_long} ({100*n_long/len(labels):.1f}%), "
                f"SHORT={n_short} ({100*n_short/len(labels):.1f}%)")

    # Compute label span for purged CV based on labeling method.
    # This determines how far forward each label's outcome extends,
    # which controls the purge radius in PurgedKFoldCV.
    if labeling_method == "triple_barrier":
        label_span = barrier_config.max_holding_period
    elif labeling_method == "trend_scanning":
        label_span = 80  # max of default horizons [5, 10, 20, 40, 80]
    else:  # directional_change — event-driven, use conservative estimate
        label_span = barrier_config.max_holding_period

    # Step 4: Compute sample weights
    logger.info("Computing sample weights...")
    # Simple label spans: each label covers max_holding_period bars
    label_spans = [
        (i, min(i + barrier_config.max_holding_period, len(bars) - 1))
        for i in range(len(bars))
    ]
    returns = bars["close"].pct_change().fillna(0).values
    timestamps = bars["timestamp"].values

    weights = compute_sample_weights(
        label_spans=label_spans,
        returns=returns,
        timestamps=timestamps,
        num_bars=len(bars),
        half_life=training_config.time_decay_half_life,
    )

    # Step 5: Train primary model (optimize Recall)
    logger.info("Training primary model (Recall-optimized)...")
    primary = PrimaryModel()

    if training_config.optuna_n_trials > 0:
        best_params = _optuna_tune(
            features, labels, weights, training_config,
            label_span=label_span,
            optimize_for="recall",
        )
        primary.params.update(best_params)

    primary.fit(features, labels, sample_weight=weights)

    # Primary model evaluation
    primary_preds = primary.predict(features)
    primary_recall = recall_score(
        (labels + 1) // 2, (primary_preds + 1) // 2, zero_division=0
    )
    logger.info(f"Primary model recall (train): {primary_recall:.4f}")

    # Step 6: Train meta-labeling model (optimize Precision)
    # Use out-of-sample primary predictions (AFML Ch.3) so the meta model
    # sees realistic primary errors instead of overfitting to in-sample accuracy.
    logger.info("Training meta-labeling model (Precision-optimized, OOS)...")

    label_ends = np.minimum(
        np.arange(len(labels)) + label_span,
        len(labels) - 1,
    )
    oos_cv = PurgedKFoldCV(
        n_splits=training_config.n_splits,
        label_ends=label_ends,
        embargo_pct=training_config.embargo_pct,
    )
    oos_preds = np.zeros(len(labels), dtype=int)
    for train_idx, test_idx in oos_cv.split(features):
        fold_model = PrimaryModel(params={**primary.params})
        fold_model.fit(
            features.iloc[train_idx], labels[train_idx],
            sample_weight=weights[train_idx],
        )
        oos_preds[test_idx] = fold_model.predict(features.iloc[test_idx])

    oos_accuracy = float((oos_preds == labels).mean())
    logger.info(f"Primary OOS accuracy: {oos_accuracy:.4f} "
                f"(in-sample recall: {primary_recall:.4f})")

    meta_model = MetaLabelingModel()

    if len(labels) < 100:
        # Too few samples for meaningful holdout evaluation
        meta_model.fit(features, oos_preds, labels, sample_weight=weights)
        meta_preds = meta_model.predict(features, oos_preds)
        meta_precision = precision_score(
            MetaLabelingModel.construct_meta_labels(oos_preds, labels),
            meta_preds, zero_division=0,
        )
        logger.warning(f"Too few samples ({len(labels)}) for meta holdout; "
                       f"meta_precision is in-sample: {meta_precision:.4f}")
    else:
        # Temporal holdout: last 20% for honest OOS evaluation
        val_size = max(int(len(labels) * 0.2), 50)
        val_size = min(val_size, len(labels) - 50)
        meta_train_idx = np.arange(len(labels) - val_size)
        meta_val_idx = np.arange(len(labels) - val_size, len(labels))

        meta_model.fit(
            features.iloc[meta_train_idx], oos_preds[meta_train_idx],
            labels[meta_train_idx], sample_weight=weights[meta_train_idx],
        )

        # Evaluate on held-out validation set
        meta_preds_val = meta_model.predict(
            features.iloc[meta_val_idx], oos_preds[meta_val_idx],
        )
        meta_precision = precision_score(
            MetaLabelingModel.construct_meta_labels(
                oos_preds[meta_val_idx], labels[meta_val_idx]
            ),
            meta_preds_val, zero_division=0,
        )
        logger.info(f"Meta-labeling precision (held-out): {meta_precision:.4f}")

        # Refit on full data for production model
        meta_model.fit(features, oos_preds, labels, sample_weight=weights)

    # Generate bet-sizing probabilities from full-data model
    meta_probs = meta_model.predict_proba(features, oos_preds)

    # Step 7: Compute bet sizes (raw continuous, no discretization)
    bet_sizes = bet_size_from_probability(meta_probs)

    # Step 8: Save artifacts
    logger.info("Saving model artifacts...")
    prefix = f"{symbol}_{bar_type}_{labeling_method}"
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    primary.save(
        models_dir / f"{prefix}_primary.joblib",
        models_dir / f"{prefix}_scaler.joblib",
        models_dir / f"{prefix}_features.json",
    )
    meta_model.save(
        models_dir / f"{prefix}_secondary.joblib",
        models_dir / f"{prefix}_meta_features.json",
    )

    results = {
        "symbol": symbol,
        "bar_type": bar_type,
        "labeling_method": labeling_method,
        "num_trades": len(trades) if trades is not None else 0,
        "num_bars": len(bars),
        "num_samples": len(features),
        "num_features": features.shape[1],
        "primary_recall": primary_recall,
        "meta_precision": meta_precision,
        "bet_size_mean": float(bet_sizes.mean()),
        "bet_size_std": float(bet_sizes.std()),
        "artifacts": {
            "primary": str(models_dir / f"{prefix}_primary.joblib"),
            "secondary": str(models_dir / f"{prefix}_secondary.joblib"),
            "scaler": str(models_dir / f"{prefix}_scaler.joblib"),
            "features": str(models_dir / f"{prefix}_features.json"),
        },
    }

    logger.info(f"Training complete: {results}")
    return results


def _optuna_tune(features: pd.DataFrame, labels: np.ndarray,
                 weights: np.ndarray, config: TrainingConfig,
                 label_span: int = 50,
                 optimize_for: str = "recall") -> dict:
    """Hyperparameter tuning with Optuna using Purged CV.

    Parameters
    ----------
    label_span : int
        Forward span of each label in bars (used for purge radius).
    optimize_for : str
        Metric to optimize: ``"recall"``, ``"precision"``, or ``"log_loss"``.
    """
    # Determine study direction based on metric
    if optimize_for in ("recall", "precision"):
        direction = "maximize"
    else:
        direction = "minimize"

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        # Create label end indices for purged CV
        label_ends = np.minimum(
            np.arange(len(labels)) + label_span,
            len(labels) - 1,
        )

        cv = PurgedKFoldCV(
            n_splits=config.n_splits,
            label_ends=label_ends,
            embargo_pct=config.embargo_pct,
        )

        scores = []
        for train_idx, test_idx in cv.split(features):
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            w_train = weights[train_idx]

            model = PrimaryModel(params={**params, "objective": "binary",
                                          "metric": "binary_logloss",
                                          "boosting_type": "gbdt",
                                          "verbose": -1, "n_jobs": -1})
            model.fit(X_train, y_train, sample_weight=w_train)

            if optimize_for == "recall":
                y_pred = model.predict(X_test)
                y_pred_binary = ((y_pred + 1) / 2).astype(int)
                y_test_binary = ((y_test + 1) / 2).astype(int)
                score = recall_score(y_test_binary, y_pred_binary, zero_division=0)
            elif optimize_for == "precision":
                y_pred = model.predict(X_test)
                y_pred_binary = ((y_pred + 1) / 2).astype(int)
                y_test_binary = ((y_test + 1) / 2).astype(int)
                score = precision_score(y_test_binary, y_pred_binary, zero_division=0)
            else:
                y_pred_proba = model.predict_proba(X_test)
                y_test_binary = ((y_test + 1) / 2).astype(int)
                score = log_loss(y_test_binary, y_pred_proba, labels=[0, 1])
            scores.append(score)

        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=config.optuna_n_trials,
        timeout=config.optuna_timeout,
        show_progress_bar=True,
    )

    logger.info(f"Best trial ({optimize_for}): {study.best_trial.value:.4f}")
    return study.best_trial.params
