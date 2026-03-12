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
from backend.data.csv_reader import read_trades_for_symbol
from backend.bars import BAR_CLASSES
from backend.labeling.triple_barrier import triple_barrier_labels
from backend.labeling.trend_scanning import trend_scanning_labels
from backend.labeling.directional_change import dc_labels_from_volatility
from backend.weights.sample_weights import compute_sample_weights
from backend.features import compute_all_features
from backend.ml.purged_cv import PurgedKFoldCV
from backend.ml.primary_model import PrimaryModel
from backend.ml.meta_labeling import MetaLabelingModel
from backend.ml.bet_sizing import compute_bet_sizes, DISCRETE_LEVELS

logger = logging.getLogger(__name__)

LABELING_FUNCTIONS = {
    "triple_barrier": lambda bars, cfg: triple_barrier_labels(bars, cfg),
    "trend_scanning": lambda bars, _: trend_scanning_labels(bars),
    "directional_change": lambda bars, _: dc_labels_from_volatility(bars),
}


def generate_bars(trades: pd.DataFrame, symbol: str, bar_type: str,
                  bar_config: BarConfig) -> pd.DataFrame:
    """Generate bars from raw trade data."""
    bar_class = BAR_CLASSES[bar_type]

    if bar_type == "time":
        generator = bar_class(symbol, bar_config.time_interval)
    elif bar_type in ("tick", "volume", "dollar"):
        thresholds = {
            "tick": bar_config.tick_count,
            "volume": bar_config.volume_threshold,
            "dollar": bar_config.dollar_threshold,
        }
        generator = bar_class(symbol, thresholds[bar_type])
    else:
        # Imbalance/run bars — concrete classes hardcode bar_type internally
        generator = bar_class(
            symbol,
            expected_num_ticks_init=bar_config.tick_count,
            num_prev_bars=bar_config.ewma_span,
        )

    bars = generator.process_ticks(
        trades["price"].values,
        trades["qty"].values,
        trades["time"].values,
        trades["is_buyer_maker"].values,
    )

    if not bars:
        return pd.DataFrame()

    records = [b.to_dict() for b in bars]
    return pd.DataFrame(records)


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

    if trades is None:
        logger.info(f"Loading trades for {symbol} from {data_dir}")
        trades = read_trades_for_symbol(symbol, data_dir)
    if trades.empty:
        raise ValueError(f"No trade data found for {symbol} in {data_dir}")
    logger.info(f"Using {len(trades):,} trades")

    # Step 1: Generate bars
    logger.info(f"Generating {bar_type} bars...")
    bars = generate_bars(trades, symbol, bar_type, bar_config)
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
    n_long = int((labels == 1).sum())
    n_short = int((labels == -1).sum())
    logger.info(f"Samples after feature warm-up: {len(features):,}, "
                f"Features: {features.shape[1]}")
    logger.info(f"Label distribution: LONG={n_long} ({100*n_long/len(labels):.1f}%), "
                f"SHORT={n_short} ({100*n_short/len(labels):.1f}%)")

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
        np.arange(len(labels)) + training_config.n_splits,
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
    meta_model.fit(features, oos_preds, labels, sample_weight=weights)

    meta_probs = meta_model.predict_proba(features, oos_preds)
    meta_preds = meta_model.predict(features, oos_preds)
    meta_precision = precision_score(
        MetaLabelingModel.construct_meta_labels(oos_preds, labels),
        meta_preds, zero_division=0,
    )
    logger.info(f"Meta-labeling precision (OOS): {meta_precision:.4f}")

    # Step 7: Compute bet sizes
    bet_sizes = compute_bet_sizes(meta_probs)

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
        "num_trades": len(trades),
        "num_bars": len(bars),
        "num_samples": len(features),
        "num_features": features.shape[1],
        "primary_recall": primary_recall,
        "meta_precision": meta_precision,
        "bet_size_distribution": {
            str(level): int((bet_sizes == level).sum())
            for level in DISCRETE_LEVELS
        },
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
                 optimize_for: str = "recall") -> dict:
    """Hyperparameter tuning with Optuna using Purged CV."""

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
            np.arange(len(labels)) + config.n_splits,
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

            y_pred_proba = model.predict_proba(X_test)
            y_test_binary = ((y_test + 1) / 2).astype(int)
            score = log_loss(y_test_binary, y_pred_proba, labels=[0, 1])
            scores.append(score)

        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=config.optuna_n_trials,
        timeout=config.optuna_timeout,
        show_progress_bar=True,
    )

    logger.info(f"Best trial: {study.best_trial.value:.4f}")
    return study.best_trial.params
