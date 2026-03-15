"""Synthetic Signal Validation — SNR sweep across the full AFML pipeline.

Generates synthetic ticks at multiple Sharpe ratio levels, runs the
complete pipeline (bars → features → labels → train → meta → equity)
at each level, and produces a "signal recovery curve" showing model
accuracy vs. injected signal strength.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score

from backend.config import BarConfig, TripleBarrierConfig, TrainingConfig
from backend.bars import BAR_CLASSES
from backend.labeling.triple_barrier import triple_barrier_labels
from backend.labeling.trend_scanning import trend_scanning_labels
from backend.labeling.directional_change import dc_labels_from_volatility
from backend.features import compute_all_features
from backend.weights.sample_weights import compute_sample_weights
from backend.ml.purged_cv import PurgedKFoldCV
from backend.ml.primary_model import PrimaryModel
from backend.ml.meta_labeling import MetaLabelingModel
from backend.ml.bet_sizing import bet_size_from_probability
from backend.simulation.equity import simulate_equity
from backend.synthetic.generator import generate_synthetic_ticks

logger = logging.getLogger(__name__)

LABELING_FUNCTIONS = {
    "triple_barrier": lambda bars, cfg: triple_barrier_labels(bars, cfg),
    "trend_scanning": lambda bars, _: trend_scanning_labels(bars),
    "directional_change": lambda bars, _: dc_labels_from_volatility(bars),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPointResult:
    sharpe: float
    oos_accuracy: float
    oos_precision: float
    oos_recall: float
    equity_sharpe: float
    total_return: float
    max_dd: float
    win_rate: float
    num_trades: int
    num_bars: int
    num_samples: int


@dataclass
class SyntheticValidationResult:
    bar_type: str
    labeling_method: str
    n_ticks: int
    sharpe_levels: list[float]
    points: list[SyntheticPointResult]
    detection_threshold: float
    created_at: str


# ---------------------------------------------------------------------------
# Per-SNR evaluation
# ---------------------------------------------------------------------------

def _make_generator(symbol: str, bar_type: str, bar_config: BarConfig):
    """Create a bar generator (mirrors training.py._make_generator)."""
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


def _evaluate_single_sharpe(
    sharpe: float,
    bar_type: str,
    labeling_method: str,
    n_ticks: int,
    bar_config: BarConfig,
    barrier_config: TripleBarrierConfig,
    training_config: TrainingConfig,
    starting_capital: float,
    fees_bps: float,
    seed: int,
) -> SyntheticPointResult:
    """Run full pipeline on synthetic ticks at a given Sharpe level."""
    logger.info(f"  Sharpe {sharpe:.1f}: generating {n_ticks:,} ticks...")

    # 1. Generate synthetic ticks
    ticks = generate_synthetic_ticks(
        n_ticks=n_ticks,
        sharpe=sharpe,
        seed=seed + int(sharpe * 1000),
    )

    # 2. Generate bars from ticks
    generator = _make_generator("SYNTHETIC", bar_type, bar_config)
    bars_list = generator.process_ticks(
        ticks["price"].values,
        ticks["qty"].values,
        ticks["time"].values,
        ticks["is_buyer_maker"].values,
    )
    if not bars_list:
        raise ValueError(f"No bars generated for sharpe={sharpe}")
    bars = pd.DataFrame([b.to_dict() for b in bars_list])
    logger.info(f"  Sharpe {sharpe:.1f}: {len(bars)} bars from {n_ticks:,} ticks")

    if len(bars) < 100:
        raise ValueError(f"Too few bars ({len(bars)}) for sharpe={sharpe}")

    # 3. Generate labels
    label_fn = LABELING_FUNCTIONS[labeling_method]
    labels_df = label_fn(bars, barrier_config)
    if labels_df.empty:
        raise ValueError(f"No labels generated for sharpe={sharpe}")

    bars = bars.merge(labels_df[["timestamp", "label"]], on="timestamp", how="inner")
    bars = bars.dropna(subset=["label"])

    # 4. Compute features
    features = compute_all_features(bars, window=training_config.feature_window)
    features = features.ffill()
    valid_mask = features.notna().all(axis=1)
    features = features[valid_mask]
    bars = bars.loc[features.index]

    if len(features) < 50:
        raise ValueError(f"Too few samples ({len(features)}) for sharpe={sharpe}")

    labels = bars["label"].values.astype(int)

    # 5. 70/30 temporal split
    split_idx = int(len(features) * 0.7)
    train_features = features.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    train_bars = bars.iloc[:split_idx]
    test_bars = bars.iloc[split_idx:]

    if len(train_features) < 30 or len(test_features) < 10:
        raise ValueError(f"Too few train/test samples for sharpe={sharpe}")

    # 6. Sample weights
    label_spans = [
        (i, min(i + barrier_config.max_holding_period, len(train_bars) - 1))
        for i in range(len(train_bars))
    ]
    returns = train_bars["close"].pct_change().fillna(0).values
    timestamps = train_bars["timestamp"].values
    weights = compute_sample_weights(
        label_spans=label_spans,
        returns=returns,
        timestamps=timestamps,
        num_bars=len(train_bars),
        half_life=training_config.time_decay_half_life,
    )

    # 7. Train primary model (no Optuna)
    primary = PrimaryModel()
    primary.fit(train_features, train_labels, sample_weight=weights)

    # 8. OOS primary predictions via purged CV for meta model training
    if labeling_method == "triple_barrier":
        label_span = barrier_config.max_holding_period
    elif labeling_method == "trend_scanning":
        label_span = 80
    else:
        label_span = barrier_config.max_holding_period

    label_ends = np.minimum(
        np.arange(len(train_labels)) + label_span,
        len(train_labels) - 1,
    )
    oos_cv = PurgedKFoldCV(
        n_splits=training_config.n_splits,
        label_ends=label_ends,
        embargo_pct=training_config.embargo_pct,
    )
    oos_preds = np.zeros(len(train_labels), dtype=int)
    for train_idx, test_idx in oos_cv.split(train_features):
        fold_model = PrimaryModel(params={**primary.params})
        fold_model.fit(
            train_features.iloc[train_idx], train_labels[train_idx],
            sample_weight=weights[train_idx],
        )
        oos_preds[test_idx] = fold_model.predict(train_features.iloc[test_idx])

    # 9. Train meta model
    meta_model = MetaLabelingModel()
    meta_model.fit(train_features, oos_preds, train_labels, sample_weight=weights)

    # 10. Evaluate on test set
    test_primary_preds = primary.predict(test_features)
    test_meta_probs = meta_model.predict_proba(test_features, test_primary_preds)

    # OOS accuracy (primary model)
    test_labels_binary = ((test_labels + 1) / 2).astype(int)
    test_preds_binary = ((test_primary_preds + 1) / 2).astype(int)
    oos_accuracy = float((test_preds_binary == test_labels_binary).mean())
    oos_prec = float(precision_score(test_labels_binary, test_preds_binary, zero_division=0))
    oos_rec = float(recall_score(test_labels_binary, test_preds_binary, zero_division=0))

    # 11. Generate signals for equity simulation
    signals_data = []
    for i in range(len(test_features)):
        if test_meta_probs[i] > 0.5:
            row = test_bars.iloc[i]
            signals_data.append({
                "timestamp": int(row["timestamp"]),
                "side": int(test_primary_preds[i]),
                "size": float(bet_size_from_probability(np.array([test_meta_probs[i]]))[0]),
                "entry_price": float(row["close"]),
                "meta_probability": float(test_meta_probs[i]),
            })

    num_trades = len(signals_data)

    # Equity metrics defaults
    eq_sharpe = 0.0
    total_return = 0.0
    max_dd = 0.0
    win_rate = 0.0

    if signals_data:
        signals_df = pd.DataFrame(signals_data)
        try:
            sim_result = simulate_equity(
                signals_df, test_bars.reset_index(drop=True),
                labeling_method, starting_capital, fees_bps,
            )
            metrics = sim_result.get("metrics", {})
            eq_sharpe = float(metrics.get("sharpe", 0.0))
            total_return = float(metrics.get("total_return", 0.0))
            max_dd = float(metrics.get("max_dd", 0.0))
            win_rate = float(metrics.get("win_rate", 0.0))
        except Exception as e:
            logger.warning(f"  Equity sim failed for sharpe={sharpe}: {e}")

    logger.info(
        f"  Sharpe {sharpe:.1f}: OOS acc={oos_accuracy:.3f}, "
        f"trades={num_trades}, eq_sharpe={eq_sharpe:.2f}"
    )

    return SyntheticPointResult(
        sharpe=sharpe,
        oos_accuracy=round(oos_accuracy, 4),
        oos_precision=round(oos_prec, 4),
        oos_recall=round(oos_rec, 4),
        equity_sharpe=round(eq_sharpe, 2),
        total_return=round(total_return, 2),
        max_dd=round(max_dd, 2),
        win_rate=round(win_rate, 1),
        num_trades=num_trades,
        num_bars=len(bars),
        num_samples=len(features),
    )


# ---------------------------------------------------------------------------
# SNR Sweep
# ---------------------------------------------------------------------------

DEFAULT_SHARPE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def compute_detection_threshold(
    points: list[SyntheticPointResult],
    accuracy_threshold: float = 0.55,
) -> float:
    """Find lowest Sharpe where OOS accuracy exceeds threshold."""
    sorted_pts = sorted(points, key=lambda p: p.sharpe)
    for pt in sorted_pts:
        if pt.oos_accuracy >= accuracy_threshold:
            return pt.sharpe
    return float("inf")


def run_synthetic_validation(
    bar_type: str = "time",
    labeling_method: str = "triple_barrier",
    sharpe_levels: list[float] | None = None,
    n_ticks: int = 500_000,
    bar_config: BarConfig | None = None,
    barrier_config: TripleBarrierConfig | None = None,
    training_config: TrainingConfig | None = None,
    starting_capital: float = 10000.0,
    fees_bps: float = 10.0,
    seed: int = 42,
) -> SyntheticValidationResult:
    """Run full pipeline at each SNR level and collect results."""
    sharpe_levels = sharpe_levels or DEFAULT_SHARPE_LEVELS
    bar_config = bar_config or BarConfig()
    barrier_config = barrier_config or TripleBarrierConfig()
    training_config = training_config or TrainingConfig(optuna_n_trials=0, optuna_timeout=0)

    # Force no Optuna
    training_config.optuna_n_trials = 0
    training_config.optuna_timeout = 0

    logger.info(f"Starting synthetic validation: {bar_type}/{labeling_method}, "
                f"{len(sharpe_levels)} SNR levels, {n_ticks:,} ticks each")

    points: list[SyntheticPointResult] = []
    for level in sharpe_levels:
        try:
            pt = _evaluate_single_sharpe(
                sharpe=level,
                bar_type=bar_type,
                labeling_method=labeling_method,
                n_ticks=n_ticks,
                bar_config=bar_config,
                barrier_config=barrier_config,
                training_config=training_config,
                starting_capital=starting_capital,
                fees_bps=fees_bps,
                seed=seed,
            )
            points.append(pt)
        except Exception as e:
            logger.error(f"  Sharpe {level:.1f} FAILED: {e}")

    if not points:
        raise ValueError("All SNR levels failed — no results produced")

    threshold = compute_detection_threshold(points)

    return SyntheticValidationResult(
        bar_type=bar_type,
        labeling_method=labeling_method,
        n_ticks=n_ticks,
        sharpe_levels=sharpe_levels,
        points=points,
        detection_threshold=round(threshold, 2),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
