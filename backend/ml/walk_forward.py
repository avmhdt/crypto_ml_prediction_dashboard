"""Walk-Forward Validation engine for out-of-sample backtesting.

Splits historical data into rolling train/test windows and evaluates the
full AFML pipeline (bars -> labels -> features -> primary -> meta -> equity)
on each test period.  Only out-of-sample test metrics are recorded, giving
an honest estimate of live-trading performance.

References
----------
- Lopez de Prado, *Advances in Financial Machine Learning* (2018), Ch. 12
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014)
"""
import logging
import json
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score

from backend.config import (
    BarConfig,
    TripleBarrierConfig,
    TrainingConfig,
    TICK_DATA_DIR,
)
from backend.data.csv_reader import iter_trade_files
from backend.ml.training import generate_bars_streaming, generate_labels
from backend.features import compute_all_features
from backend.weights.sample_weights import compute_sample_weights
from backend.ml.purged_cv import PurgedKFoldCV
from backend.ml.primary_model import PrimaryModel
from backend.ml.meta_labeling import MetaLabelingModel
from backend.ml.bet_sizing import bet_size_from_probability
from backend.simulation.equity import simulate_equity

logger = logging.getLogger(__name__)

_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")
_MS_PER_DAY = 86_400_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Metrics and equity curve for a single train/test window."""

    window_index: int
    train_start: int          # ms timestamp
    train_end: int
    test_start: int
    test_end: int
    num_train_bars: int
    num_test_bars: int
    num_train_samples: int
    num_test_signals: int
    primary_recall: float
    meta_precision: float
    oos_accuracy: float
    oos_precision: float
    oos_recall: float
    sharpe: float
    max_dd: float
    total_return: float
    win_rate: float
    num_trades: int
    timestamps: list[int] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    drawdown: list[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result across all windows."""

    symbol: str
    bar_type: str
    labeling_method: str
    train_days: int
    test_days: int
    step_days: int
    num_windows: int
    windows: list[WindowResult]
    stitched_timestamps: list[int]
    stitched_equity: list[float]
    stitched_drawdown: list[float]
    aggregate: dict  # metric -> {mean, std, ci_lower, ci_upper}
    avg_insample_recall: float
    avg_oos_accuracy: float
    overfitting_gap: float
    created_at: str


# ---------------------------------------------------------------------------
# Window boundary computation
# ---------------------------------------------------------------------------

def compute_window_boundaries(
    data_start_ms: int,
    data_end_ms: int,
    train_days: int = 90,
    test_days: int = 30,
    step_days: int = 30,
) -> list[tuple[int, int, int, int]]:
    """Return list of (train_start, train_end, test_start, test_end) in ms.

    Windows step forward by *step_days*.  Train windows may overlap.
    Test windows are non-overlapping.  Any window whose test_end
    exceeds *data_end_ms* is skipped.

    Parameters
    ----------
    data_start_ms : int
        Earliest available data timestamp in milliseconds.
    data_end_ms : int
        Latest available data timestamp in milliseconds.
    train_days : int
        Length of each training window in days.
    test_days : int
        Length of each test window in days.
    step_days : int
        How far to step forward between windows (in days).

    Returns
    -------
    list[tuple[int, int, int, int]]
        Each element is (train_start, train_end, test_start, test_end).
    """
    train_ms = train_days * _MS_PER_DAY
    test_ms = test_days * _MS_PER_DAY
    step_ms = step_days * _MS_PER_DAY

    windows: list[tuple[int, int, int, int]] = []
    offset = 0

    while True:
        train_start = data_start_ms + offset
        train_end = train_start + train_ms
        test_start = train_end
        test_end = test_start + test_ms

        if test_end > data_end_ms:
            break

        windows.append((train_start, train_end, test_start, test_end))
        offset += step_ms

    return windows


# ---------------------------------------------------------------------------
# Equity curve stitching
# ---------------------------------------------------------------------------

def stitch_equity_curves(
    windows: list[WindowResult],
    starting_capital: float = 10000.0,
) -> tuple[list[int], list[float], list[float]]:
    """Concatenate OOS equity curves across windows.

    Each window's equity is rescaled so it starts at the previous
    window's ending equity, producing a continuous stitched curve.

    Parameters
    ----------
    windows : list[WindowResult]
        Ordered list of window results with per-window equity curves.
    starting_capital : float
        Starting equity for the first window.

    Returns
    -------
    tuple[list[int], list[float], list[float]]
        (timestamps, equity, drawdown) for the full stitched curve.
    """
    all_ts: list[int] = []
    all_eq: list[float] = []
    all_dd: list[float] = []

    current_equity = starting_capital

    for w in windows:
        if not w.equity or not w.timestamps:
            continue

        # Rescale this window's equity to start at current_equity
        window_start_eq = w.equity[0]
        if window_start_eq <= 0:
            window_start_eq = 1.0  # avoid division by zero

        scale = current_equity / window_start_eq
        for i, (ts, eq) in enumerate(zip(w.timestamps, w.equity)):
            scaled_eq = eq * scale
            all_ts.append(ts)
            all_eq.append(round(scaled_eq, 2))

        # Update current equity to the end of this window
        if w.equity:
            current_equity = w.equity[-1] * scale

    # Recompute drawdown on the stitched curve
    peak = starting_capital
    for eq in all_eq:
        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak > 0 else 0.0
        all_dd.append(round(dd, 6))

    return all_ts, all_eq, all_dd


# ---------------------------------------------------------------------------
# Bootstrap aggregate statistics
# ---------------------------------------------------------------------------

def bootstrap_aggregate(
    windows: list[WindowResult],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute mean, std, and 95% CI for each metric across windows.

    Uses bootstrap resampling to estimate confidence intervals when the
    number of windows is small (often 3-10).

    Parameters
    ----------
    windows : list[WindowResult]
        Completed window results.
    n_resamples : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict]
        Mapping of metric name to {mean, std, ci_lower, ci_upper}.
    """
    metrics_to_agg = [
        "oos_accuracy", "oos_precision", "oos_recall",
        "sharpe", "max_dd", "total_return", "win_rate",
    ]

    rng = np.random.RandomState(seed)
    result: dict[str, dict] = {}
    n_windows = len(windows)

    for metric in metrics_to_agg:
        values = np.array([getattr(w, metric) for w in windows])
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if n_windows > 1 else 0.0

        # Bootstrap CI
        boot_means = np.empty(n_resamples)
        for b in range(n_resamples):
            sample = rng.choice(values, size=n_windows, replace=True)
            boot_means[b] = np.mean(sample)

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        result[metric] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    return result


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------

def run_walk_forward(
    symbol: str,
    bar_type: str,
    labeling_method: str,
    train_days: int = 90,
    test_days: int = 30,
    step_days: int = 30,
    data_dir: Path = TICK_DATA_DIR,
    bar_config: BarConfig | None = None,
    barrier_config: TripleBarrierConfig | None = None,
    training_config: TrainingConfig | None = None,
    starting_capital: float = 10000.0,
    fees_bps: float = 10.0,
) -> WalkForwardResult:
    """Full walk-forward validation loop.

    1. Scan tick-data directory to determine date range.
    2. Compute rolling window boundaries.
    3. For each window: train primary + meta models, predict OOS, simulate equity.
    4. Stitch equity curves and compute bootstrap aggregate statistics.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g. ``"BTCUSDT"``).
    bar_type : str
        Bar type (e.g. ``"time"``, ``"volume"``).
    labeling_method : str
        Labeling method (e.g. ``"triple_barrier"``).
    train_days : int
        Training window length in days.
    test_days : int
        Test window length in days.
    step_days : int
        Step size between windows in days.
    data_dir : Path
        Root directory containing tick CSV files.
    bar_config : BarConfig | None
        Bar generation config. Defaults used if ``None``.
    barrier_config : TripleBarrierConfig | None
        Barrier config for labeling. Defaults used if ``None``.
    training_config : TrainingConfig | None
        Training config. Optuna is forced off (n_trials=0).
    starting_capital : float
        Starting capital for equity simulation.
    fees_bps : float
        Transaction fee in basis points.

    Returns
    -------
    WalkForwardResult
        Aggregated walk-forward result.

    Raises
    ------
    ValueError
        If fewer than 3 windows complete successfully.
    """
    bar_config = bar_config or BarConfig()
    barrier_config = barrier_config or TripleBarrierConfig()
    training_config = training_config or TrainingConfig()

    # Force Optuna off for walk-forward (too slow per window)
    training_config.optuna_n_trials = 0

    # ------------------------------------------------------------------
    # Step 1: Scan data directory to find date range
    # ------------------------------------------------------------------
    files = iter_trade_files(symbol, data_dir)
    if not files:
        raise ValueError(f"No trade data found for {symbol} in {data_dir}")

    dates: list[datetime] = []
    for csv_path, _fmt in files:
        match = _DATE_PATTERN.search(csv_path.name)
        if match:
            dt = datetime.strptime(match.group(1), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            dates.append(dt)

    if not dates:
        raise ValueError(f"Could not extract dates from filenames for {symbol}")

    dates.sort()
    data_start_ms = int(dates[0].timestamp() * 1000)
    data_end_ms = int(dates[-1].timestamp() * 1000) + _MS_PER_DAY  # end of last day

    logger.info(
        f"Walk-forward {symbol}/{bar_type}/{labeling_method}: "
        f"data range {dates[0].date()} to {dates[-1].date()}, "
        f"train={train_days}d test={test_days}d step={step_days}d"
    )

    # ------------------------------------------------------------------
    # Step 2: Compute window boundaries
    # ------------------------------------------------------------------
    boundaries = compute_window_boundaries(
        data_start_ms, data_end_ms, train_days, test_days, step_days
    )
    logger.info(f"Generated {len(boundaries)} walk-forward windows")

    if len(boundaries) < 3:
        raise ValueError(
            f"Need at least 3 windows but only {len(boundaries)} fit in the data range. "
            f"Try reducing train_days/test_days or adding more data."
        )

    # Determine label_span for purged CV
    if labeling_method == "triple_barrier":
        label_span = barrier_config.max_holding_period
    elif labeling_method == "trend_scanning":
        label_span = 80  # max of default horizons [5, 10, 20, 40, 80]
    else:  # directional_change
        label_span = barrier_config.max_holding_period

    # ------------------------------------------------------------------
    # Step 3: Process each window
    # ------------------------------------------------------------------
    successful_windows: list[WindowResult] = []

    for idx, (train_start, train_end, test_start, test_end) in enumerate(boundaries):
        logger.info(
            f"Window {idx+1}/{len(boundaries)}: "
            f"train [{_ms_to_date(train_start)} - {_ms_to_date(train_end)}] "
            f"test [{_ms_to_date(test_start)} - {_ms_to_date(test_end)}]"
        )

        try:
            window_result = _process_single_window(
                idx=idx,
                symbol=symbol,
                bar_type=bar_type,
                labeling_method=labeling_method,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                data_dir=data_dir,
                bar_config=bar_config,
                barrier_config=barrier_config,
                training_config=training_config,
                label_span=label_span,
                starting_capital=starting_capital,
                fees_bps=fees_bps,
            )
            successful_windows.append(window_result)
            logger.info(
                f"  Window {idx+1} complete: "
                f"OOS acc={window_result.oos_accuracy:.4f} "
                f"sharpe={window_result.sharpe:.2f} "
                f"return={window_result.total_return:.2f}%"
            )

        except Exception as exc:
            logger.warning(
                f"  Window {idx+1} failed: {exc}",
                exc_info=True,
            )
            continue

    # ------------------------------------------------------------------
    # Step 4: Validate minimum window count
    # ------------------------------------------------------------------
    if len(successful_windows) < 3:
        raise ValueError(
            f"Only {len(successful_windows)} windows succeeded out of "
            f"{len(boundaries)}. Need at least 3 for meaningful statistics."
        )

    logger.info(
        f"Walk-forward complete: {len(successful_windows)}/{len(boundaries)} "
        f"windows succeeded"
    )

    # ------------------------------------------------------------------
    # Step 5: Stitch equity curves
    # ------------------------------------------------------------------
    stitched_ts, stitched_eq, stitched_dd = stitch_equity_curves(
        successful_windows, starting_capital
    )

    # ------------------------------------------------------------------
    # Step 6: Compute bootstrap aggregate
    # ------------------------------------------------------------------
    aggregate = bootstrap_aggregate(successful_windows)

    # ------------------------------------------------------------------
    # Step 7: Build result
    # ------------------------------------------------------------------
    avg_insample_recall = float(
        np.mean([w.primary_recall for w in successful_windows])
    )
    avg_oos_accuracy = float(
        np.mean([w.oos_accuracy for w in successful_windows])
    )
    overfitting_gap = avg_insample_recall - avg_oos_accuracy

    return WalkForwardResult(
        symbol=symbol,
        bar_type=bar_type,
        labeling_method=labeling_method,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        num_windows=len(successful_windows),
        windows=successful_windows,
        stitched_timestamps=stitched_ts,
        stitched_equity=stitched_eq,
        stitched_drawdown=stitched_dd,
        aggregate=aggregate,
        avg_insample_recall=round(avg_insample_recall, 4),
        avg_oos_accuracy=round(avg_oos_accuracy, 4),
        overfitting_gap=round(overfitting_gap, 4),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Internal: process a single window
# ---------------------------------------------------------------------------

def _process_single_window(
    idx: int,
    symbol: str,
    bar_type: str,
    labeling_method: str,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    data_dir: Path,
    bar_config: BarConfig,
    barrier_config: TripleBarrierConfig,
    training_config: TrainingConfig,
    label_span: int,
    starting_capital: float,
    fees_bps: float,
) -> WindowResult:
    """Train on [train_start, train_end), evaluate on [test_start, test_end).

    This mirrors the full training pipeline from ``backend.ml.training``
    but operates on a specific time window without saving model artifacts.
    """
    # ----- (a) Generate training bars -----
    train_bars = generate_bars_streaming(
        symbol, bar_type, bar_config, data_dir, train_start, train_end
    )
    if train_bars.empty or len(train_bars) < 100:
        raise ValueError(f"Insufficient training bars: {len(train_bars)}")

    # ----- (b) Generate labels -----
    labels_df = generate_labels(train_bars, labeling_method, barrier_config)
    if labels_df.empty:
        raise ValueError("No labels generated for training window")

    # ----- (c) Merge labels with bars, dropna -----
    train_bars = train_bars.merge(
        labels_df[["timestamp", "label"]], on="timestamp", how="inner"
    )
    train_bars = train_bars.dropna(subset=["label"])

    if len(train_bars) < 50:
        raise ValueError(f"Too few labeled training bars: {len(train_bars)}")

    # ----- (d) Compute features -----
    features = compute_all_features(
        train_bars, window=training_config.feature_window
    )

    # ----- (e) Forward-fill + drop NaN, align labels -----
    features = features.ffill()
    valid_mask = features.notna().all(axis=1)
    features = features[valid_mask]
    train_bars = train_bars.loc[features.index]

    if len(features) == 0:
        raise ValueError("No valid samples after feature warm-up")

    labels = train_bars["label"].values.astype(int)
    unique_labels = set(np.unique(labels))
    if not unique_labels.issubset({-1, 1}):
        raise ValueError(f"Unexpected labels: {unique_labels}")

    num_train_bars = len(train_bars)
    num_train_samples = len(features)

    # ----- (f) Compute sample weights -----
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

    # ----- (g) Train primary model (no Optuna) -----
    primary = PrimaryModel()
    primary.fit(features, labels, sample_weight=weights)

    primary_preds = primary.predict(features)
    primary_recall = float(
        recall_score(
            (labels + 1) // 2, (primary_preds + 1) // 2, zero_division=0
        )
    )

    # ----- (h) Train meta model on OOS primary predictions -----
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
            features.iloc[train_idx],
            labels[train_idx],
            sample_weight=weights[train_idx],
        )
        oos_preds[test_idx] = fold_model.predict(features.iloc[test_idx])

    meta_model = MetaLabelingModel()
    meta_model.fit(features, oos_preds, labels, sample_weight=weights)

    meta_preds_train = meta_model.predict(features, oos_preds)
    meta_precision = float(
        precision_score(
            MetaLabelingModel.construct_meta_labels(oos_preds, labels),
            meta_preds_train,
            zero_division=0,
        )
    )

    # ----- (i) Generate test bars -----
    test_bars = generate_bars_streaming(
        symbol, bar_type, bar_config, data_dir, test_start, test_end
    )
    if test_bars.empty or len(test_bars) < 10:
        raise ValueError(f"Insufficient test bars: {len(test_bars)}")

    num_test_bars = len(test_bars)

    # ----- (j) Compute test features -----
    # Generate test labels for OOS accuracy measurement
    test_labels_df = generate_labels(test_bars, labeling_method, barrier_config)
    if test_labels_df.empty:
        raise ValueError("No labels generated for test window")

    test_bars = test_bars.merge(
        test_labels_df[["timestamp", "label"]], on="timestamp", how="inner"
    )
    test_bars = test_bars.dropna(subset=["label"])

    test_features = compute_all_features(
        test_bars, window=training_config.feature_window
    )
    test_features = test_features.ffill()
    valid_test_mask = test_features.notna().all(axis=1)
    test_features = test_features[valid_test_mask]
    test_bars = test_bars.loc[test_features.index]

    if len(test_features) == 0:
        raise ValueError("No valid test samples after feature warm-up")

    test_labels = test_bars["label"].values.astype(int)

    # ----- (k) Run inference -----
    test_primary_preds = primary.predict(test_features)
    test_meta_probs = meta_model.predict_proba(test_features, test_primary_preds)

    # Filter: keep only signals where meta probability > 0.5
    signal_mask = test_meta_probs > 0.5
    signal_indices = np.where(signal_mask)[0]

    # OOS metrics on ALL test samples (before meta filtering)
    oos_accuracy = float((test_primary_preds == test_labels).mean())
    oos_precision_val = float(
        precision_score(
            (test_labels + 1) // 2,
            (test_primary_preds + 1) // 2,
            zero_division=0,
        )
    )
    oos_recall_val = float(
        recall_score(
            (test_labels + 1) // 2,
            (test_primary_preds + 1) // 2,
            zero_division=0,
        )
    )

    # ----- (l) Build signal dicts for equity simulation -----
    num_test_signals = len(signal_indices)

    if num_test_signals > 0:
        bet_sizes = bet_size_from_probability(test_meta_probs[signal_mask])
        signals_for_sim = []
        for i, sig_idx in enumerate(signal_indices):
            bar_row = test_bars.iloc[sig_idx]
            side = int(test_primary_preds[sig_idx])
            entry_price = float(bar_row["close"])
            vol = float(bar_row["high"] - bar_row["low"])
            ts = int(bar_row["timestamp"])
            size = float(bet_sizes[i])

            sig_dict = {
                "timestamp": ts,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "sl_price": None,
                "pt_price": None,
                "time_barrier": ts + barrier_config.max_holding_period * 60000,
                "labeling_method": labeling_method,
                "meta_probability": float(test_meta_probs[signal_indices[i]]),
            }

            # Add SL/PT for triple barrier
            if labeling_method == "triple_barrier" and vol > 0:
                sig_dict["sl_price"] = entry_price - side * vol * barrier_config.sl_multiplier
                sig_dict["pt_price"] = entry_price + side * vol * barrier_config.pt_multiplier

            signals_for_sim.append(sig_dict)

        signals_df = pd.DataFrame(signals_for_sim)

        # ----- (m) Simulate equity -----
        sim_result = simulate_equity(
            signals_df,
            test_bars,
            labeling_method,
            starting_capital,
            fees_bps,
        )

        sharpe = float(sim_result.metrics.get("sharpe", 0.0))
        max_dd = float(sim_result.metrics.get("max_dd", 0.0))
        total_return = float(sim_result.metrics.get("total_return", 0.0))
        win_rate = float(sim_result.metrics.get("win_rate", 0.0))
        num_trades = int(sim_result.metrics.get("num_trades", 0))
        eq_timestamps = sim_result.timestamps
        eq_equity = sim_result.equity
        eq_drawdown = sim_result.drawdown

    else:
        # No signals passed meta filter
        sharpe = 0.0
        max_dd = 0.0
        total_return = 0.0
        win_rate = 0.0
        num_trades = 0
        eq_timestamps = []
        eq_equity = []
        eq_drawdown = []

    # ----- (n) Build WindowResult -----
    return WindowResult(
        window_index=idx,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        num_train_bars=num_train_bars,
        num_test_bars=num_test_bars,
        num_train_samples=num_train_samples,
        num_test_signals=num_test_signals,
        primary_recall=round(primary_recall, 4),
        meta_precision=round(meta_precision, 4),
        oos_accuracy=round(oos_accuracy, 4),
        oos_precision=round(oos_precision_val, 4),
        oos_recall=round(oos_recall_val, 4),
        sharpe=round(sharpe, 2),
        max_dd=round(max_dd, 2),
        total_return=round(total_return, 2),
        win_rate=round(win_rate, 1),
        num_trades=num_trades,
        timestamps=eq_timestamps,
        equity=eq_equity,
        drawdown=eq_drawdown,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_date(ms: int) -> str:
    """Convert ms timestamp to YYYY-MM-DD string for logging."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
