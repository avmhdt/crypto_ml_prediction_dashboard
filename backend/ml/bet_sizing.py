"""Bet sizing: position size from meta-label probability.

Implements two approaches from ML for Asset Managers (2020):
1. Averaging across average bets — adjust for concurrency
2. Size discretization — round to practical levels
"""
import numpy as np
import pandas as pd


# Discrete bet size levels
DISCRETE_LEVELS = np.arange(0.0, 1.1, 0.1).round(1)


def bet_size_from_probability(meta_probabilities: np.ndarray,
                               num_classes: int = 2) -> np.ndarray:
    """Convert meta-label probability to raw bet size.

    Uses the sigmoid-like mapping from AFML:
    size = 2 * P(meta=1) - 1, clipped to [0, 1]

    Higher meta-probability → larger position.

    Parameters
    ----------
    meta_probabilities : array of P(meta=1) in [0, 1]
    num_classes : number of classes (default 2 for binary)

    Returns
    -------
    sizes : array of raw bet sizes in [0, 1]
    """
    sizes = 2 * meta_probabilities - 1
    return np.clip(sizes, 0.0, 1.0)


def average_across_average_bets(sizes: np.ndarray,
                                 concurrency: np.ndarray) -> np.ndarray:
    """Adjust bet sizes for concurrency (overlapping bets).

    ML for Asset Managers (2020): When multiple bets overlap in time,
    average the sizes to avoid over-concentration.

    Parameters
    ----------
    sizes : array of raw bet sizes
    concurrency : array of concurrency counts (how many bets overlap
                  at each signal's timestamp). concurrency >= 1.

    Returns
    -------
    adjusted_sizes : array of concurrency-adjusted sizes
    """
    concurrency = np.maximum(concurrency, 1)
    return sizes / concurrency


def discretize_bet_size(sizes: np.ndarray,
                        levels: np.ndarray = DISCRETE_LEVELS) -> np.ndarray:
    """Round continuous bet sizes to nearest discrete level.

    ML for Asset Managers (2020): Discretize for practical position management.
    Default levels: {0, 0.1, 0.2, ..., 1.0}

    Parameters
    ----------
    sizes : array of continuous sizes in [0, 1]
    levels : array of allowed discrete levels (sorted ascending)

    Returns
    -------
    discrete_sizes : array of sizes rounded to nearest level
    """
    # For each size, find nearest level
    sizes_clipped = np.clip(sizes, levels[0], levels[-1])
    indices = np.abs(sizes_clipped[:, None] - levels[None, :]).argmin(axis=1)
    return levels[indices]


def compute_average_exposure(active_positions: list[dict]) -> float:
    """Compute average exposure from concurrent active positions (AFML Ch.10).

    Parameters
    ----------
    active_positions : list of dicts, each with 'side' and 'size' keys

    Returns
    -------
    exposure : float in [-1.0, 1.0], clamped
    """
    if not active_positions:
        return 0.0
    raw = sum(p["side"] * p["size"] for p in active_positions) / len(active_positions)
    return max(-1.0, min(1.0, raw))


def discretize_exposure(exposure: float, step: float = 0.1) -> float:
    """Discretize signed exposure to nearest step increment.

    Parameters
    ----------
    exposure : float in [-1.0, 1.0]
    step : discretization step (default 0.1)

    Returns
    -------
    discretized : float rounded to nearest step
    """
    inv = round(1.0 / step)
    return round(exposure * inv) / inv


def compute_concurrency_at_signals(signal_timestamps: np.ndarray,
                                    label_spans: list[tuple[int, int]]) -> np.ndarray:
    """Compute how many active bets overlap at each signal's timestamp.

    Parameters
    ----------
    signal_timestamps : array of signal generation timestamps
    label_spans : list of (start_time, end_time) for each active bet

    Returns
    -------
    concurrency : array of overlap counts at each signal timestamp
    """
    concurrency = np.ones(len(signal_timestamps), dtype=int)
    for i, ts in enumerate(signal_timestamps):
        count = sum(1 for start, end in label_spans if start <= ts <= end)
        concurrency[i] = max(count, 1)
    return concurrency


def compute_bet_sizes(meta_probabilities: np.ndarray,
                      signal_timestamps: np.ndarray | None = None,
                      label_spans: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Full bet sizing pipeline.

    1. Convert meta-probability to raw size
    2. Adjust for concurrency (averaging across average bets)
    3. Discretize to practical levels

    Parameters
    ----------
    meta_probabilities : P(meta=1) from MetaLabelingModel
    signal_timestamps : timestamps of signals (for concurrency calc)
    label_spans : active bet spans (for concurrency calc)

    Returns
    -------
    sizes : array of discrete bet sizes in {0, 0.1, 0.2, ..., 1.0}
    """
    # Step 1: Raw size from probability
    raw_sizes = bet_size_from_probability(meta_probabilities)

    # Step 2: Concurrency adjustment
    if signal_timestamps is not None and label_spans is not None:
        concurrency = compute_concurrency_at_signals(
            signal_timestamps, label_spans
        )
        adjusted_sizes = average_across_average_bets(raw_sizes, concurrency)
    else:
        adjusted_sizes = raw_sizes

    # Step 3: Discretize
    return discretize_bet_size(adjusted_sizes)
