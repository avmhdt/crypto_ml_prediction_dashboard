"""
Sample weighting for ML training following AFML Chapter 4.

Implements three orthogonal weighting schemes from Lopez de Prado's
*Advances in Financial Machine Learning* (2018), Ch. 4:

  1. Average Uniqueness — down-weights labels that overlap many other labels.
  2. Return Attribution — weights labels by the absolute returns they span.
  3. Time Decay — exponential half-life decay favouring recent observations.

The final sample weight is the element-wise product of all three,
normalized to sum to 1 and clipped so no weight is exactly zero.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Concurrency matrix (internal helper)
# ---------------------------------------------------------------------------

def _bar_concurrency(
    label_spans: list[tuple[int, int]],
    num_bars: int,
) -> np.ndarray:
    """Compute per-bar concurrency count c_t using an O(n) sweep.

    ``c_t`` is the number of labels active at bar *t*.  Instead of
    materializing a full ``(num_labels, num_bars)`` matrix (which is
    370 GB for 222K labels), we use a difference-array approach that
    needs only ``O(num_bars)`` memory.

    Parameters
    ----------
    label_spans : list[tuple[int, int]]
        Half-open bar ranges ``[start, end)`` for each label.
    num_bars : int
        Total number of bars.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_bars,)`` — concurrency count at each bar.
    """
    diff = np.zeros(num_bars + 1, dtype=np.int64)
    for start, end in label_spans:
        s = max(start, 0)
        e = min(end, num_bars)
        if s < e:
            diff[s] += 1
            diff[e] -= 1
    return np.cumsum(diff[:num_bars])


# ---------------------------------------------------------------------------
# 2.  Average uniqueness weights
# ---------------------------------------------------------------------------

def compute_uniqueness_weights(
    label_spans: list[tuple[int, int]],
    num_bars: int,
) -> np.ndarray:
    """Compute average-uniqueness weights (AFML Ch. 4, Eq. 4.10).

    Uses O(num_bars) memory instead of O(num_labels * num_bars) by
    computing per-bar concurrency with a sweep, then iterating each
    label's span to compute its average uniqueness.

    Parameters
    ----------
    label_spans : list[tuple[int, int]]
        Half-open bar ranges ``[start, end)`` for each label.
    num_bars : int
        Total number of bars.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_labels,)`` with average-uniqueness weights.
    """
    # Step 1: O(num_bars) concurrency counts
    c_t = _bar_concurrency(label_spans, num_bars).astype(np.float64)

    # Precompute 1/c_t (avoid division in the loop)
    inv_c = np.zeros(num_bars, dtype=np.float64)
    mask = c_t > 0
    inv_c[mask] = 1.0 / c_t[mask]

    # Step 2: For each label, average 1/c_t over its span — O(total_span_length)
    num_labels = len(label_spans)
    weights = np.zeros(num_labels, dtype=np.float64)

    # Use prefix sums for O(1) per label instead of O(span_length)
    inv_c_cumsum = np.zeros(num_bars + 1, dtype=np.float64)
    inv_c_cumsum[1:] = np.cumsum(inv_c)

    for i, (start, end) in enumerate(label_spans):
        s = max(start, 0)
        e = min(end, num_bars)
        span_len = e - s
        if span_len > 0:
            weights[i] = (inv_c_cumsum[e] - inv_c_cumsum[s]) / span_len

    return weights


# ---------------------------------------------------------------------------
# 3.  Return-attribution weights
# ---------------------------------------------------------------------------

def compute_attribution_weights(
    returns: np.ndarray,
    label_spans: list[tuple[int, int]],
) -> np.ndarray:
    """Compute return-attribution weights (AFML Ch. 4).

    Each label's attributed return is the sum of absolute bar returns over
    the bars it spans.  The weight is the label's attributed return divided
    by the total attributed return across all labels.

    If every label has zero attributed return (e.g. flat prices), uniform
    weights ``1/N`` are returned.

    Parameters
    ----------
    returns : np.ndarray
        Per-bar returns, shape ``(num_bars,)``.
    label_spans : list[tuple[int, int]]
        Half-open bar ranges ``[start, end)`` for each label.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_labels,)`` summing to 1.
    """
    num_bars = len(returns)
    abs_ret = np.abs(returns)
    num_labels = len(label_spans)
    attributed = np.empty(num_labels, dtype=np.float64)

    for i, (start, end) in enumerate(label_spans):
        s = max(start, 0)
        e = min(end, num_bars)
        if s < e:
            attributed[i] = abs_ret[s:e].sum()
        else:
            attributed[i] = 0.0

    total = attributed.sum()
    if total == 0.0:
        return np.full(num_labels, 1.0 / num_labels, dtype=np.float64)

    return attributed / total


# ---------------------------------------------------------------------------
# 4.  Time-decay weights
# ---------------------------------------------------------------------------

def compute_time_decay_weights(
    timestamps: np.ndarray,
    half_life: float,
) -> np.ndarray:
    """Compute exponential time-decay weights.

    Newer observations receive higher weight.  The decay follows

        w_i = 2^{ -(t_max - t_i) / half_life }

    so that an observation exactly ``half_life`` time-units in the past
    receives half the weight of the most recent observation (AFML Ch. 4,
    sample-weight decay discussion).

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps (any numeric unit — milliseconds, seconds, etc.) for each
        label, shape ``(num_labels,)``.
    half_life : float
        Half-life in the same time units as *timestamps*.  If ``<= 0``, no
        decay is applied (all weights are 1).

    Returns
    -------
    np.ndarray
        Non-negative decay weights, shape ``(num_labels,)``.
    """
    if half_life <= 0:
        return np.ones(len(timestamps), dtype=np.float64)

    timestamps = np.asarray(timestamps, dtype=np.float64)
    t_max = timestamps.max()
    age = t_max - timestamps  # non-negative
    weights = np.power(2.0, -(age / half_life))
    return weights


# ---------------------------------------------------------------------------
# 5.  Combined sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(
    label_spans: list[tuple[int, int]],
    returns: np.ndarray,
    timestamps: np.ndarray,
    num_bars: int,
    half_life: float,
) -> np.ndarray:
    """Compute combined sample weights (AFML Ch. 4).

    The final weight for each sample is the product of three independent
    weighting components:

        final = uniqueness * attribution * decay

    The result is then normalized to sum to 1 and clipped so that no weight
    is smaller than ``1e-10`` (LightGBM requires strictly positive weights).

    Parameters
    ----------
    label_spans : list[tuple[int, int]]
        Half-open bar ranges ``[start, end)`` for each label.
    returns : np.ndarray
        Per-bar returns, shape ``(num_bars,)``.
    timestamps : np.ndarray
        Timestamps for each label, shape ``(num_labels,)``.
    num_bars : int
        Total number of bars.
    half_life : float
        Decay half-life (same units as *timestamps*).  ``<= 0`` disables decay.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_labels,)`` summing to 1, with all values >= 1e-10.
    """
    uniqueness = compute_uniqueness_weights(label_spans, num_bars)
    attribution = compute_attribution_weights(returns, label_spans)
    decay = compute_time_decay_weights(timestamps, half_life)

    combined = uniqueness * attribution * decay

    # Clip minimum to avoid exact zeros
    combined = np.clip(combined, 1e-10, None)

    # Normalize to sum to 1
    total = combined.sum()
    if total > 0:
        combined /= total

    return combined
