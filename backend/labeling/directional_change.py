"""Directional Change (DC) labeling method.

Operates in *intrinsic time*: events are triggered by price reversals of
magnitude ``theta`` (as a proportion of price) from running extrema.

Two modes of operation:
  - In *upturn* mode: if price drops by theta from the running high, a
    **downturn event** is emitted (label = -1).
  - In *downturn* mode: if price rises by theta from the running low, an
    **upturn event** is emitted (label = 1).

Multi-scale variant runs the algorithm for each theta in parallel and
returns events from all scales.

ALL labels are binary {-1, 1}. Never 0, never NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dc_single_theta(
    close: np.ndarray,
    timestamps: np.ndarray,
    theta: float,
) -> list[tuple[int, int, float, float]]:
    """Run directional-change detection for a single theta.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    timestamps : np.ndarray
        Corresponding timestamps.
    theta : float
        Reversal threshold as a proportion of price (e.g. 0.01 = 1%).

    Returns
    -------
    list of (bar_index, label, theta, overshoot)
        Each entry represents one DC event.
    """
    if len(close) < 2:
        return []

    events: list[tuple[int, int, float, float]] = []

    # Initialize mode from the first move.
    if close[1] >= close[0]:
        mode = 1  # upturn: tracking highs, waiting for downturn reversal
        extreme = close[0]
        extreme_idx = 0
        dc_price = close[0]
    else:
        mode = -1  # downturn: tracking lows, waiting for upturn reversal
        extreme = close[0]
        extreme_idx = 0
        dc_price = close[0]

    for i in range(1, len(close)):
        p = close[i]

        if mode == 1:
            # Upturn mode: update running high.
            if p > extreme:
                extreme = p
                extreme_idx = i
            # Check for downturn reversal.
            if extreme > 0 and (extreme - p) / extreme >= theta:
                # Downturn event.
                # Overshoot = how much price continued beyond the DC point
                # before reversing.
                overshoot = (extreme - dc_price) / (dc_price * theta) if dc_price * theta > 0 else 0.0
                events.append((i, -1, theta, overshoot))
                # Switch to downturn mode.
                mode = -1
                dc_price = p
                extreme = p
                extreme_idx = i
        else:
            # Downturn mode: update running low.
            if p < extreme:
                extreme = p
                extreme_idx = i
            # Check for upturn reversal.
            if extreme > 0 and (p - extreme) / extreme >= theta:
                # Upturn event.
                overshoot = (dc_price - extreme) / (dc_price * theta) if dc_price * theta > 0 else 0.0
                events.append((i, 1, theta, overshoot))
                # Switch to upturn mode.
                mode = 1
                dc_price = p
                extreme = p
                extreme_idx = i

    return events


def directional_change_labels(
    bars: pd.DataFrame,
    thetas: list[float],
) -> pd.DataFrame:
    """Detect directional-change events at one or more theta scales.

    Parameters
    ----------
    bars : pd.DataFrame
        Must contain columns ``[timestamp, close]``.
    thetas : list[float]
        Reversal thresholds as proportions (e.g. ``[0.01, 0.02]``).

    Returns
    -------
    pd.DataFrame
        Columns: ``[timestamp, label, theta, overshoot]``.
        One row per DC event. ``label`` is always in {-1, 1}.
        Sorted by timestamp, then theta.
    """
    close = bars["close"].values.astype(np.float64)
    timestamps = bars["timestamp"].values

    all_events: list[dict] = []

    for th in thetas:
        events = _dc_single_theta(close, timestamps, th)
        for idx, label, theta_val, overshoot in events:
            all_events.append(
                {
                    "timestamp": timestamps[idx],
                    "label": label,
                    "theta": theta_val,
                    "overshoot": overshoot,
                }
            )

    if not all_events:
        # No events detected — return empty DataFrame with correct schema.
        return pd.DataFrame(
            columns=["timestamp", "label", "theta", "overshoot"]
        )

    result = pd.DataFrame(all_events)
    result = result.sort_values(["timestamp", "theta"]).reset_index(drop=True)
    return result


def dc_labels_from_volatility(
    bars: pd.DataFrame,
    vol_window: int = 20,
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Directional-change labeling with theta derived from rolling volatility.

    Theta values are computed as multiples of the rolling standard deviation
    of log returns, matching the SPEC recommendation of using volatility-
    scaled theta levels (Section 10, Decision 5).

    Parameters
    ----------
    bars : pd.DataFrame
        Must contain columns ``[timestamp, close]``.
    vol_window : int
        Rolling window length for volatility estimation.
    multipliers : list[float], optional
        Multiples of volatility to use as theta levels.
        Defaults to ``[0.5, 1.0, 1.5, 2.0, 3.0]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[timestamp, label, theta, overshoot]``.
        ``label`` is always in {-1, 1}.
    """
    if multipliers is None:
        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]

    close = bars["close"].values.astype(np.float64)

    # Rolling volatility of log returns.
    log_ret = np.diff(np.log(close))
    if len(log_ret) < vol_window:
        vol_estimate = np.std(log_ret) if len(log_ret) > 0 else 0.01
    else:
        # Use the most recent window for a representative volatility.
        vol_estimate = np.std(log_ret[-vol_window:])

    # Ensure a positive floor.
    vol_estimate = max(vol_estimate, 1e-8)

    thetas = [m * vol_estimate for m in multipliers]
    # Clip extremely small thetas that would generate excessive events.
    thetas = [max(t, 1e-6) for t in thetas]

    return directional_change_labels(bars, thetas)
