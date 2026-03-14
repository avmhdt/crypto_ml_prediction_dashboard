"""Trend Scanning labeling method (Lopez de Prado, 2019).

For each bar, run OLS regression of close prices over multiple candidate
forward horizons.  Select the horizon with the maximum |t-value| of the
slope coefficient.  The label is ``sign(slope)`` at the selected horizon.

Labels are {-1, 1, NaN}. Bars where no horizon produces a t-value
above the minimum threshold are labeled NaN (excluded from training).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_t_value(y: np.ndarray) -> tuple[float, float]:
    """Compute OLS slope and its t-value for a simple linear regression.

    The independent variable is ``[0, 1, ..., len(y)-1]``.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (close prices over the horizon).

    Returns
    -------
    tuple[float, float]
        ``(slope, t_value)``
    """
    n = len(y)
    if n < 3:
        # Need at least 3 points for a meaningful t-stat.
        ret = y[-1] - y[0]
        slope = ret / max(n - 1, 1)
        return slope, slope  # use slope itself as a proxy t-value

    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    y_mean = y.mean()

    # Sxx = sum((x - x_mean)^2)  = n*(n-1)*(2n-1)/6 - n*x_mean^2
    # Using the closed-form: Sxx = n*(n^2-1)/12
    sxx = n * (n * n - 1) / 12.0

    sxy = np.dot(x - x_mean, y - y_mean)
    slope = sxy / sxx

    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    sse = np.dot(residuals, residuals)
    dof = n - 2
    mse = sse / dof if dof > 0 else 1e-30

    se_slope_sq = mse / sxx
    se_slope = np.sqrt(max(se_slope_sq, 1e-30))

    t_value = slope / se_slope
    return slope, t_value


def trend_scanning_labels(
    bars: pd.DataFrame,
    horizons: list[int] | None = None,
    min_t_value: float = 1.0,
) -> pd.DataFrame:
    """Apply the trend-scanning labeling method.

    Parameters
    ----------
    bars : pd.DataFrame
        Must contain columns ``[timestamp, close]``.
        Rows are assumed to be in chronological order.
    horizons : list[int], optional
        Forward-looking window lengths to evaluate.
        Defaults to ``[5, 10, 20, 40, 80]``.
    min_t_value : float, optional
        Minimum absolute t-value required to assign a directional label.
        If the best horizon's ``|t-value|`` is below this threshold, the
        label is NaN (excluded from training).  Default ``1.0`` (slope
        must be at least 1 standard error from zero).

    Returns
    -------
    pd.DataFrame
        Columns: ``[timestamp, label, best_horizon, t_value]``.
        ``label`` is in {-1, 1, NaN}. NaN when no horizon produces a
        statistically meaningful trend.
    """
    if horizons is None:
        horizons = [5, 10, 20, 40, 80]

    close = bars["close"].values.astype(np.float64)
    timestamps = bars["timestamp"].values
    n = len(close)

    labels = np.empty(n, dtype=np.float64)
    best_horizons = np.empty(n, dtype=np.int64)
    t_values = np.empty(n, dtype=np.float64)

    for i in range(n):
        best_abs_t = -1.0
        best_slope = 0.0
        best_t = 0.0
        best_h = horizons[0]

        for h in horizons:
            end = i + h
            if end >= n:
                # Not enough data for this horizon; use what remains.
                end = n - 1
            if end <= i:
                continue

            y = close[i : end + 1]
            slope, t_val = _ols_t_value(y)
            abs_t = abs(t_val)

            if abs_t > best_abs_t:
                best_abs_t = abs_t
                best_slope = slope
                best_t = t_val
                best_h = h

        # Determine label: require |t-value| >= min_t_value for a
        # directional label.  Below threshold = no meaningful trend.
        if best_abs_t >= min_t_value:
            label = 1 if best_slope > 0 else -1
        else:
            label = np.nan

        labels[i] = label
        best_horizons[i] = best_h
        t_values[i] = best_t

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "label": labels,
            "best_horizon": best_horizons,
            "t_value": t_values,
        }
    )
