"""
Volume-derived features.

Computes dollar volume, duration, velocity, acceleration, and higher-order
rolling statistics (variance, skewness, kurtosis) from bar data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_features(bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute volume-related features from bar data.

    Parameters
    ----------
    bars : pd.DataFrame
        Bar data with at least ``volume``, ``close``, and ``duration_us``
        columns.  ``dollar_volume`` is used directly if present, otherwise
        computed as ``volume * close``.
    window : int
        Rolling window for statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like *bars* with columns:
        dollar_volume, log_dollar_volume, duration_us, log_duration,
        volume_velocity, volume_acceleration, volume_variance,
        volume_skew, volume_kurtosis.
    """
    result = pd.DataFrame(index=bars.index)

    # Dollar volume
    if "dollar_volume" in bars.columns:
        dv = bars["dollar_volume"].astype(np.float64)
    else:
        dv = (bars["volume"] * bars["close"]).astype(np.float64)
    result["dollar_volume"] = dv
    result["log_dollar_volume"] = np.log1p(dv.clip(lower=0))

    # Duration
    if "duration_us" in bars.columns:
        dur = bars["duration_us"].astype(np.float64)
    else:
        dur = pd.Series(np.zeros(len(bars)), index=bars.index, dtype=np.float64)
    result["duration_us"] = dur
    result["log_duration"] = np.log1p(dur.clip(lower=0))

    # Velocity = first difference of dollar volume
    vol_velocity = dv.diff()
    result["volume_velocity"] = vol_velocity

    # Acceleration = second difference of dollar volume
    vol_accel = vol_velocity.diff()
    result["volume_acceleration"] = vol_accel

    # Rolling statistics on dollar volume
    rolling_dv = dv.rolling(window, min_periods=1)
    result["volume_variance"] = rolling_dv.var()
    result["volume_skew"] = rolling_dv.skew()
    result["volume_kurtosis"] = rolling_dv.kurt()

    return result
