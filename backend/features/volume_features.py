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

    # Duration velocity and acceleration (on log scale)
    log_dur = result["log_duration"]
    result["duration_velocity"] = log_dur.diff()
    result["duration_acceleration"] = log_dur.diff().diff()

    # Rolling statistics on duration
    rolling_dur = dur.rolling(window, min_periods=1)
    result["duration_variance"] = rolling_dur.var()
    result["duration_skew"] = rolling_dur.skew()
    result["duration_kurtosis"] = rolling_dur.kurt()

    return result


def compute_price_stats(bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling higher-order statistics on log prices.

    Computes standard deviation, skewness, and kurtosis of log prices
    over a rolling window.  These capture the distribution shape of the
    price level itself (not returns).

    Parameters
    ----------
    bars : pd.DataFrame
        Bar data with ``close`` column.
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Columns: log_price_std, log_price_skew, log_price_kurtosis.
    """
    result = pd.DataFrame(index=bars.index)
    log_prices = np.log(bars["close"].astype(np.float64))

    rolling_lp = log_prices.rolling(window, min_periods=1)
    result["log_price_std"] = rolling_lp.std()
    result["log_price_skew"] = rolling_lp.skew()
    result["log_price_kurtosis"] = rolling_lp.kurt()

    return result
