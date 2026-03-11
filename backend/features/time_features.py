"""
Cyclical time-of-day / calendar features.

Encodes temporal information using sin/cos pairs so that e.g. hour 23 is
close to hour 0, and December is close to January.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_time_features(timestamps: pd.Series) -> pd.DataFrame:
    """Compute cyclical time features from unix-millisecond timestamps.

    Parameters
    ----------
    timestamps : pd.Series
        Unix timestamps in **milliseconds** (int or float).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hour_sin, hour_cos, dow_sin, dow_cos,
        dom, month_sin, month_cos.
    """
    dt = pd.to_datetime(timestamps, unit="ms", utc=True)

    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    dow = dt.dayofweek.astype(np.float64)
    dom = dt.day.astype(np.float64)
    month = dt.month.astype(np.float64)

    result = pd.DataFrame(index=timestamps.index)
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    result["dom"] = dom
    result["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return result
