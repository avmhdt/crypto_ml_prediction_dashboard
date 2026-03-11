"""
Volatility estimators for 24/7 crypto markets.

Rogers-Satchell is the primary estimator (drift-independent, no overnight
gap required).  All estimators return rolling volatility series.

References:
    - Rogers & Satchell (1991): OHLC volatility estimator
    - Garman & Klass (1980): OHLC volatility estimator
    - Yang & Zhang (2000): Combined open-close / high-low estimator
    - Barndorff-Nielsen & Shephard (2004): Bipower variation
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rogers_satchell_vol(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rogers-Satchell volatility estimator (PRIMARY).

    RS = sqrt( rolling_mean( log(H/C)*log(H/O) + log(L/C)*log(L/O) ) )

    Drift-independent — ideal for 24/7 crypto markets with no overnight gap.

    Parameters
    ----------
    o, h, l, c : pd.Series
        Open, High, Low, Close price series.
    window : int
        Rolling window for averaging.

    Returns
    -------
    pd.Series
        Rolling Rogers-Satchell volatility.
    """
    log_hc = np.log(h / c)
    log_ho = np.log(h / o)
    log_lc = np.log(l / c)
    log_lo = np.log(l / o)

    rs_var = log_hc * log_ho + log_lc * log_lo

    rolling_mean = rs_var.rolling(window, min_periods=1).mean()
    # Clamp negative values (numerical noise) before sqrt
    return np.sqrt(rolling_mean.clip(lower=0)).rename("rogers_satchell_vol")


def garman_klass_vol(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Garman-Klass (1980) volatility estimator.

    GK = sqrt( rolling_mean( 0.5*log(H/L)^2 - (2*ln2-1)*log(C/O)^2 ) )

    Parameters
    ----------
    o, h, l, c : pd.Series
        Open, High, Low, Close price series.
    window : int
        Rolling window for averaging.

    Returns
    -------
    pd.Series
        Rolling Garman-Klass volatility.
    """
    log_hl = np.log(h / l)
    log_co = np.log(c / o)

    gk_var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2

    rolling_mean = gk_var.rolling(window, min_periods=1).mean()
    return np.sqrt(rolling_mean.clip(lower=0)).rename("garman_klass_vol")


def yang_zhang_vol(
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Yang-Zhang (2000) volatility estimator adapted for 24/7 crypto.

    In traditional markets, Yang-Zhang uses an overnight (close-to-open)
    component.  For 24/7 crypto there is no true overnight gap; we use the
    inter-bar close-to-open as the "overnight" proxy.

    YZ = sqrt( vol_overnight + k * vol_close + (1-k) * vol_RS )

    where k = 0.34 / (1.34 + (n+1)/(n-1)), and n = window.

    Parameters
    ----------
    o, h, l, c : pd.Series
        Open, High, Low, Close price series.
    window : int
        Rolling window for averaging.

    Returns
    -------
    pd.Series
        Rolling Yang-Zhang volatility.
    """
    # "Overnight" return: previous close to current open
    log_co = np.log(o / c.shift(1))
    # Open-to-close return
    log_oc = np.log(c / o)

    # Rogers-Satchell component (per-bar)
    log_hc = np.log(h / c)
    log_ho = np.log(h / o)
    log_lc = np.log(l / c)
    log_lo = np.log(l / o)
    rs_var = log_hc * log_ho + log_lc * log_lo

    n = window
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0)) if n > 1 else 0.34 / 2.34

    vol_overnight = log_co.rolling(window, min_periods=1).var()
    vol_close = log_oc.rolling(window, min_periods=1).var()
    vol_rs = rs_var.rolling(window, min_periods=1).mean()

    yz_var = vol_overnight + k * vol_close + (1.0 - k) * vol_rs.clip(lower=0)
    return np.sqrt(yz_var.clip(lower=0)).rename("yang_zhang_vol")


def realized_volatility(
    close: pd.Series, window: int = 20
) -> pd.Series:
    """Realized volatility: sqrt of the rolling sum of squared log returns.

    Parameters
    ----------
    close : pd.Series
        Close price series.
    window : int
        Rolling window.

    Returns
    -------
    pd.Series
        Rolling realized volatility.
    """
    log_ret = np.log(close / close.shift(1))
    rv = (log_ret ** 2).rolling(window, min_periods=1).sum()
    return np.sqrt(rv).rename("realized_volatility")


def bipower_variation(
    close: pd.Series, window: int = 20
) -> pd.Series:
    """Bipower variation — robust to jumps.

    BPV = (pi/2) * rolling_sum( |r_t| * |r_{t-1}| )

    Parameters
    ----------
    close : pd.Series
        Close price series.
    window : int
        Rolling window.

    Returns
    -------
    pd.Series
        Rolling bipower variation.
    """
    log_ret = np.log(close / close.shift(1))
    abs_ret = log_ret.abs()
    product = abs_ret * abs_ret.shift(1)
    bpv = (np.pi / 2.0) * product.rolling(window, min_periods=1).sum()
    return bpv.rename("bipower_variation")
