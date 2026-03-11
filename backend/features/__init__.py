"""
Feature engineering module.

Exposes ``compute_all_features`` which orchestrates every feature sub-module
and returns a single wide DataFrame with 50+ columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features.price_features import (
    compute_ffd_weights,
    ffd_transform,
    find_min_d,
    cusum_filter,
    shannon_entropy,
    plugin_entropy,
    lempel_ziv_complexity,
    kontoyiannis_entropy,
)
from backend.features.microstructural_features import (
    order_book_imbalance,
    trade_flow_imbalance,
    amihud_lambda,
    roll_spread,
    corwin_schultz_spread,
)
from backend.features.volume_features import compute_volume_features
from backend.features.volatility_features import (
    rogers_satchell_vol,
    garman_klass_vol,
    yang_zhang_vol,
    realized_volatility,
    bipower_variation,
)
from backend.features.time_features import compute_time_features


def compute_all_features(
    bars: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    window: int = 20,
) -> pd.DataFrame:
    """Compute all features and return a single wide DataFrame.

    Orchestrates every feature sub-module:
        - Price features (FFD, velocity, acceleration, entropy proxies)
        - Volatility features (RS, GK, YZ, realized vol, bipower)
        - Volume features (dollar volume, duration, rolling stats)
        - Microstructural features (if *trades* is provided)
        - Time features (cyclical encoding)

    Parameters
    ----------
    bars : pd.DataFrame
        Bar data with columns: timestamp, open, high, low, close, volume,
        dollar_volume, tick_count, duration_us.
    trades : pd.DataFrame | None
        Trade-level data with columns: is_buyer_maker, volume (qty),
        price, time.  When ``None``, microstructural features that require
        trade data are skipped; bar-level proxies are used instead.
    window : int
        Rolling window for all windowed features.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame aligned to *bars* index with 50+ feature columns.
    """
    frames: list[pd.DataFrame] = []

    # ----- Price features -----
    log_close = np.log(bars["close"])

    # FFD — find minimum d then transform
    min_d = find_min_d(log_close, adf_pvalue=0.05, d_range=(0, 1), step=0.05)
    ffd_series = ffd_transform(log_close, d=min_d)

    price_df = pd.DataFrame(index=bars.index)
    price_df["ffd_close"] = ffd_series
    price_df["ffd_d"] = min_d  # constant, but useful metadata

    # Price velocity and acceleration (on FFD series)
    price_df["price_velocity"] = ffd_series.diff()
    price_df["price_acceleration"] = ffd_series.diff().diff()

    # Rolling entropy proxies on returns
    returns = log_close.diff()
    price_df["returns"] = returns
    price_df["returns_abs"] = returns.abs()

    # Rolling Shannon entropy of returns
    ent_vals = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        chunk = returns.iloc[start : i + 1].dropna()
        if len(chunk) >= 5:
            ent_vals.append(shannon_entropy(chunk, n_bins=10))
        else:
            ent_vals.append(np.nan)
    price_df["rolling_shannon_entropy"] = ent_vals

    # Rolling plugin entropy of returns
    pent_vals = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        chunk = returns.iloc[start : i + 1].dropna()
        if len(chunk) >= 5:
            pent_vals.append(plugin_entropy(chunk, n_bins=10))
        else:
            pent_vals.append(np.nan)
    price_df["rolling_plugin_entropy"] = pent_vals

    # Rolling Lempel-Ziv complexity of returns (binary: up/down)
    lz_vals = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        chunk = returns.iloc[start : i + 1].dropna()
        if len(chunk) >= 5:
            binary = (chunk > 0).astype(int).values
            lz_vals.append(lempel_ziv_complexity(binary))
        else:
            lz_vals.append(np.nan)
    price_df["rolling_lempel_ziv"] = lz_vals

    # Rolling Kontoyiannis entropy of returns (takes pd.Series, discretizes internally)
    kont_vals = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        chunk = returns.iloc[start : i + 1].dropna()
        if len(chunk) >= window + 2:
            kont_vals.append(kontoyiannis_entropy(chunk, window=min(len(chunk) // 2, window)))
        else:
            kont_vals.append(np.nan)
    price_df["rolling_kontoyiannis_entropy"] = kont_vals

    frames.append(price_df)

    # ----- Volatility features -----
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]

    vol_df = pd.DataFrame(index=bars.index)
    vol_df["rogers_satchell_vol"] = rogers_satchell_vol(o, h, l, c, window)
    vol_df["garman_klass_vol"] = garman_klass_vol(o, h, l, c, window)
    vol_df["yang_zhang_vol"] = yang_zhang_vol(o, h, l, c, window)
    vol_df["realized_volatility"] = realized_volatility(c, window)
    vol_df["bipower_variation"] = bipower_variation(c, window)

    # Volatility velocity and acceleration (on primary RS estimator)
    vol_df["vol_velocity"] = vol_df["rogers_satchell_vol"].diff()
    vol_df["vol_acceleration"] = vol_df["rogers_satchell_vol"].diff().diff()

    frames.append(vol_df)

    # ----- Volume features -----
    vol_feat = compute_volume_features(bars, window=window)
    frames.append(vol_feat)

    # ----- Microstructural features -----
    micro_df = pd.DataFrame(index=bars.index)

    if trades is not None and len(trades) > 0:
        # Aggregate trade-level features to bar level
        # Expect trades to have bar_index or be alignable; if not, compute
        # bar-level proxies from the bars themselves.
        if "is_buyer_maker" in trades.columns:
            is_bm = trades["is_buyer_maker"].astype(bool)
            trade_vol = trades["qty"] if "qty" in trades.columns else trades["volume"]

            obi = order_book_imbalance(is_bm, trade_vol, window=window)
            tfi = trade_flow_imbalance(is_bm, window=window)

            # Resample to bar-level if trades have a bar_index column
            if "bar_index" in trades.columns:
                obi_bar = obi.groupby(trades["bar_index"]).last().reindex(bars.index)
                tfi_bar = tfi.groupby(trades["bar_index"]).last().reindex(bars.index)
                micro_df["order_book_imbalance"] = obi_bar
                micro_df["trade_flow_imbalance"] = tfi_bar
            else:
                # Fallback: use last N trade values per bar as proxy
                micro_df["order_book_imbalance"] = np.nan
                micro_df["trade_flow_imbalance"] = np.nan

    # Bar-level microstructural features (always available)
    bar_returns = bars["close"].pct_change()
    dv = bars["dollar_volume"] if "dollar_volume" in bars.columns else bars["volume"] * bars["close"]
    micro_df["amihud_lambda"] = amihud_lambda(bar_returns, dv, window=window)
    micro_df["roll_spread"] = roll_spread(bars["close"], window=window)
    micro_df["corwin_schultz_spread"] = corwin_schultz_spread(
        bars["high"], bars["low"], window=window
    )

    # Higher-order stats on micro features
    micro_df["spread_variance"] = micro_df["roll_spread"].rolling(window, min_periods=1).var()
    micro_df["spread_skew"] = micro_df["roll_spread"].rolling(window, min_periods=1).skew()
    micro_df["spread_kurtosis"] = micro_df["roll_spread"].rolling(window, min_periods=1).kurt()

    frames.append(micro_df)

    # ----- Time features -----
    if "timestamp" in bars.columns:
        time_feat = compute_time_features(bars["timestamp"])
        frames.append(time_feat)

    # ----- Combine all -----
    features = pd.concat(frames, axis=1)

    # Rolling higher-order stats on returns
    features["returns_variance"] = returns.rolling(window, min_periods=1).var()
    features["returns_skew"] = returns.rolling(window, min_periods=1).skew()
    features["returns_kurtosis"] = returns.rolling(window, min_periods=1).kurt()

    # Autocorrelation of returns
    features["returns_autocorr"] = returns.rolling(window, min_periods=3).apply(
        lambda x: x.autocorr() if len(x) > 2 else np.nan, raw=False
    )

    return features
