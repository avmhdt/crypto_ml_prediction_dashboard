"""
Entropy features with multiple encodings (AFML Ch.18).

Provides binary, quantile, and sigma encodings for computing:
- Shannon entropy (MLE)
- Redundancy (1 - H/H_max)
- Mutual information I(X;Y) between returns and lagged returns
- Normalized variation of information
- Kontoyiannis LZ entropy estimate with each encoding
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------


def binary_encode(series: pd.Series) -> np.ndarray:
    """Encode as 0/1: 1 if value >= 0, else 0."""
    return (series.values >= 0).astype(np.int64)


def quantile_encode(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Encode into quantile bins (0 to n_bins-1).

    Falls back to rank-based encoding if qcut fails (too few unique values).
    """
    try:
        encoded = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
        return encoded.values.astype(np.float64)
    except (ValueError, IndexError):
        # Fallback: rank-based
        ranks = series.rank(method="min", na_option="keep")
        n = ranks.count()
        if n == 0:
            return np.full(len(series), np.nan)
        return np.floor(ranks.values * n_bins / (n + 1)).astype(np.float64)


def sigma_encode(series: pd.Series, n_sigma: int = 3) -> np.ndarray:
    """Encode into bins based on standard deviation from mean.

    Bins: ..., [-2sigma,-1sigma), [-1sigma,0), [0,1sigma), [1sigma,2sigma), ...
    Total bins = 2*n_sigma + 1 (including tails).
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std) or std < 1e-15:
        return np.zeros(len(series), dtype=np.float64)
    boundaries = [mean + i * std for i in range(-n_sigma, n_sigma + 1)]
    return np.digitize(series.values, boundaries).astype(np.float64)


# ---------------------------------------------------------------------------
# Core entropy functions (operate on pre-encoded discrete arrays)
# ---------------------------------------------------------------------------


def discrete_shannon_entropy(encoded: np.ndarray) -> float:
    """Shannon entropy H = -sum(p * log(p)) from a discrete array."""
    clean = encoded[~np.isnan(encoded)].astype(np.int64)
    if len(clean) == 0:
        return 0.0
    _, counts = np.unique(clean, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def discrete_redundancy(encoded: np.ndarray) -> float:
    """Redundancy R = 1 - H/H_max where H_max = log(n_symbols)."""
    clean = encoded[~np.isnan(encoded)].astype(np.int64)
    if len(clean) == 0:
        return 0.0
    n_symbols = len(np.unique(clean))
    if n_symbols <= 1:
        return 0.0
    h = discrete_shannon_entropy(encoded)
    h_max = np.log(n_symbols)
    return float(1.0 - h / h_max) if h_max > 0 else 0.0


def discrete_mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc = x[mask].astype(np.int64)
    yc = y[mask].astype(np.int64)
    if len(xc) < 2:
        return 0.0
    h_x = discrete_shannon_entropy(xc)
    h_y = discrete_shannon_entropy(yc)
    # Joint encoding: Cantor-like pairing
    max_y = int(np.max(np.abs(yc))) + 1
    joint = xc * (2 * max_y + 1) + yc
    h_xy = discrete_shannon_entropy(joint)
    return float(max(0.0, h_x + h_y - h_xy))


def discrete_nvi(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized variation of information: NVI = 1 - I(X;Y) / H(X,Y)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc = x[mask].astype(np.int64)
    yc = y[mask].astype(np.int64)
    if len(xc) < 2:
        return 1.0
    mi = discrete_mutual_info(xc, yc)
    max_y = int(np.max(np.abs(yc))) + 1
    joint = xc * (2 * max_y + 1) + yc
    h_xy = discrete_shannon_entropy(joint)
    return float(1.0 - mi / h_xy) if h_xy > 0 else 1.0


def discrete_kontoyiannis(encoded: np.ndarray, window: int = 20) -> float:
    """Kontoyiannis longest-match-length entropy on a discrete sequence."""
    clean = encoded[~np.isnan(encoded)].astype(np.int64)
    n = len(clean)
    if n < window + 2:
        return 0.0

    match_lengths: list[float] = []
    for i in range(window, n):
        max_L = min(n - i, window)
        L = 0
        for l_candidate in range(1, max_L + 1):
            pattern = clean[i : i + l_candidate]
            found = False
            for j in range(i - window, i):
                if j + l_candidate > i:
                    break
                if np.array_equal(clean[j : j + l_candidate], pattern):
                    found = True
                    break
            if found:
                L = l_candidate
            else:
                break
        match_lengths.append(L + 1)

    if len(match_lengths) == 0:
        return 0.0
    avg_match = np.mean(match_lengths)
    log_window = np.log2(window) if window > 1 else 1.0
    return float(log_window / avg_match) if avg_match > 0 else 0.0


# ---------------------------------------------------------------------------
# Rolling computation
# ---------------------------------------------------------------------------


def compute_entropy_features(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """Compute all entropy features with binary, quantile, and sigma encodings.

    For each encoding and each rolling window, computes:
    - Shannon entropy
    - Redundancy
    - Mutual information (between returns and lagged returns)
    - Normalized variation of information
    - Kontoyiannis entropy estimate

    Parameters
    ----------
    returns : pd.Series
        Log returns series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        15 columns: {encoding}_{metric} for each combination.
    """
    n = len(returns)
    lagged = returns.shift(1)

    encoders = {
        "binary": binary_encode,
        "quantile": quantile_encode,
        "sigma": sigma_encode,
    }
    metrics = ["shannon", "redundancy", "mi", "nvi", "kontoyiannis"]

    # Preallocate
    columns = {}
    for enc_name in encoders:
        for metric in metrics:
            columns[f"entropy_{enc_name}_{metric}"] = np.full(n, np.nan)

    for i in range(window, n):
        start = i - window
        chunk = returns.iloc[start:i]
        chunk_lag = lagged.iloc[start:i]

        # Skip if too many NaNs
        valid = chunk.dropna()
        if len(valid) < 5:
            continue

        for enc_name, enc_fn in encoders.items():
            enc = enc_fn(chunk.dropna())
            enc_lag = enc_fn(chunk_lag.dropna())

            # Align lengths for MI/NVI
            min_len = min(len(enc), len(enc_lag))
            if min_len < 3:
                continue
            enc_aligned = enc[:min_len]
            enc_lag_aligned = enc_lag[:min_len]

            columns[f"entropy_{enc_name}_shannon"][i] = discrete_shannon_entropy(enc)
            columns[f"entropy_{enc_name}_redundancy"][i] = discrete_redundancy(enc)
            columns[f"entropy_{enc_name}_mi"][i] = discrete_mutual_info(
                enc_aligned, enc_lag_aligned
            )
            columns[f"entropy_{enc_name}_nvi"][i] = discrete_nvi(
                enc_aligned, enc_lag_aligned
            )
            kont_window = min(len(enc) // 3, max(5, window // 3))
            if len(enc) >= kont_window + 2 and kont_window >= 3:
                columns[f"entropy_{enc_name}_kontoyiannis"][i] = discrete_kontoyiannis(
                    enc, window=kont_window
                )

    return pd.DataFrame(columns, index=returns.index)
