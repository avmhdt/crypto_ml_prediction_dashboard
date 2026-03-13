"""
Price features: FFD (fractional differentiation), structural breaks, and entropy.

References:
    - AFML Ch.5: Fractionally Differentiated Features
    - AFML Ch.17: Structural Breaks (CUSUM, Chow, SADF, GSADF)
    - AFML Ch.18: Entropy Features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# FFD — Fixed-Width Window Fractional Differentiation
# ---------------------------------------------------------------------------


def compute_ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute FFD weights w_k = -w_{k-1} * (d - k + 1) / k.

    Weights are generated until |w_k| < threshold.

    Parameters
    ----------
    d : float
        Fractional differentiation order in (0, 1).
    threshold : float
        Minimum absolute weight to include.

    Returns
    -------
    np.ndarray
        1-D array of weights starting with w_0 = 1.
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights, dtype=np.float64)


def ffd_transform(
    series: pd.Series, d: float, threshold: float = 1e-5
) -> pd.Series:
    """Apply fixed-width window fractional differentiation to *series*.

    The transformation computes the dot product of the weight vector with a
    rolling window of the input series.  The first ``len(weights) - 1``
    observations are NaN (insufficient history).

    Parameters
    ----------
    series : pd.Series
        Raw price series (e.g. log prices).
    d : float
        Fractional differentiation order.
    threshold : float
        Minimum absolute weight for FFD window.

    Returns
    -------
    pd.Series
        Fractionally differentiated series (same index as *series*).
    """
    weights = compute_ffd_weights(d, threshold)
    width = len(weights)
    result = np.full(len(series), np.nan)
    values = series.values.astype(np.float64)
    # weights[0] corresponds to the current observation, weights[k] to lag k
    for i in range(width - 1, len(values)):
        result[i] = np.dot(weights, values[i - width + 1 : i + 1][::-1])
    return pd.Series(result, index=series.index, name="ffd")


def find_min_d(
    series: pd.Series,
    adf_pvalue: float = 0.05,
    d_range: tuple = (0, 1),
    step: float = 0.05,
) -> float:
    """Find the minimum *d* that makes the FFD series stationary.

    Iterates *d* from ``d_range[0]`` upward in increments of *step* and
    returns the first *d* for which the Augmented Dickey-Fuller test rejects
    the unit-root null at the given *adf_pvalue*.

    Parameters
    ----------
    series : pd.Series
        Raw (non-stationary) price series.
    adf_pvalue : float
        Significance level for the ADF test.
    d_range : tuple
        (min_d, max_d) search range.
    step : float
        Increment for *d*.

    Returns
    -------
    float
        Minimum *d* achieving stationarity.  Returns ``d_range[1]`` if no
        smaller value passes the ADF test.
    """
    d = d_range[0] + step  # d=0 is the original series, skip it
    while d <= d_range[1] + 1e-9:
        ffd_series = ffd_transform(series, d)
        clean = ffd_series.dropna()
        if len(clean) < 20:
            d += step
            continue
        adf_stat, pvalue, *_ = adfuller(clean, maxlag=1, regression="c", autolag=None)
        if pvalue < adf_pvalue:
            return round(d, 4)
        d += step
    return round(d_range[1], 4)


# ---------------------------------------------------------------------------
# Structural Breaks
# ---------------------------------------------------------------------------


def cusum_filter(series: pd.Series, threshold: float) -> list[int]:
    """Cumulative sum (CUSUM) filter for detecting structural breaks.

    Tracks positive and negative cumulative sums of demeaned returns.  When
    either |S+| or |S-| exceeds *threshold*, the index is emitted as an event
    and the accumulators are reset.

    Parameters
    ----------
    series : pd.Series
        Price or return series.
    threshold : float
        Cumulative deviation threshold for emitting an event.

    Returns
    -------
    list[int]
        Positional indices where events were detected.
    """
    diff = series.diff().dropna()
    mean_diff = diff.mean()
    demeaned = diff - mean_diff

    s_pos = 0.0
    s_neg = 0.0
    events: list[int] = []

    for i, val in enumerate(demeaned.values):
        s_pos = max(0.0, s_pos + val)
        s_neg = min(0.0, s_neg + val)

        if s_pos > threshold:
            events.append(i + 1)  # +1 because diff dropped first element
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -threshold:
            events.append(i + 1)
            s_pos = 0.0
            s_neg = 0.0

    return events


def chow_test(series: pd.Series, break_point: int) -> float:
    """Chow test F-statistic for a structural break at *break_point*.

    Compares a simple linear trend regression on the full sample against
    separate regressions before and after the break point.

    Parameters
    ----------
    series : pd.Series
        Time series (price or returns).
    break_point : int
        Positional index of the hypothesised break.

    Returns
    -------
    float
        F-statistic.  Higher values indicate a more significant break.
    """
    y = series.values.astype(np.float64)
    n = len(y)
    if break_point < 3 or break_point > n - 3:
        return 0.0

    x_full = np.column_stack([np.ones(n), np.arange(n, dtype=np.float64)])

    # Full model residuals
    beta_full = np.linalg.lstsq(x_full, y, rcond=None)[0]
    rss_full = np.sum((y - x_full @ beta_full) ** 2)

    # Pre-break
    x1 = x_full[:break_point]
    y1 = y[:break_point]
    beta1 = np.linalg.lstsq(x1, y1, rcond=None)[0]
    rss1 = np.sum((y1 - x1 @ beta1) ** 2)

    # Post-break
    x2 = x_full[break_point:]
    y2 = y[break_point:]
    beta2 = np.linalg.lstsq(x2, y2, rcond=None)[0]
    rss2 = np.sum((y2 - x2 @ beta2) ** 2)

    k = x_full.shape[1]  # number of regressors
    rss_restricted = rss_full
    rss_unrestricted = rss1 + rss2

    denom = rss_unrestricted / (n - 2 * k)
    if denom <= 0:
        return 0.0
    f_stat = ((rss_restricted - rss_unrestricted) / k) / denom
    return float(f_stat)


def sadf_test(series: pd.Series, min_window: int = 20) -> float:
    """Supremum Augmented Dickey-Fuller (SADF) test.

    Expands the estimation window from *min_window* to the full sample and
    returns the maximum ADF statistic observed.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    min_window : int
        Minimum number of observations in the expanding window.

    Returns
    -------
    float
        Maximum ADF t-statistic across all windows.
    """
    values = series.values.astype(np.float64)
    n = len(values)
    if n < min_window + 5:
        return np.nan

    max_adf = -np.inf
    for end in range(min_window, n + 1):
        window = values[:end]
        try:
            adf_stat, *_ = adfuller(window, maxlag=1, regression="c", autolag=None)
            if adf_stat > max_adf:
                max_adf = adf_stat
        except Exception:
            continue
    return float(max_adf)


def gsadf_test(
    series: pd.Series, min_window: int = 20, max_window: int = 500
) -> float:
    """Generalized SADF (GSADF) test.

    Outer loop varies the end point; inner loop varies the start point.
    The inner loop is capped at *max_window* observations to keep complexity
    at O(n * max_window) rather than O(n^2).

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    min_window : int
        Minimum number of observations per ADF window.
    max_window : int
        Maximum look-back for the inner loop (caps cost).

    Returns
    -------
    float
        Maximum ADF t-statistic across all (start, end) pairs.
    """
    values = series.values.astype(np.float64)
    n = len(values)
    if n < min_window + 5:
        return np.nan

    max_adf = -np.inf
    for end in range(min_window, n + 1):
        # Inner loop: vary start, but cap the window length
        earliest_start = max(0, end - max_window)
        latest_start = end - min_window
        for start in range(earliest_start, latest_start + 1):
            window = values[start:end]
            try:
                adf_stat, *_ = adfuller(
                    window, maxlag=1, regression="c", autolag=None
                )
                if adf_stat > max_adf:
                    max_adf = adf_stat
            except Exception:
                continue
    return float(max_adf)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def shannon_entropy(series: pd.Series, n_bins: int = 20) -> float:
    """Shannon entropy of a discretised series.

    Parameters
    ----------
    series : pd.Series
        Continuous-valued series.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    counts, _ = np.histogram(series.dropna().values, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def plugin_entropy(series: pd.Series, n_bins: int = 20) -> float:
    """Plug-in entropy estimator with Miller-Madow bias correction.

    Correction term: (m - 1) / (2 * N), where m = number of non-empty bins
    and N = number of samples.

    Parameters
    ----------
    series : pd.Series
        Continuous-valued series.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Bias-corrected Shannon entropy in nats.
    """
    clean = series.dropna().values
    n = len(clean)
    if n == 0:
        return 0.0
    counts, _ = np.histogram(clean, bins=n_bins)
    m = np.sum(counts > 0)  # non-empty bins
    h = shannon_entropy(series, n_bins)
    correction = (m - 1) / (2 * n)
    return float(h + correction)


def lempel_ziv_complexity(binary_series: np.ndarray) -> float:
    """LZ76 normalised complexity of a binary sequence.

    Parameters
    ----------
    binary_series : np.ndarray
        1-D array of 0s and 1s.

    Returns
    -------
    float
        Normalised LZ complexity: c(n) / (n / log2(n)).
    """
    s = binary_series.astype(int)
    n = len(s)
    if n <= 1:
        return 0.0

    i = 0
    c = 1  # complexity counter
    l = 1  # current substring length
    while i + l <= n:
        # Check if s[i:i+l] has been seen in s[0:i+l-1]
        substring = s[i : i + l]
        found = False
        for j in range(0, i + l - l + 1):
            if j + l > i + l - 1 and j < i:
                # Allow partial overlap but not full overlap at end
                pass
            if np.array_equal(s[j : j + l], substring) and j < i + l - l:
                if j + l <= i + l:
                    found = True
                    break
        if found:
            l += 1
        else:
            c += 1
            i += l
            l = 1

    # Normalise
    normaliser = n / np.log2(n) if n > 1 else 1.0
    return float(c / normaliser)


def kontoyiannis_entropy(series: pd.Series, window: int = 50) -> float:
    """Kontoyiannis' longest-match-length entropy estimator.

    For each position *i* in [window, n), find the length of the longest
    match of s[i:i+L] in s[i-window:i].  The entropy rate is estimated as
    the reciprocal of the average normalised match length.

    Parameters
    ----------
    series : pd.Series
        Continuous-valued series (will be discretised to binary above/below
        the rolling median).
    window : int
        Look-back window for matching.

    Returns
    -------
    float
        Estimated entropy rate.
    """
    clean = series.dropna().values
    n = len(clean)
    if n < window + 2:
        return 0.0

    # Discretise to binary: above or below median
    median = np.median(clean)
    binary = (clean >= median).astype(int)

    match_lengths: list[float] = []
    for i in range(window, n):
        # Find longest match of binary[i:i+L] in binary[i-window:i]
        max_L = min(n - i, window)
        L = 0
        for l_candidate in range(1, max_L + 1):
            pattern = binary[i : i + l_candidate]
            found = False
            search_end = i
            search_start = i - window
            for j in range(search_start, search_end):
                if j + l_candidate > search_end:
                    break
                if np.array_equal(binary[j : j + l_candidate], pattern):
                    found = True
                    break
            if found:
                L = l_candidate
            else:
                break
        match_lengths.append(L + 1)  # +1 as per Kontoyiannis convention

    if len(match_lengths) == 0:
        return 0.0

    avg_match = np.mean(match_lengths)
    log_window = np.log2(window) if window > 1 else 1.0
    # Entropy ~ log2(window) / avg_match_length
    return float(log_window / avg_match) if avg_match > 0 else 0.0


# ---------------------------------------------------------------------------
# Structural Breaks — Rolling Test Statistics
# ---------------------------------------------------------------------------


def bde_cusum_stat(log_prices: pd.Series, window: int = 50) -> pd.Series:
    """Brown-Durbin-Evans CUSUM test statistic on recursive residuals.

    For each rolling window, fits OLS on a time trend, computes recursive
    residuals, and returns max|CUSUM| / sqrt(window).

    Parameters
    ----------
    log_prices : pd.Series
        Log price series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling BDE CUSUM statistic.
    """
    values = log_prices.values.astype(np.float64)
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(window, n):
        y = values[i - window : i]
        t = np.arange(window, dtype=np.float64)
        X = np.column_stack([np.ones(window), t])

        # Recursive residuals: fit on first k obs, predict k+1
        min_obs = max(3, window // 4)
        residuals = []
        for k in range(min_obs, window):
            Xk = X[:k]
            yk = y[:k]
            try:
                beta = np.linalg.lstsq(Xk, yk, rcond=None)[0]
                pred = X[k] @ beta
                residuals.append(y[k] - pred)
            except np.linalg.LinAlgError:
                residuals.append(0.0)

        if len(residuals) < 2:
            continue
        res = np.array(residuals)
        sigma = np.std(res)
        if sigma < 1e-15:
            result[i] = 0.0
            continue
        cusum = np.cumsum(res / sigma)
        result[i] = float(np.max(np.abs(cusum)) / np.sqrt(len(cusum)))

    return pd.Series(result, index=log_prices.index, name="bde_cusum")


def csw_cusum_stat(log_prices: pd.Series, window: int = 50) -> pd.Series:
    """Chu-Stinchcombe-White CUSUM test statistic on levels.

    Measures cumulative deviations of log prices from the window mean,
    standardised by the window standard deviation.

    Parameters
    ----------
    log_prices : pd.Series
        Log price series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling CSW CUSUM statistic.
    """
    values = log_prices.values.astype(np.float64)
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(window, n):
        y = values[i - window : i]
        mu = np.mean(y)
        sigma = np.std(y, ddof=1)
        if sigma < 1e-15:
            result[i] = 0.0
            continue
        cusum = np.cumsum(y - mu) / sigma
        result[i] = float(np.max(np.abs(cusum)) / np.sqrt(window))

    return pd.Series(result, index=log_prices.index, name="csw_cusum")


def chow_type_df_stat(log_prices: pd.Series, window: int = 50) -> pd.Series:
    """Chow-type Dickey-Fuller test statistic.

    Splits each rolling window at the midpoint and returns the maximum
    ADF t-statistic from both sub-samples.

    Parameters
    ----------
    log_prices : pd.Series
        Log price series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling Chow-type DF statistic.
    """
    values = log_prices.values.astype(np.float64)
    n = len(values)
    result = np.full(n, np.nan)
    min_half = max(10, window // 4)

    for i in range(window, n):
        y = values[i - window : i]
        mid = window // 2
        if mid < min_half or (window - mid) < min_half:
            continue
        try:
            adf1, *_ = adfuller(y[:mid], maxlag=1, regression="c", autolag=None)
            adf2, *_ = adfuller(y[mid:], maxlag=1, regression="c", autolag=None)
            result[i] = float(max(adf1, adf2))
        except Exception:
            continue

    return pd.Series(result, index=log_prices.index, name="chow_type_df")


def martingale_tests(log_prices: pd.Series, window: int = 50) -> pd.DataFrame:
    """Sub- and super-martingale tests with polynomial, exponential, power trends.

    For each rolling window, regresses cumulative returns on trend functions
    and returns the t-statistic of the trend coefficient.

    Parameters
    ----------
    log_prices : pd.Series
        Log price series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Columns: martingale_poly_t, martingale_exp_t, martingale_power_t.
    """
    values = log_prices.values.astype(np.float64)
    n = len(values)
    poly_t = np.full(n, np.nan)
    exp_t = np.full(n, np.nan)
    power_t = np.full(n, np.nan)

    for i in range(window, n):
        y = values[i - window : i]
        # Cumulative returns from start of window
        cum_ret = y - y[0]
        t_arr = np.arange(1, window + 1, dtype=np.float64)

        for trend_type, trend_fn, out_arr in [
            ("poly", t_arr, poly_t),
            ("exp", np.exp(t_arr / window), exp_t),
            ("power", np.sqrt(t_arr), power_t),
        ]:
            X = np.column_stack([np.ones(window), trend_fn])
            try:
                beta, residuals_sum, rank, sv = np.linalg.lstsq(X, cum_ret, rcond=None)
                if rank < 2:
                    continue
                y_hat = X @ beta
                rss = np.sum((cum_ret - y_hat) ** 2)
                mse = rss / (window - 2)
                XtX_inv = np.linalg.inv(X.T @ X)
                se_beta = np.sqrt(mse * XtX_inv[1, 1])
                if se_beta > 1e-15:
                    out_arr[i] = float(beta[1] / se_beta)
            except (np.linalg.LinAlgError, ValueError):
                continue

    return pd.DataFrame(
        {"martingale_poly_t": poly_t, "martingale_exp_t": exp_t, "martingale_power_t": power_t},
        index=log_prices.index,
    )


def rolling_sadf_stat(log_prices: pd.Series, window: int = 50) -> pd.Series:
    """Rolling SADF (Supremum ADF) test statistic.

    Computes the SADF test over each rolling window.

    Parameters
    ----------
    log_prices : pd.Series
        Log price series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling SADF statistic.
    """
    values = log_prices.values.astype(np.float64)
    n = len(values)
    result = np.full(n, np.nan)
    min_sub = max(15, window // 3)

    for i in range(window, n):
        chunk = values[i - window : i]
        max_adf = -np.inf
        for end in range(min_sub, len(chunk) + 1):
            try:
                adf_stat, *_ = adfuller(
                    chunk[:end], maxlag=1, regression="c", autolag=None
                )
                if adf_stat > max_adf:
                    max_adf = adf_stat
            except Exception:
                continue
        if max_adf > -np.inf:
            result[i] = float(max_adf)

    return pd.Series(result, index=log_prices.index, name="rolling_sadf")
