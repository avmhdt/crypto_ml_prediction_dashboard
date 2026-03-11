"""Triple Barrier labeling method (AFML Ch. 3).

Computes labels by setting three barriers around each bar's close price:
  - Stop-loss (SL): close - sl_multiplier * daily_vol
  - Profit-target (PT): close + pt_multiplier * daily_vol
  - Time barrier: bar index + max_holding_period

The first barrier touched determines the label:
  SL hit  -> -1
  PT hit  ->  1
  Time exit -> sign(return), with zero-return fallback to last tick direction

ALL labels are binary {-1, 1}. Never 0, never NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import TripleBarrierConfig


def _daily_volatility(close: pd.Series, span: int) -> pd.Series:
    """EWMA standard deviation of log returns.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    span : int
        EWMA span for the volatility estimate.

    Returns
    -------
    pd.Series
        Per-bar volatility estimate (same index as *close*).
    """
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.ewm(span=span, min_periods=max(1, span // 2)).std()
    # Forward-fill any leading NaNs with the first valid value so every bar
    # has a usable volatility estimate.
    vol = vol.ffill().bfill()
    # Guarantee a positive floor so barriers are never degenerate.
    vol = vol.clip(lower=1e-12)
    return vol


def triple_barrier_labels(
    bars: pd.DataFrame,
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Apply the triple-barrier labeling method to a bar DataFrame.

    Parameters
    ----------
    bars : pd.DataFrame
        Must contain columns ``[timestamp, open, high, low, close, volume]``.
        Rows are assumed to be in chronological order.
    config : TripleBarrierConfig, optional
        Barrier configuration.  Uses defaults when *None*.

    Returns
    -------
    pd.DataFrame
        Columns: ``[timestamp, label, sl_price, pt_price, time_barrier_ts]``.
        ``label`` is always in {-1, 1}.
    """
    if config is None:
        config = TripleBarrierConfig()

    close = bars["close"].values.astype(np.float64)
    high = bars["high"].values.astype(np.float64)
    low = bars["low"].values.astype(np.float64)
    timestamps = bars["timestamp"].values
    n = len(close)

    vol = _daily_volatility(
        pd.Series(close, dtype=np.float64), span=config.volatility_window
    ).values

    sl_mult = config.sl_multiplier
    pt_mult = config.pt_multiplier
    max_hold = config.max_holding_period

    labels = np.empty(n, dtype=np.int64)
    sl_prices = np.empty(n, dtype=np.float64)
    pt_prices = np.empty(n, dtype=np.float64)
    time_barrier_indices = np.empty(n, dtype=np.int64)

    for i in range(n):
        sl_price = close[i] - sl_mult * vol[i]
        pt_price = close[i] + pt_mult * vol[i]
        tb_idx = min(i + max_hold, n - 1)

        sl_prices[i] = sl_price
        pt_prices[i] = pt_price
        time_barrier_indices[i] = tb_idx

        label = 0  # sentinel; will be resolved below
        hit = False

        # Scan forward from bar i+1 to the time barrier (inclusive).
        for j in range(i + 1, tb_idx + 1):
            # Check SL first (conservative: loss triggers before gain within
            # the same bar).
            if low[j] <= sl_price:
                label = -1
                hit = True
                break
            if high[j] >= pt_price:
                label = 1
                hit = True
                break

        if not hit:
            # Time barrier reached without SL/PT touch.
            ret = close[tb_idx] - close[i]
            if ret > 0:
                label = 1
            elif ret < 0:
                label = -1
            else:
                # Zero return at time barrier: use last tick direction.
                # Walk backward from tb_idx to find a non-zero move.
                direction = 0
                for k in range(tb_idx, i, -1):
                    diff = close[k] - close[k - 1]
                    if diff > 0:
                        direction = 1
                        break
                    elif diff < 0:
                        direction = -1
                        break
                # Ultimate fallback: if everything was perfectly flat,
                # assign +1 (arbitrary but valid binary label).
                label = direction if direction != 0 else 1

        labels[i] = label

    # Map time-barrier indices to their timestamps.
    time_barrier_ts = timestamps[time_barrier_indices]

    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "label": labels,
            "sl_price": sl_prices,
            "pt_price": pt_prices,
            "time_barrier_ts": time_barrier_ts,
        }
    )
    return result
