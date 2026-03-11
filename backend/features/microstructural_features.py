"""
Microstructural features adapted for crypto perpetual futures.

References:
    - AFML Ch.19: Microstructural Features
    - Roll (1984): Effective spread estimator
    - Corwin & Schultz (2012): Bid-ask spread from high-low prices
    - Amihud (2002): Illiquidity measure
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def order_book_imbalance(
    is_buyer_maker: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling order-book imbalance: (sell_vol - buy_vol) / (sell_vol + buy_vol).

    Note on Binance convention: ``is_buyer_maker=True`` means the *seller*
    was the taker (aggressor), so the trade is a **sell**.  Conversely,
    ``is_buyer_maker=False`` means the *buyer* was the taker (a **buy**).

    Parameters
    ----------
    is_buyer_maker : pd.Series
        Boolean series from trade data.
    volume : pd.Series
        Trade volume series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Order-book imbalance in [-1, 1].
    """
    # is_buyer_maker=True => sell trade, is_buyer_maker=False => buy trade
    sell_vol = volume.where(is_buyer_maker, 0.0)
    buy_vol = volume.where(~is_buyer_maker, 0.0)

    rolling_sell = sell_vol.rolling(window, min_periods=1).sum()
    rolling_buy = buy_vol.rolling(window, min_periods=1).sum()

    total = rolling_sell + rolling_buy
    imbalance = (rolling_sell - rolling_buy) / total.replace(0, np.nan)
    return imbalance.rename("order_book_imbalance")


def trade_flow_imbalance(
    is_buyer_maker: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling trade-flow imbalance (signed trade direction).

    Each trade is assigned a sign:
        -1 if ``is_buyer_maker=True``  (seller is aggressor → sell)
        +1 if ``is_buyer_maker=False`` (buyer is aggressor → buy)

    Parameters
    ----------
    is_buyer_maker : pd.Series
        Boolean series from trade data.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling mean of signed trades in [-1, 1].
    """
    signed = pd.Series(
        np.where(is_buyer_maker, -1.0, 1.0),
        index=is_buyer_maker.index,
        dtype=np.float64,
    )
    return signed.rolling(window, min_periods=1).mean().rename("trade_flow_imbalance")


def amihud_lambda(
    returns: pd.Series, dollar_volume: pd.Series, window: int = 20
) -> pd.Series:
    """Amihud (2002) illiquidity measure: rolling mean of |return| / dollar_volume.

    Parameters
    ----------
    returns : pd.Series
        Log or simple returns.
    dollar_volume : pd.Series
        Dollar volume per bar.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Amihud lambda (higher = less liquid).
    """
    safe_dv = dollar_volume.replace(0, np.nan)
    ratio = returns.abs() / safe_dv
    return ratio.rolling(window, min_periods=1).mean().rename("amihud_lambda")


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll (1984) effective spread estimator.

    Spread = 2 * sqrt(-cov(r_t, r_{t-1})) when the autocovariance is
    negative.  Set to 0 when cov >= 0 (no valid spread estimate).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int
        Rolling window size for covariance estimation.

    Returns
    -------
    pd.Series
        Estimated effective spread.
    """
    returns = close.pct_change()
    returns_lag = returns.shift(1)

    # Rolling covariance between r_t and r_{t-1}
    # Using manual rolling cov to avoid alignment issues
    r = returns.values.astype(np.float64)
    r_lag = returns_lag.values.astype(np.float64)
    n = len(r)
    spread = np.full(n, np.nan)

    for i in range(window, n):
        x = r[i - window + 1 : i + 1]
        y = r_lag[i - window + 1 : i + 1]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            spread[i] = 0.0
            continue
        cov = np.cov(x[mask], y[mask])[0, 1]
        if cov < 0:
            spread[i] = 2.0 * np.sqrt(-cov)
        else:
            spread[i] = 0.0

    return pd.Series(spread, index=close.index, name="roll_spread")


def corwin_schultz_spread(
    high: pd.Series, low: pd.Series, window: int = 20
) -> pd.Series:
    """Corwin & Schultz (2012) high-low spread estimator.

    Uses the relationship between daily high-low ranges and the bid-ask
    spread.  For 2-bar windows:

        beta  = sum of log(H_j/L_j)^2 over consecutive pairs
        gamma = log(max(H_{j}, H_{j+1}) / min(L_{j}, L_{j+1}))^2
        alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

    Parameters
    ----------
    high : pd.Series
        High prices per bar.
    low : pd.Series
        Low prices per bar.
    window : int
        Rolling window for averaging the spread estimate.

    Returns
    -------
    pd.Series
        Estimated bid-ask spread.
    """
    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)
    n = len(h)

    spread_raw = np.full(n, np.nan)

    for i in range(1, n):
        # Single-bar log range squared
        log_hl_0 = np.log(h[i - 1] / l[i - 1]) if l[i - 1] > 0 else 0.0
        log_hl_1 = np.log(h[i] / l[i]) if l[i] > 0 else 0.0

        beta = log_hl_0 ** 2 + log_hl_1 ** 2

        # 2-bar high/low
        h_max = max(h[i - 1], h[i])
        l_min = min(l[i - 1], l[i])
        gamma = (np.log(h_max / l_min) ** 2) if l_min > 0 else 0.0

        k = 3.0 - 2.0 * np.sqrt(2.0)
        if k == 0:
            spread_raw[i] = 0.0
            continue

        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

        exp_alpha = np.exp(alpha)
        denom = 1.0 + exp_alpha
        if denom > 0:
            s = 2.0 * (exp_alpha - 1.0) / denom
            spread_raw[i] = max(s, 0.0)
        else:
            spread_raw[i] = 0.0

    result = pd.Series(spread_raw, index=high.index)
    return result.rolling(window, min_periods=1).mean().rename("corwin_schultz_spread")
