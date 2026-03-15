"""Synthetic tick generator with controllable signal-to-noise ratio.

Generates artificial tick data using Geometric Brownian Motion with
regime-switching drift and correlated order flow.  The Sharpe ratio
parameter controls the signal strength — at Sharpe 0 the price is a
pure random walk; at higher values a planted directional signal emerges
that the AFML pipeline's features can detect.

References
----------
- Lopez de Prado, *Advances in Financial Machine Learning* (2018), Ch. 13
- Columbia GBM notes (Karl Sigman): SNR = mu / sigma
"""
import numpy as np
import pandas as pd


def generate_synthetic_ticks(
    n_ticks: int = 500_000,
    sharpe: float = 0.0,
    base_price: float = 40_000.0,
    annual_volatility: float = 0.80,
    regime_half_life_ticks: int = 25_000,
    ticks_per_second: float = 10.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic tick stream with controllable signal strength.

    Parameters
    ----------
    n_ticks : int
        Number of ticks to generate.
    sharpe : float
        Annualized Sharpe ratio.  0 = pure noise, 5 = very strong signal.
        Maps to GBM drift: ``mu = sharpe * sigma``.
    base_price : float
        Starting price level.
    annual_volatility : float
        Annualized volatility (e.g. 0.80 = 80%).
    regime_half_life_ticks : int
        Expected number of ticks before regime switches direction.
    ticks_per_second : float
        Tick arrival rate (used for timestamp generation).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``[price, qty, time, is_buyer_maker]``
    """
    rng = np.random.RandomState(seed)

    # Time parameters
    ticks_per_year = ticks_per_second * 86_400 * 365
    dt = 1.0 / ticks_per_year
    sigma_tick = annual_volatility * np.sqrt(dt)

    # Drift from Sharpe: mu_annual = sharpe * sigma_annual
    mu_annual = abs(sharpe) * annual_volatility
    mu_tick = mu_annual * dt

    # Regime switching: +1 (uptrend) or -1 (downtrend)
    # Transition probability per tick
    p_switch = 1.0 - np.exp(-1.0 / max(regime_half_life_ticks, 1))

    regimes = np.ones(n_ticks, dtype=np.float64)
    current_regime = rng.choice([-1.0, 1.0])
    for i in range(n_ticks):
        if rng.random() < p_switch:
            current_regime *= -1.0
        regimes[i] = current_regime

    # Price process: GBM with regime-modulated drift
    noise = rng.standard_normal(n_ticks)
    log_returns = regimes * (mu_tick - sigma_tick**2 / 2) + sigma_tick * noise

    log_prices = np.cumsum(log_returns)
    log_prices = np.insert(log_prices, 0, 0.0)[:-1]  # shift so first = 0
    prices = base_price * np.exp(log_prices)

    # Volume: exponential distribution (BTC-like trade sizes)
    qtys = rng.exponential(0.1, n_ticks).astype(np.float64)

    # Timestamps: monotonically increasing ms
    start_time = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
    interval_ms = 1000.0 / ticks_per_second
    times = (start_time + np.arange(n_ticks, dtype=np.int64) * int(interval_ms))

    # Order flow: correlated with regime direction
    # Buyer-aggressor → is_buyer_maker=False (price goes up)
    # In uptrend regime: more buyer-aggressors → lower P(is_buyer_maker)
    alpha = 0.15 * min(abs(sharpe), 3.0) / 3.0  # caps at ±0.15 bias
    p_ibm = np.where(
        regimes > 0,
        0.5 - alpha,   # uptrend: fewer buyer-makers (more buyer-aggressors)
        0.5 + alpha,   # downtrend: more buyer-makers (more seller-aggressors)
    )
    is_buyer_maker = rng.random(n_ticks) < p_ibm

    return pd.DataFrame({
        "price": prices,
        "qty": qtys,
        "time": times,
        "is_buyer_maker": is_buyer_maker,
    })
