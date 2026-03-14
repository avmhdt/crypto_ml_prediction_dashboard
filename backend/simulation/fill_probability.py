"""Fill probability estimator using Avellaneda-Stoikov arrival rate model.

Models the probability that a limit order resting at a given distance
from mid-price will be filled within a time window, based on an
exponential arrival-rate intensity:

    lambda = A * exp(-kappa * delta)

where *delta* is the distance from mid-price and *kappa* controls how
rapidly fill probability decays with distance.  An optional Order Flow
Imbalance (OFI) adjustment scales the intensity up or down.
"""
import math


class FillProbabilityEstimator:
    """Avellaneda-Stoikov fill probability model.

    Args:
        kappa: Intensity decay parameter. Higher values mean fill
            probability drops more steeply with distance from mid.
        base_intensity: Base arrival intensity ``A`` at mid-price.
    """

    def __init__(
        self,
        kappa: float = 1.5,
        base_intensity: float = 1.0,
    ) -> None:
        self._kappa = kappa
        self._base_intensity = base_intensity
        self._beta_ofi: float = 0.5  # OFI sensitivity coefficient

    def estimate(
        self,
        delta_from_mid: float,
        dt_seconds: float,
        ofi: float = 0.0,
        volatility: float = 0.01,
    ) -> float:
        """Estimate fill probability for a limit order.

        Args:
            delta_from_mid: Absolute distance of limit price from
                mid-price (always positive).
            dt_seconds: Time window in seconds.
            ofi: Order Flow Imbalance signal (positive = buy pressure).
                Adjusts arrival intensity when non-zero.
            volatility: Current volatility estimate (reserved for
                future calibration, not used in base formula).

        Returns:
            Probability of fill in [0.0, 1.0].
        """
        if dt_seconds <= 0.0 or delta_from_mid < 0.0:
            return 0.0

        # Base intensity: lambda = A * exp(-kappa * delta)
        exponent = -self._kappa * delta_from_mid
        # Clamp exponent to prevent overflow
        exponent = max(-500.0, min(500.0, exponent))
        lam = self._base_intensity * math.exp(exponent)

        # OFI adjustment: scale intensity by exp(beta * ofi)
        if ofi != 0.0:
            ofi_exponent = self._beta_ofi * ofi
            ofi_exponent = max(-500.0, min(500.0, ofi_exponent))
            lam *= math.exp(ofi_exponent)

        # P(fill) = 1 - exp(-lambda * dt)
        fill_exponent = -lam * dt_seconds
        fill_exponent = max(-500.0, min(0.0, fill_exponent))
        prob = 1.0 - math.exp(fill_exponent)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, prob))

    def calibrate(
        self,
        recent_trades: list[dict],
        bbo_history: list[dict],
    ) -> None:
        """Calibrate kappa from empirical fill rate vs distance.

        Placeholder for future implementation.  Will fit kappa by
        binning historical trades by distance-from-mid and computing
        empirical fill rates per bin.

        Args:
            recent_trades: List of trade dicts with keys
                ``price``, ``qty``, ``time_ms``.
            bbo_history: List of BBO dicts with keys
                ``bid``, ``ask``, ``time_ms``.
        """
        # TODO: Implement empirical kappa fitting
        #   1. Bin trades by distance from mid at time of trade
        #   2. Compute fill frequency per bin
        #   3. Fit exponential decay to get kappa
        pass
