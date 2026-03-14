"""Optimal limit order placement using Avellaneda-Stoikov model.

Computes the ideal limit price offset from mid-price based on the
Avellaneda-Stoikov (2008) framework for market making.  The base
offset is derived from the model's reservation price formula:

    base_offset = (1 / kappa) * ln(1 + gamma / kappa)

Urgency and signal confidence then scale this offset to produce
more aggressive (closer to mid) or more passive (further from mid)
limit prices.
"""
import math


class LimitPriceEngine:
    """Compute optimal limit order prices.

    Args:
        gamma: Risk-aversion parameter.  Higher gamma means more
            conservative offset (further from mid).
        kappa: Order-arrival intensity decay.  Higher kappa means
            fill probability drops faster with distance, so the
            optimal offset is smaller.
        urgency: Urgency factor in [0, 1].  ``0`` = fully passive
            (maximum offset), ``1`` = aggressive (minimum offset).
    """

    def __init__(
        self,
        gamma: float = 0.1,
        kappa: float = 1.5,
        urgency: float = 0.5,
    ) -> None:
        self._gamma = gamma
        self._kappa = kappa
        self._urgency = urgency

    def compute_limit_price(
        self,
        mid_price: float,
        signal_side: int,
        signal_size: float,
        spread: float,
        volatility: float,
    ) -> float:
        """Compute optimal limit price for an order.

        Args:
            mid_price: Current mid-price.
            signal_side: ``1`` for buy signal, ``-1`` for sell signal.
            signal_size: Signal confidence/size in [0, 1].
                Higher values produce tighter (more aggressive) prices.
            spread: Current bid-ask spread in price terms.
            volatility: Current volatility estimate (used as context;
                the base offset is already calibrated via gamma/kappa).

        Returns:
            Optimal limit price.  For buy orders this is below mid;
            for sell orders this is above mid.
        """
        if self._kappa <= 0.0 or mid_price <= 0.0:
            return mid_price

        # Base offset from Avellaneda-Stoikov reservation price
        base_offset = (1.0 / self._kappa) * math.log(
            1.0 + self._gamma / self._kappa
        )

        # Urgency factor: higher urgency -> smaller offset (closer to mid)
        urgency_factor = 1.0 - 0.8 * self._urgency

        # Confidence factor: stronger signal -> smaller offset
        confidence_factor = 1.0 - 0.5 * signal_size

        adjusted_offset = base_offset * urgency_factor * confidence_factor

        # Apply offset relative to spread
        if signal_side == 1:
            # Buy: place below mid-price
            limit_price = mid_price - adjusted_offset * spread
        else:
            # Sell: place above mid-price
            limit_price = mid_price + adjusted_offset * spread

        return limit_price
