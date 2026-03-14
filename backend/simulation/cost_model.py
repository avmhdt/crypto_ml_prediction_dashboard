"""Transaction cost model for realistic order fill simulation.

Decomposes the total cost of a trade into exchange fees, spread cost,
slippage (square-root impact model), market impact, and funding.
Designed for Binance USDT-M futures with VIP-tier fee support.

Slippage model (Almgren & Chriss, 2001):
    I(Q) = Y * sigma * sqrt(Q / V) * notional

where Y is a scaling constant, sigma is daily volatility, Q is order
size, and V is average daily volume.
"""
import math
from dataclasses import dataclass

from backend.simulation.config import SimulationConfig


@dataclass
class TransactionCost:
    """Breakdown of costs for a single transaction.

    All cost fields are in absolute dollar terms (positive = cost,
    negative = rebate).
    """
    exchange_fee: float = 0.0
    funding_cost: float = 0.0
    spread_cost: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    total: float = 0.0
    notional: float = 0.0

    @property
    def total_bps(self) -> float:
        """Total cost expressed in basis points of notional."""
        if self.notional <= 0.0:
            return 0.0
        return (self.total / self.notional) * 10_000.0


class CostModel:
    """Compute transaction costs for simulated order fills.

    Args:
        config: Simulation configuration providing fee rates,
            slippage scaling, and other cost parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config

    def compute_exchange_fee(
        self,
        notional: float,
        order_type: str = "taker",
    ) -> float:
        """Compute exchange fee for a trade.

        Args:
            notional: Trade notional value in USD.
            order_type: ``'maker'`` or ``'taker'``.

        Returns:
            Fee amount in USD (positive).
        """
        if order_type == "maker":
            fee_bps = self._config.maker_fee_bps
        else:
            fee_bps = self._config.taker_fee_bps
        return notional * fee_bps / 10_000.0

    def compute_spread_cost(
        self,
        spread: float,
        notional: float,
    ) -> float:
        """Compute the cost of crossing the spread.

        For a market order the expected cost is half the spread times
        the notional, since the order crosses from mid to the far side.

        Args:
            spread: Current bid-ask spread in price terms.
            notional: Trade notional value in USD.

        Returns:
            Spread cost in USD.
        """
        if spread <= 0.0 or notional <= 0.0:
            return 0.0
        # Half-spread cost: we pay half the spread per unit notional
        # spread is absolute, so spread_cost = 0.5 * spread / mid * notional
        # Approximation: mid ~ notional / qty, but we use
        # spread_fraction = spread / (2 * mid) ~ spread / (bid + ask)
        # Simplified: half_spread * notional
        half_spread_frac = spread / 2.0
        # If spread is in price terms and notional is USD,
        # cost = (spread / 2) / price * notional. For simplicity and
        # consistency with the spec: half-spread * notional.
        return half_spread_frac * notional

    def compute_slippage(
        self,
        order_size_usd: float,
        adv_usd: float,
        volatility_daily: float,
    ) -> float:
        """Compute expected slippage using square-root impact model.

        I(Q) = Y * sigma * sqrt(Q / V) * notional

        Args:
            order_size_usd: Order size in USD.
            adv_usd: Average daily volume in USD.
            volatility_daily: Daily volatility as a decimal (e.g. 0.02).

        Returns:
            Slippage cost in USD.
        """
        if order_size_usd <= 0.0 or adv_usd <= 0.0:
            return 0.0

        y = self._config.slippage_Y
        ratio = order_size_usd / adv_usd
        return y * volatility_daily * math.sqrt(ratio) * order_size_usd

    def compute_market_impact(
        self,
        order_size_usd: float,
        adv_usd: float,
        volatility_daily: float,
    ) -> float:
        """Compute permanent market impact estimate.

        Approximated as 0.1x the temporary slippage for small orders.
        At typical retail sizes (< 1% of ADV) permanent impact is
        negligible, but this provides a conservative lower bound.

        Args:
            order_size_usd: Order size in USD.
            adv_usd: Average daily volume in USD.
            volatility_daily: Daily volatility as a decimal.

        Returns:
            Market impact cost in USD.
        """
        return 0.1 * self.compute_slippage(
            order_size_usd, adv_usd, volatility_daily
        )

    def compute_total(
        self,
        notional: float,
        order_type: str,
        spread: float,
        order_size_usd: float,
        adv_usd: float,
        volatility: float,
        funding_cost: float = 0.0,
    ) -> TransactionCost:
        """Compute all cost components and return a TransactionCost.

        Args:
            notional: Trade notional value in USD.
            order_type: ``'maker'`` or ``'taker'``.
            spread: Current bid-ask spread in price terms.
            order_size_usd: Order size in USD.
            adv_usd: Average daily volume in USD.
            volatility: Daily volatility as a decimal.
            funding_cost: Pre-computed funding cost in USD (default 0).

        Returns:
            TransactionCost with all components populated.
        """
        exchange_fee = self.compute_exchange_fee(notional, order_type)
        spread_cost = self.compute_spread_cost(spread, notional)
        slippage = self.compute_slippage(
            order_size_usd, adv_usd, volatility
        )
        market_impact = self.compute_market_impact(
            order_size_usd, adv_usd, volatility
        )
        total = (
            exchange_fee
            + funding_cost
            + spread_cost
            + slippage
            + market_impact
        )

        return TransactionCost(
            exchange_fee=exchange_fee,
            funding_cost=funding_cost,
            spread_cost=spread_cost,
            slippage=slippage,
            market_impact=market_impact,
            total=total,
            notional=notional,
        )
