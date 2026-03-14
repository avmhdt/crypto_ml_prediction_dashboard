"""Simulation configuration with Binance VIP fee tiers.

Provides SimulationConfig dataclass for controlling simulation mode,
capital, fee structure (simple flat-rate or realistic VIP-tier-based),
and Avellaneda-Stoikov model parameters. VIP fee schedule mirrors
Binance USDT-M futures as of 2024.
"""
from dataclasses import dataclass, field


# Binance USDT-M Futures VIP fee schedule (maker_bps, taker_bps)
VIP_FEE_TABLE: dict[int, tuple[float, float]] = {
    0: (2.0, 5.0),
    1: (1.6, 4.0),
    2: (1.4, 3.5),
    3: (1.2, 3.2),
    4: (1.0, 3.0),
    5: (0.8, 2.7),
    6: (0.6, 2.5),
    7: (0.4, 2.2),
    8: (0.2, 2.0),
    9: (0.0, 1.7),
}

BNB_DISCOUNT_FACTOR: float = 0.75  # 25% discount when paying fees in BNB


@dataclass
class SimulationConfig:
    """Configuration for order fill simulation.

    Supports two modes:
    - 'simple': flat fee in bps (legacy, matches equity.py behaviour)
    - 'realistic': VIP-tier maker/taker fees, slippage, market impact,
      funding costs, and queue-position modelling

    Avellaneda-Stoikov parameters (gamma, kappa, urgency) control the
    optimal limit-order placement engine used in realistic mode.
    """

    # --- mode ----------------------------------------------------------
    mode: str = "simple"

    # --- capital -------------------------------------------------------
    starting_capital: float = 10000.0

    # --- simple-mode fee (bps on notional) -----------------------------
    fees_bps: float = 10.0

    # --- realistic-mode fee tier ---------------------------------------
    vip_tier: int = 0
    bnb_discount: bool = False

    # Computed from vip_tier + bnb_discount in __post_init__
    maker_fee_bps: float = field(init=False, default=0.0)
    taker_fee_bps: float = field(init=False, default=0.0)

    # --- order management ----------------------------------------------
    order_timeout_ms: int = 60000

    # --- Avellaneda-Stoikov parameters ---------------------------------
    gamma: float = 0.1       # risk-aversion parameter
    kappa: float = 1.5       # order-arrival intensity decay
    urgency: float = 0.5     # 0 = passive, 1 = aggressive

    # --- funding -------------------------------------------------------
    default_funding_rate: float = 0.0001
    funding_interval_hours: int = 8

    # --- slippage ------------------------------------------------------
    slippage_Y: float = 1.0  # scaling constant for square-root impact

    def __post_init__(self) -> None:
        """Set maker/taker fees from VIP tier and apply BNB discount."""
        if self.vip_tier not in VIP_FEE_TABLE:
            raise ValueError(
                f"vip_tier must be 0-9, got {self.vip_tier}"
            )
        maker, taker = VIP_FEE_TABLE[self.vip_tier]

        if self.bnb_discount:
            maker *= BNB_DISCOUNT_FACTOR
            taker *= BNB_DISCOUNT_FACTOR

        self.maker_fee_bps = maker
        self.taker_fee_bps = taker
