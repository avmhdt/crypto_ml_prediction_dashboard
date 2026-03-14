"""Simulation engine for crypto ML prediction dashboard.

Exports the equity curve simulator (legacy) and the Phase A
realistic order fill simulation components:

- SimulationConfig: VIP-tier fee configuration and AS parameters
- BBOTracker: Best bid/offer tracking from bookTicker stream
- QueuePositionTracker: LogProb queue position estimation
- FillProbabilityEstimator: Avellaneda-Stoikov fill probability
- CostModel / TransactionCost: Multi-component cost decomposition
- FundingRateTracker: Perpetual futures funding cost tracking
- LimitPriceEngine: Optimal limit order placement
"""
from backend.simulation.config import SimulationConfig
from backend.simulation.bbo_tracker import BBOTracker
from backend.simulation.queue_tracker import QueuePositionTracker
from backend.simulation.fill_probability import FillProbabilityEstimator
from backend.simulation.cost_model import CostModel, TransactionCost
from backend.simulation.funding_tracker import FundingRateTracker
from backend.simulation.limit_price import LimitPriceEngine
from backend.simulation.fill_simulator import OrderFillSimulator, SimulatedFill, PendingOrder
from backend.simulation.equity import SimulationResult, simulate_equity

__all__ = [
    "SimulationConfig",
    "BBOTracker",
    "QueuePositionTracker",
    "FillProbabilityEstimator",
    "CostModel",
    "TransactionCost",
    "FundingRateTracker",
    "LimitPriceEngine",
    "OrderFillSimulator",
    "SimulatedFill",
    "PendingOrder",
    "SimulationResult",
    "simulate_equity",
]
