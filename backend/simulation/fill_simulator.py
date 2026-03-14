"""Order fill simulator — orchestrates limit order lifecycle.

Per-symbol tick-level simulator that manages pending orders, tracks
queue positions, computes fill probabilities, and emits SimulatedFill
events with full cost decomposition.

Usage::

    sim = OrderFillSimulator("BTCUSDT", SimulationConfig(mode="realistic"))
    sim.on_bbo(50000, 1.5, 50001, 0.8, ts)
    sim.submit_order(signal_dict)
    fills = sim.on_tick(50000.5, 0.1, ts+100, True)
"""
import logging
from dataclasses import dataclass, field

from backend.simulation.config import SimulationConfig
from backend.simulation.bbo_tracker import BBOTracker
from backend.simulation.queue_tracker import QueuePositionTracker
from backend.simulation.fill_probability import FillProbabilityEstimator
from backend.simulation.cost_model import CostModel, TransactionCost
from backend.simulation.funding_tracker import FundingRateTracker
from backend.simulation.limit_price import LimitPriceEngine

logger = logging.getLogger(__name__)

# Default ADV for major pairs when unknown (conservative: $1B/day)
_DEFAULT_ADV_USD = 1_000_000_000.0
# Default daily volatility when unknown
_DEFAULT_VOLATILITY = 0.02


@dataclass
class SimulatedFill:
    """Record of a completed simulated fill."""
    signal_id: int | None
    symbol: str
    side: int
    fill_price: float
    fill_qty: float
    fill_time_ms: int
    order_type: str          # "maker" or "taker"
    costs: TransactionCost = field(default_factory=TransactionCost)
    queue_wait_ms: int = 0
    meta_probability: float = 0.0
    limit_price: float = 0.0
    submitted_ms: int = 0


@dataclass
class PendingOrder:
    """An order waiting to be filled."""
    signal: dict
    limit_price: float
    submitted_ms: int
    queue_tracker: QueuePositionTracker
    timeout_ms: int = 60000


class OrderFillSimulator:
    """Per-symbol orchestrator for realistic order fill simulation.

    Manages pending simulated orders, processes ticks and BBO updates,
    and emits fill events with full cost decomposition.

    Args:
        symbol: Trading symbol (e.g. ``'BTCUSDT'``).
        config: Simulation configuration.
    """

    def __init__(self, symbol: str, config: SimulationConfig) -> None:
        self.symbol = symbol
        self._config = config
        self._bbo = BBOTracker()
        self._cost_model = CostModel(config)
        self._funding = FundingRateTracker(
            default_rate=config.default_funding_rate,
        )
        self._fill_prob = FillProbabilityEstimator(
            kappa=config.kappa,
            base_intensity=1.0,
        )
        self._limit_engine = LimitPriceEngine(
            gamma=config.gamma,
            kappa=config.kappa,
            urgency=config.urgency,
        )

        self._pending: list[PendingOrder] = []
        self._total_submitted = 0
        self._total_filled = 0
        self._total_expired = 0
        self._total_slippage_bps = 0.0
        self._total_maker = 0
        self._total_taker = 0
        self._total_wait_ms = 0

    # --- order submission ------------------------------------------------

    def submit_order(self, signal: dict) -> PendingOrder | None:
        """Submit a new simulated order from a trading signal.

        Computes the optimal limit price and initialises queue tracking.
        Returns None if BBO data is not yet available.

        Args:
            signal: Dict with keys: symbol, side, size, entry_price,
                meta_probability, timestamp, and optionally signal_id.

        Returns:
            PendingOrder if submitted, None if skipped.
        """
        side = signal.get("side", 1)
        size = signal.get("size", 0.0)
        if size <= 0:
            return None

        mid = self._bbo.mid_price
        spread = self._bbo.spread
        entry_price = signal.get("entry_price", mid)

        # Use entry price as fallback if BBO not available
        if mid <= 0:
            mid = entry_price
        if spread <= 0:
            spread = mid * 0.0002  # assume 2 bps spread

        # Compute optimal limit price
        volatility = _DEFAULT_VOLATILITY
        limit_price = self._limit_engine.compute_limit_price(
            mid_price=mid,
            signal_side=side,
            signal_size=size,
            spread=spread,
            volatility=volatility,
        )

        # Estimate initial depth at limit level
        if side == 1:
            initial_depth = self._bbo._bid_qty if self._bbo._bid_qty > 0 else 1.0
        else:
            initial_depth = self._bbo._ask_qty if self._bbo._ask_qty > 0 else 1.0

        order_qty = size  # normalised [0, 1]

        queue = QueuePositionTracker(
            side=side,
            limit_price=limit_price,
            order_qty=order_qty,
            initial_depth=initial_depth,
        )

        submitted_ms = signal.get("timestamp", 0)
        pending = PendingOrder(
            signal=signal,
            limit_price=limit_price,
            submitted_ms=submitted_ms,
            queue_tracker=queue,
            timeout_ms=self._config.order_timeout_ms,
        )
        self._pending.append(pending)
        self._total_submitted += 1
        return pending

    # --- tick processing -------------------------------------------------

    def on_tick(
        self,
        price: float,
        qty: float,
        time_ms: int,
        is_buyer_maker: bool,
    ) -> list[SimulatedFill]:
        """Process a trade tick through all pending orders.

        Updates queue positions and checks for fills.

        Args:
            price: Trade price.
            qty: Trade quantity.
            time_ms: Trade timestamp in epoch ms.
            is_buyer_maker: True if buyer was the maker.

        Returns:
            List of fills that occurred on this tick.
        """
        fills: list[SimulatedFill] = []
        still_pending: list[PendingOrder] = []

        for order in self._pending:
            # Advance queue position
            order.queue_tracker.on_trade(price, qty, is_buyer_maker)

            if order.queue_tracker.is_filled:
                # Order filled via queue depletion or trade-through
                fill = self._create_fill(order, price, time_ms, "maker")
                fills.append(fill)
            elif time_ms - order.submitted_ms > order.timeout_ms:
                # Order timed out
                self._total_expired += 1
            else:
                still_pending.append(order)

        self._pending = still_pending
        return fills

    # --- BBO processing --------------------------------------------------

    def on_bbo(
        self,
        bid: float,
        bid_qty: float,
        ask: float,
        ask_qty: float,
        time_ms: int,
    ) -> None:
        """Process a BBO update.

        Args:
            bid: Best bid price.
            bid_qty: Quantity at best bid.
            ask: Best ask price.
            ask_qty: Quantity at best ask.
            time_ms: Timestamp in epoch ms.
        """
        self._bbo.on_bbo(bid, bid_qty, ask, ask_qty, time_ms)

    # --- expiry ----------------------------------------------------------

    def cancel_expired(self, current_ms: int) -> list[dict]:
        """Cancel orders that have exceeded their timeout.

        Args:
            current_ms: Current timestamp in epoch ms.

        Returns:
            List of expired signal dicts.
        """
        expired_signals: list[dict] = []
        still_pending: list[PendingOrder] = []

        for order in self._pending:
            if current_ms - order.submitted_ms > order.timeout_ms:
                expired_signals.append(order.signal)
                self._total_expired += 1
            else:
                still_pending.append(order)

        self._pending = still_pending
        return expired_signals

    # --- properties ------------------------------------------------------

    @property
    def pending_orders(self) -> list[PendingOrder]:
        """Currently pending orders."""
        return self._pending

    @property
    def bbo(self) -> BBOTracker:
        """BBO tracker for this symbol."""
        return self._bbo

    @property
    def funding(self) -> FundingRateTracker:
        """Funding rate tracker."""
        return self._funding

    @property
    def stats(self) -> dict:
        """Simulation statistics."""
        fill_rate = (
            self._total_filled / self._total_submitted * 100
            if self._total_submitted > 0 else 0.0
        )
        avg_wait = (
            self._total_wait_ms / self._total_filled
            if self._total_filled > 0 else 0.0
        )
        avg_slippage = (
            self._total_slippage_bps / self._total_filled
            if self._total_filled > 0 else 0.0
        )
        maker_ratio = (
            self._total_maker / self._total_filled * 100
            if self._total_filled > 0 else 0.0
        )
        return {
            "total_submitted": self._total_submitted,
            "total_filled": self._total_filled,
            "total_expired": self._total_expired,
            "fill_rate": round(fill_rate, 1),
            "avg_wait_ms": round(avg_wait, 0),
            "avg_slippage_bps": round(avg_slippage, 2),
            "maker_ratio": round(maker_ratio, 1),
        }

    # --- internals -------------------------------------------------------

    def _create_fill(
        self,
        order: PendingOrder,
        fill_price: float,
        fill_time_ms: int,
        order_type: str,
    ) -> SimulatedFill:
        """Build a SimulatedFill with full cost decomposition."""
        signal = order.signal
        side = signal.get("side", 1)
        size = signal.get("size", 1.0)
        entry_price = signal.get("entry_price", fill_price)
        notional = abs(size * fill_price)

        # Compute funding cost
        funding_cost = self._funding.compute_funding_cost(
            entry_ms=order.submitted_ms,
            exit_ms=fill_time_ms,
            position_notional=notional,
            side=side,
        )

        # Compute total costs
        spread = self._bbo.spread if self._bbo.spread > 0 else fill_price * 0.0002
        costs = self._cost_model.compute_total(
            notional=notional,
            order_type=order_type,
            spread=spread / fill_price,  # as fraction for spread cost
            order_size_usd=notional,
            adv_usd=_DEFAULT_ADV_USD,
            volatility=_DEFAULT_VOLATILITY,
            funding_cost=funding_cost,
        )

        wait_ms = fill_time_ms - order.submitted_ms

        # Update stats
        self._total_filled += 1
        self._total_wait_ms += wait_ms
        self._total_slippage_bps += costs.slippage / notional * 10000 if notional > 0 else 0
        if order_type == "maker":
            self._total_maker += 1
        else:
            self._total_taker += 1

        return SimulatedFill(
            signal_id=signal.get("id"),
            symbol=self.symbol,
            side=side,
            fill_price=fill_price,
            fill_qty=size,
            fill_time_ms=fill_time_ms,
            order_type=order_type,
            costs=costs,
            queue_wait_ms=wait_ms,
            meta_probability=signal.get("meta_probability", 0.0),
            limit_price=order.limit_price,
            submitted_ms=order.submitted_ms,
        )
