"""Queue position tracker using LogProb model from hftbacktest.

Estimates how far ahead in the limit-order queue an order sits,
advancing the position when trades occur at or through the limit
price, and when depth decreases on the same side.

The LogProb model (Cont & de Larrard, 2013) estimates the probability
that a depth decrease occurred in front of our order:

    P(before_me) = log(1 + back) / log(1 + front + back)

where *front* is the quantity ahead and *back* is the quantity behind.
"""
import math


# Side constants matching common convention
SIDE_BUY = 1
SIDE_SELL = -1


class QueuePositionTracker:
    """Track estimated queue position for a resting limit order.

    Args:
        side: ``1`` for buy (bid-side), ``-1`` for sell (ask-side).
        limit_price: Price level where the order rests.
        order_qty: Size of the order.
        initial_depth: Total resting quantity at the limit price level
            (including our order) at the time the order was placed.
    """

    def __init__(
        self,
        side: int,
        limit_price: float,
        order_qty: float,
        initial_depth: float,
    ) -> None:
        self._side = side
        self._limit_price = limit_price
        self._order_qty = order_qty
        # Queue ahead = depth minus our own order (we are at the back)
        self._queue_ahead = max(0.0, initial_depth - order_qty)
        self._filled_qty = 0.0
        self._is_filled = False

    # --- trade updates -------------------------------------------------

    def on_trade(
        self,
        trade_price: float,
        trade_qty: float,
        is_buyer_maker: bool,
    ) -> None:
        """Process a public trade and advance queue position.

        Trade-through: if the trade price crosses our limit level the
        order fills instantly regardless of queue position.

        For resting orders:
        - Buy side: advance when ``is_buyer_maker=True`` (sell aggressor
          hits bids) AND ``trade_price <= limit_price``.
        - Sell side: advance when ``is_buyer_maker=False`` (buy aggressor
          lifts asks) AND ``trade_price >= limit_price``.

        Args:
            trade_price: Execution price of the public trade.
            trade_qty: Executed quantity.
            is_buyer_maker: True when the buyer was the maker (trade
                was initiated by a sell-side aggressor).
        """
        if self._is_filled:
            return

        # Trade-through detection: price crosses our limit level
        if self._side == SIDE_BUY and trade_price < self._limit_price:
            self._fill()
            return
        if self._side == SIDE_SELL and trade_price > self._limit_price:
            self._fill()
            return

        # Queue advance for same-level trades
        if self._side == SIDE_BUY:
            if is_buyer_maker and trade_price <= self._limit_price:
                self._advance(trade_qty)
        else:  # SIDE_SELL
            if not is_buyer_maker and trade_price >= self._limit_price:
                self._advance(trade_qty)

    # --- depth updates -------------------------------------------------

    def on_depth_change(
        self,
        new_depth: float,
        old_depth: float,
    ) -> None:
        """Process a depth change at the limit price level.

        When depth decreases, the LogProb model estimates how much of
        the decrease occurred in front of our order vs behind.

        Args:
            new_depth: New total depth at the price level.
            old_depth: Previous total depth at the price level.
        """
        if self._is_filled:
            return

        decrease = old_depth - new_depth
        if decrease <= 0.0:
            # Depth increased or unchanged; no queue advance
            return

        # LogProb model from hftbacktest (Cont & de Larrard, 2013):
        #
        # The probability that a depth decrease is in front of our
        # order (and therefore advances our queue position):
        #
        #   P(in_front) = log(1 + front) / log(1 + front + back)
        #
        # where front = queue_ahead, back = depth behind our order.
        # When most depth is in front, cancels are more likely in
        # front and we advance more.  The spec's naming convention
        # uses P(before_me) = log(1+back)/log(1+front+back) where
        # "back" and "front" are swapped relative to queue priority;
        # we normalise to queue-position semantics here.
        front = self._queue_ahead
        back = max(0.0, old_depth - front - self._order_qty)
        total = front + back
        if total <= 0.0:
            return
        denom = math.log(1.0 + total)
        if denom <= 0.0:
            return

        p_in_front = math.log(1.0 + front) / denom
        expected_advance = decrease * p_in_front
        self._advance(expected_advance)

    # --- properties ----------------------------------------------------

    @property
    def queue_ahead(self) -> float:
        """Estimated quantity ahead of our order in the queue."""
        return self._queue_ahead

    @property
    def is_filled(self) -> bool:
        """Whether the order has been completely filled."""
        return self._is_filled

    @property
    def fill_fraction(self) -> float:
        """Fraction of the order that has been filled (0.0 to 1.0)."""
        if self._order_qty <= 0.0:
            return 0.0
        return min(1.0, self._filled_qty / self._order_qty)

    # --- internals -----------------------------------------------------

    def _advance(self, qty: float) -> None:
        """Advance queue position by *qty*; trigger fill when reached."""
        prev_ahead = self._queue_ahead
        self._queue_ahead = max(0.0, self._queue_ahead - qty)

        if prev_ahead > 0.0 and self._queue_ahead <= 0.0:
            # Queue is now empty; remaining trade qty fills our order
            overflow = qty - prev_ahead
            remaining = self._order_qty - self._filled_qty
            fill_amount = min(remaining, overflow + remaining)
            self._filled_qty += fill_amount
            if self._filled_qty >= self._order_qty:
                self._fill()

    def _fill(self) -> None:
        """Mark order as fully filled."""
        self._filled_qty = self._order_qty
        self._queue_ahead = 0.0
        self._is_filled = True
