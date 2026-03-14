"""Best Bid/Offer (BBO) tracker for Binance bookTicker stream.

Maintains the current top-of-book state and a rolling window of recent
snapshots for computing spread statistics.  Designed to consume the
Binance WebSocket ``bookTicker`` events with minimal overhead.
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class _BBOSnapshot:
    """Single point-in-time BBO observation."""
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    time_ms: int


class BBOTracker:
    """Track best bid/offer from a live bookTicker stream.

    Usage::

        tracker = BBOTracker()
        tracker.on_bbo(bid=50000.0, bid_qty=1.2,
                       ask=50001.0, ask_qty=0.8, time_ms=1700000000000)
        print(tracker.mid_price)   # 50000.5
        print(tracker.spread_bps)  # ~0.2
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0
        self._bid_qty: float = 0.0
        self._ask_qty: float = 0.0
        self._last_update_ms: int = 0
        self._buffer: deque[_BBOSnapshot] = deque(maxlen=buffer_size)

    # --- ingest --------------------------------------------------------

    def on_bbo(
        self,
        bid: float,
        bid_qty: float,
        ask: float,
        ask_qty: float,
        time_ms: int,
    ) -> None:
        """Process a new bookTicker update.

        Args:
            bid: Best bid price.
            bid_qty: Quantity at best bid.
            ask: Best ask price.
            ask_qty: Quantity at best ask.
            time_ms: Exchange timestamp in epoch milliseconds.
        """
        self._best_bid = bid
        self._best_ask = ask
        self._bid_qty = bid_qty
        self._ask_qty = ask_qty
        self._last_update_ms = time_ms
        self._buffer.append(
            _BBOSnapshot(bid=bid, bid_qty=bid_qty,
                         ask=ask, ask_qty=ask_qty, time_ms=time_ms)
        )

    # --- properties ----------------------------------------------------

    @property
    def best_bid(self) -> float:
        """Current best bid price."""
        return self._best_bid

    @property
    def best_ask(self) -> float:
        """Current best ask price."""
        return self._best_ask

    @property
    def last_update_ms(self) -> int:
        """Epoch-millisecond timestamp of last BBO update."""
        return self._last_update_ms

    @property
    def mid_price(self) -> float:
        """Mid-price: average of best bid and best ask."""
        return (self._best_bid + self._best_ask) / 2.0

    @property
    def spread(self) -> float:
        """Absolute spread (ask - bid)."""
        return self._best_ask - self._best_bid

    @property
    def spread_bps(self) -> float:
        """Current spread in basis points relative to mid-price."""
        mid = self.mid_price
        if mid <= 0.0:
            return 0.0
        return (self.spread / mid) * 10_000.0

    @property
    def avg_spread_bps(self) -> float:
        """Average spread in bps over the rolling buffer window.

        Returns 0.0 if the buffer is empty or all mid-prices are zero.
        """
        if not self._buffer:
            return 0.0

        total_bps = 0.0
        valid = 0
        for snap in self._buffer:
            mid = (snap.bid + snap.ask) / 2.0
            if mid > 0.0:
                total_bps += ((snap.ask - snap.bid) / mid) * 10_000.0
                valid += 1

        return total_bps / valid if valid > 0 else 0.0
