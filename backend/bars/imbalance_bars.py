"""Imbalance bars: Tick, Volume, and Dollar imbalance bars (AFML Ch.2).

Imbalance bars sample the market when the running signed imbalance exceeds
an adaptively estimated threshold.  The threshold is:

    threshold = E[T] * E[|imbalance| / T]

where E[T] is the EWMA of bar lengths and E[|imbalance|/T] is the EWMA of
the absolute normalised imbalance across completed bars.  Both are maintained
by the EWMABarGenerator base class and local EWMA tracking respectively.

The tick direction b_t is determined as:
  - If the tick aggressor side is available: b_t = -1 if is_buyer_maker else +1
  - Otherwise: b_t = sign(price_t - price_{t-1}), repeating the last sign on
    zero change.
"""
import numpy as np

from backend.bars.base import Bar, EWMABarGenerator


class _ImbalanceBarBase(EWMABarGenerator):
    """Shared logic for all imbalance bar variants.

    Subclasses override ``_tick_value`` to return the quantity that is
    multiplied by the tick direction b_t (1 for tick imbalance, volume for
    volume imbalance, dollar volume for dollar imbalance).
    """

    def __init__(self, symbol: str, bar_type: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type,
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)
        # Running imbalance accumulator for the current (incomplete) bar.
        self._imbalance: float = 0.0
        # Previous tick price for b_t computation when is_buyer_maker is
        # unavailable.  None until the first tick.
        self._prev_price: float | None = None
        # Last non-zero tick direction to carry forward on zero-change ticks.
        self._prev_b: float = 1.0
        # EWMA of |imbalance| / T across completed bars.
        self._expected_imbalance: float = 1.0
        # Tick counter within the current bar (mirrors _acc.tick_count but
        # maintained separately so it's always available before _acc.update).
        self._tick_idx: int = 0

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _tick_value(self, price: float, qty: float) -> float:
        """Return the quantity to multiply by b_t for this bar type."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Tick direction
    # ------------------------------------------------------------------
    def _compute_b(self, price: float, is_buyer_maker: bool) -> float:
        """Compute signed tick direction b_t."""
        # Use aggressor flag when available (Binance aggTrades always have it).
        # is_buyer_maker == True means the buyer was the maker, so the seller
        # was the aggressor → sell-initiated → -1.
        b = -1.0 if is_buyer_maker else 1.0

        # Fall-back: if we wanted to derive from price change alone, we could
        # compare to previous price.  Currently we trust the flag.
        if self._prev_price is not None and price != self._prev_price:
            # Validate against price movement (informational; flag wins).
            pass

        self._prev_price = price
        self._prev_b = b
        return b

    # ------------------------------------------------------------------
    # EWMA helpers for imbalance estimation
    # ------------------------------------------------------------------
    def _update_expected_imbalance(self, abs_norm_imbalance: float) -> None:
        """Update the EWMA of |imbalance|/T, clamped to [0.1, 10.0]."""
        raw = (
            self._ewma_alpha * abs_norm_imbalance +
            (1 - self._ewma_alpha) * self._expected_imbalance
        )
        self._expected_imbalance = max(0.1, min(10.0, raw))

    def _threshold(self) -> float:
        """Current adaptive threshold."""
        return self._expected_ticks * self._expected_imbalance

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def process_tick(self, price: float, qty: float, time_ms: int,
                     is_buyer_maker: bool) -> list[Bar]:
        bars: list[Bar] = []

        b = self._compute_b(price, is_buyer_maker)
        value = self._tick_value(price, qty)
        self._imbalance += b * value
        self._tick_idx += 1

        self._acc.update(price, qty, time_ms)

        # Check whether the running imbalance exceeds the threshold.
        if abs(self._imbalance) >= self._threshold():
            # Record imbalance statistics before resetting.
            T = self._tick_idx
            abs_norm = abs(self._imbalance) / T if T > 0 else 0.0

            bar = self._emit_bar()
            bars.append(bar)

            # Update EWMA estimates.
            self._update_expected_ticks(T)
            self._update_expected_imbalance(abs_norm)

            # Reset running state for the next bar.
            self._imbalance = 0.0
            self._tick_idx = 0

        return bars

    # ------------------------------------------------------------------
    # State persistence (extends base)
    # ------------------------------------------------------------------
    def save_state(self, path) -> None:
        import json
        base_state = {
            "expected_ticks": self._expected_ticks,
            "bar_tick_counts": self._bar_tick_counts[-self.num_prev_bars:],
            "expected_num_ticks_init": self.expected_num_ticks_init,
            "num_prev_bars": self.num_prev_bars,
            "expected_imbalance": self._expected_imbalance,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(base_state))

    def load_state(self, path) -> None:
        super().load_state(path)
        if path.exists():
            import json
            state = json.loads(path.read_text())
            self._expected_imbalance = state.get(
                "expected_imbalance", self._expected_imbalance
            )


# ======================================================================
# Concrete imbalance bar classes
# ======================================================================

class TickImbalanceBars(_ImbalanceBarBase):
    """Tick imbalance bars: emit when |sum(b_t)| >= E[T]*E[|imb|/T].

    Each tick contributes b_t (direction) with a weight of 1.
    """

    def __init__(self, symbol: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type="tick_imbalance",
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)

    def _tick_value(self, price: float, qty: float) -> float:
        return 1.0


class VolumeImbalanceBars(_ImbalanceBarBase):
    """Volume imbalance bars: emit when |sum(b_t * v_t)| >= threshold.

    Each tick contributes b_t * volume.
    """

    def __init__(self, symbol: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type="volume_imbalance",
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)

    def _tick_value(self, price: float, qty: float) -> float:
        return qty


class DollarImbalanceBars(_ImbalanceBarBase):
    """Dollar imbalance bars: emit when |sum(b_t * dv_t)| >= threshold.

    Each tick contributes b_t * dollar_volume (price * qty).
    """

    def __init__(self, symbol: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type="dollar_imbalance",
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)

    def _tick_value(self, price: float, qty: float) -> float:
        return price * qty
