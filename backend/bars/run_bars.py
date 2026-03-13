"""Run bars: Tick, Volume, and Dollar run bars (AFML Ch.2).

Run bars monitor the *runs* of consecutive same-sign ticks rather than the
net imbalance.  The emission condition is:

    max( P[b=1]*run_up,  (1-P[b=1])*run_down )  >=
        E[T] * max( P[b=1]*E[run_up/T],  (1-P[b=1])*E[run_down/T] )

Where:
  - P[b=1] is the EWMA proportion of buy-initiated ticks.
  - run_up / run_down are the cumulative value of same-sign ticks in the
    current bar (tick count, volume, or dollar volume depending on bar type).
  - E[T], E[run_up/T], E[run_down/T] are EWMA estimates across completed bars.

The tick direction b_t is computed identically to imbalance bars.
"""
import json
from pathlib import Path

import numpy as np

from backend.bars.base import Bar, EWMABarGenerator


class _RunBarBase(EWMABarGenerator):
    """Shared logic for all run bar variants.

    Subclasses override ``_tick_value`` to supply the quantity that feeds the
    run counters (1 for tick-run, volume for volume-run, dollar volume for
    dollar-run).
    """

    def __init__(self, symbol: str, bar_type: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type,
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)

        # ------ Running state within the current bar ------
        # Cumulative run values for up (+1) and down (-1) ticks.
        self._run_up: float = 0.0
        self._run_down: float = 0.0
        # Count of buy-initiated ticks in the current bar (for P[b=1]).
        self._buy_count: int = 0
        self._tick_idx: int = 0

        # ------ EWMA estimates across bars ------
        # P[b=1]: proportion of buy ticks (EWMA).
        self._p_buy: float = 0.5
        # E[run_up/T] and E[run_down/T] — None until calibrated from warmup bar.
        self._expected_run_up: float | None = None
        self._expected_run_down: float | None = None
        # Number of completed bars (for warmup detection)
        self._bars_completed: int = 0

        # ------ Tick direction state ------
        self._prev_price: float | None = None
        self._prev_b: float = 1.0

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _tick_value(self, price: float, qty: float) -> float:
        """Return the quantity to accumulate per tick for run measurement."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Tick direction
    # ------------------------------------------------------------------
    def _compute_b(self, price: float, is_buyer_maker: bool) -> float:
        b = -1.0 if is_buyer_maker else 1.0
        self._prev_price = price
        self._prev_b = b
        return b

    # ------------------------------------------------------------------
    # EWMA helpers
    # ------------------------------------------------------------------
    def _update_ewma_runs(self, norm_run_up: float, norm_run_down: float,
                          p_buy_bar: float) -> None:
        """Update EWMA estimates after a bar completes."""
        a = self._ewma_alpha
        if self._expected_run_up is None:
            # First calibration from warmup bar
            self._expected_run_up = norm_run_up if norm_run_up > 0 else 0.5
            self._expected_run_down = norm_run_down if norm_run_down > 0 else 0.5
        else:
            self._expected_run_up = a * norm_run_up + (1 - a) * self._expected_run_up
            self._expected_run_down = a * norm_run_down + (1 - a) * self._expected_run_down
        self._p_buy = max(0.01, min(0.99,
            a * p_buy_bar + (1 - a) * self._p_buy))

    def _threshold(self) -> float:
        """RHS of the emission condition (adaptive threshold)."""
        if self._expected_run_up is None:
            return float('inf')  # During warmup
        p = self._p_buy
        return self._expected_ticks * max(
            p * self._expected_run_up,
            (1.0 - p) * self._expected_run_down,
        )

    def _run_metric(self) -> float:
        """LHS of the emission condition (current bar metric)."""
        p = self._p_buy
        return max(
            p * self._run_up,
            (1.0 - p) * self._run_down,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def process_tick(self, price: float, qty: float, time_ms: int,
                     is_buyer_maker: bool) -> list[Bar]:
        bars: list[Bar] = []

        b = self._compute_b(price, is_buyer_maker)
        value = self._tick_value(price, qty)
        self._tick_idx += 1

        # Accumulate runs.
        if b > 0:
            self._run_up += value
            self._buy_count += 1
        else:
            self._run_down += value

        self._acc.update(price, qty, time_ms)

        # Warmup: force-emit first bar after expected_num_ticks_init ticks
        # to calibrate the EWMA from actual data.
        if self._bars_completed == 0:
            if self._tick_idx >= self.expected_num_ticks_init:
                T = self._tick_idx
                norm_up = self._run_up / T if T > 0 else 0.0
                norm_down = self._run_down / T if T > 0 else 0.0
                p_buy_bar = self._buy_count / T if T > 0 else 0.5

                bar = self._emit_bar()
                bars.append(bar)

                self._update_expected_ticks(T)
                self._update_ewma_runs(norm_up, norm_down, p_buy_bar)

                self._run_up = 0.0
                self._run_down = 0.0
                self._buy_count = 0
                self._tick_idx = 0
                self._bars_completed += 1
            return bars

        # Normal operation: emission check.
        if self._run_metric() >= self._threshold():
            T = self._tick_idx
            norm_up = self._run_up / T if T > 0 else 0.0
            norm_down = self._run_down / T if T > 0 else 0.0
            p_buy_bar = self._buy_count / T if T > 0 else 0.5

            bar = self._emit_bar()
            bars.append(bar)

            self._update_expected_ticks(T)
            self._update_ewma_runs(norm_up, norm_down, p_buy_bar)

            self._run_up = 0.0
            self._run_down = 0.0
            self._buy_count = 0
            self._tick_idx = 0
            self._bars_completed += 1

        return bars

    # ------------------------------------------------------------------
    # State persistence (extends base)
    # ------------------------------------------------------------------
    def save_state(self, path: Path) -> None:
        state = {
            "expected_ticks": self._expected_ticks,
            "bar_tick_counts": self._bar_tick_counts[-self.num_prev_bars:],
            "expected_num_ticks_init": self.expected_num_ticks_init,
            "num_prev_bars": self.num_prev_bars,
            "p_buy": self._p_buy,
            "expected_run_up": self._expected_run_up,
            "expected_run_down": self._expected_run_down,
            "bars_completed": self._bars_completed,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))

    def load_state(self, path: Path) -> None:
        super().load_state(path)
        if path.exists():
            state = json.loads(path.read_text())
            self._p_buy = state.get("p_buy", self._p_buy)
            self._expected_run_up = state.get("expected_run_up", self._expected_run_up)
            self._expected_run_down = state.get("expected_run_down", self._expected_run_down)
            self._bars_completed = state.get("bars_completed", self._bars_completed)


# ======================================================================
# Concrete run bar classes
# ======================================================================

class TickRunBars(_RunBarBase):
    """Tick run bars: emit based on runs of consecutive same-sign ticks.

    Each tick contributes a value of 1 to the run counter.
    """

    def __init__(self, symbol: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type="tick_run",
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)

    def _tick_value(self, price: float, qty: float) -> float:
        return 1.0


class VolumeRunBars(_RunBarBase):
    """Volume run bars: emit based on volume-weighted runs.

    Each tick contributes its base-asset volume to the run counter.
    """

    def __init__(self, symbol: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type="volume_run",
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)

    def _tick_value(self, price: float, qty: float) -> float:
        return qty


class DollarRunBars(_RunBarBase):
    """Dollar run bars: emit based on dollar-volume-weighted runs.

    Each tick contributes its dollar volume (price * qty) to the run counter.
    """

    def __init__(self, symbol: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type="dollar_run",
                         expected_num_ticks_init=expected_num_ticks_init,
                         num_prev_bars=num_prev_bars)

    def _tick_value(self, price: float, qty: float) -> float:
        return price * qty
