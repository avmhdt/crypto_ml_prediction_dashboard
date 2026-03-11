"""Base class for all bar types."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class Bar:
    """Single OHLCV bar."""
    symbol: str
    bar_type: str
    timestamp: int  # unix ms (bar close time)
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    tick_count: int
    duration_us: int  # microseconds

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bar_type": self.bar_type,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "dollar_volume": self.dollar_volume,
            "tick_count": self.tick_count,
            "duration_us": self.duration_us,
        }


@dataclass
class BarAccumulator:
    """Accumulates ticks into a bar."""
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    close: float = 0.0
    volume: float = 0.0
    dollar_volume: float = 0.0
    tick_count: int = 0
    start_time: int = 0
    end_time: int = 0

    def update(self, price: float, qty: float, time_ms: int) -> None:
        if self.tick_count == 0:
            self.open = price
            self.start_time = time_ms
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += qty
        self.dollar_volume += price * qty
        self.tick_count += 1
        self.end_time = time_ms

    def to_bar(self, symbol: str, bar_type: str) -> Bar:
        duration = (self.end_time - self.start_time) * 1000 if self.tick_count > 1 else 0
        return Bar(
            symbol=symbol,
            bar_type=bar_type,
            timestamp=self.end_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            dollar_volume=self.dollar_volume,
            tick_count=self.tick_count,
            duration_us=duration,
        )

    def reset(self) -> None:
        self.open = 0.0
        self.high = -np.inf
        self.low = np.inf
        self.close = 0.0
        self.volume = 0.0
        self.dollar_volume = 0.0
        self.tick_count = 0
        self.start_time = 0
        self.end_time = 0


class BaseBarGenerator(ABC):
    """Abstract base for all bar generators."""

    def __init__(self, symbol: str, bar_type: str):
        self.symbol = symbol
        self.bar_type = bar_type
        self._acc = BarAccumulator()

    @abstractmethod
    def process_tick(self, price: float, qty: float, time_ms: int,
                     is_buyer_maker: bool) -> list[Bar]:
        """Process a single tick. Return list of completed bars (usually 0 or 1)."""
        ...

    def process_ticks(self, prices: np.ndarray, qtys: np.ndarray,
                      times: np.ndarray, is_buyer_makers: np.ndarray) -> list[Bar]:
        """Process batch of ticks."""
        bars = []
        for i in range(len(prices)):
            result = self.process_tick(
                float(prices[i]), float(qtys[i]),
                int(times[i]), bool(is_buyer_makers[i])
            )
            bars.extend(result)
        return bars

    def _emit_bar(self) -> Bar:
        """Emit current accumulated bar and reset."""
        bar = self._acc.to_bar(self.symbol, self.bar_type)
        self._acc.reset()
        return bar


class EWMABarGenerator(BaseBarGenerator):
    """Base for bars that use EWMA threshold estimation (imbalance/run bars)."""

    def __init__(self, symbol: str, bar_type: str,
                 expected_num_ticks_init: int = 1000,
                 num_prev_bars: int = 100):
        super().__init__(symbol, bar_type)
        self.expected_num_ticks_init = expected_num_ticks_init
        self.num_prev_bars = num_prev_bars
        self._ewma_alpha = 2.0 / (num_prev_bars + 1)
        self._expected_ticks = float(expected_num_ticks_init)
        self._bar_tick_counts: list[int] = []
        # Clamp bounds to prevent bar explosion/starvation
        self._min_expected = expected_num_ticks_init * 0.5
        self._max_expected = expected_num_ticks_init * 2.0

    def _update_expected_ticks(self, actual_ticks: int) -> None:
        """Update EWMA of expected ticks per bar."""
        self._bar_tick_counts.append(actual_ticks)
        self._expected_ticks = (
            self._ewma_alpha * actual_ticks +
            (1 - self._ewma_alpha) * self._expected_ticks
        )
        # Clamp to prevent explosion/starvation
        self._expected_ticks = max(self._min_expected,
                                   min(self._max_expected, self._expected_ticks))

    def save_state(self, path: Path) -> None:
        """Save EWMA state for continuity across sessions."""
        state = {
            "expected_ticks": self._expected_ticks,
            "bar_tick_counts": self._bar_tick_counts[-self.num_prev_bars:],
            "expected_num_ticks_init": self.expected_num_ticks_init,
            "num_prev_bars": self.num_prev_bars,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))

    def load_state(self, path: Path) -> None:
        """Load EWMA state from prior session."""
        if path.exists():
            state = json.loads(path.read_text())
            self._expected_ticks = state.get("expected_ticks", self._expected_ticks)
            self._bar_tick_counts = state.get("bar_tick_counts", [])
