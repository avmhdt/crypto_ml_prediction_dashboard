"""Time bars: fixed-interval OHLCV aggregation."""
from backend.bars.base import Bar, BaseBarGenerator

# Map interval strings to milliseconds.
_INTERVAL_MS = {
    "1min": 60_000,
    "5min": 300_000,
    "15min": 900_000,
    "1h": 3_600_000,
}


def _parse_interval(interval: str) -> int:
    """Convert an interval string like '1min', '5min', '15min', '1h' to milliseconds."""
    if interval in _INTERVAL_MS:
        return _INTERVAL_MS[interval]
    # Try parsing a bare number as minutes for forward-compat.
    stripped = interval.lower().strip()
    if stripped.endswith("min"):
        return int(stripped[:-3]) * 60_000
    if stripped.endswith("h"):
        return int(stripped[:-1]) * 3_600_000
    if stripped.endswith("s"):
        return int(stripped[:-1]) * 1_000
    raise ValueError(f"Unsupported interval string: {interval!r}")


class TimeBars(BaseBarGenerator):
    """Emit a bar whenever a fixed time window closes.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. 'BTCUSDT').
    interval : str
        Human-readable interval such as '1min', '5min', '15min', or '1h'.
    """

    def __init__(self, symbol: str, interval: str = "1min"):
        super().__init__(symbol, bar_type="time")
        self._interval_ms = _parse_interval(interval)
        # The end-of-window boundary (exclusive).  Set on first tick.
        self._window_end: int | None = None

    def _align_window(self, time_ms: int) -> int:
        """Return the exclusive end of the window that *time_ms* falls into."""
        # Floor time_ms to the nearest interval, then add one interval.
        return (time_ms // self._interval_ms + 1) * self._interval_ms

    def process_tick(self, price: float, qty: float, time_ms: int,
                     is_buyer_maker: bool) -> list[Bar]:
        bars: list[Bar] = []

        # Bootstrap window on first tick.
        if self._window_end is None:
            self._window_end = self._align_window(time_ms)

        # If the incoming tick crosses one or more window boundaries we must
        # emit the accumulated bar(s) first.  A large gap can skip whole
        # windows; we only emit a bar for windows where we actually have data.
        while time_ms >= self._window_end:
            if self._acc.tick_count > 0:
                bars.append(self._emit_bar())
            # Advance window.  If the tick still falls beyond the next window
            # we loop again (but without accumulated data, so no bar emitted).
            self._window_end += self._interval_ms

        # After we have aligned the window, accumulate the tick.
        self._acc.update(price, qty, time_ms)

        return bars
