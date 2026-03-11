"""Information-driven bars: Tick, Volume, and Dollar bars (AFML Ch.2).

Each bar type samples when a threshold of the chosen activity measure is
reached.  These are the simplest information bars — thresholds are fixed and
provided at construction time.
"""
from backend.bars.base import Bar, BaseBarGenerator


class TickBars(BaseBarGenerator):
    """Emit a bar after every *tick_count* ticks.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    tick_count : int
        Number of ticks per bar.
    """

    def __init__(self, symbol: str, tick_count: int = 1000):
        super().__init__(symbol, bar_type="tick")
        self._threshold = tick_count

    def process_tick(self, price: float, qty: float, time_ms: int,
                     is_buyer_maker: bool) -> list[Bar]:
        self._acc.update(price, qty, time_ms)
        if self._acc.tick_count >= self._threshold:
            return [self._emit_bar()]
        return []


class VolumeBars(BaseBarGenerator):
    """Emit a bar when cumulative volume reaches *volume_threshold*.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    volume_threshold : float
        Cumulative base-asset volume that triggers a bar.
    """

    def __init__(self, symbol: str, volume_threshold: float = 100.0):
        super().__init__(symbol, bar_type="volume")
        self._threshold = volume_threshold

    def process_tick(self, price: float, qty: float, time_ms: int,
                     is_buyer_maker: bool) -> list[Bar]:
        self._acc.update(price, qty, time_ms)
        if self._acc.volume >= self._threshold:
            return [self._emit_bar()]
        return []


class DollarBars(BaseBarGenerator):
    """Emit a bar when cumulative dollar volume reaches *dollar_threshold*.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    dollar_threshold : float
        Cumulative quote-asset (dollar) volume that triggers a bar.
    """

    def __init__(self, symbol: str, dollar_threshold: float = 1_000_000.0):
        super().__init__(symbol, bar_type="dollar")
        self._threshold = dollar_threshold

    def process_tick(self, price: float, qty: float, time_ms: int,
                     is_buyer_maker: bool) -> list[Bar]:
        self._acc.update(price, qty, time_ms)
        if self._acc.dollar_volume >= self._threshold:
            return [self._emit_bar()]
        return []
