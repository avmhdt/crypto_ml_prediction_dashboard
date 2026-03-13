"""Binance perpetual futures WebSocket live feed.

Connects to Binance's public market data WebSocket (no API key needed)
and streams real-time trade data for a given symbol.
"""
import asyncio
import logging
from collections.abc import Callable
from binance import AsyncClient, BinanceSocketManager

logger = logging.getLogger(__name__)


class BinanceLiveFeed:
    """Streams live trade data from Binance perpetual futures."""

    def __init__(
        self,
        symbol: str,
        on_tick: Callable[[dict], None] | None = None,
        on_bbo: Callable[[dict], None] | None = None,
    ):
        self.symbol = symbol.lower()
        self.on_tick = on_tick
        self.on_bbo = on_bbo
        self._client: AsyncClient | None = None
        self._bsm: BinanceSocketManager | None = None
        self._running = False
        self._tick_buffer: list[dict] = []
        self._bbo_buffer: list[dict] = []
        self._buffer_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start streaming trade data and BBO from Binance perpetual futures."""
        self._client = await AsyncClient.create()
        self._bsm = BinanceSocketManager(self._client, max_queue_size=5000)
        self._running = True

        # Subscribe to both aggTrade and bookTicker via multiplex socket
        streams = [
            f"{self.symbol}@aggTrade",
            f"{self.symbol}@bookTicker",
        ]
        ts = self._bsm.futures_multiplex_socket(streams)
        async with ts as stream:
            while self._running:
                try:
                    raw = await asyncio.wait_for(stream.recv(), timeout=30)
                    if raw is None:
                        continue
                    # Unwrap {stream, data} envelope
                    msg = raw.get("data", raw) if isinstance(raw, dict) else raw
                    event_type = msg.get("e", "")

                    if event_type == "aggTrade":
                        tick = self._parse_trade(msg)
                        if tick:
                            async with self._buffer_lock:
                                self._tick_buffer.append(tick)
                            if self.on_tick:
                                self.on_tick(tick)
                    elif event_type == "bookTicker":
                        bbo = self._parse_bbo(msg)
                        if bbo:
                            async with self._buffer_lock:
                                self._bbo_buffer.append(bbo)
                            if self.on_bbo:
                                self.on_bbo(bbo)
                except asyncio.TimeoutError:
                    logger.debug("WebSocket timeout, reconnecting...")
                    continue
                except Exception as e:
                    logger.error(f"Live feed error: {e}")
                    if self._running:
                        await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        if self._client:
            await self._client.close_connection()

    async def flush_buffer(self) -> list[dict]:
        """Return and clear buffered ticks for micro-batch insert."""
        async with self._buffer_lock:
            ticks = self._tick_buffer.copy()
            self._tick_buffer.clear()
        return ticks

    async def flush_bbo_buffer(self) -> list[dict]:
        """Return and clear buffered BBO snapshots."""
        async with self._buffer_lock:
            bbos = self._bbo_buffer.copy()
            self._bbo_buffer.clear()
        return bbos

    def _parse_trade(self, msg: dict) -> dict | None:
        """Parse Binance WebSocket aggTrade message to internal format."""
        if msg.get("e") != "aggTrade":
            return None
        return {
            "id": msg["a"],
            "symbol": self.symbol.upper(),
            "price": float(msg["p"]),
            "qty": float(msg["q"]),
            "quote_qty": float(msg["p"]) * float(msg["q"]),
            "time": msg["T"],
            "is_buyer_maker": msg["m"],
        }

    def _parse_bbo(self, msg: dict) -> dict | None:
        """Parse Binance WebSocket bookTicker message to internal format."""
        if msg.get("e") != "bookTicker":
            return None
        bid = float(msg["b"])
        ask = float(msg["a"])
        mid = (bid + ask) / 2.0
        return {
            "symbol": self.symbol.upper(),
            "bid": bid,
            "bid_qty": float(msg["B"]),
            "ask": ask,
            "ask_qty": float(msg["A"]),
            "spread": ask - bid,
            "mid": mid,
            "time": msg.get("T", msg.get("E", 0)),
        }
