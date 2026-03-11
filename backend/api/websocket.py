"""WebSocket endpoint for live data streaming.

Streams ticks, bars, and signals to connected dashboard clients.
"""
import asyncio
import json
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.config import SYMBOLS
from backend.data.database import (
    get_connection, insert_ticks_batch, prune_old_data,
)
from backend.data.live_feed import BinanceLiveFeed

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections per symbol."""

    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        if symbol not in self.active:
            self.active[symbol] = []
        self.active[symbol].append(websocket)

    def disconnect(self, websocket: WebSocket, symbol: str):
        if symbol in self.active:
            self.active[symbol] = [
                ws for ws in self.active[symbol] if ws != websocket
            ]

    async def broadcast(self, symbol: str, message: dict):
        if symbol not in self.active:
            return
        dead = []
        for ws in self.active[symbol]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active[symbol].remove(ws)


manager = ConnectionManager()

# Track active live feeds per symbol
_active_feeds: dict[str, BinanceLiveFeed] = {}
_feed_tasks: dict[str, asyncio.Task] = {}


async def _run_live_feed(symbol: str):
    """Run live feed for a symbol, broadcasting ticks to connected clients."""
    feed = BinanceLiveFeed(symbol)
    _active_feeds[symbol] = feed

    conn = get_connection()
    last_prune = time.time()

    async def on_batch():
        nonlocal last_prune
        ticks = await feed.flush_buffer()
        if ticks:
            # Insert into DuckDB
            insert_ticks_batch(conn, ticks)

            # Broadcast latest tick to clients
            latest = ticks[-1]
            await manager.broadcast(symbol, {
                "type": "tick",
                "data": {
                    "price": latest["price"],
                    "qty": latest["qty"],
                    "time": latest["time"],
                    "is_buyer_maker": latest["is_buyer_maker"],
                },
            })

        # Prune old data every 5 minutes
        if time.time() - last_prune > 300:
            prune_old_data(conn, int(time.time() * 1000))
            last_prune = time.time()

    # Start feed in background
    feed_task = asyncio.create_task(feed.start())

    try:
        while True:
            await asyncio.sleep(1)
            await on_batch()
    except asyncio.CancelledError:
        await feed.stop()
        feed_task.cancel()
    finally:
        conn.close()


def _ensure_feed_running(symbol: str):
    """Ensure live feed is running for a symbol."""
    symbol_upper = symbol.upper()
    if symbol_upper not in _feed_tasks or _feed_tasks[symbol_upper].done():
        _feed_tasks[symbol_upper] = asyncio.create_task(
            _run_live_feed(symbol_upper)
        )


@router.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        await websocket.close(code=4004, reason=f"Unknown symbol: {symbol}")
        return

    await manager.connect(websocket, symbol)
    _ensure_feed_running(symbol)

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            # Client can send config updates or subscribe to specific bar types
            try:
                msg = json.loads(data)
                logger.debug(f"Client message: {msg}")
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
