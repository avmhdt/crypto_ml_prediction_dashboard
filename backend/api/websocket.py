"""WebSocket endpoint for live data streaming.

Streams ticks, bars, and signals to connected dashboard clients.
Full pipeline: Binance WS → ticks → bars → features → inference → signals.
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
from backend.pipeline import LivePipeline

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

# Track active live feeds and pipelines per symbol
_active_feeds: dict[str, BinanceLiveFeed] = {}
_feed_tasks: dict[str, asyncio.Task] = {}
_pipelines: dict[str, LivePipeline] = {}


def _get_pipeline(symbol: str) -> LivePipeline:
    """Get or create pipeline for a symbol."""
    if symbol not in _pipelines:
        pipeline = LivePipeline(symbol)
        pipeline.load_models()
        _pipelines[symbol] = pipeline
    return _pipelines[symbol]


async def _run_live_feed(symbol: str):
    """Run live feed for a symbol, processing ticks through the full pipeline."""
    feed = BinanceLiveFeed(symbol)
    _active_feeds[symbol] = feed
    pipeline = _get_pipeline(symbol)

    conn = get_connection()
    last_prune = time.time()

    async def on_batch():
        nonlocal last_prune
        ticks = await feed.flush_buffer()
        if ticks:
            # Insert raw ticks into DuckDB
            insert_ticks_batch(conn, ticks)

            # Process ticks through pipeline (bars + inference)
            events = pipeline.process_ticks_batch(ticks)

            # Persist and broadcast each event
            for event in events:
                if event["type"] == "bar":
                    _insert_bar(conn, event["data"])
                elif event["type"] == "signal":
                    _insert_signal(conn, event["data"])
                await manager.broadcast(symbol, event)

            # Also broadcast latest tick for real-time price display
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


def _insert_bar(conn, bar_data: dict) -> None:
    """Insert a completed bar into DuckDB."""
    try:
        conn.execute(
            """INSERT INTO bars (symbol, bar_type, timestamp, open, high, low,
               close, volume, dollar_volume, tick_count, duration_us)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                bar_data["symbol"], bar_data["bar_type"], bar_data["timestamp"],
                bar_data["open"], bar_data["high"], bar_data["low"],
                bar_data["close"], bar_data["volume"], bar_data["dollar_volume"],
                bar_data["tick_count"], bar_data["duration_us"],
            ],
        )
    except Exception as e:
        logger.error(f"Failed to insert bar: {e}")


def _insert_signal(conn, signal_data: dict) -> None:
    """Insert a signal into DuckDB."""
    try:
        conn.execute(
            """INSERT INTO signals (symbol, bar_type, labeling_method, timestamp,
               side, size, entry_price, sl_price, pt_price, time_barrier,
               meta_probability)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                signal_data["symbol"], signal_data["bar_type"],
                signal_data["labeling_method"], signal_data["timestamp"],
                signal_data["side"], signal_data["size"],
                signal_data["entry_price"], signal_data["sl_price"],
                signal_data["pt_price"], signal_data["time_barrier"],
                signal_data["meta_probability"],
            ],
        )
    except Exception as e:
        logger.error(f"Failed to insert signal: {e}")


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
            try:
                msg = json.loads(data)
                logger.debug(f"Client message: {msg}")
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
