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
    get_connection, insert_ticks_batch, insert_bbo, prune_old_data,
)
from backend.data.live_feed import BinanceLiveFeed
from backend.pipeline import LivePipeline
from backend.simulation.fill_simulator import OrderFillSimulator
from backend.simulation.config import SimulationConfig

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

# Track active live feeds, pipelines, and fill simulators per symbol
_active_feeds: dict[str, BinanceLiveFeed] = {}
_feed_tasks: dict[str, asyncio.Task] = {}
_pipelines: dict[str, LivePipeline] = {}
_fill_simulators: dict[str, OrderFillSimulator] = {}

# Shared DuckDB connection for all feed writers (avoids lock contention)
_shared_conn = None


def _get_shared_conn():
    """Get or create a shared DuckDB connection for feed writes."""
    global _shared_conn
    if _shared_conn is None:
        _shared_conn = get_connection()
        logger.info("Shared DuckDB connection opened for live feeds")
    return _shared_conn


def _get_pipeline(symbol: str) -> LivePipeline:
    """Get or create pipeline for a symbol."""
    if symbol not in _pipelines:
        pipeline = LivePipeline(symbol)
        pipeline.load_models()
        _pipelines[symbol] = pipeline
    return _pipelines[symbol]


def _get_fill_simulator(symbol: str) -> OrderFillSimulator:
    """Get or create fill simulator for a symbol."""
    if symbol not in _fill_simulators:
        config = SimulationConfig(mode="realistic")
        _fill_simulators[symbol] = OrderFillSimulator(symbol, config)
    return _fill_simulators[symbol]


async def _run_live_feed(symbol: str):
    """Run live feed for a symbol, processing ticks through the full pipeline.

    Individual batch errors are caught and logged without killing the feed.
    The feed only dies on CancelledError (shutdown) or unrecoverable init errors.
    """
    consecutive_errors = 0
    max_consecutive_errors = 30  # Give up after 30 consecutive failures

    try:
        logger.info(f"Starting live feed for {symbol}")
        feed = BinanceLiveFeed(symbol)
        _active_feeds[symbol] = feed
        pipeline = _get_pipeline(symbol)
        logger.info(f"Pipeline ready for {symbol}, models_loaded={pipeline._models_loaded}")

        conn = _get_shared_conn()
        last_prune = time.time()
        batch_count = 0

        fill_sim = _get_fill_simulator(symbol)

        async def on_batch():
            nonlocal last_prune, batch_count
            ticks = await feed.flush_buffer()
            if ticks:
                batch_count += 1
                # Insert raw ticks into DuckDB
                try:
                    insert_ticks_batch(conn, ticks)
                except Exception as e:
                    logger.error(f"[{symbol}] Failed to insert {len(ticks)} ticks: {e}")

                # Process ticks through pipeline (bars + inference)
                events = pipeline.process_ticks_batch(ticks)

                # Feed ticks to fill simulator for queue advancement
                for tick in ticks:
                    try:
                        fill_sim.on_tick(
                            tick["price"], tick["qty"],
                            tick["time"], tick["is_buyer_maker"],
                        )
                    except Exception:
                        pass  # fill sim errors don't block pipeline

                # Log periodic status
                if batch_count % 30 == 0:
                    logger.info(f"[{symbol}] batch #{batch_count}: {len(ticks)} ticks, {len(events)} events")

                # Persist and broadcast each event
                for event in events:
                    if event["type"] == "bar":
                        _insert_bar(conn, event["data"])
                        logger.info(f"New {event['data']['bar_type']} bar for {symbol}: close={event['data']['close']}")
                    elif event["type"] == "signal":
                        _insert_signal(conn, event["data"])
                        # Submit signal to fill simulator
                        try:
                            fill_sim.submit_order(event["data"])
                        except Exception:
                            pass
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

            # Process BBO updates
            bbos = await feed.flush_bbo_buffer()
            for bbo in bbos:
                try:
                    fill_sim.on_bbo(
                        bbo["bid"], bbo["bid_qty"],
                        bbo["ask"], bbo["ask_qty"],
                        bbo["time"],
                    )
                    insert_bbo(conn, bbo)
                except Exception:
                    pass
                await manager.broadcast(symbol, {
                    "type": "bbo",
                    "data": bbo,
                })

            # Prune old data every 5 minutes
            if time.time() - last_prune > 300:
                try:
                    prune_old_data(conn, int(time.time() * 1000))
                except Exception as e:
                    logger.error(f"[{symbol}] Prune failed: {e}")
                last_prune = time.time()

        # Start Binance WebSocket feed in background
        feed_task = asyncio.create_task(feed.start())
        logger.info(f"[{symbol}] Binance feed task started, entering batch loop")

        try:
            while True:
                await asyncio.sleep(1)
                try:
                    await on_batch()
                    consecutive_errors = 0  # Reset on success
                except asyncio.CancelledError:
                    raise  # Let CancelledError propagate
                except Exception:
                    consecutive_errors += 1
                    logger.exception(f"[{symbol}] on_batch error #{consecutive_errors}")
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[{symbol}] Too many consecutive errors, feed dying")
                        break
                    await asyncio.sleep(min(consecutive_errors, 5))
        except asyncio.CancelledError:
            logger.info(f"[{symbol}] Feed cancelled, shutting down")
            await feed.stop()
            feed_task.cancel()
    except Exception:
        logger.exception(f"Live feed for {symbol} crashed during init")


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
        if symbol_upper in _feed_tasks and _feed_tasks[symbol_upper].done():
            try:
                exc = _feed_tasks[symbol_upper].exception()
                if exc:
                    logger.error(f"Previous feed for {symbol_upper} failed: {exc}")
            except asyncio.CancelledError:
                pass
        logger.info(f"Launching feed task for {symbol_upper}")
        _feed_tasks[symbol_upper] = asyncio.create_task(
            _run_live_feed(symbol_upper)
        )


async def _feed_watchdog():
    """Periodically check and restart dead feed tasks."""
    while True:
        await asyncio.sleep(10)
        for symbol in SYMBOLS:
            if symbol in _feed_tasks and _feed_tasks[symbol].done():
                logger.warning(f"Watchdog: restarting dead feed for {symbol}")
                _ensure_feed_running(symbol)


def start_all_feeds():
    """Start live feeds for all configured symbols (called at app startup)."""
    for symbol in SYMBOLS:
        _ensure_feed_running(symbol)
    # Start watchdog to auto-restart dead feeds
    asyncio.create_task(_feed_watchdog())


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
