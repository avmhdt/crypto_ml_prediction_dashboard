"""DuckDB connection and schema for LIVE dashboard data only.

This database stores only recent live data (ticks, bars, signals)
for the dashboard display. Historical training data stays on D: drive
as CSV files and is never loaded here.
"""
import duckdb
from pathlib import Path
from backend.config import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables for live data storage with rolling retention."""
    # Sequence for signal IDs
    conn.execute("CREATE SEQUENCE IF NOT EXISTS signal_id_seq START 1")

    # Live ticks from Binance WebSocket (24h rolling)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            id BIGINT,
            symbol VARCHAR,
            price DOUBLE,
            qty DOUBLE,
            quote_qty DOUBLE,
            time BIGINT,
            is_buyer_maker BOOLEAN
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time
        ON ticks (symbol, time)
    """)

    # Computed bars from live ticks (7d rolling)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol VARCHAR,
            bar_type VARCHAR,
            timestamp BIGINT,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            tick_count INTEGER,
            duration_us BIGINT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bars_symbol_type_time
        ON bars (symbol, bar_type, timestamp)
    """)

    # ML signals (30d rolling)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY DEFAULT(nextval('signal_id_seq')),
            symbol VARCHAR,
            bar_type VARCHAR,
            labeling_method VARCHAR,
            timestamp BIGINT,
            side INTEGER,
            size DOUBLE,
            entry_price DOUBLE,
            sl_price DOUBLE,
            pt_price DOUBLE,
            time_barrier BIGINT,
            meta_probability DOUBLE
        )
    """)

    # Simulated fills from realistic order fill simulation (30d rolling)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS sim_fill_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sim_fills (
            id INTEGER PRIMARY KEY DEFAULT(nextval('sim_fill_id_seq')),
            symbol VARCHAR,
            signal_id INTEGER,
            side INTEGER,
            fill_price DOUBLE,
            fill_qty DOUBLE,
            fill_time BIGINT,
            order_type VARCHAR,
            limit_price DOUBLE,
            submitted_time BIGINT,
            queue_wait_ms BIGINT,
            exchange_fee DOUBLE,
            funding_cost DOUBLE,
            spread_cost DOUBLE,
            slippage DOUBLE,
            market_impact DOUBLE,
            total_cost DOUBLE,
            meta_probability DOUBLE
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sim_fills_symbol_time
        ON sim_fills (symbol, fill_time)
    """)

    # BBO (best bid/offer) snapshots for spread calibration (24h rolling)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bbo_log (
            symbol VARCHAR,
            time BIGINT,
            bid DOUBLE,
            bid_qty DOUBLE,
            ask DOUBLE,
            ask_qty DOUBLE,
            spread DOUBLE,
            mid DOUBLE
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bbo_log_symbol_time
        ON bbo_log (symbol, time)
    """)


def prune_old_data(conn: duckdb.DuckDBPyConnection, now_ms: int) -> None:
    """Remove data older than retention windows."""
    ms_per_hour = 3_600_000
    ms_per_day = 86_400_000

    # Ticks: 24h
    conn.execute(
        "DELETE FROM ticks WHERE time < ?",
        [now_ms - 24 * ms_per_hour]
    )
    # Bars: 7d
    conn.execute(
        "DELETE FROM bars WHERE timestamp < ?",
        [now_ms - 7 * ms_per_day]
    )
    # Signals: 30d
    conn.execute(
        "DELETE FROM signals WHERE timestamp < ?",
        [now_ms - 30 * ms_per_day]
    )
    # Simulated fills: 30d
    conn.execute(
        "DELETE FROM sim_fills WHERE fill_time < ?",
        [now_ms - 30 * ms_per_day]
    )
    # BBO log: 24h
    conn.execute(
        "DELETE FROM bbo_log WHERE time < ?",
        [now_ms - 24 * ms_per_hour]
    )


def insert_ticks_batch(conn: duckdb.DuckDBPyConnection,
                       ticks: list[dict]) -> int:
    """Micro-batch insert of live ticks."""
    if not ticks:
        return 0
    conn.executemany(
        """INSERT INTO ticks (id, symbol, price, qty, quote_qty, time, is_buyer_maker)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [(t["id"], t["symbol"], t["price"], t["qty"], t["quote_qty"],
          t["time"], t["is_buyer_maker"]) for t in ticks]
    )
    return len(ticks)


def insert_sim_fill(conn: duckdb.DuckDBPyConnection, fill: dict) -> None:
    """Insert a simulated fill record."""
    conn.execute(
        """INSERT INTO sim_fills (symbol, signal_id, side, fill_price, fill_qty,
           fill_time, order_type, limit_price, submitted_time, queue_wait_ms,
           exchange_fee, funding_cost, spread_cost, slippage, market_impact,
           total_cost, meta_probability)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [fill["symbol"], fill.get("signal_id"), fill["side"], fill["fill_price"],
         fill["fill_qty"], fill["fill_time"], fill["order_type"],
         fill["limit_price"], fill["submitted_time"], fill["queue_wait_ms"],
         fill["exchange_fee"], fill["funding_cost"], fill["spread_cost"],
         fill["slippage"], fill["market_impact"], fill["total_cost"],
         fill.get("meta_probability", 0.0)]
    )


def insert_bbo(conn: duckdb.DuckDBPyConnection, bbo: dict) -> None:
    """Insert a BBO snapshot."""
    conn.execute(
        """INSERT INTO bbo_log (symbol, time, bid, bid_qty, ask, ask_qty, spread, mid)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [bbo["symbol"], bbo["time"], bbo["bid"], bbo["bid_qty"],
         bbo["ask"], bbo["ask_qty"], bbo["spread"], bbo["mid"]]
    )


def load_bars(conn: duckdb.DuckDBPyConnection, symbol: str, bar_type: str,
              start_time: int | None = None, end_time: int | None = None,
              limit: int = 1000):
    query = "SELECT * FROM bars WHERE symbol = ? AND bar_type = ?"
    params: list = [symbol, bar_type]
    if start_time is not None:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time is not None:
        query += " AND timestamp <= ?"
        params.append(end_time)
    query += f" ORDER BY timestamp DESC LIMIT {limit}"
    return conn.execute(query, params).fetchdf()


def load_signals(conn: duckdb.DuckDBPyConnection, symbol: str,
                 bar_type: str | None = None,
                 labeling_method: str | None = None,
                 limit: int = 100):
    query = "SELECT * FROM signals WHERE symbol = ?"
    params: list = [symbol]
    if bar_type is not None:
        query += " AND bar_type = ?"
        params.append(bar_type)
    if labeling_method is not None:
        query += " AND labeling_method = ?"
        params.append(labeling_method)
    query += f" ORDER BY timestamp DESC LIMIT {limit}"
    return conn.execute(query, params).fetchdf()
