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
