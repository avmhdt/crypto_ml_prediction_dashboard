"""DuckDB connection and schema for LIVE dashboard data only.

This database stores only recent live data (ticks, bars, signals)
for the dashboard display. Historical training data stays on D: drive
as CSV files and is never loaded here.

Also stores walk-forward validation results for OOS performance tracking.
"""
from __future__ import annotations

import json
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

    # ------------------------------------------------------------------
    # Walk-Forward Validation tables
    # ------------------------------------------------------------------
    conn.execute("CREATE SEQUENCE IF NOT EXISTS wf_run_id_seq START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS wf_window_id_seq START 1")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS wf_runs (
            id INTEGER PRIMARY KEY DEFAULT(nextval('wf_run_id_seq')),
            symbol VARCHAR NOT NULL,
            bar_type VARCHAR NOT NULL,
            labeling_method VARCHAR NOT NULL,
            train_days INTEGER NOT NULL,
            test_days INTEGER NOT NULL,
            step_days INTEGER NOT NULL,
            num_windows INTEGER NOT NULL,
            stitched_timestamps VARCHAR,
            stitched_equity VARCHAR,
            stitched_drawdown VARCHAR,
            avg_oos_accuracy DOUBLE,
            avg_oos_sharpe DOUBLE,
            avg_oos_max_dd DOUBLE,
            avg_oos_return DOUBLE,
            avg_oos_win_rate DOUBLE,
            avg_insample_recall DOUBLE,
            overfitting_gap DOUBLE,
            aggregate_stats VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS wf_windows (
            id INTEGER PRIMARY KEY DEFAULT(nextval('wf_window_id_seq')),
            run_id INTEGER NOT NULL,
            window_index INTEGER NOT NULL,
            train_start BIGINT NOT NULL,
            train_end BIGINT NOT NULL,
            test_start BIGINT NOT NULL,
            test_end BIGINT NOT NULL,
            num_train_bars INTEGER,
            num_test_bars INTEGER,
            num_train_samples INTEGER,
            num_test_signals INTEGER,
            primary_recall DOUBLE,
            meta_precision DOUBLE,
            oos_accuracy DOUBLE,
            oos_precision DOUBLE,
            oos_recall DOUBLE,
            sharpe DOUBLE,
            max_dd DOUBLE,
            total_return DOUBLE,
            win_rate DOUBLE,
            num_trades INTEGER,
            timestamps VARCHAR,
            equity VARCHAR,
            drawdown VARCHAR
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


# ------------------------------------------------------------------
# Walk-Forward CRUD
# ------------------------------------------------------------------

def save_wf_result(conn: duckdb.DuckDBPyConnection,
                   result: "WalkForwardResult") -> int:
    """Insert a walk-forward run and its windows into the database.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    result : WalkForwardResult
        Completed walk-forward result (imported from backend.ml.walk_forward).

    Returns
    -------
    int
        The auto-generated run ID.
    """
    # Compute average OOS metrics for the run-level summary
    agg = result.aggregate
    avg_sharpe = agg.get("sharpe", {}).get("mean", 0.0)
    avg_max_dd = agg.get("max_dd", {}).get("mean", 0.0)
    avg_return = agg.get("total_return", {}).get("mean", 0.0)
    avg_win_rate = agg.get("win_rate", {}).get("mean", 0.0)

    conn.execute(
        """INSERT INTO wf_runs (
            symbol, bar_type, labeling_method,
            train_days, test_days, step_days, num_windows,
            stitched_timestamps, stitched_equity, stitched_drawdown,
            avg_oos_accuracy, avg_oos_sharpe, avg_oos_max_dd,
            avg_oos_return, avg_oos_win_rate,
            avg_insample_recall, overfitting_gap, aggregate_stats,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            result.symbol,
            result.bar_type,
            result.labeling_method,
            result.train_days,
            result.test_days,
            result.step_days,
            result.num_windows,
            json.dumps(result.stitched_timestamps),
            json.dumps(result.stitched_equity),
            json.dumps(result.stitched_drawdown),
            result.avg_oos_accuracy,
            avg_sharpe,
            avg_max_dd,
            avg_return,
            avg_win_rate,
            result.avg_insample_recall,
            result.overfitting_gap,
            json.dumps(result.aggregate),
            result.created_at,
        ],
    )

    # Retrieve the auto-generated run ID
    run_id = conn.execute(
        "SELECT currval('wf_run_id_seq')"
    ).fetchone()[0]

    # Insert each window
    for w in result.windows:
        conn.execute(
            """INSERT INTO wf_windows (
                run_id, window_index,
                train_start, train_end, test_start, test_end,
                num_train_bars, num_test_bars,
                num_train_samples, num_test_signals,
                primary_recall, meta_precision,
                oos_accuracy, oos_precision, oos_recall,
                sharpe, max_dd, total_return, win_rate, num_trades,
                timestamps, equity, drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                run_id,
                w.window_index,
                w.train_start,
                w.train_end,
                w.test_start,
                w.test_end,
                w.num_train_bars,
                w.num_test_bars,
                w.num_train_samples,
                w.num_test_signals,
                w.primary_recall,
                w.meta_precision,
                w.oos_accuracy,
                w.oos_precision,
                w.oos_recall,
                w.sharpe,
                w.max_dd,
                w.total_return,
                w.win_rate,
                w.num_trades,
                json.dumps(w.timestamps),
                json.dumps(w.equity),
                json.dumps(w.drawdown),
            ],
        )

    return int(run_id)


def load_wf_runs(
    conn: duckdb.DuckDBPyConnection,
    symbol: str | None = None,
    bar_type: str | None = None,
    labeling_method: str | None = None,
) -> list[dict]:
    """Load walk-forward run summaries (no windows, no curves).

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    symbol : str | None
        Filter by symbol.
    bar_type : str | None
        Filter by bar type.
    labeling_method : str | None
        Filter by labeling method.

    Returns
    -------
    list[dict]
        List of run summary dicts.
    """
    query = """
        SELECT id, symbol, bar_type, labeling_method,
               train_days, test_days, step_days, num_windows,
               avg_oos_accuracy, avg_oos_sharpe, avg_oos_max_dd,
               avg_oos_return, avg_oos_win_rate,
               avg_insample_recall, overfitting_gap,
               created_at
        FROM wf_runs WHERE 1=1
    """
    params: list = []

    if symbol is not None:
        query += " AND symbol = ?"
        params.append(symbol)
    if bar_type is not None:
        query += " AND bar_type = ?"
        params.append(bar_type)
    if labeling_method is not None:
        query += " AND labeling_method = ?"
        params.append(labeling_method)

    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()
    columns = [
        "id", "symbol", "bar_type", "labeling_method",
        "train_days", "test_days", "step_days", "num_windows",
        "avg_oos_accuracy", "avg_oos_sharpe", "avg_oos_max_dd",
        "avg_oos_return", "avg_oos_win_rate",
        "avg_insample_recall", "overfitting_gap",
        "created_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


def load_wf_run(
    conn: duckdb.DuckDBPyConnection,
    run_id: int,
) -> dict | None:
    """Load a full walk-forward run including windows and equity curves.

    JSON-encoded list fields (timestamps, equity, drawdown, aggregate_stats)
    are decoded back to Python objects.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    run_id : int
        The run ID to load.

    Returns
    -------
    dict | None
        Full run dict with ``windows`` list, or ``None`` if not found.
    """
    # Load run
    row = conn.execute(
        "SELECT * FROM wf_runs WHERE id = ?", [run_id]
    ).fetchone()
    if row is None:
        return None

    run_columns = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'wf_runs' ORDER BY ordinal_position"
    ).fetchall()
    run_col_names = [c[0] for c in run_columns]
    run_dict = dict(zip(run_col_names, row))

    # Decode JSON fields
    for field_name in ("stitched_timestamps", "stitched_equity", "stitched_drawdown"):
        val = run_dict.get(field_name)
        if isinstance(val, str):
            run_dict[field_name] = json.loads(val)

    agg_val = run_dict.get("aggregate_stats")
    if isinstance(agg_val, str):
        run_dict["aggregate_stats"] = json.loads(agg_val)

    # Load windows
    win_rows = conn.execute(
        "SELECT * FROM wf_windows WHERE run_id = ? ORDER BY window_index",
        [run_id],
    ).fetchall()

    win_columns = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'wf_windows' ORDER BY ordinal_position"
    ).fetchall()
    win_col_names = [c[0] for c in win_columns]

    windows = []
    for win_row in win_rows:
        win_dict = dict(zip(win_col_names, win_row))
        # Decode JSON array fields
        for field_name in ("timestamps", "equity", "drawdown"):
            val = win_dict.get(field_name)
            if isinstance(val, str):
                win_dict[field_name] = json.loads(val)
        windows.append(win_dict)

    run_dict["windows"] = windows
    return run_dict


def load_wf_latest(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    bar_type: str,
    labeling_method: str,
) -> dict | None:
    """Load the most recent walk-forward run for a given combination.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    symbol : str
        Trading pair symbol.
    bar_type : str
        Bar type.
    labeling_method : str
        Labeling method.

    Returns
    -------
    dict | None
        Full run dict (same format as :func:`load_wf_run`), or ``None``.
    """
    row = conn.execute(
        "SELECT id FROM wf_runs "
        "WHERE symbol = ? AND bar_type = ? AND labeling_method = ? "
        "ORDER BY id DESC LIMIT 1",
        [symbol, bar_type, labeling_method],
    ).fetchone()

    if row is None:
        return None

    return load_wf_run(conn, row[0])
