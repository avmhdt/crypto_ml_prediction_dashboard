"""Direct CSV reader for offline training data.

Reads historical trade CSVs directly from D:/Position.One/tick data
without loading into DuckDB. Used by the training pipeline only.
"""
import re
import duckdb
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from backend.config import TICK_DATA_DIR, SYMBOLS

# Pattern to extract date from filenames like BTCUSDT-aggTrades-2025-07-01.csv
_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")


def _file_date_in_range(
    csv_path: Path, start_time: int | None, end_time: int | None,
) -> bool:
    """Check if a CSV file's date falls within the time range.

    Filenames contain dates (e.g., BTCUSDT-aggTrades-2025-07-01.csv).
    This allows skipping files outside the requested range without opening them.
    """
    if start_time is None and end_time is None:
        return True

    match = _DATE_PATTERN.search(csv_path.name)
    if not match:
        return True  # Can't determine date, include it

    file_date = datetime.strptime(match.group(1), "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    # File covers the entire day
    file_start_ms = int(file_date.timestamp() * 1000)
    file_end_ms = file_start_ms + 86_400_000  # +24h

    if start_time is not None and file_end_ms < start_time:
        return False  # File ends before our start
    if end_time is not None and file_start_ms > end_time:
        return False  # File starts after our end
    return True


def read_trades_for_symbol(
    symbol: str,
    data_dir: Path = TICK_DATA_DIR,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Read trade data directly from CSV files on D: drive.

    Uses DuckDB's read_csv_auto for fast columnar scanning without
    loading data into a persistent database. Pre-filters files by
    filename date to avoid scanning thousands of irrelevant CSVs.
    """
    frames = []

    # Read aggTrades (older, more data)
    agg_dir = data_dir / "tick_data" / "futures_um" / symbol
    if agg_dir.exists():
        agg_files = sorted(agg_dir.glob(f"{symbol}-aggTrades-*.csv"))
        agg_files = [f for f in agg_files if _file_date_in_range(f, start_time, end_time)]
        for csv_file in agg_files:
            df = _read_aggtrades_csv(csv_file, symbol, start_time, end_time)
            if df is not None and len(df) > 0:
                frames.append(df)

    # Read trades (newer, higher fidelity)
    trades_dir = data_dir / "trades_data" / "futures_um" / symbol
    if trades_dir.exists():
        trade_files = sorted(trades_dir.glob(f"{symbol}-trades-*.csv"))
        trade_files = [f for f in trade_files if _file_date_in_range(f, start_time, end_time)]
        for csv_file in trade_files:
            df = _read_trades_csv(csv_file, symbol, start_time, end_time)
            if df is not None and len(df) > 0:
                frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["id", "symbol", "price", "qty", "quote_qty",
                      "time", "is_buyer_maker"]
        )

    result = pd.concat(frames, ignore_index=True)
    result.sort_values("time", inplace=True)
    return result


def _read_trades_csv(
    csv_path: Path, symbol: str,
    start_time: int | None, end_time: int | None,
) -> pd.DataFrame | None:
    """Read a single trades CSV: id,price,qty,quote_qty,time,is_buyer_maker"""
    path_str = str(csv_path).replace("\\", "/")
    time_filter = _build_time_filter(start_time, end_time, "column4")

    query = f"""
        SELECT
            CAST(column0 AS BIGINT) AS id,
            '{symbol}' AS symbol,
            CAST(column1 AS DOUBLE) AS price,
            CAST(column2 AS DOUBLE) AS qty,
            CAST(column3 AS DOUBLE) AS quote_qty,
            CAST(column4 AS BIGINT) AS time,
            CAST(column5 AS BOOLEAN) AS is_buyer_maker
        FROM read_csv_auto('{path_str}', header=true, all_varchar=true)
        {time_filter}
    """
    try:
        return duckdb.query(query).fetchdf()
    except Exception:
        return None


def _read_aggtrades_csv(
    csv_path: Path, symbol: str,
    start_time: int | None, end_time: int | None,
) -> pd.DataFrame | None:
    """Read aggTrades CSV: agg_id,price,qty,first_id,last_id,time,is_buyer_maker.

    Handles both headerless (pre-2025) and header (2025+) CSV formats by
    checking if the first value in the file is numeric.
    """
    path_str = str(csv_path).replace("\\", "/")

    # Detect whether file has a header by checking if first cell is numeric
    has_header = False
    try:
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()
            first_val = first_line.split(",")[0]
            has_header = not first_val.isdigit()
    except Exception:
        pass

    if has_header:
        # New format: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
        time_filter = _build_time_filter(start_time, end_time, "transact_time")
        query = f"""
            SELECT
                CAST(agg_trade_id AS BIGINT) AS id,
                '{symbol}' AS symbol,
                CAST(price AS DOUBLE) AS price,
                CAST(quantity AS DOUBLE) AS qty,
                CAST(price AS DOUBLE) * CAST(quantity AS DOUBLE) AS quote_qty,
                CAST(transact_time AS BIGINT) AS time,
                CAST(is_buyer_maker AS BOOLEAN) AS is_buyer_maker
            FROM read_csv_auto('{path_str}', header=true, all_varchar=true)
            {time_filter}
        """
    else:
        # Old format: no header, positional columns
        time_filter = _build_time_filter(start_time, end_time, "column5")
        query = f"""
            SELECT
                CAST(column0 AS BIGINT) AS id,
                '{symbol}' AS symbol,
                CAST(column1 AS DOUBLE) AS price,
                CAST(column2 AS DOUBLE) AS qty,
                CAST(column1 AS DOUBLE) * CAST(column2 AS DOUBLE) AS quote_qty,
                CAST(column5 AS BIGINT) AS time,
                CAST(column6 AS BOOLEAN) AS is_buyer_maker
            FROM read_csv_auto('{path_str}', header=false, all_varchar=true)
            {time_filter}
        """
    try:
        return duckdb.query(query).fetchdf()
    except Exception:
        return None


def _build_time_filter(
    start_time: int | None, end_time: int | None, time_col: str
) -> str:
    """Build a WHERE clause for time filtering."""
    conditions = []
    if start_time is not None:
        conditions.append(f"CAST({time_col} AS BIGINT) >= {start_time}")
    if end_time is not None:
        conditions.append(f"CAST({time_col} AS BIGINT) <= {end_time}")
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""


def list_available_symbols(data_dir: Path = TICK_DATA_DIR) -> list[str]:
    """List symbols that have CSV data available on disk."""
    available = []
    for symbol in SYMBOLS:
        trades_dir = data_dir / "trades_data" / "futures_um" / symbol
        agg_dir = data_dir / "tick_data" / "futures_um" / symbol
        if trades_dir.exists() or agg_dir.exists():
            available.append(symbol)
    return available
