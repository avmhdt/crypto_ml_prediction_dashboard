"""Unit tests for REST API routes.

Covers T-A01 through T-A08.

Uses an in-memory DuckDB database with synthetic data so tests run without
any external dependencies (no Binance feed, no model files).
"""

from __future__ import annotations

import duckdb
import pytest
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import router as api_router
from backend.config import BAR_TYPES, SYMBOLS
from backend.data.database import init_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_test_app() -> FastAPI:
    """Create a FastAPI app with an in-memory DuckDB for testing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = duckdb.connect(":memory:")
        init_schema(conn)
        _seed_test_data(conn)
        app.state.db = conn
        yield
        conn.close()

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(api_router)
    return test_app


def _seed_test_data(conn: duckdb.DuckDBPyConnection) -> None:
    """Insert synthetic bars and signals into the test database."""
    now_ms = 1_700_000_000_000

    # Insert bars for the first symbol with bar_type "time"
    symbol = SYMBOLS[0]  # BTCUSDT
    bar_type = BAR_TYPES[0]  # time
    for i in range(10):
        conn.execute(
            """INSERT INTO bars (symbol, bar_type, timestamp, open, high, low,
               close, volume, dollar_volume, tick_count, duration_us)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                symbol,
                bar_type,
                now_ms + i * 60_000,
                100.0 + i,
                102.0 + i,
                99.0 + i,
                101.0 + i,
                1000.0 + i * 10,
                100_000.0 + i * 1000,
                200 + i,
                60_000_000,
            ],
        )

    # Insert signals for the first symbol
    for i in range(5):
        conn.execute(
            """INSERT INTO signals (symbol, bar_type, labeling_method, timestamp,
               side, size, entry_price, sl_price, pt_price, time_barrier,
               meta_probability)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                symbol,
                bar_type,
                "triple_barrier",
                now_ms + i * 60_000,
                1 if i % 2 == 0 else -1,
                0.5,
                100.0,
                98.0,
                104.0,
                now_ms + 3_600_000,
                0.75,
            ],
        )


@pytest.fixture(scope="module")
def client():
    """Provide a TestClient connected to the test app."""
    app = _make_test_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# T-A01: GET /api/symbols returns 5 symbols
# ---------------------------------------------------------------------------
class TestSymbolsEndpoint:
    def test_a01_symbols_returns_5(self, client):
        resp = client.get("/api/symbols")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 5
        assert data == SYMBOLS


# ---------------------------------------------------------------------------
# T-A02: GET /api/config returns bar_types, labeling_methods, symbols
# ---------------------------------------------------------------------------
class TestConfigEndpoint:
    def test_a02_config_returns_expected_keys(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "bar_types" in data
        assert "labeling_methods" in data
        assert "symbols" in data
        assert data["symbols"] == SYMBOLS
        assert len(data["bar_types"]) == len(BAR_TYPES)


# ---------------------------------------------------------------------------
# T-A03: GET /api/bars/{symbol}/{bar_type} returns array
# ---------------------------------------------------------------------------
class TestBarsEndpoint:
    def test_a03_bars_returns_array(self, client):
        symbol = SYMBOLS[0]
        bar_type = BAR_TYPES[0]
        resp = client.get(f"/api/bars/{symbol}/{bar_type}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    # -------------------------------------------------------------------
    # T-A04: GET /api/bars respects limit parameter
    # -------------------------------------------------------------------
    def test_a04_bars_respects_limit(self, client):
        symbol = SYMBOLS[0]
        bar_type = BAR_TYPES[0]
        resp = client.get(f"/api/bars/{symbol}/{bar_type}?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 3

    # -------------------------------------------------------------------
    # T-A07: GET /api/bars with invalid symbol returns 404
    # -------------------------------------------------------------------
    def test_a07_invalid_symbol_404(self, client):
        resp = client.get(f"/api/bars/FAKECOIN/{BAR_TYPES[0]}")
        assert resp.status_code == 404

    # -------------------------------------------------------------------
    # T-A08: GET /api/bars with invalid bar_type returns 400
    # -------------------------------------------------------------------
    def test_a08_invalid_bar_type_400(self, client):
        resp = client.get(f"/api/bars/{SYMBOLS[0]}/invalid_bar_type")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# T-A05: GET /api/signals/{symbol} returns array
# ---------------------------------------------------------------------------
class TestSignalsEndpoint:
    def test_a05_signals_returns_array(self, client):
        symbol = SYMBOLS[0]
        resp = client.get(f"/api/signals/{symbol}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0


# ---------------------------------------------------------------------------
# T-A06: POST /api/config/barriers updates config
# ---------------------------------------------------------------------------
class TestBarriersEndpoint:
    def test_a06_update_barriers(self, client):
        payload = {"sl_mult": 3.0, "pt_mult": 4.0, "max_hold": 100}
        resp = client.post("/api/config/barriers", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
