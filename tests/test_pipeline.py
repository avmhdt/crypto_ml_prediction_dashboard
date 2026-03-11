"""Integration tests for the live pipeline."""
import time
import pytest
import numpy as np

from backend.pipeline import LivePipeline
from backend.bars.base import Bar
from backend.data.database import get_connection, init_schema
from pathlib import Path
import tempfile


@pytest.fixture
def pipeline():
    return LivePipeline("BTCUSDT")


@pytest.fixture
def db_conn():
    """Temporary DuckDB connection for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        conn = get_connection(db_path)
        init_schema(conn)
        yield conn
        conn.close()


class TestPipelineInit:
    def test_creates_all_generators(self, pipeline):
        assert len(pipeline._generators) == 10

    def test_generator_types(self, pipeline):
        expected = {
            "time", "tick", "volume", "dollar",
            "tick_imbalance", "volume_imbalance", "dollar_imbalance",
            "tick_run", "volume_run", "dollar_run",
        }
        assert set(pipeline._generators.keys()) == expected

    def test_no_models_loaded_by_default(self, pipeline):
        assert pipeline._models_loaded is False


class TestPipelineProcessing:
    def test_single_tick_returns_few_events(self, pipeline):
        """A single tick may trigger some bar types (e.g. dollar imbalance)
        but should not produce signals without models."""
        events = pipeline.process_tick(100000, 0.1, int(time.time() * 1000), True)
        signal_events = [e for e in events if e["type"] == "signal"]
        assert len(signal_events) == 0

    def test_many_ticks_produce_bars(self, pipeline):
        now = int(time.time() * 1000)
        events = []
        for i in range(2000):
            evts = pipeline.process_tick(
                100000 + (i % 50) * 10, 0.5,
                now + i * 500, i % 3 == 0
            )
            events.extend(evts)

        bar_events = [e for e in events if e["type"] == "bar"]
        assert len(bar_events) > 0

    def test_bar_events_have_required_fields(self, pipeline):
        now = int(time.time() * 1000)
        events = []
        for i in range(2000):
            evts = pipeline.process_tick(
                100000 + i, 0.5, now + i * 500, True
            )
            events.extend(evts)

        bar_events = [e for e in events if e["type"] == "bar"]
        assert len(bar_events) > 0
        bar = bar_events[0]["data"]
        required_fields = {
            "symbol", "bar_type", "timestamp", "open", "high",
            "low", "close", "volume", "dollar_volume", "tick_count",
        }
        assert required_fields.issubset(set(bar.keys()))

    def test_batch_processing(self, pipeline):
        now = int(time.time() * 1000)
        ticks = [
            {"price": 100000 + i, "qty": 0.1,
             "time": now + i * 1000, "is_buyer_maker": True}
            for i in range(200)
        ]
        events = pipeline.process_ticks_batch(ticks)
        assert isinstance(events, list)

    def test_no_signals_without_models(self, pipeline):
        now = int(time.time() * 1000)
        events = []
        for i in range(2000):
            evts = pipeline.process_tick(
                100000 + i, 0.5, now + i * 500, True
            )
            events.extend(evts)

        signal_events = [e for e in events if e["type"] == "signal"]
        assert len(signal_events) == 0

    def test_bar_buffers_capped(self, pipeline):
        now = int(time.time() * 1000)
        for i in range(5000):
            pipeline.process_tick(100000, 0.5, now + i * 500, True)

        for buf in pipeline._bar_buffers.values():
            assert len(buf) <= pipeline._max_buffer


class TestDatabaseIntegration:
    def test_insert_and_load_bars(self, db_conn):
        db_conn.execute(
            """INSERT INTO bars (symbol, bar_type, timestamp, open, high, low,
               close, volume, dollar_volume, tick_count, duration_us)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ["BTCUSDT", "time", 1000000, 100.0, 101.0, 99.0,
             100.5, 10.0, 1005.0, 100, 60000000],
        )
        from backend.data.database import load_bars
        df = load_bars(db_conn, "BTCUSDT", "time")
        assert len(df) == 1
        assert df.iloc[0]["close"] == 100.5

    def test_insert_and_load_signals(self, db_conn):
        db_conn.execute(
            """INSERT INTO signals (symbol, bar_type, labeling_method, timestamp,
               side, size, entry_price, sl_price, pt_price, time_barrier,
               meta_probability)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ["BTCUSDT", "time", "triple_barrier", 1000000,
             1, 0.75, 100.0, 98.0, 104.0, 1003600000, 0.85],
        )
        from backend.data.database import load_signals
        df = load_signals(db_conn, "BTCUSDT")
        assert len(df) == 1
        assert df.iloc[0]["side"] == 1
        assert df.iloc[0]["meta_probability"] == 0.85

    def test_prune_old_data(self, db_conn):
        now_ms = int(time.time() * 1000)
        old_time = now_ms - 8 * 86_400_000  # 8 days ago

        db_conn.execute(
            """INSERT INTO bars (symbol, bar_type, timestamp, open, high, low,
               close, volume, dollar_volume, tick_count, duration_us)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ["BTCUSDT", "time", old_time, 100.0, 101.0, 99.0,
             100.5, 10.0, 1005.0, 100, 60000000],
        )

        from backend.data.database import prune_old_data, load_bars
        prune_old_data(db_conn, now_ms)
        df = load_bars(db_conn, "BTCUSDT", "time")
        assert len(df) == 0


class TestBarGenerators:
    def test_time_bars_emit_on_interval(self):
        from backend.bars.time_bars import TimeBars
        gen = TimeBars(symbol="BTCUSDT", interval="1min")
        now = int(time.time() * 1000)
        bars = []
        for i in range(120):  # 2 minutes of 1-second ticks
            result = gen.process_tick(100000, 0.1, now + i * 1000, True)
            bars.extend(result)
        assert len(bars) >= 1

    def test_tick_bars_emit_on_count(self):
        from backend.bars.information_bars import TickBars
        gen = TickBars(symbol="BTCUSDT", tick_count=50)
        now = int(time.time() * 1000)
        bars = []
        for i in range(200):
            result = gen.process_tick(100000, 0.1, now + i, True)
            bars.extend(result)
        assert len(bars) == 4  # 200 / 50

    def test_volume_bars_emit_on_threshold(self):
        from backend.bars.information_bars import VolumeBars
        gen = VolumeBars(symbol="BTCUSDT", volume_threshold=10.0)
        now = int(time.time() * 1000)
        bars = []
        for i in range(100):
            result = gen.process_tick(100000, 0.5, now + i, True)
            bars.extend(result)
        assert len(bars) == 5  # 100 * 0.5 / 10.0

    def test_dollar_bars_emit_on_threshold(self):
        from backend.bars.information_bars import DollarBars
        gen = DollarBars(symbol="BTCUSDT", dollar_threshold=50000.0)
        now = int(time.time() * 1000)
        bars = []
        for i in range(100):
            result = gen.process_tick(100.0, 10.0, now + i, True)
            bars.extend(result)
        # Each tick: $100 * 10 = $1000. 100 ticks = $100k. $100k / $50k = 2
        assert len(bars) == 2

    def test_bar_ohlcv_correctness(self):
        from backend.bars.information_bars import TickBars
        gen = TickBars(symbol="BTCUSDT", tick_count=5)
        now = int(time.time() * 1000)

        prices = [100, 105, 95, 102, 98]
        bars = []
        for i, p in enumerate(prices):
            result = gen.process_tick(float(p), 1.0, now + i, True)
            bars.extend(result)

        assert len(bars) == 1
        bar = bars[0]
        assert bar.open == 100
        assert bar.high == 105
        assert bar.low == 95
        assert bar.close == 98
        assert bar.volume == 5.0
        assert bar.tick_count == 5
