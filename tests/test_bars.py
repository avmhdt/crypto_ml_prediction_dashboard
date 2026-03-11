"""Unit tests for bar generators (T-B01 through T-B14).

Covers time bars, information bars (tick/volume/dollar), imbalance bars
(tick/volume/dollar), run bars (tick/volume/dollar), OHLCV invariants,
output schema, and EWMA state persistence.
"""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from backend.bars.base import Bar, BarAccumulator, EWMABarGenerator
from backend.bars.time_bars import TimeBars
from backend.bars.information_bars import TickBars, VolumeBars, DollarBars
from backend.bars.imbalance_bars import (
    TickImbalanceBars,
    VolumeImbalanceBars,
    DollarImbalanceBars,
)
from backend.bars.run_bars import TickRunBars, VolumeRunBars, DollarRunBars
from backend.bars import BAR_CLASSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_BAR_FIELDS = {
    "symbol", "bar_type", "timestamp", "open", "high", "low", "close",
    "volume", "dollar_volume", "tick_count", "duration_us",
}


def _generate_trades(
    n: int,
    base_price: float = 50_000.0,
    base_qty: float = 0.1,
    start_ms: int = 1_700_000_000_000,
    interval_ms: int = 100,
    price_noise: float = 50.0,
    qty_noise: float = 0.05,
    seed: int = 42,
):
    """Return (prices, qtys, times, is_buyer_makers) arrays for *n* trades."""
    rng = np.random.RandomState(seed)
    prices = base_price + rng.uniform(-price_noise, price_noise, n)
    qtys = base_qty + rng.uniform(0, qty_noise, n)
    times = np.arange(n, dtype=np.int64) * interval_ms + start_ms
    is_buyer_makers = rng.choice([True, False], size=n)
    return prices.astype(np.float64), qtys.astype(np.float64), times, is_buyer_makers


def _feed_trades(gen, prices, qtys, times, makers):
    """Feed trades one-by-one and collect all emitted bars."""
    bars = []
    for i in range(len(prices)):
        result = gen.process_tick(
            float(prices[i]), float(qtys[i]),
            int(times[i]), bool(makers[i]),
        )
        bars.extend(result)
    return bars


# ===================================================================
# T-B01: Time bar aggregates trades into 1-min OHLCV
# ===================================================================

class TestTimeBars:
    def test_b01_time_bar_aggregates_into_1min_ohlcv(self):
        """100 trades spanning 3 minutes should produce 3 bars."""
        gen = TimeBars("BTCUSDT", interval="1min")
        start = 1_700_000_000_000  # arbitrary epoch ms

        # Create 100 trades: ~33 per minute spread across minutes 0, 1, 2
        n = 100
        prices = np.full(n, 50_000.0)
        qtys = np.full(n, 0.1)
        # Distribute trades: 0-32 in minute 0, 33-65 in minute 1, 66-99 in minute 2
        times = np.zeros(n, dtype=np.int64)
        makers = np.zeros(n, dtype=bool)

        # Minute 0: trades at start + 0ms .. start + 32*1800ms (still within 60s)
        for i in range(33):
            times[i] = start + i * 1800  # 1800ms apart = 59.4s total
            prices[i] = 50_000 + i * 10  # rising prices

        # Minute 1: trades starting at start + 60_000ms
        for i in range(33, 66):
            times[i] = start + 60_000 + (i - 33) * 1800
            prices[i] = 50_500 + (i - 33) * 5

        # Minute 2: trades starting at start + 120_000ms
        for i in range(66, 100):
            times[i] = start + 120_000 + (i - 66) * 1700
            prices[i] = 51_000 - (i - 66) * 3

        bars = _feed_trades(gen, prices, qtys, times, makers)

        # We should have exactly 2 completed bars (minute 0 and minute 1).
        # Minute 2's bar hasn't closed yet because no tick beyond minute 3 exists.
        # To flush the third bar, send a tick in minute 3.
        flush_result = gen.process_tick(
            51_000.0, 0.1, start + 180_000, False,
        )
        bars.extend(flush_result)

        assert len(bars) == 3, f"Expected 3 bars, got {len(bars)}"

        for bar in bars:
            assert bar.bar_type == "time"
            assert bar.symbol == "BTCUSDT"
            assert bar.volume > 0
            assert bar.tick_count > 0

    # ===================================================================
    # T-B02: Time bar handles empty interval
    # ===================================================================
    def test_b02_time_bar_empty_interval(self):
        """No trades in a window should produce no bar for that window."""
        gen = TimeBars("ETHUSDT", interval="1min")
        start = 1_700_000_000_000

        # Trades only in minute 0, spaced 2000ms apart => 10 trades, 18s total
        bars_m0 = []
        for i in range(10):
            bars_m0.extend(gen.process_tick(
                3_000.0 + i, 1.0, start + i * 2000, False,
            ))

        # Jump to minute 2 (skip minute 1 entirely) to flush minute-0 bar
        bars_gap = gen.process_tick(3_050.0, 1.0, start + 120_001, False)

        # Should have emitted exactly 1 bar (minute 0).
        # Minute 1 had no trades, so no bar for that window.
        all_bars = bars_m0 + list(bars_gap)
        assert len(all_bars) == 1, (
            f"Expected 1 bar (minute 0 only, skipping empty minute 1), got {len(all_bars)}"
        )


# ===================================================================
# T-B03: Tick bar emits after N ticks
# ===================================================================

class TestTickBars:
    def test_b03_tick_bar_emits_after_n_ticks(self):
        """2500 trades with threshold=1000 should produce 2 bars."""
        gen = TickBars("BTCUSDT", tick_count=1000)
        prices, qtys, times, makers = _generate_trades(2500)
        bars = _feed_trades(gen, prices, qtys, times, makers)
        assert len(bars) == 2, f"Expected 2 bars from 2500 ticks @ threshold 1000, got {len(bars)}"
        for bar in bars:
            assert bar.tick_count == 1000
            assert bar.bar_type == "tick"


# ===================================================================
# T-B04: Volume bar emits after threshold volume
# ===================================================================

class TestVolumeBars:
    def test_b04_volume_bar_emits_after_threshold(self):
        """Volume bars should emit when cumulative volume >= threshold."""
        gen = VolumeBars("BTCUSDT", volume_threshold=10.0)
        # Each trade has qty ~0.1-0.15, so ~10 volume per ~80 trades
        prices, qtys, times, makers = _generate_trades(500, base_qty=0.1, qty_noise=0.05)
        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 1, "Expected at least 1 volume bar"
        for bar in bars:
            assert bar.volume >= 10.0, f"Bar volume {bar.volume} < threshold 10.0"
            assert bar.bar_type == "volume"


# ===================================================================
# T-B05: Dollar bar emits after threshold dollar volume
# ===================================================================

class TestDollarBars:
    def test_b05_dollar_bar_emits_after_threshold(self):
        """Dollar bars should emit when cumulative dollar volume >= threshold."""
        # price ~50k, qty ~0.1 => each trade ~5000 dollars.
        # threshold = 100_000 => ~20 trades per bar
        gen = DollarBars("BTCUSDT", dollar_threshold=100_000.0)
        prices, qtys, times, makers = _generate_trades(500, base_price=50_000.0, base_qty=0.1)
        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 1, "Expected at least 1 dollar bar"
        for bar in bars:
            assert bar.dollar_volume >= 100_000.0, (
                f"Bar dollar_volume {bar.dollar_volume} < threshold 100000"
            )
            assert bar.bar_type == "dollar"


# ===================================================================
# T-B06: Tick imbalance bar EWMA converges
# ===================================================================

class TestTickImbalanceBars:
    def test_b06_tick_imbalance_ewma_converges(self):
        """10000 trades with biased buy/sell runs -- EWMA should converge."""
        gen = TickImbalanceBars(
            "BTCUSDT",
            expected_num_ticks_init=100,
            num_prev_bars=20,
        )
        n = 10_000
        rng = np.random.RandomState(42)
        prices = 50_000.0 + rng.uniform(-100, 100, n)
        qtys = np.full(n, 0.1)
        times = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        # Biased toward buys (70%) to create imbalance that exceeds threshold
        makers = rng.choice([True, False], size=n, p=[0.3, 0.7])

        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 2, f"Expected >=2 imbalance bars, got {len(bars)}"

        # _expected_ticks should be clamped within [50, 200] (0.5x-2x of init 100)
        assert gen._expected_ticks >= gen._min_expected
        assert gen._expected_ticks <= gen._max_expected


# ===================================================================
# T-B07: Volume imbalance bar EWMA converges
# ===================================================================

class TestVolumeImbalanceBars:
    def test_b07_volume_imbalance_ewma_converges(self):
        """Volume imbalance EWMA should converge with biased flow."""
        gen = VolumeImbalanceBars(
            "BTCUSDT",
            expected_num_ticks_init=100,
            num_prev_bars=20,
        )
        n = 10_000
        rng = np.random.RandomState(123)
        prices = 50_000.0 + rng.uniform(-100, 100, n)
        qtys = 0.1 + rng.uniform(0, 0.05, n)
        times = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        # Biased toward buys (70%) to create volume imbalance
        makers = rng.choice([True, False], size=n, p=[0.3, 0.7])

        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 2, f"Expected >=2 volume imbalance bars, got {len(bars)}"
        assert gen._expected_ticks >= gen._min_expected
        assert gen._expected_ticks <= gen._max_expected


# ===================================================================
# T-B08: Dollar imbalance bar EWMA converges
# ===================================================================

class TestDollarImbalanceBars:
    def test_b08_dollar_imbalance_ewma_converges(self):
        """Dollar imbalance EWMA should converge with balanced flow."""
        gen = DollarImbalanceBars(
            "BTCUSDT",
            expected_num_ticks_init=100,
            num_prev_bars=20,
        )
        n = 10_000
        rng = np.random.RandomState(456)
        prices = 50_000.0 + rng.uniform(-200, 200, n)
        qtys = 0.1 + rng.uniform(0, 0.05, n)
        times = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        makers = np.array([i % 2 == 0 for i in range(n)])

        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 2, f"Expected >=2 dollar imbalance bars, got {len(bars)}"
        assert gen._expected_ticks >= gen._min_expected
        assert gen._expected_ticks <= gen._max_expected


# ===================================================================
# T-B09: Tick run bar EWMA converges
# ===================================================================

class TestTickRunBars:
    def test_b09_tick_run_ewma_converges(self):
        """Tick run bars EWMA should converge over 10000 alternating trades."""
        gen = TickRunBars(
            "BTCUSDT",
            expected_num_ticks_init=100,
            num_prev_bars=20,
        )
        n = 10_000
        prices = np.full(n, 50_000.0)
        qtys = np.full(n, 0.1)
        times = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        makers = np.array([i % 2 == 0 for i in range(n)])

        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 2, f"Expected >=2 tick run bars, got {len(bars)}"
        assert gen._expected_ticks >= gen._min_expected
        assert gen._expected_ticks <= gen._max_expected


# ===================================================================
# T-B10: Volume run bar emits correctly
# ===================================================================

class TestVolumeRunBars:
    def test_b10_volume_run_bar_emits(self):
        """Volume run bars should emit bars from realistic trade data."""
        gen = VolumeRunBars(
            "BTCUSDT",
            expected_num_ticks_init=100,
            num_prev_bars=20,
        )
        n = 10_000
        rng = np.random.RandomState(789)
        prices = 50_000.0 + rng.uniform(-100, 100, n)
        qtys = 0.1 + rng.uniform(0, 0.1, n)
        times = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        # Biased toward buys to create stronger runs
        makers = rng.choice([True, False], size=n, p=[0.3, 0.7])

        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 1, "Expected at least 1 volume run bar"
        for bar in bars:
            assert bar.bar_type == "volume_run"
            assert bar.volume > 0
            assert bar.tick_count > 0


# ===================================================================
# T-B11: Dollar run bar emits correctly
# ===================================================================

class TestDollarRunBars:
    def test_b11_dollar_run_bar_emits(self):
        """Dollar run bars should emit bars from realistic trade data."""
        gen = DollarRunBars(
            "BTCUSDT",
            expected_num_ticks_init=100,
            num_prev_bars=20,
        )
        n = 10_000
        rng = np.random.RandomState(101)
        prices = 50_000.0 + rng.uniform(-200, 200, n)
        qtys = 0.1 + rng.uniform(0, 0.1, n)
        times = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        makers = rng.choice([True, False], size=n, p=[0.3, 0.7])

        bars = _feed_trades(gen, prices, qtys, times, makers)

        assert len(bars) >= 1, "Expected at least 1 dollar run bar"
        for bar in bars:
            assert bar.bar_type == "dollar_run"
            assert bar.dollar_volume > 0
            assert bar.tick_count > 0


# ===================================================================
# T-B12: Bar output schema has all required fields
# ===================================================================

class TestBarSchema:
    def test_b12_bar_output_schema(self):
        """Bar.to_dict() must contain all required OHLCV fields."""
        bar = Bar(
            symbol="BTCUSDT",
            bar_type="time",
            timestamp=1_700_000_060_000,
            open=50_000.0,
            high=50_100.0,
            low=49_900.0,
            close=50_050.0,
            volume=5.0,
            dollar_volume=250_000.0,
            tick_count=50,
            duration_us=60_000_000,
        )
        d = bar.to_dict()
        missing = REQUIRED_BAR_FIELDS - set(d.keys())
        assert not missing, f"Missing fields in bar dict: {missing}"

        # Verify types
        assert isinstance(d["symbol"], str)
        assert isinstance(d["bar_type"], str)
        assert isinstance(d["timestamp"], int)
        assert isinstance(d["tick_count"], int)
        assert isinstance(d["duration_us"], int)
        for field in ("open", "high", "low", "close", "volume", "dollar_volume"):
            assert isinstance(d[field], float), f"{field} should be float, got {type(d[field])}"

    def test_b12_all_bar_types_produce_valid_schema(self):
        """Every bar generator type should produce bars with valid schema."""
        generators = [
            TimeBars("BTCUSDT", interval="1min"),
            TickBars("BTCUSDT", tick_count=50),
            VolumeBars("BTCUSDT", volume_threshold=5.0),
            DollarBars("BTCUSDT", dollar_threshold=50_000.0),
        ]
        prices, qtys, times, makers = _generate_trades(200, interval_ms=500)

        for gen in generators:
            bars = _feed_trades(gen, prices, qtys, times, makers)
            # For time bars, flush the last bar
            if isinstance(gen, TimeBars):
                bars.extend(gen.process_tick(50_000.0, 0.1, int(times[-1]) + 120_000, False))

            assert len(bars) >= 1, f"{gen.bar_type} produced no bars"
            for bar in bars:
                d = bar.to_dict()
                missing = REQUIRED_BAR_FIELDS - set(d.keys())
                assert not missing, f"{gen.bar_type}: missing fields {missing}"


# ===================================================================
# T-B13: OHLCV invariant: low <= open,close <= high
# ===================================================================

class TestOHLCVInvariant:
    def test_b13_ohlcv_invariant(self):
        """For every bar: low <= open, low <= close, high >= open, high >= close."""
        generators = [
            TimeBars("BTCUSDT", interval="1min"),
            TickBars("BTCUSDT", tick_count=100),
            VolumeBars("BTCUSDT", volume_threshold=5.0),
            DollarBars("BTCUSDT", dollar_threshold=50_000.0),
            TickImbalanceBars("BTCUSDT", expected_num_ticks_init=50, num_prev_bars=10),
            VolumeImbalanceBars("BTCUSDT", expected_num_ticks_init=50, num_prev_bars=10),
            DollarImbalanceBars("BTCUSDT", expected_num_ticks_init=50, num_prev_bars=10),
            TickRunBars("BTCUSDT", expected_num_ticks_init=50, num_prev_bars=10),
            VolumeRunBars("BTCUSDT", expected_num_ticks_init=50, num_prev_bars=10),
            DollarRunBars("BTCUSDT", expected_num_ticks_init=50, num_prev_bars=10),
        ]

        # Generate trades with significant price movement and biased buy/sell
        # to ensure imbalance/run bars also fire
        n = 5000
        rng = np.random.RandomState(99)
        prices = (50_000.0 + rng.uniform(-500, 500, n)).astype(np.float64)
        qtys = (0.1 + rng.uniform(0, 0.05, n)).astype(np.float64)
        times = np.arange(n, dtype=np.int64) * 500 + 1_700_000_000_000
        # 75% buy-initiated to create strong imbalance/runs
        makers = rng.choice([True, False], size=n, p=[0.25, 0.75])

        for gen in generators:
            bars = _feed_trades(gen, prices, qtys, times, makers)
            # For time bars, flush the last bar
            if isinstance(gen, TimeBars):
                bars.extend(gen.process_tick(50_000.0, 0.1, int(times[-1]) + 120_000, False))

            assert len(bars) >= 1, f"{gen.bar_type} produced no bars for invariant test"
            for bar in bars:
                assert bar.low <= bar.open, (
                    f"{gen.bar_type}: low={bar.low} > open={bar.open}"
                )
                assert bar.low <= bar.close, (
                    f"{gen.bar_type}: low={bar.low} > close={bar.close}"
                )
                assert bar.high >= bar.open, (
                    f"{gen.bar_type}: high={bar.high} < open={bar.open}"
                )
                assert bar.high >= bar.close, (
                    f"{gen.bar_type}: high={bar.high} < close={bar.close}"
                )


# ===================================================================
# T-B14: Bar config save/load round-trips EWMA state
# ===================================================================

class TestEWMAStatePersistence:
    def test_b14_imbalance_bar_save_load_roundtrip(self, tmp_path):
        """Imbalance bar EWMA state should round-trip through save/load."""
        gen = TickImbalanceBars(
            "BTCUSDT", expected_num_ticks_init=200, num_prev_bars=50,
        )
        # Feed enough trades to update EWMA state
        prices, qtys, times, makers = _generate_trades(5000, seed=77)
        _feed_trades(gen, prices, qtys, times, makers)

        # Save state
        state_path = tmp_path / "tick_imbalance_state.json"
        gen.save_state(state_path)

        # Capture state before loading
        expected_ticks = gen._expected_ticks
        expected_imbalance = gen._expected_imbalance
        bar_tick_counts = gen._bar_tick_counts[-gen.num_prev_bars:]

        # Create a fresh generator and load state
        gen2 = TickImbalanceBars(
            "BTCUSDT", expected_num_ticks_init=200, num_prev_bars=50,
        )
        gen2.load_state(state_path)

        assert gen2._expected_ticks == pytest.approx(expected_ticks)
        assert gen2._expected_imbalance == pytest.approx(expected_imbalance)
        assert gen2._bar_tick_counts == bar_tick_counts

    def test_b14_run_bar_save_load_roundtrip(self, tmp_path):
        """Run bar EWMA state should round-trip through save/load."""
        gen = TickRunBars(
            "BTCUSDT", expected_num_ticks_init=200, num_prev_bars=50,
        )
        prices, qtys, times, makers = _generate_trades(5000, seed=88)
        _feed_trades(gen, prices, qtys, times, makers)

        state_path = tmp_path / "tick_run_state.json"
        gen.save_state(state_path)

        expected_ticks = gen._expected_ticks
        p_buy = gen._p_buy
        expected_run_up = gen._expected_run_up
        expected_run_down = gen._expected_run_down
        bar_tick_counts = gen._bar_tick_counts[-gen.num_prev_bars:]

        gen2 = TickRunBars(
            "BTCUSDT", expected_num_ticks_init=200, num_prev_bars=50,
        )
        gen2.load_state(state_path)

        assert gen2._expected_ticks == pytest.approx(expected_ticks)
        assert gen2._p_buy == pytest.approx(p_buy)
        assert gen2._expected_run_up == pytest.approx(expected_run_up)
        assert gen2._expected_run_down == pytest.approx(expected_run_down)
        assert gen2._bar_tick_counts == bar_tick_counts

    def test_b14_load_nonexistent_path_is_noop(self, tmp_path):
        """Loading from a nonexistent path should not crash or alter defaults."""
        gen = TickImbalanceBars("BTCUSDT", expected_num_ticks_init=500)
        default_ticks = gen._expected_ticks

        gen.load_state(tmp_path / "does_not_exist.json")

        assert gen._expected_ticks == default_ticks

    def test_b14_saved_file_is_valid_json(self, tmp_path):
        """The saved state file should be valid JSON with expected keys."""
        gen = VolumeRunBars("BTCUSDT", expected_num_ticks_init=100, num_prev_bars=20)
        prices, qtys, times, makers = _generate_trades(2000, seed=55)
        _feed_trades(gen, prices, qtys, times, makers)

        state_path = tmp_path / "state.json"
        gen.save_state(state_path)

        data = json.loads(state_path.read_text())
        assert "expected_ticks" in data
        assert "bar_tick_counts" in data
        assert "expected_num_ticks_init" in data
        assert "num_prev_bars" in data
        # Run-bar specific keys
        assert "p_buy" in data
        assert "expected_run_up" in data
        assert "expected_run_down" in data


# ===================================================================
# BAR_CLASSES registry
# ===================================================================

class TestBarClassesRegistry:
    def test_bar_classes_contains_all_types(self):
        """BAR_CLASSES should map all 10 bar types to their classes."""
        expected_keys = {
            "time", "tick", "volume", "dollar",
            "tick_imbalance", "volume_imbalance", "dollar_imbalance",
            "tick_run", "volume_run", "dollar_run",
        }
        assert set(BAR_CLASSES.keys()) == expected_keys
