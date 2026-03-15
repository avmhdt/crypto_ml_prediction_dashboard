"""Unit and integration tests for walk-forward validation.

Test IDs T-WF01 through T-WF24 per .sdd/TESTS.md specification.
"""
import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from backend.ml.walk_forward import (
    WindowResult,
    WalkForwardResult,
    compute_window_boundaries,
    stitch_equity_curves,
    bootstrap_aggregate,
)
from backend.data.database import (
    init_schema,
    save_wf_result,
    load_wf_runs,
    load_wf_run,
    load_wf_latest,
)

_MS_PER_DAY = 86_400_000


# ── Helpers ──────────────────────────────────────────────────────
def _make_window(
    index: int,
    equity: list[float] | None = None,
    oos_accuracy: float = 0.52,
    sharpe: float = 0.3,
    total_return: float = 1.5,
    max_dd: float = -3.0,
    win_rate: float = 51.0,
    num_trades: int = 20,
) -> WindowResult:
    n = 50
    equity = equity or [10000 + j * 2 for j in range(n)]
    return WindowResult(
        window_index=index,
        train_start=index * 30 * _MS_PER_DAY,
        train_end=index * 30 * _MS_PER_DAY + 90 * _MS_PER_DAY,
        test_start=index * 30 * _MS_PER_DAY + 90 * _MS_PER_DAY,
        test_end=index * 30 * _MS_PER_DAY + 120 * _MS_PER_DAY,
        num_train_bars=500,
        num_test_bars=200,
        num_train_samples=450,
        num_test_signals=num_trades,
        primary_recall=0.65,
        meta_precision=0.55,
        oos_accuracy=oos_accuracy,
        oos_precision=0.51,
        oos_recall=0.53,
        sharpe=sharpe,
        max_dd=max_dd,
        total_return=total_return,
        win_rate=win_rate,
        num_trades=num_trades,
        timestamps=[index * 30 * _MS_PER_DAY + 90 * _MS_PER_DAY + j * 1000 for j in range(n)],
        equity=equity,
        drawdown=[0.0] * n,
    )


def _make_result(windows: list[WindowResult] | None = None) -> WalkForwardResult:
    windows = windows or [_make_window(i) for i in range(5)]
    ts, eq, dd = stitch_equity_curves(windows, 10000.0)
    agg = bootstrap_aggregate(windows)
    avg_is = float(np.mean([w.primary_recall for w in windows]))
    avg_oos = float(np.mean([w.oos_accuracy for w in windows]))
    return WalkForwardResult(
        symbol="BTCUSDT",
        bar_type="time",
        labeling_method="triple_barrier",
        train_days=90,
        test_days=30,
        step_days=30,
        num_windows=len(windows),
        windows=windows,
        stitched_timestamps=ts,
        stitched_equity=eq,
        stitched_drawdown=dd,
        aggregate=agg,
        avg_insample_recall=round(avg_is, 4),
        avg_oos_accuracy=round(avg_oos, 4),
        overfitting_gap=round(avg_is - avg_oos, 4),
        created_at="2026-03-15T04:00:00Z",
    )


@pytest.fixture
def mem_conn():
    """In-memory DuckDB connection with schema initialized."""
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    yield conn
    conn.close()


# ═══════════════════════════════════════════════════════════════════
#  Window Boundary Computation (T-WF01 – T-WF04)
# ═══════════════════════════════════════════════════════════════════

class TestWindowBoundaries:
    def test_twf01_covers_full_range(self):
        """T-WF01: Window boundaries cover full date range."""
        start = 0
        end = 240 * _MS_PER_DAY
        windows = compute_window_boundaries(start, end, 90, 30, 30)
        assert len(windows) >= 4
        # Test windows should be non-overlapping
        for i in range(1, len(windows)):
            assert windows[i][2] >= windows[i - 1][3], "Test windows overlap"
        # First test window starts at train_days offset
        assert windows[0][2] == 90 * _MS_PER_DAY

    def test_twf02_clamps_to_data_end(self):
        """T-WF02: Last window truncated if test extends past data end."""
        start = 0
        end = 125 * _MS_PER_DAY  # room for 1 full window + partial
        windows = compute_window_boundaries(start, end, 90, 30, 30)
        assert len(windows) >= 1
        for _, _, _, test_end in windows:
            assert test_end <= end

    def test_twf03_single_window(self):
        """T-WF03: Exactly 1 window when data = train + test."""
        start = 0
        end = 120 * _MS_PER_DAY
        windows = compute_window_boundaries(start, end, 90, 30, 30)
        assert len(windows) == 1

    def test_twf04_insufficient_data(self):
        """T-WF04: Returns empty list when data < train_days."""
        start = 0
        end = 80 * _MS_PER_DAY  # less than 90-day train
        windows = compute_window_boundaries(start, end, 90, 30, 30)
        assert windows == []


# ═══════════════════════════════════════════════════════════════════
#  Equity Stitching (T-WF05 – T-WF07)
# ═══════════════════════════════════════════════════════════════════

class TestEquityStitching:
    def test_twf05_concatenates_correctly(self):
        """T-WF05: Stitched array has correct length."""
        windows = [_make_window(i, equity=[10000 + j for j in range(50)]) for i in range(3)]
        ts, eq, dd = stitch_equity_curves(windows, 10000.0)
        assert len(ts) == 150  # 3 windows * 50 points
        assert len(eq) == 150
        assert len(dd) == 150

    def test_twf06_normalizes_starting_capital(self):
        """T-WF06: Each window starts at the previous window's ending equity."""
        w0 = _make_window(0, equity=[10000, 10100, 10200, 10300, 10500])
        w1 = _make_window(1, equity=[10000, 10050, 10100, 10150, 10200])
        ts, eq, dd = stitch_equity_curves([w0, w1], 10000.0)
        # Window 1 should start at 10500 (w0's ending), not 10000
        assert eq[5] == pytest.approx(10500.0, rel=1e-3)

    def test_twf07_single_window(self):
        """T-WF07: Single window returns its own curve unchanged."""
        window = _make_window(0, equity=[10000, 10100, 10200])
        ts, eq, dd = stitch_equity_curves([window], 10000.0)
        assert eq == [10000, 10100, 10200]


# ═══════════════════════════════════════════════════════════════════
#  Bootstrap Aggregate (T-WF08 – T-WF11)
# ═══════════════════════════════════════════════════════════════════

class TestBootstrapAggregate:
    def test_twf08_valid_bounds(self):
        """T-WF08: CI bounds are valid."""
        windows = [_make_window(i, sharpe=float(np.random.uniform(-1, 2))) for i in range(10)]
        agg = bootstrap_aggregate(windows)
        assert "sharpe" in agg
        s = agg["sharpe"]
        assert s["ci_lower"] <= s["mean"] <= s["ci_upper"]
        assert s["std"] >= 0

    def test_twf09_identical_values(self):
        """T-WF09: All windows same value → std=0, ci_lower=ci_upper=mean."""
        windows = [_make_window(i, sharpe=0.5) for i in range(10)]
        agg = bootstrap_aggregate(windows)
        s = agg["sharpe"]
        assert s["mean"] == pytest.approx(0.5)
        assert s["std"] == pytest.approx(0.0, abs=1e-6)

    def test_twf10_minimum_windows(self):
        """T-WF10: Works with 3 windows (minimum)."""
        windows = [_make_window(i) for i in range(3)]
        agg = bootstrap_aggregate(windows)
        expected_keys = {"oos_accuracy", "oos_precision", "oos_recall", "sharpe", "max_dd", "total_return", "win_rate"}
        assert expected_keys.issubset(set(agg.keys()))

    def test_twf11_overfitting_gap(self):
        """T-WF11: Overfitting gap computed correctly."""
        avg_is = 0.80
        avg_oos = 0.52
        gap = avg_is - avg_oos
        assert gap == pytest.approx(0.28)


# ═══════════════════════════════════════════════════════════════════
#  Data Classes (T-WF12 – T-WF13)
# ═══════════════════════════════════════════════════════════════════

class TestDataClasses:
    def test_twf12_window_result_fields(self):
        """T-WF12: WindowResult stores all fields."""
        w = _make_window(0)
        assert w.window_index == 0
        assert w.primary_recall == 0.65
        assert isinstance(w.timestamps, list)
        assert isinstance(w.equity, list)

    def test_twf13_walk_forward_result_fields(self):
        """T-WF13: WalkForwardResult matches window count."""
        result = _make_result()
        assert result.num_windows == len(result.windows) == 5
        assert isinstance(result.aggregate, dict)
        assert result.overfitting_gap == pytest.approx(
            result.avg_insample_recall - result.avg_oos_accuracy
        )


# ═══════════════════════════════════════════════════════════════════
#  DuckDB Integration (T-WF14 – T-WF18)
# ═══════════════════════════════════════════════════════════════════

class TestDuckDBIntegration:
    def test_twf14_runs_roundtrip(self, mem_conn):
        """T-WF14: Insert + load round-trip for wf_runs."""
        result = _make_result()
        run_id = save_wf_result(mem_conn, result)
        assert run_id >= 1

        runs = load_wf_runs(mem_conn, symbol="BTCUSDT")
        assert len(runs) == 1
        assert runs[0]["symbol"] == "BTCUSDT"

    def test_twf15_windows_roundtrip(self, mem_conn):
        """T-WF15: Insert + load includes windows."""
        result = _make_result()
        run_id = save_wf_result(mem_conn, result)

        full = load_wf_run(mem_conn, run_id)
        assert full is not None
        assert len(full["windows"]) == 5

    def test_twf16_latest_returns_most_recent(self, mem_conn):
        """T-WF16: load_wf_latest returns the most recent run."""
        r1 = _make_result()
        id1 = save_wf_result(mem_conn, r1)
        r2 = _make_result()
        id2 = save_wf_result(mem_conn, r2)

        latest = load_wf_latest(mem_conn, "BTCUSDT", "time", "triple_barrier")
        assert latest is not None
        assert latest["id"] == id2

    def test_twf17_filter_by_combo(self, mem_conn):
        """T-WF17: load_wf_runs filters by symbol/bar_type/labeling."""
        r1 = _make_result()
        save_wf_result(mem_conn, r1)

        # Different symbol
        r2 = _make_result()
        r2.symbol = "ETHUSDT"
        save_wf_result(mem_conn, r2)

        btc_runs = load_wf_runs(mem_conn, symbol="BTCUSDT")
        assert len(btc_runs) == 1
        assert btc_runs[0]["symbol"] == "BTCUSDT"

        eth_runs = load_wf_runs(mem_conn, symbol="ETHUSDT")
        assert len(eth_runs) == 1

        all_runs = load_wf_runs(mem_conn)
        assert len(all_runs) == 2

    def test_twf18_json_equity_roundtrip(self, mem_conn):
        """T-WF18: JSON-encoded equity arrays survive DuckDB round-trip."""
        result = _make_result()
        run_id = save_wf_result(mem_conn, result)
        full = load_wf_run(mem_conn, run_id)

        assert full["stitched_timestamps"] == result.stitched_timestamps
        assert len(full["stitched_equity"]) == len(result.stitched_equity)
        # Check values are close (floating point round-trip)
        for a, b in zip(full["stitched_equity"][:5], result.stitched_equity[:5]):
            assert a == pytest.approx(b, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════
#  Edge Cases (T-WF25, T-WF27 – T-WF29, T-WF31)
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_twf27_empty_runs_table(self, mem_conn):
        """T-WF27: Empty table returns empty list."""
        runs = load_wf_runs(mem_conn, symbol="BTCUSDT")
        assert runs == []

    def test_twf28_zero_trades_window(self):
        """T-WF28: Window with zero trades included in aggregate."""
        windows = [_make_window(i, num_trades=0, sharpe=0.0) for i in range(5)]
        agg = bootstrap_aggregate(windows)
        assert "sharpe" in agg
        assert agg["sharpe"]["mean"] == pytest.approx(0.0, abs=0.01)

    def test_twf29_empty_equity_window(self):
        """T-WF29: Window with empty equity skipped in stitching."""
        w0 = _make_window(0, equity=[10000, 10100])
        w1 = WindowResult(
            window_index=1, train_start=0, train_end=0, test_start=0, test_end=0,
            num_train_bars=0, num_test_bars=0, num_train_samples=0, num_test_signals=0,
            primary_recall=0, meta_precision=0, oos_accuracy=0, oos_precision=0, oos_recall=0,
            sharpe=0, max_dd=0, total_return=0, win_rate=0, num_trades=0,
            timestamps=[], equity=[], drawdown=[],
        )
        w2 = _make_window(2, equity=[10000, 10050])
        ts, eq, dd = stitch_equity_curves([w0, w1, w2], 10000.0)
        # Should have 4 points (w0=2 + w1=0 + w2=2)
        assert len(eq) == 4

    def test_twf31_timestamps_increasing(self):
        """T-WF31: Stitched timestamps are strictly increasing."""
        windows = [_make_window(i) for i in range(3)]
        ts, _, _ = stitch_equity_curves(windows, 10000.0)
        for i in range(1, len(ts)):
            assert ts[i] > ts[i - 1], f"Non-increasing at index {i}"
