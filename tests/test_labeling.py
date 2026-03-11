"""Tests for labeling modules: triple barrier, trend scanning, directional change.

Covers TESTS.md T-L01 through T-L16.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.config import TripleBarrierConfig
from backend.labeling.triple_barrier import triple_barrier_labels, _daily_volatility
from backend.labeling.trend_scanning import trend_scanning_labels
from backend.labeling.directional_change import (
    directional_change_labels,
    dc_labels_from_volatility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(closes: list[float], *, spread: float = 0.0) -> pd.DataFrame:
    """Build a minimal bars DataFrame from a list of close prices.

    Parameters
    ----------
    closes : list[float]
        Close prices in chronological order.
    spread : float
        Amount added/subtracted to derive high/low from close.
        When 0, high == low == close (no intra-bar range).
    """
    n = len(closes)
    closes_arr = np.array(closes, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "open": closes_arr,
            "high": closes_arr + spread,
            "low": closes_arr - spread,
            "close": closes_arr,
            "volume": np.ones(n, dtype=np.float64) * 100.0,
        }
    )


def _make_random_bars(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate *n* bars with a geometric random walk (realistic crypto-like)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 0.02, size=n)
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    highs = closes * (1 + rng.uniform(0.0, 0.01, size=n))
    lows = closes * (1 - rng.uniform(0.0, 0.01, size=n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "open": closes * (1 + rng.uniform(-0.005, 0.005, size=n)),
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(50, 500, size=n),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# TRIPLE BARRIER TESTS (T-L01 .. T-L07)
# ═══════════════════════════════════════════════════════════════════════════


class TestTripleBarrier:
    """T-L01 through T-L07."""

    # T-L01: Price hits PT first -> label = 1
    def test_l01_pt_hit_first(self):
        """Construct bars where the price surges well above the PT barrier
        before any SL breach can happen."""
        # Start at 100, then jump to 200 — guaranteed to exceed PT.
        closes = [100.0] * 5 + [200.0] + [200.0] * 10
        bars = _make_bars(closes, spread=0.0)

        cfg = TripleBarrierConfig(
            sl_multiplier=2.0,
            pt_multiplier=2.0,
            max_holding_period=10,
            volatility_window=5,
        )
        result = triple_barrier_labels(bars, cfg)
        # The first bar should see a PT hit because 200 >> entry + vol*mult.
        assert result.loc[0, "label"] == 1

    # T-L02: Price hits SL first -> label = -1
    def test_l02_sl_hit_first(self):
        """Construct bars where price plunges far below the SL barrier."""
        closes = [100.0] * 5 + [10.0] + [10.0] * 10
        bars = _make_bars(closes, spread=0.0)

        cfg = TripleBarrierConfig(
            sl_multiplier=2.0,
            pt_multiplier=2.0,
            max_holding_period=10,
            volatility_window=5,
        )
        result = triple_barrier_labels(bars, cfg)
        assert result.loc[0, "label"] == -1

    # T-L03: Time exit with positive return -> label = 1
    def test_l03_time_exit_positive(self):
        """Price drifts up gently within barriers, time exit with gain."""
        # Gentle uptrend: 100 -> 100.0001 per bar (tiny moves, won't hit
        # barriers because vol will be orders of magnitude larger than move).
        n = 60
        closes = [100.0 + 0.00001 * i for i in range(n)]
        bars = _make_bars(closes, spread=0.0)

        cfg = TripleBarrierConfig(
            sl_multiplier=100.0,   # very wide barriers
            pt_multiplier=100.0,
            max_holding_period=10,
            volatility_window=5,
        )
        result = triple_barrier_labels(bars, cfg)
        # Bar 0 should reach time barrier at bar 10 with a positive return.
        assert result.loc[0, "label"] == 1

    # T-L04: Time exit with negative return -> label = -1
    def test_l04_time_exit_negative(self):
        """Price drifts down gently within barriers, time exit with loss."""
        n = 60
        closes = [100.0 - 0.00001 * i for i in range(n)]
        bars = _make_bars(closes, spread=0.0)

        cfg = TripleBarrierConfig(
            sl_multiplier=100.0,
            pt_multiplier=100.0,
            max_holding_period=10,
            volatility_window=5,
        )
        result = triple_barrier_labels(bars, cfg)
        assert result.loc[0, "label"] == -1

    # T-L05: Time exit with zero return -> label in {-1, 1}
    def test_l05_time_exit_zero_return(self):
        """Perfectly flat price should still produce a valid binary label."""
        n = 60
        closes = [100.0] * n
        bars = _make_bars(closes, spread=0.0)

        cfg = TripleBarrierConfig(
            sl_multiplier=100.0,
            pt_multiplier=100.0,
            max_holding_period=10,
            volatility_window=5,
        )
        result = triple_barrier_labels(bars, cfg)
        # Zero return fallback: label must still be in {-1, 1}.
        assert result.loc[0, "label"] in {-1, 1}

    # T-L06: SL/PT from volatility-scaled multipliers
    def test_l06_volatility_scaled_barriers(self):
        """Verify that sl_price and pt_price are close +/- multiplier * vol."""
        closes = [100.0 + np.sin(i / 5.0) * 2.0 for i in range(100)]
        bars = _make_bars(closes, spread=0.5)

        cfg = TripleBarrierConfig(
            sl_multiplier=1.5,
            pt_multiplier=2.5,
            max_holding_period=20,
            volatility_window=20,
        )
        result = triple_barrier_labels(bars, cfg)

        # Recompute the volatility the same way the module does.
        vol = _daily_volatility(
            pd.Series(bars["close"].values, dtype=np.float64),
            span=cfg.volatility_window,
        ).values

        for i in range(len(bars)):
            c = bars["close"].values[i]
            expected_sl = c - cfg.sl_multiplier * vol[i]
            expected_pt = c + cfg.pt_multiplier * vol[i]
            assert np.isclose(result.loc[i, "sl_price"], expected_sl, atol=1e-10), (
                f"Bar {i}: sl_price mismatch"
            )
            assert np.isclose(result.loc[i, "pt_price"], expected_pt, atol=1e-10), (
                f"Bar {i}: pt_price mismatch"
            )

    # T-L07: Output always binary {-1, 1} over 1000 random bars
    def test_l07_always_binary_1000_bars(self):
        """Run triple barrier on 1000 random bars; every label must be -1 or 1."""
        bars = _make_random_bars(1000, seed=7)
        cfg = TripleBarrierConfig(
            sl_multiplier=2.0,
            pt_multiplier=2.0,
            max_holding_period=30,
            volatility_window=20,
        )
        result = triple_barrier_labels(bars, cfg)
        assert len(result) == 1000
        assert set(result["label"].unique()).issubset({-1, 1})


# ═══════════════════════════════════════════════════════════════════════════
# TREND SCANNING TESTS (T-L08 .. T-L11)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrendScanning:
    """T-L08 through T-L11."""

    # T-L08: Clear uptrend -> label = 1
    def test_l08_uptrend_label(self):
        """A strong linear uptrend should yield label = 1 for the first bar."""
        n = 100
        closes = [100.0 + 2.0 * i for i in range(n)]
        bars = _make_bars(closes)
        result = trend_scanning_labels(bars, horizons=[5, 10, 20])
        assert result.loc[0, "label"] == 1

    # T-L09: Clear downtrend -> label = -1
    def test_l09_downtrend_label(self):
        """A strong linear downtrend should yield label = -1 for the first bar."""
        n = 100
        closes = [200.0 - 2.0 * i for i in range(n)]
        bars = _make_bars(closes)
        result = trend_scanning_labels(bars, horizons=[5, 10, 20])
        assert result.loc[0, "label"] == -1

    # T-L10: Max t-value selects best horizon
    def test_l10_max_t_value_best_horizon(self):
        """The returned best_horizon should match the horizon with the
        highest absolute t-value among the candidates."""
        # Build a series that trends up for the first 11 bars then reverses.
        # This should make horizon=10 have a stronger t-value than horizon=20
        # for bar 0.
        n = 100
        closes = []
        for i in range(n):
            if i <= 10:
                closes.append(100.0 + 5.0 * i)
            else:
                closes.append(closes[10] - 1.0 * (i - 10))
        bars = _make_bars(closes)
        result = trend_scanning_labels(bars, horizons=[5, 10, 20])

        # Bar 0: the slope over 10 bars (pure uptrend) should have a very
        # high |t-value| versus 20 bars (up then down).  The best_horizon
        # should be one of the shorter horizons.
        best_h = result.loc[0, "best_horizon"]
        assert best_h in [5, 10], (
            f"Expected best horizon to be 5 or 10, got {best_h}"
        )
        assert result.loc[0, "t_value"] > 0, "Expected positive t-value for uptrend"

    # T-L11: Output always binary {-1, 1}
    def test_l11_always_binary(self):
        """Run trend scanning on 500 random bars; every label must be -1 or 1."""
        bars = _make_random_bars(500, seed=11)
        result = trend_scanning_labels(bars, horizons=[5, 10, 20, 40])
        assert len(result) == 500
        assert set(result["label"].unique()).issubset({-1, 1})


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTIONAL CHANGE TESTS (T-L12 .. T-L16)
# ═══════════════════════════════════════════════════════════════════════════


class TestDirectionalChange:
    """T-L12 through T-L16."""

    # T-L12: Upturn event -> label = 1
    def test_l12_upturn_label(self):
        """After a drop, a sharp rise by theta should produce an upturn (1)."""
        # Drop from 100 to 90 (10 % drop), then rise from 90 to 100.
        # With theta=0.05 (5%), the drop triggers a downturn event, then
        # the rise triggers an upturn event.
        closes = (
            [100.0] + [100.0 - 2.0 * i for i in range(1, 7)]  # drop to 88
            + [88.0 + 3.0 * i for i in range(1, 8)]  # rise to 109
        )
        bars = _make_bars(closes)
        result = directional_change_labels(bars, thetas=[0.05])
        upturn_events = result[result["label"] == 1]
        assert len(upturn_events) > 0, "Expected at least one upturn event"

    # T-L13: Downturn event -> label = -1
    def test_l13_downturn_label(self):
        """After a rise, a sharp drop by theta should produce a downturn (-1)."""
        # Rise from 100 to 110, then drop back down.
        closes = (
            [100.0] + [100.0 + 2.0 * i for i in range(1, 7)]  # rise to 112
            + [112.0 - 3.0 * i for i in range(1, 8)]  # drop to 91
        )
        bars = _make_bars(closes)
        result = directional_change_labels(bars, thetas=[0.05])
        downturn_events = result[result["label"] == -1]
        assert len(downturn_events) > 0, "Expected at least one downturn event"

    # T-L14: Multi-scale theta produces events at all scales
    def test_l14_multi_scale_theta(self):
        """Using multiple theta values should yield events at every scale
        (assuming sufficient price movement)."""
        # Create a large zigzag pattern to trigger all scales.
        closes = []
        price = 100.0
        for cycle in range(10):
            for i in range(20):
                price += 3.0  # up
                closes.append(price)
            for i in range(20):
                price -= 3.0  # down
                closes.append(price)
        bars = _make_bars(closes)

        thetas = [0.01, 0.05, 0.10]
        result = directional_change_labels(bars, thetas=thetas)

        observed_thetas = set(result["theta"].unique())
        for th in thetas:
            assert th in observed_thetas, (
                f"Expected events at theta={th}, only got {observed_thetas}"
            )

    # T-L15: Theta derived from Rogers-Satchell volatility
    def test_l15_theta_from_volatility(self):
        """dc_labels_from_volatility should compute theta from rolling
        volatility and produce valid results."""
        bars = _make_random_bars(500, seed=15)
        result = dc_labels_from_volatility(bars, vol_window=20, multipliers=[0.5, 1.0, 2.0])
        # Must produce at least some events on 500 random bars.
        assert len(result) > 0, "Expected DC events from volatility-scaled theta"
        # All labels must be binary.
        assert set(result["label"].unique()).issubset({-1, 1})
        # The theta values should be data-derived (positive floats).
        assert all(result["theta"] > 0)

    # T-L16: Output always binary {-1, 1}
    def test_l16_always_binary(self):
        """Run DC on 1000 random bars with multiple thetas; all labels in {-1,1}."""
        bars = _make_random_bars(1000, seed=16)
        result = directional_change_labels(bars, thetas=[0.01, 0.02, 0.05])
        if len(result) > 0:
            assert set(result["label"].unique()).issubset({-1, 1})
        # Also check the volatility-derived variant.
        result2 = dc_labels_from_volatility(bars, vol_window=20)
        if len(result2) > 0:
            assert set(result2["label"].unique()).issubset({-1, 1})
