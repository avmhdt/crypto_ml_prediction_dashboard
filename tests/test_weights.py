"""Unit tests for backend/weights/sample_weights.py.

Test IDs T-W01 through T-W08 covering uniqueness, attribution,
time-decay, combined, and normalization guarantees.
"""
import numpy as np
import pytest

from backend.weights.sample_weights import (
    compute_uniqueness_weights,
    compute_attribution_weights,
    compute_time_decay_weights,
    compute_sample_weights,
)


# ── T-W01: Non-overlapping labels → uniqueness weight = 1.0 ──────────

class TestUniquenessWeights:
    def test_tw01_non_overlapping_uniqueness_is_one(self):
        """Labels that don't share any bars should all have uniqueness = 1.0."""
        label_spans = [(0, 5), (5, 10), (10, 15)]
        num_bars = 15
        weights = compute_uniqueness_weights(label_spans, num_bars)
        np.testing.assert_allclose(weights, 1.0)

    # ── T-W02: Fully overlapping labels → uniqueness weight < 1.0 ────

    def test_tw02_fully_overlapping_uniqueness_less_than_one(self):
        """Labels spanning identical bars must have uniqueness < 1.0."""
        label_spans = [(0, 10), (0, 10), (0, 10)]
        num_bars = 10
        weights = compute_uniqueness_weights(label_spans, num_bars)
        assert all(w < 1.0 for w in weights)
        # Three identical spans → each bar has concurrency 3, so avg uniqueness = 1/3
        np.testing.assert_allclose(weights, 1.0 / 3.0)


# ── T-W03: Return attribution weights sum to total return ────────────

class TestAttributionWeights:
    def test_tw03_attribution_weights_sum_to_one(self):
        """Return attribution weights must sum to 1.0."""
        np.random.seed(42)
        returns = np.random.randn(20) * 0.01
        label_spans = [(0, 5), (5, 10), (10, 15), (15, 20)]
        weights = compute_attribution_weights(returns, label_spans)
        assert weights.sum() == pytest.approx(1.0)

    def test_tw03_attribution_proportional_to_abs_return(self):
        """Label spanning higher absolute returns should get higher weight."""
        returns = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                            0.1, 0.2, 0.3, 0.4, 0.5])
        label_spans = [(0, 5), (5, 10)]
        weights = compute_attribution_weights(returns, label_spans)
        # Second span has all the return, first span has zero
        assert weights[1] > weights[0]
        # First span has zero abs return, so weight = 0
        assert weights[0] == pytest.approx(0.0)
        assert weights[1] == pytest.approx(1.0)


# ── T-W04 & T-W05: Time decay ───────────────────────────────────────

class TestTimeDecayWeights:
    def test_tw04_positive_half_life_decays_older(self):
        """With positive half_life, older timestamps get lower weights."""
        timestamps = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        half_life = 100.0
        weights = compute_time_decay_weights(timestamps, half_life)
        # Most recent (500) should have weight 1.0
        assert weights[-1] == pytest.approx(1.0)
        # Older timestamps should be strictly decreasing
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]
        # Timestamp 400 (age=100=half_life) should have weight 0.5
        assert weights[3] == pytest.approx(0.5)

    def test_tw05_negative_half_life_no_decay(self):
        """half_life = -1 means no decay, all weights should be 1.0."""
        timestamps = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        weights = compute_time_decay_weights(timestamps, half_life=-1)
        np.testing.assert_allclose(weights, 1.0)

    def test_tw05_zero_half_life_no_decay(self):
        """half_life = 0 should also disable decay (all weights 1.0)."""
        timestamps = np.array([10.0, 20.0, 30.0])
        weights = compute_time_decay_weights(timestamps, half_life=0)
        np.testing.assert_allclose(weights, 1.0)


# ── T-W06 through T-W08: Combined sample weights ────────────────────

class TestCombinedWeights:
    @pytest.fixture
    def sample_data(self):
        """Reusable synthetic data for combined weight tests."""
        np.random.seed(123)
        num_bars = 50
        label_spans = [(i, i + 5) for i in range(0, 40, 5)]  # 8 non-overlapping
        returns = np.random.randn(num_bars) * 0.01
        timestamps = np.arange(len(label_spans), dtype=np.float64) * 100.0
        return label_spans, returns, timestamps, num_bars

    def test_tw06_combined_is_product_of_components(self, sample_data):
        """Final weight is proportional to uniqueness * attribution * decay."""
        label_spans, returns, timestamps, num_bars = sample_data
        half_life = 300.0

        combined = compute_sample_weights(
            label_spans, returns, timestamps, num_bars, half_life,
        )

        uniqueness = compute_uniqueness_weights(label_spans, num_bars)
        attribution = compute_attribution_weights(returns, label_spans)
        decay = compute_time_decay_weights(timestamps, half_life)
        raw_product = uniqueness * attribution * decay
        raw_product = np.clip(raw_product, 1e-10, None)
        expected = raw_product / raw_product.sum()

        np.testing.assert_allclose(combined, expected, atol=1e-12)

    def test_tw07_final_weights_sum_to_one(self, sample_data):
        """Normalized combined weights must sum to 1.0."""
        label_spans, returns, timestamps, num_bars = sample_data
        weights = compute_sample_weights(
            label_spans, returns, timestamps, num_bars, half_life=200.0,
        )
        assert weights.sum() == pytest.approx(1.0)

    def test_tw08_no_weight_zero_or_negative(self, sample_data):
        """Every weight must be strictly positive (>= 1e-10 after clip)."""
        label_spans, returns, timestamps, num_bars = sample_data
        weights = compute_sample_weights(
            label_spans, returns, timestamps, num_bars, half_life=200.0,
        )
        assert np.all(weights > 0), "Found zero or negative weight"
        assert np.all(weights >= 1e-10 / weights.sum())
