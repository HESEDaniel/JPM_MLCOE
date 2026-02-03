"""Unit tests for metrics utility functions."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.metrics import (
    compute_mse, compute_rmse, compute_nees, stability_summary,
    compute_symmetry_error, compute_min_eigenvalues, compute_nis
)


class TestComputeMSE:
    """Tests for MSE computation."""

    def test_known_value(self):
        """MSE should match hand-computed value."""
        estimated = np.array([1.0, 2.0, 3.0])
        true = np.array([0.0, 0.0, 0.0])

        mse = compute_mse(estimated, true)

        # MSE = (1 + 4 + 9) / 3 = 14/3
        np.testing.assert_allclose(mse, 14.0 / 3.0)


class TestComputeRMSE:
    """Tests for RMSE computation."""

    def test_sqrt_of_mse(self):
        """RMSE should be square root of MSE."""
        estimated = np.array([1.0, 2.0, 3.0])
        true = np.array([0.0, 0.0, 0.0])

        rmse = compute_rmse(estimated, true)
        mse = compute_mse(estimated, true)

        np.testing.assert_allclose(rmse, np.sqrt(mse))


class TestComputeNEES:
    """Tests for NEES computation."""

    def test_output_shape(self, rng):
        """NEES should return T values."""
        T = 50
        n_x = 3
        m_filt = rng.standard_normal((T, n_x))
        P_filt = np.array([np.eye(n_x) for _ in range(T)])
        xs = rng.standard_normal((T, n_x))

        nees = compute_nees(m_filt, P_filt, xs)

        assert nees.shape == (T,)

    def test_regularization(self, rng):
        """Regularization should prevent singularity issues."""
        T = 10
        n_x = 2
        m_filt = rng.standard_normal((T, n_x))
        # Near-singular covariance
        P_filt = np.array([1e-12 * np.eye(n_x) for _ in range(T)])
        xs = rng.standard_normal((T, n_x))

        # Should not raise, uses regularization
        nees = compute_nees(m_filt, P_filt, xs, regularize=1e-8)

        assert np.all(np.isfinite(nees))

    def test_fallback_for_singular(self, rng):
        """Should use fallback for truly singular matrices."""
        T = 5
        n_x = 2
        m_filt = rng.standard_normal((T, n_x))
        # Singular covariance
        P_filt = np.zeros((T, n_x, n_x))
        xs = rng.standard_normal((T, n_x))

        nees = compute_nees(m_filt, P_filt, xs)

        assert np.all(np.isfinite(nees))


class TestStabilitySummary:
    """Tests for stability summary."""

    def test_returns_dict(self):
        """stability_summary should return a dictionary."""
        cond_nums = np.array([10.0, 20.0, 30.0])

        summary = stability_summary(cond_nums)

        assert isinstance(summary, dict)
        assert 'mean_cond' in summary
        assert 'max_cond' in summary


class TestComputeSymmetryError:
    """Tests for symmetry error computation."""

    def test_relative_error(self):
        """Error should be relative to matrix norm."""
        T = 5
        P = np.array([[10.0, 0.1], [0.2, 10.0]])  # Slightly asymmetric
        P_filt = np.array([P for _ in range(T)])

        err = compute_symmetry_error(P_filt)

        # Error should be small relative to matrix norm
        assert np.all(err < 0.1)


class TestComputeMinEigenvalues:
    """Tests for minimum eigenvalue computation."""

    def test_psd_detection(self):
        """Non-PSD matrices should have negative minimum eigenvalue."""
        T = 5
        # Non-PSD matrix
        P_bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        P_filt = np.array([P_bad for _ in range(T)])

        min_eig = compute_min_eigenvalues(P_filt)

        assert np.all(min_eig < 0)


class TestComputeNIS:
    """Tests for NIS computation."""

    def test_output_shape(self):
        """Should return T values."""
        T = 20
        n_y = 2
        innovations = np.random.randn(T, n_y)
        S_innov = np.array([np.eye(n_y) for _ in range(T)])

        nis = compute_nis(innovations, S_innov)

        assert nis.shape == (T,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
