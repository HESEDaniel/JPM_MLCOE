"""Unit tests for Unscented Kalman Filter implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.filters.ukf import (
    unscented_kalman_filter, ukf_predict, ukf_update,
    _ukf_weights, _sigma_points
)
from src.filters.kf import kalman_filter
from src.ssm import linear_gaussian_ssm


class TestUKFWeights:
    """Tests for UKF weight computation."""

    def test_weights_sum_to_one(self):
        """Mean weights should sum to 1."""
        n_x = 4
        W_m, W_c, gamma = _ukf_weights(n_x, alpha=1e-3, beta=2.0, kappa=0.0)

        assert W_m.shape == (2 * n_x + 1,)
        np.testing.assert_allclose(np.sum(W_m), 1.0, rtol=1e-10)


class TestSigmaPoints:
    """Tests for sigma point generation."""

    def test_fallback_for_non_psd(self):
        """Sigma points should handle near-singular covariance."""
        n_x = 2
        m = np.zeros(n_x)
        # Create nearly singular P
        P = np.array([[1.0, 0.999], [0.999, 1.0]])
        gamma = 1.0

        # Should not raise, uses eigendecomposition fallback
        sigma = _sigma_points(m, P, gamma)

        assert sigma.shape == (2 * n_x + 1, n_x)
        assert not np.any(np.isnan(sigma))


class TestUKFPredict:
    """Tests for UKF prediction step."""

    def test_mean_through_nonlinearity(self, nonlinear_model):
        """Mean should be weighted average through nonlinearity."""
        m = nonlinear_model

        m_pred, P_pred = ukf_predict(m['m0'], m['P0'], m['f'], m['Q'])

        # For linear f, should match simple propagation
        expected = m['A'] @ m['m0']
        np.testing.assert_allclose(m_pred, expected, rtol=1e-5)
        assert m_pred.shape == (2,)
        assert P_pred.shape == (2, 2)


class TestUKFUpdate:
    """Tests for UKF update step."""

    def test_update_reduces_covariance(self, linear_model):
        """Update should reduce covariance and produce valid output."""
        m = linear_model
        m_pred = np.array([1.0, 2.0])
        P_pred = m['P0']
        y = np.array([0.5, 0.3])

        m_upd, P_upd = ukf_update(m_pred, P_pred, y, m['h'], m['R'])

        assert m_upd.shape == (2,)
        assert P_upd.shape == (2, 2)
        assert np.trace(P_upd) <= np.trace(P_pred) + 1e-10


class TestUnscentedKalmanFilter:
    """Tests for full Unscented Kalman Filter."""

    def test_output_shapes(self, rng, linear_model):
        """Verify correct output shapes."""
        m = linear_model
        T = 50
        ys = rng.standard_normal((T, 2))

        m_filt, P_filt, cond_nums = unscented_kalman_filter(
            m['f'], m['h'], m['Q'], m['R'], m['m0'], m['P0'], ys, joseph=True
        )

        assert m_filt.shape == (T, 2)
        assert P_filt.shape == (T, 2, 2)
        assert cond_nums.shape == (T,)

    def test_no_nan(self, rng, linear_model):
        """Output should not contain NaN."""
        m = linear_model
        ys = rng.standard_normal((100, 2))

        m_filt, P_filt, _ = unscented_kalman_filter(
            m['f'], m['h'], m['Q'], m['R'], m['m0'], m['P0'], ys, joseph=True
        )

        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))

    def test_reduces_to_kf_for_linear(self, rng, kf_system):
        """UKF should match KF for linear systems."""
        A, B, C, D, Sigma = kf_system
        T = 50
        Q = B @ B.T
        R = D @ D.T

        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        # KF
        m_kf, P_kf, _ = kalman_filter(A, B, C, D, Sigma, ys, joseph=True)

        # UKF with linear functions
        def f(x):
            return A @ x

        def h(x):
            return C @ x

        m0 = np.zeros(2)
        P0 = Sigma

        m_ukf, P_ukf, _ = unscented_kalman_filter(
            f, h, Q, R, m0, P0, ys, joseph=True
        )

        np.testing.assert_allclose(m_ukf, m_kf, rtol=1e-5)
        np.testing.assert_allclose(P_ukf, P_kf, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
