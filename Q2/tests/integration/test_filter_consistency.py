"""Integration tests for filter consistency and edge cases."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import linear_gaussian_ssm
from src.filters import kalman_filter, extended_kalman_filter, unscented_kalman_filter, particle_filter


class TestAllFiltersDeterministic:
    """Test that filters are deterministic with same seed."""

    def test_all_filters_deterministic(self):
        """Same seed should produce same results."""
        # Setup
        A = np.array([[0.9, 0.1], [0.0, 0.95]])
        B = np.array([[0.1, 0], [0, 0.1]])
        C = np.eye(2)
        D = np.array([[0.1, 0], [0, 0.1]])
        Sigma = np.eye(2)
        T = 30

        # Generate data with same seed
        rng1 = np.random.default_rng(42)
        xs1, ys1 = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng1)

        rng2 = np.random.default_rng(42)
        xs2, ys2 = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng2)

        np.testing.assert_array_equal(xs1, xs2)
        np.testing.assert_array_equal(ys1, ys2)

        # KF should be deterministic
        m_kf1, P_kf1, _ = kalman_filter(A, B, C, D, Sigma, ys1)
        m_kf2, P_kf2, _ = kalman_filter(A, B, C, D, Sigma, ys2)

        np.testing.assert_array_equal(m_kf1, m_kf2)


class TestShortTrajectory:
    """Test that filters work for very short trajectories."""

    def test_short_trajectory(self, rng):
        """Filters should work for T=1."""
        A = np.array([[0.9, 0.1], [0.0, 0.95]])
        B = np.array([[0.1, 0], [0, 0.1]])
        C = np.eye(2)
        D = np.array([[0.1, 0], [0, 0.1]])
        Sigma = np.eye(2)
        T = 1

        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        # KF
        m_kf, P_kf, cond = kalman_filter(A, B, C, D, Sigma, ys)
        assert m_kf.shape == (1, 2)
        assert not np.any(np.isnan(m_kf))

        # EKF
        Q, R = B @ B.T, D @ D.T
        m_ekf, P_ekf, _ = extended_kalman_filter(
            lambda x: A @ x, lambda x: C @ x,
            lambda x: A, lambda x: C,
            Q, R, np.zeros(2), Sigma, ys
        )
        assert m_ekf.shape == (1, 2)
        assert not np.any(np.isnan(m_ekf))

        # UKF
        m_ukf, P_ukf, _ = unscented_kalman_filter(
            lambda x: A @ x, lambda x: C @ x,
            Q, R, np.zeros(2), Sigma, ys
        )
        assert m_ukf.shape == (1, 2)
        assert not np.any(np.isnan(m_ukf))


class TestHighDimensionStability:
    """Test filter stability in high dimensions."""

    def test_high_dimension_stability(self, rng):
        """Filters should be stable in higher dimensions."""
        n_x = 10
        A = 0.9 * np.eye(n_x)
        B = 0.1 * np.eye(n_x)
        C = np.eye(n_x)
        D = 0.1 * np.eye(n_x)
        Sigma = np.eye(n_x)
        T = 50

        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        # KF
        m_kf, P_kf, cond = kalman_filter(A, B, C, D, Sigma, ys)

        assert not np.any(np.isnan(m_kf))
        assert not np.any(np.isnan(P_kf))
        assert np.all(cond < 1e12)

        # Check covariances are PSD
        for t in range(T):
            eigvals = np.linalg.eigvalsh(P_kf[t])
            assert np.all(eigvals >= -1e-10)


class TestIllConditionedSystem:
    """Test Joseph update prevents divergence in ill-conditioned systems."""

    def test_ill_conditioned_system(self, rng):
        """Joseph update should prevent numerical divergence."""
        # Create ill-conditioned system
        A = np.array([[1.0, 0.0], [0.0, 0.99]])
        B = np.array([[1.0, 0], [0, 0.01]])  # Very different noise scales
        C = np.eye(2)
        D = np.array([[0.01, 0], [0, 0.01]])
        Sigma = np.eye(2)
        T = 100

        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        # Joseph update
        m_joseph, P_joseph, cond_joseph = kalman_filter(
            A, B, C, D, Sigma, ys, joseph=True
        )

        # All covariances should be PSD
        for t in range(T):
            eigvals = np.linalg.eigvalsh(P_joseph[t])
            assert np.all(eigvals >= -1e-10), f"Non-PSD at t={t}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
