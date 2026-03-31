"""Unit tests for Ensemble Kalman Filter (EnKF)."""
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.filters.enkf import enkf_update, enkf_posterior_analytical


class TestEnKFUpdate:
    """Tests for enkf_update function."""

    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        N, n_x, n_y = 50, 4, 2
        rng = np.random.default_rng(42)

        particles = rng.standard_normal((N, n_x))
        H = rng.standard_normal((n_y, n_x))
        R = np.eye(n_y) * 0.1
        y = rng.standard_normal(n_y)

        particles_post, m_post, P_post = enkf_update(particles, H, R, y)

        assert particles_post.shape == (N, n_x)
        assert m_post.shape == (n_x,)
        assert P_post.shape == (n_x, n_x)

    def test_no_nan(self):
        """Test that outputs contain no NaN values."""
        N, n_x, n_y = 50, 4, 2
        rng = np.random.default_rng(42)

        particles = rng.standard_normal((N, n_x))
        H = rng.standard_normal((n_y, n_x))
        R = np.eye(n_y) * 0.1
        y = rng.standard_normal(n_y)

        particles_post, m_post, P_post = enkf_update(particles, H, R, y)

        assert np.all(np.isfinite(particles_post))
        assert np.all(np.isfinite(m_post))
        assert np.all(np.isfinite(P_post))


class TestEnKFPosteriorAnalytical:
    """Tests for enkf_posterior_analytical function."""

    def test_matches_kf_update(self):
        """Test that analytical EnKF matches standard Kalman filter update."""
        n_x, n_y = 3, 2
        rng = np.random.default_rng(42)

        x_bar = rng.standard_normal(n_x)
        B = np.eye(n_x) * 2.0
        H = rng.standard_normal((n_y, n_x))
        R = np.eye(n_y) * 0.5
        y = rng.standard_normal(n_y)

        # EnKF analytical update
        m_enkf, P_enkf = enkf_posterior_analytical(x_bar, B, H, R, y)

        # Standard Kalman filter update
        S = H @ B @ H.T + R
        K = B @ H.T @ np.linalg.inv(S)
        m_kf = x_bar + K @ (y - H @ x_bar)
        P_kf = (np.eye(n_x) - K @ H) @ B

        assert m_enkf.shape == (n_x,)
        assert P_enkf.shape == (n_x, n_x)
        np.testing.assert_allclose(m_enkf, m_kf, atol=1e-10)
        np.testing.assert_allclose(P_enkf, P_kf, atol=1e-10)
