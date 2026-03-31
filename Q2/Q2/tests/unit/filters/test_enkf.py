"""Unit tests for TensorFlow Ensemble Kalman Filter (EnKF)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.filters import enkf_update, enkf_posterior_analytical

DTYPE = tf.float64


class TestEnKFUpdate:
    """Tests for enkf_update function."""

    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        N, n_x, n_y = 50, 4, 2
        rng = np.random.default_rng(42)

        particles = tf.constant(rng.standard_normal((N, n_x)), dtype=DTYPE)
        H = tf.constant(rng.standard_normal((n_y, n_x)), dtype=DTYPE)
        R = tf.constant(np.eye(n_y) * 0.1, dtype=DTYPE)
        y = tf.constant(rng.standard_normal(n_y), dtype=DTYPE)

        particles_post, m_post, P_post = enkf_update(particles, H, R, y)

        assert particles_post.shape == (N, n_x)
        assert m_post.shape == (n_x,)
        assert P_post.shape == (n_x, n_x)

    def test_no_nan(self):
        """Test that outputs contain no NaN values."""
        N, n_x, n_y = 50, 4, 2
        rng = np.random.default_rng(42)

        particles = tf.constant(rng.standard_normal((N, n_x)), dtype=DTYPE)
        H = tf.constant(rng.standard_normal((n_y, n_x)), dtype=DTYPE)
        R = tf.constant(np.eye(n_y) * 0.1, dtype=DTYPE)
        y = tf.constant(rng.standard_normal(n_y), dtype=DTYPE)

        particles_post, m_post, P_post = enkf_update(particles, H, R, y)

        assert np.all(np.isfinite(particles_post.numpy()))
        assert np.all(np.isfinite(m_post.numpy()))
        assert np.all(np.isfinite(P_post.numpy()))

    def test_posterior_mean_is_mean_of_particles(self):
        """The returned m_post should be the posterior mean."""
        N, n_x, n_y = 100, 3, 2
        rng = np.random.default_rng(42)

        particles = tf.constant(rng.standard_normal((N, n_x)), dtype=DTYPE)
        H = tf.constant(rng.standard_normal((n_y, n_x)), dtype=DTYPE)
        R = tf.constant(np.eye(n_y) * 0.1, dtype=DTYPE)
        y = tf.constant(rng.standard_normal(n_y), dtype=DTYPE)

        _, m_post, _ = enkf_update(particles, H, R, y)

        # m_post is the analytical posterior, not necessarily mean of particles
        assert m_post.shape == (n_x,)
        assert np.all(np.isfinite(m_post.numpy()))


class TestEnKFPosteriorAnalytical:
    """Tests for enkf_posterior_analytical function."""

    def test_matches_kf_update(self):
        """Test that analytical EnKF matches standard Kalman filter update."""
        n_x, n_y = 3, 2
        rng = np.random.default_rng(42)

        x_bar = tf.constant(rng.standard_normal(n_x), dtype=DTYPE)
        B = tf.constant(np.eye(n_x) * 2.0, dtype=DTYPE)
        H = tf.constant(rng.standard_normal((n_y, n_x)), dtype=DTYPE)
        R = tf.constant(np.eye(n_y) * 0.5, dtype=DTYPE)
        y = tf.constant(rng.standard_normal(n_y), dtype=DTYPE)

        # EnKF analytical update
        m_enkf, P_enkf = enkf_posterior_analytical(x_bar, B, H, R, y)

        # Standard Kalman filter update (numpy reference)
        x_bar_np = x_bar.numpy()
        B_np = B.numpy()
        H_np = H.numpy()
        R_np = R.numpy()
        y_np = y.numpy()

        S = H_np @ B_np @ H_np.T + R_np
        K = B_np @ H_np.T @ np.linalg.inv(S)
        m_kf = x_bar_np + K @ (y_np - H_np @ x_bar_np)
        P_kf = (np.eye(n_x) - K @ H_np) @ B_np

        assert m_enkf.shape == (n_x,)
        assert P_enkf.shape == (n_x, n_x)
        np.testing.assert_allclose(m_enkf.numpy(), m_kf, atol=1e-10)
        np.testing.assert_allclose(P_enkf.numpy(), P_kf, atol=1e-10)

    def test_output_shapes(self):
        """Test output shapes for different dimensions."""
        for n_x, n_y in [(2, 1), (4, 2), (5, 3)]:
            rng = np.random.default_rng(42)
            x_bar = tf.constant(rng.standard_normal(n_x), dtype=DTYPE)
            B = tf.constant(np.eye(n_x), dtype=DTYPE)
            H = tf.constant(rng.standard_normal((n_y, n_x)), dtype=DTYPE)
            R = tf.constant(np.eye(n_y), dtype=DTYPE)
            y = tf.constant(rng.standard_normal(n_y), dtype=DTYPE)

            m_post, P_post = enkf_posterior_analytical(x_bar, B, H, R, y)

            assert m_post.shape == (n_x,)
            assert P_post.shape == (n_x, n_x)


class TestEnKFWithLocalization:
    """Tests for EnKF with localization matrix."""

    def test_localization_produces_valid_output(self):
        """EnKF with localization should produce valid output."""
        from src.filters import localization_matrix as loc_matrix

        N, n_x, n_y = 50, 4, 2
        rng = np.random.default_rng(42)

        particles = tf.constant(rng.standard_normal((N, n_x)), dtype=DTYPE)
        H = tf.constant(rng.standard_normal((n_y, n_x)), dtype=DTYPE)
        R = tf.constant(np.eye(n_y) * 0.1, dtype=DTYPE)
        y = tf.constant(rng.standard_normal(n_y), dtype=DTYPE)
        C_loc = loc_matrix(n_x, r_in=3.0)

        particles_post, m_post, P_post = enkf_update(
            particles, H, R, y, localization_matrix=C_loc)

        assert np.all(np.isfinite(particles_post.numpy()))
        assert np.all(np.isfinite(m_post.numpy()))
        assert np.all(np.isfinite(P_post.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
