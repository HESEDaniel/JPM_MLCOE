"""Unit tests for Lorenz 96 SSM (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm.lorenz96 import (
    lorenz96_rhs, lorenz96_step, lorenz96_ssm,
    lorenz96_rhs_tf, lorenz96_step_tf,
)

DTYPE = tf.float64


class TestLorenz96RHS:
    """Tests for Lorenz 96 ODE right-hand side (numpy)."""

    def test_output_shape(self):
        """RHS should return same shape as input."""
        K = 40
        x = np.ones(K)
        F = 8.0

        dx = lorenz96_rhs(x, F)

        assert dx.shape == (K,)

    def test_cyclic_boundary(self):
        """RHS should handle cyclic boundary correctly."""
        K = 5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        F = 8.0

        dx = lorenz96_rhs(x, F)

        # dx[0] = (x[1] - x[3]) * x[4] - x[0] + F
        expected_dx0 = (x[1] - x[3]) * x[4] - x[0] + F
        np.testing.assert_allclose(dx[0], expected_dx0)


class TestLorenz96RHSTf:
    """Tests for Lorenz 96 ODE right-hand side (TF)."""

    def test_output_shape(self):
        """TF RHS should return same shape as input."""
        K = 40
        x = tf.constant(np.ones(K), dtype=DTYPE)
        F = 8.0

        dx = lorenz96_rhs_tf(x, F)

        assert dx.shape == (K,)

    def test_matches_numpy(self):
        """TF RHS should match numpy RHS exactly."""
        K = 40
        x_np = np.random.randn(K)
        F = 8.0

        dx_np = lorenz96_rhs(x_np, F)
        dx_tf = lorenz96_rhs_tf(tf.constant(x_np, dtype=DTYPE), F).numpy()

        np.testing.assert_allclose(dx_np, dx_tf, rtol=1e-12)

    def test_batch_shape(self):
        """TF RHS should handle batched input."""
        N, K = 5, 40
        x = tf.constant(np.random.randn(N, K), dtype=DTYPE)
        F = 8.0

        dx = lorenz96_rhs_tf(x, F)

        assert dx.shape == (N, K)


class TestLorenz96Step:
    """Tests for Lorenz 96 RK4 integration step (numpy)."""

    def test_output_shape(self):
        """Step should return same shape as input."""
        K = 40
        x = np.random.randn(K)
        F = 8.0
        dt = 0.05

        x_next = lorenz96_step(x, F, dt)

        assert x_next.shape == (K,)


class TestLorenz96StepTf:
    """Tests for Lorenz 96 RK4 integration step (TF)."""

    def test_output_shape(self):
        """TF step should return same shape as input."""
        K = 40
        x = tf.constant(np.random.randn(K), dtype=DTYPE)

        x_next = lorenz96_step_tf(x, 8.0, 0.05)

        assert x_next.shape == (K,)

    def test_matches_numpy(self):
        """TF step should match numpy step to high precision."""
        x_np = np.random.randn(40)

        x_np_out = lorenz96_step(x_np, 8.0, 0.05)
        x_tf_out = lorenz96_step_tf(tf.constant(x_np, dtype=DTYPE), 8.0, 0.05).numpy()

        np.testing.assert_allclose(x_np_out, x_tf_out, rtol=1e-12)

    def test_batch_shape(self):
        """TF step should handle batched input."""
        N, K = 5, 40
        x = tf.constant(np.random.randn(N, K), dtype=DTYPE)

        x_next = lorenz96_step_tf(x, 8.0, 0.05)

        assert x_next.shape == (N, K)

    def test_batch_matches_single(self):
        """Batch TF step should match individual TF steps."""
        N, K = 5, 40
        x_np = np.random.randn(N, K)
        x_batch = tf.constant(x_np, dtype=DTYPE)

        x_next_batch = lorenz96_step_tf(x_batch, 8.0, 0.05)

        for i in range(N):
            x_single = lorenz96_step_tf(tf.constant(x_np[i], dtype=DTYPE), 8.0, 0.05)
            np.testing.assert_allclose(
                x_next_batch[i].numpy(), x_single.numpy(), rtol=1e-12)


class TestLorenz96SSM:
    """Tests for Lorenz 96 state space model (data generation)."""

    def test_output_shapes(self, rng):
        """Generated data should have correct shapes."""
        T = 50
        K = 40

        xs, ys, H, Q, R = lorenz96_ssm(T, rng, K=K)

        assert xs.shape == (T, K)
        assert ys.shape == (T, K)  # Default obs_every=1
        assert H.shape == (K, K)
        assert Q.shape == (K, K)
        assert R.shape == (K, K)

    def test_H_matrix_structure(self, rng):
        """H should select observed variables."""
        T = 20
        K = 20
        obs_every = 2

        xs, ys, H, Q, R = lorenz96_ssm(T, rng, K=K, obs_every=obs_every)

        n_obs = K // obs_every
        assert H.shape == (n_obs, K)
        assert ys.shape == (T, n_obs)
        assert np.sum(H) == n_obs

    def test_no_nan(self, rng):
        """Generated data should not contain NaN."""
        T = 100
        K = 40

        xs, ys, H, Q, R = lorenz96_ssm(T, rng, K=K)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_Q_and_R_structure(self, rng):
        """Q and R should be diagonal with correct scale."""
        T = 20
        K = 20
        Q_std = 0.1
        R_std = 0.5

        _, _, _, Q, R = lorenz96_ssm(T, rng, K=K, Q_std=Q_std, R_std=R_std)

        np.testing.assert_allclose(Q, Q_std**2 * np.eye(K))
        np.testing.assert_allclose(R, R_std**2 * np.eye(K))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
