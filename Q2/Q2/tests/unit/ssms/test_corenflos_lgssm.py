"""Unit tests for CorenflosLGSSM (Corenflos et al. 2021, Section 5)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import CorenflosLGSSM


class TestMatrixConstruction:
    """Verify A, H, Q, R are built correctly."""

    def test_A_formula(self):
        """A_{ij} = base^{|i-j|+1}."""
        base, d_x = 0.42, 5
        ssm = CorenflosLGSSM(d_x=d_x, base=base)
        A_np = ssm.A.numpy()
        for i in range(d_x):
            for j in range(d_x):
                expected = base ** (abs(i - j) + 1)
                np.testing.assert_allclose(A_np[i, j], expected, rtol=1e-6)

    def test_A_symmetric(self):
        ssm = CorenflosLGSSM(d_x=10)
        np.testing.assert_allclose(
            ssm.A.numpy(), ssm.A.numpy().T, atol=1e-12)

    def test_H_is_first_rows_of_identity(self):
        d_x, d_y = 10, 3
        ssm = CorenflosLGSSM(d_x=d_x, d_y=d_y)
        expected = np.eye(d_x)[:d_y, :]
        np.testing.assert_allclose(ssm.H.numpy(), expected, atol=1e-12)

    def test_Q_is_identity(self):
        ssm = CorenflosLGSSM(d_x=5)
        np.testing.assert_allclose(ssm.Q.numpy(), np.eye(5), atol=1e-12)

    def test_R_is_sigma_sq_identity(self):
        sigma = 0.316
        ssm = CorenflosLGSSM(d_y=2, sigma_obs=sigma)
        expected = sigma ** 2 * np.eye(2)
        np.testing.assert_allclose(ssm.R.numpy(), expected, atol=1e-10)

    def test_default_dimensions(self):
        ssm = CorenflosLGSSM()
        assert ssm.state_dim == 25
        assert ssm.obs_dim == 1
        assert ssm.A.shape == (25, 25)
        assert ssm.H.shape == (1, 25)


class TestForwardFunctions:
    """Tests for f, h, f_batch, h_batch, Jacobians."""

    @pytest.fixture
    def ssm(self):
        return CorenflosLGSSM(d_x=4, d_y=2, dtype=tf.float64)

    def test_f_shape(self, ssm):
        x = tf.ones(4, dtype=tf.float64)
        assert ssm.f(x).shape == (4,)

    def test_h_shape(self, ssm):
        x = tf.ones(4, dtype=tf.float64)
        assert ssm.h(x).shape == (2,)

    def test_f_batch_shape(self, ssm):
        particles = tf.ones((10, 4), dtype=tf.float64)
        assert ssm.f_batch(particles).shape == (10, 4)

    def test_h_batch_shape(self, ssm):
        particles = tf.ones((10, 4), dtype=tf.float64)
        assert ssm.h_batch(particles).shape == (10, 2)

    def test_f_batch_matches_f(self, ssm):
        """f_batch on single particle should match f."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float64)
        f_single = ssm.f(x).numpy()
        f_batch = ssm.f_batch(x[tf.newaxis, :])[0].numpy()
        np.testing.assert_allclose(f_batch, f_single, atol=1e-12)

    def test_F_jac_is_A(self, ssm):
        """For linear SSM, F_jac(x) = A regardless of x."""
        x = tf.ones(4, dtype=tf.float64)
        np.testing.assert_allclose(
            ssm.F_jac(x).numpy(), ssm.A.numpy(), atol=1e-12)

    def test_H_jac_is_H(self, ssm):
        x = tf.ones(4, dtype=tf.float64)
        np.testing.assert_allclose(
            ssm.H_jac(x).numpy(), ssm.H.numpy(), atol=1e-12)


class TestSimulate:
    """Tests for trajectory simulation."""

    def test_output_shapes(self):
        ssm = CorenflosLGSSM(d_x=5, d_y=2)
        rng = np.random.default_rng(42)
        xs, ys = ssm.simulate(20, rng)
        assert xs.shape == (20, 5)
        assert ys.shape == (20, 2)

    def test_no_nan(self):
        ssm = CorenflosLGSSM(d_x=10, d_y=3)
        rng = np.random.default_rng(42)
        xs, ys = ssm.simulate(50, rng)
        assert not np.any(np.isnan(xs))
        assert not np.any(np.isnan(ys))


class TestLogLikelihood:
    """Tests for log_likelihood computation."""

    def test_output_shape(self):
        ssm = CorenflosLGSSM(d_x=5, d_y=2, dtype=tf.float64)
        N = 30
        y = tf.zeros(2, dtype=tf.float64)
        particles = tf.zeros((N, 5), dtype=tf.float64)
        ll = ssm.log_likelihood(y, particles)
        assert ll.shape == (N,)

    def test_finite_values(self):
        ssm = CorenflosLGSSM(d_x=5, d_y=2, dtype=tf.float64)
        rng = np.random.default_rng(0)
        particles = tf.constant(rng.standard_normal((20, 5)), dtype=tf.float64)
        y = tf.constant(rng.standard_normal(2), dtype=tf.float64)
        ll = ssm.log_likelihood(y, particles)
        assert np.all(np.isfinite(ll.numpy()))

    def test_closer_prediction_higher_likelihood(self):
        """Particle whose h(x) is closer to y should have higher log-likelihood."""
        ssm = CorenflosLGSSM(d_x=3, d_y=1, dtype=tf.float64)
        y = tf.constant([1.0], dtype=tf.float64)
        # x_close: h(x) = x[0] = 1.0 (matches y exactly)
        # x_far:   h(x) = x[0] = 10.0 (far from y)
        x_close = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float64)
        x_far = tf.constant([[10.0, 0.0, 0.0]], dtype=tf.float64)
        particles = tf.concat([x_close, x_far], axis=0)
        ll = ssm.log_likelihood(y, particles).numpy()
        assert ll[0] > ll[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
