"""Unit tests for Range-Bearing SSM (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import RangeBearing

DTYPE = tf.float64


class TestRangeBearingSSM:
    """Unit tests for Range-Bearing SSM (TF)."""

    @pytest.fixture
    def model(self):
        return RangeBearing()

    def test_init(self, model):
        """Model should initialize with correct dimensions."""
        assert model.state_dim == 4
        assert model.obs_dim == 2
        assert model.Q.shape == (4, 4)
        assert model.R.shape == (2, 2)
        assert model.P0.shape == (4, 4)

    def test_f_shape(self, model):
        """f should return (state_dim,) tensor."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.f(x)

        assert result.shape == (model.state_dim,)

    def test_h_shape(self, model):
        """h should return (obs_dim,) tensor."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.h(x)

        assert result.shape == (model.obs_dim,)

    def test_observation_function(self, model):
        """Test h() at known point: (3, -, 4, -) -> range=5, bearing=atan2(4,3)."""
        x = tf.constant([3.0, 0.0, 4.0, 0.0], dtype=DTYPE)

        y = model.h(x)

        np.testing.assert_allclose(y.numpy()[0], 5.0, atol=1e-10)
        np.testing.assert_allclose(y.numpy()[1], np.arctan2(4, 3), atol=1e-10)

    def test_F_jac_shape(self, model):
        """F_jac should return (state_dim, state_dim) matrix."""
        x = tf.constant([2.0, 0.1, 3.0, 0.1], dtype=DTYPE)

        F = model.F_jac(x)

        assert F.shape == (model.state_dim, model.state_dim)

    def test_H_jac_shape(self, model):
        """H_jac should return (obs_dim, state_dim) matrix."""
        x = tf.constant([2.0, 0.1, 3.0, 0.1], dtype=DTYPE)

        H = model.H_jac(x)

        assert H.shape == (model.obs_dim, model.state_dim)

    def test_H_jac_numerical(self, model):
        """H_jac should match numerical differentiation."""
        x_np = np.array([2.0, 0.1, 3.0, 0.1])
        x = tf.constant(x_np, dtype=DTYPE)

        H = model.H_jac(x).numpy()

        eps = 1e-6
        H_num = np.zeros((2, 4))
        for i in range(4):
            x_plus = x_np.copy()
            x_plus[i] += eps
            x_minus = x_np.copy()
            x_minus[i] -= eps
            h_plus = model.h(tf.constant(x_plus, dtype=DTYPE)).numpy()
            h_minus = model.h(tf.constant(x_minus, dtype=DTYPE)).numpy()
            H_num[:, i] = (h_plus - h_minus) / (2 * eps)

        np.testing.assert_allclose(H, H_num, atol=1e-5)

    def test_f_batch(self, model):
        """f_batch should apply f to each particle."""
        N = 10
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)

        result = model.f_batch(particles)

        assert result.shape == (N, model.state_dim)

    def test_h_batch(self, model):
        """h_batch should match single-particle h for each row."""
        N = 10
        particles_np = np.random.randn(N, model.state_dim)
        particles = tf.constant(particles_np, dtype=DTYPE)

        result = model.h_batch(particles)

        assert result.shape == (N, model.obs_dim)
        for i in range(N):
            single = model.h(tf.constant(particles_np[i], dtype=DTYPE)).numpy()
            np.testing.assert_allclose(result[i].numpy(), single, rtol=1e-12)

    def test_Q_sampler(self, model):
        """Q_sampler should return (N, state_dim) noise samples."""
        rng = tf.random.Generator.from_seed(42)
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, model.state_dim)

    def test_log_likelihood_shape(self, model):
        """Log-likelihood should return N values."""
        N = 50
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)
        y = tf.constant([5.0, 0.5], dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)

    def test_log_likelihood_bearing_wrap(self, model):
        """Log-likelihood should handle bearing angle wrapping."""
        N = 50
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)
        y = tf.constant([5.0, -np.pi + 0.1], dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik.numpy()))

    def test_simulate(self, rng, model):
        """simulate should return correct shapes without NaN."""
        T = 100

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 4)
        assert ys.shape == (T, 2)
        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_range_positive(self, rng, model):
        """Range observations should be positive."""
        T = 100
        xs, ys = model.simulate(T, rng)

        assert np.all(ys[:, 0] > 0), "Range should be positive"

    def test_bearing_range(self, rng, model):
        """Bearing should be in [-pi, pi]."""
        T = 100
        xs, ys = model.simulate(T, rng)

        assert np.all(ys[:, 1] >= -np.pi), "Bearing below -pi"
        assert np.all(ys[:, 1] <= np.pi), "Bearing above pi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
