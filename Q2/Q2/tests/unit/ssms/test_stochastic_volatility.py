"""Unit tests for Stochastic Volatility SSM (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import SVLogTransformed, SVAdditiveNoise
from src.ssm.stochastic_volatility import LOG_CHI2_MEAN

DTYPE = tf.float64


class TestSVLogTransformed:
    """Tests for log-transformed stochastic volatility model (TF)."""

    def test_init(self):
        """Model should initialize with correct dimensions."""
        model = SVLogTransformed()

        assert model.state_dim == 1
        assert model.obs_dim == 1
        assert model.Q.shape == (1, 1)
        assert model.R.shape == (1, 1)
        assert model.P0.shape == (1, 1)

    def test_f_shape(self):
        """f should return (1,) tensor."""
        model = SVLogTransformed()
        x = tf.constant([2.0], dtype=DTYPE)

        result = model.f(x)

        assert result.shape == (1,)

    def test_h_shape(self):
        """h should return (1,) tensor."""
        model = SVLogTransformed()
        x = tf.constant([1.0], dtype=DTYPE)

        result = model.h(x)

        assert result.shape == (1,)

    def test_f_transition(self):
        """f should implement x_k = alpha * x_{k-1}."""
        model = SVLogTransformed(alpha=0.9)
        x = tf.constant([2.0], dtype=DTYPE)

        x_next = model.f(x)

        np.testing.assert_allclose(x_next.numpy(), [0.9 * 2.0])

    def test_h_observation(self):
        """h should implement log-transformed observation function."""
        model = SVLogTransformed(beta=0.5)
        x = tf.constant([1.0], dtype=DTYPE)

        y = model.h(x)

        expected = np.log(0.5**2) + 1.0 + LOG_CHI2_MEAN
        np.testing.assert_allclose(y.numpy()[0], expected)

    def test_F_jac_shape(self):
        """F_jac should return (1, 1) matrix."""
        model = SVLogTransformed(alpha=0.91)
        x = tf.constant([1.0], dtype=DTYPE)

        F = model.F_jac(x)

        assert F.shape == (1, 1)
        np.testing.assert_allclose(F.numpy(), [[0.91]])

    def test_H_jac_shape(self):
        """H_jac should return (1, 1) matrix."""
        model = SVLogTransformed()
        x = tf.constant([1.0], dtype=DTYPE)

        H = model.H_jac(x)

        assert H.shape == (1, 1)
        np.testing.assert_allclose(H.numpy(), [[1.0]])

    def test_f_batch(self):
        """f_batch should apply f to each particle."""
        model = SVLogTransformed(alpha=0.9)
        N = 10
        particles = tf.constant(np.random.randn(N, 1), dtype=DTYPE)

        result = model.f_batch(particles)

        assert result.shape == (N, 1)
        np.testing.assert_allclose(result.numpy(), 0.9 * particles.numpy())

    def test_Q_sampler(self):
        """Q_sampler should return (N, 1) noise samples."""
        model = SVLogTransformed(sigma=1.0)
        rng = tf.random.Generator.from_seed(42)
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, 1)

    def test_log_likelihood_shape(self):
        """Log-likelihood should return N values for N particles."""
        model = SVLogTransformed()
        N = 50
        particles = tf.constant(np.random.randn(N, 1), dtype=DTYPE)
        y = tf.constant([0.5], dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)

    def test_simulate(self, rng):
        """simulate should return correct shapes without NaN."""
        model = SVLogTransformed()
        T = 100

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T,)
        assert ys.shape == (T,)
        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"


class TestSVAdditiveNoise:
    """Tests for additive noise stochastic volatility model (TF)."""

    def test_init(self):
        """Model should initialize with correct dimensions."""
        model = SVAdditiveNoise()

        assert model.state_dim == 1
        assert model.obs_dim == 1
        assert model.Q.shape == (1, 1)
        assert model.R.shape == (1, 1)
        assert model.P0.shape == (1, 1)

    def test_f_shape(self):
        """f should return (1,) tensor."""
        model = SVAdditiveNoise()
        x = tf.constant([2.0], dtype=DTYPE)

        result = model.f(x)

        assert result.shape == (1,)

    def test_h_shape(self):
        """h should return (1,) tensor."""
        model = SVAdditiveNoise()
        x = tf.constant([1.0], dtype=DTYPE)

        result = model.h(x)

        assert result.shape == (1,)

    def test_f_transition(self):
        """f should implement x_k = alpha * x_{k-1}."""
        model = SVAdditiveNoise(alpha=0.9)
        x = tf.constant([2.0], dtype=DTYPE)

        x_next = model.f(x)

        np.testing.assert_allclose(x_next.numpy(), [0.9 * 2.0])

    def test_h_observation(self):
        """h should implement h(x) = beta * exp(exp_scale * x)."""
        model = SVAdditiveNoise(beta=0.5, exp_scale=0.5)
        x = tf.constant([1.0], dtype=DTYPE)

        y = model.h(x)

        expected = 0.5 * np.exp(0.5 * 1.0)
        np.testing.assert_allclose(y.numpy()[0], expected)

    def test_F_jac_shape(self):
        """F_jac should return (1, 1) matrix."""
        model = SVAdditiveNoise(alpha=0.91)
        x = tf.constant([1.0], dtype=DTYPE)

        F = model.F_jac(x)

        assert F.shape == (1, 1)
        np.testing.assert_allclose(F.numpy(), [[0.91]])

    def test_H_jac_numerical(self):
        """H_jac should match numerical differentiation."""
        model = SVAdditiveNoise(beta=0.5, exp_scale=0.5)
        x_np = np.array([1.5])

        H = model.H_jac(tf.constant(x_np, dtype=DTYPE))

        # Numerical differentiation
        eps = 1e-7
        h_plus = model.h(tf.constant(x_np + eps, dtype=DTYPE)).numpy()
        h_minus = model.h(tf.constant(x_np - eps, dtype=DTYPE)).numpy()
        H_num = (h_plus - h_minus) / (2 * eps)

        np.testing.assert_allclose(H.numpy()[0, 0], H_num[0], rtol=1e-5)

    def test_f_batch(self):
        """f_batch should apply f to each particle."""
        model = SVAdditiveNoise(alpha=0.9)
        N = 10
        particles = tf.constant(np.random.randn(N, 1), dtype=DTYPE)

        result = model.f_batch(particles)

        assert result.shape == (N, 1)
        np.testing.assert_allclose(result.numpy(), 0.9 * particles.numpy())

    def test_Q_sampler(self):
        """Q_sampler should return (N, 1) noise samples."""
        model = SVAdditiveNoise(sigma=1.0)
        rng = tf.random.Generator.from_seed(42)
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, 1)

    def test_log_likelihood(self):
        """Log-likelihood should return correct shape with finite values."""
        model = SVAdditiveNoise()
        N = 50
        particles = tf.constant(np.random.randn(N, 1), dtype=DTYPE)
        y = tf.constant([0.5], dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik.numpy()))

    def test_simulate(self, rng):
        """simulate should return correct shapes without NaN."""
        model = SVAdditiveNoise()
        T = 100

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T,)
        assert ys.shape == (T,)
        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
