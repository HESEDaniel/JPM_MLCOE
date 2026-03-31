"""Unit tests for Skewed-t Poisson SSM (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import SkewedTPoissonSSM

DTYPE = tf.float64


class TestSkewedTPoissonSSM:
    """Tests for Skewed-t Poisson state space model (TF)."""

    @pytest.fixture
    def model(self):
        """Create a smaller model for faster testing (4x4 grid = 16 dims)."""
        return SkewedTPoissonSSM(d=16, alpha=0.9)

    def test_init(self, model):
        """Model should initialize with correct dimensions."""
        assert model.d == 16
        assert model.state_dim == 16
        assert model.obs_dim == 16
        assert model.alpha == 0.9
        assert model.Sigma.shape == (16, 16)
        assert model.Sigma_tilde.shape == (16, 16)
        assert model.gamma.shape == (16,)
        assert model.Q.shape == (16, 16)

    def test_non_square_d_raises(self):
        """d must be a perfect square."""
        with pytest.raises(ValueError, match="must be a perfect square"):
            SkewedTPoissonSSM(d=15)

    def test_f_shape(self, model):
        """f should return (state_dim,) tensor."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.f(x)

        assert result.shape == (model.state_dim,)

    def test_f_transition(self, model):
        """f should implement x_k = alpha * x_{k-1}."""
        x_np = np.ones(16)
        x = tf.constant(x_np, dtype=DTYPE)

        x_next = model.f(x)

        np.testing.assert_allclose(x_next.numpy(), 0.9 * x_np)

    def test_h_shape(self, model):
        """h should return (obs_dim,) tensor."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.h(x)

        assert result.shape == (model.obs_dim,)

    def test_h_observation_at_zero(self, model):
        """h at x=0 should return m1 * ones (Poisson rate function)."""
        x = tf.constant(np.zeros(16), dtype=DTYPE)

        rate = model.h(x)

        np.testing.assert_allclose(rate.numpy(), model.m1 * np.ones(16))

    def test_F_jac_shape(self, model):
        """F_jac should return (state_dim, state_dim) matrix."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        F = model.F_jac(x)

        assert F.shape == (model.state_dim, model.state_dim)
        np.testing.assert_allclose(F.numpy(), 0.9 * np.eye(16))

    def test_H_jac_shape(self, model):
        """H_jac should return (obs_dim, state_dim) diagonal matrix."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        H = model.H_jac(x)

        assert H.shape == (model.obs_dim, model.state_dim)

    def test_H_jac_numerical(self, model):
        """H_jac should match numerical differentiation."""
        x_np = np.random.randn(model.state_dim) * 0.5
        x = tf.constant(x_np, dtype=DTYPE)

        H = model.H_jac(x).numpy()

        eps = 1e-7
        H_num = np.zeros((model.obs_dim, model.state_dim))
        for i in range(model.state_dim):
            x_plus = x_np.copy()
            x_plus[i] += eps
            x_minus = x_np.copy()
            x_minus[i] -= eps
            h_plus = model.h(tf.constant(x_plus, dtype=DTYPE)).numpy()
            h_minus = model.h(tf.constant(x_minus, dtype=DTYPE)).numpy()
            H_num[:, i] = (h_plus - h_minus) / (2 * eps)

        np.testing.assert_allclose(H, H_num, rtol=1e-5, atol=1e-10)

    def test_R_state_dependent(self, model):
        """R should be state-dependent (Poisson variance = mean)."""
        x = tf.constant(np.zeros(16), dtype=DTYPE)

        R = model.R_state_dependent(x)

        expected = np.diag(model.h(x).numpy())
        np.testing.assert_allclose(R.numpy(), expected)

    def test_f_batch(self, model):
        """f_batch should apply f to each particle."""
        N = 10
        particles_np = np.random.randn(N, model.state_dim)
        particles = tf.constant(particles_np, dtype=DTYPE)

        result = model.f_batch(particles)

        assert result.shape == (N, model.state_dim)
        np.testing.assert_allclose(result.numpy(), 0.9 * particles_np)

    def test_h_batch(self, model):
        """h_batch should match single h for each particle."""
        N = 10
        particles_np = np.random.randn(N, model.state_dim) * 0.5
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
        assert not np.any(np.isnan(noise.numpy()))

    def test_log_likelihood(self, model):
        """log_likelihood should return N finite values."""
        N = 50
        particles = tf.constant(np.random.randn(N, model.state_dim) * 0.5, dtype=DTYPE)
        y = tf.constant(np.random.default_rng(0).poisson(1, size=16).astype(float), dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik.numpy()))

    def test_sample_skewed_t(self, rng, model):
        """sample_skewed_t should return valid samples."""
        mu = np.zeros(16)

        sample = model.sample_skewed_t(mu, rng)

        assert sample.shape == (16,)
        assert not np.any(np.isnan(sample))

    def test_simulate_shapes(self, rng, model):
        """simulate should return correct shapes."""
        T = 30

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 16)
        assert ys.shape == (T, 16)

    def test_simulate_no_nan(self, rng, model):
        """simulate should not produce NaN."""
        T = 30

        xs, ys = model.simulate(T, rng)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_poisson_observations(self, rng, model):
        """Observations should be non-negative integers (Poisson counts)."""
        T = 30

        xs, ys = model.simulate(T, rng)

        assert np.all(ys >= 0)
        np.testing.assert_allclose(ys, np.round(ys))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
