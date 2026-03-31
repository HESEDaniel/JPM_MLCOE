"""Unit tests for Spatial Sensor Network SSM (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import SpatialSensorNetwork

DTYPE = tf.float64


class TestSpatialSensorNetwork:
    """Tests for Spatial Sensor Network state space model (TF)."""

    @pytest.fixture
    def model(self):
        """Create a smaller model for faster testing (4x4 grid = 16 dims)."""
        return SpatialSensorNetwork(d=16, alpha=0.9)

    def test_init(self, model):
        """Model should initialize with correct dimensions."""
        assert model.d == 16
        assert model.state_dim == 16
        assert model.obs_dim == 16
        assert model.alpha == 0.9
        assert model.Q.shape == (16, 16)
        assert model.R.shape == (16, 16)
        assert model.A.shape == (16, 16)
        assert model.H.shape == (16, 16)

    def test_non_square_d_raises(self):
        """d must be a perfect square."""
        with pytest.raises(ValueError, match="must be a perfect square"):
            SpatialSensorNetwork(d=15)

    def test_spatial_covariance(self, model):
        """Q should be symmetric and positive definite with spatial structure."""
        Q_np = model.Q.numpy()

        # Symmetric
        np.testing.assert_allclose(Q_np, Q_np.T, atol=1e-12)

        # Positive definite
        eigvals = np.linalg.eigvalsh(Q_np)
        assert np.all(eigvals > 0)

        # Off-diagonal entries should be non-zero (spatial correlation)
        assert Q_np[0, 1] > 0

    def test_f_shape(self, model):
        """f should return (state_dim,) tensor."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.f(x)

        assert result.shape == (model.state_dim,)

    def test_f_correctness(self, model):
        """f should implement x_next = alpha * x."""
        x_np = np.random.randn(model.state_dim)
        x = tf.constant(x_np, dtype=DTYPE)

        result = model.f(x).numpy()

        np.testing.assert_allclose(result, 0.9 * x_np)

    def test_h_shape(self, model):
        """h should return (obs_dim,) tensor (identity)."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.h(x)

        assert result.shape == (model.obs_dim,)

    def test_h_is_identity(self, model):
        """h should be the identity function."""
        x_np = np.random.randn(model.state_dim)
        x = tf.constant(x_np, dtype=DTYPE)

        result = model.h(x).numpy()

        np.testing.assert_allclose(result, x_np)

    def test_F_jac_shape(self, model):
        """F_jac should return (state_dim, state_dim) matrix."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        F = model.F_jac(x)

        assert F.shape == (model.state_dim, model.state_dim)
        np.testing.assert_allclose(F.numpy(), 0.9 * np.eye(16))

    def test_H_jac_shape(self, model):
        """H_jac should return (obs_dim, state_dim) identity matrix."""
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        H = model.H_jac(x)

        assert H.shape == (model.obs_dim, model.state_dim)
        np.testing.assert_allclose(H.numpy(), np.eye(16))

    def test_f_batch(self, model):
        """f_batch should apply f to each particle."""
        N = 10
        particles_np = np.random.randn(N, model.state_dim)
        particles = tf.constant(particles_np, dtype=DTYPE)

        result = model.f_batch(particles)

        assert result.shape == (N, model.state_dim)
        np.testing.assert_allclose(result.numpy(), 0.9 * particles_np)

    def test_h_batch(self, model):
        """h_batch should be identity on particles."""
        N = 10
        particles_np = np.random.randn(N, model.state_dim)
        particles = tf.constant(particles_np, dtype=DTYPE)

        result = model.h_batch(particles)

        assert result.shape == (N, model.obs_dim)
        np.testing.assert_allclose(result.numpy(), particles_np)

    def test_Q_sampler(self, model):
        """Q_sampler should return (N, state_dim) samples with approximately correct covariance."""
        rng = tf.random.Generator.from_seed(42)
        N = 1000

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, model.state_dim)

        # Check empirical covariance approximates Q
        cov_empirical = np.cov(noise.numpy().T)
        np.testing.assert_allclose(cov_empirical, model.Q.numpy(), rtol=0.3)

    def test_log_likelihood(self, model):
        """log_likelihood should return N finite values."""
        N = 50
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)
        y = tf.constant(np.random.randn(model.obs_dim), dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik.numpy()))

    def test_simulate(self, rng, model):
        """simulate should return correct shapes without NaN."""
        T = 50

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 16)
        assert ys.shape == (T, 16)
        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
