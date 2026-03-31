"""Unit tests for Linear Gaussian SSM (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def kf_system():
    """KF-specific system (A, B, C, D, Sigma)."""
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.1]])
    Sigma = np.eye(2)
    return A, B, C, D, Sigma


class TestLinearGaussianSSM:
    """Unit tests for Linear Gaussian SSM (TF)."""

    def test_init(self, kf_system):
        """Model should initialize with correct dimensions and matrices."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)

        assert model.state_dim == 2
        assert model.obs_dim == 1
        assert model.Q.shape == (2, 2)
        assert model.R.shape == (1, 1)
        assert model.P0.shape == (2, 2)

        # Q = B @ B^T
        expected_Q = B @ B.T
        np.testing.assert_allclose(model.Q.numpy(), expected_Q)

        # R = D @ D^T
        expected_R = D @ D.T
        np.testing.assert_allclose(model.R.numpy(), expected_R)

    def test_f_shape(self, kf_system):
        """f should return (state_dim,) tensor."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.f(x)

        assert result.shape == (model.state_dim,)

    def test_h_shape(self, kf_system):
        """h should return (obs_dim,) tensor."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.h(x)

        assert result.shape == (model.obs_dim,)

    def test_f_correctness(self, kf_system):
        """f should implement x_next = A @ x."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        x_np = np.array([1.0, 2.0])
        x = tf.constant(x_np, dtype=DTYPE)

        result = model.f(x).numpy()

        np.testing.assert_allclose(result, A @ x_np)

    def test_h_correctness(self, kf_system):
        """h should implement y = C @ x."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        x_np = np.array([1.0, 2.0])
        x = tf.constant(x_np, dtype=DTYPE)

        result = model.h(x).numpy()

        np.testing.assert_allclose(result, C @ x_np)

    def test_F_jac_shape(self, kf_system):
        """F_jac should return (state_dim, state_dim) matrix."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        F = model.F_jac(x)

        assert F.shape == (model.state_dim, model.state_dim)
        np.testing.assert_allclose(F.numpy(), A)

    def test_H_jac_shape(self, kf_system):
        """H_jac should return (obs_dim, state_dim) matrix."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        H = model.H_jac(x)

        assert H.shape == (model.obs_dim, model.state_dim)
        np.testing.assert_allclose(H.numpy(), C)

    def test_f_batch(self, kf_system):
        """f_batch should apply f to each particle."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        N = 10
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)

        result = model.f_batch(particles)

        assert result.shape == (N, model.state_dim)
        # Verify each row matches A @ x
        for i in range(N):
            np.testing.assert_allclose(
                result[i].numpy(), A @ particles[i].numpy(), rtol=1e-12)

    def test_h_batch(self, kf_system):
        """h_batch should apply h to each particle."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        N = 10
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)

        result = model.h_batch(particles)

        assert result.shape == (N, model.obs_dim)

    def test_Q_sampler(self, kf_system):
        """Q_sampler should return (N, state_dim) noise samples."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        rng = tf.random.Generator.from_seed(42)
        N = 20

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, model.state_dim)

    def test_log_likelihood(self, kf_system):
        """log_likelihood should return (N,) values."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        N = 50
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)
        y = tf.constant([0.5], dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)

    def test_simulate(self, rng, kf_system):
        """simulate should return correct shapes without NaN."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        T = 100

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 2)
        assert ys.shape == (T, 1)
        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_reproducibility(self, kf_system):
        """Same seed should give same results."""
        A, B, C, D, Sigma = kf_system
        model = LinearGaussianSSM(A, B, C, D, Sigma)
        T = 50

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        xs1, ys1 = model.simulate(T, rng1)
        xs2, ys2 = model.simulate(T, rng2)

        np.testing.assert_array_equal(xs1, xs2)
        np.testing.assert_array_equal(ys1, ys2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
