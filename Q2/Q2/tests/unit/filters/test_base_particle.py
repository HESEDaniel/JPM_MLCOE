"""Unit tests for BaseParticleFilter static utilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.filters.base_particle import BaseParticleFilter

DTYPE = tf.float64


class TestInitParticles:
    """Tests for init_particles (Cholesky-based sampling from N(m0, P0))."""

    def test_shape(self, tf_rng):
        m0 = tf.zeros(3, dtype=DTYPE)
        P0 = tf.eye(3, dtype=DTYPE)
        particles = BaseParticleFilter.init_particles(m0, P0, 100, tf_rng)
        assert particles.shape == (100, 3)

    def test_sample_mean_converges(self, tf_rng):
        m0 = tf.constant([2.0, -1.0], dtype=DTYPE)
        P0 = tf.eye(2, dtype=DTYPE) * 0.5
        particles = BaseParticleFilter.init_particles(m0, P0, 5000, tf_rng)
        sample_mean = tf.reduce_mean(particles, axis=0).numpy()
        np.testing.assert_allclose(sample_mean, m0.numpy(), atol=0.1)

    def test_sample_cov_converges(self, tf_rng):
        m0 = tf.zeros(2, dtype=DTYPE)
        P0 = tf.constant([[1.0, 0.3], [0.3, 0.5]], dtype=DTYPE)
        particles = BaseParticleFilter.init_particles(m0, P0, 10000, tf_rng)
        diff = particles - tf.reduce_mean(particles, axis=0)
        sample_cov = (tf.transpose(diff) @ diff / 9999.0).numpy()
        np.testing.assert_allclose(sample_cov, P0.numpy(), atol=0.1)

    def test_non_diagonal_covariance(self, tf_rng):
        """Should work with non-diagonal PSD covariance."""
        m0 = tf.zeros(3, dtype=DTYPE)
        L = tf.constant([[1.0, 0, 0], [0.5, 1.0, 0], [0.2, 0.3, 1.0]], dtype=DTYPE)
        P0 = L @ tf.transpose(L)
        particles = BaseParticleFilter.init_particles(m0, P0, 50, tf_rng)
        assert particles.shape == (50, 3)
        assert not np.any(np.isnan(particles.numpy()))


class TestWeightedMoments:
    """Tests for weighted_moments."""

    def test_uniform_weights_match_mean(self):
        particles = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=DTYPE)
        w = tf.ones(3, dtype=DTYPE) / 3.0
        m, P = BaseParticleFilter.weighted_moments(particles, w)
        np.testing.assert_allclose(m.numpy(), [3.0, 4.0], atol=1e-10)

    def test_concentrated_weight(self):
        """Single dominant weight -> mean near that particle."""
        particles = tf.constant([[0.0, 0.0], [10.0, 10.0]], dtype=DTYPE)
        w = tf.constant([0.99, 0.01], dtype=DTYPE)
        m, _ = BaseParticleFilter.weighted_moments(particles, w)
        np.testing.assert_allclose(m.numpy(), [0.1, 0.1], atol=0.01)

    def test_covariance_psd(self):
        rng = tf.random.Generator.from_seed(0)
        particles = rng.normal(shape=(100, 3), dtype=DTYPE)
        w = tf.nn.softmax(rng.normal(shape=(100,), dtype=DTYPE))
        _, P = BaseParticleFilter.weighted_moments(particles, w)
        eigvals = np.linalg.eigvalsh(P.numpy())
        assert np.all(eigvals >= -1e-10)


class TestEnsembleMoments:
    """Tests for ensemble_moments (equal-weight sample moments)."""

    def test_mean_matches_reduce_mean(self):
        particles = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
        m, _ = BaseParticleFilter.ensemble_moments(particles)
        np.testing.assert_allclose(m.numpy(), [2.0, 3.0], atol=1e-10)

    def test_covariance_divisor(self):
        """Ensemble cov uses N-1 (unbiased estimator)."""
        particles = tf.constant([[0.0], [2.0]], dtype=DTYPE)
        _, P = BaseParticleFilter.ensemble_moments(particles)
        # var = (1^2 + 1^2) / (2-1) = 2.0
        np.testing.assert_allclose(P.numpy(), [[2.0]], atol=1e-10)


class TestComputeESS:
    """Tests for compute_ess."""

    def test_uniform_weights_give_N(self):
        N = 100
        w = tf.ones(N, dtype=DTYPE) / float(N)
        ess = BaseParticleFilter.compute_ess(w)
        np.testing.assert_allclose(ess.numpy(), N, rtol=1e-10)

    def test_concentrated_weight_gives_near_one(self):
        w = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        ess = BaseParticleFilter.compute_ess(w)
        np.testing.assert_allclose(ess.numpy(), 1.0, rtol=1e-10)

    def test_ess_between_1_and_N(self):
        rng = tf.random.Generator.from_seed(7)
        w = tf.nn.softmax(rng.normal(shape=(50,), dtype=DTYPE))
        ess = BaseParticleFilter.compute_ess(w).numpy()
        assert 1.0 <= ess <= 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
