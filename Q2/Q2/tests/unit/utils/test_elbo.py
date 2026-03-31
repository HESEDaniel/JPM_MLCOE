"""Unit tests for ELBO utilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.utils.elbo import log_gaussian_diag, log_gaussian_full

DTYPE = tf.float64


class TestLogGaussianDiag:
    """Tests for diagonal-covariance Gaussian log-density."""

    def test_output_shape(self):
        N, d = 10, 3
        x = tf.zeros((N, d), dtype=DTYPE)
        mean = tf.zeros((N, d), dtype=DTYPE)
        log_var = tf.zeros(d, dtype=DTYPE)
        result = log_gaussian_diag(x, mean, log_var)
        assert result.shape == (N,)

    def test_standard_normal_at_zero(self):
        """log N(0; 0, I) = -d/2 * log(2*pi)."""
        d = 3
        x = tf.zeros((1, d), dtype=DTYPE)
        mean = tf.zeros((1, d), dtype=DTYPE)
        log_var = tf.zeros(d, dtype=DTYPE)
        result = log_gaussian_diag(x, mean, log_var).numpy()[0]
        expected = -0.5 * d * np.log(2 * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_off_center_lower_density(self):
        """Points further from mean should have lower log-density."""
        d = 2
        mean = tf.zeros((2, d), dtype=DTYPE)
        log_var = tf.zeros(d, dtype=DTYPE)
        x = tf.constant([[0.0, 0.0], [3.0, 3.0]], dtype=DTYPE)
        result = log_gaussian_diag(x, mean, log_var).numpy()
        assert result[0] > result[1]

    def test_known_1d_value(self):
        """log N(1; 0, 2) = -0.5*(log(2) + log(2*pi) + 0.5)."""
        x = tf.constant([[1.0]], dtype=DTYPE)
        mean = tf.constant([[0.0]], dtype=DTYPE)
        log_var = tf.constant([np.log(2.0)], dtype=DTYPE)
        result = log_gaussian_diag(x, mean, log_var).numpy()[0]
        expected = -0.5 * (np.log(2.0) + np.log(2 * np.pi) + 1.0 / 2.0)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestLogGaussianFull:
    """Tests for full-covariance Gaussian log-density."""

    def test_output_shape(self):
        N, d = 10, 3
        x = tf.zeros((N, d), dtype=DTYPE)
        mean = tf.zeros((N, d), dtype=DTYPE)
        cov_inv = tf.eye(d, dtype=DTYPE)
        log_det = tf.constant(0.0, dtype=DTYPE)
        result = log_gaussian_full(x, mean, cov_inv, log_det)
        assert result.shape == (N,)

    def test_standard_normal_at_zero(self):
        """Should match diag version for identity covariance."""
        d = 3
        x = tf.zeros((1, d), dtype=DTYPE)
        mean = tf.zeros((1, d), dtype=DTYPE)
        cov_inv = tf.eye(d, dtype=DTYPE)
        log_det = tf.constant(0.0, dtype=DTYPE)
        result = log_gaussian_full(x, mean, cov_inv, log_det).numpy()[0]
        expected = -0.5 * d * np.log(2 * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_diag_full_equivalence(self):
        """Diagonal cov: log_gaussian_diag and log_gaussian_full should agree."""
        d = 4
        rng = np.random.default_rng(42)
        x = tf.constant(rng.standard_normal((5, d)), dtype=DTYPE)
        mean = tf.constant(rng.standard_normal((5, d)), dtype=DTYPE)
        var = np.abs(rng.standard_normal(d)) + 0.1
        log_var = tf.constant(np.log(var), dtype=DTYPE)

        val_diag = log_gaussian_diag(x, mean, log_var).numpy()

        cov_inv = tf.linalg.diag(1.0 / tf.constant(var, dtype=DTYPE))
        log_det = tf.reduce_sum(log_var)
        val_full = log_gaussian_full(x, mean, cov_inv, log_det).numpy()

        np.testing.assert_allclose(val_full, val_diag, atol=1e-10)

    def test_broadcast_mean(self):
        """Should work with [d] mean broadcast to [N, d]."""
        d = 2
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
        mean = tf.constant([0.0, 0.0], dtype=DTYPE)
        cov_inv = tf.eye(d, dtype=DTYPE)
        log_det = tf.constant(0.0, dtype=DTYPE)
        result = log_gaussian_full(x, mean, cov_inv, log_det)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
