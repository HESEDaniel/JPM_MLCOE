"""Unit tests for TensorFlow Unscented Kalman Filter implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.filters import KalmanFilter, UnscentedKalmanFilter, FilterResult
from src.filters.ukf import _ukf_weights, _sigma_points
from src.ssm import LinearGaussianSSM, RangeBearing

DTYPE = tf.float64


class TestUKFWeights:
    """Tests for UKF weight computation."""

    def test_weights_sum_to_one(self):
        """Mean weights should sum to 1."""
        n_x = 4
        W_m, W_c, gamma = _ukf_weights(n_x, alpha=1e-3, beta=2.0, kappa=0.0)

        assert W_m.shape == (2 * n_x + 1,)
        np.testing.assert_allclose(
            tf.reduce_sum(W_m).numpy(), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("n_x", [1, 2, 5, 10])
    def test_weight_count(self, n_x):
        """Should produce 2*n_x + 1 weights."""
        W_m, W_c, gamma = _ukf_weights(n_x, alpha=1e-3, beta=2.0, kappa=0.0)
        assert W_m.shape[0] == 2 * n_x + 1
        assert W_c.shape[0] == 2 * n_x + 1


class TestSigmaPoints:
    """Tests for sigma point generation."""

    def test_sigma_points_shape(self):
        """Sigma points should have correct shape."""
        n_x = 3
        m = tf.zeros(n_x, dtype=DTYPE)
        P = tf.eye(n_x, dtype=DTYPE)
        gamma = tf.constant(1.0, dtype=DTYPE)

        sigma = _sigma_points(m, P, gamma)
        assert sigma.shape == (2 * n_x + 1, n_x)

    def test_fallback_for_near_singular(self):
        """Sigma points should handle near-singular covariance."""
        n_x = 2
        m = tf.zeros(n_x, dtype=DTYPE)
        P = tf.constant([[1.0, 0.999], [0.999, 1.0]], dtype=DTYPE)
        gamma = tf.constant(1.0, dtype=DTYPE)

        sigma = _sigma_points(m, P, gamma)

        assert sigma.shape == (2 * n_x + 1, n_x)
        assert not np.any(np.isnan(sigma.numpy()))

    def test_center_is_mean(self):
        """First sigma point should be the mean."""
        n_x = 2
        m = tf.constant([3.0, 5.0], dtype=DTYPE)
        P = tf.eye(n_x, dtype=DTYPE)
        gamma = tf.constant(1.0, dtype=DTYPE)

        sigma = _sigma_points(m, P, gamma)
        np.testing.assert_allclose(sigma[0].numpy(), m.numpy())


class TestUKFPredict:
    """Tests for UKF prediction step."""

    def test_mean_through_linear(self, linear_model):
        """For linear f, predicted mean should match A @ m."""
        ssm = linear_model
        ukf = UnscentedKalmanFilter()
        m = tf.constant([1.0, 2.0], dtype=DTYPE)
        P = ssm.P0

        m_pred, P_pred = ukf.predict(m, P, ssm)

        expected = tf.linalg.matvec(ssm.A, m)
        np.testing.assert_allclose(m_pred.numpy(), expected.numpy(), rtol=1e-5)
        assert m_pred.shape == (2,)
        assert P_pred.shape == (2, 2)


class TestUKFUpdate:
    """Tests for UKF update step."""

    def test_update_reduces_covariance(self, linear_model):
        """Update should reduce covariance."""
        ssm = linear_model
        ukf = UnscentedKalmanFilter()
        m_pred = tf.constant([1.0, 2.0], dtype=DTYPE)
        P_pred = ssm.P0
        y = tf.constant([0.5, 0.3], dtype=DTYPE)

        m_upd, P_upd = ukf.update(m_pred, P_pred, y, ssm)

        assert m_upd.shape == (2,)
        assert P_upd.shape == (2, 2)
        assert np.trace(P_upd.numpy()) <= np.trace(P_pred.numpy()) + 1e-10


class TestUnscentedKalmanFilter:
    """Tests for full Unscented Kalman Filter."""

    def test_output_shapes_and_finite(self, rng, linear_model):
        """Verify correct output shapes and no NaN/Inf."""
        ssm = linear_model
        T = 30
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        ukf = UnscentedKalmanFilter()
        res = ukf.filter(ssm, ys_tf)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert res.diagnostics['cond_nums'].shape == (T,)
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))

    def test_reduces_to_kf_for_linear(self, rng):
        """UKF should match KF for linear systems."""
        A = np.array([[0.9, 0.1], [0.0, 0.95]])
        B = np.array([[0.1, 0.0], [0.0, 0.1]])
        C = np.eye(2)
        D = np.array([[0.1, 0.0], [0.0, 0.1]])
        Sigma = np.eye(2)

        ssm = LinearGaussianSSM(A, B, C, D, Sigma)
        T = 30
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        kf = KalmanFilter(joseph=True)
        ukf = UnscentedKalmanFilter()

        res_kf = kf.filter(ssm, ys_tf)
        res_ukf = ukf.filter(ssm, ys_tf)

        np.testing.assert_allclose(
            res_ukf.m_filt.numpy(), res_kf.m_filt.numpy(), rtol=1e-4)
        np.testing.assert_allclose(
            res_ukf.P_filt.numpy(), res_kf.P_filt.numpy(), rtol=1e-4)

    def test_ukf_on_range_bearing(self, rng):
        """UKF should work on nonlinear RangeBearing SSM."""
        ssm = RangeBearing(dt=1.0, q=0.1, r_range=0.1, r_bearing=0.05)
        T = 20
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        ukf = UnscentedKalmanFilter()
        res = ukf.filter(ssm, ys_tf)

        assert res.m_filt.shape == (T, 4)
        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
