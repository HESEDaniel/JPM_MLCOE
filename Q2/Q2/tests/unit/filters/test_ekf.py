"""Unit tests for TensorFlow Extended Kalman Filter implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.filters import KalmanFilter, ExtendedKalmanFilter, FilterResult
from src.ssm import LinearGaussianSSM, RangeBearing

DTYPE = tf.float64


class TestEKFPredict:
    """Tests for EKF prediction step."""

    def test_mean_propagation(self, linear_model):
        """Mean should propagate through f."""
        ssm = linear_model
        ekf = ExtendedKalmanFilter()
        m = tf.constant([1.0, 2.0], dtype=DTYPE)
        P = ssm.P0

        m_pred, P_pred = ekf.predict(m, P, ssm)

        expected = tf.linalg.matvec(ssm.A, m)
        np.testing.assert_allclose(m_pred.numpy(), expected.numpy(), rtol=1e-10)
        assert m_pred.shape == (2,)
        assert P_pred.shape == (2, 2)

    def test_covariance_grows(self, linear_model):
        """Predicted covariance should be larger than prior (adds Q)."""
        ssm = linear_model
        ekf = ExtendedKalmanFilter()
        m = tf.zeros(2, dtype=DTYPE)
        P = 0.01 * tf.eye(2, dtype=DTYPE)

        _, P_pred = ekf.predict(m, P, ssm)

        # P_pred = F P F^T + Q >= Q
        assert np.trace(P_pred.numpy()) > np.trace(P.numpy())


class TestEKFUpdate:
    """Tests for EKF update step."""

    def test_joseph_update_stability(self, linear_model):
        """Joseph update should maintain PSD."""
        ssm = linear_model
        ekf = ExtendedKalmanFilter(joseph=True)
        m_pred = tf.constant([1.0, 2.0], dtype=DTYPE)
        P_pred = ssm.P0
        y = tf.constant([0.5, 0.3], dtype=DTYPE)

        m_upd, P_upd = ekf.update(m_pred, P_pred, y, ssm)

        assert m_upd.shape == (2,)
        assert P_upd.shape == (2, 2)
        eigvals = np.linalg.eigvalsh(P_upd.numpy())
        assert np.all(eigvals >= -1e-10)

    def test_update_reduces_covariance(self, linear_model):
        """Update should reduce covariance trace."""
        ssm = linear_model
        ekf = ExtendedKalmanFilter()
        m_pred = tf.constant([1.0, 2.0], dtype=DTYPE)
        P_pred = ssm.P0
        y = tf.constant([0.5, 0.3], dtype=DTYPE)

        _, P_upd = ekf.update(m_pred, P_pred, y, ssm)

        assert np.trace(P_upd.numpy()) <= np.trace(P_pred.numpy()) + 1e-10


class TestExtendedKalmanFilter:
    """Tests for full Extended Kalman Filter."""

    def test_output_shapes_and_finite(self, rng, linear_model):
        """Verify correct output shapes and no NaN/Inf."""
        ssm = linear_model
        T = 30
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        ekf = ExtendedKalmanFilter()
        res = ekf.filter(ssm, ys_tf)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert res.diagnostics['cond_nums'].shape == (T,)
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))

    def test_reduces_to_kf_for_linear(self, rng):
        """EKF should match KF for linear systems."""
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
        ekf = ExtendedKalmanFilter(joseph=True)

        res_kf = kf.filter(ssm, ys_tf)
        res_ekf = ekf.filter(ssm, ys_tf)

        np.testing.assert_allclose(
            res_ekf.m_filt.numpy(), res_kf.m_filt.numpy(), rtol=1e-5)
        np.testing.assert_allclose(
            res_ekf.P_filt.numpy(), res_kf.P_filt.numpy(), rtol=1e-5)


class TestEKFOnRangeBearing:
    """Test EKF on nonlinear RangeBearing model."""

    def test_ekf_on_range_bearing(self, rng):
        """EKF should produce valid output on RangeBearing SSM."""
        ssm = RangeBearing(dt=1.0, q=0.1, r_range=0.1, r_bearing=0.05)
        T = 20
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        ekf = ExtendedKalmanFilter(joseph=True, angle_indices=[1])
        res = ekf.filter(ssm, ys_tf)

        assert res.m_filt.shape == (T, 4)
        assert res.P_filt.shape == (T, 4, 4)
        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))

    def test_ekf_range_bearing_tracks(self, rng):
        """EKF should track the true state on RangeBearing."""
        ssm = RangeBearing(dt=1.0, q=0.05, r_range=0.1, r_bearing=0.05)
        T = 30
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        ekf = ExtendedKalmanFilter(joseph=True, angle_indices=[1])
        res = ekf.filter(ssm, ys_tf)

        m_np = res.m_filt.numpy()
        # Check position components (indices 0, 2) have reasonable MSE
        pos_mse = np.mean((m_np[:, [0, 2]] - xs[:, [0, 2]]) ** 2)
        assert pos_mse < 10.0, f"Position MSE too high: {pos_mse}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
