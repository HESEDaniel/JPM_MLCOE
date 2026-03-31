"""Unit tests for TensorFlow Kalman Filter implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.filters import KalmanFilter, FilterResult
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def kf_ssm():
    """Construct a LinearGaussianSSM for KF tests."""
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.1]])
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


@pytest.fixture
def kf_ssm_2d_obs():
    """SSM with 2D observations for KF tests."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


class TestKalmanFilter:
    """Tests for KalmanFilter class."""

    def test_output_shapes_and_finite(self, rng, kf_ssm):
        """Verify correct output shapes and no NaN/Inf."""
        T = 30
        xs, ys = kf_ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        kf = KalmanFilter()
        res = kf.filter(kf_ssm, ys_tf)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert 'cond_nums' in res.diagnostics
        assert res.diagnostics['cond_nums'].shape == (T,)
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))
        assert np.all(np.isfinite(res.diagnostics['cond_nums'].numpy()))

    def test_joseph_vs_standard_update(self, rng, kf_ssm_2d_obs):
        """Joseph and standard update should produce similar means."""
        ssm = kf_ssm_2d_obs
        T = 30
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        kf_joseph = KalmanFilter(joseph=True)
        kf_std = KalmanFilter(joseph=False)

        res_joseph = kf_joseph.filter(ssm, ys_tf)
        res_std = kf_std.filter(ssm, ys_tf)

        np.testing.assert_allclose(
            res_joseph.m_filt.numpy(), res_std.m_filt.numpy(), rtol=1e-5)

        # Joseph covariances should all be PSD
        P_np = res_joseph.P_filt.numpy()
        for t in range(T):
            eigvals = np.linalg.eigvalsh(P_np[t])
            assert np.all(eigvals >= -1e-10), f"Non-PSD at t={t}"

    def test_mse_decreases_with_observations(self, rng, kf_ssm_2d_obs):
        """Filtered MSE should be lower than prior MSE."""
        ssm = kf_ssm_2d_obs
        T = 50
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        kf = KalmanFilter()
        res = kf.filter(ssm, ys_tf)

        m_np = res.m_filt.numpy()
        mse_filter = np.mean((m_np - xs) ** 2)
        mse_prior = np.mean(xs ** 2)  # Prior mean is 0
        assert mse_filter < mse_prior

    def test_predict_step(self, kf_ssm_2d_obs):
        """Test single predict step."""
        ssm = kf_ssm_2d_obs
        kf = KalmanFilter()
        m = tf.constant([1.0, 2.0], dtype=DTYPE)
        P = tf.eye(2, dtype=DTYPE)

        m_pred, P_pred = kf.predict(m, P, ssm)

        assert m_pred.shape == (2,)
        assert P_pred.shape == (2, 2)
        # Expected: A @ m
        expected_m = tf.linalg.matvec(ssm.A, m)
        np.testing.assert_allclose(m_pred.numpy(), expected_m.numpy())

    def test_update_step(self, kf_ssm_2d_obs):
        """Test single update step."""
        ssm = kf_ssm_2d_obs
        kf = KalmanFilter()
        m_pred = tf.constant([1.0, 2.0], dtype=DTYPE)
        P_pred = tf.eye(2, dtype=DTYPE)
        y = tf.constant([0.5, 0.3], dtype=DTYPE)

        m_post, P_post = kf.update(m_pred, P_pred, y, ssm)

        assert m_post.shape == (2,)
        assert P_post.shape == (2, 2)
        # Posterior covariance should have smaller trace
        assert np.trace(P_post.numpy()) <= np.trace(P_pred.numpy()) + 1e-10

    def test_deterministic(self, rng, kf_ssm):
        """KF should produce identical results on same data."""
        T = 20
        xs, ys = kf_ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        kf = KalmanFilter()
        res1 = kf.filter(kf_ssm, ys_tf)
        res2 = kf.filter(kf_ssm, ys_tf)

        np.testing.assert_array_equal(res1.m_filt.numpy(), res2.m_filt.numpy())


class TestKalmanFilterShortTrajectory:
    """Test KF on very short trajectories."""

    def test_single_step(self, rng, kf_ssm):
        """KF should work for T=1."""
        xs, ys = kf_ssm.simulate(1, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        kf = KalmanFilter()
        res = kf.filter(kf_ssm, ys_tf)

        assert res.m_filt.shape == (1, 2)
        assert not np.any(np.isnan(res.m_filt.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
