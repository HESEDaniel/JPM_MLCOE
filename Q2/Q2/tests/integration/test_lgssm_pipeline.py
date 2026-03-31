"""Integration tests for Linear Gaussian SSM pipeline (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import LinearGaussianSSM
from src.filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter
from src.flows import EDHFlow, RKHSFlow
from src.utils.metrics import compute_nees, compute_rmse

DTYPE = tf.float64


@pytest.fixture
def lgssm_data(rng):
    """Generate LGSSM data for testing."""
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    T = 50

    ssm = LinearGaussianSSM(A, B, C, D, Sigma)
    xs, ys = ssm.simulate(T, rng)
    ys_tf = tf.constant(ys, dtype=DTYPE)

    return {'ssm': ssm, 'xs': xs, 'ys': ys, 'ys_tf': ys_tf, 'T': T}


class TestKFOnLGSSM:
    """Test Kalman Filter on LGSSM."""

    def test_kf_nees_and_rmse(self, lgssm_data):
        """KF should be optimal for LGSSM, NEES approximately n_x."""
        d = lgssm_data
        kf = KalmanFilter(joseph=True)
        res = kf.filter(d['ssm'], d['ys_tf'])

        m_np = res.m_filt.numpy()
        P_np = res.P_filt.numpy()

        nees = compute_nees(m_np, P_np, d['xs'])
        mean_nees = np.mean(nees)
        assert 0.5 < mean_nees < 6.0, f"Mean NEES = {mean_nees}"

        rmse = compute_rmse(m_np, d['xs'])
        assert rmse < 1.0, f"RMSE = {rmse}"


class TestEKFMatchesKF:
    """Test that EKF matches KF for linear systems."""

    def test_ekf_matches_kf_on_lgssm(self, lgssm_data):
        """EKF should match KF for linear systems."""
        d = lgssm_data

        kf = KalmanFilter(joseph=True)
        ekf = ExtendedKalmanFilter(joseph=True)

        res_kf = kf.filter(d['ssm'], d['ys_tf'])
        res_ekf = ekf.filter(d['ssm'], d['ys_tf'])

        np.testing.assert_allclose(
            res_ekf.m_filt.numpy(), res_kf.m_filt.numpy(), rtol=1e-5)
        np.testing.assert_allclose(
            res_ekf.P_filt.numpy(), res_kf.P_filt.numpy(), rtol=1e-5)


class TestUKFMatchesKF:
    """Test that UKF matches KF for linear systems."""

    def test_ukf_matches_kf_on_lgssm(self, lgssm_data):
        """UKF should match KF for linear systems."""
        d = lgssm_data

        kf = KalmanFilter(joseph=True)
        ukf = UnscentedKalmanFilter()

        res_kf = kf.filter(d['ssm'], d['ys_tf'])
        res_ukf = ukf.filter(d['ssm'], d['ys_tf'])

        np.testing.assert_allclose(
            res_ukf.m_filt.numpy(), res_kf.m_filt.numpy(), rtol=1e-4)
        np.testing.assert_allclose(
            res_ukf.P_filt.numpy(), res_kf.P_filt.numpy(), rtol=1e-4)


class TestPFConvergesToKF:
    """Test that PF converges to KF with many particles."""

    def test_pf_converges_to_kf(self, rng, lgssm_data):
        """PF with N=2000 should approach KF accuracy."""
        d = lgssm_data

        kf = KalmanFilter(joseph=True)
        res_kf = kf.filter(d['ssm'], d['ys_tf'])

        pf = ParticleFilter(n_particles=2000, resample_threshold=0.5)
        tf_rng = tf.random.Generator.from_seed(42)
        res_pf = pf.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

        rmse_kf = compute_rmse(res_kf.m_filt.numpy(), d['xs'])
        rmse_pf = compute_rmse(res_pf.m_filt.numpy(), d['xs'])

        # PF should be within factor of 3 of KF
        assert rmse_pf < rmse_kf * 3.0, (
            f"PF RMSE={rmse_pf:.4f} too large vs KF RMSE={rmse_kf:.4f}")


class TestEDHOnLGSSM:
    """Test EDH flow on LGSSM."""

    def test_edh_on_lgssm(self, lgssm_data):
        """EDH should produce valid output on LGSSM."""
        d = lgssm_data

        edh = EDHFlow(n_particles=100, n_flow_steps=10)
        tf_rng = tf.random.Generator.from_seed(42)
        res = edh.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))

        rmse = compute_rmse(res.m_filt.numpy(), d['xs'])
        assert rmse < 2.0, f"EDH RMSE = {rmse}"


class TestRKHSPFFOnLGSSM:
    """Test RKHS PFF on LGSSM."""

    @pytest.mark.xfail(reason="RKHS PFF may diverge on small LGSSM")
    def test_rkhs_pff_on_lgssm(self, lgssm_data):
        """RKHS PFF should produce valid output on LGSSM."""
        d = lgssm_data

        flow = RKHSFlow(n_particles=100, n_flow_steps=10, step_size=0.1)
        tf_rng = tf.random.Generator.from_seed(42)
        res = flow.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))

        rmse = compute_rmse(res.m_filt.numpy(), d['xs'])
        assert rmse < 3.0, f"RKHS PFF RMSE = {rmse}"


class TestAllFiltersDeterministic:
    """Test that deterministic filters produce identical results on same data."""

    def test_kf_deterministic(self, lgssm_data):
        """KF should be fully deterministic."""
        d = lgssm_data
        kf = KalmanFilter()

        res1 = kf.filter(d['ssm'], d['ys_tf'])
        res2 = kf.filter(d['ssm'], d['ys_tf'])

        np.testing.assert_array_equal(
            res1.m_filt.numpy(), res2.m_filt.numpy())

    def test_ekf_deterministic(self, lgssm_data):
        """EKF should be fully deterministic."""
        d = lgssm_data
        ekf = ExtendedKalmanFilter()

        res1 = ekf.filter(d['ssm'], d['ys_tf'])
        res2 = ekf.filter(d['ssm'], d['ys_tf'])

        np.testing.assert_array_equal(
            res1.m_filt.numpy(), res2.m_filt.numpy())


class TestShortTrajectory:
    """Test filters work for very short trajectories."""

    def test_single_step_all_filters(self, rng):
        """All filters should work for T=1."""
        A = np.array([[0.9, 0.1], [0.0, 0.95]])
        B = np.array([[0.1, 0.0], [0.0, 0.1]])
        C = np.eye(2)
        D = np.array([[0.1, 0.0], [0.0, 0.1]])
        Sigma = np.eye(2)

        ssm = LinearGaussianSSM(A, B, C, D, Sigma)
        xs, ys = ssm.simulate(1, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        # KF
        res_kf = KalmanFilter().filter(ssm, ys_tf)
        assert res_kf.m_filt.shape == (1, 2)
        assert not np.any(np.isnan(res_kf.m_filt.numpy()))

        # EKF
        res_ekf = ExtendedKalmanFilter().filter(ssm, ys_tf)
        assert res_ekf.m_filt.shape == (1, 2)
        assert not np.any(np.isnan(res_ekf.m_filt.numpy()))

        # UKF
        res_ukf = UnscentedKalmanFilter().filter(ssm, ys_tf)
        assert res_ukf.m_filt.shape == (1, 2)
        assert not np.any(np.isnan(res_ukf.m_filt.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
