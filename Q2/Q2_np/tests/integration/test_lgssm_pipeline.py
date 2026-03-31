"""Integration tests for Linear Gaussian SSM pipeline."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import linear_gaussian_ssm
from src.filters import kalman_filter, extended_kalman_filter, unscented_kalman_filter, particle_filter
from src.flows import rkhs_pff_linear_gaussian
from src.utils.metrics import compute_nees, compute_rmse


@pytest.fixture
def lgssm_data(rng):
    """Generate LGSSM data for testing."""
    A = np.array([[1.0, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0], [0, 0.1]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.array([[0.1, 0], [0, 0.1]])
    Sigma = np.eye(2)
    T = 100

    xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

    Q = B @ B.T
    R = D @ D.T
    m0 = np.zeros(2)
    P0 = Sigma

    return {
        'A': A, 'B': B, 'C': C, 'D': D, 'Sigma': Sigma,
        'Q': Q, 'R': R, 'm0': m0, 'P0': P0,
        'xs': xs, 'ys': ys, 'T': T
    }


class TestKFOnLGSSM:
    """Test Kalman Filter on LGSSM."""

    def test_kf_on_lgssm(self, lgssm_data):
        """KF should be optimal for LGSSM, NEES should be approximately n_x."""
        d = lgssm_data

        m_filt, P_filt, cond_nums = kalman_filter(
            d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'], joseph=True
        )

        # Compute NEES
        nees = compute_nees(m_filt, P_filt, d['xs'])

        # For consistent filter, mean NEES should be approximately n_x = 2
        mean_nees = np.mean(nees)
        assert 0.5 < mean_nees < 4.0, f"Mean NEES = {mean_nees}"

        # RMSE should be reasonable
        rmse = compute_rmse(m_filt, d['xs'])
        assert rmse < 1.0


class TestEKFMatchesKF:
    """Test that EKF matches KF for linear systems."""

    def test_ekf_matches_kf_on_lgssm(self, lgssm_data):
        """EKF with linear functions should match KF."""
        d = lgssm_data

        # KF
        m_kf, P_kf, _ = kalman_filter(
            d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'], joseph=True
        )

        # EKF with linear functions
        def f(x):
            return d['A'] @ x

        def h(x):
            return d['C'] @ x

        def F_jac(x):
            return d['A']

        def H_jac(x):
            return d['C']

        m_ekf, P_ekf, _ = extended_kalman_filter(
            f, h, F_jac, H_jac, d['Q'], d['R'], d['m0'], d['P0'], d['ys']
        )

        np.testing.assert_allclose(m_ekf, m_kf, rtol=1e-5)
        np.testing.assert_allclose(P_ekf, P_kf, rtol=1e-5)


class TestUKFMatchesKF:
    """Test that UKF matches KF for linear systems."""

    def test_ukf_matches_kf_on_lgssm(self, lgssm_data):
        """UKF should match KF for linear systems."""
        d = lgssm_data

        # KF
        m_kf, P_kf, _ = kalman_filter(
            d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'], joseph=True
        )

        # UKF with linear functions
        def f(x):
            return d['A'] @ x

        def h(x):
            return d['C'] @ x

        m_ukf, P_ukf, _ = unscented_kalman_filter(
            f, h, d['Q'], d['R'], d['m0'], d['P0'], d['ys']
        )

        np.testing.assert_allclose(m_ukf, m_kf, rtol=1e-4)
        np.testing.assert_allclose(P_ukf, P_kf, rtol=1e-4)


class TestPFConvergesToKF:
    """Test that PF converges to KF with many particles."""

    def test_pf_converges_to_kf(self, rng, lgssm_data):
        """PF with N=5000 should approach KF accuracy."""
        d = lgssm_data

        # KF
        m_kf, P_kf, _ = kalman_filter(
            d['A'], d['B'], d['C'], d['D'], d['Sigma'], d['ys'], joseph=True
        )

        # PF
        def f(x):
            return (d['A'] @ x.T).T if x.ndim > 1 else d['A'] @ x

        def h(x):
            return (d['C'] @ x.T).T if x.ndim > 1 else d['C'] @ x

        def Q_sampler(rng, N):
            return rng.multivariate_normal(np.zeros(2), d['Q'], size=N)

        def log_likelihood(y, particles):
            diff = y - h(particles)
            R_inv = np.linalg.inv(d['R'])
            return -0.5 * np.sum(diff * (R_inv @ diff.T).T, axis=1)

        m_pf, P_pf, ess, _ = particle_filter(
            f, h, Q_sampler, log_likelihood, d['m0'], d['P0'], d['ys'],
            N_particles=5000, rng=rng
        )

        # PF RMSE should be close to KF RMSE
        rmse_kf = compute_rmse(m_kf, d['xs'])
        rmse_pf = compute_rmse(m_pf, d['xs'])

        # PF should be within factor of 2.5 of KF
        assert rmse_pf < rmse_kf * 2.5


class TestRKHSPFFOnLGSSM:
    """Test RKHS PFF on LGSSM."""

    def test_rkhs_pff_on_lgssm(self, rng, lgssm_data):
        """RKHS PFF should work on LGSSM."""
        d = lgssm_data

        m_filt, P_filt, ess, _ = rkhs_pff_linear_gaussian(
            d['ys'], d['A'], d['C'], d['Q'], d['R'], d['m0'], d['P0'],
            N_particles=200, n_flow_steps=15, rng=rng
        )

        # Should produce valid output
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))

        # RMSE should be reasonable
        rmse = compute_rmse(m_filt, d['xs'])
        assert rmse < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
