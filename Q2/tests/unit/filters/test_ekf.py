"""Unit tests for Extended Kalman Filter implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.filters.ekf import extended_kalman_filter, ekf_predict, ekf_update
from src.filters.kf import kalman_filter
from src.ssm import linear_gaussian_ssm
from tests.unit.conftest import check_psd


class TestEKFPredict:
    """Tests for EKF prediction step."""

    def test_mean_propagation(self, linear_model):
        """Mean should propagate through f."""
        m = linear_model
        m_in = np.array([1.0, 2.0])
        P_in = m['P0']

        m_pred, P_pred = ekf_predict(m_in, P_in, m['f'], m['F_jac'], m['Q'])

        expected = m['A'] @ m_in
        np.testing.assert_allclose(m_pred, expected)
        assert m_pred.shape == (2,)
        assert P_pred.shape == (2, 2)


class TestEKFUpdate:
    """Tests for EKF update step."""

    def test_joseph_update_stability(self, linear_model):
        """Joseph update should maintain PSD."""
        m = linear_model
        m_pred = m['m0']
        P_pred = m['P0']
        y = np.array([0.5, 0.3])

        m_upd, P_upd = ekf_update(m_pred, P_pred, y, m['h'], m['H_jac'],
                                   m['R'], joseph=True)

        assert m_upd.shape == (2,)
        assert P_upd.shape == (2, 2)
        assert check_psd(P_upd)


class TestExtendedKalmanFilter:
    """Tests for full Extended Kalman Filter."""

    def test_output_shapes(self, rng, linear_model):
        """Verify correct output shapes."""
        m = linear_model
        T = 50
        ys = rng.standard_normal((T, 2))

        m_filt, P_filt, cond_nums = extended_kalman_filter(
            m['f'], m['h'], m['F_jac'], m['H_jac'], m['Q'], m['R'],
            m['m0'], m['P0'], ys, joseph=True
        )

        assert m_filt.shape == (T, 2)
        assert P_filt.shape == (T, 2, 2)
        assert cond_nums.shape == (T,)

    def test_no_nan(self, rng, linear_model):
        """Output should not contain NaN."""
        m = linear_model
        ys = rng.standard_normal((100, 2))

        m_filt, P_filt, _ = extended_kalman_filter(
            m['f'], m['h'], m['F_jac'], m['H_jac'], m['Q'], m['R'],
            m['m0'], m['P0'], ys, joseph=True
        )

        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))

    def test_reduces_to_kf_for_linear(self, rng, kf_system):
        """EKF should match KF for linear systems."""
        A, B, C, D, Sigma = kf_system
        T = 50
        Q = B @ B.T
        R = D @ D.T

        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        # KF
        m_kf, P_kf, _ = kalman_filter(A, B, C, D, Sigma, ys, joseph=True)

        # EKF with linear functions
        def f(x):
            return A @ x

        def h(x):
            return C @ x

        def F_jac(x):
            return A

        def H_jac(x):
            return C

        m0 = np.zeros(2)
        P0 = Sigma

        m_ekf, P_ekf, _ = extended_kalman_filter(
            f, h, F_jac, H_jac, Q, R, m0, P0, ys, joseph=True
        )

        np.testing.assert_allclose(m_ekf, m_kf, rtol=1e-5)
        np.testing.assert_allclose(P_ekf, P_kf, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
