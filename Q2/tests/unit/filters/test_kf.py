"""Unit tests for Kalman Filter implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.filters.kf import kalman_filter, _solve_lu, _solve_cholesky, _solve_inv
from src.ssm import linear_gaussian_ssm
from tests.unit.conftest import check_psd


class TestSolvers:
    """Tests for linear system solvers."""

    @pytest.mark.parametrize("solver_func", [_solve_lu, _solve_cholesky, _solve_inv])
    def test_solver_correctness(self, solver_func):
        """Solvers should produce correct solution S @ X = B."""
        S = np.array([[4.0, 1.0], [1.0, 3.0]])
        B = np.array([[1.0], [2.0]])

        X = solver_func(S, B)

        np.testing.assert_allclose(S @ X, B, rtol=1e-10)

    def test_solver_selection(self, rng, kf_system):
        """Different solvers should produce similar results."""
        A, B, C, D, Sigma = kf_system
        T = 30
        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        m_lu, P_lu, _ = kalman_filter(A, B, C, D, Sigma, ys, solver='lu')
        m_chol, P_chol, _ = kalman_filter(A, B, C, D, Sigma, ys, solver='cholesky')
        m_inv, P_inv, _ = kalman_filter(A, B, C, D, Sigma, ys, solver='inv')

        np.testing.assert_allclose(m_lu, m_chol, rtol=1e-8)
        np.testing.assert_allclose(m_lu, m_inv, rtol=1e-8)


class TestKalmanFilter:
    """Tests for Kalman Filter."""

    def test_output_shapes(self, rng, kf_system):
        """Verify correct output shapes."""
        A, B, C, D, Sigma = kf_system
        T, n_x = 50, A.shape[0]
        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, T, rng)

        m_filt, P_filt, cond_nums = kalman_filter(A, B, C, D, Sigma, ys, joseph=True)

        assert m_filt.shape == (T, n_x)
        assert P_filt.shape == (T, n_x, n_x)
        assert cond_nums.shape == (T,)

    def test_no_nan(self, rng, kf_system):
        """Output should not contain NaN."""
        A, B, C, D, Sigma = kf_system
        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, 100, rng)

        m_filt, P_filt, cond_nums = kalman_filter(A, B, C, D, Sigma, ys, joseph=True)

        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert not np.any(np.isnan(cond_nums))

    def test_joseph_vs_standard_update(self, rng, kf_system):
        """Joseph update should produce valid covariances even when standard might not."""
        A, B, C, D, Sigma = kf_system
        xs, ys = linear_gaussian_ssm(A, B, C, D, Sigma, 50, rng)

        # Joseph update
        m_joseph, P_joseph, _ = kalman_filter(A, B, C, D, Sigma, ys, joseph=True)
        # Standard update
        m_std, P_std, _ = kalman_filter(A, B, C, D, Sigma, ys, joseph=False)

        # Both should produce similar means
        np.testing.assert_allclose(m_joseph, m_std, rtol=1e-5)

        # Joseph covariances should all be PSD
        for t in range(len(P_joseph)):
            assert check_psd(P_joseph[t])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
