"""Unit tests for PF-PF EDH implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.flows.pfpf_edh import pfpf_edh, _log_gaussian


class TestLogGaussian:
    """Tests for log-Gaussian density computation."""

    def test_correct_log_density(self):
        """Log density should match analytical formula."""
        n = 2
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        cov_inv = np.linalg.inv(cov)
        log_det_cov = np.linalg.slogdet(cov)[1]

        x = np.array([1.5, 2.5])

        result = _log_gaussian(x, mean, cov_inv, log_det_cov, n)

        # Compute expected value manually
        diff = x - mean
        expected = -0.5 * (log_det_cov + n * np.log(2 * np.pi) +
                          diff @ cov_inv @ diff)

        np.testing.assert_allclose(result, expected)


class TestPFPFEDH:
    """Tests for PF-PF EDH filter."""

    def test_output_shape(self, rng, linear_ssm):
        """PF-PF EDH should return correct shapes."""
        m = linear_ssm
        N = 50

        m_filt, P_filt, ess, _, weights = pfpf_edh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=N, n_flow_steps=10, filter_type='ekf', F_jacobian=m['F_jac'], rng=rng
        )

        assert m_filt.shape == (m['T'], 2)
        assert P_filt.shape == (m['T'], 2, 2)
        assert ess.shape == (m['T'],)
        assert weights.shape == (m['T'], N)

    def test_no_nan(self, rng, linear_ssm):
        """PF-PF EDH should not produce NaN."""
        m = linear_ssm

        m_filt, P_filt, ess, _, _ = pfpf_edh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=50, n_flow_steps=10, filter_type='ekf', F_jacobian=m['F_jac'], rng=rng
        )

        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert not np.any(np.isnan(ess))

    def test_weights_sum_to_one(self, rng, linear_ssm):
        """Weights should sum to 1 at each time step."""
        m = linear_ssm

        _, _, _, _, weights = pfpf_edh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=50, n_flow_steps=10, filter_type='ekf', F_jacobian=m['F_jac'], rng=rng
        )

        for t in range(m['T']):
            np.testing.assert_allclose(np.sum(weights[t]), 1.0, rtol=1e-10)

    def test_ess_reasonable(self, rng, linear_ssm):
        """ESS should be in valid range."""
        m = linear_ssm
        N = 100

        _, _, ess, _, _ = pfpf_edh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=N, n_flow_steps=15, filter_type='ekf', F_jacobian=m['F_jac'], rng=rng
        )

        assert np.all(ess >= 1)
        assert np.all(ess <= N)

    def test_custom_lambda_schedule(self, rng, linear_ssm):
        """Custom lambda schedule should work."""
        m = linear_ssm
        lambda_schedule = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

        m_filt, _, _, _, _ = pfpf_edh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=50, lambda_schedule=lambda_schedule,
            filter_type='ekf', F_jacobian=m['F_jac'], rng=rng
        )

        assert not np.any(np.isnan(m_filt))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
