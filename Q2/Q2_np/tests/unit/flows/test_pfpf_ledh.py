"""Unit tests for PF-PF LEDH implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.flows.pfpf_ledh import pfpf_ledh
from src.flows.pfpf_edh import pfpf_edh


class TestPFPFLEDH:
    """Tests for PF-PF LEDH filter."""

    def test_output_shape(self, rng, linear_ssm):
        """PF-PF LEDH should return correct shapes."""
        m = linear_ssm
        N = 50

        m_filt, P_filt, ess, _, weights = pfpf_ledh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=N, n_flow_steps=10, filter_type='ekf', F_jacobian=m['F_jac'], rng=rng
        )

        assert m_filt.shape == (m['T'], 2)
        assert P_filt.shape == (m['T'], 2, 2)
        assert ess.shape == (m['T'],)
        assert weights.shape == (m['T'], N)

    def test_no_nan(self, rng, linear_ssm):
        """PF-PF LEDH should not produce NaN."""
        m = linear_ssm

        m_filt, P_filt, ess, _, _ = pfpf_ledh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=50, n_flow_steps=10, filter_type='ekf', F_jacobian=m['F_jac'], rng=rng
        )

        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert not np.any(np.isnan(ess))

    def test_comparison_with_pfpf_edh(self, rng, linear_ssm):
        """For linear model, PFPF LEDH and PFPF EDH should give similar results."""
        m = linear_ssm

        m_edh, _, _, _, _ = pfpf_edh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=100, n_flow_steps=15, filter_type='ekf',
            F_jacobian=m['F_jac'], rng=np.random.default_rng(42)
        )

        m_ledh, _, _, _, _ = pfpf_ledh(
            m['f'], m['h'], m['H_jac'], m['Q'], m['R'], m['m0'], m['P0'], m['ys'],
            N_particles=100, n_flow_steps=15, filter_type='ekf',
            F_jacobian=m['F_jac'], rng=np.random.default_rng(42)
        )

        # RMSE should be similar (within a factor of 3)
        rmse_edh = np.sqrt(np.mean((m_edh - m['xs'])**2))
        rmse_ledh = np.sqrt(np.mean((m_ledh - m['xs'])**2))

        assert rmse_ledh < rmse_edh * 3  # LEDH shouldn't be much worse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
