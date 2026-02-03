"""Integration tests for Multi-Target Acoustic tracking pipeline."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm.multi_target_acoustic import (
    MultiTargetAcousticModel, multi_target_acoustic_ssm,
    sample_initial_distribution, compute_omat_trajectory
)
from src.flows import pfpf_edh, pfpf_ledh


@pytest.fixture
def multi_target_data(rng):
    """Generate multi-target acoustic tracking data."""
    T = 30
    n_targets = 4

    xs, ys, F, Q_sim, Q_filt, R, sensors = multi_target_acoustic_ssm(
        T, rng, n_targets=n_targets, max_retries_sec=60.0
    )

    m0, P0 = sample_initial_distribution(rng, n_targets)

    model = MultiTargetAcousticModel(n_targets=n_targets)

    return {
        'model': model,
        'xs': xs,
        'ys': ys,
        'T': T,
        'm0': m0,
        'P0': P0,
        'Q_filt': Q_filt,
        'R': R
    }


class TestPFPFEDHOnMultiTarget:
    """Test PF-PF EDH on multi-target tracking."""

    def test_pfpf_edh_on_multi_target(self, rng, multi_target_data):
        """PF-PF EDH should track multi-target acoustic model."""
        d = multi_target_data
        model = d['model']

        m_filt, P_filt, ess, _, _ = pfpf_edh(
            model.f, model.h, model.h_jacobian,
            d['Q_filt'], d['R'], d['m0'], d['P0'], d['ys'],
            N_particles=100, n_flow_steps=10,
            filter_type='ekf', F_jacobian=model.f_jacobian, rng=rng
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert np.all(ess >= 1)


class TestPFPFLEDHOnMultiTarget:
    """Test PF-PF LEDH on multi-target tracking."""

    def test_pfpf_ledh_on_multi_target(self, rng, multi_target_data):
        """PF-PF LEDH should track multi-target acoustic model."""
        d = multi_target_data
        model = d['model']

        m_filt, P_filt, ess, _, _ = pfpf_ledh(
            model.f, model.h, model.h_jacobian,
            d['Q_filt'], d['R'], d['m0'], d['P0'], d['ys'],
            N_particles=100, n_flow_steps=8,
            filter_type='ekf', F_jacobian=model.f_jacobian, rng=rng
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert np.all(ess >= 1)


class TestOMATMetricComputation:
    """Test OMAT metric computation on tracking results."""

    def test_omat_metric_computation(self, rng, multi_target_data):
        """OMAT metric should be computable from tracking results."""
        d = multi_target_data
        model = d['model']

        m_filt, _, _, _, _ = pfpf_edh(
            model.f, model.h, model.h_jacobian,
            d['Q_filt'], d['R'], d['m0'], d['P0'], d['ys'],
            N_particles=100, n_flow_steps=10,
            filter_type='ekf', F_jacobian=model.f_jacobian, rng=rng
        )

        # Compute OMAT trajectory
        omat = compute_omat_trajectory(d['xs'], m_filt, n_targets=4)

        assert omat.shape == (d['T'],)
        assert np.all(omat >= 0)
        assert np.all(np.isfinite(omat))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
