"""Integration tests for Stochastic Volatility pipeline."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ssm import SVLogTransformed, SVAdditiveNoise
from src.filters import extended_kalman_filter, unscented_kalman_filter, particle_filter
from src.flows import pfpf_edh, pfpf_ledh


@pytest.fixture
def sv_data(rng):
    """Generate stochastic volatility data."""
    T = 100
    model = SVLogTransformed(alpha=0.91, sigma=1.0, beta=0.5)
    xs, ys = model.simulate(T, rng)

    return {
        'model': model,
        'xs': xs,
        'ys': ys,
        'T': T
    }


@pytest.fixture
def sv_additive_data(rng):
    """Generate additive noise SV data."""
    T = 100
    model = SVAdditiveNoise(alpha=0.91, sigma=1.0, beta=0.5, obs_std=0.5)
    xs, ys = model.simulate(T, rng)

    return {
        'model': model,
        'xs': xs,
        'ys': ys,
        'T': T
    }


class TestEKFOnSV:
    """Test EKF on log-transformed stochastic volatility."""

    def test_ekf_on_sv_log_transformed(self, sv_data):
        """EKF should track log-transformed SV model."""
        d = sv_data
        model = d['model']

        # Transform observations
        zs = model.transform_obs(d['ys'])

        m_filt, P_filt, cond_nums = extended_kalman_filter(
            model.f, model.h, model.F_jac, model.H_jac,
            model.Q, model.R, model.m0, model.P0, zs
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))

        # Check correlation with true states
        corr = np.corrcoef(m_filt[:, 0], d['xs'])[0, 1]
        assert corr > 0.3, f"Correlation too low: {corr}"


class TestUKFOnSV:
    """Test UKF on log-transformed stochastic volatility."""

    def test_ukf_on_sv_log_transformed(self, sv_data):
        """UKF should track log-transformed SV model."""
        d = sv_data
        model = d['model']

        # Transform observations
        zs = model.transform_obs(d['ys'])

        m_filt, P_filt, cond_nums = unscented_kalman_filter(
            model.f, model.h, model.Q, model.R, model.m0, model.P0, zs
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))

        # Check correlation with true states
        corr = np.corrcoef(m_filt[:, 0], d['xs'])[0, 1]
        assert corr > 0.3, f"Correlation too low: {corr}"


class TestPFOnSV:
    """Test Particle Filter on stochastic volatility."""

    def test_pf_on_sv_original(self, rng, sv_data):
        """PF should track original SV model using original observations."""
        d = sv_data
        model = d['model']

        # Use 2D observations for PF interface
        ys_2d = d['ys'].reshape(-1, 1)

        def f(x):
            return model.alpha * x

        m_filt, P_filt, ess, _ = particle_filter(
            f=f, h=lambda x: x,
            Q_sampler=model.Q_sampler, log_likelihood=model.log_likelihood,
            m0=model.m0, P0=model.P0, ys=ys_2d,
            N_particles=500, rng=rng
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert np.all(ess >= 1)

        # Check correlation with true states
        corr = np.corrcoef(m_filt[:, 0], d['xs'])[0, 1]
        assert corr > 0.2, f"Correlation too low: {corr}"


class TestPFPFEDHOnSV:
    """Test PF-PF EDH on stochastic volatility."""

    def test_pfpf_edh_on_sv(self, rng, sv_additive_data):
        """PF-PF EDH should track additive noise SV model."""
        d = sv_additive_data
        model = d['model']

        # Use 2D observations
        ys_2d = d['ys'].reshape(-1, 1)

        m_filt, P_filt, ess, _, _ = pfpf_edh(
            model.f, model.h, model.H_jac, model.Q, model.R,
            model.m0, model.P0, ys_2d,
            N_particles=200, n_flow_steps=15,
            filter_type='ekf', F_jacobian=model.F_jac, rng=rng
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert np.all(ess >= 1)


class TestPFPFLEDHOnSV:
    """Test PF-PF LEDH on stochastic volatility."""

    def test_pfpf_ledh_on_sv(self, rng, sv_additive_data):
        """PF-PF LEDH should track additive noise SV model."""
        d = sv_additive_data
        model = d['model']

        # Use 2D observations
        ys_2d = d['ys'].reshape(-1, 1)

        m_filt, P_filt, ess, _, _ = pfpf_ledh(
            model.f, model.h, model.H_jac, model.Q, model.R,
            model.m0, model.P0, ys_2d,
            N_particles=200, n_flow_steps=10,
            filter_type='ekf', F_jacobian=model.F_jac, rng=rng
        )

        # Check outputs are valid
        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert np.all(ess >= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
