"""Integration tests for Stochastic Volatility pipeline (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm import SVLogTransformed, SVAdditiveNoise
from src.filters import ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter
from src.flows import PFPFEDHFilter, PFPFLEDHFilter

DTYPE = tf.float64


@pytest.fixture
def sv_data(rng):
    """Generate stochastic volatility data (log-transformed)."""
    T = 50
    model = SVLogTransformed(alpha=0.91, sigma=1.0, beta=0.5)
    xs, ys = model.simulate(T, rng)
    return {'model': model, 'xs': xs, 'ys': ys, 'T': T}


@pytest.fixture
def sv_additive_data(rng):
    """Generate additive noise SV data."""
    T = 50
    model = SVAdditiveNoise(alpha=0.91, sigma=1.0, beta=0.5, obs_std=0.5)
    xs, ys = model.simulate(T, rng)
    return {'model': model, 'xs': xs, 'ys': ys, 'T': T}


class TestEKFOnSV:
    """Test EKF on log-transformed stochastic volatility."""

    def test_ekf_on_sv_log_transformed(self, sv_data):
        """EKF should track log-transformed SV model."""
        d = sv_data
        model = d['model']
        zs = model.transform_obs(d['ys'])
        zs_tf = tf.constant(zs, dtype=DTYPE)

        ekf = ExtendedKalmanFilter(joseph=True)
        res = ekf.filter(model, zs_tf)

        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))

        # Check correlation with true states
        m_np = res.m_filt.numpy()[:, 0]
        corr = np.corrcoef(m_np, d['xs'])[0, 1]
        assert corr > 0.2, f"Correlation too low: {corr}"


class TestUKFOnSV:
    """Test UKF on log-transformed stochastic volatility."""

    def test_ukf_on_sv_log_transformed(self, sv_data):
        """UKF should track log-transformed SV model."""
        d = sv_data
        model = d['model']
        zs = model.transform_obs(d['ys'])
        zs_tf = tf.constant(zs, dtype=DTYPE)

        ukf = UnscentedKalmanFilter()
        res = ukf.filter(model, zs_tf)

        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))

        m_np = res.m_filt.numpy()[:, 0]
        corr = np.corrcoef(m_np, d['xs'])[0, 1]
        assert corr > 0.2, f"Correlation too low: {corr}"


class TestPFOnSV:
    """Test Particle Filter on stochastic volatility."""

    def test_pf_on_sv(self, sv_data):
        """PF should track SV model."""
        d = sv_data
        model = d['model']

        # PF uses original observations as 2D
        ys_2d = d['ys'].reshape(-1, 1)
        ys_tf = tf.constant(ys_2d, dtype=DTYPE)

        pf = ParticleFilter(n_particles=300, resample_threshold=0.5)
        tf_rng = tf.random.Generator.from_seed(42)
        res = pf.filter(model, ys_tf, rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))

        ess = res.diagnostics['ess'].numpy()
        assert np.all(ess >= 1)

        m_np = res.m_filt.numpy()[:, 0]
        corr = np.corrcoef(m_np, d['xs'])[0, 1]
        assert corr > 0.1, f"Correlation too low: {corr}"


class TestPFPFEDHOnSV:
    """Test PF-PF EDH on stochastic volatility."""

    def test_pfpf_edh_on_sv(self, sv_additive_data):
        """PF-PF EDH should track additive noise SV model."""
        d = sv_additive_data
        model = d['model']
        ys_2d = d['ys'].reshape(-1, 1)
        ys_tf = tf.constant(ys_2d, dtype=DTYPE)

        flow = PFPFEDHFilter(n_particles=100, n_flow_steps=10)
        tf_rng = tf.random.Generator.from_seed(42)
        res = flow.filter(model, ys_tf, rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))

        ess = res.diagnostics['ess'].numpy()
        assert np.all(ess >= 1.0)


class TestPFPFLEDHOnSV:
    """Test PF-PF LEDH on stochastic volatility."""

    def test_pfpf_ledh_on_sv(self, sv_additive_data):
        """PF-PF LEDH should track additive noise SV model."""
        d = sv_additive_data
        model = d['model']
        ys_2d = d['ys'].reshape(-1, 1)
        ys_tf = tf.constant(ys_2d, dtype=DTYPE)

        flow = PFPFLEDHFilter(n_particles=100, n_flow_steps=5)
        tf_rng = tf.random.Generator.from_seed(42)
        res = flow.filter(model, ys_tf, rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))
        assert not np.any(np.isnan(res.P_filt.numpy()))

        ess = res.diagnostics['ess'].numpy()
        assert np.all(ess >= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
