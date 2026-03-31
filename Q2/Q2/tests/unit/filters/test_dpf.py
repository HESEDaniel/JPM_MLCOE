"""Unit tests for DifferentiableParticleFilter."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.filters import DifferentiableParticleFilter, FilterResult
from src.resampling import SinkhornOTResampler, SoftResampler, MultinomialResampler
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def dpf_ssm():
    """Simple 2D linear SSM for DPF testing."""
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    B = np.sqrt(0.1) * np.eye(2)
    C = np.eye(2)
    D = np.sqrt(0.1) * np.eye(2)
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


@pytest.fixture
def dpf_data(rng, dpf_ssm):
    T = 15
    xs, ys = dpf_ssm.simulate(T, rng)
    ys_tf = tf.constant(ys, dtype=DTYPE)
    return {'ssm': dpf_ssm, 'xs': xs, 'ys_tf': ys_tf, 'T': T}


class TestDPFOutputs:
    """Tests for DPF output correctness."""

    def test_output_shapes_and_finite(self, dpf_data, tf_rng):
        """Verify correct output shapes, no NaN, and valid ESS bounds."""
        d = dpf_data
        N = 50
        dpf = DifferentiableParticleFilter(n_particles=N)
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (d['T'], 2)
        assert res.P_filt.shape == (d['T'], 2, 2)
        assert 'ess' in res.diagnostics
        assert res.diagnostics['ess'].shape == (d['T'],)
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))
        ess = res.diagnostics['ess'].numpy()
        assert np.all(np.isfinite(ess))
        assert np.all(ess >= 1.0)
        assert np.all(ess <= N + 1e-6)

    def test_tracks_signal(self, dpf_data, tf_rng):
        """DPF should produce estimates closer to truth than prior."""
        d = dpf_data
        dpf = DifferentiableParticleFilter(n_particles=200)
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

        mse_filter = np.mean((res.m_filt.numpy() - d['xs']) ** 2)
        mse_prior = np.mean(d['xs'] ** 2)
        assert mse_filter < mse_prior


class TestDPFResamplers:
    """Test DPF with different resampling strategies."""

    def test_with_ot_resampler(self, dpf_data, tf_rng):
        d = dpf_data
        dpf = DifferentiableParticleFilter(
            n_particles=50, resampler=SinkhornOTResampler(epsilon=0.5))
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=tf_rng)
        assert not np.any(np.isnan(res.m_filt.numpy()))

    def test_with_soft_resampler(self, dpf_data, tf_rng):
        d = dpf_data
        dpf = DifferentiableParticleFilter(
            n_particles=50, resampler=SoftResampler(alpha=0.5))
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=tf_rng)
        assert not np.any(np.isnan(res.m_filt.numpy()))

    def test_with_multinomial_resampler(self, dpf_data, tf_rng):
        d = dpf_data
        dpf = DifferentiableParticleFilter(
            n_particles=50, resampler=MultinomialResampler())
        res = dpf.filter(d['ssm'], d['ys_tf'], rng=tf_rng)
        assert not np.any(np.isnan(res.m_filt.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
