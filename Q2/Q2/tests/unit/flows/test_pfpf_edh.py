"""Unit tests for TensorFlow PF-PF EDH implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.flows import PFPFEDHFilter
from src.filters import FilterResult
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def pfpf_ssm():
    """Linear SSM for PFPF EDH tests."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


@pytest.fixture
def pfpf_data(rng, pfpf_ssm):
    """Generate data for PFPF tests."""
    T = 10
    xs, ys = pfpf_ssm.simulate(T, rng)
    ys_tf = tf.constant(ys, dtype=DTYPE)
    return {'ssm': pfpf_ssm, 'xs': xs, 'ys_tf': ys_tf, 'T': T}


class TestPFPFEDHFilter:
    """Tests for PF-PF EDH filter."""

    def test_output_shapes_and_finite(self, pfpf_data, tf_rng):
        """PF-PF EDH should return correct shapes, no NaN, valid ESS."""
        d = pfpf_data
        N = 50

        flow = PFPFEDHFilter(n_particles=N, n_flow_steps=5)
        res = flow.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

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

    def test_custom_lambda_schedule(self, pfpf_data, tf_rng):
        """Custom lambda schedule should work."""
        d = pfpf_data
        schedule = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

        flow = PFPFEDHFilter(n_particles=30, lambda_schedule=schedule)
        res = flow.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))

    def test_resample_count(self, pfpf_data, tf_rng):
        """Resample count should be a non-negative integer."""
        d = pfpf_data

        flow = PFPFEDHFilter(n_particles=30, n_flow_steps=5, resample_threshold=0.5)
        res = flow.filter(d['ssm'], d['ys_tf'], rng=tf_rng)

        assert 'resample_count' in res.diagnostics
        assert res.diagnostics['resample_count'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
