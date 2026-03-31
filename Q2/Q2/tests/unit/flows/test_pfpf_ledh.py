"""Unit tests for TensorFlow PF-PF LEDH implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.flows import PFPFLEDHFilter, PFPFEDHFilter
from src.filters import FilterResult
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def pfpf_ssm():
    """Linear SSM for PFPF LEDH tests."""
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


class TestPFPFLEDHFilter:
    """Tests for PF-PF LEDH filter."""

    def test_output_shapes_and_finite(self, pfpf_data, tf_rng):
        """PF-PF LEDH should return correct shapes, no NaN, valid ESS."""
        d = pfpf_data
        N = 50

        flow = PFPFLEDHFilter(n_particles=N, n_flow_steps=3)
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

    def test_comparison_with_pfpf_edh(self, pfpf_data):
        """For linear model, PFPF LEDH and EDH RMSE should be comparable."""
        d = pfpf_data

        rng_edh = tf.random.Generator.from_seed(42)
        rng_ledh = tf.random.Generator.from_seed(42)

        edh = PFPFEDHFilter(n_particles=50, n_flow_steps=5)
        ledh = PFPFLEDHFilter(n_particles=50, n_flow_steps=5)

        res_edh = edh.filter(d['ssm'], d['ys_tf'], rng=rng_edh)
        res_ledh = ledh.filter(d['ssm'], d['ys_tf'], rng=rng_ledh)

        rmse_edh = np.sqrt(np.mean((res_edh.m_filt.numpy() - d['xs']) ** 2))
        rmse_ledh = np.sqrt(np.mean((res_ledh.m_filt.numpy() - d['xs']) ** 2))

        # LEDH should not be catastrophically worse than EDH
        assert rmse_ledh < rmse_edh * 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
