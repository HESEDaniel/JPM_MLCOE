"""Unit tests for TensorFlow Particle Filter implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.filters import ParticleFilter, FilterResult, systematic_resample
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


class TestSystematicResample:
    """Tests for systematic resampling algorithm."""

    def test_preserves_count(self, tf_rng):
        """Resampling should preserve particle count."""
        N = 100
        log_w = tf.constant(np.random.default_rng(42).standard_normal(N), dtype=DTYPE)

        indices = systematic_resample(log_w, tf_rng)

        assert indices.shape == (N,)
        assert tf.reduce_all(indices >= 0).numpy()
        assert tf.reduce_all(indices < N).numpy()

    def test_uniform_weights_give_all_indices(self, tf_rng):
        """Uniform weights should (roughly) select all indices."""
        N = 50
        log_w = tf.zeros(N, dtype=DTYPE)

        indices = systematic_resample(log_w, tf_rng)

        assert indices.shape == (N,)
        # Most indices should be unique for uniform weights
        unique = tf.unique(indices).y
        assert len(unique) >= N * 0.8

    def test_concentrated_weight(self, tf_rng):
        """A single dominant weight should be selected often."""
        N = 100
        log_w = tf.constant([-100.0] * N, dtype=DTYPE)
        log_w = tf.tensor_scatter_nd_update(log_w, [[0]], [0.0])

        indices = systematic_resample(log_w, tf_rng)

        # Index 0 should dominate
        count_0 = tf.reduce_sum(tf.cast(indices == 0, tf.int32))
        assert count_0.numpy() > N * 0.5


class TestParticleFilter:
    """Tests for Particle Filter."""

    @pytest.fixture
    def pf_ssm(self):
        """Simple linear SSM for PF testing."""
        A = np.array([[0.9, 0.0], [0.0, 0.9]])
        B = np.sqrt(0.1) * np.eye(2)
        C = np.eye(2)
        D = np.sqrt(0.1) * np.eye(2)
        Sigma = np.eye(2)
        return LinearGaussianSSM(A, B, C, D, Sigma)

    def test_output_shapes_and_finite(self, rng, pf_ssm, tf_rng):
        """Verify correct output shapes, no NaN, and valid ESS bounds."""
        T, N = 20, 100
        xs, ys = pf_ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        pf = ParticleFilter(n_particles=N)
        res = pf.filter(pf_ssm, ys_tf, rng=tf_rng)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert 'ess' in res.diagnostics
        assert res.diagnostics['ess'].shape == (T,)
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))
        ess = res.diagnostics['ess'].numpy()
        assert np.all(np.isfinite(ess))
        assert np.all(ess >= 1.0)
        assert np.all(ess <= N + 1e-6)

    def test_tracks_signal(self, rng, pf_ssm, tf_rng):
        """PF with enough particles should track the signal."""
        T, N = 30, 500
        xs, ys = pf_ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        pf = ParticleFilter(n_particles=N)
        res = pf.filter(pf_ssm, ys_tf, rng=tf_rng)

        m_np = res.m_filt.numpy()
        mse_filter = np.mean((m_np - xs) ** 2)
        mse_prior = np.mean(xs ** 2)
        assert mse_filter < mse_prior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
