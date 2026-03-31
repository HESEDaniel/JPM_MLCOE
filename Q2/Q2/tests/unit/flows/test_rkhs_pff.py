"""Unit tests for TensorFlow RKHS Particle Flow Filter implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.flows import RKHSFlow, localization_matrix
from src.filters import FilterResult
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def rkhs_ssm():
    """Linear SSM for RKHS PFF tests."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


class TestLocalizationMatrix:
    """Tests for localization matrix computation."""

    def test_shape(self):
        """Localization matrix should have correct shape."""
        C = localization_matrix(10, r_in=3.0)
        assert C.shape == (10, 10)

    def test_symmetric(self):
        """Localization matrix should be symmetric."""
        C = localization_matrix(10, r_in=3.0)
        np.testing.assert_allclose(C.numpy(), tf.transpose(C).numpy(), rtol=1e-10)

    def test_diagonal_ones(self):
        """Localization matrix should have ones on diagonal."""
        C = localization_matrix(10, r_in=4.0)
        np.testing.assert_allclose(
            tf.linalg.diag_part(C).numpy(), np.ones(10), rtol=1e-10)

    def test_entries_between_zero_and_one(self):
        """All entries should be in [0, 1]."""
        C = localization_matrix(8, r_in=3.0)
        C_np = C.numpy()
        assert np.all(C_np >= 0.0)
        assert np.all(C_np <= 1.0 + 1e-10)


class TestRKHSFlow:
    """Tests for RKHS PFF filter."""

    def test_output_shapes_and_finite(self, rng, rkhs_ssm, tf_rng):
        """RKHS PFF filter should return FilterResult with correct shapes and no NaN."""
        ssm = rkhs_ssm
        T = 10
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        flow = RKHSFlow(n_particles=50, n_flow_steps=5, step_size=0.1)
        res = flow.filter(ssm, ys_tf, rng=tf_rng)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert 'ess' in res.diagnostics
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))

    @pytest.mark.xfail(reason="RKHS scalar kernel may produce NaN for small systems")
    def test_scalar_kernel(self, rng, rkhs_ssm, tf_rng):
        """RKHS PFF should work with scalar kernel type."""
        ssm = rkhs_ssm
        T = 5
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        flow = RKHSFlow(
            n_particles=50, n_flow_steps=3, step_size=0.05,
            kernel_type='scalar')
        res = flow.filter(ssm, ys_tf, rng=tf_rng)

        assert res.m_filt.shape == (T, 2)
        assert not np.any(np.isnan(res.m_filt.numpy()))

    def test_matrix_valued_kernel(self, rng, rkhs_ssm, tf_rng):
        """RKHS PFF should work with matrix-valued kernel type."""
        ssm = rkhs_ssm
        T = 5
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        flow = RKHSFlow(
            n_particles=50, n_flow_steps=3, step_size=0.05,
            kernel_type='matrix-valued')
        res = flow.filter(ssm, ys_tf, rng=tf_rng)

        assert res.m_filt.shape == (T, 2)
        assert not np.any(np.isnan(res.m_filt.numpy()))

    def test_ess_array(self, rng, rkhs_ssm, tf_rng):
        """ESS should be recorded at each time step (equal to N for RKHS)."""
        ssm = rkhs_ssm
        T = 5
        N = 40
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        flow = RKHSFlow(n_particles=N, n_flow_steps=3, step_size=0.1)
        res = flow.filter(ssm, ys_tf, rng=tf_rng)

        ess = res.diagnostics['ess'].numpy()
        assert ess.shape == (T,)
        # RKHS uses equal weights => ESS = N
        np.testing.assert_allclose(ess, N, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
