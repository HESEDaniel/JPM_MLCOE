"""Unit tests for TensorFlow Exact Daum-Huang (EDH) particle flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.flows import EDHFlow, compute_edh_matrices
from src.filters import FilterResult
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def edh_ssm():
    """Linear SSM for EDH tests."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


class TestComputeEDHMatrices:
    """Tests for EDH matrix computation."""

    def test_A_matrix_shape(self, edh_ssm):
        """A matrix should have correct shape."""
        ssm = edh_ssm
        n_x = 2
        lam = tf.constant(0.5, dtype=DTYPE)
        m = ssm.m0
        P = ssm.P0
        H = ssm.C
        R = ssm.R
        y = tf.constant([1.0, 0.5], dtype=DTYPE)

        A, b = compute_edh_matrices(m, P, H, R, y, lam, m, ssm.h)

        assert A.shape == (n_x, n_x)

    def test_b_vector_shape(self, edh_ssm):
        """b vector should have correct shape."""
        ssm = edh_ssm
        n_x = 2
        lam = tf.constant(0.5, dtype=DTYPE)
        y = tf.constant([1.0, 0.5], dtype=DTYPE)

        A, b = compute_edh_matrices(
            ssm.m0, ssm.P0, ssm.C, ssm.R, y, lam, ssm.m0, ssm.h)

        assert b.shape == (n_x,)

    @pytest.mark.parametrize("lam_val", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_numerical_stability(self, edh_ssm, lam_val):
        """Matrices should be stable for various lambda values."""
        ssm = edh_ssm
        lam = tf.constant(lam_val, dtype=DTYPE)
        y = tf.constant([1.0, 0.5], dtype=DTYPE)

        A, b = compute_edh_matrices(
            ssm.m0, ssm.P0, ssm.C, ssm.R, y, lam, ssm.m0, ssm.h)

        assert np.all(np.isfinite(A.numpy())), f"Non-finite in A at lambda={lam_val}"
        assert np.all(np.isfinite(b.numpy())), f"Non-finite in b at lambda={lam_val}"


class TestEDHFlow:
    """Tests for EDH flow filter."""

    def test_output_shapes_and_finite(self, rng, edh_ssm, tf_rng):
        """EDH filter should return FilterResult with correct shapes and no NaN."""
        ssm = edh_ssm
        T = 10
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        edh = EDHFlow(n_particles=50, n_flow_steps=5)
        res = edh.filter(ssm, ys_tf, rng=tf_rng)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert 'ess' in res.diagnostics
        assert res.diagnostics['ess'].shape == (T,)
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))

    def test_custom_lambda_schedule(self, rng, edh_ssm, tf_rng):
        """Custom lambda schedule should work."""
        ssm = edh_ssm
        T = 5
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        schedule = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        edh = EDHFlow(n_particles=30, lambda_schedule=schedule)
        res = edh.filter(ssm, ys_tf, rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
