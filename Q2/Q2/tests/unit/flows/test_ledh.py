"""Unit tests for TensorFlow Local Exact Daum-Huang (LEDH) particle flow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.flows import LEDHFlow, compute_ledh_matrices
from src.filters import FilterResult
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def ledh_ssm():
    """Linear SSM for LEDH tests."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


class TestComputeLEDHMatrices:
    """Tests for LEDH matrix computation."""

    def test_per_particle_matrices(self, ledh_ssm):
        """LEDH matrices should be computed at particle location."""
        ssm = ledh_ssm
        x_i = tf.constant([3.5, 2.5], dtype=DTYPE)
        lam = tf.constant(0.5, dtype=DTYPE)
        y = tf.constant([1.0, 0.5], dtype=DTYPE)
        R_inv = tf.linalg.inv(ssm.R)

        A_i, b_i = compute_ledh_matrices(
            x_i, ssm.m0, ssm.P0, ssm.h, ssm.H_jac, ssm.R, y, lam, R_inv)

        assert A_i.shape == (2, 2)
        assert b_i.shape == (2,)
        assert not np.any(np.isnan(A_i.numpy()))
        assert not np.any(np.isnan(b_i.numpy()))

    @pytest.mark.parametrize("lam_val", [0.0, 0.5, 1.0])
    def test_numerical_stability(self, ledh_ssm, lam_val):
        """Matrices should be stable for various lambda values."""
        ssm = ledh_ssm
        x_i = tf.constant([1.0, 0.5], dtype=DTYPE)
        lam = tf.constant(lam_val, dtype=DTYPE)
        y = tf.constant([1.0, 0.5], dtype=DTYPE)
        R_inv = tf.linalg.inv(ssm.R)

        A_i, b_i = compute_ledh_matrices(
            x_i, ssm.m0, ssm.P0, ssm.h, ssm.H_jac, ssm.R, y, lam, R_inv)

        assert np.all(np.isfinite(A_i.numpy())), f"Non-finite at lambda={lam_val}"
        assert np.all(np.isfinite(b_i.numpy())), f"Non-finite at lambda={lam_val}"


class TestLEDHFlow:
    """Tests for Local EDH flow filter."""

    def test_output_shapes_and_finite(self, rng, ledh_ssm, tf_rng):
        """LEDH filter should return FilterResult with correct shapes and no NaN."""
        ssm = ledh_ssm
        T = 5
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        ledh = LEDHFlow(n_particles=30, n_flow_steps=3)
        res = ledh.filter(ssm, ys_tf, rng=tf_rng)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert 'ess' in res.diagnostics
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))

    def test_comparison_with_edh(self, rng, ledh_ssm):
        """For linear model, LEDH and EDH posterior means should be similar."""
        from src.flows import EDHFlow

        ssm = ledh_ssm
        T = 5
        xs, ys = ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        rng_edh = tf.random.Generator.from_seed(42)
        rng_ledh = tf.random.Generator.from_seed(42)

        edh = EDHFlow(n_particles=50, n_flow_steps=5)
        ledh = LEDHFlow(n_particles=50, n_flow_steps=5)

        res_edh = edh.filter(ssm, ys_tf, rng=rng_edh)
        res_ledh = ledh.filter(ssm, ys_tf, rng=rng_ledh)

        # Posterior means should be in the same ballpark
        rmse_edh = np.sqrt(np.mean((res_edh.m_filt.numpy() - xs) ** 2))
        rmse_ledh = np.sqrt(np.mean((res_ledh.m_filt.numpy() - xs) ** 2))
        assert rmse_ledh < rmse_edh * 5.0  # LEDH should not be catastrophically worse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
