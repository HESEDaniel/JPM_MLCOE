"""Unit tests for Stochastic Particle Flow (Dai 2022)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.flows import StochasticFlow, solve_optimal_homotopy
from src.filters import FilterResult
from src.ssm import LinearGaussianSSM

DTYPE = tf.float64


@pytest.fixture
def sf_ssm():
    """2D linear SSM for stochastic flow tests."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    B = np.array([[0.1, 0.0], [0.0, 0.1]])
    C = np.eye(2)
    D = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma = np.eye(2)
    return LinearGaussianSSM(A, B, C, D, Sigma)


class TestSolveOptimalHomotopy:
    """Tests for BVP solver producing beta*(lambda)."""

    @pytest.fixture
    def homotopy_inputs(self):
        P_prior_inv = tf.eye(2, dtype=DTYPE)
        neg_hess = tf.constant([[5.0, 0.0], [0.0, 5.0]], dtype=DTYPE)
        return P_prior_inv, neg_hess

    def test_boundary_conditions(self, homotopy_inputs):
        """beta(0) ~ 0 and beta(1) ~ 1."""
        beta, bdot, lam = solve_optimal_homotopy(*homotopy_inputs, n_grid=20)
        np.testing.assert_allclose(beta[0].numpy(), 0.0, atol=1e-4)
        np.testing.assert_allclose(beta[-1].numpy(), 1.0, atol=1e-2)

    def test_output_shapes(self, homotopy_inputs):
        n_grid = 15
        beta, bdot, lam = solve_optimal_homotopy(*homotopy_inputs, n_grid=n_grid)
        assert beta.shape == (n_grid,)
        assert bdot.shape == (n_grid,)
        assert lam.shape == (n_grid,)

    def test_monotonicity(self, homotopy_inputs):
        """beta should be monotonically non-decreasing (0 -> 1)."""
        beta, _, _ = solve_optimal_homotopy(*homotopy_inputs, n_grid=30)
        diffs = np.diff(beta.numpy())
        assert np.all(diffs >= -1e-6)

    def test_lambda_grid_endpoints(self, homotopy_inputs):
        _, _, lam = solve_optimal_homotopy(*homotopy_inputs, n_grid=10)
        np.testing.assert_allclose(lam[0].numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(lam[-1].numpy(), 1.0, atol=1e-12)


class TestStochasticFlow:
    """Tests for StochasticFlow filter."""

    def test_output_shapes_and_finite(self, rng, sf_ssm, tf_rng):
        """Verify correct output shapes and no NaN."""
        T = 10
        xs, ys = sf_ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        Q_diff = tf.eye(2, dtype=DTYPE) * 0.5
        sf = StochasticFlow(n_particles=30, n_flow_steps=5, Q_diff=Q_diff)
        res = sf.filter(sf_ssm, ys_tf, rng=tf_rng)

        assert isinstance(res, FilterResult)
        assert res.m_filt.shape == (T, 2)
        assert res.P_filt.shape == (T, 2, 2)
        assert res.diagnostics['ess'].shape == (T,)
        assert np.all(np.isfinite(res.m_filt.numpy()))
        assert np.all(np.isfinite(res.P_filt.numpy()))

    def test_with_optimal_homotopy(self, rng, sf_ssm, tf_rng):
        """StochasticFlow with pre-solved beta*(lambda) should run without error."""
        T = 8
        xs, ys = sf_ssm.simulate(T, rng)
        ys_tf = tf.constant(ys, dtype=DTYPE)

        P_inv = tf.linalg.inv(sf_ssm.P0)
        H = sf_ssm.C
        R_inv = tf.linalg.inv(sf_ssm.R)
        neg_hess = tf.transpose(H) @ R_inv @ H
        beta, bdot, lam = solve_optimal_homotopy(P_inv, neg_hess, n_grid=10)

        Q_diff = tf.eye(2, dtype=DTYPE) * 0.3
        sf = StochasticFlow(
            n_particles=30, Q_diff=Q_diff,
            beta_schedule=beta, beta_dot_schedule=bdot,
            lambda_schedule=lam)
        res = sf.filter(sf_ssm, ys_tf, rng=tf_rng)

        assert not np.any(np.isnan(res.m_filt.numpy()))



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
