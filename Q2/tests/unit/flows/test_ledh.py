"""Unit tests for Local Exact Daum-Huang (LEDH) Particle Flow implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.flows.ledh import local_edh_flow, compute_ledh_matrices
from src.flows.edh import exact_daum_huang_flow


class TestComputeLEDHMatrices:
    """Tests for LEDH matrix computation."""

    def test_per_particle_matrices(self, nonlinear_model):
        """LEDH matrices should be computed at particle location."""
        m = nonlinear_model
        x_i = np.array([3.5, 2.5])
        lam = 0.5

        A_i, b_i = compute_ledh_matrices(
            x_i, m['m0'], m['P0'], m['h'], m['H_jac'], m['R'], m['y'], lam
        )

        assert A_i.shape == (2, 2)
        assert b_i.shape == (2,)
        assert not np.any(np.isnan(A_i))
        assert not np.any(np.isnan(b_i))

    def test_precomputed_R_inv(self, linear_model):
        """Precomputed R_inv should give same results."""
        m = linear_model
        x_i = np.array([1.0, 0.5])
        y = np.array([1.0, 0.5])
        lam = 0.5

        R_inv = np.linalg.inv(m['R'])
        I = np.eye(2)

        A_i_no_precompute, b_i_no_precompute = compute_ledh_matrices(
            x_i, m['m0'], m['P0'], m['h'], m['H_jac'], m['R'], y, lam
        )

        A_i_precompute, b_i_precompute = compute_ledh_matrices(
            x_i, m['m0'], m['P0'], m['h'], m['H_jac'], m['R'], y, lam,
            R_inv=R_inv, I=I
        )

        np.testing.assert_allclose(A_i_no_precompute, A_i_precompute, rtol=1e-10)
        np.testing.assert_allclose(b_i_no_precompute, b_i_precompute, rtol=1e-10)


class TestLocalEDHFlow:
    """Tests for Local EDH flow."""

    def test_output_shape(self, rng, nonlinear_model):
        """LEDH should preserve particle shape."""
        m = nonlinear_model
        N = 100
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=N)

        particles_out, log_det_J, _, _, _ = local_edh_flow(
            particles, m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], m['y'],
            n_steps=10, redraw=False, rng=rng, filter_type='ekf'
        )

        assert particles_out.shape == (N, 2)
        assert log_det_J.shape == (N,)

    def test_no_nan(self, rng, nonlinear_model):
        """LEDH should not produce NaN."""
        m = nonlinear_model
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=100)

        particles_out, log_det_J, _, _, _ = local_edh_flow(
            particles, m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], m['y'],
            n_steps=20, redraw=False, rng=rng, filter_type='ekf'
        )

        assert not np.any(np.isnan(particles_out))
        assert not np.any(np.isnan(log_det_J))

    def test_per_particle_jacobian(self, rng, nonlinear_model):
        """Log-det Jacobian should be computed per-particle."""
        m = nonlinear_model
        N = 50
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=N)

        _, log_det_J, _, _, _ = local_edh_flow(
            particles, m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], m['y'],
            n_steps=10, redraw=False, rng=rng, filter_type='ekf'
        )

        # Log-det should vary per particle
        assert len(np.unique(log_det_J.round(6))) > 1

    def test_store_history_false(self, rng, nonlinear_model):
        """store_history=False should return None for history."""
        m = nonlinear_model
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=50)

        _, _, history, _, _ = local_edh_flow(
            particles, m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], m['y'],
            n_steps=10, store_history=False, redraw=False, rng=rng
        )

        assert history is None

    def test_comparison_with_edh(self, rng, linear_model):
        """For linear model, LEDH and EDH should give similar results."""
        m = linear_model
        N = 200
        y = np.array([1.0, 0.5])
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=N)

        # Use same particles for both
        particles_edh, _, _, m_edh, _ = exact_daum_huang_flow(
            particles.copy(), m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], y,
            n_steps=20, redraw=False, rng=np.random.default_rng(42), filter_type='ekf'
        )

        particles_ledh, _, _, m_ledh, _ = local_edh_flow(
            particles.copy(), m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], y,
            n_steps=20, store_history=False, redraw=False, rng=np.random.default_rng(42),
            filter_type='ekf'
        )

        # Posterior means should be close for linear model
        np.testing.assert_allclose(m_edh, m_ledh, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
