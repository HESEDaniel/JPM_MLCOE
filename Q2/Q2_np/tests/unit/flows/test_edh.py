"""Unit tests for Exact Daum-Huang (EDH) Particle Flow implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.flows.edh import exact_daum_huang_flow, compute_edh_matrices


class TestComputeEDHMatrices:
    """Tests for EDH matrix computation."""

    def test_A_matrix_shape(self, linear_model):
        """A matrix should have correct shape."""
        m = linear_model
        n_x = 2
        lam = 0.5
        y = np.array([1.0, 0.5])

        A, b = compute_edh_matrices(
            m['m0'], m['P0'], m['H'], m['R'], y, lam, m['m0'], m['h']
        )

        assert A.shape == (n_x, n_x)

    def test_b_vector_shape(self, linear_model):
        """b vector should have correct shape."""
        m = linear_model
        n_x = 2
        lam = 0.5
        y = np.array([1.0, 0.5])

        A, b = compute_edh_matrices(
            m['m0'], m['P0'], m['H'], m['R'], y, lam, m['m0'], m['h']
        )

        assert b.shape == (n_x,)

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
    def test_numerical_stability(self, linear_model, lam):
        """Matrices should be stable for various lambda values."""
        m = linear_model
        y = np.array([1.0, 0.5])

        A, b = compute_edh_matrices(
            m['m0'], m['P0'], m['H'], m['R'], y, lam, m['m0'], m['h']
        )

        assert np.all(np.isfinite(A)), f"Non-finite in A at lambda={lam}"
        assert np.all(np.isfinite(b)), f"Non-finite in b at lambda={lam}"


class TestExactDaumHuangFlow:
    """Tests for EDH flow."""

    def test_output_shape(self, rng, linear_model):
        """Flow should preserve particle shape."""
        m = linear_model
        N = 100
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=N)
        y = np.array([1.0, 0.5])

        particles_out, log_det_J, history, _, _ = exact_daum_huang_flow(
            particles, m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], y,
            n_steps=10, redraw=False, rng=rng, filter_type='ekf'
        )

        assert particles_out.shape == (N, 2)
        assert log_det_J.shape == (N,)

    def test_no_nan(self, rng, linear_model):
        """Flow should not produce NaN."""
        m = linear_model
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=100)
        y = np.array([1.0, 0.5])

        particles_out, log_det_J, _, _, _ = exact_daum_huang_flow(
            particles, m['m0'], m['P0'], m['f'], m['F_jac'], m['Q'],
            m['h'], m['H_jac'], m['R'], y,
            n_steps=20, redraw=False, rng=rng, filter_type='ekf'
        )

        assert not np.any(np.isnan(particles_out))
        assert not np.any(np.isnan(log_det_J))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
