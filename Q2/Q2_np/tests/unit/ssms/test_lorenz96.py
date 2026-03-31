"""Unit tests for Lorenz 96 SSM."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ssm import lorenz96_ssm, lorenz96_step, lorenz96_rhs


class TestLorenz96RHS:
    """Tests for Lorenz 96 ODE right-hand side."""

    def test_output_shape(self):
        """RHS should return same shape as input."""
        K = 40
        x = np.ones(K)
        F = 8.0

        dx = lorenz96_rhs(x, F)

        assert dx.shape == (K,)

    def test_cyclic_boundary(self):
        """RHS should handle cyclic boundary correctly."""
        K = 5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        F = 8.0

        dx = lorenz96_rhs(x, F)

        # dx[0] = (x[1] - x[-2]) * x[-1] - x[0] + F
        #       = (x[1] - x[3]) * x[4] - x[0] + F
        #       = (2 - 4) * 5 - 1 + 8 = -10 - 1 + 8 = -3
        expected_dx0 = (x[1] - x[3]) * x[4] - x[0] + F
        np.testing.assert_allclose(dx[0], expected_dx0)


class TestLorenz96Step:
    """Tests for Lorenz 96 RK4 integration step."""

    def test_output_shape(self):
        """Step should return same shape as input."""
        K = 40
        x = np.random.randn(K)
        F = 8.0
        dt = 0.05

        x_next = lorenz96_step(x, F, dt)

        assert x_next.shape == (K,)


class TestLorenz96SSM:
    """Tests for Lorenz 96 state space model."""

    def test_output_shapes(self, rng):
        """Generated data should have correct shapes."""
        T = 50
        K = 40

        xs, ys, H, Q, R = lorenz96_ssm(T, rng, K=K)

        assert xs.shape == (T, K)
        assert ys.shape == (T, K)  # Default obs_every=1
        assert H.shape == (K, K)
        assert Q.shape == (K, K)
        assert R.shape == (K, K)

    def test_H_matrix_structure(self, rng):
        """H should select observed variables."""
        T = 20
        K = 20
        obs_every = 2

        xs, ys, H, Q, R = lorenz96_ssm(T, rng, K=K, obs_every=obs_every)

        n_obs = K // obs_every
        assert H.shape == (n_obs, K)
        assert ys.shape == (T, n_obs)

        # H should be sparse with ones
        assert np.sum(H) == n_obs

    def test_no_nan(self, rng):
        """Generated data should not contain NaN."""
        T = 100
        K = 40

        xs, ys, H, Q, R = lorenz96_ssm(T, rng, K=K)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
