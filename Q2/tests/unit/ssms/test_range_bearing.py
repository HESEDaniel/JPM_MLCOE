"""Unit tests for Range-Bearing SSM."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ssm import RangeBearing


class TestRangeBearingSSM:
    """Unit tests for Range-Bearing SSM."""

    @pytest.fixture
    def model(self):
        return RangeBearing()

    def test_output_shapes(self, rng, model):
        """Generated data should have correct shapes."""
        T = 100
        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 4), f"xs shape: {xs.shape}"
        assert ys.shape == (T, 2), f"ys shape: {ys.shape}"
        assert model.F.shape == (4, 4), f"F shape: {model.F.shape}"
        assert model.Q.shape == (4, 4), f"Q shape: {model.Q.shape}"
        assert model.R.shape == (2, 2), f"R shape: {model.R.shape}"

    def test_no_nan(self, rng, model):
        """Generated data should not contain NaN."""
        T = 100
        xs, ys = model.simulate(T, rng)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_range_positive(self, rng, model):
        """Range observations should be positive."""
        T = 100
        xs, ys = model.simulate(T, rng)

        # Range is in column 0
        assert np.all(ys[:, 0] > 0), "Range should be positive"

    def test_bearing_range(self, rng, model):
        """Bearing should be in [-pi, pi]."""
        T = 100
        xs, ys = model.simulate(T, rng)

        # Bearing is in column 1
        assert np.all(ys[:, 1] >= -np.pi), "Bearing below -pi"
        assert np.all(ys[:, 1] <= np.pi), "Bearing above pi"

    def test_observation_function(self, model):
        """Test h() correctness."""
        # Test at known point (3, 4) -> range=5, bearing=atan2(4,3)
        x = np.array([3.0, 0.0, 4.0, 0.0])
        y = model.h(x)

        assert np.isclose(y[0], 5.0, atol=1e-10), f"Range: {y[0]}"
        assert np.isclose(y[1], np.arctan2(4, 3), atol=1e-10), f"Bearing: {y[1]}"

    def test_jacobian_shape(self, model):
        """Test Jacobian has correct shape."""
        x = np.array([2.0, 0.1, 3.0, 0.1])
        H = model.H_jac(x)

        assert H.shape == (2, 4), f"H shape: {H.shape}"

    def test_jacobian_numerical(self, model):
        """Test Jacobian against numerical differentiation."""
        x = np.array([2.0, 0.1, 3.0, 0.1])
        H = model.H_jac(x)

        eps = 1e-6
        H_num = np.zeros((2, 4))
        for i in range(4):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            H_num[:, i] = (model.h(x_plus) - model.h(x_minus)) / (2 * eps)

        assert np.allclose(H, H_num, atol=1e-5), f"H:\n{H}\nH_num:\n{H_num}"

    def test_log_likelihood_shape(self, rng, model):
        """Log-likelihood should return N values."""
        N = 50
        particles = rng.multivariate_normal(model.m0, model.P0, size=N)
        y = np.array([5.0, 0.5])

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)

    def test_log_likelihood_bearing_wrap(self, rng, model):
        """Log-likelihood should handle bearing angle wrapping."""
        N = 50
        particles = rng.multivariate_normal(model.m0, model.P0, size=N)
        # Bearing near -pi
        y = np.array([5.0, -np.pi + 0.1])

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik))

    def test_Q_sampler_shape(self, rng, model):
        """Q_sampler should return correct shape."""
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
