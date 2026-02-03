"""Unit tests for Stochastic Volatility SSM."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ssm import SVLogTransformed, SVAdditiveNoise


class TestSVLogTransformed:
    """Tests for log-transformed stochastic volatility model."""

    def test_output_shapes(self, rng):
        """Generated data should have correct shapes."""
        T = 100
        model = SVLogTransformed()
        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T,), f"xs shape: {xs.shape}"
        assert ys.shape == (T,), f"ys shape: {ys.shape}"

    def test_no_nan(self, rng):
        """Generated data should not contain NaN."""
        T = 100
        model = SVLogTransformed()
        xs, ys = model.simulate(T, rng)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_f_transition(self):
        """f should implement x_k = alpha * x_{k-1}."""
        model = SVLogTransformed(alpha=0.9)
        x = np.array([2.0])

        x_next = model.f(x)

        np.testing.assert_allclose(x_next, np.array([0.9 * 2.0]))

    def test_h_observation(self):
        """h should implement log-transformed observation function."""
        model = SVLogTransformed(beta=0.5)
        x = np.array([1.0])

        y = model.h(x)

        # h(x) = log(beta^2) + x + LOG_CHI2_MEAN
        from src.ssm.stochastic_volatility import LOG_CHI2_MEAN
        expected = np.log(0.5**2) + 1.0 + LOG_CHI2_MEAN
        np.testing.assert_allclose(y[0], expected)

    def test_jacobians(self):
        """Jacobians should have correct values."""
        model = SVLogTransformed(alpha=0.91)
        x = np.array([1.0])

        F = model.F_jac(x)
        H = model.H_jac(x)

        np.testing.assert_allclose(F, np.array([[0.91]]))
        np.testing.assert_allclose(H, np.array([[1.0]]))

    def test_log_likelihood_shape(self, rng):
        """Log-likelihood should return N values for N particles."""
        model = SVLogTransformed()
        N = 50
        particles = rng.standard_normal((N, 1))
        y = np.array([0.5])

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)

    def test_Q_sampler_shape(self, rng):
        """Q_sampler should return correct shape."""
        model = SVLogTransformed(sigma=1.0)
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, 1)


class TestSVAdditiveNoise:
    """Tests for additive noise stochastic volatility model."""

    def test_output_shapes(self, rng):
        """Generated data should have correct shapes."""
        T = 100
        model = SVAdditiveNoise()
        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T,), f"xs shape: {xs.shape}"
        assert ys.shape == (T,), f"ys shape: {ys.shape}"

    def test_no_nan(self, rng):
        """Generated data should not contain NaN."""
        T = 100
        model = SVAdditiveNoise()
        xs, ys = model.simulate(T, rng)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_f_transition(self):
        """f should implement x_k = alpha * x_{k-1}."""
        model = SVAdditiveNoise(alpha=0.9)
        x = np.array([2.0])

        x_next = model.f(x)

        np.testing.assert_allclose(x_next, np.array([0.9 * 2.0]))

    def test_h_observation(self):
        """h should implement h(x) = beta * exp(exp_scale * x)."""
        model = SVAdditiveNoise(beta=0.5, exp_scale=0.5)
        x = np.array([1.0])

        y = model.h(x)

        expected = 0.5 * np.exp(0.5 * 1.0)
        np.testing.assert_allclose(y[0], expected)

    def test_H_jac_numerical(self, rng):
        """H_jac should match numerical differentiation."""
        model = SVAdditiveNoise(beta=0.5, exp_scale=0.5)
        x = np.array([1.5])

        H = model.H_jac(x)

        # Numerical differentiation
        eps = 1e-7
        H_num = (model.h(x + eps) - model.h(x - eps)) / (2 * eps)

        np.testing.assert_allclose(H[0, 0], H_num[0], rtol=1e-5)

    def test_log_likelihood(self, rng):
        """Log-likelihood should return correct shape."""
        model = SVAdditiveNoise()
        N = 50
        particles = rng.standard_normal((N, 1))
        y = np.array([0.5])

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
