"""Unit tests for Skewed-t Poisson SSM."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ssm.skewed_t_poisson import SkewedTPoissonSSM


class TestSkewedTPoissonSSM:
    """Tests for Skewed-t Poisson state space model."""

    @pytest.fixture
    def model(self):
        """Create a smaller model for faster testing."""
        # Use 16 dimensions (4x4 grid) instead of 144 (12x12)
        return SkewedTPoissonSSM(d=16, alpha=0.9)

    def test_initialization(self, model):
        """Model should initialize with correct dimensions."""
        assert model.d == 16
        assert model.alpha == 0.9
        assert model.Sigma.shape == (16, 16)
        assert model.Sigma_tilde.shape == (16, 16)
        assert model.gamma.shape == (16,)

    def test_f_transition(self, model):
        """f should implement x_k = alpha * x_{k-1}."""
        x = np.ones(16)

        x_next = model.f(x)

        np.testing.assert_allclose(x_next, 0.9 * x)

    def test_h_observation(self, model):
        """h should implement Poisson rate function."""
        x = np.zeros(16)

        rate = model.h(x)

        # h(x) = m1 * exp(m2 * x), at x=0: h(0) = m1
        np.testing.assert_allclose(rate, model.m1 * np.ones(16))

    def test_R_state_dependent(self, model):
        """R should be state-dependent (Poisson variance = mean)."""
        x = np.zeros(16)

        R = model.R_state_dependent(x)

        # At x=0, rate = m1, so R = diag(m1)
        expected = np.diag(model.h(x))
        np.testing.assert_allclose(R, expected)

    def test_sample_skewed_t(self, rng, model):
        """sample_skewed_t should return valid samples."""
        mu = np.zeros(16)

        sample = model.sample_skewed_t(mu, rng)

        assert sample.shape == (16,)
        assert not np.any(np.isnan(sample))

    def test_Q_sampler(self, rng, model):
        """Q_sampler should return samples from N(0, Sigma_tilde)."""
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, 16)
        assert not np.any(np.isnan(noise))

    def test_log_likelihood(self, rng, model):
        """log_likelihood should return N values."""
        N = 50
        particles = rng.standard_normal((N, 16))
        y = rng.poisson(1, size=16).astype(float)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik))

    def test_simulate_shapes(self, rng, model):
        """simulate should return correct shapes."""
        T = 30

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 16)
        assert ys.shape == (T, 16)

    def test_simulate_no_nan(self, rng, model):
        """simulate should not produce NaN."""
        T = 30

        xs, ys = model.simulate(T, rng)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_poisson_observations(self, rng, model):
        """Observations should be non-negative integers (Poisson counts)."""
        T = 30

        xs, ys = model.simulate(T, rng)

        # Observations should be non-negative
        assert np.all(ys >= 0)
        # Observations should be integers (or close to it)
        np.testing.assert_allclose(ys, np.round(ys))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
