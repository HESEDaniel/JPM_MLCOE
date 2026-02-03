"""Unit tests for Spatial Sensor Network SSM."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ssm.spatial_sensor_network import SpatialSensorNetwork


class TestSpatialSensorNetwork:
    """Tests for Spatial Sensor Network state space model."""

    @pytest.fixture
    def model(self):
        """Create a smaller model for faster testing."""
        # Use 16 dimensions (4x4 grid) instead of 64 (8x8)
        return SpatialSensorNetwork(d=16, alpha=0.9)

    def test_initialization(self, model):
        """Model should initialize with correct dimensions."""
        assert model.d == 16
        assert model.alpha == 0.9
        assert model.Q.shape == (16, 16)
        assert model.R.shape == (16, 16)
        assert model.A.shape == (16, 16)
        assert model.H.shape == (16, 16)

    def test_spatial_covariance(self, model):
        """Q should have spatial covariance structure."""
        # Q should be symmetric and positive definite
        assert np.allclose(model.Q, model.Q.T)
        eigvals = np.linalg.eigvalsh(model.Q)
        assert np.all(eigvals > 0)

        # Off-diagonal entries should be non-zero (spatial correlation)
        off_diag = model.Q[0, 1]
        assert off_diag > 0

    def test_Q_sampler(self, rng, model):
        """Q_sampler should sample from N(0, Q)."""
        N = 1000

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, 16)

        # Check covariance is approximately Q
        cov_empirical = np.cov(noise.T)
        np.testing.assert_allclose(cov_empirical, model.Q, rtol=0.3)

    def test_log_likelihood(self, rng, model):
        """log_likelihood should return N values."""
        N = 50
        particles = rng.standard_normal((N, 16))
        y = rng.standard_normal(16)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik))

    def test_simulate_shapes(self, rng, model):
        """simulate should return correct shapes."""
        T = 50

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 16)
        assert ys.shape == (T, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
