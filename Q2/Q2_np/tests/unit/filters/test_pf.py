"""Unit tests for Particle Filter implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.filters.pf import particle_filter, systematic_resample


class TestSystematicResample:
    """Tests for systematic resampling algorithm."""

    def test_preserves_count_and_valid_indices(self, rng):
        """Resampling should preserve particle count and produce valid indices."""
        N = 100
        weights = rng.dirichlet(np.ones(N))

        indices = systematic_resample(weights, rng)

        assert len(indices) == N
        assert np.all(indices >= 0)
        assert np.all(indices < N)

class TestParticleFilter:
    """Tests for Particle Filter."""

    @pytest.fixture
    def simple_pf_model(self):
        """Simple linear system for PF testing."""
        def f(x):
            return 0.9 * x

        def Q_sampler(rng, N):
            return 0.1 * rng.standard_normal((N, 1))

        def log_likelihood(y, particles):
            diff = y[0] - particles[:, 0]
            return -0.5 * diff**2 / 0.1

        m0 = np.array([0.0])
        P0 = np.array([[1.0]])

        return f, Q_sampler, log_likelihood, m0, P0

    def test_output_shapes(self, rng, simple_pf_model):
        """Verify correct output shapes."""
        f, Q_sampler, log_likelihood, m0, P0 = simple_pf_model
        T, N = 50, 100
        ys = rng.standard_normal((T, 1))

        m_filt, P_filt, ess, _ = particle_filter(
            f, lambda x: x, Q_sampler, log_likelihood, m0, P0, ys,
            N_particles=N, rng=rng
        )

        assert m_filt.shape == (T, 1)
        assert P_filt.shape == (T, 1, 1)
        assert ess.shape == (T,)

    def test_ess_bounds(self, rng, simple_pf_model):
        """ESS should be between 1 and N."""
        f, Q_sampler, log_likelihood, m0, P0 = simple_pf_model
        N = 100
        ys = rng.standard_normal((50, 1))

        _, _, ess, _ = particle_filter(
            f, lambda x: x, Q_sampler, log_likelihood, m0, P0, ys,
            N_particles=N, rng=rng
        )

        assert np.all(ess >= 1)
        assert np.all(ess <= N)

    def test_no_nan(self, rng, simple_pf_model):
        """Output should not contain NaN."""
        f, Q_sampler, log_likelihood, m0, P0 = simple_pf_model
        ys = rng.standard_normal((50, 1))

        m_filt, P_filt, ess, _ = particle_filter(
            f, lambda x: x, Q_sampler, log_likelihood, m0, P0, ys,
            N_particles=100, rng=rng
        )

        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert not np.any(np.isnan(ess))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
