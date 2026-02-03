"""Unit tests for RKHS Particle Flow Filter implementation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.flows.rkhs_pff import (
    localization_matrix, rkhs_particle_flow, rkhs_pff_linear_gaussian
)


class TestLocalizationMatrix:
    """Tests for localization matrix computation."""

    def test_localization_matrix_shape(self):
        """Localization matrix should have correct shape."""
        C = localization_matrix(10, r_in=3.0)
        assert C.shape == (10, 10)

    def test_localization_matrix_symmetric(self):
        """Localization matrix should be symmetric."""
        C = localization_matrix(10, r_in=3.0)
        np.testing.assert_allclose(C, C.T, rtol=1e-10)

    def test_localization_matrix_diagonal_ones(self):
        """Localization matrix should have ones on diagonal."""
        C = localization_matrix(10, r_in=4.0)
        np.testing.assert_allclose(np.diag(C), np.ones(10), rtol=1e-10)


class TestRKHSParticleFlow:
    """Tests for RKHS particle flow."""

    def test_output_shape(self, rng, linear_model):
        """RKHS flow should preserve particle shape."""
        m = linear_model
        particles = rng.standard_normal((100, 2))
        y = np.array([1.0, 0.5])

        particles_out = rkhs_particle_flow(particles, m['h'], m['H_jac'], m['R'], y, n_steps=10)

        assert particles_out.shape == (100, 2)

    def test_no_nan(self, rng, linear_model):
        """RKHS flow should not produce NaN."""
        m = linear_model
        particles = rng.standard_normal((100, 2))
        y = np.array([1.0, 0.5])

        particles_out = rkhs_particle_flow(particles, m['h'], m['H_jac'], m['R'], y, n_steps=10)

        assert not np.any(np.isnan(particles_out))

    def test_moves_particles(self, rng, linear_model):
        """RKHS flow should move particles."""
        m = linear_model
        particles = rng.standard_normal((100, 2))
        y = np.array([1.0, 0.5])

        particles_out = rkhs_particle_flow(
            particles, m['h'], m['H_jac'], m['R'], y, n_steps=10, step_size=0.1
        )

        assert not np.allclose(particles, particles_out)


class TestRKHSPFFLinearGaussian:
    """Tests for RKHS PFF on linear Gaussian systems."""

    def test_output_shape(self, rng, linear_ssm):
        """RKHS PFF should return correct shapes."""
        m = linear_ssm
        N = 50

        m_filt, P_filt, ess, _ = rkhs_pff_linear_gaussian(
            m['ys'], m['A'], m['H'], m['Q'], m['R'], m['m0'], m['P0'],
            N_particles=N, rng=rng
        )

        assert m_filt.shape == (m['T'], 2)
        assert P_filt.shape == (m['T'], 2, 2)
        assert ess.shape == (m['T'],)

    def test_no_nan(self, rng, linear_ssm):
        """RKHS PFF should not produce NaN."""
        m = linear_ssm

        m_filt, P_filt, ess, _ = rkhs_pff_linear_gaussian(
            m['ys'], m['A'], m['H'], m['Q'], m['R'], m['m0'], m['P0'],
            N_particles=50, rng=rng
        )

        assert not np.any(np.isnan(m_filt))
        assert not np.any(np.isnan(P_filt))
        assert not np.any(np.isnan(ess))

    def test_ess_bounds(self, rng, linear_ssm):
        """ESS should be between 1 and N."""
        m = linear_ssm
        N = 50

        _, _, ess, _ = rkhs_pff_linear_gaussian(
            m['ys'], m['A'], m['H'], m['Q'], m['R'], m['m0'], m['P0'],
            N_particles=N, rng=rng
        )

        assert np.all(ess >= 1)
        assert np.all(ess <= N)


class TestRKHSPFFNonlinear:
    """Tests for RKHS PFF on nonlinear systems."""

    def test_nonlinear_tracking(self, rng, nonlinear_model):
        """RKHS flow should move particles towards observation."""
        m = nonlinear_model
        N = 200
        particles = rng.multivariate_normal(m['m0'], m['P0'], size=N)

        particles_out = rkhs_particle_flow(
            particles, m['h'], m['H_jac'], m['R'], m['y'],
            n_steps=20, step_size=0.05
        )

        # Particles should not contain NaN
        assert not np.any(np.isnan(particles_out))

        # Mean should be different from input
        mean_in = np.mean(particles, axis=0)
        mean_out = np.mean(particles_out, axis=0)
        assert not np.allclose(mean_in, mean_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
