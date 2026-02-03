"""Unit tests for Multi-Target Acoustic Tracking SSM."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ssm.multi_target_acoustic import (
    build_block_diag, create_sensor_grid, create_transition_matrix,
    create_measurement_noise,
    acoustic_observation, acoustic_observation_batch, acoustic_jacobian,
    multi_target_acoustic_ssm, sample_initial_distribution,
    omat_distance, compute_omat_trajectory, MultiTargetAcousticModel
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_build_block_diag(self):
        """Block diagonal should repeat block correctly."""
        block = np.array([[1, 2], [3, 4]])
        n_repeats = 3

        result = build_block_diag(block, n_repeats)

        assert result.shape == (6, 6)
        np.testing.assert_array_equal(result[:2, :2], block)
        np.testing.assert_array_equal(result[2:4, 2:4], block)
        np.testing.assert_array_equal(result[4:6, 4:6], block)
        np.testing.assert_array_equal(result[:2, 2:4], np.zeros((2, 2)))

    def test_create_sensor_grid(self):
        """Sensor grid should have correct shape and positions."""
        area_size = 40.0
        n_per_side = 5

        sensors = create_sensor_grid(area_size, n_per_side)

        assert sensors.shape == (25, 2)
        assert sensors.min() == 0.0
        assert sensors.max() == 40.0

    def test_create_transition_matrix(self):
        """Transition matrix should have constant velocity structure."""
        n_targets = 2
        F = create_transition_matrix(n_targets)

        assert F.shape == (8, 8)
        # Check block structure: position integrates velocity
        np.testing.assert_allclose(F[0, 2], 1.0)  # x += vx
        np.testing.assert_allclose(F[1, 3], 1.0)  # y += vy

    def test_create_measurement_noise(self):
        """Measurement noise should be diagonal."""
        n_sensors = 25
        sigma_v = 0.1

        R = create_measurement_noise(n_sensors, sigma_v)

        assert R.shape == (25, 25)
        np.testing.assert_allclose(R, sigma_v**2 * np.eye(25))


class TestAcousticObservation:
    """Tests for acoustic observation function."""

    def test_observation_shape(self):
        """Observation should have n_sensors components."""
        n_targets = 4
        x = np.zeros(16)
        x[:2] = [20, 20]  # Target 1 at center
        sensors = create_sensor_grid(40.0, 5)

        z = acoustic_observation(x, sensors, n_targets)

        assert z.shape == (25,)

    def test_batch_observation_shape(self):
        """Batch observation should have (N, n_sensors) shape."""
        n_targets = 4
        N = 50
        particles = np.random.randn(N, 16)
        sensors = create_sensor_grid(40.0, 5)

        z_batch = acoustic_observation_batch(particles, sensors, n_targets)

        assert z_batch.shape == (N, 25)

    def test_batch_matches_single(self):
        """Batch observation should match single observations."""
        n_targets = 4
        N = 10
        particles = np.random.randn(N, 16)
        sensors = create_sensor_grid(40.0, 5)

        z_batch = acoustic_observation_batch(particles, sensors, n_targets)

        for i in range(N):
            z_single = acoustic_observation(particles[i], sensors, n_targets)
            np.testing.assert_allclose(z_batch[i], z_single)

    def test_observation_formula(self):
        """Observation should follow z = sum Psi / (d + d0)."""
        # Single target at (10, 10)
        x = np.array([10.0, 10.0, 0.0, 0.0] * 4)  # Duplicate for 4 targets
        x[4:] = [30, 30, 0, 0] * 3  # Move other targets away
        sensors = np.array([[0.0, 0.0]])  # Single sensor at origin
        psi, d0 = 10.0, 0.1

        z = acoustic_observation(x, sensors, n_targets=4, psi=psi, d0=d0)

        # Distance from (10, 10) to (0, 0) = sqrt(200)
        d1 = np.sqrt(200)
        # Distance from (30, 30) to (0, 0) = sqrt(1800)
        d2 = np.sqrt(1800)
        expected = psi / (d1 + d0) + 3 * psi / (d2 + d0)
        np.testing.assert_allclose(z[0], expected, rtol=1e-10)

    def test_jacobian_shape(self):
        """Jacobian should have (n_sensors, n_x) shape."""
        n_targets = 4
        x = np.random.randn(16)
        sensors = create_sensor_grid(40.0, 5)

        H = acoustic_jacobian(x, sensors, n_targets)

        assert H.shape == (25, 16)

    def test_jacobian_numerical(self):
        """Jacobian should match numerical differentiation."""
        n_targets = 4
        x = np.array([15.0, 15.0, 0.1, 0.1] * 4)
        sensors = create_sensor_grid(40.0, 5)

        H = acoustic_jacobian(x, sensors, n_targets)

        # Numerical differentiation
        eps = 1e-6
        H_num = np.zeros_like(H)
        for i in range(16):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            H_num[:, i] = (acoustic_observation(x_plus, sensors, n_targets) -
                          acoustic_observation(x_minus, sensors, n_targets)) / (2 * eps)

        np.testing.assert_allclose(H, H_num, rtol=1e-4, atol=1e-8)


class TestMultiTargetAcousticSSM:
    """Tests for multi-target acoustic SSM simulation."""

    def test_output_shapes(self, rng):
        """Generated data should have correct shapes."""
        T = 30
        n_targets = 4

        xs, ys, F, Q_sim, Q_filt, R, sensors = multi_target_acoustic_ssm(
            T, rng, n_targets=n_targets, max_retries_sec=30.0
        )

        assert xs.shape == (T, 16)
        assert ys.shape == (T, 25)
        assert F.shape == (16, 16)
        assert Q_sim.shape == (16, 16)
        assert Q_filt.shape == (16, 16)
        assert R.shape == (25, 25)
        assert sensors.shape == (25, 2)

    def test_no_nan(self, rng):
        """Generated data should not contain NaN."""
        T = 30

        xs, ys, _, _, _, _, _ = multi_target_acoustic_ssm(T, rng, max_retries_sec=30.0)

        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_targets_stay_in_bounds(self, rng):
        """Targets should stay within tracking area."""
        T = 30
        area_size = 40.0

        xs, _, _, _, _, _, _ = multi_target_acoustic_ssm(
            T, rng, area_size=area_size, max_retries_sec=30.0
        )

        # Extract positions
        for c in range(4):
            pos_x = xs[:, 4*c]
            pos_y = xs[:, 4*c + 1]
            assert np.all(pos_x >= 0) and np.all(pos_x <= area_size)
            assert np.all(pos_y >= 0) and np.all(pos_y <= area_size)


class TestSampleInitialDistribution:
    """Tests for initial distribution sampling."""

    def test_output_shapes(self, rng):
        """Initial mean and covariance should have correct shapes."""
        n_targets = 4

        m0, P0 = sample_initial_distribution(rng, n_targets)

        assert m0.shape == (16,)
        assert P0.shape == (16, 16)

    def test_positions_in_bounds(self, rng):
        """Sampled positions should be within area."""
        n_targets = 4
        area_size = 40.0

        m0, P0 = sample_initial_distribution(rng, n_targets, area_size=area_size)

        for c in range(n_targets):
            assert 0 <= m0[4*c] <= area_size
            assert 0 <= m0[4*c + 1] <= area_size


class TestOMATMetrics:
    """Tests for OMAT distance computation."""

    def test_omat_distance_zero(self):
        """OMAT distance should be zero for identical positions."""
        X = np.array([[10.0, 10.0], [20.0, 20.0]])

        d = omat_distance(X, X)

        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_omat_distance_symmetric(self):
        """OMAT distance should be symmetric."""
        X1 = np.array([[10.0, 10.0], [20.0, 20.0]])
        X2 = np.array([[15.0, 15.0], [25.0, 25.0]])

        d12 = omat_distance(X1, X2)
        d21 = omat_distance(X2, X1)

        np.testing.assert_allclose(d12, d21)

    def test_compute_omat_trajectory_shape(self, rng):
        """OMAT trajectory should have T components."""
        T = 20
        n_targets = 4

        xs_true = rng.standard_normal((T, 16))
        m_filt = rng.standard_normal((T, 16))

        omat = compute_omat_trajectory(xs_true, m_filt, n_targets)

        assert omat.shape == (T,)
        assert np.all(omat >= 0)


class TestMultiTargetAcousticModel:
    """Tests for MultiTargetAcousticModel class."""

    def test_initialization(self):
        """Model should initialize with default parameters."""
        model = MultiTargetAcousticModel()

        assert model.n_targets == 4
        assert model.n_x == 16
        assert model.n_sensors == 25
        assert model.F.shape == (16, 16)

    def test_h_function(self):
        """h should implement acoustic observation."""
        model = MultiTargetAcousticModel()
        x = np.random.randn(16)

        z = model.h(x)

        assert z.shape == (25,)

    def test_Q_sampler(self, rng):
        """Q_sampler should return process noise samples."""
        model = MultiTargetAcousticModel()
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, 16)

    def test_log_likelihood(self, rng):
        """log_likelihood should return N values."""
        model = MultiTargetAcousticModel()
        N = 50
        particles = np.random.randn(N, 16)
        y = np.random.randn(25)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik))

    def test_simulate(self, rng):
        """simulate should return states and observations."""
        model = MultiTargetAcousticModel()
        T = 30

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 16)
        assert ys.shape == (T, 25)

    def test_sample_initial(self, rng):
        """sample_initial should return valid initial distribution."""
        model = MultiTargetAcousticModel()

        m0, P0 = model.sample_initial(rng)

        assert m0.shape == (16,)
        assert P0.shape == (16, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
