"""Unit tests for Multi-Target Acoustic Tracking SSM (TensorFlow)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.ssm.multi_target_acoustic import (
    build_block_diag, create_sensor_grid, create_transition_matrix,
    create_measurement_noise,
    multi_target_acoustic_ssm, sample_initial_distribution,
    omat_distance, compute_omat_trajectory, MultiTargetAcousticModel,
)

DTYPE = tf.float64


class TestHelperFunctions:
    """Tests for numpy helper functions."""

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


class TestMultiTargetAcousticModel:
    """Tests for MultiTargetAcousticModel TF class."""

    def test_init(self):
        """Model should initialize with correct dimensions."""
        model = MultiTargetAcousticModel()

        assert model.n_targets == 4
        assert model.state_dim == 16
        assert model.obs_dim == 25
        assert model.Q.shape == (16, 16)
        assert model.R.shape == (25, 25)

    def test_f_shape(self):
        """f should return (state_dim,) tensor."""
        model = MultiTargetAcousticModel()
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.f(x)

        assert result.shape == (model.state_dim,)

    def test_h_shape(self):
        """h should return (obs_dim,) tensor."""
        model = MultiTargetAcousticModel()
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        result = model.h(x)

        assert result.shape == (model.obs_dim,)

    def test_F_jac_shape(self):
        """F_jac should return (state_dim, state_dim) matrix."""
        model = MultiTargetAcousticModel()
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        F = model.F_jac(x)

        assert F.shape == (model.state_dim, model.state_dim)

    def test_H_jac_shape(self):
        """H_jac should return (obs_dim, state_dim) matrix."""
        model = MultiTargetAcousticModel()
        x = tf.constant(np.random.randn(model.state_dim), dtype=DTYPE)

        H = model.H_jac(x)

        assert H.shape == (model.obs_dim, model.state_dim)

    def test_H_jac_numerical(self):
        """H_jac should match numerical differentiation."""
        model = MultiTargetAcousticModel()
        x_np = np.array([15.0, 15.0, 0.1, 0.1] * 4)
        x = tf.constant(x_np, dtype=DTYPE)

        H = model.H_jac(x).numpy()

        eps = 1e-6
        H_num = np.zeros((model.obs_dim, model.state_dim))
        for i in range(model.state_dim):
            x_plus = x_np.copy()
            x_plus[i] += eps
            x_minus = x_np.copy()
            x_minus[i] -= eps
            h_plus = model.h(tf.constant(x_plus, dtype=DTYPE)).numpy()
            h_minus = model.h(tf.constant(x_minus, dtype=DTYPE)).numpy()
            H_num[:, i] = (h_plus - h_minus) / (2 * eps)

        np.testing.assert_allclose(H, H_num, rtol=1e-4, atol=1e-8)

    def test_f_batch(self):
        """f_batch should apply f to each particle."""
        model = MultiTargetAcousticModel()
        N = 10
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)

        result = model.f_batch(particles)

        assert result.shape == (N, model.state_dim)

    def test_h_batch(self):
        """h_batch should match single h for each particle."""
        model = MultiTargetAcousticModel()
        N = 10
        particles_np = np.random.randn(N, model.state_dim)
        particles = tf.constant(particles_np, dtype=DTYPE)

        z_batch = model.h_batch(particles)

        assert z_batch.shape == (N, model.obs_dim)
        for i in range(N):
            z_single = model.h(tf.constant(particles_np[i], dtype=DTYPE)).numpy()
            np.testing.assert_allclose(z_batch[i].numpy(), z_single, rtol=1e-10)

    def test_Q_sampler(self):
        """Q_sampler should return (N, state_dim) noise samples."""
        model = MultiTargetAcousticModel()
        rng = tf.random.Generator.from_seed(42)
        N = 100

        noise = model.Q_sampler(rng, N)

        assert noise.shape == (N, model.state_dim)

    def test_log_likelihood(self):
        """log_likelihood should return N finite values."""
        model = MultiTargetAcousticModel()
        N = 50
        particles = tf.constant(np.random.randn(N, model.state_dim), dtype=DTYPE)
        y = tf.constant(np.random.randn(model.obs_dim), dtype=DTYPE)

        log_lik = model.log_likelihood(y, particles)

        assert log_lik.shape == (N,)
        assert np.all(np.isfinite(log_lik.numpy()))

    def test_simulate(self, rng):
        """simulate should return correct shapes."""
        model = MultiTargetAcousticModel()
        T = 30

        xs, ys = model.simulate(T, rng)

        assert xs.shape == (T, 16)
        assert ys.shape == (T, 25)
        assert not np.any(np.isnan(xs)), "NaN in states"
        assert not np.any(np.isnan(ys)), "NaN in observations"

    def test_sample_initial(self, rng):
        """sample_initial should return valid initial distribution."""
        model = MultiTargetAcousticModel()

        m0, P0 = model.sample_initial(rng)

        assert m0.shape == (16,)
        assert P0.shape == (16, 16)


class TestMultiTargetAcousticSSM:
    """Tests for multi_target_acoustic_ssm simulation function."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
