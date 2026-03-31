"""Multi-Target Acoustic Tracking State Space Model (TensorFlow)."""
import time
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf

DTYPE = tf.float64

# Default Constants
DEFAULT_PSI = 10.0
DEFAULT_D0 = 0.1
DEFAULT_SIGMA_V = 0.1
DEFAULT_N_TARGETS = 4
DEFAULT_AREA_SIZE = 40.0

DEFAULT_TRUE_X0 = np.array([
    12.0, 6.0, 0.001, 0.001,
    32.0, 32.0, -0.001, -0.005,
    20.0, 13.0, -0.1, 0.01,
    15.0, 35.0, 0.002, 0.002,
])


def build_block_diag(block, n_repeats):
    return np.kron(np.eye(n_repeats), block)


def create_sensor_grid(area_size=DEFAULT_AREA_SIZE, n_sensors_per_side=5):
    return np.array([
        (x, y) for x in np.linspace(0, area_size, n_sensors_per_side)
        for y in np.linspace(0, area_size, n_sensors_per_side)
    ])


def create_transition_matrix(n_targets=DEFAULT_N_TARGETS):
    F_block = np.array([
        [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ])
    return build_block_diag(F_block, n_targets)


def create_process_noise_sim(n_targets=DEFAULT_N_TARGETS):
    Q_block = (1.0 / 20.0) * np.array([
        [1/3, 0.0, 0.5, 0.0], [0.0, 1/3, 0.0, 0.5],
        [0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0],
    ])
    return build_block_diag(Q_block, n_targets)


def create_process_noise_filt(n_targets=DEFAULT_N_TARGETS):
    Q_block = np.array([
        [3.0, 0.0, 0.1, 0.0], [0.0, 3.0, 0.0, 0.1],
        [0.1, 0.0, 0.03, 0.0], [0.0, 0.1, 0.0, 0.03],
    ])
    return build_block_diag(Q_block, n_targets)


def create_measurement_noise(n_sensors, sigma_v=DEFAULT_SIGMA_V):
    return (sigma_v ** 2) * np.eye(n_sensors)


def multi_target_acoustic_ssm(T, rng, n_targets=DEFAULT_N_TARGETS,
                               area_size=DEFAULT_AREA_SIZE,
                               psi=DEFAULT_PSI, d0=DEFAULT_D0,
                               sigma_v=DEFAULT_SIGMA_V, x0=None,
                               max_retries_sec=60.0):
    """Simulate multi-target acoustic tracking SSM (numpy)."""
    if x0 is None:
        x0 = DEFAULT_TRUE_X0.copy()
    n_x = 4 * n_targets
    sensor_coords = create_sensor_grid(area_size)
    n_sensors = len(sensor_coords)
    F = create_transition_matrix(n_targets)
    Q_sim = create_process_noise_sim(n_targets)
    Q_filt = create_process_noise_filt(n_targets)
    R = create_measurement_noise(n_sensors, sigma_v)

    start = time.perf_counter()
    while time.perf_counter() - start < max_retries_sec:
        xs, ys = np.zeros((T, n_x)), np.zeros((T, n_sensors))
        x = x0.copy()
        xs[0] = x
        ys[0] = _acoustic_obs_np(x, sensor_coords, n_targets, psi, d0) + \
                rng.multivariate_normal(np.zeros(n_sensors), R)
        escaped = False
        for t in range(1, T):
            x = F @ x + rng.multivariate_normal(np.zeros(n_x), Q_sim)
            pos = x.reshape(n_targets, 4)[:, :2]
            if not np.all((pos >= 0.0) & (pos <= area_size)):
                escaped = True
                break
            xs[t] = x
            ys[t] = _acoustic_obs_np(x, sensor_coords, n_targets, psi, d0) + \
                    rng.multivariate_normal(np.zeros(n_sensors), R)
        if not escaped:
            return xs, ys, F, Q_sim, Q_filt, R, sensor_coords
    raise RuntimeError("multi_target_acoustic_ssm exceeded retry time.")


def _acoustic_obs_np(x, sensor_coords, n_targets, psi, d0):
    positions = x.reshape(n_targets, 4)[:, :2]
    diff = positions[:, np.newaxis, :] - sensor_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return np.sum(psi / (distances + d0), axis=0)


def sample_initial_distribution(rng, n_targets=DEFAULT_N_TARGETS,
                                area_size=DEFAULT_AREA_SIZE,
                                x0_true=None, pos_std=10.0, vel_std=1.0):
    """Sample initial mean and covariance for filters (numpy)."""
    if x0_true is None:
        x0_true = DEFAULT_TRUE_X0.copy()
    stds = np.tile([pos_std, pos_std, vel_std, vel_std], n_targets)
    while True:
        m0 = rng.normal(x0_true, stds)
        if all(0 <= m0[4*c] <= area_size and 0 <= m0[4*c+1] <= area_size
               for c in range(n_targets)):
            break
    P0 = np.diag(stds ** 2)
    return m0, P0


def omat_distance(X_true, X_est, p=1):
    """OMAT distance (numpy, uses scipy)."""
    from scipy.optimize import linear_sum_assignment
    C = len(X_true)
    diff = X_true[:, np.newaxis, :] - X_est[np.newaxis, :, :]
    distances_p = np.linalg.norm(diff, axis=2) ** p
    row_ind, col_ind = linear_sum_assignment(distances_p)
    min_sum = np.sum(distances_p[row_ind, col_ind])
    return (min_sum / C) ** (1 / p) if p > 0 else min_sum / C


def compute_omat_trajectory(xs_true, m_filt, n_targets=DEFAULT_N_TARGETS, p=1):
    """OMAT at each time step (numpy)."""
    T = len(xs_true)
    omat = np.zeros(T)
    for t in range(T):
        true_pos = xs_true[t].reshape(n_targets, 4)[:, :2]
        est_pos = m_filt[t].reshape(n_targets, 4)[:, :2]
        omat[t] = omat_distance(true_pos, est_pos, p=p)
    return omat


class MultiTargetAcousticModel:
    """Multi-target acoustic tracking model (TensorFlow)."""

    def __init__(self, n_targets=DEFAULT_N_TARGETS, area_size=DEFAULT_AREA_SIZE,
                 psi=DEFAULT_PSI, d0=DEFAULT_D0, sigma_v=DEFAULT_SIGMA_V):
        self.n_targets = n_targets
        self.area_size = area_size
        self._psi = tf.constant(psi, dtype=DTYPE)
        self._d0 = tf.constant(d0, dtype=DTYPE)
        self.sigma_v = sigma_v

        sensor_coords_np = create_sensor_grid(area_size)
        self._sensor_coords = tf.constant(sensor_coords_np, dtype=DTYPE)
        self.n_sensors = len(sensor_coords_np)
        self.state_dim = 4 * n_targets
        self.obs_dim = self.n_sensors

        F_np = create_transition_matrix(n_targets)
        Q_filt_np = create_process_noise_filt(n_targets)
        R_np = create_measurement_noise(self.n_sensors, sigma_v)

        self._F = tf.constant(F_np, dtype=DTYPE)
        self.Q_sim = tf.constant(create_process_noise_sim(n_targets), dtype=DTYPE)
        self.Q = tf.constant(Q_filt_np, dtype=DTYPE)  # filtering Q
        self.R = tf.constant(R_np, dtype=DTYPE)
        self._R_inv = tf.linalg.inv(self.R)
        self._log_det_R = tf.linalg.slogdet(self.R)[1]
        self._L_Q = tf.linalg.cholesky(self.Q)

        # Default initial distribution
        self.m0 = tf.constant(DEFAULT_TRUE_X0, dtype=DTYPE)
        self.P0 = tf.constant(np.diag(np.tile([100.0, 100.0, 1.0, 1.0], n_targets)),
                               dtype=DTYPE)

    def f(self, x):
        return tf.linalg.matvec(self._F, x)

    def h(self, x):
        """Acoustic observation: z_s = sum_c psi / (||p_c - r_s|| + d0)."""
        positions = tf.reshape(x, [self.n_targets, 4])[:, :2]  # [C, 2]
        diff = positions[:, tf.newaxis, :] - self._sensor_coords[tf.newaxis, :, :]  # [C, S, 2]
        distances = tf.norm(diff, axis=2)  # [C, S]
        return tf.reduce_sum(self._psi / (distances + self._d0), axis=0)  # [S]

    def F_jac(self, x):
        return self._F

    def H_jac(self, x):
        """Jacobian of acoustic observation wrt state x."""
        positions = tf.reshape(x, [self.n_targets, 4])[:, :2]  # [C, 2]
        diff = positions[:, tf.newaxis, :] - self._sensor_coords[tf.newaxis, :, :]  # [C, S, 2]
        distances = tf.maximum(tf.norm(diff, axis=2), 1e-6)  # [C, S]

        coeff = -self._psi / (distances * (distances + self._d0) ** 2)  # [C, S]
        grads = coeff[:, :, tf.newaxis] * diff  # [C, S, 2]

        # Build sparse Jacobian [S, 4*C]
        n_x = self.state_dim
        n_s = self.n_sensors
        H = tf.zeros((n_s, n_x), dtype=DTYPE)
        for c in range(self.n_targets):
            indices = tf.stack([
                tf.repeat(tf.range(n_s), 2),
                tf.tile(tf.constant([4 * c, 4 * c + 1]), [n_s])
            ], axis=1)
            values = tf.reshape(grads[c], [-1])
            H = H + tf.scatter_nd(indices, values, (n_s, n_x))
        return H

    def f_batch(self, particles):
        return particles @ tf.transpose(self._F)

    def h_batch(self, particles):
        """Batch acoustic observation [N, n_x] -> [N, n_sensors]."""
        N = tf.shape(particles)[0]
        positions = tf.reshape(particles, [N, self.n_targets, 4])[:, :, :2]  # [N, C, 2]
        diff = positions[:, :, tf.newaxis, :] - self._sensor_coords[tf.newaxis, tf.newaxis, :, :]
        distances = tf.norm(diff, axis=3)  # [N, C, S]
        return tf.reduce_sum(self._psi / (distances + self._d0), axis=1)  # [N, S]

    def Q_sampler(self, rng, N):
        z = rng.normal(shape=(N, self.state_dim), dtype=DTYPE)
        return tf.linalg.matvec(self._L_Q, z)

    def log_likelihood(self, y, particles):
        const = -0.5 * (self._log_det_R + tf.cast(self.n_sensors, DTYPE) * tf.constant(np.log(2 * np.pi), dtype=DTYPE))
        y_pred = self.h_batch(particles)
        residuals = y - y_pred
        return const - 0.5 * tf.reduce_sum(residuals * tf.linalg.matvec(self._R_inv, residuals), axis=1)

    def simulate(self, T, rng, x0=None):
        """Simulate trajectory (numpy)."""
        xs, ys, _, _, _, _, _ = multi_target_acoustic_ssm(
            T, rng, self.n_targets, self.area_size,
            float(self._psi), float(self._d0), self.sigma_v, x0)
        return xs, ys

    def sample_initial(self, rng, x0_true=None):
        """Sample initial distribution (numpy)."""
        return sample_initial_distribution(rng, self.n_targets, self.area_size, x0_true)
