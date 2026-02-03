"""Multi-Target Acoustic Tracking State Space Model."""
import time
from typing import Tuple, Optional

import numpy as np

# Default Constants

DEFAULT_PSI = 10.0        # Amplitude parameter
DEFAULT_D0 = 0.1          # Distance offset
DEFAULT_SIGMA_V = 0.1     # Measurement noise std
DEFAULT_N_TARGETS = 4     # Number of targets
DEFAULT_AREA_SIZE = 40.0  # Tracking area size (m)

# Default true initial states (from paper)
DEFAULT_TRUE_X0 = np.array([
    12.0, 6.0, 0.001, 0.001,       # Target 1
    32.0, 32.0, -0.001, -0.005,    # Target 2
    20.0, 13.0, -0.1, 0.01,        # Target 3
    15.0, 35.0, 0.002, 0.002,      # Target 4
])


def build_block_diag(block: np.ndarray, n_repeats: int) -> np.ndarray:
    """Build block diagonal matrix by repeating a block."""
    return np.kron(np.eye(n_repeats), block)


def create_sensor_grid(area_size: float = DEFAULT_AREA_SIZE,
                       n_sensors_per_side: int = 5) -> np.ndarray:
    """
    Create sensor grid coordinates.

    Parameters
    ----------
    area_size : float
        Size of tracking area (square)
    n_sensors_per_side : int
        Number of sensors per side (total = n_sensors_per_side^2)

    Returns
    -------
    sensor_coords : ndarray [n_sensors, 2]
        Sensor (x, y) coordinates
    """
    return np.array([
        (x, y) for x in np.linspace(0, area_size, n_sensors_per_side)
        for y in np.linspace(0, area_size, n_sensors_per_side)
    ])


def create_transition_matrix(n_targets: int = DEFAULT_N_TARGETS) -> np.ndarray:
    """
    Create state transition matrix for constant-velocity model.

    Returns
    -------
    F : ndarray [4*n_targets, 4*n_targets]
        Block-diagonal transition matrix
    """
    F_block = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return build_block_diag(F_block, n_targets)


def create_process_noise_sim(n_targets: int = DEFAULT_N_TARGETS) -> np.ndarray:
    """
    Create process noise covariance for SIMULATION (Section 5A.1).

    Returns
    -------
    Q : ndarray [4*n_targets, 4*n_targets]
        Process noise covariance for trajectory generation
    """
    Q_block = (1.0 / 20.0) * np.array([
        [1/3, 0.0, 0.5, 0.0],
        [0.0, 1/3, 0.0, 0.5],
        [0.5, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 1.0],
    ])
    return build_block_diag(Q_block, n_targets)


def create_process_noise_filt(n_targets: int = DEFAULT_N_TARGETS) -> np.ndarray:
    """
    Create process noise covariance for FILTERING (Section 5A.2).

    Note: "The entries are larger than those used to generate the target trajectories"

    Returns
    -------
    Q : ndarray [4*n_targets, 4*n_targets]
        Process noise covariance for filtering
    """
    Q_block = np.array([
        [3.0, 0.0, 0.1, 0.0],
        [0.0, 3.0, 0.0, 0.1],
        [0.1, 0.0, 0.03, 0.0],
        [0.0, 0.1, 0.0, 0.03],
    ])
    return build_block_diag(Q_block, n_targets)


def create_measurement_noise(n_sensors: int, sigma_v: float = DEFAULT_SIGMA_V) -> np.ndarray:
    """
    Create measurement noise covariance.

    Returns
    -------
    R : ndarray [n_sensors, n_sensors]
        Measurement noise covariance
    """
    return (sigma_v ** 2) * np.eye(n_sensors)


# --- Observation Model ---

def acoustic_observation(x: np.ndarray,
                         sensor_coords: np.ndarray,
                         n_targets: int = DEFAULT_N_TARGETS,
                         psi: float = DEFAULT_PSI,
                         d0: float = DEFAULT_D0) -> np.ndarray:
    """
    Compute acoustic amplitude measurements for all sensors.

    z_s(x) = sum_c Psi / (||p_c - r_s|| + d0)

    Parameters
    ----------
    x : ndarray [4*n_targets]
        State vector [x1, y1, vx1, vy1, ..., xC, yC, vxC, vyC]
    sensor_coords : ndarray [n_sensors, 2]
        Sensor coordinates
    n_targets : int
        Number of targets
    psi : float
        Amplitude parameter
    d0 : float
        Distance offset

    Returns
    -------
    z : ndarray [n_sensors]
        Acoustic amplitude at each sensor
    """
    positions = x.reshape(n_targets, 4)[:, :2]  # [C, 2]
    diff = positions[:, np.newaxis, :] - sensor_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # [C, n_sensors]
    z = np.sum(psi / (distances + d0), axis=0)
    return z


def acoustic_observation_batch(particles: np.ndarray,
                               sensor_coords: np.ndarray,
                               n_targets: int = DEFAULT_N_TARGETS,
                               psi: float = DEFAULT_PSI,
                               d0: float = DEFAULT_D0) -> np.ndarray:
    """
    Compute acoustic measurements for a batch of particles (fully vectorized).

    Parameters
    ----------
    particles : ndarray [N, n_x]
        N particles, each with state dimension n_x = 4 * n_targets
    sensor_coords : ndarray [n_sensors, 2]
        Sensor coordinates
    n_targets : int
        Number of targets
    psi : float
        Amplitude parameter
    d0 : float
        Distance offset

    Returns
    -------
    z : ndarray [N, n_sensors]
        Acoustic measurements for each particle
    """
    N = particles.shape[0]
    positions = particles.reshape(N, n_targets, 4)[:, :, :2]  # [N, C, 2]

    # Compute distances: [N, C, S, 2] -> [N, C, S]
    diff = positions[:, :, np.newaxis, :] - sensor_coords[np.newaxis, np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=3)  # [N, C, S]

    # Sum over targets: [N, S]
    z = np.sum(psi / (distances + d0), axis=1)
    return z


def acoustic_jacobian(x: np.ndarray,
                      sensor_coords: np.ndarray,
                      n_targets: int = DEFAULT_N_TARGETS,
                      psi: float = DEFAULT_PSI,
                      d0: float = DEFAULT_D0) -> np.ndarray:
    """
    Jacobian of acoustic_observation wrt state x.

    Parameters
    ----------
    x : ndarray [4*n_targets]
        State vector
    sensor_coords : ndarray [n_sensors, 2]
        Sensor coordinates

    Returns
    -------
    H : ndarray [n_sensors, 4*n_targets]
        Jacobian matrix
    """
    n_sensors = len(sensor_coords)
    H = np.zeros((n_sensors, 4 * n_targets))
    positions = x.reshape(n_targets, 4)[:, :2]
    diff = positions[:, np.newaxis, :] - sensor_coords[np.newaxis, :, :]
    distances = np.maximum(np.linalg.norm(diff, axis=2), 1e-6)

    # Gradient
    coeff = -psi / (distances * (distances + d0) ** 2)
    grads = coeff[:, :, np.newaxis] * diff

    for c in range(n_targets):
        H[:, 4 * c:4 * c + 2] = grads[c]

    return H


# SSM Simulation

def multi_target_acoustic_ssm(
    T: int,
    rng: np.random.Generator,
    n_targets: int = DEFAULT_N_TARGETS,
    area_size: float = DEFAULT_AREA_SIZE,
    psi: float = DEFAULT_PSI,
    d0: float = DEFAULT_D0,
    sigma_v: float = DEFAULT_SIGMA_V,
    x0: Optional[np.ndarray] = None,
    max_retries_sec: float = 60.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate multi-target acoustic tracking SSM.

    Parameters
    ----------
    T : int
        Number of time steps
    rng : np.random.Generator
        Random number generator
    n_targets : int
        Number of targets
    area_size : float
        Size of tracking area (square)
    psi : float
        Amplitude parameter
    d0 : float
        Distance offset
    sigma_v : float
        Measurement noise std
    x0 : ndarray, optional
        Initial state. If None, uses default from paper.
    max_retries_sec : float
        Maximum time to retry trajectory generation if targets escape

    Returns
    -------
    xs : ndarray [T, 4*n_targets]
        True states
    ys : ndarray [T, n_sensors]
        Observations
    F : ndarray [4*n_targets, 4*n_targets]
        State transition matrix
    Q_sim : ndarray [4*n_targets, 4*n_targets]
        Process noise covariance (simulation)
    Q_filt : ndarray [4*n_targets, 4*n_targets]
        Process noise covariance (filtering)
    R : ndarray [n_sensors, n_sensors]
        Measurement noise covariance
    sensor_coords : ndarray [n_sensors, 2]
        Sensor coordinates
    """
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
        xs = np.zeros((T, n_x))
        ys = np.zeros((T, n_sensors))
        x = x0.copy()

        xs[0] = x
        ys[0] = acoustic_observation(x, sensor_coords, n_targets, psi, d0) + \
                rng.multivariate_normal(np.zeros(n_sensors), R)

        escaped = False

        for t in range(1, T):
            x = F @ x + rng.multivariate_normal(np.zeros(n_x), Q_sim)

            pos = x.reshape(n_targets, 4)[:, :2]
            if not np.all((pos >= 0.0) & (pos <= area_size)):
                escaped = True
                break

            xs[t] = x
            ys[t] = acoustic_observation(x, sensor_coords, n_targets, psi, d0) + \
                    rng.multivariate_normal(np.zeros(n_sensors), R)

        if not escaped:
            return xs, ys, F, Q_sim, Q_filt, R, sensor_coords

    raise RuntimeError(f"multi_target_acoustic_ssm exceeded {max_retries_sec:.2f}s "
                       "without finding an in-bounds trajectory.")


def sample_initial_distribution(
    rng: np.random.Generator,
    n_targets: int = DEFAULT_N_TARGETS,
    area_size: float = DEFAULT_AREA_SIZE,
    x0_true: Optional[np.ndarray] = None,
    pos_std: float = 10.0,
    vel_std: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample initial mean and covariance for filters.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    n_targets : int
        Number of targets
    area_size : float
        Size of tracking area
    x0_true : ndarray, optional
        True initial state (default from paper)
    pos_std : float
        Position uncertainty std (default 10m from paper)
    vel_std : float
        Velocity uncertainty std (default 1 m/s from paper)

    Returns
    -------
    m0 : ndarray [4*n_targets]
        Initial mean
    P0 : ndarray [4*n_targets, 4*n_targets]
        Initial covariance
    """
    if x0_true is None:
        x0_true = DEFAULT_TRUE_X0.copy()

    stds = np.tile([pos_std, pos_std, vel_std, vel_std], n_targets)

    while True:
        m0 = rng.normal(x0_true, stds)
        positions_valid = all(
            0 <= m0[4*c] <= area_size and 0 <= m0[4*c + 1] <= area_size
            for c in range(n_targets)
        )
        if positions_valid:
            break

    P0 = np.diag(stds ** 2)
    return m0, P0


# OMAT Metric

def omat_distance(X_true: np.ndarray, X_est: np.ndarray, p: int = 1) -> float:
    """
    Compute OMAT (Optimal Mass Transfer) distance between true and estimated positions.

    Parameters
    ----------
    X_true : ndarray [C, d]
        True target positions
    X_est : ndarray [C, d]
        Estimated target positions
    p : int
        Order parameter (paper uses p=1)

    Returns
    -------
    omat : float
        OMAT distance
    """
    from scipy.optimize import linear_sum_assignment

    C = len(X_true)

    # Compute pairwise Euclidean distance matrix
    diff = X_true[:, np.newaxis, :] - X_est[np.newaxis, :, :]
    euclidean_distances = np.linalg.norm(diff, axis=2)
    distances_p = euclidean_distances ** p

    # Solve optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(distances_p)
    min_sum = np.sum(distances_p[row_ind, col_ind])

    if p > 0:
        omat = (min_sum / C) ** (1 / p)
    else:
        omat = min_sum / C

    return omat


def compute_omat_trajectory(xs_true: np.ndarray,
                            m_filt: np.ndarray,
                            n_targets: int = DEFAULT_N_TARGETS,
                            p: int = 1) -> np.ndarray:
    """
    Compute OMAT at each time step.

    Parameters
    ----------
    xs_true : ndarray [T, 4*n_targets]
        True states
    m_filt : ndarray [T, 4*n_targets]
        Filtered means
    n_targets : int
        Number of targets
    p : int
        OMAT order parameter

    Returns
    -------
    omat : ndarray [T]
        OMAT distance at each time step
    """
    T = len(xs_true)
    omat = np.zeros(T)
    for t in range(T):
        true_pos = xs_true[t].reshape(n_targets, 4)[:, :2]
        est_pos = m_filt[t].reshape(n_targets, 4)[:, :2]
        omat[t] = omat_distance(true_pos, est_pos, p=p)
    return omat


# Model Parameter Container

class MultiTargetAcousticModel:
    """
    Container for multi-target acoustic tracking model parameters.

    Provides convenient access to model functions with bound parameters.
    """

    def __init__(
        self,
        n_targets: int = DEFAULT_N_TARGETS,
        area_size: float = DEFAULT_AREA_SIZE,
        psi: float = DEFAULT_PSI,
        d0: float = DEFAULT_D0,
        sigma_v: float = DEFAULT_SIGMA_V
    ):
        """Initialize model with given number of targets and area size."""
        self.n_targets = n_targets
        self.area_size = area_size
        self.psi = psi
        self.d0 = d0
        self.sigma_v = sigma_v

        # Create model components
        self.sensor_coords = create_sensor_grid(area_size)
        self.n_sensors = len(self.sensor_coords)
        self.n_x = 4 * n_targets

        self.F = create_transition_matrix(n_targets)
        self.Q_sim = create_process_noise_sim(n_targets)
        self.Q_filt = create_process_noise_filt(n_targets)
        self.R = create_measurement_noise(self.n_sensors, sigma_v)

        # Precompute for efficiency
        self.L_Q_filt = np.linalg.cholesky(self.Q_filt)
        self.R_inv = np.linalg.inv(self.R)
        self.log_det_R = np.linalg.slogdet(self.R)[1]

    def f(self, x: np.ndarray) -> np.ndarray:
        """State transition function."""
        return self.F @ x if x.ndim == 1 else (self.F @ x.T).T

    def f_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition (constant for linear dynamics)."""
        return self.F

    def h(self, x: np.ndarray) -> np.ndarray:
        """Observation function."""
        return acoustic_observation(x, self.sensor_coords, self.n_targets,
                                    self.psi, self.d0)

    def h_batch(self, particles: np.ndarray) -> np.ndarray:
        """Batch observation function."""
        return acoustic_observation_batch(particles, self.sensor_coords,
                                          self.n_targets, self.psi, self.d0)

    def h_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of observation function."""
        return acoustic_jacobian(x, self.sensor_coords, self.n_targets,
                                 self.psi, self.d0)

    def Q_sampler(self, rng: np.random.Generator, N: int) -> np.ndarray:
        """Sample process noise for N particles."""
        return rng.standard_normal((N, self.n_x)) @ self.L_Q_filt.T

    def log_likelihood(self, y: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Compute log-likelihood for particle filter."""
        const = -0.5 * (self.log_det_R + self.n_sensors * np.log(2 * np.pi))
        y_pred = self.h_batch(particles)
        residuals = y - y_pred
        return const - 0.5 * np.sum(residuals @ self.R_inv * residuals, axis=1)

    def simulate(self, T: int, rng: np.random.Generator,
                 x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate trajectory."""
        xs, ys, _, _, _, _, _ = multi_target_acoustic_ssm(
            T, rng, self.n_targets, self.area_size, self.psi, self.d0,
            self.sigma_v, x0
        )
        return xs, ys

    def sample_initial(self, rng: np.random.Generator,
                       x0_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample initial distribution."""
        return sample_initial_distribution(rng, self.n_targets, self.area_size, x0_true)
