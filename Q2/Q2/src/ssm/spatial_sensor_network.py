"""Spatial Sensor Network State Space Model (TensorFlow)."""
import numpy as np
import tensorflow as tf

DTYPE = tf.float64


class SpatialSensorNetwork:
    """Linear Gaussian SSM with spatially correlated process noise."""

    def __init__(self, d=64, alpha=0.9, alpha0=3.0, alpha1=0.01, beta=20.0, sigma_z=1.0):
        """Initialize sensor network with d sensors on a grid.

        Parameters
        ----------
        d : int
            State/observation dimension (must be a perfect square).
        alpha : float
            AR(1) coefficient for state transition.
        alpha0 : float
            Spatial covariance amplitude.
        alpha1 : float
            Spatial covariance nugget (diagonal regularization).
        beta : float
            Spatial covariance length scale.
        sigma_z : float
            Observation noise std.
        """
        self.d = d
        self.alpha = alpha
        self.sigma_z = sigma_z
        self.state_dim = d
        self.obs_dim = d

        grid_size = int(np.sqrt(d))
        if grid_size ** 2 != d:
            raise ValueError(f"d={d} must be a perfect square.")

        # Sensor grid positions
        i, j = np.meshgrid(np.arange(1, grid_size + 1), np.arange(1, grid_size + 1))
        positions = np.column_stack([i.ravel(), j.ravel()])

        # Process noise covariance
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        Q_np = alpha0 * np.exp(-sq_dist / beta) + alpha1 * np.eye(d)

        # Measurement noise covariance
        R_np = (sigma_z ** 2) * np.eye(d)

        # Matrices
        A_np = alpha * np.eye(d)
        H_np = np.eye(d)

        self.A = tf.constant(A_np, dtype=DTYPE)
        self.C = tf.constant(H_np, dtype=DTYPE)  # observation matrix (KF compatibility)
        self.H = self.C  # alias
        self.Q = tf.constant(Q_np, dtype=DTYPE)
        self.R = tf.constant(R_np, dtype=DTYPE)
        self.m0 = tf.zeros(d, dtype=DTYPE)
        self.P0 = tf.constant(Q_np, dtype=DTYPE)

        # Precompute
        self._L_Q = tf.linalg.cholesky(self.Q)
        self._R_inv = tf.constant(np.eye(d) / (sigma_z ** 2), dtype=DTYPE)
        self._log_det_R = tf.constant(d * np.log(sigma_z ** 2), dtype=DTYPE)

    def f(self, x):
        """State transition."""
        return self.alpha * x

    def h(self, x):
        """Observation function (identity)."""
        return x

    def F_jac(self, x):
        """Jacobian of f (constant)."""
        return self.A

    def H_jac(self, x):
        """Jacobian of h (constant)."""
        return self.H

    def f_batch(self, particles):
        return self.alpha * particles

    def h_batch(self, particles):
        return particles

    def Q_sampler(self, rng, N):
        """Sample N process noise vectors."""
        z = rng.normal(shape=(N, self.d), dtype=DTYPE)
        return tf.linalg.matvec(self._L_Q, z)

    def log_likelihood(self, y, particles):
        """Compute log p(y|x) for each particle."""
        const = -0.5 * (self._log_det_R + tf.cast(self.d, DTYPE) * tf.cast(np.log(2 * np.pi), DTYPE))
        residuals = y - particles
        return const - 0.5 * tf.reduce_sum(tf.linalg.matvec(self._R_inv, residuals) * residuals, axis=1)

    def simulate(self, T, rng, x0=None):
        """Simulate T steps, return (xs, ys)."""
        Q_np = self.Q.numpy()
        R_np = self.R.numpy()
        d = self.d

        x = np.zeros(d) if x0 is None else x0.copy()
        xs, ys = np.zeros((T, d)), np.zeros((T, d))

        for t in range(T):
            x = self.alpha * x + rng.multivariate_normal(np.zeros(d), Q_np)
            xs[t] = x
            ys[t] = x + rng.multivariate_normal(np.zeros(d), R_np)

        return xs, ys

    def get_kf_params(self):
        """Return (A, L_Q, H, D) for Kalman filter."""
        D = tf.constant(self.sigma_z * np.eye(self.d), dtype=DTYPE)
        return self.A, self._L_Q, self.H, D
