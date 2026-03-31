"""Spatial Sensor Network State Space Model."""
import numpy as np


class SpatialSensorNetwork:
    """Linear Gaussian SSM with spatially correlated process noise."""

    def __init__(self, d=64, alpha=0.9, alpha0=3.0, alpha1=0.01, beta=20.0, sigma_z=1.0):
        """Initialize sensor network with d sensors on a grid."""
        self.d = d
        self.alpha = alpha
        self.sigma_z = sigma_z

        grid_size = int(np.sqrt(d))
        if grid_size ** 2 != d:
            raise ValueError(f"d={d} must be a perfect square.")

        # Sensor grid positions
        i, j = np.meshgrid(np.arange(1, grid_size + 1), np.arange(1, grid_size + 1))
        positions = np.column_stack([i.ravel(), j.ravel()])

        # Process noise covariance
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        self.Q = alpha0 * np.exp(-sq_dist / beta) + alpha1 * np.eye(d)

        # Measurement noise covariance
        self.R = (sigma_z ** 2) * np.eye(d)

        # Matrices
        self.A = alpha * np.eye(d)
        self.H = np.eye(d)

        # Precompute
        self.L_Q = np.linalg.cholesky(self.Q)
        self.R_inv = np.eye(d) / (sigma_z ** 2)
        self.log_det_R = d * np.log(sigma_z ** 2)

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

    def Q_sampler(self, rng, N):
        """Sample N process noise vectors."""
        return rng.standard_normal((N, self.d)) @ self.L_Q.T

    def log_likelihood(self, y, particles):
        """Compute log p(y|x) for each particle."""
        const = -0.5 * (self.log_det_R + self.d * np.log(2 * np.pi))
        residuals = y - particles
        return const - 0.5 * np.sum(residuals @ self.R_inv * residuals, axis=1)

    def simulate(self, T, rng, x0=None):
        """Simulate T steps, return (xs, ys)."""
        x = np.zeros(self.d) if x0 is None else x0.copy()
        xs, ys = np.zeros((T, self.d)), np.zeros((T, self.d))

        for t in range(T):
            x = self.f(x) + rng.multivariate_normal(np.zeros(self.d), self.Q)
            xs[t] = x
            ys[t] = x + rng.multivariate_normal(np.zeros(self.d), self.R)

        return xs, ys

    def get_kf_params(self):
        """Return (A, B, C, D) for Kalman filter."""
        D = self.sigma_z * np.eye(self.d)
        return self.A, self.L_Q, self.H, D
