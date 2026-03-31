"""Skewed-t Dynamics with Poisson Measurements SSM."""
import numpy as np
from scipy.special import gammaln


class SkewedTPoissonSSM:
    """GH Skewed-t dynamics with Poisson count measurements."""

    def __init__(self, d=144, alpha=0.9, alpha0=3.0, alpha1=0.01,
                 beta=20.0, gamma_val=0.3, nu=7.0, m1=1.0, m2=1/3):
        """Initialize model with d-dimensional state on a grid."""
        self.d = d
        self.alpha = alpha
        self.m1, self.m2, self.nu = m1, m2, nu

        grid_size = int(np.sqrt(d))
        if grid_size ** 2 != d:
            raise ValueError(f"d={d} must be a perfect square.")

        # Sensor grid and spatial covariance (Eq. 42)
        i, j = np.meshgrid(np.arange(1, grid_size + 1), np.arange(1, grid_size + 1))
        positions = np.column_stack([i.ravel(), j.ravel()])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        self.Sigma = alpha0 * np.exp(-sq_dist / beta) + alpha1 * np.eye(d)

        # Skewness vector
        self.gamma = gamma_val * np.ones(d)

        # Effective covariance Sigma_tilde (Eq. 44)
        coeff1 = nu / max(nu - 2, 0.1)
        coeff2 = (nu ** 2) / ((2 * nu - 8) * ((nu / 2 - 1) ** 2)) if nu > 4 else 0.0
        self.Sigma_tilde = coeff1 * self.Sigma + coeff2 * np.outer(self.gamma, self.gamma)

        # Precompute
        self.L_Sigma = np.linalg.cholesky(self.Sigma)
        self.A = alpha * np.eye(d)
        self.Q = self.Sigma_tilde.copy()
        self.L_Q = np.linalg.cholesky(self.Q + 1e-8 * np.eye(d))

    def f(self, x):
        """State transition mean: alpha * x."""
        return self.alpha * x

    def h(self, x):
        """Observation function: Poisson rate m1 * exp(m2 * x)."""
        return self.m1 * np.exp(self.m2 * x)

    def F_jac(self, x):
        """Jacobian of f (constant)."""
        return self.A

    def H_jac(self, x):
        """Jacobian of h: diag(m1 * m2 * exp(m2 * x))."""
        return np.diag(self.m1 * self.m2 * np.exp(self.m2 * x))

    def R_state_dependent(self, x):
        """State-dependent R: Poisson variance = mean."""
        return np.diag(self.h(x))

    @property
    def R(self):
        """Default R at x=0."""
        return self.R_state_dependent(np.zeros(self.d))

    def sample_skewed_t(self, mu, rng):
        """Sample from GH Skewed-t: X = mu + W*gamma + sqrt(W)*Z."""
        W = 1.0 / rng.gamma(self.nu / 2, 2.0 / self.nu)
        Z = rng.standard_normal(self.d) @ self.L_Sigma.T
        return mu + W * self.gamma + np.sqrt(W) * Z

    def Q_sampler(self, rng, N):
        """Sample N process noise from N(0, Sigma_tilde)."""
        return rng.standard_normal((N, self.d)) @ self.L_Q.T

    def log_likelihood(self, y, particles):
        """Poisson log-likelihood for each particle."""
        lam = np.clip(self.m1 * np.exp(self.m2 * particles), 1e-10, 1e10)
        return np.sum(y * np.log(lam) - lam - gammaln(y + 1), axis=1)

    def simulate(self, T, rng, x0=None):
        """Simulate T steps with skewed-t dynamics and Poisson observations."""
        x = np.zeros(self.d) if x0 is None else x0.copy()
        xs, ys = np.zeros((T, self.d)), np.zeros((T, self.d))

        for t in range(T):
            x = self.sample_skewed_t(self.f(x), rng)
            xs[t] = x
            lam = np.clip(self.h(x), 1e-10, 1e10)
            ys[t] = rng.poisson(lam)

        return xs, ys
