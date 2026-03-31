"""Skewed-t Dynamics with Poisson Measurements SSM (TensorFlow)."""
import numpy as np
import tensorflow as tf

DTYPE = tf.float64


class SkewedTPoissonSSM:
    """GH Skewed-t dynamics with Poisson count measurements."""

    def __init__(self, d=144, alpha=0.9, alpha0=3.0, alpha1=0.01,
                 beta=20.0, gamma_val=0.3, nu=7.0, m1=1.0, m2=1/3):
        """Initialize model with d-dimensional state on a grid."""
        self.d = d
        self.state_dim = d
        self.obs_dim = d
        self.alpha = alpha
        self.m1 = m1
        self.m2 = m2
        self.nu = nu

        grid_size = int(np.sqrt(d))
        if grid_size ** 2 != d:
            raise ValueError(f"d={d} must be a perfect square.")

        # Sensor grid and spatial covariance (Eq. 42)
        i, j = np.meshgrid(np.arange(1, grid_size + 1),
                           np.arange(1, grid_size + 1))
        positions = np.column_stack([i.ravel(), j.ravel()])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        Sigma_np = alpha0 * np.exp(-sq_dist / beta) + alpha1 * np.eye(d)

        # Skewness vector
        gamma_np = gamma_val * np.ones(d)

        # Effective covariance Sigma_tilde (Eq. 44)
        coeff1 = nu / max(nu - 2, 0.1)
        coeff2 = (nu ** 2) / ((2 * nu - 8) * ((nu / 2 - 1) ** 2)) if nu > 4 else 0.0
        Sigma_tilde_np = coeff1 * Sigma_np + coeff2 * np.outer(gamma_np, gamma_np)

        # Store numpy versions for simulate / sample_skewed_t
        self._Sigma_np = Sigma_np
        self._gamma_np = gamma_np
        self._L_Sigma_np = np.linalg.cholesky(Sigma_np)

        # TF constants
        self.Sigma = tf.constant(Sigma_np, dtype=DTYPE)
        self.Sigma_tilde = tf.constant(Sigma_tilde_np, dtype=DTYPE)
        self.gamma = tf.constant(gamma_np, dtype=DTYPE)
        self.A = tf.constant(alpha * np.eye(d), dtype=DTYPE)

        self.Q = tf.constant(Sigma_tilde_np, dtype=DTYPE)
        self._L_Q = tf.linalg.cholesky(
            self.Q + tf.constant(1e-8, dtype=DTYPE) * tf.eye(d, dtype=DTYPE)
        )

        R_np = np.diag(m1 * np.ones(d))
        self._R = tf.constant(R_np, dtype=DTYPE)

        # Initial state distribution
        self.m0 = tf.constant(np.zeros(d), dtype=DTYPE)
        self.P0 = tf.constant(Sigma_tilde_np, dtype=DTYPE)

    def f(self, x):
        """State transition mean: alpha * x."""
        return tf.constant(self.alpha, dtype=DTYPE) * x

    def h(self, x):
        """Observation function: Poisson rate m1 * exp(m2 * x)."""
        return tf.constant(self.m1, dtype=DTYPE) * tf.exp(
            tf.constant(self.m2, dtype=DTYPE) * x
        )

    def F_jac(self, x):
        """Jacobian of f (constant): alpha * I."""
        return self.A

    def H_jac(self, x):
        """Jacobian of h: diag(m1 * m2 * exp(m2 * x))."""
        return tf.linalg.diag(
            tf.constant(self.m1 * self.m2, dtype=DTYPE) * tf.exp(
                tf.constant(self.m2, dtype=DTYPE) * x
            )
        )

    def R_state_dependent(self, x):
        """State-dependent R: Poisson variance = mean."""
        return tf.linalg.diag(self.h(x))

    @property
    def R(self):
        """Default R at x=0."""
        return self._R

    def log_likelihood(self, y, particles):
        """Poisson log-likelihood for each particle.

        Args:
            y: observation vector, shape (d,)
            particles: shape (N, d)

        Returns:
            log-likelihoods, shape (N,)
        """
        lam = tf.clip_by_value(
            tf.constant(self.m1, dtype=DTYPE) * tf.exp(
                tf.constant(self.m2, dtype=DTYPE) * particles
            ),
            1e-10, 1e10,
        )
        return tf.reduce_sum(
            y * tf.math.log(lam) - lam - tf.math.lgamma(y + 1.0),
            axis=1,
        )

    def Q_sampler(self, rng, N):
        """Sample N process noise from N(0, Sigma_tilde).

        Args:
            rng: tf.random.Generator
            N: number of samples

        Returns:
            noise samples, shape (N, d)
        """
        z = rng.normal(shape=(N, self.d), dtype=DTYPE)
        return tf.linalg.matvec(self._L_Q, z)

    def sample_skewed_t(self, mu, rng):
        """Sample from GH Skewed-t: X = mu + W*gamma + sqrt(W)*Z.

        Uses numpy (called only from simulate).

        Args:
            mu: mean vector (numpy), shape (d,)
            rng: np.random.Generator

        Returns:
            sample, shape (d,)
        """
        W = 1.0 / rng.gamma(self.nu / 2, 2.0 / self.nu)
        Z = rng.standard_normal(self.d) @ self._L_Sigma_np.T
        return mu + W * self._gamma_np + np.sqrt(W) * Z

    def simulate(self, T, rng, x0=None):
        """Simulate T steps with skewed-t dynamics and Poisson observations.

        Uses numpy (data generation).

        Args:
            T: number of time steps
            rng: np.random.Generator
            x0: optional initial state (numpy), shape (d,)

        Returns:
            xs: states, shape (T, d)
            ys: observations, shape (T, d)
        """
        x = np.zeros(self.d) if x0 is None else x0.copy()
        xs = np.zeros((T, self.d))
        ys = np.zeros((T, self.d))

        for t in range(T):
            x = self.sample_skewed_t(self.alpha * x, rng)
            xs[t] = x
            lam = np.clip(self.m1 * np.exp(self.m2 * x), 1e-10, 1e10)
            ys[t] = rng.poisson(lam)

        return xs, ys

    def f_batch(self, particles):
        """Batch state transition.

        Args:
            particles: shape (N, d)

        Returns:
            transitioned particles, shape (N, d)
        """
        return tf.vectorized_map(self.f, particles)

    def h_batch(self, particles):
        """Batch observation function.

        Args:
            particles: shape (N, d)

        Returns:
            observation predictions, shape (N, d)
        """
        return tf.vectorized_map(self.h, particles)
