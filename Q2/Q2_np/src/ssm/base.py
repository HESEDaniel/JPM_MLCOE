"""Base State Space Model interface."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

DTYPE = tf.float64


def _ensure_tf(x, dtype=DTYPE):
    """Convert numpy array to tf.Tensor if needed."""
    if isinstance(x, tf.Tensor):
        return x
    return tf.constant(x, dtype=dtype)


class BaseSSM(ABC):
    """Abstract base class for all state space models.

    Properties Q, R, m0, P0 return numpy arrays for backward compatibility
    with the legacy functional API. The OOP filter/flow classes use
    _ensure_tf() to convert them when needed.

    Methods f, h, F_jac, H_jac accept both numpy and tf inputs.
    Subclasses must implement: state_dim, obs_dim, Q, R, m0, P0,
    f, h, F_jac, H_jac, simulate.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state vector."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of observation vector."""

    @property
    @abstractmethod
    def Q(self) -> np.ndarray:
        """Process noise covariance [n_x, n_x]."""

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Observation noise covariance [n_y, n_y]."""

    @property
    @abstractmethod
    def m0(self) -> np.ndarray:
        """Initial state mean [n_x]."""

    @property
    @abstractmethod
    def P0(self) -> np.ndarray:
        """Initial state covariance [n_x, n_x]."""

    # TF tensor properties (for OOP filters)

    @property
    def Q_tf(self) -> tf.Tensor:
        return _ensure_tf(self.Q)

    @property
    def R_tf(self) -> tf.Tensor:
        return _ensure_tf(self.R)

    @property
    def m0_tf(self) -> tf.Tensor:
        return _ensure_tf(self.m0)

    @property
    def P0_tf(self) -> tf.Tensor:
        return _ensure_tf(self.P0)

    @abstractmethod
    def f(self, x) -> Union[np.ndarray, tf.Tensor]:
        """State transition: x_{t+1} = f(x_t) + noise.

        Parameters
        ----------
        x : array-like [n_x]

        Returns
        -------
        array-like [n_x]
        """

    @abstractmethod
    def h(self, x) -> Union[np.ndarray, tf.Tensor]:
        """Observation function: y_t = h(x_t) + noise.

        Parameters
        ----------
        x : array-like [n_x]

        Returns
        -------
        array-like [n_y]
        """

    @abstractmethod
    def F_jac(self, x) -> Union[np.ndarray, tf.Tensor]:
        """Jacobian of f at x.

        Parameters
        ----------
        x : array-like [n_x]

        Returns
        -------
        array-like [n_x, n_x]
        """

    @abstractmethod
    def H_jac(self, x) -> Union[np.ndarray, tf.Tensor]:
        """Jacobian of h at x.

        Parameters
        ----------
        x : array-like [n_x]

        Returns
        -------
        array-like [n_y, n_x]
        """

    @abstractmethod
    def simulate(self, T: int, rng: np.random.Generator,
                 x0: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic states and observations.

        Parameters
        ----------
        T : int
            Number of time steps.
        rng : np.random.Generator
            NumPy random generator.
        x0 : ndarray [n_x], optional
            Initial state.

        Returns
        -------
        xs : ndarray [T, n_x]
        ys : ndarray [T, n_y]
        """

    def Q_sampler(self, rng, N: int):
        """Sample N process noise vectors.

        Supports both np.random.Generator and tf.random.Generator.
        """
        if isinstance(rng, np.random.Generator):
            return rng.multivariate_normal(
                np.zeros(self.state_dim), np.asarray(self.Q), size=N
            )
        L_Q = tf.linalg.cholesky(_ensure_tf(self.Q))
        z = rng.normal(shape=(N, self.state_dim), dtype=DTYPE)
        return tf.linalg.matvec(L_Q, z)

    def log_likelihood(self, y, particles):
        """Log p(y|x) for each particle. Default: Gaussian with covariance R.

        Supports both numpy and tf inputs.
        """
        if isinstance(particles, np.ndarray):
            y_pred = np.array([np.asarray(self.h(p)) for p in particles])
            diff = np.asarray(y) - y_pred
            R_inv_diff = np.linalg.solve(np.asarray(self.R), diff.T).T
            return -0.5 * np.sum(diff * R_inv_diff, axis=1)

        particles_tf = _ensure_tf(particles)
        y_tf = _ensure_tf(y)
        y_pred = tf.vectorized_map(self.h, particles_tf)
        diff = y_tf - y_pred
        R_tf = _ensure_tf(self.R)
        R_inv_diff = tf.linalg.solve(R_tf, tf.transpose(diff))
        return -0.5 * tf.reduce_sum(diff * tf.transpose(R_inv_diff), axis=1)

    def f_batch(self, particles):
        """Batch state transition for N particles (TF)."""
        return tf.vectorized_map(self.f, _ensure_tf(particles))

    def h_batch(self, particles):
        """Batch observation for N particles (TF)."""
        return tf.vectorized_map(self.h, _ensure_tf(particles))
