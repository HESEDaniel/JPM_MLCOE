"""Linear Gaussian SSM from Corenflos et al. (2021) Section 5.

Eq. 16: X_{t+1} | X_t=x  ~  N(A x, I_{d_x})
Eq. 17: Y_t     | X_t=x  ~  N(H x, sigma_obs^2 I_{d_y})

A_{ij} = base^{|i-j|+1},  H = I_{d_y, d_x}  (first d_y rows of identity)
"""
import numpy as np
import tensorflow as tf


class CorenflosLGSSM:
    """Corenflos(2021) linear Gaussian SSM.

    Parameters
    ----------
    d_x : int
        State dimension.
    d_y : int
        Observation dimension (d_y <= d_x).
    base : float
        Base for transition matrix: A_{ij} = base^{|i-j|+1}.
    sigma_obs : float
        Observation noise std (R = sigma_obs^2 * I).
    dtype : tf.DType
        Tensor dtype (default float32, matching author code).
    """

    def __init__(self, d_x=25, d_y=1, base=0.42, sigma_obs=0.316, dtype=tf.float32):
        self.state_dim = d_x
        self.obs_dim = d_y
        self.dtype = dtype

        idx = tf.range(d_x, dtype=dtype)
        diff = tf.abs(idx[:, tf.newaxis] - idx[tf.newaxis, :])
        self.A = tf.constant(base, dtype=dtype) ** (diff + 1.0)

        self.H = tf.eye(d_x, dtype=dtype)[:d_y, :]

        self.Q = tf.eye(d_x, dtype=dtype)
        self.R = (sigma_obs ** 2) * tf.eye(d_y, dtype=dtype)

        self.Q_inv = tf.eye(d_x, dtype=dtype)
        self.R_inv = tf.linalg.inv(self.R)
        self.log_det_Q = tf.constant(0.0, dtype=dtype)
        self.log_det_R = tf.linalg.slogdet(self.R)[1]
        self._L_Q = tf.linalg.cholesky(self.Q)

        self.m0 = tf.zeros(d_x, dtype=dtype)
        self.P0 = tf.eye(d_x, dtype=dtype)

    def f(self, x):
        return tf.linalg.matvec(self.A, x)

    def h(self, x):
        return tf.linalg.matvec(self.H, x)

    def F_jac(self, x):
        return self.A

    def H_jac(self, x):
        return self.H

    def f_batch(self, particles):
        return particles @ tf.transpose(self.A)

    def h_batch(self, particles):
        return particles @ tf.transpose(self.H)

    def Q_sampler(self, rng, N):
        return rng.normal(shape=(N, self.state_dim), dtype=self.dtype)

    def log_likelihood(self, y, particles):
        y_pred = self.h_batch(particles)
        diff = y - y_pred
        mahal = tf.reduce_sum(diff * tf.linalg.matvec(self.R_inv, diff), axis=1)
        return -0.5 * (self.log_det_R
                       + tf.cast(self.obs_dim, self.dtype) * tf.math.log(tf.constant(2.0 * np.pi, dtype=self.dtype))
                       + mahal)

    def simulate(self, T, rng, m0=None):
        """Simulate trajectory (numpy).

        Parameters
        ----------
        T : int
        rng : numpy rng

        Returns
        -------
        xs : [T, d_x], ys : [T, d_y]
        """
        A_np = self.A.numpy()
        H_np = self.H.numpy()
        Q_np = self.Q.numpy()
        R_np = self.R.numpy()
        d_x, d_y = self.state_dim, self.obs_dim

        xs = np.zeros((T, d_x))
        ys = np.zeros((T, d_y))
        x = rng.multivariate_normal(np.zeros(d_x), Q_np) if m0 is None else m0.copy()

        for t in range(T):
            x = A_np @ x + rng.multivariate_normal(np.zeros(d_x), Q_np)
            y = H_np @ x + rng.multivariate_normal(np.zeros(d_y), R_np)
            xs[t] = x
            ys[t] = y

        return xs, ys
