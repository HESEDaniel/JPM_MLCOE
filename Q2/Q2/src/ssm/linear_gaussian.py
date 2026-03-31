"""Linear Gaussian SSM (TensorFlow)."""
import numpy as np
import tensorflow as tf

DTYPE = tf.float64


class LinearGaussianSSM:
    """Linear Gaussian SSM: x_{t+1} = A x_t + B v_t, y_t = C x_t + D w_t.

    Parameters
    ----------
    A : array-like [n_x, n_x]
        State transition matrix.
    B : array-like [n_x, n_v]
        Process noise input matrix (Q = B B^T).
    C : array-like [n_y, n_x]
        Observation matrix.
    D : array-like [n_y, n_w]
        Observation noise input matrix (R = D D^T).
    Sigma : array-like [n_x, n_x]
        Initial state covariance P0.
    """

    def __init__(self, A, B, C, D, Sigma):
        self._B = tf.constant(B, dtype=DTYPE)
        self._D = tf.constant(D, dtype=DTYPE)
        self.A = tf.constant(A, dtype=DTYPE)
        self.C = tf.constant(C, dtype=DTYPE)
        self.Q = tf.linalg.matmul(self._B, self._B, transpose_b=True)
        self.R = tf.linalg.matmul(self._D, self._D, transpose_b=True)
        self.P0 = tf.constant(Sigma, dtype=DTYPE)
        self.state_dim = int(self.A.shape[0])
        self.obs_dim = int(self.C.shape[0])
        self.m0 = tf.zeros(self.state_dim, dtype=DTYPE)

    def f(self, x):
        return tf.linalg.matvec(self.A, x)

    def h(self, x):
        return tf.linalg.matvec(self.C, x)

    def F_jac(self, x):
        return self.A

    def H_jac(self, x):
        return self.C

    def f_batch(self, particles):
        return particles @ tf.transpose(self.A)

    def h_batch(self, particles):
        return particles @ tf.transpose(self.C)

    def Q_sampler(self, rng, N):
        L_Q = tf.linalg.cholesky(self.Q)
        z = rng.normal(shape=(N, self.state_dim), dtype=DTYPE)
        return tf.linalg.matvec(L_Q, z)

    def log_likelihood(self, y, particles):
        y_pred = self.h_batch(particles)
        diff = y - y_pred
        R_inv_diff = tf.linalg.solve(self.R, tf.transpose(diff))
        return -0.5 * tf.reduce_sum(diff * tf.transpose(R_inv_diff), axis=1)

    def simulate(self, T, rng, x0=None):
        A, B = self.A.numpy(), self._B.numpy()
        C, D = self.C.numpy(), self._D.numpy()
        n_v, n_w = B.shape[1], D.shape[1]

        x = rng.multivariate_normal(np.zeros(self.state_dim), self.P0.numpy())
        xs = np.zeros((T, self.state_dim))
        ys = np.zeros((T, self.obs_dim))

        for t in range(T):
            x = A @ x + B @ rng.standard_normal(n_v)
            y = C @ x + D @ rng.standard_normal(n_w)
            xs[t], ys[t] = x, y

        return xs, ys
