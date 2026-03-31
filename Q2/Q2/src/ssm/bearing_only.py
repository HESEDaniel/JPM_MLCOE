"""2D Bearing-Only SSM from Dai(22) Section 4."""
import numpy as np
import tensorflow as tf

DTYPE = tf.float64


class BearingOnly2D:
    """2D bearing-only tracking with 2 passive infrared sensors.

    State: x = [x_t, y_t] (2D position, stationary target)
    Observation: h_i = arctan((y_t - y_si) / (x_t - x_si)), i=1,2

    From Dai(22) Section 4:
    - True location: (4, 4)
    - Sensor 1 at (-3.5, 0), Sensor 2 at (3.5, 0)
      (ordering verified against paper's measurement z)
    - Prior: N([3.0, 5.0], diag(1000.0, 2.0))
    - R = diag(0.04, 0.04)
    - Fixed measurement z = [0.4754, 1.1868]
    """

    def __init__(self, sensor_pos=None, m_prior=None, P_prior=None,
                 R_diag=None, true_state=None):
        """
        Parameters
        ----------
        sensor_pos : list of [x, y], optional
            Sensor positions. Defaults to [[-3.5, 0], [3.5, 0]].
        m_prior : list [n_x], optional
            Prior mean. Defaults to [3.0, 5.0].
        P_prior : list [n_x, n_x], optional
            Prior covariance. Defaults to diag(1000, 2).
        R_diag : list [n_y], optional
            Observation noise variance diagonal. Defaults to [0.04, 0.04].
        true_state : list [n_x], optional
            True target position. Defaults to [4.0, 4.0].
        """
        if sensor_pos is None:
            sensor_pos = [[-3.5, 0.0], [3.5, 0.0]]
        if m_prior is None:
            m_prior = [3.0, 5.0]
        if P_prior is None:
            P_prior = [[1000.0, 0.0], [0.0, 2.0]]
        if R_diag is None:
            R_diag = [0.04, 0.04]
        if true_state is None:
            true_state = [4.0, 4.0]

        self._sensor_pos = tf.constant(sensor_pos, dtype=DTYPE)
        self._true_state = tf.constant(true_state, dtype=DTYPE)
        self.state_dim = 2
        self.obs_dim = 2
        self.m0 = tf.constant(m_prior, dtype=DTYPE)
        self.P0 = tf.constant(P_prior, dtype=DTYPE)
        self.R = tf.linalg.diag(tf.constant(R_diag, dtype=DTYPE))
        self.Q = tf.zeros((2, 2), dtype=DTYPE)
        self.P_prior_inv = tf.linalg.inv(self.P0)

    def f(self, x):
        """No dynamics (stationary target)."""
        return x

    def h(self, x):
        """Bearing observations: h_i = arctan((y - y_si) / (x - x_si)).

        Uses tf.math.atan (NOT atan2) to match paper's arctan convention.
        """
        dx1 = x[0] - self._sensor_pos[0, 0]
        dy1 = x[1] - self._sensor_pos[0, 1]
        dx2 = x[0] - self._sensor_pos[1, 0]
        dy2 = x[1] - self._sensor_pos[1, 1]
        return tf.stack([tf.math.atan(dy1 / dx1), tf.math.atan(dy2 / dx2)])

    def F_jac(self, x):
        return tf.eye(2, dtype=DTYPE)

    def H_jac(self, x):
        """Jacobian: d(arctan(dy/dx))/d[x_t, y_t] = [-dy/r^2, dx/r^2]."""
        rows = []
        for s in range(2):
            dx = x[0] - self._sensor_pos[s, 0]
            dy = x[1] - self._sensor_pos[s, 1]
            r2 = tf.maximum(dx**2 + dy**2, tf.constant(1e-12, dtype=DTYPE))
            rows.append(tf.stack([-dy / r2, dx / r2]))
        return tf.stack(rows)

    def f_batch(self, particles):
        return particles  # stationary target

    def h_batch(self, particles):
        return tf.vectorized_map(self.h, particles)

    def Q_sampler(self, rng, N):
        return tf.zeros((N, self.state_dim), dtype=DTYPE)  # Q = 0

    def simulate(self, T, rng, x0=None):
        """Generate bearing measurements from true state."""
        if x0 is None:
            x0 = self._true_state.numpy()
        xs = np.tile(x0, (T, 1))
        ys = np.zeros((T, 2))
        R_np = self.R.numpy()
        for t in range(T):
            h_val = self.h(tf.constant(x0, dtype=DTYPE)).numpy()
            ys[t] = h_val + rng.multivariate_normal(np.zeros(2), R_np)
        return xs, ys

    def generate_fixed_measurement(self):
        """Return the fixed measurement from Dai(22) Section 4."""
        return tf.constant([0.4754, 1.1868], dtype=DTYPE)
