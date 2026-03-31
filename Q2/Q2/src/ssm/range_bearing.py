"""Range-Bearing SSM (TensorFlow)."""
import numpy as np
import tensorflow as tf

DTYPE = tf.float64


class RangeBearing:
    """Range-Bearing tracking: state [x, vx, y, vy], obs [range, bearing].

    Parameters
    ----------
    dt : float
        Discrete time step.
    q : float
        Process noise intensity.
    r_range : float
        Range measurement noise std.
    r_bearing : float
        Bearing measurement noise std.
    sensor_pos : array-like [2], optional
        Sensor position [x, y]. Defaults to [0, 0].
    """

    def __init__(self, dt=1.0, q=0.1, r_range=0.1, r_bearing=0.05, sensor_pos=None):
        self.dt = dt
        sp = np.array(sensor_pos if sensor_pos is not None else [0.0, 0.0])
        self._sensor_pos = tf.constant(sp, dtype=DTYPE)

        F_np = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                         [0, 0, 1, dt], [0, 0, 0, 1]])
        Q_np = q**2 * np.array([
            [dt**4/4, dt**3/2, 0, 0], [dt**3/2, dt**2, 0, 0],
            [0, 0, dt**4/4, dt**3/2], [0, 0, dt**3/2, dt**2]])
        R_np = np.diag([r_range**2, r_bearing**2])

        self._F = tf.constant(F_np, dtype=DTYPE)
        self.Q = tf.constant(Q_np, dtype=DTYPE)
        self.R = tf.constant(R_np, dtype=DTYPE)
        self._R_inv = tf.linalg.inv(self.R)

        # Precompute sqrt(Q) via eigendecomposition (Q is rank-2, Cholesky fails)
        eigvals, eigvecs = tf.linalg.eigh(self.Q)
        eigvals = tf.maximum(eigvals, 0.0)  # clamp numerical negatives
        self._Q_sqrt = eigvecs @ tf.linalg.diag(tf.sqrt(eigvals))
        self.m0 = tf.constant([5.0, 0.5, 5.0, 0.5], dtype=DTYPE)
        self.P0 = tf.constant(np.diag([0.5, 0.1, 0.5, 0.1]), dtype=DTYPE)
        self.state_dim = 4
        self.obs_dim = 2

    def set_initial(self, m0, P0):
        self.m0 = tf.constant(m0, dtype=DTYPE)
        self.P0 = tf.constant(P0, dtype=DTYPE)

    def f(self, x):
        return tf.linalg.matvec(self._F, x)

    def h(self, x):
        px = x[0] - self._sensor_pos[0]
        py = x[2] - self._sensor_pos[1]
        return tf.stack([tf.sqrt(px**2 + py**2), tf.math.atan2(py, px)])

    def F_jac(self, x):
        return self._F

    def H_jac(self, x):
        px = x[0] - self._sensor_pos[0]
        py = x[2] - self._sensor_pos[1]
        r = tf.maximum(tf.sqrt(px**2 + py**2), 1e-6)
        z = tf.constant(0.0, dtype=DTYPE)
        return tf.stack([
            tf.stack([px/r, z, py/r, z]),
            tf.stack([-py/r**2, z, px/r**2, z])])

    def simulate(self, T, rng, x0=None):
        F = self._F.numpy()
        Q_np = self.Q.numpy()
        R_np = self.R.numpy()
        sp = self._sensor_pos.numpy()
        if x0 is None:
            x0 = self.m0.numpy().copy()
        xs, ys = np.zeros((T, 4)), np.zeros((T, 2))
        x = x0.copy()
        for t in range(T):
            x = F @ x + rng.multivariate_normal(np.zeros(4), Q_np)
            px, py = x[0] - sp[0], x[2] - sp[1]
            h_x = np.array([np.sqrt(px**2 + py**2), np.arctan2(py, px)])
            y = h_x + rng.multivariate_normal(np.zeros(2), R_np)
            y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))
            xs[t], ys[t] = x, y
        return xs, ys

    def log_likelihood(self, y, particles):
        y_pred = tf.vectorized_map(self.h, particles)
        diff = y - y_pred
        wrapped = tf.math.atan2(tf.math.sin(diff[:, 1]), tf.math.cos(diff[:, 1]))
        diff = tf.concat([diff[:, :1], wrapped[:, tf.newaxis]], axis=1)
        R_inv_diff = tf.linalg.matvec(self._R_inv, diff)
        return -0.5 * tf.reduce_sum(diff * R_inv_diff, axis=1)

    def Q_sampler(self, rng, N):
        z = rng.normal(shape=(N, 4), dtype=DTYPE)
        return tf.linalg.matvec(self._Q_sqrt, z)

    def f_batch(self, particles):
        return particles @ tf.transpose(self._F)

    def h_batch(self, particles):
        return tf.vectorized_map(self.h, particles)
