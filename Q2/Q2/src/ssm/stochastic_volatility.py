"""Stochastic Volatility SSM (TensorFlow)."""
import numpy as np
import tensorflow as tf

DTYPE = tf.float64

EULER_GAMMA = 0.5772156649015328606
LOG_CHI2_MEAN = -EULER_GAMMA - np.log(2)
LOG_CHI2_VAR = np.pi**2 / 2


class SVLogTransformed:
    """SV with log-transformed observations (linear in x).

    State: x_t = alpha * x_{t-1} + sigma * v_t
    Original obs: y_t = beta * exp(x_t/2) * w_t
    Log-transformed: z_t = log(y_t^2) ~ log(beta^2) + x_t + log(chi^2_1)

    Parameters
    ----------
    alpha : float
        AR(1) persistence parameter.
    sigma : float
        Process noise std.
    beta : float
        Observation scale parameter.
    """

    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5):
        self.alpha = tf.constant(alpha, dtype=DTYPE)
        self.sigma = tf.constant(sigma, dtype=DTYPE)
        self.beta = tf.constant(beta, dtype=DTYPE)
        self.state_dim = 1
        self.obs_dim = 1
        self.Q = tf.constant([[sigma**2]], dtype=DTYPE)
        self.R = tf.constant([[LOG_CHI2_VAR]], dtype=DTYPE)
        self.m0 = tf.constant([0.0], dtype=DTYPE)
        self.P0 = tf.constant([[sigma**2 / (1 - alpha**2)]], dtype=DTYPE)
        self._log_beta2 = tf.constant(np.log(beta**2), dtype=DTYPE)
        self._log_chi2_mean = tf.constant(LOG_CHI2_MEAN, dtype=DTYPE)

    def f(self, x):
        return tf.stack([self.alpha * x[0]])

    def h(self, x):
        return tf.stack([self._log_beta2 + x[0] + self._log_chi2_mean])

    def F_jac(self, x):
        return tf.reshape(self.alpha, [1, 1])

    def H_jac(self, x):
        return tf.constant([[1.0]], dtype=DTYPE)

    def simulate(self, T, rng, x0=None):
        alpha = self.alpha.numpy()
        sigma = self.sigma.numpy()
        beta = self.beta.numpy()
        xs, ys = np.zeros(T), np.zeros(T)
        xs[0] = rng.normal(0, np.sqrt(sigma**2 / (1 - alpha**2)))
        ys[0] = beta * np.exp(xs[0] / 2) * rng.standard_normal()
        for t in range(1, T):
            xs[t] = alpha * xs[t-1] + sigma * rng.standard_normal()
            ys[t] = beta * np.exp(xs[t] / 2) * rng.standard_normal()
        return xs, ys

    def transform_obs(self, ys):
        return np.log(ys**2 + 1e-10).reshape(-1, 1)

    def log_likelihood(self, y, particles):
        vol = self.beta * tf.exp(particles[:, 0] / 2.0)
        return -0.5 * (y[0]**2 / vol**2 + tf.math.log(vol**2))

    def Q_sampler(self, rng, N):
        return self.sigma * rng.normal(shape=(N, 1), dtype=DTYPE)

    def f_batch(self, particles):
        return tf.vectorized_map(self.f, particles)

    def h_batch(self, particles):
        return tf.vectorized_map(self.h, particles)


class SVAdditiveNoise:
    """SV with additive observation noise.

    State: x_t = alpha * x_{t-1} + sigma * v_t
    Obs: y_t = beta * exp(exp_scale * x_t) + obs_std * w_t

    Parameters
    ----------
    alpha : float
        AR(1) persistence parameter.
    sigma : float
        Process noise std.
    beta : float
        Observation scale parameter.
    obs_std : float
        Additive observation noise std.
    exp_scale : float
        Exponent scaling factor.
    """

    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5, obs_std=0.5, exp_scale=0.5):
        self.alpha = tf.constant(alpha, dtype=DTYPE)
        self.sigma = tf.constant(sigma, dtype=DTYPE)
        self.beta = tf.constant(beta, dtype=DTYPE)
        self.obs_std = tf.constant(obs_std, dtype=DTYPE)
        self.exp_scale = tf.constant(exp_scale, dtype=DTYPE)
        self.state_dim = 1
        self.obs_dim = 1
        self.Q = tf.constant([[sigma**2]], dtype=DTYPE)
        self.R = tf.constant([[obs_std**2]], dtype=DTYPE)
        self.m0 = tf.constant([0.0], dtype=DTYPE)
        self.P0 = tf.constant([[sigma**2 / (1 - alpha**2)]], dtype=DTYPE)

    def f(self, x):
        return tf.stack([self.alpha * x[0]])

    def h(self, x):
        return tf.stack([self.beta * tf.exp(self.exp_scale * x[0])])

    def F_jac(self, x):
        return tf.reshape(self.alpha, [1, 1])

    def H_jac(self, x):
        val = self.beta * self.exp_scale * tf.exp(self.exp_scale * x[0])
        return tf.reshape(val, [1, 1])

    def simulate(self, T, rng, x0=None):
        alpha = self.alpha.numpy()
        sigma = self.sigma.numpy()
        beta = self.beta.numpy()
        obs_std = self.obs_std.numpy()
        exp_scale = self.exp_scale.numpy()
        xs, ys = np.zeros(T), np.zeros(T)
        xs[0] = rng.normal(0, np.sqrt(sigma**2 / (1 - alpha**2)))
        ys[0] = beta * np.exp(exp_scale * xs[0]) + obs_std * rng.standard_normal()
        for t in range(1, T):
            xs[t] = alpha * xs[t-1] + sigma * rng.standard_normal()
            ys[t] = beta * np.exp(exp_scale * xs[t]) + obs_std * rng.standard_normal()
        return xs, ys

    def transform_obs(self, ys):
        return ys.reshape(-1, 1)

    def log_likelihood(self, y, particles):
        h_val = self.beta * tf.exp(self.exp_scale * particles[:, 0])
        return -0.5 * ((y[0] - h_val)**2 / self.obs_std**2)

    def Q_sampler(self, rng, N):
        return self.sigma * rng.normal(shape=(N, 1), dtype=DTYPE)

    def f_batch(self, particles):
        return tf.vectorized_map(self.f, particles)

    def h_batch(self, particles):
        return tf.vectorized_map(self.h, particles)
