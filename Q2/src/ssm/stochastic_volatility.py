"""Stochastic Volatility State Space Model."""
import numpy as np

# Log-chi-squared distribution constants
EULER_GAMMA = 0.5772156649015328606
LOG_CHI2_MEAN = -EULER_GAMMA - np.log(2)
LOG_CHI2_VAR = np.pi**2 / 2


class SVLogTransformed:
    """Stochastic Volatility with log-transformed observations (linear in x)."""

    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5):
        """Initialize SV model with given parameters."""
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta

        self.Q = np.array([[sigma**2]])
        self.R = np.array([[LOG_CHI2_VAR]])
        self.m0 = np.array([0.0])
        self.P0 = np.array([[sigma**2 / (1 - alpha**2)]])

    def simulate(self, T, rng):
        """Generate states and observations."""
        xs = np.zeros(T)
        ys = np.zeros(T)

        xs[0] = rng.normal(0, np.sqrt(self.P0[0, 0]))
        ys[0] = self.beta * np.exp(xs[0] / 2) * rng.standard_normal()

        for t in range(1, T):
            xs[t] = self.alpha * xs[t-1] + self.sigma * rng.standard_normal()
            ys[t] = self.beta * np.exp(xs[t] / 2) * rng.standard_normal()

        return xs, ys

    def transform_obs(self, ys):
        """Transform observations."""
        return np.log(ys**2 + 1e-10).reshape(-1, 1)

    def f(self, x):
        """State transition."""
        return np.array([self.alpha * x[0]])

    def h(self, x):
        """Observation function."""
        return np.array([np.log(self.beta**2) + x[0] + LOG_CHI2_MEAN])

    def F_jac(self, x):
        """State transition Jacobian."""
        return np.array([[self.alpha]])

    def H_jac(self, x):
        """Observation Jacobian."""
        return np.array([[1.0]])

    def log_likelihood(self, y, particles):
        """Log p(y|x) for original observations."""
        vol = self.beta * np.exp(particles[:, 0] / 2)
        return -0.5 * (y[0]**2 / vol**2 + np.log(vol**2))

    def Q_sampler(self, rng, N):
        """Sample process noise."""
        return self.sigma * rng.standard_normal((N, 1))


class SVAdditiveNoise:
    """Stochastic Volatility with additive observation noise."""

    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5, obs_std=0.5, exp_scale=0.5):
        """Initialize SV model with given parameters."""
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.obs_std = obs_std
        self.exp_scale = exp_scale

        self.Q = np.array([[sigma**2]])
        self.R = np.array([[obs_std**2]])
        self.m0 = np.array([0.0])
        self.P0 = np.array([[sigma**2 / (1 - alpha**2)]])

    def simulate(self, T, rng):
        """Generate states and observations."""
        xs = np.zeros(T)
        ys = np.zeros(T)

        xs[0] = rng.normal(0, np.sqrt(self.P0[0, 0]))
        ys[0] = self.beta * np.exp(self.exp_scale * xs[0]) + self.obs_std * rng.standard_normal()

        for t in range(1, T):
            xs[t] = self.alpha * xs[t-1] + self.sigma * rng.standard_normal()
            ys[t] = self.beta * np.exp(self.exp_scale * xs[t]) + self.obs_std * rng.standard_normal()

        return xs, ys

    def transform_obs(self, ys):
        """No transformation needed."""
        return ys.reshape(-1, 1)

    def f(self, x):
        """State transition."""
        return np.array([self.alpha * x[0]])

    def h(self, x):
        """Observation function."""
        return np.array([self.beta * np.exp(self.exp_scale * x[0])])

    def F_jac(self, x):
        """State transition Jacobian."""
        return np.array([[self.alpha]])

    def H_jac(self, x):
        """Observation Jacobian."""
        return np.array([[self.beta * self.exp_scale * np.exp(self.exp_scale * x[0])]])

    def log_likelihood(self, y, particles):
        """Log p(y|x) for additive Gaussian noise."""
        h_val = self.beta * np.exp(self.exp_scale * particles[:, 0])
        return -0.5 * ((y[0] - h_val)**2 / self.obs_std**2)

    def Q_sampler(self, rng, N):
        """Sample process noise."""
        return self.sigma * rng.standard_normal((N, 1))
