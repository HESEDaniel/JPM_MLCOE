"""Range-Bearing State Space Model."""
import numpy as np


class RangeBearing:
    """Range-Bearing tracking model with constant velocity dynamics.

    State: [x, vx, y, vy] - 2D position and velocity
    Observations: [range, bearing] from sensor at sensor_pos

    Parameters
    ----------
    dt : float
        Time step
    q : float
        Process noise intensity
    r_range : float
        Range observation noise std
    r_bearing : float
        Bearing observation noise std (radians)
    sensor_pos : ndarray [2]
        Sensor position [x, y]
    """

    def __init__(self, dt=1.0, q=0.1, r_range=0.1, r_bearing=0.05, sensor_pos=None):
        """Initialize model with given parameters."""
        self.dt = dt
        self.q = q
        self.r_range = r_range
        self.r_bearing = r_bearing
        self.sensor_pos = sensor_pos if sensor_pos is not None else np.array([0.0, 0.0])

        # State transition (constant velocity)
        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])

        # Process noise (discrete white noise acceleration)
        self.Q = q**2 * np.array([
            [dt**4/4, dt**3/2, 0, 0],
            [dt**3/2, dt**2, 0, 0],
            [0, 0, dt**4/4, dt**3/2],
            [0, 0, dt**3/2, dt**2]
        ])

        # Observation noise
        self.R = np.diag([r_range**2, r_bearing**2])

        # Default initial state distribution
        self.m0 = np.array([5.0, 0.5, 5.0, 0.5])
        self.P0 = np.diag([0.5, 0.1, 0.5, 0.1])

    def set_initial(self, m0, P0):
        """Set initial state distribution."""
        self.m0 = m0
        self.P0 = P0

    def simulate(self, T, rng, x0=None):
        """Generate states and observations.

        Parameters
        ----------
        T : int
            Number of time steps
        rng : numpy.random.Generator
            Random number generator
        x0 : ndarray [4], optional
            Initial state. If None, uses self.m0.

        Returns
        -------
        xs : ndarray [T, 4]
            True states
        ys : ndarray [T, 2]
            Observations [range, bearing]
        """
        if x0 is None:
            x0 = self.m0.copy()

        xs = np.zeros((T, 4))
        ys = np.zeros((T, 2))
        x = x0.copy()

        for t in range(T):
            x = self.F @ x + rng.multivariate_normal(np.zeros(4), self.Q)
            y = self.h(x) + rng.multivariate_normal(np.zeros(2), self.R)
            y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # Wrap to [-pi, pi]
            xs[t], ys[t] = x, y

        return xs, ys

    def f(self, x):
        """State transition function."""
        return self.F @ x

    def h(self, x):
        """Observation function: [range, bearing].

        Parameters
        ----------
        x : ndarray [4]
            State [x, vx, y, vy]

        Returns
        -------
        y : ndarray [2]
            [range, bearing]
        """
        px = x[0] - self.sensor_pos[0]
        py = x[2] - self.sensor_pos[1]
        return np.array([np.sqrt(px**2 + py**2), np.arctan2(py, px)])

    def F_jac(self, x):
        """State transition Jacobian (constant)."""
        return self.F

    def H_jac(self, x):
        """Observation Jacobian: d[r, theta]/d[x, vx, y, vy].

        Parameters
        ----------
        x : ndarray [4]

        Returns
        -------
        H : ndarray [2, 4]
        """
        px = x[0] - self.sensor_pos[0]
        py = x[2] - self.sensor_pos[1]
        r = max(np.sqrt(px**2 + py**2), 1e-6)

        return np.array([
            [px/r, 0, py/r, 0],
            [-py/r**2, 0, px/r**2, 0]
        ])

    def log_likelihood(self, y, particles):
        """Log p(y|x) for particle filter.

        Parameters
        ----------
        y : ndarray [2]
            Observation [range, bearing]
        particles : ndarray [N, 4]
            Particle states

        Returns
        -------
        log_p : ndarray [N]
        """
        y_pred = np.array([self.h(p) for p in particles])

        # Handle bearing angle wrapping
        diff = y - y_pred
        diff[:, 1] = np.arctan2(np.sin(diff[:, 1]), np.cos(diff[:, 1]))

        # Gaussian likelihood
        log_p = -0.5 * np.sum(diff * (np.linalg.solve(self.R, diff.T)).T, axis=1)
        return log_p

    def Q_sampler(self, rng, N):
        """Sample process noise.

        Parameters
        ----------
        rng : numpy.random.Generator
        N : int
            Number of samples

        Returns
        -------
        noise : ndarray [N, 4]
        """
        return rng.multivariate_normal(np.zeros(4), self.Q, size=N)
