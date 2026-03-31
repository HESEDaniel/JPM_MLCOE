"""Lorenz 96 State Space Model."""
import numpy as np


def lorenz96_rhs(x, F):
    """Lorenz 96 ODE right-hand side."""
    xm2 = np.roll(x, 2)
    xm1 = np.roll(x, 1)
    xp1 = np.roll(x, -1)
    return (xp1 - xm2) * xm1 - x + F


def lorenz96_step(x, F, dt):
    """RK4 integration step."""
    k1 = lorenz96_rhs(x, F)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def lorenz96_ssm(T, rng, K=40, F=8.0, dt=0.05, obs_every=1, Q_std=0.0, R_std=1.0):
    """
    Simulate Lorenz 96 SSM.

    Parameters
    ----------
    T : int
        Number of time steps
    K : int
        State dimension
    F : float
        Forcing parameter (F=8 gives chaos)
    dt : float
        Integration time step
    obs_every : int
        Observe every m-th variable (1 = all)
    Q_std, R_std : float
        Process/observation noise std

    Returns
    -------
    xs : ndarray [T, K]
        True states
    ys : ndarray [T, n_obs]
        Observations
    H : ndarray [n_obs, K]
        Observation matrix
    Q, R : ndarray
        Process/observation noise covariances
    """
    # Observation matrix
    obs_idx = np.arange(0, K, obs_every)
    n_obs = len(obs_idx)
    H = np.zeros((n_obs, K))
    for i, j in enumerate(obs_idx):
        H[i, j] = 1.0

    Q = (Q_std ** 2) * np.eye(K)
    R = (R_std ** 2) * np.eye(n_obs)

    # Spinup to attractor
    x = F * np.ones(K)
    x[0] += 0.01
    for _ in range(1000):
        x = lorenz96_step(x, F, dt)

    # Simulate
    xs = np.zeros((T, K))
    ys = np.zeros((T, n_obs))

    for t in range(T):
        x = lorenz96_step(x, F, dt)
        if Q_std > 0:
            x += rng.normal(0, Q_std, K)
        y = H @ x + rng.normal(0, R_std, n_obs)
        xs[t], ys[t] = x, y

    return xs, ys, H, Q, R
