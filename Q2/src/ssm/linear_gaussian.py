"""Linear Gaussian State Space Model (LGSSM)."""
import numpy as np


def linear_gaussian_ssm(A, B, C, D, Sigma, T, rng):
    """
    Simulate Linear Gaussian SSM.

    Parameters
    ----------
    A : ndarray [n_x, n_x]
        State transition matrix
    B : ndarray [n_x, n_v]
        Process noise coefficient
    C : ndarray [n_y, n_x]
        Observation matrix
    D : ndarray [n_y, n_w]
        Observation noise coefficient
    Sigma : ndarray [n_x, n_x]
        Initial state covariance
    T : int
        Number of time steps

    Returns
    -------
    xs : ndarray [T, n_x]
        Latent states
    ys : ndarray [T, n_y]
        Observations
    """
    n_x, n_y = A.shape[0], C.shape[0]
    n_v, n_w = B.shape[1], D.shape[1]

    x = rng.multivariate_normal(np.zeros(n_x), Sigma)

    xs = np.zeros((T, n_x))
    ys = np.zeros((T, n_y))

    for t in range(T):
        x = A @ x + B @ rng.standard_normal(n_v)
        y = C @ x + D @ rng.standard_normal(n_w)
        xs[t], ys[t] = x, y

    return xs, ys
