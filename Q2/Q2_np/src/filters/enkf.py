"""Ensemble Kalman Filter (EnKF) implementation."""
import numpy as np


def enkf_update(particles, H, R, y, localization_matrix=None):
    """
    Ensemble Kalman Filter update step.

    Parameters
    ----------
    particles : ndarray [N, n_x]
        Prior ensemble members
    H : ndarray [n_y, n_x]
        Observation matrix
    R : ndarray [n_y, n_y]
        Observation noise covariance
    y : ndarray [n_y]
        Observation
    localization_matrix : ndarray [n_x, n_x], optional
        Localization matrix C for covariance tapering (B_loc = B * C)

    Returns
    -------
    particles_post : ndarray [N, n_x]
        Posterior ensemble members
    m_post : ndarray [n_x]
        Posterior mean
    P_post : ndarray [n_x, n_x]
        Posterior covariance
    """
    N, n_x = particles.shape
    n_y = len(y)

    # Ensemble mean and perturbations
    x_bar = np.mean(particles, axis=0)
    X = (particles - x_bar).T  # [n_x, N]

    # Sample covariance B = X @ X.T / (N - 1)
    B = X @ X.T / (N - 1)

    # Apply localization
    if localization_matrix is not None:
        B_loc = B * localization_matrix
    else:
        B_loc = B

    # Kalman gain: K = B_loc @ H.T @ (H @ B_loc @ H.T + R)^{-1}
    S = H @ B_loc @ H.T + R
    K = B_loc @ H.T @ np.linalg.solve(S, np.eye(n_y))

    # Update mean
    innovation = y - H @ x_bar
    m_post = x_bar + K @ innovation

    # Update covariance (Joseph form for stability)
    IKH = np.eye(n_x) - K @ H
    P_post = IKH @ B_loc @ IKH.T + K @ R @ K.T

    # Update particles (deterministic EnKF)
    # Each particle is updated: x_i^a = x_i^f + K @ (y - H @ x_i^f)
    particles_post = particles + (K @ (y[:, None] - H @ particles.T)).T

    return particles_post, m_post, P_post


def enkf_posterior_analytical(x_bar, B, H, R, y, localization_matrix=None):
    """
    Compute EnKF posterior mean and covariance analytically.

    Parameters
    ----------
    x_bar : ndarray [n_x]
        Prior mean
    B : ndarray [n_x, n_x]
        Prior covariance
    H : ndarray [n_y, n_x]
        Observation matrix
    R : ndarray [n_y, n_y]
        Observation noise covariance
    y : ndarray [n_y]
        Observation
    localization_matrix : ndarray [n_x, n_x], optional
        Localization matrix C for covariance tapering (B_loc = B * C)

    Returns
    -------
    m_post : ndarray [n_x]
        Posterior mean
    P_post : ndarray [n_x, n_x]
        Posterior covariance
    """
    n_x = len(x_bar)
    n_y = len(y)

    # Apply localization
    if localization_matrix is not None:
        B_loc = B * localization_matrix
    else:
        B_loc = B

    # Kalman gain
    S = H @ B_loc @ H.T + R
    K = B_loc @ H.T @ np.linalg.solve(S, np.eye(n_y))

    # Posterior mean
    innovation = y - H @ x_bar
    m_post = x_bar + K @ innovation

    # Posterior covariance
    P_post = (np.eye(n_x) - K @ H) @ B_loc

    return m_post, P_post
