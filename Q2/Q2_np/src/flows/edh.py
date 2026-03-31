"""Exact Daum-Huang (EDH) particle flow."""
import numpy as np
from .flow_utils import get_lambda_schedule, propagate_particles, predict_step, update_step


def compute_edh_matrices(m, P, H, R, y, lam, eta_bar, h):
    """
    Compute A(lambda) and b(lambda) for EDH flow (Li et al. 2017, Eq. 10-11).

    Parameters
    ----------
    m : ndarray [n_x]
        Prior mean
    P : ndarray [n_x, n_x]
        Prior covariance
    H : ndarray [n_y, n_x]
        Observation Jacobian at eta_bar
    R : ndarray [n_y, n_y]
        Observation noise covariance
    y : ndarray [n_y]
        Observation
    lam : float
        Pseudo-time in [0, 1]
    eta_bar : ndarray [n_x]
        Mean of particles at current lambda
    h : callable
        Observation function

    Returns
    -------
    A : ndarray [n_x, n_x]
    b : ndarray [n_x]
    """
    n_x = len(m)
    I = np.eye(n_x)

    S_lam = lam * (H @ P @ H.T) + R
    S_inv_H = np.linalg.solve(S_lam, H)
    A = -0.5 * P @ H.T @ S_inv_H

    e_lam = h(eta_bar) - H @ eta_bar
    y_corr = y - e_lam
    R_inv_y = np.linalg.solve(R, y_corr)

    term1 = (I + lam * A) @ P @ H.T @ R_inv_y
    term2 = A @ m
    b = (I + 2 * lam * A) @ (term1 + term2)

    return A, b


def exact_daum_huang_flow(particles, m_prev, P_prev, f, F_jacobian, Q, h, H_jac, R, y,
                          n_steps=20, lambda_schedule=None, redraw=True, rng=None,
                          filter_type='ukf', alpha_ukf=1e-3, beta_ukf=2.0, kappa=0.0,
                          joseph=True, angle_indices=None):
    """
    Apply EDH flow to transport particles from prior to posterior.

    Parameters
    ----------
    particles : ndarray [N, n_x]
        Input particles
    m_prev, P_prev : ndarray
        Previous posterior mean/covariance
    f : callable
        State transition function
    F_jacobian : callable
        Jacobian of f (for EKF)
    Q : ndarray [n_x, n_x]
        Process noise covariance
    h : callable
        Observation function
    H_jac : callable
        Jacobian of h
    R : ndarray [n_y, n_y]
        Observation noise covariance
    y : ndarray [n_y]
        Current observation
    n_steps : int
        Number of integration steps
    lambda_schedule : ndarray, optional
        Custom lambda positions [0, ..., 1]. If None, uses uniform spacing.
    redraw : bool
        Redraw particles from posterior after flow
    filter_type : str
        'ekf' or 'ukf'
    joseph : bool
        Use Joseph stabilized covariance update
    angle_indices : list[int]
        Indices of angular observations

    Returns
    -------
    particles : ndarray [N, n_x]
    log_det_J : ndarray [N]
    flow_history : ndarray [n_steps+1, N, n_x]
    m_post : ndarray [n_x]
    P_post : ndarray [n_x, n_x]
    """
    N, n_x = particles.shape

    lam_pos, n_steps = get_lambda_schedule(n_steps, lambda_schedule)
    particles = propagate_particles(particles, f, Q, rng)
    m_pred, P_pred = predict_step(m_prev, P_prev, f, F_jacobian, Q, filter_type,
                                  alpha_ukf, beta_ukf, kappa)

    x = particles.copy()
    flow_history = np.zeros((n_steps + 1, N, n_x))
    flow_history[0] = x
    log_det_J = np.zeros(N)

    # Flow integration
    for j in range(1, n_steps + 1):
        lam = lam_pos[j]
        eps = lam - lam_pos[j - 1]

        eta_bar = np.mean(x, axis=0)
        H_curr = H_jac(eta_bar)
        A, b = compute_edh_matrices(m_pred, P_pred, H_curr, R, y, lam, eta_bar, h)

        x = x + eps * (x @ A.T + b)
        log_det_J += np.linalg.slogdet(np.eye(n_x) + eps * A.T)[1]
        flow_history[j] = x

    # EKF/UKF update
    m_post, P_post = update_step(m_pred, P_pred, y, h, H_jac, R, filter_type,
                                 alpha_ukf, beta_ukf, kappa, joseph, angle_indices)

    x_hat = np.mean(x, axis=0)

    if redraw:
        x = rng.multivariate_normal(x_hat, P_post, size=N)
        flow_history[-1] = x

    return x, log_det_J, flow_history, x_hat, P_post
