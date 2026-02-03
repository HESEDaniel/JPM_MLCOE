"""Local Exact Daum-Huang (LEDH) particle flow."""
import numpy as np
from .flow_utils import get_lambda_schedule, propagate_particles, predict_step, update_step


def compute_ledh_matrices(x_i, m, P, h, H_jac, R, y, lam, R_inv=None, I=None):
    """
    Compute A_i(lambda) and b_i(lambda) for LEDH flow at particle x_i.

    Parameters
    ----------
    x_i : ndarray [n_x]
        Current particle position
    m : ndarray [n_x]
        Prior mean
    P : ndarray [n_x, n_x]
        Prior covariance
    h : callable
        Observation function
    H_jac : callable
        Jacobian of h
    R : ndarray [n_y, n_y]
        Observation noise covariance
    y : ndarray [n_y]
        Observation
    lam : float
        Pseudo-time in [0, 1]
    R_inv : ndarray, optional
        Precomputed R inverse
    I : ndarray, optional
        Precomputed identity matrix

    Returns
    -------
    A_i : ndarray [n_x, n_x]
    b_i : ndarray [n_x]
    """
    n_x = len(m)
    if I is None:
        I = np.eye(n_x)
    if R_inv is None:
        R_inv = np.linalg.inv(R)

    H_i = H_jac(x_i)
    h_i = h(x_i)
    PH = P @ H_i.T
    S_lam = lam * (H_i @ PH) + R

    S_inv_H = np.linalg.solve(S_lam, H_i)
    A_i = -0.5 * PH @ S_inv_H

    e_lam = h_i - H_i @ x_i
    y_corr = y - e_lam

    IplusLA = I + lam * A_i
    term1 = IplusLA @ PH @ R_inv @ y_corr
    term2 = A_i @ m
    b_i = (I + 2 * lam * A_i) @ (term1 + term2)

    return A_i, b_i


def local_edh_flow(particles, m_prev, P_prev, f, F_jacobian, Q, h, H_jac, R, y,
                   n_steps=20, lambda_schedule=None, store_history=True, redraw=True,
                   rng=None, filter_type='ukf', alpha_ukf=1e-3, beta_ukf=2.0, kappa=0.0,
                   joseph=True, angle_indices=None):
    """
    Apply LEDH flow with per-particle local linearization.

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
    store_history : bool
        Store flow history (set False for speed)
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
    flow_history : ndarray or None
    m_post : ndarray [n_x]
    P_post : ndarray [n_x, n_x]
    """
    N, n_x = particles.shape

    lam_pos, n_steps = get_lambda_schedule(n_steps, lambda_schedule)
    particles = propagate_particles(particles, f, Q, rng)
    m_pred, P_pred = predict_step(m_prev, P_prev, f, F_jacobian, Q, filter_type,
                                  alpha_ukf, beta_ukf, kappa)

    # Precompute constants
    I = np.eye(n_x)
    R_inv = np.linalg.inv(R)

    x = particles.copy()
    flow_history = np.zeros((n_steps + 1, N, n_x)) if store_history else None
    if store_history:
        flow_history[0] = x.copy()
    log_det_J = np.zeros(N)

    # Flow integration (per-particle)
    for j in range(1, n_steps + 1):
        lam = lam_pos[j]
        eps = lam - lam_pos[j - 1]

        for i in range(N):
            A_i, b_i = compute_ledh_matrices(x[i], m_pred, P_pred, h, H_jac, R, y, lam,
                                             R_inv=R_inv, I=I)
            x[i] = x[i] + eps * (A_i @ x[i] + b_i)
            log_det_J[i] += np.linalg.slogdet(I + eps * A_i)[1]

        if store_history:
            flow_history[j] = x.copy()

    # EKF/UKF update
    m_post, P_post = update_step(m_pred, P_pred, y, h, H_jac, R, filter_type,
                                 alpha_ukf, beta_ukf, kappa, joseph, angle_indices)

    x_hat = np.mean(x, axis=0)

    if redraw:
        x = rng.multivariate_normal(x_hat, P_post, size=N)
        if store_history:
            flow_history[-1] = x.copy()

    return x, log_det_J, flow_history, x_hat, P_post
