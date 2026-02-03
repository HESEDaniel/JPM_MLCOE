"""Particle Flow Particle Filter with LEDH."""
import numpy as np
from ..filters.ekf import ekf_predict, ekf_update
from ..filters.ukf import ukf_predict, ukf_update
from ..filters.pf import systematic_resample
from .ledh import compute_ledh_matrices


def _log_gaussian(x, mean, cov_inv, log_det_cov, n):
    """Log of Gaussian density N(x; mean, cov)."""
    diff = x - mean
    return -0.5 * (log_det_cov + n * np.log(2 * np.pi) + diff @ cov_inv @ diff)


def pfpf_ledh(f, h, H_jac, Q, R, m0, P0, ys,
              N_particles=500, n_flow_steps=20, lambda_schedule=None,
              resample_threshold=0.5, filter_type='ekf', F_jacobian=None,
              alpha_ukf=1e-3, beta_ukf=2.0, kappa=0.0, rng=None):
    """
    PF-PF with LEDH flow (Algorithm 1 from Li et al. 2017).

    Parameters
    ----------
    f : callable
        State transition function
    h : callable
        Observation function
    H_jac : callable
        Jacobian of h
    Q : ndarray [n_x, n_x]
        Process noise covariance
    R : ndarray [n_y, n_y]
        Observation noise covariance
    m0 : ndarray [n_x]
        Initial mean
    P0 : ndarray [n_x, n_x]
        Initial covariance
    ys : ndarray [T, n_y]
        Observations
    N_particles : int
        Number of particles
    n_flow_steps : int
        Number of flow steps
    lambda_schedule : ndarray, optional
        Custom lambda positions [0, ..., 1]
    resample_threshold : float
        Resample when ESS < threshold * N
    filter_type : str
        'ekf' or 'ukf'
    F_jacobian : callable
        Jacobian of f (for EKF)

    Returns
    -------
    m_filt : ndarray [T, n_x]
    P_filt : ndarray [T, n_x, n_x]
    ess_history : ndarray [T]
    resample_count : int
    weights_history : ndarray [T, N]
    """
    if rng is None:
        rng = np.random.default_rng()

    T, n_x, N = len(ys), len(m0), N_particles

    # Precompute
    Q_inv = np.linalg.inv(Q)
    log_det_Q = np.linalg.slogdet(Q)[1]
    R_inv = np.linalg.inv(R)
    log_det_R = np.linalg.slogdet(R)[1]
    n_y = R.shape[0]
    I = np.eye(n_x)

    # Lambda schedule
    if lambda_schedule is not None:
        lam_pos = lambda_schedule
        n_flow_steps = len(lam_pos) - 1
    else:
        lam_pos = np.linspace(0, 1, n_flow_steps + 1)

    # Initialize
    particles = rng.multivariate_normal(m0, P0, size=N)
    weights = np.ones(N) / N
    m_prev = np.tile(m0, (N, 1))
    P_prev = np.tile(P0, (N, 1, 1))

    # Storage
    m_filt = np.zeros((T, n_x))
    P_filt = np.zeros((T, n_x, n_x))
    ess_history = np.zeros(T)
    weights_history = np.zeros((T, N))
    resample_count = 0

    for t in range(T):
        y_t = np.atleast_1d(ys[t])

        # Per-particle prediction and initialization
        eta_bar = np.zeros((N, n_x))
        eta_0_bar = np.zeros((N, n_x))
        eta_0 = np.zeros((N, n_x))
        eta_1 = np.zeros((N, n_x))
        log_theta = np.zeros(N)
        P_pred_all = np.zeros((N, n_x, n_x))
        f_x_prev = np.zeros((N, n_x))
        m_pred = np.zeros((N, n_x))

        for i in range(N):
            # EKF/UKF prediction - use particle position x^i_{k-1} per Algorithm 1 Line 5
            if filter_type.lower() == 'ekf':
                m_pred[i], P_pred_i = ekf_predict(particles[i], P_prev[i], f, F_jacobian, Q)
            else:
                m_pred[i], P_pred_i = ukf_predict(particles[i], P_prev[i], f, Q,
                                          alpha_ukf, beta_ukf, kappa)
            P_pred_all[i] = P_pred_i

            f_x_prev[i] = f(particles[i])
            eta_bar[i] = f_x_prev[i]
            eta_0_bar[i] = f_x_prev[i]
            eta_0[i] = rng.multivariate_normal(f_x_prev[i], Q)
            eta_1[i] = eta_0[i].copy()

        # Flow integration (per-particle)
        for j in range(1, n_flow_steps + 1):
            lam = lam_pos[j]
            eps = lam - lam_pos[j - 1]

            for i in range(N):
                A_i, b_i = compute_ledh_matrices(
                    eta_bar[i], eta_0_bar[i], P_pred_all[i], h, H_jac, R, y_t, lam, R_inv, I
                )
                eta_bar[i] += eps * (A_i @ eta_bar[i] + b_i)
                eta_1[i] += eps * (A_i @ eta_1[i] + b_i)
                log_theta[i] += np.linalg.slogdet(I + eps * A_i)[1]

            # Intermediate normalization to prevent numerical overflow (as in reference PFPF)
            log_theta = log_theta - np.max(log_theta)

        # Update particles
        particles = eta_1.copy()

        # Log-likelihood
        log_lik = np.array([
            _log_gaussian(y_t, h(particles[i]), R_inv, log_det_R, n_y)
            for i in range(N)
        ])

        # Transition density ratio
        log_trans_ratio = np.zeros(N)
        log_trans_ratio = np.array([
                _log_gaussian(particles[i], f_x_prev[i], Q_inv, log_det_Q, n_x) -
                _log_gaussian(eta_0[i], f_x_prev[i], Q_inv, log_det_Q, n_x)
                for i in range(N)
                ])

        # Weight update (log-sum-exp trick)
        log_weights = log_lik + log_theta + log_trans_ratio + np.log(weights)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)

        # Per-particle EKF/UKF update
        for i in range(N):
            if filter_type.lower() == 'ekf':
                m_post_i, P_post_i = ekf_update(
                    m_pred[i], P_pred_all[i], y_t, h, H_jac, R)
            elif filter_type.lower() == 'ukf':
                m_post_i, P_post_i = ukf_update(
                    m_pred[i], P_pred_all[i], y_t, h, R, alpha_ukf, beta_ukf, kappa)
            else:
                raise ValueError("filter_type must be 'ekf' or 'ukf'")
            m_prev[i], P_prev[i] = m_post_i, P_post_i
        # Estimates
        m_filt[t] = weights @ particles

        diff = particles - m_filt[t]
        P_filt[t] = np.einsum('i,ij,ik->jk', weights, diff, diff)

        # ESS and resample
        ess = 1.0 / np.sum(weights ** 2)
        ess_history[t] = ess

        if ess < resample_threshold * N:
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            m_prev = m_prev[indices]
            P_prev = P_prev[indices]
            weights = np.ones(N) / N
            resample_count += 1

        weights_history[t] = weights

    return m_filt, P_filt, ess_history, resample_count, weights_history
