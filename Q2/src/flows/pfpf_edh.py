"""Particle Flow Particle Filter with EDH."""
import numpy as np
from ..filters.ekf import ekf_predict, ekf_update
from ..filters.ukf import ukf_predict, ukf_update
from ..filters.pf import systematic_resample
from .edh import compute_edh_matrices


def _log_gaussian(x, mean, cov_inv, log_det_cov, n):
    """Log of Gaussian density N(x; mean, cov)."""
    diff = x - mean
    return -0.5 * (log_det_cov + n * np.log(2 * np.pi) + diff @ cov_inv @ diff)


def pfpf_edh(f, h, H_jac, Q, R, m0, P0, ys,
             N_particles=500, n_flow_steps=20, lambda_schedule=None,
             resample_threshold=0.5, filter_type='ekf', F_jacobian=None,
             alpha_ukf=1e-3, beta_ukf=2.0, kappa=0.0, rng=None):
    """
    PF-PF with EDH flow (Algorithm 2 from Li et al. 2017).

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

    # Lambda schedule
    if lambda_schedule is not None:
        lam_pos = lambda_schedule
        n_flow_steps = len(lam_pos) - 1
    else:
        lam_pos = np.linspace(0, 1, n_flow_steps + 1)

    # Initialize
    particles = rng.multivariate_normal(m0, P0, size=N)
    weights = np.ones(N) / N
    x_hat, P_hat = m0.copy(), P0.copy()

    # Storage
    m_filt = np.zeros((T, n_x))
    P_filt = np.zeros((T, n_x, n_x))
    ess_history = np.zeros(T)
    weights_history = np.zeros((T, N))
    resample_count = 0

    eta_0 = np.zeros((N, n_x))
    f_x_prev = np.zeros((N, n_x))
    for t in range(T):
        y_t = np.atleast_1d(ys[t])

        # EKF/UKF prediction
        if filter_type.lower() == 'ekf':
            m_pred, P_pred = ekf_predict(x_hat, P_hat, f, F_jacobian, Q)
        else:
            m_pred, P_pred = ukf_predict(x_hat, P_hat, f, Q, alpha_ukf, beta_ukf, kappa)

        # Propagate particles with noise
        for i in range(N):
            f_x_prev[i] = f(particles[i])
            eta_0[i] = rng.multivariate_normal(f_x_prev[i], Q)

        eta_1 = eta_0.copy()
        eta_bar_0 = f(x_hat)
        eta_bar = eta_bar_0.copy()

        # Flow integration
        for j in range(1, n_flow_steps + 1):
            lam = lam_pos[j]
            eps = lam - lam_pos[j - 1]

            H_curr = H_jac(eta_bar)
            A, b = compute_edh_matrices(eta_bar_0, P_pred, H_curr, R, y_t, lam, eta_bar, h)

            eta_bar += eps * (A @ eta_bar + b)
            eta_1 += eps * (eta_1 @ A.T + b)

        # Update particles and weights
        particles = eta_1.copy()

        # Log-likelihood
        log_lik = np.array([
            _log_gaussian(y_t, h(particles[i]), R_inv, log_det_R, n_y)
            for i in range(N)
        ])

        log_trans_ratio = np.array([
            _log_gaussian(particles[i], f_x_prev[i], Q_inv, log_det_Q, n_x) -
            _log_gaussian(eta_0[i], f_x_prev[i], Q_inv, log_det_Q, n_x)
            for i in range(N)
        ])

        # Weight update (log-sum-exp trick)
        log_weights = log_lik + log_trans_ratio + np.log(weights)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)

        # EKF/UKF update
        if filter_type.lower() == 'ekf':
            _, P_post = ekf_update(m_pred, P_pred, y_t, h, H_jac, R)
        elif filter_type.lower() == 'ukf':
            _, P_post = ukf_update(m_pred, P_pred, y_t, h, R, alpha_ukf, beta_ukf, kappa)
        else:
            raise ValueError("filter_type must be 'ekf' or 'ukf'")

        # Estimates
        m_filt[t] = weights @ particles
        P_filt[t] = P_post.copy()
        x_hat, P_hat = m_filt[t].copy(), P_filt[t].copy()

        # ESS and resample
        ess = 1.0 / np.sum(weights ** 2)
        ess_history[t] = ess

        if ess < resample_threshold * N:
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            weights = np.ones(N) / N
            resample_count += 1

        weights_history[t] = weights

    return m_filt, P_filt, ess_history, resample_count, weights_history
