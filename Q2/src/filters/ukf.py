"""Unscented Kalman Filter (UKF) implementation."""
import numpy as np
from scipy.linalg import cho_factor as cholesky_factor, cho_solve as cholesky_solve


def unscented_kalman_filter(f, h, Q, R, m0, P0, ys,
                            alpha=1e-3, beta=2.0, kappa=0.0, joseph=True, angle_indices=None):
    """
    Unscented Kalman Filter for nonlinear SSM.

    Parameters
    ----------
    f : callable
        State transition: f(x) -> x_next
    h : callable
        Observation: h(x) -> y
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
    alpha, beta, kappa : float
        UKF scaling parameters
    angle_indices : list[int]
        Indices of angular observations

    Returns
    -------
    m_filt : ndarray [T, n_x]
    P_filt : ndarray [T, n_x, n_x]
    cond_nums : ndarray [T]
    """
    T, n_x = ys.shape[0], len(m0)

    m, P = m0.copy(), P0.copy()
    m_filt = np.zeros((T, n_x))
    P_filt = np.zeros((T, n_x, n_x))
    cond_nums = np.zeros(T)

    for t in range(T):
        # Predict
        m_pred, P_pred = ukf_predict(m, P, f, Q, alpha, beta, kappa)
        # Update
        m, P = ukf_update(m_pred, P_pred, ys[t], h, R, alpha, beta, kappa, joseph, angle_indices)
        m_filt[t], P_filt[t] = m, P
        cond_nums[t] = np.linalg.cond(P)

    return m_filt, P_filt, cond_nums


def _ukf_weights(n_x, alpha, beta, kappa):
    """Compute UKF weights and scaling factor."""
    lam = alpha**2 * (n_x + kappa) - n_x
    gamma = np.sqrt(n_x + lam)

    W_m = np.full(2 * n_x + 1, 1 / (2 * (n_x + lam)))
    W_c = W_m.copy()
    W_m[0] = lam / (n_x + lam)
    W_c[0] = lam / (n_x + lam) + (1 - alpha**2 + beta)

    return W_m, W_c, gamma


def _sigma_points(m, P, gamma):
    """Generate sigma points."""
    n_x = len(m)
    try:
        sqrt_P = np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(P)
        sqrt_P = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10)))

    sigma = np.zeros((2 * n_x + 1, n_x))
    sigma[0] = m
    for i in range(n_x):
        sigma[i + 1] = m + gamma * sqrt_P[:, i]
        sigma[n_x + i + 1] = m - gamma * sqrt_P[:, i]
    return sigma


def ukf_predict(m, P, f, Q, alpha=1e-3, beta=2.0, kappa=0.0):
    """UKF prediction step."""
    n_x = len(m)
    W_m, W_c, gamma = _ukf_weights(n_x, alpha, beta, kappa)

    sigma = _sigma_points(m, P, gamma)
    sigma_pred = np.array([f(s) for s in sigma])
    m_pred = W_m @ sigma_pred

    P_pred = Q.copy()
    for i, w in enumerate(W_c):
        d = sigma_pred[i] - m_pred
        P_pred += w * np.outer(d, d)
    return m_pred, P_pred


def ukf_update(m_pred, P_pred, y, h, R, alpha=1e-3, beta=2.0, kappa=0.0,
               joseph=True, angle_indices=None):
    """UKF update step."""
    n_x, n_y = len(m_pred), R.shape[0]
    W_m, W_c, gamma = _ukf_weights(n_x, alpha, beta, kappa)

    sigma = _sigma_points(m_pred, P_pred, gamma)
    sigma_obs = np.array([h(s) for s in sigma])

    if angle_indices:
        y_pred = W_m @ sigma_obs
        for i in angle_indices:
            y_pred[i] = np.arctan2(W_m @ np.sin(sigma_obs[:, i]),
                                   W_m @ np.cos(sigma_obs[:, i]))
    else:
        y_pred = W_m @ sigma_obs

    P_yy, P_xy = R.copy(), np.zeros((n_x, n_y))
    for i, w in enumerate(W_c):
        dy = sigma_obs[i] - y_pred
        if angle_indices:
            for j in angle_indices:
                dy[j] = np.arctan2(np.sin(dy[j]), np.cos(dy[j]))
        P_yy += w * np.outer(dy, dy)
        P_xy += w * np.outer(sigma[i] - m_pred, dy)

    L, lower = cholesky_factor(P_yy)
    K = cholesky_solve((L, lower), P_xy.T).T
    innov = y - y_pred
    if angle_indices:
        for i in angle_indices:
            innov[i] = np.arctan2(np.sin(innov[i]), np.cos(innov[i]))

    m = m_pred + K @ innov
    P = P_pred - K @ P_yy @ K.T
    return m, P
