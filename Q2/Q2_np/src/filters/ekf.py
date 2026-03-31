"""Extended Kalman Filter (EKF) implementation."""
import numpy as np
from scipy.linalg import cho_factor as cholesky_factor, cho_solve as cholesky_solve
from .common import joseph_update, standard_update, wrap_angles


def extended_kalman_filter(f, h, F_jacobian, H_jacobian, Q, R, m0, P0, ys,
                           joseph=True, angle_indices=None):
    """
    Extended Kalman Filter for nonlinear SSM.

    Parameters
    ----------
    f : callable
        State transition: f(x) -> x_next
    h : callable
        Observation function: h(x) -> y
    F_jacobian, H_jacobian : callable
        Jacobians: F_jacobian(x) -> [n_x, n_x], H_jacobian(x) -> [n_y, n_x]
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
    joseph : bool
        Use Joseph stabilized update
    angle_indices : list[int]
        Indices of angular observations (wrapped to [-pi, pi])

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
        m_pred, P_pred = ekf_predict(m, P, f, F_jacobian, Q)
        # Update
        m, P = ekf_update(m_pred, P_pred, ys[t], h, H_jacobian, R, joseph, angle_indices)
        m_filt[t], P_filt[t] = m, P
        cond_nums[t] = np.linalg.cond(P)

    return m_filt, P_filt, cond_nums


def ekf_predict(m, P, f, F_jacobian, Q):
    """EKF prediction step."""
    F = F_jacobian(m)
    m_pred = f(m)
    P_pred = F @ P @ F.T + Q

    return m_pred, P_pred


def ekf_update(m_pred, P_pred, y, h, H_jacobian, R, joseph=True, angle_indices=None):
    """EKF update step."""
    H = H_jacobian(m_pred)
    S = H @ P_pred @ H.T + R
    L, lower = cholesky_factor(S)
    K = cholesky_solve((L, lower), H @ P_pred).T

    innov = wrap_angles(y - h(m_pred), angle_indices)
    m = m_pred + K @ innov
    P = joseph_update(P_pred, K, H, R) if joseph else standard_update(P_pred, K, H)

    return m, P
