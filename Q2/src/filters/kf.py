"""Kalman Filter (KF) implementation."""
import numpy as np
from scipy import linalg as sla
from .common import joseph_update, standard_update


def _solve_lu(S, B):
    """Solve S @ X = B using LU factorization (np.linalg.solve)."""
    return np.linalg.solve(S, B)


def _solve_cholesky(S, B):
    """Solve S @ X = B using Cholesky factorization (assumes S is SPD)."""
    L = sla.cholesky(S, lower=True)
    return sla.cho_solve((L, True), B)


def _solve_inv(S, B):
    """Solve S @ X = B using explicit matrix inversion (least stable)."""
    return np.linalg.inv(S) @ B


def kalman_filter(A, B, C, D, Sigma, ys, joseph=True, solver='cholesky'):
    """
    Kalman Filter for Linear Gaussian SSM.

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
    ys : ndarray [T, n_y]
        Observations
    joseph : bool
        Use Joseph stabilized covariance update (default: True)
    solver : str
        Solver for Kalman gain: 'lu', 'cholesky', or 'inv' (default: 'cholesky')

    Returns
    -------
    m_filt : ndarray [T, n_x]
        Filtered state means
    P_filt : ndarray [T, n_x, n_x]
        Filtered state covariances
    cond_nums : ndarray [T]
        Condition numbers of P
    """
    n_x = A.shape[0]
    T = ys.shape[0]
    Q, R = B @ B.T, D @ D.T

    if solver == 'cholesky':
        solve_fn = _solve_cholesky
    elif solver == 'inv':
        solve_fn = _solve_inv
    else:
        solve_fn = _solve_lu

    m, P = np.zeros(n_x), Sigma.copy()
    m_filt = np.zeros((T, n_x))
    P_filt = np.zeros((T, n_x, n_x))
    cond_nums = np.zeros(T)

    for t in range(T):
        # Predict
        m_pred = A @ m
        P_pred = A @ P @ A.T + Q

        # Update: K = P_pred @ C.T @ S^{-1}
        S = C @ P_pred @ C.T + R
        K = solve_fn(S.T, C @ P_pred.T).T
        m = m_pred + K @ (ys[t] - C @ m_pred)
        P = joseph_update(P_pred, K, C, R) if joseph else standard_update(P_pred, K, C)

        m_filt[t], P_filt[t] = m, P
        cond_nums[t] = np.linalg.cond(P)

    return m_filt, P_filt, cond_nums
