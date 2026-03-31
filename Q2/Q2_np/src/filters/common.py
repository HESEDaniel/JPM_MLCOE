"""Common utilities for Kalman filter variants."""
import numpy as np


def joseph_update(P_pred, K, H, R):
    """
    Compute Joseph-stabilized covariance update.
    
    Parameters
    ----------
    P_pred : ndarray [n_x, n_x]
        Predicted covariance
    K : ndarray [n_x, n_y]
        Kalman gain
    H : ndarray [n_y, n_x]
        Observation matrix/Jacobian
    R : ndarray [n_y, n_y]
        Observation noise covariance

    Returns
    -------
    ndarray [n_x, n_x]
        Updated covariance
    """
    n_x = P_pred.shape[0]
    I = np.eye(n_x)
    IKH = I - K @ H
    return IKH @ P_pred @ IKH.T + K @ R @ K.T


def standard_update(P_pred, K, H):
    """
    Compute standard covariance update: P = (I - KH) P_pred.

    Parameters
    ----------
    P_pred : ndarray [n_x, n_x]
        Predicted covariance
    K : ndarray [n_x, n_y]
        Kalman gain
    H : ndarray [n_y, n_x]
        Observation matrix/Jacobian

    Returns
    -------
    ndarray [n_x, n_x]
        Updated covariance
    """
    n_x = P_pred.shape[0]
    I = np.eye(n_x)
    return (I - K @ H) @ P_pred


def wrap_angles(innovation, angle_indices):
    """
    Wrap specified indices of innovation to [-pi, pi].
    
    Parameters
    ----------
    innovation : ndarray [n_y]
        Innovation vector (y - h(x))
    angle_indices : list of int or None
        Indices to wrap. If None, returns innovation unchanged.

    Returns
    -------
    ndarray [n_y]
        Innovation with angles wrapped
    """
    if not angle_indices:
        return innovation

    result = innovation.copy()
    for i in angle_indices:
        result[i] = np.arctan2(np.sin(result[i]), np.cos(result[i]))
    return result
