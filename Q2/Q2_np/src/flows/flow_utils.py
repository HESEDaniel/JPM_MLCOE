"""Common utilities for particle flow filters."""
import numpy as np
from ..filters.ekf import ekf_predict, ekf_update
from ..filters.ukf import ukf_predict, ukf_update


def get_lambda_schedule(n_steps, custom_schedule=None):
    """
    Get lambda schedule for flow integration.

    Parameters
    ----------
    n_steps : int
        Number of integration steps
    custom_schedule : ndarray, optional
        Custom lambda positions [0, ..., 1]. If provided, n_steps is ignored.

    Returns
    -------
    lam_pos : ndarray
        Lambda positions from 0 to 1
    n_steps : int
        Actual number of steps (may differ if custom_schedule provided)
    """
    if custom_schedule is not None:
        return custom_schedule, len(custom_schedule) - 1
    return np.linspace(0, 1, n_steps + 1), n_steps


def propagate_particles(particles, f, Q, rng):
    """
    Propagate particles through dynamics with process noise.

    Parameters
    ----------
    particles : ndarray [N, n_x]
        Current particles
    f : callable
        State transition function
    Q : ndarray [n_x, n_x]
        Process noise covariance
    rng : Generator
        NumPy random generator

    Returns
    -------
    ndarray [N, n_x]
        Propagated particles
    """
    N, n_x = particles.shape
    noise = rng.multivariate_normal(np.zeros(n_x), Q, size=N)
    return f(particles) + noise


def predict_step(m_prev, P_prev, f, F_jacobian, Q, filter_type,
                 alpha_ukf=1e-3, beta_ukf=2.0, kappa=0.0):
    """
    Perform EKF or UKF prediction step.

    Parameters
    ----------
    m_prev : ndarray [n_x]
        Previous mean
    P_prev : ndarray [n_x, n_x]
        Previous covariance
    f : callable
        State transition function
    F_jacobian : callable
        Jacobian of f (used for EKF)
    Q : ndarray [n_x, n_x]
        Process noise covariance
    filter_type : str
        'ekf' or 'ukf'
    alpha_ukf, beta_ukf, kappa : float
        UKF parameters

    Returns
    -------
    m_pred : ndarray [n_x]
        Predicted mean
    P_pred : ndarray [n_x, n_x]
        Predicted covariance
    """
    if filter_type.lower() == 'ukf':
        return ukf_predict(m_prev, P_prev, f, Q, alpha_ukf, beta_ukf, kappa)
    elif filter_type.lower() == 'ekf':
        return ekf_predict(m_prev, P_prev, f, F_jacobian, Q)
    else:
        raise ValueError(f"filter_type must be 'ekf' or 'ukf', got '{filter_type}'")


def update_step(m_pred, P_pred, y, h, H_jac, R, filter_type,
                alpha_ukf=1e-3, beta_ukf=2.0, kappa=0.0,
                joseph=True, angle_indices=None):
    """
    Perform EKF or UKF update step.

    Parameters
    ----------
    m_pred : ndarray [n_x]
        Predicted mean
    P_pred : ndarray [n_x, n_x]
        Predicted covariance
    y : ndarray [n_y]
        Observation
    h : callable
        Observation function
    H_jac : callable
        Jacobian of h (used for EKF)
    R : ndarray [n_y, n_y]
        Observation noise covariance
    filter_type : str
        'ekf' or 'ukf'
    alpha_ukf, beta_ukf, kappa : float
        UKF parameters
    joseph : bool
        Use Joseph stabilized update
    angle_indices : list of int
        Indices of angular observations

    Returns
    -------
    m_post : ndarray [n_x]
        Posterior mean
    P_post : ndarray [n_x, n_x]
        Posterior covariance
    """
    if filter_type.lower() == 'ekf':
        return ekf_update(m_pred, P_pred, y, h, H_jac, R,
                          joseph=joseph, angle_indices=angle_indices)
    elif filter_type.lower() == 'ukf':
        return ukf_update(m_pred, P_pred, y, h, R, alpha_ukf, beta_ukf, kappa,
                          joseph=joseph, angle_indices=angle_indices)
    else:
        raise ValueError(f"filter_type must be 'ekf' or 'ukf', got '{filter_type}'")
