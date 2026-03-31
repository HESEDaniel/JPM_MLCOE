"""Common TF utilities for particle flows."""
import tensorflow as tf

from ..filters.ekf import ExtendedKalmanFilter
from ..filters.ukf import UnscentedKalmanFilter
from ..filters.common import DTYPE


def get_lambda_schedule(n_steps, custom_schedule=None):
    """Get lambda schedule for flow integration.

    Returns (lam_pos, n_steps) where lam_pos is [0, ..., 1].
    """
    if custom_schedule is not None:
        lam_pos = tf.constant(custom_schedule, dtype=DTYPE)
        return lam_pos, tf.shape(lam_pos)[0] - 1
    return tf.cast(tf.linspace(0.0, 1.0, n_steps + 1), DTYPE), n_steps


def predict_step(m_prev, P_prev, ssm, filter_type='ekf', **ukf_kwargs):
    """Dispatch EKF/UKF prediction step.

    Parameters
    ----------
    m_prev : tf.Tensor [n_x]
        Previous state mean.
    P_prev : tf.Tensor [n_x, n_x]
        Previous state covariance.
    ssm : SSM
        State space model.
    filter_type : str
        'ekf' or 'ukf'.
    """
    if filter_type.lower() == 'ukf':
        ukf = UnscentedKalmanFilter(**ukf_kwargs)
        return ukf.predict(m_prev, P_prev, ssm)
    else:
        ekf = ExtendedKalmanFilter()
        return ekf.predict(m_prev, P_prev, ssm)


def update_step(m_pred, P_pred, y, ssm, filter_type='ekf',
                angle_indices=None, joseph=True, **ukf_kwargs):
    """Dispatch EKF/UKF update step.

    Parameters
    ----------
    m_pred : tf.Tensor [n_x]
        Predicted state mean.
    P_pred : tf.Tensor [n_x, n_x]
        Predicted state covariance.
    y : tf.Tensor [n_y]
        Observation.
    ssm : SSM
        State space model.
    filter_type : str
        'ekf' or 'ukf'.
    angle_indices : list of int, optional
        Indices of angular components for wrapping.
    joseph : bool
        Use Joseph-stabilized covariance update (default True for EKF).
    """
    if filter_type.lower() == 'ukf':
        ukf = UnscentedKalmanFilter(**ukf_kwargs)
        return ukf.update(m_pred, P_pred, y, ssm)
    else:
        ekf = ExtendedKalmanFilter(joseph=joseph, angle_indices=angle_indices)
        return ekf.update(m_pred, P_pred, y, ssm)
