"""Common TensorFlow utilities for filters and flows."""
import math

import tensorflow as tf

DTYPE = tf.float64


def joseph_update(P_pred, K, H, R):
    """Joseph-stabilized covariance update.

    P = (I - KH) P_pred (I - KH)^T + K R K^T
    """
    n_x = tf.shape(P_pred)[0]
    I = tf.eye(n_x, dtype=DTYPE)
    IKH = I - K @ H
    return IKH @ P_pred @ tf.transpose(IKH) + K @ R @ tf.transpose(K)


def standard_update(P_pred, K, H):
    """Standard covariance update: P = (I - KH) P_pred."""
    n_x = tf.shape(P_pred)[0]
    I = tf.eye(n_x, dtype=DTYPE)
    return (I - K @ H) @ P_pred


def wrap_angles(innovation, angle_indices):
    """Wrap specified indices of innovation to [-pi, pi]."""
    if not angle_indices:
        return innovation
    indices = tf.constant(angle_indices, dtype=tf.int32)
    vals = tf.gather(innovation, indices)
    wrapped = tf.math.atan2(tf.math.sin(vals), tf.math.cos(vals))
    return tf.tensor_scatter_nd_update(innovation, tf.expand_dims(indices, 1), wrapped)


def log_gaussian(x, mean, cov_inv, log_det_cov, n):
    """Log of multivariate Gaussian density.

    log N(x; mean, cov) = -0.5 * (log_det + n*log(2pi) + (x-m)^T S^{-1} (x-m))
    """
    diff = x - mean
    mahal = tf.reduce_sum(diff * tf.linalg.matvec(cov_inv, diff))
    log_2pi = tf.math.log(tf.constant(2.0 * math.pi, dtype=DTYPE))
    return -0.5 * (log_det_cov + tf.cast(n, DTYPE) * log_2pi + mahal)


def compute_kalman_gain(P_pred, H, R):
    """Compute Kalman gain via Cholesky solve.

    K = P_pred H^T (H P_pred H^T + R)^{-1}

    Returns (K, S) where S is the innovation covariance.
    """
    S = H @ P_pred @ tf.transpose(H) + R
    PH_T = P_pred @ tf.transpose(H)
    K = tf.transpose(tf.linalg.cholesky_solve(tf.linalg.cholesky(S), tf.transpose(PH_T)))
    return K, S


def symmetrize(P):
    """Symmetrize a matrix: P = 0.5 * (P + P^T)."""
    return 0.5 * (P + tf.transpose(P))


def cond_number(P):
    """Condition number via SVD."""
    s = tf.linalg.svd(P, compute_uv=False)
    return s[0] / tf.maximum(s[-1], 1e-15)
