"""Ensemble Kalman Filter (TensorFlow)."""
import tensorflow as tf

DTYPE = tf.float64


def enkf_update(particles, H, R, y, localization_matrix=None):
    """EnKF update step.

    Parameters
    ----------
    particles : tf.Tensor [N, n_x]
    H : tf.Tensor [n_y, n_x]
    R : tf.Tensor [n_y, n_y]
    y : tf.Tensor [n_y]
    localization_matrix : tf.Tensor [n_x, n_x], optional

    Returns
    -------
    particles_post, m_post, P_post
    """
    N = tf.shape(particles)[0]
    n_x = tf.shape(particles)[1]
    N_f = tf.cast(N, DTYPE)

    x_bar = tf.reduce_mean(particles, axis=0)
    X = tf.transpose(particles - x_bar[tf.newaxis, :])  # [n_x, N]

    B = X @ tf.transpose(X) / (N_f - 1.0)

    if localization_matrix is not None:
        B_loc = B * localization_matrix
    else:
        B_loc = B

    S = H @ B_loc @ tf.transpose(H) + R
    K = B_loc @ tf.transpose(H) @ tf.linalg.inv(S)

    innovation = y - tf.linalg.matvec(H, x_bar)
    m_post = x_bar + tf.linalg.matvec(K, innovation)

    I = tf.eye(n_x, dtype=DTYPE)
    IKH = I - K @ H
    P_post = IKH @ B_loc @ tf.transpose(IKH) + K @ R @ tf.transpose(K)

    y_pred = particles @ tf.transpose(H)
    innov_all = y[tf.newaxis, :] - y_pred
    particles_post = particles + innov_all @ tf.transpose(K)

    return particles_post, m_post, P_post


def enkf_posterior_analytical(x_bar, B, H, R, y, localization_matrix=None):
    """Compute EnKF posterior mean and covariance analytically.

    Parameters
    ----------
    x_bar : tf.Tensor [n_x]
    B : tf.Tensor [n_x, n_x]
    H : tf.Tensor [n_y, n_x]
    R : tf.Tensor [n_y, n_y]
    y : tf.Tensor [n_y]
    localization_matrix : tf.Tensor [n_x, n_x], optional

    Returns
    -------
    m_post, P_post
    """
    n_x = tf.shape(x_bar)[0]

    if localization_matrix is not None:
        B_loc = B * localization_matrix
    else:
        B_loc = B

    S = H @ B_loc @ tf.transpose(H) + R
    K = B_loc @ tf.transpose(H) @ tf.linalg.inv(S)

    innovation = y - tf.linalg.matvec(H, x_bar)
    m_post = x_bar + tf.linalg.matvec(K, innovation)

    I = tf.eye(n_x, dtype=DTYPE)
    P_post = (I - K @ H) @ B_loc

    return m_post, P_post


def localization_matrix(n_x, r_in=4.0):
    """Gaspari-Cohn localization matrix.

    Exponential decay with radius r_in. Returns tf.Tensor [n_x, n_x].
    """
    idx = tf.cast(tf.range(n_x), DTYPE)
    diff = tf.abs(idx[:, tf.newaxis] - idx[tf.newaxis, :])
    # Circular distance for periodic domain
    circ_dist = tf.minimum(diff, tf.cast(n_x, DTYPE) - diff)
    return tf.exp(-0.5 * (circ_dist / r_in) ** 2)
