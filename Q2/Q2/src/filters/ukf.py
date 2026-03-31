"""Unscented Kalman Filter (TensorFlow)."""
import tensorflow as tf

from .base import BaseFilter, FilterResult
from .common import joseph_update, wrap_angles, cond_number, DTYPE


def _ukf_weights(n_x, alpha, beta, kappa):
    """Compute UKF weights and scaling factor."""
    n_x_f = tf.cast(n_x, DTYPE)
    alpha_f = tf.constant(alpha, dtype=DTYPE)
    beta_f = tf.constant(beta, dtype=DTYPE)
    kappa_f = tf.constant(kappa, dtype=DTYPE)

    lam = alpha_f**2 * (n_x_f + kappa_f) - n_x_f
    gamma = tf.sqrt(n_x_f + lam)

    w_fill = 1.0 / (2.0 * (n_x_f + lam))
    W_m = tf.fill([2 * n_x + 1], w_fill)
    W_c = tf.identity(W_m)
    W_m = tf.tensor_scatter_nd_update(W_m, [[0]], [lam / (n_x_f + lam)])
    W_c = tf.tensor_scatter_nd_update(W_c, [[0]], [lam / (n_x_f + lam) + (1.0 - alpha_f**2 + beta_f)])
    return W_m, W_c, gamma


def _sigma_points(m, P, gamma):
    """Generate sigma points with eigenvalue fallback."""
    eigvals, eigvecs = tf.linalg.eigh(P)
    eigvals = tf.maximum(eigvals, 1e-10)
    sqrt_P = eigvecs @ tf.linalg.diag(tf.sqrt(eigvals))
    scaled = gamma * sqrt_P
    sigma_plus = m[tf.newaxis, :] + tf.transpose(scaled)
    sigma_minus = m[tf.newaxis, :] - tf.transpose(scaled)
    return tf.concat([m[tf.newaxis, :], sigma_plus, sigma_minus], axis=0)


class UnscentedKalmanFilter(BaseFilter):
    """Unscented Kalman Filter for nonlinear SSM.

    Parameters
    ----------
    alpha : float
        Spread of sigma points around the mean.
    beta : float
        Prior knowledge parameter (2.0 is optimal for Gaussian).
    kappa : float
        Secondary scaling parameter.
    joseph : bool
        Use Joseph-stabilized covariance update.
    angle_indices : list of int, optional
        Indices of angular components for wrapping.
    """

    def __init__(self, alpha=1e-3, beta=2.0, kappa=0.0,
                 joseph=False, angle_indices=None):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.joseph = joseph
        self.angle_indices = angle_indices

    def predict(self, m, P, ssm, **kwargs):
        n_x = tf.shape(m)[0]
        W_m, W_c, gamma = _ukf_weights(n_x, self.alpha, self.beta, self.kappa)
        sigma = _sigma_points(m, P, gamma)
        sigma_pred = tf.vectorized_map(ssm.f, sigma)
        m_pred = tf.einsum('i,ij->j', W_m, sigma_pred)
        diff = sigma_pred - m_pred[tf.newaxis, :]
        P_pred = ssm.Q + tf.einsum('i,ij,ik->jk', W_c, diff, diff)
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, ssm, **kwargs):
        n_x = tf.shape(m_pred)[0]
        W_m, W_c, gamma = _ukf_weights(n_x, self.alpha, self.beta, self.kappa)
        sigma = _sigma_points(m_pred, P_pred, gamma)
        sigma_obs = tf.vectorized_map(ssm.h, sigma)
        y_pred = tf.einsum('i,ij->j', W_m, sigma_obs)
        dy = sigma_obs - y_pred[tf.newaxis, :]
        if self.angle_indices:
            dy = tf.vectorized_map(
                lambda d: wrap_angles(d, self.angle_indices), dy)
        dx = sigma - m_pred[tf.newaxis, :]
        P_yy = ssm.R + tf.einsum('i,ij,ik->jk', W_c, dy, dy)
        P_xy = tf.einsum('i,ij,ik->jk', W_c, dx, dy)
        L = tf.linalg.cholesky(P_yy)
        K = tf.transpose(tf.linalg.cholesky_solve(L, tf.transpose(P_xy)))
        innov = y - y_pred
        if self.angle_indices:
            innov = wrap_angles(innov, self.angle_indices)
        m_post = m_pred + tf.linalg.matvec(K, innov)
        if self.joseph:
            # Approximate H from sigma points for Joseph update
            H_approx = tf.transpose(tf.linalg.cholesky_solve(
                tf.linalg.cholesky(P_pred), P_xy))
            P_post = joseph_update(P_pred, K, H_approx, ssm.R)
        else:
            P_post = P_pred - K @ P_yy @ tf.transpose(K)
        return m_post, P_post

    def filter(self, ssm, ys, **kwargs):
        m_filt, P_filt, cond_nums = self._filter_loop(ssm, ys)
        return FilterResult(m_filt, P_filt, {'cond_nums': cond_nums})

    @tf.function
    def _filter_loop(self, ssm, ys):
        T = tf.shape(ys)[0]
        m, P = ssm.m0, ssm.P0
        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        c_arr = tf.TensorArray(DTYPE, size=T)
        for t in tf.range(T):
            m_pred, P_pred = self.predict(m, P, ssm)
            m, P = self.update(m_pred, P_pred, ys[t], ssm)
            m_arr = m_arr.write(t, m)
            P_arr = P_arr.write(t, P)
            c_arr = c_arr.write(t, cond_number(P))
        return m_arr.stack(), P_arr.stack(), c_arr.stack()
