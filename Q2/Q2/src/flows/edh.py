"""Exact Daum-Huang (EDH) particle flow (TensorFlow)."""
import tensorflow as tf

from .base import BaseFlow
from .flow_utils import update_step
from ..filters.common import DTYPE


def compute_edh_matrices(m, P, H, R, y, lam, eta_bar, h_fn):
    """Compute A(lambda) and b(lambda) for EDH flow (Li et al. 2017, Eq. 10-11).

    Parameters
    ----------
    m : tf.Tensor [n_x]
        Prior mean.
    P : tf.Tensor [n_x, n_x]
        Prior covariance.
    H : tf.Tensor [n_y, n_x]
        Observation Jacobian at eta_bar.
    R : tf.Tensor [n_y, n_y]
        Observation noise covariance.
    y : tf.Tensor [n_y]
        Observation.
    lam : tf.Tensor scalar
        Homotopy parameter in [0, 1].
    eta_bar : tf.Tensor [n_x]
        Linearization point (particle mean).
    h_fn : callable
        Observation function x -> h(x).

    Returns
    -------
    A : tf.Tensor [n_x, n_x]
    b : tf.Tensor [n_x]
    """
    n_x = tf.shape(m)[0]
    I = tf.eye(n_x, dtype=DTYPE)

    S_lam = lam * (H @ P @ tf.transpose(H)) + R
    S_inv_H = tf.linalg.solve(S_lam, H)
    A = -0.5 * P @ tf.transpose(H) @ S_inv_H

    e_lam = h_fn(eta_bar) - tf.linalg.matvec(H, eta_bar)
    y_corr = y - e_lam
    R_inv_y = tf.linalg.solve(R, y_corr[:, tf.newaxis])[:, 0]

    term1 = tf.linalg.matvec((I + lam * A) @ P @ tf.transpose(H), R_inv_y)
    term2 = tf.linalg.matvec(A, m)
    b = tf.linalg.matvec(I + 2 * lam * A, term1 + term2)

    return A, b


class EDHFlow(BaseFlow):
    """Exact Daum-Huang particle flow filter.

    Parameters
    ----------
    redraw : bool
        If True, redraw particles from N(x_hat, P_post) after each flow step.
        Matches the "redraw strategy" in Li et al. 2017 / [50].
    """

    def __init__(self, n_particles=500, n_flow_steps=20,
                 lambda_schedule=None, filter_type='ekf', redraw=True):
        super().__init__(n_particles, n_flow_steps, lambda_schedule, filter_type)
        self.redraw = redraw

    def _transport(self, particles, m_pred, P_pred, ssm, y, rng):
        lam_pos = self.lambda_schedule
        for j in tf.range(1, self.n_flow_steps + 1):
            lam = lam_pos[j]
            eps = lam - lam_pos[j - 1]
            eta_bar = tf.reduce_mean(particles, axis=0)
            H_curr = ssm.H_jac(eta_bar)
            A, b = compute_edh_matrices(
                m_pred, P_pred, H_curr, ssm.R, y, lam, eta_bar, ssm.h)
            particles += eps * (
                particles @ tf.transpose(A) + b[tf.newaxis, :])
        return particles

    def _compute_posterior(self, particles, m_pred, P_pred, y, ssm):
        """EKF/UKF update for covariance, ensemble mean for state."""
        _, P_post = update_step(m_pred, P_pred, y, ssm, self.filter_type)
        x_hat = tf.reduce_mean(particles, axis=0)
        return x_hat, P_post

    def flow_step(self, particles, m_prev, P_prev, ssm, y, rng):
        particles, weights, x_hat, P_hat = super().flow_step(
            particles, m_prev, P_prev, ssm, y, rng)

        if self.redraw:
            N = tf.shape(particles)[0]
            L_post = tf.linalg.cholesky(P_hat)
            z = rng.normal(shape=(N, ssm.state_dim), dtype=DTYPE)
            particles = x_hat[tf.newaxis, :] + tf.linalg.matvec(L_post, z)

        return particles, weights, x_hat, P_hat
