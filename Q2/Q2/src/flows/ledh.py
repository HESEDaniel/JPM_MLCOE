"""Local Exact Daum-Huang (LEDH) particle flow (TensorFlow)."""
import tensorflow as tf

from .base import BaseFlow
from .flow_utils import update_step
from ..filters.common import DTYPE


def batch_H_jac(ssm, x_all):
    """Efficiently compute H_jac for all particles.

    Uses tiling if H_jac is constant, otherwise tf.map_fn.
    """
    N = tf.shape(x_all)[0]
    H0 = ssm.H_jac(x_all[0])
    H1 = ssm.H_jac(x_all[0] + tf.ones_like(x_all[0]))
    return tf.cond(
        tf.reduce_all(tf.equal(H0, H1)),
        lambda: tf.tile(H0[tf.newaxis], [N, 1, 1]),
        lambda: tf.map_fn(ssm.H_jac, x_all, fn_output_signature=DTYPE))


def batch_h(ssm, x_all):
    """Efficiently compute h for all particles."""
    if hasattr(ssm, 'h_batch'):
        return ssm.h_batch(x_all)
    return tf.vectorized_map(ssm.h, x_all)


def compute_ledh_matrices(x_i, m, P, h_fn, H_jac_fn, R, y, lam, R_inv):
    """Compute per-particle A_i(lambda) and b_i(lambda) for LEDH."""
    n_x = tf.shape(m)[0]
    I = tf.eye(n_x, dtype=DTYPE)

    H_i = H_jac_fn(x_i)
    PH = P @ tf.transpose(H_i)
    S_lam = lam * (H_i @ PH) + R
    S_inv_H = tf.linalg.solve(S_lam, H_i)
    A_i = -0.5 * PH @ S_inv_H

    e_lam = h_fn(x_i) - tf.linalg.matvec(H_i, x_i)
    y_corr = y - e_lam

    IplusLA = I + lam * A_i
    term1 = tf.linalg.matvec(IplusLA @ PH, tf.linalg.matvec(R_inv, y_corr))
    term2 = tf.linalg.matvec(A_i, m)
    b_i = tf.linalg.matvec(I + 2 * lam * A_i, term1 + term2)

    return A_i, b_i


def compute_ledh_matrices_batch(x_all, m, P, H_all, h_all, R, y, lam, R_inv):
    """Batch-compute A_i, b_i for all particles (no Python loop).

    Parameters
    ----------
    x_all : [N, n_x]
    H_all : [N, n_y, n_x]  -- pre-computed Jacobians
    h_all : [N, n_y]        -- pre-computed observations
    m, P, R, R_inv, y, lam : shared across particles

    Returns
    -------
    A_all : [N, n_x, n_x]
    b_all : [N, n_x]
    """
    n_x = tf.shape(m)[0]
    I = tf.eye(n_x, dtype=DTYPE)

    H_all_T = tf.transpose(H_all, [0, 2, 1])              # [N, n_x, n_y]

    PH = P[tf.newaxis] @ H_all_T                          # [N, n_x, n_y]
    S_lam = lam * (H_all @ PH) + R[tf.newaxis]            # [N, n_y, n_y]
    S_inv_H = tf.linalg.solve(S_lam, H_all)               # [N, n_y, n_x]
    A_all = -0.5 * (PH @ S_inv_H)                         # [N, n_x, n_x]

    Hx = tf.einsum('nij,nj->ni', H_all, x_all)            # [N, n_y]
    e_lam = h_all - Hx                                    # [N, n_y]
    y_corr = y[tf.newaxis] - e_lam                         # [N, n_y]
    R_inv_y = tf.einsum('ij,nj->ni', R_inv, y_corr)       # [N, n_y]

    IplusLA = I[tf.newaxis] + lam * A_all                  # [N, n_x, n_x]
    term1 = tf.einsum('nij,nj->ni', IplusLA @ PH, R_inv_y)  # [N, n_x]
    term2 = tf.einsum('nij,j->ni', A_all, m)               # [N, n_x]
    I_2lA = I[tf.newaxis] + 2 * lam * A_all                # [N, n_x, n_x]
    b_all = tf.einsum('nij,nj->ni', I_2lA, term1 + term2)  # [N, n_x]

    return A_all, b_all


class LEDHFlow(BaseFlow):
    """Local EDH particle flow (per-particle linearization).

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
        R_inv = tf.linalg.inv(ssm.R)
        lam_pos = self.lambda_schedule

        for j in tf.range(1, self.n_flow_steps + 1):
            lam = lam_pos[j]
            eps = lam - lam_pos[j - 1]
            H_all = batch_H_jac(ssm, particles)
            h_all = batch_h(ssm, particles)
            A_all, b_all = compute_ledh_matrices_batch(
                particles, m_pred, P_pred, H_all, h_all, ssm.R, y, lam, R_inv)
            particles = particles + eps * (
                tf.einsum('nij,nj->ni', A_all, particles) + b_all)

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
