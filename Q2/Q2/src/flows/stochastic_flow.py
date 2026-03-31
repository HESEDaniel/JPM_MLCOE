"""Stochastic Particle Flow - Dai(22).

SDE (8):  dx = f(x,lam)dlam + q dw_lam
Drift:    f = K_1 grad(log p) + K_2 grad(log h)

With alpha+beta=1 (Gaussian approx):
  K_1 = 0.5Q - (beta_dot/2) M^-1 neg_hess_logh M^-1       [Eq 11]
  K_2 = beta_dot M^-1                                       [Eq 12]
  M(beta) = P^-1_pred + beta neg_hess_logh,  neg_hess_logh = -grad^2(log h) = H^T R^-1 H
"""
import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseFlow
from .ledh import batch_H_jac, batch_h
from ..filters.common import DTYPE


def solve_optimal_homotopy(P_prior_inv, neg_hess_logh, mu=0.2, n_grid=30,
                           n_bisect=50, bisect_tol=1e-8):
    """Solve BVP (Eq 26-27) for optimal beta*(lam) via shooting + bisection.

    Parameters
    ----------
    P_prior_inv : tf.Tensor [n_x, n_x]
        Inverse of prior covariance.
    neg_hess_logh : tf.Tensor [n_x, n_x]
        Negative Hessian of log-likelihood: -grad^2(log h) = H^T R^-1 H.
    mu : float
        Weight on condition number penalty in cost functional (Eq 25).
    n_grid : int
        Number of lam grid points for the output schedule.
    n_bisect : int
        Maximum bisection iterations for shooting method.
    bisect_tol : float
        Convergence tolerance on |beta(1) - 1| for early stopping.

    Returns
    -------
    beta : tf.Tensor [n_grid]
        Optimal homotopy schedule beta*(lam).
    beta_dot : tf.Tensor [n_grid]
        Derivative dbeta*/dlam at each grid point.
    lam_grid : tf.Tensor [n_grid]
        Lambda grid [0, ..., 1].
    """
    tr_neg_hess_logh = tf.linalg.trace(neg_hess_logh)
    mu_tf = tf.constant(mu, dtype=DTYPE)

    def beta_ddot(beta):
        """Compute beta_ddot = mu * d(kappa_*(M))/d(beta) (Eq 26)."""
        M = P_prior_inv + beta * neg_hess_logh
        M_inv = tf.linalg.inv(M)
        M_inv_sq = M_inv @ M_inv
        t1 = tr_neg_hess_logh * tf.linalg.trace(M_inv)
        t2 = tf.linalg.trace(M) * tf.linalg.trace(M_inv_sq @ neg_hess_logh)
        return mu_tf * (t1 - t2)

    def ode_fn(lam, state):
        """1st-order ODE system from 2nd-order BVP (Eq 26)."""
        beta, beta_dot = state[0], state[1]
        return tf.stack([beta_dot, beta_ddot(beta)])

    solver = tfp.math.ode.DormandPrince(rtol=1e-4, atol=1e-4)
    lam_grid = tf.cast(tf.linspace(0.0, 1.0, n_grid), DTYPE)

    def shoot(s):
        """Shoot from beta(0)=0, beta_dot(0)=s and return beta(1)."""
        y0 = tf.stack([tf.constant(0.0, dtype=DTYPE), s])
        r = solver.solve(ode_fn, tf.constant(0.0, dtype=DTYPE), y0,
                         solution_times=tf.constant([1.0], dtype=DTYPE))
        return r.states[0, 0]

    # Bisection shooting: find beta_dot(0) such that beta(1) = 1
    lo = tf.constant(0.5, dtype=DTYPE)
    hi = tf.constant(2.0, dtype=DTYPE)
    target = tf.constant(1.0, dtype=DTYPE)

    # Phase 1: bracket -- double hi until shoot(hi) >= 1
    max_bracket = 50
    bracketed = False
    for _ in range(max_bracket):
        if shoot(hi) < target:
            hi = hi * 2.0
        else:
            bracketed = True
            break
    if not bracketed:
        raise RuntimeError(
            f"Bracketing failed: shoot({float(hi):.1e}) < 1.0 "
            f"after {max_bracket} doublings. "
            f"Check P_prior_inv and neg_hess_logh conditioning.")

    # Phase 2: bisect until |beta(1) - 1| < tol or max iterations
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        val = shoot(mid)
        if tf.abs(val - target) < bisect_tol:
            break
        if val < target:
            lo = mid
        else:
            hi = mid
    bdot0 = (lo + hi) / 2.0

    # Solve on full grid with converged initial slope
    y0 = tf.stack([tf.constant(0.0, dtype=DTYPE), bdot0])
    result = solver.solve(ode_fn, tf.constant(0.0, dtype=DTYPE), y0,
                          solution_times=lam_grid)
    return result.states[:, 0], result.states[:, 1], lam_grid


class StochasticFlow(BaseFlow):
    """Stochastic particle flow filter (Dai 2022).

    Euler-Maruyama integration of the SDE (Eq 8):
      dx = [K_1 grad(log p) + K_2 grad(log h)] dlam + sqrt(Q_diff) dw

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_flow_steps : int
        Number of Euler-Maruyama integration steps per time step.
    Q_diff : tf.Tensor [n_x, n_x]
        Diffusion matrix for the SDE (user-specified, e.g. diag(4, 0.4)).
    beta_schedule : array-like [n_flow_steps+1], optional
        Optimal homotopy beta*(lam) from solve_optimal_homotopy.
        If None, uses straight-line beta(lam) = lam.
    beta_dot_schedule : array-like [n_flow_steps+1], optional
        Derivative dbeta*/dlam. Required if beta_schedule is provided.
    lambda_schedule : array-like, optional
        Custom lam grid. If None, uses uniform [0, ..., 1].
    filter_type : str
        'ekf' or 'ukf' for the prediction step.
    """

    def __init__(self, n_particles=50, n_flow_steps=29,
                 Q_diff=None,
                 beta_schedule=None, beta_dot_schedule=None,
                 lambda_schedule=None, filter_type='ekf'):
        super().__init__(n_particles, n_flow_steps, lambda_schedule, filter_type)
        self.Q_diff = Q_diff
        if beta_schedule is not None:
            self._beta_schedule = tf.cast(beta_schedule, DTYPE)
            self._beta_dot_schedule = tf.cast(beta_dot_schedule, DTYPE)
        else:
            self._beta_schedule = None
            self._beta_dot_schedule = None

    def _transport(self, particles, m_pred, P_pred, ssm, y, rng):
        N = tf.shape(particles)[0]
        n_x = ssm.state_dim

        P_pred_inv = tf.linalg.pinv(P_pred)
        R_inv = tf.linalg.inv(ssm.R)
        L_Q_diff = tf.linalg.cholesky(self.Q_diff)

        lam = self.lambda_schedule
        beta_sched = self._beta_schedule if self._beta_schedule is not None else lam
        bdot_sched = (self._beta_dot_schedule if self._beta_dot_schedule is not None
                      else tf.ones(self.n_flow_steps + 1, dtype=DTYPE))

        for j in tf.range(self.n_flow_steps):
            j1 = j + 1
            dl = lam[j1] - lam[j]
            beta = beta_sched[j1]
            bdot = bdot_sched[j1]

            # Global K1, K2 (Hessian at particle mean)
            H = ssm.H_jac(tf.reduce_mean(particles, axis=0))
            neg_hess_logh = tf.transpose(H) @ R_inv @ H
            M_inv = tf.linalg.pinv(P_pred_inv + beta * neg_hess_logh)

            K1 = 0.5 * self.Q_diff - (bdot / 2.0) * M_inv @ neg_hess_logh @ M_inv
            K2 = bdot * M_inv

            # Per-particle gradients (batch, no closure)
            g_prior = -tf.einsum('ij,nj->ni', P_pred_inv, particles - m_pred)
            H_all = batch_H_jac(ssm, particles)
            h_all = batch_h(ssm, particles)
            H_all_T = tf.transpose(H_all, [0, 2, 1])
            residual = y[tf.newaxis] - h_all
            R_inv_res = tf.einsum('ij,nj->ni', R_inv, residual)
            g_lik = tf.einsum('nij,nj->ni', H_all_T, R_inv_res)
            g_p = g_prior + beta * g_lik

            drift = (tf.einsum('ij,nj->ni', K1, g_p)
                     + tf.einsum('ij,nj->ni', K2, g_lik))

            noise = rng.normal(shape=(N, n_x), dtype=DTYPE)
            particles = (particles + dl * drift
                         + tf.sqrt(dl) * tf.linalg.matvec(L_Q_diff, noise))

        return particles
