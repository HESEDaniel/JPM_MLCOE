"""Particle Flow Particle Filter with EDH -- Li17 Algorithm 2.

flow_type='li17': deterministic EDH ODE (Li17 Eq 10-11)
flow_type='dai22': stochastic Dai22 SDE (Eq 8, 11, 12)

Structure (shared): global EKF predict -> propagate -> particle flow -> weight -> update
"""
import math

import tensorflow as tf

from .edh import compute_edh_matrices
from .flow_utils import predict_step, update_step
from .ledh import batch_H_jac, batch_h
from ..filters.base import FilterResult
from ..filters.common import DTYPE
from .pfpf import PFPFFilter
from ..resampling import systematic_resample


class PFPFEDHFilter(PFPFFilter):
    """PF-PF Algorithm 2 (Li et al. 2017).

    Parameters
    ----------
    flow_type : str
        'li17' -- EDH deterministic flow (default).
        'dai22' -- Dai22 stochastic flow (requires Q_diff).
    weight_type : str
        SDE weight scheme (ignored for li17, which always uses paper formula).
        'uniform'       -- w = 1/N.
        'likelihood'    -- w propto p(z|eta_1).
        'det_jacobian'  -- w propto p(z|eta_1) * p(eta_1|x)/p(eta_0|x).
                          (EDH global F det cancels in normalization.)
    Q_diff : tf.Tensor [n_x, n_x], optional
        SDE diffusion matrix (required when flow_type='dai22').
    beta_schedule, beta_dot_schedule : array-like, optional
        Optimal homotopy beta*(lam) from Dai22. If None, uses beta=lam.
    """

    def __init__(self, n_particles=500, n_flow_steps=20,
                 lambda_schedule=None, filter_type='ekf',
                 resample_threshold=0.5,
                 flow_type='li17',
                 weight_type='likelihood',
                 Q_diff=None,
                 beta_schedule=None, beta_dot_schedule=None):
        super().__init__(n_particles, n_flow_steps, lambda_schedule,
                         filter_type, resample_threshold)
        self.flow_type = flow_type
        self.weight_type = weight_type
        self.Q_diff = Q_diff
        self._has_diffusion = Q_diff is not None and bool(tf.reduce_any(Q_diff > 0))
        if beta_schedule is not None:
            self._beta_schedule = tf.cast(beta_schedule, DTYPE)
            self._beta_dot_schedule = tf.cast(beta_dot_schedule, DTYPE)
        else:
            self._beta_schedule = None
            self._beta_dot_schedule = None

    def _integrate_flow(self, eta, m_pred, P_pred, ssm, y, rng):
        if self.flow_type == 'dai22':
            return self._flow_sde(eta, m_pred, P_pred, ssm, y, rng)
        return self._flow_ode(eta, m_pred, P_pred, ssm, y, rng)

    def _compute_log_weights(self, particles, f_x_prev, eta_0, y, ssm,
                             n_x, n_y, Q_inv, R_inv, log_det_Q, log_det_R):
        # li17: always use paper formula (likelihood + transition ratio)
        if self.flow_type == 'li17':
            return self._ode_log_weights(
                particles, f_x_prev, eta_0, y, ssm,
                n_x, n_y, Q_inv, R_inv, log_det_Q, log_det_R)

        # dai22: branch on weight_type
        if self.weight_type == 'uniform':
            return tf.zeros(tf.shape(particles)[0], dtype=DTYPE)

        # likelihood term (shared by 'likelihood' and 'det_jacobian')
        h_particles = batch_h(ssm, particles)
        log_2pi = tf.math.log(tf.constant(2.0 * math.pi, dtype=DTYPE))
        diff_y = y[tf.newaxis] - h_particles
        mahal_y = tf.reduce_sum(
            diff_y * tf.linalg.matvec(R_inv, diff_y), axis=1)
        ll = -0.5 * (log_det_R + tf.cast(n_y, DTYPE) * log_2pi + mahal_y)

        if self.weight_type == 'likelihood':
            return ll

        # det_jacobian: ll + transition ratio (global F det cancels)
        diff_new = particles - f_x_prev
        mahal_new = tf.reduce_sum(
            diff_new * tf.linalg.matvec(Q_inv, diff_new), axis=1)
        diff_old = eta_0 - f_x_prev
        mahal_old = tf.reduce_sum(
            diff_old * tf.linalg.matvec(Q_inv, diff_old), axis=1)
        return ll - 0.5 * (mahal_new - mahal_old)

    @staticmethod
    def _ode_log_weights(particles, f_x_prev, eta_0, y, ssm,
                         n_x, n_y, Q_inv, R_inv, log_det_Q, log_det_R):
        """Default ODE weight formula: likelihood x transition density ratio."""
        h_particles = batch_h(ssm, particles)
        log_2pi = tf.math.log(tf.constant(2.0 * math.pi, dtype=DTYPE))

        # log p(y | x_i)
        diff_y = y[tf.newaxis] - h_particles
        mahal_y = tf.reduce_sum(
            diff_y * tf.linalg.matvec(R_inv, diff_y), axis=1)
        ll = -0.5 * (log_det_R + tf.cast(n_y, DTYPE) * log_2pi + mahal_y)

        # Transition density ratio (valid for deterministic transport)
        diff_new = particles - f_x_prev
        mahal_new = tf.reduce_sum(
            diff_new * tf.linalg.matvec(Q_inv, diff_new), axis=1)
        diff_old = eta_0 - f_x_prev
        mahal_old = tf.reduce_sum(
            diff_old * tf.linalg.matvec(Q_inv, diff_old), axis=1)
        tr = -0.5 * (mahal_new - mahal_old)

        return ll + tr

    def _flow_ode(self, eta, m_pred, P_pred, ssm, y, rng):
        """EDH ODE (Li17 Eq 10-11)."""
        eta_bar = m_pred
        lam_pos = self.lambda_schedule

        for j in tf.range(1, self.n_flow_steps + 1):
            lam = lam_pos[j]
            eps = lam - lam_pos[j - 1]
            H_curr = ssm.H_jac(eta_bar)
            A, b = compute_edh_matrices(
                m_pred, P_pred, H_curr, ssm.R, y, lam, eta_bar, ssm.h)
            eta_bar = eta_bar + eps * (tf.linalg.matvec(A, eta_bar) + b)
            eta = eta + eps * (eta @ tf.transpose(A) + b[tf.newaxis, :])

        return eta

    def _flow_sde(self, eta, m_pred, P_pred, ssm, y, rng):
        """Dai22 SDE (Eq 8, 11, 12)."""
        N = tf.shape(eta)[0]
        n_x = ssm.state_dim

        I_nx = tf.eye(n_x, dtype=DTYPE)
        P_pred_inv = tf.linalg.pinv(P_pred)
        R_inv = tf.linalg.inv(ssm.R)
        Q_diff_val = self.Q_diff if self.Q_diff is not None else tf.zeros((n_x, n_x), dtype=DTYPE)
        L_Q_diff = tf.linalg.cholesky(Q_diff_val) if self._has_diffusion else None

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
            H = ssm.H_jac(tf.reduce_mean(eta, axis=0))
            neg_hess = tf.transpose(H) @ R_inv @ H
            M = P_pred_inv + beta * neg_hess
            M_inv = tf.linalg.pinv(M)
            K1 = 0.5 * Q_diff_val - (bdot / 2.0) * M_inv @ neg_hess @ M_inv
            K2 = bdot * M_inv

            # Per-particle gradients
            g_prior = -tf.einsum('ij,nj->ni', P_pred_inv, eta - m_pred)
            H_all = batch_H_jac(ssm, eta)
            h_all = batch_h(ssm, eta)
            H_all_T = tf.transpose(H_all, [0, 2, 1])
            residual = y[tf.newaxis] - h_all
            R_inv_res = tf.einsum('ij,nj->ni', R_inv, residual)
            g_lik = tf.einsum('nij,nj->ni', H_all_T, R_inv_res)
            g_p = g_prior + beta * g_lik

            drift = (tf.einsum('ij,nj->ni', K1, g_p)
                     + tf.einsum('ij,nj->ni', K2, g_lik))

            if self._has_diffusion:
                noise = tf.linalg.matvec(
                    L_Q_diff, rng.normal(shape=(N, n_x), dtype=DTYPE))
                eta = eta + dl * drift + tf.sqrt(dl) * noise
            else:
                eta = eta + dl * drift

        return eta

    def flow_step(self, particles, m_prev, P_prev, ssm, y, rng,
                  weights_prev, Q_inv, R_inv, log_det_Q, log_det_R):
        N = tf.shape(particles)[0]
        n_x = ssm.state_dim
        n_y = ssm.obs_dim

        m_pred, P_pred = predict_step(m_prev, P_prev, ssm, self.filter_type)

        # Propagate particles with noise (batch -- no per-particle loop)
        f_x_prev = ssm.f_batch(particles)
        eta_0 = f_x_prev + ssm.Q_sampler(rng, N)

        # Flow transport (subclass-specific)
        eta_1 = self._integrate_flow(tf.identity(eta_0), m_pred, P_pred, ssm, y, rng)
        particles = eta_1

        # Importance weights (batch -- no per-particle loop)
        log_w = self._compute_log_weights(
            particles, f_x_prev, eta_0, y, ssm, n_x, n_y,
            Q_inv, R_inv, log_det_Q, log_det_R)
        log_weights = log_w + tf.math.log(weights_prev)
        log_weights = log_weights - tf.reduce_max(log_weights)
        weights = tf.nn.softmax(log_weights)

        m_post, P_post = update_step(m_pred, P_pred, y, ssm, self.filter_type)
        x_hat = tf.einsum('i,ij->j', weights, particles)

        return particles, weights, x_hat, P_post

    def filter(self, ssm, ys, rng=None, **kwargs):
        if rng is None:
            rng = tf.random.Generator.from_seed(42)
        ys = tf.cast(ys, DTYPE)

        # Precompute (pinv handles rank-deficient Q)
        Q_inv = tf.linalg.pinv(ssm.Q)
        log_det_Q = tf.linalg.slogdet(ssm.Q)[1]
        R_inv = tf.linalg.inv(ssm.R)
        log_det_R = tf.linalg.slogdet(ssm.R)[1]

        m_filt, P_filt, ess, resample_count = self._filter_loop(
            ssm, ys, rng, Q_inv, R_inv, log_det_Q, log_det_R)
        return FilterResult(
            m_filt, P_filt,
            {'ess': ess, 'resample_count': resample_count})

    @tf.function
    def _filter_loop(self, ssm, ys, rng, Q_inv, R_inv, log_det_Q, log_det_R):
        T = tf.shape(ys)[0]
        N = self.n_particles
        N_f = tf.cast(N, DTYPE)

        particles = self.init_particles(ssm.m0, ssm.P0, N, rng)
        weights = tf.ones(N, dtype=DTYPE) / N_f
        x_hat, P_hat = ssm.m0, ssm.P0

        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        ess_arr = tf.TensorArray(DTYPE, size=T)
        resample_count = tf.constant(0)

        for t in tf.range(T):
            particles, weights, x_hat, P_hat = self.flow_step(
                particles, x_hat, P_hat, ssm, ys[t], rng,
                weights, Q_inv, R_inv, log_det_Q, log_det_R)

            ess_t = self.compute_ess(weights)
            ess_arr = ess_arr.write(t, ess_t)

            if ess_t < self.resample_threshold * N_f:
                idx = systematic_resample(tf.math.log(weights), rng)
                particles = tf.gather(particles, idx)
                weights = tf.ones(N, dtype=DTYPE) / N_f
                resample_count += 1

            m_arr = m_arr.write(t, x_hat)
            P_arr = P_arr.write(t, P_hat)

        return m_arr.stack(), P_arr.stack(), ess_arr.stack(), resample_count
