"""Particle Flow Particle Filter with LEDH -- Li17 Algorithm 1.

flow_type='li17': deterministic LEDH ODE (Li17 Eq 13-14)
flow_type='dai22': stochastic Dai22 SDE (Eq 8, 11, 12) with per-particle P^i

Structure (shared): per-particle EKF predict -> propagate -> particle flow -> weight -> update
"""
import math

import tensorflow as tf

from .ledh import batch_H_jac, batch_h
from ..filters.base import FilterResult
from ..filters.common import DTYPE
from .pfpf import PFPFFilter
from ..resampling import systematic_resample


def _batch_F_jac(ssm, x_all):
    """Compute F_jac for all particles (tiles if constant)."""
    N = tf.shape(x_all)[0]
    F0 = ssm.F_jac(x_all[0])
    F1 = ssm.F_jac(x_all[0] + tf.ones_like(x_all[0]))
    return tf.cond(
        tf.reduce_all(tf.equal(F0, F1)),
        lambda: tf.tile(F0[tf.newaxis], [N, 1, 1]),
        lambda: tf.map_fn(ssm.F_jac, x_all, fn_output_signature=DTYPE))


class PFPFLEDHFilter(PFPFFilter):
    """PF-PF Algorithm 1 (Li et al. 2017).

    Parameters
    ----------
    flow_type : str
        'li17' -- LEDH deterministic flow (default).
        'dai22' -- Dai22 stochastic flow with per-particle P^i (requires Q_diff).
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

    def _transport(self, eta_0, f_x_prev, P_pred_all, ssm, y, rng, R_inv):
        """Dispatch to ODE or SDE flow."""
        if self.flow_type == 'dai22':
            return self._flow_sde(eta_0, f_x_prev, P_pred_all, ssm, y, rng, R_inv)
        return self._flow_ode(eta_0, f_x_prev, P_pred_all, ssm, y, rng, R_inv)

    def _flow_ode(self, eta_0, f_x_prev, P_pred_all, ssm, y, rng, R_inv):
        """LEDH ODE (Li17 Eq 13-14)."""
        N = tf.shape(eta_0)[0]
        n_x = ssm.state_dim
        I = tf.eye(n_x, dtype=DTYPE)
        log_jacobian = tf.zeros(N, dtype=DTYPE)

        eta_bar = tf.identity(f_x_prev)   # auxiliary (noise-free)
        eta_1 = tf.identity(eta_0)         # actual (noisy)

        lam_pos = self.lambda_schedule
        for j in tf.range(1, self.n_flow_steps + 1):
            lam = lam_pos[j]
            eps = lam - lam_pos[j - 1]

            H_all = batch_H_jac(ssm, eta_bar)
            h_all = batch_h(ssm, eta_bar)
            H_all_T = tf.transpose(H_all, [0, 2, 1])

            PH = P_pred_all @ H_all_T
            S_lam = lam * (H_all @ PH) + ssm.R[tf.newaxis]
            S_inv_H = tf.linalg.solve(S_lam, H_all)
            A_all = -0.5 * (PH @ S_inv_H)

            Hx = tf.einsum('nij,nj->ni', H_all, eta_bar)
            e_lam = h_all - Hx
            y_corr = y[tf.newaxis] - e_lam
            R_inv_y = tf.einsum('ij,nj->ni', R_inv, y_corr)

            IplusLA = I[tf.newaxis] + lam * A_all
            term1 = tf.einsum('nij,nj->ni', IplusLA @ PH, R_inv_y)
            term2 = tf.einsum('nij,nj->ni', A_all, f_x_prev)
            I_2lA = I[tf.newaxis] + 2 * lam * A_all
            b_all = tf.einsum('nij,nj->ni', I_2lA, term1 + term2)

            eta_bar = eta_bar + eps * (
                tf.einsum('nij,nj->ni', A_all, eta_bar) + b_all)
            eta_1 = eta_1 + eps * (
                tf.einsum('nij,nj->ni', A_all, eta_1) + b_all)

            log_jacobian += tf.linalg.slogdet(
                I[tf.newaxis] + eps * A_all)[1]
            # Intermediate normalization to prevent overflow (matches np version)
            log_jacobian = log_jacobian - tf.reduce_max(log_jacobian)

        return eta_1, log_jacobian

    def _flow_sde(self, eta_0, f_x_prev, P_pred_all, ssm, y, rng, R_inv):
        """Dai22 SDE (Eq 8, 11, 12) with per-particle P^i."""
        N = tf.shape(eta_0)[0]
        n_x = ssm.state_dim
        I = tf.eye(n_x, dtype=DTYPE)

        P_pred_inv_all = tf.linalg.pinv(P_pred_all)
        Q_diff_val = self.Q_diff if self.Q_diff is not None else tf.zeros((n_x, n_x), dtype=DTYPE)
        L_Q_diff = tf.linalg.cholesky(Q_diff_val) if self._has_diffusion else None
        compute_det = (self.weight_type == 'det_jacobian')
        log_det_F = tf.zeros(N, dtype=DTYPE)

        lam = self.lambda_schedule
        beta_sched = self._beta_schedule if self._beta_schedule is not None else lam
        bdot_sched = (self._beta_dot_schedule if self._beta_dot_schedule is not None
                      else tf.ones(self.n_flow_steps + 1, dtype=DTYPE))

        eta = tf.identity(eta_0)

        for j in tf.range(self.n_flow_steps):
            j1 = j + 1
            dl = lam[j1] - lam[j]
            beta = beta_sched[j1]
            bdot = bdot_sched[j1]

            H_all = batch_H_jac(ssm, eta)
            h_all = batch_h(ssm, eta)
            H_all_T = tf.transpose(H_all, [0, 2, 1])

            neg_hess = H_all_T @ (R_inv[tf.newaxis] @ H_all)
            M = P_pred_inv_all + beta * neg_hess
            M_inv = tf.linalg.pinv(M)

            K1 = (0.5 * Q_diff_val[tf.newaxis]
                  - (bdot / 2.0) * M_inv @ neg_hess @ M_inv)
            K2 = bdot * M_inv

            g_prior = -tf.einsum('nij,nj->ni', P_pred_inv_all, eta - f_x_prev)
            residual = y[tf.newaxis] - h_all
            R_inv_res = tf.einsum('ij,nj->ni', R_inv, residual)
            g_lik = tf.einsum('nij,nj->ni', H_all_T, R_inv_res)
            g_p = g_prior + beta * g_lik

            drift = (tf.einsum('nij,nj->ni', K1, g_p)
                     + tf.einsum('nij,nj->ni', K2, g_lik))

            if compute_det:
                F_all = -(K1 @ M + K2 @ neg_hess)
                log_det_F += tf.linalg.slogdet(I[tf.newaxis] + dl * F_all)[1]

            if self._has_diffusion:
                noise = tf.linalg.matvec(
                    L_Q_diff, rng.normal(shape=(N, n_x), dtype=DTYPE))
                eta = eta + dl * drift + tf.sqrt(dl) * noise
            else:
                eta = eta + dl * drift

        return eta, log_det_F

    def flow_step(self, particles, P_prev_all, ssm, y, rng,
                  weights_prev, Q_inv, R_inv, log_det_R):
        """Algorithm 1 flow step (shared for ODE/SDE)."""
        N = tf.shape(particles)[0]
        n_x = ssm.state_dim
        n_y = ssm.obs_dim
        I = tf.eye(n_x, dtype=DTYPE)
        log_2pi = tf.math.log(tf.constant(2.0 * math.pi, dtype=DTYPE))

        # Per-particle EKF prediction (Alg.1 Lines 4-5)
        f_x_prev = ssm.f_batch(particles)
        F_all = _batch_F_jac(ssm, particles)
        FP = F_all @ P_prev_all
        P_pred_all = FP @ tf.transpose(F_all, [0, 2, 1]) + ssm.Q[tf.newaxis]

        # Propagate (Alg.1 Lines 6-10)
        eta_0 = f_x_prev + ssm.Q_sampler(rng, N)

        # Particle flow (ODE or SDE)
        eta_1, log_correction = self._transport(
            eta_0, f_x_prev, P_pred_all, ssm, y, rng, R_inv)

        # Importance weights
        particles_new = eta_1
        N_f = tf.cast(N, DTYPE)

        if self.flow_type == 'dai22' and self.weight_type == 'uniform':
            weights = tf.ones(N, dtype=DTYPE) / N_f
        else:
            h_particles = batch_h(ssm, particles_new)
            diff_y = y[tf.newaxis] - h_particles
            mahal_y = tf.reduce_sum(
                diff_y * tf.linalg.matvec(R_inv, diff_y), axis=1)
            ll = -0.5 * (log_det_R + tf.cast(n_y, DTYPE) * log_2pi + mahal_y)

            log_w = ll + log_correction

            # transition ratio: ODE always, SDE det_jacobian only
            if self.flow_type == 'li17' or self.weight_type == 'det_jacobian':
                diff_new = particles_new - f_x_prev
                mahal_new = tf.reduce_sum(
                    diff_new * tf.linalg.matvec(Q_inv, diff_new), axis=1)
                diff_old = eta_0 - f_x_prev
                mahal_old = tf.reduce_sum(
                    diff_old * tf.linalg.matvec(Q_inv, diff_old), axis=1)
                log_w += -0.5 * (mahal_new - mahal_old)

            log_w += tf.math.log(weights_prev)
            log_w -= tf.reduce_max(log_w)
            weights = tf.nn.softmax(log_w)

        x_hat = tf.einsum('i,ij->j', weights, particles_new)

        # Per-particle EKF update (Alg.1 Line 28)
        H_upd = batch_H_jac(ssm, f_x_prev)
        H_upd_T = tf.transpose(H_upd, [0, 2, 1])

        S_upd = H_upd @ P_pred_all @ H_upd_T + ssm.R[tf.newaxis]
        K_upd = P_pred_all @ H_upd_T @ tf.linalg.inv(S_upd)
        P_post_all = (I[tf.newaxis] - K_upd @ H_upd) @ P_pred_all

        return particles_new, weights, x_hat, P_post_all

    def filter(self, ssm, ys, rng=None, **kwargs):
        if rng is None:
            rng = tf.random.Generator.from_seed(42)
        ys = tf.cast(ys, DTYPE)
        n_x = ssm.state_dim

        Q_inv = tf.linalg.pinv(ssm.Q)
        R_inv = tf.linalg.inv(ssm.R)
        log_det_R = tf.linalg.slogdet(ssm.R)[1]

        m_filt, P_filt, ess, resample_count = self._filter_loop(
            ssm, ys, rng, Q_inv, R_inv, log_det_R)
        return FilterResult(
            m_filt, P_filt,
            {'ess': ess, 'resample_count': resample_count})

    @tf.function
    def _filter_loop(self, ssm, ys, rng, Q_inv, R_inv, log_det_R):
        T = tf.shape(ys)[0]
        N = self.n_particles
        n_x = ssm.state_dim
        N_f = tf.cast(N, DTYPE)

        particles = self.init_particles(ssm.m0, ssm.P0, N, rng)
        weights = tf.ones(N, dtype=DTYPE) / N_f
        P_prev_all = tf.tile(ssm.P0[tf.newaxis], [N, 1, 1])

        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        ess_arr = tf.TensorArray(DTYPE, size=T)
        resample_count = tf.constant(0)

        for t in tf.range(T):
            particles, weights, x_hat, P_prev_all = self.flow_step(
                particles, P_prev_all, ssm, ys[t], rng,
                weights, Q_inv, R_inv, log_det_R)

            ess_t = self.compute_ess(weights)
            ess_arr = ess_arr.write(t, ess_t)

            if ess_t < self.resample_threshold * N_f:
                idx = systematic_resample(tf.math.log(weights), rng)
                particles = tf.gather(particles, idx)
                P_prev_all = tf.gather(P_prev_all, idx)
                weights = tf.ones(N, dtype=DTYPE) / N_f
                resample_count += 1

            m_arr = m_arr.write(t, x_hat)
            diff = particles - x_hat[tf.newaxis]
            P_hat = tf.einsum('i,ij,ik->jk', weights, diff, diff)
            P_arr = P_arr.write(t, P_hat)

        return m_arr.stack(), P_arr.stack(), ess_arr.stack(), resample_count
