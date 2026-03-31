"""RKHS Particle Flow Filter (TensorFlow)."""
import tensorflow as tf

from ..filters.base import FilterResult
from ..filters.base_particle import BaseParticleFilter
from ..filters.common import DTYPE


def localization_matrix(n_x, r_in=4.0):
    """Gaussian localization: C[i,j] = exp(-((i-j)/r_in)^2)."""
    idx = tf.cast(tf.range(n_x), DTYPE)
    return tf.exp(-((idx[:, tf.newaxis] - idx[tf.newaxis, :]) / r_in)**2)


def _rkhs_flow_step(particles, h_fn, H_jac_fn, R, y, B_loc, B_loc_inv,
                     R_inv, bandwidths, kernel_type, step_size):
    """One step of RKHS particle flow (Hu21, Algorithm 1-2).

    Parameters
    ----------
    particles : tf.Tensor [N, n_x]
    h_fn, H_jac_fn : callable
    R : tf.Tensor [n_y, n_y]
    y : tf.Tensor [n_y]
    B_loc : tf.Tensor [n_x, n_x] - localized ensemble covariance
    B_loc_inv : tf.Tensor [n_x, n_x]
    R_inv : tf.Tensor [n_y, n_y]
    bandwidths : tf.Tensor [n_x] or None
    kernel_type : str
    step_size : float

    Returns
    -------
    particles : tf.Tensor [N, n_x]
    flow_mag : tf.Tensor scalar
    """
    N = tf.shape(particles)[0]
    n_x = tf.shape(particles)[1]
    N_f = tf.cast(N, DTYPE)
    x_bar = tf.reduce_mean(particles, axis=0)

    # Log-posterior gradient for each particle
    def grad_i(x_i):
        H_i = H_jac_fn(x_i)
        innov = y - h_fn(x_i)
        grad_lik = tf.linalg.matvec(tf.transpose(H_i), tf.linalg.matvec(R_inv, innov))
        grad_prior = -tf.linalg.matvec(B_loc_inv, x_i - x_bar)
        return grad_lik + grad_prior

    grad_log_post = tf.vectorized_map(grad_i, particles)  # [N, n_x]

    if kernel_type == 'scalar':
        alpha_inv_B = tf.linalg.inv(B_loc)  # A = inv(alpha * B_loc), alpha absorbed
        # Pairwise kernel
        diff = particles[:, tf.newaxis, :] - particles[tf.newaxis, :, :]  # [N, N, n_x]
        quad = tf.reduce_sum(tf.einsum('ijk,kl->ijl', diff, alpha_inv_B) * diff, axis=-1)  # [N, N]
        K = tf.exp(-0.5 * quad)  # [N, N]

        # Stein operator
        term1 = tf.linalg.matmul(K, grad_log_post)  # [N, n_x]
        # div_K: sum over j of grad_j K(x_i, x_j)
        div_K = -tf.einsum('ij,ijk,kl->il', K, diff, alpha_inv_B)  # [N, n_x]
        I_flow = (term1 + div_K) / N_f
    else:
        # Matrix-valued kernel (per-dimension)
        I_flow_parts = tf.TensorArray(DTYPE, size=n_x)
        for d in tf.range(n_x):
            diff_d = particles[:, d:d+1] - tf.transpose(particles[:, d:d+1])  # [N, N]
            K_d = tf.exp(-diff_d**2 / (2.0 * bandwidths[d]))  # [N, N]
            grad_K_d = diff_d / bandwidths[d] * K_d  # [N, N]
            term1 = tf.linalg.matvec(K_d, grad_log_post[:, d])  # [N]
            term2 = tf.reduce_sum(grad_K_d, axis=1)  # [N]
            I_flow_parts = I_flow_parts.write(d, (term1 + term2) / N_f)
        I_flow = tf.transpose(I_flow_parts.stack())  # [N, n_x]

    flow = I_flow @ tf.transpose(B_loc)  # [N, n_x]
    flow_mag = tf.reduce_mean(tf.norm(flow, axis=1))
    particles = particles + step_size * flow

    return particles, flow_mag


class RKHSFlow(BaseParticleFilter):
    """RKHS Particle Flow Filter (Hu21).

    Inherits BaseParticleFilter directly -- RKHS doesn't fit the SDE framework
    (no lam-schedule, uses kernel Stein operator instead of drift+diffusion).

    Parameters
    ----------
    n_particles : int
    n_flow_steps : int
    step_size : float
    loc_radius : float
        Localization radius for ensemble covariance.
    bandwidth_alpha : float or None
        Bandwidth scaling (default: 1/N).
    kernel_type : str
        'scalar' or 'matrix-valued'.
    adaptive_step : bool
        Enable adaptive step size.
    step_factor : float
    decrease_patience : int
    """

    def __init__(self, n_particles=100, n_flow_steps=10, step_size=0.1,
                 loc_radius=4.0, bandwidth_alpha=None,
                 kernel_type='matrix-valued', adaptive_step=False,
                 step_factor=1.4, decrease_patience=20,
                 min_step=1e-6, max_step=1.0):
        super().__init__(n_particles)
        self.n_flow_steps = n_flow_steps
        self.step_size = step_size
        self.loc_radius = loc_radius
        self.bandwidth_alpha = bandwidth_alpha
        self.kernel_type = kernel_type
        self.adaptive_step = adaptive_step
        self.step_factor = step_factor
        self.decrease_patience = decrease_patience
        self.min_step = min_step
        self.max_step = max_step

    def _apply_flow(self, particles, ssm, y):
        """Apply RKHS flow to particles for one observation."""
        N = tf.shape(particles)[0]
        n_x = ssm.state_dim
        N_f = tf.cast(N, DTYPE)

        x_bar = tf.reduce_mean(particles, axis=0)
        X = tf.transpose(particles - x_bar)
        B = X @ tf.transpose(X) / (N_f - 1.0)
        C_loc = localization_matrix(n_x, self.loc_radius)
        B_loc = B * C_loc
        B_loc_inv = tf.linalg.inv(B_loc)
        R_inv = tf.linalg.inv(ssm.R)

        alpha = self.bandwidth_alpha if self.bandwidth_alpha is not None else 1.0 / N_f

        if self.kernel_type == 'scalar':
            bandwidths = None
            B_loc_scaled = alpha * B_loc
            B_loc_inv_for_kernel = tf.linalg.inv(B_loc_scaled)
        else:
            bandwidths = alpha * tf.linalg.diag_part(B_loc)
            B_loc_inv_for_kernel = None

        current_step = self.step_size
        prev_flow_mag = float('inf')
        consecutive_decreases = 0

        for _ in range(self.n_flow_steps):
            particles, flow_mag = _rkhs_flow_step(
                particles, ssm.h, ssm.H_jac, ssm.R, y,
                B_loc, B_loc_inv, R_inv, bandwidths,
                self.kernel_type, current_step)

            if self.adaptive_step:
                fm = float(flow_mag)
                if fm < prev_flow_mag:
                    consecutive_decreases += 1
                    if consecutive_decreases >= self.decrease_patience:
                        current_step = min(current_step * self.step_factor, self.max_step)
                        consecutive_decreases = 0
                else:
                    current_step = max(current_step / self.step_factor, self.min_step)
                    consecutive_decreases = 0
                prev_flow_mag = fm

        return particles

    def flow_step(self, particles, m_prev, P_prev, ssm, y, rng):
        N = tf.shape(particles)[0]

        # Propagate with noise
        noise = ssm.Q_sampler(rng, N)
        particles = ssm.f_batch(particles) + noise

        # Apply RKHS flow
        particles = self._apply_flow(particles, ssm, y)

        # Equal weights (no resampling needed)
        x_hat, P_hat = self.ensemble_moments(particles)
        N_f = tf.cast(N, DTYPE)
        weights = tf.ones(N, dtype=DTYPE) / N_f
        return particles, weights, x_hat, P_hat

    def filter(self, ssm, ys, rng=None, **kwargs):
        if rng is None:
            rng = tf.random.Generator.from_seed(42)
        ys = tf.cast(ys, DTYPE)
        T = tf.shape(ys)[0]
        N = self.n_particles

        particles = self.init_particles(ssm.m0, ssm.P0, N, rng)

        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        ess_arr = tf.TensorArray(DTYPE, size=T)

        for t in tf.range(T):
            particles, weights, x_hat, P_hat = self.flow_step(
                particles, None, None, ssm, ys[t], rng)
            m_arr = m_arr.write(t, x_hat)
            P_arr = P_arr.write(t, P_hat)
            ess_arr = ess_arr.write(t, tf.cast(N, DTYPE))

        return FilterResult(m_arr.stack(), P_arr.stack(),
                            {'ess': ess_arr.stack(), 'resample_count': 0})
