"""Differentiable Ensemble Transform (DET) resampling via Sinkhorn OT.

Exact match to filterflow RegularisedTransform implementation:
  filterflow/resampling/differentiable/regularized_transport/

Reference: Corenflos et al. (2021), Algorithm 3.
"""
import tensorflow as tf

from .base import ResamplerBase


@tf.custom_gradient
def _clip_grad(x):
    """Identity forward, clip gradient to [-1, 1] backward.

    Matches filterflow transport() @tf.custom_gradient which clips
    d_transport to [-1, 1] before backpropagating.
    """
    def grad(dy):
        return tf.clip_by_value(dy, -1.0, 1.0)
    return x, grad


def _squared_distances(x, y):
    """[..., N, d] x [..., M, d] -> [..., N, M]"""
    xx = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    yy = tf.reduce_sum(y ** 2, axis=-1)[..., tf.newaxis, :]
    xy = tf.matmul(x, y, transpose_b=True)
    return tf.maximum(xx - 2.0 * xy + yy, 0.0)


def _cost(x, y):
    """Half squared distance."""
    return _squared_distances(x, y) / 2.0


def _diameter(x):
    """max_k std_i(X^i_k).  [..., N, d] -> [...]"""
    d = tf.reduce_max(tf.math.reduce_std(x, axis=-2), axis=-1)
    return tf.where(d == 0.0, tf.ones_like(d), d)


def _max_min(x, y):
    """max(max(x,y)) - min(min(x,y)) per batch.  Used for epsilon_0 in sinkhorn_loop."""
    max_val = tf.maximum(
        tf.reduce_max(x, axis=[-2, -1]),
        tf.reduce_max(y, axis=[-2, -1]))
    min_val = tf.minimum(
        tf.reduce_min(x, axis=[-2, -1]),
        tf.reduce_min(y, axis=[-2, -1]))
    return max_val - min_val


def _softmin(eps, C, f):
    """-eps * logsumexp((f - C/eps), axis=-1).
    eps: [..., 1], C: [..., N, M], f: [..., M] -> [..., N]
    """
    f_ = f[..., tf.newaxis, :]
    # Match filterflow: f_ - C / eps_reshaped
    eps_r = eps[..., tf.newaxis]  # [..., 1, 1]
    temp = f_ - C / eps_r
    return -eps * tf.reduce_logsumexp(temp, axis=-1)


def _sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy,
                   epsilon, particles_diameter, scaling, threshold, max_iter):
    """Sinkhorn with epsilon scaling and self-interaction costs.

    Exact match to filterflow sinkhorn_loop.

    Parameters
    ----------
    log_alpha, log_beta : [..., N]
    cost_xy, cost_yx, cost_xx, cost_yy : [..., N, N]
    epsilon : float (target)
    particles_diameter : [...] (for epsilon_0)
    scaling : float
    threshold, max_iter : float, int

    Returns
    -------
    a_y, b_x : [..., N] final potentials
    """
    dtype = log_alpha.dtype
    eps_target = tf.constant(epsilon, dtype=dtype)
    scaling_sq = tf.constant(scaling ** 2, dtype=dtype)

    eps_0 = particles_diameter ** 2
    running_eps = eps_0[..., tf.newaxis]  # [..., 1]

    # Initialize
    a_y = _softmin(running_eps, cost_yx, log_alpha)
    b_x = _softmin(running_eps, cost_xy, log_beta)
    a_x = _softmin(running_eps, cost_xx, log_alpha)
    b_y = _softmin(running_eps, cost_yy, log_beta)

    for _ in tf.range(max_iter):
        a_y_prev = a_y

        re = running_eps  # [..., 1]
        at_y = _softmin(re, cost_yx, log_alpha + b_x / re)
        bt_x = _softmin(re, cost_xy, log_beta + a_y / re)
        at_x = _softmin(re, cost_xx, log_alpha + a_x / re)
        bt_y = _softmin(re, cost_yy, log_beta + b_y / re)

        a_y = 0.5 * (a_y + at_y)
        b_x = 0.5 * (b_x + bt_x)
        a_x = 0.5 * (a_x + at_x)
        b_y = 0.5 * (b_y + bt_y)

        # Convergence check
        err = tf.reduce_max(tf.abs(a_y - a_y_prev))

        # Decrease epsilon
        new_eps = tf.maximum(running_eps * scaling_sq, eps_target)
        still_scaling = (new_eps < running_eps)
        running_eps = new_eps

        if err < threshold and not tf.reduce_any(still_scaling):
            break

    # Stop gradient + final extrapolation at target epsilon
    a_y, b_x, a_x, b_y = [tf.stop_gradient(v) for v in (a_y, b_x, a_x, b_y)]

    final_a_y = _softmin(eps_target, cost_yx, log_alpha + b_x / eps_target)
    final_b_x = _softmin(eps_target, cost_xy, log_beta + a_y / eps_target)

    return final_a_y, final_b_x


def _transport_from_potentials(x, f, g, eps, logw, N_f):
    """[..., N, d], [..., N], [..., N] -> [..., N, N]"""
    dtype = f.dtype
    eps_t = tf.constant(eps, dtype=dtype)
    log_n = tf.math.log(tf.cast(N_f, dtype))

    cost_matrix = _cost(x, x)
    fg = f[..., :, tf.newaxis] + g[..., tf.newaxis, :]
    temp = (fg - cost_matrix) / eps_t

    temp = temp - tf.reduce_logsumexp(temp, axis=-2, keepdims=True) + log_n
    temp = temp + logw[..., tf.newaxis, :]

    return tf.exp(temp)


class SinkhornOTResampler(ResamplerBase):
    """Differentiable Ensemble Transform (DET).

    Exact match to filterflow RegularisedTransform.

    Parameters
    ----------
    epsilon : float
        Target regularization (default 0.5, paper Section 5.2).
    scaling : float
        Epsilon annealing factor (default 0.9, author code).
    max_iters : int
    threshold : float
    """

    DIFFERENTIABLE = True

    def __init__(self, epsilon=0.5, scaling=0.9, max_iters=100, threshold=1e-3):
        self.epsilon = epsilon
        self.scaling = scaling
        self.max_iters = max_iters
        self.threshold = threshold

    def apply(self, log_weights, particles, rng=None):
        """Apply DET resampling.

        Parameters
        ----------
        log_weights : tf.Tensor [..., N]
        particles : tf.Tensor [..., N, d]

        Returns
        -------
        particles_new, log_weights_new : same shapes
        """
        N = tf.shape(particles)[-2]
        dtype = particles.dtype
        N_f = tf.cast(N, dtype)
        d = tf.cast(tf.shape(particles)[-1], dtype)

        logw = log_weights - tf.reduce_logsumexp(log_weights, axis=-1, keepdims=True)

        # Compute transport with custom gradient (author: clip [-1, 1])
        T = self._transport_with_clip(particles, logw, N_f, d)

        # Apply to ORIGINAL particles
        particles_new = tf.matmul(T, particles)
        log_weights_new = tf.fill(tf.shape(log_weights), -tf.math.log(N_f))

        return particles_new, log_weights_new

    def _transport_with_clip(self, particles, logw, N_f, d):
        """Compute transport matrix with gradient clipping [-1, 1].

        Following filterflow @tf.custom_gradient on transport().
        """
        log_uniform = tf.fill(tf.shape(logw), -tf.math.log(N_f))

        # Center and scale
        mean = tf.stop_gradient(tf.reduce_mean(particles, axis=-2, keepdims=True))
        centered = particles - mean
        diam = _diameter(particles)
        scale = tf.stop_gradient(diam[..., tf.newaxis, tf.newaxis] * tf.sqrt(d))
        scaled_x = centered / scale

        # Cost matrices
        sx_sg = tf.stop_gradient(scaled_x)
        cost_xy = _cost(scaled_x, sx_sg)
        cost_yx = _cost(sx_sg, scaled_x)
        cost_xx = _cost(scaled_x, sx_sg)
        cost_yy = _cost(sx_sg, scaled_x)

        mm_scale = tf.stop_gradient(_max_min(scaled_x, scaled_x))

        # Sinkhorn
        alpha, beta = _sinkhorn_loop(
            logw, log_uniform,
            cost_xy, cost_yx, cost_xx, cost_yy,
            self.epsilon, mm_scale, self.scaling,
            self.threshold, self.max_iters)

        # Transport matrix
        T = _transport_from_potentials(scaled_x, alpha, beta,
                                       self.epsilon, logw, N_f)

        # Clip gradient to [-1, 1] (filterflow @tf.custom_gradient)
        # Identity forward, clipped backward
        T = _clip_grad(T)

        return T
