"""Soft resampling (Karkus et al. 2018, Chen et al. 2023 Section 5.1).

Chen(23) Eq. 47: W_tilde = lambda * W + (1-lambda) / N
Chen(23) Eq. 48: w_new = W / W_tilde  (weight correction)
Resample indices from Mult(W_tilde) via multinomial sampling.

All computations in log domain to avoid underflow (logsumexp trick).

Gradient flow through weight correction term log(W) - log(W_soft):
    - Multinomial sampling (tf.random.categorical): non-differentiable (discrete)
    - tf.gather(log_W, indices): differentiable w.r.t. log_W, not w.r.t. indices
    - log(W) - log(W_soft): gradient pathway from new_weights back to log_weights

Supports batched [..., N] / [..., N, d] inputs.
"""
import tensorflow as tf

from .base import ResamplerBase


class SoftResampler(ResamplerBase):
    """Soft resampler with blended weight distribution.

    Parameters
    ----------
    alpha : float
        Blending coefficient in [0, 1]. alpha=1 is standard resampling,
        alpha=0 is uniform (no resampling).
    """

    DIFFERENTIABLE = True

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def apply(self, log_weights, particles, rng=None):
        """Apply soft resampling.

        Parameters
        ----------
        log_weights : tf.Tensor [..., N]
        particles : tf.Tensor [..., N, d]
        rng : unused

        Returns
        -------
        resampled_particles, new_log_weights : same shapes
        """
        dtype = particles.dtype
        N = tf.shape(particles)[-2]
        N_f = tf.cast(N, dtype)
        alpha = tf.constant(self.alpha, dtype=dtype)

        log_W = log_weights - tf.reduce_logsumexp(log_weights, axis=-1, keepdims=True)

        # Blended distribution in log domain (Chen(23) Eq. 47)
        log_alpha = tf.math.log(alpha)
        log_uniform = tf.math.log((1.0 - alpha) / N_f)
        log_W_soft = tf.reduce_logsumexp(
            tf.stack([log_alpha + log_W,
                      tf.broadcast_to(log_uniform, tf.shape(log_W))], axis=0),
            axis=0)  # [..., N]

        # Multinomial sampling from Mult(W_soft) -- non-differentiable
        orig_shape = tf.shape(log_weights)
        flat_logits = tf.reshape(log_W_soft, [-1, orig_shape[-1]])
        logits_f32 = tf.cast(flat_logits, tf.float32)
        indices = tf.cast(tf.random.categorical(logits_f32, N), tf.int32)

        # Batched gather for particles
        flat_p = tf.reshape(particles, [-1, orig_shape[-1], tf.shape(particles)[-1]])
        B = tf.shape(flat_p)[0]
        batch_idx = tf.repeat(tf.range(B)[:, tf.newaxis], N, axis=1)
        gather_idx = tf.stack([batch_idx, indices], axis=-1)
        particles_new = tf.gather_nd(flat_p, gather_idx)
        particles_new = tf.reshape(particles_new, tf.shape(particles))

        # Batched gather for log weights (all in log domain)
        flat_log_W = tf.reshape(log_W, [-1, orig_shape[-1]])
        flat_log_W_soft = tf.reshape(log_W_soft, [-1, orig_shape[-1]])
        log_W_selected = tf.gather_nd(flat_log_W, gather_idx)
        log_W_soft_selected = tf.gather_nd(flat_log_W_soft, gather_idx)

        # Weight correction (Chen(23) Eq. 48)
        new_log_weights = log_W_selected - log_W_soft_selected
        new_log_weights = new_log_weights - tf.reduce_logsumexp(new_log_weights, axis=-1, keepdims=True)
        new_log_weights = tf.reshape(new_log_weights, orig_shape)

        return particles_new, new_log_weights
