"""Standard resampling schemes wrapped as ResamplerBase."""
import tensorflow as tf

from .base import ResamplerBase
DTYPE = tf.float64


def systematic_resample(log_weights, rng):
    """Systematic resampling (low variance).

    Uses a single uniform random number and evenly spaced points
    on the CDF. Lower variance than multinomial.

    Parameters
    ----------
    log_weights : tf.Tensor [N]
    rng : tf.random.Generator

    Returns
    -------
    indices : tf.Tensor [N] int32
    """
    N = tf.shape(log_weights)[0]
    N_f = tf.cast(N, DTYPE)
    w = tf.nn.softmax(log_weights)
    cumsum = tf.cumsum(w)
    u0 = rng.uniform(shape=(), minval=0.0, maxval=1.0 / N_f, dtype=DTYPE)
    u = u0 + tf.cast(tf.range(N), DTYPE) / N_f
    indices = tf.searchsorted(cumsum, u)
    return tf.clip_by_value(indices, 0, N - 1)


def multinomial_resample(log_weights, N=None):
    """Multinomial resampling via tf.random.categorical.

    Parameters
    ----------
    log_weights : tf.Tensor [N]
        Unnormalized log weights.
    N : int, optional
        Number of samples. Defaults to len(log_weights).

    Returns
    -------
    indices : tf.Tensor [N] int32
    """
    if N is None:
        N = tf.shape(log_weights)[0]
    logits = tf.cast(log_weights[tf.newaxis, :], tf.float32)
    indices = tf.random.categorical(logits, N)
    return tf.cast(indices[0], tf.int32)


class SystematicResampler(ResamplerBase):
    """Systematic (low-variance) resampler. Not differentiable."""

    DIFFERENTIABLE = False

    def apply(self, log_weights, particles, rng):
        N = tf.shape(particles)[0]
        N_f = tf.cast(N, particles.dtype)
        indices = systematic_resample(log_weights, rng)
        particles_new = tf.gather(particles, indices)
        log_weights_new = tf.fill([N], -tf.math.log(N_f))
        return particles_new, log_weights_new


class MultinomialResampler(ResamplerBase):
    """Multinomial resampler. Not differentiable.

    Supports [..., N] / [..., N, d] shapes (batch-compatible).
    Standard resampling used in Corenflos(2021) as the PF baseline.
    """

    DIFFERENTIABLE = False

    def apply(self, log_weights, particles, rng=None):
        N = tf.shape(particles)[-2]
        N_f = tf.cast(N, particles.dtype)

        # tf.random.categorical expects [batch, num_classes]
        orig_shape = tf.shape(log_weights)
        flat_lw = tf.reshape(log_weights, [-1, orig_shape[-1]])          # [B, N]
        logits_f32 = tf.cast(flat_lw, tf.float32)
        indices = tf.cast(tf.random.categorical(logits_f32, N), tf.int32)  # [B, N]

        # Gather per batch element
        flat_p = tf.reshape(particles, [-1, orig_shape[-1], tf.shape(particles)[-1]])  # [B, N, d]
        B = tf.shape(flat_p)[0]
        batch_idx = tf.repeat(tf.range(B)[:, tf.newaxis], N, axis=1)     # [B, N]
        gather_idx = tf.stack([batch_idx, indices], axis=-1)              # [B, N, 2]
        particles_new = tf.gather_nd(flat_p, gather_idx)                  # [B, N, d]

        particles_new = tf.reshape(particles_new, tf.shape(particles))
        log_weights_new = tf.fill(tf.shape(log_weights), -tf.math.log(N_f))
        return particles_new, log_weights_new
