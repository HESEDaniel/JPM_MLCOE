"""Stop-gradient resampling trick (Scibior-Masrani-Wood 2021).

Particles are resampled by multinomial sampling (non-differentiable, like a
standard PF). The new log-weights are constructed so the *forward value* is
uniform (``-log N``) -- exactly the standard-PF behaviour, so the forward pass
is unchanged -- while the *gradient* flows through the pre-resample
``log_softmax`` rather than through the categorical sampling step. This is the
mechanism behind ``tfp.experimental.mcmc.particle_filter`` with
``unbiased_gradients=True``.

Construction (per resampled particle, indexed by its ancestor ``a_i``):

    log_probs    = log_softmax(log_w)                 # gradient flows here
    log_probs_sg = log_softmax(stop_gradient(log_w))  # forward value
    a            ~ Categorical(log_probs_sg)           # ancestor indices
    w_new_i      = gather(log_probs - log_probs_sg - log N, a)

In the forward pass ``log_probs == log_probs_sg`` so ``w_new_i == -log N``
(uniform), and resampling has reset the weights exactly as a standard PF would.
The gradient is ``d log_softmax(log_w)`` evaluated at the *resampled* ancestors,
which is the Scibior unbiased-gradient correction. Both the ``-log N`` term and
the ``gather`` by ancestor index are essential: without them the retained
weights are neither uniform (standard PF) nor the unbiased Scibior value, and
they become misaligned with the resampled particles.

This matches the Scibior et al. reference implementation, cross-checked against
TFP's ``unbiased_gradients=True`` to within ~1% on the gradient.
"""
import tensorflow as tf

from .base import ResamplerBase


class StopGradientResampler(ResamplerBase):
    """Scibior-Masrani-Wood 2021 stop-gradient trick.

    Particles are multinomial-resampled (non-differentiable). Log-weights are
    rewritten so their forward value is uniform (``-log N``) but their gradient
    flows only through ``log_softmax(log_w)`` (the differentiable part of the
    importance-weight computation), giving an unbiased gradient estimate of the
    FIVO log-marginal-likelihood objective.
    """

    DIFFERENTIABLE = True

    def apply(self, log_weights, particles, rng=None):
        # ``log_weights``: (N,), ``particles``: (N, d).
        N = tf.shape(log_weights)[-1]
        log_N = tf.math.log(tf.cast(N, log_weights.dtype))

        # 1. Multinomial resample using stop_gradient(log_w) for the index draw
        #    (the forward pass is identical to a standard bootstrap PF).
        log_probs_sg = tf.nn.log_softmax(tf.stop_gradient(log_weights), axis=-1)
        idx = tf.cast(
            tf.random.categorical(tf.cast(log_probs_sg[tf.newaxis, :], tf.float32), N)[0],
            tf.int32,
        )
        particles_new = tf.gather(particles, idx)

        # 2. Scibior importance-weight trick. Value = -log N (uniform), gradient
        #    = d log_softmax(log_w); gathered to the resampled ancestors so the
        #    weights stay aligned with ``particles_new``.
        target_log_probs = tf.nn.log_softmax(log_weights, axis=-1)
        importance_weights = target_log_probs - log_probs_sg - log_N
        log_w_new = tf.gather(importance_weights, idx)
        return particles_new, log_w_new
