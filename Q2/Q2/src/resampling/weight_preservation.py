"""Weight-preservation resampling (Chen et al. 2023, p.9, ref [73]).

Multinomial resampling of particles, but weights are set to the mean
of the pre-resampling weights. This preserves total weight mass and
creates a differentiable gradient path through the weight computation.

Formula: w_new_i = (1/N) * sum_j(w_j) for all i
In log-space: log_w_new = logsumexp(log_w) - log(N)

Gradient flows through logsumexp: d(log_w_new)/d(log_w_j) = softmax(log_w)_j
"""
import tensorflow as tf

from .base import ResamplerBase
from .systematic import MultinomialResampler


class WeightPreservationResampler(ResamplerBase):
    """Multinomial resampling with weight preservation.

    Particles are resampled via multinomial (non-differentiable).
    New weights = mean of old weights (differentiable through logsumexp).

    DIFFERENTIABLE = True because gradient flows through the weight path.
    """

    DIFFERENTIABLE = True

    def __init__(self):
        self._inner = MultinomialResampler()

    def apply(self, log_weights, particles, rng=None):
        """Apply weight-preservation resampling.

        Parameters
        ----------
        log_weights : tf.Tensor [..., N]
        particles : tf.Tensor [..., N, d]
        rng : unused

        Returns
        -------
        particles_new, log_weights_new : same shapes
        """
        particles_new, _ = self._inner.apply(log_weights, particles, rng)

        N_f = tf.cast(tf.shape(log_weights)[-1], particles.dtype)
        mean_log_w = (tf.reduce_logsumexp(log_weights, axis=-1, keepdims=True)
                      - tf.math.log(N_f))
        log_weights_new = tf.broadcast_to(mean_log_w, tf.shape(log_weights))

        return particles_new, log_weights_new
