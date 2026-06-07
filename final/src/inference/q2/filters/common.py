"""Common TensorFlow utilities for filters and flows."""
import tensorflow as tf

DTYPE = tf.float64


def masked_log_softmax(logits, mask, axis=-1):
    """Log-softmax restricted to available items, with no sentinel logit.

    Computes ``log p_j = logit_j - log(sum_{k: mask_k} exp(logit_k))`` directly.
    The availability ``mask`` (bool or {0,1}) enters as a multiplicative factor
    inside the log-sum-exp, so unavailable items contribute 0 to the normalizer
    without filling them with -Inf or a large-negative constant. Because the
    log-probability is formed directly (no intermediate probability), the
    gradient is the bounded softmax form and stays finite even when an item's
    probability underflows -- avoiding the 1/p blow-up of a probability-space
    mask, so this is safe on the differentiable PF + Adam path. The per-row max
    offset cancels analytically and is only for overflow safety. Unavailable
    entries get a finite (never-selected) value.
    """
    mask_f = tf.cast(mask, logits.dtype)
    m = tf.reduce_max(logits, axis=axis, keepdims=True)
    z = tf.reduce_sum(mask_f * tf.exp(logits - m), axis=axis, keepdims=True)
    return logits - (tf.math.log(z) + m)
