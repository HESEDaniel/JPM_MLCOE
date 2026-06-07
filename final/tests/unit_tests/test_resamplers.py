"""Regression tests for the differentiable resamplers.

Focus: the Scibior-Masrani-Wood ``StopGradientResampler`` must (1) reset the
forward log-weights to uniform ``-log N`` (standard-PF forward pass) and
(2) carry the unbiased gradient through the *gathered* log-softmax, aligned
with the resampled particles. A previous implementation returned the
pre-resample weights (non-uniform) and failed to gather by ancestor index;
these tests would catch that regression.
"""
import numpy as np
import tensorflow as tf

from src.inference.q2.resampling import StopGradientResampler
from src.inference.q2.resampling.weight_preservation import WeightPreservationResampler

DT = tf.float64


def _log_w():
    return tf.Variable(np.array([0.5, -1.0, 2.0, 0.0, -0.3, 1.2]), dtype=DT)


def test_stop_gradient_forward_value_is_uniform():
    """Forward log-weights must be uniform -log N (weights reset on resample)."""
    tf.random.set_seed(0)
    log_w = _log_w()
    N = int(log_w.shape[0])
    particles = tf.reshape(tf.range(N, dtype=DT), (N, 1))
    _, log_w_new = StopGradientResampler().apply(log_w, particles)
    assert np.allclose(log_w_new.numpy(), -np.log(N), atol=1e-9), log_w_new.numpy()


def test_stop_gradient_gradient_matches_scibior():
    """Gradient of sum(log_w_new) must equal the gathered-log-softmax form
    counts(idx) - N * softmax(log_w), i.e. the Scibior unbiased correction
    evaluated at the *resampled ancestors* (not the original index order)."""
    tf.random.set_seed(0)
    log_w = _log_w()
    N = int(log_w.shape[0])
    # particles = arange so particles_new[:, 0] recovers the internal ancestors.
    particles = tf.reshape(tf.range(N, dtype=DT), (N, 1))
    with tf.GradientTape() as tape:
        p_new, log_w_new = StopGradientResampler().apply(log_w, particles)
        loss = tf.reduce_sum(log_w_new)
    g = tape.gradient(loss, log_w)
    assert g is not None and np.all(np.isfinite(g.numpy()))

    idx = tf.cast(p_new[:, 0], tf.int32)
    sm = tf.nn.softmax(log_w)
    counts = tf.cast(tf.math.bincount(idx, minlength=N), DT)
    expected = counts - tf.cast(N, DT) * sm  # d/d log_w of sum_i log_softmax(log_w)[idx_i]
    assert np.allclose(g.numpy(), expected.numpy(), atol=1e-9), (g.numpy(), expected.numpy())


def test_stop_gradient_particles_are_gathered():
    """Resampled particles must be a subset of the originals (multinomial draw)."""
    tf.random.set_seed(1)
    log_w = _log_w()
    N = int(log_w.shape[0])
    particles = tf.reshape(tf.range(N, dtype=DT), (N, 1))
    p_new, _ = StopGradientResampler().apply(log_w, particles)
    idx = set(int(i) for i in p_new[:, 0].numpy())
    assert idx.issubset(set(range(N)))


def test_weight_preservation_forward_value_is_uniform():
    """Cross-check the other differentiable resampler also resets to uniform
    (value = mean weight = logsumexp - log N, equal across particles)."""
    tf.random.set_seed(0)
    log_w = _log_w()
    N = int(log_w.shape[0])
    particles = tf.reshape(tf.range(N, dtype=DT), (N, 1))
    _, log_w_new = WeightPreservationResampler().apply(log_w, particles)
    vals = log_w_new.numpy()
    assert np.allclose(vals, vals[0], atol=1e-12)  # all equal (uniform)
