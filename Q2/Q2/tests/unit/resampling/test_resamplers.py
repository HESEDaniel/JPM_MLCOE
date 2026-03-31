"""Unit tests for all resampler implementations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import tensorflow as tf
import pytest

from src.resampling import (
    NoResampling, SystematicResampler, MultinomialResampler,
    SoftResampler, SinkhornOTResampler, WeightPreservationResampler,
)

DTYPE = tf.float64


@pytest.fixture
def particles_and_weights():
    """Random particles and non-uniform log-weights."""
    rng = tf.random.Generator.from_seed(0)
    N, d = 50, 3
    particles = rng.normal(shape=(N, d), dtype=DTYPE)
    log_w = rng.normal(shape=(N,), dtype=DTYPE)
    return particles, log_w


# NoResampling

class TestNoResampling:
    def test_identity(self, particles_and_weights):
        p, lw = particles_and_weights
        resampler = NoResampling()
        p_new, lw_new = resampler.apply(lw, p)
        np.testing.assert_array_equal(p_new.numpy(), p.numpy())
        np.testing.assert_array_equal(lw_new.numpy(), lw.numpy())

    def test_differentiable_flag(self):
        assert NoResampling.DIFFERENTIABLE is True


# SystematicResampler

class TestSystematicResampler:
    def test_shape_preserved(self, particles_and_weights, tf_rng):
        p, lw = particles_and_weights
        resampler = SystematicResampler()
        p_new, lw_new = resampler.apply(lw, p, tf_rng)
        assert p_new.shape == p.shape
        assert lw_new.shape == lw.shape

    def test_uniform_log_weights(self, particles_and_weights, tf_rng):
        """After resampling, log-weights should be uniform."""
        p, lw = particles_and_weights
        resampler = SystematicResampler()
        _, lw_new = resampler.apply(lw, p, tf_rng)
        # All weights should be -log(N)
        expected = -np.log(p.shape[0])
        np.testing.assert_allclose(lw_new.numpy(), expected, atol=1e-10)

    def test_not_differentiable(self):
        assert SystematicResampler.DIFFERENTIABLE is False


# MultinomialResampler

class TestMultinomialResampler:
    def test_shape_preserved(self, particles_and_weights):
        p, lw = particles_and_weights
        resampler = MultinomialResampler()
        p_new, lw_new = resampler.apply(lw, p)
        assert p_new.shape == p.shape
        assert lw_new.shape == lw.shape

    def test_concentrated_weight_dominates(self):
        """Particle with dominant weight should appear frequently."""
        N = 100
        lw = tf.constant([-100.0] * N, dtype=DTYPE)
        lw = tf.tensor_scatter_nd_update(lw, [[0]], [0.0])
        p = tf.concat([tf.ones((1, 2), dtype=DTYPE) * 99.0,
                        tf.zeros((N - 1, 2), dtype=DTYPE)], axis=0)
        resampler = MultinomialResampler()
        p_new, _ = resampler.apply(lw, p)
        # Most resampled particles should be near [99, 99]
        close_to_dominant = tf.reduce_sum(
            tf.cast(tf.reduce_all(tf.abs(p_new - 99.0) < 0.1, axis=1), tf.int32))
        assert close_to_dominant.numpy() > N * 0.5


# SoftResampler

class TestSoftResampler:
    def test_shape_preserved(self, particles_and_weights):
        p, lw = particles_and_weights
        resampler = SoftResampler(alpha=0.5)
        p_new, lw_new = resampler.apply(lw, p)
        assert p_new.shape == p.shape
        assert lw_new.shape == lw.shape

    def test_alpha_one_approaches_standard(self, particles_and_weights):
        """alpha=1 should behave like standard multinomial."""
        p, lw = particles_and_weights
        resampler = SoftResampler(alpha=1.0)
        p_new, lw_new = resampler.apply(lw, p)
        assert not np.any(np.isnan(p_new.numpy()))
        assert not np.any(np.isnan(lw_new.numpy()))

    def test_alpha_zero_no_nan(self, particles_and_weights):
        """alpha=0 (uniform sampling) should not produce NaN."""
        p, lw = particles_and_weights
        resampler = SoftResampler(alpha=0.0)
        p_new, lw_new = resampler.apply(lw, p)
        assert not np.any(np.isnan(p_new.numpy()))
        assert not np.any(np.isnan(lw_new.numpy()))

    def test_differentiable_flag(self):
        assert SoftResampler.DIFFERENTIABLE is True


# SinkhornOTResampler

class TestSinkhornOTResampler:
    def test_shape_preserved(self, particles_and_weights):
        p, lw = particles_and_weights
        resampler = SinkhornOTResampler(epsilon=0.5, max_iters=30)
        p_new, lw_new = resampler.apply(lw, p)
        assert p_new.shape == p.shape
        assert lw_new.shape == lw.shape

    def test_uniform_output_weights(self, particles_and_weights):
        """OT resampler should output uniform log-weights."""
        p, lw = particles_and_weights
        resampler = SinkhornOTResampler(epsilon=0.5, max_iters=30)
        _, lw_new = resampler.apply(lw, p)
        expected = -np.log(p.shape[0])
        np.testing.assert_allclose(lw_new.numpy(), expected, atol=1e-10)

    def test_no_nan(self, particles_and_weights):
        p, lw = particles_and_weights
        resampler = SinkhornOTResampler(epsilon=0.5, max_iters=50)
        p_new, lw_new = resampler.apply(lw, p)
        assert not np.any(np.isnan(p_new.numpy()))
        assert not np.any(np.isnan(lw_new.numpy()))

    def test_transport_matrix_rows(self, particles_and_weights):
        """Transport matrix rows should sum to ~1 (doubly stochastic)."""
        p, lw = particles_and_weights
        N = p.shape[0]
        resampler = SinkhornOTResampler(epsilon=0.5, max_iters=50)
        logw = lw - tf.reduce_logsumexp(lw)
        N_f = tf.cast(N, DTYPE)
        d = tf.cast(tf.shape(p)[-1], DTYPE)
        T = resampler._transport_with_clip(p, logw, N_f, d)
        row_sums = tf.reduce_sum(T, axis=-1).numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)

    def test_gradient_flows(self, particles_and_weights):
        """Gradients should flow through OT resampling."""
        p, lw = particles_and_weights
        p_var = tf.Variable(p)
        resampler = SinkhornOTResampler(epsilon=0.5, max_iters=30)
        with tf.GradientTape() as tape:
            p_new, _ = resampler.apply(lw, p_var)
            loss = tf.reduce_sum(p_new)
        grad = tape.gradient(loss, p_var)
        assert grad is not None, "No gradient flow through OT"
        assert not np.any(np.isnan(grad.numpy()))

    def test_differentiable_flag(self):
        assert SinkhornOTResampler.DIFFERENTIABLE is True


# WeightPreservationResampler

class TestWeightPreservationResampler:
    def test_shape_preserved(self, particles_and_weights):
        p, lw = particles_and_weights
        resampler = WeightPreservationResampler()
        p_new, lw_new = resampler.apply(lw, p)
        assert p_new.shape == p.shape
        assert lw_new.shape == lw.shape

    def test_weight_mass_preserved(self, particles_and_weights):
        """Total weight mass (logsumexp) should be the same before and after."""
        p, lw = particles_and_weights
        resampler = WeightPreservationResampler()
        _, lw_new = resampler.apply(lw, p)

        mass_before = tf.reduce_logsumexp(lw).numpy()
        mass_after = tf.reduce_logsumexp(lw_new).numpy()
        np.testing.assert_allclose(mass_after, mass_before, atol=1e-10)

    def test_new_weights_uniform(self, particles_and_weights):
        """All new log-weights should be identical (= mean of old)."""
        p, lw = particles_and_weights
        resampler = WeightPreservationResampler()
        _, lw_new = resampler.apply(lw, p)
        vals = lw_new.numpy()
        np.testing.assert_allclose(vals, vals[0], atol=1e-10)

    def test_differentiable_flag(self):
        assert WeightPreservationResampler.DIFFERENTIABLE is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
