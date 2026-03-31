"""Differentiable Particle Filter with pluggable resampling."""
import tensorflow as tf

from .base import FilterResult
from .base_particle import BaseParticleFilter
from .common import DTYPE
from ..resampling import SinkhornOTResampler


class DifferentiableParticleFilter(BaseParticleFilter):
    """Differentiable Particle Filter with OT resampling.

    A bootstrap particle filter that supports differentiable resampling
    via pluggable resampler objects. Default uses Sinkhorn OT resampling
    (Corenflos et al. 2021).

    Parameters
    ----------
    n_particles : int
        Number of particles.
    resampler : ResamplerBase, optional
        Resampler instance. Defaults to SinkhornOTResampler().
    resample_threshold : float
        ESS ratio threshold for triggering resampling.
    """

    def __init__(self, n_particles=100, resampler=None,
                 resample_threshold=0.5):
        super().__init__(n_particles)
        self.resampler = resampler if resampler is not None else SinkhornOTResampler()
        self.resample_threshold = resample_threshold

    def flow_step(self, particles, m_prev, P_prev, ssm, y, rng,
                  log_weights=None):
        N = tf.shape(particles)[0]
        N_f = tf.cast(N, DTYPE)

        noise = ssm.Q_sampler(rng, N)
        particles = ssm.f_batch(particles) + noise

        log_lik = ssm.log_likelihood(y, particles)
        if log_weights is not None:
            log_w = log_lik + log_weights
        else:
            log_w = log_lik
        log_w = log_w - tf.reduce_max(log_w)

        w = tf.nn.softmax(log_w)
        ess = 1.0 / tf.reduce_sum(w ** 2)

        if ess < self.resample_threshold * N_f:
            particles, log_w = self.resampler.apply(log_w, particles, rng)

        w_final = tf.nn.softmax(log_w)
        m_post = tf.einsum('i,ij->j', w_final, particles)
        diff = particles - m_post[tf.newaxis, :]
        P_post = tf.einsum('i,ij,ik->jk', w_final, diff, diff)

        return particles, log_w, m_post, P_post

    def filter(self, ssm, ys, rng=None, **kwargs):
        if rng is None:
            rng = tf.random.Generator.from_seed(42)
        ys = tf.cast(ys, DTYPE)
        m_filt, P_filt, ess = self._filter_loop(ssm, ys, rng)
        return FilterResult(m_filt, P_filt, {'ess': ess})

    @tf.function
    def _filter_loop(self, ssm, ys, rng):
        T = tf.shape(ys)[0]
        N = self.n_particles
        N_f = tf.cast(N, DTYPE)

        particles = self.init_particles(ssm.m0, ssm.P0, N, rng)
        log_weights = tf.fill([N], -tf.math.log(N_f))

        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        ess_arr = tf.TensorArray(DTYPE, size=T)

        for t in tf.range(T):
            particles, log_weights, m_t, P_t = self.flow_step(
                particles, None, None, ssm, ys[t], rng, log_weights)

            w = tf.nn.softmax(log_weights)
            ess_t = 1.0 / tf.reduce_sum(w ** 2)
            ess_arr = ess_arr.write(t, ess_t)
            m_arr = m_arr.write(t, m_t)
            P_arr = P_arr.write(t, P_t)

        return m_arr.stack(), P_arr.stack(), ess_arr.stack()
