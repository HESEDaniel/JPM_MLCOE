"""Bootstrap Particle Filter (TensorFlow)."""
import tensorflow as tf

from .base import FilterResult
from .base_particle import BaseParticleFilter
from .common import DTYPE
from ..resampling import systematic_resample


class ParticleFilter(BaseParticleFilter):
    """Bootstrap Particle Filter.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    resample_threshold : float
        ESS ratio threshold for triggering systematic resampling.
    """

    def __init__(self, n_particles=1000, resample_threshold=0.5):
        super().__init__(n_particles)
        self.resample_threshold = resample_threshold

    def filter(self, ssm, ys, rng=None, **kwargs):
        if rng is None:
            rng = tf.random.Generator.from_seed(42)
        ys = tf.cast(ys, DTYPE)
        m_filt, P_filt, ess, resample_count = self._filter_loop(ssm, ys, rng)
        return FilterResult(
            m_filt, P_filt,
            {'ess': ess, 'resample_count': resample_count})

    @tf.function
    def _filter_loop(self, ssm, ys, rng):
        T = tf.shape(ys)[0]
        n_x = ssm.state_dim
        N = self.n_particles
        N_f = tf.cast(N, DTYPE)

        particles = self.init_particles(ssm.m0, ssm.P0, N, rng)

        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        ess_arr = tf.TensorArray(DTYPE, size=T)
        resample_count = tf.constant(0)

        for t in tf.range(T):
            noise = ssm.Q_sampler(rng, N)
            particles = ssm.f_batch(particles) + noise

            log_w = ssm.log_likelihood(ys[t], particles)
            w = tf.nn.softmax(log_w)

            ess_t = self.compute_ess(w)
            ess_arr = ess_arr.write(t, ess_t)

            if ess_t < self.resample_threshold * N_f:
                idx = systematic_resample(log_w, rng)
                particles = tf.gather(particles, idx)
                w = tf.ones(N, dtype=DTYPE) / N_f
                resample_count += 1

            m_t, P_t = self.weighted_moments(particles, w)
            m_arr = m_arr.write(t, m_t)
            P_arr = P_arr.write(t, P_t)

        return m_arr.stack(), P_arr.stack(), ess_arr.stack(), resample_count
