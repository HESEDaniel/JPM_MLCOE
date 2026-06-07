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

    def __init__(self, n_particles=1000, resample_threshold=0.5,
                 export_particles: bool = False):
        super().__init__(n_particles)
        self.resample_threshold = resample_threshold
        self.export_particles = export_particles

    def filter(self, ssm, ys, rng=None, **kwargs):
        if rng is None:
            rng = tf.random.Generator.from_seed(42)
        ys = tf.cast(ys, DTYPE)
        m_filt, P_filt, ess, resample_count, log_marg, particles, weights = (
            self._filter_loop(ssm, ys, rng)
        )
        diag = {'ess': ess, 'resample_count': resample_count,
                'log_marginal_lik': log_marg}
        if self.export_particles:
            diag['particles'] = particles  # (T, N)
            diag['weights'] = weights      # (T, N), pre-resample softmax weights
        return FilterResult(m_filt, P_filt, diag)

    @tf.function
    def _filter_loop(self, ssm, ys, rng):
        T = tf.shape(ys)[0]
        n_x = ssm.state_dim
        N = self.n_particles
        N_f = tf.cast(N, DTYPE)

        particles = self.init_particles(ssm.m0, ssm.P0, N, rng)
        log_w = tf.zeros(N, dtype=DTYPE)   # CUMULATIVE FIX: carry over

        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        ess_arr = tf.TensorArray(DTYPE, size=T)
        particles_arr = tf.TensorArray(DTYPE, size=T)   # (T, N, n_x)
        weights_arr = tf.TensorArray(DTYPE, size=T)     # (T, N) pre-resample weights
        resample_count = tf.constant(0)
        log_marg_total = tf.constant(0.0, dtype=DTYPE)   # MLE PATCH

        for t in tf.range(T):
            noise = ssm.Q_sampler(rng, N)
            particles = ssm.f_batch(particles) + noise

            log_lik = ssm.log_likelihood(ys[t], particles)             # MLE PATCH: extracted
            # Incremental log p(y_t | y_{1:t-1}) = logsumexp(log_w + log_lik) - logsumexp(log_w).
            # log_w is the cumulative weight from the previous step (=0 after resample),
            # so this is exact under both branches of the resampling rule.
            log_marg_t = (tf.reduce_logsumexp(log_w + log_lik)
                          - tf.reduce_logsumexp(log_w))               # MLE PATCH
            log_marg_total = log_marg_total + log_marg_t              # MLE PATCH

            log_w = log_w + log_lik                                    # CUMULATIVE FIX: += not =
            w = tf.nn.softmax(log_w)

            ess_t = self.compute_ess(w)
            ess_arr = ess_arr.write(t, ess_t)

            # Pre-resample particles + weights: the actual importance-sampling
            # posterior approximation at time t (before any duplicate-creating
            # systematic resample step).
            particles_arr = particles_arr.write(t, particles)
            weights_arr = weights_arr.write(t, w)

            if ess_t < self.resample_threshold * N_f:
                idx = systematic_resample(log_w, rng)
                particles = tf.gather(particles, idx)
                w = tf.ones(N, dtype=DTYPE) / N_f
                log_w = tf.zeros(N, dtype=DTYPE)                   # CUMULATIVE FIX: reset on resample
                resample_count += 1

            m_t, P_t = self.weighted_moments(particles, w)
            m_arr = m_arr.write(t, m_t)
            P_arr = P_arr.write(t, P_t)

        return (m_arr.stack(), P_arr.stack(), ess_arr.stack(),
                resample_count, log_marg_total,
                particles_arr.stack(), weights_arr.stack())
