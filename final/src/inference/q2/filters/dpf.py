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
        # FIX (M2): a non-differentiable resampler silently blocks the gradient through
        # every resample step, so the "differentiable" PF then yields a biased gradient
        # for joint MLE. Warn loudly rather than fail (the bootstrap-PF baseline may
        # intentionally use a discrete resampler for forward-only filtering).
        if not getattr(self.resampler, "DIFFERENTIABLE", False):
            import warnings
            warnings.warn(
                f"{type(self.resampler).__name__} has DIFFERENTIABLE=False; gradients "
                "through resample steps are blocked. Use a differentiable resampler "
                "(stop_gradient/soft/sinkhorn/no_resample) for joint MLE.",
                RuntimeWarning, stacklevel=2,
            )
        self.resample_threshold = resample_threshold

    def flow_step(self, particles, m_prev, P_prev, ssm, y, rng,
                  log_weights=None):
        N = tf.shape(particles)[0]
        N_f = tf.cast(N, DTYPE)

        noise = ssm.Q_sampler(rng, N)
        particles = ssm.f_batch(particles) + noise

        log_lik = ssm.log_likelihood(y, particles)
        if log_weights is not None:
            # MLE PATCH: incremental log marginal lik
            #   log p(y_t | y_{1:t-1}) = logsumexp(log_w_prev + log_lik) - logsumexp(log_w_prev).
            # Robust to the normalisation applied to log_w on the previous step
            # (the absolute scale cancels in the ratio).
            log_marg_t = (tf.reduce_logsumexp(log_lik + log_weights)
                          - tf.reduce_logsumexp(log_weights))
            log_w = log_lik + log_weights
        else:
            log_marg_t = tf.reduce_logsumexp(log_lik) - tf.math.log(N_f)  # MLE PATCH
            log_w = log_lik
        log_w = log_w - tf.reduce_max(log_w)

        w = tf.nn.softmax(log_w)
        ess = 1.0 / tf.reduce_sum(w ** 2)
        # ess is the *pre-resample* ESS --- the actual particle diversity that
        # the resampler reacts to. Post-resample ESS is artefactually N (uniform
        # weights), which is not a meaningful diagnostic.

        if ess < self.resample_threshold * N_f:
            particles, log_w = self.resampler.apply(log_w, particles, rng)

        w_final = tf.nn.softmax(log_w)
        m_post = tf.einsum('i,ij->j', w_final, particles)
        diff = particles - m_post[tf.newaxis, :]
        P_post = tf.einsum('i,ij,ik->jk', w_final, diff, diff)

        return particles, log_w, m_post, P_post, log_marg_t, ess  # MLE PATCH: 6th output = pre-resample ESS

    def filter(self, ssm, ys, rng=None, **kwargs):
        if rng is None:
            rng = tf.random.Generator.from_seed(42)
        ys = tf.cast(ys, DTYPE)
        m_filt, P_filt, ess, log_marg = self._filter_loop(ssm, ys, rng)   # MLE PATCH
        return FilterResult(m_filt, P_filt,
                            {'ess': ess, 'log_marginal_likelihood': log_marg})

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
        log_marg_total = tf.constant(0.0, dtype=DTYPE)   # MLE PATCH

        for t in tf.range(T):
            particles, log_weights, m_t, P_t, log_marg_t, ess_pre = self.flow_step(
                particles, None, None, ssm, ys[t], rng, log_weights)
            log_marg_total += log_marg_t   # MLE PATCH

            ess_arr = ess_arr.write(t, ess_pre)   # pre-resample ESS
            m_arr = m_arr.write(t, m_t)
            P_arr = P_arr.write(t, P_t)

        return m_arr.stack(), P_arr.stack(), ess_arr.stack(), log_marg_total   # MLE PATCH
