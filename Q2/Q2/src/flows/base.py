"""Base particle flow interface -- SDE framework.

BaseFlow provides the SDE integration framework:
    dx = drift(x, lam) dlam + L dW_lam

Subclasses override _transport() to define their transport mechanism.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import tensorflow as tf

from ..filters.base import FilterResult
from ..filters.base_particle import BaseParticleFilter
from ..filters.common import DTYPE
from .flow_utils import predict_step


class BaseFlow(BaseParticleFilter, ABC):
    """Abstract base class for particle flow methods (SDE framework).

    EDH, LEDH, and StochasticFlow all follow dx = drift(x,lam)dlam + L dW_lam.
    BaseFlow provides the sequential filter loop; subclasses implement _transport().
    """

    def __init__(self, n_particles: int = 500, n_flow_steps: int = 20,
                 lambda_schedule: Optional[np.ndarray] = None,
                 filter_type: str = 'ekf'):
        super().__init__(n_particles)
        self.n_flow_steps = n_flow_steps
        self.filter_type = filter_type

        if lambda_schedule is not None:
            self.lambda_schedule = tf.constant(lambda_schedule, dtype=DTYPE)
            self.n_flow_steps = len(lambda_schedule) - 1
        else:
            self.lambda_schedule = tf.cast(
                tf.linspace(0.0, 1.0, n_flow_steps + 1), dtype=DTYPE)

    @abstractmethod
    def _transport(self, particles, m_pred, P_pred, ssm, y, rng):
        """Transport particles from prior to posterior.

        Parameters
        ----------
        particles : tf.Tensor [N, n_x]
            Propagated particles (after f + noise).
        m_pred : tf.Tensor [n_x]
            Predicted mean.
        P_pred : tf.Tensor [n_x, n_x]
            Predicted covariance.
        ssm : SSM
        y : tf.Tensor [n_y]
        rng : tf.random.Generator

        Returns
        -------
        particles : tf.Tensor [N, n_x]
        """

    def _compute_posterior(self, particles, m_pred, P_pred, y, ssm):
        """Compute posterior moments from transported particles.

        Default: ensemble moments. Override for EKF/UKF-based posterior.
        """
        return self.ensemble_moments(particles)

    def flow_step(self, particles, m_prev, P_prev, ssm, y, rng):
        """Full flow step: predict -> propagate -> transport -> posterior.

        Parameters
        ----------
        particles : tf.Tensor [N, n_x]
        m_prev : tf.Tensor [n_x]
        P_prev : tf.Tensor [n_x, n_x]
        ssm : SSM
        y : tf.Tensor [n_y]
        rng : tf.random.Generator

        Returns
        -------
        particles : tf.Tensor [N, n_x]
        weights : tf.Tensor [N]
        x_hat : tf.Tensor [n_x]
        P_hat : tf.Tensor [n_x, n_x]
        """
        N = tf.shape(particles)[0]

        m_pred, P_pred = predict_step(m_prev, P_prev, ssm, self.filter_type)

        # Propagate through dynamics + noise
        particles = ssm.f_batch(particles) + ssm.Q_sampler(rng, N)

        # Transport (subclass-specific)
        particles = self._transport(particles, m_pred, P_pred, ssm, y, rng)

        # Posterior moments
        x_hat, P_hat = self._compute_posterior(particles, m_pred, P_pred, y, ssm)

        # Uniform weights (flow methods don't use importance weights)
        N_f = tf.cast(N, DTYPE)
        weights = tf.ones(N, dtype=DTYPE) / N_f

        return particles, weights, x_hat, P_hat

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

        particles = self.init_particles(ssm.m0, ssm.P0, N, rng)
        x_hat, P_hat = ssm.m0, ssm.P0

        m_arr = tf.TensorArray(DTYPE, size=T)
        P_arr = tf.TensorArray(DTYPE, size=T)
        ess_arr = tf.TensorArray(DTYPE, size=T)

        for t in tf.range(T):
            particles, weights, x_hat, P_hat = self.flow_step(
                particles, x_hat, P_hat, ssm, ys[t], rng)
            m_arr = m_arr.write(t, x_hat)
            P_arr = P_arr.write(t, P_hat)
            ess_arr = ess_arr.write(t, self.compute_ess(weights))

        return m_arr.stack(), P_arr.stack(), ess_arr.stack()
