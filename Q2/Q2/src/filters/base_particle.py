"""Base class for particle-based filters."""
from abc import ABC, abstractmethod

import tensorflow as tf

from .base import BaseFilter, FilterResult
from .common import DTYPE


class BaseParticleFilter(BaseFilter, ABC):
    """Abstract base class for all particle-based filters.

    Provides shared particle utilities used by bootstrap PF,
    flow-based filters, and PFPF methods.
    """

    def __init__(self, n_particles: int):
        self.n_particles = n_particles

    @staticmethod
    def init_particles(m0, P0, N, rng):
        """Sample initial particles from N(m0, P0).

        Parameters
        ----------
        m0 : tf.Tensor [n_x]
        P0 : tf.Tensor [n_x, n_x]
        N : int
        rng : tf.random.Generator

        Returns
        -------
        particles : tf.Tensor [N, n_x]
        """
        n_x = tf.shape(m0)[0]
        L_P0 = tf.linalg.cholesky(P0)
        z = rng.normal(shape=(N, n_x), dtype=DTYPE)
        return m0[tf.newaxis, :] + tf.linalg.matvec(L_P0, z)

    @staticmethod
    def weighted_moments(particles, weights):
        """Compute weighted mean and covariance.

        Parameters
        ----------
        particles : tf.Tensor [N, n_x]
        weights : tf.Tensor [N]

        Returns
        -------
        m : tf.Tensor [n_x]
        P : tf.Tensor [n_x, n_x]
        """
        m = tf.einsum('i,ij->j', weights, particles)
        diff = particles - m[tf.newaxis, :]
        P = tf.einsum('i,ij,ik->jk', weights, diff, diff)
        return m, P

    @staticmethod
    def ensemble_moments(particles):
        """Compute sample mean and covariance (equal weights).

        Parameters
        ----------
        particles : tf.Tensor [N, n_x]

        Returns
        -------
        m : tf.Tensor [n_x]
        P : tf.Tensor [n_x, n_x]
        """
        N_f = tf.cast(tf.shape(particles)[0], DTYPE)
        m = tf.reduce_mean(particles, axis=0)
        diff = particles - m[tf.newaxis, :]
        P = tf.einsum('ij,ik->jk', diff, diff) / (N_f - 1.0)
        return m, P

    @staticmethod
    def compute_ess(weights):
        """Compute effective sample size.

        Parameters
        ----------
        weights : tf.Tensor [N]

        Returns
        -------
        ess : tf.Tensor scalar
        """
        return 1.0 / tf.reduce_sum(weights ** 2)
