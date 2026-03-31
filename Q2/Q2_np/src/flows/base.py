"""Base particle flow interface and result container."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf


@dataclass
class FlowResult:
    """Unified container for particle flow filter outputs.

    Attributes
    ----------
    m_filt : tf.Tensor [T, n_x]
        Filtered state means.
    P_filt : tf.Tensor [T, n_x, n_x]
        Filtered state covariances.
    diagnostics : dict
        Algorithm-specific diagnostics. Common keys:
        - 'ess': tf.Tensor [T]
        - 'resample_count': int
        - 'weights_history': tf.Tensor [T, N]
        - 'flow_history': list or None
    """
    m_filt: tf.Tensor
    P_filt: tf.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class BaseFlow(ABC):
    """Abstract base class for particle flow methods.

    Subclasses implement the per-step flow transport and optionally
    override the full filter loop.
    """

    def __init__(self, n_particles: int = 500, n_flow_steps: int = 20,
                 lambda_schedule: Optional[np.ndarray] = None,
                 filter_type: str = 'ekf'):
        """
        Parameters
        ----------
        n_particles : int
            Number of particles.
        n_flow_steps : int
            Number of flow integration steps.
        lambda_schedule : ndarray, optional
            Custom lambda positions [0, ..., 1]. If None, uses uniform.
        filter_type : str
            'ekf' or 'ukf' for prediction/update.
        """
        self.n_particles = n_particles
        self.n_flow_steps = n_flow_steps
        self.filter_type = filter_type

        if lambda_schedule is not None:
            self.lambda_schedule = tf.constant(lambda_schedule, dtype=tf.float64)
            self.n_flow_steps = len(lambda_schedule) - 1
        else:
            self.lambda_schedule = tf.cast(
                tf.linspace(0.0, 1.0, n_flow_steps + 1), dtype=tf.float64
            )

    @abstractmethod
    def flow_step(self, particles: tf.Tensor, m_prev: tf.Tensor,
                  P_prev: tf.Tensor, ssm, y: tf.Tensor,
                  rng: tf.random.Generator, **kwargs):
        """Apply flow transport for one time step.

        Subclasses implement the specific flow (EDH, LEDH, stochastic, etc.).

        Parameters
        ----------
        particles : tf.Tensor [N, n_x]
            Current particles.
        m_prev : tf.Tensor [n_x]
            Previous filter mean.
        P_prev : tf.Tensor [n_x, n_x]
            Previous filter covariance.
        ssm : BaseSSM
            State space model.
        y : tf.Tensor [n_y]
            Current observation.
        rng : tf.random.Generator
            Random generator.

        Returns
        -------
        particles : tf.Tensor [N, n_x]
            Transported particles.
        weights : tf.Tensor [N]
            Updated weights.
        m_post : tf.Tensor [n_x]
            Posterior mean estimate.
        P_post : tf.Tensor [n_x, n_x]
            Posterior covariance estimate.
        """

    @abstractmethod
    def filter(self, ssm, ys: tf.Tensor,
               rng: Optional[tf.random.Generator] = None,
               **kwargs) -> FlowResult:
        """Run flow-based filtering over all time steps.

        Parameters
        ----------
        ssm : BaseSSM
            State space model.
        ys : tf.Tensor [T, n_y]
            Observations.
        rng : tf.random.Generator, optional
            Random generator. If None, creates one with default seed.

        Returns
        -------
        FlowResult
        """
