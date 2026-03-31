"""Base class for Particle Flow Particle Filters (PFPF).

PFPF methods use a particle flow as proposal distribution for importance-weighted
particle filtering. Shared base for PFPFEDHFilter (Algorithm 2) and
PFPFLEDHFilter (Algorithm 1) from Li et al. 2017.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import tensorflow as tf

from ..filters.base import FilterResult
from ..filters.base_particle import BaseParticleFilter
from ..filters.common import DTYPE


class PFPFFilter(BaseParticleFilter, ABC):
    """Abstract base class for importance-weighted PF using flow as proposal.

    Subclasses implement their own flow_step and _filter_loop.
    """

    def __init__(self, n_particles: int = 500, n_flow_steps: int = 20,
                 lambda_schedule: Optional[np.ndarray] = None,
                 filter_type: str = 'ekf',
                 resample_threshold: float = 0.5):
        super().__init__(n_particles)
        self.n_flow_steps = n_flow_steps
        self.filter_type = filter_type
        self.resample_threshold = resample_threshold

        if lambda_schedule is not None:
            self.lambda_schedule = tf.constant(lambda_schedule, dtype=DTYPE)
            self.n_flow_steps = len(lambda_schedule) - 1
        else:
            self.lambda_schedule = tf.cast(
                tf.linspace(0.0, 1.0, n_flow_steps + 1), dtype=DTYPE)

    @abstractmethod
    def filter(self, ssm, ys, rng=None, **kwargs) -> FilterResult:
        """Run filtering. Subclasses implement their own filter loop."""
