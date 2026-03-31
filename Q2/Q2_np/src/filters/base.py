"""Base filter interface and result container."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

import tensorflow as tf


@dataclass
class FilterResult:
    """Unified container for filter outputs.

    Attributes
    ----------
    m_filt : tf.Tensor [T, n_x]
        Filtered state means.
    P_filt : tf.Tensor [T, n_x, n_x]
        Filtered state covariances.
    diagnostics : dict
        Algorithm-specific diagnostics. Common keys:
        - 'cond_nums': tf.Tensor [T] (KF/EKF/UKF)
        - 'ess': tf.Tensor [T] (PF, PF-PF)
        - 'resample_count': int (PF, PF-PF)
        - 'weights_history': tf.Tensor [T, N] (PF, PF-PF)
    """
    m_filt: tf.Tensor
    P_filt: tf.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class BaseFilter(ABC):
    """Abstract base class for all filters.

    Subclasses implement predict/update steps and the full filter loop.
    All filters accept a BaseSSM and observations, returning FilterResult.
    """

    @abstractmethod
    def filter(self, ssm, ys: tf.Tensor, **kwargs) -> FilterResult:
        """Run filtering on observations.

        Parameters
        ----------
        ssm : BaseSSM
            State space model.
        ys : tf.Tensor [T, n_y]
            Observations.

        Returns
        -------
        FilterResult
        """

    def predict(self, m: tf.Tensor, P: tf.Tensor, ssm, **kwargs):
        """Single prediction step.

        Parameters
        ----------
        m : tf.Tensor [n_x]
            Current state mean.
        P : tf.Tensor [n_x, n_x]
            Current state covariance.
        ssm : BaseSSM
            State space model.

        Returns
        -------
        m_pred : tf.Tensor [n_x]
        P_pred : tf.Tensor [n_x, n_x]
        """
        raise NotImplementedError

    def update(self, m_pred: tf.Tensor, P_pred: tf.Tensor,
               y: tf.Tensor, ssm, **kwargs):
        """Single update step.

        Parameters
        ----------
        m_pred : tf.Tensor [n_x]
            Predicted state mean.
        P_pred : tf.Tensor [n_x, n_x]
            Predicted state covariance.
        y : tf.Tensor [n_y]
            Observation.
        ssm : BaseSSM
            State space model.

        Returns
        -------
        m_post : tf.Tensor [n_x]
        P_post : tf.Tensor [n_x, n_x]
        """
        raise NotImplementedError
