"""Base resampler interface."""
from abc import ABC, abstractmethod


class ResamplerBase(ABC):
    """Abstract base for particle resamplers.

    All resamplers accept log-weights and particles, returning
    resampled particles and new log-weights.
    """

    DIFFERENTIABLE = False

    @abstractmethod
    def apply(self, log_weights, particles, rng):
        """Resample particles given log-weights.

        Parameters
        ----------
        log_weights : tf.Tensor [N]
            Unnormalized log importance weights.
        particles : tf.Tensor [N, n_x]
            Current particles.
        rng : tf.random.Generator
            Random number generator.

        Returns
        -------
        particles_new : tf.Tensor [N, n_x]
            Resampled particles.
        log_weights_new : tf.Tensor [N]
            New log-weights (uniform for non-differentiable resamplers).
        """


class NoResampling(ResamplerBase):
    """Identity resampler -- returns particles and weights unchanged.

    Useful as a baseline to isolate the effect of resampling
    on filtering accuracy and gradient flow.
    """

    DIFFERENTIABLE = True

    def apply(self, log_weights, particles, rng=None):
        return particles, log_weights
